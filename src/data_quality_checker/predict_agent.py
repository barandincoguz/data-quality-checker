"""Serve G0 predictions to a remote annotation platform.

The platform runs on Hugging Face Spaces (Linux, no MLX) and cannot reach this
machine; this agent therefore runs locally and pushes outbound only. It is
stateless: it asks the platform which documents lack a prediction, runs G0, and
posts the results. A restarted agent — or a platform whose ephemeral database
was wiped — converges again by asking the same question.

Failure policy: a *model-level* error (token limit, unparseable output) is a
deterministic property of that document and is cached as `status="error"`, so the
platform reports "model kontrolü yapılamadı" instead of retrying forever. A
*raised* exception means the environment broke (MLX allocator, missing weights);
nothing is posted, because fabricating a prediction would be worse than having
none.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .config import AppConfig
from .errors import ConfigurationError, ContractError
from .processing import EchoHumanBackend, MlxG0Backend, PredictionBackend

DEFAULT_GENERATION = "G0"
MAX_BACKOFF_SECONDS = 600.0


@dataclass(eq=True)
class AgentStats:
    pending: int = 0
    predicted: int = 0
    upserted: int = 0
    failed: int = 0


def resolve_token(env_name: str, environ: Mapping[str, str]) -> str:
    token = str(environ.get(env_name, "")).strip()
    if not token:
        raise ConfigurationError(
            f"{env_name} is empty; export the platform's DQCHECK_INGEST_TOKEN value"
        )
    return token


class HttpTransport:
    """Minimal stdlib HTTP client for the platform's internal endpoints."""

    def __init__(self, *, base_url: str, token: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout

    def _request(self, method: str, path: str, payload: Any | None = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, method=method
        )
        request.add_header("Authorization", f"Bearer {self._token}")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            body = response.read().decode("utf-8")
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise ContractError(f"platform returned non-JSON body: {exc}") from exc

    def get_pending(self, limit: int) -> list[dict[str, Any]]:
        payload = self._request("GET", f"/api/internal/predictions/pending?limit={limit}")
        documents = payload.get("documents")
        if not isinstance(documents, list):
            raise ContractError("pending response has no documents list")
        return documents

    def post_predictions(self, items: list[dict[str, Any]]) -> int:
        payload = self._request("POST", "/api/internal/predictions", {"items": items})
        return int(payload.get("upserted", 0))


def build_backend(config: AppConfig, *, fake: bool = False) -> PredictionBackend:
    return EchoHumanBackend() if fake else MlxG0Backend(config)


def _predict_one(
    backend: PredictionBackend, document: Mapping[str, Any]
) -> dict[str, Any]:
    result = backend.predict(
        {"text": document["pdf_text"], "human_references": []}
    )
    return {
        "document_id": document["document_id"],
        "generation": DEFAULT_GENERATION,
        "status": result.status,
        "references": list(result.references),
        "truncated": bool(result.operational.get("truncated")),
        "model_fingerprint": backend.model_fingerprint,
        "text_sha256": document["text_sha256"],
        "error": result.error,
        "operational": dict(result.operational),
    }


def run_agent(
    *,
    transport: Any,
    backend: PredictionBackend,
    batch_size: int = 4,
    poll_seconds: float = 30.0,
    once: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
    max_cycles: int | None = None,
) -> AgentStats:
    """Pull → predict → push. `once` runs a single cycle (used by tests and cron).

    `max_cycles` bounds the loop for tests; production runs leave it None.
    """
    stats = AgentStats()
    consecutive_failures = 0
    cycles = 0
    while True:
        cycles += 1
        pending = transport.get_pending(batch_size)
        stats.pending = len(pending)
        batch_failed = False
        items: list[dict[str, Any]] = []
        for document in pending:
            try:
                items.append(_predict_one(backend, document))
            except Exception as exc:  # environment-level failure
                stats.failed += 1
                batch_failed = True
                log(f"predict failed for {document.get('document_id')}: {exc}")
                break
            stats.predicted += 1
        if items:
            try:
                upserted = transport.post_predictions(items)
                stats.upserted += upserted
                log(
                    f"pending={len(pending)} predicted={len(items)} upserted={upserted}"
                )
            except Exception as exc:
                stats.failed += 1
                batch_failed = True
                log(f"ingest failed for {len(items)} item(s): {exc}")
        elif not pending:
            log("pending=0 idle")

        consecutive_failures = consecutive_failures + 1 if batch_failed else 0
        if once or (max_cycles is not None and cycles >= max_cycles):
            return stats
        if consecutive_failures:
            delay = min(poll_seconds * (2**consecutive_failures), MAX_BACKOFF_SECONDS)
            log(f"backing off {delay:.0f}s after {consecutive_failures} failed cycle(s)")
            sleep(delay)
        elif not pending:
            sleep(poll_seconds)
