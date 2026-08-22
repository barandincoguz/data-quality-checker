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
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .config import AppConfig
from .errors import ConfigurationError, ContractError
from .fingerprints import sha256_text
from .processing import MlxG0Backend, PredictionBackend

DEFAULT_GENERATION = "G0"
PRODUCTION_BACKEND_ID = "mlx-g0"
APPROVED_PRODUCTION_MODEL_FINGERPRINTS = frozenset(
    {
        "3018af0b678572a71588f37132d0318a9eebf210193bedfd5931d9a42b4989f3",
    }
)
MAX_BACKOFF_SECONDS = 600.0
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
SUCCESS_POLL_FLOOR_SECONDS = 0.25
_OPERATIONAL_FIELDS = frozenset(
    {
        "backend",
        "input_tokens",
        "output_tokens",
        "latency_seconds",
        "truncated",
        "generation_attempted",
        "finish_reason",
        "ttft_seconds",
        "prompt_tps",
        "generation_tps",
        "peak_memory_bytes",
    }
)


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


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


class HttpTransport:
    """Minimal stdlib HTTP client for the platform's internal endpoints."""

    def __init__(self, *, base_url: str, token: str, timeout: float = 60.0) -> None:
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ConfigurationError("predict-agent --space-url must be an https URL")
        if parsed.username or parsed.password:
            raise ConfigurationError("predict-agent --space-url must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ConfigurationError("predict-agent --space-url must not contain query or fragment")
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout
        self._opener = urllib.request.build_opener(_NoRedirectHandler())

    def _request(self, method: str, path: str, payload: Any | None = None) -> Any:
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        request = urllib.request.Request(f"{self.base_url}{path}", data=data, method=method)
        request.add_header("Authorization", f"Bearer {self._token}")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        with self._opener.open(request, timeout=self._timeout) as response:
            if response.status not in (200, 201):
                raise ContractError(f"platform returned unexpected HTTP {response.status}")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ContractError("platform response exceeded 16 MiB")
        body = raw.decode("utf-8")
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise ContractError(f"platform returned non-JSON body: {exc}") from exc

    def get_pending(self, limit: int) -> list[dict[str, Any]]:
        payload = self._request("GET", f"/api/internal/predictions/pending?limit={limit}")
        if not isinstance(payload, Mapping):
            raise ContractError("pending response is not an object")
        documents = payload.get("documents")
        if not isinstance(documents, list):
            raise ContractError("pending response has no documents list")
        if len(documents) > limit:
            raise ContractError("pending response exceeded requested limit")
        if any(not isinstance(document, Mapping) for document in documents):
            raise ContractError("pending response contains a non-object document")
        return [dict(document) for document in documents]

    def post_predictions(self, items: list[dict[str, Any]]) -> int:
        payload = self._request("POST", "/api/internal/predictions", {"items": items})
        if not isinstance(payload, Mapping):
            raise ContractError("prediction ingest response is not an object")
        upserted = int(payload.get("upserted", -1))
        rejected = int(payload.get("rejected", 0))
        if rejected:
            raise ContractError(f"platform rejected {rejected} prediction item(s)")
        if upserted != len(items):
            raise ContractError(f"platform upserted {upserted} of {len(items)} prediction item(s)")
        return upserted


def build_backend(config: AppConfig, **legacy_options: Any) -> PredictionBackend:
    """Build the only backend permitted to send remote predictions.

    ``legacy_options`` is rejected deliberately so old automation carrying
    ``fake=True``/``allow_fixture=True`` fails closed instead of silently
    selecting a test backend.
    """
    if legacy_options:
        raise ConfigurationError("predict-agent never permits fixture backend ingest")
    return MlxG0Backend(config)


def _safe_operational(values: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key in _OPERATIONAL_FIELDS:
        if key not in values:
            continue
        value = values[key]
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        safe[key] = value
    return safe


def _clip_reference(reference: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    limits = {
        "kanun_no": 64,
        "kanun_ad": 512,
        "madde": 64,
        "fikra": 64,
        "bent": 64,
        "source_text": 4_000,
    }
    clipped = False
    out: dict[str, Any] = {}
    for field, limit in limits.items():
        value = reference.get(field)
        if field == "source_text" and value is None:
            value = ""
        if value is None:
            out[field] = None
            continue
        text = str(value)
        if len(text) > limit:
            text = text[:limit]
            clipped = True
        out[field] = text
    return out, clipped


def _predict_one(backend: PredictionBackend, document: Mapping[str, Any]) -> dict[str, Any]:
    if getattr(backend, "backend_id", None) != PRODUCTION_BACKEND_ID:
        raise ConfigurationError(
            "predict-agent refuses non-production prediction backend provenance"
        )
    if backend.model_fingerprint not in APPROVED_PRODUCTION_MODEL_FINGERPRINTS:
        raise ConfigurationError("predict-agent refuses an unapproved production model fingerprint")
    document_id = str(document["document_id"])
    pdf_text = str(document["pdf_text"])
    computed_hash = sha256_text(pdf_text)
    if document.get("text_sha256") != computed_hash:
        return {
            "document_id": document_id,
            "generation": DEFAULT_GENERATION,
            "status": "error",
            "references": [],
            "truncated": False,
            "model_fingerprint": backend.model_fingerprint,
            "text_sha256": computed_hash,
            "source": "dqcheck_agent",
            "error": "platform text_sha256 mismatch; prediction was not attempted",
            "operational": {
                "backend": PRODUCTION_BACKEND_ID,
                "truncated": False,
                "generation_attempted": False,
            },
        }
    result = backend.predict({"text": pdf_text, "human_references": []})
    clipped = len(result.references) > 200
    references = []
    for raw_reference in result.references[:200]:
        if not isinstance(raw_reference, Mapping):
            clipped = True
            continue
        reference, item_clipped = _clip_reference(raw_reference)
        clipped = clipped or item_clipped
        references.append(reference)
    operational = _safe_operational(result.operational)
    operational["backend"] = PRODUCTION_BACKEND_ID
    if clipped:
        operational["truncated"] = True
    return {
        "document_id": document_id,
        "generation": DEFAULT_GENERATION,
        "status": result.status,
        "references": references,
        "truncated": bool(operational.get("truncated")),
        "model_fingerprint": backend.model_fingerprint,
        "text_sha256": computed_hash,
        "source": "dqcheck_agent",
        "error": result.error[:2_000] if result.error else None,
        "operational": operational,
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
    if not 1 <= batch_size <= 16:
        raise ConfigurationError("batch_size must be between 1 and 16")
    if poll_seconds <= 0:
        raise ConfigurationError("poll_seconds must be greater than zero")
    stats = AgentStats()
    consecutive_failures = 0
    cycles = 0
    while True:
        cycles += 1
        try:
            pending = transport.get_pending(batch_size)
        except Exception as exc:
            pending = []
            stats.failed += 1
            batch_failed = True
            log(f"poll failed: {exc}")
        else:
            batch_failed = False
        stats.pending = len(pending)
        items: list[dict[str, Any]] = []
        for document in pending:
            if not isinstance(document, Mapping):
                stats.failed += 1
                batch_failed = True
                log("predict failed for <invalid-document>: pending item is not an object")
                continue
            try:
                items.append(_predict_one(backend, document))
            except Exception as exc:  # environment-level failure
                stats.failed += 1
                batch_failed = True
                log(f"predict failed for {document.get('document_id')}: {exc}")
                continue
            stats.predicted += 1
        if items:
            try:
                upserted = transport.post_predictions(items)
                if upserted != len(items):
                    raise ContractError(f"platform upserted {upserted} of {len(items)} item(s)")
                stats.upserted += upserted
                log(f"pending={len(pending)} predicted={len(items)} upserted={upserted}")
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
            exponent = min(consecutive_failures, 10)
            delay = min(poll_seconds * (2**exponent), MAX_BACKOFF_SECONDS)
            log(f"backing off {delay:.0f}s after {consecutive_failures} failed cycle(s)")
            sleep(delay)
        elif not pending:
            sleep(poll_seconds)
        else:
            sleep(min(poll_seconds, SUCCESS_POLL_FLOOR_SECONDS))
