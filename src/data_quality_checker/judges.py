"""Blind two-model judge pilot, GREEN audit sampling, and explicit judge lock."""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .atomic import write_json_atomic
from .config import AppConfig
from .constants import JUDGE_MODEL_KEY, JUDGE_MODEL_REVISION, MODEL_ID
from .contracts import validate_reference_list
from .errors import ContractError, DQCheckError, GateBlocked, IntegrityError
from .fingerprints import fingerprint_json, sha256_file
from .heartbeat import RunLease
from .judge_model import assert_gemma_thinking_is_disabled, resolve_judge_snapshot
from .normalization import compact_references, core_identity, full_identity
from .preparation import validate_ready
from .reference_policy import apply_reference_policy
from .storage import Store
from .text import evidence_match_mode

JUDGE_MODELS = ("qwen3.5:397b", "deepseek-v3.2")
_STATIC_JUDGE_MODEL_PROVIDERS: dict[str, str] = {
    "qwen3.5:397b": "ollama",
    "deepseek-v3.2": "ollama",
    JUDGE_MODEL_KEY: "mlx",
}
JUDGE_PROMPT = (
    "Act as a blind legal-reference adjudicator. Compare candidate A and B "
    "only against the Turkish document. Return JSON with verdict (A, B, TIE, "
    "or NEITHER), candidate_errors {A:[],B:[]}, final_references, evidence, "
    "and reason_codes. evidence and reason_codes must both be JSON arrays, "
    "never prose: each evidence entry is a single span quoted verbatim from "
    "the document. "
    "Each final_references entry must keep the same fields as the candidate "
    "entries it is drawn from, including source_text quoted verbatim from the "
    "document: a final reference whose source_text is missing, empty, or not a "
    "literal span of the document is rejected outright.\n\n"
)
PILOT_TARGET_PER_BUCKET = 20
PILOT_MAX_DOCS = 60


class JudgeProviderUnavailable(DQCheckError):
    """A judge provider could not be resolved or reached (e.g. missing credential).

    Deliberately based on `DQCheckError` (not `ContractError`, `ValueError`, or
    `TypeError`) so it is caught by the CLI's top-level handler as a clean
    `dqcheck: error: ...` exit, while still NOT matching the pilot's own
    `except (ContractError, ValueError, TypeError)` per-document clause, which
    would otherwise silently turn a missing credential into per-document
    "error" rows instead of aborting the run.
    """


class JudgeProvider(Protocol):
    def judge(self, *, model: str, payload: dict[str, Any]) -> tuple[Any, dict[str, Any]]: ...


@dataclass(frozen=True)
class PilotSelection:
    internal_doc_ids: tuple[str, ...]
    counts: dict[str, int]


def _hash_order(batch_id: str, internal_doc_id: str, *, salt: str) -> str:
    return hashlib.sha256(f"{batch_id}:{salt}:{internal_doc_id}".encode()).hexdigest()


def select_pilot_documents(documents: list[dict[str, Any]], *, batch_id: str) -> PilotSelection:
    by_bucket: dict[str, list[dict[str, Any]]] = {key: [] for key in ("GREEN", "YELLOW", "RED")}
    for document in documents:
        bucket = document.get("router_bucket")
        if bucket in by_bucket:
            by_bucket[bucket].append(document)
    for bucket, rows in by_bucket.items():
        rows.sort(key=lambda row: _hash_order(batch_id, row["internal_doc_id"], salt=bucket))

    selected: dict[str, list[dict[str, Any]]] = {
        bucket: rows[:PILOT_TARGET_PER_BUCKET] for bucket, rows in by_bucket.items()
    }
    target_total = min(PILOT_MAX_DOCS, sum(len(rows) for rows in by_bucket.values()))
    shortage = target_total - sum(len(rows) for rows in selected.values())
    while shortage > 0:
        progressed = False
        for bucket in ("YELLOW", "RED", "GREEN"):
            offset = len(selected[bucket])
            if offset < len(by_bucket[bucket]):
                selected[bucket].append(by_bucket[bucket][offset])
                shortage -= 1
                progressed = True
                if shortage == 0:
                    break
        if not progressed:
            break
    ordered = [
        row["internal_doc_id"] for bucket in ("GREEN", "YELLOW", "RED") for row in selected[bucket]
    ]
    return PilotSelection(tuple(ordered), {bucket: len(selected[bucket]) for bucket in selected})


def _stratified_round_robin(
    documents: list[dict[str, Any]], *, batch_id: str, count: int
) -> list[dict[str, Any]]:
    strata: defaultdict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for document in documents:
        metadata = json.loads(document["metadata_json"])
        references, _ = apply_reference_policy(json.loads(document["human_references_json"]))
        coverage = max(float(document["pdf_coverage"]), float(document["html_coverage"]))
        key = (
            metadata.get("annotation_completed"),
            min(len(references), 4),
            "full" if coverage == 1.0 else "partial" if coverage > 0 else "zero",
            document["selected_channel"],
            min(len(document["text"]) // 5000, 4),
        )
        strata[key].append(document)
    for key, rows in strata.items():
        rows.sort(
            key=lambda row: _hash_order(batch_id, row["internal_doc_id"], salt=f"green-audit:{key}")
        )
    keys = sorted(strata, key=lambda value: repr(value))
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for key in keys:
            if strata[key]:
                selected.append(strata[key].pop(0))
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    return selected


def ensure_green_audit_plan(*, config: AppConfig, store: Store, batch_id: str) -> dict[str, Any]:
    path = config.public_root / "batches" / batch_id / "green_audit.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise IntegrityError(f"invalid GREEN audit plan: {exc}") from exc
        if payload.get("batch_id") != batch_id:
            raise IntegrityError("GREEN audit batch mismatch")
        return payload
    green = store.list_documents(batch_id, router_buckets=["GREEN"])
    requested = max(math.ceil(0.10 * len(green)), 30) if green else 0
    sample_size = min(len(green), requested)
    selected = _stratified_round_robin(green, batch_id=batch_id, count=sample_size)
    payload = {
        "schema_version": 1,
        "batch_id": batch_id,
        "green_count": len(green),
        "requested_sample_size": requested,
        "sample_size": sample_size,
        "sample_internal_doc_ids": [row["internal_doc_id"] for row in selected],
        "sample_public_doc_ids": [row["public_doc_id"] for row in selected],
        "stratification_fields": [
            "annotation_completion",
            "reference_count",
            "evidence_coverage",
            "selected_channel",
            "text_length",
        ],
        "status": "pending" if sample_size else "not_applicable",
    }
    write_json_atomic(path, payload, mode=0o644)
    store.add_event(batch_id, "green_audit_planned", payload)
    return payload


def blind_candidates(
    *,
    batch_id: str,
    internal_doc_id: str,
    model: str,
    human_references: list[dict[str, str]],
    model_references: list[dict[str, str]],
) -> tuple[str, list[dict[str, str]], list[dict[str, str]]]:
    bit = int(_hash_order(batch_id, internal_doc_id, salt=f"blind:{model}"), 16) & 1
    if bit:
        return "A=model,B=human", model_references, human_references
    return "A=human,B=model", human_references, model_references


class FakeJudgeProvider:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def judge(self, *, model: str, payload: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        self.payloads.append(payload)
        references = payload["candidate_a"]
        return (
            {
                "verdict": "A",
                "candidate_errors": {"A": [], "B": ["not_selected"]},
                "final_references": references,
                "evidence": [ref["source_text"] for ref in references if ref["source_text"]],
                "reason_codes": ["fixture_prefers_a"],
            },
            {"latency_seconds": 0.0, "cost": 0.0, "provider": "fake"},
        )


class OllamaJudgeProvider:
    def __init__(self) -> None:
        self.base_url = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com").rstrip("/")
        self.timeout = float(os.environ.get("OLLAMA_TIMEOUT", "500"))
        self.keys = [
            value
            for name in (
                "OLLAMA_API_KEY",
                *[f"OLLAMA_API_KEY_V{i}" for i in range(2, 8)],
            )
            if (value := os.environ.get(name))
        ]
        if not self.keys:
            raise JudgeProviderUnavailable("OLLAMA_API_KEY is unavailable")
        self._key_index = 0

    def judge(self, *, model: str, payload: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        prompt = JUDGE_PROMPT + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        request_payload = json.dumps(
            {
                "model": model,
                "stream": False,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0},
                "format": "json",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=request_payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.keys[self._key_index % len(self.keys)]}",
                "Content-Type": "application/json",
            },
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            self._key_index += 1
            raise JudgeProviderUnavailable(str(exc)) from exc
        content = (body.get("message") or {}).get("content") or body.get("response")
        return content, {
            "latency_seconds": time.perf_counter() - started,
            "cost": body.get("cost"),
            "provider": "ollama",
        }


class MlxJudgeProvider:
    """Local adjudicator on the pinned Gemma 4 31B snapshot.

    Weights load lazily and stay loaded: a judged batch is many documents and
    reloading 21 GB per call would dominate the run. Sampling is greedy, and
    the snapshot revision is pinned, so a re-run of the same batch reproduces
    the same verdicts -- which is the property a locked judge needs and a
    cloud endpoint cannot promise.
    """

    def __init__(self) -> None:
        # 2048 truncated a real 18-reference verdict mid-JSON once every
        # final reference had to carry its own verbatim source_text. A cut-off
        # verdict is not a cheaper verdict, it is an unusable one, so the
        # default carries roughly double the observed need.
        self.max_tokens = int(os.environ.get("MLX_JUDGE_MAX_TOKENS", "4096"))
        self._snapshot = resolve_judge_snapshot()
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from mlx_lm import load
        except ImportError as exc:
            raise JudgeProviderUnavailable(f"mlx_lm is unavailable: {exc}") from exc
        self._model, self._tokenizer = load(str(self._snapshot))

    def _generate(self, prompt: str) -> tuple[str, dict[str, Any]]:
        self._load()
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        text = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        assert_gemma_thinking_is_disabled(text)
        chunks: list[str] = []
        last = None
        for response in stream_generate(
            self._model,
            self._tokenizer,
            text,
            max_tokens=self.max_tokens,
            sampler=make_sampler(temp=0.0),
        ):
            chunks.append(response.text)
            last = response
        meta = {
            "prompt_tokens": getattr(last, "prompt_tokens", None),
            "generation_tokens": getattr(last, "generation_tokens", None),
            "peak_memory_gb": getattr(last, "peak_memory", None),
            "finish_reason": getattr(last, "finish_reason", None),
        }
        return "".join(chunks), meta

    def judge(self, *, model: str, payload: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        prompt = JUDGE_PROMPT + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        started = time.perf_counter()
        content, meta = self._generate(prompt)
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("```")[1].removeprefix("json").strip()
        return stripped, {
            "latency_seconds": time.perf_counter() - started,
            "cost": 0.0,
            "provider": "mlx",
            "model_revision": JUDGE_MODEL_REVISION,
            **meta,
        }


def gemini_judge_model() -> str | None:
    """Resolve the configured Gemini judge model id, or None if unconfigured.

    There is deliberately no default. A wrong default routes silently to a
    model that cannot answer, so an unset GEMINI_JUDGE_MODEL simply leaves
    Gemini out of the registry and the Ollama judges keep working.
    """
    return os.environ.get("GEMINI_JUDGE_MODEL", "").strip() or None


class GeminiJudgeProvider:
    """Google Generative AI adjudicator, stdlib HTTP only.

    Mirrors OllamaJudgeProvider: same prompt, same return shape, same
    fail-closed behaviour. The response is handed back as raw text so
    _validate_judge_result applies the identical contract to every provider.
    """

    def __init__(self) -> None:
        self.base_url = os.environ.get(
            "GEMINI_BASE_URL",
            "https://aiplatform.googleapis.com/v1/publishers/google/models",
        ).rstrip("/")
        self.timeout = float(os.environ.get("GEMINI_TIMEOUT", "500"))
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise JudgeProviderUnavailable("GEMINI_API_KEY is unavailable")
        self.api_key = key

    def judge(self, *, model: str, payload: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        prompt = JUDGE_PROMPT + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        request_payload = json.dumps(
            {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0,
                    "responseMimeType": "application/json",
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/{model}:generateContent",
            data=request_payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
            except Exception:  # noqa: BLE001 - the body is best-effort diagnostic only
                detail = ""
            raise JudgeProviderUnavailable(f"HTTP {exc.code}: {detail or exc.reason}") from exc
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            http.client.HTTPException,
        ) as exc:
            raise JudgeProviderUnavailable(str(exc)) from exc
        candidates = body.get("candidates") or []
        if not candidates:
            raise JudgeProviderUnavailable("gemini returned no candidates")
        parts = ((candidates[0].get("content") or {}).get("parts")) or []
        content = "".join(str(part.get("text", "")) for part in parts)
        if not content:
            raise JudgeProviderUnavailable("gemini returned an empty candidate")
        usage = body.get("usageMetadata") or {}
        return content, {
            "latency_seconds": time.perf_counter() - started,
            "cost": None,
            "provider": "gemini",
            "input_tokens": usage.get("promptTokenCount"),
            "output_tokens": usage.get("candidatesTokenCount"),
        }


def judge_model_providers() -> dict[str, str]:
    """Model id to provider kind, resolved at call time.

    The Gemini entry is keyed on gemini_judge_model(), which reads the
    environment on every call, so the registry follows configuration rather
    than whatever the environment held at import. When GEMINI_JUDGE_MODEL is
    unset or blank, the Gemini entry is omitted entirely rather than falling
    back to a guessed id, so the Ollama judges keep working untouched.
    """
    providers = dict(_STATIC_JUDGE_MODEL_PROVIDERS)
    model = gemini_judge_model()
    if model is None:
        return providers
    if model in providers:
        raise ContractError(
            f"GEMINI_JUDGE_MODEL={model!r} collides with a built-in judge model; "
            f"choose an id outside {sorted(_STATIC_JUDGE_MODEL_PROVIDERS)}"
        )
    providers[model] = "gemini"
    return providers


def resolve_judge_provider(model: str, *, fake_backend: bool = False) -> JudgeProvider:
    """Return the provider that serves `model`.

    `fake_backend` short-circuits before any credential is read, so tests and
    dry runs never touch the network or require a key.
    """
    providers = judge_model_providers()
    kind = providers.get(model)
    if kind is None:
        raise ContractError(f"unknown judge model {model!r}; expected one of {sorted(providers)}")
    if fake_backend:
        return FakeJudgeProvider()
    if kind == "gemini":
        return GeminiJudgeProvider()
    if kind == "mlx":
        return MlxJudgeProvider()
    if kind == "ollama":
        return OllamaJudgeProvider()
    raise ContractError(f"judge model {model!r} maps to unknown provider kind {kind!r}")


def _validate_judge_result(payload: Any, document_text: str) -> dict[str, Any]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ContractError(f"judge output is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError("judge output root must be an object")
    verdict = str(payload.get("verdict", "")).upper()
    if verdict not in {"A", "B", "TIE", "NEITHER"}:
        raise ContractError("judge verdict must be A, B, TIE, or NEITHER")
    candidate_errors = payload.get("candidate_errors")
    if not isinstance(candidate_errors, dict) or not all(
        isinstance(candidate_errors.get(label), list) for label in ("A", "B")
    ):
        raise ContractError("candidate_errors must contain A and B lists")
    references = validate_reference_list(payload.get("final_references"))
    references, _ = apply_reference_policy(references)
    fabricated = [
        index
        for index, reference in enumerate(references)
        if not reference["source_text"]
        or evidence_match_mode(reference["source_text"], document_text) is None
    ]
    if fabricated:
        raise ContractError(f"fabricated_or_missing_evidence at references {fabricated}")
    evidence = payload.get("evidence")
    reason_codes = payload.get("reason_codes")
    if not isinstance(evidence, list) or not isinstance(reason_codes, list):
        raise ContractError("evidence and reason_codes must be lists")
    if any(evidence_match_mode(item, document_text) is None for item in evidence if item):
        raise ContractError("judge evidence contains a span absent from the document")
    return {
        "verdict": verdict,
        "candidate_errors": {
            "A": list(candidate_errors["A"]),
            "B": list(candidate_errors["B"]),
        },
        "final_references": references,
        "evidence": [str(item) for item in evidence],
        "reason_codes": [str(item) for item in reason_codes],
    }


def _safe_model_name(model: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in model)


def _run_pilot_impl(
    *,
    config: AppConfig,
    batch_id: str,
    allow_external_judge: bool,
    generation: str = "G0",
    fake_backend: bool = False,
    provider: JudgeProvider | None = None,
    lease: RunLease | None = None,
    judge_models: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if not fake_backend and not allow_external_judge:
        raise GateBlocked("--allow-external-judge is required before any external call")
    models = tuple(judge_models) if judge_models else JUDGE_MODELS
    unknown = [model for model in models if model not in judge_model_providers()]
    if unknown:
        raise ContractError(
            f"unknown judge models {unknown}; expected from {sorted(judge_model_providers())}"
        )
    with Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store:
        batch = store.get_batch(batch_id)
        if batch is None or batch["status"] != "processed":
            raise GateBlocked("judge pilot requires a fully processed batch")
        documents = store.list_documents(batch_id)
        audit = ensure_green_audit_plan(config=config, store=store, batch_id=batch_id)
        selection = select_pilot_documents(documents, batch_id=batch_id)
        selection_manifest = {
            "schema_version": 1,
            "batch_id": batch_id,
            "maximum_documents": PILOT_MAX_DOCS,
            "target_per_bucket": PILOT_TARGET_PER_BUCKET,
            "counts": selection.counts,
            "internal_doc_ids": list(selection.internal_doc_ids),
            "public_doc_ids": [
                next(row["public_doc_id"] for row in documents if row["internal_doc_id"] == doc_id)
                for doc_id in selection.internal_doc_ids
            ],
            "models": list(models),
            "local_generator_role": {
                "model": MODEL_ID,
                "role": "G0 candidate generator, never a remote judge",
            },
        }
        public_dir = config.public_root / "batches" / batch_id
        write_json_atomic(public_dir / "judge_pilot_selection.json", selection_manifest, mode=0o644)
        counts_by_model: dict[str, dict[str, int]] = {
            model: {"valid": 0, "unavailable": 0, "error": 0} for model in models
        }
        total_latency: dict[str, float] = {model: 0.0 for model in models}
        total_cost: dict[str, float] = {model: 0.0 for model in models}
        document_by_id = {row["internal_doc_id"]: row for row in documents}
        providers_by_model: dict[str, JudgeProvider] = {}

        for selection_index, internal_doc_id in enumerate(selection.internal_doc_ids, 1):
            document = document_by_id[internal_doc_id]
            human, _ = apply_reference_policy(json.loads(document["human_references_json"]))
            prediction = store.get_prediction(batch_id, internal_doc_id, generation)
            if prediction is None:
                raise IntegrityError(
                    f"missing {generation} prediction for pilot doc {internal_doc_id}"
                )
            model_references, _ = apply_reference_policy(json.loads(prediction["references_json"]))
            for model in models:
                existing = store.get_judge_result(batch_id, internal_doc_id, model)
                if existing is not None and existing["status"] == "valid":
                    response_path = Path(existing["response_path"])
                    if (
                        not response_path.is_file()
                        or sha256_file(response_path) != existing["response_sha256"]
                    ):
                        raise IntegrityError(
                            f"judge result file is missing or corrupt: {response_path}"
                        )
                    counts_by_model[model]["valid"] += 1
                    continue
                mapping, candidate_a, candidate_b = blind_candidates(
                    batch_id=batch_id,
                    internal_doc_id=internal_doc_id,
                    model=model,
                    human_references=human,
                    model_references=model_references,
                )
                external_payload = {
                    "document_text": document["text"],
                    "candidate_a": candidate_a,
                    "candidate_b": candidate_b,
                    "instructions": "Use only evidence present in document_text.",
                }
                attempts: list[dict[str, Any]] = []
                validated: dict[str, Any] | None = None
                operational: dict[str, Any] = {}
                last_error = ""
                unavailable = False
                if model not in providers_by_model:
                    providers_by_model[model] = provider or resolve_judge_provider(
                        model, fake_backend=fake_backend
                    )
                model_provider = providers_by_model[model]
                for attempt in range(1, 4):
                    try:
                        raw, operational = model_provider.judge(
                            model=model, payload=external_payload
                        )
                        validated = _validate_judge_result(raw, document["text"])
                        attempts.append({"attempt": attempt, "status": "valid"})
                        break
                    except JudgeProviderUnavailable as exc:
                        last_error = str(exc)
                        unavailable = True
                        attempts.append(
                            {
                                "attempt": attempt,
                                "status": "unavailable",
                                "error": last_error,
                            }
                        )
                    except (ContractError, ValueError, TypeError) as exc:
                        last_error = str(exc)
                        attempts.append(
                            {
                                "attempt": attempt,
                                "status": "invalid",
                                "error": last_error,
                            }
                        )
                status = (
                    "valid"
                    if validated is not None
                    else ("unavailable" if unavailable else "error")
                )
                result_payload = {
                    "schema_version": 1,
                    "batch_id": batch_id,
                    "internal_doc_id": internal_doc_id,
                    "model": model,
                    "blind_mapping": mapping,
                    "status": status,
                    "result": validated or {},
                    "attempts": attempts,
                    "operational": operational,
                    "error": None if validated is not None else last_error,
                }
                result_path = (
                    config.sensitive_root
                    / "batches"
                    / batch_id
                    / "judges"
                    / "pilot"
                    / _safe_model_name(model)
                    / f"{internal_doc_id}.json"
                )
                write_json_atomic(result_path, result_payload)
                store.persist_judge_result(
                    batch_id=batch_id,
                    internal_doc_id=internal_doc_id,
                    model=model,
                    blind_mapping=mapping,
                    status=status,
                    verdict=None if validated is None else validated["verdict"],
                    result=validated or {},
                    response_path=result_path,
                    response_sha256=sha256_file(result_path),
                    retry_count=max(0, len(attempts) - 1),
                    error=None if validated is not None else last_error,
                )
                counts_by_model[model][status] += 1
                latency = operational.get("latency_seconds")
                cost = operational.get("cost")
                if isinstance(latency, (int, float)):
                    total_latency[model] += float(latency)
                if isinstance(cost, (int, float)):
                    total_cost[model] += float(cost)
            if lease is not None:
                lease.beat(
                    stage="judge-pilot",
                    completed_units=selection_index,
                    expected_units=len(selection.internal_doc_ids),
                    last_successful_unit=internal_doc_id,
                )

        summary = {
            "schema_version": 1,
            "batch_id": batch_id,
            "selected_document_count": len(selection.internal_doc_ids),
            "selection_counts": selection.counts,
            "models": {
                model: {
                    **counts_by_model[model],
                    "valid_output_rate": (
                        counts_by_model[model]["valid"] / len(selection.internal_doc_ids)
                        if selection.internal_doc_ids
                        else None
                    ),
                    "total_latency_seconds": total_latency[model],
                    "reported_cost": total_cost[model],
                    "expert_metrics": "pending",
                }
                for model in models
            },
            "external_consent": bool(allow_external_judge),
            "fake_backend": fake_backend,
            "green_audit": audit,
            "production_judge_locked": False,
        }
        write_json_atomic(public_dir / "judge_pilot_summary.json", summary, mode=0o644)
        store.add_event(batch_id, "judge_pilot_completed", summary)
        return summary


def _run_locked_judge_coverage(
    *,
    config: AppConfig,
    batch_id: str,
    model: str,
    provider: JudgeProvider,
    fake_backend: bool,
    allow_external_judge: bool,
    lease: RunLease,
    generation: str = "G0",
) -> dict[str, Any]:
    with Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store:
        documents = store.list_documents(batch_id)
        escalation = (config.public_root / "batches" / batch_id / "green_escalation.json").exists()
        targets = [
            document
            for document in documents
            if document["router_bucket"] in {"RED", "YELLOW"}
            or (escalation and document["router_bucket"] == "GREEN")
        ]
        targets.sort(key=lambda row: row["internal_doc_id"])
        counts = {"valid": 0, "unavailable": 0, "error": 0}
        lease.beat(stage="judge-production", expected_units=len(targets), completed_units=0)
        for index, document in enumerate(targets, 1):
            internal_doc_id = document["internal_doc_id"]
            existing = store.get_judge_result(batch_id, internal_doc_id, model)
            if existing is not None and existing["status"] == "valid":
                result_path = Path(existing["response_path"])
                if (
                    not result_path.is_file()
                    or sha256_file(result_path) != existing["response_sha256"]
                ):
                    raise IntegrityError(f"judge result file is missing or corrupt: {result_path}")
                counts["valid"] += 1
                lease.beat(completed_units=index, last_successful_unit=internal_doc_id)
                continue
            prediction = store.get_prediction(batch_id, internal_doc_id, generation)
            if prediction is None:
                raise IntegrityError(f"missing {generation} prediction for {internal_doc_id}")
            human, _ = apply_reference_policy(json.loads(document["human_references_json"]))
            model_references, _ = apply_reference_policy(json.loads(prediction["references_json"]))
            mapping, candidate_a, candidate_b = blind_candidates(
                batch_id=batch_id,
                internal_doc_id=internal_doc_id,
                model=model,
                human_references=human,
                model_references=model_references,
            )
            external_payload = {
                "document_text": document["text"],
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
                "instructions": "Use only evidence present in document_text.",
            }
            attempts: list[dict[str, Any]] = []
            validated: dict[str, Any] | None = None
            operational: dict[str, Any] = {}
            last_error = ""
            unavailable = False
            for attempt in range(1, 4):
                try:
                    raw, operational = provider.judge(model=model, payload=external_payload)
                    validated = _validate_judge_result(raw, document["text"])
                    attempts.append({"attempt": attempt, "status": "valid"})
                    break
                except JudgeProviderUnavailable as exc:
                    unavailable = True
                    last_error = str(exc)
                    attempts.append(
                        {
                            "attempt": attempt,
                            "status": "unavailable",
                            "error": last_error,
                        }
                    )
                except (ContractError, ValueError, TypeError) as exc:
                    last_error = str(exc)
                    attempts.append({"attempt": attempt, "status": "invalid", "error": last_error})
            status = (
                "valid" if validated is not None else ("unavailable" if unavailable else "error")
            )
            result_payload = {
                "schema_version": 1,
                "batch_id": batch_id,
                "internal_doc_id": internal_doc_id,
                "model": model,
                "stage": "production",
                "blind_mapping": mapping,
                "status": status,
                "result": validated or {},
                "attempts": attempts,
                "operational": operational,
                "error": None if validated is not None else last_error,
            }
            result_path = (
                config.sensitive_root
                / "batches"
                / batch_id
                / "judges"
                / "production"
                / _safe_model_name(model)
                / f"{internal_doc_id}.json"
            )
            write_json_atomic(result_path, result_payload)
            store.persist_judge_result(
                batch_id=batch_id,
                internal_doc_id=internal_doc_id,
                model=model,
                blind_mapping=mapping,
                status=status,
                verdict=None if validated is None else validated["verdict"],
                result=validated or {},
                response_path=result_path,
                response_sha256=sha256_file(result_path),
                retry_count=max(0, len(attempts) - 1),
                error=None if validated is not None else last_error,
            )
            counts[status] += 1
            lease.beat(completed_units=index, last_successful_unit=internal_doc_id)
        summary = {
            "schema_version": 1,
            "batch_id": batch_id,
            "stage": "production",
            "locked_model": model,
            "required_document_count": len(targets),
            "counts": counts,
            "coverage_complete": counts["valid"] == len(targets),
            "green_escalated": escalation,
            "external_consent": bool(allow_external_judge),
            "fake_backend": fake_backend,
        }
        write_json_atomic(
            config.public_root / "batches" / batch_id / "judge_production_summary.json",
            summary,
            mode=0o644,
        )
        store.add_event(batch_id, "judge_production_completed", summary)
        return summary


def run_judge_pilot(
    *,
    config: AppConfig,
    batch_id: str,
    allow_external_judge: bool,
    generation: str = "G0",
    fake_backend: bool = False,
    provider: JudgeProvider | None = None,
    judge_models: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if not fake_backend and not allow_external_judge:
        raise GateBlocked("--allow-external-judge is required before any external call")
    ready = validate_ready(config, batch_id)
    lease = RunLease(
        lock_path=config.sensitive_root / "locks" / f"judge_{batch_id}.lock",
        heartbeat_path=config.sensitive_root / "batches" / batch_id / "judge_heartbeat.json",
        purpose="judge",
        run_id=f"judge:{batch_id}",
        input_fingerprint=str(ready["input_fingerprint"]),
        config_fingerprint=config.fingerprint,
    ).start(stage="judge-preflight")
    try:
        lock_path = config.public_root / "batches" / batch_id / "judge_lock.json"
        if lock_path.exists():
            lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
            locked_model = str(lock_payload["model"])
            result = _run_locked_judge_coverage(
                config=config,
                batch_id=batch_id,
                model=locked_model,
                provider=provider
                or resolve_judge_provider(locked_model, fake_backend=fake_backend),
                fake_backend=fake_backend,
                allow_external_judge=allow_external_judge,
                lease=lease,
                generation=generation,
            )
        else:
            result = _run_pilot_impl(
                config=config,
                batch_id=batch_id,
                allow_external_judge=allow_external_judge,
                generation=generation,
                fake_backend=fake_backend,
                provider=provider,
                lease=lease,
                judge_models=judge_models,
            )
        lease.finish(status="completed")
        return result
    except BaseException as exc:
        lease.finish(status="failed", error=str(exc))
        raise


def _set_metric(references: list[dict[str, Any]], *, core: bool) -> set[Any]:
    compacted = compact_references(references)
    return {
        core_identity(reference) if core else full_identity(reference) for reference in compacted
    }


def judge_expert_metrics(
    store: Store, *, batch_id: str, model: str, ids: list[str]
) -> dict[str, Any]:
    if not ids:
        return {"status": "complete", "document_count": 0}
    exact = 0
    tp = fp = fn = 0
    latencies: list[float] = []
    costs: list[float] = []
    for internal_doc_id in ids:
        review = store.get_review(batch_id, internal_doc_id)
        result = store.get_judge_result(batch_id, internal_doc_id, model)
        if review is None or review["status"] != "finalized" or result is None:
            return {"status": "pending"}
        if result["status"] != "valid":
            return {"status": result["status"], "document_count": len(ids)}
        expert = json.loads(review["final_references_json"] or "[]")
        judge = json.loads(result["result_json"]).get("final_references", [])
        expert_full, judge_full = (
            _set_metric(expert, core=False),
            _set_metric(judge, core=False),
        )
        exact += expert_full == judge_full
        expert_core, judge_core = (
            _set_metric(expert, core=True),
            _set_metric(judge, core=True),
        )
        tp += len(expert_core & judge_core)
        fp += len(judge_core - expert_core)
        fn += len(expert_core - judge_core)
        if result["response_path"]:
            payload = json.loads(Path(result["response_path"]).read_text(encoding="utf-8"))
            latency = (payload.get("operational") or {}).get("latency_seconds")
            cost = (payload.get("operational") or {}).get("cost")
            if isinstance(latency, (int, float)):
                latencies.append(float(latency))
            if isinstance(cost, (int, float)):
                costs.append(float(cost))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "status": "complete",
        "document_count": len(ids),
        "expert_exact_set_agreement": exact / len(ids),
        "expert_core_precision": precision,
        "expert_core_recall": recall,
        "expert_core_f1": f1,
        "fabricated_evidence_rate": 0.0,
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "reported_cost": sum(costs) if costs else None,
    }


def lock_judge(*, config: AppConfig, batch_id: str, model: str, reason: str) -> dict[str, Any]:
    if model not in judge_model_providers():
        raise ContractError(f"model must be one of {sorted(judge_model_providers())}")
    if not reason.strip():
        raise ContractError("judge lock requires a non-empty reason")
    selection_path = config.public_root / "batches" / batch_id / "judge_pilot_selection.json"
    try:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateBlocked(f"judge pilot selection is unavailable: {exc}") from exc
    ids = [str(value) for value in selection.get("internal_doc_ids", [])]
    summary_path = config.public_root / "batches" / batch_id / "judge_pilot_summary.json"
    try:
        pilot_models = tuple(json.loads(summary_path.read_text(encoding="utf-8"))["models"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        pilot_models = JUDGE_MODELS
    if model not in pilot_models:
        raise GateBlocked(f"judge {model} did not run in this batch's pilot")
    with Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store:
        all_metrics = {
            candidate: judge_expert_metrics(store, batch_id=batch_id, model=candidate, ids=ids)
            for candidate in pilot_models
        }
        metrics = all_metrics[model]
        if metrics.get("status") != "complete":
            raise GateBlocked(
                "all pilot documents require finalized expert review before judge-lock"
            )
        unavailable = [
            row
            for row in store.list_judge_results(batch_id, model=model)
            if row["status"] != "valid"
        ]
        if unavailable:
            raise GateBlocked(f"judge {model} has unavailable/invalid pilot outputs")
        payload = {
            "schema_version": 1,
            "batch_id": batch_id,
            "model": model,
            "reason": reason.strip(),
            "metrics": metrics,
            "pilot_model_metrics": all_metrics,
            "lock_fingerprint": fingerprint_json(
                {
                    "batch_id": batch_id,
                    "model": model,
                    "reason": reason.strip(),
                    "metrics": metrics,
                }
            ),
        }
        path = config.public_root / "batches" / batch_id / "judge_lock.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != payload:
                raise GateBlocked("a different production judge is already locked")
            return existing
        write_json_atomic(path, payload, mode=0o644)
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for candidate, candidate_metrics in all_metrics.items():
                if candidate in summary.get("models", {}):
                    summary["models"][candidate]["expert_metrics"] = candidate_metrics
            summary["production_judge_locked"] = True
            summary["locked_model"] = model
            write_json_atomic(summary_path, summary, mode=0o644)
        store.add_event(batch_id, "judge_locked", payload)
        return payload
