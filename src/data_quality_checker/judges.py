"""Blind judge pilot, GREEN audit sampling, and explicit judge lock."""

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
from .constants import (
    JUDGE_MODEL_KEY,
    JUDGE_MODEL_REVISION,
    MODEL_ID,
    REFERENCE_DECISION_FIELDS,
)
from .contracts import validate_reference_list
from .errors import ContractError, DQCheckError, GateBlocked, IntegrityError
from .fingerprints import fingerprint_json, sha256_file
from .heartbeat import RunLease
from .judge_model import assert_gemma_thinking_is_disabled, resolve_judge_snapshot
from .performance import _percentile
from .preparation import validate_ready
from .reference_matching import compare_with_policy
from .reference_policy import apply_reference_policy
from .storage import Store
from .text import evidence_match_mode

# The default pilot runs the local judge alone. A cloud id left in this tuple
# would not degrade gracefully: _run_pilot_impl resolves each provider outside
# its retry block, and OllamaJudgeProvider raises in __init__ when no key is
# present, so one unusable default would abort the whole pilot at the first
# document rather than being recorded as unavailable. The cloud ids stay
# registered and reachable through --judge-models wherever a key exists.
JUDGE_MODELS = (JUDGE_MODEL_KEY,)
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
    "entries it is drawn from. "
    "Reading the Turkish text: a back-reference such as aynı, anılan, mezkûr "
    "or söz konusu Kanun means the nearest law named before it, and a section "
    "heading such as KURUMLAR VERGİSİ KANUNU YÖNÜNDEN opens a new such "
    "context. Signature blocks, addresses, telephone and fax lines and page "
    "footers are not legal context and never carry a back-reference across "
    "them. A reference is a law and one article of that law; madde is the "
    "article's own designation, and a slash before an uppercase letter, as in "
    "32/A, marks an additional article in its own right. "
    "These instructions are not part of the document. Every evidence span "
    "must be copied from document_text alone; a span taken "
    "from this instruction text is rejected. Before keeping any span, find "
    "it in document_text and copy it from there; if you cannot find it, "
    "quote a passage you can find or drop the reference.\n\n"
)


def judge_prompt_fingerprint() -> str:
    """sha256 of the exact instructions every judge received.

    The judge is meant to be a fixed instrument: a pinned model revision at
    temperature zero. The prompt is the other half of that instrument and was
    recorded nowhere, so a wording change would have moved every downstream
    number invisibly. Rounds carrying different fingerprints were not judged by
    the same instrument and must not be compared as if they were.
    """

    return hashlib.sha256(JUDGE_PROMPT.encode("utf-8")).hexdigest()


# v2 (1 Oct 2026): the decision fields are the law and the article only.
CANDIDATE_PAYLOAD_CONTRACT = "decision_fields_only_v2_law_article"


def judge_instrument() -> dict[str, str]:
    """Name the whole instrument a verdict came from, not half of it.

    `judge_prompt_fingerprint` covered the instructions. The other half is
    the shape of the payload those instructions describe, and it was recorded
    nowhere. Round 1's pilot ran with `source_text` on every annotator
    candidate and recorded prompt sha `aa887e10…` -- the same sha a correctly
    blinded run with an unchanged prompt would have recorded, so the invalid
    verdicts and their replacements would have been indistinguishable in the
    record, while HANDOFF's rule that rounds carrying different fingerprints
    must not be compared quietly relied on that sha.
    """

    return {
        "prompt_sha256": judge_prompt_fingerprint(),
        "candidate_payload_contract": CANDIDATE_PAYLOAD_CONTRACT,
    }


def _instrument_matches(stored: dict[str, Any]) -> bool:
    """True when a stored verdict was produced by this exact instrument."""

    return stored.get("judge_instrument") == judge_instrument()


# Names what *accepted* a verdict, not what *produced* it.
#
# `judge_instrument()` fingerprints the prompt and candidate payload shape --
# the two things that determine what the judge was asked and shown. This is a
# different axis: which rule `_validate_judge_result` applied when deciding
# the resulting JSON was a valid verdict. It deliberately does NOT live inside
# `judge_instrument()`. The resume guard in `_run_pilot_impl` re-judges a
# document only when its stored `judge_instrument` doesn't match the current
# one, and every one of M01's 56 already-valid verdicts was produced by the
# same prompt and payload shape before and after this validator changed --
# their evidence passes under both the old whole-array check and the new
# per-span one. Folding this constant into `judge_instrument()` would change
# that fingerprint for all 56 and force them to be re-judged for no reason:
# the instrument that produced them never moved, only the rule that grades
# their output did. Recording the two side by side lets the stored payload
# answer both questions -- what asked the question, and what accepted the
# answer -- without conflating them.
VALIDATION_CONTRACT = "per_span_evidence_v2"

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
        # No stratum on quote coverage: the annotator's source_text decides nothing.
        key = (
            metadata.get("annotation_completed"),
            min(len(references), 4),
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
            "selected_channel",
            "text_length",
        ],
        "status": "pending" if sample_size else "not_applicable",
    }
    write_json_atomic(path, payload, mode=0o644)
    store.add_event(batch_id, "green_audit_planned", payload)
    return payload


def _decision_fields_only(references: list[dict[str, str]]) -> list[dict[str, str]]:
    """Project a candidate onto the five fields a verdict may rest on.

    A positive projection, not a `source_text` blacklist, so a field added to
    the reference schema later cannot leak into a comparison by default.

    Round 1 measured why this is not optional. Across all 60 pilot documents
    the annotator's 494 rows carried `source_text` and M000's 610 carried
    none, with identical key sets on both sides -- the field's value alone
    named the human. In the GREEN bucket, where the five decision fields are
    identical by construction and `source_text` is therefore the only
    difference, the judge chose the annotator in 17 of 18 decisive verdicts.
    An instruction cannot close that gap: JUDGE_PROMPT already told the judge
    a candidate's source_text is not evidence.
    """

    return [
        {field: reference.get(field, "") for field in REFERENCE_DECISION_FIELDS}
        for reference in references
    ]


def blind_candidates(
    *,
    batch_id: str,
    internal_doc_id: str,
    model: str,
    human_references: list[dict[str, str]],
    model_references: list[dict[str, str]],
) -> tuple[str, list[dict[str, str]], list[dict[str, str]]]:
    bit = int(_hash_order(batch_id, internal_doc_id, salt=f"blind:{model}"), 16) & 1
    human_blind = _decision_fields_only(human_references)
    model_blind = _decision_fields_only(model_references)
    if bit:
        return "A=model,B=human", model_blind, human_blind
    return "A=human,B=model", human_blind, model_blind


class FakeJudgeProvider:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def judge(self, *, model: str, payload: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        self.payloads.append(payload)
        references = payload["candidate_a"]
        # Evidence is quoted from the document, exactly as a real judge's
        # must be. This fixture used to build it from the candidates'
        # `source_text`, which both crashed once candidates were blinded
        # (`KeyError: 'source_text'`, uncaught by the pilot's per-document
        # clause) and kept 24 call sites testing the very shortcut the
        # blinded judge no longer has.
        document_text = str(payload.get("document_text", ""))
        lines = [line for line in document_text.splitlines() if line.strip()]
        return (
            {
                "verdict": "A",
                "candidate_errors": {"A": [], "B": ["not_selected"]},
                "final_references": references,
                "evidence": lines[:1],
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
    """Apply the judge output contract, per span rather than per array.

    Measured across batch M01's 60-document pilot: 280 evidence spans, only 4
    unverifiable (1.43%), and no document carried more than one. Rejecting the
    whole array on one bad span discarded 4 entire verdicts -- 21 verified
    spans among them -- over a guard that is right to exist (see
    `test_evidence_span_absent_from_document_is_still_rejected`: a span the
    judge could not have quoted is a real signal, most often a plausible
    reconstruction stitched from non-contiguous fragments) but was applied at
    the wrong granularity. A span the judge could not quote is a fact about
    that span, not about the eleven others in the same verdict, so it is
    quarantined into `unverified_evidence` rather than raised: the verdict
    survives, the signal is kept, and `judge_expert_metrics` re-sources the
    fabrication-rate signal from this field precisely because a verdict that
    stops failing here would otherwise stop generating the attempt-error rows
    that metric used to count.
    """
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
    evidence = payload.get("evidence")
    reason_codes = payload.get("reason_codes")
    if not isinstance(evidence, list) or not isinstance(reason_codes, list):
        raise ContractError("evidence and reason_codes must be lists")
    verified_evidence: list[str] = []
    unverified_evidence: list[str] = []
    for item in evidence:
        span = str(item)
        if not span.strip():
            continue
        if evidence_match_mode(span, document_text) is None:
            unverified_evidence.append(span)
        else:
            verified_evidence.append(span)
    return {
        "verdict": verdict,
        "candidate_errors": {
            "A": list(candidate_errors["A"]),
            "B": list(candidate_errors["B"]),
        },
        "final_references": references,
        "evidence": verified_evidence,
        "unverified_evidence": unverified_evidence,
        "reason_codes": [str(item) for item in reason_codes],
    }


def judged_prediction_provenance(prediction: dict[str, Any]) -> dict[str, str]:
    """Name the exact model output a verdict was formed against.

    `judge_results` is keyed by (batch_id, internal_doc_id, model): no
    generation. Within the loop that is sound -- a batch is judged against one
    generation -- but the stored verdict named no generation either, so a
    verdict could not be traced to what it judged. M01 already holds verdicts
    formed against G0's predictions and is now judged against M000's; without
    this, the two would be indistinguishable in the record.

    Refuses a prediction it cannot identify. A verdict whose subject is unknown
    is worse than no verdict: it looks like evidence.
    """
    required = ("generation", "model_fingerprint", "response_sha256")
    missing = [field for field in required if not prediction.get(field)]
    if missing:
        raise IntegrityError(
            f"prediction cannot be identified for a judge verdict; missing {missing}"
        )
    return {
        "generation": str(prediction["generation"]),
        "model_fingerprint": str(prediction["model_fingerprint"]),
        "prediction_sha256": str(prediction["response_sha256"]),
        "input_fingerprint": str(prediction.get("input_fingerprint", "")),
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
        instrument = judge_instrument()
        stale_by_model: dict[str, int] = {model: 0 for model in models}

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
                    stored = json.loads(response_path.read_text(encoding="utf-8"))
                    # A pilot is the run you redo when the instrument moves,
                    # so a verdict from another instrument is re-judged rather
                    # than refused -- but it is counted, because the rerun that
                    # silently reused 57 valid verdicts is exactly how round 1
                    # spent a pilot reproducing its own invalid conclusions.
                    if _instrument_matches(stored) and stored.get(
                        "judged_prediction"
                    ) == judged_prediction_provenance(prediction):
                        counts_by_model[model]["valid"] += 1
                        continue
                    stale_by_model[model] += 1
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
                                # Keep what was rejected. Without it a refusal
                                # leaves only its message, and no one can tell
                                # a judge that really fabricated a span from a
                                # validator that was too strict -- which is the
                                # difference between a judge-error metric and a
                                # number nobody can defend.
                                "rejected_output": raw,
                            }
                        )
                status = (
                    "valid"
                    if validated is not None
                    else ("unavailable" if unavailable else "error")
                )
                result_payload = {
                    "schema_version": 3,
                    "batch_id": batch_id,
                    "internal_doc_id": internal_doc_id,
                    "model": model,
                    "blind_mapping": mapping,
                    "judge_instrument": instrument,
                    "validation_contract": VALIDATION_CONTRACT,
                    "status": status,
                    # Which model output this verdict was formed against. The
                    # store's key does not carry it, so without this a verdict
                    # cannot be traced to its subject.
                    "judged_prediction": judged_prediction_provenance(prediction),
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
                    "stale_verdicts_rejudged": stale_by_model[model],
                }
                for model in models
            },
            "external_consent": bool(allow_external_judge),
            "fake_backend": fake_backend,
            "green_audit": audit,
            "judge_prompt_sha256": judge_prompt_fingerprint(),
            "judge_instrument": instrument,
            "validation_contract": VALIDATION_CONTRACT,
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
    include_green_audit: bool = False,
) -> dict[str, Any]:
    with Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store:
        documents = store.list_documents(batch_id)
        escalation = (config.public_root / "batches" / batch_id / "green_escalation.json").exists()
        # A judge fixed in advance (no pilot) must judge every document the expert
        # reviews, the GREEN audit sample included, or its errors there go unmeasured.
        audit = (
            set(
                ensure_green_audit_plan(config=config, store=store, batch_id=batch_id)[
                    "sample_internal_doc_ids"
                ]
            )
            if include_green_audit
            else set()
        )
        targets = [
            document
            for document in documents
            # QUARANTINE too: the expert reviews it (hitl._review_requirements), and
            # the judge judges every document the expert reviews.
            if document["router_bucket"] in {"RED", "YELLOW", "QUARANTINE"}
            or (escalation and document["router_bucket"] == "GREEN")
            or document["internal_doc_id"] in audit
        ]
        targets.sort(key=lambda row: row["internal_doc_id"])
        counts = {"valid": 0, "unavailable": 0, "error": 0}
        lease.beat(stage="judge-production", expected_units=len(targets), completed_units=0)
        for index, document in enumerate(targets, 1):
            internal_doc_id = document["internal_doc_id"]
            prediction = store.get_prediction(batch_id, internal_doc_id, generation)
            if prediction is None:
                raise IntegrityError(f"missing {generation} prediction for {internal_doc_id}")
            provenance = judged_prediction_provenance(prediction)
            existing = store.get_judge_result(batch_id, internal_doc_id, model)
            if existing is not None and existing["status"] == "valid":
                result_path = Path(existing["response_path"])
                if (
                    not result_path.is_file()
                    or sha256_file(result_path) != existing["response_sha256"]
                ):
                    raise IntegrityError(f"judge result file is missing or corrupt: {result_path}")
                # Resume must not hand back a verdict formed against another
                # generation's output. The store's key cannot tell them apart,
                # so the stored payload is asked directly; a verdict written
                # before provenance was recorded names nothing and is refused
                # rather than silently trusted.
                stored = json.loads(result_path.read_text(encoding="utf-8"))
                if not _instrument_matches(stored):
                    raise IntegrityError(
                        f"existing judge verdict for {internal_doc_id} was formed by a "
                        f"different judge instrument than the locked one; re-judge or clear it"
                    )
                if stored.get("judged_prediction") != provenance:
                    raise IntegrityError(
                        f"existing judge verdict for {internal_doc_id} was formed against "
                        f"{stored.get('judged_prediction', {}).get('generation', 'an unrecorded')}"
                        f" prediction, not {generation}; re-judge or clear it"
                    )
                counts["valid"] += 1
                lease.beat(completed_units=index, last_successful_unit=internal_doc_id)
                continue
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
                "schema_version": 3,
                "batch_id": batch_id,
                "internal_doc_id": internal_doc_id,
                "model": model,
                "stage": "production",
                "blind_mapping": mapping,
                "judge_instrument": judge_instrument(),
                "validation_contract": VALIDATION_CONTRACT,
                "status": status,
                "judged_prediction": judged_prediction_provenance(prediction),
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
                include_green_audit=lock_payload.get("pilot", True) is False,
            )
        elif _loop_mode(config):
            raise GateBlocked(
                "the GT750 loop has no judge pilot; `loop-judge --round N` fixes Gemma 4 "
                "and runs it on every document the expert reviews"
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


def _resolve_verdict(verdict: str | None, blind_mapping: str) -> str:
    """Translate a blind A/B verdict into the real side it favoured.

    ``blind_mapping`` values look like ``"A=human,B=model"`` or
    ``"A=model,B=human"``; which side is the annotator and which is the
    fine-tuned model is randomised per document, so a raw "A" or "B" is
    meaningless on its own -- it must be resolved through the mapping before
    it is counted. TIE and NEITHER need no translation.
    """
    if verdict in (None, "TIE", "NEITHER"):
        return verdict or "NEITHER"
    sides = dict(pair.split("=", 1) for pair in blind_mapping.split(","))
    side = sides.get(verdict)
    if side == "human":
        return "annotator"
    if side == "model":
        return "model"
    return "NEITHER"


def _error_class(error: str) -> str:
    """Collapse a free-form attempt error message to a stable class label.

    Rejection messages either read as ``"<class> at <detail>"`` (the
    fabricated/missing-evidence contract check) or ``"<class>: <detail>"`` (a
    JSON decode failure surfacing its own message); any other contract
    message carries no detail suffix and is its own class.
    """
    for separator in (" at ", ":"):
        if separator in error:
            return error.split(separator, 1)[0].strip()
    return error.strip()


def judge_expert_metrics(
    store: Store, *, batch_id: str, model: str, ids: list[str]
) -> dict[str, Any]:
    """Judge-quality numbers, including a fabrication signal sourced two ways.

    `fabricated_evidence_attempts`/`fabricated_evidence_rate` count attempts
    whose error message was the evidence-rejection class -- that signal only
    exists for an attempt that failed outright. Since `_validate_judge_result`
    now quarantines a single unquotable span into `unverified_evidence`
    instead of failing the whole attempt, a verdict that used to fail on one
    bad span can now succeed on its first try and generate no failed attempt
    at all -- which would silently zero out that counter for exactly the
    verdicts it exists to flag. `unverified_evidence_spans` and
    `unverified_evidence_span_rate` re-source the same signal from the
    `unverified_evidence` field carried on valid results, so a verdict that
    stops failing does not stop being counted. Both signals are kept: the
    attempt-based counters still reflect genuinely-failed attempts (bad JSON,
    a bad verdict token, and so on), and the span-based counters reflect
    spans a valid verdict could not have quoted.
    """
    if not ids:
        return {"status": "complete", "document_count": 0}
    exact = 0
    tp = fp = fn = 0
    latencies: list[float] = []
    costs: list[float] = []
    total_attempts = 0
    fabricated_evidence_attempts = 0
    invalid_attempts = 0
    invalid_attempts_by_error_class: dict[str, int] = defaultdict(int)
    documents_needing_retry = 0
    unverified_evidence_spans = 0
    evidence_spans_seen = 0
    verdict_distribution = {"annotator": 0, "model": 0, "TIE": 0, "NEITHER": 0}
    for internal_doc_id in ids:
        review = store.get_review(batch_id, internal_doc_id)
        result = store.get_judge_result(batch_id, internal_doc_id, model)
        if review is None or review["status"] != "finalized" or result is None:
            return {"status": "pending"}
        if result["status"] != "valid":
            return {"status": result["status"], "document_count": len(ids)}
        expert = json.loads(review["final_references_json"] or "[]")
        result_data = json.loads(result["result_json"])
        judge = result_data.get("final_references", [])
        # Re-sourced fabrication signal: spans this valid verdict could not
        # be verified against the document, counted here so the metric does
        # not depend on the verdict having failed an attempt to be seen.
        verified_spans = result_data.get("evidence") or []
        unverified_spans = result_data.get("unverified_evidence") or []
        unverified_evidence_spans += len(unverified_spans)
        evidence_spans_seen += len(verified_spans) + len(unverified_spans)
        # The project's one comparison (2 Oct 2026), 213/413 removed from both lists.
        comparison = compare_with_policy(expert, judge)
        exact += comparison.agree
        tp += comparison.tp
        fp += comparison.fp
        fn += comparison.fn
        if result["retry_count"]:
            documents_needing_retry += 1
        verdict_distribution[_resolve_verdict(result["verdict"], result["blind_mapping"])] += 1
        if result["response_path"]:
            payload = json.loads(Path(result["response_path"]).read_text(encoding="utf-8"))
            operational = payload.get("operational") or {}
            latency = operational.get("latency_seconds")
            cost = operational.get("cost")
            if isinstance(latency, (int, float)):
                latencies.append(float(latency))
            if isinstance(cost, (int, float)):
                costs.append(float(cost))
            for attempt in payload.get("attempts") or []:
                total_attempts += 1
                if attempt.get("status") != "invalid":
                    # "valid" needs no counting here; "unavailable" is a
                    # provider outage, not a rejected judge output, so it
                    # does not belong in the invalid/fabrication breakdown.
                    continue
                invalid_attempts += 1
                error_class = _error_class(str(attempt.get("error") or ""))
                invalid_attempts_by_error_class[error_class] += 1
                if error_class == "fabricated_or_missing_evidence":
                    fabricated_evidence_attempts += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    sorted_latencies = sorted(latencies)
    return {
        "status": "complete",
        "document_count": len(ids),
        "expert_exact_set_agreement": exact / len(ids),
        "expert_core_precision": precision,
        "expert_core_recall": recall,
        "expert_core_f1": f1,
        "fabricated_evidence_attempts": fabricated_evidence_attempts,
        "total_attempts": total_attempts,
        "fabricated_evidence_rate": (
            fabricated_evidence_attempts / total_attempts if total_attempts else 0.0
        ),
        "invalid_attempts": invalid_attempts,
        "invalid_attempts_by_error_class": dict(invalid_attempts_by_error_class),
        "unverified_evidence_spans": unverified_evidence_spans,
        "evidence_spans_seen": evidence_spans_seen,
        "unverified_evidence_span_rate": (
            unverified_evidence_spans / evidence_spans_seen if evidence_spans_seen else 0.0
        ),
        "documents_needing_retry": documents_needing_retry,
        "verdict_distribution": verdict_distribution,
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "latency_p50_seconds": _percentile(sorted_latencies, 50) if sorted_latencies else None,
        "latency_p95_seconds": _percentile(sorted_latencies, 95) if sorted_latencies else None,
        "reported_cost": sum(costs) if costs else None,
    }


def _loop_mode(config: AppConfig) -> bool:
    from .loop_sources import has_loop_settings

    return has_loop_settings(config)


def judge_lock_payload(config: AppConfig, batch_id: str) -> dict[str, Any] | None:
    path = config.public_root / "batches" / batch_id / "judge_lock.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def judge_visible_to_expert(config: AppConfig, batch_id: str) -> bool:
    """Whether the expert may see the judge's verdict: only after a pilot lock.

    The GT750 loop's judge is fixed in advance and never shown (`fix_loop_judge`);
    with no lock at all the judge is hidden too.
    """
    payload = judge_lock_payload(config, batch_id)
    return bool(payload) and payload.get("expert_sees_judge", True) is not False


LOOP_JUDGE_REASON = (
    "GT750 loop: the judge is fixed in advance (Gemma 4, local) -- no pilot, never "
    "shown to the expert; its errors are measured against the expert's final lists "
    "(user decision, 29 Sep 2026)"
)


def fix_loop_judge(
    *, config: AppConfig, batch_id: str, model: str = JUDGE_MODEL_KEY
) -> dict[str, Any]:
    """Fix the GT750 loop's judge for `batch_id`: Gemma 4, no pilot, hidden from the expert.

    User decision (29 Sep 2026): the pilot exam and auditing the judge were load on
    the expert, while the judge's errors are measured anyway -- every verdict is
    scored against the expert's final list (`loop-errors`). Hidden on every
    document, the judge's measurement stays blind everywhere, not only on a
    pilot. Written where the pilot's lock was (`judge_lock.json`) so coverage
    and release read one judge; recorded as `judge_fixed`, never `judge_locked`,
    since no expert decision is ever made after seeing it.
    """
    if not _loop_mode(config):
        raise ContractError("only the GT750 loop fixes its judge in advance")
    if model != JUDGE_MODEL_KEY or judge_model_providers().get(model) != "mlx":
        raise ContractError(
            f"the loop's judge is Gemma 4, run locally ({JUDGE_MODEL_KEY}); got {model!r}"
        )
    with Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store:
        batch = store.get_batch(batch_id)
        if batch is None or batch["status"] != "processed":
            raise GateBlocked(f"batch {batch_id} is not processed")
        core = {
            "schema_version": 1,
            "batch_id": batch_id,
            "model": model,
            "reason": LOOP_JUDGE_REASON,
            "pilot": False,
            "expert_sees_judge": False,
            "instrument": judge_instrument(),
        }
        payload = {**core, "lock_fingerprint": fingerprint_json(core)}
        path = config.public_root / "batches" / batch_id / "judge_lock.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != payload:
                raise GateBlocked(f"batch {batch_id} already has a different judge: {path}")
            return existing
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, payload, mode=0o644)
        store.add_event(batch_id, "judge_fixed", payload)
        return payload


def lock_judge(*, config: AppConfig, batch_id: str, model: str, reason: str) -> dict[str, Any]:
    if _loop_mode(config):
        raise GateBlocked(
            "the GT750 loop has no judge pilot to lock from; `loop-judge --round N` fixes "
            "Gemma 4 and runs it"
        )
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
