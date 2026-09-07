"""Per-document G0 inference, atomic persistence, resume, and routing."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .atomic import write_json_atomic
from .config import AppConfig
from .constants import MODEL_ID, MODEL_REVISION
from .contracts import validate_reference_list
from .errors import ContractError, FingerprintMismatch, GateBlocked, IntegrityError
from .fingerprints import fingerprint_json, sha256_file, sha256_text
from .g0 import SYSTEM_PROMPT
from .g0_finalize import ROUND_LABEL_PATTERN, round_registry_path
from .heartbeat import RunLease
from .performance import optional_float, optional_int
from .preparation import validate_ready
from .reference_policy import (
    DEFAULT_REFERENCE_POLICY_ID,
    reference_policy_fingerprint,
)
from .router import RouteDecision, route_document
from .storage import Store


@dataclass(frozen=True)
class PredictionResult:
    status: str
    references: list[dict[str, str]]
    raw_output: str
    operational: dict[str, Any]
    error: str | None = None


class PredictionBackend(Protocol):
    backend_id: str

    @property
    def model_fingerprint(self) -> str: ...

    def predict(self, document: dict[str, Any]) -> PredictionResult: ...


class EchoHumanBackend:
    """Test-only backend; never selected without the hidden CLI test flag."""

    backend_id = "echo-human-fixture-v1"
    model_fingerprint = fingerprint_json({"backend": "echo-human-fixture-v1"})

    def predict(self, document: dict[str, Any]) -> PredictionResult:
        references = validate_reference_list(document["human_references"])
        return PredictionResult(
            status="success",
            references=references,
            raw_output=json.dumps(references, ensure_ascii=False, separators=(",", ":")),
            operational={
                "backend": "echo-human-fixture-v1",
                "input_tokens": None,
                "output_tokens": None,
                "latency_seconds": 0.0,
                "truncated": False,
            },
        )


LEGAL_TUPLE_FIELDS = ("kanun_no", "kanun_ad", "madde", "fikra", "bent")
# A pair of identical references proves nothing: the pipeline deduplicates, and
# a pair can still be followed by different references. A run of three has no
# legitimate reading -- d000021 emitted the same reference 259 times before the
# 8192-token ceiling cut it off after 166 seconds, and doubling the ceiling to
# 16384 produced 524 copies in 334 seconds with the same five unique
# references. Every token after the third copy buys nothing.
REPETITION_STOP_RUN_LENGTH = 3


def deduplicate_legal_tuples(references: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for reference in references:
        identity = tuple(str(reference.get(field, "")) for field in LEGAL_TUPLE_FIELDS)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(reference)
    return unique


def _last_complete_objects(text: str, count: int) -> list[str] | None:
    """Return the last `count` complete `{...}` objects, newest last.

    Scans backwards from the end so the cost does not grow with the output --
    this runs once per generated token. A trailing incomplete object is
    ignored, which is the normal state mid-stream.
    """
    objects: list[str] = []
    end = len(text)
    for _ in range(count):
        close = text.rfind("}", 0, end)
        if close < 0:
            return None
        opened = text.rfind("{", 0, close)
        if opened < 0:
            return None
        objects.append(text[opened : close + 1])
        end = opened
    return list(reversed(objects))


def terminal_duplicate_run_reached(
    text: str, *, run_length: int = REPETITION_STOP_RUN_LENGTH
) -> bool:
    """True when the output currently ends in a run of identical references.

    Greedy decoding at temperature 0 cannot escape such a run: each copy raises
    the probability of the next, so once P(another copy) exceeds P("]") the
    inequality only widens. Raising `max_tokens` therefore buys time, not
    termination, which is why generation is stopped here instead.
    """
    objects = _last_complete_objects(text, run_length)
    if objects is None:
        return False
    identities = []
    for chunk in objects:
        try:
            payload = json.loads(chunk)
        except json.JSONDecodeError:
            return False
        if not isinstance(payload, dict):
            return False
        identities.append(tuple(str(payload.get(field, "")) for field in LEGAL_TUPLE_FIELDS))
    return len(set(identities)) == 1


def salvage_repeated_json_prefix(raw_output: str) -> dict[str, Any] | None:
    """Close only a valid JSON prefix that demonstrably ends in a tuple loop.

    Lives here, next to the parser, so the deployment path and the training
    path apply the identical rule. They did not: the same model failing the
    same way was recovered during validation and quarantined during
    deployment, purely because the recovery existed in one code path only.

    The rule refuses an enumeration runaway on purpose. d000059 counted `bent`
    upward -- 201, 202, 203 -- so no two tuples are equal and there is nothing
    to deduplicate; closing that array would inject hundreds of fabricated
    references into the batch. Such a document stays quarantined.
    """
    array_start = raw_output.find("[")
    if array_start < 0:
        return None
    search_end = len(raw_output)
    while True:
        object_end = raw_output.rfind("}", array_start, search_end)
        if object_end < 0:
            return None
        candidate = raw_output[array_start : object_end + 1].rstrip().rstrip(",") + "]"
        try:
            references = _parse_model_output(candidate)
        except ContractError:
            search_end = object_end
            continue
        if len(references) < 2:
            return None
        identities = [
            tuple(str(reference.get(field, "")) for field in LEGAL_TUPLE_FIELDS)
            for reference in references
        ]
        terminal_identity = identities[-1]
        terminal_run_length = 0
        for identity in reversed(identities):
            if identity != terminal_identity:
                break
            terminal_run_length += 1
        unique = deduplicate_legal_tuples(references)
        # The second clause is provably unreachable given the first -- a
        # terminal run of two means the last two identities are equal, so
        # deduplication must shrink the list. It is kept because this is the
        # frozen rule the sealed G0 runs were produced with, and removing dead
        # code from a frozen rule buys nothing. A mutation that deletes it
        # therefore survives, correctly; it is not a gap in the tests.
        if terminal_run_length < 2 or len(unique) == len(references):
            return None
        return {
            "references": unique,
            "complete_reference_count": len(references),
            "unique_reference_count": len(unique),
            "duplicate_reference_count": len(references) - len(unique),
            "terminal_duplicate_run_length": terminal_run_length,
            "discarded_incomplete_suffix_chars": len(raw_output) - object_end - 1,
            "synthetic_closing_bracket": True,
            "repaired_prefix_sha256": sha256_text(candidate),
        }


def _parse_model_output(raw_output: str) -> list[dict[str, str]]:
    text = raw_output.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end < start:
            raise ContractError("model output contains no JSON array")
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ContractError(f"model output JSON parse failed: {exc}") from exc
    if isinstance(payload, dict) and isinstance(payload.get("references"), list):
        payload = payload["references"]
    return validate_reference_list(payload)


class MlxG0Backend:
    backend_id = "mlx-g0"

    def __init__(self, config: AppConfig, *, registry_path: Path | None = None) -> None:
        registry_path = (
            config.public_root / "g0" / "G0.json"
            if registry_path is None
            else registry_path.resolve()
        )
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GateBlocked(f"no sealed G0 registry at {registry_path}: {exc}") from exc
        if registry.get("model_id") != MODEL_ID or registry.get("model_revision") != MODEL_REVISION:
            raise FingerprintMismatch("sealed G0 model id/revision differs from the v1 contract")
        adapter_path = Path(str(registry.get("adapter_path", ""))).resolve()
        model_path = Path(str(registry.get("model_snapshot_path", ""))).resolve()
        if not adapter_path.exists() or not model_path.is_dir():
            raise GateBlocked("sealed G0 model snapshot or adapter is unavailable")
        adapter_file = (
            adapter_path / "adapters.safetensors" if adapter_path.is_dir() else adapter_path
        )
        if not adapter_file.is_file() or sha256_file(adapter_file) != registry.get(
            "adapter_sha256"
        ):
            raise IntegrityError("sealed G0 adapter checksum mismatch")
        self.max_input_tokens = int(registry["max_sequence_length"])
        self.max_generation_tokens = int(registry.get("max_generation_tokens", 4096))
        self._model_fingerprint = fingerprint_json(
            {
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "adapter_sha256": registry["adapter_sha256"],
                "max_sequence_length": self.max_input_tokens,
            }
        )
        from mlx_lm import load

        self.model, self.tokenizer = load(str(model_path), adapter_path=str(adapter_path))

    @property
    def model_fingerprint(self) -> str:
        return self._model_fingerprint

    def predict(self, document: dict[str, Any]) -> PredictionResult:
        from mlx_lm import stream_generate

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": document["text"]},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(prompt, str):
            raise ContractError("tokenizer chat template did not return a string")
        input_tokens = len(self.tokenizer.encode(prompt))
        if input_tokens > self.max_input_tokens:
            return PredictionResult(
                status="error",
                references=[],
                raw_output="",
                error=(
                    f"input_token_count={input_tokens} exceeds pinned "
                    f"max_sequence_length={self.max_input_tokens}; not truncated"
                ),
                operational={
                    "input_tokens": input_tokens,
                    "output_tokens": 0,
                    "latency_seconds": 0.0,
                    "truncated": False,
                    "generation_attempted": False,
                },
            )
        started = time.perf_counter()
        pieces: list[str] = []
        output_tokens = 0
        finish_reason = ""
        # Performance counters reported next to accuracy; see
        # `docs/agents/local-inference-metrics.md`. The first yield gives
        # time-to-first-token; the final response carries mlx_lm's prefill and
        # decode throughput and peak memory, which used to be discarded.
        ttft_seconds: float | None = None
        prompt_tps: float | None = None
        generation_tps: float | None = None
        peak_memory_bytes: int | None = None
        repetition_stopped = False
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_generation_tokens,
        ):
            if ttft_seconds is None:
                ttft_seconds = time.perf_counter() - started
            pieces.append(response.text)
            output_tokens = int(response.generation_tokens)
            finish_reason = str(response.finish_reason or finish_reason)
            # Greedy decoding at temperature 0 cannot leave a tuple loop, so
            # generation is stopped the moment one is provable rather than paid
            # for to the ceiling. Checked only when an object just closed, and
            # the check scans backwards from the end, so it costs nothing per
            # token. See REPETITION_STOP_RUN_LENGTH for why three, not two.
            if "}" in response.text and terminal_duplicate_run_reached("".join(pieces)):
                repetition_stopped = True
                finish_reason = "repetition_stop"
                break
            prompt_tps = optional_float(getattr(response, "prompt_tps", None), prompt_tps)
            generation_tps = optional_float(
                getattr(response, "generation_tps", None), generation_tps
            )
            peak_memory_bytes = optional_int(
                getattr(response, "peak_memory", None), peak_memory_bytes
            )
        raw = "".join(pieces)
        model_runaway = finish_reason == "length" or repetition_stopped
        recovery: dict[str, Any] | None = None
        try:
            references = _parse_model_output(raw)
            status, error = (
                ("error", "model output reached generation limit")
                if finish_reason == "length"
                else ("success", None)
            )
        except ContractError as exc:
            references = []
            status = "error"
            error = (
                f"model output reached generation limit; {exc}"
                if finish_reason == "length"
                else str(exc)
            )
            # A runaway that ends in a proven tuple loop carries a usable
            # prefix: the copies are dropped and the array is closed. An
            # enumeration runaway is refused by the same rule and stays an
            # error, because closing it would invent references.
            if model_runaway:
                recovery = salvage_repeated_json_prefix(raw)
                if recovery is not None:
                    references = recovery["references"]
                    status, error = "success", None
        # A recovered document is not truncated: a complete reference set was
        # obtained from it. One that stayed an error is, and routes to
        # quarantine as before.
        truncated = bool(model_runaway and recovery is None and status == "error")
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass
        return PredictionResult(
            status=status,
            references=references,
            raw_output=raw,
            error=error,
            operational={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_seconds": time.perf_counter() - started,
                "truncated": truncated,
                "finish_reason": finish_reason,
                "repetition_stopped": repetition_stopped,
                "repetition_recovery": (
                    None
                    if recovery is None
                    else {key: value for key, value in recovery.items() if key != "references"}
                ),
                "generation_attempted": True,
                "ttft_seconds": ttft_seconds,
                "prompt_tps": prompt_tps,
                "generation_tps": generation_tps,
                "peak_memory_bytes": peak_memory_bytes,
            },
        )


def _document_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "human_references": json.loads(row["human_references_json"]),
        "metadata": json.loads(row["metadata_json"]),
        "warnings": json.loads(row["warnings_json"]),
    }


def _route(document: dict[str, Any], result: PredictionResult) -> RouteDecision:
    return route_document(
        human_references=document["human_references"],
        model_references=result.references,
        preparation_status=document["preparation_status"],
        model_status=result.status,
        model_truncated=bool(result.operational.get("truncated")),
        has_safe_text=bool(document["text"]),
    )


def _result_payload(
    *,
    batch_id: str,
    generation: str,
    document: dict[str, Any],
    result: PredictionResult,
    route: RouteDecision,
    input_fingerprint: str,
    model_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "batch_id": batch_id,
        "internal_doc_id": document["internal_doc_id"],
        "public_doc_id": document["public_doc_id"],
        "generation": generation,
        "status": result.status,
        "references": result.references,
        "raw_output": result.raw_output,
        "error": result.error,
        "operational": result.operational,
        "route": route.to_dict(),
        "input_fingerprint": input_fingerprint,
        "model_fingerprint": model_fingerprint,
        "reference_policy_id": DEFAULT_REFERENCE_POLICY_ID,
        "reference_policy_fingerprint": reference_policy_fingerprint(),
    }


def _validate_result_file(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "batch_id",
        "internal_doc_id",
        "generation",
        "status",
        "references",
        "operational",
        "route",
        "input_fingerprint",
        "model_fingerprint",
    }
    if not isinstance(payload, dict) or not required <= payload.keys():
        raise IntegrityError(f"prediction result schema failure: {path}")
    validate_reference_list(payload["references"])


def _prediction_from_payload(payload: dict[str, Any]) -> PredictionResult:
    return PredictionResult(
        status=str(payload["status"]),
        references=validate_reference_list(payload["references"]),
        raw_output=str(payload.get("raw_output") or ""),
        error=None if payload.get("error") is None else str(payload["error"]),
        operational=dict(payload.get("operational") or {}),
    )


def process_batch(
    *,
    config: AppConfig,
    batch_id: str,
    generation: str,
    resume: bool,
    fake_backend: bool = False,
    backend: PredictionBackend | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(generation, str) or (
        generation != "G0" and not ROUND_LABEL_PATTERN.match(generation)
    ):
        raise ContractError(
            f"generation must be 'G0' or a round label like 'M003', got {generation!r}"
        )
    if registry_path is None and generation != "G0":
        # A round's predictions must come from that round's own sealed model,
        # never silently from G0's default. See round_registry_path.
        registry_path = round_registry_path(config, generation)
    ready = validate_ready(config, batch_id)
    selected_backend: PredictionBackend = (
        backend
        if backend is not None
        else (
            EchoHumanBackend()
            if fake_backend
            else MlxG0Backend(config, registry_path=registry_path)
        )
    )
    model_fingerprint = selected_backend.model_fingerprint
    sensitive_dir = config.sensitive_root / "batches" / batch_id
    output_dir = sensitive_dir / "predictions" / generation
    lease = RunLease(
        lock_path=config.sensitive_root / "locks" / f"process_{batch_id}_{generation}.lock",
        heartbeat_path=sensitive_dir / f"process_{generation}_heartbeat.json",
        purpose="process",
        run_id=f"process:{batch_id}:{generation}",
        input_fingerprint=str(ready["input_fingerprint"]),
        config_fingerprint=config.fingerprint,
    ).start(stage="preflight")
    bucket_counts: dict[str, int] = {
        bucket: 0 for bucket in ("GREEN", "YELLOW", "RED", "QUARANTINE")
    }
    try:
        with Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store:
            batch = store.get_batch(batch_id)
            if batch is None or not batch["ready"]:
                raise GateBlocked(f"batch {batch_id} is not READY in SQLite")
            if batch["input_fingerprint"] != ready["input_fingerprint"]:
                raise FingerprintMismatch("READY and SQLite input fingerprints differ")
            rows = store.list_documents(batch_id)
            if len(rows) != int(ready["document_count"]):
                raise IntegrityError(
                    f"document coverage drift: SQLite={len(rows)} READY={ready['document_count']}"
                )
            lease.beat(stage="prediction", expected_units=len(rows))

            for index, row in enumerate(rows, 1):
                document = _document_payload(row)
                if sha256_text(document["text"]) != document["text_sha256"]:
                    raise IntegrityError(
                        f"prepared text checksum mismatch: {row['internal_doc_id']}"
                    )
                input_fingerprint = fingerprint_json(
                    {
                        "batch_input_fingerprint": ready["input_fingerprint"],
                        "internal_doc_id": row["internal_doc_id"],
                        "text_sha256": row["text_sha256"],
                        "human_references": document["human_references"],
                        "reference_policy_id": DEFAULT_REFERENCE_POLICY_ID,
                        "reference_policy_fingerprint": reference_policy_fingerprint(),
                    }
                )
                target = output_dir / f"{row['internal_doc_id']}.json"
                existing = store.get_prediction(batch_id, row["internal_doc_id"], generation)
                if existing is not None:
                    if not resume:
                        raise GateBlocked(
                            f"prediction already exists for {row['internal_doc_id']}; pass --resume"
                        )
                    existing_path = Path(existing["response_path"])
                    if (
                        not existing_path.is_file()
                        or sha256_file(existing_path) != existing["response_sha256"]
                    ):
                        raise IntegrityError(
                            f"completed prediction file is missing or corrupt: {existing_path}"
                        )
                    payload = json.loads(existing_path.read_text(encoding="utf-8"))
                    if (
                        payload.get("input_fingerprint") != input_fingerprint
                        or payload.get("model_fingerprint") != model_fingerprint
                    ):
                        raise FingerprintMismatch(
                            f"prediction resume fingerprint drift: {row['internal_doc_id']}"
                        )
                    result = _prediction_from_payload(payload)
                    decision = _route(document, result)
                    store.set_router_bucket(batch_id, row["internal_doc_id"], decision.bucket)
                    bucket_counts[decision.bucket] += 1
                    lease.beat(
                        completed_units=index,
                        last_successful_unit=row["internal_doc_id"],
                    )
                    continue

                if target.exists():
                    if not resume:
                        raise GateBlocked(f"orphan prediction exists at {target}; pass --resume")
                    _validate_result_file(target)
                    payload = json.loads(target.read_text(encoding="utf-8"))
                    if (
                        payload.get("input_fingerprint") != input_fingerprint
                        or payload.get("model_fingerprint") != model_fingerprint
                    ):
                        raise FingerprintMismatch(
                            f"orphan prediction fingerprint drift: {row['internal_doc_id']}"
                        )
                    result = _prediction_from_payload(payload)
                    decision = _route(document, result)
                elif document["preparation_status"] != "ready":
                    result = PredictionResult(
                        status="skipped_preparation_quarantine",
                        references=[],
                        raw_output="",
                        error="preparation hard error",
                        operational={
                            "generation_attempted": False,
                            "truncated": False,
                            "latency_seconds": 0.0,
                        },
                    )
                    decision = _route(document, result)
                    payload = _result_payload(
                        batch_id=batch_id,
                        generation=generation,
                        document=document,
                        result=result,
                        route=decision,
                        input_fingerprint=input_fingerprint,
                        model_fingerprint=model_fingerprint,
                    )
                    write_json_atomic(target, payload, validator=_validate_result_file)
                else:
                    result = selected_backend.predict(document)
                    result = PredictionResult(
                        status=result.status,
                        references=validate_reference_list(result.references),
                        raw_output=result.raw_output,
                        operational=result.operational,
                        error=result.error,
                    )
                    decision = _route(document, result)
                    payload = _result_payload(
                        batch_id=batch_id,
                        generation=generation,
                        document=document,
                        result=result,
                        route=decision,
                        input_fingerprint=input_fingerprint,
                        model_fingerprint=model_fingerprint,
                    )
                    write_json_atomic(target, payload, validator=_validate_result_file)

                response_sha256 = sha256_file(target)
                store.persist_prediction(
                    batch_id=batch_id,
                    internal_doc_id=row["internal_doc_id"],
                    generation=generation,
                    status=result.status,
                    references=result.references,
                    response_path=target,
                    response_sha256=response_sha256,
                    input_fingerprint=input_fingerprint,
                    model_fingerprint=model_fingerprint,
                    error=result.error,
                    operational=result.operational,
                )
                store.set_router_bucket(batch_id, row["internal_doc_id"], decision.bucket)
                bucket_counts[decision.bucket] += 1
                lease.beat(
                    completed_units=index,
                    last_successful_unit=row["internal_doc_id"],
                    stage="prediction",
                )

            predictions = store.list_predictions(batch_id, generation)
            if len(predictions) != len(rows):
                raise IntegrityError(
                    f"prediction coverage incomplete: {len(predictions)}/{len(rows)}"
                )
            current = store.get_batch(batch_id)
            assert current is not None
            store.update_batch(
                batch_id,
                expected_version=int(current["row_version"]),
                status="processed",
                ready=True,
            )
            summary = {
                "schema_version": 1,
                "batch_id": batch_id,
                "generation": generation,
                "model_fingerprint": model_fingerprint,
                "reference_policy_id": DEFAULT_REFERENCE_POLICY_ID,
                "reference_policy_fingerprint": reference_policy_fingerprint(),
                "expected_document_count": len(rows),
                "prediction_count": len(predictions),
                "router_counts": bucket_counts,
                "resume": resume,
                "completed": True,
            }
            write_json_atomic(
                config.public_root / "batches" / batch_id / f"process_{generation}_summary.json",
                summary,
                mode=0o644,
            )
            store.add_event(batch_id, "processing_completed", summary)
        lease.finish(status="completed")
        return summary
    except BaseException as exc:
        lease.finish(status="failed", error=str(exc))
        raise
