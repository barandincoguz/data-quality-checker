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
from .heartbeat import RunLease
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
    @property
    def model_fingerprint(self) -> str: ...

    def predict(self, document: dict[str, Any]) -> PredictionResult: ...


class EchoHumanBackend:
    """Test-only backend; never selected without the hidden CLI test flag."""

    model_fingerprint = fingerprint_json({"backend": "echo-human-fixture-v1"})

    def predict(self, document: dict[str, Any]) -> PredictionResult:
        references = validate_reference_list(document["human_references"])
        return PredictionResult(
            status="success",
            references=references,
            raw_output=json.dumps(
                references, ensure_ascii=False, separators=(",", ":")
            ),
            operational={
                "backend": "echo-human-fixture-v1",
                "input_tokens": None,
                "output_tokens": None,
                "latency_seconds": 0.0,
                "truncated": False,
            },
        )


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
    def __init__(self, config: AppConfig, *, registry_path: Path | None = None) -> None:
        registry_path = (
            config.public_root / "g0" / "G0.json"
            if registry_path is None
            else registry_path.resolve()
        )
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GateBlocked(
                f"no sealed G0 registry at {registry_path}: {exc}"
            ) from exc
        if (
            registry.get("model_id") != MODEL_ID
            or registry.get("model_revision") != MODEL_REVISION
        ):
            raise FingerprintMismatch(
                "sealed G0 model id/revision differs from the v1 contract"
            )
        adapter_path = Path(str(registry.get("adapter_path", ""))).resolve()
        model_path = Path(str(registry.get("model_snapshot_path", ""))).resolve()
        if not adapter_path.exists() or not model_path.is_dir():
            raise GateBlocked("sealed G0 model snapshot or adapter is unavailable")
        adapter_file = (
            adapter_path / "adapters.safetensors"
            if adapter_path.is_dir()
            else adapter_path
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

        self.model, self.tokenizer = load(
            str(model_path), adapter_path=str(adapter_path)
        )

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
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_generation_tokens,
        ):
            pieces.append(response.text)
            output_tokens = int(response.generation_tokens)
            finish_reason = str(response.finish_reason or finish_reason)
        raw = "".join(pieces)
        truncated = finish_reason == "length"
        try:
            references = _parse_model_output(raw)
            status, error = (
                ("error", "model output reached generation limit")
                if truncated
                else ("success", None)
            )
        except ContractError as exc:
            references, status, error = [], "error", str(exc)
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
                "generation_attempted": True,
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
) -> dict[str, Any]:
    if generation != "G0":
        raise ContractError("only G0 processing is supported")
    ready = validate_ready(config, batch_id)
    selected_backend: PredictionBackend = (
        backend
        if backend is not None
        else (EchoHumanBackend() if fake_backend else MlxG0Backend(config))
    )
    model_fingerprint = selected_backend.model_fingerprint
    sensitive_dir = config.sensitive_root / "batches" / batch_id
    output_dir = sensitive_dir / "predictions" / generation
    lease = RunLease(
        lock_path=config.sensitive_root
        / "locks"
        / f"process_{batch_id}_{generation}.lock",
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
        with Store(
            config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms
        ) as store:
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
                existing = store.get_prediction(
                    batch_id, row["internal_doc_id"], generation
                )
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
                    store.set_router_bucket(
                        batch_id, row["internal_doc_id"], decision.bucket
                    )
                    bucket_counts[decision.bucket] += 1
                    lease.beat(
                        completed_units=index,
                        last_successful_unit=row["internal_doc_id"],
                    )
                    continue

                if target.exists():
                    if not resume:
                        raise GateBlocked(
                            f"orphan prediction exists at {target}; pass --resume"
                        )
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
                store.set_router_bucket(
                    batch_id, row["internal_doc_id"], decision.bucket
                )
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
                config.public_root
                / "batches"
                / batch_id
                / f"process_{generation}_summary.json",
                summary,
                mode=0o644,
            )
            store.add_event(batch_id, "processing_completed", summary)
        lease.finish(status="completed")
        return summary
    except BaseException as exc:
        lease.finish(status="failed", error=str(exc))
        raise
