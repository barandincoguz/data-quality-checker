"""Isolated heavyweight MLX actions used by the compute preflight.

Every action runs in a child process so an allocator failure cannot corrupt the
parent's lifecycle state. Requests and results are JSON files, making failures
inspectable and resumable after terminal or process loss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .errors import ContractError, FingerprintMismatch, IntegrityError
from .fingerprints import fingerprint_json, sha256_file


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_projection(checkpoint: Path) -> dict[str, Any]:
    import mlx.core as mx
    import numpy as np

    def tensor_fingerprint(path: Path) -> str:
        rows: list[dict[str, Any]] = []
        for name, value in sorted(mx.load(str(path)).items()):
            array = np.array(value)
            rows.append(
                {
                    "name": name,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                }
            )
        return fingerprint_json(rows)

    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    return {
        "adapter_sha256": sha256_file(checkpoint / "adapters.safetensors"),
        "optimizer_sha256": sha256_file(checkpoint / "optimizer.safetensors"),
        "mlx_rng_sha256": sha256_file(checkpoint / "mlx_rng.safetensors"),
        "adapter_tensor_fingerprint": tensor_fingerprint(
            checkpoint / "adapters.safetensors"
        ),
        "optimizer_tensor_fingerprint": tensor_fingerprint(
            checkpoint / "optimizer.safetensors"
        ),
        "mlx_rng_tensor_fingerprint": tensor_fingerprint(
            checkpoint / "mlx_rng.safetensors"
        ),
        "scheduler_state": trainer_state["scheduler_state"],
        "data_cursor": trainer_state["data_cursor"],
        "python_rng_state_fingerprint": fingerprint_json(
            trainer_state["python_rng_state"]
        ),
    }


def _download(request: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id=str(request["model_id"]),
        revision=str(request["revision"]),
        cache_dir=str(request["cache_dir"]),
    )
    return {"snapshot_path": str(Path(snapshot).resolve())}


def _train(request: dict[str, Any]) -> dict[str, Any]:
    import mlx.core as mx

    from .mlx_stateful import StatefulMlxTrainer, StatefulTrainingConfig

    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    mx.reset_peak_memory()
    config = StatefulTrainingConfig(**request["training_config"])
    trainer = StatefulMlxTrainer(
        model_path=Path(request["model_path"]),
        train_path=Path(request["train_path"]),
        checkpoint_root=Path(request["checkpoint_root"]),
        config=config,
        input_fingerprint=str(request["input_fingerprint"]),
        model_fingerprint=str(request["model_fingerprint"]),
    )
    if request.get("resume_checkpoint"):
        trainer.resume(Path(request["resume_checkpoint"]))
    trained = trainer.train(target_updates=int(request["target_updates"]))
    checkpoint = Path(trained["checkpoint"])
    if not trainer.adapter_finite(checkpoint):
        raise RuntimeError("adapter contains non-finite tensors")
    return {
        **trained,
        "adapter_finite": True,
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "state_projection": _checkpoint_projection(checkpoint),
    }


def _generate_one(
    *,
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_tokens: int,
    max_input_tokens: int | None = None,
) -> dict[str, Any]:
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    from .processing import _parse_model_output

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(prompt, str):
        raise RuntimeError("chat template did not render a string prompt")
    input_tokens = len(tokenizer.encode(prompt))
    if max_input_tokens is not None and input_tokens > max_input_tokens:
        return {
            "raw_output": "",
            "parsed_references": [],
            "parse_ok": False,
            "parse_error": (
                f"input_tokens={input_tokens} exceed max_sequence_length="
                f"{max_input_tokens}; truncation forbidden"
            ),
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "finish_reason": "input_too_long",
        }
    pieces: list[str] = []
    output_tokens = 0
    finish_reason = ""
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=0.0),
    ):
        pieces.append(response.text)
        output_tokens = int(response.generation_tokens)
        finish_reason = str(response.finish_reason or finish_reason)
    raw = "".join(pieces).strip()
    parse_error: str | None = None
    try:
        parsed = _parse_model_output(raw)
    except ContractError as exc:
        parsed = []
        parse_error = str(exc)
    return {
        "raw_output": raw,
        "parsed_references": parsed,
        "parse_ok": parse_error is None,
        "parse_error": parse_error,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "finish_reason": finish_reason,
    }


def _generate(request: dict[str, Any]) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import load

    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    mx.reset_peak_memory()
    model, tokenizer = load(
        str(request["model_path"]), adapter_path=str(request["adapter_path"])
    )
    generated = _generate_one(
        model=model,
        tokenizer=tokenizer,
        messages=request["messages"],
        max_tokens=int(request.get("max_tokens", 512)),
    )
    return {
        "adapter_loaded": True,
        **generated,
        "peak_memory_bytes": int(mx.get_peak_memory()),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise IntegrityError(f"invalid validation JSONL {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise IntegrityError(f"validation row is not an object at {path}:{line_number}")
        rows.append(row)
    return rows


def _cached_validation_record(
    path: Path, *, request_fingerprint: str, doc_id: int
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"validation record cannot be parsed at {path}: {exc}") from exc
    if payload.get("request_fingerprint") != request_fingerprint:
        raise FingerprintMismatch(f"validation record request mismatch at {path}")
    if payload.get("doc_id") != doc_id:
        raise IntegrityError(f"validation record doc_id mismatch at {path}")
    return payload


def _validate(request: dict[str, Any]) -> dict[str, Any]:
    """Generate a resumable validation split and compute teacher-forced loss."""

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.tuner.trainer import evaluate

    from .mlx_stateful import NoThinkChatDataset

    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    mx.reset_peak_memory()

    data_path = Path(request["data_path"]).resolve()
    doc_ids_path = Path(request["doc_ids_path"]).resolve()
    adapter_path = Path(request["adapter_path"]).resolve()
    if sha256_file(data_path) != request.get("data_sha256"):
        raise FingerprintMismatch("validation data fingerprint mismatch")
    if sha256_file(doc_ids_path) != request.get("doc_ids_sha256"):
        raise FingerprintMismatch("validation doc-id fingerprint mismatch")
    if sha256_file(adapter_path / "adapters.safetensors") != request.get("adapter_sha256"):
        raise FingerprintMismatch("validation adapter fingerprint mismatch")

    rows = _read_jsonl(data_path)
    doc_ids = json.loads(doc_ids_path.read_text(encoding="utf-8"))
    expected_count = int(request["expected_count"])
    if (
        not isinstance(doc_ids, list)
        or any(not isinstance(doc_id, int) for doc_id in doc_ids)
        or len(doc_ids) != len(set(doc_ids))
        or len(rows) != len(doc_ids)
        or len(rows) != expected_count
    ):
        raise IntegrityError("validation rows/doc IDs do not match expected coverage")

    output_dir = Path(request["output_dir"]).resolve()
    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    request_fingerprint = str(request["request_fingerprint"])
    model, tokenizer = load(str(request["model_path"]), adapter_path=str(adapter_path))

    records: list[dict[str, Any]] = []
    for doc_id, row in zip(doc_ids, rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise IntegrityError(f"validation messages missing for doc_id={doc_id}")
        record_path = records_dir / f"doc_{doc_id}.json"
        cached = _cached_validation_record(
            record_path,
            request_fingerprint=request_fingerprint,
            doc_id=doc_id,
        )
        if cached is not None:
            records.append(cached)
            continue

        started = time.perf_counter()
        generated = _generate_one(
            model=model,
            tokenizer=tokenizer,
            messages=messages[:-1],
            max_tokens=int(request["max_generation_tokens"]),
            max_input_tokens=int(request["max_sequence_length"]),
        )
        parse_ok = bool(generated["parse_ok"])
        error = generated["parse_error"]
        runaway = generated.get("finish_reason") == "length"
        if runaway:
            error = "generation reached max_generation_tokens"
        record = {
            "schema_version": 1,
            "request_fingerprint": request_fingerprint,
            "doc_id": doc_id,
            "status": "success" if parse_ok and not runaway else "error",
            "parse_ok": parse_ok,
            "runaway": runaway,
            "references": generated["parsed_references"],
            "raw_output": generated["raw_output"],
            "error": error,
            "operational": {
                "input_tokens": generated["input_tokens"],
                "output_tokens": generated["output_tokens"],
                "finish_reason": generated["finish_reason"],
                "latency_seconds": time.perf_counter() - started,
                "truncated": False,
            },
        }
        write_json_atomic(record_path, record)
        records.append(record)
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    predictions = [
        {
            "doc_id": record["doc_id"],
            "status": record["status"],
            "references": record["references"],
            "error": record["error"],
            "operational": record["operational"],
        }
        for record in records
    ]
    predictions_path = output_dir / "predictions.json"
    write_json_atomic(predictions_path, predictions)

    dataset = NoThinkChatDataset(data_path, tokenizer)
    validation_loss = float(
        evaluate(
            model,
            dataset,
            batch_size=1,
            num_batches=-1,
            max_seq_length=int(request["max_sequence_length"]),
        )
    )
    record_files = [
        {
            "path": str((records_dir / f"doc_{doc_id}.json").relative_to(output_dir)),
            "size": (records_dir / f"doc_{doc_id}.json").stat().st_size,
            "sha256": sha256_file(records_dir / f"doc_{doc_id}.json"),
        }
        for doc_id in doc_ids
    ]
    manifest = {
        "schema_version": 1,
        "request_fingerprint": request_fingerprint,
        "coverage_count": len(records),
        "parse_count": sum(bool(record["parse_ok"]) for record in records),
        "success_count": sum(record["status"] == "success" for record in records),
        "empty_output_count": sum(not record["raw_output"].strip() for record in records),
        "zero_reference_output_count": sum(
            not record["references"] for record in records
        ),
        "parsed_zero_reference_output_count": sum(
            bool(record["parse_ok"]) and not record["references"]
            for record in records
        ),
        "predicted_reference_count": sum(
            len(record["references"]) for record in records
        ),
        "runaway_output_count": sum(bool(record["runaway"]) for record in records),
        "validation_loss": validation_loss,
        "predictions_path": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "record_files": record_files,
    }
    manifest_path = output_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    return {
        **{key: value for key, value in manifest.items() if key != "record_files"},
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "peak_memory_bytes": int(mx.get_peak_memory()),
    }


def _loss(request: dict[str, Any]) -> dict[str, Any]:
    """Run a no-truncation forward-only completion-loss smoke."""

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.tuner.trainer import evaluate

    from .mlx_stateful import NoThinkChatDataset

    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    mx.reset_peak_memory()
    data_path = Path(request["data_path"]).resolve()
    adapter_path = Path(request["adapter_path"]).resolve()
    if sha256_file(data_path) != request.get("data_sha256"):
        raise FingerprintMismatch("loss-smoke data fingerprint mismatch")
    if sha256_file(adapter_path / "adapters.safetensors") != request.get(
        "adapter_sha256"
    ):
        raise FingerprintMismatch("loss-smoke adapter fingerprint mismatch")
    model, tokenizer = load(str(request["model_path"]), adapter_path=str(adapter_path))
    dataset = NoThinkChatDataset(data_path, tokenizer)
    maximum_tokens = max(dataset.token_counts)
    max_sequence_length = int(request["max_sequence_length"])
    if maximum_tokens > max_sequence_length:
        raise IntegrityError(
            f"loss-smoke tokens={maximum_tokens} exceed max_sequence_length="
            f"{max_sequence_length}; truncation forbidden"
        )
    validation_loss = float(
        evaluate(
            model,
            dataset,
            batch_size=1,
            num_batches=-1,
            max_seq_length=max_sequence_length,
        )
    )
    if not math.isfinite(validation_loss):
        raise IntegrityError("loss-smoke result is not finite")
    return {
        "document_count": len(dataset),
        "maximum_tokens": maximum_tokens,
        "validation_loss": validation_loss,
        "peak_memory_bytes": int(mx.get_peak_memory()),
    }


def dispatch(request: dict[str, Any]) -> dict[str, Any]:
    action = request.get("action")
    if action == "download":
        return _download(request)
    if action == "train":
        return _train(request)
    if action == "generate":
        return _generate(request)
    if action == "validate":
        return _validate(request)
    if action == "loss":
        return _loss(request)
    raise ValueError(f"unsupported worker action: {action!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dqcheck-mlx-worker")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    base = {
        "schema_version": 1,
        "action": request.get("action"),
        "request_fingerprint": request.get("request_fingerprint"),
        "started_at": _utc_now(),
        "pid": os.getpid(),
        "host": socket.gethostname(),
    }
    write_json_atomic(args.result, {**base, "status": "running"})
    try:
        details = dispatch(request)
        result = {
            **base,
            "status": "passed",
            "finished_at": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            **details,
        }
        write_json_atomic(args.result, result)
        return 0
    except BaseException as exc:
        result = {
            **base,
            "status": "failed",
            "finished_at": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json_atomic(args.result, result)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
