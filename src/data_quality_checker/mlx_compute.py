"""Crash-resilient real-MLX acceptance gates for canonical-only G0.

This module deliberately imports tokenizer/model libraries only inside compute
functions. The default application and test suite remain usable without MLX.
"""

from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .config import AppConfig
from .context_windows import build_context_window_view
from .errors import ContractError, GateBlocked, IntegrityError
from .fingerprints import fingerprint_json, sha256_file
from .g0 import SEQUENCE_LENGTH_CANDIDATES
from .heartbeat import RunLease
from .locking import process_exists
from .mlx_stateful import NoThinkChatDataset, StatefulTrainingConfig


# Real probes on this run rejected 8192, 4096, and 2048 on Metal. Keep the
# accepted production gate explicit so implementation-only fingerprint changes
# do not repeat multi-minute OOM probes.
TRAINING_SEQUENCE_LENGTH_CANDIDATES = (1024,)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_snapshot_manifest(snapshot: Path, *, expected_revision: str) -> dict[str, Any]:
    snapshot = snapshot.resolve()
    if snapshot.name != expected_revision:
        raise IntegrityError(
            f"model snapshot revision mismatch: expected {expected_revision}, got {snapshot.name}"
        )
    if not snapshot.is_dir():
        raise IntegrityError(f"model snapshot directory is unavailable: {snapshot}")
    files: list[dict[str, Any]] = []
    for path in sorted(snapshot.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        files.append(
            {
                "path": path.relative_to(snapshot).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    required = {"config.json", "tokenizer_config.json"}
    observed = {row["path"] for row in files}
    missing = required - observed
    if missing:
        raise IntegrityError(
            "model snapshot is missing required files: " + ", ".join(sorted(missing))
        )
    if not any(path.endswith(".safetensors") for path in observed):
        raise IntegrityError("model snapshot contains no safetensors weights")
    return {
        "revision": expected_revision,
        "snapshot_path": str(snapshot),
        "files": files,
        "fingerprint": fingerprint_json(files),
    }


def parse_worker_result(path: Path, *, expected_action: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"worker result cannot be parsed at {path}: {exc}") from exc
    if payload.get("schema_version") != 1:
        raise IntegrityError(f"worker result schema mismatch at {path}")
    if payload.get("action") != expected_action:
        raise IntegrityError(
            f"worker action mismatch at {path}: expected {expected_action}, "
            f"got {payload.get('action')!r}"
        )
    if payload.get("status") != "passed":
        raise IntegrityError(
            f"worker {expected_action} failed at {path}: {payload.get('error', payload.get('status'))}"
        )
    return payload


def assert_equivalent_training_state(
    uninterrupted: dict[str, Any], resumed: dict[str, Any]
) -> None:
    required = (
        "adapter_tensor_fingerprint",
        "optimizer_tensor_fingerprint",
        "mlx_rng_tensor_fingerprint",
        "scheduler_state",
        "data_cursor",
        "python_rng_state_fingerprint",
    )
    for field in required:
        if uninterrupted.get(field) != resumed.get(field):
            raise IntegrityError(f"real MLX resume mismatch in {field}")


def assert_thinking_is_disabled(prompt: str) -> None:
    """Validate Qwen3.5's documented empty, closed no-thinking sentinel."""

    sentinel = "<think>\n\n</think>\n\n"
    if not prompt.endswith(sentinel):
        raise IntegrityError(
            "Qwen3.5 thinking-disable sentinel is missing, open, or non-empty"
        )


def _existing_passed_attempt(stage_dir: Path, request_fingerprint: str) -> dict[str, Any] | None:
    for attempt in sorted(stage_dir.glob("attempt_*")):
        result_path = attempt / "result.json"
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("request_fingerprint") != request_fingerprint:
            continue
        if payload.get("status") == "passed":
            return payload
        if (
            payload.get("status") == "running"
            and payload.get("host") == socket.gethostname()
            and isinstance(payload.get("pid"), int)
            and process_exists(int(payload["pid"]))
        ):
            raise GateBlocked(
                f"compute worker is already running for {stage_dir.name} "
                f"with pid={payload['pid']}"
            )
    return None


def _run_worker(
    *,
    compute_root: Path,
    stage: str,
    request: dict[str, Any],
    lease: RunLease,
    timeout_seconds: int = 7200,
) -> dict[str, Any]:
    request = dict(request)
    request_fingerprint = fingerprint_json(request)
    request["request_fingerprint"] = request_fingerprint
    stage_dir = compute_root / "workers" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    if cached := _existing_passed_attempt(stage_dir, request_fingerprint):
        return cached
    attempts = sorted(stage_dir.glob("attempt_*"))
    attempt_dir = stage_dir / f"attempt_{len(attempts) + 1:03d}"
    attempt_dir.mkdir(parents=False, exist_ok=False)
    request_path = attempt_dir / "request.json"
    result_path = attempt_dir / "result.json"
    log_temporary = attempt_dir / ".worker.log.tmp"
    log_path = attempt_dir / "worker.log"
    write_json_atomic(request_path, request)
    command = [
        sys.executable,
        "-m",
        "data_quality_checker.mlx_worker",
        "--request",
        str(request_path),
        "--result",
        str(result_path),
    ]
    started = time.monotonic()
    with log_temporary.open("wb") as log_handle:
        process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed > timeout_seconds:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise GateBlocked(
                    f"compute worker {stage} exceeded timeout={timeout_seconds}s"
                )
            lease.beat(
                stage=stage,
                child_pid=process.pid,
                eta_seconds=None,
                elapsed_seconds=elapsed,
            )
            time.sleep(5)
        log_handle.flush()
        os.fsync(log_handle.fileno())
    os.replace(log_temporary, log_path)
    if process.returncode != 0:
        try:
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-1000:].strip()
        except OSError:
            log_tail = ""
        detail = f"; log tail: {log_tail}" if log_tail else ""
        raise IntegrityError(
            f"compute worker {stage} exited {process.returncode}{detail}; see {log_path}"
        )
    result = parse_worker_result(result_path, expected_action=str(request["action"]))
    if result.get("request_fingerprint") != request_fingerprint:
        raise IntegrityError(f"compute worker request fingerprint mismatch at {result_path}")
    lease.beat(stage=stage, child_pid=None, last_successful_unit=stage)
    return result


def _model_cache_dir() -> Path:
    configured = os.environ.get("HF_HUB_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    lab_cache = Path("/opt/llm-lab/hf-cache/hub")
    if lab_cache.parent.exists() and os.access(lab_cache.parent, os.W_OK):
        return lab_cache
    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid training JSONL {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ContractError(f"training row must be an object at {path}:{line_number}")
        rows.append(row)
    return rows


def _tokenizer_preflight(model_path: Path, data_dir: Path) -> dict[str, Any]:
    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(
        str(model_path), tokenizer_config_extra={"trust_remote_code": True}
    )
    datasets = {
        split: NoThinkChatDataset(data_dir / f"{split}.jsonl", tokenizer)
        for split in ("train", "valid", "test")
    }
    rows = {
        split: _load_jsonl(data_dir / f"{split}.jsonl")
        for split in ("train", "valid", "test")
    }
    if any(
        not row["messages"][1]["content"].endswith("/no_think")
        for split_rows in rows.values()
        for row in split_rows
    ):
        raise ContractError("not every canonical chat carries the /no_think suffix")
    probe = tokenizer.apply_chat_template(
        rows["train"][0]["messages"][:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(probe, str):
        raise ContractError("tokenizer chat template did not render a string")
    assert_thinking_is_disabled(probe)
    combined: list[tuple[int, str, int]] = []
    longest_by_split: dict[str, tuple[int, str, int]] = {}
    for split, dataset in datasets.items():
        split_items = [
            (count, split, index) for index, count in enumerate(dataset.token_counts)
        ]
        combined.extend(split_items)
        longest_by_split[split] = max(split_items)
    combined.sort()
    counts = [item[0] for item in combined]
    empty_targets = [
        item
        for item in combined
        if rows[item[1]][item[2]]["messages"][-1]["content"].strip() == "[]"
    ]
    resume_item = empty_targets[0] if empty_targets else combined[0]
    longest = combined[-1]
    train_items = [
        (count, "train", index)
        for index, count in enumerate(datasets["train"].token_counts)
    ]
    train_probe_by_limit = {
        limit: max(item for item in train_items if item[0] <= limit)
        for limit in TRAINING_SEQUENCE_LENGTH_CANDIDATES
    }
    return {
        "tokenizer": tokenizer,
        "rows": rows,
        "token_counts": counts,
        "longest": longest,
        "longest_by_split": longest_by_split,
        "train_probe_by_limit": train_probe_by_limit,
        "resume": resume_item,
        "summary": {
            "chat_template_present": bool(getattr(tokenizer, "chat_template", None)),
            "thinking_disabled": True,
            "prompt_suffix": "/no_think",
            "document_count": len(counts),
            "minimum_tokens": min(counts),
            "maximum_tokens": max(counts),
            "median_tokens": counts[len(counts) // 2],
            "p95_tokens": counts[min(len(counts) - 1, math.ceil(len(counts) * 0.95) - 1)],
            "longest_split": longest[1],
            "longest_row_index": longest[2],
            "split_maximum_tokens": {
                split: item[0] for split, item in longest_by_split.items()
            },
        },
    }


def _training_request(
    *,
    model_path: Path,
    train_path: Path,
    checkpoint_root: Path,
    model_fingerprint: str,
    sequence_length: int,
    target_updates: int,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    training = StatefulTrainingConfig(
        warmup_updates=1,
        total_updates=2,
        max_sequence_length=sequence_length,
        checkpoint_every_updates=1,
        checkpoint_max_seconds=600,
    )
    request: dict[str, Any] = {
        "action": "train",
        "model_path": str(model_path),
        "train_path": str(train_path),
        "checkpoint_root": str(checkpoint_root),
        "training_config": asdict(training),
        "target_updates": target_updates,
        "input_fingerprint": sha256_file(train_path),
        "model_fingerprint": model_fingerprint,
        "trainer_implementation_sha256": sha256_file(
            Path(__file__).with_name("mlx_stateful.py")
        ),
        "worker_implementation_sha256": sha256_file(
            Path(__file__).with_name("mlx_worker.py")
        ),
    }
    if resume_checkpoint is not None:
        request["resume_checkpoint"] = str(resume_checkpoint)
    return request


def run_compute_acceptance_preflight(
    *, config: AppConfig, run_dir: Path, data_dir: Path, preflight_path: Path
) -> dict[str, Any]:
    compute_root = run_dir / "compute_acceptance"
    compute_root.mkdir(parents=True, exist_ok=True)
    try:
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"compute preflight inputs cannot be parsed: {exc}") from exc
    source_fingerprint = str(run_config["source_fingerprint"])
    lease = RunLease(
        lock_path=compute_root / "compute.lock",
        heartbeat_path=compute_root / "heartbeat.json",
        purpose="g0-compute-acceptance",
        run_id=str(preflight["run_id"]),
        input_fingerprint=source_fingerprint,
        config_fingerprint=config.fingerprint,
    ).start(stage="model_snapshot")
    preflight.update(
        {
            "status": "compute_preflight_running",
            "compute_started_at": preflight.get("compute_started_at") or _utc_now(),
            "long_run_started": False,
        }
    )
    write_json_atomic(preflight_path, preflight, mode=0o644)
    try:
        download = _run_worker(
            compute_root=compute_root,
            stage="model_snapshot",
            request={
                "action": "download",
                "model_id": config.model.model_id,
                "revision": config.model.revision,
                "cache_dir": str(_model_cache_dir()),
            },
            lease=lease,
        )
        model_path = Path(download["snapshot_path"])
        snapshot_manifest = build_snapshot_manifest(
            model_path, expected_revision=config.model.revision
        )
        write_json_atomic(
            compute_root / "model_snapshot_manifest.json", snapshot_manifest
        )
        preflight["model_snapshot"] = snapshot_manifest
        preflight["compute_gates"]["exact_model_snapshot"] = True
        write_json_atomic(preflight_path, preflight, mode=0o644)

        lease.beat(stage="tokenizer_contract")
        tokenizer_info = _tokenizer_preflight(model_path, data_dir)
        preflight["tokenizer"] = tokenizer_info["summary"]
        preflight["compute_gates"]["tokenizer_chat_template"] = True
        preflight["compute_gates"]["thinking_disabled"] = True
        write_json_atomic(preflight_path, preflight, mode=0o644)

        smoke_dir = compute_root / "fixtures"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        longest_count, longest_split, longest_index = tokenizer_info["longest"]
        resume_count, resume_split, resume_index = tokenizer_info["resume"]
        global_longest_path = smoke_dir / "global_longest.jsonl"
        resume_path = smoke_dir / "resume.jsonl"
        from .atomic import write_jsonl_atomic

        write_jsonl_atomic(
            global_longest_path,
            [tokenizer_info["rows"][longest_split][longest_index]],
        )
        write_jsonl_atomic(
            resume_path, [tokenizer_info["rows"][resume_split][resume_index]]
        )

        selected_sequence_length = next(
            candidate
            for candidate in SEQUENCE_LENGTH_CANDIDATES
            if candidate >= longest_count
        )
        memory_failures: list[dict[str, Any]] = []
        selected_training_sequence_length: int | None = None
        memory_result: dict[str, Any] | None = None
        train_probe_count: int | None = None
        for candidate in TRAINING_SEQUENCE_LENGTH_CANDIDATES:
            probe_count, probe_split, probe_index = tokenizer_info[
                "train_probe_by_limit"
            ][candidate]
            probe_path = smoke_dir / f"train_probe_{candidate}.jsonl"
            write_jsonl_atomic(
                probe_path,
                [tokenizer_info["rows"][probe_split][probe_index]],
            )
            try:
                result = _run_worker(
                    compute_root=compute_root,
                    stage=f"training_memory_smoke_{candidate}",
                    request=_training_request(
                        model_path=model_path,
                        train_path=probe_path,
                        checkpoint_root=compute_root
                        / "checkpoints"
                        / f"training_memory_{candidate}",
                        model_fingerprint=snapshot_manifest["fingerprint"],
                        sequence_length=candidate,
                        target_updates=1,
                    ),
                    lease=lease,
                )
            except (GateBlocked, IntegrityError) as exc:
                memory_failures.append(
                    {"sequence_length": candidate, "status": "failed", "error": str(exc)}
                )
                continue
            if not result.get("adapter_finite"):
                memory_failures.append(
                    {
                        "sequence_length": candidate,
                        "status": "failed",
                        "error": "adapter contains non-finite values",
                    }
                )
                continue
            selected_training_sequence_length = candidate
            memory_result = result
            train_probe_count = probe_count
            break
        if selected_training_sequence_length is None or memory_result is None:
            raise GateBlocked("no training sequence length passed the real backward smoke")
        memory_checkpoint = Path(memory_result["checkpoint"])
        longest_forward = _run_worker(
            compute_root=compute_root,
            stage=f"longest_forward_loss_{selected_sequence_length}",
            request={
                "action": "loss",
                "model_path": str(model_path),
                "adapter_path": str(memory_checkpoint),
                "adapter_sha256": sha256_file(
                    memory_checkpoint / "adapters.safetensors"
                ),
                "data_path": str(global_longest_path),
                "data_sha256": sha256_file(global_longest_path),
                "max_sequence_length": selected_sequence_length,
                "worker_implementation_sha256": sha256_file(
                    Path(__file__).with_name("mlx_worker.py")
                ),
            },
            lease=lease,
        )
        preflight["sequence_length"] = {
            "candidates": list(SEQUENCE_LENGTH_CANDIDATES),
            "selected": selected_sequence_length,
            "selected_training": selected_training_sequence_length,
            "maximum_observed_tokens": longest_count,
            "no_inference_truncation": True,
            "backward_memory_smoke": {
                "status": "passed",
                "split": "train",
                "tokens": train_probe_count,
                "sequence_length": selected_training_sequence_length,
                "peak_memory_bytes": memory_result["peak_memory_bytes"],
            },
            "backward_failures": memory_failures,
            "full_coverage_forward_smoke": {
                "status": "passed",
                "split": longest_split,
                "tokens": longest_forward["maximum_tokens"],
                "validation_loss": longest_forward["validation_loss"],
                "peak_memory_bytes": longest_forward["peak_memory_bytes"],
            },
        }
        preflight["compute_gates"]["sequence_length_memory_smoke"] = True
        write_json_atomic(preflight_path, preflight, mode=0o644)

        training_view_path = (
            data_dir / f"train_context_{selected_training_sequence_length}.jsonl"
        )
        training_view_doc_ids_path = (
            data_dir
            / f"train_context_{selected_training_sequence_length}_doc_ids.json"
        )
        training_view_manifest_path = (
            data_dir
            / f"train_context_{selected_training_sequence_length}_manifest.json"
        )
        training_view = build_context_window_view(
            source_path=data_dir / "train.jsonl",
            source_doc_ids_path=data_dir / "train_doc_ids.json",
            output_path=training_view_path,
            output_doc_ids_path=training_view_doc_ids_path,
            manifest_path=training_view_manifest_path,
            tokenizer=tokenizer_info["tokenizer"],
            model_fingerprint=snapshot_manifest["fingerprint"],
            max_sequence_length=selected_training_sequence_length,
        )
        preflight["training_view"] = {
            key: training_view[key]
            for key in (
                "source_document_count",
                "native_document_count",
                "windowed_document_count",
                "output_row_count",
                "original_reference_count",
                "chunk_reference_count",
                "empty_chunk_count",
                "maximum_tokens",
                "text_coverage",
                "reference_coverage",
                "output_jsonl_sha256",
                "output_doc_ids_sha256",
            )
        }
        preflight["compute_gates"]["training_context_view"] = True
        write_json_atomic(preflight_path, preflight, mode=0o644)

        uninterrupted = _run_worker(
            compute_root=compute_root,
            stage="resume_uninterrupted",
            request=_training_request(
                model_path=model_path,
                train_path=resume_path,
                checkpoint_root=compute_root / "checkpoints" / "uninterrupted",
                model_fingerprint=snapshot_manifest["fingerprint"],
                sequence_length=selected_training_sequence_length,
                target_updates=2,
            ),
            lease=lease,
        )
        interrupted = _run_worker(
            compute_root=compute_root,
            stage="resume_interrupted",
            request=_training_request(
                model_path=model_path,
                train_path=resume_path,
                checkpoint_root=compute_root / "checkpoints" / "resumed",
                model_fingerprint=snapshot_manifest["fingerprint"],
                sequence_length=selected_training_sequence_length,
                target_updates=1,
            ),
            lease=lease,
        )
        resumed = _run_worker(
            compute_root=compute_root,
            stage="resume_restored",
            request=_training_request(
                model_path=model_path,
                train_path=resume_path,
                checkpoint_root=compute_root / "checkpoints" / "resumed",
                model_fingerprint=snapshot_manifest["fingerprint"],
                sequence_length=selected_training_sequence_length,
                target_updates=2,
                resume_checkpoint=Path(interrupted["checkpoint"]),
            ),
            lease=lease,
        )
        assert_equivalent_training_state(
            uninterrupted["state_projection"], resumed["state_projection"]
        )
        resume_report = {
            "status": "passed",
            "fixture_tokens": resume_count,
            "uninterrupted_checkpoint": uninterrupted["checkpoint"],
            "interrupted_checkpoint": interrupted["checkpoint"],
            "resumed_checkpoint": resumed["checkpoint"],
            "state_projection": resumed["state_projection"],
        }
        write_json_atomic(compute_root / "real_resume_equivalence.json", resume_report)
        preflight["real_full_state_failure_resume"] = resume_report
        preflight["compute_gates"]["two_step_training"] = True
        preflight["compute_gates"]["real_full_state_failure_resume"] = True
        write_json_atomic(preflight_path, preflight, mode=0o644)

        generation_messages = tokenizer_info["rows"][resume_split][resume_index][
            "messages"
        ][:-1]
        generated = _run_worker(
            compute_root=compute_root,
            stage="adapter_generation",
            request={
                "action": "generate",
                "model_path": str(model_path),
                "adapter_path": uninterrupted["checkpoint"],
                "messages": generation_messages,
                "max_tokens": 512,
                "worker_implementation_sha256": sha256_file(
                    Path(__file__).with_name("mlx_worker.py")
                ),
            },
            lease=lease,
        )
        if (
            not generated.get("adapter_loaded")
            or not isinstance(generated.get("raw_output"), str)
            or int(generated.get("output_tokens", 0)) <= 0
        ):
            raise GateBlocked("adapter load/generation smoke produced no output tokens")
        preflight["generation_smoke"] = {
            key: generated[key]
            for key in (
                "adapter_loaded",
                "input_tokens",
                "output_tokens",
                "finish_reason",
                "peak_memory_bytes",
                "parse_ok",
                "parse_error",
            )
        }
        preflight["compute_gates"]["one_document_generation"] = True
        preflight["compute_gates"]["adapter_load"] = True
        if not all(preflight["compute_gates"].values()):
            raise GateBlocked("not every compute acceptance gate passed")
        preflight.update(
            {
                "status": "compute_preflight_passed_long_run_not_started",
                "compute_finished_at": _utc_now(),
                "long_run_started": False,
                "long_run_allowed": True,
            }
        )
        run_config["recovery"].update(
            {
                "real_mlx_failure_resume": resume_report,
                "long_run_allowed": True,
            }
        )
        run_config["model_snapshot"] = snapshot_manifest
        run_config["selected_sequence_length"] = selected_sequence_length
        run_config["selected_training_sequence_length"] = selected_training_sequence_length
        run_config["training_view"] = {
            "jsonl_path": str(training_view_path.resolve()),
            "jsonl_sha256": training_view["output_jsonl_sha256"],
            "doc_ids_path": str(training_view_doc_ids_path.resolve()),
            "doc_ids_sha256": training_view["output_doc_ids_sha256"],
            "manifest_path": str(training_view_manifest_path.resolve()),
            "manifest_sha256": sha256_file(training_view_manifest_path),
            "row_count": training_view["output_row_count"],
            "maximum_tokens": training_view["maximum_tokens"],
        }
        write_json_atomic(run_dir / "run_config.json", run_config)
        write_json_atomic(preflight_path, preflight, mode=0o644)
        lease.finish(status="completed")
        return preflight
    except BaseException as exc:
        preflight.update(
            {
                "status": "compute_preflight_failed",
                "compute_failed_at": _utc_now(),
                "long_run_started": False,
                "last_error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_json_atomic(preflight_path, preflight, mode=0o644)
        lease.finish(status="failed", error=str(exc))
        raise
