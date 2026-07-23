"""Segmented, resume-safe G0 development training and validation orchestration."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .atomic import fsync_directory, write_json_atomic, write_text_atomic
from .checkpoints import CheckpointManager
from .config import AppConfig
from .errors import FingerprintMismatch, GateBlocked, IntegrityError
from .fingerprints import fingerprint_json, sha256_file
from .g0 import (
    SEQUENCE_LENGTH_CANDIDATES,
    CheckpointCandidate,
    TrainingContract,
    select_checkpoint,
)
from .heartbeat import RunLease
from .mlx_compute import _run_worker
from .mlx_stateful import StatefulTrainingConfig

PILOT_LEARNING_RATES = {"lr5e-5": 5e-5, "lr1e-4": 1e-4}
PILOT_TARGET_UPDATES = 50
MAX_GENERATION_TOKENS = 2048
REFERENCE_POSTPROCESS = "canonical_set"
CORE_REFERENCE_VIEW = "row"
RUN_ID_PATTERN = re.compile(r"dqcheck_g0_qwen3_5_9b_[0-9a-f]{12}\Z")

WorkerRunner = Callable[..., dict[str, Any]]
EvaluatorRunner = Callable[..., dict[str, Any]]
CheckpointVerifier = Callable[[Path], dict[str, Any]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"JSON cannot be parsed at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise IntegrityError(f"JSON root is not an object at {path}")
    return payload


def validation_milestones(target_updates: int, *, every: int = 25) -> list[int]:
    maximum = TrainingContract().maximum_optimizer_updates
    if target_updates <= 0 or target_updates > maximum:
        raise ValueError(f"target_updates must be between 1 and {maximum}")
    milestones = list(range(every, target_updates + 1, every))
    if not milestones or milestones[-1] != target_updates:
        milestones.append(target_updates)
    return milestones


def _verified_checkpoints(
    checkpoint_root: Path,
    *,
    verifier: CheckpointVerifier,
    maximum_update: int,
) -> dict[int, Path]:
    checkpoints: dict[int, Path] = {}
    if not checkpoint_root.exists():
        return checkpoints
    for path in sorted(checkpoint_root.glob("update_*")):
        try:
            update = int(path.name.removeprefix("update_"))
        except ValueError:
            continue
        if update > maximum_update:
            continue
        manifest = verifier(path)
        if int(manifest.get("global_update", -1)) != update:
            raise IntegrityError(f"checkpoint update mismatch at {path}")
        checkpoints[update] = path
    return checkpoints


def _candidate_training_config(
    *, peak_learning_rate: float, sequence_length: int
) -> StatefulTrainingConfig:
    contract = TrainingContract()
    return StatefulTrainingConfig(
        seed=contract.seed,
        rank=contract.lora_rank,
        num_layers=contract.lora_layers,
        lora_scale=20.0,
        lora_dropout=0.0,
        batch_size=contract.batch_size,
        gradient_accumulation=contract.gradient_accumulation,
        peak_learning_rate=peak_learning_rate,
        end_learning_rate=contract.end_learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        adam_bias_correction=False,
        warmup_updates=contract.warmup_updates,
        total_updates=contract.maximum_optimizer_updates,
        max_sequence_length=sequence_length,
        gradient_checkpointing=contract.gradient_checkpointing,
        checkpoint_every_updates=contract.checkpoint_every_optimizer_updates,
        checkpoint_max_seconds=contract.checkpoint_max_minutes * 60,
    )


def _base_run_context(config: AppConfig, run_id: str) -> dict[str, Any]:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise GateBlocked(f"invalid canonical G0 run_id: {run_id!r}")
    run_dir = (config.training_runs_root / run_id).resolve()
    if run_dir.parent != config.training_runs_root.resolve():
        raise GateBlocked("G0 run path escapes the configured training root")
    run_config = _read_json(run_dir / "run_config.json")
    preflight_path = config.public_root / "g0" / run_id / "preflight.json"
    preflight = _read_json(preflight_path)
    if run_config.get("run_id") != run_id or preflight.get("run_id") != run_id:
        raise FingerprintMismatch("G0 run_id differs across run config and preflight")
    data_dir = Path(str(run_config.get("data_manifest", {}).get("path", ""))).resolve()
    if not data_dir.is_relative_to(config.sensitive_root.resolve()):
        raise GateBlocked("G0 training data is outside the configured sensitive root")
    for name in ("train.jsonl", "valid.jsonl", "valid_doc_ids.json"):
        if not (data_dir / name).is_file():
            raise GateBlocked(f"G0 training input is missing: {data_dir / name}")
    training_view = run_config.get("training_view")
    if not isinstance(training_view, dict):
        raise GateBlocked("fingerprinted training context view is missing")
    train_path = Path(str(training_view.get("jsonl_path", ""))).resolve()
    if not train_path.is_relative_to(data_dir) or not train_path.is_file():
        raise GateBlocked("training context JSONL is outside or missing from the data view")
    if training_view.get("jsonl_sha256") != sha256_file(train_path):
        raise FingerprintMismatch("training context JSONL checksum mismatch")
    return {
        "run_dir": run_dir,
        "run_config": run_config,
        "preflight": preflight,
        "preflight_path": preflight_path,
        "data_dir": data_dir,
        "train_path": train_path,
    }


def _preflight_blockers(context: dict[str, Any]) -> list[str]:
    preflight = context["preflight"]
    run_config = context["run_config"]
    blockers: list[str] = []
    if preflight.get("status") != "compute_preflight_passed_long_run_not_started":
        blockers.append(f"preflight status={preflight.get('status')!r}")
    if preflight.get("long_run_allowed") is not True:
        blockers.append("preflight long_run_allowed is not true")
    if run_config.get("recovery", {}).get("long_run_allowed") is not True:
        blockers.append("run config long_run_allowed is not true")
    if not isinstance(run_config.get("selected_sequence_length"), int):
        blockers.append("selected_sequence_length is missing")
    if not isinstance(run_config.get("selected_training_sequence_length"), int):
        blockers.append("selected_training_sequence_length is missing")
    snapshot = run_config.get("model_snapshot")
    if not isinstance(snapshot, dict) or not snapshot.get("fingerprint"):
        blockers.append("verified model snapshot is missing")
    return blockers


def _candidate_contract(
    *, config: AppConfig, context: dict[str, Any], candidate_id: str
) -> tuple[dict[str, Any], StatefulTrainingConfig]:
    if candidate_id not in PILOT_LEARNING_RATES:
        raise ValueError(f"unsupported candidate: {candidate_id}")
    run_config = context["run_config"]
    preflight = context["preflight"]
    accepted_training_sequence_length = run_config.get(
        "selected_training_sequence_length"
    )
    if isinstance(accepted_training_sequence_length, int):
        sequence_length = accepted_training_sequence_length
    else:
        maximum_tokens = preflight.get("tokenizer", {}).get("maximum_tokens")
        sequence_length = next(
            (
                candidate
                for candidate in SEQUENCE_LENGTH_CANDIDATES
                if isinstance(maximum_tokens, int) and candidate >= maximum_tokens
            ),
            0,
        )
    snapshot = run_config.get("model_snapshot") or preflight.get("model_snapshot")
    if sequence_length <= 0 or not isinstance(snapshot, dict) or not snapshot.get(
        "fingerprint"
    ):
        raise GateBlocked("not enough verified compute metadata to render candidate contract")
    training = _candidate_training_config(
        peak_learning_rate=PILOT_LEARNING_RATES[candidate_id],
        sequence_length=sequence_length,
    )
    trajectory = asdict(training)
    trajectory["sequence_length_buckets"] = list(training.sequence_length_buckets)
    contract = {
        "schema_version": 1,
        "base_run_id": run_config["run_id"],
        "candidate_id": candidate_id,
        "trajectory": trajectory,
        "validation": {
            "documents": 50,
            "every_optimizer_updates": 25,
            "max_generation_tokens": MAX_GENERATION_TOKENS,
            "temperature": 0.0,
            "reference_postprocess": REFERENCE_POSTPROCESS,
            "core_reference_view": CORE_REFERENCE_VIEW,
            "docwise_threshold": 1.0,
            "checkpoint_order": [
                "core_law_article_strict.f1",
                "docwise_core_accuracy.accuracy",
                "core_law_article_strict.recall",
                "validation_loss",
            ],
        },
        "source": {
            "source_fingerprint": run_config["source_fingerprint"],
            "train_jsonl_path": str(context["train_path"]),
            "train_jsonl_sha256": sha256_file(context["train_path"]),
            "valid_jsonl_sha256": sha256_file(context["data_dir"] / "valid.jsonl"),
            "valid_doc_ids_sha256": sha256_file(
                context["data_dir"] / "valid_doc_ids.json"
            ),
        },
        "model": {
            "id": run_config["model"]["model_id"],
            "revision": run_config["model"]["revision"],
            "snapshot_fingerprint": snapshot["fingerprint"],
            "training_sequence_length_accepted": isinstance(
                accepted_training_sequence_length, int
            ),
            "inference_sequence_length": run_config.get("selected_sequence_length"),
        },
        "implementation": {
            "trainer_sha256": sha256_file(Path(__file__).with_name("mlx_stateful.py")),
            "worker_sha256": sha256_file(Path(__file__).with_name("mlx_worker.py")),
            "orchestrator_sha256": sha256_file(Path(__file__)),
            "canonical_evaluator_sha256": sha256_file(
                config.canonical_gt_dir.resolve().parents[3]
                / "benchmark"
                / "reference"
                / "evaluate.py"
            ),
        },
    }
    contract["contract_fingerprint"] = fingerprint_json(contract)
    return contract, training


def _write_or_verify_contract(path: Path, contract: dict[str, Any]) -> None:
    if path.exists():
        existing = _read_json(path)
        if existing != contract:
            raise FingerprintMismatch(
                f"candidate contract drift at {path}; start a new reviewed candidate"
            )
        return
    write_json_atomic(path, contract)


def _checkpoint_verifier(
    *,
    checkpoint_root: Path,
    training: StatefulTrainingConfig,
    train_path: Path,
    model_fingerprint: str,
) -> CheckpointVerifier:
    manager = CheckpointManager(
        checkpoint_root,
        input_fingerprint=sha256_file(train_path),
        config_fingerprint=fingerprint_json(training.__dict__),
        model_fingerprint=model_fingerprint,
    )
    return manager.verify


def _training_request(
    *,
    context: dict[str, Any],
    training: StatefulTrainingConfig,
    checkpoint_root: Path,
    target_updates: int,
    resume_checkpoint: Path | None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "action": "train",
        "model_path": context["run_config"]["model_snapshot"]["snapshot_path"],
        "train_path": str(context["train_path"]),
        "checkpoint_root": str(checkpoint_root),
        "training_config": asdict(training),
        "target_updates": target_updates,
        "input_fingerprint": sha256_file(context["train_path"]),
        "model_fingerprint": context["run_config"]["model_snapshot"]["fingerprint"],
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


def _validation_request(
    *,
    context: dict[str, Any],
    checkpoint: Path,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "action": "validate",
        "model_path": context["run_config"]["model_snapshot"]["snapshot_path"],
        "adapter_path": str(checkpoint),
        "adapter_sha256": sha256_file(checkpoint / "adapters.safetensors"),
        "data_path": str(context["data_dir"] / "valid.jsonl"),
        "data_sha256": sha256_file(context["data_dir"] / "valid.jsonl"),
        "doc_ids_path": str(context["data_dir"] / "valid_doc_ids.json"),
        "doc_ids_sha256": sha256_file(context["data_dir"] / "valid_doc_ids.json"),
        "expected_count": 50,
        "output_dir": str(output_dir),
        "max_sequence_length": context["run_config"]["selected_sequence_length"],
        "max_generation_tokens": MAX_GENERATION_TOKENS,
        "worker_implementation_sha256": sha256_file(
            Path(__file__).with_name("mlx_worker.py")
        ),
    }


def _canonical_evaluate(
    *,
    config: AppConfig,
    predictions_path: Path,
    doc_ids_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    repo_root = config.canonical_gt_dir.resolve().parents[3]
    evaluator = repo_root / "benchmark" / "reference" / "evaluate.py"
    if not evaluator.is_file():
        raise GateBlocked(f"canonical evaluator is unavailable: {evaluator}")
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_dir, prefix=".evaluation.", suffix=".tmp.json"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    report_path = output_dir / "evaluation.json"
    command = [
        sys.executable,
        str(evaluator),
        "--predictions",
        str(predictions_path),
        "--ground-truth-dir",
        str(config.canonical_gt_dir),
        "--doc-ids-file",
        str(doc_ids_path),
        "--json-report",
        str(temporary),
        "--gt-mode",
        "approved_only",
        "--docwise-threshold",
        "1.0",
        "--reference-postprocess",
        REFERENCE_POSTPROCESS,
        "--core-reference-view",
        CORE_REFERENCE_VIEW,
        "--per-doc",
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                part
                for part in (str(repo_root), os.environ.get("PYTHONPATH", ""))
                if part
            ),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    write_text_atomic(output_dir / "evaluator.log", completed.stdout)
    try:
        if completed.returncode != 0:
            raise GateBlocked(
                f"canonical evaluator exited {completed.returncode}; "
                f"see {output_dir / 'evaluator.log'}"
            )
        report = _read_json(temporary)
        results = report.get("results")
        if not isinstance(results, list) or len(results) != 1:
            raise IntegrityError("canonical evaluator returned an unexpected result count")
        result = results[0]
        if (
            result.get("evaluated_doc_count") != 50
            or result.get("missing_prediction_doc_ids")
            or result.get("duplicate_prediction_doc_ids")
        ):
            raise IntegrityError("validation evaluator coverage gate failed")
        os.replace(temporary, report_path)
        fsync_directory(output_dir)
        return report
    finally:
        temporary.unlink(missing_ok=True)


def _validation_summary(
    *,
    update: int,
    checkpoint: Path,
    checkpoint_manifest: dict[str, Any],
    validation: dict[str, Any],
    evaluation: dict[str, Any],
    evaluation_sha256: str,
) -> dict[str, Any]:
    result = evaluation["results"][0]
    core = result["core_law_article_strict"]
    docwise = result["docwise_core_accuracy"]
    validation_loss = float(validation["validation_loss"])
    if not math.isfinite(validation_loss):
        raise IntegrityError("validation loss is not finite")
    summary = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "update": update,
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_sha256": sha256_file(checkpoint / "manifest.json"),
        "checkpoint_trainer_state_fingerprint": checkpoint_manifest[
            "trainer_state_fingerprint"
        ],
        "coverage_count": int(validation["coverage_count"]),
        "parse_count": int(validation["parse_count"]),
        "empty_output_count": int(validation["empty_output_count"]),
        "runaway_output_count": int(validation["runaway_output_count"]),
        "validation_loss": validation_loss,
        "core_law_article_strict": core,
        "docwise_core_accuracy": docwise,
        "metric_view": {
            "reference_postprocess": REFERENCE_POSTPROCESS,
            "core_reference_view": CORE_REFERENCE_VIEW,
            "docwise_threshold": 1.0,
        },
        "prediction_sha256": validation["predictions_sha256"],
        "validation_manifest_sha256": validation["manifest_sha256"],
        "evaluation_sha256": evaluation_sha256,
    }
    summary["eligible"] = (
        summary["coverage_count"] == 50
        and summary["parse_count"] >= 49
        and summary["empty_output_count"] == 0
        and summary["runaway_output_count"] == 0
    )
    return summary


def _best_summary(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        CheckpointCandidate(
            update=int(summary["update"]),
            coverage_count=int(summary["coverage_count"]),
            parse_count=int(summary["parse_count"]),
            empty_output_count=int(summary["empty_output_count"]),
            runaway_output_count=int(summary["runaway_output_count"]),
            core_f1=float(summary["core_law_article_strict"]["f1"]),
            docwise_accuracy=float(summary["docwise_core_accuracy"]["accuracy"]),
            recall=float(summary["core_law_article_strict"]["recall"]),
            validation_loss=float(summary["validation_loss"]),
        )
        for summary in summaries
    ]
    try:
        selected = select_checkpoint(candidates)
    except GateBlocked:
        return None
    return next(summary for summary in summaries if summary["update"] == selected.update)


def _load_valid_summaries(candidate_root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted((candidate_root / "validation").glob("update_*/summary.json")):
        summary = _read_json(path)
        if sha256_file(Path(summary["checkpoint"]) / "manifest.json") != summary.get(
            "checkpoint_manifest_sha256"
        ):
            raise IntegrityError(f"checkpoint drift for validation summary {path}")
        summaries.append(summary)
    return summaries


def _mark_development(
    *, context: dict[str, Any], candidate_id: str, status: str, update: int
) -> None:
    preflight = _read_json(context["preflight_path"])
    development = dict(preflight.get("development", {}))
    development[candidate_id] = {
        "status": status,
        "last_validated_update": update,
        "updated_at": _utc_now(),
    }
    preflight["development"] = development
    preflight["long_run_started"] = True
    write_json_atomic(context["preflight_path"], preflight, mode=0o644)


def _public_summary(
    *,
    config: AppConfig,
    run_id: str,
    candidate_id: str,
    state: dict[str, Any],
    summaries: list[dict[str, Any]],
) -> None:
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "candidate_id": candidate_id,
        "status": state["status"],
        "requested_target_updates": state["requested_target_updates"],
        "last_validated_update": state["last_validated_update"],
        "best_update": state.get("best_update"),
        "validations": [
            {
                key: summary[key]
                for key in (
                    "update",
                    "eligible",
                    "coverage_count",
                    "parse_count",
                    "empty_output_count",
                    "runaway_output_count",
                    "validation_loss",
                    "core_law_article_strict",
                    "docwise_core_accuracy",
                    "metric_view",
                )
            }
            for summary in summaries
        ],
        "updated_at": _utc_now(),
    }
    write_json_atomic(
        config.public_root / "g0" / run_id / "development" / f"{candidate_id}.json",
        payload,
        mode=0o644,
    )


def run_development(
    *,
    config: AppConfig,
    run_id: str,
    candidate_id: str,
    target_updates: int = PILOT_TARGET_UPDATES,
    execute: bool = False,
    resume: bool = False,
    require_tmux: bool = True,
    worker_runner: WorkerRunner | None = None,
    evaluator_runner: EvaluatorRunner | None = None,
    checkpoint_verifier: CheckpointVerifier | None = None,
) -> dict[str, Any]:
    """Plan or execute one deterministic LR candidate through validation milestones."""

    milestones = validation_milestones(target_updates)
    if candidate_id not in PILOT_LEARNING_RATES:
        raise ValueError(f"unsupported candidate: {candidate_id}")
    context = _base_run_context(config, run_id)
    blockers = _preflight_blockers(context)
    try:
        contract, training = _candidate_contract(
            config=config,
            context=context,
            candidate_id=candidate_id,
        )
    except GateBlocked as exc:
        contract = None
        training = None
        blockers.append(str(exc))
    plan = {
        "schema_version": 1,
        "status": "blocked" if blockers else "ready",
        "run_id": run_id,
        "candidate_id": candidate_id,
        "peak_learning_rate": PILOT_LEARNING_RATES[candidate_id],
        "target_updates": target_updates,
        "validation_milestones": milestones,
        "blockers": blockers,
        "contract": contract,
        "long_run_started": False,
    }
    if not execute:
        return plan
    if blockers:
        raise GateBlocked("G0 long-run preflight is blocked: " + "; ".join(blockers))
    if require_tmux and not os.environ.get("TMUX"):
        raise GateBlocked("real G0 training must run inside tmux")
    assert contract is not None and training is not None

    candidate_root = context["run_dir"] / "development" / candidate_id
    candidate_root.mkdir(parents=True, exist_ok=True)
    contract_path = candidate_root / "candidate_config.json"
    state_path = candidate_root / "state.json"
    _write_or_verify_contract(contract_path, contract)
    if state_path.exists() and not resume:
        raise GateBlocked(
            f"candidate state already exists at {state_path}; use --resume"
        )

    state = _read_json(state_path) if state_path.exists() else {
        "schema_version": 1,
        "run_id": run_id,
        "candidate_id": candidate_id,
        "contract_fingerprint": contract["contract_fingerprint"],
        "status": "planned",
        "last_validated_update": 0,
        "best_update": None,
        "recovery_history": [],
    }
    if state.get("contract_fingerprint") != contract["contract_fingerprint"]:
        raise FingerprintMismatch("candidate state contract fingerprint mismatch")
    previous_update = int(state.get("last_validated_update", 0))
    if target_updates < previous_update:
        raise GateBlocked(
            f"target_updates={target_updates} is behind completed update={previous_update}"
        )
    if resume:
        state["recovery_history"].append(
            {
                "resumed_at": _utc_now(),
                "from_validated_update": previous_update,
                "requested_target_updates": target_updates,
            }
        )
    state.update(
        {
            "status": "running",
            "requested_target_updates": target_updates,
            "started_at": state.get("started_at") or _utc_now(),
            "updated_at": _utc_now(),
            "last_error": None,
            "long_run_started": True,
        }
    )
    write_json_atomic(state_path, state)

    lease = RunLease(
        # One shared device lock prevents two LR candidates from training on
        # the same Metal device at the same time.
        lock_path=context["run_dir"] / "development" / "training.lock",
        heartbeat_path=candidate_root / "heartbeat.json",
        purpose="g0-development-training",
        run_id=f"{run_id}:{candidate_id}",
        input_fingerprint=contract["source"]["train_jsonl_sha256"],
        config_fingerprint=contract["contract_fingerprint"],
    ).start(stage="starting")
    runner = worker_runner or _run_worker
    evaluator = evaluator_runner or _canonical_evaluate
    checkpoint_root = candidate_root / "checkpoints"
    verify_checkpoint = checkpoint_verifier or _checkpoint_verifier(
        checkpoint_root=checkpoint_root,
        training=training,
        train_path=context["train_path"],
        model_fingerprint=context["run_config"]["model_snapshot"]["fingerprint"],
    )
    verified_checkpoints = _verified_checkpoints(
        checkpoint_root,
        verifier=verify_checkpoint,
        maximum_update=target_updates,
    )
    if verified_checkpoints and max(verified_checkpoints) > previous_update:
        state["recovery_history"].append(
            {
                "checkpoint_discovered_at": _utc_now(),
                "checkpoint_update": max(verified_checkpoints),
                "from_validated_update": previous_update,
            }
        )
        write_json_atomic(state_path, state)
    summaries = _load_valid_summaries(candidate_root)
    summary_by_update = {int(summary["update"]): summary for summary in summaries}
    try:
        _mark_development(
            context=context,
            candidate_id=candidate_id,
            status="running",
            update=previous_update,
        )
        for unit_number, update in enumerate(milestones, 1):
            if update in summary_by_update:
                summary_checkpoint = Path(summary_by_update[update]["checkpoint"])
                verify_checkpoint(summary_checkpoint)
                verified_checkpoints[update] = summary_checkpoint
                continue
            checkpoint = checkpoint_root / f"update_{update:07d}"
            if checkpoint.exists():
                checkpoint_manifest = verify_checkpoint(checkpoint)
                verified_checkpoints[update] = checkpoint
            else:
                resume_checkpoint = next(
                    (
                        verified_checkpoints[candidate_update]
                        for candidate_update in sorted(
                            verified_checkpoints, reverse=True
                        )
                        if candidate_update < update
                    ),
                    None,
                )
                lease.beat(
                    stage=f"train_{update:07d}",
                    expected_units=len(milestones),
                    completed_units=unit_number - 1,
                )
                result = runner(
                    compute_root=candidate_root,
                    stage=f"train_{update:07d}",
                    request=_training_request(
                        context=context,
                        training=training,
                        checkpoint_root=checkpoint_root,
                        target_updates=update,
                        resume_checkpoint=resume_checkpoint,
                    ),
                    lease=lease,
                    timeout_seconds=43_200,
                )
                if int(result.get("global_update", -1)) != update:
                    raise IntegrityError(
                        f"training worker stopped at {result.get('global_update')} not {update}"
                    )
                checkpoint = Path(result["checkpoint"])
                checkpoint_manifest = verify_checkpoint(checkpoint)
                verified_checkpoints[update] = checkpoint
            state.update(
                {
                    "stage": "validation",
                    "active_update": update,
                    "active_checkpoint": str(checkpoint),
                    "updated_at": _utc_now(),
                }
            )
            write_json_atomic(state_path, state)

            validation_dir = candidate_root / "validation" / f"update_{update:07d}"
            validation = runner(
                compute_root=candidate_root,
                stage=f"validate_{update:07d}",
                request=_validation_request(
                    context=context,
                    checkpoint=checkpoint,
                    output_dir=validation_dir,
                ),
                lease=lease,
                timeout_seconds=14_400,
            )
            evaluation = evaluator(
                config=config,
                predictions_path=Path(validation["predictions_path"]),
                doc_ids_path=context["data_dir"] / "valid_doc_ids.json",
                output_dir=validation_dir,
            )
            evaluation_path = validation_dir / "evaluation.json"
            if not evaluation_path.is_file():
                raise IntegrityError(
                    f"validation evaluator did not seal {evaluation_path}"
                )
            summary = _validation_summary(
                update=update,
                checkpoint=checkpoint,
                checkpoint_manifest=checkpoint_manifest,
                validation=validation,
                evaluation=evaluation,
                evaluation_sha256=sha256_file(evaluation_path),
            )
            write_json_atomic(validation_dir / "summary.json", summary)
            summary_by_update[update] = summary
            summaries = [summary_by_update[key] for key in sorted(summary_by_update)]
            best = _best_summary(summaries)
            state.update(
                {
                    "stage": "validated",
                    "last_validated_update": update,
                    "best_update": best["update"] if best else None,
                    "active_update": None,
                    "active_checkpoint": None,
                    "updated_at": _utc_now(),
                }
            )
            write_json_atomic(state_path, state)
            _public_summary(
                config=config,
                run_id=run_id,
                candidate_id=candidate_id,
                state=state,
                summaries=summaries,
            )
            _mark_development(
                context=context,
                candidate_id=candidate_id,
                status="running",
                update=update,
            )
            lease.beat(
                stage="validated",
                completed_units=unit_number,
                expected_units=len(milestones),
                last_successful_unit=f"validation_{update:07d}",
            )

        best = _best_summary(summaries)
        state.update(
            {
                "status": (
                    "pilot_completed"
                    if target_updates == PILOT_TARGET_UPDATES
                    else "development_completed"
                    if target_updates == TrainingContract().maximum_optimizer_updates
                    else "segment_completed"
                ),
                "best_update": best["update"] if best else None,
                "finished_at": _utc_now(),
                "updated_at": _utc_now(),
            }
        )
        write_json_atomic(state_path, state)
        _public_summary(
            config=config,
            run_id=run_id,
            candidate_id=candidate_id,
            state=state,
            summaries=summaries,
        )
        _mark_development(
            context=context,
            candidate_id=candidate_id,
            status=state["status"],
            update=int(state["last_validated_update"]),
        )
        lease.finish(status="completed")
        return {
            **state,
            "candidate_root": str(candidate_root),
            "validation_count": len(summaries),
            "best_validation": best,
        }
    except BaseException as exc:
        state.update(
            {
                "status": "failed",
                "last_error": f"{type(exc).__name__}: {exc}",
                "updated_at": _utc_now(),
            }
        )
        write_json_atomic(state_path, state)
        _public_summary(
            config=config,
            run_id=run_id,
            candidate_id=candidate_id,
            state=state,
            summaries=summaries,
        )
        _mark_development(
            context=context,
            candidate_id=candidate_id,
            status="failed",
            update=int(state.get("last_validated_update", 0)),
        )
        lease.finish(status="failed", error=str(exc))
        raise
