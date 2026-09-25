"""Operator entry points for the GT750 DQ-Loop, one per step a human starts.

Round 0 trains M000 on the 500-document train split and gates it on
validation against the untrained base model (B-4). Round k >= 1 prepares and
predicts batch R0k with M(k-1); the judge, the expert and the release run
through the existing `pilot-judges` / `judge-lock` / `serve` / `release`
commands; `loop-mark` records those stages; `loop-train` then trains M(k).
`loop-errors` scores the annotator, the model and the judge against the
expert for a round (written automatically when the round is marked
adjudicated).
The test split stays closed for the whole loop: no test prediction is even
generated until `loop-open-test --confirm-open-test`, after the last round.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .constants import JUDGE_MODEL_KEY
from .errors import ContractError, GateBlocked
from .fingerprints import sha256_file
from .g0 import assert_test_improves_base
from .g0_finalize import finalize_selection, round_label, seal_round_model
from .g0_training import (
    MAX_GENERATION_TOKENS,
    WINDOW_FALLBACK_MAX_GENERATION_TOKENS,
    WINDOW_FALLBACK_OVERLAP_TOKENS,
    WINDOW_FALLBACK_TOKENS,
    _base_run_context,
    canonical_evaluate,
)
from .heartbeat import RunLease
from .loop_bindings import write_error_report
from .loop_rounds import advance_round, resume_round, rounds_state_dir, write_round_state
from .loop_sources import validate_loop_sources
from .loop_train import train_round

LOOP_CANDIDATE = "loop-lr2.5e-5-full-epoch"


def batch_id_for(round_index: int) -> str:
    if round_index < 1:
        raise ContractError("batches start at round 1; round 0 trains M000 on the canonical split")
    return f"R{round_index:02d}"


def _loop_dir(config: Any) -> Path:
    return Path(config.public_root) / "loop"


def _plan(config: Any) -> dict[str, Any]:
    return json.loads((_loop_dir(config) / "loop_plan.json").read_text(encoding="utf-8"))


def _model_record_path(config: Any, round_index: int) -> Path:
    return _loop_dir(config) / "rounds" / f"round_{round_index:03d}_model.json"


def _worker_request(context: dict[str, Any], *, split: str, expected: int, output_dir: Path,
                    checkpoint: Path | None) -> dict[str, Any]:
    data = context["data_dir"]
    request: dict[str, Any] = {
        "action": "validate",
        "model_path": context["run_config"]["model_snapshot"]["snapshot_path"],
        "data_path": str(data / f"{split}.jsonl"),
        "data_sha256": sha256_file(data / f"{split}.jsonl"),
        "doc_ids_path": str(data / f"{split}_doc_ids.json"),
        "doc_ids_sha256": sha256_file(data / f"{split}_doc_ids.json"),
        "expected_count": expected,
        "output_dir": str(output_dir),
        "max_sequence_length": context["run_config"]["selected_sequence_length"],
        "max_generation_tokens": MAX_GENERATION_TOKENS,
        "window_fallback_enabled": True,
        "window_fallback_tokens": WINDOW_FALLBACK_TOKENS,
        "window_fallback_overlap_tokens": WINDOW_FALLBACK_OVERLAP_TOKENS,
        "window_fallback_max_generation_tokens": WINDOW_FALLBACK_MAX_GENERATION_TOKENS,
        "window_fallback_repetition_recovery_enabled": True,
        "worker_implementation_sha256": sha256_file(Path(__file__).with_name("mlx_worker.py")),
    }
    if checkpoint is not None:
        request["adapter_path"] = str(checkpoint)
        request["adapter_sha256"] = sha256_file(checkpoint / "adapters.safetensors")
    return request


def _run_split(config: Any, run_id: str, *, split: str, expected: int, output_dir: Path,
               checkpoint: Path | None, purpose: str, worker=None, evaluator=None) -> dict[str, Any]:
    if worker is None:
        from .mlx_compute import _run_worker as worker
    evaluator = evaluator or canonical_evaluate
    context = _base_run_context(config, run_id)
    lease = RunLease(
        lock_path=context["run_dir"] / "development" / "training.lock",
        heartbeat_path=output_dir / "heartbeat.json",
        purpose=purpose,
        run_id=f"{run_id}:{purpose}",
        input_fingerprint=sha256_file(context["data_dir"] / f"{split}.jsonl"),
        config_fingerprint=str(getattr(config, "fingerprint", "")),
    ).start(stage=purpose)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        result = worker(compute_root=output_dir, stage=purpose,
                        request=_worker_request(context, split=split, expected=expected,
                                                output_dir=output_dir, checkpoint=checkpoint),
                        lease=lease, timeout_seconds=14_400)
        evaluation = evaluator(config=config, predictions_path=Path(result["predictions_path"]),
                               doc_ids_path=context["data_dir"] / f"{split}_doc_ids.json",
                               output_dir=output_dir, expected_doc_count=expected)
    except BaseException as exc:
        lease.finish(status="failed", error=str(exc))
        raise
    lease.finish(status="completed")
    return {"worker": result, "evaluation": evaluation}


def _base_validation_f1(config: Any, run_id: str, *, worker=None, evaluator=None) -> float:
    output_dir = _loop_dir(config) / "baseline" / run_id
    done = output_dir / "baseline.json"
    if done.is_file():
        return float(json.loads(done.read_text(encoding="utf-8"))["core_f1"])
    run = _run_split(config, run_id, split="valid", expected=50, output_dir=output_dir,
                     checkpoint=None, purpose="loop-base-validation", worker=worker, evaluator=evaluator)
    core = run["evaluation"]["results"][0]["core_law_article_strict"]
    write_json_atomic(done, {"schema_version": 1, "run_id": run_id, "core_f1": core["f1"],
                             "precision": core["precision"], "recall": core["recall"],
                             "generated_at": datetime.now(UTC).isoformat()})
    return float(core["f1"])


def _train_and_select(config: Any, *, round_index: int, resume: bool) -> dict[str, Any]:
    source = validate_loop_sources(config)
    cleaned = [batch_id_for(k) for k in range(1, round_index + 1)]
    result = train_round(config, generation=round_label(round_index), split=source["split"],
                         cleaned_batch_ids=cleaned, execute=True, candidate_id=LOOP_CANDIDATE,
                         target_updates=None, resume=resume)
    candidate_root = Path(config.training_runs_root) / result["run_id"] / "development" / LOOP_CANDIDATE
    return {"run_id": result["run_id"], "cleaned_batch_ids": cleaned,
            "training_documents": result["training_documents"],
            "selection": finalize_selection(candidate_root)}


def train_loop_round(config: Any, *, round_index: int, resume: bool, worker=None, evaluator=None) -> dict[str, Any]:
    if round_index >= 1:
        # Check before hours of training, not after: M(k) trains only on a released R0k.
        stage = resume_round(rounds_state_dir(config), round_index).stage
        if stage != "adjudicated":
            raise GateBlocked(f"round {round_index} is at {stage!r}; run loop-mark --stage adjudicated first")
    trained = _train_and_select(config, round_index=round_index, resume=resume)
    selection = trained["selection"]
    record: dict[str, Any] = {"schema_version": 1, "round": round_index,
                              "generation": round_label(round_index), **trained,
                              "test": {"opened": False}}
    if round_index == 0:
        base = _base_validation_f1(config, trained["run_id"], worker=worker, evaluator=evaluator)
        assert_test_improves_base(base_core_f1=base, tuned_core_f1=float(selection["core_f1"]))
        record["validation_gate"] = {"base_core_f1": base, "tuned_core_f1": selection["core_f1"],
                                     "passed": True}
    context = _base_run_context(config, trained["run_id"])
    registry = seal_round_model(
        config=config, round_label=round_label(round_index),
        model_snapshot_path=Path(context["run_config"]["model_snapshot"]["snapshot_path"]),
        adapter_path=Path(selection["adapter_path"]),
        max_sequence_length=int(context["run_config"]["selected_sequence_length"]),
    )
    record["registry_sha256"] = sha256_file(registry)
    path = _model_record_path(config, round_index)
    write_json_atomic(path, record)
    if round_index >= 1:
        state_dir = rounds_state_dir(config)
        state = resume_round(state_dir, round_index)
        for stage in ("composed", "trained", "checkpoint_selected", "measured", "sealed"):
            state = advance_round(state, to=stage, artifact_sha256=sha256_file(path))
        write_round_state(state_dir, state)
    return record


def predict_loop_round(config: Any, *, round_index: int, hmac_key_file: Path | None) -> dict[str, Any]:
    from .preparation import prepare_batch
    from .processing import process_batch

    if not _model_record_path(config, round_index - 1).is_file():
        raise GateBlocked(f"M{round_index - 1:03d} is not trained yet; run loop-train --round {round_index - 1}")
    plan = _plan(config)
    manifest = json.loads(Path(plan["batch_manifest_path"]).read_text(encoding="utf-8"))
    doc_ids = set(manifest["batches"][round_index - 1])
    batch_id = batch_id_for(round_index)
    prepared = prepare_batch(config=config, annotation_zip=Path(plan["annotation_zip"]),
                             document_pool_zip=Path(plan["document_pool_zip"]), batch_id=batch_id,
                             hmac_key_file=hmac_key_file, doc_ids=doc_ids)
    processed = process_batch(config=config, batch_id=batch_id,
                              generation=round_label(round_index - 1), resume=True)
    state_dir = rounds_state_dir(config)
    state = resume_round(state_dir, round_index)
    summary = Path(config.public_root) / "batches" / batch_id / f"process_{round_label(round_index - 1)}_summary.json"
    if state.stage == "pending":
        state = advance_round(state, to="predicted", artifact_sha256=sha256_file(summary))
        state = advance_round(state, to="routed", artifact_sha256=sha256_file(summary))
        write_round_state(state_dir, state)
    return {"batch_id": batch_id, "documents": len(doc_ids), "prepared": prepared, "processed": processed}


def error_report(config: Any, *, round_index: int) -> dict[str, Any]:
    """Round k's three-way error report: batch R0k, predicted by M(k-1)."""
    return write_error_report(
        config, round_index=round_index, batch_id=batch_id_for(round_index),
        generation=round_label(round_index - 1), judge_model=JUDGE_MODEL_KEY,
        path=rounds_state_dir(config) / f"round_{round_index:03d}_errors.json")


def mark_stage(config: Any, *, round_index: int, stage: str) -> dict[str, Any]:
    batch = Path(config.public_root) / "batches" / batch_id_for(round_index)
    files = {"judged": ["judge_production_summary.json", "judge_pilot_summary.json"],
             "adjudicated": ["release.json"]}
    if stage not in files:
        raise ContractError(f"only {sorted(files)} are marked by hand; got {stage!r}")
    artifact = next((batch / name for name in files[stage] if (batch / name).is_file()), None)
    if artifact is None:
        raise GateBlocked(f"{stage} has no artifact yet under {batch}")
    state_dir = rounds_state_dir(config)
    state = advance_round(resume_round(state_dir, round_index), to=stage,
                          artifact_sha256=sha256_file(artifact))
    write_round_state(state_dir, state)
    if stage == "adjudicated":
        error_report(config, round_index=round_index)
    return {"round": round_index, "stage": state.stage, "artifact": str(artifact)}


def open_test(config: Any, *, confirm: bool, worker=None, evaluator=None) -> dict[str, Any]:
    if not confirm:
        raise GateBlocked("opening the test split is irreversible; pass --confirm-open-test")
    rounds = int(_plan(config)["rounds"])
    records = []
    for round_index in range(0, rounds + 1):
        path = _model_record_path(config, round_index)
        if not path.is_file():
            raise GateBlocked(f"round {round_index} has no trained model; the test stays closed")
        records.append(json.loads(path.read_text(encoding="utf-8")))
    curve = []
    for record in records:
        output_dir = _loop_dir(config) / "test" / f"round_{record['round']:03d}"
        run = _run_split(config, record["run_id"], split="test", expected=200, output_dir=output_dir,
                         checkpoint=Path(record["selection"]["adapter_path"]), purpose="loop-open-test",
                         worker=worker, evaluator=evaluator)
        result = run["evaluation"]["results"][0]
        curve.append({"round": record["round"], "generation": record["generation"],
                      "training_documents": record["training_documents"],
                      "core_law_article_strict": result["core_law_article_strict"],
                      "docwise_core_accuracy": result["docwise_core_accuracy"]})
    payload = {"schema_version": 1, "opened_at": datetime.now(UTC).isoformat(), "rounds": curve}
    write_json_atomic(_loop_dir(config) / "test" / "learning_curve.json", payload)
    return payload
