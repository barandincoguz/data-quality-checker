from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from data_quality_checker.config import default_config_path, load_config
from data_quality_checker.errors import GateBlocked
from data_quality_checker.fingerprints import sha256_file
from data_quality_checker.g0_training import (
    _canonical_evaluate,
    run_development,
    validation_milestones,
)

RUN_ID = "dqcheck_g0_qwen3_5_9b_aaaaaaaaaaaa"


def _fixture_context(tmp_path: Path, *, preflight_passed: bool = True):
    live = load_config()
    config_payload = json.loads(default_config_path().read_text(encoding="utf-8"))
    config_payload.update(
        {
            "canonical_gt_dir": str(live.canonical_gt_dir),
            "example_bank_path": str(live.example_bank_path),
            "reference_split_manifest_path": str(live.reference_split_manifest_path),
            "sensitive_root": str(tmp_path / "sensitive"),
            "public_root": str(tmp_path / "public"),
            "training_runs_root": str(tmp_path / "runs"),
        }
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    config = load_config(config_path)

    data_dir = config.sensitive_root / "g0" / RUN_ID / "data"
    data_dir.mkdir(parents=True)
    row = {
        "messages": [
            {"role": "system", "content": "fixture"},
            {"role": "user", "content": "fixture"},
            {"role": "assistant", "content": "[]"},
        ]
    }
    (data_dir / "train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for _ in range(394)), encoding="utf-8"
    )
    training_view_path = data_dir / "train_context_1024.jsonl"
    training_view_path.write_text(
        (data_dir / "train.jsonl").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (data_dir / "valid.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for _ in range(50)), encoding="utf-8"
    )
    (data_dir / "valid_doc_ids.json").write_text(
        json.dumps(list(range(1, 51))), encoding="utf-8"
    )
    model_path = tmp_path / "model" / ("b" * 40)
    model_path.mkdir(parents=True)
    run_dir = config.training_runs_root / RUN_ID
    run_dir.mkdir(parents=True)
    run_config = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "source_fingerprint": "c" * 64,
        "model": {
            "model_id": "mlx-community/Qwen3.5-9B-MLX-4bit",
            "revision": "b" * 40,
        },
        "model_snapshot": {
            "snapshot_path": str(model_path),
            "fingerprint": "d" * 64,
        },
        "selected_sequence_length": 10240,
        "selected_training_sequence_length": 1024,
        "training_view": {
            "jsonl_path": str(training_view_path),
            "jsonl_sha256": sha256_file(training_view_path),
            "row_count": 394,
        },
        "data_manifest": {"path": str(data_dir)},
        "recovery": {"long_run_allowed": preflight_passed},
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config), encoding="utf-8"
    )
    preflight_path = config.public_root / "g0" / RUN_ID / "preflight.json"
    preflight_path.parent.mkdir(parents=True)
    preflight_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": RUN_ID,
                "status": (
                    "compute_preflight_passed_long_run_not_started"
                    if preflight_passed
                    else "compute_preflight_failed"
                ),
                "long_run_allowed": preflight_passed,
                "long_run_started": False,
            }
        ),
        encoding="utf-8",
    )
    return config


def _evaluation(update: int) -> dict[str, Any]:
    score = update / 100
    return {
        "results": [
            {
                "core_law_article_strict": {
                    "precision": score,
                    "recall": score,
                    "f1": score,
                    "tp": update,
                    "fp": 0,
                    "fn": 0,
                },
                "docwise_core_accuracy": {
                    "threshold": 1.0,
                    "accuracy": score,
                    "passed_doc_count": update // 2,
                    "failed_doc_count": 50 - update // 2,
                    "total_docs": 50,
                },
            }
        ]
    }


def test_validation_milestones_preserve_25_update_boundaries_and_final_295() -> None:
    assert validation_milestones(50) == [25, 50]
    assert validation_milestones(295) == [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 295]
    with pytest.raises(ValueError):
        validation_milestones(296)


def test_dry_run_exposes_every_hidden_training_default(tmp_path: Path) -> None:
    config = _fixture_context(tmp_path)
    plan = run_development(
        config=config,
        run_id=RUN_ID,
        candidate_id="lr5e-5",
        target_updates=50,
    )

    assert plan["status"] == "ready"
    trajectory = plan["contract"]["trajectory"]
    assert trajectory["peak_learning_rate"] == 5e-5
    assert trajectory["rank"] == 8
    assert trajectory["num_layers"] == 16
    assert trajectory["lora_scale"] == 20.0
    assert trajectory["lora_dropout"] == 0.0
    assert trajectory["adam_beta1"] == 0.9
    assert trajectory["adam_beta2"] == 0.999
    assert trajectory["adam_eps"] == 1e-8
    assert trajectory["adam_bias_correction"] is False
    assert trajectory["compile_gradient_step"] is True
    assert trajectory["sequence_length_buckets"] == [1024, 2048, 4096, 8192, 10241]
    assert trajectory["warmup_updates"] == 5
    assert trajectory["total_updates"] == 99
    assert plan["long_run_started"] is False


def test_failed_compute_preflight_blocks_execute_without_writing_candidate(
    tmp_path: Path,
) -> None:
    config = _fixture_context(tmp_path, preflight_passed=False)
    plan = run_development(
        config=config,
        run_id=RUN_ID,
        candidate_id="lr1e-4",
        target_updates=50,
    )
    assert plan["status"] == "blocked"
    assert plan["blockers"]

    with pytest.raises(GateBlocked, match="preflight"):
        run_development(
            config=config,
            run_id=RUN_ID,
            candidate_id="lr1e-4",
            target_updates=50,
            execute=True,
            require_tmux=False,
        )
    assert not (config.training_runs_root / RUN_ID / "development").exists()


def test_validation_wrapper_uses_canonical_evaluator_and_exact_50_doc_universe(
    tmp_path: Path,
) -> None:
    config = load_config()
    split = json.loads(config.reference_split_manifest_path.read_text(encoding="utf-8"))
    doc_ids = split["splits"]["valid"]
    assert len(doc_ids) == 50
    predictions = []
    for doc_id in doc_ids:
        gold = json.loads(
            (config.canonical_gt_dir / f"doc_{doc_id}.json").read_text(encoding="utf-8")
        )
        predictions.append(
            {
                "doc_id": doc_id,
                "status": "success",
                "references": [
                    reference
                    for reference in gold["references"]
                    if str(reference.get("status", "approved")).lower() == "approved"
                ],
            }
        )
    predictions_path = tmp_path / "predictions.json"
    doc_ids_path = tmp_path / "doc_ids.json"
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    doc_ids_path.write_text(json.dumps(doc_ids), encoding="utf-8")

    report = _canonical_evaluate(
        config=config,
        predictions_path=predictions_path,
        doc_ids_path=doc_ids_path,
        output_dir=tmp_path / "evaluation",
    )
    result = report["results"][0]
    assert result["evaluated_doc_count"] == 50
    assert result["reference_postprocess_mode"] == "canonical_set"
    assert result["core_reference_view"] == "row"
    assert result["core_law_article_strict"]["recall"] == 1.0
    assert result["core_law_article_strict"]["tp"] > 0
    assert result["docwise_core_accuracy"]["total_docs"] == 50


def test_pilot_segments_train_validate_and_resume_idempotently(tmp_path: Path) -> None:
    config = _fixture_context(tmp_path)
    calls: list[tuple[str, dict[str, Any]]] = []
    fail_first_validation = True

    def worker_runner(**kwargs):
        nonlocal fail_first_validation
        stage = kwargs["stage"]
        request = kwargs["request"]
        calls.append((stage, request))
        if request["action"] == "train":
            update = int(request["target_updates"])
            checkpoint = Path(request["checkpoint_root"]) / f"update_{update:07d}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            (checkpoint / "adapters.safetensors").write_bytes(f"adapter-{update}".encode())
            (checkpoint / "manifest.json").write_text(
                json.dumps(
                    {
                        "global_update": update,
                        "trainer_state_fingerprint": f"state-{update}",
                    }
                ),
                encoding="utf-8",
            )
            return {"global_update": update, "checkpoint": str(checkpoint)}

        if fail_first_validation:
            fail_first_validation = False
            raise RuntimeError("simulated validation interruption")
        update = int(stage.rsplit("_", 1)[1])
        output_dir = Path(request["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions = output_dir / "predictions.json"
        predictions.write_text("[]", encoding="utf-8")
        manifest = output_dir / "manifest.json"
        manifest.write_text(json.dumps({"update": update}), encoding="utf-8")
        return {
            "coverage_count": 50,
            "parse_count": 50,
            "empty_output_count": 0,
            "zero_reference_output_count": 0,
            "predicted_reference_count": 1,
            "runaway_output_count": 0,
            "validation_loss": 1 / update,
            "predictions_path": str(predictions),
            "predictions_sha256": sha256_file(predictions),
            "manifest_sha256": sha256_file(manifest),
        }

    def evaluator_runner(**kwargs):
        update = int(kwargs["output_dir"].name.rsplit("_", 1)[1])
        report = _evaluation(update)
        (kwargs["output_dir"] / "evaluation.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return report

    def verify_checkpoint(path: Path):
        return json.loads((path / "manifest.json").read_text(encoding="utf-8"))

    with pytest.raises(RuntimeError, match="simulated validation interruption"):
        run_development(
            config=config,
            run_id=RUN_ID,
            candidate_id="lr1e-4",
            target_updates=50,
            execute=True,
            require_tmux=False,
            worker_runner=worker_runner,
            evaluator_runner=evaluator_runner,
            checkpoint_verifier=verify_checkpoint,
        )
    assert [stage for stage, _ in calls] == [
        "train_0000025",
        "validate_0000025",
    ]
    assert (
        config.training_runs_root
        / RUN_ID
        / "development"
        / "lr1e-4"
        / "checkpoints"
        / "update_0000025"
    ).is_dir()

    calls.clear()
    result = run_development(
        config=config,
        run_id=RUN_ID,
        candidate_id="lr1e-4",
        target_updates=50,
        execute=True,
        resume=True,
        require_tmux=False,
        worker_runner=worker_runner,
        evaluator_runner=evaluator_runner,
        checkpoint_verifier=verify_checkpoint,
    )
    assert result["status"] == "pilot_completed"
    assert result["last_validated_update"] == 50
    assert result["best_update"] == 50
    assert [stage for stage, _ in calls] == [
        "validate_0000025",
        "train_0000050",
        "validate_0000050",
    ]
    second_train = calls[1][1]
    assert second_train["resume_checkpoint"].endswith("update_0000025")

    calls.clear()
    resumed = run_development(
        config=config,
        run_id=RUN_ID,
        candidate_id="lr1e-4",
        target_updates=50,
        execute=True,
        resume=True,
        require_tmux=False,
        worker_runner=worker_runner,
        evaluator_runner=evaluator_runner,
        checkpoint_verifier=verify_checkpoint,
    )
    assert resumed["status"] == "pilot_completed"
    assert calls == []
    assert resumed["recovery_history"][-1]["from_validated_update"] == 50


def test_resume_discovers_latest_verified_time_checkpoint(tmp_path: Path) -> None:
    config = _fixture_context(tmp_path)
    phase = "crash"
    resumed_from: list[str | None] = []

    def write_checkpoint(root: str, update: int) -> Path:
        checkpoint = Path(root) / f"update_{update:07d}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "adapters.safetensors").write_bytes(
            f"adapter-{update}".encode()
        )
        (checkpoint / "manifest.json").write_text(
            json.dumps(
                {
                    "global_update": update,
                    "trainer_state_fingerprint": f"state-{update}",
                }
            ),
            encoding="utf-8",
        )
        return checkpoint

    def worker_runner(**kwargs):
        nonlocal phase
        request = kwargs["request"]
        if request["action"] == "train" and phase == "crash":
            write_checkpoint(request["checkpoint_root"], 17)
            phase = "resume"
            raise RuntimeError("simulated crash after timed checkpoint")
        if request["action"] == "train":
            resumed_from.append(request.get("resume_checkpoint"))
            checkpoint = write_checkpoint(request["checkpoint_root"], 25)
            return {"global_update": 25, "checkpoint": str(checkpoint)}
        output_dir = Path(request["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions = output_dir / "predictions.json"
        predictions.write_text("[]", encoding="utf-8")
        manifest = output_dir / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        return {
            "coverage_count": 50,
            "parse_count": 50,
            "empty_output_count": 0,
            "zero_reference_output_count": 0,
            "predicted_reference_count": 1,
            "runaway_output_count": 0,
            "validation_loss": 1.0,
            "predictions_path": str(predictions),
            "predictions_sha256": sha256_file(predictions),
            "manifest_sha256": sha256_file(manifest),
        }

    def evaluator_runner(**kwargs):
        report = _evaluation(25)
        (kwargs["output_dir"] / "evaluation.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return report

    def verify_checkpoint(path: Path):
        return json.loads((path / "manifest.json").read_text(encoding="utf-8"))

    with pytest.raises(RuntimeError, match="timed checkpoint"):
        run_development(
            config=config,
            run_id=RUN_ID,
            candidate_id="lr5e-5",
            target_updates=25,
            execute=True,
            require_tmux=False,
            worker_runner=worker_runner,
            evaluator_runner=evaluator_runner,
            checkpoint_verifier=verify_checkpoint,
        )

    result = run_development(
        config=config,
        run_id=RUN_ID,
        candidate_id="lr5e-5",
        target_updates=25,
        execute=True,
        resume=True,
        require_tmux=False,
        worker_runner=worker_runner,
        evaluator_runner=evaluator_runner,
        checkpoint_verifier=verify_checkpoint,
    )
    assert result["status"] == "segment_completed"
    assert resumed_from == [
        str(
            config.training_runs_root
            / RUN_ID
            / "development"
            / "lr5e-5"
            / "checkpoints"
            / "update_0000017"
        )
    ]
    assert any(
        entry.get("checkpoint_update") == 17
        for entry in result["recovery_history"]
    )
