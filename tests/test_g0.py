from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_quality_checker.config import default_config_path, load_config
from data_quality_checker.constants import CANONICAL_GT_MANIFEST_SHA256, EXEMPLAR_DOC_IDS
from data_quality_checker.errors import GateBlocked
from data_quality_checker.g0 import (
    CheckpointCandidate,
    TrainingContract,
    assert_test_improves_base,
    build_split,
    final_refit_updates,
    repository_manifest_sha256,
    select_checkpoint,
    select_sequence_length,
    train_bootstrap,
    validate_canonical_sources,
)


def isolated_config(tmp_path: Path):
    live = load_config()
    payload = json.loads(default_config_path().read_text(encoding="utf-8"))
    payload.update(
        {
            "canonical_gt_dir": str(live.canonical_gt_dir),
            "example_bank_path": str(live.example_bank_path),
            "reference_split_manifest_path": str(live.reference_split_manifest_path),
            "sensitive_root": str(tmp_path / "sensitive"),
            "public_root": str(tmp_path / "public"),
            "training_runs_root": str(tmp_path / "runs"),
        }
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_config(path)


def test_live_canonical_and_example_bank_contracts_match_locked_artifacts() -> None:
    config = load_config()
    assert repository_manifest_sha256(config.canonical_gt_dir) == CANONICAL_GT_MANIFEST_SHA256

    result = validate_canonical_sources(config)["summary"]
    assert result["canonical_doc_count"] == 500
    assert result["exemplar_doc_ids"] == sorted(EXEMPLAR_DOC_IDS)
    assert result["split_counts"] == {"train": 394, "valid": 50, "test": 50}


def test_seed_42_split_exactly_matches_v2_manifest() -> None:
    expected = json.loads(load_config().reference_split_manifest_path.read_text())["splits"]
    assert build_split(list(range(1, 501)), seed=42) == expected


def test_bootstrap_writes_canonical_only_data_and_fake_resume_gate(tmp_path) -> None:
    result = train_bootstrap(config=isolated_config(tmp_path), generation="G0")

    assert result["status"] == "software_preflight_passed_compute_pending"
    assert result["long_run_started"] is False
    assert result["data_manifest"]["files"]["train"]["count"] == 394
    assert result["data_manifest"]["files"]["valid"]["count"] == 50
    assert result["data_manifest"]["files"]["test"]["count"] == 50
    assert result["fake_failure_resume"]["status"] == "passed"
    assert result["compute_gates"]["real_full_state_failure_resume"] is False

    first_line = json.loads(
        (Path(result["data_manifest"]["path"]) / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert first_line["messages"][1]["content"].endswith("/no_think")
    assert first_line["messages"][2]["role"] == "assistant"


def test_sequence_length_requires_full_coverage_and_memory_pass() -> None:
    calls = []

    def smoke(candidate: int) -> bool:
        calls.append(candidate)
        return candidate >= 12288

    assert select_sequence_length([100, 9000, 11000], memory_smoke=smoke) == 12288
    assert calls == [12288]

    with pytest.raises(GateBlocked):
        select_sequence_length([40000], memory_smoke=lambda _: True)


def test_checkpoint_selection_uses_declared_lexicographic_order_and_gates() -> None:
    candidates = [
        CheckpointCandidate(25, 50, 49, 0, 0, 0.80, 0.30, 0.75, 0.1),
        CheckpointCandidate(50, 50, 50, 0, 0, 0.80, 0.31, 0.70, 0.2),
        CheckpointCandidate(75, 50, 50, 1, 0, 0.99, 0.99, 0.99, 0.01),
    ]
    assert select_checkpoint(candidates).update == 50
    with pytest.raises(GateBlocked):
        select_checkpoint([candidates[2]])


def test_refit_scaling_and_base_improvement_gate() -> None:
    assert final_refit_updates(100) == round(100 * 494 / 394)
    assert TrainingContract().maximum_optimizer_updates == 295
    assert_test_improves_base(base_core_f1=0.5, tuned_core_f1=0.6)
    with pytest.raises(GateBlocked):
        assert_test_improves_base(base_core_f1=0.5, tuned_core_f1=0.5)
