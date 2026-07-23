from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_quality_checker.errors import IntegrityError
from data_quality_checker.mlx_compute import (
    assert_thinking_is_disabled,
    assert_equivalent_training_state,
    build_snapshot_manifest,
    parse_worker_result,
)
from data_quality_checker.mlx_stateful import completion_training_view


def test_qwen_disabled_thinking_requires_the_empty_closed_sentinel() -> None:
    assert_thinking_is_disabled(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    with pytest.raises(IntegrityError, match="thinking-disable sentinel"):
        assert_thinking_is_disabled("<|im_start|>assistant\n<think>\n")
    with pytest.raises(IntegrityError, match="thinking-disable sentinel"):
        assert_thinking_is_disabled(
            "<|im_start|>assistant\n<think>reasoning</think>\n\n"
        )


def test_completion_training_view_excludes_prompt_positions() -> None:
    positions, targets = completion_training_view(
        [10, 11, 12, 13, 14], completion_offset=3
    )
    assert positions == [2, 3, 4]
    assert targets == [13, 14, 0]


def test_snapshot_manifest_is_content_addressed_and_requires_exact_revision(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / ("a" * 40)
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"model_type":"fixture"}', encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"fixture")

    first = build_snapshot_manifest(snapshot, expected_revision="a" * 40)
    second = build_snapshot_manifest(snapshot, expected_revision="a" * 40)
    assert first == second
    assert first["revision"] == "a" * 40
    assert first["fingerprint"]
    assert {row["path"] for row in first["files"]} == {
        "config.json",
        "model.safetensors",
        "tokenizer_config.json",
    }

    with pytest.raises(IntegrityError, match="revision"):
        build_snapshot_manifest(snapshot, expected_revision="b" * 40)


def test_worker_result_must_be_successful_and_match_action(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps({"schema_version": 1, "action": "train", "status": "passed"}),
        encoding="utf-8",
    )
    assert parse_worker_result(result, expected_action="train")["status"] == "passed"

    result.write_text(
        json.dumps({"schema_version": 1, "action": "generate", "status": "passed"}),
        encoding="utf-8",
    )
    with pytest.raises(IntegrityError, match="action"):
        parse_worker_result(result, expected_action="train")

    result.write_text(
        json.dumps({"schema_version": 1, "action": "train", "status": "failed"}),
        encoding="utf-8",
    )
    with pytest.raises(IntegrityError, match="failed"):
        parse_worker_result(result, expected_action="train")


def test_resume_equivalence_compares_every_recoverable_state_component() -> None:
    state = {
        "adapter_sha256": "a",
        "optimizer_sha256": "b",
        "mlx_rng_sha256": "c",
        "scheduler_state": {"global_update": 2},
        "data_cursor": {"micro_steps": 8},
        "python_rng_state_fingerprint": "d",
    }
    assert_equivalent_training_state(state, dict(state))

    changed = dict(state)
    changed["optimizer_sha256"] = "different"
    with pytest.raises(IntegrityError, match="optimizer_sha256"):
        assert_equivalent_training_state(state, changed)
