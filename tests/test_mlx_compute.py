from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_quality_checker.errors import FingerprintMismatch, IntegrityError
from data_quality_checker.mlx_compute import (
    assert_thinking_is_disabled,
    assert_equivalent_training_state,
    build_snapshot_manifest,
    parse_worker_result,
)
from data_quality_checker.mlx_worker import _cached_validation_record


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


def test_validation_record_resume_requires_the_same_request_fingerprint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "doc_7.json"
    path.write_text(
        json.dumps({"request_fingerprint": "a" * 64, "doc_id": 7}),
        encoding="utf-8",
    )
    assert _cached_validation_record(
        path, request_fingerprint="a" * 64, doc_id=7
    ) == {"request_fingerprint": "a" * 64, "doc_id": 7}
    with pytest.raises(FingerprintMismatch, match="request mismatch"):
        _cached_validation_record(path, request_fingerprint="b" * 64, doc_id=7)


def test_resume_equivalence_compares_every_recoverable_state_component() -> None:
    state = {
        "adapter_tensor_fingerprint": "a",
        "optimizer_tensor_fingerprint": "b",
        "mlx_rng_tensor_fingerprint": "c",
        "scheduler_state": {"global_update": 2},
        "data_cursor": {"micro_steps": 8},
        "python_rng_state_fingerprint": "d",
    }
    assert_equivalent_training_state(state, dict(state))

    changed = dict(state)
    changed["optimizer_tensor_fingerprint"] = "different"
    with pytest.raises(IntegrityError, match="optimizer_tensor_fingerprint"):
        assert_equivalent_training_state(state, changed)
