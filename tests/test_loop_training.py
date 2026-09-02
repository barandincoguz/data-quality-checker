from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_quality_checker.errors import ContractError
from data_quality_checker.loop_training import (
    compose_round_training_ids,
    write_round_training_manifest,
)

SPLIT = {"train": [1, 2, 3], "valid": [10, 11], "test": [20, 21, 22]}


def test_round_zero_is_canonical_train_only() -> None:
    composition = compose_round_training_ids(SPLIT, [])
    assert composition["train"] == ["canonical:1", "canonical:2", "canonical:3"]
    assert composition["valid"] == ["canonical:10", "canonical:11"]
    assert composition["test"] == ["canonical:20", "canonical:21", "canonical:22"]


def test_each_round_appends_its_cleaned_batch() -> None:
    composition = compose_round_training_ids(SPLIT, [["dA", "dB"], ["dC"]])
    assert composition["train"] == [
        "canonical:1",
        "canonical:2",
        "canonical:3",
        "dA",
        "dB",
        "dC",
    ]


def test_validation_and_test_never_grow() -> None:
    base = compose_round_training_ids(SPLIT, [])
    grown = compose_round_training_ids(SPLIT, [["dA"], ["dB"], ["dC"]])
    assert grown["valid"] == base["valid"]
    assert grown["test"] == base["test"]


def test_a_cleaned_document_may_not_enter_twice() -> None:
    with pytest.raises(ContractError):
        compose_round_training_ids(SPLIT, [["dA"], ["dA"]])


def test_a_cleaned_document_may_not_collide_with_a_canonical_id() -> None:
    with pytest.raises(ContractError):
        compose_round_training_ids(SPLIT, [["canonical:1"]])


def test_manifest_records_the_round_and_the_split_seal(tmp_path: Path) -> None:
    composition = compose_round_training_ids(SPLIT, [["dA", "dB"]])
    result = write_round_training_manifest(tmp_path, 1, composition, split_manifest_sha256="abc123")
    payload = json.loads((tmp_path / "round_001_training.json").read_text(encoding="utf-8"))
    assert payload["round"] == 1
    assert payload["split_manifest_sha256"] == "abc123"
    assert payload["batch_manifest_sha256"] is None
    assert payload["counts"] == {"train": 5, "valid": 2, "test": 3}
    assert result["manifest_sha256"]


def test_manifest_records_the_batch_manifest_seal_when_given(tmp_path: Path) -> None:
    composition = compose_round_training_ids(SPLIT, [["dA", "dB"]])
    write_round_training_manifest(
        tmp_path,
        1,
        composition,
        split_manifest_sha256="abc123",
        batch_manifest_sha256="def456",
    )
    payload = json.loads((tmp_path / "round_001_training.json").read_text(encoding="utf-8"))
    assert payload["batch_manifest_sha256"] == "def456"
