from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_quality_checker.errors import ContractError
from data_quality_checker.loop_train import train_round

try:
    from test_g0 import isolated_config as canonical_fixture
except ImportError:  # pragma: no cover - fallback if test_g0 is not importable here
    from data_quality_checker.config import default_config_path, load_config

    def canonical_fixture(tmp_path: Path):
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


SPLIT = {"train": [1, 2, 3], "valid": [10, 11], "test": [20, 21]}


def release_fixture(config, batch_id: str, doc_ids: list[str]) -> None:
    directory = Path(config.sensitive_root) / "releases" / batch_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "expert_adjudicated.jsonl").write_text(
        "".join(
            json.dumps(
                {"internal_doc_id": doc_id, "text": f"text-{doc_id}", "references": []},
                ensure_ascii=False,
            )
            + "\n"
            for doc_id in doc_ids
        ),
        encoding="utf-8",
    )
    (directory / "consensus_clean.jsonl").write_text("", encoding="utf-8")


def test_round_zero_trains_on_canonical_only(tmp_path: Path) -> None:
    config = canonical_fixture(tmp_path)
    result = train_round(config, generation="M000", split=SPLIT, cleaned_batch_ids=[])
    assert result["training_documents"] == 3
    assert result["cleaned_batch_ids"] == []


def test_a_later_round_adds_its_cleaned_documents(tmp_path: Path) -> None:
    config = canonical_fixture(tmp_path)
    release_fixture(config, "round1", ["d1", "d2"])
    result = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"])
    assert result["training_documents"] == 5


def test_validation_and_test_never_grow(tmp_path: Path) -> None:
    config = canonical_fixture(tmp_path)
    release_fixture(config, "round1", ["d1", "d2"])
    base = train_round(config, generation="M000", split=SPLIT, cleaned_batch_ids=[])
    grown = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"])
    assert grown["validation_documents"] == base["validation_documents"] == 2
    assert grown["test_documents"] == base["test_documents"] == 2


def test_two_rounds_get_different_run_directories(tmp_path: Path) -> None:
    config = canonical_fixture(tmp_path)
    release_fixture(config, "round1", ["d1", "d2"])
    first = train_round(config, generation="M000", split=SPLIT, cleaned_batch_ids=[])
    second = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"])
    assert first["run_id"] != second["run_id"]


def test_the_same_round_is_idempotent(tmp_path: Path) -> None:
    config = canonical_fixture(tmp_path)
    release_fixture(config, "round1", ["d1", "d2"])
    first = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"])
    second = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"])
    assert first["run_id"] == second["run_id"]


def test_an_invalid_generation_is_rejected(tmp_path: Path) -> None:
    config = canonical_fixture(tmp_path)
    for generation in ("G0", "m001", "M1", "round1", ""):
        with pytest.raises(ContractError):
            train_round(config, generation=generation, split=SPLIT, cleaned_batch_ids=[])
