from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_quality_checker.constants import MODEL_ID, MODEL_REVISION
from data_quality_checker.fingerprints import sha256_file
from data_quality_checker.g0 import final_refit_updates
from data_quality_checker.g0_finalize import (
    build_g0_registry,
    finalize_selection,
    seal_g0,
)


def _write_summary(candidate_root: Path, update: int, f1: float, *, eligible: bool = True) -> None:
    d = candidate_root / "validation" / f"update_{update:07d}"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(
        json.dumps(
            {
                "update": update,
                "coverage_count": 50,
                "parse_count": 50,
                "empty_output_count": 0,
                "runaway_output_count": 0,
                "eligible": eligible,
                "validation_loss": 0.1,
                "core_law_article_strict": {"f1": f1, "recall": f1 - 0.05, "precision": 0.9},
                "docwise_core_accuracy": {"accuracy": 0.3},
            }
        ),
        encoding="utf-8",
    )


def _snapshot(tmp_path: Path) -> Path:
    snap = tmp_path / "snapshot"
    snap.mkdir()
    (snap / "config.json").write_text("{}", encoding="utf-8")
    return snap


def _adapter(tmp_path: Path) -> Path:
    adapter = tmp_path / "checkpoints" / "update_0001003"
    adapter.mkdir(parents=True)
    (adapter / "adapters.safetensors").write_bytes(b"final-refit-weights")
    (adapter / "manifest.json").write_text("{}", encoding="utf-8")
    return adapter


def test_build_g0_registry_satisfies_backend_read_contract(tmp_path: Path) -> None:
    snap = _snapshot(tmp_path)
    adapter = _adapter(tmp_path)
    reg = build_g0_registry(
        model_snapshot_path=snap, adapter_path=adapter, max_sequence_length=12288
    )
    # Exactly the fields + invariants MlxG0Backend.__init__ checks (minus mlx load).
    assert reg["model_id"] == MODEL_ID
    assert reg["model_revision"] == MODEL_REVISION
    assert Path(reg["model_snapshot_path"]).is_dir()
    adapter_file = Path(reg["adapter_path"]) / "adapters.safetensors"
    assert adapter_file.is_file()
    assert sha256_file(adapter_file) == reg["adapter_sha256"]
    assert reg["max_sequence_length"] == 12288
    assert reg["max_generation_tokens"] == 4096


def test_build_g0_registry_rejects_missing_adapter(tmp_path: Path) -> None:
    snap = _snapshot(tmp_path)
    with pytest.raises(Exception):
        build_g0_registry(
            model_snapshot_path=snap,
            adapter_path=tmp_path / "does_not_exist",
            max_sequence_length=8192,
        )


def test_build_g0_registry_rejects_non_directory_snapshot(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    not_a_dir = tmp_path / "snap.txt"
    not_a_dir.write_text("x", encoding="utf-8")
    with pytest.raises(Exception):
        build_g0_registry(
            model_snapshot_path=not_a_dir,
            adapter_path=adapter,
            max_sequence_length=8192,
        )


def test_finalize_selection_picks_best_eligible_and_sizes_refit(tmp_path: Path) -> None:
    root = tmp_path / "development" / "lr2.5e-5-warm42-cos850"
    _write_summary(root, 75, 0.807)
    _write_summary(root, 800, 0.842)  # global best
    _write_summary(root, 850, 0.809)
    result = finalize_selection(root)
    assert result["selected_update"] == 800
    assert result["core_f1"] == 0.842
    assert result["refit_updates"] == final_refit_updates(800)
    assert result["adapter_path"].endswith("checkpoints/update_0000800")


def test_finalize_selection_ignores_ineligible_checkpoints(tmp_path: Path) -> None:
    root = tmp_path / "development" / "cand"
    _write_summary(root, 100, 0.90, eligible=False)  # higher F1 but ineligible
    _write_summary(root, 200, 0.80, eligible=True)
    result = finalize_selection(root)
    assert result["selected_update"] == 200


def test_finalize_selection_raises_without_summaries(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        finalize_selection(tmp_path / "empty")


def test_seal_g0_writes_registry_at_public_g0_path(tmp_path: Path) -> None:
    class _Cfg:
        public_root = tmp_path / "public"

    snap = _snapshot(tmp_path)
    adapter = _adapter(tmp_path)
    path = seal_g0(
        config=_Cfg(),
        model_snapshot_path=snap,
        adapter_path=adapter,
        max_sequence_length=10240,
    )
    assert path == tmp_path / "public" / "g0" / "G0.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["model_id"] == MODEL_ID
    assert loaded["adapter_sha256"] == sha256_file(adapter / "adapters.safetensors")
    assert loaded["max_sequence_length"] == 10240


def _sealing_fixture(tmp_path: Path):
    class _Cfg:
        public_root = tmp_path / "public"

    return _Cfg(), _snapshot(tmp_path), _adapter(tmp_path)


def test_round_label_is_zero_padded_to_three_digits() -> None:
    from data_quality_checker.g0_finalize import round_label

    assert round_label(0) == "M000"
    assert round_label(7) == "M007"
    assert round_label(12) == "M012"


def test_a_negative_round_index_has_no_label() -> None:
    from data_quality_checker.errors import ContractError
    from data_quality_checker.g0_finalize import round_label

    with pytest.raises(ContractError):
        round_label(-1)


def test_sealing_a_round_writes_its_own_registry(tmp_path) -> None:
    from data_quality_checker.g0_finalize import seal_round_model

    config, snapshot, adapter = _sealing_fixture(tmp_path)
    written = seal_round_model(
        config=config,
        round_label="M003",
        model_snapshot_path=snapshot,
        adapter_path=adapter,
        max_sequence_length=12288,
    )
    assert written.name == "M003.json"
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["adapter_sha256"]
    assert payload["max_sequence_length"] == 12288


def test_sealing_a_round_leaves_the_g0_registry_alone(tmp_path) -> None:
    from data_quality_checker.g0_finalize import seal_g0, seal_round_model

    config, snapshot, adapter = _sealing_fixture(tmp_path)
    g0_path = seal_g0(
        config=config,
        model_snapshot_path=snapshot,
        adapter_path=adapter,
        max_sequence_length=12288,
    )
    before = g0_path.read_bytes()
    seal_round_model(
        config=config,
        round_label="M003",
        model_snapshot_path=snapshot,
        adapter_path=adapter,
        max_sequence_length=12288,
    )
    assert g0_path.read_bytes() == before


def test_an_invalid_round_label_is_rejected(tmp_path) -> None:
    from data_quality_checker.errors import ContractError
    from data_quality_checker.g0_finalize import seal_round_model

    config, snapshot, adapter = _sealing_fixture(tmp_path)
    for label in ("M3", "m003", "G0", "M0003", "round-3", ""):
        with pytest.raises(ContractError):
            seal_round_model(
                config=config,
                round_label=label,
                model_snapshot_path=snapshot,
                adapter_path=adapter,
                max_sequence_length=12288,
            )
