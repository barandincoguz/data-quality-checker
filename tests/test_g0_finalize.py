from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_quality_checker.constants import MODEL_ID, MODEL_REVISION
from data_quality_checker.fingerprints import sha256_file
from data_quality_checker.g0_finalize import build_g0_registry, seal_g0


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
