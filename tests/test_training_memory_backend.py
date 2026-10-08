from __future__ import annotations

import json
from pathlib import Path

from test_g0 import isolated_config
from test_loop_train import SPLIT, release_fixture

from data_quality_checker.g0_training import _candidate_training_config
from data_quality_checker.loop_train import train_round
from data_quality_checker.mlx_compute import _training_request


def test_backend_changes_round_identity_but_preserves_bootstrap_and_data(tmp_path, monkeypatch):
    from data_quality_checker import loop_train

    config = isolated_config(tmp_path)
    monkeypatch.setattr(loop_train, "has_loop_settings", lambda _: False)
    release_fixture(config, "round1", ["d1"])
    base = train_round(config, generation="M000", split=SPLIT, cleaned_batch_ids=[])
    stock = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"])
    config.raw["loop"] = {"window_training_memory_mode": "recurrence"}
    optimized = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"])
    assert optimized["run_id"] != stock["run_id"]
    assert optimized["data_manifest"]["files"] == stock["data_manifest"]["files"]
    assert (
        json.loads(Path(optimized["run_config_path"]).read_text())["training_memory_mode"]
        == "recurrence"
    )
    assert (
        train_round(config, generation="M000", split=SPLIT, cleaned_batch_ids=[])["run_id"]
        == base["run_id"]
    )


def test_preflight_and_training_keep_the_same_recipe_except_backend(tmp_path):
    stock = _candidate_training_config(
        peak_learning_rate=2.5e-5, sequence_length=1536, training_rows=5542
    )
    optimized = _candidate_training_config(
        peak_learning_rate=2.5e-5,
        sequence_length=1536,
        training_rows=5542,
        training_memory_mode="recurrence",
    )
    assert {k: v for k, v in stock.__dict__.items() if k != "memory_mode"} == {
        k: v for k, v in optimized.__dict__.items() if k != "memory_mode"
    }
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n")
    request = _training_request(
        model_path=tmp_path,
        train_path=data,
        checkpoint_root=tmp_path / "cp",
        model_fingerprint="model",
        sequence_length=1536,
        target_updates=1,
        memory_mode="recurrence",
    )
    assert request["training_config"]["memory_mode"] == optimized.memory_mode == "recurrence"
    assert request["training_config"]["compile_gradient_step"] is True
