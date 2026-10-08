from dataclasses import replace

import pytest

from data_quality_checker.checkpoints import CheckpointManager
from data_quality_checker.errors import FingerprintMismatch
from data_quality_checker.fingerprints import fingerprint_json, sha256_file
from data_quality_checker.mlx_stateful import StatefulTrainingConfig


def _checkpoint(tmp_path, config):
    manager = CheckpointManager(
        tmp_path / "old",
        input_fingerprint="data",
        config_fingerprint=fingerprint_json(config.__dict__),
        model_fingerprint="model",
    )

    def writer(path):
        for name in ["adapters.bin", "optimizer.bin", "mlx_rng.bin"]:
            (path / name).write_bytes(name.encode())
        return ["model_or_adapter", "optimizer", "mlx_rng", "scheduler"]

    return manager.save(
        global_update=75,
        trainer_state={
            "data_cursor": {"position": 300},
            "python_rng_state": [42],
            "scheduler_state": {"global_update": 75},
            "checkpoint_boundary": "optimizer_update",
            "gradient_accumulator_empty": True,
        },
        writer=writer,
    )


def test_migration_is_verified_idempotent_and_preserves_source_and_state(tmp_path):
    from data_quality_checker.memory_recovery import migrate_checkpoint

    old = StatefulTrainingConfig()
    new = replace(old, memory_mode="recurrence")
    source = _checkpoint(tmp_path, old)
    before = {p.name: sha256_file(p) for p in source.iterdir()}
    args = dict(
        source=source,
        target_root=tmp_path / "new",
        old_training=old,
        new_training=new,
        input_fingerprint="data",
        model_fingerprint="model",
        decision="Düşük bellekli yola geç",
    )
    target = migrate_checkpoint(**args)
    assert migrate_checkpoint(**args) == target
    assert {p.name: sha256_file(p) for p in source.iterdir()} == before
    manager = CheckpointManager(
        target.parent,
        input_fingerprint="data",
        config_fingerprint=fingerprint_json(new.__dict__),
        model_fingerprint="model",
    )
    state = manager.load_trainer_state(target)
    assert state["data_cursor"] == {"position": 300}
    assert state["scheduler_state"] == {"global_update": 75}
    assert state["python_rng_state"] == [42]
    for name in ["adapters.bin", "optimizer.bin", "mlx_rng.bin"]:
        assert sha256_file(target / name) == before[name]


def test_migration_refuses_changed_recipe_or_input(tmp_path):
    from data_quality_checker.memory_recovery import migrate_checkpoint

    old = StatefulTrainingConfig()
    source = _checkpoint(tmp_path, old)
    args = dict(
        source=source,
        target_root=tmp_path / "new",
        old_training=old,
        input_fingerprint="data",
        model_fingerprint="model",
        decision="yes",
    )
    with pytest.raises(FingerprintMismatch):
        migrate_checkpoint(**args, new_training=replace(old, memory_mode="recurrence", seed=1))
    args["input_fingerprint"] = "different"
    with pytest.raises(FingerprintMismatch):
        migrate_checkpoint(**args, new_training=replace(old, memory_mode="recurrence"))
