from __future__ import annotations

import pytest

from data_quality_checker.checkpoints import CheckpointManager
from data_quality_checker.errors import IntegrityError
from data_quality_checker.fake_training import FakeStatefulTrainer


def test_uninterrupted_and_crash_resume_trajectories_are_identical(tmp_path) -> None:
    data = [0.1, 0.4, -0.2]
    direct = FakeStatefulTrainer(data=data, seed=42, checkpoint_root=tmp_path / "direct")
    direct.run(target_updates=10, checkpoint_every=5)

    interrupted = FakeStatefulTrainer(data=data, seed=42, checkpoint_root=tmp_path / "resumed")
    checkpoint = interrupted.run(target_updates=5, checkpoint_every=5)
    resumed = FakeStatefulTrainer(data=data, seed=42, checkpoint_root=tmp_path / "resumed")
    resumed.resume(checkpoint)
    resumed.run(target_updates=10, checkpoint_every=5)

    assert resumed.trajectory_fingerprint() == direct.trajectory_fingerprint()


def test_checkpoint_tampering_is_detected(tmp_path) -> None:
    trainer = FakeStatefulTrainer(data=[1.0], seed=42, checkpoint_root=tmp_path / "run")
    checkpoint = trainer.run(target_updates=2, checkpoint_every=2)
    (checkpoint / "adapter.json").write_text('{"weight":999}\n', encoding="utf-8")

    with pytest.raises(IntegrityError, match="size mismatch|SHA256 mismatch"):
        trainer.manager.verify(checkpoint)


def test_missing_state_component_cannot_be_labelled_stateful(tmp_path) -> None:
    manager = CheckpointManager(
        tmp_path / "checkpoints",
        input_fingerprint="a" * 64,
        config_fingerprint="b" * 64,
        model_fingerprint="c" * 64,
    )

    def incomplete_writer(path):
        (path / "adapter.json").write_text("{}\n", encoding="utf-8")
        return ["model_or_adapter"]

    with pytest.raises(IntegrityError, match="full state"):
        manager.save(
            global_update=1,
            trainer_state={
                "data_cursor": 1,
                "python_rng_state": [],
                "scheduler_state": {},
            },
            writer=incomplete_writer,
        )
