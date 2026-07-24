"""Deterministic fake trainer used to regression-test stateful resume semantics."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .atomic import write_json_atomic
from .checkpoints import (
    CheckpointManager,
    make_python_rng_state,
    restore_random_state,
)
from .fingerprints import fingerprint_json


@dataclass
class FakeTrainerState:
    weight: float = 0.0
    momentum: float = 0.0
    global_update: int = 0
    cursor: int = 0


class FakeStatefulTrainer:
    """Small optimizer trajectory with every required recovery component."""

    def __init__(self, *, data: list[float], seed: int, checkpoint_root: Path) -> None:
        self.data = data
        self.seed = seed
        self.state = FakeTrainerState()
        self.rng = random.Random(seed)
        self.manager = CheckpointManager(
            checkpoint_root,
            input_fingerprint=fingerprint_json(data),
            config_fingerprint=fingerprint_json({"seed": seed, "lr": 0.05}),
            model_fingerprint=fingerprint_json({"model": "fake-linear-v1"}),
        )

    def _step(self) -> None:
        value = self.data[self.state.cursor % len(self.data)]
        jitter = self.rng.random() * 1e-6
        gradient = (self.state.weight - value) + jitter
        self.state.momentum = 0.9 * self.state.momentum + 0.1 * gradient
        self.state.weight -= 0.05 * self.state.momentum
        self.state.global_update += 1
        self.state.cursor += 1

    def _checkpoint(self) -> Path:
        trainer_state = {
            "global_update": self.state.global_update,
            "data_cursor": self.state.cursor,
            "python_rng_state": make_python_rng_state(self.rng),
            "scheduler_state": {"name": "constant", "step": self.state.global_update},
        }

        def writer(path: Path) -> list[str]:
            write_json_atomic(path / "adapter.json", {"weight": self.state.weight})
            write_json_atomic(path / "optimizer.json", {"momentum": self.state.momentum})
            write_json_atomic(path / "mlx_rng.json", {"fake_state": self.state.global_update})
            return ["model_or_adapter", "optimizer", "scheduler", "mlx_rng"]

        return self.manager.save(
            global_update=self.state.global_update,
            trainer_state=trainer_state,
            writer=writer,
        )

    def run(self, *, target_updates: int, checkpoint_every: int) -> Path:
        latest: Path | None = None
        while self.state.global_update < target_updates:
            self._step()
            if self.state.global_update % checkpoint_every == 0:
                latest = self._checkpoint()
        if latest is None or self.state.global_update % checkpoint_every:
            latest = self._checkpoint()
        return latest

    def resume(self, checkpoint: Path) -> None:
        trainer = self.manager.load_trainer_state(checkpoint)
        adapter = json.loads((checkpoint / "adapter.json").read_text(encoding="utf-8"))
        optimizer = json.loads((checkpoint / "optimizer.json").read_text(encoding="utf-8"))
        self.state = FakeTrainerState(
            weight=float(adapter["weight"]),
            momentum=float(optimizer["momentum"]),
            global_update=int(trainer["global_update"]),
            cursor=int(trainer["data_cursor"]),
        )
        self.rng.setstate(restore_random_state(trainer["python_rng_state"]))

    def trajectory_fingerprint(self) -> str:
        return fingerprint_json(
            {
                "weight": self.state.weight,
                "momentum": self.state.momentum,
                "global_update": self.state.global_update,
                "cursor": self.state.cursor,
                "next_rng": self.rng.random(),
            }
        )
