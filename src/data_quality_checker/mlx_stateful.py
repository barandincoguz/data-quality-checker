"""Lazy-imported MLX LoRA trainer with real full-state checkpoint/restore.

This module deliberately does not import MLX at module import time. Default
software tests and CLI help therefore remain usable on non-Apple hosts.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

from .atomic import write_json_atomic
from .checkpoints import (
    CheckpointManager,
    make_python_rng_state,
    restore_random_state,
)
from .errors import ContractError, GateBlocked, IntegrityError
from .fingerprints import fingerprint_json, sha256_file


@dataclass(frozen=True)
class StatefulTrainingConfig:
    seed: int = 42
    rank: int = 8
    num_layers: int = 16
    batch_size: int = 1
    gradient_accumulation: int = 4
    peak_learning_rate: float = 1e-4
    end_learning_rate: float = 1e-5
    warmup_updates: int = 15
    total_updates: int = 295
    max_sequence_length: int = 8192
    gradient_checkpointing: bool = True
    checkpoint_every_updates: int = 25
    checkpoint_max_seconds: int = 600


class NoThinkChatDataset:
    def __init__(self, path: Path, tokenizer: Any) -> None:
        self.path = path.resolve()
        self.rows = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.rows:
            raise ContractError(f"training data is empty: {path}")
        self.items: list[tuple[list[int], int]] = []
        for row_index, row in enumerate(self.rows):
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ContractError(f"messages missing at {path}:{row_index + 1}")
            if messages[-1].get("role") != "assistant":
                raise ContractError(f"assistant target missing at {path}:{row_index + 1}")
            tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
                return_dict=False,
            )
            offset = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
            if not isinstance(tokens, list) or not isinstance(offset, list):
                raise ContractError("tokenizer chat template must return token lists")
            if len(tokens) <= len(offset):
                raise ContractError(f"completion-only target is empty at {path}:{row_index + 1}")
            self.items.append(([int(token) for token in tokens], len(offset)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.items[index]

    @property
    def token_counts(self) -> list[int]:
        return [len(tokens) for tokens, _ in self.items]


def completion_training_view(
    tokens: list[int], *, completion_offset: int
) -> tuple[list[int], list[int]]:
    """Return hidden-state positions and shifted targets for completion loss."""

    if not 1 <= completion_offset < len(tokens):
        raise ContractError("completion offset is outside the token sequence")
    positions = list(range(completion_offset - 1, len(tokens)))
    shifted = tokens[1:] + [0]
    return positions, [shifted[position] for position in positions]


class DeterministicCursor:
    def __init__(self, *, size: int, seed: int) -> None:
        if size <= 0:
            raise ValueError("dataset size must be positive")
        self.size = size
        self.seed = seed
        self.epoch = 0
        self.position = 0
        self.micro_steps = 0
        self._order = self._order_for_epoch(0)

    def _order_for_epoch(self, epoch: int) -> list[int]:
        order = list(range(self.size))
        random.Random(self.seed + epoch * 1_000_003).shuffle(order)
        return order

    def next_index(self) -> int:
        if self.position >= self.size:
            self.epoch += 1
            self.position = 0
            self._order = self._order_for_epoch(self.epoch)
        index = self._order[self.position]
        self.position += 1
        self.micro_steps += 1
        return index

    def state_dict(self) -> dict[str, int]:
        return {
            "size": self.size,
            "seed": self.seed,
            "epoch": self.epoch,
            "position": self.position,
            "micro_steps": self.micro_steps,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if int(payload.get("size", -1)) != self.size or int(payload.get("seed", -1)) != self.seed:
            raise IntegrityError("data cursor dataset size/seed mismatch")
        self.epoch = int(payload["epoch"])
        self.position = int(payload["position"])
        self.micro_steps = int(payload["micro_steps"])
        if not 0 <= self.position <= self.size:
            raise IntegrityError("data cursor position is out of range")
        self._order = self._order_for_epoch(self.epoch)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class StatefulMlxTrainer:
    def __init__(
        self,
        *,
        model_path: Path,
        train_path: Path,
        checkpoint_root: Path,
        config: StatefulTrainingConfig,
        input_fingerprint: str,
        model_fingerprint: str,
    ) -> None:
        if config.batch_size != 1:
            raise ContractError("v1 stateful trainer supports only batch_size=1")
        self.model_path = model_path.resolve()
        self.train_path = train_path.resolve()
        self.config = config
        self.input_fingerprint = input_fingerprint
        self.model_fingerprint = model_fingerprint
        self.config_fingerprint = fingerprint_json(config.__dict__)
        self.manager = CheckpointManager(
            checkpoint_root,
            input_fingerprint=input_fingerprint,
            config_fingerprint=self.config_fingerprint,
            model_fingerprint=model_fingerprint,
        )
        self.global_update = 0
        self.python_rng = random.Random(config.seed)
        self.last_checkpoint_time = time.monotonic()
        self._load_runtime()

    def _load_runtime(self) -> None:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_map
        from mlx_lm import load
        from mlx_lm.tuner.trainer import grad_checkpoint
        from mlx_lm.tuner.utils import linear_to_lora_layers

        mx.random.seed(self.config.seed)
        self.model, self.tokenizer = load(
            str(self.model_path), tokenizer_config={"trust_remote_code": True}
        )
        self.model.freeze()
        self.lora_parameters = {"rank": self.config.rank, "dropout": 0.0, "scale": 20.0}
        linear_to_lora_layers(
            self.model,
            self.config.num_layers,
            self.lora_parameters,
            use_dora=False,
        )
        if self.config.gradient_checkpointing:
            if not getattr(self.model, "layers", None):
                raise ContractError("model exposes no layers for gradient checkpointing")
            # mlx-lm patches the transformer-layer class, so one layer enables
            # recomputation for every block of the same class.
            grad_checkpoint(self.model.layers[0])
        self.dataset = NoThinkChatDataset(self.train_path, self.tokenizer)
        if max(self.dataset.token_counts) > self.config.max_sequence_length:
            raise GateBlocked(
                f"training example tokens={max(self.dataset.token_counts)} exceed "
                f"max_sequence_length={self.config.max_sequence_length}; truncation forbidden"
            )
        self.cursor = DeterministicCursor(size=len(self.dataset), seed=self.config.seed)
        warmup = optim.linear_schedule(
            1e-6, self.config.peak_learning_rate, self.config.warmup_updates
        )
        decay_steps = max(1, self.config.total_updates - self.config.warmup_updates)
        decay = optim.cosine_decay(
            self.config.peak_learning_rate,
            decay_steps,
            self.config.end_learning_rate,
        )
        schedule = optim.join_schedules([warmup, decay], [self.config.warmup_updates])
        self.optimizer = optim.Adam(learning_rate=schedule)

        def completion_loss(
            model: Any,
            batch: Any,
            positions: Any,
            completion_targets: Any,
        ) -> tuple[Any, Any]:
            inputs = batch[:, :-1]
            text_model = model.language_model
            hidden = text_model.model(inputs)
            selected_hidden = mx.take(hidden, positions, axis=1)
            if text_model.args.tie_word_embeddings:
                logits = text_model.model.embed_tokens.as_linear(selected_hidden)
            else:
                logits = text_model.lm_head(selected_hidden)
            ce = nn.losses.cross_entropy(logits, completion_targets)
            ntoks = mx.array(completion_targets.size)
            return ce.astype(mx.float32).mean(), ntoks

        value_and_grad = nn.value_and_grad(self.model, completion_loss)
        state = [self.model.state, self.optimizer.state, mx.random.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def step(
            batch: Any,
            positions: Any,
            completion_targets: Any,
            previous: Any,
            do_update: bool,
        ) -> tuple[Any, Any, Any]:
            (loss, tokens), gradient = value_and_grad(
                self.model, batch, positions, completion_targets
            )
            if previous is not None:
                gradient = tree_map(lambda left, right: left + right, gradient, previous)
            if do_update:
                gradient = tree_map(
                    lambda value: value / self.config.gradient_accumulation, gradient
                )
                self.optimizer.update(self.model, gradient)
                gradient = None
            return loss, tokens, gradient

        self._mx = mx
        self._tree_map = tree_map
        self._step = step
        self._compiled_state = state
        self.model.train()

    def _batch(self, dataset_index: int) -> tuple[Any, Any, Any]:
        import numpy as np

        tokens, offset = self.dataset[dataset_index]
        # One extra slot mirrors mlx-lm's completion-loss batch contract while
        # avoiding round-up beyond a boundary-sized sequence candidate.
        padded_length = len(tokens) + 1
        array = np.zeros((1, padded_length), dtype=np.int32)
        array[0, : len(tokens)] = tokens
        positions, targets = completion_training_view(
            tokens, completion_offset=offset
        )
        return (
            self._mx.array(array),
            self._mx.array(positions, dtype=self._mx.int32),
            self._mx.array([targets], dtype=self._mx.int32),
        )

    def _save_checkpoint(self) -> Path:
        mx = self._mx
        from mlx.utils import tree_flatten

        trainer_state = {
            "global_update": self.global_update,
            "data_cursor": self.cursor.state_dict(),
            "python_rng_state": make_python_rng_state(self.python_rng),
            "scheduler_state": {
                "global_update": self.global_update,
                "peak_learning_rate": self.config.peak_learning_rate,
                "end_learning_rate": self.config.end_learning_rate,
                "warmup_updates": self.config.warmup_updates,
                "total_updates": self.config.total_updates,
            },
            "checkpoint_boundary": "optimizer_update",
            "gradient_accumulator_empty": True,
        }

        def writer(path: Path) -> list[str]:
            adapter_path = path / "adapters.safetensors"
            optimizer_path = path / "optimizer.safetensors"
            mlx_rng_path = path / "mlx_rng.safetensors"
            adapter = dict(tree_flatten(self.model.trainable_parameters()))
            optimizer_state = {
                key: value
                for key, value in tree_flatten(self.optimizer.state)
                if isinstance(value, mx.array)
            }
            mlx_rng_state = {
                key: value
                for key, value in tree_flatten(mx.random.state)
                if isinstance(value, mx.array)
            }
            if not adapter or not optimizer_state or not mlx_rng_state:
                raise IntegrityError("MLX checkpoint state unexpectedly flattened to empty")
            mx.save_safetensors(str(adapter_path), adapter)
            mx.save_safetensors(str(optimizer_path), optimizer_state)
            mx.save_safetensors(str(mlx_rng_path), mlx_rng_state)
            adapter_config = {
                "model": str(self.model_path),
                "fine_tune_type": "lora",
                "num_layers": self.config.num_layers,
                "lora_parameters": self.lora_parameters,
            }
            write_json_atomic(path / "adapter_config.json", adapter_config)
            for candidate in (
                adapter_path,
                optimizer_path,
                mlx_rng_path,
                path / "adapter_config.json",
            ):
                _fsync_file(candidate)
            return ["model_or_adapter", "optimizer", "scheduler", "mlx_rng"]

        checkpoint = self.manager.save(
            global_update=self.global_update,
            trainer_state=trainer_state,
            writer=writer,
        )
        self.last_checkpoint_time = time.monotonic()
        return checkpoint

    def resume(self, checkpoint: Path) -> None:
        from mlx.utils import tree_unflatten

        trainer_state = self.manager.load_trainer_state(checkpoint)
        if trainer_state.get("checkpoint_boundary") != "optimizer_update" or not trainer_state.get(
            "gradient_accumulator_empty"
        ):
            raise IntegrityError("checkpoint is not at a safe optimizer-update boundary")
        self.model.load_weights(str(checkpoint / "adapters.safetensors"), strict=False)
        optimizer_flat = self._mx.load(str(checkpoint / "optimizer.safetensors"))
        rng_flat = self._mx.load(str(checkpoint / "mlx_rng.safetensors"))
        self.optimizer.state = tree_unflatten(optimizer_flat)
        self._mx.random.state = tree_unflatten(rng_flat)
        # The compiled function registers these mutable trees as explicit
        # state. Repoint those registrations after restoring new tree objects.
        self._compiled_state[1] = self.optimizer.state
        self._compiled_state[2] = self._mx.random.state
        self.global_update = int(trainer_state["global_update"])
        self.cursor.load_state_dict(trainer_state["data_cursor"])
        self.python_rng.setstate(restore_random_state(trainer_state["python_rng_state"]))
        self._mx.eval(self.model.state, self.optimizer.state, self._mx.random.state)
        self.last_checkpoint_time = time.monotonic()

    def train(
        self,
        *,
        target_updates: int,
        on_checkpoint: Callable[[Path, int], None] | None = None,
    ) -> dict[str, Any]:
        if target_updates <= self.global_update:
            raise ValueError("target_updates must exceed the restored global update")
        gradient_accumulator = None
        accumulated = 0
        losses: list[float] = []
        trained_tokens = 0
        latest_checkpoint: Path | None = None
        while self.global_update < target_updates:
            index = self.cursor.next_index()
            batch, positions, completion_targets = self._batch(index)
            accumulated += 1
            do_update = accumulated == self.config.gradient_accumulation
            loss, token_count, gradient_accumulator = self._step(
                batch,
                positions,
                completion_targets,
                gradient_accumulator,
                do_update,
            )
            self._mx.eval(
                self._compiled_state,
                loss,
                token_count,
                gradient_accumulator,
            )
            losses.append(float(loss.item()))
            trained_tokens += int(token_count.item())
            if not do_update:
                continue
            accumulated = 0
            self.global_update += 1
            due_count = (
                self.global_update % self.config.checkpoint_every_updates == 0
            )
            due_time = (
                time.monotonic() - self.last_checkpoint_time
                >= self.config.checkpoint_max_seconds
            )
            if due_count or due_time or self.global_update == target_updates:
                latest_checkpoint = self._save_checkpoint()
                if on_checkpoint is not None:
                    on_checkpoint(latest_checkpoint, self.global_update)
        if gradient_accumulator is not None:
            raise IntegrityError("training stopped with a non-empty gradient accumulator")
        assert latest_checkpoint is not None
        return {
            "global_update": self.global_update,
            "micro_steps": self.cursor.micro_steps,
            "mean_loss": sum(losses) / len(losses),
            "trained_tokens": trained_tokens,
            "checkpoint": str(latest_checkpoint),
            "checkpoint_manifest_sha256": sha256_file(
                latest_checkpoint / "manifest.json"
            ),
        }

    def adapter_finite(self, checkpoint: Path) -> bool:
        tensors = self._mx.load(str(checkpoint / "adapters.safetensors"))
        return all(bool(self._mx.all(self._mx.isfinite(value)).item()) for value in tensors.values())
