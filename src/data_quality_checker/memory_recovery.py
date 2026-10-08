"""Explicit, state-preserving stock → recurrence checkpoint transition.

The original checkpoint and its historical contract remain untouched. A new
checkpoint records the user decision and origin hashes under the new contract.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .checkpoints import CheckpointManager
from .errors import FingerprintMismatch
from .fingerprints import fingerprint_json, sha256_file
from .mlx_stateful import StatefulTrainingConfig


def migrate_checkpoint(
    *,
    source: Path,
    target_root: Path,
    old_training: StatefulTrainingConfig,
    new_training: StatefulTrainingConfig,
    input_fingerprint: str,
    model_fingerprint: str,
    decision: str,
) -> Path:
    if not decision.strip():
        raise FingerprintMismatch("memory recovery requires a recorded user decision")
    old = dict(old_training.__dict__)
    new = dict(new_training.__dict__)
    if old.pop("memory_mode") != "stock" or new.pop("memory_mode") != "recurrence":
        raise FingerprintMismatch("only stock to recurrence memory recovery is supported")
    if fingerprint_json(old) != fingerprint_json(new):
        raise FingerprintMismatch("memory recovery cannot change the training recipe")
    if source.parent.resolve() == target_root.resolve():
        raise FingerprintMismatch("memory recovery must preserve the original checkpoint directory")
    old_manager = CheckpointManager(
        source.parent,
        input_fingerprint=input_fingerprint,
        config_fingerprint=fingerprint_json(old_training.__dict__),
        model_fingerprint=model_fingerprint,
    )
    manifest = old_manager.verify(source)
    state = old_manager.load_trainer_state(source)
    if state.get("checkpoint_boundary") != "optimizer_update" or not state.get(
        "gradient_accumulator_empty"
    ):
        raise FingerprintMismatch("memory recovery requires a safe optimizer-update boundary")
    origin = {
        "source_checkpoint": str(source.resolve()),
        "source_manifest_sha256": sha256_file(source / "manifest.json"),
        "source_config_fingerprint": manifest["config_fingerprint"],
        "source_memory_mode": "stock",
        "resume_memory_mode": "recurrence",
        "decision": decision,
    }
    manager = CheckpointManager(
        target_root,
        input_fingerprint=input_fingerprint,
        config_fingerprint=fingerprint_json(new_training.__dict__),
        model_fingerprint=model_fingerprint,
    )
    target = manager.path_for(int(manifest["global_update"]))
    if target.exists():
        existing = manager.load_trainer_state(target)
        if existing.get("memory_backend_migration") != origin:
            raise FingerprintMismatch("existing recovery checkpoint has another origin or decision")
        return target
    state["memory_backend_migration"] = origin

    def writer(directory: Path) -> list[str]:
        for row in manifest["files"]:
            relative = Path(row["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise FingerprintMismatch("checkpoint component escapes its directory")
            if relative.as_posix() == "trainer_state.json":
                continue
            destination = directory / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, destination)
        return list(manifest["components"])

    return manager.save(
        global_update=int(manifest["global_update"]), trainer_state=state, writer=writer
    )
