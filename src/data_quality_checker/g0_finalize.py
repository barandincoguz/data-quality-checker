"""Seal a finalized G0 model registry (``public_root/g0/G0.json``).

This is the write-side counterpart to :class:`processing.MlxG0Backend`, which
only ever *reads* the sealed registry. ``build_g0_registry`` produces exactly
the payload the backend validates; ``seal_g0`` writes it atomically to the
canonical path so real G0 inference/routing can run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .constants import MODEL_ID, MODEL_REVISION
from .errors import GateBlocked
from .fingerprints import sha256_file


def _adapter_weight_file(adapter_path: Path) -> Path:
    """Resolve the adapters.safetensors the backend will checksum and load."""
    return (
        adapter_path / "adapters.safetensors"
        if adapter_path.is_dir()
        else adapter_path
    )


def build_g0_registry(
    *,
    model_snapshot_path: Path,
    adapter_path: Path,
    max_sequence_length: int,
    max_generation_tokens: int = 4096,
) -> dict[str, Any]:
    """Build the sealed G0 registry payload.

    Mirrors ``MlxG0Backend.__init__`` invariants exactly: the model id/revision
    are the frozen v1 constants; the snapshot is a directory; the adapter holds
    ``adapters.safetensors``; ``adapter_sha256`` is the checksum of that exact
    file the backend will re-verify before loading.
    """
    snapshot = model_snapshot_path.resolve()
    adapter = adapter_path.resolve()
    if not snapshot.is_dir():
        raise GateBlocked(f"model snapshot is not a directory: {snapshot}")
    adapter_file = _adapter_weight_file(adapter)
    if not adapter_file.is_file():
        raise GateBlocked(f"adapter weights are missing: {adapter_file}")
    if int(max_sequence_length) <= 0:
        raise ValueError("max_sequence_length must be positive")
    if int(max_generation_tokens) <= 0:
        raise ValueError("max_generation_tokens must be positive")
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_snapshot_path": str(snapshot),
        "adapter_path": str(adapter),
        "adapter_sha256": sha256_file(adapter_file),
        "max_sequence_length": int(max_sequence_length),
        "max_generation_tokens": int(max_generation_tokens),
    }


def seal_g0(
    *,
    config: Any,
    model_snapshot_path: Path,
    adapter_path: Path,
    max_sequence_length: int,
    max_generation_tokens: int = 4096,
) -> Path:
    """Write the sealed G0 registry to ``public_root/g0/G0.json`` atomically.

    Returns the path written. The registry is validated by construction; the
    backend re-verifies the adapter checksum and loads the model at inference
    time, so a mismatch fails closed there too.
    """
    registry = build_g0_registry(
        model_snapshot_path=model_snapshot_path,
        adapter_path=adapter_path,
        max_sequence_length=max_sequence_length,
        max_generation_tokens=max_generation_tokens,
    )
    target = Path(config.public_root) / "g0" / "G0.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, registry, mode=0o644)
    return target
