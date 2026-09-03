"""Seal a finalized G0 model registry (``public_root/g0/G0.json``).

This is the write-side counterpart to :class:`processing.MlxG0Backend`, which
only ever *reads* the sealed registry. ``build_g0_registry`` produces exactly
the payload the backend validates; ``seal_g0`` writes it atomically to the
canonical path so real G0 inference/routing can run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .constants import MODEL_ID, MODEL_REVISION
from .errors import ContractError, GateBlocked
from .fingerprints import sha256_file
from .g0 import CheckpointCandidate, final_refit_updates, select_checkpoint

ROUND_LABEL_PATTERN = re.compile(r"^M\d{3}\Z")


def finalize_selection(candidate_root: Path) -> dict[str, Any]:
    """Pick the dev-best checkpoint of a candidate and size its final refit.

    Reads every ``validation/update_*/summary.json`` under ``candidate_root``,
    selects the best eligible checkpoint with the same rule the pipeline uses
    (``select_checkpoint``: core-F1 -> docwise -> recall -> -val_loss), and
    returns the selected update, its adapter path, and the number of optimizer
    updates the final all-494 refit should run (``final_refit_updates``).
    """
    summaries: list[dict[str, Any]] = []
    for path in sorted(candidate_root.glob("validation/update_*/summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    if not summaries:
        raise GateBlocked(f"no validation summaries under {candidate_root}")
    candidates = [
        CheckpointCandidate(
            update=int(s["update"]),
            coverage_count=int(s["coverage_count"]),
            parse_count=int(s["parse_count"]),
            empty_output_count=int(s["empty_output_count"]),
            runaway_output_count=int(s["runaway_output_count"]),
            core_f1=float(s["core_law_article_strict"]["f1"]),
            docwise_accuracy=float(s["docwise_core_accuracy"]["accuracy"]),
            recall=float(s["core_law_article_strict"]["recall"]),
            validation_loss=float(s["validation_loss"]),
        )
        for s in summaries
        if s.get("eligible") is True
    ]
    selected = select_checkpoint(candidates)
    return {
        "selected_update": selected.update,
        "core_f1": selected.core_f1,
        "recall": selected.recall,
        "refit_updates": final_refit_updates(selected.update),
        "adapter_path": str(candidate_root / "checkpoints" / f"update_{selected.update:07d}"),
    }


def _adapter_weight_file(adapter_path: Path) -> Path:
    """Resolve the adapters.safetensors the backend will checksum and load."""
    return adapter_path / "adapters.safetensors" if adapter_path.is_dir() else adapter_path


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


def round_label(round_index: int) -> str:
    """The registry label for a round: `M` plus three digits.

    Zero-padded to match the round-state filenames `loop_rounds.py` writes, so
    a round's model and its state sort together and read the same way.
    """
    if round_index < 0:
        raise ContractError(f"round_index must be non-negative, got {round_index}")
    return f"M{round_index:03d}"


def round_registry_path(config: Any, label: str) -> Path:
    if not ROUND_LABEL_PATTERN.match(label):
        raise ContractError(f"invalid round label {label!r}; expected M followed by three digits")
    return Path(config.public_root) / "g0" / f"{label}.json"


def seal_round_model(
    *,
    config: Any,
    # `round_label` shadows the module-level `round_label()` function within
    # this scope; an internal call to `round_label(...)` here would bind to
    # this string and raise `TypeError: 'str' object is not callable`.
    round_label: str,
    model_snapshot_path: Path,
    adapter_path: Path,
    max_sequence_length: int,
    max_generation_tokens: int = 4096,
) -> Path:
    """Seal one round's model into its own registry.

    Each round trains a new adapter, so each round needs its own sealed
    registry rather than overwriting `G0.json`. Keeping them separate is what
    lets a finished round be re-run or audited later against exactly the model
    that produced it.
    """
    registry = build_g0_registry(
        model_snapshot_path=model_snapshot_path,
        adapter_path=adapter_path,
        max_sequence_length=max_sequence_length,
        max_generation_tokens=max_generation_tokens,
    )
    target = round_registry_path(config, round_label)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, registry, mode=0o644)
    return target
