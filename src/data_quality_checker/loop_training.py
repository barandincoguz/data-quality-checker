"""Compose round `k`'s training set: canonical train plus cleaned rounds 1..k.

Validation and test are returned unchanged on every round. They are frozen for
the life of the experiment: the loop re-runs checkpoint selection each round,
so validation can never be folded into training, and the test split may never
influence any decision at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .errors import ContractError
from .fingerprints import sha256_file

CANONICAL_PREFIX = "canonical:"


def _canonical_ids(doc_ids: Sequence[int]) -> list[str]:
    return [f"{CANONICAL_PREFIX}{doc_id}" for doc_id in doc_ids]


def compose_round_training_ids(
    split: dict[str, list[int]], cleaned_rounds: Sequence[Sequence[str]]
) -> dict[str, list[str]]:
    train = _canonical_ids(split["train"])
    seen = set(train)
    for round_index, batch in enumerate(cleaned_rounds, start=1):
        for doc_id in batch:
            if doc_id.startswith(CANONICAL_PREFIX):
                raise ContractError(
                    f"round {round_index} document {doc_id!r} uses the reserved canonical prefix"
                )
            if doc_id in seen:
                raise ContractError(
                    f"round {round_index} document {doc_id!r} is already in the training set"
                )
            seen.add(doc_id)
            train.append(doc_id)
    return {
        "train": train,
        "valid": _canonical_ids(split["valid"]),
        "test": _canonical_ids(split["test"]),
    }


def write_round_training_manifest(
    output_dir: Path,
    round_index: int,
    composition: dict[str, list[str]],
    *,
    split_manifest_sha256: str,
) -> dict[str, Any]:
    if round_index < 0:
        raise ContractError("round_index must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "round": round_index,
        "split_manifest_sha256": split_manifest_sha256,
        "counts": {name: len(ids) for name, ids in composition.items()},
        "ids": composition,
    }
    path = output_dir / f"round_{round_index:03d}_training.json"
    write_json_atomic(path, payload)
    return {"path": str(path), "manifest_sha256": sha256_file(path)}
