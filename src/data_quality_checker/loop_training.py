"""Compose round `k`'s training set: canonical train plus cleaned rounds 1..k.

Validation and test are returned unchanged on every round. They are frozen for
the life of the experiment: the loop re-runs checkpoint selection each round,
so validation can never be folded into training, and the test split may never
influence any decision at all.
"""

from __future__ import annotations

import json
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
    batch_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    if round_index < 0:
        raise ContractError("round_index must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "round": round_index,
        "split_manifest_sha256": split_manifest_sha256,
        "batch_manifest_sha256": batch_manifest_sha256,
        "counts": {name: len(ids) for name, ids in composition.items()},
        "ids": composition,
    }
    path = output_dir / f"round_{round_index:03d}_training.json"
    write_json_atomic(path, payload)
    return {"path": str(path), "manifest_sha256": sha256_file(path)}


def round_training_documents(config: Any, *, batch_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Read finished rounds' releases into training rows.

    A release already carries each document's text and the references the
    expert settled on, which is exactly what a training row needs. Only the
    two trustworthy tiers are read: `expert_adjudicated`, where a human decided,
    and `consensus_clean`, where annotator and model agreed and the round's
    GREEN sample raised nothing. Quarantined documents are excluded -- training
    on documents the pipeline could not process would poison the very data this
    loop exists to clean.
    """
    documents: dict[str, dict[str, Any]] = {}
    for batch_id in batch_ids:
        directory = Path(config.sensitive_root) / "releases" / batch_id
        found = False
        batch_document_count = 0
        for name in ("expert_adjudicated", "consensus_clean"):
            path = directory / f"{name}.jsonl"
            if not path.is_file():
                continue
            found = True
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ContractError(f"{path}:{line_number} is not JSON: {exc}") from exc
                doc_id = row.get("internal_doc_id")
                text = row.get("text")
                references = row.get("references")
                if not isinstance(doc_id, str) or not doc_id:
                    raise ContractError(f"{path}:{line_number} has no internal_doc_id")
                if not isinstance(text, str) or not text:
                    raise ContractError(f"{path}:{line_number} ({doc_id}) has no text")
                if not isinstance(references, list):
                    raise ContractError(f"{path}:{line_number} ({doc_id}) has no references list")
                if doc_id in documents:
                    raise ContractError(
                        f"document {doc_id!r} is duplicated while assembling "
                        f"batch {batch_id!r}'s release"
                    )
                documents[doc_id] = {"text": text, "references": references}
                batch_document_count += 1
        if not found:
            raise ContractError(f"no release found for batch {batch_id!r} under {directory}")
        if batch_document_count == 0:
            raise ContractError(
                f"batch {batch_id!r} release contains zero training documents "
                "-- every document was quarantined or excluded"
            )
    return documents
