"""Fixed, balanced round batches for the DQ-Loop.

The twelve batches are drawn once, before the loop runs, and never re-drawn.
Choosing each round's documents adaptively (active learning) would confound
"more cleaned data helps" with "smarter selection helps"; the experiment
claims the former, so selection must carry no signal.

`difficulty` is deliberately NOT a balance key: the pool holds two `Zor`
documents in total, which cannot be spread across twelve batches, and
pretending otherwise would put a meaningless column in the results.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .errors import ContractError
from .fingerprints import sha256_file

LOOP_BATCH_SEED = 20260902
BALANCE_FIELDS = ("length_quartile", "text_channel", "annotator", "reference_band")


def _reference_band(value: Any) -> str:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if count == 0:
        return "zero"
    if count <= 2:
        return "1-2"
    if count <= 5:
        return "3-5"
    if count <= 10:
        return "6-10"
    return "11+"


def _stratum(document: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(document.get("length_quartile", "unknown")),
        str(document.get("text_channel", "unknown")),
        str(document.get("annotator", "unknown")),
        _reference_band(document.get("reference_count")),
    )


def build_round_batches(
    documents: Sequence[dict[str, Any]], *, rounds: int, size: int, seed: int
) -> list[list[str]]:
    """Deal documents into `rounds` batches of exactly `size`, stratified.

    Documents are grouped by stratum, each group is shuffled deterministically,
    then the groups are dealt round-robin across batches. Dealing rather than
    slicing is what spreads a rare stratum -- a group of three zero-reference
    documents lands in three different batches instead of one.
    """
    if rounds <= 0 or size <= 0:
        raise ContractError("rounds and size must both be positive")
    needed = rounds * size
    if len(documents) < needed:
        raise ContractError(f"need at least {needed} documents, got {len(documents)}")

    groups: dict[tuple[str, ...], list[str]] = {}
    for document in documents:
        doc_id = document.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            raise ContractError("every document needs a non-empty string doc_id")
        groups.setdefault(_stratum(document), []).append(doc_id)

    rng = random.Random(seed)
    batches: list[list[str]] = [[] for _ in range(rounds)]
    cursor = 0
    # Sorting the strata makes the deal independent of input order; shuffling
    # within a stratum keeps it independent of the corpus's own ordering.
    for stratum in sorted(groups):
        members = sorted(groups[stratum])
        rng.shuffle(members)
        for doc_id in members:
            batches[cursor % rounds].append(doc_id)
            cursor += 1

    # The deal fills batches evenly but not to an exact `size`; trim the
    # overfull ones and top up the underfull ones from the surplus, keeping
    # every batch's stratum mix as dealt.
    surplus: list[str] = []
    for batch in batches:
        rng.shuffle(batch)
        if len(batch) > size:
            surplus.extend(batch[size:])
            del batch[size:]
    rng.shuffle(surplus)
    for batch in batches:
        while len(batch) < size:
            if not surplus:
                raise ContractError("ran out of documents while balancing batches")
            batch.append(surplus.pop())
    return [sorted(batch) for batch in batches]


def write_batch_manifest(
    output_dir: Path, batches: list[list[str]], *, seed: int
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "seed": seed,
        "rounds": len(batches),
        "size": len(batches[0]) if batches else 0,
        "balance_fields": list(BALANCE_FIELDS),
        "difficulty_balanced": False,
        "batches": batches,
    }
    path = output_dir / "round_batches_manifest.json"
    write_json_atomic(path, payload)
    return {"path": str(path), "manifest_sha256": sha256_file(path)}
