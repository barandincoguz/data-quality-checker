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

    Documents are grouped by stratum and each group is shuffled
    deterministically, as before. But the groups are no longer dealt from a
    single shared round-robin cursor and then trimmed to size: crossing
    `reference_band` with `quartile x channel x annotator` shatters rare
    strata (a handful of zero-reference documents) into near-singleton
    groups, and an uncoordinated cursor can land several of them in the same
    batch; the trim-and-top-up surplus shuffle that used to follow had no
    stratum awareness at all and could scramble that placement further.

    Instead this is a single capacity-aware greedy pass. Strata are ordered
    rarest first (ties broken by the stratum tuple), so a rare stratum is
    placed while every batch still has room -- precisely what the rare
    stratum needs. Each document then goes to whichever batch that still has
    room (`len(batch) < size`) already holds the fewest documents matching,
    in order, this document's `reference_band`, `text_channel`, `annotator`,
    `length_quartile`, then the batch's current length, then the batch index
    as a final deterministic tie-break. No batch can ever exceed `size`, so
    no trim or top-up step is needed.
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
    band_counts: list[dict[str, int]] = [{} for _ in range(rounds)]
    channel_counts: list[dict[str, int]] = [{} for _ in range(rounds)]
    annotator_counts: list[dict[str, int]] = [{} for _ in range(rounds)]
    quartile_counts: list[dict[str, int]] = [{} for _ in range(rounds)]

    # Sorting the strata (by size, then by the stratum tuple) makes the deal
    # independent of input order; shuffling within a stratum keeps it
    # independent of the corpus's own ordering.
    for stratum in sorted(groups, key=lambda item: (len(groups[item]), item)):
        quartile, channel, annotator, band = stratum
        members = sorted(groups[stratum])
        rng.shuffle(members)
        for doc_id in members:
            candidates = [index for index in range(rounds) if len(batches[index]) < size]
            if not candidates:
                # Every batch is already full. This only happens when the
                # pool has more than `needed` documents; the remainder is
                # simply not used, matching the "at least `needed`" contract
                # checked above.
                continue
            chosen = min(
                candidates,
                key=lambda index: (
                    band_counts[index].get(band, 0),
                    channel_counts[index].get(channel, 0),
                    annotator_counts[index].get(annotator, 0),
                    quartile_counts[index].get(quartile, 0),
                    len(batches[index]),
                    index,
                ),
            )
            batches[chosen].append(doc_id)
            band_counts[chosen][band] = band_counts[chosen].get(band, 0) + 1
            channel_counts[chosen][channel] = channel_counts[chosen].get(channel, 0) + 1
            annotator_counts[chosen][annotator] = annotator_counts[chosen].get(annotator, 0) + 1
            quartile_counts[chosen][quartile] = quartile_counts[chosen].get(quartile, 0) + 1
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
