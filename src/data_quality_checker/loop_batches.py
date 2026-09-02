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
from .fingerprints import fingerprint_json, sha256_file

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

    A pool larger than `rounds * size` is not dealt as-is: stratifying it
    directly would let a stratum's relative size in the pool distort its
    relative size in the batches (rarest-first can place an entire small
    stratum before a large one gets any room at all, silently re-weighting
    the population the experiment measures). Selection is kept separate from
    dealing instead -- the pool is first reduced to exactly `rounds * size`
    documents via a seeded uniform random sample (over the pool sorted by
    `doc_id`, so the sample does not depend on input order), and only that
    subset is stratified and dealt. The sample is unbiased, so the deal then
    sees exactly the documents it will place and cannot skew anything.
    """
    if rounds <= 0 or size <= 0:
        raise ContractError("rounds and size must both be positive")
    needed = rounds * size
    if len(documents) < needed:
        raise ContractError(f"need at least {needed} documents, got {len(documents)}")

    seen_doc_ids: set[str] = set()
    for document in documents:
        doc_id = document.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            raise ContractError("every document needs a non-empty string doc_id")
        if doc_id in seen_doc_ids:
            raise ContractError(f"duplicate doc_id in the pool: {doc_id!r}")
        seen_doc_ids.add(doc_id)

    pool = sorted(documents, key=lambda document: str(document["doc_id"]))
    if len(pool) > needed:
        pool = random.Random(seed).sample(pool, needed)

    groups: dict[tuple[str, ...], list[str]] = {}
    for document in pool:
        groups.setdefault(_stratum(document), []).append(str(document["doc_id"]))

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
                # Unreachable: `pool` is exactly `rounds * size` documents by
                # this point (an oversized pool was sampled down to that
                # count above), so total capacity and the number of
                # documents left to place always match -- capacity cannot
                # run out before the pool does.
                raise ContractError("internal error: ran out of batch capacity before the pool")
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
    output_dir: Path,
    batches: list[list[str]],
    *,
    seed: int,
    pool_doc_ids: Sequence[str],
) -> dict[str, Any]:
    """Seal the dealt batches, plus enough to prove where they came from.

    `pool_doc_ids` is the full pool `build_round_batches` was given for this
    round, before any oversized-pool sampling -- required, not optional: a
    manifest that cannot see the actual pool must not be assemblable at all,
    rather than quietly reporting the dealt batches' own size as `pool_size`
    with `dropped_count: 0` when documents were in fact dropped. Passing the
    true pool lets a later audit recompute `pool_fingerprint` and confirm the
    batches were dealt from the declared pool, and `dropped_count` records
    how many documents the seeded sample left out without listing which ones
    -- the seed is enough to reproduce that if ever needed. Callers whose
    pool is exactly `rounds * size` (nothing was ever dropped) pass the same
    document IDs they gave `build_round_batches`.

    Being required did not make `pool_doc_ids` truthful on its own: a caller
    could still pass a `pool_doc_ids` that does not actually contain the
    dealt batches (e.g. a stale or unrelated list), which both understates
    `pool_size` relative to what was truly available (yielding a nonsensical
    negative `dropped_count`) and lets `pool_fingerprint` attest to documents
    that were never dealt. So every dealt id must appear in `pool_doc_ids`;
    anything less means the manifest cannot honestly claim the batches were
    dealt from this pool.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dealt_ids = sorted(doc_id for batch in batches for doc_id in batch)
    selected_count = len(dealt_ids)
    pool_ids = sorted(pool_doc_ids)
    pool_size = len(pool_ids)
    if not set(dealt_ids) <= set(pool_ids):
        missing = sorted(set(dealt_ids) - set(pool_ids))
        raise ContractError(
            "pool_doc_ids does not contain every dealt document id "
            f"(missing {len(missing)}, e.g. {missing[:5]!r})"
        )
    payload = {
        "schema_version": 2,
        "seed": seed,
        "rounds": len(batches),
        "size": len(batches[0]) if batches else 0,
        "balance_fields": list(BALANCE_FIELDS),
        "difficulty_balanced": False,
        "pool_size": pool_size,
        "selected_count": selected_count,
        "dropped_count": pool_size - selected_count,
        "pool_fingerprint": fingerprint_json(pool_ids),
        "batches": batches,
    }
    path = output_dir / "round_batches_manifest.json"
    write_json_atomic(path, payload)
    return {"path": str(path), "manifest_sha256": sha256_file(path)}
