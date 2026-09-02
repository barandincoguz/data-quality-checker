from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from data_quality_checker.errors import ContractError
from data_quality_checker.loop_batches import (
    LOOP_BATCH_SEED,
    build_round_batches,
    write_batch_manifest,
)

FIXTURE_SEED = 20260902
QUARTILES = ("q1", "q2", "q3", "q4")
ANNOTATORS = tuple(f"annotator_{index:02d}" for index in range(14))
# A few annotators carry far more documents than others, mirroring the real
# corpus rather than an even 1/14 split.
ANNOTATOR_WEIGHTS = (30, 26, 22, 18, 15, 12, 10, 8, 6, 5, 4, 3, 2, 1)
REFERENCE_BAND_RANGES = ((1, 2), (3, 5), (6, 10), (11, 25))
REFERENCE_BAND_WEIGHTS = (50, 30, 15, 5)
ZERO_REFERENCE_COUNT = 9


def make_documents(count: int = 1200) -> list[dict[str, object]]:
    """Build a fixture that mirrors the measured real pool.

    An earlier fixture derived every attribute from `index % n`, which is so
    rigidly periodic that a naive contiguous slice of the sorted doc_ids
    balances every stratum by accident (see
    test_the_fixture_would_catch_a_naive_contiguous_slice below). This one is
    built from a seeded `random.Random` instead: `text_channel` is skewed
    ~64/36 rather than 50/50, the 14 annotators are unevenly loaded,
    `reference_count` is banded non-uniformly with exactly
    `ZERO_REFERENCE_COUNT` zero-reference documents at positions chosen by
    the RNG (not periodic), and `length_quartile` is correlated with
    position (the first quarter of the list is mostly `q1`, and so on) so a
    contiguous slice produces visibly skewed quartiles too.
    """
    rng = random.Random(FIXTURE_SEED)
    quarter_size = max(count // 4, 1)
    zero_reference_positions = set(rng.sample(range(count), ZERO_REFERENCE_COUNT))

    documents: list[dict[str, object]] = []
    for index in range(count):
        position_quartile = min(index // quarter_size, 3)
        length_quartile = (
            QUARTILES[position_quartile] if rng.random() < 0.70 else rng.choice(QUARTILES)
        )
        text_channel = "pdfText" if rng.random() < 0.64 else "htmlText"
        annotator = rng.choices(ANNOTATORS, weights=ANNOTATOR_WEIGHTS, k=1)[0]

        if index in zero_reference_positions:
            reference_count = 0
        else:
            low, high = rng.choices(REFERENCE_BAND_RANGES, weights=REFERENCE_BAND_WEIGHTS, k=1)[0]
            reference_count = rng.randint(low, high)

        documents.append(
            {
                "doc_id": f"d{index:05d}",
                "length_quartile": length_quartile,
                "text_channel": text_channel,
                "annotator": annotator,
                "reference_count": reference_count,
            }
        )
    return documents


def test_the_fixture_would_catch_a_naive_contiguous_slice() -> None:
    """Guard against vacuous guards.

    An earlier fixture was so regular that chopping the sorted ids into runs
    of 100 passed every balance assertion. If that ever becomes true again,
    the balance tests below stop certifying anything, so assert here that a
    naive slice genuinely violates them.
    """
    documents = make_documents()
    by_id = {str(doc["doc_id"]): doc for doc in documents}
    ordered = sorted(by_id)
    naive = [ordered[index * 100 : (index + 1) * 100] for index in range(12)]

    shares = [
        sum(1 for doc_id in batch if by_id[doc_id]["text_channel"] == "pdfText") / len(batch)
        for batch in naive
    ]
    quartile_shares = [
        sum(1 for doc_id in batch if by_id[doc_id]["length_quartile"] == "q1") / len(batch)
        for batch in naive
    ]
    assert (max(shares) - min(shares) > 0.10) or (
        max(quartile_shares) - min(quartile_shares) > 0.10
    ), "fixture is too regular: a naive contiguous slice balances it by accident"


def test_produces_the_requested_shape() -> None:
    batches = build_round_batches(make_documents(), rounds=12, size=100, seed=LOOP_BATCH_SEED)
    assert len(batches) == 12
    assert all(len(batch) == 100 for batch in batches)


def test_batches_are_disjoint() -> None:
    batches = build_round_batches(make_documents(), rounds=12, size=100, seed=LOOP_BATCH_SEED)
    flat = [doc_id for batch in batches for doc_id in batch]
    assert len(flat) == len(set(flat)) == 1200


def test_is_deterministic_for_a_seed() -> None:
    documents = make_documents()
    first = build_round_batches(documents, rounds=12, size=100, seed=LOOP_BATCH_SEED)
    second = build_round_batches(documents, rounds=12, size=100, seed=LOOP_BATCH_SEED)
    assert first == second


def test_input_order_does_not_change_the_result() -> None:
    documents = make_documents()
    shuffled = list(reversed(documents))
    assert build_round_batches(documents, rounds=12, size=100, seed=LOOP_BATCH_SEED) == (
        build_round_batches(shuffled, rounds=12, size=100, seed=LOOP_BATCH_SEED)
    )


def test_text_channel_is_balanced_across_batches() -> None:
    documents = make_documents()
    by_id = {str(doc["doc_id"]): doc for doc in documents}
    batches = build_round_batches(documents, rounds=12, size=100, seed=LOOP_BATCH_SEED)
    shares = [
        sum(1 for doc_id in batch if by_id[doc_id]["text_channel"] == "pdfText") / len(batch)
        for batch in batches
    ]
    assert max(shares) - min(shares) <= 0.10


def test_length_quartile_is_balanced_across_batches() -> None:
    documents = make_documents()
    by_id = {str(doc["doc_id"]): doc for doc in documents}
    batches = build_round_batches(documents, rounds=12, size=100, seed=LOOP_BATCH_SEED)
    for quartile in QUARTILES:
        shares = [
            sum(1 for doc_id in batch if by_id[doc_id]["length_quartile"] == quartile) / len(batch)
            for batch in batches
        ]
        assert max(shares) - min(shares) <= 0.10, f"{quartile} spread too high: {shares}"


def test_annotator_is_balanced_across_batches() -> None:
    documents = make_documents()
    by_id = {str(doc["doc_id"]): doc for doc in documents}
    batches = build_round_batches(documents, rounds=12, size=100, seed=LOOP_BATCH_SEED)
    annotators = {str(doc["annotator"]) for doc in documents}
    for annotator in annotators:
        shares = [
            sum(1 for doc_id in batch if by_id[doc_id]["annotator"] == annotator) / len(batch)
            for batch in batches
        ]
        assert max(shares) - min(shares) <= 0.10, f"annotator {annotator} is unbalanced"


def test_zero_reference_documents_are_spread_not_clumped() -> None:
    documents = make_documents()
    by_id = {str(doc["doc_id"]): doc for doc in documents}
    batches = build_round_batches(documents, rounds=12, size=100, seed=LOOP_BATCH_SEED)
    per_batch = [
        sum(1 for doc_id in batch if by_id[doc_id]["reference_count"] == 0) for batch in batches
    ]
    # 9 zero-reference documents into 12 batches permits at most 1 per batch,
    # so the correct invariant is "as even as possible", not "at most 1":
    # max(per_batch) - min(per_batch) <= 1 is what's actually satisfiable.
    assert max(per_batch) - min(per_batch) <= 1


def test_too_few_documents_is_rejected() -> None:
    with pytest.raises(ContractError):
        build_round_batches(make_documents(50), rounds=12, size=100, seed=LOOP_BATCH_SEED)


def test_a_duplicate_doc_id_in_the_pool_is_rejected() -> None:
    documents = make_documents()
    duplicate = dict(documents[0])
    duplicate["doc_id"] = documents[1]["doc_id"]
    with pytest.raises(ContractError):
        build_round_batches(documents + [duplicate], rounds=12, size=100, seed=LOOP_BATCH_SEED)


def make_skewed_pool(count: int = 2400, q1_count: int = 1300) -> list[dict[str, object]]:
    """A 2400-document pool skewed 1300 `q1` / 1100 `q2`, mirroring exactly
    the pool the reviewer measured. Every other balance field is held
    constant, so the stratum keys reduce to just the two quartiles -- this
    is what let rarest-first place the whole (smaller) `q2` stratum before
    `q1` got any room at all. A fixture that also varies `text_channel`,
    `annotator` and `reference_band` splits both quartiles into many small,
    similarly-sized strata and happens to balance fine even under the old
    bug, so it would not have caught this.
    """
    documents: list[dict[str, object]] = []
    for index in range(count):
        documents.append(
            {
                "doc_id": f"p{index:05d}",
                "length_quartile": "q1" if index < q1_count else "q2",
                "text_channel": "pdfText",
                "annotator": "annotator_00",
                "reference_count": 1,
            }
        )
    return documents


def test_oversized_pool_is_sampled_before_dealing_not_distorted() -> None:
    """Guard against the pool-distortion bug FIX 2 closed.

    A 2400-document pool skewed 1300 `q1` / 1100 `q2` used to have
    rarest-first place the entire smaller `q2` stratum before `q1` got any
    room, so the dealt batches came out at roughly 8% `q1` against a pool
    that is ~54% `q1`, with 1200 documents silently dropped. Selection is now
    separated from dealing: the pool is reduced to exactly `rounds * size`
    documents by a seeded uniform sample first, so the deal sees an unbiased
    subset and cannot skew it further.
    """
    documents = make_skewed_pool()
    pool_q1_share = sum(1 for doc in documents if doc["length_quartile"] == "q1") / len(documents)
    by_id = {str(doc["doc_id"]): doc for doc in documents}
    batches = build_round_batches(documents, rounds=12, size=100, seed=LOOP_BATCH_SEED)
    shares = [
        sum(1 for doc_id in batch if by_id[doc_id]["length_quartile"] == "q1") / len(batch)
        for batch in batches
    ]
    assert max(shares) - min(shares) <= 0.10, f"q1 spread too high across batches: {shares}"
    for share in shares:
        assert abs(share - pool_q1_share) <= 0.10, (
            f"batch q1 share {share} strays too far from the pool's own {pool_q1_share:.3f}"
        )


def test_manifest_records_pool_provenance_when_the_pool_is_oversized(tmp_path: Path) -> None:
    from data_quality_checker.fingerprints import fingerprint_json

    documents = make_skewed_pool()
    batches = build_round_batches(documents, rounds=12, size=100, seed=LOOP_BATCH_SEED)
    pool_doc_ids = [str(doc["doc_id"]) for doc in documents]
    result = write_batch_manifest(
        tmp_path, batches, seed=LOOP_BATCH_SEED, pool_doc_ids=pool_doc_ids
    )
    payload = json.loads((tmp_path / "round_batches_manifest.json").read_text(encoding="utf-8"))
    assert payload["pool_size"] == 2400
    assert payload["selected_count"] == 1200
    assert payload["dropped_count"] == 1200
    assert payload["pool_fingerprint"] == fingerprint_json(sorted(pool_doc_ids))
    assert result["manifest_sha256"]


def test_manifest_seals_itself_and_records_the_seed(tmp_path: Path) -> None:
    documents = make_documents()
    batches = build_round_batches(documents, rounds=12, size=100, seed=LOOP_BATCH_SEED)
    pool_doc_ids = [str(doc["doc_id"]) for doc in documents]
    result = write_batch_manifest(
        tmp_path, batches, seed=LOOP_BATCH_SEED, pool_doc_ids=pool_doc_ids
    )
    payload = json.loads((tmp_path / "round_batches_manifest.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["seed"] == LOOP_BATCH_SEED
    assert payload["rounds"] == 12
    assert payload["size"] == 100
    assert [len(batch) for batch in payload["batches"]] == [100] * 12
    assert payload["pool_size"] == 1200
    assert payload["selected_count"] == 1200
    assert payload["dropped_count"] == 0
    assert payload["pool_fingerprint"]
    assert result["manifest_sha256"]


def test_pool_doc_ids_disjoint_from_the_dealt_batches_is_rejected(tmp_path: Path) -> None:
    """FIX 3 regression: `pool_doc_ids` must actually contain the dealt ids.

    Requiring the argument did not make it truthful: a pool sharing zero ids
    with the dealt set used to be accepted, with `pool_fingerprint` attesting
    to documents that were never dealt. `set(dealt) <= set(pool)` must hold.
    """
    documents = make_documents()
    batches = build_round_batches(documents, rounds=12, size=100, seed=LOOP_BATCH_SEED)
    unrelated_pool = [f"unrelated{index:05d}" for index in range(1200)]
    with pytest.raises(ContractError):
        write_batch_manifest(tmp_path, batches, seed=LOOP_BATCH_SEED, pool_doc_ids=unrelated_pool)


def test_pool_doc_ids_missing_some_dealt_ids_is_rejected(tmp_path: Path) -> None:
    """FIX 3 regression, negative-count case: a too-small `pool_doc_ids`
    (e.g. only 2 ids alongside 4 dealt ids) used to be accepted and reported
    a nonsensical negative `dropped_count`.
    """
    batches = [["a", "b"], ["c", "d"]]
    with pytest.raises(ContractError):
        write_batch_manifest(tmp_path, batches, seed=LOOP_BATCH_SEED, pool_doc_ids=["a", "b"])


def test_pool_doc_ids_is_required(tmp_path: Path) -> None:
    """`pool_doc_ids` must not be assemblable without the true pool.

    Before this fix, omitting `pool_doc_ids` silently defaulted `pool_size`
    and `selected_count` to the dealt batches' own size and computed
    `pool_fingerprint` over the dealt subset -- for an oversized pool this
    reports `pool_size` far below the truth and `dropped_count: 0` when
    documents were in fact dropped, and attests to the wrong fingerprint. A
    manifest that cannot see the pool must not be assemblable at all: the
    oversized-pool reproduction (true `pool_size`/`selected_count`/
    `dropped_count`) is `test_manifest_records_pool_provenance_when_the_pool_is_oversized`
    above, which already exercises the honest, required-argument path.
    """
    batches = build_round_batches(make_documents(), rounds=12, size=100, seed=LOOP_BATCH_SEED)
    with pytest.raises(TypeError):
        write_batch_manifest(tmp_path, batches, seed=LOOP_BATCH_SEED)  # type: ignore[call-arg]
