from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_quality_checker.constants import EXEMPLAR_DOC_IDS
from data_quality_checker.errors import ContractError
from data_quality_checker.loop_split import (
    LOOP_SPLIT_SEED,
    LoopSplitSizes,
    build_loop_split,
    write_loop_split_manifest,
)

ALL_CANONICAL = list(range(1, 501))


def test_split_sizes_match_the_contract() -> None:
    split = build_loop_split(ALL_CANONICAL, sizes=LoopSplitSizes(), seed=LOOP_SPLIT_SEED)
    assert len(split["train"]) == 300
    assert len(split["valid"]) == 50
    assert len(split["test"]) == 150


def test_splits_are_disjoint_and_cover_the_whole_corpus() -> None:
    split = build_loop_split(ALL_CANONICAL, sizes=LoopSplitSizes(), seed=LOOP_SPLIT_SEED)
    train, valid, test = set(split["train"]), set(split["valid"]), set(split["test"])
    assert train & valid == set()
    assert train & test == set()
    assert valid & test == set()
    assert train | valid | test == set(ALL_CANONICAL)


def test_exemplars_are_in_train_and_never_in_the_held_out_sets() -> None:
    split = build_loop_split(ALL_CANONICAL, sizes=LoopSplitSizes(), seed=LOOP_SPLIT_SEED)
    assert EXEMPLAR_DOC_IDS <= set(split["train"])
    assert set(split["valid"]) & EXEMPLAR_DOC_IDS == set()
    assert set(split["test"]) & EXEMPLAR_DOC_IDS == set()


def test_split_is_deterministic_for_a_seed() -> None:
    first = build_loop_split(ALL_CANONICAL, sizes=LoopSplitSizes(), seed=LOOP_SPLIT_SEED)
    second = build_loop_split(ALL_CANONICAL, sizes=LoopSplitSizes(), seed=LOOP_SPLIT_SEED)
    assert first == second


def test_a_different_seed_produces_a_different_split() -> None:
    first = build_loop_split(ALL_CANONICAL, sizes=LoopSplitSizes(), seed=LOOP_SPLIT_SEED)
    second = build_loop_split(ALL_CANONICAL, sizes=LoopSplitSizes(), seed=LOOP_SPLIT_SEED + 1)
    assert first != second


def test_wrong_eligible_count_is_rejected() -> None:
    with pytest.raises(ContractError):
        build_loop_split(list(range(1, 400)), sizes=LoopSplitSizes(), seed=LOOP_SPLIT_SEED)


def test_sizes_that_do_not_fit_the_pool_are_rejected() -> None:
    with pytest.raises(ContractError):
        build_loop_split(
            ALL_CANONICAL,
            sizes=LoopSplitSizes(validation_documents=400, test_documents=300),
            seed=LOOP_SPLIT_SEED,
        )


def test_a_train_size_that_does_not_add_up_is_rejected() -> None:
    # 494 eligible - 50 - 150 = 294, plus 6 exemplars = 300. Asking for 320
    # must fail loudly rather than silently producing 300.
    with pytest.raises(ContractError):
        build_loop_split(
            ALL_CANONICAL, sizes=LoopSplitSizes(train_documents=320), seed=LOOP_SPLIT_SEED
        )


def test_the_held_out_sets_are_not_contiguous_runs() -> None:
    """Guard against vacuous guards.

    Every other test here is satisfied by any order-preserving permutation --
    a rotation instead of a shuffle passes all of them. A real shuffle
    interleaves the held-out sets with train across the whole id range; a
    contiguous run does not. This split is sealed for the life of the
    experiment, so a biased draw would be undetectable afterwards.
    """
    split = build_loop_split(ALL_CANONICAL, sizes=LoopSplitSizes(), seed=LOOP_SPLIT_SEED)
    for name in ("valid", "test"):
        ids = sorted(split[name])
        span = ids[-1] - ids[0] + 1
        assert span > len(ids) * 2, (
            f"{name} occupies a near-contiguous id range ({ids[0]}..{ids[-1]} "
            f"for {len(ids)} documents); the draw is not interleaved"
        )


def test_manifest_records_the_exemplars_and_seals_itself(tmp_path: Path) -> None:
    sizes = LoopSplitSizes()
    split = build_loop_split(ALL_CANONICAL, sizes=sizes, seed=LOOP_SPLIT_SEED)
    result = write_loop_split_manifest(tmp_path, split, sizes=sizes, seed=LOOP_SPLIT_SEED)

    payload = json.loads((tmp_path / "loop_split_manifest.json").read_text(encoding="utf-8"))
    assert payload["seed"] == LOOP_SPLIT_SEED
    assert payload["counts"] == {"train": 300, "valid": 50, "test": 150}
    assert payload["exemplars_in_train"] == sorted(EXEMPLAR_DOC_IDS)
    assert result["manifest_sha256"] == payload_sha(tmp_path)


def payload_sha(tmp_path: Path) -> str:
    from data_quality_checker.fingerprints import sha256_file

    return sha256_file(tmp_path / "loop_split_manifest.json")
