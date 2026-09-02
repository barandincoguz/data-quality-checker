from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_quality_checker.errors import ContractError
from data_quality_checker.loop_batches import (
    LOOP_BATCH_SEED,
    build_round_batches,
    write_batch_manifest,
)


def make_documents(count: int = 1200) -> list[dict[str, object]]:
    quartiles = ("q1", "q2", "q3", "q4")
    channels = ("pdfText", "htmlText")
    return [
        {
            "doc_id": f"d{index:05d}",
            "length_quartile": quartiles[index % 4],
            "text_channel": channels[index % 2],
            "annotator": f"annotator_{index % 14:02d}",
            "reference_count": 0 if index % 300 == 0 else (index % 9) + 1,
        }
        for index in range(count)
    ]


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


def test_zero_reference_documents_are_spread_not_clumped() -> None:
    documents = make_documents()
    by_id = {str(doc["doc_id"]): doc for doc in documents}
    batches = build_round_batches(documents, rounds=12, size=100, seed=LOOP_BATCH_SEED)
    per_batch = [
        sum(1 for doc_id in batch if by_id[doc_id]["reference_count"] == 0) for batch in batches
    ]
    assert max(per_batch) <= 1


def test_too_few_documents_is_rejected() -> None:
    with pytest.raises(ContractError):
        build_round_batches(make_documents(50), rounds=12, size=100, seed=LOOP_BATCH_SEED)


def test_manifest_seals_itself_and_records_the_seed(tmp_path: Path) -> None:
    batches = build_round_batches(make_documents(), rounds=12, size=100, seed=LOOP_BATCH_SEED)
    result = write_batch_manifest(tmp_path, batches, seed=LOOP_BATCH_SEED)
    payload = json.loads((tmp_path / "round_batches_manifest.json").read_text(encoding="utf-8"))
    assert payload["seed"] == LOOP_BATCH_SEED
    assert payload["rounds"] == 12
    assert payload["size"] == 100
    assert [len(batch) for batch in payload["batches"]] == [100] * 12
    assert result["manifest_sha256"]
