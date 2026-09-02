from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_quality_checker.errors import ContractError
from data_quality_checker.g0 import CheckpointCandidate
from data_quality_checker.loop_selection import write_selection_record

EXPECTED_TOP_LEVEL = {
    "schema_version",
    "round",
    "selection_basis",
    "validation_documents",
    "tie_break_order",
    "selected",
    "candidates",
}
EXPECTED_ROW = {
    "update",
    "coverage_count",
    "parse_count",
    "empty_output_count",
    "runaway_output_count",
    "core_f1",
    "docwise_accuracy",
    "recall",
    "validation_loss",
}


def candidates() -> list[CheckpointCandidate]:
    return [
        CheckpointCandidate(
            update=update,
            coverage_count=50,
            parse_count=50,
            empty_output_count=0,
            runaway_output_count=0,
            core_f1=score,
            docwise_accuracy=0.30,
            recall=0.70,
            validation_loss=0.06,
        )
        for update, score in ((50, 0.70), (100, 0.78), (150, 0.74))
    ]


def test_record_names_the_winner_and_every_candidate(tmp_path: Path) -> None:
    rows = candidates()
    result = write_selection_record(tmp_path, 3, rows[1], rows, validation_documents=50)
    payload = json.loads((tmp_path / "round_003_checkpoint.json").read_text(encoding="utf-8"))
    assert payload["round"] == 3
    assert payload["selected"]["update"] == 100
    assert [row["update"] for row in payload["candidates"]] == [50, 100, 150]
    assert result["record_sha256"]


def test_record_states_the_selection_basis_is_validation(tmp_path: Path) -> None:
    rows = candidates()
    write_selection_record(tmp_path, 3, rows[1], rows, validation_documents=50)
    payload = json.loads((tmp_path / "round_003_checkpoint.json").read_text(encoding="utf-8"))
    assert payload["selection_basis"] == "validation"
    assert payload["validation_documents"] == 50
    assert payload["tie_break_order"] == [
        "core_f1",
        "docwise_accuracy",
        "recall",
        "-validation_loss",
    ]


def test_record_keys_are_exactly_the_allowed_set(tmp_path: Path) -> None:
    """Guard against vacuous guards.

    A substring blocklist cannot anticipate the field name someone will
    add -- a reviewer once added a literal `"chosen_by": "held_out_score"`
    field to the payload and every test in this file still passed, because
    none of the forbidden substrings (`test_`, `_test`, `test_core_f1`,
    `test_score`) happened to match it. An allowlist of exactly the expected
    keys fails on ANY new top-level or row field -- leaked test-set
    reference or not -- until it is deliberately added here, which is the
    only way a real leak gets caught.
    """
    rows = candidates()
    write_selection_record(tmp_path, 3, rows[1], rows, validation_documents=50)
    payload = json.loads((tmp_path / "round_003_checkpoint.json").read_text(encoding="utf-8"))
    assert set(payload.keys()) == EXPECTED_TOP_LEVEL
    assert set(payload["selected"].keys()) == EXPECTED_ROW
    for row in payload["candidates"]:
        assert set(row.keys()) == EXPECTED_ROW


def test_write_selection_record_rejects_a_selected_that_disagrees_with_the_selector(
    tmp_path: Path,
) -> None:
    rows = candidates()
    with pytest.raises(ContractError):
        # rows[0] (core_f1 0.70) is a real candidate, so the membership
        # check passes, but select_checkpoint would choose rows[1] (0.78)
        # for this candidate set -- the record must refuse to claim
        # otherwise.
        write_selection_record(tmp_path, 3, rows[0], rows, validation_documents=50)
