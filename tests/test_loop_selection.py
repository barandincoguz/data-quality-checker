from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_quality_checker.errors import ContractError, GateBlocked
from data_quality_checker.g0 import CheckpointCandidate
from data_quality_checker.loop_selection import write_selection_record

EXPECTED_TOP_LEVEL = {
    "schema_version",
    "round",
    "selection_basis",
    "validation_documents",
    "minimum_parse_count",
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


def test_minimum_parse_count_is_forwarded_to_the_selector(tmp_path: Path) -> None:
    """FIX A regression: a caller's `minimum_parse_count` must reach the selector.

    Before this fix, `write_selection_record` re-ran `select_checkpoint` with
    only `validation_documents`, silently dropping `minimum_parse_count`. An
    operator legitimately calling `select_checkpoint(rows,
    minimum_parse_count=48)` to admit update 100 (parse_count=48) got a
    `ContractError` claiming update 100 "is not what select_checkpoint
    chooses" -- the recorder's own missing parameter, blamed on the caller.
    """
    rows = [
        CheckpointCandidate(
            update=100,
            coverage_count=50,
            parse_count=48,
            empty_output_count=0,
            runaway_output_count=0,
            core_f1=0.90,
            docwise_accuracy=0.30,
            recall=0.70,
            validation_loss=0.06,
        ),
        CheckpointCandidate(
            update=200,
            coverage_count=50,
            parse_count=50,
            empty_output_count=0,
            runaway_output_count=0,
            core_f1=0.80,
            docwise_accuracy=0.30,
            recall=0.70,
            validation_loss=0.06,
        ),
    ]
    result = write_selection_record(
        tmp_path,
        4,
        rows[0],
        rows,
        validation_documents=50,
        minimum_parse_count=48,
    )
    payload = json.loads((tmp_path / "round_004_checkpoint.json").read_text(encoding="utf-8"))
    assert payload["selected"]["update"] == 100
    assert payload["minimum_parse_count"] == 48
    assert result["record_sha256"]

    # Without the caller's minimum_parse_count, update 100 is ineligible
    # (parse_count 48 < the default 49) and the default selector would have
    # chosen update 200 instead -- confirming this genuinely exercises the
    # forwarded parameter rather than something both calls would agree on.
    with pytest.raises(ContractError):
        write_selection_record(tmp_path, 5, rows[0], rows, validation_documents=50)


def test_write_selection_record_lets_gate_blocked_propagate(tmp_path: Path) -> None:
    """FIX C: no eligible candidate means `GateBlocked`, not `ContractError`.

    `select_checkpoint` raises `GateBlocked` when no candidate passes the
    eligibility gates -- there genuinely is no selection to record, so
    `write_selection_record` lets it propagate unconverted rather than
    wrapping it as a `ContractError`. This is documented as part of the
    function's contract, not an accidental leak.
    """
    rows = candidates()
    with pytest.raises(GateBlocked):
        # parse_count=50 on every row fails a minimum_parse_count of 999,
        # so nothing is eligible and select_checkpoint raises GateBlocked.
        write_selection_record(
            tmp_path, 6, rows[0], rows, validation_documents=50, minimum_parse_count=999
        )


def test_write_selection_record_accepts_a_tied_selected(tmp_path: Path) -> None:
    """FIX D: a tie on the declared ranking key must not be rejected by identity.

    Two candidates that tie on every field in `CHECKPOINT_TIE_BREAK_ORDER`
    are indistinguishable under the ordering the record declares. `max`
    still picks whichever came first in the list, but a caller who picked
    the *other* tied candidate is not wrong -- the record must accept it
    rather than reject it for failing an object-identity check the declared
    tie-break order cannot justify.
    """
    tied_a = CheckpointCandidate(
        update=100,
        coverage_count=50,
        parse_count=50,
        empty_output_count=0,
        runaway_output_count=0,
        core_f1=0.80,
        docwise_accuracy=0.30,
        recall=0.70,
        validation_loss=0.06,
    )
    tied_b = CheckpointCandidate(
        update=200,
        coverage_count=50,
        parse_count=50,
        empty_output_count=0,
        runaway_output_count=0,
        core_f1=0.80,
        docwise_accuracy=0.30,
        recall=0.70,
        validation_loss=0.06,
    )
    rows = [tied_a, tied_b]
    # select_checkpoint's max() picks tied_a (first in the list); a caller
    # who legitimately picked tied_b must still be accepted.
    result = write_selection_record(tmp_path, 7, tied_b, rows, validation_documents=50)
    payload = json.loads((tmp_path / "round_007_checkpoint.json").read_text(encoding="utf-8"))
    assert payload["selected"]["update"] == 200
    assert result["record_sha256"]
