from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_quality_checker.errors import ContractError
from data_quality_checker.loop_rounds import (
    ROUND_STAGES,
    RoundState,
    advance_round,
    learning_curve_rows,
    new_round,
    read_round_state,
    resume_round,
    write_round_state,
)


def test_a_new_round_starts_pending() -> None:
    state = new_round(1)
    assert state.round_index == 1
    assert state.stage == "pending"
    assert state.artifacts == {}


def test_stage_order_is_total_and_starts_pending() -> None:
    assert ROUND_STAGES[0] == "pending"
    assert ROUND_STAGES[-1] == "sealed"
    assert len(set(ROUND_STAGES)) == len(ROUND_STAGES)


def test_checkpoint_selection_precedes_measurement_in_the_stage_order() -> None:
    # The whole module exists to enforce this ordering; if the constant ever
    # reversed them the gate in Task 2 would silently permit a test-informed
    # selection.
    assert ROUND_STAGES.index("checkpoint_selected") < ROUND_STAGES.index("measured")


def test_state_round_trips_through_disk(tmp_path: Path) -> None:
    state = RoundState(round_index=3, stage="routed", artifacts={"predicted": "abc123"})
    write_round_state(tmp_path, state)
    assert read_round_state(tmp_path, 3) == state


def test_written_state_is_sealed_and_names_the_round(tmp_path: Path) -> None:
    result = write_round_state(tmp_path, new_round(7))
    payload = json.loads((tmp_path / "round_007_state.json").read_text(encoding="utf-8"))
    assert payload["round"] == 7
    assert payload["stage"] == "pending"
    assert payload["schema_version"] == 1
    assert result["state_sha256"]


def test_reading_a_round_that_was_never_written_is_a_contract_error(tmp_path: Path) -> None:
    with pytest.raises(ContractError):
        read_round_state(tmp_path, 4)


def test_an_unknown_stage_is_rejected() -> None:
    with pytest.raises(ContractError):
        RoundState(round_index=1, stage="halfway", artifacts={}).validate()


def test_a_negative_round_index_is_rejected() -> None:
    with pytest.raises(ContractError):
        new_round(-1)


def _write_raw_state(output_dir: Path, round_index: int, payload: dict) -> None:
    path = output_dir / f"round_{round_index:03d}_state.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_a_state_file_missing_the_round_key_is_a_contract_error(tmp_path: Path) -> None:
    _write_raw_state(tmp_path, 1, {"stage": "pending", "artifacts": {}})
    with pytest.raises(ContractError):
        read_round_state(tmp_path, 1)


def test_a_state_file_with_artifacts_as_a_list_is_a_contract_error(tmp_path: Path) -> None:
    _write_raw_state(tmp_path, 1, {"round": 1, "stage": "pending", "artifacts": ["oops"]})
    with pytest.raises(ContractError):
        read_round_state(tmp_path, 1)


def test_a_state_file_with_a_non_numeric_round_is_a_contract_error(tmp_path: Path) -> None:
    _write_raw_state(tmp_path, 1, {"round": "seven", "stage": "pending", "artifacts": {}})
    with pytest.raises(ContractError):
        read_round_state(tmp_path, 1)


def test_a_round_advances_one_stage_and_records_its_artifact() -> None:
    state = advance_round(new_round(1), to="predicted", artifact_sha256="aaa")
    assert state.stage == "predicted"
    assert state.artifacts["predicted"] == "aaa"


def test_skipping_a_stage_is_rejected() -> None:
    with pytest.raises(ContractError):
        advance_round(new_round(1), to="routed", artifact_sha256="aaa")


def test_going_backwards_is_rejected() -> None:
    state = advance_round(new_round(1), to="predicted", artifact_sha256="aaa")
    with pytest.raises(ContractError):
        advance_round(state, to="pending", artifact_sha256="bbb")


def test_repeating_a_stage_is_rejected() -> None:
    state = advance_round(new_round(1), to="predicted", artifact_sha256="aaa")
    with pytest.raises(ContractError):
        advance_round(state, to="predicted", artifact_sha256="bbb")


def test_earlier_artifacts_survive_later_transitions() -> None:
    state = new_round(1)
    for index, stage in enumerate(ROUND_STAGES[1:], start=1):
        state = advance_round(state, to=stage, artifact_sha256=f"sha{index}")
    assert state.stage == "sealed"
    assert state.artifacts["predicted"] == "sha1"
    assert len(state.artifacts) == len(ROUND_STAGES) - 1


def test_an_empty_artifact_sha_is_rejected() -> None:
    with pytest.raises(ContractError):
        advance_round(new_round(1), to="predicted", artifact_sha256="")


def test_measurement_cannot_precede_checkpoint_selection() -> None:
    """The protocol rule, enforced.

    Reaching `measured` requires `checkpoint_selected` to be the current
    stage, so a test-set number cannot exist before the checkpoint that
    number might otherwise have influenced.
    """
    state = new_round(1)
    for stage in ("predicted", "routed", "judged", "adjudicated", "composed", "trained"):
        state = advance_round(state, to=stage, artifact_sha256="sha")
    assert state.stage == "trained"
    with pytest.raises(ContractError):
        advance_round(state, to="measured", artifact_sha256="sha")


def test_measurement_is_allowed_once_the_checkpoint_is_selected() -> None:
    state = new_round(1)
    for stage in (
        "predicted",
        "routed",
        "judged",
        "adjudicated",
        "composed",
        "trained",
        "checkpoint_selected",
    ):
        state = advance_round(state, to=stage, artifact_sha256="sha")
    measured = advance_round(state, to="measured", artifact_sha256="sha")
    assert measured.stage == "measured"


def test_a_sealed_round_cannot_advance() -> None:
    state = new_round(1)
    for stage in ROUND_STAGES[1:]:
        state = advance_round(state, to=stage, artifact_sha256="sha")
    with pytest.raises(ContractError, match="already sealed"):
        advance_round(state, to="sealed", artifact_sha256="sha")


def test_resuming_an_unwritten_round_starts_it_pending(tmp_path: Path) -> None:
    assert resume_round(tmp_path, 2) == new_round(2)


def test_resuming_a_written_round_returns_where_it_stopped(tmp_path: Path) -> None:
    state = advance_round(new_round(2), to="predicted", artifact_sha256="aaa")
    write_round_state(tmp_path, state)
    assert resume_round(tmp_path, 2) == state


def test_the_curve_reports_only_rounds_that_reached_measured(tmp_path: Path) -> None:
    measured = new_round(1)
    for stage in ROUND_STAGES[1 : ROUND_STAGES.index("measured") + 1]:
        measured = advance_round(measured, to=stage, artifact_sha256="sha")
    write_round_state(tmp_path, measured)
    write_round_state(tmp_path, advance_round(new_round(2), to="predicted", artifact_sha256="b"))

    rows = learning_curve_rows(tmp_path, rounds=[1, 2])
    assert [row["round"] for row in rows] == [1]
    assert rows[0]["stage"] == "measured"


def test_the_curve_is_empty_when_no_round_has_been_measured(tmp_path: Path) -> None:
    write_round_state(tmp_path, advance_round(new_round(1), to="predicted", artifact_sha256="a"))
    assert learning_curve_rows(tmp_path, rounds=[1]) == []
