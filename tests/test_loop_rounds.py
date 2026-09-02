from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_quality_checker.errors import ContractError
from data_quality_checker.loop_rounds import (
    ROUND_STAGES,
    RoundState,
    new_round,
    read_round_state,
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
