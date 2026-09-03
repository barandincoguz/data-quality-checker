from __future__ import annotations

from pathlib import Path

import pytest

from data_quality_checker.errors import ContractError
from data_quality_checker.loop_rounds import ROUND_STAGES, RoundState, resume_round
from data_quality_checker.loop_runner import MANUAL_STAGES, run_round


def _steps(*stages: str, recorder: list[str] | None = None):
    def make(stage: str):
        def step(state: RoundState) -> str:
            if recorder is not None:
                recorder.append(stage)
            return f"sha-{stage}"

        return step

    return {stage: make(stage) for stage in stages}


AUTOMATIC_BEFORE_ADJUDICATION = ("predicted", "routed", "judged")


def test_a_round_runs_until_it_needs_a_human(tmp_path: Path) -> None:
    seen: list[str] = []
    state = run_round(tmp_path, 1, steps=_steps(*AUTOMATIC_BEFORE_ADJUDICATION, recorder=seen))
    assert seen == list(AUTOMATIC_BEFORE_ADJUDICATION)
    assert state.stage == "judged"


def test_stopping_before_a_manual_stage_persists_what_was_done(tmp_path: Path) -> None:
    run_round(tmp_path, 1, steps=_steps(*AUTOMATIC_BEFORE_ADJUDICATION))
    assert resume_round(tmp_path, 1).stage == "judged"


def test_each_stage_records_its_artifact(tmp_path: Path) -> None:
    state = run_round(tmp_path, 1, steps=_steps(*AUTOMATIC_BEFORE_ADJUDICATION))
    assert state.artifacts["predicted"] == "sha-predicted"
    assert state.artifacts["judged"] == "sha-judged"


def test_a_second_call_resumes_rather_than_restarting(tmp_path: Path) -> None:
    first: list[str] = []
    run_round(tmp_path, 1, steps=_steps("predicted", "routed", recorder=first))
    assert first == ["predicted", "routed"]

    second: list[str] = []
    state = run_round(tmp_path, 1, steps=_steps(*AUTOMATIC_BEFORE_ADJUDICATION, recorder=second))
    assert second == ["judged"], "already-completed stages must not run again"
    assert state.stage == "judged"


def test_a_stage_with_no_step_stops_the_run(tmp_path: Path) -> None:
    state = run_round(tmp_path, 1, steps=_steps("predicted"))
    assert state.stage == "predicted"


def test_adjudicated_is_manual_by_default() -> None:
    assert "adjudicated" in MANUAL_STAGES


def test_a_manual_stage_is_not_run_even_when_a_step_is_supplied(tmp_path: Path) -> None:
    seen: list[str] = []
    state = run_round(
        tmp_path,
        1,
        steps=_steps(*AUTOMATIC_BEFORE_ADJUDICATION, "adjudicated", recorder=seen),
    )
    assert "adjudicated" not in seen
    assert state.stage == "judged"


def test_a_sealed_round_runs_nothing(tmp_path: Path) -> None:
    seen: list[str] = []
    run_round(
        tmp_path, 1, steps=_steps(*ROUND_STAGES[1:], recorder=seen), manual_stages=frozenset()
    )
    seen.clear()
    state = run_round(
        tmp_path, 1, steps=_steps(*ROUND_STAGES[1:], recorder=seen), manual_stages=frozenset()
    )
    assert seen == []
    assert state.stage == "sealed"


def test_an_unknown_stage_in_steps_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ContractError):
        run_round(tmp_path, 1, steps={"halfway": lambda state: "sha"})
