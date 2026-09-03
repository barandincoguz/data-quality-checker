from __future__ import annotations

from pathlib import Path

import pytest

from data_quality_checker.errors import ContractError
from data_quality_checker.loop_rounds import ROUND_STAGES, RoundState, resume_round
from data_quality_checker.loop_runner import MANUAL_STAGES, RoundStepFailed, run_round


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


def _boom(state: RoundState) -> str:
    raise RuntimeError("the model fell over")


def test_a_failing_stage_raises_a_typed_error(tmp_path: Path) -> None:
    with pytest.raises(RoundStepFailed) as caught:
        run_round(tmp_path, 1, steps={"predicted": _boom})
    assert caught.value.stage == "predicted"
    assert caught.value.round_index == 1


def test_a_failing_stage_leaves_the_round_where_it_was(tmp_path: Path) -> None:
    steps = dict(_steps("predicted", "routed"))
    steps["judged"] = _boom
    with pytest.raises(RoundStepFailed):
        run_round(tmp_path, 1, steps=steps)
    assert resume_round(tmp_path, 1).stage == "routed"


def test_rerunning_after_a_failure_retries_the_failed_stage(tmp_path: Path) -> None:
    steps = dict(_steps("predicted", "routed"))
    steps["judged"] = _boom
    with pytest.raises(RoundStepFailed):
        run_round(tmp_path, 1, steps=steps)

    seen: list[str] = []
    state = run_round(tmp_path, 1, steps=_steps("predicted", "routed", "judged", recorder=seen))
    assert seen == ["judged"], "the failed stage must be retried, the completed ones must not"
    assert state.stage == "judged"


def test_a_failure_on_the_very_first_stage_leaves_nothing_claimed(tmp_path: Path) -> None:
    with pytest.raises(RoundStepFailed):
        run_round(tmp_path, 1, steps={"predicted": _boom})
    assert resume_round(tmp_path, 1).artifacts == {}


def test_a_step_returning_an_empty_artifact_is_a_failure(tmp_path: Path) -> None:
    with pytest.raises(ContractError) as caught:
        run_round(tmp_path, 1, steps={"predicted": lambda state: ""})
    assert not isinstance(caught.value, RoundStepFailed), (
        "advance_round's own rejection must keep its meaning, not be relabelled as a step failure"
    )
    assert resume_round(tmp_path, 1).stage == "pending"


def test_a_step_raising_a_contract_error_keeps_its_own_meaning(tmp_path: Path) -> None:
    """A step's own ContractError must not be relabelled a step failure.

    advance_round's rejections and a step's own contract violations are both
    ContractErrors, and wrapping the latter would blame the runner's machinery
    for a problem in the step. RoundStepFailed subclasses ContractError, so
    this has to assert on the exact type, not merely that a ContractError was
    raised.
    """

    def step(state: RoundState) -> str:
        raise ContractError("the step's own contract violation")

    with pytest.raises(ContractError) as caught:
        run_round(tmp_path, 1, steps={"predicted": step})
    assert not isinstance(caught.value, RoundStepFailed)
    assert "the step's own contract violation" in str(caught.value)
    assert resume_round(tmp_path, 1).stage == "pending"


def test_status_reports_a_round_that_has_not_started(tmp_path: Path) -> None:
    from data_quality_checker.loop_runner import round_status

    assert round_status(tmp_path, 3) == {
        "round": 3,
        "stage": "pending",
        "artifacts": {},
        "next_stage": "predicted",
        "next_is_manual": False,
    }


def test_status_reports_where_a_round_stopped(tmp_path: Path) -> None:
    from data_quality_checker.loop_runner import round_status

    run_round(tmp_path, 3, steps=_steps("predicted", "routed"))
    status = round_status(tmp_path, 3)
    assert status["stage"] == "routed"
    assert status["artifacts"]["predicted"] == "sha-predicted"


def test_status_names_the_next_stage_and_whether_it_is_manual(tmp_path: Path) -> None:
    from data_quality_checker.loop_runner import round_status

    run_round(tmp_path, 3, steps=_steps("predicted", "routed", "judged"))
    status = round_status(tmp_path, 3)
    assert status["next_stage"] == "adjudicated"
    assert status["next_is_manual"] is True
