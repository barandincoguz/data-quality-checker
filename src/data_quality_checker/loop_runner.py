"""Drive a DQ-Loop round through its stages, stopping where a human is needed.

The runner deliberately knows nothing about prediction, routing, judging or
training. Every stage arrives as a callable, so this module stays testable
without a model, an archive or a GPU, and pipeline knowledge stays in one
place rather than two.

Stopping is a normal outcome. A round pauses before `adjudicated` because an
expert has to settle the disagreements, and that takes days. The caller
re-invokes `run_round` afterwards and it continues from where it stopped.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from .errors import ContractError
from .loop_rounds import ROUND_STAGES, RoundState, advance_round, resume_round, write_round_state

RoundStep = Callable[[RoundState], str]

# `adjudicated` is where an expert settles every disagreement the router and
# the judge could not. It is the one stage no automation may perform, so the
# runner stops before it even if a caller supplies a step for it.
MANUAL_STAGES = frozenset({"adjudicated"})


class RoundStepFailed(ContractError):
    """A round's stage raised, so that stage did not happen.

    The round's persisted state is left at the last stage that genuinely
    completed. Re-running retries the failed stage rather than skipping it: a
    round that skipped ahead would claim an artifact nothing produced, and
    nothing downstream would notice.
    """

    def __init__(self, round_index: int, stage: str, cause: BaseException) -> None:
        super().__init__(f"round {round_index} failed at stage {stage!r}: {cause}")
        self.round_index = round_index
        self.stage = stage
        self.cause = cause


def run_round(
    state_dir: Path,
    round_index: int,
    *,
    steps: Mapping[str, RoundStep],
    manual_stages: frozenset[str] = MANUAL_STAGES,
) -> RoundState:
    """Advance a round as far as it can go, then return where it stopped.

    Runs from wherever the round already is, so calling this twice resumes
    rather than restarting. It stops at the first stage that is manual, has no
    supplied step, or is past the end.

    State is persisted only after a stage's step returns, so a crash between
    the two re-runs that same step on resume. Every step a caller supplies
    must therefore be idempotent -- safe to run again against the state it
    was given -- since this module has no way to know what a step does and so
    cannot enforce that on its behalf.
    """
    unknown = sorted(set(steps) - set(ROUND_STAGES))
    if unknown:
        raise ContractError(f"steps names stages that do not exist: {unknown}")

    state = resume_round(state_dir, round_index)
    while state.stage != ROUND_STAGES[-1]:
        nxt = ROUND_STAGES[ROUND_STAGES.index(state.stage) + 1]
        if nxt in manual_stages:
            break
        step = steps.get(nxt)
        if step is None:
            break
        try:
            artifact = step(state)
        except ContractError:
            raise
        except Exception as exc:  # noqa: BLE001 - any step failure is a round failure
            # Failing on the round's very first stage means write_round_state below
            # never runs, so no state file exists yet; resume_round then reads that
            # back as an ordinary never-started round, not as a recorded failure.
            raise RoundStepFailed(round_index, nxt, exc) from exc
        state = advance_round(state, to=nxt, artifact_sha256=artifact)
        write_round_state(state_dir, state)
    return state
