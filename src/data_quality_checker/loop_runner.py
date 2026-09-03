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
        artifact = step(state)
        state = advance_round(state, to=nxt, artifact_sha256=artifact)
        write_round_state(state_dir, state)
    return state
