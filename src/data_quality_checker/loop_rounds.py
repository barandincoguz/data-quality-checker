"""Round lifecycle for the DQ-Loop.

A round is resumable and ordered. Each stage seals an artifact, and a stage
cannot begin until its predecessor is sealed, so a crashed round resumes
exactly where it stopped rather than silently redoing or skipping work.

The ordering is not bookkeeping. `checkpoint_selected` precedes `measured`
because the experiment's central rule is that the frozen test set never
informs a decision: the checkpoint must already be chosen, on validation
alone, before any test number exists to be tempted by.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .errors import ContractError
from .fingerprints import sha256_file

ROUND_STAGES = (
    "pending",
    "predicted",
    "routed",
    "judged",
    "adjudicated",
    "composed",
    "trained",
    "checkpoint_selected",
    "measured",
    "sealed",
)


@dataclass(frozen=True)
class RoundState:
    round_index: int
    stage: str
    artifacts: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "RoundState":
        if self.round_index < 0:
            raise ContractError(f"round_index must be non-negative, got {self.round_index}")
        if self.stage not in ROUND_STAGES:
            raise ContractError(
                f"unknown stage {self.stage!r}; expected one of {list(ROUND_STAGES)}"
            )
        return self


class RoundStateMissing(ContractError):
    """No state file exists for this round -- it has never been written.

    Distinct from a state file that exists but cannot be parsed. `resume_round`
    treats "missing" as "this round has not started yet" and returns a fresh
    pending round; a *corrupt* file must never take that path, because doing so
    would silently discard a round that had really progressed.
    """


def new_round(round_index: int) -> RoundState:
    return RoundState(round_index=round_index, stage="pending", artifacts={}).validate()


def rounds_state_dir(config: Any) -> Path:
    """Where a run's round-state files live.

    The reader and the eventual writer must agree on this, and a mismatch
    would not fail loudly: `resume_round` maps a missing file to a fresh
    `pending` round, so a writer pointed elsewhere would leave `loop-status`
    reporting every round as unstarted forever.
    """
    return Path(config.public_root) / "loop" / "rounds"


def _state_path(output_dir: Path, round_index: int) -> Path:
    return output_dir / f"round_{round_index:03d}_state.json"


def write_round_state(output_dir: Path, state: RoundState) -> dict[str, Any]:
    state.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "round": state.round_index,
        "stage": state.stage,
        "artifacts": dict(state.artifacts),
    }
    path = _state_path(output_dir, state.round_index)
    write_json_atomic(path, payload)
    return {"path": str(path), "state_sha256": sha256_file(path)}


def advance_round(state: RoundState, *, to: str, artifact_sha256: str) -> RoundState:
    """Move a round exactly one stage forward, recording that stage's artifact.

    Only forward, only one step, only once. Skipping would let a stage consume
    an artifact its predecessor never produced; repeating would overwrite a
    seal; reversing would let a sealed decision be revisited after later
    evidence -- which for the `measured` stage is precisely the contamination
    this loop is built to avoid.
    """
    state.validate()
    if to not in ROUND_STAGES:
        raise ContractError(
            f"round {state.round_index} cannot advance to unknown stage {to!r}; "
            f"expected one of {list(ROUND_STAGES)}"
        )
    if not artifact_sha256:
        raise ContractError(
            f"advancing round {state.round_index} to {to!r} requires an artifact sha256"
        )
    if state.stage == ROUND_STAGES[-1]:
        raise ContractError(f"round {state.round_index} is already sealed")
    current = ROUND_STAGES.index(state.stage)
    target = ROUND_STAGES.index(to)
    if target != current + 1:
        raise ContractError(
            f"round {state.round_index} is at {state.stage!r}; the only legal next stage is "
            f"{ROUND_STAGES[current + 1]!r}, not {to!r}"
        )
    artifacts = dict(state.artifacts)
    artifacts[to] = artifact_sha256
    return RoundState(round_index=state.round_index, stage=to, artifacts=artifacts).validate()


def read_round_state(output_dir: Path, round_index: int) -> RoundState:
    path = _state_path(output_dir, round_index)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RoundStateMissing(f"no state file for round {round_index}: {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        # The file exists but its content is not valid JSON -- truncated or
        # garbled, not absent. That is corruption, not "never started", so it
        # must not be mistaken by `resume_round` for a fresh round.
        raise ContractError(f"corrupt state for round {round_index} in {path}: {exc}") from exc
    try:
        return RoundState(
            round_index=int(payload["round"]),
            stage=str(payload["stage"]),
            artifacts=dict(payload.get("artifacts") or {}),
        ).validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"malformed state for round {round_index} in {path}: {exc}") from exc


def resume_round(output_dir: Path, round_index: int) -> RoundState:
    """Return where a round stopped, or a fresh pending round if it never started.

    This is what a crashed loop calls on restart. Returning `pending` for an
    unwritten round rather than raising is deliberate: starting round k for
    the first time and resuming it after a crash are the same call.
    """
    try:
        return read_round_state(output_dir, round_index)
    except RoundStateMissing:
        return new_round(round_index)


def learning_curve_rows(output_dir: Path, rounds: list[int]) -> list[dict[str, Any]]:
    """The curve so far: one row per round that has actually been measured.

    A round that stopped earlier -- or was never written at all -- is omitted
    rather than reported with a blank score, so a partially-run loop cannot be
    read as a complete curve. A round whose state file exists but is corrupt
    is a different failure: that round may have already been measured, and
    silently dropping the point would let a published curve go quietly
    incomplete. So only `RoundStateMissing` (never started) is swallowed here;
    any other `ContractError` (corrupt or malformed) propagates to the caller.
    """
    measured_index = ROUND_STAGES.index("measured")
    rows: list[dict[str, Any]] = []
    for round_index in sorted(set(rounds)):
        try:
            state = read_round_state(output_dir, round_index)
        except RoundStateMissing:
            continue
        if ROUND_STAGES.index(state.stage) < measured_index:
            continue
        rows.append(
            {
                "round": state.round_index,
                "stage": state.stage,
                "artifacts": dict(state.artifacts),
            }
        )
    return rows
