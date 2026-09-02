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


def new_round(round_index: int) -> RoundState:
    return RoundState(round_index=round_index, stage="pending", artifacts={}).validate()


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


def read_round_state(output_dir: Path, round_index: int) -> RoundState:
    path = _state_path(output_dir, round_index)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"no readable state for round {round_index}: {exc}") from exc
    try:
        return RoundState(
            round_index=int(payload["round"]),
            stage=str(payload["stage"]),
            artifacts=dict(payload.get("artifacts") or {}),
        ).validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"malformed state for round {round_index} in {path}: {exc}") from exc
