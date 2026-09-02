# DQ-Loop Round State Machine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a round of the DQ-Loop a resumable, ordered, self-auditing object, so the experiment's protocol rules are enforced by code rather than by discipline.

**Architecture:** One new module holding a round's persisted state and the legal transitions between its stages. The state machine does not reimplement prediction, routing, judging, review or training — those already exist. It records which stage a round has reached, seals each stage's artifact with a sha256, refuses to start a stage whose predecessor is unsealed, and — the point of the whole thing — **refuses to accept a test-set measurement until the round's checkpoint selection is already sealed.** That ordering is what makes "the test split never influences a decision" a property of the code instead of a promise in a document.

**Tech Stack:** Python 3.11, stdlib only, pytest 9, ruff, mypy. Builds on `loop_split.py`, `loop_batches.py`, `loop_training.py`, `loop_selection.py` (all merged) and `atomic.write_json_atomic` / `fingerprints.sha256_file`.

## Global Constraints

- **No GPU run, no real pipeline invocation, in scope.** This module records and orders; it does not call `process`, `pilot-judges`, `serve`, `release` or any trainer. Wiring it to those is a later plan. Everything here is testable with fixtures.
- **`g0.build_split`, `g0.write_training_data`, and the reference-split gate at `g0.py:217-222` stay untouched.** They are the deployed G0 model's provenance.
- **Stage order is fixed and total.** A round moves forward only, one stage at a time, and every transition is recorded.
- **A test measurement may only be recorded after `checkpoint_selected` is sealed.** Attempting it earlier is a `ContractError`. This is the module's reason to exist.
- **Selection reads validation only.** Nothing in a round's state may let a test metric reach a decision.
- State is persisted with `write_json_atomic` and every stage seals an artifact sha256, matching the repository's durability convention.
- No new runtime dependency. `requirements.txt` is `Flask>=3.1,<4`.
- `ruff format` and `ruff check` must pass. `mypy src` must not exceed its **37 pre-existing errors in 9 files**.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/data_quality_checker/loop_rounds.py` (create) | Round stages, persisted state, legal transitions, the ordering gate. |
| `tests/test_loop_rounds.py` (create) | |

---

### Task 1: Stages and persisted round state

**Files:**
- Create: `src/data_quality_checker/loop_rounds.py`
- Test: `tests/test_loop_rounds.py`

**Interfaces:**
- Consumes: `write_json_atomic` from `.atomic`, `sha256_file` from `.fingerprints`, `ContractError` from `.errors`.
- Produces: `ROUND_STAGES: tuple[str, ...]`, `RoundState` frozen dataclass with `round_index: int`, `stage: str`, `artifacts: dict[str, str]`; `new_round(round_index) -> RoundState`; `write_round_state(output_dir, state) -> dict[str, Any]`; `read_round_state(output_dir, round_index) -> RoundState`.

The stages, in order:

```python
ROUND_STAGES = (
    "pending",             # the round exists; its batch is drawn but nothing has run
    "predicted",           # M_{k-1} has annotated the round's 100 documents
    "routed",              # the router has bucketed human vs model
    "judged",              # the LLM judge has given its third opinion
    "adjudicated",         # the expert has settled every disagreement and the GREEN sample
    "composed",            # round k's training set has been composed
    "trained",             # M_k exists
    "checkpoint_selected", # a checkpoint was chosen, on validation only
    "measured",            # the frozen test set was scored -- recorded, never consulted
    "sealed",              # the round record is final and immutable
)
```

- [ ] **Step 1: Write the failing test**

Create `tests/test_loop_rounds.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loop_rounds.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_quality_checker.loop_rounds'`

- [ ] **Step 3: Implement stages and state**

Create `src/data_quality_checker/loop_rounds.py`:

```python
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

    def validate(self) -> "RoundState":
        if self.round_index < 0:
            raise ContractError(f"round_index must be non-negative, got {self.round_index}")
        if self.stage not in ROUND_STAGES:
            raise ContractError(f"unknown stage {self.stage!r}; expected one of {list(ROUND_STAGES)}")
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
    return RoundState(
        round_index=int(payload["round"]),
        stage=str(payload["stage"]),
        artifacts=dict(payload.get("artifacts") or {}),
    ).validate()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_loop_rounds.py -q`
Expected: PASS — 8 passed.

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/python -m ruff format src tests && .venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src 2>&1 | tail -3
git add src/data_quality_checker/loop_rounds.py tests/test_loop_rounds.py
git commit -m "feat(loop): add resumable round state for the DQ-Loop"
```

---

### Task 2: Ordered transitions and the measurement gate

**Files:**
- Modify: `src/data_quality_checker/loop_rounds.py`
- Test: `tests/test_loop_rounds.py`

**Interfaces:**
- Consumes: `ROUND_STAGES`, `RoundState` from Task 1.
- Produces: `advance_round(state, *, to, artifact_sha256) -> RoundState`.

`advance_round` moves a round exactly one stage forward, recording the artifact that stage produced. Skipping, repeating and reversing are all `ContractError`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_loop_rounds.py`:

```python
from data_quality_checker.loop_rounds import advance_round


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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loop_rounds.py -k advance -q`
Expected: FAIL — `ImportError: cannot import name 'advance_round'`

- [ ] **Step 3: Implement the transition**

Add to `loop_rounds.py`:

```python
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
        raise ContractError(f"unknown stage {to!r}; expected one of {list(ROUND_STAGES)}")
    if not artifact_sha256:
        raise ContractError(f"advancing round {state.round_index} to {to!r} requires an artifact sha256")
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
```

The `target != current + 1` rule enforces the measurement gate on its own: `measured` is reachable only from `checkpoint_selected`. Task 2's dedicated test pins that consequence explicitly, so a later reordering of `ROUND_STAGES` fails loudly rather than quietly permitting a test-informed selection.

Note the last stage: advancing from `sealed` raises, because `ROUND_STAGES[current + 1]` is out of range — add an explicit guard so the error names the situation:

```python
    if state.stage == ROUND_STAGES[-1]:
        raise ContractError(f"round {state.round_index} is already sealed")
```

Place it before the index arithmetic.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_loop_rounds.py -q`
Expected: PASS — 16 passed.

- [ ] **Step 5: Add the sealed-round guard test**

Append:

```python
def test_a_sealed_round_cannot_advance() -> None:
    state = new_round(1)
    for stage in ROUND_STAGES[1:]:
        state = advance_round(state, to=stage, artifact_sha256="sha")
    with pytest.raises(ContractError, match="already sealed"):
        advance_round(state, to="sealed", artifact_sha256="sha")
```

Run: `.venv/bin/python -m pytest tests/test_loop_rounds.py -q`
Expected: PASS — 17 passed.

- [ ] **Step 6: Lint, type-check, commit**

```bash
.venv/bin/python -m ruff format src tests && .venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src 2>&1 | tail -3
git add src/data_quality_checker/loop_rounds.py tests/test_loop_rounds.py
git commit -m "feat(loop): order round transitions and gate measurement behind selection"
```

---

### Task 3: Resume, and the learning-curve table

**Files:**
- Modify: `src/data_quality_checker/loop_rounds.py`
- Test: `tests/test_loop_rounds.py`

**Interfaces:**
- Consumes: everything from Tasks 1-2.
- Produces: `resume_round(output_dir, round_index) -> RoundState`; `learning_curve_rows(output_dir, rounds) -> list[dict[str, Any]]`.

`resume_round` returns a round's persisted state, or a fresh `pending` round if it has never been written — the call a crashed loop makes on restart. `learning_curve_rows` reads the sealed rounds and returns the curve, refusing to report a round that has not reached `measured`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_loop_rounds.py`:

```python
from data_quality_checker.loop_rounds import learning_curve_rows, resume_round


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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loop_rounds.py -k "resume or curve" -q`
Expected: FAIL — `ImportError: cannot import name 'resume_round'`

- [ ] **Step 3: Implement resume and the curve**

Add to `loop_rounds.py`:

```python
def resume_round(output_dir: Path, round_index: int) -> RoundState:
    """Return where a round stopped, or a fresh pending round if it never started.

    This is what a crashed loop calls on restart. Returning `pending` for an
    unwritten round rather than raising is deliberate: starting round k for
    the first time and resuming it after a crash are the same call.
    """
    try:
        return read_round_state(output_dir, round_index)
    except ContractError:
        return new_round(round_index)


def learning_curve_rows(output_dir: Path, rounds: list[int]) -> list[dict[str, Any]]:
    """The curve so far: one row per round that has actually been measured.

    A round that stopped earlier is omitted rather than reported with a blank
    score, so a partially-run loop cannot be read as a complete curve.
    """
    measured_index = ROUND_STAGES.index("measured")
    rows: list[dict[str, Any]] = []
    for round_index in sorted(rounds):
        try:
            state = read_round_state(output_dir, round_index)
        except ContractError:
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_loop_rounds.py -q`
Expected: PASS — 21 passed.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS — **289**. That is 265 before this plan plus 24: 8 from Task 1, 3 from the hardening commit that closed Task 1's review findings, 9 from Task 2, and 4 from Task 3.

- [ ] **Step 6: Lint, type-check, commit**

```bash
.venv/bin/python -m ruff format src tests && .venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src 2>&1 | tail -3
git add src/data_quality_checker/loop_rounds.py tests/test_loop_rounds.py
git commit -m "feat(loop): resume a crashed round and report the measured curve"
```

---

## Done When

- `loop_rounds.py` exists with stages, state, ordered transitions, resume, and the curve.
- `.venv/bin/python -m pytest -q` passes at **289** (265 + 24; the extra three come from the hardening commit that closed Task 1's review findings).
- `ruff check` passes; `mypy src` still reports 37 errors in 9 files.
- `g0.py` is byte-identical to `main`.
- No new runtime dependency.

## What the gate does and does not prove

The ordering rule makes it impossible to **record** a test measurement before a
checkpoint selection is sealed. That is worth having: it turns an ordering that
was previously a promise in a design document into a condition the code
refuses to violate, and it leaves an auditable trail — the round state names
which stage sealed which artifact, in what order.

It does **not** prove the test set was uninfluential. Nothing here stops an
operator computing a test score early and recording it later, or reading it
from a log and re-running training with a different learning rate. Code cannot
close that gap; only the discipline of the person running the loop can.

So the honest claim is: **the artifact ordering is enforced, the human
behaviour is not.** State it that way in the paper. Do not write that the
protocol is "guaranteed by construction" — a reviewer who reads this module
will see immediately that it is not, and the overclaim would cost more
credibility than the guarantee was worth.

---

## Explicitly Out Of Scope

- **Calling anything.** This module orders and records; it does not invoke prediction, routing, judging, review, release or training. Wiring is the next plan, and it is the one that will need `prepare_batch` to accept a document-id subset — today it takes ZIPs, which is the real integration constraint.
- Lifting `train_bootstrap`'s `only canonical-only G0` restriction at `g0.py:433`. It belongs with the wiring that will call it.
- Any GPU run.
- The workload and error-attribution metric tables. They need adjudication data that only a real round produces.
