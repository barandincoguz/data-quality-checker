# DQ-Loop Round Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive a round through its stages automatically, stopping where a human is genuinely required and resuming exactly there afterwards.

**Architecture:** A thin orchestrator over the pieces that already exist. `loop_rounds` owns the stages, the persisted state and the legal transitions; the runner owns only the question *"which stage runs next, and what happens when one fails."* Every stage is supplied as an injected callable, so the runner never imports the pipeline and every test runs without a model, an archive or a GPU. The stage that needs an expert is a deliberate stopping point, not a failure.

**Tech Stack:** Python 3.11, stdlib only, pytest 9, ruff, mypy. Builds on `loop_rounds.py` (merged): `ROUND_STAGES`, `RoundState`, `advance_round`, `resume_round`, `write_round_state`.

## Global Constraints

- **The runner imports no pipeline code.** Stages arrive as callables. This keeps it testable without infrastructure and stops it becoming a second place where pipeline knowledge lives.
- **Stopping is a normal outcome, not an error.** A round pauses before `adjudicated` because an expert has to settle disagreements, which takes days. `run_round` returns the state it reached; the caller re-invokes it later and it continues.
- **A failing step must not advance the round.** The state on disk stays at the last stage that genuinely completed, so re-running resumes there rather than skipping work that never happened.
- **Ordering stays `loop_rounds`' job.** The runner calls `advance_round` and lets it reject an illegal transition; it must not reimplement the rules, and in particular it must not bypass the gate that makes `measured` reachable only from `checkpoint_selected`.
- Every stage's artifact sha256 is recorded through `advance_round`, and the state is persisted after each transition so a crash mid-round loses at most the stage in flight.
- No new runtime dependency. `requirements.txt` is `Flask>=3.1,<4`.
- `ruff format` and `ruff check` must pass. `mypy src` must not exceed its **37 pre-existing errors in 9 files**.
- `g0.py` is not touched.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/data_quality_checker/loop_runner.py` (create) | Step protocol, `run_round`, stop-and-resume, failure handling. |
| `tests/test_loop_runner.py` (create) | |
| `src/data_quality_checker/cli.py` (modify, Task 3) | A `loop-run` command. |
| `src/data_quality_checker/commands.py` (modify, Task 3) | Its handler. |

---

### Task 1: Run a round until it must stop

**Files:**
- Create: `src/data_quality_checker/loop_runner.py`
- Test: `tests/test_loop_runner.py`

**Interfaces:**
- Consumes: `ROUND_STAGES`, `RoundState`, `advance_round`, `resume_round`, `write_round_state` from `.loop_rounds`; `ContractError` from `.errors`.
- Produces: `RoundStep` type alias `Callable[[RoundState], str]`; `MANUAL_STAGES: frozenset[str]`; `run_round(state_dir, round_index, *, steps, manual_stages=MANUAL_STAGES) -> RoundState`.

A step receives the round's current state and returns its artifact's sha256. The runner records it via `advance_round` and persists.

- [ ] **Step 1: Write the failing test**

Create `tests/test_loop_runner.py`:

```python
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
    state = run_round(
        tmp_path, 1, steps=_steps(*AUTOMATIC_BEFORE_ADJUDICATION, recorder=seen)
    )
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
    state = run_round(
        tmp_path, 1, steps=_steps(*AUTOMATIC_BEFORE_ADJUDICATION, recorder=second)
    )
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
    run_round(tmp_path, 1, steps=_steps(*ROUND_STAGES[1:], recorder=seen), manual_stages=frozenset())
    seen.clear()
    state = run_round(tmp_path, 1, steps=_steps(*ROUND_STAGES[1:], recorder=seen), manual_stages=frozenset())
    assert seen == []
    assert state.stage == "sealed"


def test_an_unknown_stage_in_steps_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ContractError):
        run_round(tmp_path, 1, steps={"halfway": lambda state: "sha"})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loop_runner.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_quality_checker.loop_runner'`

- [ ] **Step 3: Implement**

Create `src/data_quality_checker/loop_runner.py`:

```python
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
```

Note the persist happens **after** `advance_round`, so a state file only ever names a stage that genuinely completed.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_loop_runner.py -q`
Expected: PASS — 9 passed.

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/python -m ruff format src tests && .venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src 2>&1 | tail -3
git add src/data_quality_checker/loop_runner.py tests/test_loop_runner.py
git commit -m "feat(loop): run a round until it needs a human"
```

---

### Task 2: A failing stage must not advance the round

**Files:**
- Modify: `src/data_quality_checker/loop_runner.py`
- Test: `tests/test_loop_runner.py`

**Interfaces:**
- Produces: `RoundStepFailed(ContractError)` carrying `round_index`, `stage` and the original error.

A step that raises means that stage did not happen. The round must stay at its last completed stage on disk, so re-running does the failed stage again rather than skipping it — skipping would leave a round claiming an artifact nothing produced.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_loop_runner.py`:

```python
from data_quality_checker.loop_runner import RoundStepFailed


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
    with pytest.raises(ContractError):
        run_round(tmp_path, 1, steps={"predicted": lambda state: ""})
    assert resume_round(tmp_path, 1).stage == "pending"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loop_runner.py -k fail -q`
Expected: FAIL — `ImportError: cannot import name 'RoundStepFailed'`

- [ ] **Step 3: Implement**

Add to `loop_runner.py`:

```python
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
```

and wrap the call in `run_round`:

```python
        try:
            artifact = step(state)
        except ContractError:
            raise
        except Exception as exc:  # noqa: BLE001 - any step failure is a round failure
            raise RoundStepFailed(round_index, nxt, exc) from exc
```

`ContractError` is re-raised unchanged so `advance_round`'s own rejections keep their meaning rather than being relabelled as step failures. An empty artifact is already rejected by `advance_round`, which raises `ContractError` before any state is written — the test above pins that.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_loop_runner.py -q`
Expected: PASS — 14 passed.

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/python -m ruff format src tests && .venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src 2>&1 | tail -3
git add src/data_quality_checker/loop_runner.py tests/test_loop_runner.py
git commit -m "feat(loop): keep a round where it was when a stage fails"
```

---

### Task 3: Report a round's position from the CLI

**Files:**
- Modify: `src/data_quality_checker/cli.py`
- Modify: `src/data_quality_checker/commands.py`
- Test: `tests/test_loop_runner.py`

**Interfaces:**
- Produces: a `loop-status` subcommand taking `--round` and printing the round's stage and recorded artifacts.

**Deliberately read-only.** A `loop-run` command that actually executed stages would need the pipeline wired in, which is a later plan; shipping a command that looks like it runs a round but cannot would be worse than shipping nothing. `loop-status` answers the question an operator actually has mid-experiment — *where is round 7?* — using only what already exists.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_loop_runner.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loop_runner.py -k status -q`
Expected: FAIL — `ImportError: cannot import name 'round_status'`

- [ ] **Step 3: Implement `round_status`**

Add to `loop_runner.py`:

```python
def round_status(state_dir: Path, round_index: int) -> dict[str, object]:
    """Where a round is, and what it is waiting for.

    `next_is_manual` is the field an operator actually needs: it distinguishes
    "this round is blocked on me" from "this round is between automated
    stages".
    """
    state = resume_round(state_dir, round_index)
    if state.stage == ROUND_STAGES[-1]:
        next_stage: str | None = None
    else:
        next_stage = ROUND_STAGES[ROUND_STAGES.index(state.stage) + 1]
    return {
        "round": state.round_index,
        "stage": state.stage,
        "artifacts": dict(state.artifacts),
        "next_stage": next_stage,
        "next_is_manual": next_stage in MANUAL_STAGES if next_stage else False,
    }
```

- [ ] **Step 4: Wire the CLI**

In `cli.py`, beside the existing subparsers:

```python
    loop_status = subparsers.add_parser("loop-status", help="show where a DQ-Loop round is")
    loop_status.add_argument("--round", type=int, required=True, dest="round_index")
    loop_status.set_defaults(handler=commands.loop_status)
```

In `commands.py`. Every handler in that file has the signature
`def name(args: Namespace, config: AppConfig) -> int`, prints its result as
JSON and returns `0`; `cli.py:215` does `return int(args.handler(args, config))`.
Follow that exactly:

```python
def loop_status(args: Namespace, config: AppConfig) -> int:
    from .loop_runner import round_status

    result = round_status(config.public_root / "loop" / "rounds", args.round_index)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
```

`Namespace` and `json` are already imported at the top of `commands.py`; check
before adding either.

- [ ] **Step 5: Verify end to end**

```bash
.venv/bin/python -m data_quality_checker --config configs/presets/sample_data.json loop-status --round 3
```

Expected: exit 0, reporting round 3 as `pending` with `next_stage` `predicted`.

- [ ] **Step 6: Full suite, lint, type-check, commit**

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff format src tests && .venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src 2>&1 | tail -3
git add src/data_quality_checker/loop_runner.py src/data_quality_checker/cli.py \
        src/data_quality_checker/commands.py tests/test_loop_runner.py
git commit -m "feat(cli): report where a DQ-Loop round is"
```

---

## Done When

- `run_round` advances a round, stops before a manual stage, resumes, and leaves a failed round where it was.
- `loop-status` reports a round's position from the command line.
- `.venv/bin/python -m pytest -q` passes; report the count.
- `ruff check` passes; `mypy src` still reports 37 errors in 9 files.
- `g0.py` is byte-identical to `main`.
- No new runtime dependency.

## Explicitly Out Of Scope

- **Supplying real steps.** Binding `prepare_batch`, `process_batch`, `pilot_judges`, `release_batch`, the trainer and the evaluator to stage callables is the next plan. This one builds the thing that will call them.
- A `loop-run` CLI command that executes stages — it would need those bindings.
- Lifting `train_bootstrap`'s `only canonical-only G0` restriction (`g0.py:433`).
- Any GPU run.
