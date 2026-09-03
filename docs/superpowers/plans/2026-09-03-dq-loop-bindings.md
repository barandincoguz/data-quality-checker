# DQ-Loop Stage Bindings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the DQ-Loop's pipeline calls into the stage callables `run_round` already knows how to drive, so a round runs itself.

**Architecture:** One new module of factories. Each returns a `RoundStep` — `Callable[[RoundState], str]` — closing over the config and whatever that stage needs, calling the existing pipeline function, and returning the sha256 of the artifact it produced. The runner keeps knowing nothing about the pipeline; this module is the single place the two meet. Every binding is exercised against **real pipeline code** using the fake backends that already exist, not against a stub of the pipeline.

**Tech Stack:** Python 3.11, stdlib only, pytest 9, ruff, mypy. Binds `processing.process_batch`, `judges.run_judge_pilot`, `loop_training`, `loop_selection`, `g0.select_checkpoint` and `g0_training`'s canonical evaluator.

## Global Constraints

- **A binding calls the pipeline; it does not reimplement it.** If a binding grows logic the pipeline already has, that logic now lives in two places and will drift.
- **Every binding returns a sha256 of something that exists on disk.** `advance_round` rejects an empty artifact, and a round's whole audit value is that each stage names a file you can go and read.
- **`g0.py` is touched only to make one private evaluator helper public.** `build_split`, `write_training_data` and the reference-split gate at `g0.py:217-222` stay untouched — they are the deployed G0 model's provenance.
- **No binding for `adjudicated`.** That stage is where an expert settles what the router and judge could not, and `run_round` stops before it by default. Supplying a step would invite automating the one thing that must not be automated.
- **No binding for `trained`.** `train_bootstrap` still refuses anything but canonical-only G0 (`g0.py:449`), and training needs a GPU. Both are out of scope; see the final section.
- No new runtime dependency. `requirements.txt` is `Flask>=3.1,<4`.
- `ruff format` and `ruff check` must pass. `mypy src` must not exceed its **37 pre-existing errors in 9 files**.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/data_quality_checker/loop_bindings.py` (create) | Stage-step factories: predict, route, judge, compose, select, measure, seal. |
| `src/data_quality_checker/g0_training.py` (modify, Task 3) | Rename `_canonical_evaluate` to `canonical_evaluate` so a binding can use it. |
| `tests/test_loop_bindings.py` (create) | |

---

### Task 1: Prediction and routing

**Files:**
- Create: `src/data_quality_checker/loop_bindings.py`
- Test: `tests/test_loop_bindings.py`

**Interfaces:**
- Consumes: `process_batch` from `.processing`; `Store` from `.storage`; `RoundState` from `.loop_rounds`; `write_json_atomic` from `.atomic`; `sha256_file` from `.fingerprints`; `ContractError` from `.errors`.
- Produces: `predict_step(config, *, batch_id, generation, registry_path=None, fake_backend=False) -> RoundStep`; `route_step(config, *, batch_id, output_dir) -> RoundStep`.

**Why these are two stages when `process_batch` is one call.** `process_batch` predicts *and* routes — it writes per-document predictions and calls `store.set_router_bucket` in the same pass (`processing.py:100`, `:180`). But routing produces its own artifact that prediction does not: the bucket distribution, which is the workload axis the experiment measures. `predict_step` runs the pass and seals its result; `route_step` reads the buckets back out of the store and seals the distribution as a separate, citable file.

- [ ] **Step 1: Write the failing test**

Create `tests/test_loop_bindings.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_quality_checker.errors import ContractError
from data_quality_checker.loop_bindings import predict_step, route_step
from data_quality_checker.loop_rounds import new_round


def test_predict_step_runs_the_pipeline_and_returns_a_sha(tmp_path: Path) -> None:
    config = prepared_fixture(tmp_path)
    step = predict_step(config, batch_id="batch", generation="M003", fake_backend=True)
    artifact = step(new_round(1))
    assert artifact and len(artifact) == 64


def test_predict_step_writes_under_its_own_generation(tmp_path: Path) -> None:
    config = prepared_fixture(tmp_path)
    predict_step(config, batch_id="batch", generation="M003", fake_backend=True)(new_round(1))
    assert (config.sensitive_root / "batches" / "batch" / "predictions" / "M003").is_dir()


def test_predict_step_is_idempotent(tmp_path: Path) -> None:
    """`run_round` may re-run a step after a crash, so a repeat must not differ."""
    config = prepared_fixture(tmp_path)
    step = predict_step(config, batch_id="batch", generation="M003", fake_backend=True)
    assert step(new_round(1)) == step(new_round(1))


def test_route_step_seals_the_bucket_distribution(tmp_path: Path) -> None:
    config = prepared_fixture(tmp_path)
    predict_step(config, batch_id="batch", generation="M003", fake_backend=True)(new_round(1))

    out = tmp_path / "rounds"
    artifact = route_step(config, batch_id="batch", output_dir=out)(new_round(1))
    payload = json.loads((out / "round_001_routing.json").read_text(encoding="utf-8"))
    assert payload["round"] == 1
    assert set(payload["buckets"]) <= {"GREEN", "YELLOW", "RED", "QUARANTINE"}
    assert sum(payload["buckets"].values()) == payload["document_count"]
    assert artifact and len(artifact) == 64


def test_route_step_before_any_prediction_is_rejected(tmp_path: Path) -> None:
    config = prepared_fixture(tmp_path)
    with pytest.raises(ContractError):
        route_step(config, batch_id="batch", output_dir=tmp_path / "rounds")(new_round(1))
```

`prepared_fixture` lives in `tests/test_processing.py` with the signature
`prepared_fixture(tmp_path: Path, *, two_docs: bool = True)`. It returns the
**config alone** — not a tuple — and the batch it prepares is always the literal
id `"batch"`. Its own callers do `config = prepared_fixture(tmp_path)`. Import
it; do not rewrite it.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loop_bindings.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_quality_checker.loop_bindings'`

- [ ] **Step 3: Implement**

Create `src/data_quality_checker/loop_bindings.py`:

```python
"""Bind the pipeline's calls to the stage callables `run_round` drives.

`loop_runner` deliberately knows nothing about prediction, routing, judging or
scoring; every stage reaches it as a callable. This module is the one place
where the runner and the pipeline meet, so that knowledge lives once rather
than being spread through the orchestrator.

Each factory returns a `RoundStep`: it runs its stage and returns the sha256 of
the artifact that stage produced. `advance_round` refuses an empty artifact, so
a stage that produced nothing readable cannot advance a round.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .config import AppConfig
from .errors import ContractError
from .fingerprints import sha256_file
from .loop_rounds import RoundState
from .loop_runner import RoundStep
from .processing import process_batch
from .storage import Store

ROUTER_BUCKETS = ("GREEN", "YELLOW", "RED", "QUARANTINE")


def predict_step(
    config: AppConfig,
    *,
    batch_id: str,
    generation: str,
    registry_path: Path | None = None,
    fake_backend: bool = False,
) -> RoundStep:
    """Annotate the round's documents with the round's own model.

    `process_batch` resumes by design, so re-running after a crash re-reads the
    documents already done rather than regenerating them -- which is what makes
    this step safe for `run_round` to retry.
    """

    def step(state: RoundState) -> str:
        result = process_batch(
            config=config,
            batch_id=batch_id,
            generation=generation,
            resume=True,
            fake_backend=fake_backend,
            registry_path=registry_path,
        )
        summary = (
            config.sensitive_root / "batches" / batch_id / "predictions" / generation / "summary.json"
        )
        if not summary.is_file():
            write_json_atomic(summary, result)
        return sha256_file(summary)

    return step


def route_step(config: AppConfig, *, batch_id: str, output_dir: Path) -> RoundStep:
    """Seal the router's bucket distribution for this round.

    Prediction and routing happen in one pass, but the distribution is its own
    result: it is the workload axis the experiment measures, and it has to be a
    citable file rather than a number recomputed later from a mutable store.
    """

    def step(state: RoundState) -> str:
        with Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store:
            documents = store.list_documents(batch_id)
        if not documents:
            raise ContractError(f"batch {batch_id!r} holds no documents to route")
        buckets = Counter(str(row.get("router_bucket") or "") for row in documents)
        if "" in buckets:
            raise ContractError(
                f"{buckets['']} document(s) in batch {batch_id!r} have no router bucket; "
                "run the prediction stage first"
            )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "round": state.round_index,
            "batch_id": batch_id,
            "document_count": len(documents),
            "buckets": {name: buckets.get(name, 0) for name in ROUTER_BUCKETS},
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"round_{state.round_index:03d}_routing.json"
        write_json_atomic(path, payload)
        return sha256_file(path)

    return step
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_loop_bindings.py -q`
Expected: PASS — 5 passed.

- [ ] **Step 5: Check for a circular import**

`loop_bindings` imports from `loop_runner` (for the `RoundStep` alias) and from `processing`. Run `.venv/bin/python -c "import sys; sys.path.insert(0,'src'); import data_quality_checker.loop_bindings"` and confirm it imports cleanly. If it does not, move `RoundStep` to `loop_rounds` and import it from there in both places, and say so in your report.

- [ ] **Step 6: Lint, type-check, commit**

```bash
.venv/bin/python -m ruff format src tests && .venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src 2>&1 | tail -3
git add src/data_quality_checker/loop_bindings.py tests/test_loop_bindings.py
git commit -m "feat(loop): bind prediction and routing to round stages"
```

---

### Task 2: Judging and composition

**Files:**
- Modify: `src/data_quality_checker/loop_bindings.py`
- Test: `tests/test_loop_bindings.py`

**Interfaces:**
- Consumes: `run_judge_pilot` from `.judges`; `compose_round_training_ids`, `write_round_training_manifest` from `.loop_training`.
- Produces: `judge_step(config, *, batch_id, generation, judge_models=None, allow_external_judge=False, fake_backend=False) -> RoundStep`; `compose_step(config, *, split, cleaned_rounds, output_dir, split_manifest_sha256) -> RoundStep`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_loop_bindings.py`:

```python
from data_quality_checker.loop_bindings import compose_step, judge_step


def test_judge_step_runs_the_pilot_and_returns_a_sha(tmp_path: Path) -> None:
    config = prepared_fixture(tmp_path)
    predict_step(config, batch_id="batch", generation="M003", fake_backend=True)(new_round(1))
    artifact = judge_step(config, batch_id="batch", fake_backend=True)(new_round(1))
    assert artifact and len(artifact) == 64


def test_judge_step_refuses_an_external_call_without_consent(tmp_path: Path) -> None:
    """The pipeline's own gate must not be bypassed by wrapping it."""
    from data_quality_checker.errors import GateBlocked

    config = prepared_fixture(tmp_path)
    predict_step(config, batch_id="batch", generation="M003", fake_backend=True)(new_round(1))
    with pytest.raises(GateBlocked):
        judge_step(config, batch_id="batch", allow_external_judge=False)(new_round(1))


def test_compose_step_writes_the_round_training_manifest(tmp_path: Path) -> None:
    config = prepared_fixture(tmp_path)
    out = tmp_path / "rounds"
    split = {"train": [1, 2, 3], "valid": [10, 11], "test": [20, 21, 22]}
    artifact = compose_step(
        config,
        split=split,
        cleaned_rounds=[["dA", "dB"]],
        output_dir=out,
        split_manifest_sha256="abc123",
    )(new_round(1))
    payload = json.loads((out / "round_001_training.json").read_text(encoding="utf-8"))
    assert payload["counts"]["train"] == 5
    assert payload["split_manifest_sha256"] == "abc123"
    assert artifact and len(artifact) == 64


def test_compose_step_keeps_validation_and_test_fixed(tmp_path: Path) -> None:
    config = prepared_fixture(tmp_path)
    out = tmp_path / "rounds"
    split = {"train": [1, 2, 3], "valid": [10, 11], "test": [20, 21, 22]}
    for index, cleaned in enumerate(([], [["dA"]], [["dA"], ["dB"]]), start=1):
        compose_step(
            config,
            split=split,
            cleaned_rounds=cleaned,
            output_dir=out,
            split_manifest_sha256="abc123",
        )(new_round(index))
    counts = [
        json.loads((out / f"round_{i:03d}_training.json").read_text(encoding="utf-8"))["counts"]
        for i in (1, 2, 3)
    ]
    assert [c["valid"] for c in counts] == [2, 2, 2]
    assert [c["test"] for c in counts] == [3, 3, 3]
    assert [c["train"] for c in counts] == [3, 4, 5]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loop_bindings.py -k "judge or compose" -q`
Expected: FAIL — `ImportError: cannot import name 'judge_step'`

- [ ] **Step 3: Implement**

Add to `loop_bindings.py`:

```python
def judge_step(
    config: AppConfig,
    *,
    batch_id: str,
    judge_models: tuple[str, ...] | None = None,
    allow_external_judge: bool = False,
    fake_backend: bool = False,
) -> RoundStep:
    """Get the judge's third opinion on this round's disagreements.

    `allow_external_judge` is passed straight through rather than defaulted to
    True: the pipeline refuses an external call without it, and a binding that
    quietly supplied consent would move that decision out of the operator's
    hands.
    """

    def step(state: RoundState) -> str:
        run_judge_pilot(
            config=config,
            batch_id=batch_id,
            allow_external_judge=allow_external_judge,
            fake_backend=fake_backend,
            judge_models=judge_models,
        )
        summary = config.public_root / "batches" / batch_id / "judge_pilot_summary.json"
        if not summary.is_file():
            raise ContractError(f"judge pilot wrote no summary at {summary}")
        return sha256_file(summary)

    return step


def compose_step(
    config: AppConfig,
    *,
    split: dict[str, list[int]],
    cleaned_rounds: list[list[str]],
    output_dir: Path,
    split_manifest_sha256: str,
) -> RoundStep:
    """Compose this round's training set: canonical train plus cleaned rounds.

    Validation and test come back unchanged on every round; the loop re-runs
    checkpoint selection each round, so validation can never be folded in, and
    the test split may never influence any decision.
    """

    def step(state: RoundState) -> str:
        composition = compose_round_training_ids(split, cleaned_rounds)
        result = write_round_training_manifest(
            output_dir,
            state.round_index,
            composition,
            split_manifest_sha256=split_manifest_sha256,
        )
        return str(result["manifest_sha256"])

    return step
```

Add the two imports at the top of the module.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_loop_bindings.py -q`
Expected: PASS — 9 passed.

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/python -m ruff format src tests && .venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src 2>&1 | tail -3
git add src/data_quality_checker/loop_bindings.py tests/test_loop_bindings.py
git commit -m "feat(loop): bind judging and training composition to round stages"
```

---

### Task 3: Selection, measurement and sealing

**Files:**
- Modify: `src/data_quality_checker/g0_training.py` — rename `_canonical_evaluate` to `canonical_evaluate`
- Modify: `src/data_quality_checker/loop_bindings.py`
- Test: `tests/test_loop_bindings.py`

**Interfaces:**
- Consumes: `select_checkpoint` from `.g0`; `write_selection_record` from `.loop_selection`; `canonical_evaluate` from `.g0_training`.
- Produces: `select_step(config, *, candidates, output_dir, validation_documents, minimum_parse_count=None) -> RoundStep`; `measure_step(config, *, predictions_path, test_doc_ids_path, output_dir) -> RoundStep`; `seal_step(config, *, output_dir) -> RoundStep`.

**On the rename.** `g0_training._canonical_evaluate` already runs `benchmark/reference/evaluate.py` — which lives in this repository — against a predictions file, a ground-truth directory and a `--doc-ids-file`. That is exactly what measuring a round on the frozen 150-document test split needs. Rename it rather than writing a second evaluator: a duplicate would drift from the one the training loop already trusts, and the two would silently disagree. Do not change its body. Find every reference with `grep -rn "_canonical_evaluate" src/ tests/` and update them all.

**On `measure_step`'s ordering.** `run_round` reaches `measured` only from `checkpoint_selected`, so by the time this step runs, the round's checkpoint is already sealed on validation alone. This binding must not read the test result for any purpose other than writing it down. It returns the sha of the evaluation report and nothing else — no threshold, no comparison, no decision.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_loop_bindings.py`:

```python
from data_quality_checker.g0 import CheckpointCandidate
from data_quality_checker.loop_bindings import measure_step, seal_step, select_step


def _candidates() -> list[CheckpointCandidate]:
    return [
        CheckpointCandidate(
            update=update,
            coverage_count=50,
            parse_count=50,
            empty_output_count=0,
            runaway_output_count=0,
            core_f1=score,
            docwise_accuracy=0.30,
            recall=0.70,
            validation_loss=0.06,
        )
        for update, score in ((50, 0.70), (100, 0.78))
    ]


def test_select_step_records_the_winner(tmp_path: Path) -> None:
    config = prepared_fixture(tmp_path)
    out = tmp_path / "rounds"
    artifact = select_step(config, candidates=_candidates(), output_dir=out)(new_round(3))
    payload = json.loads((out / "round_003_checkpoint.json").read_text(encoding="utf-8"))
    assert payload["selected"]["update"] == 100
    assert payload["selection_basis"] == "validation"
    assert artifact and len(artifact) == 64


def test_select_step_records_only_validation_metrics(tmp_path: Path) -> None:
    config = prepared_fixture(tmp_path)
    out = tmp_path / "rounds"
    select_step(config, candidates=_candidates(), output_dir=out)(new_round(3))
    raw = (out / "round_003_checkpoint.json").read_text(encoding="utf-8").lower()
    assert "test" not in raw


def test_seal_step_writes_the_round_record(tmp_path: Path) -> None:
    config = prepared_fixture(tmp_path)
    out = tmp_path / "rounds"
    state = new_round(4)
    artifact = seal_step(config, output_dir=out)(state)
    payload = json.loads((out / "round_004_record.json").read_text(encoding="utf-8"))
    assert payload["round"] == 4
    assert payload["stage"] == state.stage
    assert artifact and len(artifact) == 64


def test_canonical_evaluate_is_public() -> None:
    from data_quality_checker.g0_training import canonical_evaluate

    assert callable(canonical_evaluate)
```

`measure_step` runs a real subprocess against this repository's own evaluator
(`benchmark/reference/evaluate.py`), so its test is an integration test, not a
mock. **Write one.**

The predictions file is a JSON **array** of objects, per
`evaluate.py::load_predictions` (line 358): each has an integer `doc_id`, an
optional `status` defaulting to `"success"`, and a `references` list. A row
whose `doc_id` is not an `int` is skipped silently, so getting the type wrong
would produce an empty, passing-looking evaluation:

```json
[
  {"doc_id": 3, "status": "success", "references": [{"kanun_no": "193", "kanun_ad": "Gelir Vergisi Kanunu", "madde": "94", "fikra": "", "bent": "", "source_text": "..."}]},
  {"doc_id": 7, "status": "success", "references": []}
]
```

Build it from a handful of real canonical documents: read a few `doc_*.json`
files out of `config.canonical_gt_dir`, take their ids and references, write the
array above and a matching doc-ids file, then assert the step returns a 64-char
sha and that `evaluation.json` exists with a non-zero result count.

**Assert the evaluation is non-empty**, not merely that a file appeared — a
predictions file with the wrong `doc_id` type yields an empty report that would
otherwise pass. If the evaluator cannot run here, report BLOCKED with the exact
error rather than stubbing it; a stubbed evaluator test would certify nothing.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loop_bindings.py -k "select or seal or canonical" -q`
Expected: FAIL — `ImportError: cannot import name 'select_step'`

- [ ] **Step 3: Make the evaluator public**

In `g0_training.py`, rename `_canonical_evaluate` to `canonical_evaluate` and update every reference. Do not change the body. Run `.venv/bin/python -m pytest tests/test_g0_training.py -q` and confirm the count is unchanged.

- [ ] **Step 4: Implement the three bindings**

Add to `loop_bindings.py`:

```python
def select_step(
    config: AppConfig,
    *,
    candidates: list[CheckpointCandidate],
    output_dir: Path,
    validation_documents: int = 50,
    minimum_parse_count: int = 49,
) -> RoundStep:
    """Choose this round's checkpoint, on validation alone, and record why."""

    def step(state: RoundState) -> str:
        selected = select_checkpoint(
            list(candidates),
            validation_documents=validation_documents,
            minimum_parse_count=minimum_parse_count,
        )
        result = write_selection_record(
            output_dir,
            state.round_index,
            selected,
            candidates,
            validation_documents=validation_documents,
            minimum_parse_count=minimum_parse_count,
        )
        return str(result["record_sha256"])

    return step


def measure_step(
    config: AppConfig,
    *,
    predictions_path: Path,
    test_doc_ids_path: Path,
    output_dir: Path,
) -> RoundStep:
    """Score this round's model on the frozen test split, and only record it.

    `run_round` reaches `measured` only from `checkpoint_selected`, so the
    checkpoint is already sealed on validation by the time this runs. This step
    writes the number down and returns its sha; it compares nothing and decides
    nothing, because the test split may never influence a decision.
    """

    def step(state: RoundState) -> str:
        canonical_evaluate(
            config=config,
            predictions_path=predictions_path,
            doc_ids_path=test_doc_ids_path,
            output_dir=output_dir,
        )
        report = output_dir / "evaluation.json"
        if not report.is_file():
            raise ContractError(f"evaluator wrote no report at {report}")
        return sha256_file(report)

    return step


def seal_step(config: AppConfig, *, output_dir: Path) -> RoundStep:
    """Freeze the round: write what it did and what each stage produced."""

    def step(state: RoundState) -> str:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "round": state.round_index,
            "stage": state.stage,
            "artifacts": dict(state.artifacts),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"round_{state.round_index:03d}_record.json"
        write_json_atomic(path, payload)
        return sha256_file(path)

    return step
```

Add the imports these need.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_loop_bindings.py -q`
Expected: PASS.

- [ ] **Step 6: Prove the bindings compose**

Write one test that drives `run_round` with the real bindings — predict, route, judge — against a prepared fixture with fake backends, and asserts the round reaches `judged` and stops there because `adjudicated` is manual. This is the only test that proves the runner and the pipeline actually fit together; nothing else does.

```python
def test_the_runner_drives_the_real_bindings_up_to_adjudication(tmp_path: Path) -> None:
    from data_quality_checker.loop_runner import run_round

    config = prepared_fixture(tmp_path)
    out = tmp_path / "rounds"
    state = run_round(
        out,
        1,
        steps={
            "predicted": predict_step(
                config, batch_id="batch", generation="M003", fake_backend=True
            ),
            "routed": route_step(config, batch_id="batch", output_dir=out),
            "judged": judge_step(config, batch_id="batch", fake_backend=True),
        },
    )
    assert state.stage == "judged"
    assert set(state.artifacts) == {"predicted", "routed", "judged"}
    assert all(len(sha) == 64 for sha in state.artifacts.values())
```

- [ ] **Step 7: Full suite, lint, type-check, commit**

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff format src tests && .venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src 2>&1 | tail -3
git add src/data_quality_checker/loop_bindings.py src/data_quality_checker/g0_training.py \
        tests/test_loop_bindings.py
git commit -m "feat(loop): bind selection, measurement and sealing to round stages"
```

---

## Done When

- Seven stages have bindings: predict, route, judge, compose, select, measure, seal.
- One test drives `run_round` with real bindings through to the manual stop.
- `.venv/bin/python -m pytest -q` passes; report the count.
- `ruff check` passes; `mypy src` still reports 37 errors in 9 files.
- `g0.py` is byte-identical to `main`.
- No new runtime dependency.

## Explicitly Out Of Scope, And Why

- **`adjudicated` has no binding, deliberately.** It is where a human expert settles what the router and the judge could not, and `run_round` stops before it. Automating it would remove the only step that makes the cleaned data trustworthy.
- **`trained` has no binding yet.** `train_bootstrap` raises `only canonical-only G0 is supported in v1` (`g0.py:449`), so it cannot train `M001`..`M012` at all, and training needs the shared GPU. Lifting that restriction is real work — the training export has to accept a round's composed set instead of the canonical 494 — and it deserves its own plan rather than being smuggled into a binding.

  **Update:** a later plan (`2026-09-03-dq-loop-round-training`, Task 3) lifted this. `train_round` in `loop_train.py` composes a round's training set (canonical plus every cleaned round) without touching `train_bootstrap` or `g0.py`, and `train_step` in `loop_bindings.py` now binds it. `train_bootstrap` still refuses anything but canonical-only G0, unchanged — the round path is a separate function, not a widened `train_bootstrap`. `execute=True` still hands off to the existing compute path rather than running a real training job.
- **No `loop-run` CLI command.** Running a round end to end needs `trained`, and a command that stopped halfway without saying so would mislead.
- Any GPU run.

## Post-Plan: What The Merge-Gate Review Found

This plan's draft signatures above (now corrected) are left as evidence that the
final code differs from what was first written here. Three defects surfaced
between the draft and the shipped code, none visible from reading a single
task in isolation:

1. **`judge_step` re-introduced the `"G0"` default the pipeline had just shed.**
   The draft's `judge_step(config, *, batch_id, judge_models=None, ...)` took no
   `generation`, so it would have silently re-judged generation `"G0"` no matter
   which round was actually running — exactly the hardcoded-generation
   assumption the rest of this plan's bindings were written to remove. Fixed by
   making `generation` a required keyword argument with no default.
2. **The binding fingerprinted a summary the run may not have written.**
   `run_judge_pilot` takes one of two branches — pilot or already-locked
   production — and writes a different summary file on each. A binding that
   always fingerprinted `judge_pilot_summary.json` would raise on a batch whose
   judge was already locked. Fixed by choosing the filename from
   `result["stage"]`, the branch the call actually took.
3. **The parse-quality threshold failed open at any validation size other than
   50.** `select_step`'s `minimum_parse_count: int = 49` was a literal tied to
   an assumed 50-document split. Because `select_checkpoint` gates on
   `parse_count` as an absolute count, not a ratio, a 120-document round would
   accept a checkpoint that parsed only 49/120 (41%) documents, and any round
   with fewer than 49 validation documents could never select a checkpoint at
   all. Fixed by deriving `minimum_parse_count` as `validation_documents - 1`
   when the caller leaves it unstated, keeping the two numbers coupled instead
   of letting one drift from the other.
