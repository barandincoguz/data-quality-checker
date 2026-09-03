# DQ-Loop Round Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the trainer build a round's model from the canonical training split plus every round already cleaned, so `M1 … M12` can exist and the learning curve can be drawn.

**Architecture:** Three additions, no rewrite. `write_training_data` becomes id-agnostic so a training set can mix canonical documents with cleaned external ones. A new reader turns a finished round's release into training rows — the release already carries each document's text and its expert-adjudicated references, which is exactly what a training row needs. A new `train_round` entry point assembles the two, fingerprints the composition so every round gets its own run directory, and hands off to the existing preflight and compute path.

**Tech Stack:** Python 3.11, stdlib only, pytest 9, ruff, mypy. Builds on `g0.write_training_data`, `g0.validate_canonical_sources`, `loop_split`, `loop_training`, and the release output written by `release.release_batch`.

## Global Constraints

- **`train_bootstrap` keeps working exactly as it does today.** It reproduces the historical G0 run and its `run_id` must not move. `train_round` is a separate entry point; G0's path is not rerouted through it.
- **`write_training_data`'s behaviour for G0 must be byte-identical.** Generalising its key type is a typing change, not a behavioural one — dict lookups do not care. Prove it: the existing training data for a canonical split must hash the same before and after.
- **Only trustworthy rows become training data.** A release separates `expert_adjudicated` and `consensus_clean` from quarantine. Quarantined documents are excluded — a round that trained on documents the pipeline could not process would poison the very data the loop exists to clean.
- **The run fingerprint must include the round's composition.** Two rounds train on different data; if they fingerprinted the same, the second would resume into the first's run directory and silently train the wrong model.
- No new runtime dependency. `requirements.txt` is `Flask>=3.1,<4`.
- `ruff format` and `ruff check` must pass. `mypy src` must not exceed its **37 pre-existing errors in 9 files**.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/data_quality_checker/g0.py` (modify) | `write_training_data` accepts any id type. |
| `src/data_quality_checker/loop_training.py` (modify) | `round_training_documents` reads a finished round's release into training rows. |
| `src/data_quality_checker/loop_train.py` (create) | `train_round`: assemble, fingerprint, write, preflight. |
| `tests/test_g0.py` (modify) | |
| `tests/test_loop_training.py` (modify) | |
| `tests/test_loop_train.py` (create) | |

---

### Task 1: A training set that can mix id spaces

**Files:**
- Modify: `src/data_quality_checker/g0.py` — `write_training_data` only
- Modify: `src/data_quality_checker/loop_training.py` — add `round_training_documents`
- Test: `tests/test_g0.py`, `tests/test_loop_training.py`

**Interfaces:**
- Produces: `write_training_data(*, output_dir, documents: Mapping[Any, dict[str, Any]], split: dict[str, list[Any]]) -> dict[str, Any]`; `round_training_documents(config, *, batch_ids: Sequence[str]) -> dict[str, dict[str, Any]]`.

**On the typing change.** `write_training_data` is annotated `documents: dict[int, ...]` and `split: dict[str, list[int]]` because it has only ever seen canonical documents. A round's training set also holds external documents whose ids are strings. The function's body does nothing but look ids up in a dict, so widening the annotation changes no behaviour — but leaving it narrow would force a second writer, and two writers producing training rows is how a subtle format difference gets into one model and not another.

**On the release reader.** `release_batch` writes `sensitive_root/releases/{batch_id}/expert_adjudicated.jsonl` and `consensus_clean.jsonl`. Every row already carries `internal_doc_id`, `text` and `references` — the three fields a training row needs. Read both files; **ignore any other file in that directory**, in particular anything quarantine-related.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_g0.py`:

```python
def test_write_training_data_accepts_string_ids(tmp_path: Path) -> None:
    from data_quality_checker.g0 import write_training_data

    documents = {
        "canonical:1": {"text": "birinci belge", "references": []},
        "d000304": {"text": "ikinci belge", "references": []},
        "canonical:2": {"text": "ucuncu belge", "references": []},
    }
    split = {"train": ["canonical:1", "d000304"], "valid": ["canonical:2"], "test": []}
    result = write_training_data(output_dir=tmp_path, documents=documents, split=split)
    assert result["files"]["train"]["count"] == 2
    assert result["files"]["valid"]["count"] == 1
    assert result["files"]["test"]["count"] == 0
```

Append to `tests/test_loop_training.py`:

```python
import json
from pathlib import Path

import pytest

from data_quality_checker.errors import ContractError
from data_quality_checker.loop_training import round_training_documents


def _release(root: Path, batch_id: str, expert: list[dict], consensus: list[dict]) -> None:
    directory = root / "releases" / batch_id
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in (("expert_adjudicated", expert), ("consensus_clean", consensus)):
        (directory / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )


class _Cfg:
    def __init__(self, root: Path) -> None:
        self.sensitive_root = root


def test_reads_both_trust_levels(tmp_path: Path) -> None:
    _release(
        tmp_path,
        "round1",
        expert=[{"internal_doc_id": "d1", "text": "bir", "references": [{"kanun_no": "193"}]}],
        consensus=[{"internal_doc_id": "d2", "text": "iki", "references": []}],
    )
    documents = round_training_documents(_Cfg(tmp_path), batch_ids=["round1"])
    assert set(documents) == {"d1", "d2"}
    assert documents["d1"]["text"] == "bir"
    assert documents["d1"]["references"] == [{"kanun_no": "193"}]


def test_reads_several_rounds(tmp_path: Path) -> None:
    _release(tmp_path, "round1", expert=[{"internal_doc_id": "d1", "text": "bir", "references": []}], consensus=[])
    _release(tmp_path, "round2", expert=[{"internal_doc_id": "d2", "text": "iki", "references": []}], consensus=[])
    documents = round_training_documents(_Cfg(tmp_path), batch_ids=["round1", "round2"])
    assert set(documents) == {"d1", "d2"}


def test_a_missing_release_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ContractError):
        round_training_documents(_Cfg(tmp_path), batch_ids=["never-released"])


def test_the_same_document_in_two_rounds_is_rejected(tmp_path: Path) -> None:
    _release(tmp_path, "round1", expert=[{"internal_doc_id": "d1", "text": "bir", "references": []}], consensus=[])
    _release(tmp_path, "round2", expert=[{"internal_doc_id": "d1", "text": "bir", "references": []}], consensus=[])
    with pytest.raises(ContractError):
        round_training_documents(_Cfg(tmp_path), batch_ids=["round1", "round2"])


def test_a_row_missing_text_or_references_is_rejected(tmp_path: Path) -> None:
    _release(tmp_path, "round1", expert=[{"internal_doc_id": "d1", "references": []}], consensus=[])
    with pytest.raises(ContractError):
        round_training_documents(_Cfg(tmp_path), batch_ids=["round1"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_g0.py tests/test_loop_training.py -q`
Expected: FAIL — `ImportError: cannot import name 'round_training_documents'`, and the string-id test raising a typing-unrelated failure only if the body actually depends on int keys (it should not; if it passes immediately, say so — that is the behaviour-unchanged evidence).

- [ ] **Step 3: Widen `write_training_data`'s annotations**

In `g0.py`, change only the signature:

```python
def write_training_data(
    *,
    output_dir: Path,
    documents: Mapping[Any, dict[str, Any]],
    split: dict[str, list[Any]],
) -> dict[str, Any]:
```

Add `Mapping` to the `collections.abc` import and `Any` if absent. **Change nothing in the body.**

- [ ] **Step 4: Prove G0's output is byte-identical**

Before and after your change, run the existing canonical training-data path and compare. The simplest honest check: `git stash` your edit, run the existing `tests/test_g0.py` tests that exercise `write_training_data`, record the `jsonl_sha256` values they produce, restore, re-run, compare. Report both sets. If any hash moves, stop — the change was not behaviour-neutral.

- [ ] **Step 5: Implement `round_training_documents`**

Add to `loop_training.py`:

```python
def round_training_documents(
    config: Any, *, batch_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Read finished rounds' releases into training rows.

    A release already carries each document's text and the references the
    expert settled on, which is exactly what a training row needs. Only the
    two trustworthy tiers are read: `expert_adjudicated`, where a human decided,
    and `consensus_clean`, where annotator and model agreed and the round's
    GREEN sample raised nothing. Quarantined documents are excluded -- training
    on documents the pipeline could not process would poison the very data this
    loop exists to clean.
    """
    documents: dict[str, dict[str, Any]] = {}
    for batch_id in batch_ids:
        directory = Path(config.sensitive_root) / "releases" / batch_id
        found = False
        for name in ("expert_adjudicated", "consensus_clean"):
            path = directory / f"{name}.jsonl"
            if not path.is_file():
                continue
            found = True
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ContractError(f"{path}:{line_number} is not JSON: {exc}") from exc
                doc_id = row.get("internal_doc_id")
                text = row.get("text")
                references = row.get("references")
                if not isinstance(doc_id, str) or not doc_id:
                    raise ContractError(f"{path}:{line_number} has no internal_doc_id")
                if not isinstance(text, str) or not text:
                    raise ContractError(f"{path}:{line_number} ({doc_id}) has no text")
                if not isinstance(references, list):
                    raise ContractError(f"{path}:{line_number} ({doc_id}) has no references list")
                if doc_id in documents:
                    raise ContractError(
                        f"document {doc_id!r} appears in more than one round's release"
                    )
                documents[doc_id] = {"text": text, "references": references}
        if not found:
            raise ContractError(f"no release found for batch {batch_id!r} under {directory}")
    return documents
```

Add the imports it needs.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_g0.py tests/test_loop_training.py -q`
Expected: PASS.

- [ ] **Step 7: Full suite, lint, type-check, commit**

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff format src tests && .venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src 2>&1 | tail -3
git add src/data_quality_checker/g0.py src/data_quality_checker/loop_training.py \
        tests/test_g0.py tests/test_loop_training.py
git commit -m "feat(loop): read a finished round's release into training rows"
```

---

### Task 2: `train_round`

**Files:**
- Create: `src/data_quality_checker/loop_train.py`
- Test: `tests/test_loop_train.py`

**Interfaces:**
- Consumes: `validate_canonical_sources`, `write_training_data`, `SYSTEM_PROMPT`, `PROMPT_VARIANT`, `EXAMPLE_BANK_SHA256`, `TRAINING_VIEW_POLICY` from `.g0`; `compose_round_training_ids`, `round_training_documents` from `.loop_training`; `fingerprint_json`, `sha256_text` from `.fingerprints`.
- Produces: `train_round(config, *, generation, split, cleaned_batch_ids, execute=False) -> dict[str, Any]`.

`split` is the loop split from `loop_split.build_loop_split` — `{"train": [int], "valid": [int], "test": [int]}` over canonical ids. `cleaned_batch_ids` are the batches of rounds already finished, in order.

**Why the fingerprint must include the composition.** `train_bootstrap` derives its `run_id` from the canonical manifest, the example bank, the split manifest and the prompt. Every round shares all four. If `train_round` fingerprinted the same inputs, round 2 would resume into round 1's run directory, find a finished training run there, and silently produce round 1's model labelled as round 2's. The composition — the ordered cleaned batch ids and the resulting training-set size — must be part of it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_loop_train.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_quality_checker.errors import ContractError
from data_quality_checker.loop_train import train_round

SPLIT = {"train": [1, 2, 3], "valid": [10, 11], "test": [20, 21]}


def test_round_zero_trains_on_canonical_only(tmp_path: Path) -> None:
    config = canonical_fixture(tmp_path)
    result = train_round(config, generation="M000", split=SPLIT, cleaned_batch_ids=[])
    assert result["training_documents"] == 3
    assert result["cleaned_batch_ids"] == []


def test_a_later_round_adds_its_cleaned_documents(tmp_path: Path) -> None:
    config = canonical_fixture(tmp_path)
    release_fixture(config, "round1", ["d1", "d2"])
    result = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"])
    assert result["training_documents"] == 5


def test_validation_and_test_never_grow(tmp_path: Path) -> None:
    config = canonical_fixture(tmp_path)
    release_fixture(config, "round1", ["d1", "d2"])
    base = train_round(config, generation="M000", split=SPLIT, cleaned_batch_ids=[])
    grown = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"])
    assert grown["validation_documents"] == base["validation_documents"] == 2
    assert grown["test_documents"] == base["test_documents"] == 2


def test_two_rounds_get_different_run_directories(tmp_path: Path) -> None:
    config = canonical_fixture(tmp_path)
    release_fixture(config, "round1", ["d1", "d2"])
    first = train_round(config, generation="M000", split=SPLIT, cleaned_batch_ids=[])
    second = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"])
    assert first["run_id"] != second["run_id"]


def test_the_same_round_is_idempotent(tmp_path: Path) -> None:
    config = canonical_fixture(tmp_path)
    release_fixture(config, "round1", ["d1", "d2"])
    first = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"])
    second = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"])
    assert first["run_id"] == second["run_id"]


def test_an_invalid_generation_is_rejected(tmp_path: Path) -> None:
    config = canonical_fixture(tmp_path)
    for generation in ("G0", "m001", "M1", "round1", ""):
        with pytest.raises(ContractError):
            train_round(config, generation=generation, split=SPLIT, cleaned_batch_ids=[])
```

`canonical_fixture` must satisfy `validate_canonical_sources`, which requires
500 `doc_*.json` files and an exact manifest hash — so point at the
repository's real canonical GT rather than synthesising it. `tests/test_g0.py`
already has exactly this helper at lines 26-40: it calls `load_config()` to get
the live config, copies the preset JSON, then overrides the roots to `tmp_path`
while keeping `"canonical_gt_dir": str(live.canonical_gt_dir)`. **Import that
helper if it is importable; otherwise copy its shape verbatim.**

`release_fixture(config, batch_id, doc_ids)` writes the two release files under
`config.sensitive_root / "releases" / batch_id`, in the shape Task 1's
`round_training_documents` reads: one JSON object per line with
`internal_doc_id`, `text` and `references`.

If `validate_canonical_sources` cannot be satisfied in a test, report BLOCKED with the exact error rather than weakening the function — that gate is what stops a round training against a drifted ground truth.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loop_train.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_quality_checker.loop_train'`

- [ ] **Step 3: Implement**

Create `src/data_quality_checker/loop_train.py`. `train_bootstrap` at
`g0.py:447` is the model to follow — read it first. Its directory layout is:

```python
    run_dir = config.training_runs_root / run_id
    data_dir = config.sensitive_root / "g0" / run_id / "data"
    public_dir = config.public_root / "g0" / run_id
```

Use the same shape but under `"loop"` rather than `"g0"`, so a round's run
directory can never collide with the historical G0 run's:

```python
    run_dir = config.training_runs_root / run_id
    data_dir = config.sensitive_root / "loop" / run_id / "data"
    public_dir = config.public_root / "loop" / run_id
```

Assemble in this order:

1. Validate `generation` against `ROUND_LABEL_PATTERN` from `.g0_finalize`, and
   reject `"G0"` explicitly — this entry point is for rounds, and a caller
   reaching for G0 here wants `train_bootstrap`.
2. `validate_canonical_sources(config)` — the gate that stops a round training
   against a drifted ground truth.
3. `round_training_documents(config, batch_ids=cleaned_batch_ids)`.
4. `compose_round_training_ids(split, [[...]])` — one inner list per cleaned
   batch, in order, holding that batch's document ids.
5. Build the document mapping: canonical documents keyed `f"canonical:{id}"`
   from the validated source, merged with the cleaned ones keyed by their own
   ids.
6. Fingerprint. Start from the same payload `train_bootstrap` uses — canonical
   manifest, example bank, split manifest, prompt, training-view policy — and
   add `generation`, the ordered `cleaned_batch_ids`, and the composed
   training-set size. `run_id = f"dqcheck_loop_{fingerprint[:12]}"`.
7. `write_training_data(output_dir=data_dir, documents=..., split=composed)`.

Return `run_id`, `training_documents`, `validation_documents`,
`test_documents`, `cleaned_batch_ids`, `data_dir`, and the manifest
`write_training_data` produced.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_loop_train.py -q`
Expected: PASS — 6 passed.

- [ ] **Step 5: Full suite, lint, type-check, commit**

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff format src tests && .venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src 2>&1 | tail -3
git add src/data_quality_checker/loop_train.py tests/test_loop_train.py
git commit -m "feat(loop): assemble and fingerprint a round's training run"
```

---

### Task 3: The `trained` binding, and the records

**Files:**
- Modify: `src/data_quality_checker/loop_bindings.py`
- Modify: `docs/superpowers/plans/2026-09-03-dq-loop-bindings.md`
- Test: `tests/test_loop_bindings.py`

**Interfaces:**
- Produces: `train_step(config, *, generation, split, cleaned_batch_ids, execute=False) -> RoundStep`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_loop_bindings.py`:

```python
def test_train_step_returns_the_sha_of_its_data_manifest(tmp_path: Path) -> None:
    from data_quality_checker.loop_bindings import train_step

    config = canonical_and_release_fixture(tmp_path)
    split = {"train": [1, 2, 3], "valid": [10, 11], "test": [20, 21]}
    step = train_step(config, generation="M001", split=split, cleaned_batch_ids=["round1"])
    artifact = step(new_round(1))

    result = train_round(
        config, generation="M001", split=split, cleaned_batch_ids=["round1"]
    )
    manifest = Path(result["data_dir"]) / "split_manifest.json"
    assert artifact == sha256_file(manifest)
```

`write_training_data` writes `split_manifest.json` into the directory it is
given (`g0.py:280`), and `train_round` returns that directory as `data_dir`.
Calling `train_round` a second time to learn the path is safe because the run
is idempotent — Task 2 pins that. Assert the value, never merely
`len(...) == 64`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loop_bindings.py -k train_step -q`
Expected: FAIL — `ImportError: cannot import name 'train_step'`

- [ ] **Step 3: Implement the binding**

Add `train_step` to `loop_bindings.py`, calling `train_round` and returning the sha of a file it wrote. Make `generation`, `split` and `cleaned_batch_ids` **required** keyword parameters — each is a fact the caller knows, and every default on this branch that guessed one of them produced a defect.

- [ ] **Step 4: Update the records**

Four documents state that the training path is frozen and that `trained` cannot be bound. That is no longer true and a reader would be misled. Update:

- `docs/superpowers/plans/2026-09-03-dq-loop-bindings.md` — its "Explicitly Out Of Scope" says `trained` cannot be bound. Add a note that a later plan lifted it.
- `/Users/student2/ner-project/Journal/experiments/dq_loop/DESIGN.md` §6.4 — "artımlı eğitim" is listed as an open precondition. Mark it closed, naming `train_round`.
- `/Users/student2/ner-project/docs/ROADMAP.md` §2.-1 — the same precondition in the checklist.
- `/Users/student2/ner-project/Journal/JOURNAL_LEDGER.md` — append an entry recording that the canonical-only restriction was lifted, that `train_bootstrap` and G0's `run_id` are unchanged, and that `write_training_data`'s widening was proven byte-neutral.

Do **not** touch `Journal/evidence/claims/CLAIMS_REGISTER.csv` or `Journal/reproducibility/input_artifacts.lock.json`. Run `make -C Journal audit` from `/Users/student2/ner-project` afterwards and confirm it still passes.

- [ ] **Step 5: Full suite, lint, type-check, audit, commit**

```bash
cd /Users/student2/data-quality-checker
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff format src tests && .venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src 2>&1 | tail -3
cd /Users/student2/ner-project && make -C Journal audit
```

Commit the code in `data-quality-checker` and the records in `ner-project` separately.

---

## Done When

- `train_round` builds a round's training set from the canonical split plus every cleaned round, and each round gets its own run directory.
- `train_step` binds it, with no guessed defaults.
- `train_bootstrap` is unchanged and G0's `run_id` has not moved.
- `write_training_data`'s output for a canonical split is byte-identical to before.
- `.venv/bin/python -m pytest -q` passes; report the count.
- `ruff check` passes; `mypy src` still reports 37 errors in 9 files.
- `make -C Journal audit` passes.

## Explicitly Out Of Scope

- **Running a real training job.** `train_round` prepares and fingerprints; `execute=True` hands off to the existing compute path, which needs the shared GPU. No GPU run is part of this plan.
- The `serve` subcommand's missing generation flag, which still stops an expert adjudicating a round from the CLI.
- Verifying the Gemini judge model id.
