# DQ-Loop Wiring Unblock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the three things that make it impossible to run a DQ-Loop round at all, so a later plan can chain the round end to end.

**Architecture:** Three surgical changes to existing modules, no new subsystem. Today the pipeline can only ever predict with one sealed model (`g0/G0.json`), can only ever process a generation literally named `G0`, and can only ever prepare a batch from whole ZIP archives. A loop needs a different model each round, a generation label per round, and a batch restricted to that round's hundred documents. Each change is additive with a default that reproduces today's behaviour exactly.

**Tech Stack:** Python 3.11, stdlib only, pytest 9, ruff, mypy. Touches `g0_finalize.py`, `processing.py`, `preparation.py`.

## Global Constraints

- **Every change is backward compatible by default.** Existing callers and tests must keep working unedited: `seal_g0` still writes `g0/G0.json`, `process_batch(generation="G0")` still behaves identically, `prepare_batch` without a subset still ingests the whole archive.
- **`g0.build_split`, `g0.write_training_data` and the reference-split gate at `g0.py:217-222` stay untouched.** They are the deployed G0 model's provenance. This plan does not edit `g0.py`.
- **The model registry stays fail-closed.** A round's registry is checksum-verified exactly as `G0.json` is; a missing or mismatched adapter raises rather than falling back to another model. Silently predicting with the wrong round's model would corrupt the learning curve invisibly.
- **A round label is `M` followed by three digits** (`M000` … `M012`), matching the zero-padded round filenames `loop_rounds.py` already writes.
- No new runtime dependency. `requirements.txt` is `Flask>=3.1,<4`.
- `ruff format` and `ruff check` must pass. `mypy src` must not exceed its **37 pre-existing errors in 9 files**.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/data_quality_checker/g0_finalize.py` (modify) | `seal_round_model` writing `g0/M{k:03d}.json`, reusing the existing registry builder. |
| `src/data_quality_checker/processing.py` (modify) | Accept a round generation label and a registry path; pass the path to the backend. |
| `src/data_quality_checker/preparation.py` (modify) | Accept an optional document-id subset. |
| `tests/test_g0_finalize.py` (modify) | |
| `tests/test_processing.py` (modify or create) | |
| `tests/test_preparation.py` (modify or create) | |

---

### Task 1: A sealed registry per round

**Files:**
- Modify: `src/data_quality_checker/g0_finalize.py`
- Test: `tests/test_g0_finalize.py`

**Interfaces:**
- Consumes: `build_g0_registry` (existing, unchanged), `write_json_atomic`.
- Produces: `ROUND_LABEL_PATTERN: re.Pattern[str]`; `round_label(round_index: int) -> str`; `round_registry_path(config, round_label) -> Path`; `seal_round_model(*, config, round_label, model_snapshot_path, adapter_path, max_sequence_length, max_generation_tokens=4096) -> Path`.

`seal_g0` stays exactly as it is — it writes `G0.json` and nothing about it changes.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_g0_finalize.py`. The file already has `_snapshot(tmp_path)` and `_adapter(tmp_path)` helpers, and `test_seal_g0_writes_registry_at_public_g0_path` builds its config as a local throwaway class. Mirror that exactly:

```python
def _sealing_fixture(tmp_path: Path):
    class _Cfg:
        public_root = tmp_path / "public"

    return _Cfg(), _snapshot(tmp_path), _adapter(tmp_path)
```

```python
def test_round_label_is_zero_padded_to_three_digits() -> None:
    from data_quality_checker.g0_finalize import round_label

    assert round_label(0) == "M000"
    assert round_label(7) == "M007"
    assert round_label(12) == "M012"


def test_a_negative_round_index_has_no_label() -> None:
    from data_quality_checker.errors import ContractError
    from data_quality_checker.g0_finalize import round_label

    with pytest.raises(ContractError):
        round_label(-1)


def test_sealing_a_round_writes_its_own_registry(tmp_path) -> None:
    from data_quality_checker.g0_finalize import seal_round_model

    config, snapshot, adapter = _sealing_fixture(tmp_path)
    written = seal_round_model(
        config=config,
        round_label="M003",
        model_snapshot_path=snapshot,
        adapter_path=adapter,
        max_sequence_length=12288,
    )
    assert written.name == "M003.json"
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["adapter_sha256"]
    assert payload["max_sequence_length"] == 12288


def test_sealing_a_round_leaves_the_g0_registry_alone(tmp_path) -> None:
    from data_quality_checker.g0_finalize import seal_g0, seal_round_model

    config, snapshot, adapter = _sealing_fixture(tmp_path)
    g0_path = seal_g0(
        config=config,
        model_snapshot_path=snapshot,
        adapter_path=adapter,
        max_sequence_length=12288,
    )
    before = g0_path.read_bytes()
    seal_round_model(
        config=config,
        round_label="M003",
        model_snapshot_path=snapshot,
        adapter_path=adapter,
        max_sequence_length=12288,
    )
    assert g0_path.read_bytes() == before


def test_an_invalid_round_label_is_rejected(tmp_path) -> None:
    from data_quality_checker.errors import ContractError
    from data_quality_checker.g0_finalize import seal_round_model

    config, snapshot, adapter = _sealing_fixture(tmp_path)
    for label in ("M3", "m003", "G0", "M0003", "round-3", ""):
        with pytest.raises(ContractError):
            seal_round_model(
                config=config,
                round_label=label,
                model_snapshot_path=snapshot,
                adapter_path=adapter,
                max_sequence_length=12288,
            )
```

`_snapshot` and `_adapter` already exist in that file; do not redefine them.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_g0_finalize.py -q`
Expected: FAIL — `ImportError: cannot import name 'round_label'`

- [ ] **Step 3: Implement**

Add to `g0_finalize.py`:

```python
ROUND_LABEL_PATTERN = re.compile(r"^M\d{3}$")


def round_label(round_index: int) -> str:
    """The registry label for a round: `M` plus three digits.

    Zero-padded to match the round-state filenames `loop_rounds.py` writes, so
    a round's model and its state sort together and read the same way.
    """
    if round_index < 0:
        raise ContractError(f"round_index must be non-negative, got {round_index}")
    return f"M{round_index:03d}"


def round_registry_path(config: Any, label: str) -> Path:
    if not ROUND_LABEL_PATTERN.match(label):
        raise ContractError(f"invalid round label {label!r}; expected M followed by three digits")
    return Path(config.public_root) / "g0" / f"{label}.json"


def seal_round_model(
    *,
    config: Any,
    round_label: str,
    model_snapshot_path: Path,
    adapter_path: Path,
    max_sequence_length: int,
    max_generation_tokens: int = 4096,
) -> Path:
    """Seal one round's model into its own registry.

    Each round trains a new adapter, so each round needs its own sealed
    registry rather than overwriting `G0.json`. Keeping them separate is what
    lets a finished round be re-run or audited later against exactly the model
    that produced it.
    """
    registry = build_g0_registry(
        model_snapshot_path=model_snapshot_path,
        adapter_path=adapter_path,
        max_sequence_length=max_sequence_length,
        max_generation_tokens=max_generation_tokens,
    )
    target = round_registry_path(config, round_label)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, registry, mode=0o644)
    return target
```

Add `import re` and `from .errors import ContractError` at the top if not already present — check before adding, the module may already import `ContractError`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_g0_finalize.py -q`
Expected: PASS, existing tests unchanged in count.

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/python -m ruff format src tests && .venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src 2>&1 | tail -3
git add src/data_quality_checker/g0_finalize.py tests/test_g0_finalize.py
git commit -m "feat(loop): seal a model registry per round"
```

---

### Task 2: Predict with a round's own model

**Files:**
- Modify: `src/data_quality_checker/processing.py` — the generation guard at line 325 and the backend construction at lines 328-332
- Test: `tests/test_processing.py`

**Interfaces:**
- Consumes: `ROUND_LABEL_PATTERN` from Task 1.
- Produces: `process_batch(..., registry_path: Path | None = None)`; the generation guard widened to accept `"G0"` or a round label.

`MlxG0Backend.__init__` already accepts `registry_path: Path | None = None` — the plumbing exists, `process_batch` simply never passes it. That is the whole change on the backend side.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_processing.py`. It already has `config_for(tmp_path)` and `prepared_fixture(tmp_path, *, two_docs=True)` — use `prepared_fixture` rather than building a batch yourself, and match how the file's existing tests unpack its return value.

Note the file also already contains `test_mlx_backend_accepts_an_explicit_isolated_registry_path`, which proves `MlxG0Backend` honours an explicit `registry_path`. Your job is only to make `process_batch` pass one through; do not duplicate that backend-level test.

```python
def test_a_round_generation_is_accepted(tmp_path) -> None:
    config, batch_id = prepared_fixture(tmp_path)
    result = process_batch(
        config=config, batch_id=batch_id, generation="M003", resume=False, fake_backend=True
    )
    assert result["generation"] == "M003"


def test_g0_still_works_unchanged(tmp_path) -> None:
    config, batch_id = prepared_fixture(tmp_path)
    result = process_batch(
        config=config, batch_id=batch_id, generation="G0", resume=False, fake_backend=True
    )
    assert result["generation"] == "G0"


def test_an_unknown_generation_is_rejected(tmp_path) -> None:
    config, batch_id = prepared_fixture(tmp_path)
    for generation in ("G1", "m003", "M3", "round3", ""):
        with pytest.raises(ContractError):
            process_batch(
                config=config,
                batch_id=batch_id,
                generation=generation,
                resume=False,
                fake_backend=True,
            )


def test_round_predictions_land_under_their_own_generation(tmp_path) -> None:
    config, batch_id = prepared_fixture(tmp_path)
    process_batch(
        config=config, batch_id=batch_id, generation="M003", resume=False, fake_backend=True
    )
    predictions = config.sensitive_root / "batches" / batch_id / "predictions" / "M003"
    assert predictions.is_dir()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_processing.py -q`
Expected: FAIL — `ContractError: only G0 processing is supported`

- [ ] **Step 3: Widen the guard and thread the registry**

Replace:

```python
    if generation != "G0":
        raise ContractError("only G0 processing is supported")
```

with:

```python
    if generation != "G0" and not ROUND_LABEL_PATTERN.match(generation):
        raise ContractError(
            f"generation must be 'G0' or a round label like 'M003', got {generation!r}"
        )
```

Add `registry_path: Path | None = None` to `process_batch`'s keyword-only parameters and pass it through to the backend:

```python
        else (EchoHumanBackend() if fake_backend else MlxG0Backend(config, registry_path=registry_path))
```

Import `ROUND_LABEL_PATTERN` from `.g0_finalize`. **Check for a circular import** — if `g0_finalize` imports from `processing`, move the pattern to `constants.py` instead and import it from there in both places. Report which you did.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_processing.py tests/test_g0_finalize.py -q`
Expected: PASS.

- [ ] **Step 5: Run the full suite for regressions**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS. Every existing caller of `process_batch` passes `generation="G0"` positionally or by keyword and must be unaffected.

- [ ] **Step 6: Lint, type-check, commit**

```bash
.venv/bin/python -m ruff format src tests && .venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src 2>&1 | tail -3
git add src/data_quality_checker/processing.py tests/test_processing.py
git commit -m "feat(loop): predict a round with its own sealed model"
```

---

### Task 3: Prepare a batch from a document subset

**Files:**
- Modify: `src/data_quality_checker/preparation.py` — `prepare_batch`
- Test: `tests/test_preparation.py`

**Interfaces:**
- Produces: `prepare_batch(..., doc_ids: set[str] | None = None)`.

A round is a hundred named documents, not a whole archive. `prepare_batch` reads both ZIPs and keeps every record; the subset filter restricts which annotation records become the batch, leaving the archive itself untouched.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_preparation.py`. It already has `make_config(tmp_path)`, `write_payload_zip(path, name, payload)` and `key_file(tmp_path)` — build the archives with those, matching how the file's existing tests do it, rather than inventing a new helper.

```python
def test_a_subset_restricts_the_batch(tmp_path) -> None:
    config = make_config(tmp_path)
    annotation_zip, pool_zip, keyfile = _archives(tmp_path, config, count=5)
    result = prepare_batch(
        config=config,
        annotation_zip=annotation_zip,
        document_pool_zip=pool_zip,
        batch_id="subset",
        hmac_key_file=keyfile,
        doc_ids={"private-id-1", "private-id-3"},
    )
    assert result["document_count"] == 2


def test_no_subset_keeps_every_document(tmp_path) -> None:
    config = make_config(tmp_path)
    annotation_zip, pool_zip, keyfile = _archives(tmp_path, config, count=5)
    result = prepare_batch(
        config=config,
        annotation_zip=annotation_zip,
        document_pool_zip=pool_zip,
        batch_id="whole",
        hmac_key_file=keyfile,
    )
    assert result["document_count"] == 5


def test_a_subset_naming_an_absent_document_is_rejected(tmp_path) -> None:
    config = make_config(tmp_path)
    annotation_zip, pool_zip, keyfile = _archives(tmp_path, config, count=5)
    with pytest.raises(ContractError):
        prepare_batch(
            config=config,
            annotation_zip=annotation_zip,
            document_pool_zip=pool_zip,
            batch_id="missing",
            hmac_key_file=keyfile,
            doc_ids={"private-id-1", "not-in-the-archive"},
        )


def test_an_empty_subset_is_rejected(tmp_path) -> None:
    config = make_config(tmp_path)
    annotation_zip, pool_zip, keyfile = _archives(tmp_path, config, count=5)
    with pytest.raises(ContractError):
        prepare_batch(
            config=config,
            annotation_zip=annotation_zip,
            document_pool_zip=pool_zip,
            batch_id="empty",
            hmac_key_file=keyfile,
            doc_ids=set(),
        )
```

**`prepare_batch` does not return a document count.** I checked: its return value carries the two ZIP fingerprints and an HMAC key fingerprint, nothing about how many documents landed. So `result["document_count"]` above is a placeholder you must replace. Assert on the batch's actual contents instead — open the batch with `Store` and count its documents, the way `tests/test_processing.py` and `tests/test_judges.py` already do. Adjust the assertions to what the code really exposes; do not add a count to `prepare_batch`'s return value just to make the test convenient.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_preparation.py -q`
Expected: FAIL — `TypeError: prepare_batch() got an unexpected keyword argument 'doc_ids'`

- [ ] **Step 3: Implement the filter**

Add `doc_ids: set[str] | None = None` to `prepare_batch`'s keyword-only parameters. After the `annotations` list is built and before it is used, filter it:

```python
    if doc_ids is not None:
        if not doc_ids:
            raise ContractError("doc_ids was supplied but empty; omit it to prepare the whole archive")
        wanted = {str(value) for value in doc_ids}
        annotations = [
            record for record in annotations if _record_document_id(record) in wanted
        ]
        found = {_record_document_id(record) for record in annotations}
        missing = sorted(wanted - found)
        if missing:
            raise ContractError(f"doc_ids names {len(missing)} document(s) absent from the archive: {missing[:5]}")
```

`_record_document_id(record)` does not exist yet. **Read how `prepare_batch` already extracts a document id from a record** — it checks keys like `evrakId` and `document_id` — and factor that logic into a small helper rather than duplicating it. If the existing extraction is inline and non-trivial, extract it first as a pure function, confirm the existing tests still pass, then use it here.

An empty subset is rejected rather than silently preparing nothing, and a subset naming an absent document is rejected rather than silently preparing fewer: a round that quietly ran on 97 documents instead of 100 would corrupt the learning curve with nothing to show for it.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_preparation.py -q`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS. Report the count.

- [ ] **Step 6: Lint, type-check, commit**

```bash
.venv/bin/python -m ruff format src tests && .venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src 2>&1 | tail -3
git add src/data_quality_checker/preparation.py tests/test_preparation.py
git commit -m "feat(loop): prepare a batch from a document subset"
```

---

## Done When

- A round can be sealed, predicted with, and prepared for, without touching `G0.json` or the whole archive.
- `.venv/bin/python -m pytest -q` passes; report the count.
- `ruff check` passes; `mypy src` still reports 37 errors in 9 files.
- `g0.py` is byte-identical to `main`.
- Every pre-existing caller and test of `seal_g0`, `process_batch` and `prepare_batch` is unedited and passing.
- No new runtime dependency.

## Explicitly Out Of Scope

- **The round runner.** Chaining prepare → process → reroute → judge → serve → release → compose → train → select → measure, and driving `loop_rounds`, is the next plan. This one only removes the blockers.
- Lifting `train_bootstrap`'s `only canonical-only G0` restriction (`g0.py:433`). It belongs with the runner that will call it.
- Any GPU run or real model training.
- Verifying the Gemini judge model id — the available key is blocked service-wide and that is tracked in the runbook.
