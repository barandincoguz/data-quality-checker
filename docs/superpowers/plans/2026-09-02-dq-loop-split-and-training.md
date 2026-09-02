# DQ-Loop Split, Round Batches and Checkpoint Selection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic data scaffolding and checkpoint-selection machinery the DQ-Loop needs, so that training round `k` is fully specified by manifests written before the loop starts.

**Architecture:** Four independent, file-producing units. A canonical split (`300/50/150`) written once as a sealed manifest; twelve round batches of exactly `100` drawn from the external pool, balanced and seed-fixed; a round-aware training-data assembler that composes `300 canonical + cleaned rounds 1..k`; and a contract-driven checkpoint selector that records why each round's checkpoint won. Everything is testable with fixtures and the existing deterministic fake trainer — **no GPU run is part of this plan.**

**Tech Stack:** Python 3.11, stdlib only, pytest 9, ruff, mypy. Existing modules: `g0.py` (splits, contracts, selection), `g0_training.py` (orchestration), `fake_training.py` (deterministic trainer), `atomic.py` (`write_json_atomic`, `write_jsonl_atomic`), `fingerprints.py` (`sha256_file`).

## Global Constraints

- **Do not touch `build_split`, `write_training_data`, or the reference-split gate.** `g0.py:217-222` regenerates the historical `394/50/50` split and raises `"generated 394/50/50 split differs from cleaned-GT v2 manifest"` if it drifts from `config.reference_split_manifest_path`. That gate is the deployed `G0`'s provenance. The DQ-Loop gets **new, parallel** functions and its own manifests; the historical path stays byte-identical.
- **Split sizes: `300` train / `50` validation / `150` test**, over the canonical `500`. Validation and test are drawn only from the `494` eligible documents; the six few-shot exemplars `EXEMPLAR_DOC_IDS = frozenset({1, 10, 16, 18, 36, 77})` go into **train**. `294 eligible + 6 exemplars = 300`. Exemplars in train are harmless — the model sees them in the prompt at inference anyway — while a exemplar in validation or test would be a leak, so those two stay exemplar-free.
- **Validation and test are never trained on**, for the life of the experiment.
- **The test split may never influence a decision.** Any selection record that references a test metric is a defect. Task 4 enforces this with a test.
- **Round batches are fixed before the loop starts** and never re-drawn. Active learning is deliberately excluded: the claim is "more cleaned data helps", not "smarter selection helps".
- **Balance round batches** on `length_quartile`, `text_channel`, `annotator`, `reference_count` (banded), and spread zero-reference documents. **Do not balance `difficulty`** — the pool holds only 2 `Zor` documents, which cannot be spread over 12 batches.
- No new runtime dependency. `requirements.txt` is `Flask>=3.1,<4`.
- `ruff format` and `ruff check` must pass. `mypy src` must not exceed its **37 pre-existing errors in 9 files**.
- Every artifact is written with `write_json_atomic` / `write_jsonl_atomic` and sealed with a `sha256`, matching the repository's existing durability convention.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/data_quality_checker/loop_split.py` (create) | The DQ-Loop's canonical split and its sealed manifest. Separate from `g0.build_split`. |
| `src/data_quality_checker/loop_batches.py` (create) | Round batch construction, balancing, and manifest. |
| `src/data_quality_checker/loop_training.py` (create) | Round-aware training-set composition (`canonical + cleaned rounds 1..k`). |
| `src/data_quality_checker/g0.py` (modify) | Make `select_checkpoint` contract-driven instead of hardcoding `coverage_count == 50`. |
| `src/data_quality_checker/loop_selection.py` (create) | Per-round checkpoint selection record. |
| `tests/test_loop_split.py` (create) | |
| `tests/test_loop_batches.py` (create) | |
| `tests/test_loop_training.py` (create) | |
| `tests/test_loop_selection.py` (create) | |
| `tests/test_g0.py` (modify) | Extend for the contract-driven selector. |

---

### Task 1: The DQ-Loop canonical split

**Files:**
- Create: `src/data_quality_checker/loop_split.py`
- Test: `tests/test_loop_split.py`

**Interfaces:**
- Consumes: `EXEMPLAR_DOC_IDS` from `.constants`, `ContractError` from `.errors`, `write_json_atomic` from `.atomic`, `sha256_file` from `.fingerprints`.
- Produces: `LOOP_SPLIT_SEED: int`, `LoopSplitSizes` dataclass, `build_loop_split(doc_ids, *, sizes, seed) -> dict[str, list[int]]`, `write_loop_split_manifest(output_dir, split, *, sizes, seed) -> dict[str, Any]`.

**The arithmetic, resolved.** The canonical corpus is `500` documents, of which
six are few-shot exemplars carried inside the G0 prompt. A exemplar in
validation or test would be a leak: the model has already seen it at inference
time. A exemplar in *train* is harmless for exactly the same reason — it sees
them either way.

So the six go to train and the numbers come out exactly as the design states:

```
500 canonical
 ├── 150  test        drawn from the 494 eligible, exemplar-free
 ├──  50  validation  drawn from the 494 eligible, exemplar-free
 └── 300  train       294 eligible + the 6 exemplars
```

Training therefore starts at `300` and reaches `300 + 1200 = 1500` after twelve
rounds, matching `DESIGN.md` §2.2. The manifest records `exemplars_in_train`
explicitly so a reader can see why train is six larger than the eligible
remainder.

- [ ] **Step 1: Write the failing test**

Create `tests/test_loop_split.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_quality_checker.constants import EXEMPLAR_DOC_IDS
from data_quality_checker.errors import ContractError
from data_quality_checker.loop_split import (
    LOOP_SPLIT_SEED,
    LoopSplitSizes,
    build_loop_split,
    write_loop_split_manifest,
)

ALL_CANONICAL = list(range(1, 501))


def test_split_sizes_match_the_contract() -> None:
    split = build_loop_split(ALL_CANONICAL, sizes=LoopSplitSizes(), seed=LOOP_SPLIT_SEED)
    assert len(split["train"]) == 300
    assert len(split["valid"]) == 50
    assert len(split["test"]) == 150


def test_splits_are_disjoint_and_cover_the_whole_corpus() -> None:
    split = build_loop_split(ALL_CANONICAL, sizes=LoopSplitSizes(), seed=LOOP_SPLIT_SEED)
    train, valid, test = set(split["train"]), set(split["valid"]), set(split["test"])
    assert train & valid == set()
    assert train & test == set()
    assert valid & test == set()
    assert train | valid | test == set(ALL_CANONICAL)


def test_exemplars_are_in_train_and_never_in_the_held_out_sets() -> None:
    split = build_loop_split(ALL_CANONICAL, sizes=LoopSplitSizes(), seed=LOOP_SPLIT_SEED)
    assert EXEMPLAR_DOC_IDS <= set(split["train"])
    assert set(split["valid"]) & EXEMPLAR_DOC_IDS == set()
    assert set(split["test"]) & EXEMPLAR_DOC_IDS == set()


def test_split_is_deterministic_for_a_seed() -> None:
    first = build_loop_split(ALL_CANONICAL, sizes=LoopSplitSizes(), seed=LOOP_SPLIT_SEED)
    second = build_loop_split(ALL_CANONICAL, sizes=LoopSplitSizes(), seed=LOOP_SPLIT_SEED)
    assert first == second


def test_a_different_seed_produces_a_different_split() -> None:
    first = build_loop_split(ALL_CANONICAL, sizes=LoopSplitSizes(), seed=LOOP_SPLIT_SEED)
    second = build_loop_split(ALL_CANONICAL, sizes=LoopSplitSizes(), seed=LOOP_SPLIT_SEED + 1)
    assert first != second


def test_wrong_eligible_count_is_rejected() -> None:
    with pytest.raises(ContractError):
        build_loop_split(list(range(1, 400)), sizes=LoopSplitSizes(), seed=LOOP_SPLIT_SEED)


def test_sizes_that_do_not_fit_the_pool_are_rejected() -> None:
    with pytest.raises(ContractError):
        build_loop_split(
            ALL_CANONICAL,
            sizes=LoopSplitSizes(validation_documents=400, test_documents=300),
            seed=LOOP_SPLIT_SEED,
        )


def test_a_train_size_that_does_not_add_up_is_rejected() -> None:
    # 494 eligible - 50 - 150 = 294, plus 6 exemplars = 300. Asking for 320
    # must fail loudly rather than silently producing 300.
    with pytest.raises(ContractError):
        build_loop_split(
            ALL_CANONICAL, sizes=LoopSplitSizes(train_documents=320), seed=LOOP_SPLIT_SEED
        )


def test_manifest_records_the_exemplars_and_seals_itself(tmp_path: Path) -> None:
    sizes = LoopSplitSizes()
    split = build_loop_split(ALL_CANONICAL, sizes=sizes, seed=LOOP_SPLIT_SEED)
    result = write_loop_split_manifest(tmp_path, split, sizes=sizes, seed=LOOP_SPLIT_SEED)

    payload = json.loads((tmp_path / "loop_split_manifest.json").read_text(encoding="utf-8"))
    assert payload["seed"] == LOOP_SPLIT_SEED
    assert payload["counts"] == {"train": 300, "valid": 50, "test": 150}
    assert payload["exemplars_in_train"] == sorted(EXEMPLAR_DOC_IDS)
    assert result["manifest_sha256"] == payload_sha(tmp_path)


def payload_sha(tmp_path: Path) -> str:
    from data_quality_checker.fingerprints import sha256_file

    return sha256_file(tmp_path / "loop_split_manifest.json")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loop_split.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_quality_checker.loop_split'`

- [ ] **Step 3: Implement the module**

Create `src/data_quality_checker/loop_split.py`:

```python
"""Canonical train/validation/test split for the DQ-Loop experiment.

Deliberately separate from `g0.build_split`. That function regenerates the
deployed G0's historical 394/50/50 split and is checked against a frozen
reference manifest, so it is provenance and must not move. The loop needs a
different shape and gets its own function, seed and manifest.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .constants import EXEMPLAR_DOC_IDS
from .errors import ContractError
from .fingerprints import sha256_file

LOOP_SPLIT_SEED = 20260902
ELIGIBLE_CANONICAL_DOCUMENTS = 494


@dataclass(frozen=True)
class LoopSplitSizes:
    """Requested split sizes over the canonical 500.

    Validation and test are drawn from the 494 eligible documents only. The six
    few-shot exemplars go to train, so `train_documents` is the eligible
    remainder plus six.
    """

    train_documents: int = 300
    validation_documents: int = 50
    test_documents: int = 150


def build_loop_split(
    doc_ids: list[int], *, sizes: LoopSplitSizes, seed: int
) -> dict[str, list[int]]:
    """Split the canonical corpus, keeping every exemplar in train.

    Test is drawn first and validation second, so both keep exactly their
    requested sizes; train takes the eligible remainder and then the exemplars
    are appended to it.
    """
    everything = set(doc_ids)
    eligible = sorted(everything - EXEMPLAR_DOC_IDS)
    if len(eligible) != ELIGIBLE_CANONICAL_DOCUMENTS:
        raise ContractError(
            f"expected {ELIGIBLE_CANONICAL_DOCUMENTS} eligible canonical docs, found {len(eligible)}"
        )
    missing_exemplars = EXEMPLAR_DOC_IDS - everything
    if missing_exemplars:
        raise ContractError(f"exemplars absent from the corpus: {sorted(missing_exemplars)}")
    held_out = sizes.validation_documents + sizes.test_documents
    if held_out >= len(eligible):
        raise ContractError(
            f"validation+test ({held_out}) leaves no training documents in a pool of {len(eligible)}"
        )
    shuffled = eligible[:]
    random.Random(seed).shuffle(shuffled)
    test = shuffled[: sizes.test_documents]
    valid = shuffled[sizes.test_documents : held_out]
    train = shuffled[held_out:] + sorted(EXEMPLAR_DOC_IDS)
    if len(train) != sizes.train_documents:
        raise ContractError(
            f"train came out at {len(train)}, expected {sizes.train_documents}; "
            "sizes do not fit the corpus"
        )
    return {"train": sorted(train), "valid": sorted(valid), "test": sorted(test)}


def write_loop_split_manifest(
    output_dir: Path,
    split: dict[str, list[int]],
    *,
    sizes: LoopSplitSizes,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {name: len(split[name]) for name in ("train", "valid", "test")}
    payload = {
        "schema_version": 1,
        "seed": seed,
        "eligible_documents": ELIGIBLE_CANONICAL_DOCUMENTS,
        "counts": counts,
        "exemplars_in_train": sorted(EXEMPLAR_DOC_IDS),
        "held_out_are_exemplar_free": True,
        "splits": split,
    }
    path = output_dir / "loop_split_manifest.json"
    write_json_atomic(path, payload)
    return {"path": str(path), "manifest_sha256": sha256_file(path), "counts": counts}
```

Note two things. Test is drawn first and validation second, so both hold exactly their requested sizes. And `build_loop_split` asserts the resulting train size matches the contract rather than trusting the arithmetic — if someone changes a size and the numbers stop adding up, it fails loudly instead of producing a quietly wrong split.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_loop_split.py -q`
Expected: PASS — 9 passed.

- [ ] **Step 5: Confirm the historical split is untouched**

Run: `.venv/bin/python -m pytest tests/test_g0.py -q`
Expected: PASS, unchanged count. `build_split` and its reference-manifest gate must not have moved.

- [ ] **Step 6: Lint, type-check, commit**

```bash
.venv/bin/python -m ruff format src tests && .venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src 2>&1 | tail -3
git add src/data_quality_checker/loop_split.py tests/test_loop_split.py
git commit -m "feat(loop): add the DQ-Loop canonical split beside the frozen G0 split"
```

---

### Task 2: Round batch manifests

**Files:**
- Create: `src/data_quality_checker/loop_batches.py`
- Test: `tests/test_loop_batches.py`

**Interfaces:**
- Consumes: `ContractError`, `write_json_atomic`, `sha256_file`.
- Produces: `LOOP_BATCH_SEED: int`, `BALANCE_FIELDS: tuple[str, ...]`, `build_round_batches(documents, *, rounds, size, seed) -> list[list[str]]`, `write_batch_manifest(output_dir, batches, *, seed) -> dict[str, Any]`.

Each document is a dict carrying at least `doc_id` (str), and optionally `length_quartile`, `text_channel`, `annotator`, `reference_count`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_loop_batches.py`:

```python
from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

from data_quality_checker.errors import ContractError
from data_quality_checker.loop_batches import (
    LOOP_BATCH_SEED,
    build_round_batches,
    write_batch_manifest,
)


def make_documents(count: int = 1200) -> list[dict[str, object]]:
    quartiles = ("q1", "q2", "q3", "q4")
    channels = ("pdfText", "htmlText")
    return [
        {
            "doc_id": f"d{index:05d}",
            "length_quartile": quartiles[index % 4],
            "text_channel": channels[index % 2],
            "annotator": f"annotator_{index % 14:02d}",
            "reference_count": 0 if index % 300 == 0 else (index % 9) + 1,
        }
        for index in range(count)
    ]


def test_produces_the_requested_shape() -> None:
    batches = build_round_batches(make_documents(), rounds=12, size=100, seed=LOOP_BATCH_SEED)
    assert len(batches) == 12
    assert all(len(batch) == 100 for batch in batches)


def test_batches_are_disjoint() -> None:
    batches = build_round_batches(make_documents(), rounds=12, size=100, seed=LOOP_BATCH_SEED)
    flat = [doc_id for batch in batches for doc_id in batch]
    assert len(flat) == len(set(flat)) == 1200


def test_is_deterministic_for_a_seed() -> None:
    documents = make_documents()
    first = build_round_batches(documents, rounds=12, size=100, seed=LOOP_BATCH_SEED)
    second = build_round_batches(documents, rounds=12, size=100, seed=LOOP_BATCH_SEED)
    assert first == second


def test_input_order_does_not_change_the_result() -> None:
    documents = make_documents()
    shuffled = list(reversed(documents))
    assert build_round_batches(documents, rounds=12, size=100, seed=LOOP_BATCH_SEED) == (
        build_round_batches(shuffled, rounds=12, size=100, seed=LOOP_BATCH_SEED)
    )


def test_text_channel_is_balanced_across_batches() -> None:
    documents = make_documents()
    by_id = {str(doc["doc_id"]): doc for doc in documents}
    batches = build_round_batches(documents, rounds=12, size=100, seed=LOOP_BATCH_SEED)
    shares = [
        sum(1 for doc_id in batch if by_id[doc_id]["text_channel"] == "pdfText") / len(batch)
        for batch in batches
    ]
    assert max(shares) - min(shares) <= 0.10


def test_zero_reference_documents_are_spread_not_clumped() -> None:
    documents = make_documents()
    by_id = {str(doc["doc_id"]): doc for doc in documents}
    batches = build_round_batches(documents, rounds=12, size=100, seed=LOOP_BATCH_SEED)
    per_batch = [
        sum(1 for doc_id in batch if by_id[doc_id]["reference_count"] == 0) for batch in batches
    ]
    assert max(per_batch) <= 1


def test_too_few_documents_is_rejected() -> None:
    with pytest.raises(ContractError):
        build_round_batches(make_documents(50), rounds=12, size=100, seed=LOOP_BATCH_SEED)


def test_manifest_seals_itself_and_records_the_seed(tmp_path: Path) -> None:
    batches = build_round_batches(make_documents(), rounds=12, size=100, seed=LOOP_BATCH_SEED)
    result = write_batch_manifest(tmp_path, batches, seed=LOOP_BATCH_SEED)
    payload = json.loads((tmp_path / "round_batches_manifest.json").read_text(encoding="utf-8"))
    assert payload["seed"] == LOOP_BATCH_SEED
    assert payload["rounds"] == 12
    assert payload["size"] == 100
    assert [len(batch) for batch in payload["batches"]] == [100] * 12
    assert result["manifest_sha256"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loop_batches.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_quality_checker.loop_batches'`

- [ ] **Step 3: Implement the module**

Create `src/data_quality_checker/loop_batches.py`:

```python
"""Fixed, balanced round batches for the DQ-Loop.

The twelve batches are drawn once, before the loop runs, and never re-drawn.
Choosing each round's documents adaptively (active learning) would confound
"more cleaned data helps" with "smarter selection helps"; the experiment
claims the former, so selection must carry no signal.

`difficulty` is deliberately NOT a balance key: the pool holds two `Zor`
documents in total, which cannot be spread across twelve batches, and
pretending otherwise would put a meaningless column in the results.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .errors import ContractError
from .fingerprints import sha256_file

LOOP_BATCH_SEED = 20260902
BALANCE_FIELDS = ("length_quartile", "text_channel", "annotator", "reference_band")


def _reference_band(value: Any) -> str:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if count == 0:
        return "zero"
    if count <= 2:
        return "1-2"
    if count <= 5:
        return "3-5"
    if count <= 10:
        return "6-10"
    return "11+"


def _stratum(document: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(document.get("length_quartile", "unknown")),
        str(document.get("text_channel", "unknown")),
        str(document.get("annotator", "unknown")),
        _reference_band(document.get("reference_count")),
    )


def build_round_batches(
    documents: Sequence[dict[str, Any]], *, rounds: int, size: int, seed: int
) -> list[list[str]]:
    """Deal documents into `rounds` batches of exactly `size`, stratified.

    Documents are grouped by stratum, each group is shuffled deterministically,
    then the groups are dealt round-robin across batches. Dealing rather than
    slicing is what spreads a rare stratum -- a group of three zero-reference
    documents lands in three different batches instead of one.
    """
    if rounds <= 0 or size <= 0:
        raise ContractError("rounds and size must both be positive")
    needed = rounds * size
    if len(documents) < needed:
        raise ContractError(f"need at least {needed} documents, got {len(documents)}")

    groups: dict[tuple[str, ...], list[str]] = {}
    for document in documents:
        doc_id = document.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            raise ContractError("every document needs a non-empty string doc_id")
        groups.setdefault(_stratum(document), []).append(doc_id)

    rng = random.Random(seed)
    batches: list[list[str]] = [[] for _ in range(rounds)]
    cursor = 0
    # Sorting the strata makes the deal independent of input order; shuffling
    # within a stratum keeps it independent of the corpus's own ordering.
    for stratum in sorted(groups):
        members = sorted(groups[stratum])
        rng.shuffle(members)
        for doc_id in members:
            batches[cursor % rounds].append(doc_id)
            cursor += 1

    # The deal fills batches evenly but not to an exact `size`; trim the
    # overfull ones and top up the underfull ones from the surplus, keeping
    # every batch's stratum mix as dealt.
    surplus: list[str] = []
    for batch in batches:
        rng.shuffle(batch)
        if len(batch) > size:
            surplus.extend(batch[size:])
            del batch[size:]
    rng.shuffle(surplus)
    for batch in batches:
        while len(batch) < size:
            if not surplus:
                raise ContractError("ran out of documents while balancing batches")
            batch.append(surplus.pop())
    return [sorted(batch) for batch in batches]


def write_batch_manifest(
    output_dir: Path, batches: list[list[str]], *, seed: int
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "seed": seed,
        "rounds": len(batches),
        "size": len(batches[0]) if batches else 0,
        "balance_fields": list(BALANCE_FIELDS),
        "difficulty_balanced": False,
        "batches": batches,
    }
    path = output_dir / "round_batches_manifest.json"
    write_json_atomic(path, payload)
    return {"path": str(path), "manifest_sha256": sha256_file(path)}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_loop_batches.py -q`
Expected: PASS — 8 passed. If the balance or zero-reference assertions fail, the dealing order is wrong; fix the implementation, do not relax the assertion.

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/python -m ruff format src tests && .venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src 2>&1 | tail -3
git add src/data_quality_checker/loop_batches.py tests/test_loop_batches.py
git commit -m "feat(loop): deal fixed, balanced round batches for the DQ-Loop"
```

---

### Task 3: Round-aware training-set composition

**Files:**
- Create: `src/data_quality_checker/loop_training.py`
- Test: `tests/test_loop_training.py`

**Interfaces:**
- Consumes: `build_loop_split` output shape from Task 1, `write_jsonl_atomic`/`write_json_atomic` from `.atomic`, `sha256_file`, `ContractError`.
- Produces: `compose_round_training_ids(split, cleaned_rounds) -> dict[str, list[str]]`, `write_round_training_manifest(output_dir, round_index, composition, *, split_manifest_sha256) -> dict[str, Any]`.

`cleaned_rounds` is an ordered list of lists of `doc_id` strings — the batches already adjudicated, rounds `1..k`. Canonical ids are ints, external ids are strings; the composition normalises canonical ids to strings prefixed `canonical:` so the two pools can never collide.

- [ ] **Step 1: Write the failing test**

Create `tests/test_loop_training.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_quality_checker.errors import ContractError
from data_quality_checker.loop_training import (
    compose_round_training_ids,
    write_round_training_manifest,
)

SPLIT = {"train": [1, 2, 3], "valid": [10, 11], "test": [20, 21, 22]}


def test_round_zero_is_canonical_train_only() -> None:
    composition = compose_round_training_ids(SPLIT, [])
    assert composition["train"] == ["canonical:1", "canonical:2", "canonical:3"]
    assert composition["valid"] == ["canonical:10", "canonical:11"]
    assert composition["test"] == ["canonical:20", "canonical:21", "canonical:22"]


def test_each_round_appends_its_cleaned_batch() -> None:
    composition = compose_round_training_ids(SPLIT, [["dA", "dB"], ["dC"]])
    assert composition["train"] == [
        "canonical:1",
        "canonical:2",
        "canonical:3",
        "dA",
        "dB",
        "dC",
    ]


def test_validation_and_test_never_grow() -> None:
    base = compose_round_training_ids(SPLIT, [])
    grown = compose_round_training_ids(SPLIT, [["dA"], ["dB"], ["dC"]])
    assert grown["valid"] == base["valid"]
    assert grown["test"] == base["test"]


def test_a_cleaned_document_may_not_enter_twice() -> None:
    with pytest.raises(ContractError):
        compose_round_training_ids(SPLIT, [["dA"], ["dA"]])


def test_a_cleaned_document_may_not_collide_with_a_canonical_id() -> None:
    with pytest.raises(ContractError):
        compose_round_training_ids(SPLIT, [["canonical:1"]])


def test_manifest_records_the_round_and_the_split_seal(tmp_path: Path) -> None:
    composition = compose_round_training_ids(SPLIT, [["dA", "dB"]])
    result = write_round_training_manifest(
        tmp_path, 1, composition, split_manifest_sha256="abc123"
    )
    payload = json.loads((tmp_path / "round_001_training.json").read_text(encoding="utf-8"))
    assert payload["round"] == 1
    assert payload["split_manifest_sha256"] == "abc123"
    assert payload["counts"] == {"train": 5, "valid": 2, "test": 3}
    assert result["manifest_sha256"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loop_training.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_quality_checker.loop_training'`

- [ ] **Step 3: Implement the module**

Create `src/data_quality_checker/loop_training.py`:

```python
"""Compose round `k`'s training set: canonical train plus cleaned rounds 1..k.

Validation and test are returned unchanged on every round. They are frozen for
the life of the experiment: the loop re-runs checkpoint selection each round,
so validation can never be folded into training, and the test split may never
influence any decision at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .errors import ContractError
from .fingerprints import sha256_file

CANONICAL_PREFIX = "canonical:"


def _canonical_ids(doc_ids: Sequence[int]) -> list[str]:
    return [f"{CANONICAL_PREFIX}{doc_id}" for doc_id in doc_ids]


def compose_round_training_ids(
    split: dict[str, list[int]], cleaned_rounds: Sequence[Sequence[str]]
) -> dict[str, list[str]]:
    train = _canonical_ids(split["train"])
    seen = set(train)
    for round_index, batch in enumerate(cleaned_rounds, start=1):
        for doc_id in batch:
            if doc_id.startswith(CANONICAL_PREFIX):
                raise ContractError(
                    f"round {round_index} document {doc_id!r} uses the reserved canonical prefix"
                )
            if doc_id in seen:
                raise ContractError(
                    f"round {round_index} document {doc_id!r} is already in the training set"
                )
            seen.add(doc_id)
            train.append(doc_id)
    return {
        "train": train,
        "valid": _canonical_ids(split["valid"]),
        "test": _canonical_ids(split["test"]),
    }


def write_round_training_manifest(
    output_dir: Path,
    round_index: int,
    composition: dict[str, list[str]],
    *,
    split_manifest_sha256: str,
) -> dict[str, Any]:
    if round_index < 0:
        raise ContractError("round_index must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "round": round_index,
        "split_manifest_sha256": split_manifest_sha256,
        "counts": {name: len(ids) for name, ids in composition.items()},
        "ids": composition,
    }
    path = output_dir / f"round_{round_index:03d}_training.json"
    write_json_atomic(path, payload)
    return {"path": str(path), "manifest_sha256": sha256_file(path)}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_loop_training.py -q`
Expected: PASS — 6 passed.

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/python -m ruff format src tests && .venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src 2>&1 | tail -3
git add src/data_quality_checker/loop_training.py tests/test_loop_training.py
git commit -m "feat(loop): compose round training sets from canonical plus cleaned rounds"
```

---

### Task 4: Contract-driven checkpoint selection and the per-round record

**Files:**
- Modify: `src/data_quality_checker/g0.py` — `select_checkpoint`
- Create: `src/data_quality_checker/loop_selection.py`
- Test: `tests/test_g0.py` (extend), `tests/test_loop_selection.py` (create)

**Interfaces:**
- Consumes: `CheckpointCandidate` and `select_checkpoint` from `g0.py`, `GateBlocked` from `.errors`.
- Produces: `select_checkpoint(candidates, *, validation_documents=50, minimum_parse_count=49)`; `write_selection_record(output_dir, round_index, selected, candidates, *, validation_documents) -> dict[str, Any]`.

**Why `select_checkpoint` must change.** It currently hardcodes `candidate.coverage_count == 50`. With the loop's validation set also sized `50` this happens to pass, but the coincidence is invisible and brittle: a later change to the validation size would silently reject every checkpoint with a `GateBlocked` that names nothing about sizes. Make the gate explicit and defaulted, so existing callers are unaffected.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_g0.py`:

```python
def test_select_checkpoint_honours_a_custom_validation_size() -> None:
    from data_quality_checker.g0 import CheckpointCandidate, select_checkpoint

    candidate = CheckpointCandidate(
        update=100,
        coverage_count=150,
        parse_count=150,
        empty_output_count=0,
        runaway_output_count=0,
        core_f1=0.80,
        docwise_accuracy=0.40,
        recall=0.75,
        validation_loss=0.05,
    )
    assert (
        select_checkpoint(
            [candidate], validation_documents=150, minimum_parse_count=149
        ).update
        == 100
    )


def test_select_checkpoint_rejects_a_candidate_that_misses_the_declared_coverage() -> None:
    from data_quality_checker.errors import GateBlocked
    from data_quality_checker.g0 import CheckpointCandidate, select_checkpoint

    candidate = CheckpointCandidate(
        update=100,
        coverage_count=149,
        parse_count=149,
        empty_output_count=0,
        runaway_output_count=0,
        core_f1=0.80,
        docwise_accuracy=0.40,
        recall=0.75,
        validation_loss=0.05,
    )
    with pytest.raises(GateBlocked):
        select_checkpoint([candidate], validation_documents=150, minimum_parse_count=149)
```

Create `tests/test_loop_selection.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from data_quality_checker.g0 import CheckpointCandidate
from data_quality_checker.loop_selection import write_selection_record

FORBIDDEN = ("test_", "_test", "test_core_f1", "test_score")


def candidates() -> list[CheckpointCandidate]:
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
        for update, score in ((50, 0.70), (100, 0.78), (150, 0.74))
    ]


def test_record_names_the_winner_and_every_candidate(tmp_path: Path) -> None:
    rows = candidates()
    result = write_selection_record(tmp_path, 3, rows[1], rows, validation_documents=50)
    payload = json.loads((tmp_path / "round_003_checkpoint.json").read_text(encoding="utf-8"))
    assert payload["round"] == 3
    assert payload["selected"]["update"] == 100
    assert [row["update"] for row in payload["candidates"]] == [50, 100, 150]
    assert result["record_sha256"]


def test_record_states_the_selection_basis_is_validation(tmp_path: Path) -> None:
    rows = candidates()
    write_selection_record(tmp_path, 3, rows[1], rows, validation_documents=50)
    payload = json.loads((tmp_path / "round_003_checkpoint.json").read_text(encoding="utf-8"))
    assert payload["selection_basis"] == "validation"
    assert payload["validation_documents"] == 50
    assert payload["tie_break_order"] == ["core_f1", "docwise_accuracy", "recall", "-validation_loss"]


def test_record_never_mentions_a_test_metric(tmp_path: Path) -> None:
    rows = candidates()
    write_selection_record(tmp_path, 3, rows[1], rows, validation_documents=50)
    raw = (tmp_path / "round_003_checkpoint.json").read_text(encoding="utf-8").lower()
    for token in FORBIDDEN:
        assert token not in raw, f"selection record leaked a test-set reference: {token}"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_g0.py -k validation_size tests/test_loop_selection.py -q`
Expected: FAIL — `TypeError: select_checkpoint() got an unexpected keyword argument 'validation_documents'` and `ModuleNotFoundError: No module named 'data_quality_checker.loop_selection'`

- [ ] **Step 3: Make `select_checkpoint` contract-driven**

In `g0.py`, change the signature and the eligibility filter:

```python
def select_checkpoint(
    candidates: list[CheckpointCandidate],
    *,
    validation_documents: int = 50,
    minimum_parse_count: int = 49,
) -> CheckpointCandidate:
    eligible = [
        candidate
        for candidate in candidates
        if candidate.coverage_count == validation_documents
        and candidate.parse_count >= minimum_parse_count
        and candidate.empty_output_count == 0
        and candidate.runaway_output_count == 0
    ]
    if not eligible:
        raise GateBlocked(
            "no checkpoint passes coverage/parse/collapse eligibility gates "
            f"(need coverage=={validation_documents}, parse>={minimum_parse_count})"
        )
    return max(
        eligible,
        key=lambda item: (
            item.core_f1,
            item.docwise_accuracy,
            item.recall,
            -item.validation_loss,
        ),
    )
```

The defaults reproduce the previous behaviour exactly, so `g0_training.py:589` and `g0_finalize.py:51` need no change.

- [ ] **Step 4: Implement the selection record**

Create `src/data_quality_checker/loop_selection.py`:

```python
"""Per-round record of which checkpoint was selected and on what basis.

The record deliberately carries validation metrics only. The test split is
measured every round for the learning curve, but it may never influence a
decision -- checkpoint choice, learning rate, repeating a round, stopping the
loop. A selection record that mentioned a test metric would be evidence the
protocol had been broken, so `tests/test_loop_selection.py` asserts it cannot.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .errors import ContractError
from .fingerprints import sha256_file
from .g0 import CheckpointCandidate

TIE_BREAK_ORDER = ("core_f1", "docwise_accuracy", "recall", "-validation_loss")


def _row(candidate: CheckpointCandidate) -> dict[str, Any]:
    return {
        "update": candidate.update,
        "coverage_count": candidate.coverage_count,
        "parse_count": candidate.parse_count,
        "empty_output_count": candidate.empty_output_count,
        "runaway_output_count": candidate.runaway_output_count,
        "core_f1": candidate.core_f1,
        "docwise_accuracy": candidate.docwise_accuracy,
        "recall": candidate.recall,
        "validation_loss": candidate.validation_loss,
    }


def write_selection_record(
    output_dir: Path,
    round_index: int,
    selected: CheckpointCandidate,
    candidates: Sequence[CheckpointCandidate],
    *,
    validation_documents: int,
) -> dict[str, Any]:
    if round_index < 0:
        raise ContractError("round_index must be non-negative")
    if selected not in candidates:
        raise ContractError("the selected checkpoint must be one of the candidates")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "round": round_index,
        "selection_basis": "validation",
        "validation_documents": validation_documents,
        "tie_break_order": list(TIE_BREAK_ORDER),
        "selected": _row(selected),
        "candidates": [_row(candidate) for candidate in candidates],
    }
    path = output_dir / f"round_{round_index:03d}_checkpoint.json"
    write_json_atomic(path, payload)
    return {"path": str(path), "record_sha256": sha256_file(path)}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_g0.py tests/test_loop_selection.py -q`
Expected: PASS, including the pre-existing `test_g0.py` tests unchanged.

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS — 216 before this plan, plus 9 + 8 + 6 + 5 = **244**.

- [ ] **Step 7: Lint, type-check, commit**

```bash
.venv/bin/python -m ruff format src tests && .venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src 2>&1 | tail -3
git add src/data_quality_checker/g0.py src/data_quality_checker/loop_selection.py \
        tests/test_g0.py tests/test_loop_selection.py
git commit -m "feat(loop): make checkpoint selection contract-driven and record its basis"
```

---

## Done When

- `loop_split.py`, `loop_batches.py`, `loop_training.py`, `loop_selection.py` exist with the interfaces above.
- `.venv/bin/python -m pytest -q` passes at **244**.
- `ruff check` passes; `mypy src` still reports 37 errors in 9 files.
- `g0.build_split`, `write_training_data` and the reference-split gate are byte-identical to `main`.
- No new runtime dependency.

## Explicitly Out Of Scope

- **Any real training run.** Everything here is manifest and selection machinery, testable with fixtures and the existing deterministic fake trainer. Running `M₀ … M₁₂` is an operational step with its own GPU scheduling, crash-recovery and provenance requirements.
- Round orchestration — chaining rounds, invoking the judge, driving expert review. A later plan.
- Lifting the `only canonical-only G0 is supported in v1` restriction in `train_bootstrap` (`g0.py:433`). That belongs with the orchestration plan, which is what will actually call it.
- Repairing the 563 ungrounded human quotes. An open decision recorded in the design, not code work.
