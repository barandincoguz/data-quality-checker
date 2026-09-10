# `source_text` Off Every Model Surface — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `source_text` becomes a build-time locator and a database field only — never text a model is served, in the judge's payload or in a fine-tuning row.

**Architecture:** Three surfaces currently serve `source_text` to a model: the judge's `candidate_a`/`candidate_b` lists, dense-replay training rows (whose user turn is annotator spans joined by newlines), and reference-rescue rows (whose user turn is a bare annotator span). A fourth spends token budget on it — the native/windowed sizing decision counts a target the builder then strips. Each is replaced by real, contiguous document text; `source_text` stays only as the locator that finds that text. One invariant covers all four and is enforced by test: **every user turn a model is served is a verbatim substring of its own document, and every candidate a judge is served carries exactly the five decision fields.**

**Tech Stack:** Python 3.11, pytest 9, ruff, MLX / mlx-lm 0.31.2 (Qwen3.5-9B-MLX-4bit), SQLite, stdlib-only HTTP.

## Global Constraints

- Interpreter is `./.venv/bin/python` (shared lab venv via repo symlink). **No `pip install`, no venv creation.**
- Tests: `make test` (= `.venv/bin/python -m pytest -q`). Baseline before this plan: **637 collected, 636 pass, 1 skip.**
- Lint: `make lint` (= `ruff check src/ tests/`), line-length 100.
- `source_text` is never a comparison criterion, a router-bucket input, or model-served text. It stays in the database and the expert UI for the project owner's annotator-error analysis.
- The 150-document test split stays closed. Nothing in this plan reads it.
- Checkpoint selection is validation-only. `build_context_window_view` is called on `train.jsonl` alone (`mlx_compute.py:419-421`) — verified — so nothing here touches a measurement input.
- Every guard is verified by mutation, not by reading.
- Each accuracy number ships with its cost: latency p50/p95, tokens, memory.
- `32/A` is an additional article (never split); `32/a` is a sub-part (splittable). Case decides.
- Sealed and not to be regenerated: run `dqcheck_loop_fd1abc3af6b7`, checkpoint update **325** (core F1 **0.853933**, P 0.894118, R 0.817204), adapter sha `0089e99a…`, split 300/50/150 seed `20260902`.

## Measured Baseline (read from live artefacts, 2026-09-07)

| Fact | Value | Source |
| --- | --- | --- |
| Annotator reference rows in M01 carrying `source_text` | 494 / 494 (100 of 100 docs) | `dqcheck.sqlite3` |
| M000 prediction rows carrying `source_text` | 0 / 610 (99 of 99 docs) | `dqcheck.sqlite3` |
| Key sets on both sides | identical — leak is by **value**, not key presence | `dqcheck.sqlite3` |
| GREEN decisive verdicts choosing the annotator | 17 / 18 | `judge_results` |
| Overall pilot verdicts | 38 annotator / 13 model / 4 TIE / 2 NEITHER / 3 error | `judge_results` |
| Stale `valid` M01 verdicts that a rerun would skip | **57** | `judge_results` |
| Live judge prompt sha == invalid run's recorded sha | `aa887e10…` both | `judge_pilot_summary.json` |
| Pilot latency | 2050.6 s (~34 min) | `judge_pilot_summary.json` |
| Training view rows | 2022 | `train_context_1536_manifest.json` |
| Rows whose user turn is **not** a substring of its document | **324 / 2022** | measured on `train_context_1536.jsonl` |
| Assistant targets carrying `source_text` | 0 | measured — already clean |
| Native documents at 1536 tokens, with `source_text` in the sizing target | **28 / 300** | reproduces the sealed manifest exactly |
| Native documents at 1536 tokens, with it stripped first | **35 / 300** | measured with the pinned tokenizer |
| Token inflation from `source_text` in the sizing target | median 132, mean 158, max 580 | measured |
| Dense replay today | 333 rows, 2075 references, synthetic input | sealed manifest |
| Dense replay on real spanning passages | **513 rows, 2075 references** (median pack 4, median passage 1836 chars) | measured |
| References not locatable by their span | **0 / 2075** | measured |
| Reference rescue today | 2 rows | sealed manifest |

Already clean, verified, and **not** to be changed: `SYSTEM_PROMPT` (no `source_text`), the few-shot example bank (19 hits, but `g0.py:209-214` uses it only for a checksum and the exemplar id set — it never reaches a prompt), `predict_agent._clip_reference` (output-side cap only), the expert UI (`hitl.py:455`/`:816` build their own `candidate_a, candidate_b = human, model` and never call `blind_candidates`), and every identity/metric path (`core_identity`, `full_identity`, `_set_metric`, `attribute_errors` read no `source_text`).

## File Structure

| File | Responsibility after this plan |
| --- | --- |
| `src/data_quality_checker/constants.py` | Adds `REFERENCE_DECISION_FIELDS`; `REFERENCE_FIELDS` derives from it. |
| `src/data_quality_checker/judges.py` | Projects candidates onto the decision fields; names the whole instrument; refuses a stale verdict on resume. |
| `src/data_quality_checker/context_windows.py` | Sizes rows on what is served; rescue and replay rows carry real passages. |
| `src/data_quality_checker/g0.py` | `TRAINING_VIEW_POLICY` → v11, which re-keys the run id. |
| `tests/test_source_text_is_not_model_input.py` | **New.** The cross-cutting invariant for both surfaces. |
| `tests/test_judges.py` | Candidate-content assertion; instrument and stale-resume behaviour. |
| `tests/test_context_windows.py` | Real-passage rescue and replay; sizing on the served target. |
| `tests/test_g0.py` | Re-pins the run id to the v11 policy. |

---

### Task 1: The invariant, failing on today's code

This task writes only tests. They must fail against unmodified `src/`. That failure is the mutation proof for every task after it.

**Files:**
- Create: `tests/test_source_text_is_not_model_input.py`

**Interfaces:**
- Consumes: `data_quality_checker.judges.blind_candidates`, `data_quality_checker.context_windows.build_context_window_view`.
- Produces: nothing importable. Later tasks make these tests pass.

- [ ] **Step 1: Write the failing invariant tests**

Create `tests/test_source_text_is_not_model_input.py`:

```python
"""`source_text` is a locator and a database field, never model-served text.

The annotators clip their evidence from several places in a ruling, so
`source_text` is their prose about a document, not the document. It was
nonetheless reaching two models: the judge saw it on every annotator
candidate and none of M000's, which identified the human side outright, and
the trainee was served 324 of 2022 rows whose user turn was annotator spans
joined by newlines -- an input shape that never occurs at inference.

Both guards here are content assertions on purpose. `test_judges.py`'s
`test_blind_pilot_sends_only_text_and_candidates_and_persists_both_models`
already asserted the payload's *keys* and passed throughout the leak.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from data_quality_checker import context_windows
from data_quality_checker.context_windows import build_context_window_view
from data_quality_checker.judges import blind_candidates

ANNOTATOR_REFERENCE = {
    "kanun_no": "213",
    "kanun_ad": "Vergi Usul Kanunu",
    "madde": "mükerrer 20",
    "fikra": "",
    "bent": "",
    "source_text": "mükerrer 20 nci maddesi",
}
MODEL_REFERENCE = {**ANNOTATOR_REFERENCE, "source_text": ""}

# The five decision fields, written out rather than imported from
# `constants.REFERENCE_DECISION_FIELDS`. A guard that compared a payload
# against that tuple would still pass if the tuple itself ever regained
# `source_text` -- the exact class of defect this file exists to catch, and
# the reason the previous guard (which asserted the payload's keys but never
# a candidate's contents) let a broken judge pilot through. Writing them out
# also keeps this file collectable, so its behavioural failure is
# reproducible in the repo instead of only in a scratch copy.
DECISION_FIELDS = frozenset({"kanun_no", "kanun_ad", "madde", "fikra", "bent"})


def test_judge_candidates_carry_only_the_five_decision_fields() -> None:
    """Round 1 measured the tell: 494 annotator rows carried `source_text`
    and 610 model rows carried none, so in the GREEN bucket -- where the five
    decision fields are identical by construction -- the judge chose the
    annotator in 17 of 18 decisive verdicts."""
    mapping, candidate_a, candidate_b = blind_candidates(
        batch_id="M01",
        internal_doc_id="d000001",
        model="gemma-4-31b-it-optiq-4bit",
        human_references=[dict(ANNOTATOR_REFERENCE)],
        model_references=[dict(MODEL_REFERENCE)],
    )
    assert mapping in {"A=human,B=model", "A=model,B=human"}
    for candidate in (candidate_a, candidate_b):
        for reference in candidate:
            assert set(reference) == DECISION_FIELDS
    # The two sides are now indistinguishable, which is what blind means.
    assert candidate_a == candidate_b


def test_blinding_survives_a_field_added_to_the_reference_schema() -> None:
    """A positive projection, not a `source_text` blacklist: a field added
    to the schema later must not leak into a comparison by default."""
    _, candidate_a, candidate_b = blind_candidates(
        batch_id="M01",
        internal_doc_id="d000002",
        model="gemma-4-31b-it-optiq-4bit",
        human_references=[{**ANNOTATOR_REFERENCE, "annotator_note": "kırpıldı"}],
        model_references=[dict(MODEL_REFERENCE)],
    )
    assert candidate_a == candidate_b
    assert all("annotator_note" not in reference for reference in candidate_a)


class _OffsetTokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, list[tuple[int, int]]]:
        return {"offset_mapping": [match.span() for match in re.finditer(r"\S+", text)]}


class _Tokenizer:
    def __init__(self) -> None:
        self._tokenizer = _OffsetTokenizer()

    def apply_chat_template(self, messages, **_):
        tokens: list[int] = []
        for message in messages:
            tokens.extend(range(len(str(message["content"]).split())))
        return tokens


def _source_row(user: str, references: list[dict[str, str]]) -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": "extract"},
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": json.dumps(references, ensure_ascii=False, separators=(",", ":")),
            },
        ]
    }


def test_every_training_user_turn_is_real_document_text(tmp_path: Path, monkeypatch) -> None:
    """The trainee is served document prose or nothing.

    The document below is long enough to be windowed and carries two
    references far apart, which is exactly the shape that produced a
    dense-replay row: on round 0's 300 documents that shape accounted for
    324 of 2022 training rows whose user turn appeared nowhere in the ruling.
    """
    monkeypatch.setattr(context_windows, "_primary_window_geometry", lambda _: (6, 2))
    monkeypatch.setattr(context_windows, "FALLBACK_WINDOW_TOKENS", 3)
    monkeypatch.setattr(context_windows, "FALLBACK_OVERLAP_TOKENS", 1)
    body = "a0 a1 madde bir a4 a5 a6 a7 a8 a9 b0 b1 madde iki b4 b5 b6 b7"
    references = [
        {**ANNOTATOR_REFERENCE, "madde": "1", "source_text": "madde bir"},
        {**ANNOTATOR_REFERENCE, "madde": "2", "source_text": "madde iki"},
    ]
    source = tmp_path / "train.jsonl"
    source.write_text(
        json.dumps(_source_row(body, references), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    doc_ids = tmp_path / "train_doc_ids.json"
    doc_ids.write_text("[1]\n", encoding="utf-8")
    output = tmp_path / "view.jsonl"

    manifest = build_context_window_view(
        source_path=source,
        source_doc_ids_path=doc_ids,
        output_path=output,
        output_doc_ids_path=tmp_path / "view_doc_ids.json",
        manifest_path=tmp_path / "view_manifest.json",
        tokenizer=_Tokenizer(),
        model_fingerprint="a" * 64,
        max_sequence_length=12,
    )
    assert manifest["windowed_document_count"] == 1
    assert manifest["reference_coverage"] == "complete"

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows
    for row in rows:
        user_turn = row["messages"][1]["content"]
        assert user_turn in body, f"user turn is not document text: {user_turn!r}"
        for reference in json.loads(row["messages"][2]["content"]):
            assert "source_text" not in reference
```

- [ ] **Step 2: Run the tests and confirm each one fails for the right reason**

Run: `.venv/bin/python -m pytest tests/test_source_text_is_not_model_input.py -v`

Expected: **3 failed**, each a behavioural assertion, not an import or collection error. The file must stay collectable so the whole suite still runs (`--collect-only` must report 640, up from 637) and so this proof is reproducible in the repo rather than only in a scratch copy:

- `test_judge_candidates_carry_only_the_five_decision_fields` — `AssertionError: assert {'bent', 'fik...'source_text'} == frozenset({'b...no', 'madde'})`, the extra key being `'source_text'`
- `test_blinding_survives_a_field_added_to_the_reference_schema` — `assert candidate_a == candidate_b` fails; the annotator side still carries `'source_text': 'mükerrer 20 nci maddesi'` and `'annotator_note': 'kırpıldı'`
- `test_every_training_user_turn_is_real_document_text` — `AssertionError: user turn is not document text: 'madde bir\nmadde iki'`

If any of the three passes, stop: the guard does not detect the defect it was written for, which is the only thing this task exists to establish.

- [ ] **Step 3: Commit the failing guard**

```bash
git add tests/test_source_text_is_not_model_input.py
git commit -m "test: source_text must reach no model surface (failing)"
```

---

### Task 2: Blind the judge's candidates

**Files:**
- Modify: `src/data_quality_checker/constants.py:19-27`
- Modify: `src/data_quality_checker/judges.py:43-68` (prompt), `:218-231` (blinding), `:236-249` (fake provider)
- Test: `tests/test_source_text_is_not_model_input.py`, `tests/test_judges.py:145-169`

**Interfaces:**
- Produces: `constants.REFERENCE_DECISION_FIELDS: tuple[str, ...]` — the five fields a verdict may rest on, in `REFERENCE_FIELDS` order. `judges._decision_fields_only(references: list[dict[str, str]]) -> list[dict[str, str]]`. `blind_candidates` keeps its signature and returns projected candidates.

- [ ] **Step 1: Add the decision-field tuple**

In `src/data_quality_checker/constants.py`, replace the `REFERENCE_FIELDS` block:

```python
# The five fields a verdict, a router bucket, or a metric may rest on.
# `source_text` is deliberately absent: it is the annotators' prose about a
# document, clipped from several places in it, and it identified the human
# side of every blind comparison in round 1's judge pilot.
REFERENCE_DECISION_FIELDS = (
    "kanun_no",
    "kanun_ad",
    "madde",
    "fikra",
    "bent",
)
REFERENCE_FIELDS = (*REFERENCE_DECISION_FIELDS, "source_text")
```

- [ ] **Step 2: Project the candidates**

In `src/data_quality_checker/judges.py`, extend the constants import to include `REFERENCE_DECISION_FIELDS`, then insert above `blind_candidates` and rewrite it:

```python
def _decision_fields_only(references: list[dict[str, str]]) -> list[dict[str, str]]:
    """Project a candidate onto the five fields a verdict may rest on.

    A positive projection, not a `source_text` blacklist, so a field added to
    the reference schema later cannot leak into a comparison by default.

    Round 1 measured why this is not optional. Across all 60 pilot documents
    the annotator's 494 rows carried `source_text` and M000's 610 carried
    none, with identical key sets on both sides -- the field's value alone
    named the human. In the GREEN bucket, where the five decision fields are
    identical by construction and `source_text` is therefore the only
    difference, the judge chose the annotator in 17 of 18 decisive verdicts.
    An instruction cannot close that gap: JUDGE_PROMPT already told the judge
    a candidate's source_text is not evidence.
    """

    return [
        {field: reference.get(field, "") for field in REFERENCE_DECISION_FIELDS}
        for reference in references
    ]


def blind_candidates(
    *,
    batch_id: str,
    internal_doc_id: str,
    model: str,
    human_references: list[dict[str, str]],
    model_references: list[dict[str, str]],
) -> tuple[str, list[dict[str, str]], list[dict[str, str]]]:
    bit = int(_hash_order(batch_id, internal_doc_id, salt=f"blind:{model}"), 16) & 1
    human_blind = _decision_fields_only(human_references)
    model_blind = _decision_fields_only(model_references)
    if bit:
        return "A=model,B=human", model_blind, human_blind
    return "A=human,B=model", human_blind, model_blind
```

The A/B bit is untouched, so the rerun keeps the same 60 documents and the same side assignment as the invalid run. `source_text`'s presence is the only variable that moves.

- [ ] **Step 3: Drop the now-dead prompt sentence**

In `JUDGE_PROMPT` (`judges.py:43-68`), delete this sentence entirely:

```
    "A candidate's source_text is not "
    "evidence either: it is that candidate's claim about the document and may "
    "not appear there. "
```

so the passage reads `"...a span taken from this instruction text is rejected. Before keeping any span, find it in document_text and copy it from there; ..."`. It described a field the payload no longer has, and a live instruction about an absent field invites the judge to look for it. Removing it also moves `judge_prompt_fingerprint()` off `aa887e10…`, which Task 3 relies on.

- [ ] **Step 4: Stop the fake provider modelling the leak**

Replace `FakeJudgeProvider.judge` (`judges.py:236-249`):

```python
    def judge(self, *, model: str, payload: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        self.payloads.append(payload)
        references = payload["candidate_a"]
        # Evidence is quoted from the document, exactly as a real judge's
        # must be. This fixture used to build it from the candidates'
        # `source_text`, which both crashed once candidates were blinded
        # (`KeyError: 'source_text'`, uncaught by the pilot's per-document
        # clause) and kept 24 call sites testing the very shortcut the
        # blinded judge no longer has.
        document_text = str(payload.get("document_text", ""))
        lines = [line for line in document_text.splitlines() if line.strip()]
        return (
            {
                "verdict": "A",
                "candidate_errors": {"A": [], "B": ["not_selected"]},
                "final_references": references,
                "evidence": lines[:1],
                "reason_codes": ["fixture_prefers_a"],
            },
            {"latency_seconds": 0.0, "cost": 0.0, "provider": "fake"},
        )
```

- [ ] **Step 5: Give the existing pilot test a content assertion**

In `tests/test_judges.py`, inside `test_blind_pilot_sends_only_text_and_candidates_and_persists_both_models`, after the existing `assert "private-id" not in json.dumps(payload)` line, add:

```python
        # The keys were always right; the contents were not. Round 1 shipped
        # a judge pilot through this test with the annotator's source_text on
        # every candidate. The five fields are written out rather than read
        # from REFERENCE_DECISION_FIELDS, so this assertion still fails if
        # that tuple ever regains source_text.
        for candidate in (payload["candidate_a"], payload["candidate_b"]):
            for reference in candidate:
                assert set(reference) == {"kanun_no", "kanun_ad", "madde", "fikra", "bent"}
```

No new import is needed in that file: the expected fields are written out in the assertion on purpose, for the same reason as in `tests/test_source_text_is_not_model_input.py`.

- [ ] **Step 6: Run the judge tests**

Run: `.venv/bin/python -m pytest tests/test_source_text_is_not_model_input.py -k judge tests/test_judges.py tests/test_judge_provenance.py tests/test_judge_provider_registry.py tests/test_judge_metrics.py tests/test_judge_model.py -q`

Expected: PASS. The two judge invariant tests from Task 1 now pass; the training one still fails.

- [ ] **Step 7: Commit**

```bash
git add src/data_quality_checker/constants.py src/data_quality_checker/judges.py tests/test_judges.py
git commit -m "fix(judges): a candidate carries only the five decision fields"
```

---

### Task 3: Name the whole instrument, and refuse a stale verdict

Without this, HANDOFF step 2 is a no-op: 57 M01 verdicts sit at `status='valid'`, and `_run_pilot_impl` skips any valid result after a file-hash check — unlike `_run_locked_judge_coverage`, it does not even compare `judged_prediction`. The rerun would re-judge 3 documents and re-emit the same conclusions.

**Files:**
- Modify: `src/data_quality_checker/judges.py:72-86` (fingerprint), `:571-670` (pilot resume), `:704-720` (pilot payload), `:820-845` (production resume), `:884-896` (production payload), `:1055-1075` (summary)
- Test: `tests/test_judges.py`

**Interfaces:**
- Produces: `judges.CANDIDATE_PAYLOAD_CONTRACT: str` = `"decision_fields_only_v1"`. `judges.judge_instrument() -> dict[str, str]` returning `{"prompt_sha256": str, "candidate_payload_contract": str}`. Result payloads gain a `"judge_instrument"` key; the pilot summary gains `"judge_instrument"` and per-model `"stale_verdicts_rejudged"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_judges.py`:

```python
def test_judge_instrument_names_the_payload_shape_not_only_the_prompt() -> None:
    from data_quality_checker.judges import CANDIDATE_PAYLOAD_CONTRACT, judge_instrument

    instrument = judge_instrument()
    assert set(instrument) == {"prompt_sha256", "candidate_payload_contract"}
    assert instrument["candidate_payload_contract"] == CANDIDATE_PAYLOAD_CONTRACT
    # Round 1's invalid pilot recorded this sha. A run that changed only the
    # payload would otherwise have been indistinguishable from it.
    assert instrument["prompt_sha256"] != (
        "aa887e106fd908222c4afe368b5a598695ff31ca86fd35ccdbce0097760b3d74"
    )


def test_pilot_rejudges_a_verdict_formed_by_another_instrument(tmp_path) -> None:
    """The invalid round-1 pilot left 57 verdicts at status='valid'. A rerun
    that trusted them would have spent 35 minutes reproducing them."""
    import json as _json

    from data_quality_checker.judges import run_judge_pilot

    config = prepared_processed_fixture(tmp_path)
    first = run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        provider=FakeJudgeProvider(),
        judge_models=("qwen3.5:397b",),
    )
    assert first["models"]["qwen3.5:397b"]["valid"] == 1
    assert first["models"]["qwen3.5:397b"]["stale_verdicts_rejudged"] == 0

    with Store(config.database_path) as store:
        row = store.list_judge_results("batch")[0]
    path = Path(row["response_path"])
    stored = _json.loads(path.read_text(encoding="utf-8"))
    stored["judge_instrument"] = {
        "prompt_sha256": "0" * 64,
        "candidate_payload_contract": "spans_included_v0",
    }
    path.write_text(_json.dumps(stored, ensure_ascii=False), encoding="utf-8")
    with Store(config.database_path) as store:
        store.persist_judge_result(
            batch_id="batch",
            internal_doc_id=row["internal_doc_id"],
            model="qwen3.5:397b",
            blind_mapping=row["blind_mapping"],
            status="valid",
            verdict=row["verdict"],
            result=_json.loads(row["result_json"]),
            response_path=path,
            response_sha256=sha256_file(path),
            retry_count=0,
            error=None,
        )

    second = run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        provider=FakeJudgeProvider(),
        judge_models=("qwen3.5:397b",),
    )
    assert second["models"]["qwen3.5:397b"]["stale_verdicts_rejudged"] == 1
    assert second["models"]["qwen3.5:397b"]["valid"] == 1
    assert second["judge_instrument"] == judge_instrument()


def test_locked_coverage_refuses_a_verdict_from_another_instrument(tmp_path) -> None:
    """Production coverage is one instrument by definition, so a foreign
    verdict is an error there rather than something to redo silently."""
    from data_quality_checker.judges import _instrument_matches, judge_instrument

    assert _instrument_matches({"judge_instrument": judge_instrument()}) is True
    assert _instrument_matches({"judge_instrument": {"prompt_sha256": "0" * 64}}) is False
    assert _instrument_matches({}) is False
```

That last test covers the helper's contract, not the guard's wiring — and the two are not the same thing. Verified by mutation during implementation: with only that unit test present, deleting the whole `if not _instrument_matches(stored): raise IntegrityError(...)` block from `_run_locked_judge_coverage` left all 38 targeted tests passing. The production refusal therefore also needs an integration test, in `tests/test_judge_provenance.py` immediately after its sibling `test_a_verdict_from_another_generation_is_refused_on_resume`, which already establishes the pattern: build the fixture, force the document into the RED bucket so the production path targets it, run the pilot, finalize the review so `lock_judge`'s gate passes, lock the judge, confirm a re-run is clean while the payload still matches, then rewrite the stored payload's `judge_instrument` to a foreign value

```python
    payload["judge_instrument"] = {
        "prompt_sha256": "0" * 64,
        "candidate_payload_contract": "spans_included_v0",
    }
```

re-persist it with a recomputed sha, and assert the re-run raises `IntegrityError` with `"different judge instrument"` in the message. Do not hardcode the real prompt sha anywhere in that test: compare against a foreign value so the test survives the next prompt change. The guard-deletion mutation must make this test fail — if it does not, the test proves nothing.

Add `from data_quality_checker.fingerprints import sha256_file` and `from data_quality_checker.judges import judge_instrument` to the file's imports if absent.

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/bin/python -m pytest tests/test_judges.py -k "instrument or rejudge" -v`
Expected: FAIL — `ImportError: cannot import name 'judge_instrument'`.

- [ ] **Step 3: Add the instrument**

In `judges.py`, after `judge_prompt_fingerprint`, add:

```python
CANDIDATE_PAYLOAD_CONTRACT = "decision_fields_only_v1"


def judge_instrument() -> dict[str, str]:
    """Name the whole instrument a verdict came from, not half of it.

    `judge_prompt_fingerprint` covered the instructions. The other half is
    the shape of the payload those instructions describe, and it was recorded
    nowhere. Round 1's pilot ran with `source_text` on every annotator
    candidate and recorded prompt sha `aa887e10…` -- the same sha a correctly
    blinded run with an unchanged prompt would have recorded, so the invalid
    verdicts and their replacements would have been indistinguishable in the
    record, while HANDOFF's rule that rounds carrying different fingerprints
    must not be compared quietly relied on that sha.
    """

    return {
        "prompt_sha256": judge_prompt_fingerprint(),
        "candidate_payload_contract": CANDIDATE_PAYLOAD_CONTRACT,
    }


def _instrument_matches(stored: dict[str, Any]) -> bool:
    """True when a stored verdict was produced by this exact instrument."""

    return stored.get("judge_instrument") == judge_instrument()
```

- [ ] **Step 4: Record it and act on it in the pilot**

In `_run_pilot_impl`, after `providers_by_model: dict[str, JudgeProvider] = {}` add:

```python
        instrument = judge_instrument()
        stale_by_model: dict[str, int] = {model: 0 for model in models}
```

Replace the resume block (the `if existing is not None and existing["status"] == "valid":` body) with:

```python
                existing = store.get_judge_result(batch_id, internal_doc_id, model)
                if existing is not None and existing["status"] == "valid":
                    response_path = Path(existing["response_path"])
                    if (
                        not response_path.is_file()
                        or sha256_file(response_path) != existing["response_sha256"]
                    ):
                        raise IntegrityError(
                            f"judge result file is missing or corrupt: {response_path}"
                        )
                    stored = json.loads(response_path.read_text(encoding="utf-8"))
                    # A pilot is the run you redo when the instrument moves,
                    # so a verdict from another instrument is re-judged rather
                    # than refused -- but it is counted, because the rerun that
                    # silently reused 57 valid verdicts is exactly how round 1
                    # spent a pilot reproducing its own invalid conclusions.
                    if _instrument_matches(stored) and stored.get(
                        "judged_prediction"
                    ) == judged_prediction_provenance(prediction):
                        counts_by_model[model]["valid"] += 1
                        continue
                    stale_by_model[model] += 1
```

In the pilot's `result_payload`, add `"judge_instrument": instrument,` directly after `"blind_mapping": mapping,`, and bump that payload's `"schema_version": 2` to `"schema_version": 3`. The version exists to name the payload's shape, and the resume logic now depends on the new field: a payload written without it names no instrument and is therefore re-judged rather than trusted, which is what makes M01's 57 stale verdicts legible in the record as "v2, no instrument". No test pins this value (checked).

In the pilot `summary`, add `"judge_instrument": instrument,` beside `"judge_prompt_sha256"`, and inside the per-model dict add `"stale_verdicts_rejudged": stale_by_model[model],`.

- [ ] **Step 5: Record it and refuse it in production coverage**

In `_run_locked_judge_coverage`, inside the existing resume block, immediately after the `stored = json.loads(...)` line and before the `judged_prediction` check, add:

```python
                if not _instrument_matches(stored):
                    raise IntegrityError(
                        f"existing judge verdict for {internal_doc_id} was formed by a "
                        f"different judge instrument than the locked one; re-judge or clear it"
                    )
```

In that function's `result_payload`, add `"judge_instrument": judge_instrument(),` after `"blind_mapping": mapping,`, and bump that payload's `"schema_version": 2` to `"schema_version": 3`, for the same reason as the pilot's.

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_judges.py tests/test_judge_provenance.py tests/test_loop_bindings.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/data_quality_checker/judges.py tests/test_judges.py
git commit -m "fix(judges): a verdict records the instrument that formed it"
```

---

### Task 4: Size a training row on what the model is served

**Files:**
- Modify: `src/data_quality_checker/context_windows.py:344-383`
- Test: `tests/test_context_windows.py`

**Interfaces:**
- Consumes: `_without_source_text` (unchanged). Produces: no new symbols; `native_document_count` rises from 28 to 35 on the sealed 300-document split.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_context_windows.py`:

```python
def test_native_decision_ignores_the_source_text_it_is_about_to_strip(
    tmp_path: Path,
) -> None:
    """A document is windowed only if what the model is served is too long.

    The sizing count used to include `source_text`, a field the very next
    statement strips. On the sealed 300-document split that spent a median
    132 tokens per document on text the model never sees and fragmented 7
    documents that fit whole.
    """
    reference = {
        "kanun_no": "213",
        "kanun_ad": "Vergi Usul Kanunu",
        "madde": "5",
        "fikra": "",
        "bent": "",
        "source_text": "w0 w1 w2 w3 w4 w5 w6 w7",
    }
    row = _row("w0 w1 w2 w3 w4 w5 w6 w7", [reference])
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    doc_ids = tmp_path / "train_doc_ids.json"
    doc_ids.write_text("[1]\n", encoding="utf-8")

    manifest = build_context_window_view(
        source_path=source,
        source_doc_ids_path=doc_ids,
        output_path=tmp_path / "view.jsonl",
        output_doc_ids_path=tmp_path / "view_doc_ids.json",
        manifest_path=tmp_path / "view_manifest.json",
        tokenizer=_Tokenizer(),
        # 15 is chosen, not arbitrary: this row renders to 19 tokens with
        # `source_text` in the assistant target and 12 without it. Any limit
        # from 12 to 18 therefore separates the two; at 19 or above the
        # document fits either way and the test passes without measuring
        # anything. 22 was the original value and was exactly that mistake.
        max_sequence_length=15,
        model_fingerprint="a" * 64,
    )
    assert manifest["native_document_count"] == 1
    assert manifest["windowed_document_count"] == 0
```

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/bin/python -m pytest tests/test_context_windows.py::test_native_decision_ignores_the_source_text_it_is_about_to_strip -v`
Expected: FAIL — `assert 0 == 1`, because the `source_text` words push the row past 22 tokens and it is windowed.

- [ ] **Step 3: Size on the served target**

In `build_context_window_view`, replace the block from `references = _compact_references(...)` through the end of the native branch's `output_rows.append(native_row)` with:

```python
        references = _compact_references(messages[2]["content"])
        original_reference_count += len(references)
        # Size the row on exactly what the model is served. `source_text` is
        # stripped from every assistant target below, so counting it here
        # spent budget on a field the model never sees -- a median 132 tokens
        # per document on the sealed 300-document split, which pushed 7 of
        # them out of a whole-document row and into windows.
        served_messages = [
            messages[0],
            messages[1],
            {
                "role": "assistant",
                "content": json.dumps(
                    [_without_source_text(reference) for reference in references],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
        full_tokens = _rendered_token_count(tokenizer, served_messages)
        if full_tokens <= max_sequence_length:
            native_documents += 1
            output_rows.append({"messages": served_messages})
```

The remainder of the native branch (`output_doc_ids.append`, counters, `chunks.append`, `continue`) is unchanged.

- [ ] **Step 4: Run the context-window tests**

Run: `.venv/bin/python -m pytest tests/test_context_windows.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data_quality_checker/context_windows.py tests/test_context_windows.py
git commit -m "fix(training-view): size a row on the target the model is served"
```

---

### Task 5: A rescued reference is taught on its real window

**Files:**
- Modify: `src/data_quality_checker/context_windows.py` — add two helpers above `_window_row`; rewrite the rescue loop at `:482-528`
- Test: `tests/test_context_windows.py`

**Interfaces:**
- Produces: `_token_span_for_chars(offsets: list[tuple[int, int]], char_start: int, char_end: int) -> tuple[int, int]` — smallest token interval fully covering a character range. `_expanded_interval(start: int, end: int, total: int, minimum_tokens: int) -> tuple[int, int]` — grows an interval to a minimum width, still containing the original and inside `[0, total)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_context_windows.py`:

```python
def test_reference_rescue_serves_a_real_window_not_the_annotator_span(
    tmp_path: Path, monkeypatch
) -> None:
    """A reference no window makes visible is still taught -- from the
    ruling. The rescue row used to serve the annotator's bare span as the
    user turn, which is prose about the document, not the document."""
    monkeypatch.setattr(context_windows, "_primary_window_geometry", lambda _: (5, 1))
    monkeypatch.setattr(context_windows, "FALLBACK_WINDOW_TOKENS", 5)
    monkeypatch.setattr(context_windows, "FALLBACK_OVERLAP_TOKENS", 1)
    body = "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9"
    # Two things must be true at once or this test means nothing.
    #
    # Rescue must fire: no ordinary window may contain the span as a
    # substring, which a primary geometry of (5, 1) achieves for "w3 w4 w5".
    #
    # And the rescue window must be strictly WIDER than the span. With a
    # fallback as narrow as the span, the old bare-span row and a real
    # window are byte identical, and only the strategy-id string would
    # differ -- a label, not a behaviour. A fallback of 5 against the
    # 3-token span yields "w2 w3 w4 w5 w6".
    #
    # The budget is the whole three-message chat, not the user turn: this
    # reference row costs a fixed 4 tokens (system "extract" = 1, stripped
    # assistant target = 3, because "Vergi Usul Kanunu" holds two spaces the
    # toy tokenizer splits on). So a 5-token window needs 9, and a smaller
    # limit makes the builder raise "reference rescue tokens=N exceed
    # max_sequence_length" before any assertion runs.
    reference = {
        "kanun_no": "213",
        "kanun_ad": "Vergi Usul Kanunu",
        "madde": "5",
        "fikra": "",
        "bent": "",
        "source_text": "w3 w4 w5",
    }
    source = tmp_path / "train.jsonl"
    source.write_text(
        json.dumps(_row(body, [reference]), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    doc_ids = tmp_path / "train_doc_ids.json"
    doc_ids.write_text("[1]\n", encoding="utf-8")
    output = tmp_path / "view.jsonl"

    manifest = build_context_window_view(
        source_path=source,
        source_doc_ids_path=doc_ids,
        output_path=output,
        output_doc_ids_path=tmp_path / "view_doc_ids.json",
        manifest_path=tmp_path / "view_manifest.json",
        tokenizer=_Tokenizer(),
        model_fingerprint="a" * 64,
        max_sequence_length=9,
    )
    assert manifest["reference_rescue_count"] >= 1
    assert manifest["request"]["reference_rescue"] == "real_window_v2"
    # The count alone doesn't prove rescue covered this document -- confirm a
    # chunk actually recorded that mode.
    rescue_chunks = [chunk for chunk in manifest["chunks"] if chunk["mode"] == "reference_rescue"]
    assert rescue_chunks

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    rescue_user = rows[rescue_chunks[0]["output_row_index"]]["messages"][1]["content"]
    # These three together are what the strategy id cannot fake: the served
    # turn is real document text, it contains the span, and it is not merely
    # the span. That last one is the whole point -- source_text located the
    # passage and did not become it.
    assert rescue_user in body
    assert "w3 w4 w5" in rescue_user
    assert rescue_user != "w3 w4 w5"
    # Every user turn served for this document is real document text.
    assert all(row["messages"][1]["content"] in body for row in rows)
    # And the reference is genuinely taught, not merely counted.
    assert any(json.loads(row["messages"][2]["content"]) for row in rows)
```

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/bin/python -m pytest tests/test_context_windows.py::test_reference_rescue_serves_a_real_window_not_the_annotator_span -v`
Expected: FAIL — `assert manifest["request"]["reference_rescue"] == "real_window_v2"` gets `"source_text_v1"`.

- [ ] **Step 3: Add the two interval helpers**

Insert into `context_windows.py` above `_window_row`:

```python
def _token_span_for_chars(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int, int]:
    """Smallest token interval fully covering a character range."""

    start = next(
        (index for index, (_, end) in enumerate(offsets) if end > char_start),
        max(0, len(offsets) - 1),
    )
    end = next(
        (
            index + 1
            for index in range(len(offsets) - 1, -1, -1)
            if offsets[index][0] < char_end
        ),
        start + 1,
    )
    return start, max(end, start + 1)


def _expanded_interval(
    start: int, end: int, total: int, minimum_tokens: int
) -> tuple[int, int]:
    """Grow a token interval to a minimum width, inside the body, still
    containing the interval it was given."""

    width = min(total, max(minimum_tokens, end - start))
    pad = width - (end - start)
    begin = max(0, min(start - pad // 2, total - width))
    return begin, begin + width
```

- [ ] **Step 4: Serve a window instead of the span**

In `build_context_window_view`, replace the body of the `for reference_index in missing_references:` loop with:

```python
        for reference_index in missing_references:
            if reference_index in visible_union:
                continue
            reference = references[reference_index]
            source_text = reference.get("source_text")
            if not isinstance(source_text, str) or not source_text:
                raise IntegrityError(
                    f"reference rescue has no source_text at source row {row_index}"
                )
            char_start = body.find(source_text)
            if char_start < 0:
                raise IntegrityError(f"reference rescue span is absent at source row {row_index}")
            # The span locates the passage; the passage is what is served.
            # A rescue row used to put the annotator's bare span in the user
            # turn -- text that never occurs at inference, from a field the
            # system prompt does not name.
            span_start, span_end = _token_span_for_chars(
                offsets, char_start, char_start + len(source_text)
            )
            interval = _expanded_interval(
                span_start, span_end, len(offsets), FALLBACK_WINDOW_TOKENS
            )
            rescue_row, rescue_metadata, rescue_visible = _window_row(
                tokenizer=tokenizer,
                system_message=messages[0],
                body=body,
                offsets=offsets,
                interval=interval,
                references=references,
            )
            if rescue_metadata["tokens"] > max_sequence_length:
                raise ContractError(
                    f"reference rescue tokens={rescue_metadata['tokens']} exceed "
                    f"max_sequence_length={max_sequence_length} at row {row_index}"
                )
            if reference_index not in rescue_visible:
                raise IntegrityError(
                    f"reference rescue window does not contain reference "
                    f"{reference_index} at source row {row_index}"
                )
            output_rows.append(rescue_row)
            output_doc_ids.append(doc_id)
            chunk_reference_count += rescue_metadata["reference_count"]
            reference_rescue_count += 1
            maximum_tokens = max(maximum_tokens, rescue_metadata["tokens"])
            chunks.append(
                {
                    "output_row_index": len(output_rows) - 1,
                    "source_row_index": row_index,
                    "doc_id": doc_id,
                    "mode": "reference_rescue",
                    **rescue_metadata,
                }
            )
            visible_union.update(rescue_visible)
```

A rescue window now labels every reference it actually contains, not just the one that triggered it, so a window carrying two rescued references is no longer a false negative for the second. The `if reference_index in visible_union: continue` guard is what makes that safe — an earlier rescue may already have covered this reference.

- [ ] **Step 5: Update the strategy id**

In the `request` dict, change `"reference_rescue": "source_text_v1",` to `"reference_rescue": "real_window_v2",`.

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_context_windows.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/data_quality_checker/context_windows.py tests/test_context_windows.py
git commit -m "fix(training-view): a rescued reference is taught on its real window"
```

---

### Task 6: Dense replay on real passages

**Files:**
- Modify: `src/data_quality_checker/context_windows.py:162-231` (`_dense_replay_rows`), the `dense_replay` strategy id in `request`
- Test: `tests/test_source_text_is_not_model_input.py` (Task 1's remaining failure), `tests/test_context_windows.py`

**Interfaces:**
- `_dense_replay_rows` keeps its signature — `(*, tokenizer, system_message, body, references, max_sequence_length) -> list[tuple[dict[str, Any], int, int]]` of `(row, tokens, reference_count)`. Its user turn is now `body[min_start:max_end]` for the packed group.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_context_windows.py`:

```python
def test_dense_replay_serves_the_real_passage_spanning_its_pack(
    tmp_path: Path, monkeypatch
) -> None:
    """Replay teaches many references at once, from continuous ruling prose.

    The pack used to be served as the annotator's spans joined by newlines:
    324 of the sealed view's 2022 rows were input the model will never meet.
    Rebuilt on real spanning passages the same 2075 reference instances
    survive across 513 rows.
    """
    monkeypatch.setattr(context_windows, "_primary_window_geometry", lambda _: (4, 2))
    monkeypatch.setattr(context_windows, "FALLBACK_WINDOW_TOKENS", 3)
    monkeypatch.setattr(context_windows, "FALLBACK_OVERLAP_TOKENS", 1)
    body = "a0 madde bir a3 a4 a5 madde iki a8 a9"
    references = [
        {
            "kanun_no": "213",
            "kanun_ad": "Vergi Usul Kanunu",
            "madde": "1",
            "fikra": "",
            "bent": "",
            "source_text": "madde bir",
        },
        {
            "kanun_no": "213",
            "kanun_ad": "Vergi Usul Kanunu",
            "madde": "2",
            "fikra": "",
            "bent": "",
            "source_text": "madde iki",
        },
    ]
    source = tmp_path / "train.jsonl"
    source.write_text(
        json.dumps(_row(body, references), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    doc_ids = tmp_path / "train_doc_ids.json"
    doc_ids.write_text("[1]\n", encoding="utf-8")
    output = tmp_path / "view.jsonl"

    manifest = build_context_window_view(
        source_path=source,
        source_doc_ids_path=doc_ids,
        output_path=output,
        output_doc_ids_path=tmp_path / "view_doc_ids.json",
        manifest_path=tmp_path / "view_manifest.json",
        tokenizer=_Tokenizer(),
        model_fingerprint="a" * 64,
        max_sequence_length=14,
    )
    assert manifest["dense_replay_row_count"] >= 1
    assert manifest["dense_replay_reference_count"] == 2
    assert manifest["request"]["dense_replay"] == "real_passage_span_pack_max10_full_v6"

    replay = [
        json.loads(line)
        for line, mode in zip(
            output.read_text(encoding="utf-8").splitlines(),
            [chunk["mode"] for chunk in manifest["chunks"]],
            strict=True,
        )
        if mode == "reference_dense_replay"
    ]
    assert replay
    for row in replay:
        assert row["messages"][1]["content"] in body
```

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/bin/python -m pytest tests/test_context_windows.py::test_dense_replay_serves_the_real_passage_spanning_its_pack -v`
Expected: FAIL — the strategy id is `windowed_documents_source_text_pack_max10_full_v5` and a replay user turn is `"madde bir\nmadde iki"`, absent from `body`.

- [ ] **Step 3: Rebuild `_dense_replay_rows`**

Replace the whole function:

```python
def _dense_replay_rows(
    *,
    tokenizer: Any,
    system_message: dict[str, str],
    body: str,
    references: list[dict[str, Any]],
    max_sequence_length: int,
) -> list[tuple[dict[str, Any], int, int]]:
    """Replay a document's references over the real passages that carry them.

    The pack used to be served as the annotator's `source_text` spans joined
    by newlines. That input never occurs at inference -- the model is given
    continuous ruling prose, not a list of clipped phrases -- and the
    annotators clip from several places in a document, so the joined blob was
    synthetic text carrying a many-reference answer. On the sealed
    300-document view it was 324 of 2022 training rows.

    `source_text` now only locates each reference; the row serves the
    shortest real passage that spans its pack. Measured on the same 300
    documents: all 2075 reference instances survive, across 513 rows with a
    median pack of 4 and a median passage of 1836 characters.
    """

    located: list[tuple[int, int, dict[str, Any]]] = []
    for reference_index, reference in enumerate(references):
        source_text = reference.get("source_text")
        if not isinstance(source_text, str) or not source_text.strip():
            raise IntegrityError(f"dense replay reference {reference_index} has no source_text")
        source_text = source_text.strip()
        char_start = body.find(source_text)
        if char_start < 0:
            raise IntegrityError(
                f"dense replay reference {reference_index} is absent from source text"
            )
        located.append((char_start, char_start + len(source_text), reference))
    located.sort(key=lambda item: (item[0], item[1]))

    def render(
        pack: list[tuple[int, int, dict[str, Any]]],
    ) -> tuple[dict[str, Any], int]:
        passage = body[min(start for start, _, _ in pack) : max(end for _, end, _ in pack)]
        messages = [
            system_message,
            {"role": "user", "content": passage},
            {
                "role": "assistant",
                "content": json.dumps(
                    [_without_source_text(reference) for _, _, reference in pack],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
        return {"messages": messages}, _rendered_token_count(tokenizer, messages)

    packed: list[tuple[dict[str, Any], int, int]] = []
    index = 0
    while index < len(located):
        pack = [located[index]]
        row, tokens = render(pack)
        if tokens > max_sequence_length:
            raise ContractError(
                f"dense replay row tokens={tokens} exceed "
                f"max_sequence_length={max_sequence_length}"
            )
        cursor = index + 1
        while cursor < len(located) and len(pack) < DENSE_REPLAY_MAX_REFERENCES:
            candidate_pack = [*pack, located[cursor]]
            candidate_row, candidate_tokens = render(candidate_pack)
            if candidate_tokens > max_sequence_length:
                break
            pack, row, tokens = candidate_pack, candidate_row, candidate_tokens
            cursor += 1
        packed.append((row, tokens, len(pack)))
        index = cursor
    return packed
```

The pack shrinks until its spanning passage fits, so a group whose references are far apart simply becomes several rows — which is why the row count rises from 333 to 513 while the reference count is unchanged at 2075.

- [ ] **Step 4: Update the strategy id**

In `request`, change `"dense_replay": "windowed_documents_source_text_pack_max10_full_v5",` to `"dense_replay": "real_passage_span_pack_max10_full_v6",`.

- [ ] **Step 5: Run the full context-window and invariant suites**

Run: `.venv/bin/python -m pytest tests/test_context_windows.py tests/test_source_text_is_not_model_input.py -q`
Expected: PASS — including Task 1's `test_every_training_user_turn_is_real_document_text`, which has been failing since Task 1.

- [ ] **Step 6: Commit**

```bash
git add src/data_quality_checker/context_windows.py tests/test_context_windows.py
git commit -m "fix(training-view): dense replay teaches on real document passages"
```

---

### Task 7: Re-key the training-view policy

The recipe changed, so the run id must change — that is what keeps the sealed `dqcheck_loop_fd1abc3af6b7` artefacts from being overwritten and what stops a recipe change from being invisible. `TRAINING_VIEW_POLICY` is inside `train_bootstrap`'s fingerprint payload (`g0.py:466`) and the round's (`loop_train.py:222`).

**Files:**
- Modify: `src/data_quality_checker/g0.py:34-36`
- Test: `tests/test_g0.py:112-118`

**Interfaces:**
- Produces: `g0.TRAINING_VIEW_POLICY` = `"adaptive-context-fallback128-positive-only-dense-max10-tokenmean-looprepair-realtext-v11"`.

- [ ] **Step 1: Bump the policy**

```python
TRAINING_VIEW_POLICY = (
    "adaptive-context-fallback128-positive-only-dense-max10-tokenmean-looprepair-realtext-v11"
)
```

- [ ] **Step 2: Run the pinned-run-id test and read the new id**

Run: `.venv/bin/python -m pytest tests/test_g0.py::test_bootstrap_run_id_is_unchanged_for_development -v`
Expected: FAIL with `assert 'dqcheck_g0_qwen3_5_9b_<new12hex>' == 'dqcheck_g0_qwen3_5_9b_458edd8b15f9'`. Copy the left-hand value.

- [ ] **Step 3: Re-pin the lock with the new id and say why**

In `tests/test_g0.py`, update the assertion to the value from Step 2 and extend the comment:

```python
def test_bootstrap_run_id_is_unchanged_for_development(tmp_path: Path) -> None:
    # Regression lock: pins the development run id to today's SYSTEM_PROMPT
    # and TRAINING_VIEW_POLICY. source_text left the model's field list and
    # few-shot examples, and then left the trainee's input entirely -- dense
    # replay and reference rescue now serve real document passages. Both are
    # deliberate changes to the sealed fingerprint, which is exactly why a
    # new run id is right: the sealed loop run dqcheck_loop_fd1abc3af6b7 and
    # its update-325 checkpoint keep their own directory untouched.
    result = train_bootstrap(config=isolated_config(tmp_path), generation="G0")
    assert result["run_id"] == "dqcheck_g0_qwen3_5_9b_<new12hex>"
```

- [ ] **Step 4: Run the g0 and loop-train suites**

Run: `.venv/bin/python -m pytest tests/test_g0.py tests/test_g0_training.py tests/test_loop_train.py tests/test_loop_training.py tests/test_q36_p1.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data_quality_checker/g0.py tests/test_g0.py
git commit -m "chore(training-view): re-key the view policy for real-text inputs"
```

---

### Task 8: End-to-end verification on real data

No code changes. This is the gate.

**Files:**
- Create: `artifacts/data_quality_checker/evidence/2026-09-07_source_text_off_model_surfaces/verification.json`

- [ ] **Step 1: Whole suite and lint**

Run: `make test && make lint`
Expected: 637 collected; **636 passed, 1 skipped**; ruff clean. A different pass count means a test was silently dropped — stop and find it.

- [ ] **Step 2: Rebuild the training view on the sealed 300-document split and diff the manifests**

Run:

```bash
.venv/bin/python - <<'PY'
import json, os
os.environ.setdefault("HF_HOME", "/opt/llm-lab/hf-cache")
from pathlib import Path
# The pipeline's own loader, not AutoTokenizer. `build_context_window_view`
# needs `tokenizer._tokenizer` to be callable for offset mapping; a bare
# AutoTokenizer hands back a raw tokenizers.Tokenizer and the build dies
# with "'tokenizers.Tokenizer' object is not callable".
from mlx_lm.utils import load_tokenizer
from data_quality_checker.context_windows import build_context_window_view

SNAP = ("/opt/llm-lab/hf-cache/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/"
        "snapshots/938d8919941c6e7efd3c7150eff7fe9d12afa631")
D = Path("data/sensitive/data_quality_checker/loop/dqcheck_loop_fd1abc3af6b7/data")
OUT = Path("/private/tmp/claude-504/-Users-student2-data-quality-checker/"
           "42c05135-5b37-4bc9-960b-9e56d7d6b0de/scratchpad/view_v11")
OUT.mkdir(parents=True, exist_ok=True)
new = build_context_window_view(
    source_path=D / "train.jsonl",
    source_doc_ids_path=D / "train_doc_ids.json",
    output_path=OUT / "train_context_1536.jsonl",
    output_doc_ids_path=OUT / "train_context_1536_doc_ids.json",
    manifest_path=OUT / "train_context_1536_manifest.json",
    tokenizer=load_tokenizer(SNAP, tokenizer_config_extra={"trust_remote_code": True}),
    model_fingerprint="877ac3b405738c16501b2638574676ce7a41f6bb67cdbd75a9bffacc1d32ec56",
    max_sequence_length=1536,
)
old = json.loads((D / "train_context_1536_manifest.json").read_text(encoding="utf-8"))
keys = ["output_row_count", "native_document_count", "windowed_document_count",
        "original_reference_count", "chunk_reference_count", "reference_rescue_count",
        "dense_replay_row_count", "dense_replay_reference_count",
        "training_reference_count", "maximum_tokens"]
for key in keys:
    print(f"{key:34s} {old.get(key)!s:>12s} -> {new.get(key)!s}")

body = {}
for row, doc in zip(
    (json.loads(line) for line in (D / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()),
    json.loads((D / "train_doc_ids.json").read_text(encoding="utf-8")), strict=True,
):
    body[doc] = row["messages"][1]["content"].rstrip()
rows = [json.loads(line) for line in (OUT / "train_context_1536.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
ids = json.loads((OUT / "train_context_1536_doc_ids.json").read_text(encoding="utf-8"))
synthetic = sum(1 for row, doc in zip(rows, ids, strict=True) if row["messages"][1]["content"] not in body[doc])
leaked = sum(1 for row in rows for ref in json.loads(row["messages"][2]["content"]) if "source_text" in ref)

# Real reference coverage, computed. The manifest's own
# `reference_coverage: "complete"` is an unconditional literal
# (context_windows.py:591) -- a label on an invariant enforced by an
# IntegrityError elsewhere, so asserting it proves nothing. This recomputes
# it: every reference of every source document must appear, by its five
# decision fields, in at least one output row's assistant target.
FIELDS = ("kanun_no", "kanun_ad", "madde", "fikra", "bent")
def ident(reference):
    return tuple(str(reference.get(field, "")) for field in FIELDS)
want, got = {}, {}
for row, doc in zip(
    (json.loads(line) for line in (D / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()),
    json.loads((D / "train_doc_ids.json").read_text(encoding="utf-8")), strict=True,
):
    want.setdefault(doc, set()).update(ident(r) for r in json.loads(row["messages"][2]["content"]))
for row, doc in zip(rows, ids, strict=True):
    got.setdefault(doc, set()).update(ident(r) for r in json.loads(row["messages"][2]["content"]))
uncovered = {doc: sorted(want[doc] - got.get(doc, set())) for doc in want if want[doc] - got.get(doc, set())}
print()
print("user turns that are NOT document text:", synthetic, "(was 324)")
print("assistant references carrying source_text:", leaked)
print("documents with an uncovered reference:", len(uncovered))
for doc, missing in list(uncovered.items())[:5]:
    print("   ", doc, missing)
PY
```

Measured on 2026-09-07 against the sealed 300-document set. Read every line; only the last one is a stop-the-line gate.

| Manifest key | v10 (sealed) | v11 (measured) |
| --- | --- | --- |
| `output_row_count` | 2022 | **2194** |
| `native_document_count` | 28 | **35** |
| `windowed_document_count` | 272 | **265** |
| `original_reference_count` | 2152 | 2152 |
| `chunk_reference_count` | 3060 | **3061** |
| `reference_rescue_count` | 2 | **1** |
| `dense_replay_row_count` | 333 | **506** |
| `dense_replay_reference_count` | 2075 | **2041** |
| `training_reference_count` | 5135 | **5102** |
| `maximum_tokens` | 1536 | 1536 |

Two of those moves look like loss and are not. Both were verified, not assumed:

- **`dense_replay_reference_count` 2075 → 2041.** The 34-instance drop is exactly the dense-replay instances of the seven documents Task 4 moved from windowed to native — ids 9, 109, 166, 213, 363, 443 and 477. A native document is served whole and is never replayed, so its references are now taught once with full document context instead of once per window plus a replay row. Summing those seven documents' v10 replay `reference_count` values gives exactly 34.
- **`reference_rescue_count` 2 → 1, with `chunk_reference_count` 3060 → 3061.** The same single document (148) is still rescued. Task 5's rescue window labels every reference visible in it rather than only the one that triggered it, so one window now retires what previously took two rows: one fewer row, one more labelled instance.

The three gates, and the only reason to stop:

- **user turns that are NOT document text: 0** (v10 had 324) — this branch's whole purpose
- **assistant references carrying source_text: 0**
- **documents with an uncovered reference: 0** — computed from the output rows by five-field identity, not read from the manifest's `reference_coverage`, which is an unconditional literal

**Stop only on the third gate.** "References were lost" means a document has a reference that appears in no training row. The instance counters above move for structural reasons; explain any deviation from the table, never simply accept it.

- [ ] **Step 3: Record the new optimizer-step geometry**

The sealed run was 2022 rows → 506 optimizer updates (batch 1 × grad-accum 4) with warmup 25. The rebuilt view is **2194 rows → `ceil(2194 / 4)` = 549 optimizer updates**. Record that in the verification artefact. M₀′'s and M₁'s schedules must be derived from the new count, never carried over from 506 — and the M₀′ − M₀ comparison is therefore "recipe plus the schedule length the recipe implies", exactly as Task 9 states.

- [ ] **Step 4: Re-run the judge pilot**

Run inside tmux (~35 min; the sealed baseline was 2050.6 s and the payload is now smaller, so expect less):

```bash
.venv/bin/dqcheck pilot-judges --batch-id M01 --generation M000 --allow-external-judge
```

Read `artifacts/data_quality_checker/batches/M01/judge_pilot_summary.json` and gate on:
- `models[...]["stale_verdicts_rejudged"] == 57` — proof the rerun actually re-judged rather than reusing the invalid verdicts
- `judge_instrument.prompt_sha256 != "aa887e10…"`
- `judge_instrument.candidate_payload_contract == "decision_fields_only_v1"`
- GREEN verdicts turn **TIE-heavy**. The baseline was 17 annotator / 1 model / 4 TIE / 1 NEITHER over 18 decisive verdicts, where the five decision fields are identical by construction. If the annotator still wins GREEN decisively, blinding was not the whole story and that is a real finding — record it, do not paper over it.
- `error` may rise above 3. The judge previously had the annotator's verbatim spans to crib its own `evidence` array from and must now find spans in `document_text` itself, which `evidence_match_mode` checks. A rise is the measured cost of blinding.

Publish the run's cost alongside: latency p50/p95, prompt/generation tokens, peak memory (all present in each result's `operational` block).

- [ ] **Step 5: Write the evidence artefact**

Create `artifacts/data_quality_checker/evidence/2026-09-07_source_text_off_model_surfaces/verification.json` holding: the manifest diff from Step 2, the step geometry from Step 3, the pilot summary from Step 4, the suite result from Step 1, and the sha256 of the rebuilt view.

- [ ] **Step 6: Commit**

```bash
git add artifacts/data_quality_checker/evidence/2026-09-07_source_text_off_model_surfaces/
git commit -m "docs: evidence that source_text reaches no model surface"
```

---

### Task 9: Re-baseline M₀ under the new recipe before M01's documents join

**This task is a measurement, not a code change, and it is the one that keeps the loop's headline claim intact.**

The DQ-Loop measures one thing over 12 rounds: what more cleaned data buys. M₀ was trained under recipe v10 on 300 documents. If M₁ is trained under v11 on 400, then `M₁ − M₀` mixes the extra 100 documents with a changed training recipe, and no later round can separate them.

Because `build_context_window_view` is applied to `train.jsonl` alone — verified at `mlx_compute.py:419-421`, with validation and test never windowed — a v11 run on the identical sealed 300/50 split is directly comparable to M₀'s **0.853933** core F1 at update 325. One training run buys a clean baseline; the 150-document test split stays closed.

- [ ] **Step 1: Train M₀′ on the sealed 300-document split under v11**

Same split (300/50/150, seed `20260902`, fingerprint `7967d589…`), same base model and revision, same optimisation contract — only the view policy differs. The run lands under its own new `dqcheck_loop_<fingerprint>` directory, so `dqcheck_loop_fd1abc3af6b7` is untouched. Derive the schedule from Task 8 Step 3's step count rather than reusing 506.

- [ ] **Step 2: Select a checkpoint on validation alone and record both numbers**

Report M₀′ core F1 / P / R beside M₀'s 0.853933 / 0.894118 / 0.817204. Publish latency p50/p95, tokens and peak memory with them.

- [ ] **Step 3: Write down what the number means before reading it**

Recall is the number at risk and the number this project has been fighting for. Dense replay taught many-references-at-once, and its packs shrank from up to 10 references to a median of 4 because a real spanning passage is longer than a list of clipped phrases. If M₀′ recall falls materially below 0.817204, the honest options are to raise `DENSE_REPLAY_MAX_REFERENCES`'s effective density another way or to accept the trade — not to put annotator prose back in front of the model.

- [ ] **Step 4: Only then continue round 1**

With M₀′ sealed, round 1 proceeds: expert adjudication of all 60 pilot documents → `judge-lock` → release → M₁. Note the ordering correction: `lock_judge` calls `judge_expert_metrics`, which returns `"pending"` unless **every** pilot document has a `finalized` review, and it refuses any model with a non-`valid` pilot output (`judges.py:1147-1163`). All 100 M01 reviews are `pending` today and 3 verdicts are `error`, so **judge-lock cannot precede expert adjudication** — HANDOFF §8's order is reversed.

---

## Open item this plan deliberately does not close

`judge_override` (`hitl.py:842`) writes the judge's `final_references` as the expert's decision; `release.py:242` puts that into `training_rows[].references`. A blinded judge cannot return `source_text`, and `context_windows.py` needs one to locate a reference — `_dense_replay_rows` and the rescue path raise `IntegrityError`, and `_window_row`'s visibility filter drops the reference silently. This is already latent (`tests/test_judges.py:610` permits an absent `source_text` in a verdict); blinding makes it systematic.

It cannot fire in round 1 — `judge_override` requires a locked judge, and Task 9 Step 4 shows the lock comes after expert adjudication — but it must be settled before the expert starts. The recommended fix is to re-attach `source_text` at `judge_override` time by matching each of the judge's final references against the annotator's and the model's rows on `full_identity` (the five decision fields only, so `source_text` never influences the match), leaving a genuinely novel judge row blank. The alternative is to keep `judge_override` documents out of `training_rows` and accept the per-round attrition.

---

### Task 10: Locate a known reference's span, so the expert need not type 231 of them

**Why this task exists.** The loop's purpose is to harvest references the annotator missed. Measured on M01: **231 references across 80 of 100 documents** are in M000's output and absent from the annotator's by five-field identity. None of them carries a `source_text` from anyone — the model does not emit one and the annotator never wrote the reference down. And a reference with an empty `source_text` on a document that must be windowed fails the training build outright; verified by running it:

```
IntegrityError: reference rescue has no source_text at source row 0
```

M01's documents are a median 5432 characters and only 35 of the sealed 300 fit natively, so effectively every one of them is windowed. Without this task the expert must type 231 spans, or the M₁ build dies.

**What was measured before choosing an approach.** The existing `annotator/reference/regex.py` engine was the obvious candidate and it is not good enough: it locates 129 of the 231 (55.8%), and only 62 of those spans are verbatim substrings of the raw document, because it extracts from `normalize_reference_text`'s output rather than the raw body. `context_windows.py` needs a verbatim substring (`body.find`), so that path yields 27% usable spans.

Finding *where a known reference is* is a much narrower problem than finding *which references exist*. A purpose-built locator, measured on the same 231:

| outcome | count | share |
| --- | --- | --- |
| article-level span, automatic | **197** | 85.3% |
| law-level reference (no article), law mention is the correct span | **23** | 10.0% |
| refused, goes to the expert | **11** | 4.8% |

All 220 automatic spans are verbatim substrings of the raw text. This also removes an architectural problem: nothing needs to import `annotator/`, so the two engines stay independent as `benchmark/` already does.

**The refusals are a signal, not a gap.** Each of the 11 was inspected in the document text. `d000052`'s six rows claim VUK articles 350/362/381/396, but the document says *"sırasıyla 350, 362, 381 ve 396 Sıra No.lu Vergi Usul Kanunu Genel Tebliğleri"* — those are communiqué serial numbers, not articles. `d000031`'s `488/34` points at *"(34) numaralı fıkrasında"* of a table section, not article 34. So an unlocatable article is evidence the reference is wrong, and the right response is to show it to the expert rather than invent a span. Most of these will disappear during adjudication anyway, because they are rows the expert deletes.

**Files:**
- Create: `src/data_quality_checker/reference_locator.py`
- Modify: `src/data_quality_checker/loop_training.py` (`round_training_documents`)
- Test: `tests/test_reference_locator.py`

**Interfaces:**
- Produces `locate_reference_span(document_text: str, reference: Mapping[str, str]) -> str | None` — a verbatim substring of `document_text` that supports `reference`, or `None` when the reference names an article the text does not place. Never returns a non-substring; never guesses an article-level span from a law mention.
- Produces `attach_missing_spans(documents: Mapping[Any, dict[str, Any]]) -> tuple[dict[Any, dict[str, Any]], list[dict[str, str]]]` — fills empty `source_text` values in each document's `references` and returns the documents alongside a refusal list of `{"doc_id", "kanun_no", "kanun_ad", "madde", "fikra", "bent"}`.
- `round_training_documents` raises `ContractError` naming **every** refusal at once, so the expert gets one worklist instead of one error per run.

**The seam, and why it is there.** The locator runs in `round_training_documents`, between the immutable release and the training view. The release stays a faithful record of what the expert decided, with no machine-added spans; the located span exists only as the build-time locator this whole branch reduced `source_text` to. It must not be written back into the release or the database.

- [ ] **Step 1: Write the failing tests**

`tests/test_reference_locator.py`, with real Turkish drafting shapes rather than invented ones — each of these was taken from an M01 document:

```python
def test_locates_an_ordinary_article_mention() -> None:
    text = "213 sayılı Vergi Usul Kanununun 413 üncü maddesinde belirtilen usul"
    span = locate_reference_span(text, {"kanun_no": "213", "kanun_ad": "Vergi Usul Kanunu",
                                        "madde": "413", "fikra": "", "bent": ""})
    assert span in text
    assert "413" in span


def test_locates_each_number_of_a_shared_ordinal_series() -> None:
    """"2, 3 ve 4 üncü maddeleri": only the last number carries the suffix,
    yet every number in the series names an article. Taken from d000014."""
    text = "Mezkûr Kararın 2, 3 ve 4 üncü maddeleri gereğince, internet ortamında"
    for madde in ("2", "3", "4"):
        span = locate_reference_span(text, {"kanun_no": "476", "kanun_ad": "",
                                            "madde": madde, "fikra": "", "bent": ""})
        assert span is not None and span in text


def test_a_law_level_reference_is_located_by_its_law_mention() -> None:
    text = "5957 sayılı Sebze ve Meyveler ile Yeterli Arz ve Talep Derinliği bulunan"
    span = locate_reference_span(text, {"kanun_no": "5957", "kanun_ad": "Sebze ve Meyveler",
                                        "madde": "", "fikra": "", "bent": ""})
    assert span is not None and span in text


def test_refuses_a_serial_number_masquerading_as_an_article() -> None:
    """d000052: the model read communiqué serial numbers as VUK articles. The
    locator must refuse rather than hand back the law mention, because an
    unlocatable article is evidence the reference itself is wrong."""
    text = ("uyulacak usul ve esaslar sırasıyla 350, 362, 381 ve 396 Sıra No.lu "
            "Vergi Usul Kanunu Genel Tebliğleri ile duyurulmuştur")
    span = locate_reference_span(text, {"kanun_no": "213", "kanun_ad": "Vergi Usul Kanunu",
                                        "madde": "350", "fikra": "", "bent": ""})
    assert span is None


def test_every_returned_span_is_a_verbatim_substring() -> None:
    """The only property context_windows.py actually depends on."""
    text = "3065 sayılı Katma Değer Vergisi Kanununun 17/4-k maddesi uyarınca"
    for madde in ("17/4-k", "17"):
        span = locate_reference_span(text, {"kanun_no": "3065", "kanun_ad": "Katma Değer Vergisi Kanunu",
                                            "madde": madde, "fikra": "", "bent": ""})
        if span is not None:
            assert span in text
```

- [ ] **Step 2: Run them and confirm they fail on the missing module**

Run: `.venv/bin/python -m pytest tests/test_reference_locator.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'data_quality_checker.reference_locator'`.

- [ ] **Step 3: Write the locator**

The probe that produced the measured table above is at `scratchpad/locate_probe.py` and is the reference implementation: an ordinal-suffix alternation, three article patterns, the comma-series pattern, law-mention scoring that prefers the article mention nearest after a law mention, and a 90-character context window. Port it into `reference_locator.py` with the measured numbers in the module docstring, and **refuse rather than fall back to a law mention whenever `madde` is non-empty**.

- [ ] **Step 4: Wire it into `round_training_documents` and raise on refusals**

- [ ] **Step 5: Mutation proof**

Remove the refusal branch so an unlocatable article falls back to the law mention; `test_refuses_a_serial_number_masquerading_as_an_article` must fail. Remove the comma-series pattern; `test_locates_each_number_of_a_shared_ordinal_series` must fail for `"2"` and `"3"`.

- [ ] **Step 6: Re-measure against the live 231 and confirm the table above reproduces**
