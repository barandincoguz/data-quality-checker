# CoT-Enumeration Recall Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, zero-shot on the base model, whether a 3-step enumerate→resolve→filter CoT prompt recovers genuinely-missed law references (recall up without a precision collapse), on validation-50, in three arms.

**Architecture:** A standalone experiment (not part of the `data_quality_checker` package): pure prompt/parse helpers with unit tests, plus a resumable inference driver that loads the base `Qwen3.5-9B-MLX-4bit` once, runs each arm over the 50 validation docs, writes evaluator-format predictions, and runs the canonical evaluator. No change to the fine-tune contract, `g0.py`, or the sealed test.

**Tech Stack:** Python 3.11, `mlx_lm` (load / stream_generate / make_sampler), the repo's `benchmark/reference/evaluate.py` evaluator, pytest.

## Global Constraints

- Base model, **no adapter**: `mlx-community/Qwen3.5-9B-MLX-4bit` (local snapshot; run offline with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`).
- Decoding: temperature 0.0; `enable_thinking=False`; inference context 12288.
- `max_generation_tokens`: plain arm 2048, cot_enum arm 4096.
- Documents: validation-50 only. **Do NOT touch the sealed test-50.**
- Evaluation flags (identical to fine-tune validation): `--reference-postprocess canonical_set --core-reference-view row --gt-mode approved_only --docwise-threshold 1.0 --per-doc`, primary metric `core_law_article_strict`.
- Python interpreter: `/opt/llm-lab/.venv/bin/python`; unit tests: `.venv-dqcheck/bin/python -m pytest`.
- GPU is shared with another user; the driver MUST cache per-doc and resume, so repeated invocations accumulate progress.
- Data sources (run `955f5a14e275`): docs `data/sensitive/data_quality_checker/g0/dqcheck_g0_qwen3_5_9b_955f5a14e275/data/valid.jsonl` (user message = document text) aligned by index with `valid_doc_ids.json`; gold = `data/ground_truth/gt_v3_triangulated_2026-05-15/validated/doc_<id>.json`.

**Base directory for all created files:** `Fine-Tuning/experiments/2026-07-25_cot_enum_probe/`

---

### Task 1: Pure prompts + parsing helpers

**Files:**
- Create: `Fine-Tuning/experiments/2026-07-25_cot_enum_probe/prompts.py`
- Test: `Fine-Tuning/experiments/2026-07-25_cot_enum_probe/test_prompts.py`

**Interfaces:**
- Produces: `PLAIN_PROMPT: str`, `COT_ENUM_PROMPT: str`, `REFERENCE_FIELDS: tuple[str,...]`, `parse_final_json(text: str) -> tuple[list[dict[str,str]], bool]`, `to_prediction_record(doc_id: int, references: list[dict[str,str]], parse_ok: bool) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
# test_prompts.py
from prompts import (
    PLAIN_PROMPT, COT_ENUM_PROMPT, REFERENCE_FIELDS,
    parse_final_json, to_prediction_record,
)

def test_parse_extracts_reference_with_all_six_fields():
    text = ('reasoning...\nFINAL_JSON: [{"kanun_no":"213","kanun_ad":"Vergi Usul Kanunu",'
            '"madde":"413","fikra":"","bent":"","source_text":"x"}]')
    refs, ok = parse_final_json(text)
    assert ok is True
    assert len(refs) == 1
    assert refs[0]["kanun_no"] == "213"
    assert set(refs[0].keys()) == set(REFERENCE_FIELDS)

def test_parse_empty_array_is_ok():
    refs, ok = parse_final_json("blah\nFINAL_JSON: []")
    assert ok is True and refs == []

def test_parse_uses_last_marker_and_first_balanced_array():
    text = 'FINAL_JSON: [old]\nmore\nFINAL_JSON: [{"kanun_no":"1","kanun_ad":"a","madde":"2","fikra":"","bent":"","source_text":"s"}] trailing text'
    refs, ok = parse_final_json(text)
    assert ok is True and len(refs) == 1 and refs[0]["madde"] == "2"

def test_parse_missing_marker_is_not_ok():
    refs, ok = parse_final_json("no marker here [1,2]")
    assert ok is False and refs == []

def test_parse_malformed_json_is_not_ok():
    refs, ok = parse_final_json("FINAL_JSON: [not json")
    assert ok is False and refs == []

def test_to_prediction_record_shapes_for_evaluator():
    rec = to_prediction_record(5, [{"kanun_no": "1", "kanun_ad": "a", "madde": "2",
                                     "fikra": "", "bent": "", "source_text": "s"}], True)
    assert rec == {"doc_id": 5, "status": "success",
                   "references": [{"kanun_no": "1", "kanun_ad": "a", "madde": "2",
                                   "fikra": "", "bent": "", "source_text": "s"}]}

def test_to_prediction_record_parse_error_status():
    rec = to_prediction_record(9, [], False)
    assert rec["status"] == "parse_error" and rec["references"] == []

def test_prompts_are_nonempty_and_cot_requires_final_json():
    assert PLAIN_PROMPT.strip()
    assert "FINAL_JSON:" in COT_ENUM_PROMPT
    assert "STEP 1" in COT_ENUM_PROMPT and "FILTER" in COT_ENUM_PROMPT
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd Fine-Tuning/experiments/2026-07-25_cot_enum_probe && /Users/student2/ner-project/data-quality-checker-weak-learning-program/.venv-dqcheck/bin/python -m pytest test_prompts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'prompts'`.

- [ ] **Step 3: Write `prompts.py`**

`PLAIN_PROMPT` is the winning recall-v2 system prompt copied verbatim (retrieve exact text with `git show dbe82d3^:data-quality-checker-weak-learning-program/src/data_quality_checker/g0.py` and copy the `SYSTEM_PROMPT` string). The rest:

```python
"""Prompts and parsing for the CoT-enumeration recall probe (zero-shot base model)."""
from __future__ import annotations
import json

REFERENCE_FIELDS = ("kanun_no", "kanun_ad", "madde", "fikra", "bent", "source_text")

# recall-v2 winning system prompt, copied verbatim from
# git show dbe82d3^:.../g0.py  (SYSTEM_PROMPT). Paste the exact string here.
PLAIN_PROMPT = (
    "You extract every statutory law reference from Turkish tax rulings (ozelgeler).\n\n"
    "Return one flat JSON array and include ALL references in the document. Each item has "
    "exactly these string fields: kanun_no, kanun_ad, madde, fikra, bent, source_text. Use an "
    "empty string when a field is absent. Resolve locally supported anaphora; retain "
    "table/cetvel/list references in the same contract; never invent a law identity or evidence. "
    "Deduplicate the same legal tuple and suppress a generic law-only row when that law has a "
    "specific article row. Output only the JSON array; if no references exist, output [].\n\n"
    "Recall rules adapted from the official few-shot-cot-v3-en prompt:\n"
    "- Do not become overly conservative. Extract every explicit article, paragraph, and "
    "subparagraph reference, even when secondary regulations appear nearby.\n"
    "- Preserve every distinct explicit legal tuple. Never replace a specific tuple with a "
    "generic law-only row, and never return [] when an explicit statutory reference exists.\n"
    "- Scan the entire document once more for dropped madde, fikra, and bent details before "
    "returning JSON.\n"
)

COT_ENUM_PROMPT = (
    "You extract every statutory law reference from Turkish tax rulings (ozelgeler).\n\n"
    "Work through three explicit steps in your response, then output the final answer.\n\n"
    "STEP 1 - ENUMERATE (be exhaustive; over-inclusion is fine here): list every span in the "
    "document that mentions an article/madde, paragraph/fikra, or subparagraph/bent - including "
    "bare 'N inci/uncu maddesi', anaphoric 'ayni maddenin ... fikrasi', and 'gecici N nci "
    "maddesi'. Skip nothing; if in doubt, list it.\n"
    "STEP 2 - RESOLVE: for each enumerated span, determine kanun_no and kanun_ad from the nearest "
    "supporting context; never invent a law identity or evidence.\n"
    "STEP 3 - FILTER: drop spans with no resolvable law identity or that are not genuine statutory "
    "article references; deduplicate the same (kanun_no, kanun_ad, madde) identity; suppress a "
    "generic law-only row when a specific-article row for that law exists.\n\n"
    "Then output exactly one line starting with 'FINAL_JSON:' followed by a single JSON array. "
    "Each item has exactly these string fields: kanun_no, kanun_ad, madde, fikra, bent, "
    "source_text (empty string when a field is absent). If no references exist, output "
    "'FINAL_JSON: []'.\n"
)


def parse_final_json(text):
    """Return (references, parse_ok): the JSON array after the LAST 'FINAL_JSON:' marker,
    taking the first balanced [...] and normalising each item to the six string fields."""
    marker = "FINAL_JSON:"
    idx = text.rfind(marker)
    if idx < 0:
        return [], False
    tail = text[idx + len(marker):]
    start = tail.find("[")
    if start < 0:
        return [], False
    depth = 0
    blob = None
    for i in range(start, len(tail)):
        if tail[i] == "[":
            depth += 1
        elif tail[i] == "]":
            depth -= 1
            if depth == 0:
                blob = tail[start:i + 1]
                break
    if blob is None:
        return [], False
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return [], False
    if not isinstance(data, list):
        return [], False
    refs = []
    for item in data:
        if isinstance(item, dict):
            refs.append({f: str(item.get(f, "")) for f in REFERENCE_FIELDS})
    return refs, True


def to_prediction_record(doc_id, references, parse_ok):
    """Evaluator-format record: matches the fine-tune validation predictions.json schema."""
    return {
        "doc_id": int(doc_id),
        "status": "success" if parse_ok else "parse_error",
        "references": references,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd Fine-Tuning/experiments/2026-07-25_cot_enum_probe && /Users/student2/ner-project/data-quality-checker-weak-learning-program/.venv-dqcheck/bin/python -m pytest test_prompts.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add Fine-Tuning/experiments/2026-07-25_cot_enum_probe/prompts.py Fine-Tuning/experiments/2026-07-25_cot_enum_probe/test_prompts.py
git commit -m "feat(cot-probe): prompts + FINAL_JSON parser with tests"
```

---

### Task 2: Resumable base-model inference driver

**Files:**
- Create: `Fine-Tuning/experiments/2026-07-25_cot_enum_probe/run_probe.py`

**Interfaces:**
- Consumes: `prompts.PLAIN_PROMPT`, `prompts.COT_ENUM_PROMPT`, `prompts.parse_final_json`, `prompts.to_prediction_record`.
- Produces: CLI `python run_probe.py --arm {plain,cot_enum}` that writes
  `<base>/arms/<arm>/records/doc_<id>.json` (per-doc cache) and, when all 50 exist,
  `<base>/arms/<arm>/predictions.json` (list of prediction records).

- [ ] **Step 1: Write `run_probe.py`**

```python
"""Zero-shot base-model inference for the CoT-enumeration recall probe.

Resumable: each doc's record is cached; re-running skips completed docs, so a
SIGKILL from GPU contention only costs the in-flight doc.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

from prompts import PLAIN_PROMPT, COT_ENUM_PROMPT, parse_final_json, to_prediction_record

REPO = Path("/Users/student2/ner-project")
DATA = REPO / "data/sensitive/data_quality_checker/g0/dqcheck_g0_qwen3_5_9b_955f5a14e275/data"
MODEL = ("/opt/llm-lab/hf-cache/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/"
         "snapshots/938d8919941c6e7efd3c7150eff7fe9d12afa631")
BASE = Path(__file__).resolve().parent
PROMPTS = {"plain": PLAIN_PROMPT, "cot_enum": COT_ENUM_PROMPT}
MAX_TOKENS = {"plain": 2048, "cot_enum": 4096}


def load_docs():
    rows = [json.loads(l) for l in (DATA / "valid.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    doc_ids = json.loads((DATA / "valid_doc_ids.json").read_text(encoding="utf-8"))
    assert len(rows) == len(doc_ids), "valid rows/doc_ids misaligned"
    return [(int(d), r["messages"][1]["content"]) for d, r in zip(doc_ids, rows)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["plain", "cot_enum"], required=True)
    args = ap.parse_args()
    arm_dir = BASE / "arms" / args.arm
    records_dir = arm_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    docs = load_docs()
    pending = [(d, t) for d, t in docs if not (records_dir / f"doc_{d}.json").exists()]
    print(f"[{args.arm}] {len(docs)-len(pending)}/{len(docs)} cached; {len(pending)} to run", flush=True)

    if pending:
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
        model, tokenizer = load(MODEL)  # no adapter -> base model
        sampler = make_sampler(temp=0.0)
        system = PROMPTS[args.arm]
        max_tokens = MAX_TOKENS[args.arm]
        for doc_id, text in pending:
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": text}],
                add_generation_prompt=True, enable_thinking=False, tokenize=False,
            )
            parts = []
            for resp in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler):
                parts.append(resp.text)
            raw = "".join(parts)
            refs, ok = parse_final_json(raw)
            record = to_prediction_record(doc_id, refs, ok)
            record["raw_output"] = raw
            (records_dir / f"doc_{doc_id}.json").write_text(
                json.dumps(record, ensure_ascii=False), encoding="utf-8")
            print(f"[{args.arm}] doc {doc_id}: parse_ok={ok} refs={len(refs)}", flush=True)

    # assemble predictions.json when all 50 present
    ids = [d for d, _ in docs]
    if all((records_dir / f"doc_{d}.json").exists() for d in ids):
        preds = []
        for d in ids:
            rec = json.loads((records_dir / f"doc_{d}.json").read_text(encoding="utf-8"))
            preds.append({k: rec[k] for k in ("doc_id", "status", "references")})
        (arm_dir / "predictions.json").write_text(json.dumps(preds, ensure_ascii=False), encoding="utf-8")
        parsed = sum(1 for d in ids if json.loads((records_dir/f'doc_{d}.json').read_text())["status"] == "success")
        print(f"[{args.arm}] predictions.json written: {len(preds)} docs, parse_ok {parsed}/{len(preds)}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run the plain arm on a tiny slice to verify plumbing**

Temporarily verify generation works by running the driver and interrupting after a few docs (Ctrl-C or let GPU contention stop it), then confirm cache files appear:

Run:
```bash
cd Fine-Tuning/experiments/2026-07-25_cot_enum_probe
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /opt/llm-lab/.venv/bin/python run_probe.py --arm plain
```
Expected: prints `[plain] doc <id>: parse_ok=True refs=N` lines; `arms/plain/records/doc_*.json` files are created. (May be SIGKILLed by GPU contention — that is fine; re-running resumes.)

- [ ] **Step 3: Verify a cached record is well-formed**

Run: `python -c "import json,glob; d=json.load(open(sorted(glob.glob('arms/plain/records/doc_*.json'))[0])); print(d['doc_id'], d['status'], len(d['references']), list(d['references'][0].keys()) if d['references'] else [])"`
Expected: a doc_id, `success`, a reference count, and the six field names.

- [ ] **Step 4: Commit**

```bash
git add Fine-Tuning/experiments/2026-07-25_cot_enum_probe/run_probe.py
git commit -m "feat(cot-probe): resumable base-model inference driver"
```

---

### Task 3: Run both arms to completion + evaluate + compare

**Files:**
- Create: `Fine-Tuning/experiments/2026-07-25_cot_enum_probe/evaluate_arms.sh`
- Create (output): `Fine-Tuning/experiments/2026-07-25_cot_enum_probe/COMPARISON.md`

**Interfaces:**
- Consumes: `arms/<arm>/predictions.json` from Task 2.
- Produces: `arms/<arm>/evaluation.json` and a written 3-arm comparison table.

- [ ] **Step 1: Drive both arms to 50/50 with an auto-resume loop**

Run (repeats through GPU-contention SIGKILLs until predictions.json exists for both arms):
```bash
cd Fine-Tuning/experiments/2026-07-25_cot_enum_probe
for arm in plain cot_enum; do
  for attempt in $(seq 1 40); do
    [ -f "arms/$arm/predictions.json" ] && break
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /opt/llm-lab/.venv/bin/python run_probe.py --arm "$arm" || true
    sleep 8
  done
done
```
Expected: both `arms/plain/predictions.json` and `arms/cot_enum/predictions.json` exist, each 50 docs.

- [ ] **Step 2: Write `evaluate_arms.sh` (canonical evaluator per arm)**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /Users/student2/ner-project
GT=data/ground_truth/gt_v3_triangulated_2026-05-15/validated
IDS=data/sensitive/data_quality_checker/g0/dqcheck_g0_qwen3_5_9b_955f5a14e275/data/valid_doc_ids.json
BASE=Fine-Tuning/experiments/2026-07-25_cot_enum_probe/arms
for arm in plain cot_enum; do
  /opt/llm-lab/.venv/bin/python benchmark/reference/evaluate.py \
    --predictions "$BASE/$arm/predictions.json" \
    --ground-truth-dir "$GT" \
    --doc-ids-file "$IDS" \
    --json-report "$BASE/$arm/evaluation.json" \
    --gt-mode approved_only \
    --docwise-threshold 1.0 \
    --reference-postprocess canonical_set \
    --core-reference-view row \
    --per-doc
done
```
(If `benchmark/reference/evaluate.py` rejects an unknown flag, read `benchmark/reference/evaluate.py --help` and align flag names before proceeding — the set above mirrors `g0_training._canonical_evaluate`.)

- [ ] **Step 3: Run the evaluator**

Run: `bash Fine-Tuning/experiments/2026-07-25_cot_enum_probe/evaluate_arms.sh`
Expected: `arms/plain/evaluation.json` and `arms/cot_enum/evaluation.json` written.

- [ ] **Step 4: Emit the 3-arm comparison table**

Run:
```bash
cd /Users/student2/ner-project
/opt/llm-lab/.venv/bin/python - <<'PY'
import json
B="Fine-Tuning/experiments/2026-07-25_cot_enum_probe/arms"
def row(tag, path):
    r=json.load(open(path))["results"][0]["core_law_article_strict"]
    dw=json.load(open(path))["results"][0]["docwise_core_accuracy"]
    return f"| {tag} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['tp']}/{r['fp']}/{r['fn']} | {dw['passed_doc_count']}/50 |"
print("| arm | P | R | F1 | tp/fp/fn | docwise |")
print("|---|---|---|---|---|---|")
print(row("base+plain (zero-shot)", f"{B}/plain/evaluation.json"))
print(row("base+cot_enum (zero-shot)", f"{B}/cot_enum/evaluation.json"))
print("| fine-tuned warm42-cos150@75 (val, reference) | 0.895 | 0.735 | 0.807 | 239/28/86 | 10/50 |")
PY
```
Expected: a 3-row markdown table printed. **Verdict rule:** the CoT mechanism *works* iff `cot_enum` recall > `plain` recall AND `cot_enum` precision does not collapse (not merely more predictions at lower precision).

- [ ] **Step 5: Write `COMPARISON.md` and commit**

Paste the printed table into `COMPARISON.md` with a one-paragraph verdict (does CoT recover misses vs plain-base? does precision hold? how does it compare to the fine-tuned reference?).

```bash
git add Fine-Tuning/experiments/2026-07-25_cot_enum_probe/evaluate_arms.sh Fine-Tuning/experiments/2026-07-25_cot_enum_probe/COMPARISON.md
git commit -m "feat(cot-probe): 3-arm evaluation + comparison"
```

---

### Task 4 (deferred, do NOT skip at the end): Reporting

After Task 3's verdict, update the study report and ledger (the user explicitly asked to remember this **after** the run):

- [ ] Add a "Study V — CoT-enumeration zero-shot probe" section to `Fine-Tuning/reports/2026-07-25_g0_finetuning_recall_study/REPORT.md` (3-arm table + verdict; note the base+plain number now also answers `assert_test_improves_base`).
- [ ] Add an `EXPERIMENT_LEDGER.md` entry (next free 2.x number) summarising the probe and its decision (mechanism works → bake into fine-tune via a new spec; or fails → data scale-up is the only recall lever).
- [ ] Commit: `git commit -m "docs(finetuning): CoT-enumeration zero-shot probe results"`.

---

## Self-Review

**Spec coverage:** mechanism (Task 1 prompts) ✓; zero-shot base-model 3-arm test (Tasks 2–3) ✓; validation-50 + canonical evaluator flags (Global Constraints + Task 3) ✓; GPU-contention resume (Task 2 cache + Task 3 loop) ✓; success criterion/verdict (Task 3 Step 4) ✓; reporting deferred (Task 4) ✓; out-of-scope (no g0.py/contract/sealed-test change) respected — the probe is standalone.

**Placeholder scan:** `PLAIN_PROMPT` is given verbatim with a git command to confirm the exact source; all other code is complete; the only "fill-in" is `COMPARISON.md` prose, which is a human-judgment artifact with explicit content instructions. No forbidden placeholders.

**Type consistency:** `parse_final_json -> (list[dict], bool)` consumed correctly by the driver; `to_prediction_record(doc_id, references, parse_ok)` signature matches its call; predictions.json keys `doc_id/status/references` match the evaluator input used by the fine-tune validation path.
