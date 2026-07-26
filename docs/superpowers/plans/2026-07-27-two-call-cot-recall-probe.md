# Two-Call CoT Recall Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, zero-shot on the base model, whether a two-call terse pipeline (Call 1 enumerate → Call 2 resolve+filter → code dedup net) recovers recall without a precision collapse and without truncation, beating the plain prompt and Study V's single-call CoT.

**Architecture:** Extend the Study V experiment (`Fine-Tuning/experiments/2026-07-25_cot_enum_probe/`) with a new `two_call_cot` arm. Two bounded base-model calls per document plus a deterministic code dedup net reusing `data_quality_checker.normalization.compact_references`. Reuse the Study V plain-arm baseline, evaluator, and cache/resume pattern.

**Tech Stack:** Python 3.11, `mlx_lm` (load / stream_generate / make_sampler), `data_quality_checker.normalization`, the repo evaluator `benchmark/reference/evaluate.py`, pytest.

## Global Constraints

- Base model, **no adapter**: `mlx-community/Qwen3.5-9B-MLX-4bit`, snapshot `/opt/llm-lab/hf-cache/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/938d8919941c6e7efd3c7150eff7fe9d12afa631`. Run offline: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
- Decoding: temperature 0.0, `enable_thinking=False`. Token budgets: **Call 1 (enumerate) 2048, Call 2 (resolve+filter) 3072**.
- Documents: **validation-50 only.** Do NOT touch sealed test-50.
- Evaluation (identical to fine-tune validation): `PYTHONPATH=/Users/student2/ner-project`, flags `--gt-mode approved_only --docwise-threshold 1.0 --reference-postprocess canonical_set --core-reference-view row --per-doc`, metric `core_law_article_strict`.
- Interpreters: GPU/mlx + evaluator `/opt/llm-lab/.venv/bin/python`; unit tests `data-quality-checker-weak-learning-program/.venv-dqcheck/bin/python -m pytest`.
- GPU shared with another user → SIGKILL possible; the driver MUST cache per-doc and resume; wrap runs in a bash retry-loop until `predictions.json` exists.
- Data: `data/sensitive/data_quality_checker/g0/dqcheck_g0_qwen3_5_9b_955f5a14e275/data/valid.jsonl` (user msg = doc text) aligned by index with `valid_doc_ids.json`; gold `data/ground_truth/gt_v3_triangulated_2026-05-15/validated/doc_<id>.json`.
- Git: commit only predictions/evaluation/COMPARISON (raw `records/` is gitignored). Push to the **`baran`** remote (`origin` is 403).

**Base directory for created/modified files:** `Fine-Tuning/experiments/2026-07-25_cot_enum_probe/`
**Repo root:** `/Users/student2/ner-project` · **dqcheck src:** `<repo>/data-quality-checker-weak-learning-program/src`

---

### Task 1: Two-call prompts + pure helpers

**Files:**
- Modify: `Fine-Tuning/experiments/2026-07-25_cot_enum_probe/prompts.py` (append; keep existing `PLAIN_PROMPT`, `COT_ENUM_PROMPT`, `parse_final_json`, `to_prediction_record`, `REFERENCE_FIELDS` unchanged)
- Modify: `Fine-Tuning/experiments/2026-07-25_cot_enum_probe/test_prompts.py` (append tests)

**Interfaces:**
- Consumes (existing): `REFERENCE_FIELDS`.
- Produces: `ENUMERATE_PROMPT: str`, `RESOLVE_FILTER_PROMPT: str`, `parse_enumeration(text: str) -> list[str]`, `compact_via_normalization(refs: list[dict]) -> list[dict]`.

- [ ] **Step 1: Write the failing tests (append to test_prompts.py)**

```python
from prompts import (
    ENUMERATE_PROMPT, RESOLVE_FILTER_PROMPT, parse_enumeration, compact_via_normalization,
)

def test_parse_enumeration_strips_markers_and_blanks():
    text = "1. 213 sayılı Vergi Usul Kanununun 413. maddesi\n- 17 nci maddesi\n\n* aynı maddenin 2 nci fıkrası\n"
    spans = parse_enumeration(text)
    assert spans == [
        "213 sayılı Vergi Usul Kanununun 413. maddesi",
        "17 nci maddesi",
        "aynı maddenin 2 nci fıkrası",
    ]

def test_parse_enumeration_strips_surrounding_quotes():
    assert parse_enumeration('- "9 uncu maddesi"') == ["9 uncu maddesi"]

def test_parse_enumeration_empty():
    assert parse_enumeration("\n  \n") == []

def test_compact_via_normalization_dedups_identical_core():
    ref = {"kanun_no": "213", "kanun_ad": "Vergi Usul Kanunu", "madde": "413",
           "fikra": "", "bent": "", "source_text": "213 sayılı VUK 413"}
    out = compact_via_normalization([dict(ref), dict(ref)])
    assert len(out) == 1

def test_compact_via_normalization_suppresses_generic_law_only():
    specific = {"kanun_no": "193", "kanun_ad": "Gelir Vergisi Kanunu", "madde": "94",
                "fikra": "", "bent": "", "source_text": "GVK 94"}
    generic = {"kanun_no": "193", "kanun_ad": "Gelir Vergisi Kanunu", "madde": "",
               "fikra": "", "bent": "", "source_text": "193 sayılı Kanun"}
    out = compact_via_normalization([specific, generic])
    assert all(r["madde"] for r in out)  # the law-only generic row is dropped

def test_two_call_prompts_shape():
    assert "one per line" in ENUMERATE_PROMPT.lower() or "one span per line" in ENUMERATE_PROMPT.lower()
    assert "JSON" in RESOLVE_FILTER_PROMPT and "drop" in RESOLVE_FILTER_PROMPT.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd Fine-Tuning/experiments/2026-07-25_cot_enum_probe && /Users/student2/ner-project/data-quality-checker-weak-learning-program/.venv-dqcheck/bin/python -m pytest test_prompts.py -q`
Expected: FAIL — `ImportError: cannot import name 'ENUMERATE_PROMPT'`.

- [ ] **Step 3: Append implementation to prompts.py**

```python
import re
import sys
from pathlib import Path

_DQCHECK_SRC = Path("/Users/student2/ner-project/data-quality-checker-weak-learning-program/src")

ENUMERATE_PROMPT = (
    "You are the ENUMERATE step of a Turkish legal-reference extractor over tax rulings "
    "(ozelgeler).\n\n"
    "List every span in the document that mentions a law article (madde), paragraph (fikra), "
    "or subparagraph (bent). Be exhaustive: include bare 'N inci/uncu maddesi', anaphoric "
    "'ayni maddenin ... fikrasi', and 'gecici N nci maddesi'. Over-inclusion is fine; skip "
    "nothing.\n\n"
    "Output format: ONE span per line, copied verbatim from the document, nothing else. "
    "No numbering commentary, no explanations, no JSON. If the document has no such spans, "
    "output nothing.\n"
)

RESOLVE_FILTER_PROMPT = (
    "You are the RESOLVE+FILTER step of a Turkish legal-reference extractor. You are given a "
    "document and a list of candidate spans enumerated from it.\n\n"
    "For each candidate span, using the document for context, produce a JSON reference object "
    "with exactly these string fields: kanun_no, kanun_ad, madde, fikra, bent, source_text "
    "(empty string when a field is absent). Resolve elided or anaphoric law names from the "
    "nearest supporting context; never invent a law identity or evidence.\n"
    "FILTER: drop a candidate only if it is not a genuine statutory article reference (e.g. a "
    "non-law mention or a duplicate you already emitted).\n\n"
    "Output only a single JSON array of the reference objects, on one line prefixed with "
    "'FINAL_JSON:'. No prose. If nothing qualifies, output 'FINAL_JSON: []'.\n"
)


def parse_enumeration(text):
    """Split Call-1's terse span list into spans: strip list markers/numbering and quotes,
    drop blank lines."""
    spans = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        s = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", s).strip()
        s = s.strip("\"'`").strip()
        if s:
            spans.append(s)
    return spans


def compact_via_normalization(refs):
    """Deterministic precision net: dedup + generic-law-only suppression via the canonical
    normalization used by the evaluator. Falls back to the input on any normalization error."""
    if str(_DQCHECK_SRC) not in sys.path:
        sys.path.insert(0, str(_DQCHECK_SRC))
    try:
        from data_quality_checker.normalization import compact_references
        return compact_references(refs)
    except Exception:
        return refs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd Fine-Tuning/experiments/2026-07-25_cot_enum_probe && /Users/student2/ner-project/data-quality-checker-weak-learning-program/.venv-dqcheck/bin/python -m pytest test_prompts.py -q`
Expected: PASS (all — 9 prior + 6 new = 15).

- [ ] **Step 5: Commit**

```bash
git add Fine-Tuning/experiments/2026-07-25_cot_enum_probe/prompts.py Fine-Tuning/experiments/2026-07-25_cot_enum_probe/test_prompts.py
git commit -m "feat(cot-probe): two-call prompts + enumeration parser + normalization dedup net"
```

---

### Task 2: Two-call resumable driver

**Files:**
- Create: `Fine-Tuning/experiments/2026-07-25_cot_enum_probe/run_two_call.py`

**Interfaces:**
- Consumes: `prompts.ENUMERATE_PROMPT`, `prompts.RESOLVE_FILTER_PROMPT`, `prompts.parse_enumeration`, `prompts.parse_final_json`, `prompts.to_prediction_record`, `prompts.compact_via_normalization`.
- Produces: `python run_two_call.py` that writes `arms/two_call_cot/records/doc_<id>.json` (cache) and, at 50/50, `arms/two_call_cot/predictions.json`.

- [ ] **Step 1: Write `run_two_call.py`**

```python
"""Two-call terse CoT pipeline, zero-shot base model. Resumable per-doc cache."""
from __future__ import annotations
import json
from pathlib import Path

from prompts import (
    ENUMERATE_PROMPT, RESOLVE_FILTER_PROMPT, parse_enumeration,
    parse_final_json, to_prediction_record, compact_via_normalization,
)

REPO = Path("/Users/student2/ner-project")
DATA = REPO / "data/sensitive/data_quality_checker/g0/dqcheck_g0_qwen3_5_9b_955f5a14e275/data"
MODEL = ("/opt/llm-lab/hf-cache/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/"
         "snapshots/938d8919941c6e7efd3c7150eff7fe9d12afa631")
BASE = Path(__file__).resolve().parent
ARM = BASE / "arms" / "two_call_cot"
ENUM_TOKENS, RESOLVE_TOKENS = 2048, 3072


def load_docs():
    rows = [json.loads(l) for l in (DATA / "valid.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    doc_ids = json.loads((DATA / "valid_doc_ids.json").read_text(encoding="utf-8"))
    assert len(rows) == len(doc_ids)
    return [(int(d), r["messages"][1]["content"]) for d, r in zip(doc_ids, rows)]


def main():
    records_dir = ARM / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    docs = load_docs()
    pending = [(d, t) for d, t in docs if not (records_dir / f"doc_{d}.json").exists()]
    print(f"[two_call] {len(docs)-len(pending)}/{len(docs)} cached; {len(pending)} to run", flush=True)

    if pending:
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
        model, tokenizer = load(MODEL)
        sampler = make_sampler(temp=0.0)

        def generate(system, user, max_tokens):
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                add_generation_prompt=True, enable_thinking=False, tokenize=False,
            )
            return "".join(r.text for r in stream_generate(
                model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler))

        for doc_id, text in pending:
            enum_raw = generate(ENUMERATE_PROMPT, text, ENUM_TOKENS)
            spans = parse_enumeration(enum_raw)
            span_block = "\n".join(f"{i+1}. {s}" for i, s in enumerate(spans))
            user2 = f"{text}\n\n--- Candidate spans to resolve ---\n{span_block}"
            resolve_raw = generate(RESOLVE_FILTER_PROMPT, user2, RESOLVE_TOKENS)
            refs, ok = parse_final_json(resolve_raw)
            refs = compact_via_normalization(refs) if ok else refs
            record = to_prediction_record(doc_id, refs, ok)
            record["n_spans"] = len(spans)
            record["enum_raw"] = enum_raw
            record["resolve_raw"] = resolve_raw
            (records_dir / f"doc_{doc_id}.json").write_text(
                json.dumps(record, ensure_ascii=False), encoding="utf-8")
            print(f"[two_call] doc {doc_id}: spans={len(spans)} parse_ok={ok} refs={len(refs)}", flush=True)

    ids = [d for d, _ in docs]
    if all((records_dir / f"doc_{d}.json").exists() for d in ids):
        preds = []
        for d in ids:
            rec = json.loads((records_dir / f"doc_{d}.json").read_text(encoding="utf-8"))
            preds.append({k: rec[k] for k in ("doc_id", "status", "references")})
        (ARM / "predictions.json").write_text(json.dumps(preds, ensure_ascii=False), encoding="utf-8")
        parsed = sum(1 for d in ids if json.loads((records_dir/f'doc_{d}.json').read_text())["status"] == "success")
        print(f"[two_call] predictions.json written: {len(preds)} docs, parse_ok {parsed}/{len(preds)}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check (no GPU run here — the controller runs the GPU loop)**

Run: `/opt/llm-lab/.venv/bin/python -c "import ast; ast.parse(open('/Users/student2/ner-project/Fine-Tuning/experiments/2026-07-25_cot_enum_probe/run_two_call.py').read())"`
Expected: exit 0.

- [ ] **Step 3: Commit**

```bash
git add Fine-Tuning/experiments/2026-07-25_cot_enum_probe/run_two_call.py
git commit -m "feat(cot-probe): two-call terse pipeline driver (enumerate -> resolve+filter -> dedup net)"
```

---

### Task 3: Run the arm + evaluate + 3-arm comparison

**Files:**
- Create (output): `Fine-Tuning/experiments/2026-07-25_cot_enum_probe/arms/two_call_cot/{predictions,evaluation}.json`
- Modify: `Fine-Tuning/experiments/2026-07-25_cot_enum_probe/COMPARISON.md` (append a two-call section)

- [ ] **Step 1: Drive the arm to 50/50 (GPU retry-loop; controller runs this)**

```bash
cd /Users/student2/ner-project/Fine-Tuning/experiments/2026-07-25_cot_enum_probe
for attempt in $(seq 1 60); do
  [ -f arms/two_call_cot/predictions.json ] && break
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /opt/llm-lab/.venv/bin/python run_two_call.py || true
  sleep 8
done
```
Expected: `arms/two_call_cot/predictions.json` exists (50 docs). Note the printed `parse_ok N/50` — expect ≈ 50/50 (truncation eliminated).

- [ ] **Step 2: Evaluate the arm**

```bash
cd /Users/student2/ner-project
D=Fine-Tuning/experiments/2026-07-25_cot_enum_probe
PYTHONPATH=/Users/student2/ner-project /opt/llm-lab/.venv/bin/python benchmark/reference/evaluate.py \
  --predictions "$D/arms/two_call_cot/predictions.json" \
  --ground-truth-dir data/ground_truth/gt_v3_triangulated_2026-05-15/validated \
  --doc-ids-file data/sensitive/data_quality_checker/g0/dqcheck_g0_qwen3_5_9b_955f5a14e275/data/valid_doc_ids.json \
  --json-report "$D/arms/two_call_cot/evaluation.json" \
  --gt-mode approved_only --docwise-threshold 1.0 \
  --reference-postprocess canonical_set --core-reference-view row --per-doc
```
Expected: `arms/two_call_cot/evaluation.json` written.

- [ ] **Step 3: Emit the 3-arm comparison**

```bash
cd /Users/student2/ner-project
/opt/llm-lab/.venv/bin/python - <<'PY'
import json
B="Fine-Tuning/experiments/2026-07-25_cot_enum_probe/arms"
def row(tag, path):
    r=json.load(open(path))["results"][0]; c=r["core_law_article_strict"]; dw=r["docwise_core_accuracy"]
    return f"| {tag} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} | {c['tp']}/{c['fp']}/{c['fn']} | {dw['passed_doc_count']}/50 |"
print("| arm | P | R | F1 | tp/fp/fn | docwise |")
print("|---|---|---|---|---|---|")
print(row("base+plain", f"{B}/plain/evaluation.json"))
print(row("base+cot_enum (single-call, Study V)", f"{B}/cot_enum/evaluation.json"))
print(row("base+two_call_cot", f"{B}/two_call_cot/evaluation.json"))
print("| fine-tuned warm42@75 (reference) | 0.895 | 0.735 | 0.807 | 239/28/86 | 10/50 |")
PY
```
Expected: a 4-row markdown table. **Verdict rule:** the pipeline works iff two_call_cot recall ≥ plain recall AND precision not materially below plain AND parse ≈ 50/50.

- [ ] **Step 4: Append verdict to COMPARISON.md and commit**

Append a "## Two-call pipeline" section to `COMPARISON.md` with the table and a one-paragraph verdict (recall vs plain? precision held? parse rate? beats single-call CoT? gap to fine-tuned?).

```bash
git add Fine-Tuning/experiments/2026-07-25_cot_enum_probe/arms/two_call_cot/predictions.json \
        Fine-Tuning/experiments/2026-07-25_cot_enum_probe/arms/two_call_cot/evaluation.json \
        Fine-Tuning/experiments/2026-07-25_cot_enum_probe/COMPARISON.md
git commit -m "feat(cot-probe): two-call arm results + 3-arm comparison"
git push baran HEAD
```

---

### Task 4 (deferred, do NOT skip at the end): Reporting

After Task 3's verdict (the user asked to report **after** the run):

- [ ] Add a "Study VI — two-call CoT pipeline" section to `Fine-Tuning/reports/2026-07-25_g0_finetuning_recall_study/REPORT.md` (4-arm table + verdict; whether the truncation/precision fixes made CoT viable).
- [ ] Add an `EXPERIMENT_LEDGER.md` entry (next free 2.x) with the decision (viable recall mechanism → integration design; or CoT exhausted → data scale-up).
- [ ] Commit + `git push baran HEAD`.

---

## Self-Review

**Spec coverage:** Call 1 enumerate (ENUMERATE_PROMPT, Task 1) ✓; Call 2 resolve+filter (RESOLVE_FILTER_PROMPT + run_two_call, Tasks 1–2) ✓; code dedup net (compact_via_normalization reusing compact_references, Task 1) ✓; two-call bounded driver with cache/resume (Task 2 + Task 3 loop) ✓; validation-50 + canonical evaluator flags (Global Constraints + Task 3) ✓; 3-arm comparison + parse-rate success signal (Task 3) ✓; reporting deferred (Task 4) ✓; out-of-scope (no g0.py/contract/sealed-test change) respected.

**Placeholder scan:** all code complete; only `COMPARISON.md` prose is a human-judgment artifact with explicit content instructions. No forbidden placeholders.

**Type consistency:** `parse_enumeration -> list[str]` feeds the Call-2 span block; `parse_final_json -> (list[dict], bool)` and `compact_via_normalization(list[dict]) -> list[dict]` compose in the driver; `to_prediction_record(doc_id, refs, ok)` and the `{doc_id,status,references}` predictions schema match the evaluator input used by the plain/cot_enum arms.
