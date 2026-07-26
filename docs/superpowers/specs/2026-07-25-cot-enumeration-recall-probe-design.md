# CoT-Enumeration Recall Probe — Design

**Date:** 2026-07-25
**Status:** approved (brainstorming complete; awaiting spec review before writing-plans)
**Owner track:** G0 fine-tuning recall study (follow-on to `Fine-Tuning/reports/2026-07-25_g0_finetuning_recall_study/`).

## Problem and motivation

The best fine-tuned checkpoint (`warm42-cos150 @ update 75`) tops out at recall ≈ 0.735 (validation-50) / 0.728 (sealed test). Error analysis showed the misses are **diffuse omissions** — the model drops ~1–2 real references across 36/50 docs, `fn_missed` 93 vs `fp_spurious` 16, with no single targetable category. A stronger recall-instruction prompt (`recall-v3`) made this **worse**: it produced *more* predictions (285 vs 271 at update 75) but *lower* recall (0.711) and precision — the extra output was false positives, not recovered misses.

Two facts frame the next step:
1. **Fine-tuning washes out prompt instructions.** The model is trained (completion-only, teacher-forced) to imitate the gold JSON regardless of the prompt's plea, so instruction-level aggression has little grip on the tuned model.
2. **"Tell it to output more" is a dead end** — it trades precision for noise, not recall.

Therefore we test a different *mechanism* — chain-of-thought **enumerate → resolve → filter** — where the aggression lives in an over-generating enumeration step and precision is protected by an explicit filter step. And we test it **zero-shot on the base model first** (cheap-before-expensive), where the prompt has full effect and no format change is needed, before considering the far larger effort of baking CoT into the fine-tuning data contract.

## Goal

Determine, cheaply and empirically, whether a CoT enumerate→resolve→filter prompt recovers genuinely-missed references (recall up **without** a precision collapse), zero-shot on the base model. The result decides whether the mechanism is worth integrating into fine-tuning.

## The mechanism: 3-step CoT prompt

The model reasons, then emits JSON:

1. **ENUMERATE (aggressive, maximally inclusive).** Scan the document start-to-end and list *every* span bearing a madde/fıkra/bent — including bare `"17 nci maddesi"`, anaphoric `"aynı maddenin ... fıkrası"`, and `"geçici 67 nci maddesi"`. Over-generation is explicitly allowed here; the objective is to skip nothing.
2. **RESOLVE.** For each enumerated span, resolve `kanun_no` / `kanun_ad` from the nearest supporting context (anaphora resolution happens here, explicitly).
3. **FILTER (precision guard).** Drop spans with no resolvable law identity or that are not genuine statutory article references; dedup by core identity `(kanun_no, kanun_ad, madde)`; suppress a generic law-only row when a specific-article row for that law exists → final JSON.

**Output contract.** After the reasoning, the last line is `FINAL_JSON: [ ... ]` (a single-line JSON array of the six-field references). The harness extracts the array following the `FINAL_JSON:` marker. This test runs through a **standalone inference script**, not the fine-tune worker (which enforces JSON-only, no-think output).

**Why this beats recall-v3:** aggression is confined to step 1 (over-generate candidates) and paid back in step 3 (filter), so it does not simply inflate false positives — it is architecturally a recall-then-precision pipeline rather than a "try harder" instruction.

## Test design

- **Model:** base `mlx-community/Qwen3.5-9B-MLX-4bit`, **no adapter**, temperature 0, full 12288-token context, `max_generation_tokens ≈ 4096` (CoT output is long).
- **Documents:** validation-50 (the sealed test-50 stays reserved for the final headline; the mechanism comparison is done on validation to keep it honest and reusable).
- **Three arms:**
  1. **base + plain prompt (recall-v2)**, zero-shot — the base model's non-CoT recall. Doubles as the base measurement needed for `assert_test_improves_base`.
  2. **base + CoT-enum prompt**, zero-shot — the mechanism's net effect.
  3. **fine-tuned reference** — validation-50 numbers already in hand (F1 0.807 / R 0.735).
- **Evaluation:** the canonical evaluator (`benchmark/reference/evaluate.py`) with the identical validation flags — `--reference-postprocess canonical_set`, `--core-reference-view row`, `core_law_article_strict`, `--gt-mode approved_only`, `--docwise-threshold 1.0`, `--per-doc`.
- **Primary metric:** `core_law_article_strict` recall and F1; watch precision and predicted-reference count to detect the recall-v3 failure mode (more output, no recall gain).

## Success criterion and decision

- **Mechanism works** iff arm 2 recall > arm 1 recall (CoT recovers misses zero-shot) **and** precision does not collapse (arm 2 is not merely more predictions at lower precision). Bonus signal: arm 2 approaches or exceeds the fine-tuned recall (arm 3).
- **If it works:** open a *separate* design to bake enumerate→resolve→filter into the fine-tuning data contract (a two-stage output format / gold reformatting + fresh run + evaluator alignment) — explicitly out of scope here.
- **If it does not:** the prompt/CoT lever is confirmed exhausted; training-data scale-up (the planned ~2000 documents) is the only remaining recall lever. Either outcome is a publishable result.

## Components (for the implementation plan)

1. **CoT-enum prompt text** — a new prompt string (not committed into `g0.py`'s `SYSTEM_PROMPT`, since this is a standalone probe, not a training prompt).
2. **Standalone zero-shot inference script** — loads the base model once, iterates the 50 validation docs, runs each arm (plain / CoT-enum), parses `FINAL_JSON:`, writes predictions in the evaluator's expected format (`[{doc_id, references[...]}]`). Robust to a missing/blank `FINAL_JSON` line (treat as `[]`, count as a parse issue).
3. **Evaluation + comparison** — run the canonical evaluator on each arm's predictions; emit a 3-arm comparison table (P/R/F1/docwise/#pred).
4. **GPU-contention resilience** — the sealed-test run was intermittently SIGKILLed by another user's concurrent MLX jobs on the shared 96 GB machine; the script must cache per-doc records and resume, so repeated invocations accumulate progress.

## Out of scope

- Baking CoT into fine-tuning (separate spec, only if the probe succeeds).
- Any change to `g0.py`'s training `SYSTEM_PROMPT` or the fine-tune contract.
- Reopening the sealed test-50.

## Deferred (do NOT forget)

- **Reporting.** After the probe runs, update `Fine-Tuning/reports/2026-07-25_g0_finetuning_recall_study/` with a "Study V — CoT-enumeration zero-shot probe" section (3-arm table + verdict) and add an EXPERIMENT_LEDGER entry. The user explicitly asked to remember reporting **after** the experiment, not now.

## Reproducibility / evidence

- Base model snapshot: the pinned `Qwen3.5-9B-MLX-4bit` snapshot already local (offline-capable).
- Validation docs + gold: the run's `data/valid.jsonl` / `valid_doc_ids.json` and canonical GT.
- Outputs land under a new `Fine-Tuning/runs/.../cot_probe/` (or a dated `artifacts/` dir), with per-arm `predictions.json` + `evaluation.json`.
