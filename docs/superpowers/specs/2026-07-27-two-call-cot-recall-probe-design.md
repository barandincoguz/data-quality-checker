# Two-Call CoT Recall Probe — Design

**Date:** 2026-07-27
**Status:** approved (brainstorming complete; awaiting spec review before writing-plans)
**Owner track:** G0 fine-tuning recall study — follow-on to Study V (`Fine-Tuning/experiments/2026-07-25_cot_enum_probe/`, `Fine-Tuning/reports/2026-07-25_g0_finetuning_recall_study/` §7b).

## Problem and motivation

Study V's single-call CoT enumerate→resolve→filter probe found a real signal: on the 38 documents where it parsed, the enumerate step **raised recall 0.695 → 0.733** vs the plain prompt — the first prompt-side recall gain in the study. But it failed net for two reasons: (1) the CoT is so verbose that **24% (12/50) of outputs truncated before `FINAL_JSON:` even at 8192 tokens**, scoring 0 predictions; (2) precision dropped 0.816 → 0.744 because the in-line filter under-removed over-generated candidates. Full-50 F1 collapsed to 0.579 vs plain 0.706.

The goal is to keep the enumerate recall gain while eliminating the truncation and recovering precision.

## Goal

Test, zero-shot on the base model, a **two-call terse pipeline** that structurally removes truncation (each call is bounded) and adds a deterministic precision guard, and measure whether it recovers recall **without** a precision collapse — beating both the plain prompt and Study V's single-call CoT.

## Design: two-call terse pipeline

**Call 1 — ENUMERATE (terse, exhaustive; recall driver).**
- Input: an enumerate system prompt + the document.
- Output: a compact list, **one span per line, verbatim from the document, no prose or deliberation** — including bare `"N inci maddesi"`, anaphoric `"aynı maddenin ... fıkrası"`, and `"geçici N nci maddesi"`. Over-inclusion is explicitly allowed.
- Parse: split lines, strip list markers → `list[str]` of spans. A blank/empty result is allowed (document with no references).
- Token budget: ~2048. A terse list does not truncate the way verbose prose did.

**Call 2 — RESOLVE + FILTER (compact JSON out).**
- Input: a resolve system prompt + the document + the Call-1 span list.
- For each listed span, using the document for context: resolve `kanun_no` / `kanun_ad` (elided/anaphoric law names resolved here), `madde` / `fikra` / `bent`. **Drop a span only if it is not a genuine statutory article reference** (LLM judgment). Output: a compact JSON array of six-field references, **no prose**.
- Token budget: ~3072. JSON-only output stays bounded.
- Parse: reuse `prompts.parse_final_json` (marker-optional bare-array parser) from the Study V probe.

**Code safety net (deterministic precision guard, after Call 2).**
- Deduplicate by core identity `(kanun_no, kanun_ad, madde)` and suppress a generic law-only row when a specific-article row for that law exists — by reusing `data_quality_checker.normalization.compact_references`. This is the same `canonical_set` logic the evaluator applies, run at prediction time so precision does not depend on the LLM getting dedup right.

**Why this fixes Study V:** truncation is impossible (both calls bounded and terse); recall is preserved because Call 1 still enumerates exhaustively; precision is protected by an explicit filter (Call 2) plus a deterministic dedup net (code) rather than the single under-powered in-line filter.

## Test design

- **Model:** base `mlx-community/Qwen3.5-9B-MLX-4bit`, no adapter, temperature 0, offline. Reuse the Study V generation pattern (`mlx_lm.load` once, `stream_generate`, `make_sampler(temp=0)`, `apply_chat_template(enable_thinking=False)`), but issue two calls per document.
- **Documents:** validation-50. Reuse the Study V plain-arm predictions/evaluation as the baseline (no re-run needed); the fine-tuned checkpoint (F1 0.807 / R 0.735) is the reference.
- **Three arms:** base+plain (from Study V), **base+two_call_cot** (new), fine-tuned reference.
- **Evaluation:** the canonical evaluator with the identical validation flags (`canonical_set`, row view, `core_law_article_strict`, `approved_only`, docwise 1.0), run with `PYTHONPATH=<repo root>`.
- **Parse-reliability check:** report how many of the 50 docs completed both calls and parsed (expect ~50/50 — the truncation fix is a primary success signal).

## Success criterion and decision

- **The pipeline works** iff, on the full 50: `two_call_cot` recall ≥ plain recall (the enumerate gain survives) **and** precision is not materially below plain (the filter + dedup net hold) **and** parse rate ≈ 50/50 (truncation eliminated). Bonus: it beats Study V single-call CoT (F1 0.579) and narrows the gap to the fine-tuned reference.
- **If it works:** a viable recall-boosting inference mechanism exists; open a separate design for integrating it into the fine-tuned extractor (run the 2-call pipeline at inference, or distill its outputs into single-call fine-tune targets) — out of scope here.
- **If it does not:** the CoT lever is exhausted even in its best form; training-data scale-up (the planned ~2000 documents) remains the only recall path. Either outcome is a publishable result.

## Components (for the implementation plan)

1. **Two prompts** — `ENUMERATE_PROMPT` (terse, exhaustive span list) and `RESOLVE_FILTER_PROMPT` (spans + doc → filtered JSON) — added to the Study V `prompts.py` (or a sibling module in a new experiment dir).
2. **`parse_enumeration(text) -> list[str]`** — pure, unit-tested: split the terse list into spans, strip markers/numbering, drop blanks.
3. **`compact_via_normalization(refs) -> list[dict]`** — pure wrapper reusing `data_quality_checker.normalization.compact_references` for the code dedup net; unit-tested on a dup + generic-law-only case.
4. **`run_two_call.py`** — resumable driver: for each doc, Call 1 → parse spans → Call 2 (doc + spans) → `parse_final_json` → code dedup net → cache per-doc record; assemble `arms/two_call_cot/predictions.json` at 50/50. Per-doc cache + retry-loop for GPU-contention resume (see [[finetune-infra-gotchas]]).
5. **Evaluation + comparison** — reuse `evaluate_arms.sh` pattern (PYTHONPATH-fixed) for the new arm; emit an updated 3-arm table.

## Out of scope

- Baking the pipeline into fine-tuning (separate design if the probe succeeds).
- Any change to `g0.py`'s training prompt or the fine-tune contract.
- Reopening the sealed test-50.

## Deferred (do NOT forget)

- **Reporting.** After the probe runs, add a "Study VI — two-call CoT pipeline" section to `Fine-Tuning/reports/2026-07-25_g0_finetuning_recall_study/REPORT.md` (3-arm table + verdict) and an EXPERIMENT_LEDGER entry (next free 2.x). The user asks to remember reporting **after** the run.

## Reproducibility / evidence

- Base model snapshot + venvs + evaluator invocation: as in the Study V probe (see the fine-tune infra notes).
- Outputs under `Fine-Tuning/experiments/2026-07-27_two_call_cot_probe/` (new dir) or a new `two_call_cot` arm beside the Study V arms; per-doc records gitignored, predictions/evaluation/COMPARISON committed. Push to the `baran` remote (origin is 403).
