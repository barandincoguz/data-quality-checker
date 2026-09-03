# DQ-Loop Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Emit, per round, every number the paper's claim needs — model improvement, judge quality, annotator quality, three-way error attribution, and expert workload — as one audited artifact.

**Architecture:** A new `loop_metrics.py` of pure functions over records already in the store, plus a `metrics_step` binding that writes `round_metrics.json` into the round's artifact chain. No new data collection: everything below is derivable from `documents`, `predictions`, `judge_results` and `reviews`, or from the frozen test split.

**Tech Stack:** Python stdlib, the repo's existing evaluator, SQLite store.

## Global Constraints

- **The expert is the sole truth source.** Annotator, model and judge are all *rated*; none may define correctness. A metric that scores the judge against the model, or the model against the judge, is a defect.
- Every accuracy number ships with latency p50/p95, token counts and peak memory beside it.
- The frozen 150-document test split may never influence a decision; test metrics are computed only after `checkpoint_selected`.
- No metric may be hardcoded. A constant standing in for a computation is the defect this plan exists to remove.
- Reference comparison uses the existing `core_identity` / `full_identity` and `apply_reference_policy`; do not write a second comparison.
- All rates are emitted with their numerator and denominator, never as a bare float.

---

### Task 1: Fix `fabricated_evidence_rate`, and report what the judge actually attempted

**Files:** Modify `src/data_quality_checker/judges.py`; Test `tests/test_judge_metrics.py`

**The defect:** `judges.py:982` emits `"fabricated_evidence_rate": 0.0` as a literal. It is never computed. It reads as "the judge never fabricates" when it means "outputs containing fabricated evidence were rejected upstream, so none survive to be counted."

**Where the real data lives (verified against a live run, do not assume otherwise):** the `judge_results` DB row carries only `retry_count`, `status`, `error`, `verdict`, `blind_mapping` and `response_path`. The per-attempt history is in the response JSON file named by `response_path`, whose keys are exactly:
`attempts`, `batch_id`, `blind_mapping`, `error`, `internal_doc_id`, `model`, `operational`, `result`, `schema_version`, `status`.
`attempts` is a list of `{"attempt": 1, "status": "valid"}`; a rejected attempt carries `status: "invalid"` and an `error` string beginning `fabricated_or_missing_evidence`. `operational` carries `latency_seconds`, `generation_tokens`, `finish_reason`, `model_revision` and `cost`. Read the response files; the counts are not in the database.

**Emit instead, all with explicit numerator and denominator:**
- `fabricated_evidence_attempts` / `total_attempts`
- `invalid_attempts` / `total_attempts`, split by error class
- `documents_needing_retry` / `document_count`
- `verdict_distribution` — counts for A, B, TIE, NEITHER, resolved through `blind_mapping` into `annotator` / `model` / `TIE` / `NEITHER`, because raw A/B is meaningless once the mapping is randomised
- `latency_p50_seconds`, `latency_p95_seconds` beside the existing mean

**Tests must include:** a fixture whose stored attempts contain one fabricated rejection followed by a valid retry, asserting the rate is 1/2 and not 0. Prove it discriminates by reverting to the literal and watching it fail.

---

### Task 2: Three-way error attribution and rater agreement

**Files:** Create `src/data_quality_checker/loop_metrics.py`; Test `tests/test_loop_metrics.py`

**Interfaces produced:**
```python
def attribute_errors(*, expert, annotator, model, judge) -> dict
def rater_agreement(records) -> dict
```

`attribute_errors` scores each rater against the expert's finalized references on one document, at core identity, returning per rater: `correct`, `missed` (in expert, not in rater), `spurious` (in rater, not in expert).

**Where each rater's references live (verified against the live store, do not assume otherwise):**

| rater | source | notes |
|---|---|---|
| expert (truth) | `reviews.final_references_json` | only rows whose `status` is `finalized`; the row also carries `action` and both epoch timestamps |
| annotator | `documents.human_references_json` | |
| model | `predictions.references_json` | one row per `(batch_id, internal_doc_id, generation)` |
| judge | `judge_results.result_json` -> `final_references` | only rows whose `status` is `valid`; `blind_mapping` says which side was which |

Compare with the existing `core_identity` (a 4-tuple) from `normalization.py`; apply `apply_reference_policy` to every list before comparing, exactly as `_validate_judge_result` does, so the VUK 213/413 policy is not applied to one rater and not another. Do not write a second comparison function.

**Aggregate and emit:**
- per rater: precision, recall, F1, exact-set-agreement, each with counts
- `joint_miss_count` — references the expert kept that *no* rater produced. This is the number GREEN sampling hides, and it is the honest ceiling on what the loop can find.
- `annotator_only_correct`, `model_only_correct`, `judge_only_correct` — where each rater was the sole source of a correct reference
- pairwise agreement and Cohen's kappa for each pair; Fleiss' kappa across the three

**Tests must include:** a case where all three raters miss a reference the expert kept, asserting `joint_miss_count == 1` — a metric that only counts disagreements would report 0 here.

---

### Task 3: Expert workload

**Files:** Modify `loop_metrics.py`; Test `tests/test_loop_metrics.py`

The paper's claim has two halves and this is the neglected one. Derive from `reviews`:
- `documents_requiring_review` / `document_count`. The required set is the one
  `hitl._review_requirements` actually enforces, verified at `hitl.py:375-400`:
  the GREEN **audit sample** (`ensure_green_audit_plan`'s
  `sample_internal_doc_ids`) union RED union YELLOW union escalated GREEN.
  The audit sample is required regardless of escalation. Omitting it would
  understate workload precisely where the paper's claim is strongest: as the
  model improves more documents route GREEN, but the audit sample does not
  shrink with them, so expert review has a floor the metric must show.
- `action_distribution` — the actual action set, verified in `hitl.py:789-812`, is exactly:
  `accept_human`, `accept_model`, `revise`, `defer`, `judge_override`.
  (`accept_candidate_a` / `accept_candidate_b` arrive from the blind UI and are
  normalised to `accept_human` / `accept_model` before storage, so they never
  appear in the `reviews` table.) Count each, with the total.
  These are not interchangeable for workload: `revise` means the expert wrote
  the references themselves and is the most expensive outcome; `defer` leaves
  the document unresolved and stores `final_references_json` as null, so a
  deferred row is **not** expert truth and must be excluded from every accuracy
  computation while still counting as work performed.
- `seconds_to_decision` p50/p95 from `updated_at_epoch - created_at_epoch`, reported with the caveat that it measures wall time on an open review, not attention
- `edit_distance_from_best_candidate` — how many references the expert added, removed or altered relative to whichever candidate was closest

**The workload claim is `documents_requiring_review` falling round over round while accuracy holds or rises. Both halves must appear in the same artifact.**

---

### Task 4: Model round metrics on the frozen test split

**Files:** Modify `loop_metrics.py`; Test `tests/test_loop_metrics.py`

Wrap the repo's own `canonical_evaluate`; do not reimplement scoring. Emit core and full P/R/F1, the docwise distribution the evaluator already computes (`perfect_f1_count`, `zero_f1_count`, min/mean/median/p25/p75), `parse_failure_count`, `truncation_count`, and operational latency p50/p95, tokens and peak memory.

Add `paired_delta(previous, current)`: a paired bootstrap over the shared documents returning ΔF1 with a 95% interval. **A round that reports an improvement without an interval is not evidence**; emit `inconclusive` when the interval spans zero.

**Replicate the project's existing method exactly** — it is `paired_core_bootstrap` in `ner-project/scripts/dqcheck_g0_external_selection.py:910`, which produced the only external selection number the project has published. That script lives in another repository, so reimplement rather than import, but keep every parameter identical:

- `samples = 10_000`, `seed = 42` (`random.Random(seed)`), both echoed in the output
- resample document ids **with replacement** from the shared id universe, `[rng.choice(doc_ids) for _ in doc_ids]`
- for each resample, sum `tp`/`fp`/`fn` across the sampled ids and compute F1 from the sums — never average per-document F1
- raise if the two document universes differ; a paired test over mismatched universes is not paired
- emit `lower_2_5`, `upper_97_5`, `bootstrap_mean`, `probability_delta_gt_zero`, and per-document `challenger_win` / `tie` / `incumbent_win`

Add on top of that method: `verdict`, one of `improved`, `regressed`, `inconclusive` — `inconclusive` whenever the interval spans zero, regardless of the sign of the mean.

---

### Task 5: Bind the metrics into the round artifact chain

**Files:** Modify `src/data_quality_checker/loop_bindings.py`, `loop_rounds.py`; Test `tests/test_loop_bindings.py`

Add `metrics_step(config, *, round_index, ...)` writing `round_metrics.json` under the round's directory, fingerprinted like every other step via `sha256_file`. It runs at the `measured` stage, which is already reachable only from `checkpoint_selected` — so test-split numbers cannot leak into selection. The seal must cover the metrics artifact.

**Test:** mutate one metric value and assert the round's seal changes; a seal that does not cover the metrics certifies nothing.
