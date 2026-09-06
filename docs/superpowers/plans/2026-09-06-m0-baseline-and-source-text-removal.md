# M₀ Baseline and `source_text` Removal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Take `source_text` out of every decision the pipeline makes, and train the loop's real baseline `M₀` on the 300-document canonical train split.

**Why now:** two findings converged. `source_text` is annotator-authored prose — the scholarship annotators cut and pasted from different parts of the ruling — so it cannot be a comparison criterion, a routing signal, or a training label. Separately, the round-1 batch was predicted by the deployed G0, which was refit on **all 494** canonical documents and has therefore seen every one of the 150 frozen test documents. `DESIGN.md` already requires otherwise: `M₀` is trained on 300, and `batch#1` is cleaned by `M₀`.

**Tech Stack:** MLX/Qwen3.5-9B LoRA, existing `g0.py` compute path.

## Global Constraints

- `source_text` stays in the database and in the HITL display. It is the project owner's material for annotator error analysis. It must not reach the router, the model's prompt or target, the judge's contract, or any identity comparison.
- The judge's own `evidence` array stays and stays validated. That is the judge quoting the document to justify itself — a hallucination guard on a machine's own output, not a use of annotator prose.
- `M₀` trains on the 300-document train split only (`loop_split`, seed `20260902`). The 50 validation documents select the checkpoint. The 150 test documents are never touched until `measured`.
- The deployed G0 (`refit-cos1003`) is not deleted and not reused as a round predictor. It remains the historical artifact behind published numbers.
- Interpreter `.venv/bin/python`; never `pip install`.
- Suite is at **525 passed, 1 skipped**; leave it higher and green.

---

### Task 1: Take `source_text` out of routing

**Files:** `src/data_quality_checker/router.py`, `tests/test_router.py`

Remove the `evidence_mismatch` branch: when the annotator's and the model's reference sets agree at full identity, the document is GREEN regardless of how each side quoted the ruling. `_evidence_compatible` and its helpers go with it if nothing else uses them.

**Measured effect on the round-1 batch:** 10 of 100 documents leave YELLOW for GREEN. Across the 23 reference pairs whose spans differed, at least one side's span was verbatim in the document in all 23 — there was nothing for an expert to decide.

**Tests:** two references identical in all five fields but with different `source_text` route GREEN. A `fikra`/`bent` difference still routes YELLOW, and a core difference still routes RED.

---

### Task 2: Take `source_text` out of the model contract

**Files:** `src/data_quality_checker/g0.py`, `src/data_quality_checker/context_windows.py`, their tests

`SYSTEM_PROMPT` lists the fields the model must emit; drop `source_text` from that list and from the few-shot examples. The assistant target in `context_windows.py` serialises the reference dicts directly — strip `source_text` before serialising.

`context_windows` also uses `source_text` to locate a reference inside the ruling when it builds a training window. **That use stays.** It is preprocessing that decides which passage to show, not a label the model learns.

**Note the coupling:** `SYSTEM_PROMPT` is one constant shared by training (`g0.py:247`) and inference (`processing.py:142`), and its sha256 is sealed into the G0 registry. Changing it makes the deployed G0 off-distribution, which is why Task 4 trains a new model rather than reusing it.

**Tests:** a built training row's assistant target parses to references with no `source_text` key; the system prompt does not name the field.

---

### Task 3: Take `source_text` out of the judge contract

**Files:** `src/data_quality_checker/judges.py`, `tests/test_judges.py`

`_validate_judge_result` requires every `final_references` entry to carry a `source_text` that occurs verbatim in the ruling. Remove that requirement and the prompt sentences that demand it. Keep the `evidence` array and keep validating that each of its spans occurs in the ruling.

**Why this matters concretely:** round 1's judge refused `d000081` three times because it adopted the annotator's span, which was stitched from two places in the ruling. The annotator's *reference* was correct — the model found the same provision. The contract rejected a right answer over annotator prose.

**Tests:** a judge answer whose `final_references` carry no `source_text` validates; one whose `evidence` contains a span absent from the ruling is still rejected.

---

### Task 4: Write the loop split, and train `M₀`

**Files:** `src/data_quality_checker/loop_train.py`, `loop_bindings.py`, tests

Write `build_loop_split`'s output as a sealed artifact so every later round reads one split rather than recomputing it.

Then make `train_round` able to launch compute instead of raising, and train `M₀` on the 300 train documents, selecting the checkpoint on the 50 validation documents with the existing contract-driven `select_checkpoint`. The 150 test documents stay untouched.

**Tests:** the training data contains exactly the 300 train document ids and none from validation or test; `execute=True` reaches the compute path; the selection record still refuses a checkpoint that is not eligible.

---

### Task 5: Re-run round 1 on `M₀`

No code. Re-predict the M01 batch with `M₀`, re-route, re-judge, and record the round-1 numbers that will actually appear in the paper. The batch itself does not change — the same 100 documents, the same manifest, the same seed.
