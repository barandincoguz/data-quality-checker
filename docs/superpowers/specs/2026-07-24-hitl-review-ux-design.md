# HITL Review UX — A/B Diff, Queue Navigation, Structured Editor

Date: 2026-07-24
Status: approved (user delegated autonomous execution)
Scope: `data_quality_checker.hitl` review UI only. No change to the review
server contract, blind A/B protocol, auth/CSRF, or storage.

## Problem

The single-file HITL review UI (`hitl.py`, `REVIEW_TEMPLATE`) shows the two
blind candidate reference sets as raw JSON blobs and offers revision only
through a free-form JSON textarea. A reviewer of Turkish legal references must
eyeball two JSON documents to spot disagreements and hand-edit JSON to revise —
slow and error-prone. There is no progress indicator and no keyboard flow.

## Goals (user-selected)

1. **A/B diff table** — field-level comparison of the two blind candidates.
2. **Queue navigation + shortcuts** — progress indicator, keyboard actions,
   auto-advance (auto-advance already exists via `next_review` redirect).
3. **Structured revision editor** — per-field reference rows instead of raw JSON.

Out of scope (deliberately deferred): evidence highlighting in the document
text (the `evidence_spans` helper stays computed-but-unused for now).

## Approach: server-render + progressive-enhancement JS

Keep the minimal single-file Flask style. Push logic into pure, unit-tested
functions; keep the browser layer thin and degradable.

### 1. Pure functions (testable, no Flask)

- `ab_diff(candidate_a, candidate_b) -> list[DiffRow]`
  - Group refs by `core_identity` (reuse `normalization.core_identity`),
    preserving first-appearance order across A then B.
  - Per group: `status` in `{"same","only_a","only_b","differs"}`.
    `same` when the `full_identity` multiset of A and B match; `only_a`/`only_b`
    when the group exists on one side; else `differs`.
  - When `differs` and the group is 1-to-1, emit `field_diffs`: the subset of
    `{fikra,bent,source_text}` whose values differ (core fields are equal by
    construction).
  - Each row carries the underlying `a`/`b` reference lists for rendering.
- `review_position(queue, internal_doc_id) -> {"index": int, "total": int} | None`
  - 1-based index of the doc within the ordered review queue; `None` if absent.

### 2. Template (`REVIEW_TEMPLATE`)

- Header shows `index/total · bucket`.
- **Diff table**: one row per `ab_diff` group; A and B columns; colour by
  `status` (same=neutral, only_a/only_b=side colour, differs=warn); differing
  fields marked. Raw A/B JSON preserved inside a collapsed `<details>`.
- **Structured editor**: an HTML table of reference rows with inputs
  (`kanun_no, kanun_ad, madde, fikra, bent, source_text`), plus
  "satır ekle / sil" and "A'dan / B'den / Judge'dan doldur" buttons. Vanilla
  inline JS manages rows and, on the `revise` submit, serialises rows into the
  existing `references_json` hidden field — the server contract is unchanged and
  `validate_final_references` still guards it.
- **No-JS fallback**: the current raw `references_json` textarea remains inside a
  `<details>`, so revision still works with JS disabled.
- **Keyboard shortcuts**: inline JS `keydown` — `a`=accept A, `b`=accept B,
  `d`=defer — ignored while typing in an `input`/`textarea`/`select`.
- Candidate A/B and judge suggestion embedded in a
  `<script type="application/json">` block for the JS to read (no new endpoint).

### 3. Route

`review_document` computes the queue once, derives `review_position`, and passes
it to the template alongside the existing `_document_for_review` payload.

## Testing

- Unit tests for `ab_diff` (same / only_a / only_b / differs + field_diffs) and
  `review_position` (present / absent / ordering).
- Flask-client tests asserting the review page renders the diff table, the
  progress indicator, and the structured-editor inputs; and that a revise POST
  built the way the editor serialises still finalises (guards the contract).
- Existing `test_hitl.py` behaviour (auth, CSRF, blinding, versioning,
  escalation) must stay green.

## Non-goals / invariants

- No change to blind mapping, auth, CSRF, optimistic versioning, or storage.
- No new dependency, no build step, no JS framework.
- Diff never reveals which candidate is human vs model.
