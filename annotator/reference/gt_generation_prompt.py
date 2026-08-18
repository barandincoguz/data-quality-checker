"""GT v2 prompt builder.

Explicitly derived from the benchmark prompt
``annotator.reference.llm_prompt.build_system_prompt("few-shot-cot-v3-en")``.

The GT v2 prompt:

- Reuses every benchmark **extraction rule** verbatim (role+schema,
  baseline rules, negative-guard rules, recall-recovery rules, final-lock
  rules, negative mini-examples, hidden checklists, all 11 curated
  few-shot examples).
- Adds four GT-only blocks on top: stricter source-text policy + audit-
  tagging rules, raw audit-field contract, raw output contract, and a
  GT-only final silent checklist.
- Replaces only the benchmark output contract (because the raw GT output
  carries audit fields ``confidence`` / ``evidence_type`` / ``reasoning``
  that the benchmark does not).
- Does NOT instruct the model to emit pipeline-managed metadata fields
  (model id, prompt hash, input hash, json-repaired flag, raw response
  excerpt). Those are added by trusted pipeline code, never by the model.

The final validated GT downstream of this raw output strips the audit
fields so the validated rows are evaluator-compatible.

CLI:

    venv/bin/python -m annotator.reference.gt_generation_prompt --render
    venv/bin/python -m annotator.reference.gt_generation_prompt --length

See ``docs/superpowers/specs/2026-04-30-gt-v2-design.md`` §§1, 5.1, 5.3.
"""

from __future__ import annotations

import argparse

from annotator.reference.llm_prompt import (
    ENGLISH_BASELINE_RULES_BLOCK,
    ENGLISH_FINAL_CHECKLIST_BLOCK,
    ENGLISH_FINAL_LOCK_RULES_BLOCK,
    ENGLISH_NEGATIVE_GUARD_RULES_BLOCK,
    ENGLISH_NEGATIVE_MINI_EXAMPLES_BLOCK,
    ENGLISH_RECALL_RECOVERY_CHECKLIST_BLOCK,
    ENGLISH_RECALL_RECOVERY_RULES_BLOCK,
    ENGLISH_ROLE_AND_SCHEMA_BLOCK,
    FEW_SHOT_COT_V3_EN_PROMPT_VARIANT,
    _assemble_prompt,
    _build_examples_block,
    _load_prompt_examples,
)

GT_PROMPT_VERSION = "gt-generation-v1"


# ── GT-only blocks ──────────────────────────────────────────────────────


GT_GENERATION_STRICTER_BLOCK = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## GROUND-TRUTH-ONLY STRICTER RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This is a research-grade ground-truth generation task, not a benchmark
prediction. Output will be audited row by row, including by humans.
Apply the following rules in addition to every benchmark rule above.

GT-S1 Source-text policy is STRICT.
  - source_text MUST be a verbatim substring of the input document.
  - No ellipses, no paraphrase, no reconstructed span, no edits.
  - Target ≤ 120 characters. If the only natural span is longer,
    shorten by trimming the trailing edge only — never edit the
    middle. The validator does not auto-reject on length, but rows
    with source_text > 120 chars will be flagged for review.
  - If you cannot produce a verbatim span, set source_text="" and
    set confidence="low".

GT-S2 Empty-field bias.
  - When in doubt, leave kanun_no or kanun_ad empty rather than
    guessing. Hallucinated identifiers are the most expensive class
    of error. Empty fields are auditable; wrong fields are not.

GT-S3 Local-anaphora audit tagging.
  - Any reference whose law identity (kanun_no / kanun_ad) was
    resolved from "Aynı Kanunun", "anılan Kanun", "bu Kanunun", or any
    other same-law cue rather than from an explicit law mention in
    the same span MUST carry "local_anaphora" in evidence_type.
    The pipeline forces human review on every such row, regardless
    of your confidence.

GT-S4 Amendment-chain audit tagging.
  - When the source pattern is "<X> sayılı Kanunun <Y> inci maddesi
    ile <HOST> ... eklenen (W) numaralı bent" (or equivalent),
    emit TWO references:
      (a) AMENDING_LAW / madde=Y                evidence_type=["amendment_chain"]
      (b) HOST_LAW / host_madde / host_fikra / W  evidence_type=["amendment_chain"]
    Both rows MUST carry "amendment_chain" so the validator forces
    review on both. Never attach the host's bent/fıkra to the
    amending law.

GT-S5 Footer policy is explicit.
  - KEEP the standard VUK 413 özelge footer:
      "(*) Bu Özelge 213 sayılı Vergi Usul Kanununun 413. maddesine
       dayanılarak verilmiştir."
    Emit it as a normal kanun_madde_referansi with
    evidence_type=["footer"].
  - DROP the 5070 Elektronik İmza Kanunu signature attestation
    boilerplate (e.g. "5070 sayılı Elektronik İmza Kanunu'nun 5.
    Maddesi gereğince güvenli elektronik imza..."). This is a
    document-signing artifact, not a substantive legal reference of
    the ruling. Do NOT emit a reference for it.

GT-S6 alt_bent policy.
  - There is no alt_bent field in the schema and one will not be
    added. The final validated GT schema is fixed at
    (type, kanun_no, kanun_ad, madde, fikra, bent, source_text).
  - If the document explicitly references a sub-subparagraph
    (e.g. "(2) numaralı bendin (b) alt bendi"), store the most
    specific supported value in `bent` (here `bent="b"`) and add
    "uncertain" to evidence_type so the validator flags the row
    for human review.

GT-S7 Other audit-tagging rules.
  - Tag tablo / cetvel / liste references with
    evidence_type=["table_reference"].
  - Tag collapsed article ranges
    (e.g. "1 ila 24 üncü maddeleri" → ONE row with madde="1 ila 24")
    with evidence_type=["range"].
  - Tag rows from explicit enumerations
    (e.g. one of the rows from "61, 94, 103 ve 104 üncü maddeleri")
    with evidence_type=["enumeration"].
  - Tag generic law-only rows (only allowed when no specific tuple
    exists for the same law in the document) with
    evidence_type=["generic_law_only"].
  - Tag canonical abbreviation resolutions (KDV → 3065, VUK → 213,
    GVK → 193, KVK → 5520, ÖTV → 4760, DVK → 488) with
    evidence_type=["abbreviation"].
  - Tag fully explicit references (no anaphora, no abbreviation,
    no amendment chain) with evidence_type=["explicit"].
  - A row may carry MULTIPLE evidence_type values when more than one
    applies (e.g. ["abbreviation", "local_anaphora"]).
  - Use evidence_type=["uncertain"] only when none of the above fits
    or when you are not confident enough to assert a specific tag."""


GT_GENERATION_AUDIT_FIELDS_BLOCK = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## RAW AUDIT FIELDS (per-row, mandatory)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Every reference object in the raw output MUST carry these three
audit fields in addition to the benchmark schema fields:

  confidence
    One of "high" / "medium" / "low".
      high   = explicit reference, complete fields, verbatim span,
               no anaphora resolution required, no abbreviation
               guesswork.
      medium = required mild abbreviation resolution OR mild local
               anaphora OR a long source_text (≥ 80 chars);
               otherwise unambiguous.
      low    = empty law field that you couldn't safely fill,
               ambiguous anaphora, amendment-chain target uncertainty,
               non-verbatim source_text candidate, suspected
               secondary-regulation leak, range/enumeration ambiguity,
               or any other low-evidence case.

  evidence_type
    A non-empty JSON list. Allowed values:
      explicit, abbreviation, local_anaphora, footer,
      amendment_chain, table_reference, range, enumeration,
      generic_law_only, uncertain.
    Multiple values are allowed and encouraged when more than one
    applies. At least one value is required per row.

  reasoning
    One short sentence in English explaining how this row was
    extracted. Mention which evidence type was used and any
    non-trivial choice (anaphora resolution, abbreviation expansion,
    amendment-chain split, range collapse, enumeration expansion).
    Keep this concise.

Confidence semantics are DESCRIPTIVE, not prescriptive. A row tagged
amendment_chain or local_anaphora MAY still be confidence="high" if
you have strong evidence — the validator will flag those rows for
review regardless of confidence, because they belong to high-risk
classes.

Pipeline-managed fields. The following metadata fields are added
later by trusted pipeline code, NOT by you. Do NOT emit any of
them in your output:
  - model
  - prompt_version
  - prompt_hash
  - input_hash
  - json_repaired
  - raw_response_excerpt"""


GT_GENERATION_OUTPUT_CONTRACT_BLOCK = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## GT-V1 OUTPUT CONTRACT (overrides any earlier output contract)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return JSON only. No markdown fences, no prose, no explanations
outside the JSON object. No leading or trailing text. The schema:

{
  "doc_id": <int|null>,
  "ozelge_no": "<string>",
  "references": [
    {
      "type": "kanun_madde_referansi",
      "kanun_no": "213",
      "kanun_ad": "Vergi Usul Kanunu",
      "madde": "mükerrer 298",
      "fikra": "A",
      "bent": "1",
      "source_text": "213 sayılı Vergi Usul Kanununun mükerrer 298 inci maddesinin (A) fıkrasının (1) numaralı bendinde",
      "confidence": "high",
      "evidence_type": ["explicit"],
      "reasoning": "Explicit law name, article, paragraph, and bent in a single span."
    }
  ]
}

This schema SUPERSEDES any earlier output contract. Do not emit the
benchmark output contract format. Do not emit the LangExtract wrapper
format.

The validated benchmark GT downstream of this raw output will strip
confidence, evidence_type, and reasoning so each row matches the
project's evaluator schema (type, kanun_no, kanun_ad, madde, fikra,
bent, source_text). Your job here is the audit-rich raw output."""


GT_GENERATION_FINAL_CHECKLIST_BLOCK = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## GT-V1 FINAL SILENT CHECKLIST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

In addition to the benchmark final checklists above, silently confirm
all of the following before returning:

1. Every source_text is a verbatim substring of the input document.
2. No source_text contains "..." or paraphrased text.
3. Every row has a non-empty evidence_type list.
4. Every confidence value is one of: high, medium, low.
5. Every unambiguous canonical tax-law abbreviation has both
   kanun_no and kanun_ad filled
   (213/Vergi Usul Kanunu, 193/Gelir Vergisi Kanunu,
    5520/Kurumlar Vergisi Kanunu, 3065/Katma Değer Vergisi Kanunu,
    4760/Özel Tüketim Vergisi Kanunu, 488/Damga Vergisi Kanunu).
6. Every amendment chain emitted exactly two rows
   (amending + host), both tagged "amendment_chain".
7. The 5070 Elektronik İmza Kanunu signature boilerplate is NOT in
   the output.
8. The standard VUK 413 footer IS in the output exactly once if the
   document contains it, tagged "footer".
9. No quoted-article list item became a fake reference.
10. No standalone secondary-regulation object
    (Tebliğ, Yönetmelik, Karar, Genelge, Sirküler, Tarife, KHK).
11. No two rows in the output have the same
    (kanun_no, kanun_ad, madde, fikra, bent) tuple.
12. JSON only, schema-valid, no trailing commas, no markdown fences.
13. No pipeline-managed metadata fields are emitted (model,
    prompt_version, prompt_hash, input_hash, json_repaired,
    raw_response_excerpt)."""


# ── Builder ────────────────────────────────────────────────────────────


def _benchmark_variant_rules_block() -> str:
    """Return the v3-en variant rules block, assembled from the same
    constants the benchmark uses. Mirrors
    ``llm_prompt._build_variant_rules_block("few-shot-cot-v3-en")``.
    """
    return _assemble_prompt(
        ENGLISH_BASELINE_RULES_BLOCK,
        ENGLISH_NEGATIVE_GUARD_RULES_BLOCK,
        ENGLISH_RECALL_RECOVERY_RULES_BLOCK,
        ENGLISH_FINAL_LOCK_RULES_BLOCK,
        ENGLISH_NEGATIVE_MINI_EXAMPLES_BLOCK,
    )


def _benchmark_reasoning_block() -> str:
    """Return the v3-en reasoning block. Mirrors
    ``llm_prompt._build_reasoning_block("few-shot-cot-v3-en")``.
    """
    return _assemble_prompt(
        ENGLISH_RECALL_RECOVERY_CHECKLIST_BLOCK,
        ENGLISH_FINAL_CHECKLIST_BLOCK,
    )


def _benchmark_examples_block() -> str:
    """Return the v3-en curated few-shot examples block. Mirrors
    ``llm_prompt._build_examples_block`` for the v3-en variant.
    """
    return _build_examples_block(
        FEW_SHOT_COT_V3_EN_PROMPT_VARIANT,
        _load_prompt_examples(FEW_SHOT_COT_V3_EN_PROMPT_VARIANT),
    )


def build_gt_generation_prompt() -> str:
    """Assemble the GT v2 prompt deterministically.

    Block order (mirrors ``llm_prompt.build_system_prompt`` for v3-en
    plus the four GT-only blocks):

    1. ENGLISH_ROLE_AND_SCHEMA_BLOCK                     (benchmark)
    2. variant rules: BASELINE + NEGATIVE_GUARD +
       RECALL_RECOVERY + FINAL_LOCK + NEGATIVE_MINI_EXAMPLES (benchmark)
    3. reasoning checklists                              (benchmark)
    4. GT_GENERATION_STRICTER_BLOCK                      (GT-only)
    5. GT_GENERATION_AUDIT_FIELDS_BLOCK                  (GT-only)
    6. curated few-shot examples (v3-en, 11 entries)     (benchmark)
    7. GT_GENERATION_OUTPUT_CONTRACT_BLOCK               (GT-only;
       supersedes benchmark output contract)
    8. GT_GENERATION_FINAL_CHECKLIST_BLOCK               (GT-only)
    """
    return _assemble_prompt(
        ENGLISH_ROLE_AND_SCHEMA_BLOCK,
        _benchmark_variant_rules_block(),
        _benchmark_reasoning_block(),
        GT_GENERATION_STRICTER_BLOCK,
        GT_GENERATION_AUDIT_FIELDS_BLOCK,
        _benchmark_examples_block(),
        GT_GENERATION_OUTPUT_CONTRACT_BLOCK,
        GT_GENERATION_FINAL_CHECKLIST_BLOCK,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the GT v2 generation prompt without calling any API."
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Print the assembled GT prompt to stdout (default action).",
    )
    parser.add_argument(
        "--length",
        action="store_true",
        help="Print only a length / line-count summary.",
    )
    args = parser.parse_args(argv)

    prompt = build_gt_generation_prompt()
    if args.length and not args.render:
        print(f"prompt_version: {GT_PROMPT_VERSION}")
        print(f"chars         : {len(prompt)}")
        print(f"lines         : {len(prompt.splitlines())}")
        return 0
    print(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
