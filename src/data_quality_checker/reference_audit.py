"""Self-consistency audit and candidate-vs-expert triage over reference lists.

Two pure, database-free functions that encode the reference audits a human
expert would otherwise perform by hand each cleaning round:

- `audit_reference_set` checks one document's reference list against itself
  (does each span really occur in the document? does a stated law number
  disagree with its stated name? is a row duplicated? does an article's own
  number actually appear in its own span?).
- `triage_disagreement` classifies, for one document, every provision a
  candidate rater (annotator, model or judge) proposes that the expert's
  finalized truth does not have -- the triage step a reviewer performs on
  every disagreement.

Both take references in the repo's normal shape (`kanun_no`, `kanun_ad`,
`madde`, `fikra`, `bent`, `source_text`) and run them through
`reference_policy.apply_reference_policy` and `normalization.compact_references`
before anything is checked, so the shared VUK-213/article-413 policy and
ordinary normalization artefacts (whitespace, casing, alias spelling,
embedded-bent splitting, ...) are already gone before a finding is reported.
`duplicate_row` is the one exception, and is documented at its own check
below: `compact_references`'s own dedup pass collapses every row sharing a
`full_identity`, regardless of span, down to one -- so a check that needs to
tell *"two rows, same span"* (a defect) apart from *"two rows, different
spans"* (two legitimate mentions of one provision) cannot be computed from
the compacted output at all, and instead uses `normalize_reference` directly
(the same per-row normalization `compact_references` uses internally) over
the policy-kept, still-undeduplicated rows.

**`article_number_absent_from_span` never asserts an error.** Run over the
project's real 500-document ground truth this check fires 172 times, and
every single case inspected by hand was legitimate: anaphora (the article was
named earlier and this span only refers back to it -- 167 of the 172), a bent
reference under the same article (`"(2/b) bendinde"`), an amending law's own
article number, or a number the source PDF glued to the previous word with no
space (`"Kanununun267 inci maddesi"`). A naive version of this check that
reported these as mismatches produced five false alarms out of five. This
function therefore reports evidence -- the numbers actually found in the span,
sub-classified as `anaphoric` (no digits at all) or `other_number_present`
(some other digits, possibly even the same digits glued without a boundary)
-- and asserts nothing about correctness. The same restraint applies to
`triage_disagreement`'s `label_inconsistent_with_span`: it is named for what
it is (a label that does not match its own evidence), never for who is at
fault, and `candidate_addition` is explicitly named a candidate, never a
truth error -- of eight such candidates found on real data, verification
showed three were already in the ground truth (blocked only by a since-fixed
normalization defect), three were annotator mistakes, and one was genuinely
ambiguous.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .normalization import (
    CANONICAL_LAW_BY_NO,
    compact_references,
    core_identity,
    full_identity,
    normalize_reference,
)
from .reference_policy import apply_reference_policy
from .text import folded_text, normalize_text

Identity = tuple[str, str, str, str]

_PLAIN_NUMBER_RE = re.compile(r"^\d+$")
_NUMBER_RUN_RE = re.compile(r"\d+")
# A law is referred back to anaphorically -- without restating its number --
# by one of these words immediately followed by some inflected form of
# "kanun" ("kanunun", "kanununda", "kanununca", ...). Matched against
# case-folded text so casing never defeats it.
_BACK_REFERENCE_RE = re.compile(r"(aynı|anılan|mezkur|mezkûr|söz konusu|bu)\s+kanun")


def _numbers_in_span(span: str) -> list[str]:
    return _NUMBER_RUN_RE.findall(span)


def _number_present_in_span(number: str, span: str) -> bool:
    """Whether `number` occurs in `span` as its own token, not glued to a word.

    Plain substring search would call "267" present inside "Kanununun267"
    just because the digits happen to appear there; `\\b` requires a
    transition between a word character and a non-word one, and since digits
    are word characters too, a number stuck directly onto the preceding word
    has no such transition and correctly reads as absent.
    """
    if not number:
        return False
    return re.search(rf"\b{re.escape(number)}\b", span) is not None


def _refers_back_to_a_law(span: str) -> bool:
    return _BACK_REFERENCE_RE.search(folded_text(span)) is not None


def _leading_number(article: str) -> str:
    match = _NUMBER_RUN_RE.search(article)
    return match.group(0) if match else ""


def _is_grounded(source_text: str, normalized_document_text: str) -> bool:
    return bool(source_text) and source_text in normalized_document_text


def _law_number_conflict(reference: Mapping[str, str]) -> dict[str, str] | None:
    kanun_no = reference["kanun_no"]
    kanun_ad = reference["kanun_ad"]
    canonical = CANONICAL_LAW_BY_NO.get(kanun_no)
    if canonical is None or not kanun_ad or kanun_ad == canonical:
        return None
    return {"kanun_no": kanun_no, "declared_law_name": kanun_ad, "canonical_law_name": canonical}


def _article_absent_finding(reference: Mapping[str, str]) -> dict[str, Any] | None:
    article = reference["madde"]
    if not _PLAIN_NUMBER_RE.match(article):
        return None
    span = reference["source_text"]
    if _number_present_in_span(article, span):
        return None
    numbers = _numbers_in_span(span)
    classification = "other_number_present" if numbers else "anaphoric"
    return {"article": article, "classification": classification, "numbers_in_span": numbers}


def _duplicate_groups(
    normalized_rows: list[dict[str, str]],
) -> list[list[dict[str, str]]]:
    groups: dict[tuple[Any, ...], list[dict[str, str]]] = defaultdict(list)
    for reference in normalized_rows:
        key = (full_identity(reference), reference["source_text"])
        groups[key].append(reference)
    return [group for group in groups.values() if len(group) > 1]


def audit_reference_set(
    references: Iterable[Mapping[str, Any]], *, document_text: str
) -> dict[str, Any]:
    """Self-consistency checks over one document's reference list.

    Returns counts (each with its denominator alongside it) plus a
    `findings` dict keyed by check name, each entry a list of
    `{"reference": <normalized reference>, ...evidence}` dicts:

    - `ungrounded_span`: `source_text` is empty, or does not occur in
      `document_text` once both sides have had runs of whitespace collapsed
      to a single space (`normalization.normalize_reference`'s own
      `source_text` normalization already does this; here it is applied to
      `document_text` too before the substring check).
    - `law_number_name_conflict`: `kanun_no` is one of the eight canonical
      statutes and the (already alias-resolved) `kanun_ad` disagrees with
      the canonical name for that number.
    - `duplicate_row`: raw rows sharing a `full_identity` *and* the same
      collapsed `source_text` -- see the module docstring for why this one
      check is computed from the normalized-but-undeduplicated rows rather
      than the compacted list every other check uses.
    - `article_number_absent_from_span`: see the module docstring. Never a
      claim that the reference is wrong.

    `reference_count` is the compacted reference count the first three
    per-provision checks are evaluated over; `raw_reference_count` is the
    policy-kept, normalized-but-undeduplicated row count `duplicate_row` is
    evaluated over.
    """
    kept, _ = apply_reference_policy(list(references))
    normalized_rows = [normalize_reference(reference) for reference in kept]
    compacted = compact_references(kept)
    normalized_document_text = normalize_text(document_text)

    ungrounded_items: list[dict[str, Any]] = []
    law_conflict_items: list[dict[str, Any]] = []
    article_absent_items: list[dict[str, Any]] = []
    for reference in compacted:
        if not _is_grounded(reference["source_text"], normalized_document_text):
            ungrounded_items.append(
                {"reference": reference, "source_text": reference["source_text"]}
            )
        conflict = _law_number_conflict(reference)
        if conflict is not None:
            law_conflict_items.append({"reference": reference, **conflict})
        absent = _article_absent_finding(reference)
        if absent is not None:
            article_absent_items.append({"reference": reference, **absent})

    duplicate_items = [
        {"reference": group[0], "duplicate_count": len(group)}
        for group in _duplicate_groups(normalized_rows)
    ]

    return {
        "reference_count": len(compacted),
        "raw_reference_count": len(normalized_rows),
        "ungrounded_span_count": len(ungrounded_items),
        "law_number_name_conflict_count": len(law_conflict_items),
        "duplicate_row_count": len(duplicate_items),
        "article_number_absent_from_span_count": len(article_absent_items),
        "findings": {
            "ungrounded_span": ungrounded_items,
            "law_number_name_conflict": law_conflict_items,
            "duplicate_row": duplicate_items,
            "article_number_absent_from_span": article_absent_items,
        },
    }


def _classify_candidate_only(
    reference: Mapping[str, str], *, normalized_document_text: str
) -> tuple[str, dict[str, Any]]:
    span = reference["source_text"]
    if not _is_grounded(span, normalized_document_text):
        return "ungrounded", {"source_text": span}

    kanun_no = reference["kanun_no"]
    law_number_absent = bool(kanun_no) and not _number_present_in_span(kanun_no, span)
    refers_back = _refers_back_to_a_law(span)

    leading_article_number = _leading_number(reference["madde"])
    article_number_absent = bool(leading_article_number) and not _number_present_in_span(
        leading_article_number, span
    )

    if (law_number_absent and not refers_back) or article_number_absent:
        return "label_inconsistent_with_span", {
            "kanun_no": kanun_no,
            "law_number_absent": law_number_absent,
            "refers_back_to_a_law": refers_back,
            "article": reference["madde"],
            "article_number_absent": article_number_absent,
        }
    return "candidate_addition", {}


def triage_disagreement(
    expert: Iterable[Mapping[str, Any]],
    candidate: Iterable[Mapping[str, Any]],
    *,
    document_text: str,
) -> dict[str, Any]:
    """Classify every provision `candidate` has that `expert` does not.

    `expert` is truth. For each core identity (`normalization.core_identity`)
    present in the compacted `candidate` list but absent from the compacted
    `expert` list, this is the triage a reviewer performs on every
    disagreement:

    - `ungrounded`: the span is empty or does not occur in `document_text`
      (whitespace-collapsed on both sides, exactly as
      `audit_reference_set`'s `ungrounded_span`).
    - `label_inconsistent_with_span`: the span is grounded, but neither the
      stated law number appears in it as its own token nor does the span
      refer back to a law anaphorically (`aynı|anılan|mezkur|mezkûr|söz
      konusu|bu` + `kanun`), or the article's leading number is absent from
      the span. Each finding carries the sub-signals (`law_number_absent`,
      `refers_back_to_a_law`, `article_number_absent`) that produced the
      classification, for a human to read -- never a verdict on which rater
      is at fault.
    - `candidate_addition`: grounded and self-consistent; named a candidate
      addition, never a truth error, because on real data most such
      candidates turned out to be legitimate (see the module docstring).

    Returns per-classification counts and `findings` (as above, plus
    `candidate_reference_count`, `candidate_matched_count` -- candidate
    identities the expert *does* also have, so
    `candidate_only_count == candidate_reference_count -
    candidate_matched_count` always -- and the mirror direction:
    `expert_reference_count`, `expert_only_count` and `expert_only` (the
    sorted core identities the expert has that the candidate lacks), so a
    caller can compute recall without a second pass over the same data.
    """
    expert_kept, _ = apply_reference_policy(list(expert))
    candidate_kept, _ = apply_reference_policy(list(candidate))
    expert_compacted = compact_references(expert_kept)
    candidate_compacted = compact_references(candidate_kept)
    normalized_document_text = normalize_text(document_text)

    expert_core: set[Identity] = {core_identity(reference) for reference in expert_compacted}
    candidate_core: set[Identity] = {core_identity(reference) for reference in candidate_compacted}

    findings: dict[str, list[dict[str, Any]]] = {
        "ungrounded": [],
        "label_inconsistent_with_span": [],
        "candidate_addition": [],
    }
    matched_count = 0
    for reference in candidate_compacted:
        if core_identity(reference) in expert_core:
            matched_count += 1
            continue
        classification, evidence = _classify_candidate_only(
            reference, normalized_document_text=normalized_document_text
        )
        findings[classification].append({"reference": reference, **evidence})

    expert_only = sorted(expert_core - candidate_core)

    return {
        "candidate_reference_count": len(candidate_compacted),
        "expert_reference_count": len(expert_compacted),
        "candidate_matched_count": matched_count,
        "candidate_only_count": len(candidate_compacted) - matched_count,
        "ungrounded_count": len(findings["ungrounded"]),
        "label_inconsistent_with_span_count": len(findings["label_inconsistent_with_span"]),
        "candidate_addition_count": len(findings["candidate_addition"]),
        "expert_only_count": len(expert_only),
        "expert_only": expert_only,
        "findings": findings,
    }
