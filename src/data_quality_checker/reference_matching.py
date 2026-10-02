"""The one reference comparison of the project.

User decision (2 Oct 2026): "döngü motoru ve değerlendiriciyi artık farklı iki şey
gibi görmeyelim; ikisi de en iyisi hangisiyse ondan olsun." Every comparison of
reference lists -- the validation F1 (`benchmark/reference/evaluate.py`), the
router's buckets, the expert screen, the journal table, the three-source error
attribution and its kappas, the reference audit, the judge's metrics -- is this
module, so they cannot answer "is this the same reference?" differently.

A reference is a law and an article of it (1 Oct 2026). Lists of one document
are compared like this:

1. **One spelling** (`canonical_reference`): `normalization.normalize_reference`
   with the number as written -- list marks, ordinals, sub-parts, "ve müteakip",
   spelling of "geçici"/"mükerrer"/"ek", inserted articles ("32/A"), "193
   sayılı", decree numbers, a number stated in the name.
2. **Alignment** (`DocumentContext`): a row with no written number takes the
   number of its law when the law's name points to exactly one number -- among
   the numbered rows of the lists being compared, and the project's canonical
   laws. This is the name fallback (user decision, 2 Oct 2026): "Harçlar
   Kanunu" 57 is "492 Harçlar Kanunu" 57. A name that points to two numbers
   ("Kurumlar Vergisi Kanunu": 5422 and its successor 5520, both in one ruling)
   is never resolved to either; the row keeps its name, and two such rows match
   by name and article.
3. **Identity**: (law number, article) when the row has a number, else (folded
   name, article); a row without an article is a law-level reference. A row
   naming neither a number nor a law is no reference: ignored as truth, a false
   positive as a candidate.
4. **One row per reference** within a list.
5. **A law-only row beside an article of the same law, in the same list, is
   extra, not wrong** (user decision, 2 Oct 2026). In the truth it is not a
   reference (never found, never missed). In a candidate it is a true positive
   when the truth holds that law-level reference, and otherwise neither a false
   positive nor anything else (`excused_*`). So tp + fn is always the truth's
   number of references. A law-only row with no article of its law in its own
   list is a claim like any other.

TP = references in both lists; FP = the candidate's others; FN = the reference
list's others. All of it is set arithmetic on aligned identities, so it does
not depend on row order, and the counts every module reports are the same
counts.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .normalization import (
    CANONICAL_NO_BY_NAME,
    _fold,
    names_a_particular_law,
    normalize_law_name,
    normalize_reference,
)

LAW_LEVEL_KIND = "law_level"
LAW_ARTICLE_KIND = "law_article"

Identity = tuple[str, str, str, str]
"""("no:N" or "name:K", law number or "", folded name K when there is no number,
article or "") -- `normalization.core_identity`'s shape, so a numbered reference
has exactly the identity it always had."""


def canonical_reference(reference: Mapping[str, Any]) -> dict[str, str]:
    """The comparison's one spelling of a row, with the law number as written."""
    raw = {name: str(reference.get(name, "") or "") for name in ("kanun_no", "kanun_ad", "madde", "source_text")}
    return normalize_reference(raw, resolve_number_from_name=False)


def effective_law_name(ref: Mapping[str, str]) -> str:
    """The row's law name, or "" when it names no particular law ("Kanun", "Aynı Kanun")."""
    name = ref.get("kanun_ad", "")
    return name if name and names_a_particular_law(name) else ""


def law_name_key(ref: Mapping[str, str]) -> str:
    """The name compared: folded, letters and digits only."""
    return re.sub(r"[^0-9a-z]+", "", _fold(effective_law_name(ref)))


_CANONICAL_NUMBER_BY_NAME_KEY = {
    re.sub(r"[^0-9a-z]+", "", _fold(name)): number for name, number in CANONICAL_NO_BY_NAME.items()
}


class DocumentContext:
    """Which number a law name stands for, in the lists of one document being compared."""

    def __init__(self, *lists: Iterable[Mapping[str, str]]) -> None:
        numbers: dict[str, set[str]] = defaultdict(set)
        for rows in lists:
            for row in rows:
                if row.get("kanun_no") and law_name_key(row):
                    numbers[law_name_key(row)].add(row["kanun_no"])
        self._numbers = numbers

    def number_for(self, row: Mapping[str, str]) -> str:
        if row.get("kanun_no"):
            return row["kanun_no"]
        key = law_name_key(row)
        if not key:
            return ""
        candidates = set(self._numbers.get(key, set()))
        if key in _CANONICAL_NUMBER_BY_NAME_KEY:
            candidates.add(_CANONICAL_NUMBER_BY_NAME_KEY[key])
        return next(iter(candidates)) if len(candidates) == 1 else ""

    def identity(self, row: Mapping[str, str]) -> Identity:
        number = self.number_for(row)
        name = "" if number else law_name_key(row)
        law = f"no:{number}" if number else (f"name:{name}" if name else "unknown")
        return (law, number, name, row.get("madde", ""))


def law_of(identity: Identity) -> str:
    return identity[0]


def has_law(identity: Identity) -> bool:
    return identity[0] != "unknown"


@dataclass(frozen=True)
class Comparison:
    """How a candidate list compares with a reference list (gold, truth, expert)."""

    tp: int
    fp: int
    fn: int
    matched: tuple[tuple[dict[str, str], dict[str, str]], ...] = ()
    missed: tuple[dict[str, str], ...] = ()
    extra: tuple[dict[str, str], ...] = ()
    excused_reference: tuple[dict[str, str], ...] = ()
    excused_candidate: tuple[dict[str, str], ...] = ()
    matched_identities: tuple[Identity, ...] = ()
    missed_identities: tuple[Identity, ...] = ()
    extra_identities: tuple[Identity, ...] = ()
    diagnostics: dict[str, int] = field(default_factory=dict)

    @property
    def agree(self) -> bool:
        return self.fp == 0 and self.fn == 0

    @property
    def similarity(self) -> float:
        total = self.tp + self.fp + self.fn
        return self.tp / total if total else 1.0


def _claims(rows: Sequence[dict[str, str]], context: DocumentContext) -> tuple[dict[Identity, dict[str, str]], set[Identity], int]:
    """A list's references by identity (one row each, the first written), the
    law-only ones that are extra (an article of their law is in the same list),
    and how many rows repeated a reference."""
    by_identity: dict[Identity, dict[str, str]] = {}
    for row in rows:
        by_identity.setdefault(context.identity(row), row)
    laws_with_an_article = {law_of(i) for i in by_identity if i[3] and has_law(i)}
    redundant = {i for i in by_identity if not i[3] and has_law(i) and law_of(i) in laws_with_an_article}
    return by_identity, redundant, len(rows) - len(by_identity)


def empty_diagnostics() -> dict[str, int]:
    return {
        "eligible_gold_count": 0,
        "eligible_gold_law_article_count": 0,
        "eligible_gold_law_level_count": 0,
        "eligible_prediction_count": 0,
        "eligible_prediction_law_article_count": 0,
        "eligible_prediction_law_level_count": 0,
        "excluded_gold_missing_law_count": 0,
        "invalid_prediction_missing_law_count": 0,
        "matched_law_article_count": 0,
        "matched_law_level_count": 0,
        "matched_exact_law_identity_count": 0,
        "matched_by_law_name_count": 0,
        "matched_with_conflicting_kanun_ad_count": 0,
        "deduplicated_gold_rows": 0,
        "deduplicated_prediction_rows": 0,
        "excused_redundant_gold_law_level_count": 0,
        "excused_redundant_prediction_law_level_count": 0,
    }


def compare_reference_lists(
    reference: Iterable[Mapping[str, Any]],
    candidate: Iterable[Mapping[str, Any]],
    *,
    context_lists: Sequence[Iterable[Mapping[str, Any]]] = (),
) -> Comparison:
    """Compare a candidate list with a reference list of one document, the project's one way.

    `context_lists`: further lists of the same document compared alongside (for
    instance the other raters), so that one law name resolves the same way in
    every pairwise comparison of the document.
    """
    reference_rows = [canonical_reference(r) for r in reference]
    candidate_rows = [canonical_reference(r) for r in candidate]
    context = DocumentContext(reference_rows, candidate_rows, *([canonical_reference(r) for r in rows] for rows in context_lists))
    ref_claims, ref_redundant, ref_repeated = _claims(reference_rows, context)
    cand_claims, cand_redundant, cand_repeated = _claims(candidate_rows, context)

    diagnostics = empty_diagnostics()
    diagnostics["deduplicated_gold_rows"] = ref_repeated
    diagnostics["deduplicated_prediction_rows"] = cand_repeated
    ref_ids = {i for i in ref_claims if has_law(i)}
    cand_ids = {i for i in cand_claims if has_law(i)}
    no_law_candidates = [cand_claims[i] for i in cand_claims if not has_law(i)]
    diagnostics["excluded_gold_missing_law_count"] = sum(1 for i in ref_claims if not has_law(i))
    diagnostics["invalid_prediction_missing_law_count"] = len(no_law_candidates)
    for side, ids in (("gold", ref_ids), ("prediction", cand_ids)):
        diagnostics[f"eligible_{side}_count"] = len(ids)
        for kind, count in Counter(LAW_ARTICLE_KIND if i[3] else LAW_LEVEL_KIND for i in ids).items():
            diagnostics[f"eligible_{side}_{kind}_count"] = count

    # The truth's claims leave out its own law-only rows beside an article of
    # their law: those are extra in the truth too, never a reference to find.
    truth = ref_ids - ref_redundant
    common = sorted(truth & cand_ids)
    matched = []
    for identity in common:
        left, right = ref_claims[identity], cand_claims[identity]
        matched.append((left, right))
        diagnostics["matched_law_article_count" if identity[3] else "matched_law_level_count"] += 1
        if left.get("kanun_no") and right.get("kanun_no"):
            diagnostics["matched_exact_law_identity_count"] += 1
        else:
            diagnostics["matched_by_law_name_count"] += 1
        if law_name_key(left) and law_name_key(right) and law_name_key(left) != law_name_key(right):
            # "213 sayılı Gelir Vergisi Kanunu" is VUK by its number: counted, never hidden.
            diagnostics["matched_with_conflicting_kanun_ad_count"] += 1

    missed_ids = sorted(truth - cand_ids)
    extra_ids = sorted(i for i in cand_ids - truth if i not in cand_redundant)
    excused_reference = sorted(ref_redundant)
    excused_candidate = sorted(i for i in cand_redundant if i not in truth)
    diagnostics["excused_redundant_gold_law_level_count"] = len(excused_reference)
    diagnostics["excused_redundant_prediction_law_level_count"] = len(excused_candidate)
    return Comparison(
        tp=len(common),
        fp=len(extra_ids) + len(no_law_candidates),
        fn=len(missed_ids),
        matched=tuple(matched),
        missed=tuple(ref_claims[i] for i in missed_ids),
        extra=tuple([cand_claims[i] for i in extra_ids] + no_law_candidates),
        excused_reference=tuple(ref_claims[i] for i in excused_reference),
        excused_candidate=tuple(cand_claims[i] for i in excused_candidate),
        matched_identities=tuple(common),
        missed_identities=tuple(missed_ids),
        extra_identities=tuple(extra_ids),
        diagnostics=diagnostics,
    )


def same_reference(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Are two rows, compared on their own, the same reference?"""
    return compare_reference_lists([left], [right]).agree


def reference_identities(
    references: Iterable[Mapping[str, Any]],
    *,
    context_lists: Sequence[Iterable[Mapping[str, Any]]] = (),
) -> set[Identity]:
    """The references a list claims, aligned with the other lists of its document:
    one identity per reference, law-only rows beside an article of their law left
    out (extra, not a claim)."""
    rows = [canonical_reference(r) for r in references]
    context = DocumentContext(rows, *([canonical_reference(r) for r in other] for other in context_lists))
    claims, redundant, _ = _claims(rows, context)
    return {i for i in claims if i not in redundant}


def display_name(identity: Identity) -> str:
    """A readable label for an identity (journal and screen tables)."""
    _, number, name, article = identity
    law = number or name or "?"
    return f"{law}/{article}" if article else law


def law_name_for_display(name: str, number: str) -> str:
    return normalize_law_name(name, law_number=number)


def compare_with_policy(
    reference: Iterable[Mapping[str, Any]],
    candidate: Iterable[Mapping[str, Any]],
    *,
    context_lists: Sequence[Iterable[Mapping[str, Any]]] = (),
    policy_id: str | None = None,
) -> Comparison:
    """`compare_reference_lists` after the reference policy (VUK 213/413) is removed
    from every list alike -- the comparison the loop's modules make."""
    from .reference_policy import DEFAULT_REFERENCE_POLICY_ID, apply_reference_policy

    policy = policy_id or DEFAULT_REFERENCE_POLICY_ID
    kept = [apply_reference_policy(list(rows), policy_id=policy)[0] for rows in (reference, candidate, *context_lists)]
    return compare_reference_lists(kept[0], kept[1], context_lists=kept[2:])
