"""The evaluator's matcher: the project's one reference comparison.

Since 2 Oct 2026 (user decision: "döngü motoru ve değerlendiriciyi artık farklı iki
şey gibi görmeyelim") the matching itself lives in
`data_quality_checker.reference_matching`, which every comparison of the project
uses -- the router, the expert screen, the journal table, the error attribution,
the judge's metrics, and this evaluator. This module keeps the names the
evaluator and its tests import.

The semantics (metric profile `strict-v2`, order-independent optimal matching)
are unchanged except where that module's docstring says otherwise: one
normalizer, the name fallback for rows without a number, and law-only rows
beside an article of their law in the same list scored as extra, not wrong. The
archived five-field matcher is at git tag archive/gt750-v1-fikra-bent.
"""

from __future__ import annotations

from typing import Any

from data_quality_checker.normalization import normalize_article
from data_quality_checker.reference_matching import (  # noqa: F401 -- re-exported
    LAW_ARTICLE_KIND,
    LAW_LEVEL_KIND,
    Comparison,
    DocumentContext,
    canonical_reference,
    compare_reference_lists,
    effective_law_name,
    law_name_key,
    same_reference,
)


def core_identity_key(ref: dict[str, str]) -> tuple[str, str, str, str]:
    """A row's identity, aligned on its own (no other list in the context)."""
    row = canonical_reference(ref)
    return DocumentContext([row]).identity(row)


def law_identity_compatible(left: dict[str, str], right: dict[str, str]) -> bool:
    """Do two rows, aligned with each other, name the same law?"""
    rows = [canonical_reference(left), canonical_reference(right)]
    context = DocumentContext(rows)
    a, b = (context.identity(row) for row in rows)
    return a[0] != "unknown" and a[0] == b[0]


from data_quality_checker.reference_matching import (  # noqa: E402,F401 -- re-exported
    empty_diagnostics as empty_optimal_diagnostics,
)

canonicalize_reference = canonical_reference


def canonical_article(ref: dict[str, str]) -> str:
    """The article a row names, in the comparison's one spelling."""
    return normalize_article(ref.get("madde", ""))


canonical_article_key = canonical_article


def evaluate_doc_optimal(
    pred_refs: list[dict[str, str]],
    gold_refs: list[dict[str, str]],
) -> tuple[int, int, int, dict[str, int], list[tuple[dict[str, str], dict[str, str]]]]:
    """Score one document: `(tp, fp, fn, diagnostics, matched (gold, prediction) pairs)`."""
    comparison = compare_reference_lists(gold_refs, pred_refs)
    return comparison.tp, comparison.fp, comparison.fn, dict(comparison.diagnostics), list(comparison.matched)


def optimal_diagnostics_payload(diagnostics: dict[str, int]) -> dict[str, Any]:
    """Add derived rates to the matcher's diagnostics."""
    payload: dict[str, Any] = dict(diagnostics)
    matched_total = diagnostics["matched_law_article_count"] + diagnostics["matched_law_level_count"]
    payload["matched_total"] = matched_total
    payload["matched_exact_law_identity_rate"] = (
        diagnostics["matched_exact_law_identity_count"] / matched_total if matched_total else 0.0
    )
    return payload
