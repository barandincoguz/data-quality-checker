"""Tests for legal-reference normalization defects.

Each task's tests are written to fail against the pre-fix code first, per the
project's TDD discipline: some of these defects are invisible on inspection
(a combining character that does not render, a strip applied to a copy
rather than the returned value), so seeing red first is the only proof.
"""

from __future__ import annotations

from data_quality_checker.normalization import _fold

# ---------------------------------------------------------------------------
# Task 1: Turkish-aware case folding
# ---------------------------------------------------------------------------


def test_fold_matches_uppercase_turkish_i_with_dot() -> None:
    assert _fold("GELİR VERGİSİ KANUNU") == _fold("Gelir Vergisi Kanunu")


def test_fold_matches_uppercase_law_with_no_canonical_entry() -> None:
    # No alias-table entry for this law, so the alias table cannot mask the bug.
    assert _fold("KOOPERATİFLER KANUNU") == _fold("Kooperatifler Kanunu")


def test_fold_matches_dotless_i_both_directions() -> None:
    assert _fold("ışık") == _fold("IŞIK")


def test_fold_negative_distinct_law_names_stay_distinct() -> None:
    assert _fold("Gelir Vergisi Kanunu") != _fold("Kurumlar Vergisi Kanunu")
