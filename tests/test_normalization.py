"""Tests for legal-reference normalization defects.

Each task's tests are written to fail against the pre-fix code first, per the
project's TDD discipline: some of these defects are invisible on inspection
(a combining character that does not render, a strip applied to a copy
rather than the returned value), so seeing red first is the only proof.
"""

from __future__ import annotations

from data_quality_checker.normalization import _fold, core_identity, normalize_reference


def _ref(**updates: str) -> dict[str, str]:
    payload = {
        "kanun_no": "",
        "kanun_ad": "",
        "madde": "1",
        "fikra": "",
        "bent": "",
        "source_text": "x",
    }
    payload.update(updates)
    return payload


def _core(kanun_ad: str) -> tuple[str, str, str, str]:
    return core_identity(normalize_reference(_ref(kanun_ad=kanun_ad)))


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


# ---------------------------------------------------------------------------
# Task 2: Law-name prefixes and stray punctuation
# ---------------------------------------------------------------------------


def test_numbered_prefix_does_not_defeat_alias_lookup() -> None:
    assert _core("193 sayılı Gelir Vergisi Kanunu") == _core("Gelir Vergisi Kanunu")


def test_returned_law_name_has_stray_punctuation_stripped() -> None:
    # 'Kooperatifler Kanunu' has no alias-table entry, so this also proves the
    # punctuation strip applies to the value actually returned, not a copy.
    assert _core("1163 sayılı Kooperatifler Kanunu") == _core('Kooperatifler Kanunu"')


def test_numbered_prefix_negative_distinct_laws_stay_distinct() -> None:
    assert _core("193 sayılı Gelir Vergisi Kanunu") != _core("213 sayılı Vergi Usul Kanunu")
