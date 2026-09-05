"""Tests for legal-reference normalization defects.

Each task's tests are written to fail against the pre-fix code first, per the
project's TDD discipline: some of these defects are invisible on inspection
(a combining character that does not render, a strip applied to a copy
rather than the returned value), so seeing red first is the only proof.
"""

from __future__ import annotations

from data_quality_checker.normalization import (
    _fold,
    core_identity,
    normalize_article,
    normalize_extension,
    normalize_law_number,
    normalize_reference,
)


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


# ---------------------------------------------------------------------------
# Task 3a: "N ve müteakip" ("N and following") reduces to the bare article
# ---------------------------------------------------------------------------


def test_ve_muteakip_suffix_reduces_to_bare_article() -> None:
    assert normalize_article("229 ve müteakip") == normalize_article("229")


def test_ve_muteakip_suffix_negative_distinct_articles_stay_distinct() -> None:
    assert normalize_article("229") != normalize_article("230")


# ---------------------------------------------------------------------------
# Task 3b: ordinals beyond the tenth, concatenated and space-separated
# ---------------------------------------------------------------------------


def test_ordinal_eleventh_concatenated_and_spaced_match_digit_form() -> None:
    assert normalize_extension("onbirinci") == "11"
    assert normalize_extension("on birinci") == "11"


def test_ordinal_twentieth_matches_digit_form() -> None:
    assert normalize_extension("yirminci") == "20"


def test_ordinal_negative_distinct_ordinals_stay_distinct() -> None:
    assert normalize_extension("onbirinci") != normalize_extension("onikinci")


# ---------------------------------------------------------------------------
# Task 3c: leading zeros on the law number
# ---------------------------------------------------------------------------


def test_law_number_strips_leading_zeros() -> None:
    assert normalize_law_number("0193") == "193"


def test_law_number_all_zeros_stays_a_single_zero() -> None:
    assert normalize_law_number("0000") == "0"
    assert normalize_law_number("0") == "0"


def test_law_number_negative_distinct_numbers_stay_distinct() -> None:
    assert normalize_law_number("193") != normalize_law_number("1930")
