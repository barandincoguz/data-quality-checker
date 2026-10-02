"""Tests for legal-reference normalization defects.

Each task's tests are written to fail against the pre-fix code first, per the
project's TDD discipline: some of these defects are invisible on inspection
(a combining character that does not render, a strip applied to a copy
rather than the returned value), so seeing red first is the only proof.
"""

from __future__ import annotations

from data_quality_checker.normalization import (
    _fold,
    compact_references,
    core_identity,
    normalize_article,
    normalize_law_number,
    normalize_reference,
)


def _ref(**updates: str) -> dict[str, str]:
    payload = {
        "kanun_no": "",
        "kanun_ad": "",
        "madde": "1",
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
    assert _core("Kooperatifler Kanunu") == _core('Kooperatifler Kanunu"')


def test_a_number_written_into_the_name_is_the_law_number() -> None:
    # Engine audit (2 Oct 2026): "1163 sayılı Kooperatifler Kanunu" with an empty
    # number field names law 1163 -- the same reference as a row numbered 1163,
    # as the evaluator already read it.
    from data_quality_checker.normalization import core_identity, normalize_reference

    numbered = core_identity(normalize_reference({"kanun_no": "1163", "kanun_ad": "Kooperatifler Kanunu", "madde": "1", "source_text": ""}))
    assert _core("1163 sayılı Kooperatifler Kanunu") == numbered


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
# Task 3c: leading zeros on the law number
# ---------------------------------------------------------------------------


def test_law_number_strips_leading_zeros() -> None:
    assert normalize_law_number("0193") == "193"


def test_law_number_all_zeros_stays_a_single_zero() -> None:
    assert normalize_law_number("0000") == "0"
    assert normalize_law_number("0") == "0"


def test_law_number_negative_distinct_numbers_stay_distinct() -> None:
    assert normalize_law_number("193") != normalize_law_number("1930")


# ---------------------------------------------------------------------------
# A reference is a law and an article (1 Oct 2026): a part of the article
# written into `madde` is cut off; an inserted article keeps its letter.
# ---------------------------------------------------------------------------


def test_a_part_of_the_article_written_into_madde_is_cut_off() -> None:
    for written in ("6/b", "6/ç", "6 /b", "6/4", "6/4-a", "6 (2)", "6(a)", "6-a", "6/b maddesi"):
        assert normalize_article(written) == "6", written


def test_a_list_mark_before_the_article_is_not_part_of_it() -> None:
    """KDV rulings list provisions as "-1/1 inci maddesinde"; M000 v2 copied the
    mark into `madde` at update 25 (1 Oct 2026). No article starts with one."""
    for written in ("-1", "-1/1", "– 1", "• 1", "*1"):
        assert normalize_article(written) == "1", written


def test_an_inserted_article_keeps_its_uppercase_letter() -> None:
    # 32/A ("İndirimli kurumlar vergisi") is a distinct article inserted by
    # amendment, not a part of article 32 -- it must never be cut off.
    assert normalize_article("32/A") == "32/A"
    assert normalize_article("32/ A") == "32/A"
    assert normalize_article("mükerrer 20/A-B") == "mükerrer 20/A-B"
    assert normalize_article("5/A-1") == "5/A"
    assert normalize_article("32/A") != normalize_article("32")


def test_normalizing_twice_changes_nothing() -> None:
    """The inserted-article letter stays upper case, so a normalized "32/A"
    is not read as a part of article 32 on a second pass (it was: "32/a")."""
    for written in ("32/A", "Geçici 21", "mükerrer 20/A-B", "13/a", "466-469", "Ek 5", "229 ve müteakip"):
        once = normalize_article(written)
        assert normalize_article(once) == once, written
        reference = normalize_reference(_ref(madde=written))
        assert normalize_reference(reference) == reference, written


def test_a_stored_row_with_paragraph_fields_is_read_as_its_law_and_article() -> None:
    """Lists written before 1 Oct 2026 still carry `fikra`/`bent`; they are
    projected away on read, so the same provision at two depths is one."""
    deep = normalize_reference({**_ref(kanun_no="193", madde="94"), "fikra": "1", "bent": "a"})
    plain = normalize_reference(_ref(kanun_no="193", madde="94"))
    assert deep == plain
    assert set(deep) == {"kanun_no", "kanun_ad", "madde", "source_text"}


def test_uppercase_prefixed_law_with_embedded_part_matches_its_plain_twin() -> None:
    """One row combining every defect this plan fixes."""
    a = compact_references(
        [{"kanun_no": "3065", "kanun_ad": "3065 SAYILI KATMA DEĞER VERGİSİ KANUNU", "madde": "6/b", "source_text": "x"}]
    )
    b = compact_references([{"kanun_no": "3065", "kanun_ad": "Katma Değer Vergisi Kanunu", "madde": "6", "source_text": "y"}])
    assert core_identity(a[0]) == core_identity(b[0])


def test_law_name_is_not_part_of_identity_when_the_number_is_present() -> None:
    """The number identifies the law; the name beside it is redundant.

    Only eight laws sit in CANONICAL_LAW_BY_NO, so for every other statute an
    empty name on one side and a written name on the other made two references
    to the same provision compare as different. Measured on real data: three of
    the eight apparent ground-truth gaps were this and nothing else.
    """
    for number, written in (
        ("506", "506 Sayılı Kanun"),
        ("1163", "Kooperatifler Kanunu"),
        ("7194", "7194 sayılı kanun"),
        ("4857", "İş Kanunu"),
    ):
        blank = compact_references(
            [
                {
                    "kanun_no": number,
                    "kanun_ad": "",
                    "madde": "60",
                    "source_text": "x",
                }
            ]
        )
        named = compact_references(
            [
                {
                    "kanun_no": number,
                    "kanun_ad": written,
                    "madde": "60",
                    "source_text": "y",
                }
            ]
        )
        assert core_identity(blank[0]) == core_identity(named[0]), number


def test_distinct_laws_stay_distinct_without_the_name() -> None:
    """Dropping the name must not merge two different statutes."""
    a = compact_references(
        [
            {
                "kanun_no": "506",
                "kanun_ad": "",
                "madde": "60",
                "source_text": "x",
            }
        ]
    )
    b = compact_references(
        [
            {
                "kanun_no": "1163",
                "kanun_ad": "",
                "madde": "60",
                "source_text": "x",
            }
        ]
    )
    assert core_identity(a[0]) != core_identity(b[0])


def test_the_name_still_identifies_a_law_that_carries_no_number() -> None:
    """With kanun_no empty, law_identity falls back to the name -- keep that."""
    a = compact_references(
        [
            {
                "kanun_no": "",
                "kanun_ad": "Kooperatifler Kanunu",
                "madde": "93",
                "source_text": "x",
            }
        ]
    )
    b = compact_references(
        [
            {
                "kanun_no": "",
                "kanun_ad": "İş Kanunu",
                "madde": "93",
                "source_text": "x",
            }
        ]
    )
    assert core_identity(a[0]) != core_identity(b[0])
