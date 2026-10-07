"""Comprehensive stress tests and edge case audit for the unified reference matching engine.

Tests all categories:
1. Law name and number fallback & resolution
2. Article (madde) normalization & variations
3. Subsumption (law-only row beside article in same list)
4. Deduplication & cardinality
5. DocumentContext multi-list consistency
6. Malformed inputs, fuzzing, and scale stress tests
"""

from __future__ import annotations

import random
import pytest

from data_quality_checker.normalization import (
    normalize_article,
    normalize_law_name,
    normalize_law_number,
    normalize_reference,
)
from data_quality_checker.reference_matching import (
    DocumentContext,
    canonical_reference,
    compare_reference_lists,
)


def ref(no: str, ad: str = "", md: str = "") -> dict[str, str]:
    return {"kanun_no": no, "kanun_ad": ad, "madde": md, "source_text": ""}


def counts(reference, candidate, context_lists=()) -> tuple[int, int, int]:
    comp = compare_reference_lists(reference, candidate, context_lists=context_lists)
    return comp.tp, comp.fp, comp.fn


# ==============================================================================
# 1. LAW NAME & NUMBER FALLBACK
# ==============================================================================

def test_numbered_vs_named_fallback_for_canonical_laws():
    """Laws in CANONICAL_LAW_BY_NO: GVK (193), VUK (213), KVK (5520), KDVK (3065), etc."""
    assert counts([ref("193", "", "94")], [ref("", "Gelir Vergisi Kanunu", "94")]) == (1, 0, 0)
    assert counts([ref("213", "", "5")], [ref("", "Vergi Usul Kanunu", "5")]) == (1, 0, 0)
    assert counts([ref("3065", "", "1")], [ref("", "Katma Değer Vergisi Kanunu", "1")]) == (1, 0, 0)
    assert counts([ref("488", "", "1")], [ref("", "Damga Vergisi Kanunu", "1")]) == (1, 0, 0)
    assert counts([ref("4760", "", "1")], [ref("", "Özel Tüketim Vergisi Kanunu", "1")]) == (1, 0, 0)
    assert counts([ref("6183", "", "1")], [ref("", "Amme Alacaklarının Tahsil Usulü Hakkında Kanun", "1")]) == (1, 0, 0)


def test_alias_abbreviations():
    """Common abbreviations like VUK, GVK, KDVK, ÖTVK, DVK."""
    assert counts([ref("193", "", "94")], [ref("", "GVK", "94")]) == (1, 0, 0)
    assert counts([ref("213", "", "5")], [ref("", "VUK", "5")]) == (1, 0, 0)
    assert counts([ref("3065", "", "1")], [ref("", "KDVK", "1")]) == (1, 0, 0)
    assert counts([ref("4760", "", "1")], [ref("", "ÖTVK", "1")]) == (1, 0, 0)
    assert counts([ref("488", "", "1")], [ref("", "DVK", "1")]) == (1, 0, 0)


def test_ambiguous_laws_not_resolved():
    """Kurumlar Vergisi Kanunu (5422 vs 5520) must NOT resolve to either number if both are in ruling."""
    # When both only name Kurumlar Vergisi Kanunu, they match by name
    assert counts([ref("", "Kurumlar Vergisi Kanunu", "14")], [ref("", "Kurumlar Vergisi Kanunu", "14")]) == (1, 0, 0)
    # When both 5422 and 5520 are in the ruling context, the name-only row cannot be guessed
    gold = [ref("5422", "Kurumlar Vergisi Kanunu", ""), ref("5520", "Kurumlar Vergisi Kanunu", "14")]
    cand = [ref("", "Kurumlar Vergisi Kanunu", "14"), ref("5422", "", "")]
    tp, fp, fn = counts(gold, cand)
    assert (tp, fp, fn) == (1, 1, 1)


def test_number_prefix_in_name_field():
    """'6183 sayılı Kanun' in the name field carries the number."""
    assert counts([ref("6183", "", "5")], [ref("", "6183 sayılı Kanun", "5")]) == (1, 0, 0)
    assert counts([ref("6183", "", "5")], [ref("", "6183 sayılı Amme Alacakları Kanunu", "5")]) == (1, 0, 0)


def test_name_written_in_number_field():
    """A name written in kanun_no field ('Gelir Vergisi Kanunu') is read as name."""
    cand = [{"kanun_no": "Gelir Vergisi Kanunu", "kanun_ad": "", "madde": "94"}]
    gold = [ref("193", "", "94")]
    assert counts(gold, cand) == (1, 0, 0)


def test_leading_zeros_and_empty_numbers():
    assert counts([ref("193", "", "94")], [ref("0193", "", "94")]) == (1, 0, 0)
    assert counts([ref("193", "", "94")], [ref("00193", "", "94")]) == (1, 0, 0)
    # Words meaning 'no number'
    for empty in ["null", "none", "nan", "yok", "-", "—", "n/a"]:
        assert counts([ref("193", "", "94")], [ref(empty, "Gelir Vergisi Kanunu", "94")]) == (1, 0, 0)


# ==============================================================================
# 2. ARTICLE (MADDE) NORMALIZATION & VARIATIONS
# ==============================================================================

def test_article_basic_and_ordinals():
    assert counts([ref("193", "", "94")], [ref("193", "", "94.")]) == (1, 0, 0)
    assert counts([ref("193", "", "94")], [ref("193", "", "94. madde")]) == (1, 0, 0)
    assert counts([ref("193", "", "94")], [ref("193", "", "94 üncü madde")]) == (1, 0, 0)
    assert counts([ref("193", "", "94")], [ref("193", "", "94'üncü maddesi")]) == (1, 0, 0)
    assert counts([ref("193", "", "94")], [ref("193", "", "md. 94")]) == (1, 0, 0)
    assert counts([ref("193", "", "94")], [ref("193", "", "m. 94")]) == (1, 0, 0)
    assert counts([ref("193", "", "94")], [ref("193", "", "94 md.")]) == (1, 0, 0)


def test_article_subparts_stripped():
    """1 Oct 2026 rule: fıkra/bent in madde are stripped."""
    assert counts([ref("3065", "", "13")], [ref("3065", "", "13/a")]) == (1, 0, 0)
    assert counts([ref("3065", "", "13")], [ref("3065", "", "13/b")]) == (1, 0, 0)
    assert counts([ref("3065", "", "17")], [ref("3065", "", "17/4-r")]) == (1, 0, 0)
    assert counts([ref("3065", "", "17")], [ref("3065", "", "(17/4-r)")]) == (1, 0, 0)
    assert counts([ref("3065", "", "1")], [ref("3065", "", "-1/1")]) == (1, 0, 0)


def test_inserted_uppercase_article_preserved():
    """32/A is part of the article name, not a subpart! 32/a is article 32, so they differ."""
    assert counts([ref("5520", "", "32/A")], [ref("5520", "", "32/a")]) == (0, 1, 1)
    assert counts([ref("5520", "", "32/A")], [ref("5520", "", "32 / A")]) == (1, 0, 0)
    assert counts([ref("5520", "", "32/A")], [ref("5520", "", "32/A")]) == (1, 0, 0)


def test_qualified_articles():
    assert counts([ref("193", "", "geçici 67")], [ref("193", "", "Geçici 67")]) == (1, 0, 0)
    assert counts([ref("193", "", "geçici 67")], [ref("193", "", "geç. 67")]) == (1, 0, 0)
    assert counts([ref("193", "", "geçici 67")], [ref("193", "", "Geçici Madde 67")]) == (1, 0, 0)
    assert counts([ref("213", "", "mükerrer 298")], [ref("213", "", "Mükerrer 298")]) == (1, 0, 0)
    assert counts([ref("213", "", "mükerrer 298")], [ref("213", "", "mük. 298")]) == (1, 0, 0)
    assert counts([ref("488", "", "ek 1")], [ref("488", "", "Ek 1")]) == (1, 0, 0)
    assert counts([ref("488", "", "ek 1")], [ref("488", "", "Ek Madde 1")]) == (1, 0, 0)


def test_ranges_normalized():
    assert counts([ref("193", "", "94 ila 96")], [ref("193", "", "94-96")]) == (1, 0, 0)
    assert counts([ref("193", "", "94 ila 96")], [ref("193", "", "94–96")]) == (1, 0, 0)
    assert counts([ref("193", "", "94 ila 96")], [ref("193", "", "94 ilâ 96")]) == (1, 0, 0)


def test_open_ended_ranges():
    assert counts([ref("213", "", "229")], [ref("213", "", "229 ve müteakip")]) == (1, 0, 0)
    assert counts([ref("213", "", "229")], [ref("213", "", "229 ve izleyen")]) == (1, 0, 0)
    assert counts([ref("213", "", "229")], [ref("213", "", "229 ve devamı")]) == (1, 0, 0)


def test_bare_paragraph_number_never_becomes_law_level():
    """'(17)' without article is an unreadable article, not a law-level match!"""
    gold = [ref("3065", "", "")]  # law-level
    cand = [ref("3065", "", "(17)")]
    tp, fp, fn = counts(gold, cand)
    assert tp == 0
    assert fp == 1
    assert fn == 1


# ==============================================================================
# 3. SUBSUMPTION (LAW-ONLY ROW BESIDE ARTICLE IN SAME LIST)
# ==============================================================================

def test_law_only_row_beside_article_excused():
    gold = [ref("193", "", "94")]
    cand = [ref("193", "", ""), ref("193", "", "94")]
    assert counts(gold, cand) == (1, 0, 0)

    # In truth, extra law-only is not a reference to find
    truth = [ref("193", "", ""), ref("193", "", "94")]
    assert counts(truth, [ref("193", "", "94")]) == (1, 0, 0)


def test_law_only_alone_is_a_claim():
    gold = [ref("193", "", "94")]
    cand = [ref("193", "", "")]
    assert counts(gold, cand) == (0, 1, 1)

    gold = [ref("193", "", "")]
    cand = [ref("193", "", "94")]
    assert counts(gold, cand) == (0, 1, 1)


def test_multiple_articles_subsume_single_law_level():
    gold = [ref("193", "", "94"), ref("193", "", "103"), ref("193", "", "104")]
    cand = [ref("193", "", "94"), ref("193", "", "103"), ref("193", "", "104"), ref("193", "", "")]
    assert counts(gold, cand) == (3, 0, 0)


def test_duplicate_law_level_subsumed():
    gold = [ref("193", "", "94")]
    cand = [ref("193", "", ""), ref("193", "", ""), ref("193", "", "94")]
    assert counts(gold, cand) == (1, 0, 0)


# ==============================================================================
# 4. DEDUPLICATION & PERMUTATION INVARIANCE
# ==============================================================================

def test_deduplication_exact_and_spelling():
    gold = [ref("193", "", "94")]
    cand = [ref("193", "", "94"), ref("193", "", "94"), ref("193", "", "94.")]
    assert counts(gold, cand) == (1, 0, 0)


def test_permutation_invariance_stress():
    gold = [
        ref("193", "Gelir Vergisi Kanunu", "94"),
        ref("193", "Gelir Vergisi Kanunu", "103"),
        ref("213", "Vergi Usul Kanunu", "5"),
        ref("5520", "Kurumlar Vergisi Kanunu", "32/A"),
        ref("193", "", ""),
        ref("6770", "", ""),
    ]
    cand = [
        ref("", "Gelir Vergisi Kanunu", "94"),
        ref("193", "", "103"),
        ref("213", "Vergi Usul Kanunu", "5"),
        ref("5520", "", "32/A"),
        ref("193", "", ""),
        ref("6770", "", ""),
        ref("3065", "KDV", "1"),
    ]
    base = counts(gold, cand)
    rng = random.Random(12345)
    for _ in range(200):
        g = list(gold)
        c = list(cand)
        rng.shuffle(g)
        rng.shuffle(c)
        assert counts(g, c) == base


# ==============================================================================
# 5. MULTI-LIST DOCUMENT CONTEXT & TRANSITIVITY
# ==============================================================================

def test_context_lists_align_law_names_transitively():
    """If Rater C connects 492 with 'Harçlar Kanunu', pairwise A and B benefit."""
    rater_a = [ref("492", "", "57")]
    rater_b = [ref("", "Harçlar Kanunu", "57")]
    rater_c = [ref("492", "Harçlar Kanunu", "57")]

    # With rater_c in context_lists, Harçlar Kanunu resolves to 492
    assert counts(rater_a, rater_b, context_lists=[rater_c]) == (1, 0, 0)


# ==============================================================================
# 6. FUZZING, MALFORMED INPUTS & SCALE STRESS
# ==============================================================================

def test_malformed_and_extreme_inputs():
    """Empty, None, unicode garbage, whitespace, huge text."""
    weird_candidates = [
        {},
        {"kanun_no": None, "kanun_ad": None, "madde": None},
        {"kanun_no": "   ", "kanun_ad": "\t\n", "madde": ""},
        {"kanun_no": "???", "kanun_ad": "###", "madde": "!!!"},
        {"kanun_no": "193", "kanun_ad": "A" * 5000, "madde": "94"},
    ]
    gold = [ref("193", "", "94")]
    # Should not raise exception
    res = compare_reference_lists(gold, weird_candidates)
    assert res.tp == 1  # the long name one still has 193/94


def test_large_scale_stress():
    """Document with 50 gold references and 60 candidate references."""
    gold = [ref("193", "", str(i)) for i in range(1, 51)]
    cand = [ref("193", "", str(i)) for i in range(1, 41)] + [ref("213", "", str(i)) for i in range(1, 21)]
    res = compare_reference_lists(gold, cand)
    assert res.tp == 40
    assert res.fn == 10
    assert res.fp == 20


# ==============================================================================
# 7. ENGINE HARDENING REGRESSION TESTS (AUDIT GAPS RESOLVED)
# ==============================================================================

def test_expanded_canonical_laws_direct_fallback():
    """All 14 newly added canonical laws and aliases resolve without ruling context."""
    # 492 Harçlar Kanunu (105 refs in GT v8)
    assert counts([ref("492", "", "57")], [ref("", "Harçlar Kanunu", "57")]) == (1, 0, 0)
    assert counts([ref("492", "", "57")], [ref("", "harclar kanunu", "57")]) == (1, 0, 0)

    # 6802 Gider Vergileri Kanunu (36 refs)
    assert counts([ref("6802", "", "29")], [ref("", "Gider Vergileri Kanunu", "29")]) == (1, 0, 0)
    assert counts([ref("6802", "", "29")], [ref("", "gider vergisi kanunu", "29")]) == (1, 0, 0)

    # 1319 Emlak Vergisi Kanunu (34 refs)
    assert counts([ref("1319", "", "8")], [ref("", "Emlak Vergisi Kanunu", "8")]) == (1, 0, 0)

    # 2464 Belediye Gelirleri Kanunu (34 refs)
    assert counts([ref("2464", "", "12")], [ref("", "Belediye Gelirleri Kanunu", "12")]) == (1, 0, 0)

    # 7338 Veraset ve İntikal Vergisi Kanunu (18 refs)
    assert counts([ref("7338", "", "4")], [ref("", "Veraset ve İntikal Vergisi Kanunu", "4")]) == (1, 0, 0)

    # 197 Motorlu Taşıtlar Vergisi Kanunu
    assert counts([ref("197", "", "5")], [ref("", "Motorlu Taşıtlar Vergisi Kanunu", "5")]) == (1, 0, 0)
    assert counts([ref("197", "", "5")], [ref("", "MTVK", "5")]) == (1, 0, 0)

    # 6102 Türk Ticaret Kanunu & TTK
    assert counts([ref("6102", "", "18")], [ref("", "Türk Ticaret Kanunu", "18")]) == (1, 0, 0)
    assert counts([ref("6102", "", "18")], [ref("", "TTK", "18")]) == (1, 0, 0)

    # 5510 Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu & SSGSK
    assert counts([ref("5510", "", "4")], [ref("", "Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu", "4")]) == (1, 0, 0)
    assert counts([ref("5510", "", "4")], [ref("", "SSGSK", "4")]) == (1, 0, 0)

    # 4721 Türk Medeni Kanunu & TMK
    assert counts([ref("4721", "", "1")], [ref("", "Türk Medeni Kanunu", "1")]) == (1, 0, 0)
    assert counts([ref("4721", "", "1")], [ref("", "TMK", "1")]) == (1, 0, 0)

    # 6098 Türk Borçlar Kanunu & TBK
    assert counts([ref("6098", "", "10")], [ref("", "Türk Borçlar Kanunu", "10")]) == (1, 0, 0)
    assert counts([ref("6098", "", "10")], [ref("", "TBK", "10")]) == (1, 0, 0)

    # 3218 Serbest Bölgeler Kanunu & SBK
    assert counts([ref("3218", "", "6")], [ref("", "Serbest Bölgeler Kanunu", "6")]) == (1, 0, 0)
    assert counts([ref("3218", "", "6")], [ref("", "SBK", "6")]) == (1, 0, 0)

    # 4458 Gümrük Kanunu, 3194 İmar Kanunu, 4857 İş Kanunu
    assert counts([ref("4458", "", "1")], [ref("", "Gümrük Kanunu", "1")]) == (1, 0, 0)
    assert counts([ref("3194", "", "1")], [ref("", "İmar Kanunu", "1")]) == (1, 0, 0)
    assert counts([ref("4857", "", "1")], [ref("", "İş Kanunu", "1")]) == (1, 0, 0)

    # Standard Acronyms for existing laws
    assert counts([ref("6183", "", "48")], [ref("", "AATUHK", "48")]) == (1, 0, 0)
    assert counts([ref("6183", "", "48")], [ref("", "AATK", "48")]) == (1, 0, 0)
    assert counts([ref("5018", "", "1")], [ref("", "KMYKK", "1")]) == (1, 0, 0)


def test_qualified_article_abbreviations():
    """'geçici md. 67', 'ek md. 1', 'mük. md. 298' match their canonical forms."""
    assert counts([ref("193", "", "geçici 67")], [ref("193", "", "geçici md. 67")]) == (1, 0, 0)
    assert counts([ref("193", "", "geçici 67")], [ref("193", "", "geçici m. 67")]) == (1, 0, 0)
    assert counts([ref("193", "", "geçici 67")], [ref("193", "", "geçici md: 67")]) == (1, 0, 0)
    assert counts([ref("193", "", "geçici 67")], [ref("193", "", "geçici madde: 67")]) == (1, 0, 0)
    assert counts([ref("488", "", "ek 1")], [ref("488", "", "ek md. 1")]) == (1, 0, 0)
    assert counts([ref("213", "", "mükerrer 298")], [ref("213", "", "mük. md. 298")]) == (1, 0, 0)


def test_qualified_article_ranges():
    """'geçici 1-3' normalizes to 'geçici 1 ila 3'."""
    assert counts([ref("193", "", "geçici 1 ila 3")], [ref("193", "", "geçici 1-3")]) == (1, 0, 0)
    assert counts([ref("193", "", "geçici 1 ila 3")], [ref("193", "", "geçici 1 – 3")]) == (1, 0, 0)
    assert counts([ref("193", "", "geçici 1 ila 3")], [ref("193", "", "geçici 1 — 3")]) == (1, 0, 0)
    assert counts([ref("193", "", "geçici 1 ila 3")], [ref("193", "", "geçici 1 ila 3")]) == (1, 0, 0)


def test_article_prefix_colons_and_hyphens():
    """'Madde: 94', 'Md: 94', 'Md.-94', 'Madde - 94' strip prefixes cleanly."""
    assert counts([ref("193", "", "94")], [ref("193", "", "madde: 94")]) == (1, 0, 0)
    assert counts([ref("193", "", "94")], [ref("193", "", "Madde: 94")]) == (1, 0, 0)
    assert counts([ref("193", "", "94")], [ref("193", "", "md: 94")]) == (1, 0, 0)
    assert counts([ref("193", "", "94")], [ref("193", "", "Md: 94")]) == (1, 0, 0)
    assert counts([ref("193", "", "94")], [ref("193", "", "md.-94")]) == (1, 0, 0)
    assert counts([ref("193", "", "94")], [ref("193", "", "madde - 94")]) == (1, 0, 0)
    assert counts([ref("193", "", "94")], [ref("193", "", "m: 94")]) == (1, 0, 0)


def test_em_dash_range_normalization():
    """Em-dash '—' is treated identically to hyphen and en-dash."""
    assert counts([ref("193", "", "94 ila 96")], [ref("193", "", "94 — 96")]) == (1, 0, 0)
    assert counts([ref("193", "", "94 ila 96")], [ref("193", "", "94—96")]) == (1, 0, 0)


def test_float_law_number_preservation():
    """'193.0' or float 193.0 is not converted to 1930."""
    assert counts([ref("193", "", "94")], [ref("193.0", "", "94")]) == (1, 0, 0)
    assert counts([ref("193", "", "94")], [ref("193.00", "", "94")]) == (1, 0, 0)
    assert counts([ref("193", "", "94")], [ref(193.0, "", "94")]) == (1, 0, 0)


def test_equal_range_collapses_to_single_article():
    assert counts([ref("193", "", "94")], [ref("193", "", "94-94")]) == (1, 0, 0)
    assert counts([ref("193", "", "geçici 1")], [ref("193", "", "geçici 1-1")]) == (1, 0, 0)


def test_subsumption_with_expanded_canonical_law():
    """Subsumption works seamlessly for newly canonical laws."""
    gold = [ref("492", "Harçlar Kanunu", "57")]
    cand = [ref("", "Harçlar Kanunu", ""), ref("492", "", "57")]
    # Extra law-only is excused, article matches
    assert counts(gold, cand) == (1, 0, 0)

