"""The two comparison engines must answer the identity question the same way.

The project runs two of them. `data_quality_checker.normalization` decides the
router's buckets, the judge's output contract and every round metric;
`benchmark.reference.matching` scores the model and produced the published
benchmarks. Having two *matching policies* is defensible -- set identity is
cheap and symmetric, optimal bipartite assignment maximises cardinality and
understands law-level subsumption. Having two answers to "are these the same
provision?" is not.

They drifted: a defect fixed in one was still live in the other, in a different
form, and each had gaps the other did not. This file pins the agreement so the
next drift fails a test instead of quietly moving a number.
"""

from __future__ import annotations

import pytest

from annotator.reference.common import normalize_reference as annotator_normalize
from benchmark.reference.matching import (
    canonical_article_key,
    canonicalize_reference,
    law_identity_compatible,
)
from data_quality_checker.normalization import compact_references, core_identity


def _row(kanun_no: str, kanun_ad: str, madde: str, **legacy: str) -> dict[str, str]:
    """A reference; `legacy` adds the paragraph fields a list stored before
    1 Oct 2026 still carries, which both engines must read past."""
    return {"kanun_no": kanun_no, "kanun_ad": kanun_ad, "madde": madde, "source_text": "s", **legacy}


def _loop_engine_says_same(left: dict[str, str], right: dict[str, str]) -> bool:
    return core_identity(compact_references([left])[0]) == core_identity(
        compact_references([right])[0]
    )


def _evaluator_says_same(left: dict[str, str], right: dict[str, str]) -> bool:
    prepared = [
        canonicalize_reference(annotator_normalize(dict(row, type="kanun_madde_referansi")))
        for row in (left, right)
    ]
    return law_identity_compatible(*prepared) and canonical_article_key(
        prepared[0]
    ) == canonical_article_key(prepared[1])


#: (label, left, right, are they the same provision)
CASES = [
    (
        "turkish uppercase",
        _row("193", "Gelir Vergisi Kanunu", "37"),
        _row("193", "GELİR VERGİSİ KANUNU", "37"),
        True,
    ),
    (
        "numbered-law prefix",
        _row("193", "Gelir Vergisi Kanunu", "37"),
        _row("193", "193 sayılı Gelir Vergisi Kanunu", "37"),
        True,
    ),
    (
        "blank name against a written one",
        _row("1163", "", "93"),
        _row("1163", "Kooperatifler Kanunu", "93"),
        True,
    ),
    (
        "number outranks a contradicting name",
        _row("213", "Vergi Usul Kanunu", "5"),
        _row("213", "Gelir Vergisi Kanunu", "5"),
        True,
    ),
    (
        "sub-part embedded in the article",
        _row("3065", "KDV Kanunu", "6", bent="b"),
        _row("3065", "KDV Kanunu", "6/b"),
        True,
    ),
    (
        "article suffix",
        _row("3065", "KDV Kanunu", "37 nci maddesinde"),
        _row("3065", "KDV Kanunu", "37"),
        True,
    ),
    (
        "ve müteakip",
        _row("213", "Vergi Usul Kanunu", "229"),
        _row("213", "Vergi Usul Kanunu", "229 ve müteakip maddelerinde"),
        True,
    ),
    (
        "dotless mükerrer",
        _row("213", "Vergi Usul Kanunu", "Mükerrer 242"),
        _row("213", "Vergi Usul Kanunu", "mukerrer 242"),
        True,
    ),
    (
        "dotless geçici",
        _row("193", "Gelir Vergisi Kanunu", "geçici 67"),
        _row("193", "Gelir Vergisi Kanunu", "gecici 67"),
        True,
    ),
    (
        "zero-padded number",
        _row("193", "Gelir Vergisi Kanunu", "37"),
        _row("0193", "Gelir Vergisi Kanunu", "37"),
        True,
    ),
    # -- spellings found in the GT750 data that name one article (27 Sep 2026) --
    ("qualifier typo geçiçi", _row("3218", "", "geçiçi 3"), _row("3218", "", "geçici 3"), True),
    (
        "qualifier typo mükrrer",
        _row("213", "", "mükrrer 242"),
        _row("213", "", "Mükerrer 242"),
        True,
    ),
    ("dotted capital GEÇİCİ", _row("193", "", "GEÇİCİ 67"), _row("193", "", "geçici 67"), True),
    ("geçici madde N", _row("193", "", "Geçici Madde 67"), _row("193", "", "geçici 67"), True),
    ("ek madde N", _row("213", "", "Ek Madde 1"), _row("213", "", "ek 1"), True),
    ("ve mütakip typo", _row("213", "", "229 ve mütakip"), _row("213", "", "229"), True),
    ("ve izleyen", _row("6098", "", "794 ve izleyen"), _row("6098", "", "794"), True),
    ("ve devamı", _row("6098", "", "603 ve devamı"), _row("6098", "", "603"), True),
    ("ve takip eden", _row("213", "", "229 ve takip eden maddeleri"), _row("213", "", "229"), True),
    ("değişik N is article N", _row("3065", "", "Değişik 3"), _row("3065", "", "3"), True),
    ("dash range", _row("6098", "", "466-469"), _row("6098", "", "466 ila 469"), True),
    ("ilâ range", _row("6098", "", "473 ilâ 475"), _row("6098", "", "473 ila 475"), True),
    # -- a reference is a law and an article (1 Oct 2026): parts are cut off --
    ("paragraph number in madde", _row("213", "", "17/4"), _row("213", "", "17"), True),
    ("paragraph and item in madde", _row("213", "", "17/4-a"), _row("213", "", "17"), True),
    ("parenthesised paragraph", _row("213", "", "10 (2)"), _row("213", "", "10"), True),
    ("dashed item", _row("5510", "", "4-a"), _row("5510", "", "4"), True),
    ("item of an inserted article", _row("5520", "", "5/A-1"), _row("5520", "", "5/A"), True),
    ("stored paragraph fields", _row("193", "", "94", fikra="1", bent="a"), _row("193", "", "94"), True),
    ("inserted pair stays", _row("193", "", "mükerrer 20/A-B"), _row("193", "", "mükerrer 20"), False),
    ("list mark before the article", _row("3065", "", "-1/1"), _row("3065", "", "1"), True),
    ("en-dash list mark", _row("3065", "", "– 13"), _row("3065", "", "13"), True),
    # -- engine audit (2 Oct 2026): spellings a model writes, read alike --
    ("apostrophe ordinal", _row("193", "", "94'üncü"), _row("193", "", "94"), True),
    ("curly apostrophe ordinal", _row("193", "", "41’inci maddesinde"), _row("193", "", "41"), True),
    ("apostrophe after geçici N", _row("193", "", "geçici 9'uncu"), _row("193", "", "geçici 9"), True),
    ("madde N", _row("193", "", "madde 94"), _row("193", "", "94"), True),
    ("N. madde", _row("193", "", "94. madde"), _row("193", "", "94"), True),
    ("md. N", _row("193", "", "md. 94"), _row("193", "", "94"), True),
    ("N md.", _row("193", "", "94 md."), _row("193", "", "94"), True),
    ("KDV bracketed article", _row("3065", "", "(17/4-r)"), _row("3065", "", "17"), True),
    ("bracketed article and its word", _row("4760", "", "(1/1-b) maddesinde"), _row("4760", "", "1"), True),
    ("spaced inserted letter", _row("5520", "", "32 / A"), _row("5520", "", "32/A"), True),
    ("Mük. N", _row("213", "", "Mük. 298"), _row("213", "", "mükerrer 298"), True),
    ("a dashed part is not a span", _row("213", "", "17-4-r"), _row("213", "", "17"), True),
    ("N sayılı in the number field", _row("193 sayılı", "", "94"), _row("193", "", "94"), True),
    ("No: in the number field", _row("No:193", "", "94"), _row("193", "", "94"), True),
    ("number only in the name", _row("", "6183 sayılı Kanun", "48"), _row("6183", "", "48"), True),
    ("name in the number field", _row("Gelir Vergisi Kanunu", "", "94"), _row("193", "Gelir Vergisi Kanunu", "94"), True),
    ("dotted abbreviation", _row("", "G.V.K.", "94"), _row("193", "Gelir Vergisi Kanunu", "94"), True),
    ("two decrees are two", _row("2004/6887", "", "1"), _row("2004/8301", "", "1"), False),
    ("a decree is not the year's law", _row("2007/13033", "", "1"), _row("2007", "", "1"), False),
    ("a bracketed number is not that article", _row("193", "", "(94)"), _row("193", "", "94"), False),
    # -- and the provisions that must stay apart --
    ("geçici N is not article N", _row("193", "", "geçici 3"), _row("193", "", "3"), False),
    ("ek N is not article N", _row("213", "", "ek 3"), _row("213", "", "3"), False),
    ("mükerrer N is not article N", _row("193", "", "mükerrer 20"), _row("193", "", "20"), False),
    ("geçici 3 is not geçici 4", _row("193", "", "geçiçi 3"), _row("193", "", "geçici 4"), False),
    (
        "ek madde A is not article 32",
        _row("5520", "Kurumlar Vergisi Kanunu", "32"),
        _row("5520", "Kurumlar Vergisi Kanunu", "32/A"),
        False,
    ),
    (
        "inserted 32/A is not a part of article 32",
        _row("5520", "Kurumlar Vergisi Kanunu", "32/A"),
        _row("5520", "Kurumlar Vergisi Kanunu", "32/a"),
        False,
    ),
    (
        "neighbouring articles",
        _row("213", "Vergi Usul Kanunu", "229"),
        _row("213", "Vergi Usul Kanunu", "230"),
        False,
    ),
    (
        "different laws",
        _row("193", "Gelir Vergisi Kanunu", "37"),
        _row("213", "Vergi Usul Kanunu", "37"),
        False,
    ),
    (
        "conflicting numbers",
        _row("213", "Vergi Usul Kanunu", "5"),
        _row("193", "Vergi Usul Kanunu", "5"),
        False,
    ),
]


@pytest.mark.parametrize(("label", "left", "right", "same"), CASES)
def test_both_engines_agree(label: str, left: dict, right: dict, same: bool) -> None:
    loop = _loop_engine_says_same(left, right)
    evaluator = _evaluator_says_same(left, right)
    assert loop == same, f"{label}: loop engine"
    assert evaluator == same, f"{label}: evaluator engine"


def test_both_engines_keep_an_inserted_article_through_a_second_pass() -> None:
    """A normalized "32/A" must still be article 32/A when normalized again --
    the case-folded "32/a" the loop engine once returned read as a part of
    article 32 on the second pass."""
    from data_quality_checker.normalization import normalize_reference

    for madde in ("32/A", "mükerrer 20/A-B", "geçici 1/A"):
        once = normalize_reference(_row("5520", "", madde))
        assert normalize_reference(once) == once, madde
        prepared = canonicalize_reference(annotator_normalize(dict(_row("5520", "", madde), type="kanun_madde_referansi")))
        assert canonicalize_reference(prepared) == prepared, madde
        assert canonical_article_key(prepared) != canonical_article_key(
            canonicalize_reference(annotator_normalize(dict(_row("5520", "", "32"), type="kanun_madde_referansi")))
        )


#: Every spelling above and the forms the stored predictions carried (2 Oct 2026).
FORMS = sorted({row["madde"] for _, left, right, _ in CASES for row in (left, right)} | {
    "(II/B-10)", "II/B-5.1", "I/A-1", "(17)", "(1)", "ekli (1) sayılı tablo", "null", "mükerrer", "VIII.",
    "(2) sayılı tarifede yazılı olanları", "30 Seri No.lu (D)", "-", "", "Geçici Madde 9", "20/A-B",
})


@pytest.mark.parametrize("madde", FORMS)
def test_both_engines_are_a_fixed_point(madde: str) -> None:
    from data_quality_checker.normalization import normalize_article, normalize_reference

    once = normalize_reference(_row("213", "", madde))
    assert normalize_reference(once) == once, madde
    assert normalize_article(once["madde"]) == once["madde"], madde
    prepared = canonicalize_reference(annotator_normalize(dict(_row("213", "", madde), type="kanun_madde_referansi")))
    assert canonicalize_reference(prepared) == prepared, madde


def test_an_unreadable_article_is_never_a_law_level_row() -> None:
    """"(17)" (a paragraph number) once normalized to an empty article in both
    engines: the row became a law-level claim and could match a law-level gold
    row of its law (engine audit, 2 Oct 2026)."""
    from benchmark.reference.matching import evaluate_doc_optimal
    from data_quality_checker.normalization import normalize_reference

    assert normalize_reference(_row("6824", "", "(17)"))["madde"]
    gold = [_row("6824", "", "")]
    tp, fp, fn, _, _ = evaluate_doc_optimal([_row("6824", "", "(17)")], gold)
    assert (tp, fp, fn) == (0, 1, 1)
    tp, fp, fn, _, _ = evaluate_doc_optimal([_row("6824", "", "")], gold)
    assert (tp, fp, fn) == (1, 0, 0)


def test_the_reference_policy_is_removed_from_both_sides(tmp_path) -> None:
    """VUK 213/413 is dropped from the gold and the predictions alike, so a gold
    row of it can never be charged as a miss (engine audit, 2 Oct 2026)."""
    import json

    from benchmark.reference.evaluate import evaluate_prediction_file, parse_ignored_law_articles

    gold = {1: [annotator_normalize(_row("213", "Vergi Usul Kanunu", "413")),
                annotator_normalize(_row("193", "", "94"))]}
    predictions = tmp_path / "p.json"
    for refs, expected in (([_row("213", "", "413 üncü maddesi"), _row("193", "", "94")], (1, 0, 0)),
                           ([_row("193", "", "94")], (1, 0, 0)),
                           ([_row("", "Vergi Usul Kanunu", "413"), _row("193", "", "94")], (1, 0, 0))):
        predictions.write_text(json.dumps([{"doc_id": 1, "status": "success", "references": refs}]))
        result = evaluate_prediction_file(predictions, gold, None, ignore_law_articles=parse_ignored_law_articles(["213:413"]))
        core = result["core_law_article_strict"]
        assert (core["tp"], core["fp"], core["fn"]) == expected
        assert result["ignored_law_articles"]["gold_rows_removed"] == 1
    predictions.write_text(json.dumps([{"doc_id": 1, "status": "success", "references": [_row("193", "", "94")]}]))
    plain = evaluate_prediction_file(predictions, gold, None)
    assert "ignored_law_articles" not in plain and plain["core_law_article_strict"]["fn"] == 1
