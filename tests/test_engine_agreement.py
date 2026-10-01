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
