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


def _row(
    kanun_no: str, kanun_ad: str, madde: str, fikra: str = "", bent: str = ""
) -> dict[str, str]:
    return {
        "kanun_no": kanun_no,
        "kanun_ad": kanun_ad,
        "madde": madde,
        "fikra": fikra,
        "bent": bent,
        "source_text": "s",
    }


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
    # -- and the provisions that must stay apart --
    (
        "ek madde A is not article 32",
        _row("5520", "Kurumlar Vergisi Kanunu", "32"),
        _row("5520", "Kurumlar Vergisi Kanunu", "32/A"),
        False,
    ),
    (
        "ek madde A is not bent a",
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
