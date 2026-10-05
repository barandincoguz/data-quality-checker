"""One row per reference in a list, by the comparison engine's own identity.

Since 1 Oct 2026 a reference is a law and an article. An annotator's rows that
differed only in fıkra or bent ("193/94/1", "193/94/2") became the same row,
and R01's annotator lists repeat a reference in 88 of 250 documents (5 Oct
2026). The engine has always counted a repeated row once; a training target
must say each reference once too, as the gold training data (GT v7/v8) does.
"""

from __future__ import annotations

import random

from data_quality_checker.reference_matching import distinct_rows, reference_identities


def ref(no: str, ad: str = "", md: str = "") -> dict[str, str]:
    return {"kanun_no": no, "kanun_ad": ad, "madde": md, "source_text": ""}


def test_a_repeated_reference_keeps_its_first_row() -> None:
    rows = [ref("193", "", "94"), ref("5520", "", "10"), ref("193", "", "94")]
    assert distinct_rows(rows) == [0, 1]


def test_spelling_does_not_make_a_second_reference() -> None:
    rows = [ref("193", "Gelir Vergisi Kanunu", "94"), ref("193", "", "94 üncü madde")]
    assert distinct_rows(rows) == [0]


def test_a_law_row_beside_an_article_of_that_law_goes() -> None:
    rows = [ref("193", "", ""), ref("193", "", "94")]
    assert distinct_rows(rows) == [1]


def test_a_law_row_alone_stays() -> None:
    assert distinct_rows([ref("193", "", ""), ref("5520", "", "10")]) == [0, 1]


def test_a_named_row_is_the_numbered_row_of_its_law_in_the_same_list() -> None:
    rows = [ref("193", "Gelir Vergisi Kanunu", "94"), ref("", "Gelir Vergisi Kanunu", "94")]
    assert distinct_rows(rows) == [0]


def test_the_references_claimed_never_change() -> None:
    rng = random.Random(7)
    pool = [ref("193", "", "94"), ref("193", "", ""), ref("5520", "", "10"), ref("", "Gelir Vergisi Kanunu", "94"),
            ref("213", "", "359"), ref("3065", "", "")]
    for _ in range(200):
        rows = [rng.choice(pool) for _ in range(rng.randint(0, 8))]
        kept = [rows[i] for i in distinct_rows(rows)]
        assert reference_identities(kept) == reference_identities(rows)
        assert len(reference_identities(kept)) == len(kept)  # no reference twice


def test_an_empty_list_has_no_rows() -> None:
    assert distinct_rows([]) == []
