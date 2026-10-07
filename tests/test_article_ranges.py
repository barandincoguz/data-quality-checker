"""An article range is every article in it (user decision, 7 Oct 2026).

"Mevzuatta “473 ila 475 inci maddeler” denildiğinde 473, 474 ve 475. maddelerin
tamamı kastedilir ... HER MADDE AYRI SATIR OLARAK TUTULUR". Applied everywhere a
list is compared or trained on: the engine (scores, routing, the screen) and the
round training data, gold included from M001 on. Before, "473 ila 475" was one
identity of its own: the gold training split kept 20 such rows (16 documents) and
taught the model to write them; a model row "473 ila 475" against 473, 474, 475
counted as three misses and one extra.
"""

from __future__ import annotations

from data_quality_checker.hitl import ab_diff
from data_quality_checker.normalization import expand_article_range
from data_quality_checker.reference_matching import compare_reference_lists, expand_ranges, reference_identities


def ref(no: str, md: str, quote: str = "") -> dict[str, str]:
    return {"kanun_no": no, "kanun_ad": "", "madde": md, "source_text": quote}


def test_a_closed_range_is_every_article_in_it() -> None:
    assert expand_article_range("473 ila 475") == ["473", "474", "475"]
    assert expand_article_range("473-475") == ["473", "474", "475"]
    assert len(expand_article_range("620 ila 645")) == 26


def test_a_range_ending_at_a_repeated_article_includes_it() -> None:
    assert expand_article_range("256 ila mükerrer 257") == ["256", "257", "mükerrer 257"]


def test_anything_else_is_not_a_range() -> None:
    for madde in ("473", "5 ila 3", "32/A ila 32/C", "geçici 1", "", "1 ila 400"):
        assert expand_article_range(madde) is None, madde


def test_expanding_a_list_keeps_order_fields_and_where_each_row_came_from() -> None:
    rows = expand_ranges([ref("6102", "473 ila 475", "q"), ref("193", "94")])
    assert [(r["kanun_no"], r["madde"]) for r in rows] == [("6102", "473"), ("6102", "474"), ("6102", "475"), ("193", "94")]
    assert rows[0]["source_text"] == "q" and rows[0]["expanded_from"] == "473 ila 475"
    assert "expanded_from" not in rows[3]


def test_the_engine_compares_a_range_with_its_articles() -> None:
    articles = [ref("6102", "473"), ref("6102", "474"), ref("6102", "475")]
    same = compare_reference_lists(articles, [ref("6102", "473 ila 475")])
    assert (same.tp, same.fp, same.fn) == (3, 0, 0)
    part = compare_reference_lists([ref("6102", "473 ila 475")], [ref("6102", "473")])
    assert (part.tp, part.fp, part.fn) == (1, 0, 2)
    assert reference_identities([ref("6102", "473 ila 475")]) == reference_identities(articles)


def test_the_screen_shows_a_range_as_its_articles() -> None:
    rows = ab_diff([ref("6102", "473 ila 475")], [ref("6102", "473"), ref("6102", "474")])
    assert [(r["core"]["madde"], r["status"]) for r in rows] == [("473", "same"), ("474", "same"), ("475", "only_a")]
    assert rows[2]["a"][0]["expanded_from"] == "473 ila 475"
