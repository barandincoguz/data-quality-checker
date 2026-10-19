"""Every mention of a round document's reference, not only the best one (6 Oct 2026).

A training window lists a reference when one of its quotes is in the window's
text. Gold documents carry every quote the annotators wrote (`source_texts`); a
round document's reference used to carry the locator's single best span, so a
window citing it a second time ("aynı Kanunun 94 üncü maddesi") was trained to
say nothing. On the 500 gold training documents the single span lost 289 of
4,968 window-reference pairs; every mention whose law is written or inferred
lost 96 (user decision: round documents only, gold unchanged).
"""

from __future__ import annotations

from data_quality_checker.reference_locator import (
    attach_mention_spans,
    locate_reference_span,
    locate_reference_spans,
)

TEXT = (
    "193 sayılı Gelir Vergisi Kanununun 94 üncü maddesinde tevkifat düzenlenmiştir. "
    "Ödemeler bakımından aynı Kanunun 94 üncü maddesi uygulanır. "
    "3065 sayılı Katma Değer Vergisi Kanununun 94 üncü maddesi ise başka bir konudadır."
)


def ref(no: str, md: str = "") -> dict[str, str]:
    return {"kanun_no": no, "kanun_ad": "", "madde": md, "source_text": ""}


def test_the_best_span_comes_first() -> None:
    spans = locate_reference_spans(TEXT, ref("193", "94"))
    assert spans[0] == locate_reference_span(TEXT, ref("193", "94"))


def test_a_second_mention_of_the_same_law_and_article_is_kept() -> None:
    spans = locate_reference_spans(TEXT, ref("193", "94"))
    assert len(spans) == 2
    assert any(
        "aynı Kanunun 94 üncü maddesi" in TEXT[TEXT.find(s) - 20 : TEXT.find(s) + len(s)]
        for s in spans[1:]
    )


def test_another_law_s_article_of_the_same_number_is_not_a_mention() -> None:
    spans = locate_reference_spans(TEXT, ref("193", "94"))
    third = TEXT.index("3065 sayılı")
    assert all(TEXT.find(s) < third for s in spans)


def test_an_unplaceable_reference_has_no_spans() -> None:
    assert locate_reference_spans(TEXT, ref("5520", "10")) == []


def test_attach_fills_source_texts_and_keeps_the_best_as_source_text() -> None:
    located, refusals = attach_mention_spans(
        {"d1": {"text": TEXT, "references": [ref("193", "94"), ref("5520", "10")]}}
    )
    first, second = located["d1"]["references"]
    assert first["source_text"] == first["source_texts"][0] and len(first["source_texts"]) == 2
    assert second["source_text"] == "" and "source_texts" not in second
    assert refusals == [{"doc_id": "d1", "kanun_no": "5520", "kanun_ad": "", "madde": "10"}]


def test_attach_never_mutates_its_input() -> None:
    documents = {"d1": {"text": TEXT, "references": [ref("193", "94")]}}
    attach_mention_spans(documents)
    assert documents["d1"]["references"][0] == ref("193", "94")
