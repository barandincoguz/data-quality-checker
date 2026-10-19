"""The expert screen highlights every place each compared reference is cited.

Before (6 Oct 2026, seen on the screen): a highlight came only from the
annotator's quote, found once with `str.find` -- so the model's references were
never highlighted (its output has no quote), a reference both sides agreed on
was not highlighted when the annotator left the quote empty, only the first
mention was marked, and a quote not written verbatim marked nothing.
Now every row of the comparison is placed by the locator at each mention whose
law is written or inferred, the annotator's verbatim quotes are marked too, and
a row cited nowhere says so.
"""

from __future__ import annotations

from data_quality_checker.hitl import ab_diff, evidence_segments, row_highlights, segment_kind
from data_quality_checker.reference_locator import mention_offsets

TEXT = (
    "193 sayılı Gelir Vergisi Kanununun 94 üncü maddesinde tevkifat düzenlenmiştir. "
    "Ödemeler bakımından aynı Kanunun 94 üncü maddesi uygulanır. "
    "3065 sayılı Katma Değer Vergisi Kanununun 94 üncü maddesi ise başka bir konudadır. "
    "Ücret Gelir Vergisi Kanununun 61 inci maddesinde tanımlanmıştır."
)


def ref(no: str, md: str, quote: str = "", ad: str = "") -> dict[str, str]:
    return {"kanun_no": no, "kanun_ad": ad, "madde": md, "source_text": quote}


def test_every_mention_of_a_reference_is_found_and_another_law_s_article_is_not() -> None:
    offsets = mention_offsets(TEXT, ref("193", "94"))
    assert len(offsets) == 2
    kdv = TEXT.index("3065 sayılı")
    assert all(end <= kdv for _, end in offsets)


def test_a_reference_cited_nowhere_has_no_mentions() -> None:
    assert mention_offsets(TEXT, ref("5520", "10")) == []


def test_a_row_only_the_model_has_is_highlighted() -> None:
    rows = ab_diff(
        [ref("193", "94", "193 sayılı Gelir Vergisi Kanununun 94 üncü maddesinde")],
        [ref("193", "94"), ref("193", "61")],
    )
    spans, counts = row_highlights(TEXT, rows)
    model_only = next(i for i, r in enumerate(rows) if r["status"] == "only_b")
    assert counts[model_only] == 1
    assert any(s["candidate"] == str(model_only) for s in spans)


def test_a_row_both_agree_on_is_highlighted_without_a_quote() -> None:
    rows = ab_diff([ref("193", "61")], [ref("193", "61")])
    spans, counts = row_highlights(TEXT, rows)
    assert counts == [1] and spans


def test_the_annotator_s_verbatim_quote_is_marked_too() -> None:
    quote = "Ödemeler bakımından"
    rows = ab_diff([ref("193", "94", quote)], [])
    spans, _ = row_highlights(TEXT, rows)
    start = TEXT.index(quote)
    assert any(s["start"] == start and s["end"] == start + len(quote) for s in spans)


def test_a_quote_not_in_the_text_marks_nothing_but_the_reference_is_still_placed() -> None:
    rows = ab_diff([ref("193", "61", "Gelir Vergisi Kanunu'nun 61. maddesi")], [])
    spans, counts = row_highlights(TEXT, rows)
    assert counts == [1]


def test_a_row_cited_nowhere_counts_zero() -> None:
    rows = ab_diff([], [ref("5520", "10")])
    spans, counts = row_highlights(TEXT, rows)
    assert counts == [0] and spans == []


def test_segments_carry_the_rows_that_cover_them() -> None:
    rows = ab_diff([ref("193", "94")], [ref("193", "94"), ref("193", "61")])
    spans, _ = row_highlights(TEXT, rows)
    segments = evidence_segments(TEXT, spans)
    assert "".join(s["text"] for s in segments) == TEXT
    covered = {c for s in segments for c in s["candidates"]}
    assert covered == {"0", "1"}


def test_the_colour_of_a_mark_follows_the_rows_it_covers() -> None:
    assert segment_kind(["same"]) == "both"
    assert segment_kind(["only_a"]) == "a"
    assert segment_kind(["only_b"]) == "b"
    assert segment_kind(["only_a", "only_b"]) == "mixed"
    assert segment_kind(["only_a", "same"]) == "both"


def test_a_law_only_row_beside_its_article_is_shown_as_no_difference_not_as_same() -> None:
    """The engine excuses it (extra, not a claim); the screen must not call a
    one-sided row 'Aynı' nor colour it as cited by both sides."""
    from data_quality_checker.hitl import display_status

    rows = ab_diff([ref("193", ""), ref("193", "94")], [ref("193", "94")])
    law_only = next(r for r in rows if not r["core"]["madde"])
    assert law_only["status"] == "same"
    assert display_status(law_only) == "excused"
    assert display_status(next(r for r in rows if r["core"]["madde"] == "94")) == "same"
    assert segment_kind(["excused"]) == "neutral"
    assert segment_kind(["excused", "only_b"]) == "b"
    assert segment_kind(["excused", "same"]) == "both"


def test_a_locator_failure_costs_the_highlight_not_the_page(monkeypatch) -> None:
    import data_quality_checker.hitl as hitl

    def broken(text, reference):
        raise ValueError("unexpected reference shape")

    monkeypatch.setattr(hitl, "mention_offsets", broken)
    rows = ab_diff([ref("193", "94", "Ödemeler bakımından")], [ref("193", "61")])
    spans, counts = row_highlights(TEXT, rows)
    assert counts == [0, 0]
    assert any(
        s["start"] == TEXT.index("Ödemeler bakımından") for s in spans
    )  # the quote still marks
