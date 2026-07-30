from __future__ import annotations

from data_quality_checker.hitl import (
    ab_diff,
    evidence_segments,
    review_guidance,
    review_position,
    summarize_diff,
)


def _ref(no="213", ad="Vergi Usul Kanunu", madde="413", fikra="", bent="", src="k"):
    return {
        "kanun_no": no,
        "kanun_ad": ad,
        "madde": madde,
        "fikra": fikra,
        "bent": bent,
        "source_text": src,
    }


def test_ab_diff_marks_identical_reference_as_same() -> None:
    ref = _ref(src="213 sayılı Vergi Usul Kanununun 413. maddesi")
    rows = ab_diff([dict(ref)], [dict(ref)])

    assert len(rows) == 1
    assert rows[0]["status"] == "same"
    assert rows[0]["status_label"] == "Aynı"
    assert rows[0]["field_diffs"] == []
    assert rows[0]["core"]["kanun_no"] == "213"
    assert rows[0]["core"]["madde"] == "413"


def test_ab_diff_flags_only_a_and_only_b() -> None:
    a_only = _ref(madde="413")
    b_only = _ref(no="193", ad="Gelir Vergisi Kanunu", madde="94")
    rows = ab_diff([a_only], [b_only])

    by_status = {row["status"] for row in rows}
    assert by_status == {"only_a", "only_b"}
    only_a = next(r for r in rows if r["status"] == "only_a")
    assert only_a["a"] and not only_a["b"]
    assert only_a["status_label"] == "Yalnız insan anotasyonunda"


def test_ab_diff_reports_field_level_difference_for_matched_core() -> None:
    a = _ref(fikra="1", bent="a", src="metin A")
    b = _ref(fikra="2", bent="a", src="metin B")
    rows = ab_diff([a], [b])

    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "differs"
    assert set(row["field_diffs"]) == {"fikra", "source_text"}
    assert "bent" not in row["field_diffs"]


def test_ab_diff_differs_without_field_diffs_when_group_is_not_one_to_one() -> None:
    # Same core identity on both sides, but two refs on A vs one on B whose
    # full identities do not match as a multiset -> "differs" with no
    # per-field diffs (field_diffs is only defined for 1-to-1 groups).
    a1 = _ref(fikra="1", src="A birinci fıkra")
    a2 = _ref(fikra="2", src="A ikinci fıkra")
    b1 = _ref(fikra="1", src="B birinci fıkra")
    rows = ab_diff([a1, a2], [b1])

    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "differs"
    assert row["field_diffs"] == []
    assert len(row["a"]) == 2
    assert len(row["b"]) == 1


def test_ab_diff_preserves_first_appearance_order_across_a_then_b() -> None:
    first = _ref(madde="413")
    second = _ref(no="193", ad="Gelir Vergisi Kanunu", madde="94")
    rows = ab_diff([first], [second])

    assert [row["core"]["madde"] for row in rows] == ["413", "94"]


def test_ab_diff_handles_empty_candidates() -> None:
    assert ab_diff([], []) == []


def test_summarize_diff_counts_same_and_different_groups() -> None:
    rows = ab_diff(
        [_ref(madde="1"), _ref(madde="2")],
        [_ref(madde="1"), _ref(madde="3")],
    )

    assert summarize_diff(rows) == {
        "same_count": 1,
        "difference_count": 2,
        "total_count": 3,
    }


def test_review_guidance_explains_red_yellow_and_green_purpose() -> None:
    assert "Kanun veya madde" in review_guidance("RED")["title"]
    assert "fıkra, bent" in review_guidance("YELLOW")["explanation"]
    assert "kalite kontrol" in review_guidance("GREEN")["title"]


def test_evidence_segments_returns_single_plain_run_without_spans() -> None:
    assert evidence_segments("abc", []) == [{"text": "abc", "candidates": []}]


def test_evidence_segments_empty_text_returns_empty() -> None:
    assert evidence_segments("", []) == []


def test_evidence_segments_marks_a_single_span_and_leaves_context_plain() -> None:
    # text indices: 0123456789
    spans = [{"start": 2, "end": 5, "candidate": "A"}]
    segs = evidence_segments("0123456789", spans)

    assert segs == [
        {"text": "01", "candidates": []},
        {"text": "234", "candidates": ["A"]},
        {"text": "56789", "candidates": []},
    ]


def test_evidence_segments_splits_overlapping_a_and_b() -> None:
    spans = [
        {"start": 2, "end": 6, "candidate": "A"},
        {"start": 4, "end": 8, "candidate": "B"},
    ]
    segs = evidence_segments("0123456789", spans)

    assert segs == [
        {"text": "01", "candidates": []},
        {"text": "23", "candidates": ["A"]},
        {"text": "45", "candidates": ["A", "B"]},
        {"text": "67", "candidates": ["B"]},
        {"text": "89", "candidates": []},
    ]


def test_evidence_segments_covers_full_text_and_preserves_it() -> None:
    text = "213 sayılı Vergi Usul Kanununun 413. maddesi uyarınca"
    spans = [{"start": 0, "end": 10, "candidate": "A"}]
    segs = evidence_segments(text, spans)

    assert "".join(s["text"] for s in segs) == text


def test_review_position_is_one_based_with_total() -> None:
    queue = [
        {"internal_doc_id": "x"},
        {"internal_doc_id": "y"},
        {"internal_doc_id": "z"},
    ]
    assert review_position(queue, "x") == {"index": 1, "total": 3}
    assert review_position(queue, "z") == {"index": 3, "total": 3}


def test_review_position_returns_none_when_absent() -> None:
    assert review_position([{"internal_doc_id": "x"}], "missing") is None
