from __future__ import annotations

from data_quality_checker.hitl import ab_diff, review_position


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
