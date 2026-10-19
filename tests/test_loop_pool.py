import pytest

from data_quality_checker import loop_pool


def _docs(n):
    return [
        {
            "doc_id": f"oid{i:04d}",
            "length_quartile": f"Q{i % 4 + 1}",
            "text_channel": "htmlText",
            "annotator": f"u{i % 3}",
            "reference_count": i % 7,
        }
        for i in range(n)
    ]


def test_four_full_rounds_and_the_rest_as_a_fifth():
    batches = loop_pool.deal_with_remainder(_docs(1103), full_rounds=4, size=250, seed=20260925)

    assert [len(b) for b in batches] == [250, 250, 250, 250, 103]
    flat = [d for b in batches for d in b]
    assert sorted(flat) == sorted(d["doc_id"] for d in _docs(1103))


def test_an_exact_pool_has_no_remainder_round():
    batches = loop_pool.deal_with_remainder(_docs(1000), full_rounds=4, size=250, seed=1)

    assert [len(b) for b in batches] == [250, 250, 250, 250]


def test_dealing_is_deterministic():
    assert loop_pool.deal_with_remainder(
        _docs(1103), full_rounds=4, size=250, seed=3
    ) == loop_pool.deal_with_remainder(_docs(1103), full_rounds=4, size=250, seed=3)


def test_human_references_parse_the_export_forms():
    assert loop_pool.human_references(
        {"current_references": "[{'madde': '5', 'source_text': 'x'}]"}
    ) == [{"madde": "5", "source_text": "x"}]
    assert loop_pool.human_references({"current_references": []}) == []
    with pytest.raises(Exception):
        loop_pool.human_references({"current_references": "{bozuk"})
