"""The screen explains preparation warnings in plain Turkish instead of raw JSON.

User (6 Oct 2026): "review kısmında bu yazan sorunları da incele bu neden veriliyor
amaç ne ben bişey anlamıyorum daha iyi anlatsın" -- the screen showed
`[{"code": "reference_null_field", "field": "kanun_ad", "reference_index": 9}, ...]`.
Preparation records where the annotation export gave a field as null (it became
empty): harmless for the law name, but an empty article makes the row law-level
and an empty law number leaves only the name to match by. In R01: kanun_ad 46
rows / 32 documents, madde 55 / 40, kanun_no 6 / 5.
"""

from __future__ import annotations

from data_quality_checker.hitl import explain_warnings

HUMAN = [
    {"kanun_no": "193", "kanun_ad": "Gelir Vergisi Kanunu", "madde": "94", "source_text": ""},
    {"kanun_no": "193", "kanun_ad": "", "madde": "61", "source_text": ""},
    {"kanun_no": "213", "kanun_ad": "", "madde": "", "source_text": ""},
    {"kanun_no": "", "kanun_ad": "Harçlar Kanunu", "madde": "57", "source_text": ""},
]


def null(field: str, index: int) -> dict:
    return {"code": "reference_null_field", "field": field, "reference_index": index}


def test_an_empty_law_name_is_explained_as_harmless_with_its_rows() -> None:
    [note] = explain_warnings([null("kanun_ad", 1), null("kanun_ad", 2)], HUMAN)
    assert note["level"] == "info"
    assert "2. ve 3. satır" in note["text"]
    assert "193 · madde 61" in note["text"] and "213 · madde —" in note["text"]
    assert "etkilemez" in note["text"]


def test_an_empty_article_asks_for_attention() -> None:
    [note] = explain_warnings([null("madde", 2)], HUMAN)
    assert note["level"] == "attention"
    assert "3. satır" in note["text"] and "kanun düzeyinde" in note["text"]


def test_an_empty_law_number_asks_for_attention() -> None:
    [note] = explain_warnings([null("kanun_no", 3)], HUMAN)
    assert note["level"] == "attention"
    assert "Harçlar Kanunu · madde 57" in note["text"] and "adıyla" in note["text"]


def test_attention_comes_before_information() -> None:
    notes = explain_warnings([null("kanun_ad", 1), null("madde", 2)], HUMAN)
    assert [n["level"] for n in notes] == ["attention", "info"]


def test_quote_and_paragraph_fields_are_not_shown() -> None:
    assert explain_warnings([null("source_text", 0), null("fikra", 0), null("bent", 1),
                             {"code": "source_text_not_in_selected_channel", "reference_index": 0}], HUMAN) == []


def test_a_missing_field_reads_like_a_null_one() -> None:
    [note] = explain_warnings([{"code": "reference_missing_field", "field": "madde", "reference_index": 0}], HUMAN)
    assert note["level"] == "attention" and "1. satır" in note["text"]


def test_an_unfinished_or_missing_annotation_is_explained() -> None:
    [unfinished] = explain_warnings([{"code": "annotation_incomplete"}], HUMAN)
    assert unfinished["level"] == "attention" and "tamamlandı" in unfinished["text"]
    [missing] = explain_warnings([{"code": "current_references_missing_or_null"}], [])
    assert missing["level"] == "attention" and "liste" in missing["text"]


def test_an_unknown_warning_is_still_shown() -> None:
    [note] = explain_warnings([{"code": "something_new", "reference_index": 0}], HUMAN)
    assert "something_new" in note["text"]


def test_an_index_outside_the_list_does_not_break_the_page() -> None:
    [note] = explain_warnings([null("madde", 40)], HUMAN)
    assert "41. satır" in note["text"]
