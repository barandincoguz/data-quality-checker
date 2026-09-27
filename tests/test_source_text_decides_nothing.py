"""The annotator's `source_text` decides nothing (user rule, 27 September 2026).

The annotators did not use the quote field as intended, so it enters no
comparison, routing, error attribution, sampling or channel choice. It may
still be shown to highlight the passage on the expert screen, nothing more.
"""

from __future__ import annotations

from data_quality_checker.hitl import ab_diff
from data_quality_checker.loop_metrics import attribute_errors, rater_agreement


def ref(madde: str, fikra: str = "", source: str = "") -> dict[str, str]:
    return {
        "kanun_no": "193",
        "kanun_ad": "Gelir Vergisi Kanunu",
        "madde": madde,
        "fikra": fikra,
        "bent": "",
        "source_text": source,
    }


def test_the_expert_screen_never_reports_a_quote_difference() -> None:
    rows = ab_diff([ref("94", "1", "bursiyerin alıntısı")], [ref("94", "2", "")])
    assert rows[0]["status"] == "differs"
    assert rows[0]["field_diffs"] == ["fikra"]
    same = ab_diff([ref("94", source="bir alıntı")], [ref("94", source="başka alıntı")])
    assert same[0]["status"] == "same" and same[0]["field_diffs"] == []


def test_error_attribution_ignores_the_quote() -> None:
    expert = [ref("94", "1", "uzmanın yazdığı")]
    annotator = [ref("94", "1", "metinde olmayan bir alıntı")]
    result = attribute_errors(expert=expert, annotator=annotator, model=None, judge=None)
    assert result["annotator"]["missed"] == [] and result["annotator"]["spurious"] == []
    assert result["annotator"]["detail_mismatch"] == []
    per_rater = rater_agreement([{"expert": expert, "annotator": annotator}])["per_rater"]
    assert per_rater["annotator"]["exact_set_agreement"] == 1.0
    assert "grounding" not in per_rater["annotator"]


def test_green_audit_sampling_is_not_stratified_on_quote_coverage(tmp_path) -> None:
    from test_hitl import fixture

    from data_quality_checker.judges import ensure_green_audit_plan
    from data_quality_checker.storage import Store

    config, _ = fixture(tmp_path, count=3)
    with Store(config.database_path) as store:
        plan = ensure_green_audit_plan(config=config, store=store, batch_id="batch")
    assert "evidence_coverage" not in plan["stratification_fields"]


def test_the_text_channel_is_chosen_without_reading_the_quotes() -> None:
    from data_quality_checker.preparation import _select_channel

    quoted = ref("5", source="213 sayılı Vergi Usul Kanununun 5 inci maddesi")
    pdf_covers_quote = {
        "pdfText": f"PDF metni {quoted['source_text']} devam",
        "htmlText": "<p>HTML metni, alıntı burada geçmiyor</p>",
    }
    _, channel, _, _ = _select_channel(pdf_covers_quote, [quoted])
    assert channel == "htmlText"
    _, channel, _, _ = _select_channel({"pdfText": "yalnız PDF"}, [quoted])
    assert channel == "pdfText"


def test_the_expert_is_not_shown_quote_warnings(tmp_path) -> None:
    import json

    from test_hitl import authenticated, fixture

    from data_quality_checker.storage import Store

    config, app = fixture(tmp_path)
    with Store(config.database_path) as store:
        doc_id = store.list_documents("batch")[0]["internal_doc_id"]
        store.connection.execute(
            "UPDATE documents SET warnings_json=? WHERE internal_doc_id=?",
            (
                json.dumps([{"code": "source_text_not_in_selected_channel", "reference_index": 0}]),
                doc_id,
            ),
        )
        store.connection.commit()
    client = app.test_client()
    authenticated(client)
    page = client.get(f"/review/{doc_id}").get_data(as_text=True)
    assert "source_text_not_in_selected_channel" not in page
    assert "hazırlama uyarısı" not in page
