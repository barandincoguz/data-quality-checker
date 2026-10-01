"""Carrying the expert's R01 decisions from the archived loop (1 Oct 2026).

The five-field GT750 loop is archived untouched; the law-and-article loop starts
over on the same documents. The expert's decisions there are kept: projected to
the law and the article, with their history, original times and a mark of where
they came from -- and only when the two batches are provably the same documents.
"""

from __future__ import annotations

import json

import pytest
from test_judges import prepared_processed_fixture

from data_quality_checker.errors import GateBlocked, IntegrityError
from data_quality_checker.loop_carry import carry_decisions
from data_quality_checker.storage import Store

DEEP = {"kanun_no": "193", "kanun_ad": "", "madde": "94", "fikra": "1", "bent": "a", "source_text": "q"}
OTHER = {"kanun_no": "213", "kanun_ad": "", "madde": "30", "fikra": "2", "bent": "", "source_text": ""}


def _pair(tmp_path):
    for name in ("v1", "v2"):
        (tmp_path / name).mkdir()
    source = prepared_processed_fixture(tmp_path / "v1", count=3, references=[DEEP])
    target = prepared_processed_fixture(tmp_path / "v2", count=3, references=[DEEP])
    target.raw["loop"] = {}
    with Store(source.database_path) as store:
        ids = sorted(row["internal_doc_id"] for row in store.list_documents("batch"))
        for doc_id, action, final in ((ids[0], "accept_human", [DEEP]), (ids[1], "revise", [DEEP, dict(DEEP, fikra="2"), OTHER])):
            review = store.get_review("batch", doc_id)
            store.update_review_with_event(
                batch_id="batch", internal_doc_id=doc_id, expected_version=review["row_version"],
                status="finalized", action=action, final_references=final, reason=None, reviewer="expert",
                event_payload={"internal_doc_id": doc_id, "status": "finalized", "action": action,
                               "row_version": 1, "seconds_on_page": 42.0},
            )
    return source, target, ids


def _carry(source, target):
    return carry_decisions(target, batch_id="batch", source_database=source.database_path, source_label="gt750_loop")


def test_decisions_arrive_as_laws_and_articles_with_their_history(tmp_path) -> None:
    source, target, ids = _pair(tmp_path)
    manifest = _carry(source, target)
    assert manifest["carried_document_count"] == 2 and manifest["by_action"] == {"accept_human": 1, "revise": 1}
    with Store(target.database_path) as store:
        first = store.get_review("batch", ids[0])
        second = store.get_review("batch", ids[1])
        events = [json.loads(row["payload_json"]) for row in store.connection.execute(
            "SELECT payload_json FROM batch_events WHERE event_type='expert_review_updated' ORDER BY event_id")]
        times = [row[0] for row in store.connection.execute(
            "SELECT created_at_epoch FROM batch_events WHERE event_type='expert_review_updated' ORDER BY event_id")]
    with Store(source.database_path) as store:
        original_times = [row[0] for row in store.connection.execute(
            "SELECT created_at_epoch FROM batch_events WHERE event_type='expert_review_updated' ORDER BY event_id")]
    law_article = {"kanun_no": "193", "kanun_ad": "", "madde": "94", "source_text": "q"}
    assert (first["status"], first["action"], json.loads(first["final_references_json"])) == (
        "finalized", "accept_human", [law_article])
    # The two depths of 193/94 are one row; nothing else changes.
    assert json.loads(second["final_references_json"]) == [
        law_article, {"kanun_no": "213", "kanun_ad": "", "madde": "30", "source_text": ""}]
    assert times == original_times and len(events) == 2
    for event in events:
        assert event["carried_from"]["loop"] == "gt750_loop" and event["seconds_on_page"] == 42.0
        assert all("fikra" not in r and "bent" not in r for r in event["final_references"])
    assert store_unchanged(source, ids)


def store_unchanged(source, ids) -> bool:
    with Store(source.database_path) as store:
        return json.loads(store.get_review("batch", ids[0])["final_references_json"]) == [DEEP]


def test_carrying_twice_changes_nothing(tmp_path) -> None:
    source, target, _ = _pair(tmp_path)
    assert _carry(source, target) == _carry(source, target)


def test_nothing_is_carried_onto_a_different_document(tmp_path) -> None:
    source, target, ids = _pair(tmp_path)
    with Store(target.database_path) as store:
        store.connection.execute("UPDATE documents SET text_sha256='x' WHERE internal_doc_id=?", (ids[2],))
        store.connection.commit()
    with pytest.raises(IntegrityError, match="text_sha256"):
        _carry(source, target)


def test_nothing_is_carried_over_a_decision_already_made_here(tmp_path) -> None:
    source, target, ids = _pair(tmp_path)
    with Store(target.database_path) as store:
        review = store.get_review("batch", ids[1])
        store.update_review(batch_id="batch", internal_doc_id=ids[1], expected_version=review["row_version"],
                            status="finalized", action="accept_model", final_references=[], reason=None, reviewer="x")
    with pytest.raises(GateBlocked, match="already reviewed"):
        _carry(source, target)


def test_an_audited_green_document_whose_carried_list_differs_escalates(tmp_path) -> None:
    source, target, ids = _pair(tmp_path)
    manifest = _carry(source, target)
    # Every document is GREEN and audited here (the echo backend agrees with the
    # annotator); the revised list adds 213/30, so the agreed list was wrong.
    assert manifest["green_escalated"] == [ids[1]]
    assert (target.public_root / "batches" / "batch" / "green_escalation.json").is_file()
