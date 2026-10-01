"""The loop's three error sources -- annotator, model, judge -- scored against the expert.

The paper's error table (`Journal/experiments/dq_loop/DESIGN.md` §5.2) has one
column per source: missed, spurious and P/R/F1. A reference is a law and an
article of it (1 Oct 2026); the old "detail" row (right article, wrong
paragraph) and the grounding row are gone. The expert screen must let the expert give any of
the three sources' answer as the truth, and the judge's pilot exam must reach
the expert at all.
"""

from __future__ import annotations

import json

from test_judges import prepared_processed_fixture

from data_quality_checker.hitl import create_hitl_app, review_queue
from data_quality_checker.judges import FakeJudgeProvider, lock_judge, run_judge_pilot
from data_quality_checker.loop_bindings import write_error_report
from data_quality_checker.loop_metrics import attribute_errors, expert_workload, rater_agreement
from data_quality_checker.normalization import core_identity, normalize_reference
from data_quality_checker.storage import Store

SECRET = "s" * 32
TOKEN = "t" * 32
JUDGE = "gemma-4-31b-it-optiq-4bit"


def ref(no: str, madde: str, source: str = "", **legacy: str) -> dict[str, str]:
    return {"kanun_no": no, "kanun_ad": "", "madde": madde, "source_text": source, **legacy}


def test_a_right_article_is_right_whatever_paragraph_a_stored_row_names() -> None:
    """Rows stored before 1 Oct 2026 still carry paragraph fields: the same law
    and article is correct whatever they say, and no detail error exists."""
    result = attribute_errors(
        expert=[ref("193", "51", fikra="1"), ref("193", "94")],
        annotator=[ref("193", "51", fikra="2"), ref("193", "94")],
        model=[ref("193", "51/1"), ref("193", "94")],
        judge=None,
    )
    for name in ("annotator", "model"):
        assert result[name]["missed"] == [] and result[name]["spurious"] == []
        assert result[name]["correct"] == sorted(
            {core_identity(normalize_reference(ref("193", m))) for m in ("51", "94")}
        )
        assert "detail_mismatch" not in result[name]
    assert result["judge"]["available"] is False


def test_rater_agreement_has_no_detail_errors() -> None:
    result = rater_agreement(
        [
            {
                "expert": [ref("193", "51", fikra="1")],
                "annotator": [ref("193", "51", fikra="2")],
                "model": [ref("193", "51")],
                "judge": [ref("193", "51", fikra="1")],
            }
        ]
    )
    for name in ("annotator", "model", "judge"):
        assert result["per_rater"][name]["f1"] == 1.0
        assert "detail_errors" not in result["per_rater"][name]


def test_workload_counts_a_judge_pilot_document_as_required_review() -> None:
    records = [
        {"router_bucket": "GREEN", "in_judge_pilot": True, "status": "pending"},
        {"router_bucket": "GREEN", "status": "pending"},
    ]
    result = expert_workload(records)
    assert result["documents_requiring_review_count"] == 1
    assert result["judge_pilot_only_review_count"] == 1


def _green_batch_with_pilot(tmp_path, *, count: int):
    config = prepared_processed_fixture(
        tmp_path, count=count, references=[ref("193", "94", source="Güvenli belge metni")]
    )
    run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        provider=FakeJudgeProvider(),
        judge_models=(JUDGE,),
    )
    return config


def _set_buckets(config, buckets: dict[str, str]) -> None:
    with Store(config.database_path) as store:
        for doc_id, bucket in buckets.items():
            store.set_router_bucket("batch", doc_id, bucket)


def test_every_judge_pilot_document_is_in_the_expert_queue_and_comes_first(tmp_path) -> None:
    """A GREEN pilot document outside the GREEN audit sample used to be absent
    from the queue, so `judge-lock` (which needs all 60 pilot documents
    finalized) could never open. R01 had 12 such documents."""
    config = prepared_processed_fixture(tmp_path, count=80)
    with Store(config.database_path) as store:
        ids = [row["internal_doc_id"] for row in store.list_documents("batch")]
    _set_buckets(config, {doc_id: "GREEN" for doc_id in ids})
    run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        provider=FakeJudgeProvider(),
        judge_models=(JUDGE,),
    )
    pilot = json.loads(
        (config.public_root / "batches" / "batch" / "judge_pilot_selection.json").read_text()
    )["internal_doc_ids"]
    with Store(config.database_path) as store:
        queue = [
            row["internal_doc_id"]
            for row in review_queue(config=config, store=store, batch_id="batch")
        ]
    assert set(pilot) <= set(queue)
    assert set(queue[: len(pilot)]) == set(pilot)


def _finalize(config, doc_ids, *, action="accept_human") -> None:
    with Store(config.database_path) as store:
        for doc_id in doc_ids:
            review = store.get_review("batch", doc_id)
            store.update_review(
                batch_id="batch",
                internal_doc_id=doc_id,
                expected_version=review["row_version"],
                status="finalized",
                action=action,
                final_references=json.loads(
                    store.get_document("batch", doc_id)["human_references_json"]
                ),
                reason=None,
                reviewer="fixture",
            )


def _client(config):
    app = create_hitl_app(
        config=config, batch_id="batch", testing=True, session_secret=SECRET, access_token=TOKEN
    )
    client = app.test_client()
    csrf = client.get("/api/session", headers={"Authorization": f"Bearer {TOKEN}"}).get_json()[
        "csrf_token"
    ]
    return client, csrf


def test_the_expert_can_give_the_locked_judges_answer_as_the_truth(tmp_path) -> None:
    config = prepared_processed_fixture(
        tmp_path, count=62, references=[ref("193", "94", source="Güvenli belge metni")]
    )
    with Store(config.database_path) as store:
        ids = [row["internal_doc_id"] for row in store.list_documents("batch")]
    _set_buckets(config, {doc_id: "RED" for doc_id in ids})
    run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        provider=FakeJudgeProvider(),
        judge_models=(JUDGE,),
    )
    pilot = json.loads(
        (config.public_root / "batches" / "batch" / "judge_pilot_selection.json").read_text()
    )["internal_doc_ids"]
    rest = [doc_id for doc_id in ids if doc_id not in pilot]

    # Before the lock the judge is hidden: the pilot is its exam.
    client, _ = _client(config)
    page = client.get(f"/review/{pilot[0]}").get_data(as_text=True)
    assert "Hakemin önerisi doğru" not in page

    _finalize(config, pilot)
    lock_judge(config=config, batch_id="batch", model=JUDGE, reason="fixture")
    run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        provider=FakeJudgeProvider(),
        judge_models=(JUDGE,),
    )

    client, csrf = _client(config)
    page = client.get(f"/review/{rest[0]}").get_data(as_text=True)
    for label in (
        "İnsan anotasyonu doğru",
        "Model çıktısı doğru",
        "Hakemin önerisi doğru",
        "referansları düzelt",
        "ertele",
    ):
        assert label in page
    assert 'value="judge_override"' in page

    response = client.post(
        f"/api/reviews/{rest[0]}",
        json={"row_version": 0, "action": "judge_override"},
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 200, response.get_json()
    with Store(config.database_path) as store:
        review = store.get_review("batch", rest[0])
        judge = json.loads(store.get_judge_result("batch", rest[0], JUDGE)["result_json"])
    assert review["status"] == "finalized" and review["action"] == "judge_override"
    assert [r["madde"] for r in json.loads(review["final_references_json"])] == [
        r["madde"] for r in judge["final_references"]
    ]


def test_error_report_separates_decisions_made_before_and_after_the_judge_lock(tmp_path) -> None:
    config = prepared_processed_fixture(
        tmp_path, count=62, references=[ref("193", "94", source="Güvenli belge metni")]
    )
    with Store(config.database_path) as store:
        ids = [row["internal_doc_id"] for row in store.list_documents("batch")]
    _set_buckets(config, {doc_id: "RED" for doc_id in ids})
    run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        provider=FakeJudgeProvider(),
        judge_models=(JUDGE,),
    )
    pilot = json.loads(
        (config.public_root / "batches" / "batch" / "judge_pilot_selection.json").read_text()
    )["internal_doc_ids"]
    rest = [doc_id for doc_id in ids if doc_id not in pilot]
    _finalize(config, pilot)
    lock_judge(config=config, batch_id="batch", model=JUDGE, reason="fixture")
    run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        provider=FakeJudgeProvider(),
        judge_models=(JUDGE,),
    )
    _finalize(config, rest[:1], action="judge_override")

    path = tmp_path / "errors.json"
    report = write_error_report(
        config, round_index=1, batch_id="batch", generation="G0", judge_model=JUDGE, path=path
    )
    assert json.loads(path.read_text()) == report
    assert report["reviews"] == {"finalized": 61, "deferred": 0, "pending": 1}
    assert report["before_judge_lock"]["document_count"] == 60
    assert report["after_judge_lock"]["document_count"] == 1
    assert report["all_finalized"]["document_count"] == 61
    assert report["all_finalized"]["per_rater"]["judge"]["available_document_count"] == 61
    assert report["expert_workload"]["action_distribution"]["judge_override"] == 1
    assert report["judge_locked_at_epoch"] is not None
