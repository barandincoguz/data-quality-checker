"""The GT750 loop's judge: Gemma 4, fixed in advance, no pilot, never shown to the expert.

User decision (29 Sep 2026): the pilot exam and auditing the judge were load on
the expert, and the judge's errors are measured anyway -- every judge verdict is
scored against the expert's final list (`loop-errors`). So in the loop the judge
is fixed beforehand (Gemma 4, local), runs on every document the expert reviews,
and is hidden from the expert on all of them: the measurement stays blind, and
the expert's queue loses the pilot documents.
"""

from __future__ import annotations

import json

import pytest
from test_judges import prepared_processed_fixture

from data_quality_checker.constants import JUDGE_MODEL_KEY
from data_quality_checker.errors import ContractError, GateBlocked
from data_quality_checker.hitl import create_hitl_app, review_queue
from data_quality_checker.judges import (
    FakeJudgeProvider,
    fix_loop_judge,
    lock_judge,
    run_judge_pilot,
)
from data_quality_checker.loop_bindings import write_error_report
from data_quality_checker.storage import Store

SECRET = "s" * 32
TOKEN = "t" * 32


def _ref(no: str, madde: str) -> dict[str, str]:
    return {"kanun_no": no, "kanun_ad": "", "madde": madde, "fikra": "", "bent": "", "source_text": ""}


def _loop_batch(tmp_path, *, buckets: dict[str, int]):
    """A processed batch under a loop config, routed into the given bucket counts."""
    count = sum(buckets.values())
    config = prepared_processed_fixture(tmp_path, count=count, references=[_ref("193", "94")])
    config.raw["loop"] = {}
    with Store(config.database_path) as store:
        ids = sorted(row["internal_doc_id"] for row in store.list_documents("batch"))
        routed, start = {}, 0
        for bucket, n in buckets.items():
            for doc_id in ids[start : start + n]:
                store.set_router_bucket("batch", doc_id, bucket)
                routed[doc_id] = bucket
            start += n
    return config, routed


def _audit_ids(config) -> set[str]:
    from data_quality_checker.judges import ensure_green_audit_plan

    with Store(config.database_path) as store:
        return set(ensure_green_audit_plan(config=config, store=store, batch_id="batch")["sample_internal_doc_ids"])


def _queue(config) -> list[str]:
    with Store(config.database_path) as store:
        return [row["internal_doc_id"] for row in review_queue(config=config, store=store, batch_id="batch")]


def test_the_loop_judge_is_gemma_4_and_nothing_else(tmp_path) -> None:
    config, _ = _loop_batch(tmp_path, buckets={"RED": 2})
    with pytest.raises(ContractError, match="Gemma"):
        fix_loop_judge(config=config, batch_id="batch", model="qwen3.5:397b")
    payload = fix_loop_judge(config=config, batch_id="batch")
    assert payload["model"] == JUDGE_MODEL_KEY and "gemma-4" in payload["model"]
    assert payload["pilot"] is False and payload["expert_sees_judge"] is False
    stored = json.loads((config.public_root / "batches" / "batch" / "judge_lock.json").read_text())
    assert stored == payload
    assert fix_loop_judge(config=config, batch_id="batch") == payload  # idempotent
    with Store(config.database_path) as store:
        events = [row[0] for row in store.connection.execute(
            "SELECT event_type FROM batch_events WHERE batch_id='batch'")]
    # No "judge_locked": nothing the expert decides is ever made after seeing the judge.
    assert "judge_fixed" in events and "judge_locked" not in events


def test_the_expert_queue_has_no_pilot_in_the_loop(tmp_path) -> None:
    """R01 already carries a pilot selection from before the decision; the queue
    ignores it: RED, YELLOW and the GREEN audit sample only."""
    config, routed = _loop_batch(tmp_path, buckets={"RED": 3, "YELLOW": 2, "GREEN": 40})
    pilot = sorted(d for d, b in routed.items() if b == "GREEN")[:5]
    (config.public_root / "batches" / "batch" / "judge_pilot_selection.json").write_text(
        json.dumps({"internal_doc_ids": pilot})
    )
    audit = _audit_ids(config)
    expected = {d for d, b in routed.items() if b in ("RED", "YELLOW")} | audit
    assert set(_queue(config)) == expected


def test_the_loop_refuses_the_pilot_and_the_old_lock(tmp_path) -> None:
    config, _ = _loop_batch(tmp_path, buckets={"RED": 2})
    with pytest.raises(GateBlocked, match="loop-judge"):
        run_judge_pilot(config=config, batch_id="batch", allow_external_judge=True,
                        provider=FakeJudgeProvider(), judge_models=(JUDGE_MODEL_KEY,))
    with pytest.raises(GateBlocked, match="loop-judge"):
        lock_judge(config=config, batch_id="batch", model=JUDGE_MODEL_KEY, reason="x")


def test_the_loop_judge_covers_every_document_the_expert_reviews(tmp_path) -> None:
    config, routed = _loop_batch(tmp_path, buckets={"RED": 2, "YELLOW": 1, "GREEN": 35})
    fix_loop_judge(config=config, batch_id="batch")
    summary = run_judge_pilot(config=config, batch_id="batch", allow_external_judge=True,
                              provider=FakeJudgeProvider())
    reviewed = set(_queue(config))
    with Store(config.database_path) as store:
        judged = {row["internal_doc_id"] for row in store.list_judge_results("batch", model=JUDGE_MODEL_KEY)
                  if row["status"] == "valid"}
    assert reviewed <= judged
    assert summary["required_document_count"] == len(reviewed) and summary["coverage_complete"]


def _client(config):
    app = create_hitl_app(config=config, batch_id="batch", testing=True, session_secret=SECRET, access_token=TOKEN)
    client = app.test_client()
    csrf = client.get("/api/session", headers={"Authorization": f"Bearer {TOKEN}"}).get_json()["csrf_token"]
    return client, csrf


def test_the_expert_never_sees_the_loop_judge(tmp_path) -> None:
    config, routed = _loop_batch(tmp_path, buckets={"RED": 2})
    fix_loop_judge(config=config, batch_id="batch")
    run_judge_pilot(config=config, batch_id="batch", allow_external_judge=True, provider=FakeJudgeProvider())
    doc_id = sorted(routed)[0]
    client, csrf = _client(config)
    page = client.get(f"/review/{doc_id}").get_data(as_text=True)
    for hidden in ("Hakemin önerisi", 'value="judge_override"', "Yardımcı görüşten başla", "Otomatik yardımcı görüş"):
        assert hidden not in page
    assert client.get(f"/api/documents/{doc_id}").get_json()["judge_suggestion"] is None
    response = client.post(f"/api/reviews/{doc_id}", json={"row_version": 0, "action": "judge_override"},
                           headers={"X-CSRF-Token": csrf})
    assert response.status_code >= 400
    with Store(config.database_path) as store:
        assert store.get_review("batch", doc_id)["status"] == "pending"


def test_release_needs_the_judge_on_every_reviewed_document(tmp_path) -> None:
    from data_quality_checker.release import release_batch

    config, routed = _loop_batch(tmp_path, buckets={"RED": 1, "GREEN": 31})
    fix_loop_judge(config=config, batch_id="batch")
    audit = sorted(_audit_ids(config))
    red = [d for d, b in routed.items() if b == "RED"]
    # Judge only the RED document, then finalize everything the expert must see.
    run_judge_pilot(config=config, batch_id="batch", allow_external_judge=True, provider=FakeJudgeProvider())
    with Store(config.database_path) as store:
        store.connection.execute(
            "DELETE FROM judge_results WHERE batch_id='batch' AND internal_doc_id=?", (audit[0],)
        )
        store.connection.commit()
        for doc_id in red + audit:
            review = store.get_review("batch", doc_id)
            store.update_review(batch_id="batch", internal_doc_id=doc_id, expected_version=review["row_version"],
                                status="finalized", action="accept_human",
                                final_references=json.loads(store.get_document("batch", doc_id)["human_references_json"]),
                                reason=None, reviewer="fixture")
    with pytest.raises(GateBlocked, match="judge"):
        release_batch(config=config, batch_id="batch")


def test_every_expert_decision_in_the_loop_is_blind_to_the_judge(tmp_path) -> None:
    config, routed = _loop_batch(tmp_path, buckets={"RED": 3})
    fix_loop_judge(config=config, batch_id="batch")
    run_judge_pilot(config=config, batch_id="batch", allow_external_judge=True, provider=FakeJudgeProvider())
    with Store(config.database_path) as store:
        for doc_id in routed:
            review = store.get_review("batch", doc_id)
            store.update_review(batch_id="batch", internal_doc_id=doc_id, expected_version=review["row_version"],
                                status="finalized", action="accept_human",
                                final_references=json.loads(store.get_document("batch", doc_id)["human_references_json"]),
                                reason=None, reviewer="fixture")
    report = write_error_report(config, round_index=1, batch_id="batch", generation="G0",
                                judge_model=JUDGE_MODEL_KEY, path=tmp_path / "errors.json")
    assert report["judge_visible_to_expert"] is False
    assert report["before_judge_lock"]["document_count"] == 3
    assert report["after_judge_lock"]["document_count"] == 0
    assert report["all_finalized"]["per_rater"]["judge"]["available_document_count"] == 3


def test_marking_a_loop_round_judged_needs_the_judge_run_not_the_pilot(tmp_path) -> None:
    from types import SimpleNamespace

    from data_quality_checker import loop_ops

    public = tmp_path / "public"
    batch = public / "batches" / "R01"
    batch.mkdir(parents=True)
    (batch / "judge_pilot_summary.json").write_text("{}")
    config = SimpleNamespace(public_root=public, sensitive_root=tmp_path / "s", raw={"loop": {}})
    with pytest.raises(GateBlocked, match="judge_production_summary"):
        loop_ops.mark_stage(config, round_index=1, stage="judged")


def test_loop_judge_fixes_runs_and_marks_the_round_judged(tmp_path, monkeypatch) -> None:
    from data_quality_checker import loop_ops
    from data_quality_checker.fingerprints import sha256_file
    from data_quality_checker.loop_rounds import (
        advance_round,
        resume_round,
        rounds_state_dir,
        write_round_state,
    )

    config = prepared_processed_fixture(tmp_path, count=3, generation="M000", references=[_ref("193", "94")])
    config.raw["loop"] = {}
    with Store(config.database_path) as store:
        for row in store.list_documents("batch"):
            store.set_router_bucket("batch", row["internal_doc_id"], "RED")
    monkeypatch.setattr(loop_ops, "batch_id_for", lambda k: "batch")
    state_dir = rounds_state_dir(config)
    state = resume_round(state_dir, 1)
    marker = tmp_path / "routed.json"
    marker.write_text("{}")
    for stage in ("predicted", "routed"):
        state = advance_round(state, to=stage, artifact_sha256=sha256_file(marker))
    write_round_state(state_dir, state)

    summary = loop_ops.judge_round(config, round_index=1, fake_backend=True)
    assert summary["coverage_complete"] and summary["required_document_count"] == 3
    assert resume_round(state_dir, 1).stage == "judged"
    assert json.loads((config.public_root / "batches" / "batch" / "judge_lock.json").read_text())["model"] == JUDGE_MODEL_KEY


def test_a_quarantined_document_is_reviewed_by_the_expert(tmp_path) -> None:
    """R01 d000112 (3 Oct 2026): the model failed, only the annotator's list was left
    -- nothing agreed with it, so the expert reviews it; left out, the batch never
    had a truth for it."""
    config, routed = _loop_batch(tmp_path, buckets={"RED": 1, "QUARANTINE": 1, "GREEN": 2})
    quarantined = next(d for d, b in routed.items() if b == "QUARANTINE")
    red = next(d for d, b in routed.items() if b == "RED")
    queue = _queue(config)
    assert quarantined in queue and red in queue
    assert queue.index(red) < len(queue) and set(queue) >= {quarantined, red} | _audit_ids(config)
