"""Tests for `judge_expert_metrics`'s judge-quality numbers.

These exercise the function directly against a bare `Store`, without running the
full annotate/process/judge pipeline, so each fixture can control exactly which
attempts, verdicts and blind mappings are on record. The response JSON files
mirror the shape a live judge run actually writes: `attempts`, `blind_mapping`,
`operational` and `result`, per the schema documented in the loop-metrics plan.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_quality_checker.fingerprints import sha256_file
from data_quality_checker.judges import judge_expert_metrics
from data_quality_checker.storage import Store

BATCH_ID = "batch"
MODEL = "test-judge"


def _make_store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "store.db")
    store.create_batch(batch_id=BATCH_ID, input_fingerprint="fp", config_fingerprint="cfg")
    return store


@pytest.fixture
def store(tmp_path: Path):
    instance = _make_store(tmp_path)
    try:
        yield instance
    finally:
        instance.close()


def _add_document(store: Store, doc_id: str) -> None:
    store.add_document(
        BATCH_ID,
        {
            "internal_doc_id": doc_id,
            "public_doc_id": doc_id,
            "raw_document_id": doc_id,
            "selected_channel": "pdf",
            "pdf_coverage": 1.0,
            "html_coverage": 0.0,
            "text": f"document text {doc_id}",
            "text_sha256": "0" * 64,
            "human_references": [],
        },
    )


def _finalize_review(store: Store, doc_id: str) -> None:
    review = store.get_review(BATCH_ID, doc_id)
    assert review is not None
    store.update_review(
        batch_id=BATCH_ID,
        internal_doc_id=doc_id,
        expected_version=review["row_version"],
        status="finalized",
        action="accept_human",
        final_references=[],
        reason=None,
        reviewer="fixture",
    )


def _record_judge_result(
    store: Store,
    tmp_path: Path,
    doc_id: str,
    *,
    verdict: str = "TIE",
    blind_mapping: str = "A=human,B=model",
    attempts: list[dict] | None = None,
    latency_seconds: float | None = 1.0,
    cost: float | None = 0.01,
    model: str = MODEL,
) -> None:
    """Persist a valid judge result plus the response file its metrics are read from.

    Mirrors what `_run_pilot_impl`/`_run_locked_judge_coverage` actually write: a
    DB row carrying `blind_mapping`/`verdict`/`retry_count`, and a JSON file at
    `response_path` carrying `attempts` and `operational`.
    """
    attempts = attempts if attempts is not None else [{"attempt": 1, "status": "valid"}]
    retry_count = max(0, len(attempts) - 1)
    operational: dict[str, float] = {}
    if latency_seconds is not None:
        operational["latency_seconds"] = latency_seconds
    if cost is not None:
        operational["cost"] = cost
    response_payload = {
        "schema_version": 1,
        "batch_id": BATCH_ID,
        "internal_doc_id": doc_id,
        "model": model,
        "blind_mapping": blind_mapping,
        "status": "valid",
        "result": {"verdict": verdict, "final_references": []},
        "attempts": attempts,
        "operational": operational,
        "error": None,
    }
    response_path = tmp_path / f"{doc_id}-{model}.json"
    response_path.write_text(json.dumps(response_payload), encoding="utf-8")
    store.persist_judge_result(
        batch_id=BATCH_ID,
        internal_doc_id=doc_id,
        model=model,
        blind_mapping=blind_mapping,
        status="valid",
        verdict=verdict,
        result={"verdict": verdict, "final_references": []},
        response_path=response_path,
        response_sha256=sha256_file(response_path),
        retry_count=retry_count,
        error=None,
    )


def test_status_is_pending_when_review_not_finalized(store: Store) -> None:
    _add_document(store, "doc-1")
    metrics = judge_expert_metrics(store, batch_id=BATCH_ID, model=MODEL, ids=["doc-1"])
    assert metrics == {"status": "pending"}


def test_fabricated_evidence_rate_counts_the_rejected_attempt_not_zero(
    store: Store, tmp_path: Path
) -> None:
    """The rate must reflect a real rejection, 1/2, not the old hardcoded 0.0."""
    _add_document(store, "doc-1")
    _finalize_review(store, "doc-1")
    _record_judge_result(
        store,
        tmp_path,
        "doc-1",
        attempts=[
            {
                "attempt": 1,
                "status": "invalid",
                "error": "fabricated_or_missing_evidence at references [0]",
            },
            {"attempt": 2, "status": "valid"},
        ],
    )
    metrics = judge_expert_metrics(store, batch_id=BATCH_ID, model=MODEL, ids=["doc-1"])
    assert metrics["status"] == "complete"
    assert metrics["total_attempts"] == 2
    assert metrics["fabricated_evidence_attempts"] == 1
    assert metrics["fabricated_evidence_rate"] == pytest.approx(0.5)
    assert metrics["fabricated_evidence_rate"] != 0.0
    assert metrics["documents_needing_retry"] == 1
    assert metrics["document_count"] == 1


def test_documents_needing_retry_only_counts_documents_that_retried(
    store: Store, tmp_path: Path
) -> None:
    for doc_id in ("doc-1", "doc-2"):
        _add_document(store, doc_id)
        _finalize_review(store, doc_id)
    _record_judge_result(store, tmp_path, "doc-1", attempts=[{"attempt": 1, "status": "valid"}])
    _record_judge_result(
        store,
        tmp_path,
        "doc-2",
        attempts=[
            {"attempt": 1, "status": "invalid", "error": "judge output is not JSON: boom"},
            {"attempt": 2, "status": "valid"},
        ],
    )
    metrics = judge_expert_metrics(store, batch_id=BATCH_ID, model=MODEL, ids=["doc-1", "doc-2"])
    assert metrics["document_count"] == 2
    assert metrics["documents_needing_retry"] == 1
    assert metrics["total_attempts"] == 3
    assert metrics["invalid_attempts"] == 1
    assert metrics["invalid_attempts_by_error_class"] == {"judge output is not JSON": 1}
    assert metrics["fabricated_evidence_attempts"] == 0
    assert metrics["fabricated_evidence_rate"] == 0.0


def test_unavailable_attempts_count_toward_total_but_not_invalid_breakdown(
    store: Store, tmp_path: Path
) -> None:
    """An unavailable provider is an outage, not a rejected output -- it must not
    inflate the invalid/fabrication breakdown, but it still happened."""
    _add_document(store, "doc-1")
    _finalize_review(store, "doc-1")
    _record_judge_result(
        store,
        tmp_path,
        "doc-1",
        attempts=[
            {"attempt": 1, "status": "unavailable", "error": "paywall"},
            {"attempt": 2, "status": "valid"},
        ],
    )
    metrics = judge_expert_metrics(store, batch_id=BATCH_ID, model=MODEL, ids=["doc-1"])
    assert metrics["total_attempts"] == 2
    assert metrics["invalid_attempts"] == 0
    assert metrics["invalid_attempts_by_error_class"] == {}
    assert metrics["fabricated_evidence_attempts"] == 0
    assert metrics["documents_needing_retry"] == 1


@pytest.mark.parametrize(
    ("blind_mapping", "verdict", "expected"),
    [
        ("A=human,B=model", "A", "annotator"),
        ("A=human,B=model", "B", "model"),
        ("A=model,B=human", "A", "model"),
        ("A=model,B=human", "B", "annotator"),
        ("A=human,B=model", "TIE", "TIE"),
        ("A=human,B=model", "NEITHER", "NEITHER"),
    ],
)
def test_verdict_distribution_resolves_raw_letters_through_blind_mapping(
    store: Store,
    tmp_path: Path,
    blind_mapping: str,
    verdict: str,
    expected: str,
) -> None:
    """Raw A/B is meaningless once blind_mapping randomises which side is which;
    only the resolved annotator/model/TIE/NEITHER counts should appear."""
    _add_document(store, "doc-1")
    _finalize_review(store, "doc-1")
    _record_judge_result(store, tmp_path, "doc-1", verdict=verdict, blind_mapping=blind_mapping)
    metrics = judge_expert_metrics(store, batch_id=BATCH_ID, model=MODEL, ids=["doc-1"])
    assert metrics["verdict_distribution"] == {
        "annotator": 0,
        "model": 0,
        "TIE": 0,
        "NEITHER": 0,
        expected: 1,
    }


def test_verdict_distribution_aggregates_across_documents(store: Store, tmp_path: Path) -> None:
    cases = [
        ("doc-1", "A=human,B=model", "A"),  # annotator
        ("doc-2", "A=human,B=model", "B"),  # model
        ("doc-3", "A=model,B=human", "A"),  # model
        ("doc-4", "TIE-mapping-does-not-matter", "TIE"),
        ("doc-5", "NEITHER-mapping-does-not-matter", "NEITHER"),
    ]
    for doc_id, blind_mapping, verdict in cases:
        _add_document(store, doc_id)
        _finalize_review(store, doc_id)
        _record_judge_result(store, tmp_path, doc_id, verdict=verdict, blind_mapping=blind_mapping)
    metrics = judge_expert_metrics(
        store, batch_id=BATCH_ID, model=MODEL, ids=[case[0] for case in cases]
    )
    assert metrics["verdict_distribution"] == {
        "annotator": 1,
        "model": 2,
        "TIE": 1,
        "NEITHER": 1,
    }
    assert metrics["document_count"] == 5


def test_latency_percentiles_and_mean(store: Store, tmp_path: Path) -> None:
    latencies = [1.0, 2.0, 3.0, 4.0, 5.0]
    doc_ids = []
    for index, latency in enumerate(latencies):
        doc_id = f"doc-{index}"
        doc_ids.append(doc_id)
        _add_document(store, doc_id)
        _finalize_review(store, doc_id)
        _record_judge_result(store, tmp_path, doc_id, latency_seconds=latency)
    metrics = judge_expert_metrics(store, batch_id=BATCH_ID, model=MODEL, ids=doc_ids)
    assert metrics["mean_latency_seconds"] == pytest.approx(3.0)
    assert metrics["latency_p50_seconds"] == pytest.approx(3.0)
    assert metrics["latency_p95_seconds"] == pytest.approx(5.0)


def test_existing_metric_keys_and_meanings_are_preserved(store: Store, tmp_path: Path) -> None:
    """The pre-existing keys must keep working: this task only adds to the dict."""
    _add_document(store, "doc-1")
    _finalize_review(store, "doc-1")
    _record_judge_result(store, tmp_path, "doc-1", cost=0.02, latency_seconds=2.5)
    metrics = judge_expert_metrics(store, batch_id=BATCH_ID, model=MODEL, ids=["doc-1"])
    assert metrics["status"] == "complete"
    assert metrics["document_count"] == 1
    assert metrics["expert_exact_set_agreement"] == 1.0
    assert metrics["expert_core_precision"] == 1.0
    assert metrics["expert_core_recall"] == 1.0
    assert metrics["expert_core_f1"] == 1.0
    assert metrics["mean_latency_seconds"] == pytest.approx(2.5)
    assert metrics["reported_cost"] == pytest.approx(0.02)
