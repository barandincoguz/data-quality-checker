"""Predict-agent loop: pull, predict, push. No network, no MLX."""
import pytest

from data_quality_checker.errors import ConfigurationError
from data_quality_checker.predict_agent import (
    AgentStats,
    resolve_token,
    run_agent,
)
from data_quality_checker.processing import PredictionResult

DOCUMENT = {
    "document_id": "d1",
    "pdf_text": "Vergi Usul Kanunu'nun 114 uncu maddesi.",
    "text_sha256": "a" * 64,
}
REFERENCE = {
    "kanun_no": "213", "kanun_ad": "Vergi Usul Kanunu", "madde": "114",
    "fikra": "", "bent": "", "source_text": "114 uncu maddesi",
}


class FakeTransport:
    def __init__(self, batches):
        self._batches = list(batches)
        self.posted = []
        self.requested_limits = []

    def get_pending(self, limit):
        self.requested_limits.append(limit)
        return self._batches.pop(0) if self._batches else []

    def post_predictions(self, items):
        self.posted.append(items)
        return len(items)


class FakeBackend:
    model_fingerprint = "fake-fingerprint"

    def __init__(self, result=None, raises=None):
        self._result = result or PredictionResult(
            status="success", references=[REFERENCE], raw_output="[]",
            operational={"truncated": False, "latency_seconds": 1.0},
        )
        self._raises = raises
        self.calls = []

    def predict(self, document):
        self.calls.append(document)
        if self._raises is not None:
            raise self._raises
        return self._result


def test_resolve_token_requires_a_non_empty_environment_variable():
    assert resolve_token("TOKEN", {"TOKEN": "abc"}) == "abc"
    with pytest.raises(ConfigurationError):
        resolve_token("TOKEN", {})
    with pytest.raises(ConfigurationError):
        resolve_token("TOKEN", {"TOKEN": "   "})


def test_empty_pending_posts_nothing():
    transport = FakeTransport([[]])
    stats = run_agent(
        transport=transport, backend=FakeBackend(), once=True, sleep=lambda _s: None
    )
    assert transport.posted == []
    assert stats == AgentStats(pending=0, predicted=0, upserted=0, failed=0)


def test_successful_batch_posts_the_ingest_payload():
    transport = FakeTransport([[DOCUMENT]])
    backend = FakeBackend()
    stats = run_agent(
        transport=transport, backend=backend, batch_size=4, once=True,
        sleep=lambda _s: None,
    )
    (batch,) = transport.posted
    (item,) = batch
    assert item == {
        "document_id": "d1",
        "generation": "G0",
        "status": "success",
        "references": [REFERENCE],
        "truncated": False,
        "model_fingerprint": "fake-fingerprint",
        "text_sha256": "a" * 64,
        "error": None,
        "operational": {"truncated": False, "latency_seconds": 1.0},
    }
    assert stats.upserted == 1
    assert transport.requested_limits == [4]
    # The model reads exactly the text the platform sent.
    assert backend.calls[0]["text"] == DOCUMENT["pdf_text"]


def test_model_level_error_is_cached_so_the_document_is_not_retried_forever():
    result = PredictionResult(
        status="error", references=[], raw_output="",
        error="input_token_count=9001 exceeds pinned max_sequence_length",
        operational={"truncated": False},
    )
    transport = FakeTransport([[DOCUMENT]])
    run_agent(
        transport=transport, backend=FakeBackend(result=result), once=True,
        sleep=lambda _s: None,
    )
    (batch,) = transport.posted
    assert batch[0]["status"] == "error"
    assert batch[0]["references"] == []


def test_environment_failure_posts_nothing_and_never_fabricates_a_prediction():
    transport = FakeTransport([[DOCUMENT]])
    stats = run_agent(
        transport=transport,
        backend=FakeBackend(raises=RuntimeError("Metal allocator died")),
        once=True,
        sleep=lambda _s: None,
    )
    assert transport.posted == []
    assert stats.failed == 1
    assert stats.predicted == 0


def test_post_failure_is_counted_and_does_not_crash_the_loop():
    class ExplodingTransport(FakeTransport):
        def post_predictions(self, items):
            raise OSError("connection reset")

    transport = ExplodingTransport([[DOCUMENT]])
    stats = run_agent(
        transport=transport, backend=FakeBackend(), once=True, sleep=lambda _s: None
    )
    assert stats.failed == 1
    assert stats.upserted == 0


def test_loop_backs_off_after_a_failed_batch():
    delays = []
    transport = FakeTransport([[DOCUMENT], []])
    run_agent(
        transport=transport,
        backend=FakeBackend(raises=RuntimeError("boom")),
        poll_seconds=30.0,
        once=False,
        sleep=delays.append,
        log=lambda _m: None,
        max_cycles=2,
    )
    assert delays and delays[0] == 60.0
