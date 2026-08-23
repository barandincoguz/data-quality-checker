"""Predict-agent loop: pull, predict, push. No network, no MLX."""

import pytest

from data_quality_checker.cli import build_parser
from data_quality_checker.errors import ConfigurationError, ContractError
from data_quality_checker.fingerprints import sha256_text
from data_quality_checker.predict_agent import (
    APPROVED_PRODUCTION_MODEL_FINGERPRINTS,
    AgentStats,
    HttpTransport,
    build_backend,
    resolve_token,
    run_agent,
)
from data_quality_checker.processing import PredictionResult

DOCUMENT = {
    "document_id": "d1",
    "pdf_text": "Vergi Usul Kanunu'nun 114 uncu maddesi.",
    "text_sha256": sha256_text("Vergi Usul Kanunu'nun 114 uncu maddesi."),
}
REFERENCE = {
    "kanun_no": "213",
    "kanun_ad": "Vergi Usul Kanunu",
    "madde": "114",
    "fikra": "",
    "bent": "",
    "source_text": "114 uncu maddesi",
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
    backend_id = "mlx-g0"
    model_fingerprint = next(iter(APPROVED_PRODUCTION_MODEL_FINGERPRINTS))

    def __init__(self, result=None, raises=None):
        self._result = result or PredictionResult(
            status="success",
            references=[REFERENCE],
            raw_output="[]",
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
    stats = run_agent(transport=transport, backend=FakeBackend(), once=True, sleep=lambda _s: None)
    assert transport.posted == []
    assert stats == AgentStats(pending=0, predicted=0, upserted=0, failed=0)


def test_successful_batch_posts_the_ingest_payload():
    transport = FakeTransport([[DOCUMENT]])
    backend = FakeBackend()
    stats = run_agent(
        transport=transport,
        backend=backend,
        batch_size=4,
        once=True,
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
        "model_fingerprint": FakeBackend.model_fingerprint,
        "text_sha256": DOCUMENT["text_sha256"],
        "source": "dqcheck_agent",
        "error": None,
        "operational": {
            "backend": "mlx-g0",
            "truncated": False,
            "latency_seconds": 1.0,
        },
    }
    assert stats.upserted == 1
    assert transport.requested_limits == [4]
    # The model reads exactly the text the platform sent.
    assert backend.calls[0]["text"] == DOCUMENT["pdf_text"]


def test_model_level_error_is_cached_so_the_document_is_not_retried_forever():
    result = PredictionResult(
        status="error",
        references=[],
        raw_output="",
        error="input_token_count=9001 exceeds pinned max_sequence_length",
        operational={"truncated": False},
    )
    transport = FakeTransport([[DOCUMENT]])
    run_agent(
        transport=transport,
        backend=FakeBackend(result=result),
        once=True,
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
    stats = run_agent(transport=transport, backend=FakeBackend(), once=True, sleep=lambda _s: None)
    assert stats.failed == 1
    assert stats.upserted == 0


def test_partial_consumer_success_is_counted_without_livelock_backoff():
    class PartialTransport(FakeTransport):
        def post_predictions(self, items):
            self.posted.append(items)
            return len(items) - 1

    second = {**DOCUMENT, "document_id": "d2"}
    transport = PartialTransport([[DOCUMENT, second]])
    logs = []

    stats = run_agent(
        transport=transport,
        backend=FakeBackend(),
        once=True,
        sleep=lambda _s: None,
        log=logs.append,
    )

    assert stats.upserted == 1
    assert stats.failed == 0
    assert any("skipped=1" in line for line in logs)


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


def test_poll_failure_is_retried_instead_of_escaping_the_daemon():
    class PollFailure(FakeTransport):
        def __init__(self):
            super().__init__([[]])
            self.calls = 0

        def get_pending(self, limit):
            self.calls += 1
            if self.calls == 1:
                raise OSError("503 Space is waking")
            return []

    delays = []
    stats = run_agent(
        transport=PollFailure(),
        backend=FakeBackend(),
        poll_seconds=30,
        sleep=delays.append,
        log=lambda _m: None,
        max_cycles=2,
    )

    assert stats.failed == 1
    assert delays == [60.0]


def test_bad_middle_document_does_not_starve_good_document_after_it():
    second = {
        "document_id": "d2",
        "pdf_text": "bad",
        "text_sha256": sha256_text("bad"),
    }
    third = {
        "document_id": "d3",
        "pdf_text": "good again",
        "text_sha256": sha256_text("good again"),
    }

    class SelectiveBackend(FakeBackend):
        def predict(self, document):
            if document["text"] == "bad":
                raise RuntimeError("deterministic bad document")
            return super().predict(document)

    transport = FakeTransport([[DOCUMENT, second, third]])
    stats = run_agent(
        transport=transport,
        backend=SelectiveBackend(),
        once=True,
        sleep=lambda _s: None,
        log=lambda _m: None,
    )

    assert [item["document_id"] for item in transport.posted[0]] == ["d1", "d3"]
    assert stats.predicted == 2
    assert stats.failed == 1


def test_hash_mismatch_is_cached_as_error_without_running_model():
    bad = {**DOCUMENT, "text_sha256": "0" * 64}
    backend = FakeBackend()
    transport = FakeTransport([[bad]])

    run_agent(
        transport=transport,
        backend=backend,
        once=True,
        sleep=lambda _s: None,
    )

    assert backend.calls == []
    (item,) = transport.posted[0]
    assert item["status"] == "error"
    assert item["text_sha256"] == sha256_text(DOCUMENT["pdf_text"])
    assert "text_sha256 mismatch" in item["error"]


def test_successful_nonempty_cycle_has_a_poll_floor_delay():
    delays = []
    run_agent(
        transport=FakeTransport([[DOCUMENT], []]),
        backend=FakeBackend(),
        poll_seconds=30,
        sleep=delays.append,
        log=lambda _m: None,
        max_cycles=2,
    )
    assert delays[0] > 0


def test_http_transport_requires_https():
    with pytest.raises(ConfigurationError, match="https"):
        HttpTransport(base_url="http://example.test", token="secret")


def test_fixture_backend_is_never_available_to_remote_agent():
    with pytest.raises(ConfigurationError, match="never permits fixture"):
        build_backend(  # type: ignore[arg-type]
            None,
            fake=True,
            allow_fixture=True,
        )


def test_remote_agent_fails_loudly_for_unapproved_model_fingerprint():
    class UnknownModel(FakeBackend):
        model_fingerprint = "f" * 64

    transport = FakeTransport([[DOCUMENT]])
    with pytest.raises(ConfigurationError, match="unapproved"):
        run_agent(
            transport=transport,
            backend=UnknownModel(),
            once=True,
            sleep=lambda _s: None,
            log=lambda _m: None,
        )

    assert transport.posted == []


@pytest.mark.parametrize("forbidden_flag", ["--fake-backend", "--allow-fixture-ingest"])
def test_predict_agent_cli_has_no_fake_backend_escape_hatch(forbidden_flag):
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "predict-agent",
                "--space-url",
                "https://example.test",
                forbidden_flag,
            ]
        )


def test_schema_overflow_is_cached_as_error_not_generation_truncation():
    oversized = {**REFERENCE, "source_text": "x" * 4_001}
    result = PredictionResult(
        status="success",
        references=[oversized],
        raw_output="[]",
        operational={"truncated": False, "finish_reason": "stop"},
    )
    transport = FakeTransport([[DOCUMENT]])

    run_agent(
        transport=transport,
        backend=FakeBackend(result=result),
        once=True,
        sleep=lambda _s: None,
    )

    (item,) = transport.posted[0]
    assert item["status"] == "error"
    assert item["references"] == []
    assert item["truncated"] is False
    assert item["operational"]["truncated"] is False
    assert "schema limits" in item["error"]


def test_http_transport_accepts_valid_partial_upsert_response(monkeypatch):
    transport = HttpTransport(base_url="https://example.test", token="secret")
    monkeypatch.setattr(
        transport,
        "_request",
        lambda *_args, **_kwargs: {"upserted": 1, "rejected": 0},
    )
    assert transport.post_predictions([{}, {}]) == 1


def test_http_transport_still_fails_closed_on_rejected_item(monkeypatch):
    transport = HttpTransport(base_url="https://example.test", token="secret")
    monkeypatch.setattr(
        transport,
        "_request",
        lambda *_args, **_kwargs: {"upserted": 1, "rejected": 1},
    )
    with pytest.raises(ContractError, match="rejected 1"):
        transport.post_predictions([{}, {}])


def test_backoff_exponent_cannot_overflow_after_long_outage():
    class AlwaysFails:
        def get_pending(self, _limit):
            raise OSError("offline")

    delays = []
    run_agent(
        transport=AlwaysFails(),
        backend=FakeBackend(),
        poll_seconds=30,
        sleep=delays.append,
        log=lambda _m: None,
        max_cycles=1_025,
    )
    assert len(delays) == 1_024
    assert max(delays) == 600.0
