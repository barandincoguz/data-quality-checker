from __future__ import annotations

import json

import pytest

from data_quality_checker.performance import (
    SCHEMA_VERSION,
    distribution,
    summarize_operational_records,
    write_performance_summary,
)


def _record(**overrides):
    base = {
        "input_tokens": 1000,
        "output_tokens": 200,
        "latency_seconds": 5.0,
        "truncated": False,
        "finish_reason": "stop",
        "generation_attempted": True,
    }
    base.update(overrides)
    return base


def test_distribution_reports_nearest_rank_percentiles():
    summary = distribution([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert summary["count"] == 10
    assert summary["min"] == 1.0
    assert summary["max"] == 10.0
    assert summary["mean"] == 5.5
    assert summary["p50"] == 5.0
    assert summary["p90"] == 9.0
    assert summary["p95"] == 10.0


def test_distribution_of_empty_sample_reports_zero_count():
    assert distribution([]) == {"count": 0}
    assert distribution([None, None]) == {"count": 0}


def test_percentiles_are_reported_because_the_mean_hides_the_tail():
    # The audited live batch had p95 at roughly 2.3x the mean; a summary that
    # only carried the mean would understate the slow tail.
    # Nearest-rank p95 is the 95th of 100 ordered values, so the slow tail has to
    # start at or before that rank to be reported: 94 fast and 6 slow requests.
    latencies = [1.0] * 94 + [50.0] * 6
    summary = distribution(latencies)
    assert summary["mean"] < 5.0
    assert summary["p95"] == 50.0
    assert summary["p50"] == 1.0


def test_summary_counts_documents_tokens_and_finish_reasons():
    records = [
        _record(),
        _record(output_tokens=4096, truncated=True, finish_reason="length"),
        _record(generation_attempted=False, latency_seconds=0.0, output_tokens=0),
    ]
    summary = summarize_operational_records(records, provenance={"universe": "test"})

    assert summary["schema_version"] == SCHEMA_VERSION
    assert summary["document_count"] == 3
    assert summary["provenance"]["universe"] == "test"
    assert summary["output_tokens"]["total"] == 200 + 4096 + 0
    assert summary["reliability"]["truncated_count"] == 1
    assert summary["reliability"]["generation_skipped_count"] == 1
    assert summary["reliability"]["finish_reason_counts"] == {"stop": 2, "length": 1}


def test_unmeasured_fields_are_named_not_zeroed():
    """An old run must never be mistaken for an infinitely fast one."""
    summary = summarize_operational_records([_record()], provenance={})
    assert set(summary["not_measured"]) == {
        "ttft_seconds",
        "prompt_tps",
        "generation_tps",
        "peak_memory_bytes",
    }
    assert "ttft_seconds" not in summary


def test_measured_performance_fields_are_summarized():
    records = [
        _record(ttft_seconds=0.4, prompt_tps=900.0, generation_tps=40.0, peak_memory_bytes=11_000),
        _record(ttft_seconds=0.6, prompt_tps=700.0, generation_tps=30.0, peak_memory_bytes=12_000),
    ]
    summary = summarize_operational_records(records, provenance={})
    assert "not_measured" not in summary
    assert summary["ttft_seconds"]["count"] == 2
    assert summary["generation_tps"]["mean"] == 35.0
    assert summary["peak_memory_bytes"]["max"] == 12_000


def test_throughput_uses_wall_clock_when_supplied():
    records = [_record(output_tokens=100, latency_seconds=10.0) for _ in range(6)]
    summary = summarize_operational_records(
        records, provenance={}, wall_clock_seconds=120.0
    )
    throughput = summary["throughput"]
    assert throughput["wall_clock_seconds"] == 120.0
    assert throughput["documents_per_hour"] == pytest.approx(180.0)
    assert throughput["output_tokens_per_second_wall_clock"] == pytest.approx(5.0)
    # Concurrency makes the sum of latencies differ from wall clock; both are kept.
    assert throughput["output_tokens_per_second_sum_of_latencies"] == pytest.approx(10.0)


def test_summary_is_written_atomically_and_reloadable(tmp_path):
    target = tmp_path / "nested" / "performance.json"
    summary = summarize_operational_records([_record()], provenance={"universe": "u"})
    write_performance_summary(target, summary)
    assert json.loads(target.read_text(encoding="utf-8"))["document_count"] == 1


def test_summary_carries_no_document_text_or_identity():
    """The summary is published outside the sensitive root, so it must be numbers only."""
    records = [_record(raw_output="gizli belge metni", annotator="ali")]
    summary = summarize_operational_records(records, provenance={"universe": "u"})
    serialized = json.dumps(summary, ensure_ascii=False)
    assert "gizli belge metni" not in serialized
    assert "ali" not in serialized
