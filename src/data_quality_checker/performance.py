"""Run-level inference performance summary for the public artifact tree.

Accuracy alone cannot settle a local-versus-cloud choice, so every run publishes
latency, token, throughput and resource counters next to its scores. See
`docs/agents/local-inference-metrics.md` for the reporting standard.

This summary carries numbers and configuration only — never document text,
annotator identity or raw model output — so it is safe outside the sensitive
root, which is exactly where the trade-off tables need it.
"""

from __future__ import annotations

import json
import math
import statistics
from typing import Any, Iterable, Sequence

from .atomic import write_json_atomic

SCHEMA_VERSION = 1
PERCENTILES = (50, 90, 95, 99)


def _percentile(ordered: Sequence[float], percentile: float) -> float:
    """Nearest-rank percentile; deterministic and dependency-free."""
    if not ordered:
        raise ValueError("percentile of an empty sample")
    rank = max(1, min(len(ordered), math.ceil(percentile / 100.0 * len(ordered))))
    return float(ordered[rank - 1])


def distribution(samples: Iterable[float]) -> dict[str, Any]:
    values = sorted(float(sample) for sample in samples if sample is not None)
    if not values:
        return {"count": 0}
    summary: dict[str, Any] = {
        "count": len(values),
        "mean": round(statistics.fmean(values), 4),
        "min": round(values[0], 4),
        "max": round(values[-1], 4),
        "total": round(sum(values), 4),
    }
    for percentile in PERCENTILES:
        summary[f"p{percentile}"] = round(_percentile(values, percentile), 4)
    return summary


def summarize_operational_records(
    records: Iterable[dict[str, Any]],
    *,
    provenance: dict[str, Any],
    wall_clock_seconds: float | None = None,
) -> dict[str, Any]:
    """Build the publishable performance summary from per-document counters.

    `records` are the `operational` dictionaries persisted per prediction. Fields
    that a given run did not measure stay absent rather than being coerced to
    zero, so an old run is never mistaken for a fast one.
    """
    rows = list(records)
    latency = [row.get("latency_seconds") for row in rows]
    output_tokens = [row.get("output_tokens") for row in rows]
    input_tokens = [row.get("input_tokens") for row in rows]

    latency_total = sum(value for value in latency if value is not None)
    output_total = sum(value for value in output_tokens if value is not None)

    finish_reasons: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("finish_reason") or "unknown")
        finish_reasons[reason] = finish_reasons.get(reason, 0) + 1

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "document_count": len(rows),
        "provenance": provenance,
        "latency_seconds": distribution(latency),
        "output_tokens": distribution(output_tokens),
        "input_tokens": distribution(input_tokens),
        "reliability": {
            "finish_reason_counts": finish_reasons,
            "truncated_count": sum(1 for row in rows if row.get("truncated")),
            "generation_skipped_count": sum(
                1 for row in rows if row.get("generation_attempted") is False
            ),
        },
    }

    for field in ("ttft_seconds", "prompt_tps", "generation_tps"):
        measured = [row.get(field) for row in rows if row.get(field) is not None]
        if measured:
            summary[field] = distribution(measured)
        else:
            summary.setdefault("not_measured", []).append(field)

    peak_memory = [row.get("peak_memory_bytes") for row in rows if row.get("peak_memory_bytes")]
    if peak_memory:
        summary["peak_memory_bytes"] = {"max": int(max(peak_memory))}
    else:
        summary.setdefault("not_measured", []).append("peak_memory_bytes")

    throughput: dict[str, Any] = {}
    if latency_total > 0:
        throughput["output_tokens_per_second_sum_of_latencies"] = round(output_total / latency_total, 3)
        throughput["seconds_per_document_mean"] = round(latency_total / max(1, len(rows)), 3)
    if wall_clock_seconds:
        throughput["wall_clock_seconds"] = round(float(wall_clock_seconds), 3)
        throughput["documents_per_hour"] = round(len(rows) / (float(wall_clock_seconds) / 3600.0), 2)
        throughput["output_tokens_per_second_wall_clock"] = round(output_total / float(wall_clock_seconds), 3)
    if throughput:
        summary["throughput"] = throughput

    return summary


def write_performance_summary(path, summary: dict[str, Any]) -> dict[str, Any]:
    write_json_atomic(path, summary)
    return summary


def load_operational_records_from_sqlite(database_path, batch_id: str) -> list[dict[str, Any]]:
    """Read persisted per-document counters for an existing batch, read-only."""
    import sqlite3

    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "select operational_json from predictions where batch_id = ?", (batch_id,)
        ).fetchall()
    finally:
        connection.close()
    return [json.loads(row[0]) for row in rows if row[0]]
