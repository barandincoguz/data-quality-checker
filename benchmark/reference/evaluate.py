"""Benchmark evaluator for Gemini-style reference annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from annotator.reference.common import (
    CANONICAL_LAW_BY_NO,
    CORE_FIELDS,
    compact_law_references,
    canonical_law_name_by_no,
    normalize_kanun_adi,
    normalize_kanun_no,
    normalize_reference,
)
from benchmark.reference.matching import (
    empty_optimal_diagnostics as _empty_optimal_diagnostics,
    evaluate_doc_optimal,
    optimal_diagnostics_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_GROUND_TRUTH_DIR = (
    PROJECT_ROOT / "data" / "ground_truth" / "gt_v3_triangulated_2026-05-15" / "validated"
)
DEFAULT_DOCS_PATH = PROJECT_ROOT / "data" / "test_data.json"
PRIMARY_METRIC_NAME = "core_law_article_strict"
PRIMARY_METRIC_FIELDS = ("kanun_no", "kanun_ad", "madde")
DOCWISE_CORE_ACCURACY_NAME = "docwise_core_accuracy"
DEFAULT_DOCWISE_THRESHOLD = 1.0
SUPPORTED_GT_MODES = ("approved_only", "approved_plus_draft")
REFERENCE_POSTPROCESS_LEGACY = "legacy"
REFERENCE_POSTPROCESS_CANONICAL_SET = "canonical_set"
SUPPORTED_REFERENCE_POSTPROCESS_MODES = (
    REFERENCE_POSTPROCESS_LEGACY,
    REFERENCE_POSTPROCESS_CANONICAL_SET,
)
CORE_REFERENCE_VIEW_ROW = "row"
CORE_REFERENCE_VIEW_CORE_SET = "core_set"
SUPPORTED_CORE_REFERENCE_VIEW_MODES = (
    CORE_REFERENCE_VIEW_ROW,
    CORE_REFERENCE_VIEW_CORE_SET,
)
MATCHING_ENGINE_GREEDY = "greedy"
MATCHING_ENGINE_OPTIMAL = "optimal"
SUPPORTED_MATCHING_ENGINES = (
    MATCHING_ENGINE_GREEDY,
    MATCHING_ENGINE_OPTIMAL,
)
GENERIC_LAW_NAMES = {"", "kanun", "kanun hükmünde kararname"}
LAW_NAME_TO_CANONICAL_NO = {
    normalize_kanun_adi(law_name, kanun_no=law_no): law_no
    for law_no, law_name in CANONICAL_LAW_BY_NO.items()
    if law_name and normalize_kanun_adi(law_name, kanun_no=law_no).lower() not in GENERIC_LAW_NAMES
}


@dataclass
class Metrics:
    """Precision/recall/F1 container."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


@dataclass
class PredictionLoadResult:
    """Loaded predictions and quality diagnostics."""

    predictions: dict[int, list[dict[str, str]]]
    status_counts: dict[str, int]
    duplicate_doc_ids: list[int]
    error_doc_ids: list[int]
    total_rows: int


@dataclass
class GroundTruthLoadResult:
    """Loaded ground truth references plus audit metadata."""

    gold: dict[int, list[dict[str, str]]]
    audit: dict[str, Any]


@dataclass
class ErrorBucket:
    """Grouped mismatch bucket for diagnostic reporting."""

    bucket_type: str
    entity_type: str
    count: int
    sample_doc_ids: list[int]
    wrong_fields: list[str] = dc_field(default_factory=list)


@dataclass
class DocLevelMetrics:
    """Document-level metric distribution summary."""

    total_docs: int
    perfect_f1_count: int
    zero_f1_count: int
    min_f1: float
    max_f1: float
    mean_f1: float
    median_f1: float
    p25_f1: float
    p75_f1: float
    worst_docs: list[dict]
    per_doc: list[dict] | None = None


@dataclass
class DocwiseCoreAccuracy:
    """Document-level pass/fail accuracy over core law/article matches."""

    threshold: float
    accuracy: float
    passed_doc_count: int
    failed_doc_count: int
    total_docs: int
    per_doc: list[dict] | None = None


@dataclass
class PartialMatchStats:
    """Statistics about near-miss partial matches."""

    total_unmatched_pairs: int
    avg_overlap_score: float
    overlap_distribution: dict[int, int]
    near_miss_count: int


@dataclass(frozen=True)
class CoreReference:
    """Reference prepared for core law/article evaluation."""

    ref: dict[str, str]
    kind: str


def compute_metrics(tp: int, fp: int, fn: int) -> Metrics:
    """Compute precision, recall and F1 safely."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return Metrics(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)


def load_json_array(path: Path) -> list[dict[str, Any]]:
    """Load a JSON array and validate shape."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def parse_cli_bool(value: str | bool | None) -> bool:
    """Parse flexible CLI boolean values."""
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def normalize_ground_truth_mode(mode: str) -> str:
    """Normalize user-facing GT mode values to supported internal values."""
    normalized = str(mode).strip().lower()
    if normalized not in SUPPORTED_GT_MODES:
        raise ValueError(f"Unsupported ground truth mode: {mode}")
    return normalized


def validate_docwise_threshold(value: float) -> float:
    """Validate document-wise accuracy threshold."""
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--docwise-threshold must be between 0.0 and 1.0")
    return threshold


def normalize_reference_postprocess_mode(mode: str) -> str:
    """Normalize reference postprocess mode values."""
    normalized = str(mode or REFERENCE_POSTPROCESS_LEGACY).strip().lower()
    if normalized not in SUPPORTED_REFERENCE_POSTPROCESS_MODES:
        raise ValueError(f"Unsupported reference postprocess mode: {mode}")
    return normalized


def normalize_core_reference_view_mode(mode: str) -> str:
    """Normalize core metric reference view values."""
    normalized = str(mode or CORE_REFERENCE_VIEW_ROW).strip().lower()
    if normalized not in SUPPORTED_CORE_REFERENCE_VIEW_MODES:
        raise ValueError(f"Unsupported core reference view: {mode}")
    return normalized


def normalize_matching_engine(engine: str) -> str:
    """Normalize core matching engine values."""
    normalized = str(engine or MATCHING_ENGINE_GREEDY).strip().lower()
    if normalized not in SUPPORTED_MATCHING_ENGINES:
        raise ValueError(f"Unsupported matching engine: {engine}")
    return normalized


def include_drafts_for_ground_truth_mode(mode: str) -> bool:
    """Return whether the GT mode includes draft rows."""
    return normalize_ground_truth_mode(mode) == "approved_plus_draft"


def load_docs_universe(path: Path | None) -> list[int] | None:
    """Load expected doc ids from document source file."""
    if path is None:
        return None
    docs = load_json_array(path)
    doc_ids: list[int] = []
    for doc in docs:
        doc_id = doc.get("doc_id")
        if isinstance(doc_id, int):
            doc_ids.append(doc_id)
    return sorted(set(doc_ids))


def load_doc_ids_file(path: Path | None) -> list[int] | None:
    """Load doc ids from JSON array file."""
    if path is None:
        return None
    payload = load_json_array(path)
    doc_ids: list[int] = []
    for item in payload:
        if isinstance(item, int):
            doc_ids.append(item)
            continue
        if isinstance(item, dict) and isinstance(item.get("doc_id"), int):
            doc_ids.append(item["doc_id"])
    return sorted(set(doc_ids))


def compute_docs_fingerprint(doc_ids: list[int]) -> str:
    """Build deterministic fingerprint from evaluation doc ids."""
    normalized = ",".join(str(doc_id) for doc_id in sorted(set(doc_ids)))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_json_report(path: Path) -> dict[str, Any]:
    """Load JSON report payload."""
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON report payload: {path}")
    return payload


def _normalize_annotation_status(raw_status: Any) -> str:
    """Normalize row status values used by annotation files."""
    normalized = str(raw_status or "approved").strip().lower()
    return normalized or "approved"


def load_ground_truth_bundle(
    directory: Path,
    *,
    include_drafts: bool = False,
) -> GroundTruthLoadResult:
    """Load per-document ground truth annotations plus audit metadata."""
    if not directory.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {directory}")

    gold: dict[int, list[dict[str, str]]] = {}
    row_status_counts: Counter[str] = Counter()
    approved_doc_ids: list[int] = []
    draft_only_doc_ids: list[int] = []
    mixed_doc_ids: list[int] = []
    empty_doc_ids: list[int] = []

    included_statuses = {"approved"}
    if include_drafts:
        included_statuses.add("draft")

    for file_path in sorted(directory.glob("doc_*.json")):
        try:
            doc_id = int(file_path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue

        with file_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        raw_refs = payload.get("references", [])
        doc_row_statuses: Counter[str] = Counter()
        references = []
        for raw_ref in raw_refs:
            status = _normalize_annotation_status(raw_ref.get("status", "approved"))
            row_status_counts[status] += 1
            doc_row_statuses[status] += 1
            if status not in included_statuses:
                continue
            references.append(normalize_reference(raw_ref))

        approved_count = doc_row_statuses.get("approved", 0)
        draft_count = doc_row_statuses.get("draft", 0)
        total_doc_rows = sum(doc_row_statuses.values())
        if total_doc_rows == 0:
            empty_doc_ids.append(doc_id)
        elif approved_count > 0 and draft_count > 0:
            mixed_doc_ids.append(doc_id)
            approved_doc_ids.append(doc_id)
        elif approved_count > 0:
            approved_doc_ids.append(doc_id)
        elif draft_count > 0:
            draft_only_doc_ids.append(doc_id)

        gold[doc_id] = references

    audit = {
        "ground_truth_mode": "approved_plus_draft" if include_drafts else "approved_only",
        "included_statuses": sorted(included_statuses),
        "row_status_counts": dict(row_status_counts),
        "approved_doc_count": len(approved_doc_ids),
        "draft_only_doc_count": len(draft_only_doc_ids),
        "mixed_doc_count": len(mixed_doc_ids),
        "empty_doc_count": len(empty_doc_ids),
        "included_reference_count": sum(len(refs) for refs in gold.values()),
        "approved_doc_ids": approved_doc_ids,
        "draft_only_doc_ids": draft_only_doc_ids,
        "mixed_doc_ids": mixed_doc_ids,
        "empty_doc_ids": empty_doc_ids,
    }
    return GroundTruthLoadResult(gold=gold, audit=audit)


def load_ground_truth(directory: Path, *, include_drafts: bool = False) -> dict[int, list[dict[str, str]]]:
    """Load per-document ground truth annotations from `data/annotations` style files."""
    return load_ground_truth_bundle(directory, include_drafts=include_drafts).gold


def load_predictions(path: Path) -> PredictionLoadResult:
    """Load predicted annotations from baseline output file."""
    payload = load_json_array(path)
    predictions: dict[int, list[dict[str, str]]] = {}
    status_counter: Counter[str] = Counter()
    duplicate_doc_ids: list[int] = []
    error_doc_ids: list[int] = []

    for item in payload:
        doc_id = item.get("doc_id")
        if not isinstance(doc_id, int):
            continue

        status = str(item.get("status", "success")).strip().lower() or "success"
        status_counter[status] += 1
        if status == "error":
            error_doc_ids.append(doc_id)

        if doc_id in predictions:
            duplicate_doc_ids.append(doc_id)

        # A repeated doc_id keeps the last block; the earlier rows are dropped.
        # No shipped prediction file contains duplicates, and `duplicate_doc_ids`
        # is reported so the loss can never pass unnoticed.
        references = [normalize_reference(ref) for ref in (item.get("references") or [])]
        predictions[doc_id] = references

    return PredictionLoadResult(
        predictions=predictions,
        status_counts=dict(status_counter),
        duplicate_doc_ids=sorted(set(duplicate_doc_ids)),
        error_doc_ids=sorted(set(error_doc_ids)),
        total_rows=len(payload),
    )


def _empty_postprocess_summary(mode: str) -> dict[str, Any]:
    """Initialize reference postprocess diagnostics."""
    return {
        "mode": mode,
        "doc_count": 0,
        "input_reference_count": 0,
        "output_reference_count": 0,
        "removed_reference_count": 0,
        "compact_summary": {
            "dropped_generic_due_to_specific": 0,
            "collapsed_generic_only_duplicates": 0,
            "collapsed_same_reference_duplicates": 0,
            "removed_existing_rows": 0,
            "added_new_rows": 0,
        },
        "changed_doc_ids": [],
    }


def postprocess_reference_maps(
    reference_maps: dict[int, list[dict[str, str]]],
    *,
    mode: str = REFERENCE_POSTPROCESS_LEGACY,
) -> tuple[dict[int, list[dict[str, str]]], dict[str, Any]]:
    """Apply the selected document-level reference postprocess view."""
    normalized_mode = normalize_reference_postprocess_mode(mode)
    summary = _empty_postprocess_summary(normalized_mode)
    processed: dict[int, list[dict[str, str]]] = {}

    for doc_id, refs in reference_maps.items():
        normalized_refs = [normalize_reference(ref) for ref in refs]
        summary["doc_count"] += 1
        summary["input_reference_count"] += len(normalized_refs)

        if normalized_mode == REFERENCE_POSTPROCESS_CANONICAL_SET:
            output_refs, compact_summary = compact_law_references(normalized_refs)
            for key, value in compact_summary.items():
                summary["compact_summary"][key] = summary["compact_summary"].get(key, 0) + value
        else:
            output_refs = normalized_refs

        summary["output_reference_count"] += len(output_refs)
        if output_refs != normalized_refs:
            summary["changed_doc_ids"].append(doc_id)
        processed[doc_id] = output_refs

    summary["removed_reference_count"] = (
        summary["input_reference_count"] - summary["output_reference_count"]
    )
    summary["changed_doc_ids"] = sorted(summary["changed_doc_ids"])
    summary["changed_doc_count"] = len(summary["changed_doc_ids"])
    return processed, summary


def _empty_core_reference_view_summary(mode: str) -> dict[str, Any]:
    """Initialize core reference view diagnostics."""
    return {
        "mode": mode,
        "doc_count": 0,
        "input_reference_count": 0,
        "output_reference_count": 0,
        "removed_reference_count": 0,
        "collapsed_same_core_reference_duplicates": 0,
        "changed_doc_ids": [],
    }


def _core_reference_view_key(ref: dict[str, str]) -> tuple[str, str, str] | None:
    """Return the core identity key for rows eligible for core evaluation."""
    normalized = normalize_reference(ref)
    madde = normalized.get("madde", "")
    kanun_no = normalized.get("kanun_no", "")
    kanun_ad = normalized.get("kanun_ad", "")
    if not madde or not (kanun_no or kanun_ad):
        return None
    return (kanun_no, kanun_ad, madde)


def _core_reference_survivor_rank(row: dict[str, str], order: int) -> tuple[int, int, int, int]:
    """Rank duplicate same-core rows for the row kept in core-set view."""
    status = str(row.get("status", "")).strip().lower()
    status_score = 2 if status == "approved" else 1 if status else 0
    detail_score = sum(1 for field in ("fikra", "bent") if row.get(field))
    source_score = 1 if row.get("source_text") else 0
    order_score = -order
    return (status_score, detail_score, source_score, order_score)


def apply_core_reference_view_maps(
    reference_maps: dict[int, list[dict[str, str]]],
    *,
    mode: str = CORE_REFERENCE_VIEW_ROW,
) -> tuple[dict[int, list[dict[str, str]]], dict[str, Any]]:
    """Apply the selected reference view used by core/docwise metrics."""
    normalized_mode = normalize_core_reference_view_mode(mode)
    summary = _empty_core_reference_view_summary(normalized_mode)
    processed: dict[int, list[dict[str, str]]] = {}

    for doc_id, refs in reference_maps.items():
        normalized_refs = [normalize_reference(ref) for ref in refs]
        summary["doc_count"] += 1
        summary["input_reference_count"] += len(normalized_refs)

        if normalized_mode == CORE_REFERENCE_VIEW_ROW:
            output_refs = normalized_refs
        else:
            output_refs = []
            core_positions: dict[tuple[str, str, str], int] = {}
            core_ranks: dict[tuple[str, str, str], tuple[int, int, int, int]] = {}
            removed_count = 0

            for order, row in enumerate(normalized_refs):
                key = _core_reference_view_key(row)
                if key is None:
                    output_refs.append(row)
                    continue

                rank = _core_reference_survivor_rank(row, order)
                existing_position = core_positions.get(key)
                if existing_position is None:
                    core_positions[key] = len(output_refs)
                    core_ranks[key] = rank
                    output_refs.append(row)
                    continue

                removed_count += 1
                if rank > core_ranks[key]:
                    output_refs[existing_position] = row
                    core_ranks[key] = rank

            summary["collapsed_same_core_reference_duplicates"] += removed_count

        summary["output_reference_count"] += len(output_refs)
        if output_refs != normalized_refs:
            summary["changed_doc_ids"].append(doc_id)
        processed[doc_id] = output_refs

    summary["removed_reference_count"] = (
        summary["input_reference_count"] - summary["output_reference_count"]
    )
    summary["changed_doc_ids"] = sorted(summary["changed_doc_ids"])
    summary["changed_doc_count"] = len(summary["changed_doc_ids"])
    return processed, summary


def _empty_core_exclusion_diagnostics() -> dict[str, int]:
    """Initialize flat count diagnostics for core metric exclusions/eligibility."""
    return {
        "eligible_gold_count": 0,
        "eligible_gold_triplet_required_count": 0,
        "eligible_gold_no_pair_required_count": 0,
        "eligible_gold_ad_pair_required_count": 0,
        "eligible_prediction_count": 0,
        "eligible_prediction_triplet_candidate_count": 0,
        "eligible_prediction_no_pair_candidate_count": 0,
        "eligible_prediction_ad_pair_candidate_count": 0,
        "excluded_no_madde_gold_count": 0,
        "excluded_missing_law_gold_count": 0,
        "invalid_core_prediction_no_madde_count": 0,
        "invalid_core_prediction_no_law_count": 0,
    }


def _empty_ignored_extra_law_field_diagnostics() -> dict[str, int]:
    """Initialize flat count diagnostics for ignored extra law fields."""
    return {
        "ignored_extra_pred_kanun_ad_total": 0,
        "ignored_extra_pred_kanun_ad_inconsistent": 0,
        "ignored_extra_pred_kanun_ad_unknown_consistency": 0,
        "ignored_extra_pred_kanun_no_total": 0,
        "ignored_extra_pred_kanun_no_inconsistent": 0,
        "ignored_extra_pred_kanun_no_unknown_consistency": 0,
    }


def _empty_soft_extension_counts() -> dict[str, int]:
    """Initialize flat count diagnostics for fikra/bent soft evaluation."""
    return {
        "core_tp_count": 0,
        "fikra_gold_present_count": 0,
        "fikra_exact_when_gold_present_count": 0,
        "fikra_missing_when_gold_present_count": 0,
        "fikra_wrong_nonempty_when_gold_present_count": 0,
        "fikra_gold_empty_count": 0,
        "fikra_extra_when_gold_empty_count": 0,
        "bent_gold_present_count": 0,
        "bent_exact_when_gold_present_count": 0,
        "bent_missing_when_gold_present_count": 0,
        "bent_wrong_nonempty_when_gold_present_count": 0,
        "bent_gold_empty_count": 0,
        "bent_extra_when_gold_empty_count": 0,
        "full_extension_exact_count": 0,
    }


def _merge_flat_counts(target: dict[str, int], source: dict[str, int]) -> None:
    """Add flat integer dict counts into `target`."""
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def _normalized_non_generic_law_name(kanun_ad: str, kanun_no: str = "") -> str:
    """Return normalized law name unless it is generic/unknown."""
    normalized = normalize_kanun_adi(kanun_ad, kanun_no=kanun_no)
    return normalized if normalized.lower() not in GENERIC_LAW_NAMES else ""


def _canonical_law_no_by_name(kanun_ad: str) -> str:
    """Resolve canonical law number by normalized law name when known."""
    normalized_ad = _normalized_non_generic_law_name(kanun_ad)
    return LAW_NAME_TO_CANONICAL_NO.get(normalized_ad, "")


def _predicted_kanun_ad_consistency(ref: dict[str, str]) -> bool | None:
    """Check whether predicted `kanun_ad` is consistent with predicted `kanun_no`."""
    kanun_no = normalize_kanun_no(ref.get("kanun_no", ""))
    kanun_ad = _normalized_non_generic_law_name(ref.get("kanun_ad", ""), kanun_no=kanun_no)
    canonical_ad = _normalized_non_generic_law_name(canonical_law_name_by_no(kanun_no), kanun_no=kanun_no)
    if not kanun_no or not kanun_ad or not canonical_ad:
        return None
    return kanun_ad == canonical_ad


def _predicted_kanun_no_consistency(ref: dict[str, str]) -> bool | None:
    """Check whether predicted `kanun_no` is consistent with predicted `kanun_ad`."""
    kanun_no = normalize_kanun_no(ref.get("kanun_no", ""))
    kanun_ad = _normalized_non_generic_law_name(ref.get("kanun_ad", ""), kanun_no=kanun_no)
    canonical_no = _canonical_law_no_by_name(kanun_ad)
    if not kanun_no or not kanun_ad or not canonical_no:
        return None
    return kanun_no == canonical_no


def _core_triplet_key(ref: dict[str, str]) -> tuple[str, str, str]:
    """Return triplet key for core matching."""
    return (ref.get("kanun_no", ""), ref.get("kanun_ad", ""), ref.get("madde", ""))


def _core_no_pair_key(ref: dict[str, str]) -> tuple[str, str]:
    """Return `(kanun_no, madde)` key for no-pair core matching."""
    return (ref.get("kanun_no", ""), ref.get("madde", ""))


def _core_ad_pair_key(ref: dict[str, str]) -> tuple[str, str]:
    """Return `(kanun_ad, madde)` key for ad-pair core matching."""
    return (ref.get("kanun_ad", ""), ref.get("madde", ""))


def _classify_core_gold_reference(
    ref: dict[str, str],
    diagnostics: dict[str, int],
) -> CoreReference | None:
    """Classify one gold reference for core law/article evaluation."""
    madde = ref.get("madde", "")
    kanun_no = ref.get("kanun_no", "")
    kanun_ad = ref.get("kanun_ad", "")

    if not madde:
        diagnostics["excluded_no_madde_gold_count"] += 1
        return None
    if kanun_no and kanun_ad:
        diagnostics["eligible_gold_count"] += 1
        diagnostics["eligible_gold_triplet_required_count"] += 1
        return CoreReference(ref=ref, kind="triplet_required")
    if kanun_no:
        diagnostics["eligible_gold_count"] += 1
        diagnostics["eligible_gold_no_pair_required_count"] += 1
        return CoreReference(ref=ref, kind="no_pair_required")
    if kanun_ad:
        diagnostics["eligible_gold_count"] += 1
        diagnostics["eligible_gold_ad_pair_required_count"] += 1
        return CoreReference(ref=ref, kind="ad_pair_required")

    diagnostics["excluded_missing_law_gold_count"] += 1
    return None


def _classify_core_prediction_reference(
    ref: dict[str, str],
    diagnostics: dict[str, int],
) -> CoreReference | None:
    """Classify one prediction reference for core law/article evaluation."""
    madde = ref.get("madde", "")
    kanun_no = ref.get("kanun_no", "")
    kanun_ad = ref.get("kanun_ad", "")

    if not madde:
        diagnostics["invalid_core_prediction_no_madde_count"] += 1
        return None
    if kanun_no and kanun_ad:
        diagnostics["eligible_prediction_count"] += 1
        diagnostics["eligible_prediction_triplet_candidate_count"] += 1
        return CoreReference(ref=ref, kind="triplet_candidate")
    if kanun_no:
        diagnostics["eligible_prediction_count"] += 1
        diagnostics["eligible_prediction_no_pair_candidate_count"] += 1
        return CoreReference(ref=ref, kind="no_pair_candidate")
    if kanun_ad:
        diagnostics["eligible_prediction_count"] += 1
        diagnostics["eligible_prediction_ad_pair_candidate_count"] += 1
        return CoreReference(ref=ref, kind="ad_pair_candidate")

    diagnostics["invalid_core_prediction_no_law_count"] += 1
    return None


def _consume_first_available(
    lookup: dict[tuple[str, ...], list[int]],
    key: tuple[str, ...],
    used_pred_ids: set[int],
) -> int | None:
    """Consume the first unused prediction id under `key`."""
    bucket = lookup.get(key)
    if not bucket:
        return None
    while bucket and bucket[0] in used_pred_ids:
        bucket.pop(0)
    if not bucket:
        return None
    pred_idx = bucket.pop(0)
    used_pred_ids.add(pred_idx)
    return pred_idx


def _accumulate_soft_extension_counts(
    counts: dict[str, int],
    gold_ref: dict[str, str],
    pred_ref: dict[str, str],
) -> None:
    """Accumulate fikra/bent diagnostics for one matched core reference pair."""
    counts["core_tp_count"] += 1

    gold_fikra = gold_ref.get("fikra", "")
    pred_fikra = pred_ref.get("fikra", "")
    if gold_fikra:
        counts["fikra_gold_present_count"] += 1
        if pred_fikra == gold_fikra:
            counts["fikra_exact_when_gold_present_count"] += 1
        elif not pred_fikra:
            counts["fikra_missing_when_gold_present_count"] += 1
        else:
            counts["fikra_wrong_nonempty_when_gold_present_count"] += 1
    else:
        counts["fikra_gold_empty_count"] += 1
        if pred_fikra:
            counts["fikra_extra_when_gold_empty_count"] += 1

    gold_bent = gold_ref.get("bent", "")
    pred_bent = pred_ref.get("bent", "")
    if gold_bent:
        counts["bent_gold_present_count"] += 1
        if pred_bent == gold_bent:
            counts["bent_exact_when_gold_present_count"] += 1
        elif not pred_bent:
            counts["bent_missing_when_gold_present_count"] += 1
        else:
            counts["bent_wrong_nonempty_when_gold_present_count"] += 1
    else:
        counts["bent_gold_empty_count"] += 1
        if pred_bent:
            counts["bent_extra_when_gold_empty_count"] += 1

    if gold_fikra == pred_fikra and gold_bent == pred_bent:
        counts["full_extension_exact_count"] += 1


def _finalize_soft_extension_diagnostics(counts: dict[str, int]) -> dict[str, float | int]:
    """Add rate fields to fikra/bent soft diagnostics."""
    diagnostics: dict[str, float | int] = dict(counts)

    def rate(numerator_key: str, denominator_key: str, rate_key: str) -> None:
        denominator = counts.get(denominator_key, 0)
        diagnostics[rate_key] = counts.get(numerator_key, 0) / denominator if denominator else 0.0

    rate(
        "fikra_exact_when_gold_present_count",
        "fikra_gold_present_count",
        "fikra_exact_when_gold_present_rate",
    )
    rate(
        "fikra_missing_when_gold_present_count",
        "fikra_gold_present_count",
        "fikra_missing_when_gold_present_rate",
    )
    rate(
        "fikra_wrong_nonempty_when_gold_present_count",
        "fikra_gold_present_count",
        "fikra_wrong_nonempty_when_gold_present_rate",
    )
    rate(
        "fikra_extra_when_gold_empty_count",
        "fikra_gold_empty_count",
        "fikra_extra_when_gold_empty_rate",
    )
    rate(
        "bent_exact_when_gold_present_count",
        "bent_gold_present_count",
        "bent_exact_when_gold_present_rate",
    )
    rate(
        "bent_missing_when_gold_present_count",
        "bent_gold_present_count",
        "bent_missing_when_gold_present_rate",
    )
    rate(
        "bent_wrong_nonempty_when_gold_present_count",
        "bent_gold_present_count",
        "bent_wrong_nonempty_when_gold_present_rate",
    )
    rate(
        "bent_extra_when_gold_empty_count",
        "bent_gold_empty_count",
        "bent_extra_when_gold_empty_rate",
    )
    rate("full_extension_exact_count", "core_tp_count", "full_extension_exact_rate")
    return diagnostics


def _evaluate_doc_core(
    pred_refs: list[dict[str, str]],
    gold_refs: list[dict[str, str]],
) -> tuple[Metrics, dict[str, int], dict[str, int], dict[str, int]]:
    """Evaluate one document with the primary core law/article metric."""
    exclusion_diagnostics = _empty_core_exclusion_diagnostics()
    ignored_extra_law_diagnostics = _empty_ignored_extra_law_field_diagnostics()
    soft_extension_counts = _empty_soft_extension_counts()

    gold_entries = [
        entry
        for ref in gold_refs
        if (entry := _classify_core_gold_reference(ref, exclusion_diagnostics)) is not None
    ]
    pred_entries = [
        entry
        for ref in pred_refs
        if (entry := _classify_core_prediction_reference(ref, exclusion_diagnostics)) is not None
    ]

    used_pred_ids: set[int] = set()
    matched_pairs: list[tuple[CoreReference, CoreReference]] = []

    triplet_pred_lookup: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for pred_idx, pred_entry in enumerate(pred_entries):
        if pred_entry.kind == "triplet_candidate":
            triplet_pred_lookup[_core_triplet_key(pred_entry.ref)].append(pred_idx)

    for gold_entry in gold_entries:
        if gold_entry.kind != "triplet_required":
            continue
        pred_idx = _consume_first_available(
            triplet_pred_lookup,
            _core_triplet_key(gold_entry.ref),
            used_pred_ids,
        )
        if pred_idx is not None:
            matched_pairs.append((gold_entry, pred_entries[pred_idx]))

    no_pair_pred_lookup: dict[tuple[str, str], list[int]] = defaultdict(list)
    ad_pair_pred_lookup: dict[tuple[str, str], list[int]] = defaultdict(list)
    triplet_pred_by_no_pair_lookup: dict[tuple[str, str], list[int]] = defaultdict(list)
    triplet_pred_by_ad_pair_lookup: dict[tuple[str, str], list[int]] = defaultdict(list)
    for pred_idx, pred_entry in enumerate(pred_entries):
        if pred_idx in used_pred_ids:
            continue
        if pred_entry.kind == "no_pair_candidate":
            no_pair_pred_lookup[_core_no_pair_key(pred_entry.ref)].append(pred_idx)
        elif pred_entry.kind == "ad_pair_candidate":
            ad_pair_pred_lookup[_core_ad_pair_key(pred_entry.ref)].append(pred_idx)
        elif pred_entry.kind == "triplet_candidate":
            triplet_pred_by_no_pair_lookup[_core_no_pair_key(pred_entry.ref)].append(pred_idx)
            triplet_pred_by_ad_pair_lookup[_core_ad_pair_key(pred_entry.ref)].append(pred_idx)

    for gold_entry in gold_entries:
        if gold_entry.kind != "no_pair_required":
            continue
        match_key = _core_no_pair_key(gold_entry.ref)
        pred_idx = _consume_first_available(no_pair_pred_lookup, match_key, used_pred_ids)
        if pred_idx is None:
            pred_idx = _consume_first_available(triplet_pred_by_no_pair_lookup, match_key, used_pred_ids)
            if pred_idx is not None:
                ignored_extra_law_diagnostics["ignored_extra_pred_kanun_ad_total"] += 1
                consistency = _predicted_kanun_ad_consistency(pred_entries[pred_idx].ref)
                if consistency is False:
                    ignored_extra_law_diagnostics["ignored_extra_pred_kanun_ad_inconsistent"] += 1
                elif consistency is None:
                    ignored_extra_law_diagnostics["ignored_extra_pred_kanun_ad_unknown_consistency"] += 1
        if pred_idx is not None:
            matched_pairs.append((gold_entry, pred_entries[pred_idx]))

    for gold_entry in gold_entries:
        if gold_entry.kind != "ad_pair_required":
            continue
        match_key = _core_ad_pair_key(gold_entry.ref)
        pred_idx = _consume_first_available(ad_pair_pred_lookup, match_key, used_pred_ids)
        if pred_idx is None:
            pred_idx = _consume_first_available(triplet_pred_by_ad_pair_lookup, match_key, used_pred_ids)
            if pred_idx is not None:
                ignored_extra_law_diagnostics["ignored_extra_pred_kanun_no_total"] += 1
                consistency = _predicted_kanun_no_consistency(pred_entries[pred_idx].ref)
                if consistency is False:
                    ignored_extra_law_diagnostics["ignored_extra_pred_kanun_no_inconsistent"] += 1
                elif consistency is None:
                    ignored_extra_law_diagnostics["ignored_extra_pred_kanun_no_unknown_consistency"] += 1
        if pred_idx is not None:
            matched_pairs.append((gold_entry, pred_entries[pred_idx]))

    for gold_entry, pred_entry in matched_pairs:
        _accumulate_soft_extension_counts(soft_extension_counts, gold_entry.ref, pred_entry.ref)

    tp = len(matched_pairs)
    fp = (
        len(pred_entries) - tp
        + exclusion_diagnostics["invalid_core_prediction_no_madde_count"]
        + exclusion_diagnostics["invalid_core_prediction_no_law_count"]
    )
    fn = len(gold_entries) - tp

    return (
        compute_metrics(tp, fp, fn),
        exclusion_diagnostics,
        ignored_extra_law_diagnostics,
        soft_extension_counts,
    )


def evaluate_core_maps(
    predictions: dict[int, list[dict[str, str]]],
    gold: dict[int, list[dict[str, str]]],
    doc_ids: list[int],
) -> tuple[Metrics, dict[int, Metrics], dict[str, int], dict[str, int], dict[str, float | int]]:
    """Evaluate primary core law/article metric across documents."""
    overall_tp = overall_fp = overall_fn = 0
    per_doc_metrics: dict[int, Metrics] = {}
    exclusion_diagnostics = _empty_core_exclusion_diagnostics()
    ignored_extra_law_diagnostics = _empty_ignored_extra_law_field_diagnostics()
    soft_extension_counts = _empty_soft_extension_counts()

    for doc_id in doc_ids:
        doc_metrics, doc_exclusion, doc_ignored_extra, doc_soft_counts = _evaluate_doc_core(
            predictions.get(doc_id, []),
            gold.get(doc_id, []),
        )
        overall_tp += doc_metrics.tp
        overall_fp += doc_metrics.fp
        overall_fn += doc_metrics.fn
        per_doc_metrics[doc_id] = doc_metrics
        _merge_flat_counts(exclusion_diagnostics, doc_exclusion)
        _merge_flat_counts(ignored_extra_law_diagnostics, doc_ignored_extra)
        _merge_flat_counts(soft_extension_counts, doc_soft_counts)

    return (
        compute_metrics(overall_tp, overall_fp, overall_fn),
        per_doc_metrics,
        exclusion_diagnostics,
        ignored_extra_law_diagnostics,
        _finalize_soft_extension_diagnostics(soft_extension_counts),
    )


def evaluate_optimal_maps(
    predictions: dict[int, list[dict[str, str]]],
    gold: dict[int, list[dict[str, str]]],
    doc_ids: list[int],
) -> tuple[Metrics, dict[int, Metrics], dict[str, int], dict[str, float | int]]:
    """Evaluate the core metric with the order-independent optimal matcher.

    Mirrors :func:`evaluate_core_maps` but replaces staged-greedy pairing with a
    maximum-cardinality assignment and scores law-level references instead of
    discarding them.
    """
    overall_tp = overall_fp = overall_fn = 0
    per_doc_metrics: dict[int, Metrics] = {}
    diagnostics = _empty_optimal_diagnostics()
    soft_extension_counts = _empty_soft_extension_counts()

    for doc_id in doc_ids:
        tp, fp, fn, doc_diagnostics, matched_pairs = evaluate_doc_optimal(
            predictions.get(doc_id, []),
            gold.get(doc_id, []),
        )
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn
        per_doc_metrics[doc_id] = compute_metrics(tp, fp, fn)
        _merge_flat_counts(diagnostics, doc_diagnostics)
        for gold_ref, pred_ref in matched_pairs:
            _accumulate_soft_extension_counts(soft_extension_counts, gold_ref, pred_ref)

    return (
        compute_metrics(overall_tp, overall_fp, overall_fn),
        per_doc_metrics,
        diagnostics,
        _finalize_soft_extension_diagnostics(soft_extension_counts),
    )


def reference_counter(references: list[dict[str, str]]) -> Counter[tuple[str, str, str, str, str]]:
    """Convert references to a multiset of core tuples."""
    return Counter(tuple(ref[field] for field in CORE_FIELDS) for ref in references)


def ref_to_tuple(ref: dict[str, str]) -> tuple[str, str, str, str, str]:
    """Convert a reference dict to its core tuple."""
    return tuple(ref.get(field, "") for field in CORE_FIELDS)


def _tuple_overlap_score(
    pred_tuple: tuple[str, str, str, str, str],
    gold_tuple: tuple[str, str, str, str, str],
) -> int:
    """Count how many CORE_FIELDS match between two tuples (0-5)."""
    return sum(
        1 for p, g in zip(pred_tuple, gold_tuple)
        if p == g and (p != "" or g != "")
    )


def _tuple_wrong_fields(
    pred_tuple: tuple[str, str, str, str, str],
    gold_tuple: tuple[str, str, str, str, str],
) -> list[str]:
    """Return names of fields that differ between two tuples."""
    return [
        CORE_FIELDS[i]
        for i in range(len(CORE_FIELDS))
        if pred_tuple[i] != gold_tuple[i]
    ]


def _classify_mismatch(
    pred_tuple: tuple[str, str, str, str, str] | None,
    gold_tuple: tuple[str, str, str, str, str] | None,
) -> tuple[str, list[str]]:
    """Classify a mismatch pair into a granular bucket type.

    Returns (bucket_type, wrong_fields).
    """
    if gold_tuple is None and pred_tuple is not None:
        return "spurious", []
    if pred_tuple is None and gold_tuple is not None:
        return "missed", []

    assert pred_tuple is not None and gold_tuple is not None
    overlap = _tuple_overlap_score(pred_tuple, gold_tuple)
    wrong = _tuple_wrong_fields(pred_tuple, gold_tuple)

    law_match = pred_tuple[0] == gold_tuple[0] and pred_tuple[1] == gold_tuple[1]

    if overlap >= 4:
        return "partial_4of5", wrong
    if overlap >= 3:
        return "partial_3of5", wrong
    if law_match:
        return "wrong_subfields", wrong
    return "wrong_law", wrong


def _greedy_pair_unmatched(
    unmatched_pred: list[tuple[str, str, str, str, str]],
    unmatched_gold: list[tuple[str, str, str, str, str]],
) -> tuple[
    list[tuple[tuple[str, str, str, str, str], tuple[str, str, str, str, str], int]],
    list[tuple[str, str, str, str, str]],
    list[tuple[str, str, str, str, str]],
]:
    """Greedy-pair unmatched predictions to unmatched gold by overlap score.

    Returns (paired, remaining_pred, remaining_gold).
    """
    if not unmatched_pred or not unmatched_gold:
        return [], list(unmatched_pred), list(unmatched_gold)

    candidates: list[tuple[int, int, int]] = []
    for pi, pt in enumerate(unmatched_pred):
        for gi, gt in enumerate(unmatched_gold):
            score = _tuple_overlap_score(pt, gt)
            if score > 0:
                candidates.append((score, pi, gi))

    candidates.sort(key=lambda x: x[0], reverse=True)

    used_pred: set[int] = set()
    used_gold: set[int] = set()
    paired: list[tuple[tuple[str, str, str, str, str], tuple[str, str, str, str, str], int]] = []

    for score, pi, gi in candidates:
        if pi in used_pred or gi in used_gold:
            continue
        paired.append((unmatched_pred[pi], unmatched_gold[gi], score))
        used_pred.add(pi)
        used_gold.add(gi)

    remaining_pred = [t for i, t in enumerate(unmatched_pred) if i not in used_pred]
    remaining_gold = [t for i, t in enumerate(unmatched_gold) if i not in used_gold]
    return paired, remaining_pred, remaining_gold


def field_counter(references: list[dict[str, str]], field: str) -> Counter[str]:
    """Convert one field to a multiset while ignoring empty values."""
    values = [str(ref.get(field, "")).strip() for ref in references]
    return Counter(value for value in values if value)


def _evaluate_doc_aligned(
    pred_refs: list[dict[str, str]],
    gold_refs: list[dict[str, str]],
) -> tuple[int, int, int, dict[str, dict[str, int]]]:
    """Evaluate one document with reference-aligned per-entity counting.

    Returns (tp, fp, fn, entity_totals_dict).
    """
    pred_counter = reference_counter(pred_refs)
    gold_counter = reference_counter(gold_refs)
    matched = pred_counter & gold_counter

    tp = sum(matched.values())
    fp = sum(pred_counter.values()) - tp
    fn = sum(gold_counter.values()) - tp

    entity_totals = {field: {"tp": 0, "fp": 0, "fn": 0} for field in CORE_FIELDS}

    # Matched tuples: every non-empty field is a TP
    for ref_tuple, count in matched.items():
        for idx, field in enumerate(CORE_FIELDS):
            if ref_tuple[idx]:
                entity_totals[field]["tp"] += count

    # FP tuples: unmatched predictions — every non-empty field is an FP
    fp_counter = _count_multiset_difference(pred_counter, matched)
    for ref_tuple, count in fp_counter.items():
        for idx, field in enumerate(CORE_FIELDS):
            if ref_tuple[idx]:
                entity_totals[field]["fp"] += count

    # FN tuples: unmatched gold — every non-empty field is an FN
    fn_counter = _count_multiset_difference(gold_counter, matched)
    for ref_tuple, count in fn_counter.items():
        for idx, field in enumerate(CORE_FIELDS):
            if ref_tuple[idx]:
                entity_totals[field]["fn"] += count

    return tp, fp, fn, entity_totals


def evaluate_maps(
    predictions: dict[int, list[dict[str, str]]],
    gold: dict[int, list[dict[str, str]]],
    doc_ids: list[int],
) -> tuple[Metrics, dict[str, Metrics], dict[int, Metrics]]:
    """Evaluate reference-level and reference-aligned entity-level metrics.

    Returns (overall, per_entity, per_doc).
    """
    overall_tp = overall_fp = overall_fn = 0
    entity_totals = {field: {"tp": 0, "fp": 0, "fn": 0} for field in CORE_FIELDS}
    per_doc_metrics: dict[int, Metrics] = {}

    for doc_id in doc_ids:
        pred_refs = predictions.get(doc_id, [])
        gold_refs = gold.get(doc_id, [])

        tp, fp, fn, doc_entity_totals = _evaluate_doc_aligned(pred_refs, gold_refs)

        overall_tp += tp
        overall_fp += fp
        overall_fn += fn

        per_doc_metrics[doc_id] = compute_metrics(tp, fp, fn)

        for field in CORE_FIELDS:
            for key in ("tp", "fp", "fn"):
                entity_totals[field][key] += doc_entity_totals[field][key]

    overall = compute_metrics(overall_tp, overall_fp, overall_fn)
    per_entity = {
        field: compute_metrics(values["tp"], values["fp"], values["fn"])
        for field, values in entity_totals.items()
    }
    return overall, per_entity, per_doc_metrics


def _count_multiset_difference(
    first: Counter[tuple[str, str, str, str, str]],
    second: Counter[tuple[str, str, str, str, str]],
) -> Counter[tuple[str, str, str, str, str]]:
    """Return multiset difference `first - second` preserving positive counts only."""
    out: Counter[tuple[str, str, str, str, str]] = Counter()
    for key, count in first.items():
        diff = count - second.get(key, 0)
        if diff > 0:
            out[key] = diff
    return out


def build_error_buckets(
    predictions: dict[int, list[dict[str, str]]],
    gold: dict[int, list[dict[str, str]]],
    doc_ids: list[int],
) -> tuple[list[ErrorBucket], PartialMatchStats]:
    """Aggregate granular mismatch buckets with sample doc ids.

    Returns (buckets, partial_match_stats).
    """
    counts: Counter[tuple[str, str]] = Counter()
    sample_docs: dict[tuple[str, str], list[int]] = {}
    wrong_fields_agg: dict[tuple[str, str], Counter[str]] = {}

    total_paired = 0
    total_overlap = 0
    overlap_dist: Counter[int] = Counter()
    near_miss = 0

    for doc_id in doc_ids:
        pred_counter = reference_counter(predictions.get(doc_id, []))
        gold_counter = reference_counter(gold.get(doc_id, []))
        matched = pred_counter & gold_counter
        unmatched_pred_counter = _count_multiset_difference(pred_counter, matched)
        unmatched_gold_counter = _count_multiset_difference(gold_counter, matched)

        unmatched_pred_items = list(unmatched_pred_counter.elements())
        unmatched_gold_items = list(unmatched_gold_counter.elements())

        paired, remaining_pred, remaining_gold = _greedy_pair_unmatched(
            unmatched_pred_items, unmatched_gold_items
        )

        # Process paired mismatches
        for pred_t, gold_t, score in paired:
            bucket_type, wrong = _classify_mismatch(pred_t, gold_t)
            total_paired += 1
            total_overlap += score
            overlap_dist[score] += 1
            if score >= 4:
                near_miss += 1

            # FN side
            fn_key = ("overall_reference", f"fn_{bucket_type}")
            counts[fn_key] += 1
            sample_docs.setdefault(fn_key, [])
            wrong_fields_agg.setdefault(fn_key, Counter())
            for wf in wrong:
                wrong_fields_agg[fn_key][wf] += 1
            if doc_id not in sample_docs[fn_key] and len(sample_docs[fn_key]) < 10:
                sample_docs[fn_key].append(doc_id)

            # FP side
            fp_key = ("overall_reference", f"fp_{bucket_type}")
            counts[fp_key] += 1
            sample_docs.setdefault(fp_key, [])
            wrong_fields_agg.setdefault(fp_key, Counter())
            for wf in wrong:
                wrong_fields_agg[fp_key][wf] += 1
            if doc_id not in sample_docs[fp_key] and len(sample_docs[fp_key]) < 10:
                sample_docs[fp_key].append(doc_id)

        # Remaining unpaired predictions → spurious
        for _ in remaining_pred:
            key = ("overall_reference", "fp_spurious")
            counts[key] += 1
            sample_docs.setdefault(key, [])
            if doc_id not in sample_docs[key] and len(sample_docs[key]) < 10:
                sample_docs[key].append(doc_id)

        # Remaining unpaired gold → missed
        for _ in remaining_gold:
            key = ("overall_reference", "fn_missed")
            counts[key] += 1
            sample_docs.setdefault(key, [])
            if doc_id not in sample_docs[key] and len(sample_docs[key]) < 10:
                sample_docs[key].append(doc_id)

    buckets = [
        ErrorBucket(
            entity_type=entity_type,
            bucket_type=bucket_type,
            count=count,
            sample_doc_ids=sample_docs.get((entity_type, bucket_type), []),
            wrong_fields=[
                f"{wf}={wc}"
                for wf, wc in wrong_fields_agg.get((entity_type, bucket_type), Counter()).most_common(5)
            ],
        )
        for (entity_type, bucket_type), count in counts.items()
    ]
    buckets.sort(key=lambda item: item.count, reverse=True)

    partial_stats = PartialMatchStats(
        total_unmatched_pairs=total_paired,
        avg_overlap_score=total_overlap / total_paired if total_paired else 0.0,
        overlap_distribution=dict(sorted(overlap_dist.items())),
        near_miss_count=near_miss,
    )
    return buckets, partial_stats


def compute_doc_level_diagnostics(
    per_doc_metrics: dict[int, Metrics],
    worst_n: int = 10,
    include_per_doc: bool = False,
) -> DocLevelMetrics:
    """Compute document-level F1 distribution diagnostics."""
    if not per_doc_metrics:
        return DocLevelMetrics(
            total_docs=0, perfect_f1_count=0, zero_f1_count=0,
            min_f1=0.0, max_f1=0.0, mean_f1=0.0, median_f1=0.0,
            p25_f1=0.0, p75_f1=0.0, worst_docs=[], per_doc=None,
        )

    f1_values = [m.f1 for m in per_doc_metrics.values()]
    sorted_f1 = sorted(f1_values)
    n = len(sorted_f1)

    p25_idx = max(0, int(n * 0.25) - 1)
    p75_idx = min(n - 1, int(n * 0.75))

    worst_docs_sorted = sorted(
        per_doc_metrics.items(), key=lambda x: (x[1].f1, -x[1].fn)
    )[:worst_n]

    per_doc_list = None
    if include_per_doc:
        per_doc_list = [
            {"doc_id": doc_id, **m.__dict__}
            for doc_id, m in sorted(per_doc_metrics.items())
        ]

    return DocLevelMetrics(
        total_docs=n,
        perfect_f1_count=sum(1 for f in f1_values if f == 1.0),
        zero_f1_count=sum(1 for f in f1_values if f == 0.0),
        min_f1=sorted_f1[0],
        max_f1=sorted_f1[-1],
        mean_f1=statistics.mean(f1_values),
        median_f1=statistics.median(f1_values),
        p25_f1=sorted_f1[p25_idx],
        p75_f1=sorted_f1[p75_idx],
        worst_docs=[
            {"doc_id": doc_id, **m.__dict__}
            for doc_id, m in worst_docs_sorted
        ],
        per_doc=per_doc_list,
    )


def _docwise_core_score(metric: Metrics) -> float:
    """Return the document score used by docwise core accuracy."""
    if metric.tp == 0 and metric.fp == 0 and metric.fn == 0:
        return 1.0
    return metric.f1


def compute_docwise_core_accuracy(
    core_per_doc_metrics: dict[int, Metrics],
    threshold: float = DEFAULT_DOCWISE_THRESHOLD,
    include_per_doc: bool = False,
) -> DocwiseCoreAccuracy:
    """Compute document-wise pass/fail accuracy from core per-document metrics."""
    threshold = validate_docwise_threshold(threshold)
    per_doc_rows: list[dict] = []
    passed_doc_count = 0

    for doc_id, metric in sorted(core_per_doc_metrics.items()):
        score = _docwise_core_score(metric)
        passed = score >= threshold
        if passed:
            passed_doc_count += 1
        if include_per_doc:
            per_doc_rows.append(
                {
                    "doc_id": doc_id,
                    "core_f1": score,
                    "tp": metric.tp,
                    "fp": metric.fp,
                    "fn": metric.fn,
                    "passed": passed,
                }
            )

    total_docs = len(core_per_doc_metrics)
    failed_doc_count = total_docs - passed_doc_count
    return DocwiseCoreAccuracy(
        threshold=threshold,
        accuracy=passed_doc_count / total_docs if total_docs else 0.0,
        passed_doc_count=passed_doc_count,
        failed_doc_count=failed_doc_count,
        total_docs=total_docs,
        per_doc=per_doc_rows if include_per_doc else None,
    )


def _serialize_doc_level_metrics(doc_diagnostics: DocLevelMetrics) -> dict[str, Any]:
    """Convert doc-level diagnostics dataclass to JSON-serializable dict."""
    return {
        "total_docs": doc_diagnostics.total_docs,
        "perfect_f1_count": doc_diagnostics.perfect_f1_count,
        "zero_f1_count": doc_diagnostics.zero_f1_count,
        "f1_distribution": {
            "min": doc_diagnostics.min_f1,
            "max": doc_diagnostics.max_f1,
            "mean": doc_diagnostics.mean_f1,
            "median": doc_diagnostics.median_f1,
            "p25": doc_diagnostics.p25_f1,
            "p75": doc_diagnostics.p75_f1,
        },
        "worst_docs": doc_diagnostics.worst_docs,
    }


def _serialize_docwise_core_accuracy(docwise_accuracy: DocwiseCoreAccuracy) -> dict[str, Any]:
    """Convert docwise core accuracy dataclass to JSON-serializable dict."""
    return {
        "threshold": docwise_accuracy.threshold,
        "accuracy": docwise_accuracy.accuracy,
        "passed_doc_count": docwise_accuracy.passed_doc_count,
        "failed_doc_count": docwise_accuracy.failed_doc_count,
        "total_docs": docwise_accuracy.total_docs,
    }


def _compute_metric_block_delta(
    current_block: dict[str, Any] | None,
    baseline_block: dict[str, Any] | None,
) -> dict[str, float | int] | None:
    """Compute deltas for one metric block when both sides exist."""
    if not isinstance(current_block, dict) or not isinstance(baseline_block, dict):
        return None
    return {
        "precision": current_block.get("precision", 0.0) - baseline_block.get("precision", 0.0),
        "recall": current_block.get("recall", 0.0) - baseline_block.get("recall", 0.0),
        "f1": current_block.get("f1", 0.0) - baseline_block.get("f1", 0.0),
        "tp": current_block.get("tp", 0) - baseline_block.get("tp", 0),
        "fp": current_block.get("fp", 0) - baseline_block.get("fp", 0),
        "fn": current_block.get("fn", 0) - baseline_block.get("fn", 0),
    }


def compute_metric_deltas(
    current: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Compute overall and per-entity metric deltas from baseline."""
    if not baseline:
        return None

    delta_payload: dict[str, Any] = {}
    overall_delta = _compute_metric_block_delta(current.get("overall"), baseline.get("overall"))
    if overall_delta is not None:
        delta_payload["overall"] = overall_delta

    baseline_per_entity = baseline.get("per_entity", {})
    current_per_entity = current.get("per_entity", {})
    delta_per_entity: dict[str, dict[str, float | int]] = {}
    for field in CORE_FIELDS:
        delta = _compute_metric_block_delta(current_per_entity.get(field), baseline_per_entity.get(field))
        if delta is not None:
            delta_per_entity[field] = delta
    if delta_per_entity:
        delta_payload["per_entity"] = delta_per_entity

    core_delta = _compute_metric_block_delta(
        current.get(PRIMARY_METRIC_NAME),
        baseline.get(PRIMARY_METRIC_NAME),
    )
    if core_delta is not None:
        delta_payload[PRIMARY_METRIC_NAME] = core_delta

    current_docwise = current.get(DOCWISE_CORE_ACCURACY_NAME)
    baseline_docwise = baseline.get(DOCWISE_CORE_ACCURACY_NAME)
    if isinstance(current_docwise, dict) and isinstance(baseline_docwise, dict):
        delta_payload[DOCWISE_CORE_ACCURACY_NAME] = {
            "accuracy": current_docwise.get("accuracy", 0.0) - baseline_docwise.get("accuracy", 0.0),
            "passed_doc_count": current_docwise.get("passed_doc_count", 0)
            - baseline_docwise.get("passed_doc_count", 0),
            "failed_doc_count": current_docwise.get("failed_doc_count", 0)
            - baseline_docwise.get("failed_doc_count", 0),
            "total_docs": current_docwise.get("total_docs", 0) - baseline_docwise.get("total_docs", 0),
        }

    return delta_payload or None


def format_docwise_core_accuracy(docwise_accuracy: dict[str, Any]) -> str:
    """Create a compact console line for docwise core accuracy."""
    threshold = docwise_accuracy.get("threshold", DEFAULT_DOCWISE_THRESHOLD)
    accuracy = docwise_accuracy.get("accuracy", 0.0)
    passed = docwise_accuracy.get("passed_doc_count", 0)
    total = docwise_accuracy.get("total_docs", 0)
    return f"Docwise core accuracy@{threshold:g}: {accuracy:.4f} ({passed}/{total})"


def format_metrics_table(
    method: str,
    core_metric: Metrics,
    overall: Metrics,
    per_entity: dict[str, Metrics],
) -> str:
    """Create a console-friendly plain text metrics table."""
    headers = ["Metric", "Precision", "Recall", "F1", "TP", "FP", "FN"]
    rows = [
        [
            PRIMARY_METRIC_NAME,
            f"{core_metric.precision:.4f}",
            f"{core_metric.recall:.4f}",
            f"{core_metric.f1:.4f}",
            str(core_metric.tp),
            str(core_metric.fp),
            str(core_metric.fn),
        ],
        [
            "overall_reference",
            f"{overall.precision:.4f}",
            f"{overall.recall:.4f}",
            f"{overall.f1:.4f}",
            str(overall.tp),
            str(overall.fp),
            str(overall.fn),
        ]
    ]

    for field in CORE_FIELDS:
        metric = per_entity[field]
        rows.append(
            [
                field,
                f"{metric.precision:.4f}",
                f"{metric.recall:.4f}",
                f"{metric.f1:.4f}",
                str(metric.tp),
                str(metric.fp),
                str(metric.fn),
            ]
        )

    widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) for i in range(len(headers))]

    def fmt(row: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    lines = [f"Method: {method}", fmt(headers), separator]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def evaluate_prediction_file(
    prediction_path: Path,
    gold: dict[int, list[dict[str, str]]],
    docs_universe: list[int] | None,
    include_per_doc: bool = False,
    ground_truth_mode: str = "approved_only",
    ground_truth_audit: dict[str, Any] | None = None,
    docwise_threshold: float = DEFAULT_DOCWISE_THRESHOLD,
    reference_postprocess: str = REFERENCE_POSTPROCESS_LEGACY,
    core_reference_view: str = CORE_REFERENCE_VIEW_ROW,
    matching_engine: str = MATCHING_ENGINE_GREEDY,
) -> dict[str, Any]:
    """Evaluate one prediction file and return a serializable result."""
    docwise_threshold = validate_docwise_threshold(docwise_threshold)
    reference_postprocess = normalize_reference_postprocess_mode(reference_postprocess)
    core_reference_view = normalize_core_reference_view_mode(core_reference_view)
    matching_engine = normalize_matching_engine(matching_engine)
    load_result = load_predictions(prediction_path)
    predictions = load_result.predictions

    if docs_universe is None:
        eval_doc_ids = sorted(gold.keys())
    else:
        eval_doc_ids = sorted(doc_id for doc_id in docs_universe if doc_id in gold)

    missing_gold_doc_ids = sorted(doc_id for doc_id in (docs_universe or []) if doc_id not in gold)
    missing_prediction_doc_ids = sorted(doc_id for doc_id in eval_doc_ids if doc_id not in predictions)

    processed_predictions, prediction_postprocess_summary = postprocess_reference_maps(
        predictions,
        mode=reference_postprocess,
    )
    processed_gold, gold_postprocess_summary = postprocess_reference_maps(
        gold,
        mode=reference_postprocess,
    )
    core_predictions, core_prediction_view_summary = apply_core_reference_view_maps(
        processed_predictions,
        mode=core_reference_view,
    )
    core_gold, core_gold_view_summary = apply_core_reference_view_maps(
        processed_gold,
        mode=core_reference_view,
    )

    overall, per_entity, per_doc_metrics = evaluate_maps(processed_predictions, processed_gold, eval_doc_ids)
    optimal_diagnostics: dict[str, Any] | None = None
    if matching_engine == MATCHING_ENGINE_OPTIMAL:
        # The optimal engine owns its whole preprocessing chain and is fed the
        # normalized references directly. `--reference-postprocess` and
        # `--core-reference-view` shape the legacy `greedy` path only: their
        # duplicate-survivor choice depends on row order, and their generic
        # suppression deletes zero gold rows but 385 Regex / 144 spaCy prediction
        # rows while touching none of the seven LLM methods. Consuming them would
        # import both the nondeterminism and that asymmetry into the metric.
        core_metrics, core_per_doc_metrics, raw_optimal_diagnostics, soft_extension_diagnostics = evaluate_optimal_maps(
            predictions,
            gold,
            eval_doc_ids,
        )
        optimal_diagnostics = optimal_diagnostics_payload(raw_optimal_diagnostics)
        core_exclusion_diagnostics = _empty_core_exclusion_diagnostics()
        ignored_extra_law_field_diagnostics = _empty_ignored_extra_law_field_diagnostics()
    else:
        core_metrics, core_per_doc_metrics, core_exclusion_diagnostics, ignored_extra_law_field_diagnostics, soft_extension_diagnostics = evaluate_core_maps(
            core_predictions,
            core_gold,
            eval_doc_ids,
        )
    error_buckets, partial_stats = build_error_buckets(processed_predictions, processed_gold, eval_doc_ids)
    doc_diagnostics = compute_doc_level_diagnostics(
        per_doc_metrics, include_per_doc=include_per_doc
    )
    core_doc_diagnostics = compute_doc_level_diagnostics(
        core_per_doc_metrics, include_per_doc=include_per_doc
    )
    docwise_core_accuracy = compute_docwise_core_accuracy(
        core_per_doc_metrics,
        threshold=docwise_threshold,
        include_per_doc=include_per_doc,
    )

    result: dict[str, Any] = {
        "method": prediction_path.stem,
        "prediction_path": str(prediction_path),
        "ground_truth_mode": ground_truth_mode,
        "reference_postprocess_mode": reference_postprocess,
        "core_reference_view": core_reference_view,
        "matching_engine": matching_engine,
        "prediction_row_count": load_result.total_rows,
        "evaluated_doc_count": len(eval_doc_ids),
        "docs_fingerprint": compute_docs_fingerprint(eval_doc_ids),
        "missing_gold_doc_ids": missing_gold_doc_ids,
        "missing_prediction_doc_ids": missing_prediction_doc_ids,
        "prediction_status_counts": load_result.status_counts,
        "prediction_error_doc_ids": load_result.error_doc_ids,
        "duplicate_prediction_doc_ids": load_result.duplicate_doc_ids,
        PRIMARY_METRIC_NAME: core_metrics.__dict__,
        "overall": overall.__dict__,
        "per_entity": {field: metric.__dict__ for field, metric in per_entity.items()},
        "error_buckets": [bucket.__dict__ for bucket in error_buckets],
        "top_error_buckets": [bucket.__dict__ for bucket in error_buckets[:5]],
        "partial_match_stats": partial_stats.__dict__,
        "doc_level_diagnostics": _serialize_doc_level_metrics(doc_diagnostics),
        "core_doc_level_diagnostics": _serialize_doc_level_metrics(core_doc_diagnostics),
        DOCWISE_CORE_ACCURACY_NAME: _serialize_docwise_core_accuracy(docwise_core_accuracy),
        "core_exclusion_diagnostics": core_exclusion_diagnostics,
        "ignored_extra_law_field_diagnostics": ignored_extra_law_field_diagnostics,
        "soft_extension_diagnostics": soft_extension_diagnostics,
        "reference_postprocess": {
            "mode": reference_postprocess,
            "prediction": prediction_postprocess_summary,
            "ground_truth": gold_postprocess_summary,
        },
        "core_reference_postprocess": {
            "mode": core_reference_view,
            "prediction": core_prediction_view_summary,
            "ground_truth": core_gold_view_summary,
        },
    }
    if optimal_diagnostics is not None:
        result["optimal_matching_diagnostics"] = optimal_diagnostics
        result["optimal_matching_preprocessing"] = {
            "owned_by_engine": True,
            "consumes_reference_postprocess": False,
            "consumes_core_reference_view": False,
            "steps": [
                "normalize_reference",
                "canonical_article_fields",
                "deduplicate_core_identities",
                "absorb_subsumed_identities",
                "canonical_sort",
            ],
            "generic_suppression_applied": False,
            "note": (
                "reference_postprocess and core_reference_view describe the greedy "
                "path only; they do not affect this result."
            ),
        }
    if ground_truth_audit is not None:
        result["ground_truth_audit"] = ground_truth_audit
    if doc_diagnostics.per_doc is not None:
        result["per_doc_metrics"] = doc_diagnostics.per_doc
    if core_doc_diagnostics.per_doc is not None:
        result["core_per_doc_metrics"] = core_doc_diagnostics.per_doc
    if docwise_core_accuracy.per_doc is not None:
        result["docwise_core_per_doc"] = docwise_core_accuracy.per_doc
    return result


def save_json_report(
    results: list[dict[str, Any]],
    path: Path,
    baseline_snapshot: dict[str, Any] | None = None,
) -> None:
    """Save benchmark output as JSON."""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    if baseline_snapshot is not None:
        payload["baseline_snapshot"] = baseline_snapshot
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_csv_report(results: list[dict[str, Any]], path: Path) -> None:
    """Save benchmark output as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "metric",
                "precision",
                "recall",
                "f1",
                "tp",
                "fp",
                "fn",
                "accuracy",
                "threshold",
                "passed_doc_count",
                "failed_doc_count",
                "total_docs",
                "evaluated_doc_count",
                "error_doc_count",
                "matching_engine",
                "core_reference_view",
                "reference_postprocess_mode",
                "ground_truth_mode",
            ],
        )
        writer.writeheader()

        for result in results:
            method = result["method"]
            doc_count = result["evaluated_doc_count"]
            error_doc_count = len(result.get("prediction_error_doc_ids", []))
            # Provenance must travel with every row: a `greedy` and an `optimal`
            # number are different metrics and must never be read as one series.
            provenance = {
                "matching_engine": result.get("matching_engine", MATCHING_ENGINE_GREEDY),
                "core_reference_view": result.get("core_reference_view", CORE_REFERENCE_VIEW_ROW),
                "reference_postprocess_mode": result.get(
                    "reference_postprocess_mode", REFERENCE_POSTPROCESS_LEGACY
                ),
                "ground_truth_mode": result.get("ground_truth_mode", "approved_only"),
            }
            core_metric = result.get(PRIMARY_METRIC_NAME)
            if isinstance(core_metric, dict):
                writer.writerow(
                    {
                        "method": method,
                        "metric": PRIMARY_METRIC_NAME,
                        "precision": f"{core_metric['precision']:.6f}",
                        "recall": f"{core_metric['recall']:.6f}",
                        "f1": f"{core_metric['f1']:.6f}",
                        "tp": core_metric["tp"],
                        "fp": core_metric["fp"],
                        "fn": core_metric["fn"],
                        "accuracy": "",
                        "threshold": "",
                        "passed_doc_count": "",
                        "failed_doc_count": "",
                        "total_docs": "",
                        "evaluated_doc_count": doc_count,
                        "error_doc_count": error_doc_count,
                        **provenance,
                    }
                )
            docwise_accuracy = result.get(DOCWISE_CORE_ACCURACY_NAME)
            if isinstance(docwise_accuracy, dict):
                writer.writerow(
                    {
                        "method": method,
                        "metric": DOCWISE_CORE_ACCURACY_NAME,
                        "precision": "",
                        "recall": "",
                        "f1": "",
                        "tp": "",
                        "fp": "",
                        "fn": "",
                        "accuracy": f"{docwise_accuracy['accuracy']:.6f}",
                        "threshold": f"{docwise_accuracy['threshold']:.6f}",
                        "passed_doc_count": docwise_accuracy["passed_doc_count"],
                        "failed_doc_count": docwise_accuracy["failed_doc_count"],
                        "total_docs": docwise_accuracy["total_docs"],
                        "evaluated_doc_count": doc_count,
                        "error_doc_count": error_doc_count,
                        **provenance,
                    }
                )
            overall = result["overall"]
            writer.writerow(
                {
                    "method": method,
                    "metric": "overall_reference",
                    "precision": f"{overall['precision']:.6f}",
                    "recall": f"{overall['recall']:.6f}",
                    "f1": f"{overall['f1']:.6f}",
                    "tp": overall["tp"],
                    "fp": overall["fp"],
                    "fn": overall["fn"],
                    "accuracy": "",
                    "threshold": "",
                    "passed_doc_count": "",
                    "failed_doc_count": "",
                    "total_docs": "",
                    "evaluated_doc_count": doc_count,
                    "error_doc_count": error_doc_count,
                    **provenance,
                }
            )
            for field in CORE_FIELDS:
                metric = result["per_entity"][field]
                writer.writerow(
                    {
                        "method": method,
                        "metric": field,
                        "precision": f"{metric['precision']:.6f}",
                        "recall": f"{metric['recall']:.6f}",
                        "f1": f"{metric['f1']:.6f}",
                        "tp": metric["tp"],
                        "fp": metric["fp"],
                        "fn": metric["fn"],
                        "accuracy": "",
                        "threshold": "",
                        "passed_doc_count": "",
                        "failed_doc_count": "",
                        "total_docs": "",
                        "evaluated_doc_count": doc_count,
                        "error_doc_count": error_doc_count,
                        **provenance,
                    }
                )


def save_error_bucket_csv(results: list[dict[str, Any]], path: Path) -> None:
    """Save error bucket aggregation rows as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["method", "entity_type", "bucket_type", "count", "wrong_fields", "sample_doc_ids"],
        )
        writer.writeheader()
        for result in results:
            for bucket in result.get("error_buckets", []):
                writer.writerow(
                    {
                        "method": result["method"],
                        "entity_type": bucket.get("entity_type", "overall_reference"),
                        "bucket_type": bucket.get("bucket_type", ""),
                        "count": bucket.get("count", 0),
                        "wrong_fields": ",".join(bucket.get("wrong_fields", [])),
                        "sample_doc_ids": ",".join(str(doc_id) for doc_id in bucket.get("sample_doc_ids", [])),
                    }
                )


def print_comparison(results: list[dict[str, Any]]) -> None:
    """Print compact multi-method comparison."""
    print(f"\nSummary ({PRIMARY_METRIC_NAME}):")
    print("method | precision | recall | f1 | tp | fp | fn | overall_f1")
    print("-------|-----------|--------|----|----|----|---|-----------")
    ordered = sorted(results, key=lambda item: item.get(PRIMARY_METRIC_NAME, {}).get("f1", 0.0), reverse=True)
    for item in ordered:
        primary_metric = item.get(PRIMARY_METRIC_NAME, item["overall"])
        overall = item["overall"]
        print(
            f"{item['method']} | {primary_metric['precision']:.4f} | {primary_metric['recall']:.4f} | "
            f"{primary_metric['f1']:.4f} | {primary_metric['tp']} | {primary_metric['fp']} | "
            f"{primary_metric['fn']} | {overall['f1']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Evaluate Gemini-style reference annotations")
    parser.add_argument(
        "--predictions",
        "-p",
        nargs="+",
        type=Path,
        required=True,
        help="Prediction JSON files (one or more)",
    )
    parser.add_argument(
        "--ground-truth-dir",
        "-g",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_DIR,
        help="Ground truth annotation directory",
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=DEFAULT_DOCS_PATH,
        help="Document source JSON used for evaluation universe",
    )
    parser.add_argument(
        "--doc-ids-file",
        type=Path,
        help="Optional JSON file containing explicit doc ids to evaluate (hard-case subset)",
    )
    parser.add_argument(
        "--exclude-doc-ids-file",
        type=Path,
        help="Optional JSON file containing doc ids to exclude from evaluation universe",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        help="Optional JSON baseline report path used to compute metric deltas",
    )
    parser.add_argument("--json-report", type=Path, help="Optional JSON report path")
    parser.add_argument("--csv-report", type=Path, help="Optional CSV report path")
    parser.add_argument("--error-bucket-csv", type=Path, help="Optional error bucket CSV report path")
    parser.add_argument(
        "--per-doc",
        action="store_true",
        default=False,
        help="Include per-document metrics in JSON report",
    )
    parser.add_argument(
        "--with-drafts",
        "--withDrafts",
        nargs="?",
        const=True,
        default=False,
        type=parse_cli_bool,
        dest="with_drafts",
        metavar="BOOL",
        help="Deprecated compatibility flag; equivalent to --gt-mode approved_plus_draft when true",
    )
    parser.add_argument(
        "--gt-mode",
        choices=list(SUPPORTED_GT_MODES),
        default=SUPPORTED_GT_MODES[0],
        help="Ground truth loading mode",
    )
    parser.add_argument(
        "--docwise-threshold",
        type=float,
        default=DEFAULT_DOCWISE_THRESHOLD,
        help="Document-wise core accuracy pass threshold between 0.0 and 1.0",
    )
    parser.add_argument(
        "--reference-postprocess",
        choices=list(SUPPORTED_REFERENCE_POSTPROCESS_MODES),
        default=REFERENCE_POSTPROCESS_LEGACY,
        help=(
            "Reference view before scoring. `legacy` preserves historical metrics; "
            "`canonical_set` compacts generic rows and deduplicates legal identity tuples."
        ),
    )
    parser.add_argument(
        "--core-reference-view",
        choices=list(SUPPORTED_CORE_REFERENCE_VIEW_MODES),
        default=CORE_REFERENCE_VIEW_ROW,
        help=(
            "Reference view for core/docwise metrics. `row` preserves Run 019/020 row-level "
            "core matching; `core_set` deduplicates document-local core identities "
            "(`kanun_no`, `kanun_ad`, `madde`) after reference postprocess."
        ),
    )
    parser.add_argument(
        "--matching-engine",
        choices=list(SUPPORTED_MATCHING_ENGINES),
        default=MATCHING_ENGINE_GREEDY,
        help=(
            "Core matcher. `greedy` reproduces the frozen Run 015-021 staged first-fit "
            "pairing; `optimal` uses order-independent maximum-cardinality matching with "
            "symmetric law-identity unification, and scores law-level (madde-less) "
            "references on both sides. CRITICAL: GT v4 ground truth MUST be evaluated "
            "with --matching-engine optimal. Greedy matching on GT v4 is invalid due to "
            "subparagraph splits."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    if not args.ground_truth_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {args.ground_truth_dir}")
    if args.docs and not args.docs.exists():
        raise FileNotFoundError(f"Docs file not found: {args.docs}")
    for prediction_path in args.predictions:
        if not prediction_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {prediction_path}")
    if args.doc_ids_file and not args.doc_ids_file.exists():
        raise FileNotFoundError(f"Doc ids file not found: {args.doc_ids_file}")
    if args.exclude_doc_ids_file and not args.exclude_doc_ids_file.exists():
        raise FileNotFoundError(f"Exclude doc ids file not found: {args.exclude_doc_ids_file}")
    if args.baseline_report and not args.baseline_report.exists():
        raise FileNotFoundError(f"Baseline report not found: {args.baseline_report}")
    docwise_threshold = validate_docwise_threshold(args.docwise_threshold)

    gt_mode = normalize_ground_truth_mode(
        "approved_plus_draft" if args.with_drafts else args.gt_mode
    )
    ground_truth_bundle = load_ground_truth_bundle(
        args.ground_truth_dir,
        include_drafts=include_drafts_for_ground_truth_mode(gt_mode),
    )
    gold = ground_truth_bundle.gold
    docs_universe = load_docs_universe(args.docs) if args.docs else None
    subset_doc_ids = load_doc_ids_file(args.doc_ids_file)
    if subset_doc_ids is not None:
        subset_doc_id_set = set(subset_doc_ids)
        if docs_universe is None:
            docs_universe = subset_doc_ids
        else:
            docs_universe = sorted(doc_id for doc_id in docs_universe if doc_id in subset_doc_id_set)
    exclude_doc_ids = load_doc_ids_file(args.exclude_doc_ids_file)
    if exclude_doc_ids is not None:
        exclude_doc_id_set = set(exclude_doc_ids)
        if docs_universe is None:
            docs_universe = sorted(doc_id for doc_id in gold if doc_id not in exclude_doc_id_set)
        else:
            docs_universe = sorted(doc_id for doc_id in docs_universe if doc_id not in exclude_doc_id_set)

    baseline_payload: dict[str, Any] | None = None
    baseline_by_method: dict[str, dict[str, Any]] = {}
    if args.baseline_report:
        baseline_payload = load_json_report(args.baseline_report)
        for item in baseline_payload.get("results", []):
            method_name = item.get("method")
            if isinstance(method_name, str):
                baseline_by_method[method_name] = item

    results = []
    for prediction_path in args.predictions:
        result = evaluate_prediction_file(
            prediction_path,
            gold,
            docs_universe,
            include_per_doc=args.per_doc,
            ground_truth_mode=gt_mode,
            ground_truth_audit=ground_truth_bundle.audit,
            docwise_threshold=docwise_threshold,
            reference_postprocess=args.reference_postprocess,
            core_reference_view=args.core_reference_view,
            matching_engine=args.matching_engine,
        )
        deltas = compute_metric_deltas(result, baseline_by_method.get(result["method"]))
        if deltas is not None:
            result["delta_from_baseline"] = deltas
        results.append(result)

        core_metric = Metrics(**result[PRIMARY_METRIC_NAME])
        overall = Metrics(**result["overall"])
        per_entity = {field: Metrics(**item) for field, item in result["per_entity"].items()}
        print(format_metrics_table(result["method"], core_metric, overall, per_entity))
        print(
            "Ground-truth mode: "
            f"{result.get('ground_truth_mode', 'approved_only')} | "
            f"reference_postprocess={result.get('reference_postprocess_mode', REFERENCE_POSTPROCESS_LEGACY)} | "
            f"core_reference_view={result.get('core_reference_view', CORE_REFERENCE_VIEW_ROW)} | "
            f"matching_engine={result.get('matching_engine', MATCHING_ENGINE_GREEDY)} | "
            f"approved_docs={ground_truth_bundle.audit.get('approved_doc_count', 0)} | "
            f"draft_only_docs={ground_truth_bundle.audit.get('draft_only_doc_count', 0)} | "
            f"mixed_docs={ground_truth_bundle.audit.get('mixed_doc_count', 0)}"
        )

        if result["missing_gold_doc_ids"]:
            print(f"Missing ground truth docs in directory: {result['missing_gold_doc_ids'][:10]}")
        if result["missing_prediction_doc_ids"]:
            print(f"Missing prediction docs: {result['missing_prediction_doc_ids'][:10]}")

        status_counts = result.get("prediction_status_counts", {})
        if status_counts:
            print(f"Prediction status counts: {status_counts}")
        if result.get("prediction_error_doc_ids"):
            print(f"Prediction error docs (first 10): {result['prediction_error_doc_ids'][:10]}")
        if result.get("duplicate_prediction_doc_ids"):
            print(f"Duplicate prediction doc ids: {result['duplicate_prediction_doc_ids'][:10]}")
        if result.get("top_error_buckets"):
            top_bucket_text = ", ".join(
                f"{bucket['bucket_type']}={bucket['count']}" for bucket in result["top_error_buckets"][:3]
            )
            print(f"Top error buckets: {top_bucket_text}")

        # Print doc-level diagnostics
        diag = result.get("doc_level_diagnostics", {})
        if diag:
            f1d = diag.get("f1_distribution", {})
            print(
                f"Doc-level F1: mean={f1d.get('mean', 0):.4f}  "
                f"median={f1d.get('median', 0):.4f}  "
                f"min={f1d.get('min', 0):.4f}  max={f1d.get('max', 0):.4f}  "
                f"p25={f1d.get('p25', 0):.4f}  p75={f1d.get('p75', 0):.4f}"
            )
            print(
                f"Perfect-F1 docs: {diag.get('perfect_f1_count', 0)}  "
                f"Zero-F1 docs: {diag.get('zero_f1_count', 0)}"
            )
        core_diag = result.get("core_doc_level_diagnostics", {})
        if core_diag:
            f1d = core_diag.get("f1_distribution", {})
            print(
                f"Core doc-level F1: mean={f1d.get('mean', 0):.4f}  "
                f"median={f1d.get('median', 0):.4f}  "
                f"min={f1d.get('min', 0):.4f}  max={f1d.get('max', 0):.4f}"
            )
        docwise_accuracy = result.get(DOCWISE_CORE_ACCURACY_NAME)
        if isinstance(docwise_accuracy, dict):
            print(format_docwise_core_accuracy(docwise_accuracy))
        core_exclusions = result.get("core_exclusion_diagnostics", {})
        if core_exclusions:
            print(
                "Core exclusions: "
                f"gold_no_madde={core_exclusions.get('excluded_no_madde_gold_count', 0)}  "
                f"gold_missing_law={core_exclusions.get('excluded_missing_law_gold_count', 0)}  "
                f"pred_no_madde={core_exclusions.get('invalid_core_prediction_no_madde_count', 0)}  "
                f"pred_no_law={core_exclusions.get('invalid_core_prediction_no_law_count', 0)}"
            )

        # Print partial match stats
        pms = result.get("partial_match_stats", {})
        if pms and pms.get("total_unmatched_pairs", 0) > 0:
            print(
                f"Partial match: {pms['total_unmatched_pairs']} pairs  "
                f"avg_overlap={pms['avg_overlap_score']:.2f}  "
                f"near_miss(4+of5)={pms['near_miss_count']}"
            )
        print()

    if len(results) > 1:
        print_comparison(results)

    if args.json_report:
        baseline_snapshot = None
        if baseline_payload is not None:
            baseline_snapshot = {
                "source_report": str(args.baseline_report),
                "generated_at": baseline_payload.get("generated_at"),
                "method_count": len(baseline_payload.get("results", [])),
            }
        save_json_report(results, args.json_report, baseline_snapshot=baseline_snapshot)
        print(f"JSON report saved: {args.json_report}")

    if args.csv_report:
        save_csv_report(results, args.csv_report)
        print(f"CSV report saved: {args.csv_report}")

    if args.error_bucket_csv:
        save_error_bucket_csv(results, args.error_bucket_csv)
        print(f"Error bucket CSV report saved: {args.error_bucket_csv}")


if __name__ == "__main__":
    main()
