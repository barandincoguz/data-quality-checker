"""Run regex, spaCy and LLM reference annotators, then optionally benchmark them."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from annotator.reference.llm import annotate_documents as annotate_llm
from annotator.reference.llm import build_checkpoint_path as build_llm_checkpoint_path
from annotator.reference.llm import delete_checkpoint_file as delete_llm_checkpoint_file
from annotator.reference.llm import save_run_summary as save_llm_run_summary
from annotator.reference.llm_prompt import DEFAULT_PROMPT_VARIANT, SUPPORTED_PROMPT_VARIANTS
from annotator.reference.llm_providers import DEFAULT_GEMINI_MODEL, build_prediction_filename
from annotator.reference.regex import annotate_documents as annotate_regex
from annotator.reference.spacy import annotate_documents as annotate_spacy
from benchmark.reference.evaluate import (
    DEFAULT_DOCWISE_THRESHOLD,
    DOCWISE_CORE_ACCURACY_NAME,
    CORE_REFERENCE_VIEW_ROW,
    MATCHING_ENGINE_GREEDY,
    Metrics,
    PRIMARY_METRIC_NAME,
    REFERENCE_POSTPROCESS_LEGACY,
    SUPPORTED_CORE_REFERENCE_VIEW_MODES,
    SUPPORTED_MATCHING_ENGINES,
    SUPPORTED_REFERENCE_POSTPROCESS_MODES,
    compute_metric_deltas,
    evaluate_prediction_file,
    format_docwise_core_accuracy,
    format_metrics_table,
    include_drafts_for_ground_truth_mode,
    load_doc_ids_file,
    load_docs_universe,
    load_ground_truth_bundle,
    load_json_report,
    normalize_ground_truth_mode,
    normalize_core_reference_view_mode,
    normalize_matching_engine,
    normalize_reference_postprocess_mode,
    parse_cli_bool,
    save_error_bucket_csv,
    save_csv_report,
    save_json_report,
    validate_docwise_threshold,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "test_data.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "predictions"
DEFAULT_GROUND_TRUTH_DIR = PROJECT_ROOT / "data" / "annotations"
DEFAULT_FEATURE_TAG = "adhoc"  # neutral default; pass --feature-tag for a real benchmark run (was the deprecated 008-common344-benchmark-withDrafts)
DEFAULT_LLM_RUN_SUMMARY = PROJECT_ROOT / "benchmark" / "results" / "llm_run_summary.json"
DEFAULT_LLM_MAX_RETRIES = 3


def load_documents(path: Path) -> list[dict[str, Any]]:
    """Load input docs from JSON file."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Input JSON must be an array: {path}")
    return data


def save_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    """Persist prediction rows to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)


def summarize_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build status and size summary for annotation rows."""
    status_counts = Counter(str(item.get("status", "unknown")).lower() for item in rows)
    total_refs = sum(len(item.get("references", [])) for item in rows)
    error_docs = [item.get("doc_id") for item in rows if str(item.get("status", "")).lower() == "error"]
    return {
        "status_counts": dict(status_counts),
        "total_refs": total_refs,
        "error_docs": [doc_id for doc_id in error_docs if isinstance(doc_id, int)],
    }


def ensure_required_artifacts(args: argparse.Namespace) -> None:
    """Validate required input artifacts before executing annotation/evaluation."""
    if not args.input.exists():
        raise FileNotFoundError(f"Input document file not found: {args.input}")
    if args.input_doc_ids_file and not args.input_doc_ids_file.exists():
        raise FileNotFoundError(f"Input doc ids file not found: {args.input_doc_ids_file}")
    if args.evaluate:
        if not args.ground_truth_dir.exists():
            raise FileNotFoundError(f"Ground truth directory not found: {args.ground_truth_dir}")
        if not any(args.ground_truth_dir.glob("doc_*.json")):
            raise FileNotFoundError(f"No ground truth files found in: {args.ground_truth_dir}")
    if args.baseline_report and not args.baseline_report.exists():
        raise FileNotFoundError(
            f"Baseline report not found: {args.baseline_report} (provide an existing JSON report path)"
        )
    if args.doc_ids_file and not args.doc_ids_file.exists():
        raise FileNotFoundError(f"Doc ids file not found: {args.doc_ids_file}")
    if args.exclude_doc_ids_file and not args.exclude_doc_ids_file.exists():
        raise FileNotFoundError(f"Exclude doc ids file not found: {args.exclude_doc_ids_file}")


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Run reference annotators and optional benchmark evaluation")
    parser.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT, help="Input documents JSON")
    parser.add_argument("--output-dir", "-o", type=Path, default=DEFAULT_OUTPUT_DIR, help="Prediction output directory")
    parser.add_argument(
        "--input-doc-ids-file",
        type=Path,
        help="Optional JSON file listing the input doc ids to annotate/evaluate",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["regex", "spacy", "llm"],
        default=["regex", "spacy"],
        help="Baselines to run",
    )
    parser.add_argument("--spacy-batch-size", type=int, default=256, help="spaCy nlp.pipe batch size")
    parser.add_argument("--spacy-n-process", type=int, default=1, help="spaCy nlp.pipe process count")
    parser.add_argument("--llm-provider", default="gemini", help="LLM provider used when `llm` is selected")
    parser.add_argument("--llm-model", default=DEFAULT_GEMINI_MODEL, help="LLM model name")
    parser.add_argument("--llm-temperature", type=float, default=0.0, help="LLM sampling temperature")
    parser.add_argument("--llm-run-label", default="", help="Optional label appended to LLM prediction filename")
    parser.add_argument(
        "--llm-prompt-variant",
        default=DEFAULT_PROMPT_VARIANT,
        choices=list(SUPPORTED_PROMPT_VARIANTS),
        help="LLM prompt variant",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=DEFAULT_LLM_MAX_RETRIES,
        help="Retry count for retryable LLM errors",
    )
    parser.add_argument("--llm-min-delay", type=float, default=0.0, help="Minimum delay between LLM documents")
    parser.add_argument("--llm-max-delay", type=float, default=0.0, help="Maximum delay between LLM documents")
    parser.add_argument("--llm-run-summary", type=Path, help="Optional LLM run summary JSON path")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable LLM checkpoint/resume support")
    parser.add_argument("--evaluate", action="store_true", help="Run benchmark after annotation")
    parser.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GROUND_TRUTH_DIR, help="Ground truth folder")
    parser.add_argument("--json-report", type=Path, help="Optional JSON report path")
    parser.add_argument("--csv-report", type=Path, help="Optional CSV report path")
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
        choices=["approved_only", "approved_plus_draft"],
        default="approved_only",
        help="Ground truth loading mode used during evaluation",
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
            "pairing; `optimal` uses order-independent maximum-cardinality matching and "
            "scores law-level (madde-less) references on both sides."
        ),
    )
    parser.add_argument("--baseline-report", type=Path, help="Optional baseline JSON report for delta comparison")
    parser.add_argument("--doc-ids-file", type=Path, help="Optional doc id subset JSON (e.g., hard cases)")
    parser.add_argument(
        "--exclude-doc-ids-file",
        type=Path,
        help="Optional JSON file listing doc ids to exclude from evaluation",
    )
    parser.add_argument("--error-bucket-csv", type=Path, help="Optional error bucket CSV output path")
    parser.add_argument("--hard-case-json-report", type=Path, help="Optional hard-case JSON report output path")
    parser.add_argument("--hard-case-csv-report", type=Path, help="Optional hard-case CSV report output path")
    parser.add_argument("--hard-case-error-bucket-csv", type=Path, help="Optional hard-case error bucket CSV path")
    parser.add_argument(
        "--feature-tag",
        default=DEFAULT_FEATURE_TAG,
        help=f"Feature tag used for default report paths (default: {DEFAULT_FEATURE_TAG})",
    )
    parser.add_argument(
        "--emit-default-reports",
        action="store_true",
        help="Emit default JSON/CSV/error-bucket reports under benchmark/results/per_method/<feature-tag>/",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    ensure_required_artifacts(args)
    if args.llm_temperature < 0:
        raise ValueError("--llm-temperature negatif olamaz.")
    docwise_threshold = validate_docwise_threshold(args.docwise_threshold)
    reference_postprocess = normalize_reference_postprocess_mode(args.reference_postprocess)
    core_reference_view = normalize_core_reference_view_mode(args.core_reference_view)
    matching_engine = normalize_matching_engine(args.matching_engine)

    if args.emit_default_reports:
        default_dir = PROJECT_ROOT / "benchmark" / "results" / "per_method" / args.feature_tag
        default_dir.mkdir(parents=True, exist_ok=True)
        if args.json_report is None:
            args.json_report = default_dir / "reference_eval_improved.json"
        if args.csv_report is None:
            args.csv_report = default_dir / "reference_eval_improved.csv"
        if args.error_bucket_csv is None:
            args.error_bucket_csv = default_dir / "error_buckets.csv"
        if "llm" in args.methods and args.llm_run_summary is None:
            args.llm_run_summary = default_dir / "llm_run_summary.json"
        if args.doc_ids_file:
            if args.hard_case_json_report is None:
                args.hard_case_json_report = default_dir / "hard_case_eval.json"
            if args.hard_case_csv_report is None:
                args.hard_case_csv_report = default_dir / "hard_case_eval.csv"
            if args.hard_case_error_bucket_csv is None:
                args.hard_case_error_bucket_csv = default_dir / "hard_case_error_buckets.csv"
    if "llm" in args.methods and args.llm_run_summary is None:
        args.llm_run_summary = DEFAULT_LLM_RUN_SUMMARY

    documents = load_documents(args.input)
    input_doc_ids = load_doc_ids_file(args.input_doc_ids_file)
    if input_doc_ids is not None:
        input_doc_id_set = set(input_doc_ids)
        documents = [document for document in documents if document.get("doc_id") in input_doc_id_set]

    prediction_files: list[Path] = []

    if "regex" in args.methods:
        regex_rows = annotate_regex(documents)
        regex_path = args.output_dir / "regex_reference_annotations.json"
        save_predictions(regex_path, regex_rows)
        prediction_files.append(regex_path)

        summary = summarize_prediction_rows(regex_rows)
        print(f"Regex annotations saved: {regex_path}")
        print(f"Regex status counts: {summary['status_counts']} | references: {summary['total_refs']}")
        if summary["error_docs"]:
            print(f"Regex error docs (first 10): {summary['error_docs'][:10]}")

    if "spacy" in args.methods:
        spacy_rows = annotate_spacy(
            documents,
            batch_size=max(1, args.spacy_batch_size),
            n_process=max(1, args.spacy_n_process),
        )
        spacy_path = args.output_dir / "spacy_reference_annotations.json"
        save_predictions(spacy_path, spacy_rows)
        prediction_files.append(spacy_path)

        summary = summarize_prediction_rows(spacy_rows)
        print(f"spaCy annotations saved: {spacy_path}")
        print(f"spaCy status counts: {summary['status_counts']} | references: {summary['total_refs']}")
        if summary["error_docs"]:
            print(f"spaCy error docs (first 10): {summary['error_docs'][:10]}")

    if "llm" in args.methods:
        llm_path = args.output_dir / build_prediction_filename(
            args.llm_provider,
            args.llm_prompt_variant,
            args.llm_run_label,
        )
        llm_rows, llm_summary = annotate_llm(
            documents,
            provider_name=args.llm_provider,
            model=args.llm_model,
            prompt_variant=args.llm_prompt_variant,
            temperature=max(0.0, args.llm_temperature),
            min_delay=max(0.0, args.llm_min_delay),
            max_delay=max(0.0, args.llm_max_delay),
            max_retries=max(0, args.llm_max_retries),
            run_label=args.llm_run_label,
            output_path=llm_path,
            checkpoint_enabled=not args.no_checkpoint,
        )
        save_predictions(llm_path, llm_rows)
        if not args.no_checkpoint and llm_summary["error_count"] == 0:
            delete_llm_checkpoint_file(build_llm_checkpoint_path(llm_path))
        prediction_files.append(llm_path)

        summary = summarize_prediction_rows(llm_rows)
        print(f"LLM annotations saved: {llm_path}")
        print(
            "LLM status counts: "
            f"{summary['status_counts']} | references: {summary['total_refs']} | "
            f"provider={llm_summary['provider']} model={llm_summary['model']} "
            f"prompt={llm_summary['prompt_variant']} temperature={llm_summary['temperature']}"
        )
        if summary["error_docs"]:
            print(f"LLM error docs (first 10): {summary['error_docs'][:10]}")

        if args.llm_run_summary:
            save_llm_run_summary(llm_summary, args.llm_run_summary)
            print(f"LLM run summary saved: {args.llm_run_summary}")

    if args.evaluate and prediction_files:
        gt_mode = normalize_ground_truth_mode(
            "approved_plus_draft" if args.with_drafts else args.gt_mode
        )
        ground_truth_bundle = load_ground_truth_bundle(
            args.ground_truth_dir,
            include_drafts=include_drafts_for_ground_truth_mode(gt_mode),
        )
        gold = ground_truth_bundle.gold
        docs_universe = sorted({doc_id for doc in documents if isinstance((doc_id := doc.get("doc_id")), int)})
        exclude_doc_ids = set(load_doc_ids_file(args.exclude_doc_ids_file) or [])
        if docs_universe is None:
            docs_universe = sorted(doc_id for doc_id in gold if doc_id not in exclude_doc_ids)
        elif exclude_doc_ids:
            docs_universe = [doc_id for doc_id in docs_universe if doc_id not in exclude_doc_ids]

        baseline_by_method: dict[str, dict[str, Any]] = {}
        baseline_payload: dict[str, Any] | None = None
        if args.baseline_report:
            baseline_payload = load_json_report(args.baseline_report)
            for item in baseline_payload.get("results", []):
                method_name = item.get("method")
                if isinstance(method_name, str):
                    baseline_by_method[method_name] = item

        evaluation_results = []
        for prediction_file in prediction_files:
            result = evaluate_prediction_file(
                prediction_file,
                gold,
                docs_universe,
                ground_truth_mode=gt_mode,
                ground_truth_audit=ground_truth_bundle.audit,
                docwise_threshold=docwise_threshold,
                reference_postprocess=reference_postprocess,
                core_reference_view=core_reference_view,
                matching_engine=matching_engine,
            )
            deltas = compute_metric_deltas(result, baseline_by_method.get(result["method"]))
            if deltas is not None:
                result["delta_from_baseline"] = deltas
            evaluation_results.append(result)

            core_metric = Metrics(**result[PRIMARY_METRIC_NAME])
            overall = Metrics(**result["overall"])
            per_entity = {field: Metrics(**values) for field, values in result["per_entity"].items()}
            print()
            print(format_metrics_table(result["method"], core_metric, overall, per_entity))
            print(
                "Ground-truth mode: "
                f"{result.get('ground_truth_mode', 'approved_only')} | "
                f"reference_postprocess={result.get('reference_postprocess_mode', REFERENCE_POSTPROCESS_LEGACY)} | "
                f"core_reference_view={result.get('core_reference_view', CORE_REFERENCE_VIEW_ROW)} | "
                f"approved_docs={ground_truth_bundle.audit.get('approved_doc_count', 0)} | "
                f"draft_only_docs={ground_truth_bundle.audit.get('draft_only_doc_count', 0)} | "
                f"mixed_docs={ground_truth_bundle.audit.get('mixed_doc_count', 0)}"
            )

            status_counts = result.get("prediction_status_counts", {})
            if status_counts:
                print(f"Prediction status counts: {status_counts}")
            if result.get("prediction_error_doc_ids"):
                print(f"Prediction error docs (first 10): {result['prediction_error_doc_ids'][:10]}")
            if result.get("duplicate_prediction_doc_ids"):
                print(f"Duplicate prediction doc ids: {result['duplicate_prediction_doc_ids'][:10]}")
            if result.get("missing_prediction_doc_ids"):
                print(f"Missing prediction docs (first 10): {result['missing_prediction_doc_ids'][:10]}")
            if result.get("top_error_buckets"):
                top_bucket_text = ", ".join(
                    f"{bucket['bucket_type']}={bucket['count']}" for bucket in result["top_error_buckets"][:3]
                )
                print(f"Top error buckets: {top_bucket_text}")
            docwise_accuracy = result.get(DOCWISE_CORE_ACCURACY_NAME)
            if isinstance(docwise_accuracy, dict):
                print(format_docwise_core_accuracy(docwise_accuracy))
            core_exclusions = result.get("core_exclusion_diagnostics", {})
            if core_exclusions:
                print(
                    "Core exclusions: "
                    f"gold_no_madde={core_exclusions.get('excluded_no_madde_gold_count', 0)} | "
                    f"pred_no_madde={core_exclusions.get('invalid_core_prediction_no_madde_count', 0)} | "
                    f"pred_no_law={core_exclusions.get('invalid_core_prediction_no_law_count', 0)}"
                )
            if result.get("delta_from_baseline"):
                core_delta = result["delta_from_baseline"].get(PRIMARY_METRIC_NAME)
                overall_delta = result["delta_from_baseline"].get("overall")
                if core_delta:
                    print(
                        f"Delta from baseline ({PRIMARY_METRIC_NAME}): "
                        f"precision={core_delta['precision']:+.4f}, "
                        f"recall={core_delta['recall']:+.4f}, "
                        f"f1={core_delta['f1']:+.4f}"
                    )
                elif overall_delta:
                    print(
                        "Delta from baseline (overall): "
                        f"precision={overall_delta['precision']:+.4f}, "
                        f"recall={overall_delta['recall']:+.4f}, "
                        f"f1={overall_delta['f1']:+.4f}"
                    )

        if args.json_report:
            baseline_snapshot = None
            if baseline_payload is not None:
                baseline_snapshot = {
                    "source_report": str(args.baseline_report),
                    "generated_at": baseline_payload.get("generated_at"),
                    "method_count": len(baseline_payload.get("results", [])),
                }
            save_json_report(evaluation_results, args.json_report, baseline_snapshot=baseline_snapshot)
            print(f"JSON report saved: {args.json_report}")

        if args.csv_report:
            save_csv_report(evaluation_results, args.csv_report)
            print(f"CSV report saved: {args.csv_report}")

        if args.error_bucket_csv:
            save_error_bucket_csv(evaluation_results, args.error_bucket_csv)
            print(f"Error bucket CSV report saved: {args.error_bucket_csv}")

        if args.doc_ids_file and (
            args.hard_case_json_report or args.hard_case_csv_report or args.hard_case_error_bucket_csv
        ):
            hard_case_ids = set(load_doc_ids_file(args.doc_ids_file) or [])
            hard_case_universe = sorted(
                doc_id
                for doc_id in (load_docs_universe(args.input) or [])
                if doc_id in hard_case_ids and doc_id not in exclude_doc_ids
            )

            hard_case_results = []
            for prediction_file in prediction_files:
                result = evaluate_prediction_file(
                    prediction_file,
                    gold,
                    hard_case_universe,
                    ground_truth_mode=gt_mode,
                    ground_truth_audit=ground_truth_bundle.audit,
                    docwise_threshold=docwise_threshold,
                    reference_postprocess=reference_postprocess,
                    core_reference_view=core_reference_view,
                )
                hard_case_results.append(result)

            if args.hard_case_json_report:
                save_json_report(hard_case_results, args.hard_case_json_report)
                print(f"Hard-case JSON report saved: {args.hard_case_json_report}")
            if args.hard_case_csv_report:
                save_csv_report(hard_case_results, args.hard_case_csv_report)
                print(f"Hard-case CSV report saved: {args.hard_case_csv_report}")
            if args.hard_case_error_bucket_csv:
                save_error_bucket_csv(hard_case_results, args.hard_case_error_bucket_csv)
                print(f"Hard-case error bucket CSV report saved: {args.hard_case_error_bucket_csv}")


if __name__ == "__main__":
    main()
