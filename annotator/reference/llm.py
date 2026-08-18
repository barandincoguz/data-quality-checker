"""LLM-based reference annotator with provider registry support."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from annotator.reference.llm_common import (
    build_error_result,
    build_success_result,
    enforce_output_contract,
    is_retryable_error,
    parse_response_text,
)
from annotator.reference.llm_prompt import (
    DEFAULT_PROMPT_VARIANT,
    SUPPORTED_PROMPT_VARIANTS,
    build_system_prompt,
    normalize_prompt_variant,
)
from annotator.reference.llm_providers import (
    DEFAULT_GEMINI_MODEL,
    GenerationResult,
    GenerationUsage,
    build_prediction_filename,
    get_provider,
    normalize_provider_name,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_dotenv(env_path: Path | None = None) -> None:
    """Load variables from a .env file into os.environ (no-op if file absent)."""
    path = env_path or PROJECT_ROOT / ".env"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
DEFAULT_INPUT = PROJECT_ROOT / "data" / "test_data.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "predictions"
DEFAULT_PROVIDER = "gemini"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / build_prediction_filename(DEFAULT_PROVIDER, DEFAULT_PROMPT_VARIANT)
DEFAULT_MAX_RETRIES = 3


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile on a sorted copy. q in [0, 100].

    Deliberately identical to the helper that produced the local models'
    latency profile, so cloud and local percentiles in the same table are
    computed by one formula and stay comparable.
    """
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * (q / 100.0)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(ordered[low])
    frac = pos - low
    return float(ordered[low] * (1 - frac) + ordered[high] * frac)


def _generate_with_usage(provider_instance: Any, **kwargs: Any) -> GenerationResult:
    """Call a provider, preferring the usage-reporting entry point.

    Providers registered by duck typing (LangExtract) may only expose
    `generate_content`; those report usage as unavailable rather than failing.
    """
    method = getattr(provider_instance, "generate_with_usage", None)
    if callable(method):
        return method(**kwargs)
    return GenerationResult(
        text=provider_instance.generate_content(**kwargs),
        usage=GenerationUsage.unavailable(),
    )


def _aggregate_usage(usages: Sequence[GenerationUsage]) -> dict[str, Any]:
    """Sum token counts over every attempt made for one document.

    Retries are real generations and are really charged, so a document that
    needed three attempts reports the tokens of all three. A counter no
    attempt reported stays None instead of collapsing to zero.
    """
    inputs = [u.input_tokens for u in usages if u.input_tokens is not None]
    outputs = [u.output_tokens for u in usages if u.output_tokens is not None]
    source = next((u.source for u in usages if u.source != "unavailable"), "unavailable")
    return {
        "input_tokens": sum(inputs) if inputs else None,
        "output_tokens": sum(outputs) if outputs else None,
        "usage_source": source,
    }


def _latency_summary(latencies: Sequence[float]) -> dict[str, Any]:
    """Latency distribution for the run, or None fields when nothing ran."""
    if not latencies:
        return {
            "latency_p50_seconds": None,
            "latency_p90_seconds": None,
            "latency_p95_seconds": None,
            "latency_max_seconds": None,
        }
    return {
        "latency_p50_seconds": round(percentile(latencies, 50), 4),
        "latency_p90_seconds": round(percentile(latencies, 90), 4),
        "latency_p95_seconds": round(percentile(latencies, 95), 4),
        "latency_max_seconds": round(max(latencies), 4),
    }


def _token_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Token totals across the run.

    Totals stay None when no document carried a provider-reported count, so a
    run against a provider that reports nothing cannot be mistaken for a run
    that measured zero tokens.
    """
    inputs = [r["input_tokens"] for r in records if r.get("input_tokens") is not None]
    outputs = [r["output_tokens"] for r in records if r.get("output_tokens") is not None]
    return {
        "documents_with_usage": len(
            [r for r in records if r.get("usage_source", "unavailable") != "unavailable"]
        ),
        "input_tokens_total": sum(inputs) if inputs else None,
        "output_tokens_total": sum(outputs) if outputs else None,
        "output_tokens_mean": round(sum(outputs) / len(outputs), 4) if outputs else None,
    }


def _timestamp() -> str:
    """Return a human-friendly local timestamp for progress logs."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _doc_label(document: dict[str, Any]) -> str:
    """Build a compact document label for progress output."""
    doc_id = document.get("doc_id", "?")
    text = " ".join(str(document.get("text", "")).split())
    if len(text) > 72:
        text = f"{text[:69]}..."
    return f"Doc {doc_id}: {text or '<bos metin>'}"


def _progress(message: str) -> None:
    """Print a timestamped progress line for long-running LLM jobs."""
    print(f"[{_timestamp()}] {message}", flush=True)


def build_checkpoint_path(output_path: Path) -> Path:
    """Return the adjacent checkpoint path for an output file."""
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.ckpt{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}.ckpt.json")


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace a text file via temporary file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp_file:
        tmp_file.write(content)
        tmp_path = Path(tmp_file.name)
    tmp_path.replace(path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically persist JSON content to disk."""
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    _atomic_write_text(path, serialized)


def load_checkpoint_rows(path: Path) -> list[dict[str, Any]]:
    """Load checkpointed successful rows from disk."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Checkpoint JSON must be an array: {path}")

    rows: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            rows.append(item)
    return rows


def save_checkpoint_rows(rows: Iterable[dict[str, Any]], path: Path) -> None:
    """Persist successful checkpoint rows atomically."""
    _atomic_write_json(path, list(rows))


def delete_checkpoint_file(path: Path) -> None:
    """Remove a checkpoint file if it exists."""
    if path.exists():
        path.unlink()


def _merge_results_in_document_order(
    documents: list[dict[str, Any]],
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return rows ordered to match the original input document sequence."""
    rows_by_doc_id: dict[Any, dict[str, Any]] = {}
    for row in rows:
        doc_id = row.get("doc_id")
        if doc_id is not None:
            rows_by_doc_id[doc_id] = row

    ordered_rows: list[dict[str, Any]] = []
    for document in documents:
        doc_id = document.get("doc_id")
        if doc_id in rows_by_doc_id:
            ordered_rows.append(rows_by_doc_id[doc_id])
    return ordered_rows


def annotate_document(
    text: str,
    *,
    provider_name: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_GEMINI_MODEL,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
    temperature: float = 0.0,
    provider: Any | None = None,
    usage_sink: list[GenerationUsage] | None = None,
) -> dict[str, Any]:
    """Annotate a single document with an LLM provider.

    `usage_sink`, when given, receives the provider-reported usage for this
    call. It is an out-parameter rather than part of the return value so that
    every existing caller keeps its current contract.
    """
    normalized_provider = normalize_provider_name(provider_name)
    normalized_variant = normalize_prompt_variant(prompt_variant)
    system_prompt = build_system_prompt(normalized_variant)
    raw_response = ""

    try:
        provider_instance = provider or get_provider(normalized_provider)
        generated = _generate_with_usage(
            provider_instance,
            text=text,
            system_prompt=system_prompt,
            model=model,
            temperature=max(0.0, float(temperature)),
        )
        if usage_sink is not None:
            usage_sink.append(generated.usage)
        raw_response = generated.text
        references = parse_response_text(raw_response)
        return build_success_result(text, references)
    except json.JSONDecodeError as exc:
        return build_error_result(text, f"JSON parse hatası: {exc}. Raw: {raw_response[:200]}")
    except Exception as exc:
        return build_error_result(text, str(exc))


def _annotate_with_retry(
    text: str,
    *,
    provider_name: str,
    model: str,
    prompt_variant: str,
    temperature: float,
    provider: Any,
    max_retries: int,
    doc_label: str,
    usage_sink: list[GenerationUsage] | None = None,
) -> tuple[dict[str, Any], int]:
    """Run one LLM annotation attempt with bounded retries."""
    last_result = build_error_result(text, "LLM annotator çalıştırılamadı.")
    attempts = 0

    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        if attempts == 1:
            _progress(f"  -> {doc_label} | attempt {attempts}/{max_retries + 1}")
        else:
            _progress(f"  -> {doc_label} | retry {attempts}/{max_retries + 1}")
        result = annotate_document(
            text,
            provider_name=provider_name,
            model=model,
            prompt_variant=prompt_variant,
            temperature=temperature,
            provider=provider,
            usage_sink=usage_sink,
        )
        if result.get("status") == "success":
            return result, attempts

        last_result = result
        if attempt >= max_retries or not is_retryable_error(result.get("error", "")):
            break

        wait_seconds = 10 + random.uniform(0, 5)
        error_preview = str(result.get("error", "")).strip().replace("\n", " ")
        if len(error_preview) > 160:
            error_preview = f"{error_preview[:157]}..."
        _progress(
            f"  !! Retryable error on {doc_label}: {error_preview or 'bilinmeyen hata'} | "
            f"backoff {wait_seconds:.1f}s"
        )
        time.sleep(wait_seconds)

    return last_result, attempts


def annotate_documents(
    documents: list[dict[str, Any]],
    *,
    provider_name: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_GEMINI_MODEL,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
    temperature: float = 0.0,
    min_delay: float = 0.0,
    max_delay: float = 0.0,
    max_retries: int = DEFAULT_MAX_RETRIES,
    run_label: str = "",
    output_path: Path | None = None,
    checkpoint_enabled: bool = True,
    provider: Any | None = None,
    provider_kwargs: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Annotate a list of input documents and return results with run summary."""
    normalized_provider = normalize_provider_name(provider_name)
    normalized_variant = normalize_prompt_variant(prompt_variant)
    started_at = time.time()
    per_doc_latencies: list[float] = []
    per_doc_records: list[dict[str, Any]] = []
    total_attempts = 0
    attempt_counts: Counter[int] = Counter()
    results: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    checkpoint_path = build_checkpoint_path(output_path) if checkpoint_enabled and output_path else None
    checkpoint_rows: list[dict[str, Any]] = []
    skipped_doc_ids: set[Any] = set()

    if checkpoint_path is not None:
        checkpoint_rows = load_checkpoint_rows(checkpoint_path)
        skipped_doc_ids = {
            row.get("doc_id")
            for row in checkpoint_rows
            if row.get("doc_id") is not None and row.get("status") == "success"
        }
        if skipped_doc_ids:
            _progress(
                f"Checkpoint resume active | path={checkpoint_path} "
                f"skipping {len(skipped_doc_ids)} completed docs"
            )
        results.extend(checkpoint_rows)

    pending_documents = [document for document in documents if document.get("doc_id") not in skipped_doc_ids]

    try:
        provider_instance = provider or get_provider(normalized_provider, **(provider_kwargs or {}))
        provider_init_error = ""
    except Exception as exc:
        provider_instance = None
        provider_init_error = str(exc)
        _progress(f"LLM provider initialization failed: {provider_init_error}")

    key_provenance = getattr(provider_instance, "key_provenance", None)
    provider_key_provenance = key_provenance() if callable(key_provenance) else None

    _progress(
        "LLM run started | "
        f"provider={normalized_provider} model={model} prompt={normalized_variant} "
        f"temperature={max(0.0, float(temperature)):.3f} "
        f"docs={len(documents)} pending={len(pending_documents)} "
        f"retries={max(0, max_retries)} delay=[{min_delay}, {max_delay}]"
    )

    for index, document in enumerate(pending_documents):
        doc_id = document.get("doc_id")
        text = str(document.get("text", ""))
        doc_started_at = time.time()
        doc_label = _doc_label(document)
        _progress(f"[{index + 1}/{len(pending_documents)}] {doc_label}")

        doc_usages: list[GenerationUsage] = []
        if provider_instance is None:
            result = build_error_result(text, provider_init_error or "LLM provider başlatılamadı.")
            attempts = 0
        else:
            result, attempts = _annotate_with_retry(
                text,
                provider_name=normalized_provider,
                model=model,
                prompt_variant=normalized_variant,
                temperature=max(0.0, float(temperature)),
                provider=provider_instance,
                max_retries=max(0, max_retries),
                doc_label=doc_label,
                usage_sink=doc_usages,
            )

        normalized_result = enforce_output_contract(result, text, doc_id=doc_id)
        results.append(normalized_result)
        status_counts[normalized_result["status"]] += 1
        total_attempts += attempts
        attempt_counts[attempts] += 1
        doc_latency = time.time() - doc_started_at
        per_doc_latencies.append(doc_latency)
        per_doc_records.append(
            {
                "doc_id": doc_id,
                "status": normalized_result["status"],
                "attempts": attempts,
                "latency_seconds": round(doc_latency, 4),
                "reference_count": len(normalized_result.get("references", [])),
                **_aggregate_usage(doc_usages),
            }
        )

        if (
            checkpoint_path is not None
            and normalized_result["status"] == "success"
            and normalized_result.get("doc_id") is not None
        ):
            checkpoint_rows.append(normalized_result)
            save_checkpoint_rows(checkpoint_rows, checkpoint_path)

        if normalized_result["status"] == "success":
            _progress(
                f"  OK {doc_label} | refs={len(normalized_result.get('references', []))} "
                f"| latency={doc_latency:.2f}s | attempts={attempts}"
            )
        else:
            error_preview = str(normalized_result.get("error", "")).strip().replace("\n", " ")
            if len(error_preview) > 200:
                error_preview = f"{error_preview[:197]}..."
            _progress(
                f"  XX {doc_label} | latency={doc_latency:.2f}s | attempts={attempts} "
                f"| error={error_preview or 'bilinmeyen hata'}"
            )

        if index < len(pending_documents) - 1 and max_delay > 0:
            sleep_seconds = random.uniform(max(0.0, min_delay), max(min_delay, max_delay))
            _progress(f"  .. Inter-document delay: {sleep_seconds:.2f}s")
            time.sleep(sleep_seconds)

    elapsed_seconds = time.time() - started_at
    merged_results = _merge_results_in_document_order(documents, results)
    merged_status_counts = Counter(str(item.get("status", "unknown")).lower() for item in merged_results)
    summary = {
        "provider": normalized_provider,
        "model": model,
        "prompt_variant": normalized_variant,
        "temperature": round(max(0.0, float(temperature)), 4),
        "run_label": str(run_label or "").strip(),
        "processed_target": len(documents),
        "pending_docs": len(pending_documents),
        "skipped_from_checkpoint": len(skipped_doc_ids),
        "success_count": merged_status_counts.get("success", 0),
        "error_count": merged_status_counts.get("error", 0),
        "status_counts": dict(merged_status_counts),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "elapsed_minutes": round(elapsed_seconds / 60.0, 3),
        "avg_doc_latency_seconds": round(sum(per_doc_latencies) / len(per_doc_latencies), 3)
        if per_doc_latencies
        else 0.0,
        **_latency_summary(per_doc_latencies),
        **_token_summary(per_doc_records),
        "per_document": per_doc_records,
        "max_retries": max(0, max_retries),
        "total_attempts": total_attempts,
        "retried_doc_count": sum(count for attempts, count in attempt_counts.items() if attempts > 1),
        "attempt_histogram": {str(attempts): count for attempts, count in sorted(attempt_counts.items())},
        "provider_init_error": provider_init_error,
        # Variable names and counts only, never key values. Lets a cost figure
        # be traced back to the billing tier that produced it.
        "provider_key_provenance": provider_key_provenance,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _progress(
        "LLM run completed | "
        f"success={summary['success_count']} error={summary['error_count']} "
        f"elapsed={summary['elapsed_seconds']:.2f}s avg_doc={summary['avg_doc_latency_seconds']:.2f}s"
    )
    return merged_results, summary


def load_documents(path: Path) -> list[dict[str, Any]]:
    """Load input documents from a JSON array file."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Input JSON must be an array: {path}")
    return data


def load_doc_ids(path: Path | None) -> list[int] | None:
    """Load an optional document id allowlist from a JSON array file."""
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Doc id file must be a JSON array: {path}")

    doc_ids: list[int] = []
    for item in data:
        if isinstance(item, int):
            doc_ids.append(item)
        elif isinstance(item, dict) and isinstance(item.get("doc_id"), int):
            doc_ids.append(item["doc_id"])
    return sorted(set(doc_ids))


def filter_documents_by_doc_ids(documents: list[dict[str, Any]], doc_ids: list[int] | None) -> list[dict[str, Any]]:
    """Filter documents to an explicit doc id subset while preserving input order."""
    if doc_ids is None:
        return documents
    doc_id_set = set(doc_ids)
    return [document for document in documents if document.get("doc_id") in doc_id_set]


def save_results(results: list[dict[str, Any]], path: Path) -> None:
    """Persist annotations to disk."""
    _atomic_write_json(path, results)


def save_run_summary(summary: dict[str, Any], path: Path) -> None:
    """Persist a run summary to disk."""
    _atomic_write_json(path, summary)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="LLM reference annotator")
    parser.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT, help="Input documents JSON")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT, help="Output annotations JSON")
    parser.add_argument("--doc-ids-file", type=Path, help="Optional JSON array file to filter input doc ids")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help="LLM provider name")
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL, help="LLM model name")
    parser.add_argument(
        "--prompt-variant",
        default=DEFAULT_PROMPT_VARIANT,
        choices=list(SUPPORTED_PROMPT_VARIANTS),
        help="Prompt variant to use",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM sampling temperature")
    parser.add_argument("--run-label", default="", help="Optional label appended to prediction filename/summary")
    parser.add_argument("--min-delay", type=float, default=0.0, help="Minimum delay between documents in seconds")
    parser.add_argument("--max-delay", type=float, default=0.0, help="Maximum delay between documents in seconds")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Retry count for retryable LLM errors",
    )
    parser.add_argument(
        "--ollama-key-env",
        default=None,
        help=(
            "Use only the Ollama key in this environment variable, with rotation off "
            "(e.g. OLLAMA_API_KEY_PRO). Pins the run to one billing tier so a cost "
            "figure can be attributed. Fails loudly if the variable is unset."
        ),
    )
    parser.add_argument("--run-summary", type=Path, help="Optional run summary JSON path")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable per-doc checkpoint/resume support")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for LLM annotation."""
    _load_dotenv()
    args = parse_args()
    if args.temperature < 0:
        raise ValueError("--temperature negatif olamaz.")
    if args.doc_ids_file and not args.doc_ids_file.exists():
        raise FileNotFoundError(f"Doc ids file not found: {args.doc_ids_file}")

    documents = load_documents(args.input)
    documents = filter_documents_by_doc_ids(documents, load_doc_ids(args.doc_ids_file))
    resolved_output = args.output
    if resolved_output == DEFAULT_OUTPUT and str(args.run_label or "").strip():
        resolved_output = DEFAULT_OUTPUT_DIR / build_prediction_filename(
            args.provider,
            args.prompt_variant,
            args.run_label,
        )
    results, summary = annotate_documents(
        documents,
        provider_name=args.provider,
        model=args.model,
        prompt_variant=args.prompt_variant,
        temperature=max(0.0, args.temperature),
        min_delay=max(0.0, args.min_delay),
        max_delay=max(0.0, args.max_delay),
        max_retries=max(0, args.max_retries),
        run_label=args.run_label,
        output_path=resolved_output,
        checkpoint_enabled=not args.no_checkpoint,
        provider_kwargs={"key_env": args.ollama_key_env} if args.ollama_key_env else None,
    )
    save_results(results, resolved_output)
    if not args.no_checkpoint and summary["error_count"] == 0:
        delete_checkpoint_file(build_checkpoint_path(resolved_output))
    if args.run_summary:
        save_run_summary(summary, args.run_summary)

    print(f"Input docs: {len(documents)}")
    print(
        f"Provider: {summary['provider']} | model: {summary['model']} | prompt: {summary['prompt_variant']} "
        f"| temperature: {summary['temperature']}"
    )
    print(f"Status counts: {summary['status_counts']}")
    print(f"Average doc latency (s): {summary['avg_doc_latency_seconds']}")
    print(f"Saved predictions: {resolved_output}")
    if args.run_summary:
        print(f"Saved run summary: {args.run_summary}")


if __name__ == "__main__":
    main()
