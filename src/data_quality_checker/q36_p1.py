"""Fail-closed control plane for the controlled Qwen3.6-27B SFT experiment.

Q36-P1 deliberately reuses the byte-identical G0 compact prompt and stateful
MLX trainer, while keeping its data, model revision, outputs, and deployment
views separate from G0.  No long compute is started by this module: bootstrap
seals the inputs and refuses data export until the human GREEN audit passes.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .atomic import (
    write_bytes_atomic,
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)
from .config import AppConfig
from .contracts import validate_reference_list
from .errors import ContractError, FingerprintMismatch, GateBlocked, IntegrityError
from .fingerprints import fingerprint_json, sha256_file, sha256_text
from .g0 import (
    PROMPT_VARIANT,
    SYSTEM_PROMPT,
    TRAINING_VIEW_POLICY,
    validate_canonical_sources,
)
from .mlx_compute import build_snapshot_manifest
from .mlx_stateful import StatefulTrainingConfig
from .normalization import full_identity, normalize_reference
from .reference_policy import (
    DEFAULT_REFERENCE_POLICY_ID,
    apply_reference_policy,
    reference_policy_spec,
)
from .storage import Store

EXPERIMENT_ID = "Q36-P1"
Q36_MODEL_ID = "mlx-community/Qwen3.6-27B-OptiQ-4bit"
Q36_MODEL_REVISION = "4e9a1cafd9c8bea42ce7f014e1231a4650010fc9"
LOCKED_PROMPT_SHA256 = "9608317a5f77544279b6e46aaf059f99db2820de182068685b5df3d5529b7a8c"
TRAINING_CONTEXT_TOKENS = 1536
INFERENCE_INPUT_TOKENS = 16384
INFERENCE_OUTPUT_TOKENS = 4096
INFERENCE_INPUT_FALLBACK_TOKENS = 32768
INFERENCE_OUTPUT_FALLBACK_TOKENS = 8192
GREEN_AUDIT_SAMPLE_SIZE = 35
DEVELOPMENT_CANONICAL_DOCUMENTS = 394
DEVELOPMENT_EXTERNAL_DOCUMENTS = 156
DEVELOPMENT_DOCUMENTS = 550
REFIT_CANONICAL_DOCUMENTS = 494
REFIT_EXTERNAL_DOCUMENTS = 309
REFIT_DOCUMENTS = 803

EXPECTED_GREEN_DISTRIBUTION = {
    "external_train": 156,
    "historical_validation": 34,
    "historical_sealed_test": 119,
    "canonical_duplicate_cluster": 11,
    "reference_validation_failed": 9,
    "incomplete_annotation": 13,
}

_VUK_413_EVIDENCE_RE = re.compile(
    r"\bBu\s+Özelge\s+213\s+sayılı\s+Vergi\s+Usul\s+"
    r"Kanunu(?:nun|'nun)?\s+413\s*\.?\s*"
    r"(?:üncü|uncu|inci|ıncı|nci|ncı|ncü|ncu)?\s*"
    r"maddesine\s+dayanılarak\s+verilmiştir\.?",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class Q36TrainingContract:
    seed: int = 42
    lora_rank: int = 8
    lora_layers: int = 16
    lora_scale: float = 20.0
    lora_dropout: float = 0.0
    batch_size: int = 1
    gradient_accumulation: int = 4
    peak_learning_rate: float = 2.5e-5
    warmup_updates: int = 42
    end_learning_rate: float = 1e-5
    gradient_checkpointing: bool = True
    completion_only_loss: bool = True
    loss_reduction: str = "target_token_mean_v1"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    adam_bias_correction: bool = False
    max_sequence_length: int = TRAINING_CONTEXT_TOKENS
    checkpoint_every_updates: int = 25
    checkpoint_max_seconds: int = 600
    validation_every_updates: int = 25

    def optimizer_updates(self, training_view_rows: int) -> int:
        if training_view_rows <= 0:
            raise ValueError("training_view_rows must be positive")
        return math.ceil(training_view_rows / self.gradient_accumulation)

    def stateful_config(self, training_view_rows: int) -> StatefulTrainingConfig:
        total_updates = self.optimizer_updates(training_view_rows)
        return StatefulTrainingConfig(
            seed=self.seed,
            rank=self.lora_rank,
            num_layers=self.lora_layers,
            lora_scale=self.lora_scale,
            lora_dropout=self.lora_dropout,
            batch_size=self.batch_size,
            gradient_accumulation=self.gradient_accumulation,
            peak_learning_rate=self.peak_learning_rate,
            end_learning_rate=self.end_learning_rate,
            adam_beta1=self.adam_beta1,
            adam_beta2=self.adam_beta2,
            adam_eps=self.adam_eps,
            adam_bias_correction=self.adam_bias_correction,
            loss_reduction=self.loss_reduction,
            warmup_updates=self.warmup_updates,
            total_updates=total_updates,
            max_sequence_length=self.max_sequence_length,
            gradient_checkpointing=self.gradient_checkpointing,
            checkpoint_every_updates=self.checkpoint_every_updates,
            checkpoint_max_seconds=self.checkpoint_max_seconds,
        )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _repo_root(config: AppConfig) -> Path:
    for candidate in (config.source_path.resolve().parent, *config.source_path.resolve().parents):
        if (candidate / "AGENTS.md").is_file() or (candidate / "pyproject.toml").is_file():
            return candidate
    raise IntegrityError(f"cannot resolve repository root from {config.source_path}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot parse JSON at {path}: {exc}") from exc


def _historical_paths(repo_root: Path) -> dict[str, Path]:
    root = repo_root / "artifacts/qwen3_14b_neon_external_eval_2026_07_16"
    sensitive = repo_root / "data/sensitive/neon_external_eval/prepared"
    return {
        "split_manifest": root / "split_manifest.json",
        "canonical_overlap": root / "canonical_overlap.json",
        "quarantine": root / "quarantine.json",
        "external_validation_docs": sensitive / "validation/docs.json",
        "external_validation_gold": sensitive / "validation/gold.json",
        "external_sealed_docs": sensitive / "sealed/docs.json",
        "external_sealed_gold": sensitive / "sealed/gold.json",
    }


def _model_snapshot_path() -> Path:
    return Path(
        "/opt/llm-lab/hf-cache/hub/"
        "models--mlx-community--Qwen3.6-27B-OptiQ-4bit/"
        f"snapshots/{Q36_MODEL_REVISION}"
    )


def _neon_document_hash(raw_document_id: str) -> str:
    return sha256_text(f"neon-document:{raw_document_id}")


def _classification_inputs(repo_root: Path) -> dict[str, Any]:
    paths = _historical_paths(repo_root)
    split_manifest = _read_json(paths["split_manifest"])
    documents = split_manifest.get("documents") if isinstance(split_manifest, dict) else None
    if not isinstance(documents, list):
        raise ContractError("historical split manifest has no documents list")
    by_text: dict[str, dict[str, Any]] = {}
    for row in documents:
        if not isinstance(row, dict) or row.get("source") != "external":
            continue
        text_sha256 = row.get("text_sha256")
        if not isinstance(text_sha256, str) or text_sha256 in by_text:
            raise IntegrityError("historical external text hashes are missing or duplicated")
        by_text[text_sha256] = row
    overlap_rows = _read_json(paths["canonical_overlap"])
    quarantine_rows = _read_json(paths["quarantine"])
    if not isinstance(overlap_rows, list) or not isinstance(quarantine_rows, list):
        raise ContractError("historical overlap/quarantine artifacts must be lists")
    overlap = {
        str(row["document_id_hash"]): row
        for row in overlap_rows
        if isinstance(row, dict) and row.get("document_id_hash")
    }
    quarantine = {
        str(row["document_id_hash"]): row
        for row in quarantine_rows
        if isinstance(row, dict) and row.get("document_id_hash")
    }
    return {
        "paths": paths,
        "by_text": by_text,
        "overlap": overlap,
        "quarantine": quarantine,
    }


def classify_green_pool(
    green_documents: Iterable[dict[str, Any]], *, repo_root: Path
) -> dict[str, Any]:
    """Classify every live GREEN record against the sealed historical split."""

    inputs = _classification_inputs(repo_root)
    entries: list[dict[str, Any]] = []
    counts = {key: 0 for key in EXPECTED_GREEN_DISTRIBUTION}
    split_names = {
        "train": "external_train",
        "validation": "historical_validation",
        "sealed_test": "historical_sealed_test",
    }
    seen_internal_ids: set[str] = set()
    for document in green_documents:
        internal_doc_id = str(document.get("internal_doc_id", ""))
        if not internal_doc_id or internal_doc_id in seen_internal_ids:
            raise IntegrityError("GREEN documents contain a missing/duplicate internal ID")
        seen_internal_ids.add(internal_doc_id)
        metadata = _read_json_text(document.get("metadata_json"), "metadata_json")
        text_sha256 = str(document.get("text_sha256", ""))
        historical = inputs["by_text"].get(text_sha256)
        document_hash = _neon_document_hash(str(document.get("raw_document_id", "")))
        historical_doc_id: int | None = None
        if historical is not None:
            try:
                category = split_names[str(historical["split"])]
            except KeyError as exc:
                raise IntegrityError(f"unknown historical split for {internal_doc_id}") from exc
            historical_doc_id = int(historical["doc_id"])
        elif document_hash in inputs["overlap"]:
            category = "canonical_duplicate_cluster"
        elif document_hash in inputs["quarantine"]:
            reason = str(inputs["quarantine"][document_hash].get("reason", ""))
            if reason != "reference_validation_failed":
                raise IntegrityError(
                    f"unexpected GREEN quarantine reason {reason!r} for {internal_doc_id}"
                )
            category = "reference_validation_failed"
        elif metadata.get("annotation_completed") is not True:
            category = "incomplete_annotation"
        else:
            raise IntegrityError(f"unclassified GREEN document: {internal_doc_id}")
        counts[category] += 1
        entries.append(
            {
                "internal_doc_id": internal_doc_id,
                "public_doc_id": str(document.get("public_doc_id", "")),
                "text_sha256": text_sha256,
                "category": category,
                "historical_doc_id": historical_doc_id,
            }
        )
    if counts != EXPECTED_GREEN_DISTRIBUTION:
        raise FingerprintMismatch(
            f"GREEN distribution drift: expected {EXPECTED_GREEN_DISTRIBUTION}, got {counts}"
        )
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "counts": counts,
        "document_count": len(entries),
        "entries": sorted(entries, key=lambda row: row["internal_doc_id"]),
        "source_artifacts": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in inputs["paths"].items()
        },
    }


def _read_json_text(value: Any, label: str) -> Any:
    if not isinstance(value, str):
        raise ContractError(f"{label} is not serialized JSON")
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ContractError(f"{label} is invalid JSON: {exc}") from exc


def _membership(references: list[dict[str, Any]]) -> set[tuple[str, ...]]:
    filtered, _ = apply_reference_policy(
        validate_reference_list(references), policy_id=DEFAULT_REFERENCE_POLICY_ID
    )
    return {full_identity(normalize_reference(reference)) for reference in filtered}


def inspect_green_audit(
    *,
    audit: dict[str, Any],
    green_documents: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    escalation_exists: bool,
) -> dict[str, Any]:
    sample_ids = audit.get("sample_internal_doc_ids")
    if not isinstance(sample_ids, list) or len(sample_ids) != GREEN_AUDIT_SAMPLE_SIZE:
        raise IntegrityError(f"GREEN audit sample must contain {GREEN_AUDIT_SAMPLE_SIZE} documents")
    if audit.get("green_count") != sum(EXPECTED_GREEN_DISTRIBUTION.values()):
        raise FingerprintMismatch("GREEN audit population count drift")
    documents = {str(row["internal_doc_id"]): row for row in green_documents}
    review_by_id = {str(row["internal_doc_id"]): row for row in reviews}
    if not set(map(str, sample_ids)).issubset(documents):
        raise FingerprintMismatch("GREEN audit sample is no longer a subset of live GREEN")
    details: list[dict[str, Any]] = []
    membership_changes = 0
    finalized = 0
    deferred = 0
    for raw_doc_id in sample_ids:
        doc_id = str(raw_doc_id)
        review = review_by_id.get(doc_id)
        if review is None:
            raise IntegrityError(f"GREEN audit review row is missing: {doc_id}")
        status = str(review.get("status"))
        changed: bool | None = None
        if status == "finalized":
            finalized += 1
            original = _read_json_text(
                documents[doc_id].get("human_references_json"),
                "human_references_json",
            )
            final = _read_json_text(
                review.get("final_references_json") or "[]",
                "final_references_json",
            )
            changed = _membership(original) != _membership(final)
            membership_changes += int(changed)
        elif status == "deferred":
            deferred += 1
        elif status != "pending":
            raise IntegrityError(f"unknown GREEN audit review status: {status!r}")
        details.append(
            {
                "internal_doc_id": doc_id,
                "status": status,
                "legal_membership_changed": changed,
            }
        )
    if escalation_exists or membership_changes:
        status = "escalation_required"
    elif finalized == len(sample_ids) and deferred == 0:
        status = "passed"
    else:
        status = "pending"
    return {
        "schema_version": 1,
        "status": status,
        "sample_size": len(sample_ids),
        "finalized_count": finalized,
        "pending_count": len(sample_ids) - finalized - deferred,
        "deferred_count": deferred,
        "legal_membership_change_count": membership_changes,
        "escalation_artifact_exists": escalation_exists,
        "details": details,
    }


def reconstruct_vuk_213_article_413(
    references: Iterable[dict[str, Any]], *, document_text: str
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Add the footer target only from an exact, deterministic text span."""

    validated = validate_reference_list(list(references))
    already_present = any(
        normalize_reference(reference)["kanun_no"] == "213"
        and normalize_reference(reference)["madde"] == "413"
        for reference in validated
    )
    matches = list(_VUK_413_EVIDENCE_RE.finditer(document_text))
    if already_present or not matches:
        return validated, {
            "policy": "derived_policy_reconstruction_v1",
            "added": False,
            "reason": "already_present" if already_present else "no_exact_evidence",
            "evidence_match_count": len(matches),
        }
    evidence = matches[0].group(0)
    if evidence not in document_text:
        raise IntegrityError("derived VUK 213/413 evidence is not byte-present in text")
    derived = {
        "kanun_no": "213",
        "kanun_ad": "Vergi Usul Kanunu",
        "madde": "413",
        "fikra": "",
        "bent": "",
        "source_text": evidence,
    }
    output = [*validated, derived]
    return output, {
        "policy": "derived_policy_reconstruction_v1",
        "added": True,
        "provenance": "derived_policy_reconstruction",
        "evidence_match": "exact_regex_span",
        "evidence_sha256": sha256_text(evidence),
        "evidence_match_count": len(matches),
    }


def choose_inference_contract(
    *, input_token_counts: Iterable[int], serialized_gold_token_counts: Iterable[int]
) -> dict[str, Any]:
    inputs = list(input_token_counts)
    outputs = list(serialized_gold_token_counts)
    if not inputs or not outputs or min(inputs) <= 0 or min(outputs) <= 0:
        raise ContractError("inference token counts must be non-empty positive integers")
    maximum_input = max(inputs)
    if maximum_input <= INFERENCE_INPUT_TOKENS:
        input_tokens = INFERENCE_INPUT_TOKENS
    elif maximum_input <= INFERENCE_INPUT_FALLBACK_TOKENS:
        input_tokens = INFERENCE_INPUT_FALLBACK_TOKENS
    else:
        raise GateBlocked(
            f"prompt+document tokens={maximum_input} exceed the common 32768 input cap"
        )
    maximum_output = max(outputs)
    output_tokens = (
        INFERENCE_OUTPUT_FALLBACK_TOKENS
        if maximum_output > int(INFERENCE_OUTPUT_TOKENS * 0.80)
        else INFERENCE_OUTPUT_TOKENS
    )
    if maximum_output > output_tokens:
        raise GateBlocked(
            f"serialized gold tokens={maximum_output} exceed output cap={output_tokens}"
        )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "maximum_observed_input_tokens": maximum_input,
        "maximum_serialized_gold_tokens": maximum_output,
        "output_upgrade_threshold_tokens": int(INFERENCE_OUTPUT_TOKENS * 0.80),
        "temperature": 0.0,
        "thinking": False,
        "window_fallback": "lossless_on_parse_fail_or_true_length_truncation",
    }


def measure_inference_contract(
    *, tokenizer: Any, universes: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    input_counts: list[int] = []
    output_counts: list[int] = []
    summaries: dict[str, Any] = {}
    for name, rows in universes.items():
        universe_inputs: list[int] = []
        universe_outputs: list[int] = []
        for row in rows:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(row["text"])},
            ]
            tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
            if not isinstance(tokens, list):
                raise ContractError("tokenizer did not return input token IDs")
            target = json.dumps(
                validate_reference_list(row["references"]),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            target_tokens = tokenizer.encode(target)
            if not isinstance(target_tokens, list):
                raise ContractError("tokenizer did not return target token IDs")
            universe_inputs.append(len(tokens))
            universe_outputs.append(len(target_tokens))
        if not universe_inputs:
            raise ContractError(f"inference universe is empty: {name}")
        input_counts.extend(universe_inputs)
        output_counts.extend(universe_outputs)
        summaries[name] = {
            "document_count": len(rows),
            "maximum_input_tokens": max(universe_inputs),
            "maximum_serialized_gold_tokens": max(universe_outputs),
        }
    return {
        "selected": choose_inference_contract(
            input_token_counts=input_counts,
            serialized_gold_token_counts=output_counts,
        ),
        "universes": summaries,
    }


def _load_inference_universes(
    *, source: dict[str, Any], repo_root: Path
) -> dict[str, list[dict[str, Any]]]:
    split = source["summary"]["split"]
    documents = source["documents"]
    universes = {"canonical_test_50": [documents[doc_id] for doc_id in split["test"]]}
    paths = _historical_paths(repo_root)
    for label, docs_key, gold_key in (
        (
            "external_validation_100",
            "external_validation_docs",
            "external_validation_gold",
        ),
        ("external_sealed_400", "external_sealed_docs", "external_sealed_gold"),
    ):
        docs = _read_json(paths[docs_key])
        gold = _read_json(paths[gold_key])
        if not isinstance(docs, list) or not isinstance(gold, list):
            raise ContractError(f"{label} docs/gold must be JSON arrays")
        gold_by_id = {int(row["doc_id"]): row for row in gold}
        rows = []
        for document in docs:
            doc_id = int(document["doc_id"])
            if doc_id not in gold_by_id:
                raise IntegrityError(f"{label} gold missing doc_id={doc_id}")
            rows.append(
                {
                    "text": document["text"],
                    "references": gold_by_id[doc_id].get("references", []),
                }
            )
        if len(rows) != len(gold) or len({int(row["doc_id"]) for row in docs}) != len(rows):
            raise IntegrityError(f"{label} docs/gold coverage mismatch")
        universes[label] = rows
    return universes


def refit_optimizer_updates(
    selected_development_update: int, *, refit_training_view_rows: int | None = None
) -> int:
    if selected_development_update <= 0:
        raise ValueError("selected_development_update must be positive")
    scaled = round(selected_development_update * REFIT_DOCUMENTS / DEVELOPMENT_DOCUMENTS)
    if refit_training_view_rows is None:
        return scaled
    return max(scaled, Q36TrainingContract().optimizer_updates(refit_training_view_rows))


def promotion_decision(
    *,
    base: dict[str, float],
    canonical_only: dict[str, float],
    augmented: dict[str, float],
    g0: dict[str, float],
    paired_ci_lower: float,
    coverage_count: int,
    parse_count: int,
    truncation_count: int,
    runaway_count: int,
) -> dict[str, Any]:
    gates = {
        "sealed_coverage_400": coverage_count == 400,
        "sealed_parse_400": parse_count == 400,
        "unresolved_truncation_zero": truncation_count == 0,
        "unresolved_runaway_zero": runaway_count == 0,
        "core_f1_above_base": augmented["core_f1"] > base["core_f1"],
        "paired_ci_lower_nonnegative": paired_ci_lower >= 0,
        "docwise_not_below_base": augmented["docwise"] >= base["docwise"],
        "recall_drop_vs_base_at_most_0_005": (augmented["recall"] >= base["recall"] - 0.005),
        "f1_drop_vs_canonical_at_most_0_005": (
            augmented["core_f1"] >= canonical_only["core_f1"] - 0.005
        ),
        "recall_drop_vs_canonical_at_most_0_005": (
            augmented["recall"] >= canonical_only["recall"] - 0.005
        ),
        "core_f1_above_g0": augmented["core_f1"] > g0["core_f1"],
        "docwise_above_g0": augmented["docwise"] > g0["docwise"],
    }
    return {
        "promote": all(gates.values()),
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
    }


def _validate_prediction_array(path: Path) -> None:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ContractError("prediction file root must be a list")
    doc_ids: set[int] = set()
    for index, row in enumerate(payload):
        if not isinstance(row, dict) or not isinstance(row.get("doc_id"), int):
            raise ContractError(f"prediction[{index}] has no integer doc_id")
        doc_id = int(row["doc_id"])
        if doc_id in doc_ids:
            raise ContractError(f"duplicate prediction doc_id={doc_id}")
        doc_ids.add(doc_id)
        validate_reference_list(row.get("references", []))


def _seal_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise FingerprintMismatch(f"immutable output already differs: {path}")
        return
    write_bytes_atomic(path, payload, validator=_validate_prediction_array)


def write_prediction_views(*, raw_predictions: Path, output_dir: Path) -> dict[str, Any]:
    """Seal exact raw bytes and derive the symmetric production prediction view."""

    raw_predictions = raw_predictions.resolve()
    _validate_prediction_array(raw_predictions)
    rows = _read_json(raw_predictions)
    raw_bytes = raw_predictions.read_bytes()
    output_dir = output_dir.resolve()
    raw_output = output_dir / "raw_full_extraction.json"
    production_output = output_dir / "production_filtered_213_413.json"
    _seal_bytes(raw_output, raw_bytes)
    production_rows: list[dict[str, Any]] = []
    removed = 0
    for row in rows:
        filtered, audit = apply_reference_policy(
            row.get("references", []), policy_id=DEFAULT_REFERENCE_POLICY_ID
        )
        removed += int(audit["removed_reference_count"])
        production_rows.append({**row, "references": filtered})
    serialized = (
        json.dumps(
            production_rows,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _seal_bytes(production_output, serialized)
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "source_path": str(raw_predictions),
        "document_count": len(rows),
        "raw_full_extraction": {
            "path": str(raw_output),
            "sha256": sha256_file(raw_output),
            "byte_identical_to_source": raw_output.read_bytes() == raw_bytes,
        },
        "production_filtered_213_413": {
            "path": str(production_output),
            "sha256": sha256_file(production_output),
            "policy": reference_policy_spec(DEFAULT_REFERENCE_POLICY_ID),
            "removed_reference_count": removed,
        },
        "raw_prediction_modified": False,
    }
    manifest_path = output_dir / "VIEWS_MANIFEST.json"
    if manifest_path.exists():
        existing = _read_json(manifest_path)
        if existing != manifest:
            raise FingerprintMismatch("immutable prediction-view manifest differs")
    else:
        write_json_atomic(manifest_path, manifest)
    return manifest


def _filtered_gold_payload(payload: Any) -> tuple[Any, int]:
    if isinstance(payload, list):
        output = []
        removed = 0
        for index, row in enumerate(payload):
            if not isinstance(row, dict):
                raise ContractError(f"gold[{index}] must be an object")
            filtered, audit = apply_reference_policy(
                row.get("references", []), policy_id=DEFAULT_REFERENCE_POLICY_ID
            )
            removed += int(audit["removed_reference_count"])
            output.append({**row, "references": filtered})
        return output, removed
    if isinstance(payload, dict):
        filtered, audit = apply_reference_policy(
            payload.get("references", []), policy_id=DEFAULT_REFERENCE_POLICY_ID
        )
        return {**payload, "references": filtered}, int(audit["removed_reference_count"])
    raise ContractError("gold root must be an object or list")


def _seal_json_payload(path: Path, payload: Any) -> None:
    serialized = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    if path.exists():
        if path.read_bytes() != serialized:
            raise FingerprintMismatch(f"immutable JSON output already differs: {path}")
        return
    write_bytes_atomic(path, serialized, validator=lambda candidate: _read_json(candidate))


def write_production_gold_view(*, source: Path, output: Path) -> dict[str, Any]:
    """Derive the symmetric 213/413-filtered gold view without touching source."""

    source = source.resolve()
    output = output.resolve()
    source_files = sorted(source.glob("doc_*.json")) if source.is_dir() else [source]
    if not source_files or any(not path.is_file() for path in source_files):
        raise ContractError(f"gold source has no readable JSON files: {source}")
    source_hashes = {str(path): sha256_file(path) for path in source_files}
    outputs: list[Path] = []
    removed = 0
    if source.is_dir():
        output.mkdir(parents=True, exist_ok=True)
        for path in source_files:
            filtered, count = _filtered_gold_payload(_read_json(path))
            target = output / path.name
            _seal_json_payload(target, filtered)
            outputs.append(target)
            removed += count
        manifest_path = output / "VIEW_MANIFEST.json"
    else:
        filtered, removed = _filtered_gold_payload(_read_json(source))
        _seal_json_payload(output, filtered)
        outputs.append(output)
        manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if source_hashes != {str(path): sha256_file(path) for path in source_files}:
        raise IntegrityError("gold source changed while deriving production view")
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "view": "production_filtered_213_413",
        "source": str(source),
        "source_files": [
            {"path": path, "sha256": digest} for path, digest in source_hashes.items()
        ],
        "output_files": [{"path": str(path), "sha256": sha256_file(path)} for path in outputs],
        "document_file_count": len(outputs),
        "removed_reference_count": removed,
        "policy": reference_policy_spec(DEFAULT_REFERENCE_POLICY_ID),
        "source_modified": False,
    }
    _seal_json_payload(manifest_path, manifest)
    return manifest


def _message_example(text: str, references: list[dict[str, Any]]) -> dict[str, Any]:
    target = json.dumps(
        validate_reference_list(references),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": target},
        ]
    }


def _write_training_sources(
    *,
    data_root: Path,
    source: dict[str, Any],
    green_documents: list[dict[str, Any]],
    classification: dict[str, Any],
) -> dict[str, Any]:
    entry_by_id = {row["internal_doc_id"]: row for row in classification["entries"]}
    canonical_split = source["summary"]["split"]
    canonical_documents = source["documents"]
    canonical_train_ids = canonical_split["train"]
    canonical_train = [
        _message_example(
            canonical_documents[doc_id]["text"],
            canonical_documents[doc_id]["references"],
        )
        for doc_id in canonical_train_ids
    ]
    external_rows: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    provenance: list[dict[str, Any]] = []
    for document in green_documents:
        entry = entry_by_id[str(document["internal_doc_id"])]
        if entry["category"] != "external_train":
            continue
        historical_doc_id = entry["historical_doc_id"]
        if not isinstance(historical_doc_id, int):
            raise IntegrityError("external-train row has no historical doc ID")
        human = _read_json_text(document["human_references_json"], "human_references_json")
        references, reconstruction = reconstruct_vuk_213_article_413(
            human, document_text=str(document["text"])
        )
        example = _message_example(str(document["text"]), references)
        external_rows.append((historical_doc_id, example, reconstruction))
        provenance.append(
            {
                "internal_doc_id": str(document["internal_doc_id"]),
                "public_doc_id": str(document["public_doc_id"]),
                "historical_doc_id": historical_doc_id,
                "text_sha256": str(document["text_sha256"]),
                "target_reference_count": len(references),
                "reconstruction": reconstruction,
                "model_prediction_used": False,
            }
        )
    external_rows.sort(key=lambda item: item[0])
    provenance.sort(key=lambda item: item["historical_doc_id"])
    if len(external_rows) != DEVELOPMENT_EXTERNAL_DOCUMENTS:
        raise IntegrityError(
            f"external development export must contain 156 docs, got {len(external_rows)}"
        )
    external_ids = [row[0] for row in external_rows]
    if len(set(external_ids)) != len(external_ids):
        raise IntegrityError("external development export contains duplicate doc IDs")
    canonical_dir = data_root / "canonical_only"
    augmented_dir = data_root / "augmented"
    shared_dir = data_root / "shared"
    write_jsonl_atomic(canonical_dir / "train.jsonl", canonical_train)
    write_json_atomic(canonical_dir / "train_doc_ids.json", canonical_train_ids)
    augmented = [*canonical_train, *(row[1] for row in external_rows)]
    augmented_ids = [*canonical_train_ids, *external_ids]
    write_jsonl_atomic(augmented_dir / "train.jsonl", augmented)
    write_json_atomic(augmented_dir / "train_doc_ids.json", augmented_ids)
    write_jsonl_atomic(data_root / "external_train_provenance.jsonl", provenance)
    for split_name in ("valid", "test"):
        ids = canonical_split[split_name]
        rows = [
            _message_example(
                canonical_documents[doc_id]["text"],
                canonical_documents[doc_id]["references"],
            )
            for doc_id in ids
        ]
        write_jsonl_atomic(shared_dir / f"canonical_{split_name}.jsonl", rows)
        write_json_atomic(shared_dir / f"canonical_{split_name}_doc_ids.json", ids)
    paths = sorted(path for path in data_root.rglob("*") if path.is_file())
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "prompt_variant": PROMPT_VARIANT,
        "prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "counts": {
            "canonical_only_source_documents": len(canonical_train),
            "augmented_source_documents": len(augmented),
            "external_train_source_documents": len(external_rows),
            "derived_213_413_rows": sum(row[2]["added"] for row in external_rows),
            "external_docs_without_213_413_evidence": sum(
                row[2]["reason"] == "no_exact_evidence"
                for row in external_rows
                if not row[2]["added"]
            ),
        },
        "leakage": {
            "historical_validation_in_train": 0,
            "historical_sealed_test_in_train": 0,
            "canonical_duplicate_cluster_in_train": 0,
            "validation_failed_in_train": 0,
            "incomplete_annotation_in_train": 0,
        },
        "files": [
            {
                "path": path.relative_to(data_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ],
    }
    write_json_atomic(data_root / "MANIFEST.json", manifest)
    return manifest


def bootstrap_q36_p1(
    *,
    config: AppConfig,
    batch_id: str = "neon_wl_v1",
    build_data: bool = False,
    measure_tokens: bool = True,
) -> dict[str, Any]:
    """Seal the Q36-P1 plan and stop before data/compute when a gate is closed."""

    if sha256_text(SYSTEM_PROMPT) != LOCKED_PROMPT_SHA256:
        raise FingerprintMismatch("G0 compact prompt bytes/SHA256 drifted")
    repo_root = _repo_root(config)
    source = validate_canonical_sources(config)
    with Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store:
        batch = store.get_batch(batch_id)
        if batch is None or batch.get("status") not in {"processed", "released"}:
            raise GateBlocked("Q36-P1 requires the processed neon_wl_v1 batch")
        green_documents = store.list_documents(batch_id, router_buckets=["GREEN"])
        reviews = store.list_reviews(batch_id)
    classification = classify_green_pool(green_documents, repo_root=repo_root)
    audit_path = config.public_root / "batches" / batch_id / "green_audit.json"
    audit_payload = _read_json(audit_path)
    escalation_path = config.public_root / "batches" / batch_id / "green_escalation.json"
    audit = inspect_green_audit(
        audit=audit_payload,
        green_documents=green_documents,
        reviews=reviews,
        escalation_exists=escalation_path.exists(),
    )
    snapshot = build_snapshot_manifest(_model_snapshot_path(), expected_revision=Q36_MODEL_REVISION)
    cache_ref = _model_snapshot_path().parents[1] / "refs/main"
    cache_head = cache_ref.read_text(encoding="utf-8").strip() if cache_ref.is_file() else None
    prompt = {
        "variant": PROMPT_VARIANT,
        "sha256": LOCKED_PROMPT_SHA256,
        "bytes": len(SYSTEM_PROMPT.encode("utf-8")),
        "byte_identical_to_g0": True,
    }
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "model_id": Q36_MODEL_ID,
        "model_revision": Q36_MODEL_REVISION,
        "model_fingerprint": snapshot["fingerprint"],
        "prompt": prompt,
        "canonical_manifest_sha256": source["summary"]["canonical_manifest_sha256"],
        "historical_split_sha256": classification["source_artifacts"]["split_manifest"]["sha256"],
        "training_view_policy": TRAINING_VIEW_POLICY,
        "green_distribution": classification["counts"],
    }
    run_id = f"q36_p1_{fingerprint_json(identity)[:12]}"
    sensitive_root = config.sensitive_root / "q36_p1" / run_id
    public_root = config.public_root / "q36_p1" / run_id
    run_root = config.training_runs_root / run_id
    write_json_atomic(sensitive_root / "GREEN_POOL_CLASSIFICATION.json", classification)
    write_json_atomic(sensitive_root / "GREEN_AUDIT_INSPECTION.json", audit)
    inference: dict[str, Any] = {"status": "not_measured"}
    if measure_tokens:
        from mlx_lm.utils import load_tokenizer

        tokenizer = load_tokenizer(
            str(_model_snapshot_path()),
            tokenizer_config_extra={"trust_remote_code": True},
        )
        inference = {
            "status": "measured",
            **measure_inference_contract(
                tokenizer=tokenizer,
                universes=_load_inference_universes(source=source, repo_root=repo_root),
            ),
        }
    blockers: list[str] = []
    if audit["status"] == "pending":
        blockers.append(
            f"GREEN audit incomplete: {audit['finalized_count']}/{audit['sample_size']} finalized"
        )
    elif audit["status"] == "escalation_required":
        blockers.append(
            "GREEN audit found a legal-membership change; full-pool escalation required"
        )
    if inference["status"] != "measured":
        blockers.append("common inference token contract has not been measured")
    data_manifest: dict[str, Any] | None = None
    if build_data:
        if blockers:
            data_manifest = None
        else:
            data_manifest = _write_training_sources(
                data_root=sensitive_root / "data",
                source=source,
                green_documents=green_documents,
                classification=classification,
            )
    training_contract = Q36TrainingContract()
    run_config = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "created_at": _utc_now(),
        "identity": identity,
        "model_snapshot": snapshot,
        "moving_cache_head": cache_head,
        "moving_cache_head_used": False,
        "prompt": prompt,
        "training_contract": asdict(training_contract),
        "development_arms": {
            "base": {"training": "none"},
            "canonical_only": {"source_documents": DEVELOPMENT_CANONICAL_DOCUMENTS},
            "augmented": {"source_documents": DEVELOPMENT_DOCUMENTS},
        },
        "training_horizon": "ceil(training_view_row_count / 4)",
        "inference": inference,
        "evaluation": {
            "views": ["full_extraction", "production_filtered_213_413"],
            "reference_postprocess": "canonical_set",
            "core_reference_view": "row",
            "matching_engine": "greedy",
            "docwise_threshold": 1.0,
            "paired_bootstrap_iterations": 10000,
            "paired_bootstrap_seed": 42,
        },
        "recovery": {
            "checkpoint_every_updates": 25,
            "checkpoint_max_seconds": 600,
            "full_state_required": True,
            "long_run_allowed": False,
            "long_run_started": False,
        },
        "data_manifest": data_manifest,
    }
    write_json_atomic(run_root / "run_config.json", run_config)
    public_preflight = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "generated_at": _utc_now(),
        "status": "blocked" if blockers else "software_preflight_passed_compute_pending",
        "blockers": blockers,
        "long_run_allowed": False,
        "long_run_started": False,
        "prompt": prompt,
        "model": {
            "id": Q36_MODEL_ID,
            "revision": Q36_MODEL_REVISION,
            "snapshot_fingerprint": snapshot["fingerprint"],
            "moving_cache_head": cache_head,
            "moving_cache_head_used": False,
        },
        "canonical": {
            "split_counts": source["summary"]["split_counts"],
            "manifest_sha256": source["summary"]["canonical_manifest_sha256"],
        },
        "green_pool": {
            "counts": classification["counts"],
            "sensitive_manifest_sha256": sha256_file(
                sensitive_root / "GREEN_POOL_CLASSIFICATION.json"
            ),
        },
        "green_audit": {
            key: audit[key]
            for key in (
                "status",
                "sample_size",
                "finalized_count",
                "pending_count",
                "deferred_count",
                "legal_membership_change_count",
                "escalation_artifact_exists",
            )
        },
        "inference": inference,
        "data_export_written": data_manifest is not None,
        "compute_gates": {
            "prompt_byte_sha_locked": True,
            "exact_model_snapshot": True,
            "green_distribution_locked": True,
            "green_audit_passed": audit["status"] == "passed",
            "split_leakage_check": classification["counts"] == EXPECTED_GREEN_DISTRIBUTION,
            "common_inference_contract_measured": inference["status"] == "measured",
            "training_data_export": data_manifest is not None,
            "longest_row_forward_backward_optimizer_smoke": False,
            "minimum_headroom_12_gib": False,
            "real_full_state_resume_equivalence": False,
        },
    }
    public_preflight_path = public_root / "PREFLIGHT.json"
    run_config_path = run_root / "run_config.json"
    write_json_atomic(public_preflight_path, public_preflight, mode=0o644)
    write_text_atomic(
        public_root / "SHA256SUMS.txt",
        (
            f"{sha256_file(public_preflight_path)}  "
            f"{public_preflight_path.relative_to(repo_root).as_posix()}\n"
            f"{sha256_file(run_config_path)}  "
            f"{run_config_path.relative_to(repo_root).as_posix()}\n"
        ),
        mode=0o644,
    )
    if build_data and blockers:
        raise GateBlocked("; ".join(blockers))
    return public_preflight
