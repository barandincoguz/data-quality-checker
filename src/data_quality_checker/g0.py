"""Canonical-only Qwen3.5-9B G0 data and training gates."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import shutil
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic, write_jsonl_atomic
from .config import AppConfig
from .constants import (
    CANONICAL_GT_MANIFEST_SHA256,
    EXAMPLE_BANK_SHA256,
    EXEMPLAR_DOC_IDS,
    MINIMUM_MLX_LM,
    MODEL_ID,
    MODEL_REVISION,
)
from .contracts import validate_reference_list
from .errors import ContractError, GateBlocked, IntegrityError
from .fake_training import FakeStatefulTrainer
from .fingerprints import fingerprint_json, sha256_bytes, sha256_file, sha256_text

PROMPT_VARIANT = "few-shot-cot-v3-en-compact-recall-v2"
TRAINING_VIEW_POLICY = (
    "adaptive-context-fallback128-positive-only-dense-max10-tokenmean-looprepair-v10"
)
REFIT_SPLIT_POLICY = "refit-all-494-nominal-valid-and-test"

SYSTEM_PROMPT = (
    "You extract every statutory law reference from Turkish tax rulings "
    "(ozelgeler).\n\n"
    "Return one flat JSON array and include ALL references in the document. "
    "Each item has exactly these string fields: kanun_no, kanun_ad, madde, "
    "fikra, bent, source_text. Use an empty string when a field is absent. "
    "Resolve locally supported anaphora; retain table/cetvel/list references "
    "in the same contract; never invent a law identity or evidence. Deduplicate "
    "the same legal tuple and suppress a generic law-only row when that law has "
    "a specific article row. Output only the JSON array; if no references exist, "
    "output [].\n\n"
    "Recall rules adapted from the official few-shot-cot-v3-en prompt:\n"
    "- Do not become overly conservative. Extract every explicit article, "
    "paragraph, and subparagraph reference, even when secondary regulations "
    "appear nearby.\n"
    "- Preserve every distinct explicit legal tuple. Never replace a specific "
    "tuple with a generic law-only row, and never return [] when an explicit "
    "statutory reference exists.\n"
    "- Scan the entire document once more for dropped madde, fikra, and bent "
    "details before returning JSON.\n\n"
    "Compact demonstration:\n"
    "Input: 488 sayılı Damga Vergisi Kanununun 3 üncü ve 9 uncu maddeleri\n"
    'Output: [{"kanun_no":"488","kanun_ad":"Damga Vergisi Kanunu",'
    '"madde":"3","fikra":"","bent":"","source_text":"488 sayılı '
    'Damga Vergisi Kanununun 3 üncü maddesi"},{"kanun_no":"488",'
    '"kanun_ad":"Damga Vergisi Kanunu","madde":"9","fikra":"",'
    '"bent":"","source_text":"9 uncu maddesi"}]'
)

SEQUENCE_LENGTH_CANDIDATES = (8192, 10240, 12288, 16384, 32768)


@dataclass(frozen=True)
class TrainingContract:
    seed: int = 42
    train_documents: int = 394
    validation_documents: int = 50
    test_documents: int = 50
    maximum_epochs: int = 3
    lora_rank: int = 8
    lora_layers: int = 16
    batch_size: int = 1
    gradient_accumulation: int = 4
    peak_learning_rate: float = 1e-4
    warmup_fraction: float = 0.05
    end_learning_rate: float = 1e-5
    gradient_checkpointing: bool = True
    completion_only_loss: bool = True
    temperature: float = 0.0
    checkpoint_every_optimizer_updates: int = 25
    checkpoint_max_minutes: int = 10
    validation_generation_every_optimizer_updates: int = 25
    minimum_validation_parse_count: int = 49

    @property
    def maximum_optimizer_updates(self) -> int:
        return self.maximum_optimizer_updates_for_rows(self.train_documents)

    def maximum_optimizer_updates_for_rows(self, training_rows: int) -> int:
        if training_rows <= 0:
            raise ValueError("training_rows must be positive")
        micro_steps = self.maximum_epochs * training_rows
        # Never consume examples beyond the declared three-epoch ceiling.
        return micro_steps // self.gradient_accumulation

    def warmup_updates_for(self, total_updates: int) -> int:
        if total_updates <= 0:
            raise ValueError("total_updates must be positive")
        return max(1, round(total_updates * self.warmup_fraction))

    @property
    def warmup_updates(self) -> int:
        return self.warmup_updates_for(self.maximum_optimizer_updates)


@dataclass(frozen=True)
class CheckpointCandidate:
    update: int
    coverage_count: int
    parse_count: int
    empty_output_count: int
    runaway_output_count: int
    core_f1: float
    docwise_accuracy: float
    recall: float
    validation_loss: float


def _version_tuple(value: str) -> tuple[int, ...]:
    pieces: list[int] = []
    for component in value.split("."):
        digits = "".join(char for char in component if char.isdigit())
        if not digits:
            break
        pieces.append(int(digits))
    return tuple(pieces)


def repository_manifest_sha256(gt_dir: Path) -> str:
    """Reproduce the live `shasum files | shasum` canonical GT seal."""

    gt_dir = gt_dir.resolve()
    try:
        repo_root = gt_dir.parents[3]
    except IndexError as exc:
        raise IntegrityError(f"canonical GT path is unexpectedly shallow: {gt_dir}") from exc
    lines = bytearray()
    for path in sorted(gt_dir.glob("doc_*.json")):
        relative = path.relative_to(repo_root).as_posix()
        lines.extend(f"{sha256_file(path)}  {relative}\n".encode())
    return sha256_bytes(bytes(lines))


def _example_doc_ids(payload: Any) -> set[int]:
    if not isinstance(payload, dict) or not isinstance(payload.get("examples"), list):
        raise ContractError("few-shot example bank schema is invalid")
    return {
        int(example["doc_id"])
        for example in payload["examples"]
        if isinstance(example, dict)
        and isinstance(example.get("doc_id"), int)
        and int(example["doc_id"]) != 0
    }


def build_split(doc_ids: list[int], *, seed: int = 42) -> dict[str, list[int]]:
    eligible = sorted(set(doc_ids) - EXEMPLAR_DOC_IDS)
    if len(eligible) != 494:
        raise ContractError(f"expected 494 eligible canonical docs, found {len(eligible)}")
    shuffled = eligible[:]
    random.Random(seed).shuffle(shuffled)
    return {
        "train": sorted(shuffled[:394]),
        "valid": sorted(shuffled[394:444]),
        "test": sorted(shuffled[444:494]),
    }


def validate_canonical_sources(config: AppConfig) -> dict[str, Any]:
    gt_paths = sorted(config.canonical_gt_dir.glob("doc_*.json"))
    if len(gt_paths) != 500:
        raise ContractError(
            f"canonical GT must contain 500 doc_*.json files, found {len(gt_paths)}"
        )
    digest = repository_manifest_sha256(config.canonical_gt_dir)
    if digest != CANONICAL_GT_MANIFEST_SHA256:
        raise IntegrityError(
            f"canonical GT manifest drift: expected {CANONICAL_GT_MANIFEST_SHA256}, got {digest}"
        )
    documents: dict[int, dict[str, Any]] = {}
    reference_count = 0
    for path in gt_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid canonical JSON {path}: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("doc_id"), int):
            raise ContractError(f"canonical doc_id missing in {path}")
        doc_id = int(payload["doc_id"])
        if path.name != f"doc_{doc_id}.json":
            raise ContractError(f"canonical filename/doc_id mismatch in {path}")
        if doc_id in documents:
            raise ContractError(f"duplicate canonical doc_id {doc_id}")
        if not isinstance(payload.get("text"), str) or not payload["text"].strip():
            raise ContractError(f"canonical document {doc_id} has no text")
        references = validate_reference_list(payload.get("references"))
        reference_count += len(references)
        documents[doc_id] = {**payload, "references": references}
    if set(documents) != set(range(1, 501)):
        raise ContractError("canonical doc_id coverage must be exactly 1..500")

    if sha256_file(config.example_bank_path) != EXAMPLE_BANK_SHA256:
        raise IntegrityError("few-shot example bank checksum drift")
    examples = json.loads(config.example_bank_path.read_text(encoding="utf-8"))
    real_example_ids = _example_doc_ids(examples)
    if real_example_ids != set(EXEMPLAR_DOC_IDS):
        raise ContractError(f"few-shot exemplar IDs drifted: {sorted(real_example_ids)}")

    split = build_split(list(documents), seed=42)
    reference_manifest = json.loads(
        config.reference_split_manifest_path.read_text(encoding="utf-8")
    )
    if reference_manifest.get("seed") != 42 or reference_manifest.get("splits") != split:
        raise IntegrityError("generated 394/50/50 split differs from cleaned-GT v2 manifest")
    if set(reference_manifest.get("exemplars_excluded", [])) != set(EXEMPLAR_DOC_IDS):
        raise IntegrityError("reference split exemplar exclusion differs from live example bank")
    return {
        "documents": documents,
        "summary": {
            "canonical_gt_dir": str(config.canonical_gt_dir),
            "canonical_doc_count": len(documents),
            "canonical_reference_count": reference_count,
            "canonical_manifest_sha256": digest,
            "example_bank_path": str(config.example_bank_path),
            "example_bank_sha256": EXAMPLE_BANK_SHA256,
            "exemplar_doc_ids": sorted(EXEMPLAR_DOC_IDS),
            "split_manifest_path": str(config.reference_split_manifest_path),
            "split_manifest_sha256": sha256_file(config.reference_split_manifest_path),
            "split_counts": {name: len(ids) for name, ids in split.items()},
            "split": split,
        },
    }


def _training_example(document: dict[str, Any]) -> dict[str, Any]:
    target = json.dumps(document["references"], ensure_ascii=False, separators=(",", ":"))
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": document["text"]},
            {"role": "assistant", "content": target},
        ]
    }


def write_training_data(
    *,
    output_dir: Path,
    documents: dict[int, dict[str, Any]],
    split: dict[str, list[int]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for split_name in ("train", "valid", "test"):
        ids = split[split_name]
        examples = [_training_example(documents[doc_id]) for doc_id in ids]
        data_path = output_dir / f"{split_name}.jsonl"
        ids_path = output_dir / f"{split_name}_doc_ids.json"
        write_jsonl_atomic(data_path, examples)
        write_json_atomic(ids_path, ids)
        files[split_name] = {
            "count": len(examples),
            "jsonl_sha256": sha256_file(data_path),
            "doc_ids_sha256": sha256_file(ids_path),
        }
    split_manifest = {
        "schema_version": 1,
        "seed": 42,
        "splits": split,
        "exemplars_excluded": sorted(EXEMPLAR_DOC_IDS),
    }
    write_json_atomic(output_dir / "split_manifest.json", split_manifest)
    return {
        "path": str(output_dir),
        "split_manifest_sha256": sha256_file(output_dir / "split_manifest.json"),
        "files": files,
    }


def environment_preflight(config: AppConfig) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in ("mlx", "mlx-lm", "huggingface-hub", "pytest", "Flask"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    mlx_lm_version = versions["mlx-lm"]
    version_ok = bool(
        mlx_lm_version
        and _version_tuple(mlx_lm_version) >= _version_tuple(config.model.minimum_mlx_lm)
    )
    cache_name = "models--" + config.model.model_id.replace("/", "--")
    cache_roots = []
    for candidate in (
        os.environ.get("HF_HUB_CACHE"),
        "/opt/llm-lab/hf-cache/hub",
        str(Path.home() / ".cache" / "huggingface" / "hub"),
    ):
        if candidate and candidate not in cache_roots:
            cache_roots.append(candidate)
    snapshots = [
        str(Path(root) / cache_name / "snapshots" / config.model.revision)
        for root in cache_roots
        if (Path(root) / cache_name / "snapshots" / config.model.revision).is_dir()
    ]
    usage = shutil.disk_usage(config.source_path.parent)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "minimum_mlx_lm": MINIMUM_MLX_LM,
        "mlx_lm_version_ok": version_ok,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "exact_model_snapshots": snapshots,
        "model_cached": bool(snapshots),
        "disk": {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        },
    }


def select_sequence_length(
    token_counts: list[int],
    *,
    memory_smoke: Callable[[int], bool],
) -> int:
    if not token_counts or min(token_counts) <= 0:
        raise ContractError("token counts must be non-empty positive integers")
    maximum = max(token_counts)
    for candidate in SEQUENCE_LENGTH_CANDIDATES:
        if candidate >= maximum and memory_smoke(candidate):
            return candidate
    raise GateBlocked(
        f"no sequence-length candidate covers max_tokens={maximum} and passes memory smoke"
    )


# The order checkpoints are ranked by once they pass the eligibility gates.
# A leading "-" means lower is better. `loop_selection.py` imports this
# constant rather than declaring its own copy, so the per-round selection
# record's declared ordering can never drift from what `select_checkpoint`
# actually does.
CHECKPOINT_TIE_BREAK_ORDER = ("core_f1", "docwise_accuracy", "recall", "-validation_loss")


def _checkpoint_tie_break_key(candidate: CheckpointCandidate) -> tuple[float, ...]:
    key: list[float] = []
    for field in CHECKPOINT_TIE_BREAK_ORDER:
        descending, name = (True, field[1:]) if field.startswith("-") else (False, field)
        value = float(getattr(candidate, name))
        key.append(-value if descending else value)
    return tuple(key)


def select_checkpoint(
    candidates: list[CheckpointCandidate],
    *,
    validation_documents: int = 50,
    minimum_parse_count: int = 49,
) -> CheckpointCandidate:
    eligible = [
        candidate
        for candidate in candidates
        if candidate.coverage_count == validation_documents
        and candidate.parse_count >= minimum_parse_count
        and candidate.empty_output_count == 0
        and candidate.runaway_output_count == 0
    ]
    if not eligible:
        raise GateBlocked(
            "no checkpoint passes coverage/parse/collapse eligibility gates "
            f"(need coverage=={validation_documents}, parse>={minimum_parse_count})"
        )
    return max(eligible, key=_checkpoint_tie_break_key)


def final_refit_updates(selected_updates: int) -> int:
    if selected_updates <= 0:
        raise ValueError("selected_updates must be positive")
    return round(selected_updates * 494 / 394)


def assert_test_improves_base(*, base_core_f1: float, tuned_core_f1: float) -> None:
    if tuned_core_f1 <= base_core_f1:
        raise GateBlocked(
            f"G0 test gate failed: tuned core-F1 {tuned_core_f1:.6f} "
            f"does not exceed base {base_core_f1:.6f}"
        )


def _fake_resume_regression(root: Path) -> dict[str, Any]:
    data = [0.25, -0.5, 1.25, 0.75]
    uninterrupted = FakeStatefulTrainer(data=data, seed=42, checkpoint_root=root / "uninterrupted")
    uninterrupted.run(target_updates=12, checkpoint_every=4)
    uninterrupted_fingerprint = uninterrupted.trajectory_fingerprint()

    interrupted = FakeStatefulTrainer(data=data, seed=42, checkpoint_root=root / "resumed")
    checkpoint = interrupted.run(target_updates=4, checkpoint_every=4)
    resumed = FakeStatefulTrainer(data=data, seed=42, checkpoint_root=root / "resumed")
    resumed.resume(checkpoint)
    resumed.run(target_updates=12, checkpoint_every=4)
    resumed_fingerprint = resumed.trajectory_fingerprint()
    result = {
        "status": "passed" if uninterrupted_fingerprint == resumed_fingerprint else "failed",
        "uninterrupted_fingerprint": uninterrupted_fingerprint,
        "resumed_fingerprint": resumed_fingerprint,
        "checkpoint": str(checkpoint),
        "scope": "software_fake_backend_only",
    }
    if result["status"] != "passed":
        raise IntegrityError("fake full-state failure-resume equivalence failed")
    return result


def refit_split(split: dict[str, list[int]]) -> dict[str, list[int]]:
    """Derive the final-refit split: train on every canonical document.

    Development selection is already finished when a refit runs, so the refit
    trains on all 494 documents (train+valid+test) and keeps the canonical
    validation ids only as a *nominal* validation view that drives checkpoint
    cadence — never selection. ``test`` is empty by construction: a refit has
    no held-out split, which is why its checkpoint is chosen by update count
    (``final_refit_updates``) rather than by a validation metric.

    ``valid`` and ``test`` keep the canonical ids so the pipeline's non-empty
    split gates still hold, but both are NOMINAL: every one of their documents
    is also in ``train``, so neither may ever be reported as a holdout score.
    ``split_policy`` records this explicitly in run_config and preflight.
    """
    all_ids = sorted(set(split["train"]) | set(split["valid"]) | set(split["test"]))
    return {"train": all_ids, "valid": list(split["valid"]), "test": list(split["test"])}


def train_bootstrap(
    *, config: AppConfig, generation: str, execute: bool = False, refit: bool = False
) -> dict[str, Any]:
    if generation != "G0":
        raise ContractError("only canonical-only G0 is supported in v1")
    source = validate_canonical_sources(config)
    source_summary = source["summary"]
    prompt = {
        "variant": PROMPT_VARIANT,
        "sha256": sha256_text(SYSTEM_PROMPT),
        "example_bank_sha256": EXAMPLE_BANK_SHA256,
    }
    fingerprint_payload: dict[str, Any] = {
        "canonical_manifest_sha256": source_summary["canonical_manifest_sha256"],
        "example_bank_sha256": source_summary["example_bank_sha256"],
        "split_manifest_sha256": source_summary["split_manifest_sha256"],
        "prompt": prompt,
        "training_view_policy": TRAINING_VIEW_POLICY,
    }
    if refit:
        # Only a refit adds this key, so existing development run ids stay byte
        # identical while the refit necessarily gets its own run directory.
        fingerprint_payload["split_policy"] = REFIT_SPLIT_POLICY
    source_fingerprint = fingerprint_json(fingerprint_payload)
    run_id = f"dqcheck_g0_qwen3_5_9b_{source_fingerprint[:12]}"
    run_dir = config.training_runs_root / run_id
    data_dir = config.sensitive_root / "g0" / run_id / "data"
    public_dir = config.public_root / "g0" / run_id
    effective_split = refit_split(source_summary["split"]) if refit else source_summary["split"]
    data_manifest = write_training_data(
        output_dir=data_dir,
        documents=source["documents"],
        split=effective_split,
    )
    training_contract = TrainingContract()
    environment = environment_preflight(config)
    # The default software-preflight path (execute=False) must pass without MLX
    # installed (see plan Global Constraints: local default tests import no MLX).
    # The mlx-lm version gate only applies to the real compute run.
    if execute and not environment["mlx_lm_version_ok"]:
        raise GateBlocked(
            f"mlx-lm>={MINIMUM_MLX_LM} is required; observed {environment['packages']['mlx-lm']}"
        )
    fake_resume = _fake_resume_regression(run_dir / "software_resume_regression")
    model_fingerprint = fingerprint_json(
        {"model_id": config.model.model_id, "revision": config.model.revision}
    )
    run_config = {
        "schema_version": 1,
        "run_id": run_id,
        "generation": generation,
        "source_fingerprint": source_fingerprint,
        "config_fingerprint": config.fingerprint,
        "model_fingerprint": model_fingerprint,
        "model": asdict(config.model),
        "prompt": prompt,
        "training_view_policy": TRAINING_VIEW_POLICY,
        "split_policy": REFIT_SPLIT_POLICY if refit else "development-394-50-50",
        "training": asdict(training_contract),
        "sequence_length_candidates": list(SEQUENCE_LENGTH_CANDIDATES),
        "source": {key: value for key, value in source_summary.items() if key != "split"},
        "data_manifest": data_manifest,
        "recovery": {
            "checkpoint_every_optimizer_updates": 25,
            "checkpoint_max_minutes": 10,
            "required_components": [
                "model_or_adapter",
                "optimizer",
                "scheduler",
                "global_step",
                "python_rng",
                "mlx_rng",
                "data_cursor",
            ],
            "fake_failure_resume": fake_resume,
            "real_mlx_failure_resume": "pending",
            "long_run_allowed": False,
        },
    }
    write_json_atomic(run_dir / "run_config.json", run_config)
    preflight = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "software_preflight_passed_compute_pending",
        "long_run_started": False,
        "prompt": prompt,
        "training_view_policy": TRAINING_VIEW_POLICY,
        "split_policy": REFIT_SPLIT_POLICY if refit else "development-394-50-50",
        "source": {key: value for key, value in source_summary.items() if key != "split"},
        "environment": environment,
        "training_contract": asdict(training_contract),
        "data_manifest": data_manifest,
        "fake_failure_resume": fake_resume,
        "compute_gates": {
            "exact_model_snapshot": environment["model_cached"],
            "tokenizer_chat_template": False,
            "thinking_disabled": False,
            "sequence_length_memory_smoke": False,
            "two_step_training": False,
            "one_document_generation": False,
            "adapter_load": False,
            "real_full_state_failure_resume": False,
        },
    }
    write_json_atomic(public_dir / "preflight.json", preflight, mode=0o644)
    if execute:
        from .mlx_compute import run_compute_acceptance_preflight

        return run_compute_acceptance_preflight(
            config=config,
            run_dir=run_dir,
            data_dir=data_dir,
            preflight_path=public_dir / "preflight.json",
        )
    return preflight
