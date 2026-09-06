"""Assemble round `k`'s training run: canonical train plus every round already
cleaned, fingerprinted so a different composition always gets a different run.

`train_bootstrap` derives its `run_id` from the canonical manifest, the
example bank, the split manifest and the prompt. Every round shares all four.
If `train_round` fingerprinted only those same inputs, round 2 would resume
into round 1's run directory, find a finished training run already sitting
there, and silently produce round 1's model labelled as round 2's -- a flat
learning curve nobody could explain. The composition -- the ordered cleaned
batch ids and the resulting training-set size -- is folded into the
fingerprint precisely to rule that out.

This entry point is for rounds only. `train_bootstrap` stays exactly as it
is and reproduces the deployed G0 model; a caller reaching for `"G0"` here
wants that function instead, so it is rejected explicitly. Round runs live
under `sensitive_root / "loop"`, never `"g0"`, so their directories can never
collide with G0's.

`execute=True` hands off to `g0_training.run_development` -- the same
compute driver G0's own development candidates run through. `run_development`
gates on its own preflight (a verified model snapshot, a memory-smoke-accepted
sequence length, `long_run_allowed`); this module writes only the scaffolding
`run_development`'s entry gate (`_base_run_context`) demands before it will
even look at that preflight. Actually launching training is therefore still a
separate, later step -- exactly as it is for G0, where `train_bootstrap` and
`run_development` are two separate CLI commands.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .config import AppConfig
from .errors import ContractError
from .fingerprints import fingerprint_json, sha256_text
from .g0 import (
    EXAMPLE_BANK_SHA256,
    PROMPT_VARIANT,
    SYSTEM_PROMPT,
    TRAINING_VIEW_POLICY,
    validate_canonical_sources,
    write_training_data,
)
from .g0_finalize import ROUND_LABEL_PATTERN
from .g0_training import (
    PILOT_LEARNING_RATES,
    PILOT_TARGET_UPDATES,
    CheckpointVerifier,
    EvaluatorRunner,
    WorkerRunner,
    run_development,
)
from .loop_training import CANONICAL_PREFIX, compose_round_training_ids, round_training_documents


def _write_round_scaffolding(
    *,
    config: AppConfig,
    run_id: str,
    generation: str,
    run_fingerprint: str,
    data_dir: Path,
    data_manifest: dict[str, Any],
) -> tuple[Path, Path]:
    """Write exactly what `g0_training._base_run_context` demands of a run.

    Mirrors the parts of `g0.train_bootstrap`'s software-preflight write that a
    round actually needs: a `run_config.json` whose `data_manifest.path` holds
    the training data and whose `training_view` names an in-data-dir jsonl
    with a matching checksum, plus a `preflight.json` carrying the same
    `run_id`. Whatever a real launch still needs beyond that -- the verified
    model snapshot, the memory-smoke-accepted sequence length,
    `long_run_allowed` -- is exactly what real compute preflight measures,
    which is compute and therefore not this function's job; `run_development`
    itself refuses to launch until that measurement has actually happened.

    `preflight.json` is written under `public_root / "g0" / run_id`, not
    `"loop"`: `_base_run_context` hardcodes that path for every run_id it
    accepts, G0's and a round's alike, so a round's preflight has to sit there
    too or `run_development` could never find it. The training data itself
    stays under `sensitive_root / "loop"`, as it always has.
    """
    run_dir = config.training_runs_root / run_id
    preflight_path = config.public_root / "g0" / run_id / "preflight.json"
    train_path = data_dir / "train.jsonl"
    run_config = {
        "schema_version": 1,
        "run_id": run_id,
        "generation": generation,
        "source_fingerprint": run_fingerprint,
        "config_fingerprint": config.fingerprint,
        "model": asdict(config.model),
        "data_manifest": data_manifest,
        "training_view": {
            "jsonl_path": str(train_path),
            "jsonl_sha256": data_manifest["files"]["train"]["jsonl_sha256"],
            "row_count": data_manifest["files"]["train"]["count"],
        },
        "recovery": {"long_run_allowed": False},
    }
    write_json_atomic(run_dir / "run_config.json", run_config)
    preflight = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "software_preflight_passed_compute_pending",
        "long_run_started": False,
    }
    write_json_atomic(preflight_path, preflight, mode=0o644)
    return run_dir, preflight_path


def train_round(
    config: AppConfig,
    *,
    generation: str,
    split: dict[str, list[int]],
    cleaned_batch_ids: Sequence[str],
    execute: bool = False,
    candidate_id: str | None = None,
    target_updates: int = PILOT_TARGET_UPDATES,
    resume: bool = False,
    worker_runner: WorkerRunner | None = None,
    evaluator_runner: EvaluatorRunner | None = None,
    checkpoint_verifier: CheckpointVerifier | None = None,
) -> dict[str, Any]:
    if generation == "G0":
        raise ContractError("train_round does not train G0 -- use train_bootstrap for G0")
    if not isinstance(generation, str) or not ROUND_LABEL_PATTERN.match(generation):
        raise ContractError(f"generation must be a round label like 'M003', got {generation!r}")
    if execute and candidate_id not in PILOT_LEARNING_RATES:
        raise ContractError(
            "execute=True requires an explicit candidate_id, one of "
            f"{sorted(PILOT_LEARNING_RATES)}; got {candidate_id!r}"
        )

    source = validate_canonical_sources(config)
    source_summary = source["summary"]

    cleaned_batches: list[list[str]] = []
    documents: dict[str, dict[str, Any]] = {
        f"{CANONICAL_PREFIX}{doc_id}": {
            "text": payload["text"],
            "references": payload["references"],
        }
        for doc_id, payload in source["documents"].items()
    }
    for batch_id in cleaned_batch_ids:
        batch_documents = round_training_documents(config, batch_ids=[batch_id])
        cleaned_batches.append(list(batch_documents))
        documents.update(batch_documents)

    composed = compose_round_training_ids(split, cleaned_batches)

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
        "generation": generation,
        "cleaned_batch_ids": list(cleaned_batch_ids),
        "training_set_size": len(composed["train"]),
    }
    run_fingerprint = fingerprint_json(fingerprint_payload)
    run_id = f"dqcheck_loop_{run_fingerprint[:12]}"
    data_dir = config.sensitive_root / "loop" / run_id / "data"

    data_manifest = write_training_data(
        output_dir=data_dir,
        documents=documents,
        split=composed,
    )

    run_dir, preflight_path = _write_round_scaffolding(
        config=config,
        run_id=run_id,
        generation=generation,
        run_fingerprint=run_fingerprint,
        data_dir=data_dir,
        data_manifest=data_manifest,
    )

    result: dict[str, Any] = {
        "run_id": run_id,
        "training_documents": len(composed["train"]),
        "validation_documents": len(composed["valid"]),
        "test_documents": len(composed["test"]),
        "cleaned_batch_ids": list(cleaned_batch_ids),
        "data_dir": str(data_dir),
        "data_manifest": data_manifest,
        "run_config_path": str(run_dir / "run_config.json"),
        "preflight_path": str(preflight_path),
    }
    if not execute:
        return result

    assert candidate_id is not None  # guarded above
    development = run_development(
        config=config,
        run_id=run_id,
        candidate_id=candidate_id,
        target_updates=target_updates,
        execute=True,
        resume=resume,
        worker_runner=worker_runner,
        evaluator_runner=evaluator_runner,
        checkpoint_verifier=checkpoint_verifier,
    )
    return {**result, "development": development}
