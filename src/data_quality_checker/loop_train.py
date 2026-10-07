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

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .config import AppConfig
from .constants import MINIMUM_MLX_LM
from .errors import ContractError, GateBlocked
from .fingerprints import fingerprint_json, sha256_text
from .g0 import (
    EXAMPLE_BANK_SHA256,
    TRAINING_VIEW_POLICY,
    TrainingContract,
    environment_preflight,
    loop_prompt,
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
from .loop_sources import has_loop_settings, loop_split_policy, validate_loop_sources
from .loop_training import (
    RAW_PLACEMENT_RULE,
    compose_round_training_ids,
    round_raw_training_documents,
    round_training_documents,
)

# A round trains on the loop split, not on G0's. The label goes into the
# fingerprint-adjacent record so a run can never be mistaken for a G0 run
# that happened to see fewer documents.
LOOP_SPLIT_POLICY = "loop-300-50-150"


TRAINING_VIEWS = ("windows", "full_documents")

# Where a round's cleaned batches take their references from. "clean" is the
# loop: the released lists. "raw" is the noisy-vs-clean ablation's `M_k_raw`
# (pre-registered 4 Oct 2026): the same documents, the annotator's raw lists.
LABEL_SOURCES = ("clean", "raw")


def training_view_mode(config: AppConfig, view: str | None = None) -> str:
    """How a round's documents reach the trainer: `view` if given, else "windows".

    Both views were trained for the GT750 M000 (28 Sep 2026): windows of 1,536
    tokens (`dqcheck_loop_519d9b43e168`, validation F1 0.927) and whole
    documents with the chunk-parallel recurrence (`dqcheck_loop_02fd08a7a9d1`,
    0.920; paired bootstrap 95% [-0.026, +0.014], not significant). The
    pre-registered rule -- validation F1 -- chose the windowed model: the loop's
    sealed models are windowed, so every round trains exactly as M000 did. The
    whole-document model is kept as a second line and trained every round on
    the same data (user decision, 28 Sep 2026), never sealed
    (`loop_ops.train_loop_candidate`).
    """
    if view is None:
        loop = (getattr(config, "raw", None) or {}).get("loop") or {}
        view = str(loop.get("training_view", "windows"))
    if view not in TRAINING_VIEWS:
        raise ContractError(f"training view must be one of {TRAINING_VIEWS}, got {view!r}")
    return view


def full_epoch_updates(training_view: Path, *, gradient_accumulation: int) -> int:
    """One pass over this round's windowed view.

    Derived from the view's own row count every round (HANDOFF: never carried
    over from a previous round's 506 or 549), so the cosine schedule anneals
    across exactly one epoch whatever the round's size.
    """
    rows = sum(1 for line in training_view.read_text(encoding="utf-8").splitlines() if line.strip())
    if rows <= 0:
        raise ContractError(f"training view {training_view} is empty")
    return math.ceil(rows / gradient_accumulation)


def _write_round_scaffolding(
    *,
    config: AppConfig,
    run_id: str,
    generation: str,
    run_fingerprint: str,
    data_dir: Path,
    data_manifest: dict[str, Any],
    composed: dict[str, list[str]],
    prompt: dict[str, Any],
    source_summary: dict[str, Any],
    execute: bool,
    split_policy: str = LOOP_SPLIT_POLICY,
    view_mode: str = "windows",
    recurrence: str | None = None,
    label_source: str = "clean",
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
    run_config = {
        "schema_version": 1,
        "run_id": run_id,
        "generation": generation,
        "source_fingerprint": run_fingerprint,
        "config_fingerprint": config.fingerprint,
        "model": asdict(config.model),
        "data_manifest": data_manifest,
        # No `training_view` here on purpose. The trainer reads that key as the
        # file it trains on, and the file it must train on is the windowed view
        # that the compute acceptance preflight builds from train.jsonl with the
        # tokenizer -- for G0 that turned 394 documents into 3399 windows.
        # Naming the flat train.jsonl instead would train on untruncated
        # documents whose targets still carry `source_text`, a field the system
        # prompt no longer mentions. The preflight writes the real view.
        "recovery": {"long_run_allowed": False},
        "training_view_mode": view_mode,
        **({"recurrence": recurrence} if recurrence else {}),
        **({"label_source": label_source} if label_source != "clean" else {}),
    }
    write_json_atomic(run_dir / "run_config.json", run_config)
    # The contract states this round's own split. A bare TrainingContract()
    # claims G0's geometry -- 394 train and 50 test documents -- and the
    # optimizer-update ceiling is derived from it. Capacity and optimisation
    # stay at their defaults on purpose: if they moved between rounds, a
    # round-over-round difference would stop meaning "more data".
    training_contract = TrainingContract(
        train_documents=len(composed["train"]),
        validation_documents=len(composed["valid"]),
        test_documents=len(composed["test"]),
    )
    environment = environment_preflight(config)
    if execute and not environment["mlx_lm_version_ok"]:
        raise GateBlocked(
            f"mlx-lm>={MINIMUM_MLX_LM} is required; observed {environment['packages']['mlx-lm']}"
        )
    preflight = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "software_preflight_passed_compute_pending",
        "long_run_started": False,
        "prompt": prompt,
        "training_view_policy": TRAINING_VIEW_POLICY,
        "split_policy": split_policy,
        "source": {key: value for key, value in source_summary.items() if key != "split"},
        "environment": environment,
        "training_contract": asdict(training_contract),
        "data_manifest": data_manifest,
        # The compute preflight flips each of these as its gate passes. It
        # writes into this dict immediately, so the keys must exist first.
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
    target_updates: int | None = PILOT_TARGET_UPDATES,
    resume: bool = False,
    view: str | None = None,
    labels: str = "clean",
    compute_preflight_runner: Callable[..., dict[str, Any]] | None = None,
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
    if labels not in LABEL_SOURCES:
        raise ContractError(f"labels must be one of {LABEL_SOURCES}, got {labels!r}")
    if labels == "raw" and not cleaned_batch_ids:
        raise ContractError("the raw-label arm needs at least one cleaned batch; round 0 has none")

    if has_loop_settings(config):
        source = validate_loop_sources(config)
        if {name: list(split[name]) for name in ("train", "valid", "test")} != source["split"]:
            raise ContractError("split differs from the sealed loop split in SPLIT.json")
        split_policy = loop_split_policy(config)
    else:
        source = validate_canonical_sources(config)
        split_policy = LOOP_SPLIT_POLICY
    source_summary = source["summary"]

    cleaned_batches: list[list[str]] = []
    # Canonical documents keep their integer id: that is what the ground-truth
    # directory and the evaluator resolve, and what G0's own training data
    # used. Round-cleaned documents keep their own `d000001`-shaped id, so the
    # two kinds stay distinguishable without decorating either.
    documents: dict[int | str, dict[str, Any]] = {
        doc_id: {
            "text": payload["text"],
            "references": payload["references"],
        }
        for doc_id, payload in source["documents"].items()
    }
    raw_label_audit: dict[str, Any] | None = None
    if labels == "raw":
        raw_documents, raw_label_audit = round_raw_training_documents(
            config, batch_ids=list(cleaned_batch_ids)
        )
    for batch_id in cleaned_batch_ids:
        if raw_label_audit is not None:
            # The release's documents once it exists, the whole batch before it
            # (`round_raw_training_documents`); in id order either way.
            batch_documents = {
                doc_id: raw_documents[doc_id] for doc_id in raw_label_audit["documents_by_batch"][batch_id]
            }
        else:
            batch_documents = round_training_documents(config, batch_ids=[batch_id])
        cleaned_batches.append(list(batch_documents))
        documents.update(batch_documents)

    composed = compose_round_training_ids(split, cleaned_batches)

    prompt_variant, system_prompt = loop_prompt(config)
    prompt = {
        "variant": prompt_variant,
        "sha256": sha256_text(system_prompt),
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
    view_mode = training_view_mode(config, view)
    recurrence: str | None = None
    if view_mode != "windows":
        # A different view is a different run: it must never resume into the
        # windowed run's directory (absent for "windows", so those ids stand).
        # So is a different training recurrence: the preflight measured memory
        # and resume equivalence under one, and a run is never continued under
        # another.
        from .mlx_memory import RECURRENCE_ID

        recurrence = RECURRENCE_ID
        fingerprint_payload["training_view_mode"] = view_mode
        fingerprint_payload["recurrence"] = recurrence
    if cleaned_batch_ids:
        # How round documents' references are placed in training windows: every
        # located mention since 6 Oct 2026 (the best span alone before). Absent
        # for round 0, which has no round documents, so M000's run id stands.
        fingerprint_payload["round_reference_placement"] = "located_mentions_v1"
    if labels != "clean":
        # A raw-label run is never the clean run (absent for "clean", so the
        # loop's run ids stand), and a different placement rule is another run.
        fingerprint_payload["label_source"] = labels
        fingerprint_payload["raw_placement_rule"] = RAW_PLACEMENT_RULE
        # Which documents and which raw lists: a run built before the release and
        # one built after it are one run exactly when these agree.
        fingerprint_payload["raw_training_sha256"] = fingerprint_json(
            {
                "documents_by_batch": raw_label_audit["documents_by_batch"],
                "references": {doc_id: raw_documents[doc_id]["references"] for doc_id in raw_documents},
            }
        )
    run_fingerprint = fingerprint_json(fingerprint_payload)
    run_id = f"dqcheck_loop_{run_fingerprint[:12]}"
    data_dir = config.sensitive_root / "loop" / run_id / "data"

    data_manifest = write_training_data(
        output_dir=data_dir,
        documents=documents,
        split=composed,
        system_prompt=system_prompt,
    )
    if raw_label_audit is not None:
        write_json_atomic(data_dir / "raw_label_audit.json", raw_label_audit)

    run_dir, preflight_path = _write_round_scaffolding(
        config=config,
        run_id=run_id,
        generation=generation,
        run_fingerprint=run_fingerprint,
        data_dir=data_dir,
        data_manifest=data_manifest,
        composed=composed,
        prompt=prompt,
        source_summary=source_summary,
        execute=execute,
        split_policy=split_policy,
        view_mode=view_mode,
        recurrence=recurrence,
        label_source=labels,
    )

    result: dict[str, Any] = {
        "run_id": run_id,
        "training_documents": len(composed["train"]),
        "validation_documents": len(composed["valid"]),
        "test_documents": len(composed["test"]),
        "cleaned_batch_ids": list(cleaned_batch_ids),
        "label_source": labels,
        **({"raw_label_audit": raw_label_audit["batches"]} if raw_label_audit is not None else {}),
        "data_dir": str(data_dir),
        "data_manifest": data_manifest,
        "run_config_path": str(run_dir / "run_config.json"),
        "preflight_path": str(preflight_path),
    }
    if not execute:
        return result

    assert candidate_id is not None  # guarded above
    # Build the windowed training view before the trainer looks for it. This is
    # the same order train_bootstrap follows for G0: scaffolding, then compute
    # acceptance preflight, then development.
    if compute_preflight_runner is None:
        from .mlx_compute import run_compute_acceptance_preflight as compute_preflight_runner_

        compute_preflight_runner = compute_preflight_runner_

    compute_preflight_runner(
        config=config,
        run_dir=run_dir,
        data_dir=data_dir,
        preflight_path=preflight_path,
    )
    if target_updates is None:
        run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        target_updates = full_epoch_updates(
            Path(run_config["training_view"]["jsonl_path"]),
            gradient_accumulation=TrainingContract().gradient_accumulation,
        )
        result["target_updates"] = target_updates
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
