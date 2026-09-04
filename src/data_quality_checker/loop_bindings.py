"""Bind the pipeline's calls to the stage callables `run_round` drives.

`loop_runner` deliberately knows nothing about prediction, routing, judging or
scoring; every stage reaches it as a callable. This module is the one place
where the runner and the pipeline meet, so that knowledge lives once rather
than being spread through the orchestrator.

Each factory returns a `RoundStep`: it runs its stage and returns the sha256 of
the artifact that stage produced. `advance_round` refuses an empty artifact, so
a stage that produced nothing readable cannot advance a round.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .config import AppConfig
from .constants import JUDGE_MODEL_KEY
from .errors import ContractError
from .fingerprints import sha256_file
from .g0 import CheckpointCandidate, select_checkpoint
from .g0_training import canonical_evaluate
from .judges import ensure_green_audit_plan, run_judge_pilot
from .loop_metrics import (
    documents_with_finalized_truth,
    expert_workload,
    model_round_metrics,
    paired_delta,
    rater_agreement,
)
from .loop_rounds import RoundState
from .loop_runner import RoundStep
from .loop_selection import write_selection_record
from .loop_train import train_round
from .loop_training import compose_round_training_ids, write_round_training_manifest
from .processing import process_batch
from .storage import Store

ROUTER_BUCKETS = ("GREEN", "YELLOW", "RED", "QUARANTINE")


def predict_step(
    config: AppConfig,
    *,
    batch_id: str,
    generation: str,
    registry_path: Path | None = None,
    fake_backend: bool = False,
) -> RoundStep:
    """Annotate the round's documents with the round's own model.

    `process_batch` resumes by design, so re-running after a crash re-reads the
    documents already done rather than regenerating them -- which is what makes
    this step safe for `run_round` to retry.

    `process_batch` already writes its own completion summary at
    ``{public_root}/batches/{batch_id}/process_{generation}_summary.json`` --
    unconditionally, with deterministic (sorted-key) JSON serialisation, so a
    repeat run with the same inputs reproduces it byte-for-byte. That file is
    fingerprinted directly rather than writing a second summary alongside the
    per-document predictions under ``sensitive_root``.
    """

    def step(state: RoundState) -> str:
        process_batch(
            config=config,
            batch_id=batch_id,
            generation=generation,
            resume=True,
            fake_backend=fake_backend,
            registry_path=registry_path,
        )
        summary = config.public_root / "batches" / batch_id / f"process_{generation}_summary.json"
        return sha256_file(summary)

    return step


def route_step(config: AppConfig, *, batch_id: str, output_dir: Path) -> RoundStep:
    """Seal the router's bucket distribution for this round.

    Prediction and routing happen in one pass, but the distribution is its own
    result: it is the workload axis the experiment measures, and it has to be a
    citable file rather than a number recomputed later from a mutable store.
    """

    def step(state: RoundState) -> str:
        with Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store:
            documents = store.list_documents(batch_id)
        if not documents:
            raise ContractError(f"batch {batch_id!r} holds no documents to route")
        buckets = Counter(str(row.get("router_bucket") or "") for row in documents)
        if "" in buckets:
            raise ContractError(
                f"{buckets['']} document(s) in batch {batch_id!r} have no router bucket; "
                "run the prediction stage first"
            )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "round": state.round_index,
            "batch_id": batch_id,
            "document_count": len(documents),
            "buckets": {name: buckets.get(name, 0) for name in ROUTER_BUCKETS},
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"round_{state.round_index:03d}_routing.json"
        write_json_atomic(path, payload)
        return sha256_file(path)

    return step


def judge_step(
    config: AppConfig,
    *,
    batch_id: str,
    generation: str,
    judge_models: tuple[str, ...] | None = None,
    allow_external_judge: bool = False,
    fake_backend: bool = False,
) -> RoundStep:
    """Get the judge's third opinion on this round's disagreements.

    `allow_external_judge` is passed straight through rather than defaulted to
    True: the pipeline refuses an external call without it, and a binding that
    quietly supplied consent would move that decision out of the operator's
    hands.

    `generation` names the round whose predictions the judge should look at --
    it takes no default because a round's generation is a fact the caller
    knows and must state; without it the judge would silently re-judge G0 no
    matter which round is actually running.
    """

    def step(state: RoundState) -> str:
        result = run_judge_pilot(
            config=config,
            batch_id=batch_id,
            allow_external_judge=allow_external_judge,
            generation=generation,
            fake_backend=fake_backend,
            judge_models=judge_models,
        )
        # `run_judge_pilot` takes one of two branches depending on whether a
        # judge is already locked for this batch: the locked path writes
        # `judge_production_summary.json` and never touches the pilot
        # summary, so the file to fingerprint has to be chosen from what the
        # call actually did, not assumed to always be the pilot summary.
        filename = (
            "judge_production_summary.json"
            if result.get("stage") == "production"
            else "judge_pilot_summary.json"
        )
        summary = config.public_root / "batches" / batch_id / filename
        if not summary.is_file():
            raise ContractError(f"judge pilot wrote no summary at {summary}")
        return sha256_file(summary)

    return step


def train_step(
    config: AppConfig,
    *,
    generation: str,
    split: dict[str, list[int]],
    cleaned_batch_ids: list[str],
    execute: bool = False,
) -> RoundStep:
    """Assemble and fingerprint this round's training run.

    `generation`, `split` and `cleaned_batch_ids` take no default: each is a
    fact only the caller can state, and every default guessed on this line of
    work produced a defect elsewhere in the pipeline -- a round predicting
    with the wrong model, a judge reading the wrong generation, an evaluator
    gating the wrong split size. `train_round` itself already rejects
    `execute=True` loudly, so it is passed straight through rather than
    re-validated here.
    """

    def step(state: RoundState) -> str:
        result = train_round(
            config,
            generation=generation,
            split=split,
            cleaned_batch_ids=cleaned_batch_ids,
            execute=execute,
        )
        manifest = Path(result["data_dir"]) / "split_manifest.json"
        return sha256_file(manifest)

    return step


def compose_step(
    config: AppConfig,
    *,
    split: dict[str, list[int]],
    cleaned_rounds: list[list[str]],
    output_dir: Path,
    split_manifest_sha256: str,
) -> RoundStep:
    """Compose this round's training set: canonical train plus cleaned rounds.

    Validation and test come back unchanged on every round; the loop re-runs
    checkpoint selection each round, so validation can never be folded in, and
    the test split may never influence any decision.
    """

    def step(state: RoundState) -> str:
        composition = compose_round_training_ids(split, cleaned_rounds)
        result = write_round_training_manifest(
            output_dir,
            state.round_index,
            composition,
            split_manifest_sha256=split_manifest_sha256,
        )
        return str(result["manifest_sha256"])

    return step


def select_step(
    config: AppConfig,
    *,
    candidates: list[CheckpointCandidate],
    output_dir: Path,
    validation_documents: int,
    minimum_parse_count: int | None = None,
) -> RoundStep:
    """Choose this round's checkpoint, on validation alone, and record why.

    `validation_documents` takes no default: a round's validation split size
    is a fact its caller must know and state, exactly as `measure_step`
    requires `expected_doc_count` -- a baked-in 50 would fail closed but
    misdirect the reader when a round's split is not 50.

    `minimum_parse_count` is derived, not defaulted: when left `None` it is
    computed as ``validation_documents - 1``, the same 50 -> 49 relationship
    this step used to hardcode. `select_checkpoint` gates on `parse_count` as
    an absolute count, not a ratio, so a literal 49 threshold silently fails
    open once `validation_documents` is anything other than 50 -- at 120
    documents, a checkpoint parsing barely a third of the validation set
    (49/120) would still pass, and below 49 documents no candidate could ever
    pass at all. Deriving the threshold from whatever `validation_documents`
    this call actually states keeps the two numbers coupled so they cannot
    drift apart; a caller that wants a different tolerance still states one
    explicitly and it is used as given.
    """

    resolved_minimum_parse_count = (
        validation_documents - 1 if minimum_parse_count is None else minimum_parse_count
    )

    def step(state: RoundState) -> str:
        selected = select_checkpoint(
            list(candidates),
            validation_documents=validation_documents,
            minimum_parse_count=resolved_minimum_parse_count,
        )
        result = write_selection_record(
            output_dir,
            state.round_index,
            selected,
            candidates,
            validation_documents=validation_documents,
            minimum_parse_count=resolved_minimum_parse_count,
        )
        return str(result["record_sha256"])

    return step


def measure_step(
    config: AppConfig,
    *,
    predictions_path: Path,
    test_doc_ids_path: Path,
    output_dir: Path,
    expected_doc_count: int,
) -> RoundStep:
    """Score this round's model on the frozen test split, and only record it.

    `run_round` reaches `measured` only from `checkpoint_selected`, so the
    checkpoint is already sealed on validation by the time this runs. This step
    writes the number down and returns its sha; it compares nothing and decides
    nothing, because the test split may never influence a decision.
    """

    def step(state: RoundState) -> str:
        canonical_evaluate(
            config=config,
            predictions_path=predictions_path,
            doc_ids_path=test_doc_ids_path,
            output_dir=output_dir,
            expected_doc_count=expected_doc_count,
        )
        report = output_dir / "evaluation.json"
        if not report.is_file():
            raise ContractError(f"evaluator wrote no report at {report}")
        return sha256_file(report)

    return step


def metrics_step(
    config: AppConfig,
    *,
    batch_id: str,
    generation: str,
    predictions_path: Path,
    test_doc_ids_path: Path,
    expected_doc_count: int,
    output_dir: Path,
    judge_model: str = JUDGE_MODEL_KEY,
) -> RoundStep:
    """Assemble this round's audited metrics and fingerprint the result.

    `loop_metrics.py` is deliberately pure and database-free (Tasks 2-4): every
    function there takes plain records and computes over them, none of them
    knows the store exists. This is the one place that reads the store (and
    the two filesystem facts the store cannot answer) and shapes what those
    functions expect -- so database access lives in exactly one place rather
    than being duplicated into a metrics module that is supposed to stay
    testable without one.

    Runs at the `measured` stage, reachable only from `checkpoint_selected` --
    the checkpoint is already sealed on validation alone by the time any test
    number exists, so nothing this step computes can feed back into
    selection. Writing `round_metrics.json` here, and only here, is also what
    lets `seal_step` cover it: the seal already serialises `state.artifacts`
    wholesale, so once this step's sha lands at `artifacts["measured"]`, any
    later change to this artifact necessarily changes the seal too.

    `expected_doc_count` takes no default, exactly as `measure_step` already
    requires for the same reason: `canonical_evaluate` defaults it to 50 (the
    G0 development-validation size), which is the wrong count for the loop's
    frozen test split (150 documents), and a caller-stated fact must not be
    silently guessed here either.

    `judge_model` defaults to the project's one pinned judge model constant
    (`constants.JUDGE_MODEL_KEY`) -- the same default `judge_step` already
    applies via `judges.JUDGE_MODELS` -- rather than a per-round fact a caller
    could get wrong; a round that pilots a different judge still states one
    explicitly and it is used as given.
    """

    def step(state: RoundState) -> str:
        with Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store:
            documents = store.list_documents(batch_id)
            if not documents:
                raise ContractError(f"batch {batch_id!r} holds no documents to measure")

            # Both booleans are batch-wide facts `loop_metrics.expert_workload`
            # cannot derive on its own -- resolved once per batch, here, and
            # stamped onto every one of the batch's records below, rather than
            # re-checked (or re-sampled) once per document.
            escalated = (
                config.public_root / "batches" / batch_id / "green_escalation.json"
            ).is_file()
            audit = ensure_green_audit_plan(config=config, store=store, batch_id=batch_id)
            audit_sample = {str(doc_id) for doc_id in audit["sample_internal_doc_ids"]}

            records: list[dict[str, Any]] = []
            for document in documents:
                internal_doc_id = document["internal_doc_id"]
                review = store.get_review(batch_id, internal_doc_id)
                if review is None:
                    raise ContractError(f"document {internal_doc_id!r} carries no review row")

                prediction = store.get_prediction(batch_id, internal_doc_id, generation)
                # A prediction whose status is not "success" produced no
                # usable output for this document -- that is a rater with
                # nothing to say, not a rater that reported an empty list.
                model = (
                    json.loads(prediction["references_json"])
                    if prediction is not None and prediction["status"] == "success"
                    else None
                )

                judge_row = store.get_judge_result(batch_id, internal_doc_id, judge_model)
                judge = (
                    json.loads(judge_row["result_json"]).get("final_references", [])
                    if judge_row is not None and judge_row["status"] == "valid"
                    else None
                )

                final = (
                    json.loads(review["final_references_json"])
                    if review["final_references_json"] is not None
                    else None
                )

                records.append(
                    {
                        "internal_doc_id": internal_doc_id,
                        "router_bucket": document["router_bucket"],
                        "escalated": escalated,
                        "in_green_audit_sample": internal_doc_id in audit_sample,
                        "status": review["status"],
                        "action": review["action"],
                        "created_at_epoch": review["created_at_epoch"],
                        "updated_at_epoch": review["updated_at_epoch"],
                        "annotator": json.loads(document["human_references_json"]),
                        "model": model,
                        "judge": judge,
                        "final": final,
                    }
                )

        workload = expert_workload(records)
        # A `defer` row (or a still-`pending` one) is real work but never
        # expert truth: `documents_with_finalized_truth` is the filter that
        # keeps such a row counted above, in workload, while excluding it here,
        # from every accuracy input.
        agreement = rater_agreement(
            [
                {
                    "expert": record["final"],
                    "annotator": record["annotator"],
                    "model": record["model"],
                    "judge": record["judge"],
                }
                for record in documents_with_finalized_truth(records)
            ]
        )

        # A round-specific subdirectory, not `output_dir` itself: canonical_evaluate
        # always names its report "evaluation.json", so reusing `output_dir`
        # across rounds would let round N+1 silently overwrite round N's report.
        # The report's content is fully captured inside `round_metrics.json`
        # below regardless, but the raw evaluator output is still worth
        # keeping unclobbered for audit.
        evaluation_dir = output_dir / f"round_{state.round_index:03d}_evaluation"
        model_metrics = model_round_metrics(
            config=config,
            predictions_path=predictions_path,
            doc_ids_path=test_doc_ids_path,
            output_dir=evaluation_dir,
            expected_doc_count=expected_doc_count,
        )

        paired: dict[str, Any] | None = None
        if state.round_index <= 1:
            paired_delta_reason = "round 1 has no predecessor round"
        else:
            previous_path = output_dir / f"round_{state.round_index - 1:03d}_metrics.json"
            if not previous_path.is_file():
                paired_delta_reason = (
                    f"no metrics artifact found for the previous round at {previous_path}"
                )
            else:
                previous_payload = json.loads(previous_path.read_text(encoding="utf-8"))
                # JSON object keys are always strings; `paired_delta` requires
                # the same document-id universe on both sides, and this
                # round's own `core_per_document` (produced above, never
                # round-tripped through JSON) still keys by int.
                previous_core = {
                    int(doc_id): counts
                    for doc_id, counts in previous_payload["model"]["core_per_document"].items()
                }
                paired = paired_delta(previous_core, model_metrics["core_per_document"])
                paired_delta_reason = None

        payload: dict[str, Any] = {
            "schema_version": 1,
            "round": state.round_index,
            "batch_id": batch_id,
            "generation": generation,
            "rater_agreement": agreement,
            "expert_workload": workload,
            "model": model_metrics,
            "paired_delta": paired,
            "paired_delta_reason": paired_delta_reason,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"round_{state.round_index:03d}_metrics.json"
        write_json_atomic(path, payload)
        return sha256_file(path)

    return step


def seal_step(config: AppConfig, *, output_dir: Path) -> RoundStep:
    """Freeze the round: write what it did and what each stage produced."""

    def step(state: RoundState) -> str:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "round": state.round_index,
            "stage": state.stage,
            "artifacts": dict(state.artifacts),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"round_{state.round_index:03d}_record.json"
        write_json_atomic(path, payload)
        return sha256_file(path)

    return step
