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

from collections import Counter
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .config import AppConfig
from .errors import ContractError
from .fingerprints import sha256_file
from .loop_rounds import RoundState
from .loop_runner import RoundStep
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
