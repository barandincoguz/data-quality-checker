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
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

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
from .loop_training import CANONICAL_PREFIX, compose_round_training_ids, round_training_documents


def train_round(
    config: AppConfig,
    *,
    generation: str,
    split: dict[str, list[int]],
    cleaned_batch_ids: Sequence[str],
    execute: bool = False,
) -> dict[str, Any]:
    if execute:
        raise ContractError(
            "train_round does not launch compute yet; it assembles and fingerprints "
            "a round's training run. Launch it through the existing compute path."
        )
    if generation == "G0":
        raise ContractError("train_round does not train G0 -- use train_bootstrap for G0")
    if not isinstance(generation, str) or not ROUND_LABEL_PATTERN.match(generation):
        raise ContractError(f"generation must be a round label like 'M003', got {generation!r}")

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

    return {
        "run_id": run_id,
        "training_documents": len(composed["train"]),
        "validation_documents": len(composed["valid"]),
        "test_documents": len(composed["test"]),
        "cleaned_batch_ids": list(cleaned_batch_ids),
        "data_dir": str(data_dir),
        "data_manifest": data_manifest,
    }
