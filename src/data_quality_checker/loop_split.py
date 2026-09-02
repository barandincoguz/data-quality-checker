"""Canonical train/validation/test split for the DQ-Loop experiment.

Deliberately separate from `g0.build_split`. That function regenerates the
deployed G0's historical 394/50/50 split and is checked against a frozen
reference manifest, so it is provenance and must not move. The loop needs a
different shape and gets its own function, seed and manifest.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .constants import EXEMPLAR_DOC_IDS
from .errors import ContractError
from .fingerprints import sha256_file

LOOP_SPLIT_SEED = 20260902
ELIGIBLE_CANONICAL_DOCUMENTS = 494


@dataclass(frozen=True)
class LoopSplitSizes:
    """Requested split sizes over the canonical 500.

    Validation and test are drawn from the 494 eligible documents only. The six
    few-shot exemplars go to train, so `train_documents` is the eligible
    remainder plus six.
    """

    train_documents: int = 300
    validation_documents: int = 50
    test_documents: int = 150


def build_loop_split(
    doc_ids: list[int], *, sizes: LoopSplitSizes, seed: int
) -> dict[str, list[int]]:
    """Split the canonical corpus, keeping every exemplar in train.

    Test is drawn first and validation second, so both keep exactly their
    requested sizes; train takes the eligible remainder and then the exemplars
    are appended to it.
    """
    everything = set(doc_ids)
    eligible = sorted(everything - EXEMPLAR_DOC_IDS)
    if len(eligible) != ELIGIBLE_CANONICAL_DOCUMENTS:
        raise ContractError(
            f"expected {ELIGIBLE_CANONICAL_DOCUMENTS} eligible canonical docs, found {len(eligible)}"
        )
    missing_exemplars = EXEMPLAR_DOC_IDS - everything
    if missing_exemplars:
        raise ContractError(f"exemplars absent from the corpus: {sorted(missing_exemplars)}")
    held_out = sizes.validation_documents + sizes.test_documents
    if held_out >= len(eligible):
        raise ContractError(
            f"validation+test ({held_out}) leaves no training documents in a pool of {len(eligible)}"
        )
    shuffled = eligible[:]
    random.Random(seed).shuffle(shuffled)
    test = shuffled[: sizes.test_documents]
    valid = shuffled[sizes.test_documents : held_out]
    train = shuffled[held_out:] + sorted(EXEMPLAR_DOC_IDS)
    if len(train) != sizes.train_documents:
        raise ContractError(
            f"train came out at {len(train)}, expected {sizes.train_documents}; "
            "sizes do not fit the corpus"
        )
    return {"train": sorted(train), "valid": sorted(valid), "test": sorted(test)}


def write_loop_split_manifest(
    output_dir: Path,
    split: dict[str, list[int]],
    *,
    sizes: LoopSplitSizes,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {name: len(split[name]) for name in ("train", "valid", "test")}
    payload = {
        "schema_version": 1,
        "seed": seed,
        "eligible_documents": ELIGIBLE_CANONICAL_DOCUMENTS,
        "counts": counts,
        "exemplars_in_train": sorted(EXEMPLAR_DOC_IDS),
        "held_out_are_exemplar_free": True,
        "splits": split,
    }
    path = output_dir / "loop_split_manifest.json"
    write_json_atomic(path, payload)
    return {"path": str(path), "manifest_sha256": sha256_file(path), "counts": counts}
