"""Canonical train/validation/test split for the DQ-Loop experiment.

Deliberately separate from `g0.build_split`. That function regenerates the
deployed G0's historical 394/50/50 split and is checked against a frozen
reference manifest, so it is provenance and must not move. The loop needs a
different shape and gets its own function, seed and manifest.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .constants import EXEMPLAR_DOC_IDS
from .errors import ContractError, FingerprintMismatch, IntegrityError
from .fingerprints import fingerprint_json, sha256_file

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


def _manifest_path(output_dir: Path) -> Path:
    return output_dir / "loop_split_manifest.json"


def _split_fingerprint(*, sizes: dict[str, int], seed: int, split: dict[str, list[int]]) -> str:
    """Fingerprint exactly the parts that define the split.

    Deliberately excludes decorative/derived fields (``counts``,
    ``exemplars_in_train``) so the fingerprint only ever moves when the seed,
    the requested sizes, or one of the three id lists actually changes.
    """
    return fingerprint_json({"seed": seed, "sizes": sizes, "splits": split})


def write_loop_split_manifest(
    output_dir: Path,
    split: dict[str, list[int]],
    *,
    sizes: LoopSplitSizes,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {name: len(split[name]) for name in ("train", "valid", "test")}
    sizes_payload = asdict(sizes)
    fingerprint = _split_fingerprint(sizes=sizes_payload, seed=seed, split=split)
    payload = {
        "schema_version": 1,
        "seed": seed,
        "sizes": sizes_payload,
        "eligible_documents": ELIGIBLE_CANONICAL_DOCUMENTS,
        "counts": counts,
        "exemplars_in_train": sorted(EXEMPLAR_DOC_IDS),
        "held_out_are_exemplar_free": True,
        "splits": split,
        "fingerprint": fingerprint,
    }
    path = _manifest_path(output_dir)
    write_json_atomic(path, payload)
    return {
        "path": str(path),
        "manifest_sha256": sha256_file(path),
        "counts": counts,
        "fingerprint": fingerprint,
    }


def read_loop_split_manifest(output_dir: Path) -> dict[str, list[int]]:
    """Read back a sealed split manifest, refusing one that has been tampered with.

    Recomputes the same fingerprint `write_loop_split_manifest` sealed in and
    compares it against the stored value -- if a single id in any of the three
    lists (or the seed, or the requested sizes) was edited after the fact, the
    recomputed fingerprint no longer matches and this raises rather than
    silently handing back the tampered split.
    """
    path = _manifest_path(output_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"loop split manifest cannot be read at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise IntegrityError(f"loop split manifest is not a JSON object at {path}")
    seed = payload.get("seed")
    sizes_payload = payload.get("sizes")
    split_payload = payload.get("splits")
    stored_fingerprint = payload.get("fingerprint")
    if (
        not isinstance(seed, int)
        or not isinstance(sizes_payload, dict)
        or not isinstance(split_payload, dict)
        or not isinstance(stored_fingerprint, str)
        or {"train", "valid", "test"} - set(split_payload)
    ):
        raise IntegrityError(f"loop split manifest is missing required fields at {path}")
    recomputed_fingerprint = _split_fingerprint(sizes=sizes_payload, seed=seed, split=split_payload)
    if recomputed_fingerprint != stored_fingerprint:
        raise FingerprintMismatch(f"loop split manifest fingerprint mismatch at {path}")
    return {name: list(split_payload[name]) for name in ("train", "valid", "test")}
