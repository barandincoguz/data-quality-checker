"""Compose round `k`'s training set: canonical train plus cleaned rounds 1..k.

Validation and test are returned unchanged on every round. They are frozen for
the life of the experiment: the loop re-runs checkpoint selection each round,
so validation can never be folded into training, and the test split may never
influence any decision at all.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .errors import ContractError
from .fingerprints import sha256_file
from .reference_locator import attach_missing_spans


def compose_round_training_ids(
    split: dict[str, list[int]], cleaned_rounds: Sequence[Sequence[str]]
) -> dict[str, list[int | str]]:
    """Compose the round's three splits, each document keeping its own identity.

    A canonical document is an integer here, everywhere downstream, and in G0's
    own training data, because that is the identity the ground-truth directory
    resolves (`doc_<id>.json`) and the identity the evaluator matches a
    prediction against. A document cleaned by a DQ-Loop round is a string like
    `d000001`. The two can never be confused, so the type itself records where
    a training document came from -- no decorated key is needed, and none is
    used, because a decorated key would stop the evaluator finding the
    document's ground truth.

    A cleaned document whose id reads as a bare integer is refused for exactly
    that reason: it would be indistinguishable from canonical document `id`.
    """
    train: list[int | str] = list(split["train"])
    seen: set[int | str] = set(train)
    for round_index, batch in enumerate(cleaned_rounds, start=1):
        for doc_id in batch:
            if isinstance(doc_id, int) or (
                isinstance(doc_id, str) and doc_id.strip().lstrip("+-").isdigit()
            ):
                raise ContractError(
                    f"round {round_index} document {doc_id!r} reads as a canonical "
                    "document id; a cleaned document must carry a non-numeric id"
                )
            if not isinstance(doc_id, str) or not doc_id:
                raise ContractError(f"round {round_index} has an unusable document id {doc_id!r}")
            if doc_id in seen:
                raise ContractError(
                    f"round {round_index} document {doc_id!r} is already in the training set"
                )
            seen.add(doc_id)
            train.append(doc_id)
    return {
        "train": train,
        "valid": list(split["valid"]),
        "test": list(split["test"]),
    }


def write_round_training_manifest(
    output_dir: Path,
    round_index: int,
    composition: dict[str, list[str]],
    *,
    split_manifest_sha256: str,
    batch_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    if round_index < 0:
        raise ContractError("round_index must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "round": round_index,
        "split_manifest_sha256": split_manifest_sha256,
        "batch_manifest_sha256": batch_manifest_sha256,
        "counts": {name: len(ids) for name, ids in composition.items()},
        "ids": composition,
    }
    path = output_dir / f"round_{round_index:03d}_training.json"
    write_json_atomic(path, payload)
    return {"path": str(path), "manifest_sha256": sha256_file(path)}


def sealed_release_dir(config: Any, batch_id: str) -> Path:
    """The one sealed release `release_batch` wrote for this batch, verified.

    `release_batch` promotes each release into `releases/<batch>/<release_id>/`
    and seals it with a manifest and `SHA256SUMS.txt`. This used to read
    `releases/<batch>/*.jsonl` directly -- a layout only hand-written test
    fixtures ever had -- so the first real round would have found no release.
    Every file is checked against the seal before a row is trusted.
    """
    from .release import _verify_release

    parent = Path(config.sensitive_root) / "releases" / batch_id
    sealed = (
        sorted(path for path in parent.glob("release_*") if (path / "manifest.json").is_file())
        if parent.is_dir()
        else []
    )
    if not sealed:
        raise ContractError(f"no release found for batch {batch_id!r} under {parent}")
    if len(sealed) > 1:
        raise ContractError(
            f"batch {batch_id!r} has {len(sealed)} sealed releases under {parent}; expected one"
        )
    manifest = json.loads((sealed[0] / "manifest.json").read_text(encoding="utf-8"))
    _verify_release(sealed[0], expected_document_count=int(manifest["counts"]["documents"]))
    return sealed[0]


def round_training_documents(config: Any, *, batch_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Read finished rounds' releases into training rows.

    A release already carries each document's text and the references the
    expert settled on, which is exactly what a training row needs. Only the
    two trustworthy tiers are read: `expert_adjudicated`, where a human decided,
    and `consensus_clean`, where annotator and model agreed and the round's
    GREEN sample raised nothing. Quarantined documents are excluded -- training
    on documents the pipeline could not process would poison the very data this
    loop exists to clean.

    `context_windows.py` needs a verbatim substring to place each reference in
    a training window. The annotator's `source_text` is not used for that (it
    decides nothing, user rule 27 Sep 2026) and a reference taken from the model
    or the judge has none, so every released reference is placed by the
    build-time locator, `attach_missing_spans` -- without
    ever writing the located span back into the release or the database, which
    stay the faithful record of what the expert decided. Every reference the
    locator could not place is collected into one `ContractError` naming all
    of them at once, so the expert gets one worklist instead of one crash per
    run.
    """
    documents: dict[str, dict[str, Any]] = {}
    for batch_id in batch_ids:
        directory = sealed_release_dir(config, batch_id)
        found = False
        batch_document_count = 0
        for name in ("expert_adjudicated", "consensus_clean"):
            path = directory / f"{name}.jsonl"
            if not path.is_file():
                continue
            found = True
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ContractError(f"{path}:{line_number} is not JSON: {exc}") from exc
                doc_id = row.get("internal_doc_id")
                text = row.get("text")
                references = row.get("references")
                if not isinstance(doc_id, str) or not doc_id:
                    raise ContractError(f"{path}:{line_number} has no internal_doc_id")
                if not isinstance(text, str) or not text:
                    raise ContractError(f"{path}:{line_number} ({doc_id}) has no text")
                if not isinstance(references, list):
                    raise ContractError(f"{path}:{line_number} ({doc_id}) has no references list")
                if doc_id in documents:
                    raise ContractError(
                        f"document {doc_id!r} is duplicated while assembling "
                        f"batch {batch_id!r}'s release"
                    )
                # The annotator's quote decides nothing, not even placement (user
                # rule, 27 Sep 2026): every released reference is placed by the
                # locator below. The release on disk keeps what was written.
                documents[doc_id] = {
                    "text": text,
                    "references": [
                        {**reference, "source_text": ""}
                        if isinstance(reference, dict)
                        else reference
                        for reference in references
                    ],
                }
                batch_document_count += 1
        if not found:
            raise ContractError(f"no release found for batch {batch_id!r} under {directory}")
        if batch_document_count == 0:
            raise ContractError(
                f"batch {batch_id!r} release contains zero training documents "
                "-- every document was quarantined or excluded"
            )
    documents, refusals = attach_missing_spans(documents)
    if refusals:
        worklist = "\n".join(
            f"  {refusal['doc_id']}: kanun_no={refusal['kanun_no']!r} "
            f"kanun_ad={refusal['kanun_ad']!r} madde={refusal['madde']!r}"
            for refusal in refusals
        )
        raise ContractError(
            f"{len(refusals)} reference(s) have no source_text and the locator "
            "could not place them in their document's text -- an expert must "
            f"review them:\n{worklist}"
        )
    return documents
