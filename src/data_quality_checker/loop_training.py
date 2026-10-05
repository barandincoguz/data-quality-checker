"""Compose round `k`'s training set: canonical train plus cleaned rounds 1..k.

Validation and test are returned unchanged on every round. They are frozen for
the life of the experiment: the loop re-runs checkpoint selection each round,
so validation can never be folded into training, and the test split may never
influence any decision at all.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .errors import ContractError
from .fingerprints import sha256_file
from .reference_locator import attach_missing_spans
from .reference_policy import DEFAULT_REFERENCE_POLICY_ID, apply_reference_policy


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


def _released_rows(config: Any, batch_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Every training document of these batches' sealed releases, checked.

    Only the two trustworthy tiers are read: `expert_adjudicated` and
    `consensus_clean`; quarantined documents never are. Both arms of the
    noisy-vs-clean ablation read their documents here, so they always train on
    the same documents.
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
                documents[doc_id] = {"batch_id": batch_id, "text": text, "references": references}
                batch_document_count += 1
        if not found:
            raise ContractError(f"no release found for batch {batch_id!r} under {directory}")
        if batch_document_count == 0:
            raise ContractError(
                f"batch {batch_id!r} release contains zero training documents "
                "-- every document was quarantined or excluded"
            )
    return documents


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
    # The annotator's quote decides nothing, not even placement (user rule,
    # 27 Sep 2026): every released reference is placed by the locator below.
    # The release on disk keeps what was written.
    documents: dict[str, dict[str, Any]] = {
        doc_id: {
            "text": row["text"],
            "references": [
                {**reference, "source_text": ""} if isinstance(reference, dict) else reference
                for reference in row["references"]
            ],
        }
        for doc_id, row in _released_rows(config, batch_ids).items()
    }
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


# The raw arm's placement rule (user decision, 5 Oct 2026). Part of the raw
# run's fingerprint: a different rule is a different run.
RAW_PLACEMENT_RULE = "locator_then_annotator_verbatim_quote_v1"


def _annotator_lists(database_path: Path, batch_id: str) -> dict[str, dict[str, Any]]:
    """The annotator's list of every document in a batch, read-only.

    `documents.human_references_json` is written once, at batch preparation,
    as `contracts.normalize_references` of the immutable annotation export
    (verified for R01 250/250, 4 Oct 2026); no code path writes cleaned labels
    into it. The connection is opened read-only: training never writes the
    loop's database.
    """
    if not database_path.is_file():
        raise ContractError(f"loop database not found at {database_path}")
    connection = sqlite3.connect(f"{database_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT internal_doc_id, text, human_references_json FROM documents WHERE batch_id = ?",
            (batch_id,),
        ).fetchall()
    finally:
        connection.close()
    return {
        doc_id: {"text": text, "references": json.loads(references_json)}
        for doc_id, text, references_json in rows
    }


def _has_sealed_release(config: Any, batch_id: str) -> bool:
    parent = Path(config.sensitive_root) / "releases" / batch_id
    return parent.is_dir() and any((path / "manifest.json").is_file() for path in parent.glob("release_*"))


def round_raw_training_documents(
    config: Any, *, batch_ids: Sequence[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """The raw-label arm (`M_k_raw`): the release's documents, the annotator's lists.

    Pre-registered 4 Oct 2026 (ner `Journal/experiments/dq_loop_gt750/
    ablation_noisy_vs_clean/preregistration.json`): the same documents as the
    clean arm -- read by the same `_released_rows` -- with the annotator's raw
    list in place of the released one, under the release's reference policy
    (the policy `release_batch` applies to the annotator's list).

    Before a batch is released (5 Oct 2026) its documents are every document of
    the batch: the loop's release trains on all of them -- RED and QUARANTINE
    documents are required reviews, a deferred review blocks the release, GREEN
    ones are reviewed or released as consensus -- so the raw arm needs no expert
    decision. `check_raw_candidate_against_release` compares the two sets once
    the release exists. Either way documents are in id order, so a run built
    before the release and one built after it on the same documents are the
    same run.

    Placement (user decision, 5 Oct 2026): the locator first, exactly as in
    the clean arm. In the clean arm a reference the locator cannot place goes
    to the expert as a worklist; here no one corrects it, so it is placed by
    the annotator's own quote when that quote occurs verbatim in the text --
    the one place the 27 Sep 2026 rule ("the quote decides nothing") is set
    aside, and only in this arm -- and otherwise dropped, because a window can
    only teach a reference it can place. Every count, and every dropped
    reference, is returned in the audit.
    """
    batch_of: dict[str, str] = {}
    documents_from: dict[str, str] = {}
    prepared_texts: dict[str, dict[str, Any]] = {}
    for batch_id in batch_ids:
        annotator = _annotator_lists(Path(config.database_path), batch_id)
        if _has_sealed_release(config, batch_id):
            for doc_id, row in _released_rows(config, [batch_id]).items():
                prepared = annotator.get(doc_id)
                if prepared is None:
                    raise ContractError(
                        f"released document {doc_id!r} is not in batch {batch_id!r} of the loop database"
                    )
                if prepared["text"] != row["text"]:
                    raise ContractError(
                        f"released document {doc_id!r} differs from the text prepared in batch {batch_id!r}"
                    )
            ids = sorted(_released_rows(config, [batch_id]))
            documents_from[batch_id] = "release"
        else:
            ids = sorted(annotator)
            if not ids:
                raise ContractError(
                    f"batch {batch_id!r} has no release and no documents in the loop database"
                )
            documents_from[batch_id] = "batch"
        for doc_id in ids:
            if doc_id in batch_of:
                raise ContractError(
                    f"document {doc_id!r} is in batch {batch_of[doc_id]!r} and batch {batch_id!r}"
                )
            batch_of[doc_id] = batch_id
            prepared_texts[doc_id] = annotator[doc_id]
    quotes: dict[str, list[str]] = {}
    unplaced: dict[str, dict[str, Any]] = {}
    for doc_id, prepared in prepared_texts.items():
        references, _ = apply_reference_policy(prepared["references"])
        quotes[doc_id] = [str(reference.get("source_text") or "").strip() for reference in references]
        unplaced[doc_id] = {
            "text": prepared["text"],
            "references": [{**reference, "source_text": ""} for reference in references],
        }
    located, _ = attach_missing_spans(unplaced)

    documents: dict[str, dict[str, Any]] = {}
    batches = {
        batch_id: {
            "documents_from": documents_from[batch_id],
            "documents": 0,
            "references": 0,
            "placed_by_locator": 0,
            "placed_by_annotator_quote": 0,
            "dropped_unplaceable": 0,
        }
        for batch_id in batch_ids
    }
    dropped: list[dict[str, str]] = []
    for doc_id, document in located.items():
        batch_id = batch_of[doc_id]
        counts = batches[batch_id]
        counts["documents"] += 1
        kept: list[dict[str, Any]] = []
        for reference, quote in zip(document["references"], quotes[doc_id], strict=True):
            counts["references"] += 1
            if reference["source_text"]:
                counts["placed_by_locator"] += 1
                kept.append(reference)
            elif quote and quote in document["text"]:
                counts["placed_by_annotator_quote"] += 1
                kept.append({**reference, "source_text": quote})
            else:
                counts["dropped_unplaceable"] += 1
                dropped.append(
                    {
                        "batch_id": batch_id,
                        "doc_id": doc_id,
                        "kanun_no": str(reference.get("kanun_no", "")),
                        "kanun_ad": str(reference.get("kanun_ad", "")),
                        "madde": str(reference.get("madde", "")),
                    }
                )
        documents[doc_id] = {"text": document["text"], "references": kept}
    audit = {
        "label_source": "raw",
        "reference_policy_id": DEFAULT_REFERENCE_POLICY_ID,
        "placement_rule": RAW_PLACEMENT_RULE,
        "batches": batches,
        "documents_by_batch": {
            batch_id: [doc_id for doc_id in documents if batch_of[doc_id] == batch_id] for batch_id in batch_ids
        },
        "dropped": dropped,
    }
    return documents, audit
