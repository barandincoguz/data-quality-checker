"""Immutable, checksummed, atomically promoted sensitive releases."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .atomic import fsync_directory, write_json_atomic, write_jsonl_atomic, write_text_atomic
from .config import AppConfig
from .errors import GateBlocked, IntegrityError
from .fingerprints import fingerprint_json, sha256_file
from .heartbeat import RunLease
from .hitl import _review_requirements, validate_final_references
from .judges import ensure_green_audit_plan
from .preparation import validate_ready
from .storage import Store

LAYER_FILES = (
    "expert_adjudicated.jsonl",
    "consensus_clean.jsonl",
    "quarantine.jsonl",
    "training_export.jsonl",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IntegrityError(f"invalid release JSONL {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise IntegrityError(f"release JSONL row must be an object: {path}:{line_number}")
            rows.append(payload)
    return rows


def _verify_release(path: Path, *, expected_document_count: int) -> dict[str, Any]:
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"release manifest cannot be parsed: {exc}") from exc
    files = manifest.get("files")
    if not isinstance(files, list):
        raise IntegrityError("release manifest files must be a list")
    for row in files:
        candidate = path / str(row.get("path", ""))
        if not candidate.is_file():
            raise IntegrityError(f"release file missing: {candidate}")
        if candidate.stat().st_size != row.get("size"):
            raise IntegrityError(f"release file size mismatch: {candidate}")
        if sha256_file(candidate) != row.get("sha256"):
            raise IntegrityError(f"release file checksum mismatch: {candidate}")
    checksum_path = path / "SHA256SUMS.txt"
    try:
        checksum_lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise IntegrityError(f"release checksum list is missing: {exc}") from exc
    expected_names = {*LAYER_FILES, "manifest.json"}
    observed_names: set[str] = set()
    for line in checksum_lines:
        digest, separator, name = line.partition("  ")
        if not separator or name not in expected_names:
            raise IntegrityError(f"invalid release checksum row: {line!r}")
        observed_names.add(name)
        if sha256_file(path / name) != digest:
            raise IntegrityError(f"release SHA256SUMS mismatch: {name}")
    if observed_names != expected_names:
        raise IntegrityError("release SHA256SUMS coverage is incomplete")
    layers = {name: _read_jsonl(path / name) for name in LAYER_FILES}
    release_rows = (
        layers["expert_adjudicated.jsonl"]
        + layers["consensus_clean.jsonl"]
        + layers["quarantine.jsonl"]
    )
    ids = [str(row.get("internal_doc_id")) for row in release_rows]
    if len(ids) != expected_document_count or len(set(ids)) != expected_document_count:
        raise IntegrityError(
            f"release coverage/duplicate failure: {len(set(ids))}/{expected_document_count}"
        )
    training_ids = {
        str(row.get("internal_doc_id")) for row in layers["training_export.jsonl"]
    }
    eligible_ids = {
        str(row.get("internal_doc_id"))
        for row in layers["expert_adjudicated.jsonl"] + layers["consensus_clean.jsonl"]
    }
    if training_ids != eligible_ids:
        raise IntegrityError("training export does not exactly equal eligible trust layers")
    return manifest


def _judge_required_ids(
    *, documents: list[dict[str, Any]], escalation: bool
) -> set[str]:
    required = {
        row["internal_doc_id"]
        for row in documents
        if row["router_bucket"] in {"RED", "YELLOW"}
    }
    if escalation:
        required.update(
            row["internal_doc_id"]
            for row in documents
            if row["router_bucket"] == "GREEN"
        )
    return required


def release_batch(*, config: AppConfig, batch_id: str) -> dict[str, Any]:
    ready = validate_ready(config, batch_id)
    sensitive_batch = config.sensitive_root / "batches" / batch_id
    public_batch = config.public_root / "batches" / batch_id
    lease = RunLease(
        lock_path=config.sensitive_root / "locks" / f"release_{batch_id}.lock",
        heartbeat_path=sensitive_batch / "release_heartbeat.json",
        purpose="release",
        run_id=f"release:{batch_id}",
        input_fingerprint=str(ready["input_fingerprint"]),
        config_fingerprint=config.fingerprint,
    ).start(stage="release-gates")
    temporary: Path | None = None
    try:
        with Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store:
            batch = store.get_batch(batch_id)
            if batch is None or batch["status"] not in {"processed", "released"}:
                raise GateBlocked("release requires a processed batch")
            documents = store.list_documents(batch_id)
            predictions = store.list_predictions(batch_id, "G0")
            if len(documents) != int(ready["document_count"]):
                raise GateBlocked("release document coverage differs from READY")
            if len(predictions) != len(documents):
                raise GateBlocked(
                    f"prediction coverage incomplete: {len(predictions)}/{len(documents)}"
                )
            prediction_by_id = {row["internal_doc_id"]: row for row in predictions}
            for prediction in predictions:
                path = Path(prediction["response_path"])
                if not path.is_file() or sha256_file(path) != prediction["response_sha256"]:
                    raise IntegrityError(f"prediction seal failure: {path}")

            audit = ensure_green_audit_plan(config=config, store=store, batch_id=batch_id)
            required_review_ids, _ = _review_requirements(
                config=config, store=store, batch_id=batch_id
            )
            review_by_id = {row["internal_doc_id"]: row for row in store.list_reviews(batch_id)}
            incomplete = [
                doc_id
                for doc_id in sorted(required_review_ids)
                if review_by_id[doc_id]["status"] != "finalized"
            ]
            deferred = [
                doc_id
                for doc_id in sorted(required_review_ids)
                if review_by_id[doc_id]["status"] == "deferred"
            ]
            if deferred:
                raise GateBlocked(f"deferred reviews block release: {deferred[:10]}")
            if incomplete:
                raise GateBlocked(f"required reviews are incomplete: {incomplete[:10]}")

            escalation = (public_batch / "green_escalation.json").exists()
            judge_required = _judge_required_ids(documents=documents, escalation=escalation)
            locked_model: str | None = None
            if judge_required:
                lock_path = public_batch / "judge_lock.json"
                if not lock_path.is_file():
                    raise GateBlocked("RED/YELLOW or escalated GREEN records require judge-lock")
                lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
                locked_model = str(lock_payload["model"])
                missing_judges = [
                    doc_id
                    for doc_id in sorted(judge_required)
                    if (
                        (result := store.get_judge_result(batch_id, doc_id, locked_model)) is None
                        or result["status"] != "valid"
                    )
                ]
                if missing_judges:
                    raise GateBlocked(
                        f"locked judge coverage incomplete: {missing_judges[:10]}"
                    )

            audit_complete = all(
                review_by_id[str(doc_id)]["status"] == "finalized"
                for doc_id in audit["sample_internal_doc_ids"]
            )
            if not audit_complete:
                raise GateBlocked("GREEN audit is incomplete")
            audit = {**audit, "status": "passed", "escalated": escalation}
            write_json_atomic(public_batch / "green_audit.json", audit, mode=0o644)

            expert_rows: list[dict[str, Any]] = []
            consensus_rows: list[dict[str, Any]] = []
            quarantine_rows: list[dict[str, Any]] = []
            training_rows: list[dict[str, Any]] = []
            decision_fingerprints: list[dict[str, Any]] = []
            for document in documents:
                doc_id = document["internal_doc_id"]
                human = json.loads(document["human_references_json"])
                review = review_by_id[doc_id]
                prediction = prediction_by_id[doc_id]
                common = {
                    "schema_version": 1,
                    "batch_id": batch_id,
                    "internal_doc_id": doc_id,
                    "public_doc_id": document["public_doc_id"],
                    "raw_document_id": document["raw_document_id"],
                    "text": document["text"],
                    "text_sha256": document["text_sha256"],
                    "router_bucket": document["router_bucket"],
                    "warnings": json.loads(document["warnings_json"]),
                    "prediction_sha256": prediction["response_sha256"],
                }
                if document["router_bucket"] == "QUARANTINE":
                    row = {
                        **common,
                        "release_trust": "quarantine",
                        "references": [],
                        "quarantine_reason": document["preparation_status"],
                    }
                    quarantine_rows.append(row)
                    decision_fingerprints.append(
                        {"id": doc_id, "trust": "quarantine", "references": []}
                    )
                    continue
                if review["status"] == "finalized":
                    references = validate_final_references(
                        json.loads(review["final_references_json"] or "[]"),
                        document_text=document["text"],
                    )
                    row = {
                        **common,
                        "release_trust": "expert_adjudicated",
                        "references": references,
                        "expert_action": review["action"],
                        "expert_reason": review["reason"],
                        "reviewer": review["reviewer"],
                    }
                    expert_rows.append(row)
                elif document["router_bucket"] == "GREEN" and not escalation:
                    references = human
                    row = {
                        **common,
                        "release_trust": "consensus_clean",
                        "references": references,
                    }
                    consensus_rows.append(row)
                else:
                    raise GateBlocked(f"unresolved record blocks release: {doc_id}")
                training_rows.append(
                    {
                        "schema_version": 1,
                        "batch_id": batch_id,
                        "internal_doc_id": doc_id,
                        "public_doc_id": document["public_doc_id"],
                        "text": document["text"],
                        "text_sha256": document["text_sha256"],
                        "references": references,
                        "release_trust": row["release_trust"],
                        "provenance": {
                            "prediction_sha256": prediction["response_sha256"],
                            "config_fingerprint": config.fingerprint,
                            "input_fingerprint": ready["input_fingerprint"],
                        },
                    }
                )
                decision_fingerprints.append(
                    {"id": doc_id, "trust": row["release_trust"], "references": references}
                )

            release_fingerprint = fingerprint_json(
                {
                    "batch_id": batch_id,
                    "input_fingerprint": ready["input_fingerprint"],
                    "config_fingerprint": config.fingerprint,
                    "model_fingerprints": sorted(
                        {row["model_fingerprint"] for row in predictions}
                    ),
                    "decisions": decision_fingerprints,
                    "green_escalated": escalation,
                    "locked_judge": locked_model,
                }
            )
            release_id = f"release_{release_fingerprint[:16]}"
            release_parent = config.sensitive_root / "releases" / batch_id
            target = release_parent / release_id
            release_parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                manifest = _verify_release(target, expected_document_count=len(documents))
                manifest_sha256 = sha256_file(target / "manifest.json")
                existing_release = store.get_release_for_batch(batch_id)
                if existing_release is None:
                    store.record_release(
                        release_id=release_id,
                        batch_id=batch_id,
                        release_path=target,
                        manifest_sha256=manifest_sha256,
                    )
                elif (
                    existing_release["release_id"] != release_id
                    or existing_release["manifest_sha256"] != manifest_sha256
                ):
                    raise IntegrityError("SQLite release registry differs from immutable release")
                current = store.get_batch(batch_id)
                assert current is not None
                if current["status"] != "released":
                    store.update_batch(
                        batch_id,
                        expected_version=int(current["row_version"]),
                        status="released",
                        ready=True,
                    )
                summary = {
                    "schema_version": 1,
                    "batch_id": batch_id,
                    "release_id": release_id,
                    "release_path": str(target),
                    "manifest_sha256": manifest_sha256,
                    "counts": manifest["counts"],
                    "status": "released",
                    "idempotent_existing_release": True,
                }
                write_json_atomic(public_batch / "release.json", summary, mode=0o644)
                lease.finish(status="completed")
                return summary

            temporary = Path(
                tempfile.mkdtemp(
                    dir=release_parent, prefix=f".{release_id}.", suffix=".tmp"
                )
            )
            write_jsonl_atomic(temporary / "expert_adjudicated.jsonl", expert_rows)
            write_jsonl_atomic(temporary / "consensus_clean.jsonl", consensus_rows)
            write_jsonl_atomic(temporary / "quarantine.jsonl", quarantine_rows)
            write_jsonl_atomic(temporary / "training_export.jsonl", training_rows)
            file_rows = [
                {
                    "path": name,
                    "size": (temporary / name).stat().st_size,
                    "sha256": sha256_file(temporary / name),
                }
                for name in LAYER_FILES
            ]
            manifest = {
                "schema_version": 1,
                "release_id": release_id,
                "batch_id": batch_id,
                "created_at": _utc_now(),
                "release_fingerprint": release_fingerprint,
                "input_fingerprint": ready["input_fingerprint"],
                "config_fingerprint": config.fingerprint,
                "green_audit": {
                    "sample_size": audit["sample_size"],
                    "status": "passed",
                    "escalated": escalation,
                },
                "locked_judge": locked_model,
                "counts": {
                    "documents": len(documents),
                    "expert_adjudicated": len(expert_rows),
                    "consensus_clean": len(consensus_rows),
                    "quarantine": len(quarantine_rows),
                    "training_export": len(training_rows),
                },
                "files": file_rows,
            }
            write_json_atomic(temporary / "manifest.json", manifest)
            checksum_paths = [*LAYER_FILES, "manifest.json"]
            write_text_atomic(
                temporary / "SHA256SUMS.txt",
                "".join(
                    f"{sha256_file(temporary / name)}  {name}\n" for name in checksum_paths
                ),
            )
            _verify_release(temporary, expected_document_count=len(documents))
            os.rename(temporary, target)
            temporary = None
            fsync_directory(release_parent)
            sealed_manifest = _verify_release(target, expected_document_count=len(documents))
            manifest_sha256 = sha256_file(target / "manifest.json")
            store.record_release(
                release_id=release_id,
                batch_id=batch_id,
                release_path=target,
                manifest_sha256=manifest_sha256,
            )
            current = store.get_batch(batch_id)
            assert current is not None
            store.update_batch(
                batch_id,
                expected_version=int(current["row_version"]),
                status="released",
                ready=True,
            )
            summary = {
                "schema_version": 1,
                "batch_id": batch_id,
                "release_id": release_id,
                "release_path": str(target),
                "manifest_sha256": manifest_sha256,
                "counts": sealed_manifest["counts"],
                "status": "released",
                "idempotent_existing_release": False,
            }
            write_json_atomic(public_batch / "release.json", summary, mode=0o644)
            store.add_event(batch_id, "release_completed", summary)
        lease.finish(status="completed")
        return summary
    except BaseException as exc:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        lease.finish(status="failed", error=str(exc))
        raise
