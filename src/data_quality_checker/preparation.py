"""Stage 0: secure, resumable annotation/document-pool preparation."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic, write_jsonl_atomic, write_text_atomic
from .config import AppConfig
from .contracts import normalize_references
from .errors import ConfigurationError, ContractError, GateBlocked, IntegrityError
from .fingerprints import fingerprint_json, sha256_file, sha256_text
from .heartbeat import RunLease
from .safezip import ZipRecord, read_json_records
from .storage import Store
from .text import evidence_coverage, evidence_match_mode, html_to_text, normalize_text


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _read_hmac_key(path: Path | None) -> bytes:
    if path is not None:
        try:
            key = path.read_bytes().strip()
        except OSError as exc:
            raise ConfigurationError(f"cannot read HMAC key file {path}: {exc}") from exc
    else:
        value = os.environ.get("DQCHECK_HMAC_KEY")
        if value is None:
            raise ConfigurationError(
                "provide --hmac-key-file or DQCHECK_HMAC_KEY; the key is never persisted"
            )
        key = value.encode("utf-8")
    if len(key) < 32:
        raise ConfigurationError("HMAC key must contain at least 32 bytes")
    return key


def _public_doc_id(key: bytes, raw_id: str) -> str:
    digest = hmac.new(key, raw_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"dq_{digest}"


def _record_document_id(record: dict[str, Any]) -> str:
    evrak_id = normalize_text(record.get("evrakId"))
    document_id = normalize_text(record.get("document_id"))
    if evrak_id and document_id and evrak_id != document_id:
        raise ContractError("evrakId and document_id disagree")
    return evrak_id or document_id


def _pool_id(record: dict[str, Any]) -> str:
    return normalize_text(record.get("evrakOid"))


def _annotation_completed(record: dict[str, Any]) -> bool | None:
    nested = record.get("annotation")
    if isinstance(nested, dict) and isinstance(nested.get("is_completed"), bool):
        return bool(nested["is_completed"])
    if isinstance(record.get("is_completed"), bool):
        return bool(record["is_completed"])
    return None


def annotation_attribution(record: dict[str, Any]) -> dict[str, Any]:
    """Return the non-secret human attribution fields carried by an export row."""

    nested = record.get("annotation")
    if not isinstance(nested, dict):
        return {}

    def user(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        username = normalize_text(value.get("username"))
        user_id = value.get("id")
        if not username and user_id is None:
            return None
        result: dict[str, Any] = {"username": username}
        if isinstance(user_id, (int, str)) and not isinstance(user_id, bool):
            result["id"] = user_id
        return result

    result: dict[str, Any] = {}
    for key in ("completed_by", "last_editor"):
        normalized_user = user(nested.get(key))
        if normalized_user is not None:
            result[key] = normalized_user
    for key in ("edit_count", "unique_users_count"):
        value = nested.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            result[key] = value
    return result


def annotation_attribution_path(config: AppConfig, batch_id: str) -> Path:
    return config.sensitive_root / "batches" / batch_id / "annotation_attribution.json"


def import_annotation_attribution(
    *, config: AppConfig, batch_id: str, annotation_zip: Path
) -> dict[str, Any]:
    """Import per-document annotator identity into a private, atomic sidecar."""

    source_sha256 = sha256_file(annotation_zip)
    target = annotation_attribution_path(config, batch_id)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("batch_id") == batch_id and existing.get("source_sha256") == source_sha256:
            return existing

    _, records = read_json_records(annotation_zip, config.security)
    by_raw_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        raw_id = _record_document_id(record.payload)
        if raw_id:
            by_raw_id[raw_id].append(record.payload)

    with Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store:
        documents = store.list_documents(batch_id)
        if not documents:
            raise GateBlocked(f"batch has no documents: {batch_id}")
        attributions: dict[str, dict[str, Any]] = {}
        failures: list[str] = []
        for document in documents:
            matches = by_raw_id.get(str(document["raw_document_id"]), [])
            if len(matches) != 1:
                failures.append(str(document["internal_doc_id"]))
                continue
            attribution = annotation_attribution(matches[0])
            if not attribution:
                failures.append(str(document["internal_doc_id"]))
                continue
            attributions[str(document["internal_doc_id"])] = attribution
        if failures:
            raise GateBlocked(
                "annotation attribution coverage failed for "
                f"{len(failures)} documents; no sidecar was written"
            )
        payload = {
            "schema_version": 1,
            "batch_id": batch_id,
            "source_sha256": source_sha256,
            "created_at": _utc_now(),
            "document_count": len(documents),
            "attributed_document_count": len(attributions),
            "attributions": attributions,
        }
        write_json_atomic(target, payload, mode=0o600)
        store.add_event(
            batch_id,
            "annotation_attribution_imported",
            {
                "source_sha256": source_sha256,
                "document_count": len(documents),
                "attributed_document_count": len(attributions),
            },
        )
    return payload


def _annotation_text_digest(record: dict[str, Any]) -> str | None:
    candidates: list[Any] = [
        record.get("text"),
        record.get("document_text"),
        record.get("annotation_text"),
    ]
    nested = record.get("document")
    if isinstance(nested, dict):
        candidates.extend([nested.get("text"), nested.get("document_text")])
    for candidate in candidates:
        normalized = normalize_text(candidate)
        if normalized:
            return sha256_text(normalized)
    return None


def _select_channel(
    pool_record: dict[str, Any], references: list[dict[str, str]]
) -> tuple[str, str, float, float]:
    pdf_text = normalize_text(pool_record.get("pdfText"))
    html_text = html_to_text(pool_record.get("htmlText"))
    pdf_coverage = evidence_coverage(references, pdf_text) if pdf_text else 0.0
    html_coverage = evidence_coverage(references, html_text) if html_text else 0.0
    if html_text and html_coverage > pdf_coverage:
        return html_text, "htmlText", pdf_coverage, html_coverage
    if pdf_text:
        return pdf_text, "pdfText", pdf_coverage, html_coverage
    if html_text:
        return html_text, "htmlText", pdf_coverage, html_coverage
    return "", "missing", pdf_coverage, html_coverage


def _warning_codes(warnings: list[dict[str, Any]]) -> list[str]:
    return sorted({str(warning.get("code", "unknown")) for warning in warnings})


def _quarantine_document(
    *,
    internal_doc_id: str,
    public_doc_id: str,
    raw_document_id: str,
    reason: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "internal_doc_id": internal_doc_id,
        "public_doc_id": public_doc_id,
        "raw_document_id": raw_document_id,
        "selected_channel": "missing",
        "pdf_coverage": 0.0,
        "html_coverage": 0.0,
        "text": "",
        "text_sha256": sha256_text(""),
        "human_references": [],
        "metadata": {},
        "warnings": [{"code": reason, **(details or {})}],
        "preparation_status": "quarantine",
        "router_bucket": "QUARANTINE",
    }


def _validate_prepared_file(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "internal_doc_id",
        "public_doc_id",
        "raw_document_id",
        "selected_channel",
        "pdf_coverage",
        "html_coverage",
        "text",
        "text_sha256",
        "human_references",
        "warnings",
        "preparation_status",
    }
    if not isinstance(payload, dict) or not required <= payload.keys():
        raise IntegrityError(f"prepared document schema failure: {path}")
    if sha256_text(str(payload["text"])) != payload["text_sha256"]:
        raise IntegrityError(f"prepared document text checksum mismatch: {path}")


def _input_contract(
    annotation_zip: Path,
    document_pool_zip: Path,
    *,
    key: bytes,
    doc_ids: set[str] | None = None,
) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "annotation_zip": {
            "size": annotation_zip.stat().st_size,
            "sha256": sha256_file(annotation_zip),
        },
        "document_pool_zip": {
            "size": document_pool_zip.stat().st_size,
            "sha256": sha256_file(document_pool_zip),
        },
        "hmac_key_fingerprint": sha256_text(key.hex()),
    }
    # A subset must fingerprint differently from the whole archive and from
    # every other subset, or two rounds of a training loop drawing different
    # hundred-document windows from the same archives collide on one batch
    # id (`create_batch` sees a matching fingerprint and hands back whichever
    # round prepared first). Omit the key entirely when no subset was given
    # so today's no-subset fingerprint - and every batch already prepared
    # under it - is untouched by this change.
    if doc_ids is not None:
        contract["doc_ids"] = sorted(doc_ids)
    return contract


def ready_path(config: AppConfig, batch_id: str) -> Path:
    return config.public_root / "batches" / batch_id / "READY.json"


def validate_ready(config: AppConfig, batch_id: str) -> dict[str, Any]:
    path = ready_path(config, batch_id)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateBlocked(f"batch {batch_id} has no valid READY.json: {exc}") from exc
    if (
        payload.get("batch_id") != batch_id
        or payload.get("config_fingerprint") != config.fingerprint
    ):
        raise GateBlocked(f"batch {batch_id} READY contract does not match this config")
    public_dir = path.parent
    manifest = public_dir / "manifest.json"
    checksums = public_dir / "SHA256SUMS.txt"
    if sha256_file(manifest) != payload.get("manifest_sha256"):
        raise GateBlocked(f"batch {batch_id} public manifest checksum mismatch")
    if sha256_file(checksums) != payload.get("checksums_sha256"):
        raise GateBlocked(f"batch {batch_id} checksum-list mismatch")
    return payload


def prepare_batch(
    *,
    config: AppConfig,
    annotation_zip: Path,
    document_pool_zip: Path,
    batch_id: str | None,
    hmac_key_file: Path | None,
    doc_ids: set[str] | None = None,
) -> dict[str, Any]:
    annotation_zip = annotation_zip.resolve()
    document_pool_zip = document_pool_zip.resolve()
    if not annotation_zip.is_file() or not document_pool_zip.is_file():
        raise FileNotFoundError("both annotation and document-pool ZIP files must exist")
    key = _read_hmac_key(hmac_key_file)

    annotation_audit, raw_annotation_records = read_json_records(annotation_zip, config.security)
    pool_audit, raw_pool_records = read_json_records(document_pool_zip, config.security)
    annotations = [
        record
        for record in raw_annotation_records
        if any(
            key_name in record.payload
            for key_name in (
                "evrakId",
                "document_id",
                "current_references",
                "annotation",
            )
        )
    ]
    if doc_ids is not None:
        # A round is a hundred named documents, not a whole archive: restrict
        # which annotation records become the batch. An empty subset or one
        # naming an absent document is rejected rather than silently
        # preparing nothing or fewer documents than the round expects.
        if not doc_ids:
            raise ContractError(
                "doc_ids was supplied but empty; omit it to prepare the whole archive"
            )
        wanted = {str(value) for value in doc_ids}
        annotations = [
            record for record in annotations if _record_document_id(record.payload) in wanted
        ]
        found = {_record_document_id(record.payload) for record in annotations}
        missing = sorted(wanted - found)
        if missing:
            raise ContractError(
                f"doc_ids names {len(missing)} document(s) absent from the archive: {missing[:5]}"
            )
    pool_records = [record for record in raw_pool_records if "evrakOid" in record.payload]
    if not annotations:
        raise ContractError("annotation ZIP contains no recognizable annotation records")
    if not pool_records:
        raise ContractError("document-pool ZIP contains no evrakOid records")

    input_contract = _input_contract(annotation_zip, document_pool_zip, key=key, doc_ids=doc_ids)
    input_fingerprint = fingerprint_json(input_contract)
    effective_batch_id = batch_id or f"dq_{input_fingerprint[:16]}"
    if not effective_batch_id.replace("-", "").replace("_", "").isalnum():
        raise ContractError("batch-id may contain only letters, numbers, '-' and '_'")

    sensitive_dir = config.sensitive_root / "batches" / effective_batch_id
    public_dir = config.public_root / "batches" / effective_batch_id
    document_dir = sensitive_dir / "documents"
    lease = RunLease(
        lock_path=config.sensitive_root / "locks" / f"{effective_batch_id}.lock",
        heartbeat_path=sensitive_dir / "heartbeat.json",
        purpose="prepare",
        run_id=f"prepare:{effective_batch_id}",
        input_fingerprint=input_fingerprint,
        config_fingerprint=config.fingerprint,
    ).start(stage="archive-audit")

    try:
        with Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store:
            batch = store.create_batch(
                batch_id=effective_batch_id,
                input_fingerprint=input_fingerprint,
                config_fingerprint=config.fingerprint,
                metadata={
                    "input_contract": input_contract,
                    "annotation_zip_audit": annotation_audit.__dict__
                    | {"path": str(annotation_audit.path)},
                    "pool_zip_audit": pool_audit.__dict__ | {"path": str(pool_audit.path)},
                    "started_at": _utc_now(),
                },
            )
            if batch["ready"]:
                ready = validate_ready(config, effective_batch_id)
                lease.finish(status="completed")
                return ready

            annotation_groups: defaultdict[str, list[ZipRecord]] = defaultdict(list)
            missing_annotation_records: list[ZipRecord] = []
            annotation_id_errors: dict[str, str] = {}
            for record in annotations:
                try:
                    raw_id = _record_document_id(record.payload)
                except ContractError as exc:
                    origin = f"{record.entry_name}:{record.record_index}"
                    annotation_id_errors[origin] = str(exc)
                    missing_annotation_records.append(record)
                    continue
                if raw_id:
                    annotation_groups[raw_id].append(record)
                else:
                    missing_annotation_records.append(record)

            pool_groups: defaultdict[str, list[ZipRecord]] = defaultdict(list)
            for record in pool_records:
                raw_id = _pool_id(record.payload)
                if raw_id:
                    pool_groups[raw_id].append(record)

            work_items: list[tuple[str, list[ZipRecord] | None]] = [
                (raw_id, records) for raw_id, records in annotation_groups.items()
            ]
            for record in missing_annotation_records:
                pseudo = f"__missing__:{record.entry_name}:{record.record_index}"
                work_items.append((pseudo, [record]))
            work_items.sort(key=lambda item: item[0])

            lease.beat(stage="prepare-documents", expected_units=len(work_items))
            private_mapping: list[dict[str, str]] = []
            quarantine_rows: list[dict[str, Any]] = []
            public_rows: list[dict[str, Any]] = []
            warning_counts: Counter[str] = Counter()
            prepared_paths: list[Path] = []

            for ordinal, (group_key, records) in enumerate(work_items, 1):
                assert records is not None
                internal_id = f"d{ordinal:06d}"
                is_missing = group_key.startswith("__missing__:")
                raw_id = "" if is_missing else group_key
                hmac_source = group_key
                public_id = _public_doc_id(key, hmac_source)

                if is_missing:
                    origin = f"{records[0].entry_name}:{records[0].record_index}"
                    reason = (
                        "annotation_identifier_conflict"
                        if origin in annotation_id_errors
                        else "annotation_identifier_missing"
                    )
                    document = _quarantine_document(
                        internal_doc_id=internal_id,
                        public_doc_id=public_id,
                        raw_document_id="",
                        reason=reason,
                        details={"origin": origin},
                    )
                elif len(records) != 1:
                    document = _quarantine_document(
                        internal_doc_id=internal_id,
                        public_doc_id=public_id,
                        raw_document_id=raw_id,
                        reason="duplicate_annotation_identifier",
                        details={"duplicate_count": len(records)},
                    )
                elif not pool_groups.get(raw_id):
                    document = _quarantine_document(
                        internal_doc_id=internal_id,
                        public_doc_id=public_id,
                        raw_document_id=raw_id,
                        reason="missing_document_pool_join",
                    )
                elif len(pool_groups[raw_id]) != 1:
                    document = _quarantine_document(
                        internal_doc_id=internal_id,
                        public_doc_id=public_id,
                        raw_document_id=raw_id,
                        reason="ambiguous_document_pool_identifier",
                        details={"duplicate_count": len(pool_groups[raw_id])},
                    )
                else:
                    annotation = records[0].payload
                    warnings: list[dict[str, Any]] = []
                    try:
                        references, ref_warnings = normalize_references(
                            annotation.get("current_references")
                            if "current_references" in annotation
                            else None
                        )
                        warnings.extend(ref_warnings)
                    except ContractError as exc:
                        document = _quarantine_document(
                            internal_doc_id=internal_id,
                            public_doc_id=public_id,
                            raw_document_id=raw_id,
                            reason="invalid_current_references",
                            details={"error": str(exc)},
                        )
                    else:
                        text, channel, pdf_coverage, html_coverage = _select_channel(
                            pool_groups[raw_id][0].payload, references
                        )
                        if not text:
                            document = _quarantine_document(
                                internal_doc_id=internal_id,
                                public_doc_id=public_id,
                                raw_document_id=raw_id,
                                reason="safe_document_text_missing",
                            )
                        else:
                            for index, reference in enumerate(references):
                                source = reference["source_text"]
                                if source and evidence_match_mode(source, text) is None:
                                    warnings.append(
                                        {
                                            "code": "source_text_not_in_selected_channel",
                                            "reference_index": index,
                                            "source_text_sha256": sha256_text(
                                                normalize_text(source)
                                            ),
                                        }
                                    )
                            completed = _annotation_completed(annotation)
                            if completed is False:
                                warnings.append({"code": "annotation_incomplete"})
                            document = {
                                "schema_version": 1,
                                "internal_doc_id": internal_id,
                                "public_doc_id": public_id,
                                "raw_document_id": raw_id,
                                "selected_channel": channel,
                                "pdf_coverage": pdf_coverage,
                                "html_coverage": html_coverage,
                                "text": text,
                                "text_sha256": sha256_text(text),
                                "human_references": references,
                                "annotation_text_sha256": _annotation_text_digest(annotation),
                                "metadata": {
                                    "annotation_completed": completed,
                                    "annotation_attribution": annotation_attribution(annotation),
                                    "reference_count": len(references),
                                    "source_entry": records[0].entry_name,
                                    "pool_entry": pool_groups[raw_id][0].entry_name,
                                },
                                "warnings": warnings,
                                "preparation_status": "ready",
                                "router_bucket": None,
                            }

                for code in _warning_codes(document["warnings"]):
                    warning_counts[code] += 1
                if document["preparation_status"] == "quarantine":
                    quarantine_rows.append(
                        {
                            "internal_doc_id": internal_id,
                            "public_doc_id": public_id,
                            "raw_document_id": raw_id,
                            "reasons": document["warnings"],
                        }
                    )
                target = document_dir / f"{internal_id}.json"
                write_json_atomic(target, document, validator=_validate_prepared_file)
                prepared_paths.append(target)
                store.add_document(effective_batch_id, document)
                private_mapping.append(
                    {
                        "internal_doc_id": internal_id,
                        "public_doc_id": public_id,
                        "raw_document_id": raw_id,
                    }
                )
                public_rows.append(
                    {
                        "internal_doc_id": internal_id,
                        "public_doc_id": public_id,
                        "preparation_status": document["preparation_status"],
                        "router_bucket": document.get("router_bucket"),
                        "selected_channel": document["selected_channel"],
                        "pdf_coverage": document["pdf_coverage"],
                        "html_coverage": document["html_coverage"],
                        "text_sha256": document["text_sha256"],
                        "warning_codes": _warning_codes(document["warnings"]),
                    }
                )
                lease.beat(
                    completed_units=ordinal,
                    last_successful_unit=internal_id,
                    stage="prepare-documents",
                )

            if len({row["internal_doc_id"] for row in public_rows}) != len(public_rows):
                raise IntegrityError("duplicate internal_doc_id after preparation")
            if len({row["public_doc_id"] for row in public_rows}) != len(public_rows):
                raise IntegrityError("duplicate HMAC public_doc_id after preparation")
            stored_count = len(store.list_documents(effective_batch_id))
            if stored_count != len(public_rows):
                raise IntegrityError(
                    f"SQLite document coverage {stored_count}/{len(public_rows)} is incomplete"
                )
            for path in prepared_paths:
                _validate_prepared_file(path)

            write_json_atomic(sensitive_dir / "private_mapping.json", private_mapping)
            write_jsonl_atomic(sensitive_dir / "quarantine.jsonl", quarantine_rows)
            prepared_checksums = [
                {
                    "path": path.relative_to(sensitive_dir).as_posix(),
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                }
                for path in sorted(prepared_paths)
            ]
            private_manifest = {
                "schema_version": 1,
                "batch_id": effective_batch_id,
                "input_fingerprint": input_fingerprint,
                "config_fingerprint": config.fingerprint,
                "documents": prepared_checksums,
                "private_mapping_sha256": sha256_file(sensitive_dir / "private_mapping.json"),
                "quarantine_sha256": sha256_file(sensitive_dir / "quarantine.jsonl"),
            }
            write_json_atomic(sensitive_dir / "manifest.json", private_manifest)

            public_manifest = {
                "schema_version": 1,
                "batch_id": effective_batch_id,
                "created_at": _utc_now(),
                "input_fingerprint": input_fingerprint,
                "config_fingerprint": config.fingerprint,
                "counts": {
                    "annotations": len(annotations),
                    "unique_work_items": len(work_items),
                    "ready": sum(row["preparation_status"] == "ready" for row in public_rows),
                    "quarantine": len(quarantine_rows),
                },
                "warning_counts": dict(sorted(warning_counts.items())),
                "archive_audits": {
                    "annotation": {
                        "entry_count": annotation_audit.entry_count,
                        "file_count": annotation_audit.file_count,
                        "total_uncompressed_bytes": annotation_audit.total_uncompressed_bytes,
                        "json_entry_count": annotation_audit.json_entry_count,
                    },
                    "document_pool": {
                        "entry_count": pool_audit.entry_count,
                        "file_count": pool_audit.file_count,
                        "total_uncompressed_bytes": pool_audit.total_uncompressed_bytes,
                        "json_entry_count": pool_audit.json_entry_count,
                    },
                },
                "documents": public_rows,
            }
            manifest_path = public_dir / "manifest.json"
            write_json_atomic(manifest_path, public_manifest, mode=0o644)
            checksum_lines = [
                f"{sha256_file(manifest_path)}  manifest.json",
                f"{sha256_file(sensitive_dir / 'manifest.json')}  sensitive-manifest.json",
            ]
            checksums_path = public_dir / "SHA256SUMS.txt"
            write_text_atomic(checksums_path, "\n".join(checksum_lines) + "\n", mode=0o644)
            ready = {
                "schema_version": 1,
                "batch_id": effective_batch_id,
                "status": "ready",
                "created_at": _utc_now(),
                "input_fingerprint": input_fingerprint,
                "config_fingerprint": config.fingerprint,
                "document_count": len(public_rows),
                "ready_count": public_manifest["counts"]["ready"],
                "quarantine_count": public_manifest["counts"]["quarantine"],
                "manifest_sha256": sha256_file(manifest_path),
                "checksums_sha256": sha256_file(checksums_path),
                "sensitive_manifest_sha256": sha256_file(sensitive_dir / "manifest.json"),
            }
            write_json_atomic(public_dir / "READY.json", ready, mode=0o644)
            validate_ready(config, effective_batch_id)
            current = store.get_batch(effective_batch_id)
            assert current is not None
            store.update_batch(
                effective_batch_id,
                expected_version=int(current["row_version"]),
                status="ready",
                ready=True,
                metadata={
                    "input_contract": input_contract,
                    "ready_path": str(public_dir / "READY.json"),
                    "completed_at": _utc_now(),
                },
            )
            store.add_event(effective_batch_id, "preparation_completed", ready)
        lease.finish(status="completed")
        return ready
    except BaseException as exc:
        lease.finish(status="failed", error=str(exc))
        raise
