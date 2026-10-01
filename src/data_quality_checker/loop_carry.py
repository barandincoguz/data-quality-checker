"""Carry the expert's decisions on a batch from the archived loop into this one.

User decision, 1 Oct 2026: a reference is a law and an article of it; the
GT750 loop that ran with paragraph and sub-item fields is archived untouched
(`configs/presets/gt750_loop.json`, its own roots) and the law-and-article loop
starts over on the same split, pool, batches and document key. The expert had
already decided 23 R01 documents there. Those decisions are the expert's truth
about which laws and articles each ruling cites, and are kept: each is read
from the archived database (opened read-only), projected to the law and the
article -- the paragraph fields are dropped, nothing else is changed -- and
written here with its whole history.

Every carried event is an ordinary `expert_review_updated` event, so every
consumer (decision history, time on page, the journal) counts it, and carries
`carried_from` naming the archived database, its event id and its original
time; `created_at_epoch` is the original time. The carried action keeps its
archived meaning: "accept_model" chose the archived M000's list, which is
stored here as the final list itself.

Nothing is carried unless the two batches are provably the same documents:
the same internal ids, public ids (the HMAC of the source id), text hashes and
annotator lists (projected). A target review must still be untouched. The
database is backed up first. A GREEN document of the audit sample whose carried
list disagrees with the agreed list escalates exactly as a screen decision does.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .config import AppConfig
from .contracts import validate_reference_list
from .errors import ContractError, GateBlocked, IntegrityError
from .fingerprints import fingerprint_json, sha256_file
from .loop_sources import has_loop_settings
from .sqlite_backup import backup_database
from .storage import Store

CARRIED_STATUSES = ("finalized", "deferred")
PROJECTION = "law_article_v1"


def project(references: list[dict[str, Any]] | None) -> list[dict[str, str]] | None:
    """A stored list read as laws and articles: paragraph fields dropped, exact
    repeats that leaves behind dropped, order kept."""
    if references is None:
        return None
    out: list[dict[str, str]] = []
    for reference in validate_reference_list(references):
        if reference not in out:
            out.append(reference)
    return out


def _project_event(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    if "final_references" in out:
        out["final_references"] = project(out["final_references"])
    previous = out.get("previous")
    if isinstance(previous, dict) and "final_references" in previous:
        out["previous"] = {**previous, "final_references": project(previous["final_references"])}
    return out


def _source_rows(path: Path, batch_id: str) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        documents = {
            row["internal_doc_id"]: dict(row)
            for row in connection.execute("SELECT * FROM documents WHERE batch_id=?", (batch_id,))
        }
        reviews = {
            row["internal_doc_id"]: dict(row)
            for row in connection.execute("SELECT * FROM reviews WHERE batch_id=?", (batch_id,))
        }
        events = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM batch_events WHERE batch_id=? AND event_type='expert_review_updated' "
                "ORDER BY event_id",
                (batch_id,),
            )
        ]
    finally:
        connection.close()
    return {"documents": documents, "reviews": reviews, "events": events}


def _carried_events(store: Store, batch_id: str) -> list[dict[str, Any]]:
    rows = store.connection.execute(
        "SELECT event_id, payload_json FROM batch_events WHERE batch_id=? AND event_type='expert_review_updated'",
        (batch_id,),
    ).fetchall()
    return [json.loads(row["payload_json"]) for row in rows]


def carry_decisions(
    config: AppConfig, *, batch_id: str, source_database: Path, source_label: str
) -> dict[str, Any]:
    if not has_loop_settings(config):
        raise GateBlocked("decisions are carried into a GT750 loop only")
    source_database = Path(source_database).resolve()
    if source_database == Path(config.database_path).resolve():
        raise ContractError("the source database is this loop's own")
    if not source_database.is_file():
        raise ContractError(f"source database not found: {source_database}")
    source_sha = sha256_file(source_database)
    source = _source_rows(source_database, batch_id)
    to_carry = sorted(d for d, row in source["reviews"].items() if row["status"] in CARRIED_STATUSES)
    manifest_path = config.public_root / "batches" / batch_id / "carried_decisions.json"

    with Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store:
        batch = store.get_batch(batch_id)
        if batch is None or batch["status"] != "processed":
            raise GateBlocked(f"batch {batch_id} must be prepared and processed here first")
        documents = {row["internal_doc_id"]: row for row in store.list_documents(batch_id)}
        if set(documents) != set(source["documents"]):
            raise IntegrityError("the two batches do not hold the same internal document ids")
        for doc_id, mine in documents.items():
            theirs = source["documents"][doc_id]
            for field in ("public_doc_id", "text_sha256"):
                if mine[field] != theirs[field]:
                    raise IntegrityError(f"{doc_id}: {field} differs between the two batches")
            if project(json.loads(mine["human_references_json"])) != project(
                json.loads(theirs["human_references_json"])
            ):
                raise IntegrityError(f"{doc_id}: the annotator's list differs between the two batches")
            if mine["router_bucket"] is None:
                raise GateBlocked(f"{doc_id} is not routed yet")

        already = [p for p in _carried_events(store, batch_id) if "carried_from" in p]
        if already:
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("source_database_sha256") == source_sha:
                    return manifest
            raise IntegrityError("decisions were already carried into this batch from another source")
        reviews = {row["internal_doc_id"]: row for row in store.list_reviews(batch_id)}
        for doc_id in to_carry:
            review = reviews[doc_id]
            if review["status"] != "pending" or review["row_version"] != 0:
                raise GateBlocked(f"{doc_id} was already reviewed here; nothing is carried over a decision")

        from .judges import ensure_green_audit_plan

        audit = ensure_green_audit_plan(config=config, store=store, batch_id=batch_id)
        backup = backup_database(
            store.connection,
            config.sensitive_root / "backups" / f"pre_carry_{batch_id}_{source_sha[:12]}.sqlite3",
        )
        carried: list[dict[str, Any]] = []
        with store.transaction():
            for event in source["events"]:
                payload = json.loads(event["payload_json"])
                doc_id = payload.get("internal_doc_id")
                if doc_id not in to_carry:
                    continue
                payload = {
                    **_project_event(payload),
                    "carried_from": {
                        "loop": source_label,
                        "database_sha256": source_sha,
                        "event_id": event["event_id"],
                        "created_at_epoch": event["created_at_epoch"],
                        "projection": PROJECTION,
                    },
                }
                store.connection.execute(
                    "INSERT INTO batch_events(batch_id, event_type, payload_json, created_at_epoch) "
                    "VALUES (?, 'expert_review_updated', ?, ?)",
                    (batch_id, json.dumps(payload, ensure_ascii=False, sort_keys=True), event["created_at_epoch"]),
                )
            for doc_id in to_carry:
                row = source["reviews"][doc_id]
                final = project(None if row["final_references_json"] is None else json.loads(row["final_references_json"]))
                cursor = store.connection.execute(
                    """UPDATE reviews
                          SET status=?, action=?, final_references_json=?, reason=?, reviewer=?,
                              row_version=?, updated_at_epoch=?
                        WHERE batch_id=? AND internal_doc_id=? AND row_version=0 AND status='pending'""",
                    (
                        row["status"],
                        row["action"],
                        None if final is None else json.dumps(final, ensure_ascii=False, separators=(",", ":")),
                        row["reason"],
                        row["reviewer"],
                        row["row_version"],
                        row["updated_at_epoch"],
                        batch_id,
                        doc_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise IntegrityError(f"{doc_id}: review changed while carrying")
                carried.append(
                    {
                        "internal_doc_id": doc_id,
                        "status": row["status"],
                        "action": row["action"],
                        "archived_bucket": source["documents"][doc_id]["router_bucket"],
                        "bucket": documents[doc_id]["router_bucket"],
                        "in_green_audit_sample": doc_id in set(audit["sample_internal_doc_ids"]),
                        "final_references_fingerprint": fingerprint_json(final),
                        "events": sum(
                            1 for e in source["events"] if json.loads(e["payload_json"]).get("internal_doc_id") == doc_id
                        ),
                    }
                )
            store.connection.execute(
                "INSERT INTO batch_events(batch_id, event_type, payload_json, created_at_epoch) "
                "VALUES (?, 'expert_decisions_carried', ?, strftime('%s','now'))",
                (batch_id, json.dumps({"source": source_label, "source_database_sha256": source_sha,
                                       "documents": len(carried), "projection": PROJECTION}, sort_keys=True)),
            )

        from .hitl import _trigger_green_escalation

        escalated = []
        for row in carried:
            if row["status"] != "finalized":
                continue
            review = store.get_review(batch_id, row["internal_doc_id"])
            final = json.loads(review["final_references_json"])
            if _trigger_green_escalation(
                config=config, store=store, batch_id=batch_id, document=documents[row["internal_doc_id"]], final=final
            ):
                escalated.append(row["internal_doc_id"])

    manifest = {
        "schema_version": 1,
        "batch_id": batch_id,
        "source_label": source_label,
        "source_database": str(source_database),
        "source_database_sha256": source_sha,
        "projection": PROJECTION,
        "database_backup": {"path": backup["path"], "sha256": backup["sha256"]},
        "carried_document_count": len(carried),
        "carried_event_count": sum(row["events"] for row in carried),
        "by_bucket": dict(sorted(Counter(row["bucket"] for row in carried).items())),
        "by_action": dict(sorted(Counter(row["action"] for row in carried).items())),
        "green_escalated": escalated,
        "documents": carried,
    }
    write_json_atomic(manifest_path, manifest, mode=0o644)
    return manifest
