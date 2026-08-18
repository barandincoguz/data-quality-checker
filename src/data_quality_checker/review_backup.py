"""Verified SQLite snapshots for durable HITL review decisions."""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .atomic import fsync_directory, write_json_atomic
from .config import AppConfig
from .errors import IntegrityError
from .fingerprints import fingerprint_json, sha256_file
from .sqlite_backup import backup_database
from .storage import Store

_REVIEW_STATE_SQL = """
SELECT internal_doc_id, status, action, final_references_json, reason,
       reviewer, row_version, updated_at_epoch
  FROM reviews
 WHERE batch_id=? AND status!='pending'
 ORDER BY internal_doc_id
"""
_SNAPSHOT_RETENTION = 5


@contextmanager
def review_backup_lock(*, config: AppConfig, batch_id: str):
    """Serialize review commit+snapshot sections across threads and processes."""

    lock_root = config.sensitive_root / "locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_id = fingerprint_json({"batch_id": batch_id})[:16]
    lock_path = lock_root / f"review_backup_{lock_id}.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _review_state(connection: sqlite3.Connection, batch_id: str) -> list[dict[str, Any]]:
    connection.row_factory = sqlite3.Row
    return [dict(row) for row in connection.execute(_REVIEW_STATE_SQL, (batch_id,))]


def _latest_review_event_id(connection: sqlite3.Connection, batch_id: str) -> int:
    row = connection.execute(
        """
        SELECT COALESCE(MAX(event_id), 0)
          FROM batch_events
         WHERE batch_id=? AND event_type='expert_review_updated'
        """,
        (batch_id,),
    ).fetchone()
    return int(row[0])


def _prune_verified_snapshots(root: Path, *, keep: int) -> None:
    manifest_dir = root / "manifests"
    manifests: list[tuple[int, str, Path, Path]] = []
    for manifest_path in manifest_dir.glob("reviews_*.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            snapshot_path = Path(str(payload["snapshot_path"])).resolve()
            review_count = int(payload["review_count"])
            created_at = str(payload["created_at"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if snapshot_path.parent != (root / "snapshots").resolve():
            continue
        manifests.append((review_count, created_at, manifest_path, snapshot_path))
    manifests.sort()
    for _, _, manifest_path, snapshot_path in manifests[:-keep]:
        snapshot_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    if len(manifests) > keep:
        fsync_directory(root / "snapshots")
        fsync_directory(manifest_dir)


def create_review_backup(*, config: AppConfig, store: Store, batch_id: str) -> dict[str, Any]:
    """Create and verify a content-addressed snapshot of current review state."""

    source_state = _review_state(store.connection, batch_id)
    latest_review_event_id = _latest_review_event_id(store.connection, batch_id)
    state_fingerprint = fingerprint_json(source_state)
    root = config.sensitive_root / "review_backups" / batch_id
    snapshot_path = (
        root / "snapshots" / f"reviews_{len(source_state):06d}_{state_fingerprint[:16]}.sqlite3"
    )
    backup = backup_database(store.connection, snapshot_path)

    with sqlite3.connect(f"file:{snapshot_path}?mode=ro", uri=True) as snapshot:
        snapshot.row_factory = sqlite3.Row
        integrity = str(snapshot.execute("PRAGMA integrity_check").fetchone()[0])
        foreign_key_violations = snapshot.execute("PRAGMA foreign_key_check").fetchall()
        copied_state = _review_state(snapshot, batch_id)
        copied_event_id = _latest_review_event_id(snapshot, batch_id)
    if integrity != "ok":
        raise IntegrityError(f"review backup integrity failure: {snapshot_path}")
    if foreign_key_violations:
        raise IntegrityError(f"review backup foreign-key failure: {snapshot_path}")
    if copied_state != source_state:
        raise IntegrityError(f"review backup state mismatch: {snapshot_path}")
    if copied_event_id != latest_review_event_id:
        raise IntegrityError(f"review backup event mismatch: {snapshot_path}")

    payload = {
        "schema_version": 1,
        "status": "verified",
        "batch_id": batch_id,
        "created_at": datetime.now(UTC).isoformat(),
        "snapshot_path": str(snapshot_path.resolve()),
        "snapshot_sha256": sha256_file(snapshot_path),
        "snapshot_size": snapshot_path.stat().st_size,
        "review_count": len(source_state),
        "finalized_review_count": sum(row["status"] == "finalized" for row in source_state),
        "review_state_fingerprint": state_fingerprint,
        "latest_review_event_id": latest_review_event_id,
        "integrity_check": integrity,
        "foreign_key_violation_count": 0,
        "reused_snapshot": bool(backup["reused"]),
    }
    write_json_atomic(root / "manifests" / f"{snapshot_path.stem}.json", payload, mode=0o600)
    write_json_atomic(root / "LATEST.json", payload, mode=0o600)
    _prune_verified_snapshots(root, keep=_SNAPSHOT_RETENTION)
    return payload


def review_backup_status(*, config: AppConfig, store: Store, batch_id: str) -> dict[str, Any]:
    """Verify that LATEST protects the exact current non-pending review state."""

    latest_path = config.sensitive_root / "review_backups" / batch_id / "LATEST.json"
    try:
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
        snapshot_path = Path(str(payload["snapshot_path"]))
        expected_sha256 = str(payload["snapshot_sha256"])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise IntegrityError(f"review backup LATEST is invalid: {latest_path}") from exc
    if not snapshot_path.is_file() or sha256_file(snapshot_path) != expected_sha256:
        raise IntegrityError("review backup snapshot checksum mismatch")
    source_state = _review_state(store.connection, batch_id)
    if fingerprint_json(source_state) != payload.get("review_state_fingerprint"):
        raise IntegrityError("review backup is stale relative to live review state")
    with sqlite3.connect(f"file:{snapshot_path}?mode=ro", uri=True) as snapshot:
        snapshot.row_factory = sqlite3.Row
        if snapshot.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise IntegrityError("review backup integrity check failed")
        if snapshot.execute("PRAGMA foreign_key_check").fetchall():
            raise IntegrityError("review backup foreign-key check failed")
        if _review_state(snapshot, batch_id) != source_state:
            raise IntegrityError("review backup content differs from live review state")
        copied_event_id = _latest_review_event_id(snapshot, batch_id)
    live_event_id = _latest_review_event_id(store.connection, batch_id)
    if copied_event_id != live_event_id or payload.get("latest_review_event_id") != live_event_id:
        raise IntegrityError("review backup event sequence differs from live state")
    return {
        **payload,
        "retained_snapshot_count": len(list((latest_path.parent / "snapshots").glob("*.sqlite3"))),
        "backup_lag": 0,
    }


def restore_review_backup_smoke(
    *, config: AppConfig, store: Store, batch_id: str
) -> dict[str, Any]:
    """Restore LATEST into an isolated temporary database and verify its state."""

    status = review_backup_status(config=config, store=store, batch_id=batch_id)
    snapshot_path = Path(str(status["snapshot_path"]))
    backup_root = config.sensitive_root / "review_backups" / batch_id
    with tempfile.TemporaryDirectory(prefix="restore_smoke_", dir=backup_root) as work:
        restored_path = Path(work) / "restored.sqlite3"
        with sqlite3.connect(f"file:{snapshot_path}?mode=ro", uri=True) as source:
            backup_database(source, restored_path)
        with sqlite3.connect(f"file:{restored_path}?mode=ro", uri=True) as restored:
            restored.row_factory = sqlite3.Row
            integrity = str(restored.execute("PRAGMA integrity_check").fetchone()[0])
            foreign_key_violations = restored.execute("PRAGMA foreign_key_check").fetchall()
            restored_state = _review_state(restored, batch_id)
            restored_event_id = _latest_review_event_id(restored, batch_id)

    restored_fingerprint = fingerprint_json(restored_state)
    if integrity != "ok":
        raise IntegrityError("restored review backup failed integrity check")
    if foreign_key_violations:
        raise IntegrityError("restored review backup failed foreign-key check")
    if restored_fingerprint != status["review_state_fingerprint"]:
        raise IntegrityError("restored review backup state fingerprint mismatch")
    if restored_event_id != status["latest_review_event_id"]:
        raise IntegrityError("restored review backup event sequence mismatch")

    return {
        "schema_version": 1,
        "status": "passed",
        "batch_id": batch_id,
        "source_snapshot_sha256": status["snapshot_sha256"],
        "restored_review_count": len(restored_state),
        "review_state_fingerprint": restored_fingerprint,
        "latest_review_event_id": restored_event_id,
        "integrity_check": integrity,
        "foreign_key_violation_count": 0,
    }
