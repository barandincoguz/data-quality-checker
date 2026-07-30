"""Schema-versioned SQLite store for sensitive workflow state."""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from .constants import SCHEMA_VERSION
from .errors import FingerprintMismatch, IntegrityError, VersionConflict
from .fingerprints import canonical_json_bytes


MIGRATIONS: dict[int, str] = {
    1: """
        CREATE TABLE batches (
            batch_id TEXT PRIMARY KEY,
            input_fingerprint TEXT NOT NULL,
            config_fingerprint TEXT NOT NULL,
            status TEXT NOT NULL,
            ready INTEGER NOT NULL DEFAULT 0 CHECK (ready IN (0, 1)),
            metadata_json TEXT NOT NULL DEFAULT '{}',
            row_version INTEGER NOT NULL DEFAULT 0,
            created_at_epoch REAL NOT NULL,
            updated_at_epoch REAL NOT NULL
        );

        CREATE TABLE documents (
            batch_id TEXT NOT NULL,
            internal_doc_id TEXT NOT NULL,
            public_doc_id TEXT NOT NULL,
            raw_document_id TEXT NOT NULL,
            selected_channel TEXT NOT NULL,
            pdf_coverage REAL NOT NULL,
            html_coverage REAL NOT NULL,
            text TEXT NOT NULL,
            text_sha256 TEXT NOT NULL,
            human_references_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            warnings_json TEXT NOT NULL DEFAULT '[]',
            preparation_status TEXT NOT NULL,
            router_bucket TEXT,
            row_version INTEGER NOT NULL DEFAULT 0,
            created_at_epoch REAL NOT NULL,
            updated_at_epoch REAL NOT NULL,
            PRIMARY KEY (batch_id, internal_doc_id),
            UNIQUE (batch_id, public_doc_id),
            FOREIGN KEY (batch_id) REFERENCES batches(batch_id) ON DELETE RESTRICT
        );

        CREATE TABLE predictions (
            batch_id TEXT NOT NULL,
            internal_doc_id TEXT NOT NULL,
            generation TEXT NOT NULL,
            status TEXT NOT NULL,
            references_json TEXT NOT NULL,
            response_path TEXT NOT NULL,
            response_sha256 TEXT NOT NULL,
            input_fingerprint TEXT NOT NULL,
            model_fingerprint TEXT NOT NULL,
            error TEXT,
            operational_json TEXT NOT NULL DEFAULT '{}',
            created_at_epoch REAL NOT NULL,
            updated_at_epoch REAL NOT NULL,
            PRIMARY KEY (batch_id, internal_doc_id, generation),
            FOREIGN KEY (batch_id, internal_doc_id)
              REFERENCES documents(batch_id, internal_doc_id) ON DELETE RESTRICT
        );

        CREATE TABLE judge_results (
            batch_id TEXT NOT NULL,
            internal_doc_id TEXT NOT NULL,
            model TEXT NOT NULL,
            blind_mapping TEXT NOT NULL,
            status TEXT NOT NULL,
            verdict TEXT,
            result_json TEXT NOT NULL,
            response_path TEXT,
            response_sha256 TEXT,
            retry_count INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            created_at_epoch REAL NOT NULL,
            updated_at_epoch REAL NOT NULL,
            PRIMARY KEY (batch_id, internal_doc_id, model),
            FOREIGN KEY (batch_id, internal_doc_id)
              REFERENCES documents(batch_id, internal_doc_id) ON DELETE RESTRICT
        );

        CREATE TABLE reviews (
            batch_id TEXT NOT NULL,
            internal_doc_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            action TEXT,
            final_references_json TEXT,
            reason TEXT,
            reviewer TEXT,
            row_version INTEGER NOT NULL DEFAULT 0,
            created_at_epoch REAL NOT NULL,
            updated_at_epoch REAL NOT NULL,
            PRIMARY KEY (batch_id, internal_doc_id),
            FOREIGN KEY (batch_id, internal_doc_id)
              REFERENCES documents(batch_id, internal_doc_id) ON DELETE RESTRICT
        );

        CREATE TABLE batch_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at_epoch REAL NOT NULL,
            FOREIGN KEY (batch_id) REFERENCES batches(batch_id) ON DELETE RESTRICT
        );

        CREATE TABLE releases (
            release_id TEXT PRIMARY KEY,
            batch_id TEXT NOT NULL UNIQUE,
            release_path TEXT NOT NULL UNIQUE,
            manifest_sha256 TEXT NOT NULL,
            created_at_epoch REAL NOT NULL,
            FOREIGN KEY (batch_id) REFERENCES batches(batch_id) ON DELETE RESTRICT
        );

        CREATE INDEX idx_documents_batch_router
          ON documents(batch_id, router_bucket, internal_doc_id);
        CREATE INDEX idx_reviews_batch_status
          ON reviews(batch_id, status, internal_doc_id);
        CREATE INDEX idx_events_batch
          ON batch_events(batch_id, event_id);
    """,
}


def _json(payload: Any) -> str:
    return canonical_json_bytes(payload).decode("utf-8")


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class Store:
    def __init__(self, path: Path, *, busy_timeout_ms: int = 5000) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            self.path,
            timeout=busy_timeout_ms / 1000,
            isolation_level=None,
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        self.connection.execute("PRAGMA synchronous = FULL")
        journal_mode = self.connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        if str(journal_mode).lower() != "wal":
            raise IntegrityError(f"SQLite refused WAL mode: {journal_mode}")
        self._migrate()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        self.connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield self.connection
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")

    def _migrate(self) -> None:
        current = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        if current > SCHEMA_VERSION:
            raise IntegrityError(
                f"database schema {current} is newer than supported {SCHEMA_VERSION}"
            )
        for version in range(current + 1, SCHEMA_VERSION + 1):
            sql = MIGRATIONS.get(version)
            if sql is None:
                raise IntegrityError(f"missing migration {version}")
            try:
                self.connection.executescript(
                    f"BEGIN IMMEDIATE;\n{sql}\nPRAGMA user_version = {version};\nCOMMIT;"
                )
            except BaseException:
                if self.connection.in_transaction:
                    self.connection.execute("ROLLBACK")
                raise

    def create_batch(
        self,
        *,
        batch_id: str,
        input_fingerprint: str,
        config_fingerprint: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        with self.transaction():
            existing = self.connection.execute(
                "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if existing is not None:
                if (
                    existing["input_fingerprint"] != input_fingerprint
                    or existing["config_fingerprint"] != config_fingerprint
                ):
                    raise FingerprintMismatch(
                        f"batch {batch_id} exists with different input/config fingerprints"
                    )
                return dict(existing)
            self.connection.execute(
                """
                INSERT INTO batches(
                  batch_id, input_fingerprint, config_fingerprint, status,
                  metadata_json, created_at_epoch, updated_at_epoch
                ) VALUES (?, ?, ?, 'preparing', ?, ?, ?)
                """,
                (batch_id, input_fingerprint, config_fingerprint, _json(metadata or {}), now, now),
            )
        result = self.get_batch(batch_id)
        assert result is not None
        return result

    def get_batch(self, batch_id: str) -> dict[str, Any] | None:
        return _row(
            self.connection.execute(
                "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
        )

    def update_batch(
        self,
        batch_id: str,
        *,
        expected_version: int,
        status: str,
        ready: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = self.get_batch(batch_id)
        if current is None:
            raise KeyError(batch_id)
        ready_value = int(current["ready"] if ready is None else ready)
        metadata_json = current["metadata_json"] if metadata is None else _json(metadata)
        now = time.time()
        with self.transaction():
            cursor = self.connection.execute(
                """
                UPDATE batches
                   SET status = ?, ready = ?, metadata_json = ?,
                       row_version = row_version + 1, updated_at_epoch = ?
                 WHERE batch_id = ? AND row_version = ?
                """,
                (status, ready_value, metadata_json, now, batch_id, expected_version),
            )
            if cursor.rowcount != 1:
                raise VersionConflict(f"batch {batch_id} changed concurrently")
        result = self.get_batch(batch_id)
        assert result is not None
        return result

    def add_document(self, batch_id: str, document: dict[str, Any]) -> None:
        now = time.time()
        with self.transaction():
            self.connection.execute(
                """
                INSERT INTO documents(
                  batch_id, internal_doc_id, public_doc_id, raw_document_id,
                  selected_channel, pdf_coverage, html_coverage, text, text_sha256,
                  human_references_json, metadata_json, warnings_json,
                  preparation_status, router_bucket, created_at_epoch, updated_at_epoch
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(batch_id, internal_doc_id) DO UPDATE SET
                  public_doc_id=excluded.public_doc_id,
                  raw_document_id=excluded.raw_document_id,
                  selected_channel=excluded.selected_channel,
                  pdf_coverage=excluded.pdf_coverage,
                  html_coverage=excluded.html_coverage,
                  text=excluded.text,
                  text_sha256=excluded.text_sha256,
                  human_references_json=excluded.human_references_json,
                  metadata_json=excluded.metadata_json,
                  warnings_json=excluded.warnings_json,
                  preparation_status=excluded.preparation_status,
                  router_bucket=excluded.router_bucket,
                  row_version=documents.row_version+1,
                  updated_at_epoch=excluded.updated_at_epoch
                """,
                (
                    batch_id,
                    document["internal_doc_id"],
                    document["public_doc_id"],
                    document["raw_document_id"],
                    document["selected_channel"],
                    float(document["pdf_coverage"]),
                    float(document["html_coverage"]),
                    document["text"],
                    document["text_sha256"],
                    _json(document["human_references"]),
                    _json(document.get("metadata", {})),
                    _json(document.get("warnings", [])),
                    document.get("preparation_status", "ready"),
                    document.get("router_bucket"),
                    now,
                    now,
                ),
            )
            self.connection.execute(
                """
                INSERT INTO reviews(
                  batch_id, internal_doc_id, status, created_at_epoch, updated_at_epoch
                ) VALUES (?, ?, 'pending', ?, ?)
                ON CONFLICT(batch_id, internal_doc_id) DO NOTHING
                """,
                (batch_id, document["internal_doc_id"], now, now),
            )

    def get_document(self, batch_id: str, internal_doc_id: str) -> dict[str, Any] | None:
        return _row(
            self.connection.execute(
                "SELECT * FROM documents WHERE batch_id = ? AND internal_doc_id = ?",
                (batch_id, internal_doc_id),
            ).fetchone()
        )

    def list_documents(
        self,
        batch_id: str,
        *,
        router_buckets: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        if router_buckets:
            placeholders = ",".join("?" for _ in router_buckets)
            rows = self.connection.execute(
                f"""SELECT * FROM documents
                     WHERE batch_id = ? AND router_bucket IN ({placeholders})
                     ORDER BY internal_doc_id""",
                (batch_id, *router_buckets),
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM documents WHERE batch_id = ? ORDER BY internal_doc_id",
                (batch_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def persist_prediction(
        self,
        *,
        batch_id: str,
        internal_doc_id: str,
        generation: str,
        status: str,
        references: list[dict[str, str]],
        response_path: Path,
        response_sha256: str,
        input_fingerprint: str,
        model_fingerprint: str,
        error: str | None = None,
        operational: dict[str, Any] | None = None,
    ) -> None:
        now = time.time()
        with self.transaction():
            existing = self.connection.execute(
                """SELECT input_fingerprint, model_fingerprint, response_sha256
                     FROM predictions
                    WHERE batch_id=? AND internal_doc_id=? AND generation=?""",
                (batch_id, internal_doc_id, generation),
            ).fetchone()
            if existing is not None and (
                existing["input_fingerprint"] != input_fingerprint
                or existing["model_fingerprint"] != model_fingerprint
            ):
                raise FingerprintMismatch(
                    f"prediction resume drift for {batch_id}/{internal_doc_id}/{generation}"
                )
            self.connection.execute(
                """
                INSERT INTO predictions(
                  batch_id, internal_doc_id, generation, status, references_json,
                  response_path, response_sha256, input_fingerprint,
                  model_fingerprint, error, operational_json,
                  created_at_epoch, updated_at_epoch
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(batch_id, internal_doc_id, generation) DO UPDATE SET
                  status=excluded.status,
                  references_json=excluded.references_json,
                  response_path=excluded.response_path,
                  response_sha256=excluded.response_sha256,
                  error=excluded.error,
                  operational_json=excluded.operational_json,
                  updated_at_epoch=excluded.updated_at_epoch
                """,
                (
                    batch_id,
                    internal_doc_id,
                    generation,
                    status,
                    _json(references),
                    str(response_path),
                    response_sha256,
                    input_fingerprint,
                    model_fingerprint,
                    error,
                    _json(operational or {}),
                    now,
                    now,
                ),
            )

    def get_prediction(
        self, batch_id: str, internal_doc_id: str, generation: str
    ) -> dict[str, Any] | None:
        return _row(
            self.connection.execute(
                """SELECT * FROM predictions
                    WHERE batch_id=? AND internal_doc_id=? AND generation=?""",
                (batch_id, internal_doc_id, generation),
            ).fetchone()
        )

    def list_predictions(self, batch_id: str, generation: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT * FROM predictions
                WHERE batch_id=? AND generation=? ORDER BY internal_doc_id""",
            (batch_id, generation),
        ).fetchall()
        return [dict(row) for row in rows]

    def set_router_bucket(self, batch_id: str, internal_doc_id: str, bucket: str) -> None:
        with self.transaction():
            cursor = self.connection.execute(
                """UPDATE documents SET router_bucket=?, row_version=row_version+1,
                           updated_at_epoch=?
                     WHERE batch_id=? AND internal_doc_id=?""",
                (bucket, time.time(), batch_id, internal_doc_id),
            )
            if cursor.rowcount != 1:
                raise KeyError((batch_id, internal_doc_id))

    def update_review(
        self,
        *,
        batch_id: str,
        internal_doc_id: str,
        expected_version: int,
        status: str,
        action: str | None,
        final_references: list[dict[str, str]] | None,
        reason: str | None,
        reviewer: str | None,
    ) -> dict[str, Any]:
        with self.transaction():
            cursor = self.connection.execute(
                """
                UPDATE reviews
                   SET status=?, action=?, final_references_json=?, reason=?, reviewer=?,
                       row_version=row_version+1, updated_at_epoch=?
                 WHERE batch_id=? AND internal_doc_id=? AND row_version=?
                """,
                (
                    status,
                    action,
                    None if final_references is None else _json(final_references),
                    reason,
                    reviewer,
                    time.time(),
                    batch_id,
                    internal_doc_id,
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                raise VersionConflict(
                    f"review {batch_id}/{internal_doc_id} changed concurrently"
                )
        result = _row(
            self.connection.execute(
                "SELECT * FROM reviews WHERE batch_id=? AND internal_doc_id=?",
                (batch_id, internal_doc_id),
            ).fetchone()
        )
        assert result is not None
        return result

    def update_review_with_event(
        self,
        *,
        batch_id: str,
        internal_doc_id: str,
        expected_version: int,
        status: str,
        action: str | None,
        final_references: list[dict[str, str]] | None,
        reason: str | None,
        reviewer: str | None,
        event_payload: dict[str, Any],
    ) -> tuple[dict[str, Any], int]:
        """Commit a review mutation and its audit event as one unit."""

        now = time.time()
        with self.transaction():
            cursor = self.connection.execute(
                """
                UPDATE reviews
                   SET status=?, action=?, final_references_json=?, reason=?, reviewer=?,
                       row_version=row_version+1, updated_at_epoch=?
                 WHERE batch_id=? AND internal_doc_id=? AND row_version=?
                """,
                (
                    status,
                    action,
                    None if final_references is None else _json(final_references),
                    reason,
                    reviewer,
                    now,
                    batch_id,
                    internal_doc_id,
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                raise VersionConflict(
                    f"review {batch_id}/{internal_doc_id} changed concurrently"
                )
            event = self.connection.execute(
                """
                INSERT INTO batch_events(
                  batch_id, event_type, payload_json, created_at_epoch
                ) VALUES (?, 'expert_review_updated', ?, ?)
                """,
                (batch_id, _json(event_payload), now),
            )
            event_id = int(event.lastrowid)
        result = self.get_review(batch_id, internal_doc_id)
        assert result is not None
        return result, event_id

    def get_review(self, batch_id: str, internal_doc_id: str) -> dict[str, Any] | None:
        return _row(
            self.connection.execute(
                "SELECT * FROM reviews WHERE batch_id=? AND internal_doc_id=?",
                (batch_id, internal_doc_id),
            ).fetchone()
        )

    def list_reviews(self, batch_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM reviews WHERE batch_id=? ORDER BY internal_doc_id",
            (batch_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def persist_judge_result(
        self,
        *,
        batch_id: str,
        internal_doc_id: str,
        model: str,
        blind_mapping: str,
        status: str,
        verdict: str | None,
        result: dict[str, Any],
        response_path: Path | None,
        response_sha256: str | None,
        retry_count: int,
        error: str | None,
    ) -> None:
        now = time.time()
        with self.transaction():
            self.connection.execute(
                """
                INSERT INTO judge_results(
                  batch_id, internal_doc_id, model, blind_mapping, status,
                  verdict, result_json, response_path, response_sha256,
                  retry_count, error, created_at_epoch, updated_at_epoch
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(batch_id, internal_doc_id, model) DO UPDATE SET
                  blind_mapping=excluded.blind_mapping,
                  status=excluded.status,
                  verdict=excluded.verdict,
                  result_json=excluded.result_json,
                  response_path=excluded.response_path,
                  response_sha256=excluded.response_sha256,
                  retry_count=excluded.retry_count,
                  error=excluded.error,
                  updated_at_epoch=excluded.updated_at_epoch
                """,
                (
                    batch_id,
                    internal_doc_id,
                    model,
                    blind_mapping,
                    status,
                    verdict,
                    _json(result),
                    None if response_path is None else str(response_path),
                    response_sha256,
                    retry_count,
                    error,
                    now,
                    now,
                ),
            )

    def get_judge_result(
        self, batch_id: str, internal_doc_id: str, model: str
    ) -> dict[str, Any] | None:
        return _row(
            self.connection.execute(
                """SELECT * FROM judge_results
                    WHERE batch_id=? AND internal_doc_id=? AND model=?""",
                (batch_id, internal_doc_id, model),
            ).fetchone()
        )

    def list_judge_results(
        self, batch_id: str, *, model: str | None = None
    ) -> list[dict[str, Any]]:
        if model is None:
            rows = self.connection.execute(
                "SELECT * FROM judge_results WHERE batch_id=? ORDER BY model,internal_doc_id",
                (batch_id,),
            ).fetchall()
        else:
            rows = self.connection.execute(
                """SELECT * FROM judge_results
                    WHERE batch_id=? AND model=? ORDER BY internal_doc_id""",
                (batch_id, model),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_event(self, batch_id: str, event_type: str, payload: dict[str, Any]) -> int:
        with self.transaction():
            cursor = self.connection.execute(
                """INSERT INTO batch_events(batch_id,event_type,payload_json,created_at_epoch)
                     VALUES (?,?,?,?)""",
                (batch_id, event_type, _json(payload), time.time()),
            )
        return int(cursor.lastrowid)

    def record_release(
        self,
        *,
        release_id: str,
        batch_id: str,
        release_path: Path,
        manifest_sha256: str,
    ) -> None:
        with self.transaction():
            self.connection.execute(
                """INSERT INTO releases(
                       release_id,batch_id,release_path,manifest_sha256,created_at_epoch
                     ) VALUES (?,?,?,?,?)""",
                (release_id, batch_id, str(release_path), manifest_sha256, time.time()),
            )

    def get_release_for_batch(self, batch_id: str) -> dict[str, Any] | None:
        return _row(
            self.connection.execute(
                "SELECT * FROM releases WHERE batch_id=?", (batch_id,)
            ).fetchone()
        )

    def status_summary(self, batch_id: str) -> dict[str, Any]:
        batch = self.get_batch(batch_id)
        if batch is None:
            raise KeyError(batch_id)
        document_counts = {
            (row["router_bucket"] or "UNROUTED"): row["count"]
            for row in self.connection.execute(
                """SELECT router_bucket, COUNT(*) AS count FROM documents
                    WHERE batch_id=? GROUP BY router_bucket""",
                (batch_id,),
            ).fetchall()
        }
        review_counts = {
            row["status"]: row["count"]
            for row in self.connection.execute(
                """SELECT status, COUNT(*) AS count FROM reviews
                    WHERE batch_id=? GROUP BY status""",
                (batch_id,),
            ).fetchall()
        }
        predictions = self.connection.execute(
            "SELECT COUNT(*) FROM predictions WHERE batch_id=?", (batch_id,)
        ).fetchone()[0]
        return {
            "batch_id": batch_id,
            "status": batch["status"],
            "ready": bool(batch["ready"]),
            "documents": document_counts,
            "reviews": review_counts,
            "prediction_count": int(predictions),
            "row_version": int(batch["row_version"]),
        }
