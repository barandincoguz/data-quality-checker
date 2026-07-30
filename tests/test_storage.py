from __future__ import annotations

import sqlite3

import pytest

from data_quality_checker.errors import FingerprintMismatch, VersionConflict
from data_quality_checker.storage import Store


def sample_document() -> dict:
    return {
        "internal_doc_id": "d000001",
        "public_doc_id": "pub_abc",
        "raw_document_id": "raw-1",
        "selected_channel": "pdfText",
        "pdf_coverage": 1.0,
        "html_coverage": 0.0,
        "text": "213 sayılı Vergi Usul Kanunu",
        "text_sha256": "c" * 64,
        "human_references": [],
        "metadata": {"source": "fixture"},
        "warnings": [],
        "preparation_status": "ready",
    }


def test_migration_enables_wal_foreign_keys_and_schema(tmp_path) -> None:
    with Store(tmp_path / "db.sqlite3") as store:
        assert store.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert store.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert store.connection.execute("PRAGMA user_version").fetchone()[0] == 1


def test_batch_resume_rejects_fingerprint_drift(tmp_path) -> None:
    with Store(tmp_path / "db.sqlite3") as store:
        store.create_batch(
            batch_id="batch-1", input_fingerprint="a" * 64, config_fingerprint="b" * 64
        )
        store.create_batch(
            batch_id="batch-1", input_fingerprint="a" * 64, config_fingerprint="b" * 64
        )
        with pytest.raises(FingerprintMismatch):
            store.create_batch(
                batch_id="batch-1",
                input_fingerprint="changed",
                config_fingerprint="b" * 64,
            )


def test_document_foreign_key_and_optimistic_batch_version(tmp_path) -> None:
    with Store(tmp_path / "db.sqlite3") as store:
        with pytest.raises(sqlite3.IntegrityError):
            store.add_document("missing", sample_document())

        batch = store.create_batch(
            batch_id="batch-1", input_fingerprint="a" * 64, config_fingerprint="b" * 64
        )
        store.add_document("batch-1", sample_document())
        updated = store.update_batch(
            "batch-1",
            expected_version=batch["row_version"],
            status="ready",
            ready=True,
        )
        assert updated["ready"] == 1
        assert store.status_summary("batch-1")["documents"] == {"UNROUTED": 1}

        with pytest.raises(VersionConflict):
            store.update_batch(
                "batch-1", expected_version=batch["row_version"], status="stale-write"
            )


def test_prediction_resume_rejects_model_drift(tmp_path) -> None:
    with Store(tmp_path / "db.sqlite3") as store:
        store.create_batch(
            batch_id="batch-1", input_fingerprint="a" * 64, config_fingerprint="b" * 64
        )
        store.add_document("batch-1", sample_document())
        kwargs = {
            "batch_id": "batch-1",
            "internal_doc_id": "d000001",
            "generation": "G0",
            "status": "success",
            "references": [],
            "response_path": tmp_path / "response.json",
            "response_sha256": "d" * 64,
            "input_fingerprint": "e" * 64,
            "model_fingerprint": "f" * 64,
        }
        store.persist_prediction(**kwargs)
        with pytest.raises(FingerprintMismatch):
            store.persist_prediction(**{**kwargs, "model_fingerprint": "changed"})


def test_review_and_audit_event_commit_atomically(tmp_path) -> None:
    with Store(tmp_path / "db.sqlite3") as store:
        store.create_batch(
            batch_id="batch-1",
            input_fingerprint="a" * 64,
            config_fingerprint="b" * 64,
        )
        store.add_document("batch-1", sample_document())
        store.connection.execute(
            """
            CREATE TRIGGER reject_review_event
            BEFORE INSERT ON batch_events
            WHEN NEW.event_type='expert_review_updated'
            BEGIN
              SELECT RAISE(ABORT, 'injected event failure');
            END
            """
        )

        with pytest.raises(sqlite3.IntegrityError, match="injected event failure"):
            store.update_review_with_event(
                batch_id="batch-1",
                internal_doc_id="d000001",
                expected_version=0,
                status="finalized",
                action="accept_human",
                final_references=[],
                reason=None,
                reviewer="fixture",
                event_payload={"internal_doc_id": "d000001"},
            )

        review = store.get_review("batch-1", "d000001")
        assert review is not None
        assert review["status"] == "pending"
        assert review["row_version"] == 0
