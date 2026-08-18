from __future__ import annotations

import json
import sqlite3
import threading

from data_quality_checker.config import default_config_path, load_config
from data_quality_checker.review_backup import review_backup_lock
from data_quality_checker.sqlite_backup import backup_database


def _config(tmp_path):
    source = json.loads(default_config_path().read_text(encoding="utf-8"))
    live = load_config()
    source.update(
        {
            "canonical_gt_dir": str(live.canonical_gt_dir),
            "example_bank_path": str(live.example_bank_path),
            "reference_split_manifest_path": str(live.reference_split_manifest_path),
            "sensitive_root": str(tmp_path / "sensitive"),
            "public_root": str(tmp_path / "public"),
            "training_runs_root": str(tmp_path / "runs"),
        }
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return load_config(path)


def test_review_backup_lock_serializes_two_writers(tmp_path) -> None:
    config = _config(tmp_path)
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def first_writer() -> None:
        with review_backup_lock(config=config, batch_id="batch"):
            first_entered.set()
            assert release_first.wait(timeout=2)

    def second_writer() -> None:
        assert first_entered.wait(timeout=2)
        with review_backup_lock(config=config, batch_id="batch"):
            second_entered.set()

    first = threading.Thread(target=first_writer)
    second = threading.Thread(target=second_writer)
    first.start()
    second.start()
    assert first_entered.wait(timeout=2)
    assert not second_entered.wait(timeout=0.1)
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)
    assert second_entered.is_set()


def test_sqlite_backup_leaves_no_temporary_or_journal_sidecars(tmp_path) -> None:
    source_path = tmp_path / "source.sqlite3"
    target_path = tmp_path / "snapshots" / "backup.sqlite3"
    with sqlite3.connect(source_path) as source:
        source.execute("PRAGMA journal_mode=WAL")
        source.execute("CREATE TABLE decisions (value TEXT NOT NULL)")
        source.execute("INSERT INTO decisions VALUES ('kept')")
        source.commit()

        backup_database(source, target_path)

    assert target_path.is_file()
    leftovers = [path.name for path in target_path.parent.iterdir() if path != target_path]
    assert leftovers == []
