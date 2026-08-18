"""Atomic, integrity-checked SQLite online backups."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from .atomic import fsync_directory
from .errors import IntegrityError
from .fingerprints import sha256_file


def _readonly_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _remove_sidecars(path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


def backup_database(source: sqlite3.Connection, target: Path) -> dict[str, Any]:
    """Online-backup ``source`` and promote only an integrity-checked file."""

    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            with _readonly_connection(target) as check:
                integrity = check.execute("PRAGMA integrity_check").fetchone()[0]
        except sqlite3.DatabaseError as exc:
            raise IntegrityError(f"existing SQLite backup cannot be opened: {target}") from exc
        if integrity != "ok":
            raise IntegrityError(f"existing SQLite backup failed integrity check: {target}")
        return {"path": str(target), "sha256": sha256_file(target), "reused": True}

    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with sqlite3.connect(temporary) as destination:
            source.backup(destination)
            destination.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            destination.execute("PRAGMA journal_mode=DELETE")
        with _readonly_connection(temporary) as check:
            if check.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise IntegrityError("SQLite backup failed integrity check")
        _remove_sidecars(temporary)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        fsync_directory(target.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        _remove_sidecars(temporary)
        raise
    return {"path": str(target), "sha256": sha256_file(target), "reused": False}
