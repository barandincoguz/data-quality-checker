"""Single-writer lock and explicit stale-owner inspection."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any

from .errors import LockUnavailable


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


def read_lock_metadata(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"status": "invalid_metadata"}
    return payload if isinstance(payload, dict) else {"status": "invalid_metadata"}


def owner_is_stale(metadata: dict[str, Any] | None, *, now: float | None = None) -> bool:
    if not metadata:
        return True
    current_time = time.time() if now is None else now
    host = metadata.get("host")
    pid = metadata.get("pid")
    updated_at = metadata.get("updated_at_epoch", metadata.get("acquired_at_epoch"))
    if host == socket.gethostname() and isinstance(pid, int) and not process_exists(pid):
        return True
    if isinstance(updated_at, (int, float)) and current_time < float(updated_at):
        return False
    return False


class FileLock:
    def __init__(self, path: Path, *, purpose: str) -> None:
        self.path = path.resolve()
        self.purpose = purpose
        self._handle: Any | None = None
        self.owner_token = uuid.uuid4().hex
        self.previous_metadata: dict[str, Any] | None = None

    def acquire(self) -> FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        self.previous_metadata = read_lock_metadata(self.path)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            owner = self.previous_metadata or {}
            raise LockUnavailable(
                f"writer lock is active at {self.path}; "
                f"pid={owner.get('pid')} host={owner.get('host')}"
            ) from exc
        self._handle = handle
        self._write_metadata(status="running")
        return self

    def _write_metadata(self, **updates: Any) -> None:
        if self._handle is None:
            raise RuntimeError("lock is not acquired")
        now = time.time()
        metadata = {
            "schema_version": 1,
            "purpose": self.purpose,
            "owner_token": self.owner_token,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "host": socket.gethostname(),
            "acquired_at_epoch": now,
            "updated_at_epoch": now,
            **updates,
        }
        self._handle.seek(0)
        self._handle.truncate()
        json.dump(metadata, self._handle, ensure_ascii=False, sort_keys=True)
        self._handle.write("\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def touch(self, **updates: Any) -> None:
        self._write_metadata(status="running", **updates)

    def release(self, *, status: str = "released") -> None:
        if self._handle is None:
            return
        try:
            self._write_metadata(status=status, released_at_epoch=time.time())
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> FileLock:
        return self.acquire()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release(status="failed" if exc_type else "released")
