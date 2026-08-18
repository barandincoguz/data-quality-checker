"""Atomically persisted run lifecycle and heartbeat state."""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .locking import FileLock, process_exists


def load_heartbeat(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def heartbeat_is_stale(
    payload: dict[str, Any] | None,
    *,
    max_age_seconds: float,
    now: float | None = None,
) -> bool:
    if not payload:
        return True
    current = time.time() if now is None else now
    updated = payload.get("updated_at_epoch")
    if not isinstance(updated, (int, float)) or current - float(updated) > max_age_seconds:
        return True
    if payload.get("host") == socket.gethostname():
        pid = payload.get("pid")
        return not isinstance(pid, int) or not process_exists(pid)
    return False


@dataclass
class RunLease:
    lock_path: Path
    heartbeat_path: Path
    purpose: str
    run_id: str
    input_fingerprint: str
    config_fingerprint: str
    lock: FileLock = field(init=False)
    started_at_epoch: float = field(init=False, default=0.0)
    state: dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.lock = FileLock(self.lock_path, purpose=self.purpose)

    def start(self, *, stage: str) -> RunLease:
        self.lock.acquire()
        self.started_at_epoch = time.time()
        self.state = {
            "schema_version": 1,
            "run_id": self.run_id,
            "purpose": self.purpose,
            "status": "running",
            "stage": stage,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "host": socket.gethostname(),
            "started_at_epoch": self.started_at_epoch,
            "updated_at_epoch": self.started_at_epoch,
            "last_successful_unit": None,
            "completed_units": 0,
            "expected_units": None,
            "eta_seconds": None,
            "last_error": None,
            "input_fingerprint": self.input_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "recovery_history": [],
        }
        self._persist()
        return self

    def _persist(self) -> None:
        write_json_atomic(self.heartbeat_path, self.state)
        self.lock.touch(stage=self.state.get("stage"), run_id=self.run_id)

    def beat(self, **updates: Any) -> None:
        self.state.update(updates)
        self.state["updated_at_epoch"] = time.time()
        self._persist()

    def finish(self, *, status: str = "completed", error: str | None = None) -> None:
        self.beat(status=status, last_error=error, finished_at_epoch=time.time())
        self.lock.release(status=status)

    def __enter__(self) -> RunLease:
        return self.start(stage="starting")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type:
            self.finish(status="failed", error=str(exc))
        else:
            self.finish()
