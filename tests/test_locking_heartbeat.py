from __future__ import annotations

import os
import time

import pytest

from data_quality_checker.errors import LockUnavailable
from data_quality_checker.heartbeat import RunLease, heartbeat_is_stale, load_heartbeat
from data_quality_checker.locking import FileLock


def test_second_writer_is_rejected(tmp_path) -> None:
    first = FileLock(tmp_path / "writer.lock", purpose="test").acquire()
    try:
        with pytest.raises(LockUnavailable):
            FileLock(tmp_path / "writer.lock", purpose="test-2").acquire()
    finally:
        first.release()


def test_run_lease_persists_progress_and_terminal_state(tmp_path) -> None:
    heartbeat = tmp_path / "heartbeat.json"
    lease = RunLease(
        lock_path=tmp_path / "writer.lock",
        heartbeat_path=heartbeat,
        purpose="fixture",
        run_id="run-1",
        input_fingerprint="a" * 64,
        config_fingerprint="b" * 64,
    ).start(stage="prepare")
    lease.beat(expected_units=2, completed_units=1, last_successful_unit="doc-1")

    running = load_heartbeat(heartbeat)
    assert running is not None
    assert running["completed_units"] == 1
    assert heartbeat_is_stale(running, max_age_seconds=60) is False

    lease.finish()
    finished = load_heartbeat(heartbeat)
    assert finished is not None
    assert finished["status"] == "completed"


def test_old_or_dead_heartbeat_is_stale() -> None:
    payload = {
        "host": __import__("socket").gethostname(),
        "pid": 2**30,
        "updated_at_epoch": time.time(),
    }
    assert heartbeat_is_stale(payload, max_age_seconds=60)
    assert heartbeat_is_stale(
        {"host": "remote", "pid": os.getpid(), "updated_at_epoch": 1},
        max_age_seconds=60,
    )
