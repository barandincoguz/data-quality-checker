"""Portable unit fixtures; real model/cache checks remain compute integration tests."""

import pytest


@pytest.fixture
def judge_snapshot_stub(tmp_path, monkeypatch):
    snapshot = tmp_path / "judge-snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"model_type": "gemma4"}')
    monkeypatch.setattr("data_quality_checker.judges.resolve_judge_snapshot", lambda: snapshot)
    return snapshot
