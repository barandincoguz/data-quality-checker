from __future__ import annotations

import json
import os

import pytest

from data_quality_checker.atomic import write_json_atomic, write_jsonl_atomic


def test_atomic_json_is_parseable_and_leaves_no_temporary_file(tmp_path) -> None:
    target = tmp_path / "state.json"
    write_json_atomic(target, {"coverage": 3, "ids": [1, 2, 3]})

    assert json.loads(target.read_text(encoding="utf-8"))["coverage"] == 3
    assert list(tmp_path.glob(".*.tmp")) == []


def test_replace_failure_preserves_previous_target(monkeypatch, tmp_path) -> None:
    target = tmp_path / "state.json"
    target.write_text('{"stable":true}\n', encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        write_json_atomic(target, {"stable": False})

    assert json.loads(target.read_text(encoding="utf-8")) == {"stable": True}
    assert list(tmp_path.glob(".*.tmp")) == []


def test_jsonl_is_validated_before_promotion(tmp_path) -> None:
    target = tmp_path / "records.jsonl"
    write_jsonl_atomic(target, [{"id": 1}, {"id": 2}])

    assert [json.loads(line) for line in target.read_text().splitlines()] == [
        {"id": 1},
        {"id": 2},
    ]
