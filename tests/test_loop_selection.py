from __future__ import annotations

import json
from pathlib import Path

from data_quality_checker.g0 import CheckpointCandidate
from data_quality_checker.loop_selection import write_selection_record

FORBIDDEN = ("test_", "_test", "test_core_f1", "test_score")


def candidates() -> list[CheckpointCandidate]:
    return [
        CheckpointCandidate(
            update=update,
            coverage_count=50,
            parse_count=50,
            empty_output_count=0,
            runaway_output_count=0,
            core_f1=score,
            docwise_accuracy=0.30,
            recall=0.70,
            validation_loss=0.06,
        )
        for update, score in ((50, 0.70), (100, 0.78), (150, 0.74))
    ]


def test_record_names_the_winner_and_every_candidate(tmp_path: Path) -> None:
    rows = candidates()
    result = write_selection_record(tmp_path, 3, rows[1], rows, validation_documents=50)
    payload = json.loads((tmp_path / "round_003_checkpoint.json").read_text(encoding="utf-8"))
    assert payload["round"] == 3
    assert payload["selected"]["update"] == 100
    assert [row["update"] for row in payload["candidates"]] == [50, 100, 150]
    assert result["record_sha256"]


def test_record_states_the_selection_basis_is_validation(tmp_path: Path) -> None:
    rows = candidates()
    write_selection_record(tmp_path, 3, rows[1], rows, validation_documents=50)
    payload = json.loads((tmp_path / "round_003_checkpoint.json").read_text(encoding="utf-8"))
    assert payload["selection_basis"] == "validation"
    assert payload["validation_documents"] == 50
    assert payload["tie_break_order"] == [
        "core_f1",
        "docwise_accuracy",
        "recall",
        "-validation_loss",
    ]


def test_record_never_mentions_a_test_metric(tmp_path: Path) -> None:
    rows = candidates()
    write_selection_record(tmp_path, 3, rows[1], rows, validation_documents=50)
    raw = (tmp_path / "round_003_checkpoint.json").read_text(encoding="utf-8").lower()
    for token in FORBIDDEN:
        assert token not in raw, f"selection record leaked a test-set reference: {token}"
