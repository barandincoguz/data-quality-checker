from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_processing import prepared_fixture

from data_quality_checker.errors import ContractError
from data_quality_checker.fingerprints import sha256_file
from data_quality_checker.loop_bindings import predict_step, route_step
from data_quality_checker.loop_rounds import new_round


def test_predict_step_runs_the_pipeline_and_returns_a_sha(tmp_path: Path) -> None:
    config = prepared_fixture(tmp_path)
    step = predict_step(config, batch_id="batch", generation="M003", fake_backend=True)
    artifact = step(new_round(1))
    summary = config.public_root / "batches" / "batch" / "process_M003_summary.json"
    assert artifact == sha256_file(summary)


def test_predict_step_writes_under_its_own_generation(tmp_path: Path) -> None:
    config = prepared_fixture(tmp_path)
    predict_step(config, batch_id="batch", generation="M003", fake_backend=True)(new_round(1))
    assert (config.sensitive_root / "batches" / "batch" / "predictions" / "M003").is_dir()


def test_predict_step_is_idempotent(tmp_path: Path) -> None:
    """`run_round` may re-run a step after a crash, so a repeat must not differ."""
    config = prepared_fixture(tmp_path)
    step = predict_step(config, batch_id="batch", generation="M003", fake_backend=True)
    assert step(new_round(1)) == step(new_round(1))


def test_route_step_seals_the_bucket_distribution(tmp_path: Path) -> None:
    config = prepared_fixture(tmp_path)
    predict_step(config, batch_id="batch", generation="M003", fake_backend=True)(new_round(1))

    out = tmp_path / "rounds"
    artifact = route_step(config, batch_id="batch", output_dir=out)(new_round(1))
    routing_path = out / "round_001_routing.json"
    payload = json.loads(routing_path.read_text(encoding="utf-8"))
    assert payload["round"] == 1
    assert set(payload["buckets"]) <= {"GREEN", "YELLOW", "RED", "QUARANTINE"}
    assert sum(payload["buckets"].values()) == payload["document_count"]
    assert artifact == sha256_file(routing_path)


def test_route_step_before_any_prediction_is_rejected(tmp_path: Path) -> None:
    config = prepared_fixture(tmp_path)
    with pytest.raises(ContractError):
        route_step(config, batch_id="batch", output_dir=tmp_path / "rounds")(new_round(1))


def test_predict_step_does_not_swallow_a_pipeline_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A swallowed failure would return the sha of a *previous* run's summary
    while the current run had actually failed -- so the exception must escape."""
    config = prepared_fixture(tmp_path)

    def boom(**kwargs: object) -> None:
        raise RuntimeError("process_batch exploded")

    monkeypatch.setattr("data_quality_checker.loop_bindings.process_batch", boom)
    step = predict_step(config, batch_id="batch", generation="M003", fake_backend=True)
    with pytest.raises(RuntimeError, match="process_batch exploded"):
        step(new_round(1))
