from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_processing import prepared_fixture

from data_quality_checker.errors import ContractError
from data_quality_checker.fingerprints import sha256_file
from data_quality_checker.g0 import CheckpointCandidate
from data_quality_checker.loop_bindings import (
    compose_step,
    judge_step,
    measure_step,
    predict_step,
    route_step,
    seal_step,
    select_step,
)
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


def test_judge_step_runs_the_pilot_and_returns_a_sha(tmp_path: Path) -> None:
    config = prepared_fixture(tmp_path)
    # judge_step now forwards `generation` to run_judge_pilot, so the round
    # under test must have a prediction pass under that same round's generation
    # label rather than the fixed "G0".
    predict_step(config, batch_id="batch", generation="M003", fake_backend=True)(new_round(1))
    artifact = judge_step(config, batch_id="batch", generation="M003", fake_backend=True)(
        new_round(1)
    )
    summary = config.public_root / "batches" / "batch" / "judge_pilot_summary.json"
    assert artifact == sha256_file(summary)


def test_judge_step_refuses_an_external_call_without_consent(tmp_path: Path) -> None:
    """The pipeline's own gate must not be bypassed by wrapping it."""
    from data_quality_checker.errors import GateBlocked

    config = prepared_fixture(tmp_path)
    predict_step(config, batch_id="batch", generation="M003", fake_backend=True)(new_round(1))
    with pytest.raises(GateBlocked):
        judge_step(config, batch_id="batch", generation="M003", allow_external_judge=False)(
            new_round(1)
        )


def test_compose_step_writes_the_round_training_manifest(tmp_path: Path) -> None:
    config = prepared_fixture(tmp_path)
    out = tmp_path / "rounds"
    split = {"train": [1, 2, 3], "valid": [10, 11], "test": [20, 21, 22]}
    artifact = compose_step(
        config,
        split=split,
        cleaned_rounds=[["dA", "dB"]],
        output_dir=out,
        split_manifest_sha256="abc123",
    )(new_round(1))
    manifest_path = out / "round_001_training.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["counts"]["train"] == 5
    assert payload["split_manifest_sha256"] == "abc123"
    assert artifact == sha256_file(manifest_path)


def test_compose_step_keeps_validation_and_test_fixed(tmp_path: Path) -> None:
    config = prepared_fixture(tmp_path)
    out = tmp_path / "rounds"
    split = {"train": [1, 2, 3], "valid": [10, 11], "test": [20, 21, 22]}
    for index, cleaned in enumerate(([], [["dA"]], [["dA"], ["dB"]]), start=1):
        compose_step(
            config,
            split=split,
            cleaned_rounds=cleaned,
            output_dir=out,
            split_manifest_sha256="abc123",
        )(new_round(index))
    counts = [
        json.loads((out / f"round_{i:03d}_training.json").read_text(encoding="utf-8"))["counts"]
        for i in (1, 2, 3)
    ]
    assert [c["valid"] for c in counts] == [2, 2, 2]
    assert [c["test"] for c in counts] == [3, 3, 3]
    assert [c["train"] for c in counts] == [3, 4, 5]


def _candidates() -> list[CheckpointCandidate]:
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
        for update, score in ((50, 0.70), (100, 0.78))
    ]


def test_select_step_records_the_winner(tmp_path: Path) -> None:
    config = prepared_fixture(tmp_path)
    out = tmp_path / "rounds"
    artifact = select_step(config, candidates=_candidates(), output_dir=out)(new_round(3))
    payload = json.loads((out / "round_003_checkpoint.json").read_text(encoding="utf-8"))
    assert payload["selected"]["update"] == 100
    assert payload["selection_basis"] == "validation"
    assert artifact == sha256_file(out / "round_003_checkpoint.json")


def test_select_step_records_only_validation_metrics(tmp_path: Path) -> None:
    config = prepared_fixture(tmp_path)
    out = tmp_path / "rounds"
    select_step(config, candidates=_candidates(), output_dir=out)(new_round(3))
    raw = (out / "round_003_checkpoint.json").read_text(encoding="utf-8").lower()
    assert "test" not in raw


def test_seal_step_writes_the_round_record(tmp_path: Path) -> None:
    config = prepared_fixture(tmp_path)
    out = tmp_path / "rounds"
    state = new_round(4)
    artifact = seal_step(config, output_dir=out)(state)
    payload = json.loads((out / "round_004_record.json").read_text(encoding="utf-8"))
    assert payload["round"] == 4
    assert payload["stage"] == state.stage
    assert artifact == sha256_file(out / "round_004_record.json")


def test_canonical_evaluate_is_public() -> None:
    from data_quality_checker.g0_training import canonical_evaluate

    assert callable(canonical_evaluate)


def test_measure_step_runs_the_real_evaluator_and_returns_a_sha(tmp_path: Path) -> None:
    """Integration test against this repo's own evaluator (no mock, no stub).

    The evaluator gate inside `canonical_evaluate` requires the evaluated
    document count to match the doc-ids file exactly, so this test drives it
    with the project's real frozen "test" split (50 documents) rather than an
    arbitrary handful -- anything else trips the coverage gate before we ever
    get to assert non-emptiness.
    """
    config = prepared_fixture(tmp_path)
    split = json.loads(config.reference_split_manifest_path.read_text(encoding="utf-8"))
    doc_ids = split["splits"]["test"]
    assert len(doc_ids) == 50

    predictions = []
    for doc_id in doc_ids:
        gold = json.loads(
            (config.canonical_gt_dir / f"doc_{doc_id}.json").read_text(encoding="utf-8")
        )
        predictions.append(
            {
                "doc_id": doc_id,  # must stay an int -- evaluate.py skips non-int doc_ids
                "status": "success",
                "references": [
                    reference
                    for reference in gold["references"]
                    if str(reference.get("status", "approved")).lower() == "approved"
                ],
            }
        )
    assert all(isinstance(row["doc_id"], int) for row in predictions)

    predictions_path = tmp_path / "predictions.json"
    doc_ids_path = tmp_path / "test_doc_ids.json"
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    doc_ids_path.write_text(json.dumps(doc_ids), encoding="utf-8")

    out = tmp_path / "measurement"
    artifact = measure_step(
        config,
        predictions_path=predictions_path,
        test_doc_ids_path=doc_ids_path,
        output_dir=out,
    )(new_round(5))

    report_path = out / "evaluation.json"
    assert artifact == sha256_file(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    results = report["results"]
    assert len(results) > 0
    assert results[0]["evaluated_doc_count"] == 50


def test_the_runner_drives_the_real_bindings_up_to_adjudication(tmp_path: Path) -> None:
    """The only test proving the runner and the pipeline actually fit together.

    Drives `run_round` with the real predict/route/judge bindings against a
    prepared fixture and confirms it reaches `judged` and stops there because
    `adjudicated` is manual. Uses a round generation of "M003" throughout
    (rather than "G0") now that the judge stage can look at any round's own
    predictions.
    """
    from data_quality_checker.loop_runner import run_round

    config = prepared_fixture(tmp_path)
    out = tmp_path / "rounds"
    state = run_round(
        out,
        1,
        steps={
            "predicted": predict_step(
                config, batch_id="batch", generation="M003", fake_backend=True
            ),
            "routed": route_step(config, batch_id="batch", output_dir=out),
            "judged": judge_step(config, batch_id="batch", generation="M003", fake_backend=True),
        },
    )
    assert state.stage == "judged"
    assert set(state.artifacts) == {"predicted", "routed", "judged"}
    assert all(len(sha) == 64 for sha in state.artifacts.values())
