from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_processing import prepared_fixture

from data_quality_checker.errors import ContractError, IntegrityError
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


def test_judge_step_requires_the_caller_to_state_a_generation() -> None:
    """`generation` has no default -- the previous commit removed the same
    default from `judges.py` precisely because a round's predictions never
    live under "G0", so a binding-layer default would silently reintroduce
    the bug it just closed."""
    import inspect

    signature = inspect.signature(judge_step)
    assert signature.parameters["generation"].default is inspect.Parameter.empty


def test_judge_step_judges_the_rounds_own_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A round's judge must see that round's predictions, not G0's -- predict
    under both G0 and M003, judge with generation="M003", and confirm the
    judged candidate references trace back to the M003 prediction.

    The fixture's fake backend echoes the human references regardless of
    generation, so G0 and M003 predictions would be identical without help;
    this test gives the M003 prediction distinguishable content directly
    through the store so the two generations can actually be told apart.
    """
    from data_quality_checker import judges as judges_module
    from data_quality_checker.storage import Store

    config = prepared_fixture(tmp_path, two_docs=False)
    predict_step(config, batch_id="batch", generation="G0", fake_backend=True)(new_round(1))
    predict_step(config, batch_id="batch", generation="M003", fake_backend=True)(new_round(1))

    m003_references = [
        {
            "kanun_no": "999",
            "kanun_ad": "M003 Test Kanunu",
            "madde": "413",
            "fikra": "",
            "bent": "",
            "source_text": "213 sayılı Vergi Usul Kanununun 413. maddesi",
        }
    ]
    with Store(config.database_path) as store:
        internal_doc_id = store.list_documents("batch")[0]["internal_doc_id"]
        existing = store.get_prediction("batch", internal_doc_id, "M003")
        assert existing is not None
        store.persist_prediction(
            batch_id="batch",
            internal_doc_id=internal_doc_id,
            generation="M003",
            status="success",
            references=m003_references,
            response_path=Path(existing["response_path"]),
            response_sha256=existing["response_sha256"],
            input_fingerprint=existing["input_fingerprint"],
            model_fingerprint=existing["model_fingerprint"],
            error=None,
            operational={},
        )

    # Blind order is otherwise a doc/model hash coin flip unrelated to what
    # this test checks; force the model candidate to land as "A" so the
    # fixture's FakeJudgeProvider (which always verdicts for "A") reliably
    # surfaces it.
    def forced_model_first(**kwargs: Any) -> tuple[str, list[dict[str, str]], list[dict[str, str]]]:
        return "A=model,B=human", kwargs["model_references"], kwargs["human_references"]

    monkeypatch.setattr(judges_module, "blind_candidates", forced_model_first)

    judge_step(config, batch_id="batch", generation="M003", fake_backend=True)(new_round(1))

    result_path = (
        config.sensitive_root
        / "batches"
        / "batch"
        / "judges"
        / "pilot"
        / "qwen3_5_397b"
        / f"{internal_doc_id}.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["blind_mapping"] == "A=model,B=human"
    final_references = result["result"]["final_references"]
    assert final_references, "expected the M003 reference to survive policy filtering"
    assert final_references[0]["kanun_no"] == "999"


def test_judge_step_fingerprints_the_summary_the_run_actually_wrote(tmp_path: Path) -> None:
    """Once a judge is locked for this batch, `run_judge_pilot` takes the
    locked-coverage branch and writes `judge_production_summary.json`
    instead of `judge_pilot_summary.json` -- `judge_step` must fingerprint
    whichever file the run actually produced, not assume the pilot path,
    or a round's `judged` artifact could certify work this run never did.
    """
    from data_quality_checker.judges import lock_judge
    from data_quality_checker.storage import Store

    config = prepared_fixture(tmp_path, two_docs=False)
    predict_step(config, batch_id="batch", generation="G0", fake_backend=True)(new_round(1))

    with Store(config.database_path) as store:
        internal_doc_id = store.list_documents("batch")[0]["internal_doc_id"]
        store.set_router_bucket("batch", internal_doc_id, "RED")

    # First pass: no judge is locked yet, so this runs (and seals) the pilot.
    judge_step(config, batch_id="batch", generation="G0", fake_backend=True)(new_round(1))

    with Store(config.database_path) as store:
        review = store.get_review("batch", internal_doc_id)
        assert review is not None
        store.update_review(
            batch_id="batch",
            internal_doc_id=internal_doc_id,
            expected_version=review["row_version"],
            status="finalized",
            action="accept_human",
            final_references=[],
            reason=None,
            reviewer="fixture",
        )

    lock_judge(
        config=config,
        batch_id="batch",
        model="qwen3.5:397b",
        reason="fixture metrics reviewed",
    )

    pilot_summary = config.public_root / "batches" / "batch" / "judge_pilot_summary.json"
    production_summary = config.public_root / "batches" / "batch" / "judge_production_summary.json"
    pilot_sha_before = sha256_file(pilot_summary)

    artifact = judge_step(config, batch_id="batch", generation="G0", fake_backend=True)(
        new_round(1)
    )

    assert production_summary.is_file()
    assert artifact == sha256_file(production_summary)
    assert artifact != pilot_sha_before
    assert artifact != sha256_file(pilot_summary)


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
    artifact = select_step(
        config, candidates=_candidates(), output_dir=out, validation_documents=50
    )(new_round(3))
    payload = json.loads((out / "round_003_checkpoint.json").read_text(encoding="utf-8"))
    assert payload["selected"]["update"] == 100
    assert payload["selection_basis"] == "validation"
    assert artifact == sha256_file(out / "round_003_checkpoint.json")


def test_select_step_records_only_validation_metrics(tmp_path: Path) -> None:
    config = prepared_fixture(tmp_path)
    out = tmp_path / "rounds"
    select_step(config, candidates=_candidates(), output_dir=out, validation_documents=50)(
        new_round(3)
    )
    raw = (out / "round_003_checkpoint.json").read_text(encoding="utf-8").lower()
    assert "test" not in raw


def _candidate_at(
    *, update: int, coverage_count: int, parse_count: int, core_f1: float = 0.75
) -> CheckpointCandidate:
    return CheckpointCandidate(
        update=update,
        coverage_count=coverage_count,
        parse_count=parse_count,
        empty_output_count=0,
        runaway_output_count=0,
        core_f1=core_f1,
        docwise_accuracy=0.30,
        recall=0.70,
        validation_loss=0.06,
    )


def test_select_step_derives_the_parse_threshold_and_rejects_below_it(tmp_path: Path) -> None:
    """FIX: at validation_documents=120, a literal minimum_parse_count=49 let a
    checkpoint that parsed only 49/120 (41%) documents pass the parse-quality
    gate. Deriving the threshold as validation_documents - 1 (119) rejects it.
    """
    from data_quality_checker.errors import GateBlocked

    config = prepared_fixture(tmp_path)
    out = tmp_path / "rounds"
    candidate = _candidate_at(update=50, coverage_count=120, parse_count=49)
    with pytest.raises(GateBlocked):
        select_step(config, candidates=[candidate], output_dir=out, validation_documents=120)(
            new_round(1)
        )


def test_select_step_derives_the_parse_threshold_and_accepts_near_full_coverage(
    tmp_path: Path,
) -> None:
    config = prepared_fixture(tmp_path)
    out = tmp_path / "rounds"
    candidate = _candidate_at(update=50, coverage_count=120, parse_count=119)
    artifact = select_step(
        config, candidates=[candidate], output_dir=out, validation_documents=120
    )(new_round(1))
    payload = json.loads((out / "round_001_checkpoint.json").read_text(encoding="utf-8"))
    assert payload["selected"]["update"] == 50
    assert artifact == sha256_file(out / "round_001_checkpoint.json")


def test_select_step_honours_an_explicit_minimum_parse_count_over_the_derivation(
    tmp_path: Path,
) -> None:
    """An explicit `minimum_parse_count` still overrides the validation_documents - 1
    derivation -- here a candidate with parse_count=49 at validation_documents=120
    (which the derivation alone would reject, per the test above) is accepted
    because the caller deliberately stated a looser tolerance.
    """
    config = prepared_fixture(tmp_path)
    out = tmp_path / "rounds"
    candidate = _candidate_at(update=50, coverage_count=120, parse_count=49)
    artifact = select_step(
        config,
        candidates=[candidate],
        output_dir=out,
        validation_documents=120,
        minimum_parse_count=49,
    )(new_round(1))
    payload = json.loads((out / "round_001_checkpoint.json").read_text(encoding="utf-8"))
    assert payload["selected"]["update"] == 50
    assert artifact == sha256_file(out / "round_001_checkpoint.json")


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
    document count to match `expected_doc_count` exactly, so this test drives
    it with the project's real frozen "test" split (50 documents) and states
    that count explicitly -- `measure_step` takes no default, since a round's
    split size is a fact its caller must know and state.
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
        expected_doc_count=50,
    )(new_round(5))

    report_path = out / "evaluation.json"
    assert artifact == sha256_file(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    results = report["results"]
    assert len(results) > 0
    assert results[0]["evaluated_doc_count"] == 50


def _canonical_predictions(config, doc_ids: list[int]) -> list[dict[str, Any]]:
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
    return predictions


def test_measure_step_succeeds_on_a_split_that_is_not_fifty_documents(tmp_path: Path) -> None:
    """The coverage gate must discriminate on whatever count the round states,
    not just the historical G0 validation size of 50 -- a DQ-Loop round's
    frozen test split is 150 documents, so 50 can never be assumed.

    Uses 40 real documents drawn from the reference manifest's "train" split
    (which has 394 entries, so a count other than 50 is available) with real
    ground truth, rather than mocking the evaluator.
    """
    config = prepared_fixture(tmp_path)
    split = json.loads(config.reference_split_manifest_path.read_text(encoding="utf-8"))
    doc_ids = split["splits"]["train"][:40]
    assert len(doc_ids) == 40

    predictions = _canonical_predictions(config, doc_ids)

    predictions_path = tmp_path / "predictions_40.json"
    doc_ids_path = tmp_path / "test_doc_ids_40.json"
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    doc_ids_path.write_text(json.dumps(doc_ids), encoding="utf-8")

    out = tmp_path / "measurement_40"
    artifact = measure_step(
        config,
        predictions_path=predictions_path,
        test_doc_ids_path=doc_ids_path,
        output_dir=out,
        expected_doc_count=40,
    )(new_round(6))

    report_path = out / "evaluation.json"
    assert artifact == sha256_file(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # Assert the count the evaluator actually reports, not merely that a
    # report file appeared: evaluate.py::load_predictions silently drops any
    # row whose doc_id is not an int, which would otherwise let a wrong-typed
    # doc_id pass as an empty, vacuously "successful" report.
    assert report["results"][0]["evaluated_doc_count"] == 40


def test_measure_step_raises_when_the_prediction_count_does_not_match(tmp_path: Path) -> None:
    config = prepared_fixture(tmp_path)
    split = json.loads(config.reference_split_manifest_path.read_text(encoding="utf-8"))
    doc_ids = split["splits"]["test"]
    assert len(doc_ids) == 50

    predictions = _canonical_predictions(config, doc_ids)

    predictions_path = tmp_path / "predictions.json"
    doc_ids_path = tmp_path / "test_doc_ids.json"
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    doc_ids_path.write_text(json.dumps(doc_ids), encoding="utf-8")

    out = tmp_path / "measurement_mismatch"
    with pytest.raises(
        IntegrityError,
        match="expected 42 evaluated documents, got 50",
    ):
        measure_step(
            config,
            predictions_path=predictions_path,
            test_doc_ids_path=doc_ids_path,
            output_dir=out,
            expected_doc_count=42,
        )(new_round(7))


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
