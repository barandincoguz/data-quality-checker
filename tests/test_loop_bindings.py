from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_loop_train import canonical_fixture, release_fixture
from test_processing import prepared_fixture

from data_quality_checker.constants import JUDGE_MODEL_KEY
from data_quality_checker.errors import ContractError, IntegrityError
from data_quality_checker.fingerprints import sha256_file
from data_quality_checker.g0 import CheckpointCandidate
from data_quality_checker.loop_bindings import (
    compose_step,
    judge_step,
    measure_step,
    metrics_step,
    predict_step,
    route_step,
    seal_step,
    select_step,
)
from data_quality_checker.loop_rounds import advance_round, new_round
from data_quality_checker.loop_train import train_round
from data_quality_checker.storage import Store


def canonical_and_release_fixture(tmp_path: Path) -> Any:
    """A canonical config with round "round1" already released and cleaned,
    for exercising `train_step` without a real G0 training run."""
    config = canonical_fixture(tmp_path)
    release_fixture(config, "round1", ["d1", "d2"])
    return config


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
        / "gemma-4-31b-it-optiq-4bit"
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
        model="gemma-4-31b-it-optiq-4bit",
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


def test_train_step_returns_the_sha_of_its_data_manifest(tmp_path: Path) -> None:
    from data_quality_checker.loop_bindings import train_step

    config = canonical_and_release_fixture(tmp_path)
    split = {"train": [1, 2, 3], "valid": [10, 11], "test": [20, 21]}
    step = train_step(config, generation="M001", split=split, cleaned_batch_ids=["round1"])
    artifact = step(new_round(1))

    result = train_round(config, generation="M001", split=split, cleaned_batch_ids=["round1"])
    manifest = Path(result["data_dir"]) / "split_manifest.json"
    assert artifact == sha256_file(manifest)


def test_train_step_requires_the_caller_to_state_every_fact() -> None:
    """Every default this line of work guessed for `generation`, `split` or
    `cleaned_batch_ids` produced a defect -- a round predicting with the
    wrong model, a judge reading the wrong generation, an evaluator gating
    the wrong split size. The caller knows all three; none may default."""
    import inspect

    from data_quality_checker.loop_bindings import train_step

    signature = inspect.signature(train_step)
    for name in ("generation", "split", "cleaned_batch_ids"):
        assert signature.parameters[name].default is inspect.Parameter.empty


# ---------------------------------------------------------------------------
# metrics_step (Task 5): assemble the round's audited metrics artifact
# ---------------------------------------------------------------------------

_VUK_REFERENCE = {
    "kanun_no": "213",
    "kanun_ad": "Vergi Usul Kanunu",
    "madde": "413",
    "fikra": "",
    "bent": "",
    "source_text": "213 sayılı Vergi Usul Kanununun 413. maddesi",
}


def _metrics_batch_fixture(config: Any) -> tuple[str, str]:
    """Wire one finalized document and one deferred document directly through
    the store, bypassing predict/route/judge -- `metrics_step` only reads what
    those stages would have written, so this exercises that contract
    directly instead of re-running the whole pipeline.

    Returns ``(finalized_internal_doc_id, deferred_internal_doc_id)``.
    """
    with Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store:
        documents = store.list_documents("batch")
        finalized_doc, deferred_doc = documents[0], documents[1]
        finalized_id = finalized_doc["internal_doc_id"]
        deferred_id = deferred_doc["internal_doc_id"]

        # `finalized_id`'s bucket (RED) makes it required review on its own;
        # `deferred_id` is the batch's only GREEN document, so
        # `ensure_green_audit_plan` samples it deterministically (a
        # 1-document GREEN pool always yields a 1-document sample) --
        # exercising `in_green_audit_sample` without faking the sampler.
        store.set_router_bucket("batch", finalized_id, "RED")
        store.set_router_bucket("batch", deferred_id, "GREEN")

        for internal_doc_id, references in (
            (finalized_id, [_VUK_REFERENCE]),
            (deferred_id, []),
        ):
            store.persist_prediction(
                batch_id="batch",
                internal_doc_id=internal_doc_id,
                generation="M001",
                status="success",
                references=references,
                response_path=Path("/dev/null"),
                response_sha256="0" * 64,
                input_fingerprint="fp",
                model_fingerprint="fp",
                error=None,
                operational={"latency_seconds": 1.0, "input_tokens": 10, "output_tokens": 5},
            )

        store.persist_judge_result(
            batch_id="batch",
            internal_doc_id=finalized_id,
            model=JUDGE_MODEL_KEY,
            blind_mapping="A=model,B=human",
            status="valid",
            verdict="A",
            result={"final_references": [_VUK_REFERENCE]},
            response_path=None,
            response_sha256=None,
            retry_count=0,
            error=None,
        )
        # No judge result at all for `deferred_id` -- the judge is a rater
        # with nothing to say for it, not one that reported an empty list.

        finalized_review = store.get_review("batch", finalized_id)
        assert finalized_review is not None
        store.update_review(
            batch_id="batch",
            internal_doc_id=finalized_id,
            expected_version=finalized_review["row_version"],
            status="finalized",
            action="accept_model",
            final_references=[_VUK_REFERENCE],
            reason=None,
            reviewer="fixture",
        )

        deferred_review = store.get_review("batch", deferred_id)
        assert deferred_review is not None
        store.update_review(
            batch_id="batch",
            internal_doc_id=deferred_id,
            expected_version=deferred_review["row_version"],
            status="deferred",
            action="defer",
            final_references=None,
            reason="fixture defer",
            reviewer="fixture",
        )

    escalation_path = config.public_root / "batches" / "batch" / "green_escalation.json"
    escalation_path.parent.mkdir(parents=True, exist_ok=True)
    escalation_path.write_text("{}", encoding="utf-8")

    return finalized_id, deferred_id


# Real, approved-only ground truth from the reference split (mirrors
# `test_loop_metrics.py`'s own `_frozen_test_fixture`, trimmed to two
# documents so the real `canonical_evaluate` subprocess this exercises stays
# fast): doc 5 the model always gets right, doc 20 it either misses
# (`miss_status="error"`) or also gets right (`"success"`) -- giving
# `paired_delta` a genuine, non-fabricated improvement to detect across two
# rounds sharing the same document universe.
_PERFECT_DOC_ID = 5
_MISS_DOC_ID = 20


def _frozen_split_fixture(
    config: Any, tmp_path: Path, *, miss_status: str = "error"
) -> tuple[Path, Path]:
    predictions = []
    for doc_id, status in ((_PERFECT_DOC_ID, "success"), (_MISS_DOC_ID, miss_status)):
        gold = json.loads(
            (config.canonical_gt_dir / f"doc_{doc_id}.json").read_text(encoding="utf-8")
        )
        approved = [
            reference
            for reference in gold["references"]
            if str(reference.get("status", "approved")).lower() == "approved"
        ]
        predictions.append(
            {
                "doc_id": doc_id,
                "status": status,
                "references": approved if status == "success" else [],
                "operational": {
                    "latency_seconds": 1.0,
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "peak_memory_bytes": 1_000_000,
                    "truncated": False,
                },
            }
        )
    predictions_path = tmp_path / f"predictions_{miss_status}.json"
    doc_ids_path = tmp_path / "doc_ids.json"
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    doc_ids_path.write_text(json.dumps([_PERFECT_DOC_ID, _MISS_DOC_ID]), encoding="utf-8")
    return predictions_path, doc_ids_path


def test_metrics_step_has_no_default_for_expected_doc_count() -> None:
    """`canonical_evaluate` defaults `expected_doc_count` to 50 -- the wrong
    number for the loop's 150-document frozen test split. A default here
    would silently reintroduce the exact defect `measure_step` already
    guards against."""
    import inspect

    signature = inspect.signature(metrics_step)
    assert signature.parameters["expected_doc_count"].default is inspect.Parameter.empty


def test_metrics_step_writes_round_metrics_and_returns_its_sha(tmp_path: Path) -> None:
    config = prepared_fixture(tmp_path, two_docs=True)
    _metrics_batch_fixture(config)
    predictions_path, doc_ids_path = _frozen_split_fixture(config, tmp_path)

    out = tmp_path / "rounds"
    artifact = metrics_step(
        config,
        batch_id="batch",
        generation="M001",
        predictions_path=predictions_path,
        test_doc_ids_path=doc_ids_path,
        expected_doc_count=2,
        output_dir=out,
    )(new_round(1))

    path = out / "round_001_metrics.json"
    assert artifact == sha256_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["round"] == 1
    assert payload["batch_id"] == "batch"
    assert payload["generation"] == "M001"

    # Round 1 has no predecessor: paired_delta is null with a stated reason,
    # never a fabricated baseline.
    assert payload["paired_delta"] is None
    assert payload["paired_delta_reason"] == "round 1 has no predecessor round"

    # The deferred document counts as workload but never as truth.
    workload = payload["expert_workload"]
    assert workload["document_count"] == 2
    assert workload["deferred_count"] == 1
    assert workload["action_distribution"]["defer"] == 1
    assert workload["action_distribution"]["accept_model"] == 1
    # Required regardless of router bucket: the finalized document via RED,
    # the deferred one via the GREEN audit sample (its escalation file also
    # exists, so either signal alone would already make it required).
    assert workload["documents_requiring_review_count"] == 2

    agreement = payload["rater_agreement"]
    assert agreement["document_count"] == 1  # only the finalized document
    per_rater = agreement["per_rater"]
    assert per_rater["model"]["available_document_count"] == 1
    assert per_rater["judge"]["available_document_count"] == 1
    assert per_rater["model"]["f1"] == 1.0
    assert per_rater["judge"]["f1"] == 1.0

    model_metrics = payload["model"]
    assert model_metrics["document_count"] == 2
    assert set(model_metrics["core_per_document"]) == {"5", "20"}


def test_metrics_step_computes_paired_delta_against_the_previous_round(tmp_path: Path) -> None:
    config = prepared_fixture(tmp_path, two_docs=True)
    _metrics_batch_fixture(config)
    out = tmp_path / "rounds"

    round1_predictions, doc_ids_path = _frozen_split_fixture(config, tmp_path, miss_status="error")
    metrics_step(
        config,
        batch_id="batch",
        generation="M001",
        predictions_path=round1_predictions,
        test_doc_ids_path=doc_ids_path,
        expected_doc_count=2,
        output_dir=out,
    )(new_round(1))

    round2_predictions, _ = _frozen_split_fixture(config, tmp_path, miss_status="success")
    artifact2 = metrics_step(
        config,
        batch_id="batch",
        generation="M001",
        predictions_path=round2_predictions,
        test_doc_ids_path=doc_ids_path,
        expected_doc_count=2,
        output_dir=out,
    )(new_round(2))

    path2 = out / "round_002_metrics.json"
    assert artifact2 == sha256_file(path2)
    payload2 = json.loads(path2.read_text(encoding="utf-8"))

    assert payload2["paired_delta_reason"] is None
    paired = payload2["paired_delta"]
    assert paired is not None
    assert paired["document_count"] == 2
    # Round 2 strictly improves on the miss document and holds the perfect
    # one steady, so the observed mean delta must be positive -- whether the
    # bootstrap interval at n=2 calls that "improved" or "inconclusive" is a
    # separate, legitimate small-sample question this assertion need not
    # settle.
    assert paired["bootstrap_mean"] > 0
    assert paired["verdict"] in {"improved", "inconclusive"}


def test_metrics_step_round_two_reports_no_predecessor_when_the_file_is_missing(
    tmp_path: Path,
) -> None:
    """A round whose predecessor never wrote `round_metrics.json` (skipped,
    or simply absent) must not fabricate a baseline either -- only round 1
    is exempt from stating a reason by construction."""
    config = prepared_fixture(tmp_path, two_docs=True)
    _metrics_batch_fixture(config)
    predictions_path, doc_ids_path = _frozen_split_fixture(config, tmp_path)
    out = tmp_path / "rounds"

    artifact = metrics_step(
        config,
        batch_id="batch",
        generation="M001",
        predictions_path=predictions_path,
        test_doc_ids_path=doc_ids_path,
        expected_doc_count=2,
        output_dir=out,
    )(new_round(2))

    payload = json.loads((out / "round_002_metrics.json").read_text(encoding="utf-8"))
    assert payload["paired_delta"] is None
    assert "round_001_metrics.json" in payload["paired_delta_reason"]
    assert artifact == sha256_file(out / "round_002_metrics.json")


def test_seal_step_seal_changes_when_the_round_metrics_artifact_changes(tmp_path: Path) -> None:
    """The required proof for Task 5: mutate one value inside
    `round_metrics.json` and the round's seal must change. A seal that does
    not cover the metrics artifact -- because the step bound at `measured`
    never wrote it, or because `seal_step` stopped serialising it -- would
    certify nothing.
    """
    from data_quality_checker.loop_rounds import ROUND_STAGES, RoundState

    config = prepared_fixture(tmp_path, two_docs=True)
    _metrics_batch_fixture(config)
    predictions_path, doc_ids_path = _frozen_split_fixture(config, tmp_path)

    rounds_dir = tmp_path / "rounds"
    metrics_artifact = metrics_step(
        config,
        batch_id="batch",
        generation="M001",
        predictions_path=predictions_path,
        test_doc_ids_path=doc_ids_path,
        expected_doc_count=2,
        output_dir=rounds_dir,
    )(new_round(1))

    metrics_path = rounds_dir / "round_001_metrics.json"
    assert metrics_artifact == sha256_file(metrics_path)

    state = new_round(1)
    for stage in ROUND_STAGES[1 : ROUND_STAGES.index("measured")]:
        state = advance_round(state, to=stage, artifact_sha256="sha")
    state = advance_round(state, to="measured", artifact_sha256=metrics_artifact)

    seal_before = seal_step(config, output_dir=rounds_dir)(state)

    # Mutate one value inside round_metrics.json -- nothing about its shape,
    # just one number -- and recompute the sha `measured` would have carried
    # had this been the artifact all along.
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["expert_workload"]["document_count"] += 1
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    mutated_artifact = sha256_file(metrics_path)
    assert mutated_artifact != metrics_artifact

    mutated_state = RoundState(
        round_index=1,
        stage="measured",
        artifacts={**state.artifacts, "measured": mutated_artifact},
    )
    seal_after = seal_step(config, output_dir=rounds_dir)(mutated_state)

    assert seal_before != seal_after
