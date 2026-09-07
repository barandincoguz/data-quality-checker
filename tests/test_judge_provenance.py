"""A judge verdict records which predictions it judged.

`judge_results` is keyed by (batch_id, internal_doc_id, model) -- there is no
generation in the key, and the stored payload named none either. A verdict
therefore could not be traced back to the model output it was given, and a
re-run against a different generation would silently reuse or overwrite the
earlier one. M01 already holds 27 verdicts produced against G0's predictions,
so this is not hypothetical.

The key is left alone: within the loop a batch is judged against exactly one
generation, so the collision only arises from the recorded G0-predicts-M01
design violation. What must not survive is the untraceability -- a verdict
whose subject is unknown cannot be defended in a paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_quality_checker.judges import judged_prediction_provenance


def _prediction(generation: str = "M000") -> dict[str, object]:
    return {
        "generation": generation,
        "model_fingerprint": "f" * 64,
        "input_fingerprint": "a" * 64,
        "response_sha256": "b" * 64,
        "status": "success",
    }


def test_provenance_names_the_generation_and_what_produced_it() -> None:
    provenance = judged_prediction_provenance(_prediction())
    assert provenance["generation"] == "M000"
    assert provenance["model_fingerprint"] == "f" * 64
    assert provenance["prediction_sha256"] == "b" * 64
    assert provenance["input_fingerprint"] == "a" * 64


def test_provenance_distinguishes_two_generations_of_the_same_document() -> None:
    """The exact confusion this exists to prevent."""
    assert judged_prediction_provenance(_prediction("G0")) != judged_prediction_provenance(
        _prediction("M000")
    )


def test_provenance_survives_a_json_round_trip() -> None:
    provenance = judged_prediction_provenance(_prediction())
    assert json.loads(json.dumps(provenance)) == provenance


@pytest.mark.parametrize("missing", ["generation", "model_fingerprint", "response_sha256"])
def test_a_prediction_that_cannot_be_identified_is_refused(missing: str) -> None:
    """Better to fail here than to write a verdict nobody can trace."""
    from data_quality_checker.errors import IntegrityError

    prediction = _prediction()
    del prediction[missing]
    with pytest.raises(IntegrityError):
        judged_prediction_provenance(prediction)


def test_a_verdict_from_another_generation_is_refused_on_resume(tmp_path) -> None:
    """The load-bearing guard.

    `judge_results` is keyed without a generation, so a production run that
    resumes would hand back a verdict formed against a different model's
    output and count it as covered. M01 is the live case: 27 verdicts formed
    against G0's predictions, and the batch is now judged against M000's.
    """
    import json as json_module

    from test_judges import FakeJudgeProvider, prepared_processed_fixture

    from data_quality_checker.errors import IntegrityError
    from data_quality_checker.judges import lock_judge, run_judge_pilot
    from data_quality_checker.storage import Store

    config = prepared_processed_fixture(tmp_path)
    with Store(config.database_path) as store:
        document = store.list_documents("batch")[0]
        doc_id = document["internal_doc_id"]
        store.set_router_bucket("batch", doc_id, "RED")

    run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        provider=FakeJudgeProvider(),
        judge_models=("qwen3.5:397b",),
    )
    with Store(config.database_path) as store:
        review = store.get_review("batch", doc_id)
        store.update_review(
            batch_id="batch",
            internal_doc_id=doc_id,
            expected_version=review["row_version"],
            status="finalized",
            action="accept_human",
            final_references=[],
            reason=None,
            reviewer="fixture",
        )
    lock_judge(config=config, batch_id="batch", model="qwen3.5:397b", reason="fixture")
    run_judge_pilot(
        config=config, batch_id="batch", allow_external_judge=True, provider=FakeJudgeProvider()
    )

    # Re-running is fine while the provenance still matches.
    run_judge_pilot(
        config=config, batch_id="batch", allow_external_judge=True, provider=FakeJudgeProvider()
    )

    # Now make the stored verdict claim it judged a different generation's
    # output. The fixture judges G0, so M01's live situation is the mirror of
    # this: verdicts formed against G0 while the batch is judged against M000.
    with Store(config.database_path) as store:
        stored = store.get_judge_result("batch", doc_id, "qwen3.5:397b")
    path = Path(stored["response_path"])
    payload = json_module.loads(path.read_text(encoding="utf-8"))
    payload["judged_prediction"]["generation"] = "M000"
    path.write_text(json_module.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with Store(config.database_path) as store:
        store.persist_judge_result(
            batch_id="batch",
            internal_doc_id=doc_id,
            model="qwen3.5:397b",
            blind_mapping=stored["blind_mapping"],
            status="valid",
            verdict=stored["verdict"],
            result=json_module.loads(stored["result_json"]),
            response_path=path,
            response_sha256=__import__(
                "data_quality_checker.fingerprints", fromlist=["sha256_file"]
            ).sha256_file(path),
            retry_count=0,
            error=None,
        )

    with pytest.raises(IntegrityError) as excinfo:
        run_judge_pilot(
            config=config,
            batch_id="batch",
            allow_external_judge=True,
            provider=FakeJudgeProvider(),
        )
    assert "M000" in str(excinfo.value)


def test_a_verdict_from_another_instrument_is_refused_on_resume(tmp_path) -> None:
    """Locked coverage over a batch must be one instrument.

    A judge verdict names the prompt sha and candidate-payload contract it
    was formed under (`judge_instrument`), but the store's key -- (batch_id,
    internal_doc_id, model) -- cannot tell two instruments apart. M01 is the
    live case: its 57 stored verdicts were formed with the annotator's
    `source_text` on every candidate, a blinding-contract violation Task 3
    fixed. Without this guard, a resumed production run over M01 would count
    those 57 verdicts as covered by the current, corrected instrument and
    never re-judge them.
    """
    import json as json_module

    from test_judges import FakeJudgeProvider, prepared_processed_fixture

    from data_quality_checker.errors import IntegrityError
    from data_quality_checker.judges import lock_judge, run_judge_pilot
    from data_quality_checker.storage import Store

    config = prepared_processed_fixture(tmp_path)
    with Store(config.database_path) as store:
        document = store.list_documents("batch")[0]
        doc_id = document["internal_doc_id"]
        store.set_router_bucket("batch", doc_id, "RED")

    run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        provider=FakeJudgeProvider(),
        judge_models=("qwen3.5:397b",),
    )
    with Store(config.database_path) as store:
        review = store.get_review("batch", doc_id)
        store.update_review(
            batch_id="batch",
            internal_doc_id=doc_id,
            expected_version=review["row_version"],
            status="finalized",
            action="accept_human",
            final_references=[],
            reason=None,
            reviewer="fixture",
        )
    lock_judge(config=config, batch_id="batch", model="qwen3.5:397b", reason="fixture")
    run_judge_pilot(
        config=config, batch_id="batch", allow_external_judge=True, provider=FakeJudgeProvider()
    )

    # Re-running is fine while the stored instrument still matches.
    run_judge_pilot(
        config=config, batch_id="batch", allow_external_judge=True, provider=FakeJudgeProvider()
    )

    # Now make the stored verdict claim it was formed by a different
    # instrument -- the mirror of M01's live situation, whose 57 verdicts
    # were formed with source_text included on every candidate.
    with Store(config.database_path) as store:
        stored = store.get_judge_result("batch", doc_id, "qwen3.5:397b")
    path = Path(stored["response_path"])
    payload = json_module.loads(path.read_text(encoding="utf-8"))
    payload["judge_instrument"] = {
        "prompt_sha256": "0" * 64,
        "candidate_payload_contract": "spans_included_v0",
    }
    path.write_text(json_module.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with Store(config.database_path) as store:
        store.persist_judge_result(
            batch_id="batch",
            internal_doc_id=doc_id,
            model="qwen3.5:397b",
            blind_mapping=stored["blind_mapping"],
            status="valid",
            verdict=stored["verdict"],
            result=json_module.loads(stored["result_json"]),
            response_path=path,
            response_sha256=__import__(
                "data_quality_checker.fingerprints", fromlist=["sha256_file"]
            ).sha256_file(path),
            retry_count=0,
            error=None,
        )

    with pytest.raises(IntegrityError) as excinfo:
        run_judge_pilot(
            config=config,
            batch_id="batch",
            allow_external_judge=True,
            provider=FakeJudgeProvider(),
        )
    assert "different judge instrument" in str(excinfo.value)



def test_a_verdict_written_before_provenance_existed_is_refused(tmp_path) -> None:
    """M01's 27 stored verdicts name no prediction at all. They must not be
    trusted silently just because the key matches.
    """
    from data_quality_checker.judges import judged_prediction_provenance

    legacy: dict[str, object] = {"schema_version": 1, "status": "valid"}
    assert legacy.get("judged_prediction") != judged_prediction_provenance(_prediction())
