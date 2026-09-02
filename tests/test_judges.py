from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_quality_checker.commands import pilot_judges
from data_quality_checker.config import default_config_path, load_config
from data_quality_checker.errors import ContractError, GateBlocked
from data_quality_checker.judges import (
    FakeJudgeProvider,
    JudgeProviderUnavailable,
    ensure_green_audit_plan,
    lock_judge,
    run_judge_pilot,
    select_pilot_documents,
)
from data_quality_checker.preparation import prepare_batch
from data_quality_checker.processing import process_batch
from data_quality_checker.storage import Store


def config_for(tmp_path: Path):
    live = load_config()
    payload = json.loads(default_config_path().read_text(encoding="utf-8"))
    payload.update(
        {
            "canonical_gt_dir": str(live.canonical_gt_dir),
            "example_bank_path": str(live.example_bank_path),
            "reference_split_manifest_path": str(live.reference_split_manifest_path),
            "sensitive_root": str(tmp_path / "sensitive"),
            "public_root": str(tmp_path / "public"),
            "training_runs_root": str(tmp_path / "runs"),
        }
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_config(path)


def prepared_processed_fixture(tmp_path: Path, *, count: int = 1):
    config = config_for(tmp_path)
    annotations = [
        {"document_id": f"private-id-{index}", "current_references": []} for index in range(count)
    ]
    pool = [
        {"evrakOid": f"private-id-{index}", "pdfText": f"Güvenli belge metni {index}."}
        for index in range(count)
    ]
    annotation_zip, pool_zip = tmp_path / "a.zip", tmp_path / "p.zip"
    for path, payload in (
        (annotation_zip, {"annotations": annotations}),
        (pool_zip, pool),
    ):
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("data.json", json.dumps(payload, ensure_ascii=False))
    key = tmp_path / "key"
    key.write_bytes(b"j" * 32)
    prepare_batch(
        config=config,
        annotation_zip=annotation_zip,
        document_pool_zip=pool_zip,
        batch_id="batch",
        hmac_key_file=key,
    )
    process_batch(
        config=config,
        batch_id="batch",
        generation="G0",
        resume=False,
        fake_backend=True,
    )
    return config


def test_pilot_selection_transfers_missing_quota_deterministically() -> None:
    documents = (
        [{"internal_doc_id": f"g{i}", "router_bucket": "GREEN"} for i in range(5)]
        + [{"internal_doc_id": f"y{i}", "router_bucket": "YELLOW"} for i in range(30)]
        + [{"internal_doc_id": f"r{i}", "router_bucket": "RED"} for i in range(40)]
    )
    first = select_pilot_documents(documents, batch_id="batch")
    second = select_pilot_documents(list(reversed(documents)), batch_id="batch")
    assert first == second
    assert len(first.internal_doc_ids) == 60
    assert first.counts["GREEN"] == 5
    assert first.counts["YELLOW"] + first.counts["RED"] == 55


def test_green_audit_uses_minimum_thirty_capped_by_available(tmp_path) -> None:
    config = prepared_processed_fixture(tmp_path, count=3)
    with Store(config.database_path) as store:
        plan = ensure_green_audit_plan(config=config, store=store, batch_id="batch")
    assert plan["green_count"] == 3
    assert plan["requested_sample_size"] == 30
    assert plan["sample_size"] == 3


def test_external_consent_gate_fires_before_provider_call(tmp_path) -> None:
    config = prepared_processed_fixture(tmp_path)
    provider = FakeJudgeProvider()
    with pytest.raises(GateBlocked, match="allow-external"):
        run_judge_pilot(
            config=config,
            batch_id="batch",
            allow_external_judge=False,
            provider=provider,
        )
    assert provider.payloads == []


def test_blind_pilot_sends_only_text_and_candidates_and_persists_both_models(tmp_path) -> None:
    config = prepared_processed_fixture(tmp_path)
    provider = FakeJudgeProvider()
    summary = run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        provider=provider,
    )
    assert summary["selected_document_count"] == 1
    assert len(provider.payloads) == 2
    for payload in provider.payloads:
        assert set(payload) == {
            "document_text",
            "candidate_a",
            "candidate_b",
            "instructions",
        }
        assert "private-id" not in json.dumps(payload)
    with Store(config.database_path) as store:
        results = store.list_judge_results("batch")
    assert len(results) == 2
    assert {row["status"] for row in results} == {"valid"}


def test_run_judge_pilot_resolves_each_model_once_via_the_registry(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from data_quality_checker import judges as judges_module

    config = prepared_processed_fixture(tmp_path, count=3)
    calls: list[str] = []

    def recording_resolve(model: str, *, fake_backend: bool = False) -> FakeJudgeProvider:
        calls.append(model)
        return FakeJudgeProvider()

    monkeypatch.setattr(judges_module, "resolve_judge_provider", recording_resolve)

    summary = run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        fake_backend=True,
    )

    assert set(calls) == set(judges_module.JUDGE_MODELS)
    # One resolution per model, not once per document: with three documents
    # and two models a per-document rebuild would show six calls here.
    assert len(calls) == len(judges_module.JUDGE_MODELS)
    assert summary["selected_document_count"] == 3


class RetryThenValid:
    def __init__(self):
        self.calls = 0

    def judge(self, *, model, payload):
        self.calls += 1
        if self.calls % 3 != 0:
            return "not-json", {"latency_seconds": 0.1, "cost": 0.0}
        return (
            {
                "verdict": "TIE",
                "candidate_errors": {"A": [], "B": []},
                "final_references": [],
                "evidence": [],
                "reason_codes": ["same"],
            },
            {"latency_seconds": 0.1, "cost": 0.0},
        )


def test_malformed_output_retries_up_to_three_and_records_retry_count(tmp_path) -> None:
    config = prepared_processed_fixture(tmp_path)
    provider = RetryThenValid()
    run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        provider=provider,
    )
    assert provider.calls == 6
    with Store(config.database_path) as store:
        assert {row["retry_count"] for row in store.list_judge_results("batch")} == {2}


class AlwaysUnavailable:
    def judge(self, *, model, payload):
        raise JudgeProviderUnavailable("paywall")


def test_unavailable_model_is_recorded_without_fallback(tmp_path) -> None:
    config = prepared_processed_fixture(tmp_path)
    summary = run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        provider=AlwaysUnavailable(),
    )
    assert all(model["unavailable"] == 1 for model in summary["models"].values())
    with pytest.raises(GateBlocked):
        lock_judge(config=config, batch_id="batch", model="qwen3.5:397b", reason="test")


def test_second_run_after_explicit_lock_executes_production_coverage(tmp_path) -> None:
    config = prepared_processed_fixture(tmp_path)
    with Store(config.database_path) as store:
        document = store.list_documents("batch")[0]
        store.set_router_bucket("batch", document["internal_doc_id"], "RED")

    run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        provider=FakeJudgeProvider(),
    )
    with Store(config.database_path) as store:
        review = store.get_review("batch", document["internal_doc_id"])
        assert review is not None
        store.update_review(
            batch_id="batch",
            internal_doc_id=document["internal_doc_id"],
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
    production = run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        provider=FakeJudgeProvider(),
    )
    assert production["stage"] == "production"
    assert production["locked_model"] == "qwen3.5:397b"
    assert production["required_document_count"] == 1
    assert production["coverage_complete"] is True


def test_pilot_honours_a_model_set_override(tmp_path) -> None:
    from data_quality_checker.judges import gemini_judge_model

    config = prepared_processed_fixture(tmp_path)
    summary = run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        provider=FakeJudgeProvider(),
        judge_models=("qwen3.5:397b", gemini_judge_model()),
    )
    assert set(summary["models"]) == {"qwen3.5:397b", gemini_judge_model()}


def test_pilot_rejects_an_unregistered_model_in_the_override(tmp_path) -> None:
    config = prepared_processed_fixture(tmp_path)
    with pytest.raises(ContractError):
        run_judge_pilot(
            config=config,
            batch_id="batch",
            allow_external_judge=True,
            provider=FakeJudgeProvider(),
            judge_models=("made-up-model",),
        )


def test_lock_judge_accepts_a_gemini_model_that_ran_in_the_pilot(tmp_path) -> None:
    from data_quality_checker.judges import gemini_judge_model

    config = prepared_processed_fixture(tmp_path)
    with Store(config.database_path) as store:
        document = store.list_documents("batch")[0]
        store.set_router_bucket("batch", document["internal_doc_id"], "RED")

    run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        provider=FakeJudgeProvider(),
        judge_models=("qwen3.5:397b", gemini_judge_model()),
    )
    with Store(config.database_path) as store:
        review = store.get_review("batch", document["internal_doc_id"])
        assert review is not None
        store.update_review(
            batch_id="batch",
            internal_doc_id=document["internal_doc_id"],
            expected_version=review["row_version"],
            status="finalized",
            action="accept_human",
            final_references=[],
            reason=None,
            reviewer="fixture",
        )

    payload = lock_judge(
        config=config,
        batch_id="batch",
        model=gemini_judge_model(),
        reason="dq-loop production judge",
    )
    assert payload["model"] == gemini_judge_model()
    assert set(payload["pilot_model_metrics"]) == {"qwen3.5:397b", gemini_judge_model()}
    written = json.loads(
        (config.public_root / "batches" / "batch" / "judge_lock.json").read_text(encoding="utf-8")
    )
    assert written["model"] == gemini_judge_model()


def test_lock_judge_rejects_an_unregistered_model(tmp_path) -> None:
    config = prepared_processed_fixture(tmp_path)
    with pytest.raises(ContractError):
        lock_judge(config=config, batch_id="batch", model="made-up-model", reason="test")


def test_cli_rejects_an_explicitly_supplied_empty_judge_models_flag() -> None:
    # "" is falsy, unlike ",, " (already rejected), so a truthiness check
    # would silently fall back to the default pair instead of raising.
    args = SimpleNamespace(
        batch_id="batch",
        allow_external_judge=False,
        fake_backend=False,
        judge_models="",
    )
    with pytest.raises(ContractError):
        pilot_judges(args, config=None)
