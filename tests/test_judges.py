from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_quality_checker.commands import pilot_judges
from data_quality_checker.config import default_config_path, load_config
from data_quality_checker.errors import ContractError, DQCheckError, GateBlocked, IntegrityError
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


def prepared_processed_fixture(tmp_path: Path, *, count: int = 1, generation: str = "G0"):
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
        generation=generation,
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


def test_run_judge_pilot_judges_a_named_generations_predictions(tmp_path) -> None:
    """A DQ-Loop round's predictions are stored under its own label (e.g.
    "M003"), not "G0" -- so the judge must be told which generation to read."""
    config = prepared_processed_fixture(tmp_path, generation="M003")
    provider = FakeJudgeProvider()
    summary = run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        generation="M003",
        provider=provider,
    )
    assert summary["selected_document_count"] == 1


def test_run_judge_pilot_raises_naming_the_missing_generation(tmp_path) -> None:
    """The judge fails closed rather than silently judging a different
    generation's predictions -- and the error must name the generation an
    operator actually asked for, not a hardcoded "G0"."""
    config = prepared_processed_fixture(tmp_path)  # predictions stored under G0
    provider = FakeJudgeProvider()
    with pytest.raises(IntegrityError, match="M003"):
        run_judge_pilot(
            config=config,
            batch_id="batch",
            allow_external_judge=True,
            generation="M003",
            provider=provider,
        )


def test_blind_pilot_sends_only_text_and_candidates_and_persists_both_models(tmp_path) -> None:
    config = prepared_processed_fixture(tmp_path)
    provider = FakeJudgeProvider()
    summary = run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        provider=provider,
        judge_models=("qwen3.5:397b", "deepseek-v3.2"),
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
        judge_models=("qwen3.5:397b", "deepseek-v3.2"),
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
        judge_models=("qwen3.5:397b", "deepseek-v3.2"),
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


def test_pilot_honours_a_model_set_override(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from data_quality_checker.judges import gemini_judge_model

    monkeypatch.setenv("GEMINI_JUDGE_MODEL", "gemini-test-model")
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


def test_lock_judge_accepts_a_gemini_model_that_ran_in_the_pilot(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from data_quality_checker.judges import gemini_judge_model

    monkeypatch.setenv("GEMINI_JUDGE_MODEL", "gemini-test-model")
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


_OLLAMA_ENV_NAMES = ("OLLAMA_API_KEY", *[f"OLLAMA_API_KEY_V{i}" for i in range(2, 8)])


def test_missing_credential_aborts_the_pilot_instead_of_recording_error_rows(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins the FIX 1 contract at the library boundary.

    A missing OLLAMA_API_KEY makes `resolve_judge_provider` raise
    `JudgeProviderUnavailable` from *outside* the per-document retry loop in
    `_run_pilot_impl`, so it must abort the whole pilot rather than being
    caught by the loop's `except (ContractError, ValueError, TypeError)` and
    recorded as a per-document "error" row. `JudgeProviderUnavailable` is also
    a `DQCheckError` so the CLI's top-level handler renders it cleanly.
    """
    config = prepared_processed_fixture(tmp_path)
    for name in _OLLAMA_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(JudgeProviderUnavailable) as exc_info:
        run_judge_pilot(
            config=config,
            batch_id="batch",
            allow_external_judge=True,
            judge_models=("qwen3.5:397b",),
        )
    assert isinstance(exc_info.value, DQCheckError)
    assert "OLLAMA_API_KEY" in str(exc_info.value)

    with Store(config.database_path) as store:
        assert store.list_judge_results("batch") == []


def test_missing_credential_surfaces_as_a_clean_cli_error(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pins the FIX 1 contract at the CLI boundary: no raw traceback, exit 2."""
    from data_quality_checker.cli import main

    prepared_processed_fixture(tmp_path)
    config_path = tmp_path / "config.json"
    for name in _OLLAMA_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)

    exit_code = main(
        [
            "--config",
            str(config_path),
            "pilot-judges",
            "--batch-id",
            "batch",
            "--allow-external-judge",
            "--judge-models",
            "qwen3.5:397b",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err.startswith("dqcheck: error:")
    assert "OLLAMA_API_KEY" in captured.err

    with Store(load_config(config_path).database_path) as store:
        assert store.list_judge_results("batch") == []


def test_local_gemma_is_registered_as_an_mlx_provider() -> None:
    from data_quality_checker.constants import JUDGE_MODEL_KEY
    from data_quality_checker.judges import judge_model_providers

    assert judge_model_providers()[JUDGE_MODEL_KEY] == "mlx"


def test_resolve_returns_the_mlx_provider_not_an_ollama_fallthrough(monkeypatch) -> None:
    """Regression: dispatch fell through to Ollama for every non-gemini kind."""
    from data_quality_checker.constants import JUDGE_MODEL_KEY
    from data_quality_checker.judges import MlxJudgeProvider, resolve_judge_provider

    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    for index in range(2, 8):
        monkeypatch.delenv(f"OLLAMA_API_KEY_V{index}", raising=False)
    provider = resolve_judge_provider(JUDGE_MODEL_KEY)
    assert isinstance(provider, MlxJudgeProvider)


def test_constructing_the_mlx_provider_does_not_load_weights() -> None:
    """A 21 GB load in __init__ would make every test that builds a provider unusable."""
    from data_quality_checker.judges import MlxJudgeProvider

    provider = MlxJudgeProvider()
    assert provider._model is None
    assert provider._tokenizer is None


def test_mlx_provider_strips_a_code_fence_before_returning(monkeypatch) -> None:
    """Ollama and Gemini enforce JSON server-side; a local model has no such gate."""
    from data_quality_checker.judges import MlxJudgeProvider

    provider = MlxJudgeProvider()
    monkeypatch.setattr(provider, "_generate", lambda prompt: ('```json\n{"verdict":"A"}\n```', {}))
    content, _ = provider.judge(model="gemma-4-31b-it-optiq-4bit", payload={"x": 1})
    assert content == '{"verdict":"A"}'


def test_unknown_judge_model_still_raises(monkeypatch) -> None:
    from data_quality_checker.errors import ContractError
    from data_quality_checker.judges import resolve_judge_provider

    with pytest.raises(ContractError):
        resolve_judge_provider("no-such-judge-model")
