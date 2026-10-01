from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_quality_checker.commands import pilot_judges
from data_quality_checker.config import default_config_path, load_config
from data_quality_checker.errors import ContractError, DQCheckError, GateBlocked, IntegrityError
from data_quality_checker.fingerprints import sha256_file
from data_quality_checker.judges import (
    FakeJudgeProvider,
    JudgeProviderUnavailable,
    ensure_green_audit_plan,
    judge_instrument,
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


def prepared_processed_fixture(
    tmp_path: Path,
    *,
    count: int = 1,
    generation: str = "G0",
    references: list[dict[str, str]] | None = None,
):
    config = config_for(tmp_path)
    annotations = [
        {
            "document_id": f"private-id-{index}",
            "current_references": [dict(row) for row in (references or [])],
        }
        for index in range(count)
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
    config = prepared_processed_fixture(
        tmp_path,
        references=[
            {
                # madde is 353, not the task-example 413: kanun_no 213 /
                # madde 413 is exactly DEFAULT_REFERENCE_POLICY_ID's ignored
                # VUK boilerplate identity (reference_policy.py), so that row
                # is filtered to an empty candidate list before blinding and
                # makes the guard assertion below fail vacuously the other
                # way. 353 is a real, non-boilerplate VUK article and survives
                # apply_reference_policy so the loop actually executes.
                "kanun_no": "213",
                "kanun_ad": "Vergi Usul Kanunu",
                "madde": "353",
                "source_text": "Güvenli belge metni",
            }
        ],
    )
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
        # The keys were always right; the contents were not. Round 1 shipped
        # a judge pilot through this test with the annotator's source_text on
        # every candidate. The five fields are written out rather than read
        # from REFERENCE_DECISION_FIELDS, so this assertion still fails if
        # that tuple ever regains source_text.
        # Guard against this assertion going vacuous again: it previously sat
        # in a test whose fixture carried no references at all, so the loop
        # below never ran and could not have failed however broken the code.
        assert payload["candidate_a"] and payload["candidate_b"]
        for candidate in (payload["candidate_a"], payload["candidate_b"]):
            for reference in candidate:
                assert set(reference) == {"kanun_no", "kanun_ad", "madde"}
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
            "--generation",
            "G0",
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


def test_pilot_summary_records_the_prompt_the_judges_were_given(tmp_path) -> None:
    """The prompt is half the instrument; a silent edit would move every number.

    A pinned model revision at temperature zero only makes the judge
    reproducible if the instructions are pinned too. Rounds whose summaries
    carry different fingerprints were not judged by the same instrument.
    """
    import hashlib

    from data_quality_checker.judges import JUDGE_PROMPT, judge_prompt_fingerprint

    config = prepared_processed_fixture(tmp_path)
    summary = run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        provider=FakeJudgeProvider(),
        judge_models=("qwen3.5:397b",),
    )
    assert summary["judge_prompt_sha256"] == judge_prompt_fingerprint()
    assert (
        summary["judge_prompt_sha256"] == hashlib.sha256(JUDGE_PROMPT.encode("utf-8")).hexdigest()
    )


def test_a_rejected_judge_output_is_kept_for_audit(tmp_path) -> None:
    """A refusal that keeps no evidence cannot be audited.

    When the contract rejects a judge's answer, only the error message used to
    survive. That is not enough to tell a judge that really produced a
    malformed verdict from a validator that was too strict -- and the
    difference is exactly what a judge-error metric claims to measure. The
    rejected text is now kept beside the attempt that produced it.

    Uses an invalid verdict token rather than an unfindable evidence span, as
    this test did before the per-span partition: a single unverifiable span
    is no longer fatal to the whole verdict (see the evidence tests above), so
    it can no longer stand in here for "a contract violation that is still
    fatal." An invalid verdict token still is.
    """
    import json as _json

    from data_quality_checker.judges import JudgeProviderUnavailable  # noqa: F401

    class AlwaysMalformed:
        def judge(self, *, model, payload):
            return (
                _json.dumps(
                    {
                        "verdict": "MAYBE",
                        "candidate_errors": {"A": [], "B": []},
                        "final_references": [],
                        "evidence": [],
                        "reason_codes": ["test"],
                    },
                    ensure_ascii=False,
                ),
                {"latency_seconds": 0.0, "cost": 0.0, "provider": "fake"},
            )

    config = prepared_processed_fixture(tmp_path)
    run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        provider=AlwaysMalformed(),
        judge_models=("qwen3.5:397b",),
    )
    with Store(config.database_path) as store:
        rows = store.list_judge_results("batch")
    assert rows and rows[0]["status"] == "error"
    response = _json.loads(Path(rows[0]["response_path"]).read_text(encoding="utf-8"))
    attempts = response["attempts"]
    assert attempts, "an error must record the attempts that produced it"
    assert all(a["status"] == "invalid" for a in attempts)
    for attempt in attempts:
        assert "MAYBE" in attempt["rejected_output"]


_DOCUMENT_TEXT = "213 sayılı Vergi Usul Kanununun mükerrer 20 nci maddesi uyarınca."


def _judge_payload(**final_reference_updates):
    reference = {
        "kanun_no": "213",
        "kanun_ad": "Vergi Usul Kanunu",
        "madde": "mükerrer 20",
    }
    reference.update(final_reference_updates)
    return {
        "verdict": "A",
        "candidate_errors": {"A": [], "B": []},
        "final_references": [reference],
        "evidence": ["213 sayılı Vergi Usul Kanununun mükerrer 20 nci maddesi"],
        "reason_codes": ["ok"],
    }


def test_final_references_with_no_source_text_validate() -> None:
    """source_text is annotator prose for error analysis, not a judge
    contract field: a final reference that carries none must still pass."""
    from data_quality_checker.judges import _validate_judge_result

    validated = _validate_judge_result(_judge_payload(), _DOCUMENT_TEXT)
    assert validated["verdict"] == "A"
    assert (
        "source_text" not in validated["final_references"][0]
        or not validated["final_references"][0]["source_text"]
    )


def test_final_references_source_text_need_not_be_verbatim() -> None:
    """Round 1 (d000081): the judge adopted the annotator's stitched span and
    was refused three times, though the underlying reference was correct.
    A non-verbatim, even absent, source_text must no longer block a verdict."""
    from data_quality_checker.judges import _validate_judge_result

    payload = _judge_payload(
        source_text="stitched together from two different places in the ruling"
    )
    validated = _validate_judge_result(payload, _DOCUMENT_TEXT)
    assert validated["verdict"] == "A"
    assert validated["final_references"][0]["madde"] == "mükerrer 20"


def test_evidence_span_absent_from_document_is_quarantined_not_rejected() -> None:
    """The judge's own evidence array is still a hallucination guard on a
    machine's output: a span it could not quote from the document is still
    identified as such, exactly as before. What changed is the consequence.

    Measured on batch M01: 280 evidence spans, 4 unverifiable (1.43%), no
    document with more than one -- yet discarding the whole verdict on that
    1.43% also discarded 21 verified spans riding along with the 4 bad ones.
    The guard now acts per span: the unquotable span is quarantined into
    `unverified_evidence` (so the signal this test used to prove -- that a
    fabricated span is caught -- still holds), and the verdict that carried
    it is validated normally rather than thrown away whole."""
    from data_quality_checker.judges import _validate_judge_result

    payload = _judge_payload()
    payload["evidence"] = ["bu parça belgede kesinlikle yok"]
    validated = _validate_judge_result(payload, _DOCUMENT_TEXT)
    assert validated["verdict"] == "A"
    assert validated["evidence"] == []
    assert validated["unverified_evidence"] == ["bu parça belgede kesinlikle yok"]


def test_one_unverifiable_span_among_several_does_not_discard_the_verdict() -> None:
    """The all-or-nothing check this replaces discarded 4 whole M01 verdicts
    -- 21 verified spans between them -- over 4 unverifiable spans out of 280
    (1.43%). One bad span among several must not cost the good ones: the
    verdict stays valid, `evidence` keeps only what verified, and the bad
    span is recorded separately rather than silently dropped."""
    from data_quality_checker.judges import _validate_judge_result

    payload = _judge_payload()
    payload["evidence"] = [
        "213 sayılı Vergi Usul Kanununun mükerrer 20 nci maddesi",
        "bu parça belgede kesinlikle yok",
    ]
    validated = _validate_judge_result(payload, _DOCUMENT_TEXT)
    assert validated["verdict"] == "A"
    assert validated["evidence"] == ["213 sayılı Vergi Usul Kanununun mükerrer 20 nci maddesi"]
    assert validated["unverified_evidence"] == ["bu parça belgede kesinlikle yok"]


def test_entirely_unverifiable_evidence_still_yields_a_valid_verdict() -> None:
    """Every span here is unverifiable, not just one -- the per-span partition
    still returns a valid verdict with an empty `evidence` and both spans
    quarantined into `unverified_evidence`.

    Opinion, spelled out further in the fix report: this is the right default
    rather than a gap. `verdict` and `final_references` are validated on their
    own terms by `validate_reference_list`/`apply_reference_policy`; nothing
    here makes a verdict's substance depend on its evidence array being
    non-empty. Refusing outright at 100% unverifiable would also resurrect the
    same all-or-nothing failure this fix removes, just moved to an
    unprincipled threshold (refuse at 100%? 90%? 50%?) -- a policy call that
    belongs where the full pilot's per-span rate is visible
    (`judge_expert_metrics`/`lock_judge`), not inside a single verdict's
    validator, which only ever sees one document."""
    from data_quality_checker.judges import _validate_judge_result

    payload = _judge_payload()
    payload["evidence"] = ["bu parça belgede kesinlikle yok", "başka bir uydurma parça"]
    validated = _validate_judge_result(payload, _DOCUMENT_TEXT)
    assert validated["verdict"] == "A"
    assert validated["evidence"] == []
    assert validated["unverified_evidence"] == [
        "bu parça belgede kesinlikle yok",
        "başka bir uydurma parça",
    ]


def test_judge_expert_metrics_counts_an_unverifiable_span_on_a_valid_result(
    tmp_path: Path,
) -> None:
    """A verdict can now carry an unverifiable span and still be `valid`, so
    the old attempt-based fabrication counters go silent for it -- no attempt
    ever failed, so nothing feeds `fabricated_evidence_attempts`. That is
    exactly the trap: once a bad-span verdict stops failing, a metric that
    only watched failed attempts would silently report zero fabrication while
    fabrication is exactly what happened. `judge_expert_metrics` must
    re-source the signal from the `unverified_evidence` field carried on the
    valid result itself."""
    from data_quality_checker.judges import judge_expert_metrics

    batch_id = "batch"
    model = "test-judge"
    store = Store(tmp_path / "store.db")
    try:
        store.create_batch(batch_id=batch_id, input_fingerprint="fp", config_fingerprint="cfg")
        store.add_document(
            batch_id,
            {
                "internal_doc_id": "doc-1",
                "public_doc_id": "doc-1",
                "raw_document_id": "doc-1",
                "selected_channel": "pdf",
                "pdf_coverage": 1.0,
                "html_coverage": 0.0,
                "text": "document text doc-1",
                "text_sha256": "0" * 64,
                "human_references": [],
            },
        )
        review = store.get_review(batch_id, "doc-1")
        assert review is not None
        store.update_review(
            batch_id=batch_id,
            internal_doc_id="doc-1",
            expected_version=review["row_version"],
            status="finalized",
            action="accept_human",
            final_references=[],
            reason=None,
            reviewer="fixture",
        )
        store.persist_judge_result(
            batch_id=batch_id,
            internal_doc_id="doc-1",
            model=model,
            blind_mapping="A=human,B=model",
            status="valid",
            verdict="TIE",
            result={
                "verdict": "TIE",
                "final_references": [],
                "evidence": ["kept span"],
                "unverified_evidence": ["fabricated span"],
            },
            response_path=None,
            response_sha256=None,
            retry_count=0,
            error=None,
        )
        metrics = judge_expert_metrics(store, batch_id=batch_id, model=model, ids=["doc-1"])
    finally:
        store.close()

    assert metrics["status"] == "complete"
    assert metrics["unverified_evidence_spans"] == 1
    assert metrics["evidence_spans_seen"] == 2
    assert metrics["unverified_evidence_span_rate"] == pytest.approx(0.5)
    # No attempt ever failed for this document, so the old attempt-based
    # signal is silent here -- surviving exactly that silence is the point.
    assert metrics["fabricated_evidence_attempts"] == 0


def test_judge_instrument_names_the_payload_shape_not_only_the_prompt() -> None:
    from data_quality_checker.judges import CANDIDATE_PAYLOAD_CONTRACT, judge_instrument

    instrument = judge_instrument()
    assert set(instrument) == {"prompt_sha256", "candidate_payload_contract"}
    assert instrument["candidate_payload_contract"] == CANDIDATE_PAYLOAD_CONTRACT
    # Round 1's invalid pilot recorded this sha. A run that changed only the
    # payload would otherwise have been indistinguishable from it.
    assert instrument["prompt_sha256"] != (
        "aa887e106fd908222c4afe368b5a598695ff31ca86fd35ccdbce0097760b3d74"
    )


def test_pilot_rejudges_a_verdict_formed_by_another_instrument(tmp_path) -> None:
    """The invalid round-1 pilot left 57 verdicts at status='valid'. A rerun
    that trusted them would have spent 35 minutes reproducing them."""
    import json as _json

    from data_quality_checker.judges import run_judge_pilot

    config = prepared_processed_fixture(tmp_path)
    first = run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        provider=FakeJudgeProvider(),
        judge_models=("qwen3.5:397b",),
    )
    assert first["models"]["qwen3.5:397b"]["valid"] == 1
    assert first["models"]["qwen3.5:397b"]["stale_verdicts_rejudged"] == 0

    with Store(config.database_path) as store:
        row = store.list_judge_results("batch")[0]
    path = Path(row["response_path"])
    stored = _json.loads(path.read_text(encoding="utf-8"))
    stored["judge_instrument"] = {
        "prompt_sha256": "0" * 64,
        "candidate_payload_contract": "spans_included_v0",
    }
    path.write_text(_json.dumps(stored, ensure_ascii=False), encoding="utf-8")
    with Store(config.database_path) as store:
        store.persist_judge_result(
            batch_id="batch",
            internal_doc_id=row["internal_doc_id"],
            model="qwen3.5:397b",
            blind_mapping=row["blind_mapping"],
            status="valid",
            verdict=row["verdict"],
            result=_json.loads(row["result_json"]),
            response_path=path,
            response_sha256=sha256_file(path),
            retry_count=0,
            error=None,
        )

    second = run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        provider=FakeJudgeProvider(),
        judge_models=("qwen3.5:397b",),
    )
    assert second["models"]["qwen3.5:397b"]["stale_verdicts_rejudged"] == 1
    assert second["models"]["qwen3.5:397b"]["valid"] == 1
    assert second["judge_instrument"] == judge_instrument()


def test_locked_coverage_refuses_a_verdict_from_another_instrument(tmp_path) -> None:
    """Production coverage is one instrument by definition, so a foreign
    verdict is an error there rather than something to redo silently."""
    from data_quality_checker.judges import _instrument_matches, judge_instrument

    assert _instrument_matches({"judge_instrument": judge_instrument()}) is True
    assert _instrument_matches({"judge_instrument": {"prompt_sha256": "0" * 64}}) is False
    assert _instrument_matches({}) is False


def test_the_judge_pilot_exam_is_served_blind_until_a_judge_is_locked(tmp_path) -> None:
    """The 60 pilot documents are the judge's exam: `judge_expert_metrics`
    grades the judge by comparing its `final_references` against the expert's
    decision on those same documents. So the expert must not be shown the
    judge's answer while deciding them, or the judge scores for having been
    copied.

    Today that holds only incidentally -- `_document_for_review` gates the
    suggestion on `_locked_judge`, and a lock cannot exist until every pilot
    document is finalized. Nothing stated or tested it, so a future change
    offering the pilot verdict as pre-lock decision support would contaminate
    the exam silently. This pins it, and the second half proves the assertion
    is not vacuous by showing the suggestion does appear once a judge is
    locked.
    """
    from data_quality_checker.hitl import _document_for_review, _locked_judge

    config = prepared_processed_fixture(tmp_path)
    with Store(config.database_path) as store:
        internal_doc_id = store.list_documents("batch")[0]["internal_doc_id"]
        store.set_router_bucket("batch", internal_doc_id, "RED")

    run_judge_pilot(
        config=config,
        batch_id="batch",
        allow_external_judge=True,
        provider=FakeJudgeProvider(),
        judge_models=("qwen3.5:397b",),
    )
    with Store(config.database_path) as store:
        stored = store.get_judge_result("batch", internal_doc_id, "qwen3.5:397b")
        assert stored is not None and stored["status"] == "valid"

        assert _locked_judge(config, "batch") is None
        served = _document_for_review(
            config=config, store=store, batch_id="batch", internal_doc_id=internal_doc_id
        )
        assert served["judge_suggestion"] is None

    # Finalize the review so lock_judge's gate opens, then lock.
    with Store(config.database_path) as store:
        review = store.get_review("batch", internal_doc_id)
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
    lock_judge(config=config, batch_id="batch", model="qwen3.5:397b", reason="fixture")

    with Store(config.database_path) as store:
        after = _document_for_review(
            config=config, store=store, batch_id="batch", internal_doc_id=internal_doc_id
        )
    assert after["judge_suggestion"] is not None
