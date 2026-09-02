from __future__ import annotations

import pytest

from data_quality_checker.errors import ContractError
from data_quality_checker.judges import (
    FakeJudgeProvider,
    GeminiJudgeProvider,
    OllamaJudgeProvider,
    gemini_judge_model,
    judge_model_providers,
    resolve_judge_provider,
)


def test_registry_covers_every_pilot_model() -> None:
    from data_quality_checker.judges import JUDGE_MODELS

    for model in JUDGE_MODELS:
        assert model in judge_model_providers()


def test_gemini_model_is_registered() -> None:
    assert judge_model_providers()[gemini_judge_model()] == "gemini"


def test_fake_backend_short_circuits_every_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    provider = resolve_judge_provider(gemini_judge_model(), fake_backend=True)
    assert isinstance(provider, FakeJudgeProvider)


def test_resolves_ollama_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    provider = resolve_judge_provider("qwen3.5:397b")
    assert isinstance(provider, OllamaJudgeProvider)


def test_resolves_gemini_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    provider = resolve_judge_provider(gemini_judge_model())
    assert isinstance(provider, GeminiJudgeProvider)


def test_unknown_model_is_a_contract_error() -> None:
    with pytest.raises(ContractError):
        resolve_judge_provider("not-a-registered-model")
