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


def test_gemini_model_is_registered_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_JUDGE_MODEL", "gemini-test-model")
    assert judge_model_providers()[gemini_judge_model()] == "gemini"


def test_fake_backend_short_circuits_every_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_JUDGE_MODEL", "gemini-test-model")
    provider = resolve_judge_provider(gemini_judge_model(), fake_backend=True)
    assert isinstance(provider, FakeJudgeProvider)


def test_resolves_ollama_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    provider = resolve_judge_provider("qwen3.5:397b")
    assert isinstance(provider, OllamaJudgeProvider)


def test_resolves_gemini_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_JUDGE_MODEL", "gemini-test-model")
    provider = resolve_judge_provider(gemini_judge_model())
    assert isinstance(provider, GeminiJudgeProvider)


def test_unknown_model_is_a_contract_error() -> None:
    with pytest.raises(ContractError):
        resolve_judge_provider("not-a-registered-model")


def test_gemini_judge_model_collision_with_a_static_model_id_is_a_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_JUDGE_MODEL", "qwen3.5:397b")
    with pytest.raises(ContractError, match="qwen3.5:397b"):
        judge_model_providers()


def test_fake_backend_does_not_rescue_an_unregistered_model() -> None:
    # Intentional: fake_backend short-circuits credential lookup for a known
    # model, but an unregistered model id must still fail closed instead of
    # silently returning a FakeJudgeProvider. Pinned so a later refactor
    # cannot flip this without a deliberate test change.
    with pytest.raises(ContractError):
        resolve_judge_provider("bogus", fake_backend=True)


def test_unset_gemini_judge_model_leaves_ollama_judges_working(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression this change most risks: no Gemini config must not
    break resolving the Ollama judges."""
    monkeypatch.delenv("GEMINI_JUDGE_MODEL", raising=False)
    assert gemini_judge_model() is None
    assert judge_model_providers() == {
        "qwen3.5:397b": "ollama",
        "deepseek-v3.2": "ollama",
        "gemma-4-31b-it-optiq-4bit": "mlx",
    }
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    provider = resolve_judge_provider("qwen3.5:397b")
    assert isinstance(provider, OllamaJudgeProvider)
    with pytest.raises(ContractError):
        resolve_judge_provider("gemini-anything")


def test_default_pilot_is_the_local_judge() -> None:
    from data_quality_checker.constants import JUDGE_MODEL_KEY
    from data_quality_checker.judges import JUDGE_MODELS

    assert JUDGE_MODELS == (JUDGE_MODEL_KEY,)


def test_every_default_judge_resolves_without_any_cloud_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default pilot must be runnable on this machine with no credentials.

    resolve_judge_provider is called outside the pilot's retry block, and the
    cloud providers raise in __init__ when their key is missing, so a cloud id
    in JUDGE_MODELS aborts the entire pilot at the first document instead of
    recording an unavailable judge.
    """
    from data_quality_checker.judges import JUDGE_MODELS, resolve_judge_provider

    for name in ("OLLAMA_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_JUDGE_MODEL"):
        monkeypatch.delenv(name, raising=False)
    for index in range(2, 8):
        monkeypatch.delenv(f"OLLAMA_API_KEY_V{index}", raising=False)

    for model in JUDGE_MODELS:
        resolve_judge_provider(model)
