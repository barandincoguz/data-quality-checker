from __future__ import annotations

import io
import json
from typing import Any

import pytest

from data_quality_checker.judges import GeminiJudgeProvider, JudgeProviderUnavailable


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def _gemini_body(text: str) -> bytes:
    return json.dumps({"candidates": [{"content": {"parts": [{"text": text}]}}]}).encode("utf-8")


def test_judge_returns_model_text_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeResponse:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(_gemini_body('{"verdict":"A"}'))

    monkeypatch.setattr("data_quality_checker.judges.urllib.request.urlopen", fake_urlopen)

    provider = GeminiJudgeProvider()
    raw, meta = provider.judge(model="gemini-3.1-pro", payload={"document": "metin"})

    assert raw == '{"verdict":"A"}'
    assert meta["provider"] == "gemini"
    assert meta["latency_seconds"] >= 0.0
    assert "gemini-3.1-pro:generateContent" in captured["url"]
    assert captured["body"]["generationConfig"]["temperature"] == 0
    assert captured["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert "metin" in captured["body"]["contents"][0]["parts"][0]["text"]


def test_missing_api_key_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(JudgeProviderUnavailable):
        GeminiJudgeProvider()


def test_network_failure_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def boom(request: Any, timeout: float | None = None) -> None:
        raise TimeoutError("timed out")

    monkeypatch.setattr("data_quality_checker.judges.urllib.request.urlopen", boom)
    provider = GeminiJudgeProvider()
    with pytest.raises(JudgeProviderUnavailable):
        provider.judge(model="gemini-3.1-pro", payload={"document": "metin"})


def test_empty_candidates_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeResponse:
        return _FakeResponse(json.dumps({"candidates": []}).encode("utf-8"))

    monkeypatch.setattr("data_quality_checker.judges.urllib.request.urlopen", fake_urlopen)
    provider = GeminiJudgeProvider()
    with pytest.raises(JudgeProviderUnavailable):
        provider.judge(model="gemini-3.1-pro", payload={"document": "metin"})


def test_gemini_judge_model_reads_the_environment_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from data_quality_checker.judges import DEFAULT_GEMINI_JUDGE_MODEL, gemini_judge_model

    monkeypatch.delenv("GEMINI_JUDGE_MODEL", raising=False)
    assert gemini_judge_model() == DEFAULT_GEMINI_JUDGE_MODEL

    monkeypatch.setenv("GEMINI_JUDGE_MODEL", "gemini-custom-id")
    assert gemini_judge_model() == "gemini-custom-id"
