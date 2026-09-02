from __future__ import annotations

import http.client
import io
import json
import urllib.error
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
    # urllib.request.Request normalises header names to capitalised form
    # (e.g. "x-goog-api-key" -> "X-goog-api-key"), so check what the object
    # actually holds rather than the case we sent.
    assert captured["headers"]["X-goog-api-key"] == "test-key"
    assert "Authorization" not in captured["headers"]


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


def test_http_error_body_is_surfaced_in_the_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def fake_urlopen(request: Any, timeout: float | None = None) -> None:
        raise urllib.error.HTTPError(
            request.full_url,
            404,
            "Not Found",
            {},
            io.BytesIO(b'{"error":{"message":"model not found"}}'),
        )

    monkeypatch.setattr("data_quality_checker.judges.urllib.request.urlopen", fake_urlopen)
    provider = GeminiJudgeProvider()
    with pytest.raises(JudgeProviderUnavailable) as exc_info:
        provider.judge(model="gemini-3.1-pro", payload={"document": "metin"})

    message = str(exc_info.value)
    assert "model not found" in message
    assert message != "HTTP Error 404: Not Found"


def test_incomplete_read_during_response_read_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class _BrokenResponse:
        def __enter__(self) -> "_BrokenResponse":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def read(self) -> bytes:
            raise http.client.IncompleteRead(partial=b"{")

    def fake_urlopen(request: Any, timeout: float | None = None) -> _BrokenResponse:
        return _BrokenResponse()

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
