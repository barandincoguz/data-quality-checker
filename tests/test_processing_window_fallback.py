"""Batch prediction retries a failed document in the validation worker's windows.

R01 d000112 ran to the generation limit with no JSON array and went to
quarantine; validation would have retried it in token windows (user decision,
3 Oct 2026). A document the primary pass answered is never re-generated.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from data_quality_checker import mlx_worker
from data_quality_checker.errors import IntegrityError
from data_quality_checker.processing import MlxG0Backend

ANSWER = '[{"kanun_no":"193","kanun_ad":"Gelir Vergisi Kanunu","madde":"94"}]'


class _Tokenizer:
    def apply_chat_template(self, messages: list[dict[str, str]], **_: Any) -> str:
        return "".join(message["content"] for message in messages)

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


def _backend(monkeypatch: pytest.MonkeyPatch, chunks: list[tuple[str, int, str | None]]) -> MlxG0Backend:
    def stream_generate(*_: Any, **__: Any):
        for text, tokens, finish in chunks:
            yield types.SimpleNamespace(text=text, generation_tokens=tokens, finish_reason=finish)

    monkeypatch.setitem(sys.modules, "mlx_lm", types.SimpleNamespace(stream_generate=stream_generate))
    backend = object.__new__(MlxG0Backend)
    backend.system_prompt = "sistem"
    backend.max_input_tokens = 100
    backend.max_generation_tokens = 50
    backend.model = object()
    backend.tokenizer = _Tokenizer()
    return backend


def _fallback(monkeypatch: pytest.MonkeyPatch, result: dict[str, Any] | Exception) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(mlx_worker, "_generate_window_fallback", fake)
    return calls


def _recovered() -> dict[str, Any]:
    return {
        "raw_output": ANSWER,
        "parsed_references": [{"kanun_no": "193", "kanun_ad": "Gelir Vergisi Kanunu", "madde": "94"}],
        "parse_ok": True,
        "parse_error": None,
        "input_tokens": 40,
        "output_tokens": 30,
        "finish_reason": "stop",
        "fallback": {
            "strategy": "lossless_token_windows_v1",
            "recovered": True,
            "windows": [{"window_index": 0, "parse_ok": True, "raw_output": ANSWER}],
        },
    }


def test_a_runaway_without_json_is_recovered_in_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(monkeypatch, [("aaa " * 10, 50, "length")])
    calls = _fallback(monkeypatch, _recovered())
    result = backend.predict({"text": "metin"})
    assert result.status == "success" and result.error is None
    assert result.references == [{"kanun_no": "193", "kanun_ad": "Gelir Vergisi Kanunu", "madde": "94"}]
    fallback = result.operational["window_fallback"]
    assert fallback["recovered"] is True
    assert fallback["primary"]["finish_reason"] == "length"
    assert "raw_output" not in fallback["windows"][0]
    assert result.operational["finish_reason"] == "window_fallback_complete"
    assert result.operational["truncated"] is False
    assert result.operational["output_tokens"] == 80
    (call,) = calls
    assert (call["window_tokens"], call["overlap_tokens"], call["max_generation_tokens"]) == (1024, 256, 1024)
    assert call["recover_repetition"] is True
    assert call["messages"][1] == {"role": "user", "content": "metin"}


def test_an_answered_document_is_never_regenerated(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(monkeypatch, [(ANSWER, 20, "stop")])
    calls = _fallback(monkeypatch, AssertionError("must not be called"))
    result = backend.predict({"text": "metin"})
    assert result.status == "success" and calls == []
    assert result.operational["window_fallback"] is None
    assert result.raw_output == ANSWER


def test_a_failed_fallback_keeps_the_primary_error_and_records_the_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _backend(monkeypatch, [("aaa " * 10, 50, "length")])
    failed = {**_recovered(), "parse_ok": False, "parsed_references": [], "finish_reason": "length",
              "parse_error": "window fallback failed at window=0: length"}
    failed["fallback"] = {**failed["fallback"], "recovered": False}
    _fallback(monkeypatch, failed)
    result = backend.predict({"text": "metin"})
    assert result.status == "error"
    assert result.error.startswith("model output reached generation limit")
    assert result.operational["truncated"] is True
    attempt = result.operational["window_fallback"]
    assert attempt["attempted"] is True and attempt["recovered"] is False
    assert attempt["error"] == "window fallback failed at window=0: length"
    assert "raw_output" not in attempt["windows"][0]


def test_a_fallback_that_raises_keeps_the_primary_error(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(monkeypatch, [("aaa " * 10, 50, "length")])
    _fallback(monkeypatch, IntegrityError("window fallback tokenizer exposes no offsets"))
    result = backend.predict({"text": "metin"})
    assert result.status == "error"
    assert result.operational["window_fallback"] == {
        "attempted": True,
        "recovered": False,
        "error": "window fallback tokenizer exposes no offsets",
    }


def test_an_over_long_input_is_tried_in_windows_without_a_primary_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _backend(monkeypatch, [])
    backend.max_input_tokens = 3
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        types.SimpleNamespace(stream_generate=lambda *a, **k: pytest.fail("primary pass ran")),
    )
    calls = _fallback(monkeypatch, _recovered())
    result = backend.predict({"text": "çok uzun bir belge metni burada"})
    assert result.status == "success" and len(calls) == 1
    assert result.operational["window_fallback"]["primary"]["finish_reason"] == "input_too_long"
