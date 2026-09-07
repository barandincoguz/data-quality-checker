from __future__ import annotations

import pytest

from data_quality_checker.processing import (
    REPETITION_STOP_RUN_LENGTH,
    salvage_repeated_json_prefix,
    terminal_duplicate_run_reached,
)


def _objects(*tuples: tuple[str, str]) -> str:
    return ",".join(
        '{"kanun_no":"%s","kanun_ad":"K","madde":"%s","fikra":"","bent":""}' % (no, madde)
        for no, madde in tuples
    )


class TestEarlyStop:
    """The streaming loop stops a tuple loop instead of paying for it.

    d000021 emitted the same reference 259 times before the 8192-token ceiling
    cut it off after 166 seconds; doubling the ceiling to 16384 produced 524
    copies in 334 seconds and still no closing bracket. The unique set was five
    references both times, so every token after the third copy bought nothing.
    """

    def test_a_run_of_identical_tuples_stops_generation(self) -> None:
        text = "[" + _objects(("488", "1"), ("488", "6"), ("488", "6"), ("488", "6"))
        assert terminal_duplicate_run_reached(text) is True

    def test_two_identical_tuples_are_not_enough(self) -> None:
        """A pair can still be followed by different references, and the
        pipeline deduplicates anyway, so a pair is not evidence of a loop. A
        run of three has no legitimate reading.
        """
        text = "[" + _objects(("488", "1"), ("488", "6"), ("488", "6"))
        assert terminal_duplicate_run_reached(text) is False
        assert REPETITION_STOP_RUN_LENGTH == 3

    def test_distinct_references_never_stop_generation(self) -> None:
        text = "[" + _objects(("488", "1"), ("488", "3"), ("488", "4"), ("193", "9"))
        assert terminal_duplicate_run_reached(text) is False

    def test_a_repeat_that_is_not_terminal_does_not_stop(self) -> None:
        """The run must be at the end. An earlier triple that the model has
        already moved past is history, not a loop in progress.
        """
        text = "[" + _objects(("488", "6"), ("488", "6"), ("488", "6"), ("193", "9"))
        assert terminal_duplicate_run_reached(text) is False

    def test_an_incomplete_trailing_object_is_ignored(self) -> None:
        text = "[" + _objects(("488", "6"), ("488", "6"), ("488", "6")) + ',{"kanun_no":"48'
        assert terminal_duplicate_run_reached(text) is True

    def test_a_short_or_empty_output_never_stops(self) -> None:
        assert terminal_duplicate_run_reached("") is False
        assert terminal_duplicate_run_reached("[") is False
        assert terminal_duplicate_run_reached("[" + _objects(("488", "6"))) is False


class TestSalvage:
    """The deployment path recovers what the validation path already recovered.

    The same model, failing the same way, was recovered during validation and
    quarantined during deployment, purely because the recovery lived in one
    code path and not the other.
    """

    def test_a_terminal_duplicate_loop_is_salvaged_to_its_unique_prefix(self) -> None:
        raw = "[" + _objects(
            ("488", "1"), ("488", "3"), ("488", "6"), ("488", "6"), ("488", "6")
        ) + ',{"kanun_no":"48'
        result = salvage_repeated_json_prefix(raw)
        assert result is not None
        assert [reference["madde"] for reference in result["references"]] == ["1", "3", "6"]
        assert result["duplicate_reference_count"] == 2
        assert result["synthetic_closing_bracket"] is True

    def test_an_enumeration_runaway_is_refused(self) -> None:
        """d000059 counted `bent` upward -- 201, 202, 203 -- so no two tuples
        are equal and there is nothing to deduplicate. Closing the array would
        inject fabricated references into the batch, so the rule must refuse
        and let the document be quarantined.
        """
        raw = "[" + ",".join(
            '{"kanun_no":"5520","kanun_ad":"K","madde":"30","fikra":"1","bent":"%d"}' % index
            for index in range(200, 260)
        )
        assert salvage_repeated_json_prefix(raw) is None

    def test_output_with_no_array_is_refused(self) -> None:
        assert salvage_repeated_json_prefix("I cannot answer that") is None

    def test_a_single_object_is_refused(self) -> None:
        assert salvage_repeated_json_prefix("[" + _objects(("488", "1"))) is None


@pytest.mark.parametrize("raw", ["", "[", "[]"])
def test_neither_guard_crashes_on_degenerate_output(raw: str) -> None:
    assert terminal_duplicate_run_reached(raw) is False
    assert salvage_repeated_json_prefix(raw) is None


class _StubResponse:
    def __init__(self, text: str, generation_tokens: int, finish_reason: str | None) -> None:
        self.text = text
        self.generation_tokens = generation_tokens
        self.finish_reason = finish_reason
        self.prompt_tps = 1.0
        self.generation_tps = 1.0
        self.peak_memory = 1


class _StubTokenizer:
    def apply_chat_template(self, messages, **_):
        return "PROMPT"

    def encode(self, text: str):
        return [0] * 10


def _backend(monkeypatch, chunks: list[str], finish_reason: str, module) -> object:
    """A backend whose only real part is the code under test.

    `predict` is where the two guards actually take effect; the helpers passing
    in isolation would not have caught a mis-wiring, which is the class of
    defect this whole change is about.
    """
    backend = object.__new__(module.MlxG0Backend)
    backend.tokenizer = _StubTokenizer()
    backend.model = object()
    backend.max_input_tokens = 4096
    backend.max_generation_tokens = 64

    def fake_stream_generate(model, tokenizer, prompt, max_tokens):
        for index, chunk in enumerate(chunks, start=1):
            last = index == len(chunks)
            yield _StubResponse(chunk, index, finish_reason if last else None)

    monkeypatch.setattr("mlx_lm.stream_generate", fake_stream_generate, raising=False)
    import sys
    import types

    fake_module = types.ModuleType("mlx_lm")
    fake_module.stream_generate = fake_stream_generate
    fake_module.load = lambda *a, **k: (None, None)
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_module)
    return backend


def _object(no: str, madde: str) -> str:
    return '{"kanun_no":"%s","kanun_ad":"K","madde":"%s","fikra":"","bent":""}' % (no, madde)


def test_predict_stops_a_tuple_loop_and_returns_the_unique_references(monkeypatch) -> None:
    from data_quality_checker import processing

    chunks = ["[", _object("488", "1"), ",", _object("488", "6"), ",", _object("488", "6"),
              ",", _object("488", "6"), ",", _object("488", "6")]
    backend = _backend(monkeypatch, chunks, "length", processing)
    result = backend.predict({"text": "belge"})

    assert result.operational["repetition_stopped"] is True
    assert result.operational["finish_reason"] == "repetition_stop"
    # The loop was cut short, so the last chunk was never generated.
    assert result.operational["output_tokens"] < len(chunks)
    assert result.status == "success"
    assert [reference["madde"] for reference in result.references] == ["1", "6"]
    assert result.operational["truncated"] is False
    assert result.operational["repetition_recovery"]["duplicate_reference_count"] >= 1


def test_predict_leaves_an_enumeration_runaway_as_an_error(monkeypatch) -> None:
    """The document that must stay quarantined. Nothing here is a duplicate,
    so there is no safe way to close the array.
    """
    from data_quality_checker import processing

    chunks = ["["] + [
        ',{"kanun_no":"5520","kanun_ad":"K","madde":"30","fikra":"1","bent":"%d"}' % index
        for index in range(200, 240)
    ]
    chunks[1] = chunks[1].lstrip(",")
    backend = _backend(monkeypatch, chunks, "length", processing)
    result = backend.predict({"text": "belge"})

    assert result.operational["repetition_stopped"] is False
    assert result.status == "error"
    assert result.references == []
    assert result.operational["truncated"] is True
    assert result.operational["repetition_recovery"] is None


def test_predict_leaves_a_normal_generation_untouched(monkeypatch) -> None:
    """Both guards must be invisible to a document that terminates properly --
    otherwise re-running only the failed documents would not be sound.
    """
    from data_quality_checker import processing

    chunks = ["[", _object("488", "1"), ",", _object("193", "9"), ",", _object("213", "3"), "]"]
    backend = _backend(monkeypatch, chunks, "stop", processing)
    result = backend.predict({"text": "belge"})

    assert result.status == "success"
    assert result.operational["repetition_stopped"] is False
    assert result.operational["repetition_recovery"] is None
    assert result.operational["truncated"] is False
    assert [reference["madde"] for reference in result.references] == ["1", "9", "3"]
    assert result.operational["output_tokens"] == len(chunks)
