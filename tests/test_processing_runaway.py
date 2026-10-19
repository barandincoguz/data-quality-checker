from __future__ import annotations

from pathlib import Path

import pytest

from data_quality_checker.processing import (
    REPETITION_STOP_RUN_LENGTH,
    salvage_repeated_json_prefix,
    terminal_duplicate_run_reached,
)


def _objects(*tuples: tuple[str, str]) -> str:
    return ",".join(
        '{"kanun_no":"%s","kanun_ad":"K","madde":"%s"}' % (no, madde) for no, madde in tuples
    )


class TestEarlyStop:
    """The streaming loop stops a tuple loop instead of paying for it.

    d000021 emitted the same reference 259 times before the 8192-token ceiling
    cut it off after 166 seconds; doubling the ceiling to 16384 produced 524
    copies in 334 seconds and still no closing bracket. The unique set was five
    references both times, so every token after the third copy bought nothing.
    """

    def test_a_run_at_the_threshold_stops_generation(self) -> None:
        text = "[" + _objects(("488", "1"), *((("488", "6"),) * REPETITION_STOP_RUN_LENGTH))
        assert terminal_duplicate_run_reached(text) is True

    def test_a_run_shorter_than_the_threshold_is_not_enough(self) -> None:
        """The threshold comes from the data. Across 1,248 normally-terminating
        generations the longest run of identical consecutive references was 5,
        so anything at or below that must pass -- d000052 emitted four in a row
        and then produced eleven more distinct references.
        """
        for run in range(2, 6):
            text = "[" + _objects(("488", "1"), *((("488", "6"),) * run))
            assert terminal_duplicate_run_reached(text) is False, run
        assert REPETITION_STOP_RUN_LENGTH == 12

    def test_the_real_false_positive_that_set_the_threshold(self) -> None:
        """d000052, reduced to its shape: a run of four mid-list, then recovery.

        The first threshold stopped here and cut the document from eighteen
        references to seven. Nothing about this output is a loop.
        """
        text = "[" + _objects(
            ("213", "396"),
            ("213", "396"),
            ("213", "396"),
            ("213", "396"),
            ("4631", "4"),
            ("488", "1"),
        )
        assert terminal_duplicate_run_reached(text) is False

    def test_distinct_references_never_stop_generation(self) -> None:
        text = "[" + _objects(("488", "1"), ("488", "3"), ("488", "4"), ("193", "9"))
        assert terminal_duplicate_run_reached(text) is False

    def test_a_repeat_that_is_not_terminal_does_not_stop(self) -> None:
        """The run must be at the end. An earlier triple that the model has
        already moved past is history, not a loop in progress.
        """
        text = "[" + _objects(*((("488", "6"),) * REPETITION_STOP_RUN_LENGTH), ("193", "9"))
        assert terminal_duplicate_run_reached(text) is False

    def test_an_incomplete_trailing_object_is_ignored(self) -> None:
        text = "[" + _objects(*((("488", "6"),) * REPETITION_STOP_RUN_LENGTH)) + ',{"kanun_no":"48'
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
        raw = (
            "["
            + _objects(("488", "1"), ("488", "3"), ("488", "6"), ("488", "6"), ("488", "6"))
            + ',{"kanun_no":"48'
        )
        result = salvage_repeated_json_prefix(raw)
        assert result is not None
        assert [reference["madde"] for reference in result["references"]] == ["1", "3", "6"]
        assert result["duplicate_reference_count"] == 2
        assert result["synthetic_closing_bracket"] is True

    def test_a_cyclic_runaway_is_salvaged_to_its_unique_prefix(self) -> None:
        """d000031: the tail cycles through three tuples -- Damga Vergisi
        Kanunu articles 1, 3, and 9 -- instead of repeating one of them. There
        is no terminal *run* (the last two objects differ), but the final
        tuple recurs earlier in the array, and that recurrence is exactly as
        provable a duplication as a run: deduplication must still drop it, so
        closing the array cannot invent anything the model did not emit.
        """
        cycle = (("488", "1"), ("488", "3"), ("488", "9"))
        emitted = {(no, madde) for no, madde in cycle}
        raw = "[" + _objects(*(cycle * 4)) + ',{"kanun_no":"48'

        result = salvage_repeated_json_prefix(raw)

        assert result is not None
        recovered = {
            (reference["kanun_no"], reference["madde"]) for reference in result["references"]
        }
        assert recovered == emitted
        assert len(result["references"]) == 3
        # Nothing outside the model's own three-tuple cycle appears.
        assert recovered <= emitted
        assert result["terminal_duplicate_run_length"] == 1
        assert result["terminal_duplicate_repeat_count"] == 4

    def test_an_enumeration_runaway_is_refused(self) -> None:
        """A loop that counts the article upward -- 201, 202, 203 -- repeats no
        tuple, so there is nothing to deduplicate. Closing the array would
        inject fabricated references into the batch, so the rule must refuse
        and let the document be quarantined.
        """
        raw = "[" + ",".join(
            '{"kanun_no":"5520","kanun_ad":"K","madde":"%d"}' % index for index in range(200, 260)
        )
        assert salvage_repeated_json_prefix(raw) is None

    def test_the_real_archived_enumeration_runaway_is_one_article_repeated(self) -> None:
        """Regression pin on the real d000059 output, not a reduced shape.

        Under M000 (2026-09-07) it was 424 objects, 424 unique: `bent` counted
        upward on 5520 article 30, so no tuple recurred and the document stayed
        quarantined. A tuple is now a law and an article (1 Oct 2026), so that
        tail is one provision written 415 times: the rule proves it a
        duplicate and returns the model's own four provisions, nothing else.
        Skips rather than passing silently if the archived evidence file is
        not present.
        """
        archive = (
            Path(__file__).resolve().parent.parent
            / "artifacts"
            / "data_quality_checker"
            / "evidence"
            / "2026-09-07_m01_m000_runaway"
            / "d000059_cap16384.txt"
        )
        if not archive.is_file():
            pytest.skip(f"archived evidence file not present at {archive}")
        raw = archive.read_text(encoding="utf-8")
        result = salvage_repeated_json_prefix(raw)
        assert result is not None
        assert result["complete_reference_count"] == 424 and result["unique_reference_count"] == 4
        recovered = {(r["kanun_no"], r["madde"]) for r in result["references"]}
        assert recovered == {("5520", "3"), ("5520", "1"), ("193", "7"), ("5520", "30")}
        assert all(f'"madde":"{madde}"' in raw for _, madde in recovered)

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
    backend.system_prompt = "system"
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
    return '{"kanun_no":"%s","kanun_ad":"K","madde":"%s"}' % (no, madde)


def test_predict_stops_a_tuple_loop_and_returns_the_unique_references(monkeypatch) -> None:
    from data_quality_checker import processing
    from data_quality_checker.processing import REPETITION_STOP_RUN_LENGTH

    chunks = ["[", _object("488", "1")]
    for _ in range(REPETITION_STOP_RUN_LENGTH + 2):
        chunks += [",", _object("488", "6")]
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
        ',{"kanun_no":"5520","kanun_ad":"K","madde":"%d"}' % index for index in range(200, 240)
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
