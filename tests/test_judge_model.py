import pytest

from data_quality_checker.errors import IntegrityError
from data_quality_checker.judge_model import (
    assert_gemma_thinking_is_disabled,
    resolve_judge_snapshot,
)

SENTINEL = "<|channel>thought\n<channel|>"


def test_sentinel_present_is_accepted() -> None:
    assert_gemma_thinking_is_disabled("<|turn>model\n" + SENTINEL)


def test_missing_sentinel_is_rejected() -> None:
    with pytest.raises(IntegrityError):
        assert_gemma_thinking_is_disabled("<|turn>model\n")


def test_open_thought_channel_is_rejected() -> None:
    """A thinking channel that was opened and left open must not pass."""
    with pytest.raises(IntegrityError):
        assert_gemma_thinking_is_disabled("<|turn>model\n<|channel>thought\n")


def test_non_empty_thought_channel_is_rejected() -> None:
    """Content inside the thought channel means thinking is on."""
    with pytest.raises(IntegrityError):
        assert_gemma_thinking_is_disabled(
            "<|turn>model\n<|channel>thought\nreasoning here\n<channel|>"
        )


def test_snapshot_resolves_from_the_lab_cache() -> None:
    snapshot = resolve_judge_snapshot()
    assert snapshot.is_dir()
    assert (snapshot / "config.json").is_file()
    assert snapshot.name == "23616162c5a8f928cac5b21d3e974d1dbc0b9877"


def test_missing_snapshot_raises_judge_provider_unavailable(monkeypatch, tmp_path) -> None:
    from data_quality_checker.judges import JudgeProviderUnavailable

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    with pytest.raises(JudgeProviderUnavailable):
        resolve_judge_snapshot()
