"""Locate and validate the local adjudicator model.

The judge is pinned to one snapshot revision on purpose: a locked judge
whose weights can change is not a fixed instrument, and every round's
adjudication would be measured against a different rater.
"""

from __future__ import annotations

from pathlib import Path

from .constants import JUDGE_MODEL_ID, JUDGE_MODEL_REVISION
from .errors import IntegrityError

GEMMA_THINKING_SENTINEL = "<|channel>thought\n<channel|>"
_THOUGHT_CHANNEL_OPEN = "<|channel>thought\n"
_THOUGHT_CHANNEL_CLOSE = "<channel|>"


def assert_gemma_thinking_is_disabled(prompt: str) -> None:
    """Validate Gemma 4's empty, closed no-thinking sentinel.

    Counterpart to mlx_compute.assert_thinking_is_disabled, which encodes
    Qwen3.5's different sentinel. Measured behaviour: with thinking left on,
    Gemma spends its whole budget in the thought channel and returns no JSON
    at all, so this is a correctness gate, not a cosmetic one.

    A naive `prompt.endswith(GEMMA_THINKING_SENTINEL)` check is insufficient:
    a *filled* thought channel also ends with the closing tag `<channel|>`,
    because any text can sit between the opening tag and the close. So this
    checks that the last opening of the thought channel is immediately
    followed by the closing tag, i.e. nothing was written in between.
    """

    last_open = prompt.rfind(_THOUGHT_CHANNEL_OPEN)
    if last_open == -1:
        raise IntegrityError("Gemma thinking-disable sentinel is missing, open, or non-empty")
    after_open = last_open + len(_THOUGHT_CHANNEL_OPEN)
    if not prompt.endswith(_THOUGHT_CHANNEL_CLOSE):
        raise IntegrityError("Gemma thinking-disable sentinel is missing, open, or non-empty")
    close_start = len(prompt) - len(_THOUGHT_CHANNEL_CLOSE)
    if close_start != after_open:
        raise IntegrityError("Gemma thinking-disable sentinel is missing, open, or non-empty")


def resolve_judge_snapshot() -> Path:
    """Return the pinned judge snapshot from the local cache, offline."""

    from huggingface_hub import snapshot_download

    from .judges import JudgeProviderUnavailable
    from .mlx_compute import _model_cache_dir

    try:
        path = snapshot_download(
            repo_id=JUDGE_MODEL_ID,
            revision=JUDGE_MODEL_REVISION,
            cache_dir=str(_model_cache_dir()),
            local_files_only=True,
        )
    except Exception as exc:  # huggingface_hub raises several unrelated types offline
        raise JudgeProviderUnavailable(
            f"judge snapshot {JUDGE_MODEL_ID}@{JUDGE_MODEL_REVISION[:12]} "
            f"is not in the local cache: {exc}"
        ) from exc
    return Path(path).resolve()
