from __future__ import annotations

import pytest

from data_quality_checker.mlx_stateful import (
    LOSS_REDUCTION,
    StatefulTrainingConfig,
    _completion_target_span,
    _normalise_accumulated_gradient,
)


def test_completion_span_excludes_padding_after_the_chat() -> None:
    start, end = _completion_target_span(completion_offset=5, total_tokens=8)

    assert (start, end) == (5, 8)
    assert end - start == 3


def test_target_token_loss_reduction_is_part_of_the_training_config() -> None:
    assert StatefulTrainingConfig().loss_reduction == LOSS_REDUCTION


def test_accumulated_gradient_is_normalised_by_all_target_tokens() -> None:
    # A three-token empty target and a 123-token positive target contribute
    # their summed token gradients, then share one token-count denominator.
    accumulated = {"weight": 3 * 1.0 + 123 * 2.0}

    normalised = _normalise_accumulated_gradient(
        lambda fn, tree: {key: fn(value) for key, value in tree.items()},
        accumulated,
        target_tokens=126,
    )

    assert normalised["weight"] == pytest.approx(249 / 126)
    assert normalised["weight"] != pytest.approx((1.0 + 2.0) / 2)
