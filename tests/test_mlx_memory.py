"""Full-document training memory: same arithmetic, less memory.

`mlx_lm` trains Qwen3.5's linear-attention layers through a per-token Python
recurrence whose backward keeps every step's state -- 66.6 GB at 1,536 tokens,
112.9 GB at 4,096 (measured 27 Sep 2026). Chunked, checkpointed recurrence and a
chunked loss must give exactly the numbers the stock path gives.
"""

from __future__ import annotations

from functools import partial

import pytest

mx = pytest.importorskip("mlx.core")

from data_quality_checker import mlx_memory  # noqa: E402


def _inputs(T: int = 300, H: int = 4, Dk: int = 16, Dv: int = 16):
    mx.random.seed(0)
    return (
        mx.random.normal((1, T, H, Dk)),
        mx.random.normal((1, T, H, Dk)),
        mx.random.normal((1, T, H, Dv)),
        mx.sigmoid(mx.random.normal((1, T, H))),
        mx.sigmoid(mx.random.normal((1, T, H))),
    )


def _loss(fn, q, k, v, g, beta):
    y, state = fn(q, k, v, g, beta)
    return (y.astype(mx.float32) ** 2).sum() + state.sum()


def test_the_chunked_recurrence_gives_the_stock_outputs_and_gradients_exactly() -> None:
    import mlx_lm.models.gated_delta as gated_delta

    args = _inputs()
    stock = mx.value_and_grad(partial(_loss, gated_delta.gated_delta_ops), argnums=(0, 1, 2, 3, 4))(
        *args
    )
    chunked = mx.value_and_grad(
        partial(_loss, partial(mlx_memory.chunked_gated_delta_ops, chunk=64)),
        argnums=(0, 1, 2, 3, 4),
    )(*args)
    assert float(stock[0]) == float(chunked[0])
    for left, right in zip(stock[1], chunked[1]):
        assert float(mx.abs(left - right).max()) == 0.0


def test_installing_the_chunked_recurrence_is_reversible() -> None:
    import mlx_lm.models.gated_delta as gated_delta

    stock = gated_delta.gated_delta_ops
    with mlx_memory.chunked_recurrence():
        assert gated_delta.gated_delta_ops is not stock
    assert gated_delta.gated_delta_ops is stock


def test_the_chunked_loss_is_the_full_loss() -> None:
    mx.random.seed(1)
    hidden = mx.random.normal((1, 2500, 32))
    weight = mx.random.normal((50, 32))
    targets = mx.random.randint(0, 50, (1, 2500))
    mask = (mx.arange(2500) > 1800)[None]

    def head(h):
        return h @ weight.T

    import mlx.nn as nn

    full = (nn.losses.cross_entropy(head(hidden), targets) * mask).astype(mx.float32).sum()
    chunked = mlx_memory.chunked_completion_loss(hidden, head, targets, mask, chunk=1024)
    assert abs(float(full) - float(chunked)) <= 1e-4 * abs(float(full))
