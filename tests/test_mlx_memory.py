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


def _model_like_inputs(T: int, Hk: int = 2, Hv: int = 4, Dk: int = 16, Dv: int = 8, seed: int = 3):
    """Shaped and scaled like GatedDeltaNet's call: rms-normed q/k, decay in (0, 1)."""
    mx.random.seed(seed)
    q = mx.random.normal((1, T, Hk, Dk))
    k = mx.random.normal((1, T, Hk, Dk))
    inv = Dk**-0.5
    q = (inv**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv * mx.fast.rms_norm(k, None, 1e-6)
    v = mx.random.normal((1, T, Hv, Dv))
    g = mx.exp(-mx.exp(mx.random.normal((Hv,))) * mx.log1p(mx.exp(mx.random.normal((1, T, Hv)))))
    beta = mx.sigmoid(mx.random.normal((1, T, Hv)))
    state = 0.1 * mx.random.normal((1, Hv, Dv, Dk))
    return q, k, v, g, beta, state


@pytest.mark.parametrize("T", [5, 64, 300])
def test_the_chunk_parallel_recurrence_equals_the_stock_one(T: int) -> None:
    """Same recurrence, computed 64 steps at a time with matrix products (the WY
    form): outputs, final state and every gradient agree to float32 precision."""
    import mlx_lm.models.gated_delta as gated_delta

    args = _model_like_inputs(T)

    def loss(fn, q, k, v, g, beta, state):
        y, s = fn(q, k, v, g, beta, state)
        return (y.astype(mx.float32) * mx.cos(mx.arange(y.size).reshape(y.shape))).sum() + (
            s * s
        ).sum()

    stock_y, stock_s = gated_delta.gated_delta_ops(*args)
    wy_y, wy_s = mlx_memory.chunk_parallel_gated_delta_ops(*args)
    assert float(mx.abs(stock_y - wy_y).max()) <= 1e-4 * max(1.0, float(mx.abs(stock_y).max()))
    assert float(mx.abs(stock_s - wy_s).max()) <= 1e-4 * max(1.0, float(mx.abs(stock_s).max()))
    argnums = (0, 1, 2, 3, 4, 5)
    stock = mx.value_and_grad(partial(loss, gated_delta.gated_delta_ops), argnums=argnums)(*args)
    wy = mx.value_and_grad(
        partial(loss, mlx_memory.chunk_parallel_gated_delta_ops), argnums=argnums
    )(*args)
    for left, right in zip(stock[1], wy[1]):
        scale = max(1.0, float(mx.abs(left).max()))
        assert float(mx.abs(left - right).max()) <= 1e-3 * scale


@pytest.mark.parametrize("decay_scale", [1.0, 40.0])
def test_the_chunk_parallel_update_is_stable_where_the_decay_vanishes(decay_scale: float) -> None:
    """Training differentiates the decay's own parameters (A_log, a, dt_bias). With
    a strong decay (g ~ e^-100, below float32) a log taken of g would divide the
    gradient by ~0; the update-level path takes log g = -exp(A_log) softplus(a +
    dt_bias) directly, so every gradient stays finite and matches the stock path."""
    import mlx_lm.models.gated_delta as gated_delta

    mx.random.seed(5)
    T, Hk, Hv, Dk, Dv = 150, 2, 4, 16, 8
    inv = Dk**-0.5
    q = (inv**2) * mx.fast.rms_norm(mx.random.normal((1, T, Hk, Dk)), None, 1e-6)
    k = inv * mx.fast.rms_norm(mx.random.normal((1, T, Hk, Dk)), None, 1e-6)
    v = mx.random.normal((1, T, Hv, Dv))
    a = mx.random.normal((1, T, Hv))
    b = mx.random.normal((1, T, Hv))
    A_log = mx.log(mx.full((Hv,), decay_scale))
    dt_bias = mx.zeros((Hv,))

    def loss(fn, q, k, v, a, b, A_log, dt_bias):
        y, s = fn(q, k, v, a, b, A_log, dt_bias, None, None, use_kernel=False)
        return (y.astype(mx.float32) * mx.cos(mx.arange(y.size).reshape(y.shape))).sum() + (
            s * s
        ).sum()

    argnums = (0, 1, 2, 3, 4, 5, 6)
    stock = mx.value_and_grad(partial(loss, gated_delta.gated_delta_update), argnums=argnums)(
        q, k, v, a, b, A_log, dt_bias
    )
    wy = mx.value_and_grad(
        partial(loss, mlx_memory.chunk_parallel_gated_delta_update), argnums=argnums
    )(q, k, v, a, b, A_log, dt_bias)
    assert abs(float(stock[0]) - float(wy[0])) <= 1e-4 * max(1.0, abs(float(stock[0])))
    for left, right in zip(stock[1], wy[1]):
        assert bool(mx.all(mx.isfinite(right)))
        scale = max(1.0, float(mx.abs(left).max()))
        assert float(mx.abs(left - right).max()) <= 1e-3 * scale


def test_installing_patches_the_models_own_reference_and_is_reversible() -> None:
    import mlx_lm.models.gated_delta as gated_delta
    import mlx_lm.models.qwen3_5 as qwen3_5

    before = (gated_delta.gated_delta_ops, qwen3_5.gated_delta_update)
    with mlx_memory.chunked_recurrence():
        assert qwen3_5.gated_delta_update is not before[1]
    assert (gated_delta.gated_delta_ops, qwen3_5.gated_delta_update) == before


def test_nearly_identical_keys_do_not_break_the_chunk_parallel_update() -> None:
    """Repeated text gives nearly identical keys; with a strong update and weak
    decay the lower-triangular system is ill-suited to repeated squaring (the
    powers of L grow like binomial coefficients and cancel) -- the first real
    preflight (28 Sep 2026) got a non-finite adapter from exactly this. Forward
    substitution is the recurrence itself, and must match the stock path."""
    import mlx_lm.models.gated_delta as gated_delta

    mx.random.seed(11)
    T, Hk, Hv, Dk, Dv = 256, 2, 4, 128, 32
    inv = Dk**-0.5
    base = mx.random.normal((1, 1, Hk, Dk))
    k = inv * mx.fast.rms_norm(base + 0.05 * mx.random.normal((1, T, Hk, Dk)), None, 1e-6)
    q = (inv**2) * mx.fast.rms_norm(mx.random.normal((1, T, Hk, Dk)), None, 1e-6)
    v = mx.random.normal((1, T, Hv, Dv))
    a = mx.full((1, T, Hv), -6.0)
    b = mx.full((1, T, Hv), 4.0)
    A_log, dt_bias = mx.zeros((Hv,)), mx.zeros((Hv,))

    def loss(fn, q, k, v, a, b):
        y, s = fn(q, k, v, a, b, A_log, dt_bias, None, None, use_kernel=False)
        return (y.astype(mx.float32) ** 2).sum() + (s * s).sum()

    stock = mx.value_and_grad(
        partial(loss, gated_delta.gated_delta_update), argnums=(0, 1, 2, 3, 4)
    )(q, k, v, a, b)
    wy = mx.value_and_grad(
        partial(loss, mlx_memory.chunk_parallel_gated_delta_update), argnums=(0, 1, 2, 3, 4)
    )(q, k, v, a, b)
    assert abs(float(stock[0]) - float(wy[0])) <= 1e-3 * abs(float(stock[0]))
    for left, right in zip(stock[1], wy[1]):
        assert bool(mx.all(mx.isfinite(right)))
        assert float(mx.abs(left - right).max()) <= 1e-2 * max(1.0, float(mx.abs(left).max()))
