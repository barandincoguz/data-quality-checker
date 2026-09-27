"""Train whole documents on this machine: the same mathematics in far less memory.

`mlx_lm` trains the model's 24 linear-attention (Gated DeltaNet) layers
through `gated_delta_ops`, a per-token Python recurrence: the fast Metal kernel
has no backward, so `GatedDeltaNet` sets `use_kernel=not self.training`. Every
token is a dozen small arrays per layer, and the backward keeps each step's
32x128x128 state. Measured on Qwen3.5-9B-MLX-4bit, one LoRA step, per-layer
gradient checkpointing on (27 September 2026): 1,536 tokens 66.6 GB, 4,096
tokens 112.9 GB -- which is why training was locked at 1,536 tokens and long
documents were cut into windows. Checkpointing the per-token loop in chunks
lowered MLX's own count (8,192 tokens: 79 GB) but not the process: the arrays'
own overhead took it to a 120 GB footprint and into swap, and a compiled step
kept one such graph per padded length until Metal ran out of resources.

`chunk_parallel_gated_delta_update` computes the same recurrence 64 steps at a
time with matrix products (the WY form used by Gated DeltaNet/FLA), carrying
only the state across chunks. One layer, 2,048 tokens, real sizes: 0.2 s and
2.1 GB against 1.5 s and 22.1 GB. The decay enters in log space straight from
the model's parameters, so its gradient stays finite where the decay itself
underflows. `chunked_completion_loss` computes the 248,320-word vocabulary
logits a chunk of positions at a time -- the same sum of per-token cross
entropies. `metal_cache_limit` releases freed buffers. Agreement with the stock
path is pinned by `tests/test_mlx_memory.py`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

LOSS_CHUNK = 1024


def chunked_completion_loss(
    hidden: Any, head: Callable[[Any], Any], targets: Any, mask: Any, *, chunk: int = LOSS_CHUNK
) -> Any:
    """Sum of masked token cross entropies, the vocabulary logits a chunk at a time."""
    import mlx.core as mx
    import mlx.nn as nn

    def piece(h: Any, t: Any, m: Any) -> Any:
        return (nn.losses.cross_entropy(head(h), t) * m).astype(mx.float32).sum()

    piece_checkpointed = mx.checkpoint(piece)
    total = mx.array(0.0, dtype=mx.float32)
    for start in range(0, targets.shape[1], chunk):
        end = min(targets.shape[1], start + chunk)
        total = total + piece_checkpointed(
            hidden[:, start:end], targets[:, start:end], mask[:, start:end]
        )
    return total


def hidden_and_head(model: Any) -> tuple[Callable[[Any], Any], Callable[[Any], Any]]:
    """The model's final hidden states and its vocabulary head, taken apart."""
    text = getattr(model, "language_model", model)
    inner = getattr(text, "model", None)
    args = getattr(text, "args", None)
    if inner is None or args is None:
        from .errors import ContractError

        raise ContractError("model does not expose hidden states and a head for the chunked loss")
    head = (
        inner.embed_tokens.as_linear
        if getattr(args, "tie_word_embeddings", False)
        else text.lm_head
    )
    return inner, head


def metal_cache_limit(limit_bytes: int = 0) -> None:
    import mlx.core as mx

    mx.set_cache_limit(limit_bytes)


WY_CHUNK = 64
# Names the training recurrence a whole-document run was measured and trained
# under; it is part of that run's identity (`loop_train.train_round`).
# v1 inverted by repeated squaring and got a non-finite adapter on real text;
# v2 inverts by forward substitution (28 Sep 2026).
RECURRENCE_ID = "wy_chunk_parallel_v2"
_LOG_FLOOR = 1e-30
# Modules that bind `gated_delta_update` by name at import (qwen3_5 does:
# `from .gated_delta import gated_delta_update`) -- each must be patched.
_UPDATE_HOMES = ("gated_delta", "qwen3_5", "qwen3_next")
_STOCK: dict[str, Any] = {}


def _unit_lower_inverse(L: Any) -> Any:
    """(I + L)^-1 for strictly lower-triangular L, by forward substitution.

    Row i of X = (I + L)^-1 is e_i - L[i, :i] X[:i]: the recurrence itself,
    grouped by chunk, so exactly as stable as it. Repeated squaring,
    (I - L)(I + L^2)(I + L^4)..., is exact in theory but its powers grow like
    binomial coefficients when keys repeat -- the first real preflight got a
    non-finite adapter from it (28 Sep 2026).
    """
    import mlx.core as mx

    C = L.shape[-1]
    eye = mx.broadcast_to(mx.eye(C, dtype=mx.float32), L.shape)
    rows = [eye[..., 0:1, :]]
    for i in range(1, C):
        solved = mx.concatenate(rows, axis=-2)  # X[:i]
        rows.append(eye[..., i : i + 1, :] - L[..., i : i + 1, :i] @ solved)
    return mx.concatenate(rows, axis=-2)


def _chunk_parallel(
    q: Any, k: Any, v: Any, log_g: Any, beta: Any, state: Any, chunk: int
) -> tuple[Any, Any]:
    """The WY-form recurrence. `log_g` is the per-step log decay, [B, T, Hv].

    Per step (state S is Dv x Dk, decay a_t, strength b_t):

        S_t = a_t S_{t-1} + u_t k_t^T,   u_t = b_t (v_t - a_t S_{t-1} k_t),   y_t = S_t q_t

    Inside a chunk starting from S_0, with A_t the decay product up to step t:
    (I + L) U = diag(b) (V - diag(A) K S_0^T),  L_tj = b_t (A_t / A_j)(k_t . k_j)
    for j < t. L is strictly lower triangular: (I + L)^-1 by forward
    substitution (`_unit_lower_inverse`). Then Y = diag(A) Q S_0^T +
    ((Q K^T) * D) U with D_tj = A_t / A_j (j <= t), and S_C = A_C S_0 +
    (U * A_C / A)^T K. Only the state crosses chunk boundaries.
    """
    import mlx.core as mx

    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    if (repeat := Hv // Hk) > 1:
        q = mx.repeat(q, repeat, -2)
        k = mx.repeat(k, repeat, -2)
    out_dtype = q.dtype
    C = chunk
    N = -(-T // C)
    pad = N * C - T

    def blocks(x: Any) -> Any:
        """[B, T, H, ...] -> [B, H, N, C, ...], float32; padded steps are zeros --
        no decay (log 0), no strength, zero vectors: they carry nothing."""
        x = x.astype(mx.float32)
        if pad:
            x = mx.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2))
        x = mx.moveaxis(x, 2, 1)
        return x.reshape(B, x.shape[1], N, C, *x.shape[3:])

    qc, kc, vc, bc = blocks(q), blocks(k), blocks(v), blocks(beta)
    log_a = mx.cumsum(blocks(log_g), axis=-1)  # [B,H,N,C]
    a = mx.exp(log_a)
    idx = mx.arange(C)
    lower = idx[:, None] >= idx[None, :]
    strict = idx[:, None] > idx[None, :]
    decay_lower = mx.exp(mx.where(lower, log_a[..., :, None] - log_a[..., None, :], -1e30))
    decay_strict = mx.where(strict, decay_lower, 0.0)

    L = bc[..., :, None] * (kc @ mx.swapaxes(kc, -1, -2)) * decay_strict
    inverse = _unit_lower_inverse(L)

    u_values = inverse @ (bc[..., None] * vc)  # [B,H,N,C,Dv]
    w_keys = inverse @ (bc[..., None] * a[..., None] * kc)  # [B,H,N,C,Dk]
    q_decay = a[..., None] * qc
    attend = (qc @ mx.swapaxes(kc, -1, -2)) * decay_lower
    tail = mx.exp(log_a[..., -1:] - log_a)  # A_C / A_t
    a_last = a[..., -1]  # [B,H,N]

    s = state.astype(mx.float32)
    outputs = []
    for n in range(N):
        s_t = mx.swapaxes(s, -1, -2)  # [B,H,Dk,Dv]
        u = u_values[:, :, n] - w_keys[:, :, n] @ s_t
        outputs.append(q_decay[:, :, n] @ s_t + attend[:, :, n] @ u)
        s = (
            a_last[:, :, n, None, None] * s
            + mx.swapaxes(u * tail[:, :, n, :, None], -1, -2) @ kc[:, :, n]
        )
    y = mx.concatenate(outputs, axis=2)[:, :, :T]  # [B,H,T,Dv]
    return mx.moveaxis(y, 1, 2).astype(out_dtype), s


def chunk_parallel_gated_delta_ops(
    q: Any,
    k: Any,
    v: Any,
    g: Any,
    beta: Any,
    state: Any = None,
    mask: Any = None,
    *,
    chunk: int = WY_CHUNK,
) -> tuple[Any, Any]:
    """`gated_delta_ops`' signature over the WY form (decay given as g itself).

    For comparison with the stock path; training goes through
    `chunk_parallel_gated_delta_update`, whose log decay has a stable gradient.
    Masked calls and per-key-dimension decay fall back to the stock path.
    """
    import mlx.core as mx
    import mlx_lm.models.gated_delta as gated_delta

    if mask is not None or g.ndim != 3:
        return gated_delta.gated_delta_ops(q, k, v, g, beta, state, mask)
    return _chunk_parallel(
        q, k, v, mx.log(mx.maximum(g.astype(mx.float32), _LOG_FLOOR)), beta, state, chunk
    )


def chunk_parallel_gated_delta_update(
    q: Any,
    k: Any,
    v: Any,
    a: Any,
    b: Any,
    A_log: Any,
    dt_bias: Any,
    state: Any = None,
    mask: Any = None,
    use_kernel: bool = True,
) -> tuple[Any, Any]:
    """`gated_delta_update` for training: the WY form, log decay from the parameters.

    log g = -exp(A_log) softplus(a + dt_bias), taken directly, never log(g): with
    a strong decay g underflows in float32 and a log of it would divide the
    gradient by ~0. Inference (`use_kernel`, the fast Metal kernel) and masked
    calls go to the stock update untouched.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx_lm.models.gated_delta as gated_delta

    stock = _STOCK.get("gated_delta") or gated_delta.gated_delta_update
    if use_kernel or mask is not None or a.ndim != 3:
        return stock(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=use_kernel)
    log_g = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)
    return _chunk_parallel(q, k, v, log_g, mx.sigmoid(b), state, WY_CHUNK)


def _homes() -> dict[str, Any]:
    import importlib

    homes = {}
    for name in _UPDATE_HOMES:
        try:
            homes[name] = importlib.import_module(f"mlx_lm.models.{name}")
        except ImportError:
            continue
    return homes


def install_chunked_recurrence() -> None:
    """Route every model's training recurrence through the WY form (a worker that
    trains once and exits: installed, never restored)."""
    for name, module in _homes().items():
        _STOCK.setdefault(name, module.gated_delta_update)
        module.gated_delta_update = chunk_parallel_gated_delta_update


@contextmanager
def chunked_recurrence() -> Iterator[None]:
    """`install_chunked_recurrence` for a `with` block, restored afterwards."""
    homes = _homes()
    saved = {name: module.gated_delta_update for name, module in homes.items()}
    install_chunked_recurrence()
    try:
        yield
    finally:
        for name, module in homes.items():
            module.gated_delta_update = saved[name]
            _STOCK.pop(name, None)
