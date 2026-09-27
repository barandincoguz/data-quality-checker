"""Train whole documents on this machine: the same arithmetic in less memory.

Measured on Qwen3.5-9B-MLX-4bit, one LoRA training step, per-layer gradient
checkpointing on, compiled step (27 September 2026):

    tokens   stock recurrence            chunked recurrence
    1,024    34.4 GB   21 s              --
    1,536    66.6 GB   49 s              19.8 GB   28 s
    4,096    112.9 GB  (past RAM, swap)  44.1 GB   83 s
    8,192    --                          79 GB     ~400 s (with the chunked loss
                                                   and the cache limit below)

`mlx_lm` trains the model's 24 linear-attention (Gated DeltaNet) layers
through `gated_delta_ops`, a per-token Python recurrence: the fast Metal kernel
has no backward, so `GatedDeltaNet` sets `use_kernel=not self.training`. Its
backward keeps the 32x128x128 state of every step of the layer being
differentiated, so memory grew faster than the sequence and training was
locked at 1,536 tokens, with long documents cut into windows. (Per-layer
checkpointing does help -- 86.7 GB without it at 1,024 tokens -- but it cannot
reach inside one layer's recurrence.)

`chunked_gated_delta_ops` runs the same recurrence in checkpointed chunks: the
state is carried between chunks, and a chunk's steps are recomputed during its
own backward. Outputs and gradients are bit-identical to the stock path
(`tests/test_mlx_memory.py`). `chunked_completion_loss` computes the logits of
the 248,320-word vocabulary a chunk of positions at a time -- the same sum of
per-token cross entropies. `metal_cache_limit` releases freed buffers so a long
step stays under Metal's buffer-count limit.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any

RECURRENCE_CHUNK = 128
LOSS_CHUNK = 1024


def chunked_gated_delta_ops(
    q: Any,
    k: Any,
    v: Any,
    g: Any,
    beta: Any,
    state: Any = None,
    mask: Any = None,
    *,
    chunk: int = RECURRENCE_CHUNK,
) -> tuple[Any, Any]:
    """`mlx_lm.models.gated_delta.gated_delta_ops`, in checkpointed chunks."""
    import mlx.core as mx
    import mlx_lm.models.gated_delta as gated_delta

    stock = _STOCK.get("ops") or gated_delta.gated_delta_ops
    B, T, _, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    if T <= chunk:
        return stock(q, k, v, g, beta, state, mask)

    def run(*args: Any) -> tuple[Any, Any]:
        qc, kc, vc, gc, bc, st, *rest = args
        return stock(qc, kc, vc, gc, bc, st, rest[0] if rest else None)

    run_checkpointed = mx.checkpoint(run)
    outputs = []
    for start in range(0, T, chunk):
        end = min(T, start + chunk)
        args = [
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            state,
        ]
        if mask is not None:
            args.append(mask[:, start:end])
        y, state = run_checkpointed(*args)
        outputs.append(y)
    return mx.concatenate(outputs, axis=1), state


_STOCK: dict[str, Any] = {}


@contextmanager
def chunked_recurrence(chunk: int = RECURRENCE_CHUNK) -> Iterator[None]:
    """Route `mlx_lm`'s training recurrence through the chunked version."""
    import mlx_lm.models.gated_delta as gated_delta

    stock = gated_delta.gated_delta_ops
    _STOCK["ops"] = stock
    gated_delta.gated_delta_ops = partial(chunked_gated_delta_ops, chunk=chunk)
    try:
        yield
    finally:
        gated_delta.gated_delta_ops = stock
        _STOCK.pop("ops", None)


def install_chunked_recurrence(chunk: int = RECURRENCE_CHUNK) -> None:
    """For a worker process that trains once and exits: install, never restore."""
    import mlx_lm.models.gated_delta as gated_delta

    _STOCK.setdefault("ops", gated_delta.gated_delta_ops)
    gated_delta.gated_delta_ops = partial(chunked_gated_delta_ops, chunk=chunk)


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
