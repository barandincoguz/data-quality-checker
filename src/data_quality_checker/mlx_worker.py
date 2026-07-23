"""Isolated heavyweight MLX actions used by the compute preflight.

Every action runs in a child process so an allocator failure cannot corrupt the
parent's lifecycle state. Requests and results are JSON files, making failures
inspectable and resumable after terminal or process loss.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .fingerprints import fingerprint_json, sha256_file


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_projection(checkpoint: Path) -> dict[str, Any]:
    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    return {
        "adapter_sha256": sha256_file(checkpoint / "adapters.safetensors"),
        "optimizer_sha256": sha256_file(checkpoint / "optimizer.safetensors"),
        "mlx_rng_sha256": sha256_file(checkpoint / "mlx_rng.safetensors"),
        "scheduler_state": trainer_state["scheduler_state"],
        "data_cursor": trainer_state["data_cursor"],
        "python_rng_state_fingerprint": fingerprint_json(
            trainer_state["python_rng_state"]
        ),
    }


def _download(request: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id=str(request["model_id"]),
        revision=str(request["revision"]),
        cache_dir=str(request["cache_dir"]),
    )
    return {"snapshot_path": str(Path(snapshot).resolve())}


def _train(request: dict[str, Any]) -> dict[str, Any]:
    import mlx.core as mx

    from .mlx_stateful import StatefulMlxTrainer, StatefulTrainingConfig

    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    mx.reset_peak_memory()
    config = StatefulTrainingConfig(**request["training_config"])
    trainer = StatefulMlxTrainer(
        model_path=Path(request["model_path"]),
        train_path=Path(request["train_path"]),
        checkpoint_root=Path(request["checkpoint_root"]),
        config=config,
        input_fingerprint=str(request["input_fingerprint"]),
        model_fingerprint=str(request["model_fingerprint"]),
    )
    if request.get("resume_checkpoint"):
        trainer.resume(Path(request["resume_checkpoint"]))
    trained = trainer.train(target_updates=int(request["target_updates"]))
    checkpoint = Path(trained["checkpoint"])
    if not trainer.adapter_finite(checkpoint):
        raise RuntimeError("adapter contains non-finite tensors")
    return {
        **trained,
        "adapter_finite": True,
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "state_projection": _checkpoint_projection(checkpoint),
    }


def _generate(request: dict[str, Any]) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    from .processing import _parse_model_output

    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    mx.reset_peak_memory()
    model, tokenizer = load(
        str(request["model_path"]), adapter_path=str(request["adapter_path"])
    )
    prompt = tokenizer.apply_chat_template(
        request["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(prompt, str):
        raise RuntimeError("chat template did not render a string prompt")
    pieces: list[str] = []
    output_tokens = 0
    finish_reason = ""
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=int(request.get("max_tokens", 512)),
        sampler=make_sampler(temp=0.0),
    ):
        pieces.append(response.text)
        output_tokens = int(response.generation_tokens)
        finish_reason = str(response.finish_reason or finish_reason)
    raw = "".join(pieces).strip()
    parsed = _parse_model_output(raw)
    return {
        "adapter_loaded": True,
        "raw_output": raw,
        "parsed_references": parsed,
        "input_tokens": len(tokenizer.encode(prompt)),
        "output_tokens": output_tokens,
        "finish_reason": finish_reason,
        "peak_memory_bytes": int(mx.get_peak_memory()),
    }


def dispatch(request: dict[str, Any]) -> dict[str, Any]:
    action = request.get("action")
    if action == "download":
        return _download(request)
    if action == "train":
        return _train(request)
    if action == "generate":
        return _generate(request)
    raise ValueError(f"unsupported worker action: {action!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dqcheck-mlx-worker")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    base = {
        "schema_version": 1,
        "action": request.get("action"),
        "request_fingerprint": request.get("request_fingerprint"),
        "started_at": _utc_now(),
        "pid": os.getpid(),
        "host": socket.gethostname(),
    }
    write_json_atomic(args.result, {**base, "status": "running"})
    try:
        details = dispatch(request)
        result = {
            **base,
            "status": "passed",
            "finished_at": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            **details,
        }
        write_json_atomic(args.result, result)
        return 0
    except BaseException as exc:
        result = {
            **base,
            "status": "failed",
            "finished_at": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json_atomic(args.result, result)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
