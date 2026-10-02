"""The prompt screen: which system prompt the law-and-article loop is fixed with.

User decision (1 Oct 2026): before the loop's work starts, candidate prompts
are tried on small subsets, the best one is chosen, and everything is fixed
with it. Two stages, both written down before anything ran (ner-project
`docs/superpowers/plans/2026-10-01-kanun-madde-only.md` §8):

1. Elimination (this module, `stage_one`): the untrained base model reads 100
   training documents under every registered prompt (`g0.PROMPTS`). Training
   documents, because the base model has never been trained on them and the
   validation split is not spent. Generation and scoring are exactly
   validation's: the same worker (`mlx_worker`, window fallback included) and
   the canonical evaluator on GT v7.
2. Short fine-tunes (`loop-train --round 0 --stop-at 150` under a candidate's
   own config): the current prompt and the two best others, each on the whole
   training split with its own full-epoch schedule cut at update 150, compared
   on the 50 validation documents. The current prompt's run is M000 v2 itself.

Selection: F1 at update 150; ties by recall, then the shorter prompt. The test
split stays sealed throughout.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic, write_jsonl_atomic
from .constants import EXEMPLAR_DOC_IDS
from .errors import GateBlocked
from .fingerprints import sha256_file
from .g0 import PROMPT_VARIANT, PROMPTS
from .heartbeat import RunLease
from .loop_sources import validate_loop_sources

SAMPLE_NAMESPACE = "prompt-screen-v1"
SAMPLE_SIZE = 100
SHORT_RUN_UPDATES = 150


def screen_documents(split: dict[str, list[int]], *, size: int = SAMPLE_SIZE) -> list[int]:
    """A fixed sample of training documents, chosen by hash, never by content."""

    pool = [doc_id for doc_id in split["train"] if doc_id not in EXEMPLAR_DOC_IDS]
    ranked = sorted(pool, key=lambda d: hashlib.sha256(f"{SAMPLE_NAMESPACE}:{d}".encode()).hexdigest())
    return sorted(ranked[:size])


def _rows(documents: dict[int, dict[str, Any]], doc_ids: list[int], system_prompt: str) -> list[dict[str, Any]]:
    rows = []
    for doc_id in doc_ids:
        target = [
            {field: reference.get(field, "") for field in ("kanun_no", "kanun_ad", "madde")}
            for reference in documents[doc_id]["references"]
        ]
        rows.append({"messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": documents[doc_id]["text"]},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))},
        ]})
    return rows


def rank(results: dict[str, dict[str, Any]]) -> list[str]:
    """Best first: F1, then recall, then the shorter prompt."""

    return sorted(results, key=lambda v: (-results[v]["f1"], -results[v]["recall"], len(PROMPTS[v])))


def stage_one(config: Any, *, base_run_id: str, worker=None, evaluator=None) -> dict[str, Any]:
    from .g0_training import (
        MAX_GENERATION_TOKENS,
        WINDOW_FALLBACK_MAX_GENERATION_TOKENS,
        WINDOW_FALLBACK_OVERLAP_TOKENS,
        WINDOW_FALLBACK_TOKENS,
        _base_run_context,
        canonical_evaluate,
    )
    from .mlx_compute import _run_worker

    worker = worker or _run_worker
    evaluator = evaluator or canonical_evaluate
    source = validate_loop_sources(config)
    doc_ids = screen_documents(source["split"])
    context = _base_run_context(config, base_run_id)
    root = Path(config.public_root) / "loop" / "prompt_screen" / "stage1"
    results: dict[str, dict[str, Any]] = {}
    for variant, system_prompt in PROMPTS.items():
        out = root / variant
        done = out / "result.json"
        if done.is_file():
            results[variant] = json.loads(done.read_text(encoding="utf-8"))
            continue
        out.mkdir(parents=True, exist_ok=True)
        data, ids = out / "screen.jsonl", out / "screen_doc_ids.json"
        write_jsonl_atomic(data, _rows(source["documents"], doc_ids, system_prompt))
        write_json_atomic(ids, doc_ids)
        request = {
            "action": "validate",
            "model_path": context["run_config"]["model_snapshot"]["snapshot_path"],
            "data_path": str(data),
            "data_sha256": sha256_file(data),
            "doc_ids_path": str(ids),
            "doc_ids_sha256": sha256_file(ids),
            "expected_count": len(doc_ids),
            "output_dir": str(out),
            "max_sequence_length": context["run_config"]["selected_sequence_length"],
            "max_generation_tokens": MAX_GENERATION_TOKENS,
            "window_fallback_enabled": True,
            "window_fallback_tokens": WINDOW_FALLBACK_TOKENS,
            "window_fallback_overlap_tokens": WINDOW_FALLBACK_OVERLAP_TOKENS,
            "window_fallback_max_generation_tokens": WINDOW_FALLBACK_MAX_GENERATION_TOKENS,
            "window_fallback_repetition_recovery_enabled": True,
            "worker_implementation_sha256": sha256_file(Path(__file__).with_name("mlx_worker.py")),
        }
        lease = RunLease(
            lock_path=root / "screen.lock",
            heartbeat_path=out / "heartbeat.json",
            purpose="prompt-screen-stage1",
            run_id=f"prompt-screen:{variant}",
            input_fingerprint=sha256_file(data),
            config_fingerprint=str(getattr(config, "fingerprint", "")),
        ).start(stage=variant)
        try:
            produced = worker(compute_root=out, stage=f"screen_{variant}", request=request, lease=lease,
                              timeout_seconds=14_400)
            report = evaluator(config=config, predictions_path=Path(produced["predictions_path"]),
                               doc_ids_path=ids, output_dir=out, expected_doc_count=len(doc_ids))
        except BaseException as exc:
            lease.finish(status="failed", error=str(exc))
            raise
        lease.finish(status="completed")
        core = report["results"][0]["core_law_article_strict"]
        results[variant] = {
            "variant": variant,
            "prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            "prompt_characters": len(system_prompt),
            **{key: core[key] for key in ("tp", "fp", "fn", "precision", "recall", "f1")},
            "docwise_accuracy": report["results"][0]["docwise_core_accuracy"]["accuracy"],
            "documents": len(doc_ids),
        }
        write_json_atomic(done, results[variant], mode=0o644)
    order = rank(results)
    others = [variant for variant in order if variant != PROMPT_VARIANT]
    selection = {
        "schema_version": 1,
        "stage": 1,
        "model": "untrained base (zero-shot)",
        "documents": "training split, 100 by sha256(prompt-screen-v1:id), exemplars excluded",
        "doc_ids": doc_ids,
        "ranking": [results[variant] for variant in order],
        "to_stage_two": [PROMPT_VARIANT, *others[:2]],
        "rule": "F1, then recall, then the shorter prompt; the current prompt always goes on",
        "generated_at": datetime.now(UTC).isoformat(),
    }
    write_json_atomic(root / "selection.json", selection, mode=0o644)
    return selection


def screen_config_payload(base: dict[str, Any], variant: str) -> dict[str, Any]:
    """A stage-two config: the loop's own, with the candidate prompt and its own roots."""

    if variant not in PROMPTS:
        raise GateBlocked(f"unknown prompt variant {variant!r}")
    payload = json.loads(json.dumps(base))
    payload["sensitive_root"] = f"{base['sensitive_root']}_prompt_{variant}"
    payload["public_root"] = f"{base['public_root']}_prompt_{variant}"
    payload["loop"] = {**base["loop"], "prompt_variant": variant}
    return payload


def write_stage_two_configs(config_path: Path, selection: dict[str, Any]) -> list[Path]:
    """One config per stage-two candidate other than the current prompt, beside the loop's."""

    base = json.loads(Path(config_path).read_text(encoding="utf-8"))
    written = []
    for variant in selection["to_stage_two"]:
        if variant == PROMPT_VARIANT:
            continue
        path = Path(config_path).with_name(f"{Path(config_path).stem}_prompt_{variant}.json")
        write_json_atomic(path, screen_config_payload(base, variant), mode=0o644)
        written.append(path)
    return written


def _curve(candidate_root: Path) -> dict[int, dict[str, Any]]:
    """{update: {"f1", "recall", "per_document"}} from a run's validation summaries."""

    curve = {}
    for path in sorted(candidate_root.glob("validation/update_*/evaluation.json")):
        result = json.loads(path.read_text(encoding="utf-8"))["results"][0]
        core = result["core_law_article_strict"]
        curve[int(path.parent.name.split("_")[1])] = {
            "f1": core["f1"],
            "recall": core["recall"],
            "per_document": {
                int(e["doc_id"]): {"tp": int(e["tp"]), "fp": int(e["fp"]), "fn": int(e["fn"])}
                for e in result["core_per_doc_metrics"]
            },
        }
    return curve


def stage_two(runs: dict[str, Path], *, output: Path, at: int = SHORT_RUN_UPDATES) -> dict[str, Any]:
    """Compare the stage-two runs at update `at` (pre-registered) and pick the winner.

    `runs` maps each prompt variant to its run's candidate root (the current
    prompt's is M000 v2's). Primary: F1 at `at`; ties by recall, then the
    shorter prompt. Secondary, reported: the best F1 up to `at`, and each
    candidate against the current prompt by the published paired bootstrap.
    """

    from .loop_metrics import paired_delta

    if PROMPT_VARIANT not in runs:
        raise GateBlocked("stage two compares against the current prompt; its run is missing")
    curves = {variant: _curve(root) for variant, root in runs.items()}
    for variant, curve in curves.items():
        if at not in curve:
            raise GateBlocked(f"{variant} has no validation at update {at}")
    results = {
        variant: {
            "variant": variant,
            "f1": curve[at]["f1"],
            "recall": curve[at]["recall"],
            "best_f1_up_to": max(point["f1"] for update, point in curve.items() if update <= at),
            "curve": {update: round(point["f1"], 4) for update, point in sorted(curve.items()) if update <= at},
        }
        for variant, curve in curves.items()
    }
    current = curves[PROMPT_VARIANT][at]["per_document"]
    for variant in results:
        if variant != PROMPT_VARIANT:
            results[variant]["vs_current"] = paired_delta(current, curves[variant][at]["per_document"])
    order = rank(results)
    payload = {
        "schema_version": 1,
        "stage": 2,
        "at_update": at,
        "documents": "validation split (50), GT v7",
        "ranking": [results[variant] for variant in order],
        "winner": order[0],
        "rule": "F1 at update %d, then recall, then the shorter prompt" % at,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    write_json_atomic(output, payload, mode=0o644)
    return payload
