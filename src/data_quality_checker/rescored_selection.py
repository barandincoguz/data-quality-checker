"""Select a run's checkpoint from its rescored validation curve.

A run's own selection (`g0_finalize.finalize_selection`) reads the summaries
written while it trained, each scored by the evaluator of that moment. When a
comparison must score its sides under one engine -- M001_raw, trained before
the engine proposal is applied (pre-registered 5 Oct 2026, ner
`Journal/experiments/dq_loop_gt750/ablation_noisy_vs_clean/preregistration.json`,
amendment 2) -- `loop-rescore` scores every point again, and this module selects
on that curve with the trainer's own code: each point's summary is rebuilt by
`g0_training._validation_summary` from the run's own validation record (the
worker's counts and loss, which no engine touches) and the rescored evaluation,
so eligibility -- the collapse gate reads the evaluation -- is applied anew;
then `g0.select_checkpoint` picks. The run's files are only read; the selection
is written beside the rescore, never into the run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .errors import GateBlocked, IntegrityError
from .fingerprints import sha256_file
from .g0 import CheckpointCandidate, select_checkpoint
from .g0_training import _validation_summary

LOOP_CANDIDATE = "loop-lr2.5e-5-full-epoch"


def rebuilt_summary(
    candidate_root: Path, update: int, evaluation: dict[str, Any], *, evaluation_sha256: str
) -> dict[str, Any]:
    """The summary the trainer would have written for `update` with `evaluation`."""
    name = f"update_{update:07d}"
    directory = candidate_root / "validation" / name
    checkpoint = candidate_root / "checkpoints" / name
    manifest_path = directory / "manifest.json"
    stored = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != stored["validation_manifest_sha256"]:
        raise IntegrityError(
            f"{name}: the validation record differs from the one its summary sealed"
        )
    validation = {
        **json.loads(manifest_path.read_text(encoding="utf-8")),
        "manifest_sha256": manifest_sha256,
    }
    checkpoint_manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    return _validation_summary(
        update=update,
        checkpoint=Path(stored["checkpoint"]),
        checkpoint_manifest=checkpoint_manifest,
        validation=validation,
        evaluation=evaluation,
        evaluation_sha256=evaluation_sha256,
    )


def _candidate(summary: dict[str, Any]) -> CheckpointCandidate:
    return CheckpointCandidate(
        update=int(summary["update"]),
        coverage_count=int(summary["coverage_count"]),
        parse_count=int(summary["parse_count"]),
        empty_output_count=int(summary["empty_output_count"]),
        runaway_output_count=int(summary["runaway_output_count"]),
        core_f1=float(summary["core_law_article_strict"]["f1"]),
        docwise_accuracy=float(summary["docwise_core_accuracy"]["accuracy"]),
        recall=float(summary["core_law_article_strict"]["recall"]),
        validation_loss=float(summary["validation_loss"]),
    )


def select_rescored_checkpoint(config: Any, *, run_id: str, label: str) -> dict[str, Any]:
    candidate_root = Path(config.training_runs_root) / run_id / "development" / LOOP_CANDIDATE
    rescored = Path(config.public_root) / "loop" / "rescored" / label
    summary = json.loads((rescored / "summary.json").read_text(encoding="utf-8"))
    if summary.get("run_id") != run_id:
        raise IntegrityError(
            f"rescore {label!r} is of run {summary.get('run_id')!r}, not {run_id!r}"
        )
    rows = {row["checkpoint"]: row for row in summary["rows"]}
    points = sorted(path.name for path in candidate_root.glob("validation/update_*"))
    if not points:
        raise GateBlocked(f"{run_id} has no validation checkpoints")
    missing = [name for name in points if name not in rows]
    if missing:
        raise GateBlocked(f"rescore {label!r} lacks {missing}; rescore the finished run")

    rebuilt: list[dict[str, Any]] = []
    for name in points:
        predictions = candidate_root / "validation" / name / "predictions.json"
        if sha256_file(predictions) != rows[name]["predictions_sha256"]:
            raise IntegrityError(
                f"{name}: rescore {label!r} scored other predictions than the run's"
            )
        evaluation_path = rescored / name / "evaluation.json"
        rebuilt.append(
            rebuilt_summary(
                candidate_root,
                int(name.removeprefix("update_")),
                json.loads(evaluation_path.read_text(encoding="utf-8")),
                evaluation_sha256=sha256_file(evaluation_path),
            )
        )
    selected = select_checkpoint([_candidate(s) for s in rebuilt if s["eligible"] is True])
    # The run's own choice, by the same rule on the summaries it wrote while training.
    stored = [
        json.loads(
            (candidate_root / "validation" / name / "summary.json").read_text(encoding="utf-8")
        )
        for name in points
    ]
    try:
        in_run: int | None = select_checkpoint(
            [_candidate(s) for s in stored if s.get("eligible") is True]
        ).update
    except GateBlocked:
        in_run = None
    selection = {
        "schema_version": 1,
        "run_id": run_id,
        "rescore_label": label,
        "evaluator_commit": summary.get("evaluator_commit", ""),
        "rule": "g0.select_checkpoint on summaries rebuilt by g0_training._validation_summary",
        "selected_update": selected.update,
        "core_f1": selected.core_f1,
        "recall": selected.recall,
        "docwise_accuracy": selected.docwise_accuracy,
        "adapter_path": str(candidate_root / "checkpoints" / f"update_{selected.update:07d}"),
        "in_run_selected_update": in_run,
        "points": [
            {
                "checkpoint": f"update_{int(s['update']):07d}",
                "eligible": s["eligible"],
                "core_f1": s["core_law_article_strict"]["f1"],
                "recall": s["core_law_article_strict"]["recall"],
                "docwise_accuracy": s["docwise_core_accuracy"]["accuracy"],
            }
            for s in rebuilt
        ],
    }
    write_json_atomic(rescored / "selection.json", selection, mode=0o644)
    return selection
