"""Re-score a run's validation checkpoints under this config's ground truth.

A run's own evaluations are never rewritten. When the ground truth changes
(GT v7 → v8, 2 Oct 2026) a comparison a decision rests on must score both of
its sides the same way, so this writes the run's checkpoints, scored again,
beside the config's own records: `loop/rescored/<label>/update_NNNNNNN/` with
each checkpoint's evaluation, and `summary.json` with the run's own (old) and
the new counts. The predictions scored are the run's, byte for byte (their
sha256 is recorded); the evaluator and the reference policy are the loop's
(`g0_training.canonical_evaluate`).
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic
from .errors import GateBlocked, IntegrityError
from .fingerprints import sha256_file

CORE = ("tp", "fp", "fn", "precision", "recall", "f1")


def _doc_ids(candidate: Path) -> Path:
    for request in sorted(
        candidate.glob("workers/validate_*/attempt_*/request.json"), reverse=True
    ):
        payload = json.loads(request.read_text(encoding="utf-8"))
        path = Path(payload["doc_ids_path"])
        if sha256_file(path) != payload["doc_ids_sha256"]:
            raise IntegrityError(f"validation doc ids drifted: {path}")
        return path
    raise GateBlocked(f"no validation request under {candidate}")


def _commit(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def rescore_run(
    config: Any, *, run_id: str, label: str, reason: str, evaluator=None
) -> dict[str, Any]:
    from .g0_training import canonical_evaluate
    from .loop_ops import LOOP_CANDIDATE

    evaluator = evaluator or canonical_evaluate
    candidate = Path(config.training_runs_root) / run_id / "development" / LOOP_CANDIDATE
    checkpoints = sorted(candidate.glob("validation/update_*"))
    if not checkpoints:
        raise GateBlocked(f"{run_id} has no validation checkpoints")
    out = Path(config.public_root) / "loop" / "rescored" / label
    if (out / "summary.json").exists():
        raise GateBlocked(f"{out} exists; a rescore is written once")
    doc_ids = _doc_ids(candidate)
    expected = len(json.loads(doc_ids.read_text(encoding="utf-8")))
    rows = []
    for directory in checkpoints:
        predictions = directory / "predictions.json"
        old = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))["results"][0]
        new = evaluator(
            config=config,
            predictions_path=predictions,
            doc_ids_path=doc_ids,
            output_dir=out / directory.name,
            expected_doc_count=expected,
        )["results"][0]
        rows.append(
            {
                "checkpoint": directory.name,
                "predictions_sha256": sha256_file(predictions),
                **{f"old_{k}": old["core_law_article_strict"][k] for k in CORE},
                **{f"new_{k}": new["core_law_article_strict"][k] for k in CORE},
            }
        )
    state = json.loads((candidate / "state.json").read_text(encoding="utf-8"))
    best = max(rows, key=lambda r: (r["new_f1"], r["new_recall"]))
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "label": label,
        "reason": reason,
        "ground_truth": str(config.canonical_gt_dir),
        "gt_manifest_sha256": config.raw.get("loop", {}).get("gt_manifest_sha256", ""),
        "evaluator_commit": _commit(Path(__file__).resolve().parents[2]),
        "documents": expected,
        "selected_before": f"update_{int(state['best_update']):07d}"
        if state.get("best_update")
        else "",
        "selected_by_new_f1_for_information_only": best["checkpoint"],
        "generated_at": datetime.now(UTC).isoformat(),
        "rows": rows,
    }
    write_json_atomic(out / "summary.json", summary, mode=0o644)
    return summary
