"""`loop-rescore`: a run's checkpoints scored again under the config's GT (GT v8, 2 Oct 2026).

Pinned: the run's own evaluations are never touched; every checkpoint is scored
from the run's own predictions with the run's validation doc ids; the summary
carries both the old and the new counts; a rescore is written once.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_quality_checker.errors import GateBlocked
from data_quality_checker.rescore import rescore_run


def _run(tmp_path: Path) -> SimpleNamespace:
    candidate = tmp_path / "runs" / "dqcheck_loop_x" / "development" / "loop-lr2.5e-5-full-epoch"
    ids = tmp_path / "valid_doc_ids.json"
    ids.write_text(json.dumps([1, 2]))
    for update, f1 in ((25, 0.5), (50, 0.75)):
        d = candidate / "validation" / f"update_{update:07d}"
        d.mkdir(parents=True)
        (d / "predictions.json").write_text(json.dumps([{"doc_id": 1, "references": []}]))
        core = {"tp": 1, "fp": 1, "fn": 1, "precision": f1, "recall": f1, "f1": f1}
        (d / "evaluation.json").write_text(
            json.dumps({"results": [{"core_law_article_strict": core}]})
        )
        w = candidate / "workers" / f"validate_{update:07d}" / "attempt_001"
        w.mkdir(parents=True)
        (w / "request.json").write_text(
            json.dumps(
                {
                    "doc_ids_path": str(ids),
                    "doc_ids_sha256": hashlib.sha256(ids.read_bytes()).hexdigest(),
                }
            )
        )
    (candidate / "state.json").write_text(json.dumps({"best_update": 50}))
    return SimpleNamespace(
        training_runs_root=tmp_path / "runs",
        public_root=tmp_path / "public",
        canonical_gt_dir=tmp_path / "gt",
        raw={"loop": {"gt_manifest_sha256": "abc"}},
    )


def test_every_checkpoint_is_scored_again_from_its_own_predictions(tmp_path: Path) -> None:
    config = _run(tmp_path)
    seen = []

    def evaluator(*, config, predictions_path, doc_ids_path, output_dir, expected_doc_count):
        seen.append((predictions_path.parent.name, doc_ids_path.name, expected_doc_count))
        f1 = 0.9 if "0000025" in str(predictions_path) else 0.8
        return {
            "results": [
                {
                    "core_law_article_strict": {
                        "tp": 2,
                        "fp": 0,
                        "fn": 0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": f1,
                    }
                }
            ]
        }

    before = {p: p.read_bytes() for p in (tmp_path / "runs").rglob("*.json")}
    summary = rescore_run(
        config, run_id="dqcheck_loop_x", label="r", reason="why", evaluator=evaluator
    )
    assert seen == [
        ("update_0000025", "valid_doc_ids.json", 2),
        ("update_0000050", "valid_doc_ids.json", 2),
    ]
    assert [(r["old_f1"], r["new_f1"]) for r in summary["rows"]] == [(0.5, 0.9), (0.75, 0.8)]
    assert summary["selected_before"] == "update_0000050"
    assert summary["selected_by_new_f1_for_information_only"] == "update_0000025"
    assert summary["gt_manifest_sha256"] == "abc"
    assert {p: p.read_bytes() for p in (tmp_path / "runs").rglob("*.json")} == before
    assert (
        json.loads((tmp_path / "public/loop/rescored/r/summary.json").read_text())["rows"]
        == summary["rows"]
    )
    with pytest.raises(GateBlocked, match="written once"):
        rescore_run(config, run_id="dqcheck_loop_x", label="r", reason="why", evaluator=evaluator)
