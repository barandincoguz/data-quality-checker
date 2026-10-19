"""Select a run's checkpoint from its rescored validation curve.

A run trained while the main tree's evaluator was the old engine (M001_raw,
5 Oct 2026) is selected under the engine its comparison uses: every point is
rescored (`loop-rescore`), each point's summary is rebuilt by the trainer's own
`_validation_summary` from the run's own validation record and the rescored
evaluation, and the trainer's own `select_checkpoint` picks the checkpoint.
Pre-registered: ner ablation_noisy_vs_clean/preregistration.json, amendment 2.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_quality_checker.errors import GateBlocked, IntegrityError
from data_quality_checker.fingerprints import sha256_file
from data_quality_checker.g0_training import _validation_summary
from data_quality_checker.rescored_selection import select_rescored_checkpoint

CANDIDATE = "loop-lr2.5e-5-full-epoch"


def _evaluation(
    f1: float, *, recall: float | None = None, docwise: float = 0.7, empty_docs: int = 0
) -> dict:
    per_doc = [{"tp": 0, "fp": 0, "fn": 1}] * empty_docs + [{"tp": 1, "fp": 0, "fn": 0}] * (
        50 - empty_docs
    )
    return {
        "results": [
            {
                "core_law_article_strict": {
                    "f1": f1,
                    "recall": f1 if recall is None else recall,
                    "precision": f1,
                    "tp": 1,
                    "fp": 0,
                    "fn": 0,
                },
                "docwise_core_accuracy": {"accuracy": docwise},
                "core_per_doc_metrics": per_doc,
            }
        ]
    }


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(public_root=tmp_path / "public", training_runs_root=tmp_path / "runs")


def _run(
    tmp_path: Path, points: dict[int, dict], *, coverage: dict[int, int] | None = None
) -> Path:
    """A finished run: per update a checkpoint, the worker's validation record, its evaluation and summary."""
    candidate = tmp_path / "runs" / "dqcheck_loop_r" / "development" / CANDIDATE
    for update, evaluation in points.items():
        checkpoint = candidate / "checkpoints" / f"update_{update:07d}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "manifest.json").write_text(
            json.dumps({"trainer_state_fingerprint": f"t{update}"})
        )
        directory = candidate / "validation" / f"update_{update:07d}"
        directory.mkdir(parents=True)
        predictions = directory / "predictions.json"
        predictions.write_text(json.dumps([{"doc_id": update}]))
        validation = {
            "coverage_count": (coverage or {}).get(update, 50),
            "parse_count": 50,
            "empty_output_count": 0,
            "zero_reference_output_count": 0,
            "predicted_reference_count": 9,
            "runaway_output_count": 0,
            "validation_loss": 1.0 / update,
            "predictions_path": str(predictions),
            "predictions_sha256": sha256_file(predictions),
        }
        (directory / "manifest.json").write_text(json.dumps(validation))
        (directory / "evaluation.json").write_text(json.dumps(evaluation))
        summary = _validation_summary(
            update=update,
            checkpoint=checkpoint,
            checkpoint_manifest={"trainer_state_fingerprint": f"t{update}"},
            validation={**validation, "manifest_sha256": sha256_file(directory / "manifest.json")},
            evaluation=evaluation,
            evaluation_sha256=sha256_file(directory / "evaluation.json"),
        )
        (directory / "summary.json").write_text(json.dumps(summary))
    return candidate


def _rescore(tmp_path: Path, label: str, candidate: Path, points: dict[int, dict]) -> Path:
    """What `rescore_run` leaves: per point the new evaluation, and a summary row with the predictions' sha256."""
    out = tmp_path / "public" / "loop" / "rescored" / label
    rows = []
    for update, evaluation in points.items():
        name = f"update_{update:07d}"
        (out / name).mkdir(parents=True)
        (out / name / "evaluation.json").write_text(json.dumps(evaluation))
        rows.append(
            {
                "checkpoint": name,
                "predictions_sha256": sha256_file(
                    candidate / "validation" / name / "predictions.json"
                ),
            }
        )
    (out / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "dqcheck_loop_r",
                "label": label,
                "evaluator_commit": "abc1234",
                "rows": rows,
            }
        )
    )
    return out


def test_the_checkpoint_is_selected_on_the_rescored_curve(tmp_path: Path) -> None:
    candidate = _run(tmp_path, {25: _evaluation(0.80), 50: _evaluation(0.90)})
    _rescore(tmp_path, "L", candidate, {25: _evaluation(0.92), 50: _evaluation(0.88)})
    selection = select_rescored_checkpoint(_config(tmp_path), run_id="dqcheck_loop_r", label="L")
    assert selection["selected_update"] == 25
    assert selection["core_f1"] == 0.92
    assert selection["in_run_selected_update"] == 50
    assert selection["adapter_path"] == str(candidate / "checkpoints" / "update_0000025")
    assert selection["evaluator_commit"] == "abc1234"


def test_the_selection_is_written_beside_the_rescore(tmp_path: Path) -> None:
    candidate = _run(tmp_path, {25: _evaluation(0.80)})
    out = _rescore(tmp_path, "L", candidate, {25: _evaluation(0.85)})
    selection = select_rescored_checkpoint(_config(tmp_path), run_id="dqcheck_loop_r", label="L")
    assert json.loads((out / "selection.json").read_text()) == selection


def test_the_trainer_s_tie_break_order_is_kept(tmp_path: Path) -> None:
    """Equal F1: docwise accuracy decides before recall (g0.CHECKPOINT_TIE_BREAK_ORDER)."""
    candidate = _run(tmp_path, {25: _evaluation(0.9), 50: _evaluation(0.9)})
    _rescore(
        tmp_path,
        "L",
        candidate,
        {
            25: _evaluation(0.9, recall=0.99, docwise=0.6),
            50: _evaluation(0.9, recall=0.80, docwise=0.8),
        },
    )
    assert (
        select_rescored_checkpoint(_config(tmp_path), run_id="dqcheck_loop_r", label="L")[
            "selected_update"
        ]
        == 50
    )


def test_an_ineligible_point_stays_ineligible(tmp_path: Path) -> None:
    candidate = _run(tmp_path, {25: _evaluation(0.80), 50: _evaluation(0.70)}, coverage={25: 49})
    _rescore(tmp_path, "L", candidate, {25: _evaluation(0.99), 50: _evaluation(0.70)})
    assert (
        select_rescored_checkpoint(_config(tmp_path), run_id="dqcheck_loop_r", label="L")[
            "selected_update"
        ]
        == 50
    )


def test_eligibility_is_recomputed_on_the_rescored_evaluation(tmp_path: Path) -> None:
    """The collapse gate reads the evaluation: it is applied to the rescored one."""
    candidate = _run(tmp_path, {25: _evaluation(0.80), 50: _evaluation(0.70)})
    _rescore(tmp_path, "L", candidate, {25: _evaluation(0.95, empty_docs=5), 50: _evaluation(0.70)})
    assert (
        select_rescored_checkpoint(_config(tmp_path), run_id="dqcheck_loop_r", label="L")[
            "selected_update"
        ]
        == 50
    )


def test_a_point_missing_from_the_rescore_is_refused(tmp_path: Path) -> None:
    candidate = _run(tmp_path, {25: _evaluation(0.80), 50: _evaluation(0.90)})
    _rescore(tmp_path, "L", candidate, {25: _evaluation(0.92)})
    with pytest.raises(GateBlocked, match="update_0000050"):
        select_rescored_checkpoint(_config(tmp_path), run_id="dqcheck_loop_r", label="L")


def test_a_rescore_of_other_predictions_is_refused(tmp_path: Path) -> None:
    candidate = _run(tmp_path, {25: _evaluation(0.80)})
    _rescore(tmp_path, "L", candidate, {25: _evaluation(0.92)})
    (candidate / "validation" / "update_0000025" / "predictions.json").write_text("[]")
    with pytest.raises(IntegrityError, match="update_0000025"):
        select_rescored_checkpoint(_config(tmp_path), run_id="dqcheck_loop_r", label="L")


def test_a_rescore_of_another_run_is_refused(tmp_path: Path) -> None:
    candidate = _run(tmp_path, {25: _evaluation(0.80)})
    out = _rescore(tmp_path, "L", candidate, {25: _evaluation(0.92)})
    summary = json.loads((out / "summary.json").read_text())
    (out / "summary.json").write_text(json.dumps({**summary, "run_id": "dqcheck_loop_other"}))
    with pytest.raises(IntegrityError, match="dqcheck_loop_other"):
        select_rescored_checkpoint(_config(tmp_path), run_id="dqcheck_loop_r", label="L")


def test_a_drifted_validation_record_is_refused(tmp_path: Path) -> None:
    candidate = _run(tmp_path, {25: _evaluation(0.80)})
    _rescore(tmp_path, "L", candidate, {25: _evaluation(0.92)})
    manifest = candidate / "validation" / "update_0000025" / "manifest.json"
    manifest.write_text(manifest.read_text().replace('"parse_count": 50', '"parse_count": 49'))
    with pytest.raises(IntegrityError, match="update_0000025"):
        select_rescored_checkpoint(_config(tmp_path), run_id="dqcheck_loop_r", label="L")


def test_rebuilding_with_the_run_s_own_evaluation_reproduces_its_summaries(tmp_path: Path) -> None:
    """Fidelity: the rebuilt summary of an unchanged evaluation is the stored one."""
    from data_quality_checker.rescored_selection import rebuilt_summary

    candidate = _run(tmp_path, {25: _evaluation(0.80, empty_docs=1)})
    directory = candidate / "validation" / "update_0000025"
    stored = json.loads((directory / "summary.json").read_text())
    rebuilt = rebuilt_summary(
        candidate,
        25,
        json.loads((directory / "evaluation.json").read_text()),
        evaluation_sha256=stored["evaluation_sha256"],
    )
    assert {k: v for k, v in rebuilt.items() if k != "generated_at"} == {
        k: v for k, v in stored.items() if k != "generated_at"
    }


def test_loop_select_rescored_is_a_command(monkeypatch):
    from data_quality_checker import commands, rescored_selection
    from data_quality_checker.cli import build_parser

    args = build_parser().parse_args(
        ["loop-select-rescored", "--run-id", "dqcheck_loop_r", "--label", "L"]
    )
    seen = []
    monkeypatch.setattr(
        rescored_selection,
        "select_rescored_checkpoint",
        lambda config, **kw: seen.append(kw) or {"selected_update": 25, "points": []},
    )
    assert args.handler is commands.loop_select_rescored
    assert args.handler(args, None) == 0
    assert seen == [{"run_id": "dqcheck_loop_r", "label": "L"}]
