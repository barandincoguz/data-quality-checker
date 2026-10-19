"""The prompt screen (user decision, 1 Oct 2026): the loop's system prompt is chosen
on small subsets before the work starts, then fixed.

Pinned here: every candidate is registered and free of any unit below the
article; a config chooses the prompt for training data, the run id and batch
prediction alike; the elimination sample is fixed, training-only and never an
exemplar; the ranking rule; and a stage-two config isolates its own roots.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_quality_checker.constants import EXEMPLAR_DOC_IDS
from data_quality_checker.errors import ContractError
from data_quality_checker.g0 import (
    PROMPT_VARIANT,
    PROMPTS,
    SYSTEM_PROMPT,
    _training_example,
    loop_prompt,
)
from data_quality_checker.prompt_screen import (
    rank,
    screen_config_payload,
    screen_documents,
    stage_one,
)

ROOT = Path(__file__).resolve().parents[1]


def test_the_current_prompt_is_unchanged_and_every_candidate_is_clean() -> None:
    from test_no_paragraph_anywhere import FORBIDDEN_IN_PROMPTS

    assert PROMPTS[PROMPT_VARIANT] == SYSTEM_PROMPT
    # The run M000 v2 trains under must keep its id: its prompt bytes are fixed.
    import hashlib

    assert hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest().startswith("15403a53ee4a")
    assert len(PROMPTS) == 7 and len(set(PROMPTS.values())) == 7
    for variant, text in PROMPTS.items():
        assert not FORBIDDEN_IN_PROMPTS.search(text), variant
        assert "kanun_no" in text and "madde" in text, variant
    assert "retain table/cetvel/list references" not in PROMPTS["law-article-p3-scope"]
    assert "retain table/cetvel/list references" in PROMPTS["law-article-p1-binding"]


def test_a_config_chooses_the_prompt_and_an_unknown_one_is_refused() -> None:
    assert loop_prompt(SimpleNamespace(raw={"loop": {}})) == (PROMPT_VARIANT, SYSTEM_PROMPT)
    chosen = SimpleNamespace(raw={"loop": {"prompt_variant": "law-article-p2-complete"}})
    assert loop_prompt(chosen) == ("law-article-p2-complete", PROMPTS["law-article-p2-complete"])
    with pytest.raises(ContractError, match="unknown prompt"):
        loop_prompt(SimpleNamespace(raw={"loop": {"prompt_variant": "nope"}}))
    row = _training_example({"text": "t", "references": []}, PROMPTS["law-article-p6-minimal"])
    assert row["messages"][0]["content"] == PROMPTS["law-article-p6-minimal"]


def test_the_elimination_sample_is_fixed_training_only_and_never_an_exemplar() -> None:
    split = json.loads(
        (ROOT / "data/ground_truth/gt_v7_gt750_law_article_2026-10-01/SPLIT.json").read_text()
    )
    split = {"train": [int(d) for d in split["train"]]}
    sample = screen_documents(split)
    assert len(sample) == 100 and sample == screen_documents(split)
    assert set(sample) <= set(split["train"]) and not set(sample) & EXEMPLAR_DOC_IDS


def test_ranking_is_f1_then_recall_then_the_shorter_prompt() -> None:
    results = {
        "law-article-p4-combined": {"f1": 0.8, "recall": 0.7},
        "law-article-p6-minimal": {"f1": 0.8, "recall": 0.7},
        "law-article-p1-binding": {"f1": 0.8, "recall": 0.75},
        PROMPT_VARIANT: {"f1": 0.9, "recall": 0.1},
    }
    assert rank(results) == [
        PROMPT_VARIANT,
        "law-article-p1-binding",
        "law-article-p6-minimal",
        "law-article-p4-combined",
    ]


def test_a_stage_two_config_has_its_own_roots_and_prompt() -> None:
    # The base the screen ran on (2 Oct 2026): the loop's config before P4 and GT v8.
    base = json.loads((ROOT / "configs/presets/gt750_law_article_loop_p0_gtv7.json").read_text())
    payload = screen_config_payload(base, "law-article-p1-binding")
    assert payload["loop"]["prompt_variant"] == "law-article-p1-binding"
    assert (
        payload["public_root"] != base["public_root"]
        and payload["sensitive_root"] != base["sensitive_root"]
    )
    assert {k: v for k, v in payload["loop"].items() if k != "prompt_variant"} == base["loop"]
    for variant in ("law-article-p1-binding", "law-article-p4-combined"):
        written = json.loads(
            (ROOT / f"configs/presets/gt750_law_article_loop_prompt_{variant}.json").read_text()
        )
        assert written == screen_config_payload(base, variant)


def test_the_loop_is_fixed_with_the_screen_winner_on_gt_v8() -> None:
    preset = json.loads((ROOT / "configs/presets/gt750_law_article_loop.json").read_text())
    winner = (
        json.loads(
            (
                ROOT
                / "artifacts/data_quality_checker_gt750_law_article/loop/prompt_screen/stage2/selection.json"
            ).read_text()
        )["winner"]
        if (ROOT / "artifacts/data_quality_checker_gt750_law_article").is_dir()
        else None
    )
    assert (
        preset["loop"]["prompt_variant"]
        == "law-article-p4-combined"
        == (winner or "law-article-p4-combined")
    )
    assert (
        "gt_v8_gt750_law_article" in preset["canonical_gt_dir"]
        and "gt_v8_gt750_law_article" in preset["loop"]["split_path"]
    )


def test_stage_one_runs_every_prompt_and_sends_the_current_one_and_the_best_two_on(
    tmp_path, monkeypatch
) -> None:
    from data_quality_checker import prompt_screen

    docs = {
        i: {
            "text": f"metin {i}",
            "references": [{"kanun_no": "193", "kanun_ad": "", "madde": str(i)}],
        }
        for i in range(1, 201)
    }
    split = {
        "train": list(range(1, 181)),
        "valid": list(range(181, 191)),
        "test": list(range(191, 201)),
    }
    monkeypatch.setattr(
        prompt_screen, "validate_loop_sources", lambda config: {"documents": docs, "split": split}
    )
    import data_quality_checker.g0_training as g0_training

    monkeypatch.setattr(
        g0_training,
        "_base_run_context",
        lambda config, run_id: {
            "run_config": {
                "model_snapshot": {"snapshot_path": "/m"},
                "selected_sequence_length": 8192,
            }
        },
    )
    scores = {"law-article-p5-turkish": 0.9, "law-article-p3-scope": 0.85}
    seen = []

    def worker(*, compute_root, stage, request, lease, timeout_seconds):
        system = json.loads(Path(request["data_path"]).read_text().splitlines()[0])["messages"][0][
            "content"
        ]
        seen.append(system)
        (Path(compute_root) / "predictions.json").write_text("[]")
        return {"predictions_path": str(Path(compute_root) / "predictions.json")}

    def evaluator(*, config, predictions_path, doc_ids_path, output_dir, expected_doc_count):
        variant = Path(output_dir).name
        f1 = scores.get(variant, 0.5)
        return {
            "results": [
                {
                    "core_law_article_strict": {
                        "tp": 1,
                        "fp": 0,
                        "fn": 0,
                        "precision": 1.0,
                        "recall": f1,
                        "f1": f1,
                    },
                    "docwise_core_accuracy": {"accuracy": 0.5},
                }
            ]
        }

    config = SimpleNamespace(public_root=tmp_path, raw={"loop": {}}, fingerprint="f")
    selection = stage_one(
        config, base_run_id="dqcheck_loop_000000000000", worker=worker, evaluator=evaluator
    )
    assert sorted(seen) == sorted(PROMPTS.values())
    assert selection["to_stage_two"] == [
        PROMPT_VARIANT,
        "law-article-p5-turkish",
        "law-article-p3-scope",
    ]
    assert len(selection["doc_ids"]) == 100 and set(selection["doc_ids"]) <= set(split["train"])
    # Resumable: a finished prompt is not run again.
    again = stage_one(
        config, base_run_id="dqcheck_loop_000000000000", worker=worker, evaluator=evaluator
    )
    assert len(seen) == len(PROMPTS) and again["to_stage_two"] == selection["to_stage_two"]


def test_a_short_run_is_a_candidate_that_stops_at_its_update_and_is_never_sealed(
    monkeypatch,
) -> None:
    from argparse import Namespace

    from data_quality_checker import commands, loop_ops

    seen = {}
    monkeypatch.setattr(
        loop_ops,
        "train_loop_candidate",
        lambda config, **kwargs: seen.update(kwargs) or {"ok": True},
    )
    monkeypatch.setattr(
        loop_ops, "train_loop_round", lambda *a, **k: pytest.fail("a short run must never seal")
    )
    commands.loop_train(
        Namespace(round=0, resume=False, view=None, no_seal=False, stop_at=150), SimpleNamespace()
    )
    assert seen == {"round_index": 0, "resume": False, "view": None, "stop_at": 150}


def test_batch_prediction_asks_with_the_configs_prompt() -> None:
    import inspect

    from data_quality_checker.processing import MlxG0Backend

    source = inspect.getsource(MlxG0Backend)
    assert "loop_prompt(config)" in source and '"content": self.system_prompt' in source


def test_stage_two_picks_by_f1_at_the_registered_update_and_pairs_against_the_current(
    tmp_path,
) -> None:
    from data_quality_checker.prompt_screen import stage_two

    def run(name, points):
        root = tmp_path / name
        for update, (tp, fp, fn) in points.items():
            d = root / "validation" / f"update_{update:07d}"
            d.mkdir(parents=True)
            per = [{"doc_id": i, "tp": tp, "fp": fp, "fn": fn} for i in range(20)]
            f1 = 2 * tp / (2 * tp + fp + fn)
            (d / "evaluation.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "core_law_article_strict": {"f1": f1, "recall": tp / (tp + fn)},
                                "core_per_doc_metrics": per,
                            }
                        ]
                    }
                )
            )
        return root

    runs = {
        PROMPT_VARIANT: run("p0", {125: (5, 1, 1), 150: (4, 1, 1)}),
        "law-article-p1-binding": run("p1", {125: (2, 2, 2), 150: (5, 0, 1)}),
        "law-article-p6-minimal": run("p6", {125: (9, 0, 0), 150: (3, 2, 2)}),
    }
    result = stage_two(runs, output=tmp_path / "selection.json")
    assert result["winner"] == "law-article-p1-binding"
    p6 = next(r for r in result["ranking"] if r["variant"] == "law-article-p6-minimal")
    assert p6["best_f1_up_to"] == 1.0  # reported, but the update-150 F1 decides
    p1 = next(r for r in result["ranking"] if r["variant"] == "law-article-p1-binding")
    assert p1["vs_current"]["bootstrap_mean"] > 0 and p1["vs_current"]["verdict"] == "improved"
    assert (
        json.loads((tmp_path / "selection.json").read_text())["winner"] == "law-article-p1-binding"
    )
