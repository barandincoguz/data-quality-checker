import json
from types import SimpleNamespace

import pytest

from data_quality_checker import loop_ops
from data_quality_checker.errors import ContractError, GateBlocked


def _config(tmp_path):
    return SimpleNamespace(
        public_root=tmp_path / "public",
        sensitive_root=tmp_path / "sensitive",
        training_runs_root=tmp_path / "runs",
        raw={"loop": {}},
    )


def test_batch_ids_follow_the_round():
    assert [loop_ops.batch_id_for(k) for k in (1, 2, 5)] == ["R01", "R02", "R05"]
    with pytest.raises(ContractError):
        loop_ops.batch_id_for(0)


def test_round_k_trains_on_every_batch_up_to_k(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        loop_ops,
        "validate_loop_sources",
        lambda c: {"split": {"train": [], "valid": [], "test": []}},
    )

    def fake_train_round(config, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(loop_ops, "train_round", fake_train_round)
    with pytest.raises(RuntimeError):
        loop_ops._train_and_select(_config(tmp_path), round_index=3, resume=False)

    assert seen["cleaned_batch_ids"] == ["R01", "R02", "R03"]
    assert seen["generation"] == "M003"
    assert seen["target_updates"] is None


def test_a_round_is_not_trained_before_its_release_is_marked(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_ops, "_train_and_select", lambda config, **kw: pytest.fail("trained"))

    with pytest.raises(GateBlocked, match="adjudicated"):
        loop_ops.train_loop_round(_config(tmp_path), round_index=1, resume=False)


def test_m0_must_beat_the_base_model_on_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        loop_ops,
        "_train_and_select",
        lambda config, **kw: {
            "run_id": "dqcheck_loop_x",
            "selection": {
                "core_f1": 0.70,
                "selected_update": 25,
                "adapter_path": "a",
                "recall": 0.6,
            },
        },
    )
    monkeypatch.setattr(loop_ops, "_base_validation_f1", lambda config, run_id, **kw: 0.75)

    with pytest.raises(GateBlocked):
        loop_ops.train_loop_round(_config(tmp_path), round_index=0, resume=False)


def test_the_test_split_stays_closed_without_confirmation(tmp_path):
    with pytest.raises(GateBlocked, match="confirm"):
        loop_ops.open_test(_config(tmp_path), confirm=False)


def test_the_test_split_opens_only_when_every_round_has_a_model(tmp_path):
    config = _config(tmp_path)
    plan = config.public_root / "loop" / "loop_plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text(
        json.dumps(
            {
                "rounds": 5,
                "batch_manifest_path": "x",
                "annotation_zip": "x",
                "document_pool_zip": "x",
            }
        )
    )

    with pytest.raises(GateBlocked, match="round"):
        loop_ops.open_test(config, confirm=True)


def test_round_k_error_report_scores_batch_r0k_against_m_k_minus_1(tmp_path, monkeypatch):
    seen = {}

    def fake_report(config, **kwargs):
        seen.update(kwargs)
        return {"written": True}

    monkeypatch.setattr(loop_ops, "write_error_report", fake_report)
    loop_ops.error_report(_config(tmp_path), round_index=2)
    assert seen["batch_id"] == "R02" and seen["generation"] == "M001"
    assert seen["path"] == tmp_path / "public" / "loop" / "rounds" / "round_002_errors.json"


def test_marking_a_round_adjudicated_writes_its_error_report(tmp_path, monkeypatch):
    config = _config(tmp_path)
    batch = config.public_root / "batches" / "R01"
    batch.mkdir(parents=True)
    (batch / "release.json").write_text("{}")
    monkeypatch.setattr(loop_ops, "advance_round", lambda state, **kw: state)
    monkeypatch.setattr(loop_ops, "resume_round", lambda d, k: SimpleNamespace(stage="adjudicated"))
    monkeypatch.setattr(loop_ops, "write_round_state", lambda d, s: None)
    calls = []
    monkeypatch.setattr(
        loop_ops, "error_report", lambda config, round_index: calls.append(round_index)
    )
    loop_ops.mark_stage(config, round_index=1, stage="adjudicated")
    assert calls == [1]


def test_a_round_batch_is_only_ever_handled_with_its_rounds_model():
    loop = SimpleNamespace(raw={"loop": {}})
    loop_ops.assert_round_generation(loop, "R01", "M000")
    loop_ops.assert_round_generation(loop, "R03", "M002")
    for wrong in ("G0", "M001"):
        with pytest.raises(GateBlocked, match="M000"):
            loop_ops.assert_round_generation(loop, "R01", wrong)
    # Outside the loop config, and for non-round batch ids, nothing is imposed.
    loop_ops.assert_round_generation(SimpleNamespace(raw={}), "R01", "G0")
    loop_ops.assert_round_generation(loop, "batch", "G0")


def test_serve_judge_and_release_commands_refuse_the_wrong_model(monkeypatch):
    from argparse import Namespace

    from data_quality_checker import commands

    loop = SimpleNamespace(raw={"loop": {}})
    args = Namespace(
        batch_id="R01",
        generation="G0",
        port=0,
        judge_models=None,
        allow_external_judge=True,
        fake_backend=True,
    )
    for handler in (commands.serve, commands.pilot_judges, commands.release):
        with pytest.raises(GateBlocked, match="M000"):
            handler(args, loop)


def _second_line_fakes(monkeypatch, *, seen=None):
    def fake_train(config, **kw):
        if seen is not None:
            seen.update(kw)
        return {
            "run_id": "dqcheck_loop_x",
            "training_documents": 500,
            "selection": {
                "core_f1": 0.9,
                "selected_update": 125,
                "adapter_path": "/runs/x/update_125",
            },
        }

    monkeypatch.setattr(loop_ops, "_train_and_select", fake_train)
    monkeypatch.setattr(loop_ops, "seal_round_model", lambda **kw: pytest.fail("sealed"))
    monkeypatch.setattr(
        loop_ops,
        "_base_run_context",
        lambda config, run_id: {
            "run_config": {
                "model_snapshot": {"snapshot_path": "/snap"},
                "selected_sequence_length": 10240,
            }
        },
    )
    monkeypatch.setattr(
        loop_ops,
        "build_g0_registry",
        lambda **kw: {
            "adapter_path": str(kw["adapter_path"]),
            "adapter_sha256": "ab" * 32,
            "max_sequence_length": kw["max_sequence_length"],
        },
    )


def test_the_whole_document_line_is_trained_on_the_same_round_and_never_sealed(
    tmp_path, monkeypatch
):
    """Every round also trains the kept whole-document model (28 Sep 2026): same
    round, same data, its own run; nothing may touch round_NNN_model.json or g0/,
    and its inference registry (adapter checksum included) is kept beside it."""
    config = _config(tmp_path)
    seen = {}
    _second_line_fakes(monkeypatch, seen=seen)
    record = loop_ops.train_loop_candidate(
        config, round_index=0, resume=False, view="full_documents"
    )
    assert seen == {
        "round_index": 0,
        "resume": False,
        "view": "full_documents",
        "stop_at": None,
        "labels": "clean",
    }
    path = tmp_path / "public" / "loop" / "candidates" / "round_000_full_documents.json"
    assert json.loads(path.read_text()) == record
    assert record["sealed"] is False and record["training_view_mode"] == "full_documents"
    registry = json.loads(path.with_name("round_000_full_documents_registry.json").read_text())
    assert registry == record["registry"] and registry["adapter_path"] == "/runs/x/update_125"
    assert registry["max_sequence_length"] == 10240
    assert not (tmp_path / "public" / "loop" / "rounds" / "round_000_model.json").exists()
    assert not (tmp_path / "public" / "g0").exists()


@pytest.mark.parametrize("stage", ["adjudicated", "trained", "sealed"])
def test_second_line_can_follow_main_training_without_changing_its_seal(
    tmp_path, monkeypatch, stage
):
    config = _config(tmp_path)
    _second_line_fakes(monkeypatch)
    monkeypatch.setattr(loop_ops, "resume_round", lambda d, k: SimpleNamespace(stage=stage))
    model = tmp_path / "public/loop/rounds/round_001_model.json"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"original sealed model record")
    record = loop_ops.train_loop_candidate(
        config, round_index=1, resume=True, view="full_documents"
    )
    assert record["sealed"] is False
    assert model.read_bytes() == b"original sealed model record"


def test_the_training_view_reaches_the_trainer(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        loop_ops,
        "validate_loop_sources",
        lambda c: {"split": {"train": [], "valid": [], "test": []}},
    )

    def fake_train_round(config, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(loop_ops, "train_round", fake_train_round)
    with pytest.raises(RuntimeError):
        loop_ops._train_and_select(
            _config(tmp_path), round_index=1, resume=False, view="full_documents"
        )
    assert seen["view"] == "full_documents" and seen["cleaned_batch_ids"] == ["R01"]


def test_only_the_windowed_line_is_ever_sealed(tmp_path, monkeypatch):
    """The model that predicts the next batch is the windowed one; a config asking
    for whole documents must not seal a whole-document model as M(k)."""
    config = _config(tmp_path)
    config.raw = {"loop": {"training_view": "full_documents"}}
    monkeypatch.setattr(loop_ops, "_train_and_select", lambda config, **kw: pytest.fail("trained"))
    with pytest.raises(GateBlocked, match="--view full_documents"):
        loop_ops.train_loop_round(config, round_index=0, resume=False)


def test_the_view_flag_never_seals_the_whole_document_line(monkeypatch):
    from data_quality_checker import commands

    calls = []
    monkeypatch.setattr(
        loop_ops, "train_loop_candidate", lambda config, **kw: calls.append(("candidate", kw)) or {}
    )
    monkeypatch.setattr(
        loop_ops, "train_loop_round", lambda config, **kw: calls.append(("sealed", kw)) or {}
    )
    args = SimpleNamespace(round=1, resume=True, no_seal=False, view="full_documents")
    commands.loop_train(args, None)
    commands.loop_train(SimpleNamespace(round=1, resume=False, no_seal=False, view=None), None)
    assert calls == [
        (
            "candidate",
            {"round_index": 1, "resume": True, "view": "full_documents", "stop_at": None},
        ),
        ("sealed", {"round_index": 1, "resume": False}),
    ]


def _write_line(tmp_path, *, round_index, view, run_id, selected, updates):
    public = tmp_path / "public" / "loop"
    record = {
        "run_id": run_id,
        "training_documents": 500,
        "selection": {"selected_update": selected},
    }
    if view == "windows":
        path = public / "rounds" / f"round_{round_index:03d}_model.json"
    else:
        path = public / "candidates" / f"round_{round_index:03d}_full_documents.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    run = tmp_path / "runs" / run_id
    run.mkdir(parents=True)
    (run / "run_config.json").write_text(
        json.dumps({"recurrence": None if view == "windows" else "wy"})
    )
    for update, tp in updates.items():
        directory = (
            run / "development" / loop_ops.LOOP_CANDIDATE / "validation" / f"update_{update:07d}"
        )
        directory.mkdir(parents=True)
        (directory / "predictions.json").write_text(json.dumps({"tp": tp}))
    ids = tmp_path / "sensitive" / "loop" / run_id / "data"
    ids.mkdir(parents=True)
    (ids / "valid_doc_ids.json").write_text(json.dumps([1, 2, 3, 4]))


def test_a_round_compares_both_lines_and_each_with_its_previous_round(tmp_path, monkeypatch):
    """Validation only: each line's checkpoints re-scored, the lines paired against
    each other, and from round 1 each line against its own previous round."""
    config = _config(tmp_path)
    _write_line(
        tmp_path, round_index=0, view="windows", run_id="w0", selected=50, updates={25: 1, 50: 2}
    )
    _write_line(
        tmp_path, round_index=0, view="full_documents", run_id="f0", selected=25, updates={25: 2}
    )
    _write_line(tmp_path, round_index=1, view="windows", run_id="w1", selected=25, updates={25: 3})
    _write_line(
        tmp_path, round_index=1, view="full_documents", run_id="f1", selected=25, updates={25: 4}
    )
    scored = []

    def fake_metrics(*, config, predictions_path, doc_ids_path, output_dir, expected_doc_count):
        tp = json.loads(predictions_path.read_text())["tp"]
        scored.append(predictions_path.parts[-6])
        per_doc = {
            d: {"tp": 1 if d <= tp else 0, "fp": 0, "fn": 0 if d <= tp else 1} for d in (1, 2, 3, 4)
        }
        metric = {"tp": tp, "fp": 0, "fn": 4 - tp, "precision": 1.0, "recall": tp / 4, "f1": tp / 4}
        return {
            "core": metric,
            "full": metric,
            "core_per_document": per_doc,
            "docwise_f1_distribution": {"perfect_f1_count": tp, "total_docs": 4},
        }

    monkeypatch.setattr(loop_ops, "model_round_metrics", fake_metrics)
    round0 = loop_ops.compare_views(config, round_index=0)
    assert [p["update"] for p in round0["validation_curves"]["windows"]] == [25, 50]
    assert round0["selected"]["windows"]["core"]["tp"] == 2
    assert (
        round0["runs"]["windows"]["sealed_as"] == "M000"
        and round0["runs"]["full_documents"]["sealed_as"] is None
    )
    assert round0["paired_bootstrap_full_minus_windows"]["document_count"] == 4
    assert "paired_bootstrap_vs_previous_round" not in round0

    scored.clear()
    round1 = loop_ops.compare_views(config, round_index=1)
    assert round1["previous_round"] == {
        "windows": {"run_id": "w0", "selected_update": 50},
        "full_documents": {"run_id": "f0", "selected_update": 25},
    }
    # The previous round's selected checkpoint only, not its whole curve.
    assert sorted(scored) == ["f0", "f1", "w0", "w1"]
    gains = round1["paired_bootstrap_vs_previous_round"]
    assert gains["windows"]["bootstrap_mean"] > 0 and gains["full_documents"]["bootstrap_mean"] > 0
    path = tmp_path / "public" / "loop" / "candidates" / "round_001_ablation.json"
    assert json.loads(path.read_text())["round"] == 1


def test_comparing_a_round_needs_both_lines(tmp_path):
    config = _config(tmp_path)
    _write_line(tmp_path, round_index=0, view="windows", run_id="w0", selected=25, updates={25: 1})
    with pytest.raises(GateBlocked, match="round_000_full_documents"):
        loop_ops.compare_views(config, round_index=0)


def test_the_raw_label_arm_is_a_candidate_and_never_sealed(tmp_path, monkeypatch):
    """`M_k_raw` (noisy-vs-clean ablation, pre-registered 4 Oct 2026): its own
    record beside the clean candidates, never round_NNN_model.json or g0/."""
    config = _config(tmp_path)
    seen = {}
    _second_line_fakes(monkeypatch, seen=seen)
    monkeypatch.setattr(loop_ops, "resume_round", lambda d, k: SimpleNamespace(stage="adjudicated"))
    record = loop_ops.train_loop_candidate(config, round_index=1, resume=False, labels="raw")
    assert seen["labels"] == "raw"
    path = tmp_path / "public" / "loop" / "candidates" / "round_001_windows_raw.json"
    assert json.loads(path.read_text()) == record
    assert record["sealed"] is False and record["label_source"] == "raw"
    assert path.with_name("round_001_windows_raw_registry.json").is_file()
    assert not (tmp_path / "public" / "loop" / "rounds" / "round_001_model.json").exists()
    assert not (tmp_path / "public" / "g0").exists()


def test_the_label_source_reaches_the_trainer(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        loop_ops,
        "validate_loop_sources",
        lambda c: {"split": {"train": [], "valid": [], "test": []}},
    )

    def fake_train_round(config, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(loop_ops, "train_round", fake_train_round)
    with pytest.raises(RuntimeError):
        loop_ops._train_and_select(_config(tmp_path), round_index=1, resume=False, labels="raw")
    assert seen["labels"] == "raw" and seen["cleaned_batch_ids"] == ["R01"]


def test_the_labels_flag_never_seals_the_raw_arm(monkeypatch):
    from data_quality_checker import commands

    calls = []
    monkeypatch.setattr(
        loop_ops, "train_loop_candidate", lambda config, **kw: calls.append(("candidate", kw)) or {}
    )
    monkeypatch.setattr(
        loop_ops, "train_loop_round", lambda config, **kw: calls.append(("sealed", kw)) or {}
    )
    commands.loop_train(
        SimpleNamespace(round=1, resume=False, no_seal=False, view=None, labels="raw"), None
    )
    assert calls == [
        (
            "candidate",
            {"round_index": 1, "resume": False, "view": None, "stop_at": None, "labels": "raw"},
        ),
    ]


def test_loop_train_takes_a_labels_flag():
    from data_quality_checker.cli import build_parser

    args = build_parser().parse_args(["loop-train", "--round", "1", "--labels", "raw"])
    assert args.labels == "raw"
    assert build_parser().parse_args(["loop-train", "--round", "1"]).labels == "clean"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["loop-train", "--round", "1", "--labels", "noisy"])


def test_the_raw_arm_trains_once_the_batch_is_routed_without_waiting_for_the_expert(
    tmp_path, monkeypatch
):
    """Its labels are the annotator's: no expert decision enters it (5 Oct 2026)."""
    config = _config(tmp_path)
    seen = {}
    _second_line_fakes(monkeypatch, seen=seen)
    monkeypatch.setattr(loop_ops, "resume_round", lambda d, k: SimpleNamespace(stage="judged"))
    record = loop_ops.train_loop_candidate(config, round_index=1, resume=False, labels="raw")
    assert seen["labels"] == "raw" and record["label_source"] == "raw"


def test_the_raw_arm_waits_for_the_routing(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr(loop_ops, "_train_and_select", lambda config, **kw: pytest.fail("trained"))
    monkeypatch.setattr(loop_ops, "resume_round", lambda d, k: SimpleNamespace(stage="predicted"))
    with pytest.raises(GateBlocked, match="routed"):
        loop_ops.train_loop_candidate(config, round_index=1, resume=False, labels="raw")


def test_the_clean_candidate_still_waits_for_the_expert(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr(loop_ops, "_train_and_select", lambda config, **kw: pytest.fail("trained"))
    monkeypatch.setattr(loop_ops, "resume_round", lambda d, k: SimpleNamespace(stage="judged"))
    with pytest.raises(GateBlocked, match="adjudicated"):
        loop_ops.train_loop_candidate(config, round_index=1, resume=False)


def _raw_candidate(tmp_path, train_ids):
    record = {"run_id": "dqcheck_loop_raw", "label_source": "raw"}
    path = tmp_path / "public" / "loop" / "candidates" / "round_001_windows_raw.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    data = tmp_path / "sensitive" / "loop" / "dqcheck_loop_raw" / "data"
    data.mkdir(parents=True)
    (data / "train_doc_ids.json").write_text(json.dumps(train_ids))


def test_a_raw_run_trained_before_the_release_is_checked_against_it(tmp_path, monkeypatch):
    from test_loop_training import write_sealed_release

    config = _config(tmp_path)
    _raw_candidate(tmp_path, [1, 2, "d1", "d2"])
    write_sealed_release(
        tmp_path / "sensitive",
        "R01",
        [{"internal_doc_id": "d2", "text": "iki", "references": []}],
        [{"internal_doc_id": "d1", "text": "bir", "references": []}],
    )
    report = loop_ops.check_raw_candidate_against_release(config, round_index=1)
    assert report["same_documents"] is True
    assert report["run_documents"] == report["release_documents"] == 2
    written = (
        tmp_path / "public" / "loop" / "candidates" / "round_001_windows_raw_release_check.json"
    )
    assert json.loads(written.read_text()) == report


def test_a_raw_run_whose_documents_differ_from_the_release_is_flagged(tmp_path, monkeypatch):
    from test_loop_training import write_sealed_release

    config = _config(tmp_path)
    _raw_candidate(tmp_path, [1, 2, "d1", "d2", "d3"])
    write_sealed_release(
        tmp_path / "sensitive",
        "R01",
        [{"internal_doc_id": "d2", "text": "iki", "references": []}],
        [{"internal_doc_id": "d1", "text": "bir", "references": []}],
    )
    report = loop_ops.check_raw_candidate_against_release(config, round_index=1)
    assert report["same_documents"] is False
    assert report["only_in_run"] == ["d3"] and report["only_in_release"] == []


def test_loop_raw_check_is_a_command(monkeypatch):
    from data_quality_checker import commands
    from data_quality_checker.cli import build_parser

    args = build_parser().parse_args(["loop-raw-check", "--round", "1"])
    seen = []
    monkeypatch.setattr(
        loop_ops,
        "check_raw_candidate_against_release",
        lambda config, **kw: seen.append(kw) or {"same_documents": True},
    )
    assert args.handler(args, None) == 0
    assert seen == [{"round_index": 1}]
    assert args.handler is commands.loop_raw_check
