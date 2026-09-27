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
