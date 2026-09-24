import json
from types import SimpleNamespace

import pytest

from data_quality_checker import loop_ops
from data_quality_checker.errors import ContractError, GateBlocked


def _config(tmp_path):
    return SimpleNamespace(public_root=tmp_path / "public", sensitive_root=tmp_path / "sensitive",
                           training_runs_root=tmp_path / "runs", raw={"loop": {}})


def test_batch_ids_follow_the_round():
    assert [loop_ops.batch_id_for(k) for k in (1, 2, 5)] == ["R01", "R02", "R05"]
    with pytest.raises(ContractError):
        loop_ops.batch_id_for(0)


def test_round_k_trains_on_every_batch_up_to_k(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(loop_ops, "validate_loop_sources", lambda c: {"split": {"train": [], "valid": [], "test": []}})

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
    monkeypatch.setattr(loop_ops, "_train_and_select", lambda config, **kw: {
        "run_id": "dqcheck_loop_x", "selection": {"core_f1": 0.70, "selected_update": 25,
                                                   "adapter_path": "a", "recall": 0.6}})
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
    plan.write_text(json.dumps({"rounds": 5, "batch_manifest_path": "x",
                                "annotation_zip": "x", "document_pool_zip": "x"}))

    with pytest.raises(GateBlocked, match="round"):
        loop_ops.open_test(config, confirm=True)
