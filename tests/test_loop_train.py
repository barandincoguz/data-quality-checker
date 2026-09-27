from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_quality_checker.errors import ContractError
from data_quality_checker.loop_train import train_round

try:
    from test_g0 import isolated_config as canonical_fixture
except ImportError:  # pragma: no cover - fallback if test_g0 is not importable here
    from data_quality_checker.config import default_config_path, load_config

    def canonical_fixture(tmp_path: Path):
        live = load_config()
        payload = json.loads(default_config_path().read_text(encoding="utf-8"))
        payload.update(
            {
                "canonical_gt_dir": str(live.canonical_gt_dir),
                "example_bank_path": str(live.example_bank_path),
                "reference_split_manifest_path": str(live.reference_split_manifest_path),
                "sensitive_root": str(tmp_path / "sensitive"),
                "public_root": str(tmp_path / "public"),
                "training_runs_root": str(tmp_path / "runs"),
            }
        )
        path = tmp_path / "config.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return load_config(path)


SPLIT = {"train": [1, 2, 3], "valid": [10, 11], "test": [20, 21]}


def release_fixture(config, batch_id: str, doc_ids: list[str]) -> None:
    from test_loop_training import write_sealed_release

    write_sealed_release(
        Path(config.sensitive_root),
        batch_id,
        [{"internal_doc_id": d, "text": f"text-{d}", "references": []} for d in doc_ids],
        [],
    )


def test_round_zero_trains_on_canonical_only(tmp_path: Path) -> None:
    config = canonical_fixture(tmp_path)
    result = train_round(config, generation="M000", split=SPLIT, cleaned_batch_ids=[])
    assert result["training_documents"] == 3
    assert result["cleaned_batch_ids"] == []


def test_a_later_round_adds_its_cleaned_documents(tmp_path: Path) -> None:
    config = canonical_fixture(tmp_path)
    release_fixture(config, "round1", ["d1", "d2"])
    result = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"])
    assert result["training_documents"] == 5


def test_validation_and_test_never_grow(tmp_path: Path) -> None:
    config = canonical_fixture(tmp_path)
    release_fixture(config, "round1", ["d1", "d2"])
    base = train_round(config, generation="M000", split=SPLIT, cleaned_batch_ids=[])
    grown = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"])
    assert grown["validation_documents"] == base["validation_documents"] == 2
    assert grown["test_documents"] == base["test_documents"] == 2


def test_two_rounds_get_different_run_directories(tmp_path: Path) -> None:
    config = canonical_fixture(tmp_path)
    release_fixture(config, "round1", ["d1", "d2"])
    first = train_round(config, generation="M000", split=SPLIT, cleaned_batch_ids=[])
    second = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"])
    assert first["run_id"] != second["run_id"]


def test_the_same_round_is_idempotent(tmp_path: Path) -> None:
    config = canonical_fixture(tmp_path)
    release_fixture(config, "round1", ["d1", "d2"])
    first = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"])
    second = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"])
    assert first["run_id"] == second["run_id"]


def test_an_invalid_generation_is_rejected(tmp_path: Path) -> None:
    config = canonical_fixture(tmp_path)
    for generation in ("G0", "m001", "M1", "round1", ""):
        with pytest.raises(ContractError):
            train_round(config, generation=generation, split=SPLIT, cleaned_batch_ids=[])


def test_the_entry_gate_refuses_until_the_windowed_view_exists(tmp_path: Path) -> None:
    """Scaffolding alone is not enough to start training, and must not be.

    `_base_run_context` reads `training_view.jsonl_path` and hands it to the
    trainer as the file it trains on. That file is the windowed view the
    compute acceptance preflight builds from train.jsonl with the tokenizer.
    Until it exists the gate must refuse, because the only way to pass it
    earlier is to name the flat train.jsonl instead -- which trains on
    untruncated documents whose targets still carry `source_text`.
    """
    from data_quality_checker.errors import GateBlocked
    from data_quality_checker.g0_training import _base_run_context

    config = canonical_fixture(tmp_path)
    result = train_round(config, generation="M000", split=SPLIT, cleaned_batch_ids=[])
    run_config = json.loads(Path(result["run_config_path"]).read_text(encoding="utf-8"))
    assert run_config["run_id"] == result["run_id"]
    assert (
        json.loads(Path(result["preflight_path"]).read_text(encoding="utf-8"))["run_id"]
        == (result["run_id"])
    )
    with pytest.raises(GateBlocked):
        _base_run_context(config, result["run_id"])


def test_execute_true_without_a_candidate_id_is_rejected(tmp_path: Path) -> None:
    """`execute=True` used to be an unconditional refusal; now that it hands
    off to real compute, the caller must state which LR candidate to run --
    exactly the same "no guessed facts" rule every other round-step binding in
    this codebase already follows."""
    config = canonical_fixture(tmp_path)
    with pytest.raises(ContractError, match="candidate_id"):
        train_round(config, generation="M000", split=SPLIT, cleaned_batch_ids=[], execute=True)
    with pytest.raises(ContractError, match="candidate_id"):
        train_round(
            config,
            generation="M000",
            split=SPLIT,
            cleaned_batch_ids=[],
            execute=True,
            candidate_id="not-a-real-candidate",
        )


def test_execute_true_still_refuses_g0_and_requires_a_round_label(tmp_path: Path) -> None:
    config = canonical_fixture(tmp_path)
    with pytest.raises(ContractError):
        train_round(
            config,
            generation="G0",
            split=SPLIT,
            cleaned_batch_ids=[],
            execute=True,
            candidate_id="lr1e-5",
        )
    with pytest.raises(ContractError):
        train_round(
            config,
            generation="round1",
            split=SPLIT,
            cleaned_batch_ids=[],
            execute=True,
            candidate_id="lr1e-5",
        )


def test_execute_true_reaches_run_development_with_the_injected_fake_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`execute=True` must call the compute driver instead of raising. No
    compute may run from a test, so the driver itself is replaced with a fake
    that records its call and never touches real compute; a `worker_runner`
    is still threaded through untouched, exactly as a caller would inject one
    to keep a *real* call to `run_development` from launching real training."""
    config = canonical_fixture(tmp_path)
    calls: list[dict] = []

    def fake_run_development(**kwargs):
        calls.append(kwargs)
        return {"status": "fake_completed", "run_id": kwargs["run_id"]}

    monkeypatch.setattr("data_quality_checker.loop_train.run_development", fake_run_development)

    def fake_worker_runner(**kwargs):  # pragma: no cover - never actually called
        raise AssertionError("real compute must never run from a test")

    preflights: list[dict] = []

    def fake_compute_preflight(**kwargs):
        """The real one loads a tokenizer and windows the corpus; not in a test."""
        preflights.append(kwargs)
        return {"status": "fake_preflight"}

    result = train_round(
        config,
        generation="M000",
        split=SPLIT,
        cleaned_batch_ids=[],
        execute=True,
        candidate_id="lr1e-5",
        target_updates=25,
        compute_preflight_runner=fake_compute_preflight,
        worker_runner=fake_worker_runner,
    )

    # The windowed view must be built before the trainer is asked for it.
    assert len(preflights) == 1
    assert preflights[0]["run_dir"].name == result["run_id"]
    assert len(calls) == 1
    call = calls[0]
    assert call["run_id"] == result["run_id"]
    assert call["candidate_id"] == "lr1e-5"
    assert call["target_updates"] == 25
    assert call["execute"] is True
    assert call["worker_runner"] is fake_worker_runner
    assert result["development"] == {"status": "fake_completed", "run_id": result["run_id"]}


def test_test_documents_never_appear_in_train_or_valid_files(tmp_path: Path) -> None:
    """The 150 (here: 2) frozen test documents must never leak into what the
    model actually trains or is selected on."""
    config = canonical_fixture(tmp_path)
    result = train_round(config, generation="M000", split=SPLIT, cleaned_batch_ids=[])
    data_dir = Path(result["data_dir"])
    train_ids = json.loads((data_dir / "train_doc_ids.json").read_text(encoding="utf-8"))
    valid_ids = json.loads((data_dir / "valid_doc_ids.json").read_text(encoding="utf-8"))
    test_ids = json.loads((data_dir / "test_doc_ids.json").read_text(encoding="utf-8"))
    assert set(test_ids) == {20, 21}
    assert set(test_ids) & set(train_ids) == set()
    assert set(test_ids) & set(valid_ids) == set()


def test_round_run_directory_is_pinned_to_the_loop_namespace(tmp_path: Path) -> None:
    """A round's run directory must live under `sensitive_root / "loop"` and
    never under `sensitive_root / "g0"`, so it can never collide with G0's
    historical run directory."""
    config = canonical_fixture(tmp_path)
    result = train_round(config, generation="M000", split=SPLIT, cleaned_batch_ids=[])
    data_dir = Path(result["data_dir"])
    loop_root = Path(config.sensitive_root) / "loop"
    g0_root = Path(config.sensitive_root) / "g0"
    assert loop_root in data_dir.parents
    assert g0_root not in data_dir.parents


def test_the_scaffolding_does_not_fabricate_a_training_view(tmp_path: Path) -> None:
    """A round must not claim a training view before the window is built.

    `_base_run_context` reads `training_view.jsonl_path` and hands it to the
    trainer as the training input. The windowed view is built by the compute
    acceptance preflight, from `train.jsonl`, with the tokenizer -- for G0 it
    turned 394 documents into 3399 windows. Pointing the view at the flat file
    instead would train the model on untruncated documents whose targets still
    carry `source_text`, a field the system prompt no longer even mentions.

    So the scaffolding leaves the view out, and `execute=True` runs the compute
    preflight before the trainer.
    """
    config = canonical_fixture(tmp_path)
    result = train_round(config, generation="M000", split=SPLIT, cleaned_batch_ids=[])
    run_config = json.loads(Path(result["run_config_path"]).read_text(encoding="utf-8"))
    assert "training_view" not in run_config, (
        "the windowed view does not exist yet; claiming one would train on the flat file"
    )


def test_the_preflight_carries_the_gates_and_the_round_s_own_document_counts(
    tmp_path: Path,
) -> None:
    """The compute preflight reads this file; a thin one stops the run dead.

    `run_compute_acceptance_preflight` writes into `preflight["compute_gates"]`
    straight away, so a preflight without that block fails with a KeyError
    before any window is built. And the training contract inside it must state
    the round's own split -- a bare `TrainingContract()` claims 394 train and 50
    test documents, which is G0's geometry, not a round's.
    """
    config = canonical_fixture(tmp_path)
    result = train_round(config, generation="M000", split=SPLIT, cleaned_batch_ids=[])
    preflight = json.loads(Path(result["preflight_path"]).read_text(encoding="utf-8"))

    assert "compute_gates" in preflight
    contract = preflight["training_contract"]
    assert contract["train_documents"] == len(SPLIT["train"])
    assert contract["validation_documents"] == len(SPLIT["valid"])
    assert contract["test_documents"] == len(SPLIT["test"])
    # Capacity and optimisation stay fixed across generations, or a round-over-
    # round difference stops meaning "more data".
    assert contract["lora_rank"] == 8
    assert contract["lora_layers"] == 16
    assert contract["batch_size"] == 1
    assert contract["gradient_accumulation"] == 4
    assert contract["seed"] == 42


def test_a_loop_config_trains_on_the_sealed_split_only(tmp_path, monkeypatch):
    from data_quality_checker import loop_train

    sealed = {"train": [1, 2], "valid": [3], "test": [4]}
    monkeypatch.setattr(loop_train, "has_loop_settings", lambda config: True)
    monkeypatch.setattr(
        loop_train,
        "validate_loop_sources",
        lambda config: {"documents": {}, "split": sealed, "summary": {}},
    )

    with pytest.raises(ContractError, match="sealed"):
        loop_train.train_round(
            object(),
            generation="M000",
            split={"train": [1], "valid": [3], "test": [4]},
            cleaned_batch_ids=[],
        )


def test_full_epoch_horizon_is_derived_from_the_training_view(tmp_path):
    from data_quality_checker.loop_train import full_epoch_updates

    view = tmp_path / "view.jsonl"
    view.write_text("\n".join("{}" for _ in range(2194)) + "\n")

    assert full_epoch_updates(view, gradient_accumulation=4) == 549


def test_a_loop_round_trains_whole_documents_in_its_own_run(tmp_path: Path, monkeypatch) -> None:
    """The GT750 loop trains without cutting documents into windows (27 Sep 2026).
    A different training view must never resume into the windowed run's directory."""
    from data_quality_checker import loop_train
    from data_quality_checker.g0 import validate_canonical_sources

    config = canonical_fixture(tmp_path)
    windowed = train_round(config, generation="M000", split=SPLIT, cleaned_batch_ids=[])
    source = validate_canonical_sources(config)
    monkeypatch.setattr(loop_train, "has_loop_settings", lambda config: True)
    monkeypatch.setattr(
        loop_train, "validate_loop_sources", lambda config: {**source, "split": SPLIT}
    )
    monkeypatch.setattr(loop_train, "loop_split_policy", lambda config: "loop-test")
    whole = loop_train.train_round(config, generation="M000", split=SPLIT, cleaned_batch_ids=[])
    assert whole["run_id"] != windowed["run_id"]
    run_config = json.loads(Path(whole["run_config_path"]).read_text())
    assert run_config["training_view_mode"] == "full_documents"
    # The recurrence implementation is part of the run: a run measured under one
    # (preflight memory, resume equivalence) is never continued under another.
    from data_quality_checker.mlx_memory import RECURRENCE_ID

    assert run_config["recurrence"] == RECURRENCE_ID
    assert (
        json.loads(Path(windowed["run_config_path"]).read_text())["training_view_mode"] == "windows"
    )


def test_whole_document_training_uses_the_chunked_memory_mode_and_finer_buckets() -> None:
    from data_quality_checker.g0_training import _candidate_training_config

    whole = _candidate_training_config(
        peak_learning_rate=2.5e-5,
        sequence_length=8192,
        training_rows=500,
        training_view_mode="full_documents",
    )
    assert whole.memory_mode == "chunked"
    assert 7168 in whole.sequence_length_buckets and 5120 in whole.sequence_length_buckets
    windowed = _candidate_training_config(
        peak_learning_rate=2.5e-5, sequence_length=1536, training_rows=2823
    )
    assert windowed.memory_mode == "stock"


def test_whole_document_training_takes_the_length_of_the_longest_document() -> None:
    from data_quality_checker.errors import GateBlocked
    from data_quality_checker.mlx_compute import training_sequence_candidates

    assert training_sequence_candidates("full_documents", longest_train_tokens=7109) == (8192,)
    assert training_sequence_candidates("windows", longest_train_tokens=7109) == (1536,)
    with pytest.raises(GateBlocked):
        training_sequence_candidates("full_documents", longest_train_tokens=40000)
