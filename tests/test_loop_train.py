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
    directory = Path(config.sensitive_root) / "releases" / batch_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "expert_adjudicated.jsonl").write_text(
        "".join(
            json.dumps(
                {"internal_doc_id": doc_id, "text": f"text-{doc_id}", "references": []},
                ensure_ascii=False,
            )
            + "\n"
            for doc_id in doc_ids
        ),
        encoding="utf-8",
    )
    (directory / "consensus_clean.jsonl").write_text("", encoding="utf-8")


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


def test_execute_false_scaffolding_validates_against_base_run_context(tmp_path: Path) -> None:
    """The run_config.json/preflight.json `train_round` writes even when it
    never executes must already be exactly what `run_development`'s entry gate
    (`_base_run_context`) demands, so a later `execute=True` call has nothing
    left to assemble -- only real compute preflight left to pass."""
    from data_quality_checker.g0_training import _base_run_context

    config = canonical_fixture(tmp_path)
    result = train_round(config, generation="M000", split=SPLIT, cleaned_batch_ids=[])
    context = _base_run_context(config, result["run_id"])
    assert context["run_config"]["run_id"] == result["run_id"]
    assert context["preflight"]["run_id"] == result["run_id"]
    assert context["data_dir"] == Path(result["data_dir"])


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

    result = train_round(
        config,
        generation="M000",
        split=SPLIT,
        cleaned_batch_ids=[],
        execute=True,
        candidate_id="lr1e-5",
        target_updates=25,
        worker_runner=fake_worker_runner,
    )

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
    assert set(test_ids) == {"canonical:20", "canonical:21"}
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
