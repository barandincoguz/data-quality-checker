"""Batch-local IDs must stay paired with their source in both training arms."""

import json
from pathlib import Path

import pytest
from test_loop_train import canonical_fixture
from test_loop_training import write_sealed_release
from test_loop_training_raw import LOCATABLE, SPLIT, _prepare, _reference

from data_quality_checker.errors import ContractError
from data_quality_checker.loop_train import train_round
from data_quality_checker.loop_training import (
    round_raw_training_documents,
    round_training_documents,
)
from data_quality_checker.storage import Store


def prepare(config, batch, *, local_id="d1", source_id=None, release=True):
    root = Path(config.sensitive_root)
    text = f"{batch}. {LOCATABLE}"
    refs = [_reference("193", "94", "")]
    _prepare(root, batch, {local_id: (text, refs)})
    with Store(config.database_path) as store:
        store.connection.execute(
            "UPDATE documents SET raw_document_id=? WHERE batch_id=? AND internal_doc_id=?",
            (source_id or f"source-{batch}", batch, local_id),
        )
    if release:
        write_sealed_release(root, batch, [{"internal_doc_id": local_id,
                             "raw_document_id": source_id or f"source-{batch}",
                             "text": text, "references": refs}], [])


@pytest.mark.parametrize("labels", ["clean", "raw"])
def test_two_batches_with_the_same_local_id_reach_training_without_overwrite(tmp_path, labels):
    config = canonical_fixture(tmp_path)
    prepare(config, "R01")
    prepare(config, "R02")
    result = train_round(config, generation="M002", split=SPLIT,
                         cleaned_batch_ids=["R01", "R02"], labels=labels)
    ids = json.loads((Path(result["data_dir"]) / "train_doc_ids.json").read_text())
    assert ids == [*SPLIT["train"], '["R01","d1"]', '["R02","d1"]']
    assert result["training_documents"] == 5
    rows = (Path(result["data_dir"]) / "train.jsonl").read_text()
    assert "R01." in rows and "R02." in rows
    for name in ("valid", "test"):
        assert json.loads((Path(result["data_dir"]) / f"{name}_doc_ids.json").read_text()) == SPLIT[name]


def test_multi_batch_raw_run_is_identical_before_and_after_release(tmp_path):
    config = canonical_fixture(tmp_path)
    prepare(config, "R01")
    prepare(config, "R02", release=False)
    before = train_round(config, generation="M002", split=SPLIT,
                         cleaned_batch_ids=["R01", "R02"], labels="raw")
    write_sealed_release(Path(config.sensitive_root), "R02",
                         [{"internal_doc_id": "d1", "raw_document_id": "source-R02",
                           "text": f"R02. {LOCATABLE}", "references": []}], [])
    after = train_round(config, generation="M002", split=SPLIT,
                        cleaned_batch_ids=["R01", "R02"], labels="raw")
    assert before["run_id"] == after["run_id"]
    clean = round_training_documents(config, batch_ids=["R01", "R02"])
    raw, audit = round_raw_training_documents(config, batch_ids=["R01", "R02"])
    assert set(raw) == set(clean)
    assert audit["documents_by_batch"] == {"R01": ['["R01","d1"]'], "R02": ['["R02","d1"]']}


@pytest.mark.parametrize("labels", ["clean", "raw"])
def test_a_real_source_document_repeated_across_batches_is_rejected(tmp_path, labels):
    config = canonical_fixture(tmp_path)
    prepare(config, "R01", local_id="d1", source_id="same-source")
    prepare(config, "R02", local_id="d2", source_id="same-source")
    with pytest.raises(ContractError, match="source document"):
        train_round(config, generation="M002", split=SPLIT,
                    cleaned_batch_ids=["R01", "R02"], labels=labels)


def test_corrections_still_use_local_ids_in_their_own_batch(tmp_path, monkeypatch):
    from data_quality_checker import truth_corrections

    config = canonical_fixture(tmp_path)
    prepare(config, "R01")
    prepare(config, "R02")
    monkeypatch.setattr(truth_corrections, "load_corrections", lambda config, batch:
                        {"d1": [{"kind": "remove", "row": {"kanun_no": "193", "madde": "94"}}]}
                        if batch == "R01" else {})
    clean = round_training_documents(config, batch_ids=["R01", "R02"])
    assert clean['["R01","d1"]']["references"] == []
    assert clean['["R02","d1"]']["references"][0]["madde"] == "94"
    raw, _ = round_raw_training_documents(config, batch_ids=["R01", "R02"])
    assert raw['["R01","d1"]']["references"][0]["madde"] == "94"


def test_release_source_mapping_must_match_the_prepared_document(tmp_path):
    config = canonical_fixture(tmp_path)
    prepare(config, "R01")
    with Store(config.database_path) as store:
        store.connection.execute("UPDATE documents SET raw_document_id='different-source'")
    with pytest.raises(ContractError, match="source document"):
        round_raw_training_documents(config, batch_ids=["R01"])
