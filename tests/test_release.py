from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

import data_quality_checker.release as release_module
from data_quality_checker.config import default_config_path, load_config
from data_quality_checker.errors import GateBlocked, IntegrityError
from data_quality_checker.judges import ensure_green_audit_plan
from data_quality_checker.preparation import prepare_batch
from data_quality_checker.processing import process_batch
from data_quality_checker.release import release_batch
from data_quality_checker.storage import Store


def config_for(tmp_path: Path):
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


def fixture(tmp_path: Path, *, count: int):
    config = config_for(tmp_path)
    reference = {
        "kanun_no": "213",
        "kanun_ad": "Vergi Usul Kanunu",
        "madde": "413",
        "fikra": "",
        "bent": "",
        "source_text": "213 sayılı Vergi Usul Kanununun 413. maddesi",
    }
    annotations = [
        {"document_id": f"private-{index}", "current_references": [reference]}
        for index in range(count)
    ]
    pool = [
        {
            "evrakOid": f"private-{index}",
            "pdfText": "213 sayılı Vergi Usul Kanununun 413. maddesi uygulanır.",
        }
        for index in range(count)
    ]
    annotation_zip, pool_zip = tmp_path / "a.zip", tmp_path / "p.zip"
    for path, payload in (
        (annotation_zip, {"annotations": annotations}),
        (pool_zip, pool),
    ):
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("data.json", json.dumps(payload, ensure_ascii=False))
    key = tmp_path / "key"
    key.write_bytes(b"r" * 32)
    prepare_batch(
        config=config,
        annotation_zip=annotation_zip,
        document_pool_zip=pool_zip,
        batch_id="batch",
        hmac_key_file=key,
    )
    process_batch(
        config=config,
        batch_id="batch",
        generation="G0",
        resume=False,
        fake_backend=True,
    )
    return config


def finalize_green_audit(config) -> list[str]:
    with Store(config.database_path) as store:
        audit = ensure_green_audit_plan(config=config, store=store, batch_id="batch")
        for doc_id in audit["sample_internal_doc_ids"]:
            document = store.get_document("batch", doc_id)
            review = store.get_review("batch", doc_id)
            assert document is not None and review is not None
            store.update_review(
                batch_id="batch",
                internal_doc_id=doc_id,
                expected_version=review["row_version"],
                status="finalized",
                action="accept_human",
                final_references=json.loads(document["human_references_json"]),
                reason=None,
                reviewer="fixture",
            )
        return list(audit["sample_internal_doc_ids"])


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_release_separates_expert_and_consensus_and_exact_training_export(tmp_path) -> None:
    config = fixture(tmp_path, count=31)
    audited = finalize_green_audit(config)
    assert len(audited) == 30

    summary = release_batch(config=config, batch_id="batch")
    assert summary["status"] == "released"
    assert summary["counts"] == {
        "documents": 31,
        "expert_adjudicated": 30,
        "consensus_clean": 1,
        "quarantine": 0,
        "training_export": 31,
    }
    release_path = Path(summary["release_path"])
    expert = read_jsonl(release_path / "expert_adjudicated.jsonl")
    consensus = read_jsonl(release_path / "consensus_clean.jsonl")
    training = read_jsonl(release_path / "training_export.jsonl")
    assert {row["release_trust"] for row in expert} == {"expert_adjudicated"}
    assert {row["release_trust"] for row in consensus} == {"consensus_clean"}
    assert {row["internal_doc_id"] for row in training} == {
        row["internal_doc_id"] for row in expert + consensus
    }
    assert (release_path / "SHA256SUMS.txt").is_file()

    again = release_batch(config=config, batch_id="batch")
    assert again["release_id"] == summary["release_id"]
    assert again["idempotent_existing_release"] is True


def test_pending_or_deferred_review_blocks_release(tmp_path) -> None:
    config = fixture(tmp_path, count=1)
    with pytest.raises(GateBlocked, match="incomplete"):
        release_batch(config=config, batch_id="batch")

    with Store(config.database_path) as store:
        review = store.list_reviews("batch")[0]
        store.update_review(
            batch_id="batch",
            internal_doc_id=review["internal_doc_id"],
            expected_version=review["row_version"],
            status="deferred",
            action="defer",
            final_references=None,
            reason="needs research",
            reviewer="fixture",
        )
    with pytest.raises(GateBlocked, match="deferred"):
        release_batch(config=config, batch_id="batch")


def test_release_directory_promotion_failure_leaves_no_partial_release(
    monkeypatch, tmp_path
) -> None:
    config = fixture(tmp_path, count=1)
    finalize_green_audit(config)

    def fail_rename(source, target):
        raise OSError("simulated atomic directory promotion failure")

    monkeypatch.setattr(release_module.os, "rename", fail_rename)
    with pytest.raises(OSError, match="simulated"):
        release_batch(config=config, batch_id="batch")
    release_parent = config.sensitive_root / "releases" / "batch"
    assert list(release_parent.iterdir()) == []


def test_immutable_release_tampering_is_detected(tmp_path) -> None:
    config = fixture(tmp_path, count=1)
    finalize_green_audit(config)
    summary = release_batch(config=config, batch_id="batch")
    release_path = Path(summary["release_path"])
    (release_path / "training_export.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(IntegrityError, match="size mismatch|checksum mismatch"):
        release_batch(config=config, batch_id="batch")
