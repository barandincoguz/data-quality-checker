from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from data_quality_checker.config import default_config_path, load_config
from data_quality_checker.errors import ContractError
from data_quality_checker.fingerprints import fingerprint_json, sha256_file, sha256_text
from data_quality_checker.preparation import (
    annotation_attribution_path,
    import_annotation_attribution,
    prepare_batch,
    validate_ready,
)
from data_quality_checker.storage import Store

FIELDS = {
    "kanun_no": "213",
    "kanun_ad": "Vergi Usul Kanunu",
    "madde": "1",
    "fikra": "",
    "bent": "",
}


def make_config(tmp_path: Path):
    source = json.loads(default_config_path().read_text(encoding="utf-8"))
    live = load_config()
    source.update(
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
    path.write_text(json.dumps(source), encoding="utf-8")
    return load_config(path)


def write_payload_zip(path: Path, name: str, payload) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, json.dumps(payload, ensure_ascii=False))


def key_file(tmp_path: Path) -> Path:
    path = tmp_path / "hmac.key"
    path.write_bytes(b"k" * 32)
    return path


def _archives(tmp_path: Path, config, count: int) -> tuple[Path, Path, Path]:
    """Build an annotation/pool ZIP pair of `count` documents named
    private-id-1 .. private-id-<count>, for the doc_ids subset tests."""
    annotations = [
        {"document_id": f"private-id-{i}", "current_references": []} for i in range(1, count + 1)
    ]
    pool = [
        {"evrakOid": f"private-id-{i}", "pdfText": f"document text {i}"}
        for i in range(1, count + 1)
    ]
    annotation_zip = tmp_path / "subset-annotations.zip"
    pool_zip = tmp_path / "subset-pool.zip"
    write_payload_zip(annotation_zip, "annotations.json", annotations)
    write_payload_zip(pool_zip, "pool.json", pool)
    return annotation_zip, pool_zip, key_file(tmp_path)


def test_prepare_selects_highest_evidence_channel_and_never_uses_annotation_text(
    tmp_path,
) -> None:
    config = make_config(tmp_path)
    annotations = {
        "annotations": [
            {
                "document_id": "secret-oid-html",
                "text": "ANNOTATION ZIP TEXT MUST NEVER BECOME MODEL INPUT",
                "annotation": {
                    "is_completed": True,
                    "completed_by": {"id": 12, "username": "Irmak Tanrıverdi"},
                    "last_editor": {"id": 12, "username": "Irmak Tanrıverdi"},
                    "edit_count": 1,
                    "unique_users_count": 1,
                },
                "current_references": [{**FIELDS, "source_text": "HTML evidence phrase"}],
            },
            {
                "evrakId": "secret-oid-tie",
                "current_references": [{**FIELDS, "source_text": "shared evidence"}],
            },
            {
                "document_id": "secret-oid-zero",
                "current_references": None,
            },
        ]
    }
    pool = [
        {
            "evrakOid": "secret-oid-html",
            "pdfText": "unrelated PDF content",
            "htmlText": "<p>HTML evidence phrase is present.</p><script>bad()</script>",
        },
        {
            "evrakOid": "secret-oid-tie",
            "pdfText": "shared evidence in PDF",
            "htmlText": "<p>shared evidence in HTML</p>",
        },
        {
            "evrakOid": "secret-oid-zero",
            "pdfText": "pool-only zero-reference text",
            "htmlText": "<p>alternative text</p>",
        },
    ]
    annotation_zip = tmp_path / "annotations.zip"
    pool_zip = tmp_path / "pool.zip"
    write_payload_zip(annotation_zip, "annotations.json", annotations)
    write_payload_zip(pool_zip, "pool.json", pool)

    ready = prepare_batch(
        config=config,
        annotation_zip=annotation_zip,
        document_pool_zip=pool_zip,
        batch_id="fixture",
        hmac_key_file=key_file(tmp_path),
    )

    assert ready["document_count"] == 3
    assert ready["ready_count"] == 3
    assert validate_ready(config, "fixture") == ready
    documents = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(
            (config.sensitive_root / "batches" / "fixture" / "documents").glob("*.json")
        )
    ]
    by_raw_id = {document["raw_document_id"]: document for document in documents}
    assert by_raw_id["secret-oid-html"]["selected_channel"] == "htmlText"
    assert by_raw_id["secret-oid-html"]["text"] == "HTML evidence phrase is present."
    assert "ANNOTATION ZIP TEXT" not in by_raw_id["secret-oid-html"]["text"]
    assert by_raw_id["secret-oid-tie"]["selected_channel"] == "pdfText"
    assert by_raw_id["secret-oid-zero"]["selected_channel"] == "pdfText"
    assert by_raw_id["secret-oid-zero"]["human_references"] == []
    assert by_raw_id["secret-oid-html"]["metadata"]["annotation_attribution"] == {
        "completed_by": {"id": 12, "username": "Irmak Tanrıverdi"},
        "last_editor": {"id": 12, "username": "Irmak Tanrıverdi"},
        "edit_count": 1,
        "unique_users_count": 1,
    }
    assert {warning["code"] for warning in by_raw_id["secret-oid-zero"]["warnings"]} == {
        "current_references_missing_or_null"
    }

    public_text = (config.public_root / "batches" / "fixture" / "manifest.json").read_text(
        encoding="utf-8"
    )
    assert "secret-oid-html" not in public_text
    assert "ANNOTATION ZIP TEXT" not in public_text


def test_import_annotation_attribution_writes_private_idempotent_sidecar(
    tmp_path,
) -> None:
    config = make_config(tmp_path)
    annotations = {
        "annotations": [
            {
                "document_id": "annotated-doc",
                "annotation": {
                    "is_completed": True,
                    "completed_by": {"id": 8, "username": "Uzman Kişi"},
                    "last_editor": {"id": 9, "username": "Son Editör"},
                    "edit_count": 3,
                    "unique_users_count": 2,
                },
                "current_references": [],
            }
        ]
    }
    annotation_zip = tmp_path / "annotations.zip"
    pool_zip = tmp_path / "pool.zip"
    write_payload_zip(annotation_zip, "annotations.json", annotations)
    write_payload_zip(
        pool_zip,
        "pool.json",
        [{"evrakOid": "annotated-doc", "pdfText": "document text"}],
    )
    prepare_batch(
        config=config,
        annotation_zip=annotation_zip,
        document_pool_zip=pool_zip,
        batch_id="attribution-fixture",
        hmac_key_file=key_file(tmp_path),
    )

    first = import_annotation_attribution(
        config=config,
        batch_id="attribution-fixture",
        annotation_zip=annotation_zip,
    )
    second = import_annotation_attribution(
        config=config,
        batch_id="attribution-fixture",
        annotation_zip=annotation_zip,
    )

    assert second == first
    assert first["attributed_document_count"] == 1
    attribution = next(iter(first["attributions"].values()))
    assert attribution["completed_by"]["username"] == "Uzman Kişi"
    assert attribution["last_editor"]["username"] == "Son Editör"
    assert annotation_attribution_path(config, "attribution-fixture").exists()


def test_missing_reference_fields_warn_while_invalid_types_and_duplicates_quarantine(
    tmp_path,
) -> None:
    config = make_config(tmp_path)
    annotations = [
        {
            "document_id": "duplicate-annotation",
            "current_references": [],
        },
        {
            "document_id": "duplicate-annotation",
            "current_references": [],
        },
        {
            "document_id": "ambiguous-pool",
            "current_references": [],
        },
        {
            "document_id": "invalid-refs",
            "current_references": {"not": "a-list"},
        },
        {
            "document_id": "missing-field-warning",
            "current_references": [{"kanun_no": "213", "source_text": None}],
        },
    ]
    pool = [
        {"evrakOid": "duplicate-annotation", "pdfText": "text"},
        {"evrakOid": "ambiguous-pool", "pdfText": "one"},
        {"evrakOid": "ambiguous-pool", "pdfText": "two"},
        {"evrakOid": "invalid-refs", "pdfText": "text"},
        {"evrakOid": "missing-field-warning", "pdfText": "text"},
    ]
    annotation_zip = tmp_path / "annotations.zip"
    pool_zip = tmp_path / "pool.zip"
    write_payload_zip(annotation_zip, "annotations.json", annotations)
    write_payload_zip(pool_zip, "pool.json", pool)

    ready = prepare_batch(
        config=config,
        annotation_zip=annotation_zip,
        document_pool_zip=pool_zip,
        batch_id="quarantine-fixture",
        hmac_key_file=key_file(tmp_path),
    )

    assert ready["document_count"] == 4
    assert ready["quarantine_count"] == 3
    with Store(config.database_path) as store:
        documents = store.list_documents("quarantine-fixture")
    statuses = {row["raw_document_id"]: row["preparation_status"] for row in documents}
    assert statuses["duplicate-annotation"] == "quarantine"
    assert statuses["ambiguous-pool"] == "quarantine"
    assert statuses["invalid-refs"] == "quarantine"
    assert statuses["missing-field-warning"] == "ready"

    ready_document = next(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (config.sensitive_root / "batches" / "quarantine-fixture" / "documents").glob(
            "*.json"
        )
        if "missing-field-warning" in path.read_text(encoding="utf-8")
    )
    reference = ready_document["human_references"][0]
    assert reference == {
        "kanun_no": "213",
        "kanun_ad": "",
        "madde": "",
        "fikra": "",
        "bent": "",
        "source_text": "",
    }
    assert "reference_missing_field" in {warning["code"] for warning in ready_document["warnings"]}


def test_prepare_is_idempotent_for_same_fingerprints(tmp_path) -> None:
    config = make_config(tmp_path)
    annotations = [{"document_id": "one", "current_references": []}]
    pool = [{"evrakOid": "one", "pdfText": "document"}]
    annotation_zip = tmp_path / "annotations.zip"
    pool_zip = tmp_path / "pool.zip"
    key = key_file(tmp_path)
    write_payload_zip(annotation_zip, "annotations.json", annotations)
    write_payload_zip(pool_zip, "pool.json", pool)

    first = prepare_batch(
        config=config,
        annotation_zip=annotation_zip,
        document_pool_zip=pool_zip,
        batch_id="resume-fixture",
        hmac_key_file=key,
    )
    second = prepare_batch(
        config=config,
        annotation_zip=annotation_zip,
        document_pool_zip=pool_zip,
        batch_id="resume-fixture",
        hmac_key_file=key,
    )
    assert second == first


def test_a_subset_restricts_the_batch(tmp_path) -> None:
    config = make_config(tmp_path)
    annotation_zip, pool_zip, keyfile = _archives(tmp_path, config, count=5)

    prepare_batch(
        config=config,
        annotation_zip=annotation_zip,
        document_pool_zip=pool_zip,
        batch_id="subset",
        hmac_key_file=keyfile,
        doc_ids={"private-id-1", "private-id-3"},
    )

    with Store(config.database_path) as store:
        documents = store.list_documents("subset")
    assert len(documents) == 2
    assert {row["raw_document_id"] for row in documents} == {"private-id-1", "private-id-3"}


def test_no_subset_keeps_every_document(tmp_path) -> None:
    config = make_config(tmp_path)
    annotation_zip, pool_zip, keyfile = _archives(tmp_path, config, count=5)

    prepare_batch(
        config=config,
        annotation_zip=annotation_zip,
        document_pool_zip=pool_zip,
        batch_id="whole",
        hmac_key_file=keyfile,
    )

    with Store(config.database_path) as store:
        documents = store.list_documents("whole")
    assert len(documents) == 5


def test_a_subset_naming_an_absent_document_is_rejected(tmp_path) -> None:
    config = make_config(tmp_path)
    annotation_zip, pool_zip, keyfile = _archives(tmp_path, config, count=5)

    with pytest.raises(ContractError):
        prepare_batch(
            config=config,
            annotation_zip=annotation_zip,
            document_pool_zip=pool_zip,
            batch_id="missing",
            hmac_key_file=keyfile,
            doc_ids={"private-id-1", "not-in-the-archive"},
        )


def test_an_empty_subset_is_rejected(tmp_path) -> None:
    config = make_config(tmp_path)
    annotation_zip, pool_zip, keyfile = _archives(tmp_path, config, count=5)

    with pytest.raises(ContractError):
        prepare_batch(
            config=config,
            annotation_zip=annotation_zip,
            document_pool_zip=pool_zip,
            batch_id="empty",
            hmac_key_file=keyfile,
            doc_ids=set(),
        )


def test_two_different_subsets_of_the_same_archives_get_different_batch_ids(
    tmp_path,
) -> None:
    """FIX 1 regression: before the fix, `input_fingerprint` (and the batch id
    derived from it when `batch_id=None`) was computed from the archives only,
    so two different `doc_ids` subsets of the same archives collided on one
    batch id - `create_batch` saw a matching fingerprint, found the batch
    already READY, and handed round 2 round 1's batch."""

    config = make_config(tmp_path)
    annotation_zip, pool_zip, keyfile = _archives(tmp_path, config, count=5)

    round1 = prepare_batch(
        config=config,
        annotation_zip=annotation_zip,
        document_pool_zip=pool_zip,
        batch_id=None,
        hmac_key_file=keyfile,
        doc_ids={"private-id-1"},
    )
    round2 = prepare_batch(
        config=config,
        annotation_zip=annotation_zip,
        document_pool_zip=pool_zip,
        batch_id=None,
        hmac_key_file=keyfile,
        doc_ids={"private-id-4", "private-id-5"},
    )

    assert round1["batch_id"] != round2["batch_id"]
    assert round1["input_fingerprint"] != round2["input_fingerprint"]
    with Store(config.database_path) as store:
        round1_documents = store.list_documents(round1["batch_id"])
        round2_documents = store.list_documents(round2["batch_id"])
    assert {row["raw_document_id"] for row in round1_documents} == {"private-id-1"}
    assert {row["raw_document_id"] for row in round2_documents} == {
        "private-id-4",
        "private-id-5",
    }


def test_the_same_subset_prepared_twice_is_idempotent(tmp_path) -> None:
    config = make_config(tmp_path)
    annotation_zip, pool_zip, keyfile = _archives(tmp_path, config, count=5)

    first = prepare_batch(
        config=config,
        annotation_zip=annotation_zip,
        document_pool_zip=pool_zip,
        batch_id=None,
        hmac_key_file=keyfile,
        doc_ids={"private-id-2", "private-id-3"},
    )
    second = prepare_batch(
        config=config,
        annotation_zip=annotation_zip,
        document_pool_zip=pool_zip,
        batch_id=None,
        hmac_key_file=keyfile,
        doc_ids={"private-id-2", "private-id-3"},
    )

    assert second == first


def test_a_no_subset_prepare_keeps_todays_fingerprint(tmp_path) -> None:
    """FIX 1 must not change the fingerprint (and therefore the batch id) of
    an ordinary whole-archive prepare: an existing batch must not be
    invalidated by folding `doc_ids` into the input contract. This recomputes
    the pre-fix contract shape by hand (no `doc_ids` key at all) and checks
    it against what `prepare_batch` produces today."""

    config = make_config(tmp_path)
    annotation_zip, pool_zip, keyfile = _archives(tmp_path, config, count=3)
    key = keyfile.read_bytes().strip()

    todays_contract = {
        "annotation_zip": {
            "size": annotation_zip.stat().st_size,
            "sha256": sha256_file(annotation_zip),
        },
        "document_pool_zip": {
            "size": pool_zip.stat().st_size,
            "sha256": sha256_file(pool_zip),
        },
        "hmac_key_fingerprint": sha256_text(key.hex()),
    }
    todays_fingerprint = fingerprint_json(todays_contract)

    ready = prepare_batch(
        config=config,
        annotation_zip=annotation_zip,
        document_pool_zip=pool_zip,
        batch_id=None,
        hmac_key_file=keyfile,
    )

    assert ready["input_fingerprint"] == todays_fingerprint
    assert ready["batch_id"] == f"dq_{todays_fingerprint[:16]}"


def test_a_malformed_record_elsewhere_does_not_block_an_unrelated_subset(
    tmp_path,
) -> None:
    """FIX 2 regression: the subset filter used to call `_record_document_id`
    outside the try/except the main ingestion loop wraps it in, so one record
    anywhere in the pool whose evrakId/document_id disagree aborted every
    subset prepare - even one that never asked for that record. Reproduced
    here with two unrelated documents requested and a third, unrelated,
    id-disagreeing record present in the same archive."""

    config = make_config(tmp_path)
    annotations = [
        {"document_id": "target-a", "current_references": []},
        {"document_id": "target-b", "current_references": []},
        {"evrakId": "bad-1", "document_id": "bad-2", "current_references": []},
    ]
    pool = [
        {"evrakOid": "target-a", "pdfText": "text a"},
        {"evrakOid": "target-b", "pdfText": "text b"},
        {"evrakOid": "bad-1", "pdfText": "text bad"},
    ]
    annotation_zip = tmp_path / "annotations.zip"
    pool_zip = tmp_path / "pool.zip"
    write_payload_zip(annotation_zip, "annotations.json", annotations)
    write_payload_zip(pool_zip, "pool.json", pool)

    ready = prepare_batch(
        config=config,
        annotation_zip=annotation_zip,
        document_pool_zip=pool_zip,
        batch_id="subset-with-bad-record",
        hmac_key_file=key_file(tmp_path),
        doc_ids={"target-a", "target-b"},
    )

    assert ready["document_count"] == 2
    with Store(config.database_path) as store:
        documents = store.list_documents("subset-with-bad-record")
    assert {row["raw_document_id"] for row in documents} == {"target-a", "target-b"}
