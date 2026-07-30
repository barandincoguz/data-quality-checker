from __future__ import annotations

import json
import zipfile
from pathlib import Path

from data_quality_checker.config import default_config_path, load_config
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
                "current_references": [
                    {**FIELDS, "source_text": "HTML evidence phrase"}
                ],
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
    assert {
        warning["code"] for warning in by_raw_id["secret-oid-zero"]["warnings"]
    } == {"current_references_missing_or_null"}

    public_text = (
        config.public_root / "batches" / "fixture" / "manifest.json"
    ).read_text(encoding="utf-8")
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
        for path in (
            config.sensitive_root / "batches" / "quarantine-fixture" / "documents"
        ).glob("*.json")
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
    assert "reference_missing_field" in {
        warning["code"] for warning in ready_document["warnings"]
    }


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
