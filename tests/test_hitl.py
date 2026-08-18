from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from data_quality_checker.cli import main
from data_quality_checker.config import default_config_path, load_config
from data_quality_checker.errors import IntegrityError
from data_quality_checker.fingerprints import sha256_file
from data_quality_checker.hitl import create_hitl_app
from data_quality_checker.preparation import prepare_batch
from data_quality_checker.processing import process_batch
from data_quality_checker.storage import Store

SECRET = "s" * 32
TOKEN = "t" * 32


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


def fixture(tmp_path: Path, *, count: int = 1):
    config = config_for(tmp_path)
    references = [
        {
            "kanun_no": "213",
            "kanun_ad": "Vergi Usul Kanunu",
            "madde": "413",
            "fikra": "",
            "bent": "",
            "source_text": "213 sayılı Vergi Usul Kanununun 413. maddesi",
        },
        {
            "kanun_no": "193",
            "kanun_ad": "Gelir Vergisi Kanunu",
            "madde": "94",
            "fikra": "",
            "bent": "",
            "source_text": "193 sayılı Gelir Vergisi Kanununun 94. maddesi",
        },
    ]
    annotations = [
        {
            "document_id": f"doc-{index}",
            "annotation": {
                "is_completed": True,
                "completed_by": {"id": 7, "username": "Test Anotatör"},
                "last_editor": {"id": 7, "username": "Test Anotatör"},
                "edit_count": 2,
                "unique_users_count": 1,
            },
            "current_references": references,
        }
        for index in range(count)
    ]
    pool = [
        {
            "evrakOid": f"doc-{index}",
            "pdfText": (
                "213 sayılı Vergi Usul Kanununun 413. maddesi uygulanır. "
                "193 sayılı Gelir Vergisi Kanununun 94. maddesi de uygulanır."
            ),
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
    key.write_bytes(b"h" * 32)
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
    app = create_hitl_app(
        config=config,
        batch_id="batch",
        testing=True,
        session_secret=SECRET,
        access_token=TOKEN,
    )
    return config, app


def authenticated(client):
    response = client.get("/api/session", headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 200
    return response.get_json()["csrf_token"]


def test_authentication_attribution_and_optimistic_version(tmp_path) -> None:
    config, app = fixture(tmp_path)
    client = app.test_client()
    assert client.get("/api/queue").status_code == 401
    csrf = authenticated(client)
    queue = client.get("/api/queue").get_json()["queue"]
    assert len(queue) == 1
    doc_id = queue[0]["internal_doc_id"]
    document = client.get(f"/api/documents/{doc_id}").get_json()
    assert document["candidate_mapping"] == "A=human,B=model"
    assert document["human_attribution"]["display_name"] == "Test Anotatör"
    assert document["candidate_a"] and document["candidate_b"]
    assert document["reference_policy"]["removed_reference_count"] == 2
    assert all(
        not (row["kanun_no"] == "213" and row["madde"] == "413")
        for candidate in (document["candidate_a"], document["candidate_b"])
        for row in candidate
    )

    no_csrf = client.post(
        f"/api/reviews/{doc_id}",
        json={"action": "accept_candidate_a", "row_version": 0},
    )
    assert no_csrf.status_code == 403

    accepted = client.post(
        f"/api/reviews/{doc_id}",
        json={"action": "accept_candidate_a", "row_version": 0},
        headers={"X-CSRF-Token": csrf},
    )
    assert accepted.status_code == 200
    assert accepted.get_json()["action"] == "accept_human"
    assert accepted.get_json()["candidate_mapping"] == "A=human,B=model"

    stale = client.post(
        f"/api/reviews/{doc_id}",
        json={"action": "accept_candidate_a", "row_version": 0},
        headers={"X-CSRF-Token": csrf},
    )
    assert stale.status_code == 409


def test_successful_review_is_present_in_a_verified_sqlite_backup(tmp_path) -> None:
    config, app = fixture(tmp_path)
    client = app.test_client()
    csrf = authenticated(client)
    doc_id = client.get("/api/queue").get_json()["queue"][0]["internal_doc_id"]

    response = client.post(
        f"/api/reviews/{doc_id}",
        json={"action": "accept_human", "row_version": 0},
        headers={"X-CSRF-Token": csrf},
    )

    assert response.status_code == 200
    durability = response.get_json()["durability"]
    assert durability["status"] == "verified"
    latest_path = config.sensitive_root / "review_backups" / "batch" / "LATEST.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    assert latest["latest_review_event_id"] == response.get_json()["review_event_id"]
    snapshot_path = Path(latest["snapshot_path"])
    assert snapshot_path.is_file()
    assert sha256_file(snapshot_path) == latest["snapshot_sha256"]
    with Store(snapshot_path) as snapshot:
        review = snapshot.get_review("batch", doc_id)
        assert review is not None
        assert review["status"] == "finalized"
        assert review["action"] == "accept_human"


def test_app_startup_recreates_a_missing_review_backup(tmp_path) -> None:
    config, app = fixture(tmp_path)
    client = app.test_client()
    csrf = authenticated(client)
    doc_id = client.get("/api/queue").get_json()["queue"][0]["internal_doc_id"]
    response = client.post(
        f"/api/reviews/{doc_id}",
        json={"action": "accept_human", "row_version": 0},
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 200
    backup_root = config.sensitive_root / "review_backups" / "batch"
    shutil.rmtree(backup_root)

    create_hitl_app(
        config=config,
        batch_id="batch",
        testing=True,
        session_secret=SECRET,
        access_token=TOKEN,
    )

    latest = json.loads((backup_root / "LATEST.json").read_text(encoding="utf-8"))
    assert latest["status"] == "verified"
    assert latest["review_count"] == 1


def test_app_startup_rejects_a_corrupted_latest_review_snapshot(tmp_path) -> None:
    config, _ = fixture(tmp_path)
    latest_path = config.sensitive_root / "review_backups" / "batch" / "LATEST.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    Path(latest["snapshot_path"]).write_bytes(b"corrupted snapshot")

    with pytest.raises(IntegrityError, match="backup"):
        create_hitl_app(
            config=config,
            batch_id="batch",
            testing=True,
            session_secret=SECRET,
            access_token=TOKEN,
        )


def test_committed_review_reports_durability_pending_when_backup_fails(
    tmp_path,
) -> None:
    config, app = fixture(tmp_path)
    client = app.test_client()
    csrf = authenticated(client)
    doc_id = client.get("/api/queue").get_json()["queue"][0]["internal_doc_id"]
    backup_root = config.sensitive_root / "review_backups"
    shutil.rmtree(backup_root)
    backup_root.write_text("blocks directory creation", encoding="utf-8")

    response = client.post(
        f"/api/reviews/{doc_id}",
        json={"action": "accept_human", "row_version": 0},
        headers={"X-CSRF-Token": csrf},
    )

    assert response.status_code == 503
    payload = response.get_json()
    assert payload["error"] == "durability_pending"
    assert payload["review_saved"] is True
    assert payload["durability"]["status"] == "pending"
    document = client.get(f"/api/documents/{doc_id}").get_json()
    assert document["review_status"] == "finalized"


def test_startup_catches_up_after_a_committed_review_backup_failure(tmp_path) -> None:
    config, app = fixture(tmp_path)
    client = app.test_client()
    csrf = authenticated(client)
    doc_id = client.get("/api/queue").get_json()["queue"][0]["internal_doc_id"]
    backup_root = config.sensitive_root / "review_backups"
    shutil.rmtree(backup_root)
    backup_root.write_text("blocks directory creation", encoding="utf-8")
    response = client.post(
        f"/api/reviews/{doc_id}",
        json={"action": "accept_human", "row_version": 0},
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 503

    backup_root.unlink()
    create_hitl_app(
        config=config,
        batch_id="batch",
        testing=True,
        session_secret=SECRET,
        access_token=TOKEN,
    )

    latest = json.loads((backup_root / "batch" / "LATEST.json").read_text(encoding="utf-8"))
    assert latest["status"] == "verified"
    assert latest["review_count"] == 1
    with Store(config.database_path) as store:
        assert store.get_review("batch", doc_id)["status"] == "finalized"


def test_form_review_backup_failure_warns_not_to_resubmit(tmp_path) -> None:
    config, app = fixture(tmp_path)
    client = app.test_client()
    csrf = authenticated(client)
    doc_id = client.get("/api/queue").get_json()["queue"][0]["internal_doc_id"]
    backup_root = config.sensitive_root / "review_backups"
    shutil.rmtree(backup_root)
    backup_root.write_text("blocks directory creation", encoding="utf-8")

    response = client.post(
        f"/review/{doc_id}",
        data={
            "action": "accept_human",
            "row_version": "0",
            "csrf_token": csrf,
        },
    )

    assert response.status_code == 503
    assert "Karar ana veritabanına kaydedildi" in response.get_data(as_text=True)
    assert "Kararı tekrar göndermeyin" in response.get_data(as_text=True)


def test_review_backup_retains_the_latest_five_verified_snapshots(tmp_path) -> None:
    config, app = fixture(tmp_path, count=7)
    client = app.test_client()
    csrf = authenticated(client)
    for _ in range(7):
        item = client.get("/api/queue").get_json()["queue"][0]
        response = client.post(
            f"/api/reviews/{item['internal_doc_id']}",
            json={"action": "accept_human", "row_version": item["row_version"]},
            headers={"X-CSRF-Token": csrf},
        )
        assert response.status_code == 200

    backup_root = config.sensitive_root / "review_backups" / "batch"
    snapshots = sorted((backup_root / "snapshots").glob("*.sqlite3"))
    latest = json.loads((backup_root / "LATEST.json").read_text(encoding="utf-8"))
    assert len(snapshots) == 5
    assert Path(latest["snapshot_path"]) in snapshots
    for snapshot_path in snapshots:
        with Store(snapshot_path) as snapshot:
            assert snapshot.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_review_backup_status_cli_reports_verified_current_state(tmp_path, capsys) -> None:
    config, _ = fixture(tmp_path)

    exit_code = main(
        [
            "--config",
            str(config.source_path),
            "review-backup",
            "status",
            "--batch-id",
            "batch",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "verified"
    assert payload["review_count"] == 0
    assert payload["retained_snapshot_count"] == 1


def test_review_backup_restore_smoke_cli_reloads_latest_snapshot(tmp_path, capsys) -> None:
    config, _ = fixture(tmp_path)

    exit_code = main(
        [
            "--config",
            str(config.source_path),
            "review-backup",
            "restore-smoke",
            "--batch-id",
            "batch",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert payload["restored_review_count"] == 0
    assert payload["review_state_fingerprint"]


def test_review_backup_create_and_verify_cli_are_explicit_operations(tmp_path, capsys) -> None:
    config, _ = fixture(tmp_path)

    create_exit = main(
        [
            "--config",
            str(config.source_path),
            "review-backup",
            "create",
            "--batch-id",
            "batch",
        ]
    )
    created = json.loads(capsys.readouterr().out)
    verify_exit = main(
        [
            "--config",
            str(config.source_path),
            "review-backup",
            "verify",
            "--batch-id",
            "batch",
        ]
    )
    verified = json.loads(capsys.readouterr().out)

    assert create_exit == 0
    assert created["status"] == "verified"
    assert verify_exit == 0
    assert verified["status"] == "verified"
    assert verified["snapshot_sha256"] == created["snapshot_sha256"]


def test_invalid_evidence_and_defer_without_reason_are_rejected(tmp_path) -> None:
    _, app = fixture(tmp_path)
    client = app.test_client()
    csrf = authenticated(client)
    doc_id = client.get("/api/queue").get_json()["queue"][0]["internal_doc_id"]
    bad_reference = {
        "kanun_no": "999",
        "kanun_ad": "Uydurma Kanun",
        "madde": "1",
        "fikra": "",
        "bent": "",
        "source_text": "belgede olmayan kanıt",
    }
    response = client.post(
        f"/api/reviews/{doc_id}",
        json={"action": "revise", "references": [bad_reference], "row_version": 0},
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 400
    deferred = client.post(
        f"/api/reviews/{doc_id}",
        json={"action": "defer", "row_version": 0},
        headers={"X-CSRF-Token": csrf},
    )
    assert deferred.status_code == 400


def test_evidence_only_green_edit_does_not_escalate(tmp_path) -> None:
    config, app = fixture(tmp_path)
    client = app.test_client()
    csrf = authenticated(client)
    item = client.get("/api/queue").get_json()["queue"][0]
    document = client.get(f"/api/documents/{item['internal_doc_id']}").get_json()
    revised = document["candidate_a"]
    revised[0]["source_text"] = "213 sayılı Vergi Usul Kanununun 413. maddesi uygulanır"
    response = client.post(
        f"/api/reviews/{item['internal_doc_id']}",
        json={"action": "revise", "references": revised, "row_version": 0},
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 200
    assert response.get_json()["green_escalated"] is False
    assert not (config.public_root / "batches" / "batch" / "green_escalation.json").exists()


def test_review_page_renders_diff_table_progress_and_editor(tmp_path) -> None:
    _, app = fixture(tmp_path)
    client = app.test_client()
    authenticated(client)
    doc_id = client.get("/api/queue").get_json()["queue"][0]["internal_doc_id"]
    page = client.get(f"/review/{doc_id}")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "1. Belge metnini kontrol et" in html
    assert "2. İnsan anotasyonu ile modeli karşılaştır" in html
    assert "3. Karar ver" in html
    assert "İnsan anotasyonu doğru" in html
    assert "Model çıktısı doğru" in html
    assert "Test Anotatör" in html
    assert "Tamamlayan: Test Anotatör" in html
    assert "Sol sütun insan anotasyonudur" in html
    assert "Kayıt koruması: Güncel" in html
    assert "0 karar yedeklendi" in html
    assert "İkisi de tam doğru değil — referansları düzelt" in html
    assert 'id="ref-rows"' in html
    assert 'id="fill-a"' in html
    assert 'id="cand"' in html
    assert "Belge 1 / 1" in html
    # No-JS fallback: the raw references_json textarea stays available so
    # revision still works with JavaScript disabled (spec invariant).
    assert 'name="references_json"' in html
    assert 'value="revise"' in html
    # Evidence highlighting: the document text renders with <mark> spans and a
    # legend; the candidate source_text is highlighted within the doc text.
    assert "İnsan anotasyonu kanıtı" in html
    assert '<mark class="ev' in html
    assert 'class="doctext"' in html
    assert '<details class="technical">' in html


def test_form_revise_via_references_json_finalizes(tmp_path) -> None:
    _, app = fixture(tmp_path)
    client = app.test_client()
    csrf = authenticated(client)
    doc_id = client.get("/api/queue").get_json()["queue"][0]["internal_doc_id"]
    document = client.get(f"/api/documents/{doc_id}").get_json()
    reference = document["candidate_a"][0]
    response = client.post(
        f"/review/{doc_id}",
        data={
            "csrf_token": csrf,
            "row_version": "0",
            "action": "revise",
            "references_json": json.dumps([reference]),
        },
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 302  # form submit redirects to next review
    assert client.get("/api/queue").get_json()["queue"] == []


def test_green_membership_change_escalates_all_remaining_green(tmp_path) -> None:
    config, app = fixture(tmp_path, count=35)
    client = app.test_client()
    csrf = authenticated(client)
    queue_before = client.get("/api/queue").get_json()["queue"]
    assert len(queue_before) == 30
    sampled_ids = {row["internal_doc_id"] for row in queue_before}
    first = queue_before[0]
    document = client.get(f"/api/documents/{first['internal_doc_id']}").get_json()
    assert document["candidate_a"]
    revised = []
    response = client.post(
        f"/api/reviews/{first['internal_doc_id']}",
        json={"action": "revise", "references": revised, "row_version": 0},
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 200
    assert response.get_json()["green_escalated"] is True
    queue_after = client.get("/api/queue").get_json()["queue"]
    assert len(queue_after) == 34
    assert any(row["internal_doc_id"] not in sampled_ids for row in queue_after)
    assert (config.public_root / "batches" / "batch" / "green_escalation.json").is_file()
