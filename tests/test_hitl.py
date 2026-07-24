from __future__ import annotations

import json
import zipfile
from pathlib import Path

from data_quality_checker.config import default_config_path, load_config
from data_quality_checker.hitl import create_hitl_app
from data_quality_checker.preparation import prepare_batch
from data_quality_checker.processing import process_batch


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
    reference = {
        "kanun_no": "213",
        "kanun_ad": "Vergi Usul Kanunu",
        "madde": "413",
        "fikra": "",
        "bent": "",
        "source_text": "213 sayılı Vergi Usul Kanununun 413. maddesi",
    }
    annotations = [
        {"document_id": f"doc-{index}", "current_references": [reference]}
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


def test_authentication_csrf_blinding_and_optimistic_version(tmp_path) -> None:
    config, app = fixture(tmp_path)
    client = app.test_client()
    assert client.get("/api/queue").status_code == 401
    csrf = authenticated(client)
    queue = client.get("/api/queue").get_json()["queue"]
    assert len(queue) == 1
    doc_id = queue[0]["internal_doc_id"]
    document = client.get(f"/api/documents/{doc_id}").get_json()
    assert "blind_mapping_revealed" not in document
    assert document["candidate_a"] and document["candidate_b"]

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
    assert accepted.get_json()["action"] in {"accept_human", "accept_model"}
    assert accepted.get_json()["blind_mapping_revealed"] in {
        "A=human,B=model",
        "A=model,B=human",
    }

    stale = client.post(
        f"/api/reviews/{doc_id}",
        json={"action": "accept_candidate_a", "row_version": 0},
        headers={"X-CSRF-Token": csrf},
    )
    assert stale.status_code == 409


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
    revised[0]["source_text"] = (
        "213 sayılı Vergi Usul Kanununun 413. maddesi uygulanır"
    )
    response = client.post(
        f"/api/reviews/{item['internal_doc_id']}",
        json={"action": "revise", "references": revised, "row_version": 0},
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 200
    assert response.get_json()["green_escalated"] is False
    assert not (
        config.public_root / "batches" / "batch" / "green_escalation.json"
    ).exists()


def test_review_page_renders_diff_table_progress_and_editor(tmp_path) -> None:
    _, app = fixture(tmp_path)
    client = app.test_client()
    authenticated(client)
    doc_id = client.get("/api/queue").get_json()["queue"][0]["internal_doc_id"]
    page = client.get(f"/review/{doc_id}")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "A / B karşılaştırması" in html
    assert "Revize editörü" in html
    assert 'id="ref-rows"' in html
    assert 'id="fill-a"' in html
    assert 'id="cand"' in html
    assert "(1/1)" in html


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
    revised = list(document["candidate_a"])
    revised.append(
        {
            "kanun_no": "193",
            "kanun_ad": "Gelir Vergisi Kanunu",
            "madde": "94",
            "fikra": "",
            "bent": "",
            "source_text": "193 sayılı Gelir Vergisi Kanununun 94. maddesi",
        }
    )
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
    assert (
        config.public_root / "batches" / "batch" / "green_escalation.json"
    ).is_file()
