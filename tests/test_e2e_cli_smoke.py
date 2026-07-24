"""Plan acceptance Gate 3: fixture-based end-to-end smoke through the CLI.

Drives the public `dqcheck` lifecycle `prepare -> process -> status -> release`
via `cli.main()` (argument wiring included), so the acceptance gate is enforced
by a single named artifact rather than only implied by per-stage library tests.

The review/finalize step has no CLI subcommand by design (it happens through the
local-only `serve` HITL UI), so it is driven directly against the Store here.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from data_quality_checker.cli import main
from data_quality_checker.config import default_config_path, load_config
from data_quality_checker.judges import ensure_green_audit_plan
from data_quality_checker.storage import Store


def _write_config(tmp_path: Path) -> Path:
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
    return path


def _zip(path: Path, payload) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data.json", json.dumps(payload, ensure_ascii=False))


def _finalize_green_audit(config) -> None:
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
                reviewer="e2e-smoke",
            )


def test_cli_prepare_process_status_release_smoke(tmp_path) -> None:
    config_path = _write_config(tmp_path)
    config = load_config(config_path)

    reference = {
        "kanun_no": "213",
        "kanun_ad": "Vergi Usul Kanunu",
        "madde": "413",
        "fikra": "",
        "bent": "",
        "source_text": "213 sayılı Vergi Usul Kanununun 413. maddesi",
    }
    count = 31
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
    _zip(annotation_zip, {"annotations": annotations})
    _zip(pool_zip, pool)
    key = tmp_path / "key"
    key.write_bytes(b"e" * 32)

    base = ["--config", str(config_path)]

    assert main(
        base
        + [
            "prepare",
            "--annotation-zip",
            str(annotation_zip),
            "--document-pool-zip",
            str(pool_zip),
            "--batch-id",
            "batch",
            "--hmac-key-file",
            str(key),
        ]
    ) == 0
    assert main(
        base
        + ["process", "--prepared-batch", "batch", "--generation", "G0", "--fake-backend"]
    ) == 0
    assert main(base + ["status", "--batch-id", "batch"]) == 0

    _finalize_green_audit(config)

    assert main(base + ["release", "--batch-id", "batch"]) == 0

    release_root = config.sensitive_root / "releases" / "batch"
    releases = [p for p in release_root.iterdir() if p.is_dir()]
    assert len(releases) == 1
    release_path = releases[0]
    for artifact in (
        "expert_adjudicated.jsonl",
        "consensus_clean.jsonl",
        "quarantine.jsonl",
        "training_export.jsonl",
        "manifest.json",
        "SHA256SUMS.txt",
    ):
        assert (release_path / artifact).is_file(), f"missing release artifact: {artifact}"

    # Release is idempotent: re-running the CLI must not create a second release.
    assert main(base + ["release", "--batch-id", "batch"]) == 0
    assert len([p for p in release_root.iterdir() if p.is_dir()]) == 1
