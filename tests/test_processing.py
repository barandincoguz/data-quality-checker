from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_quality_checker.config import default_config_path, load_config
from data_quality_checker.errors import GateBlocked
from data_quality_checker.fingerprints import fingerprint_json
from data_quality_checker.preparation import prepare_batch
from data_quality_checker.processing import (
    MlxG0Backend,
    PredictionResult,
    process_batch,
)
from data_quality_checker.rerouting import reroute_batch
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


def zip_payload(path: Path, payload) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("payload.json", json.dumps(payload, ensure_ascii=False))


def prepared_fixture(tmp_path: Path, *, two_docs: bool = True):
    config = config_for(tmp_path)
    references = [
        {
            "kanun_no": "213",
            "kanun_ad": "Vergi Usul Kanunu",
            "madde": "413",
            "fikra": "",
            "bent": "",
            "source_text": "213 sayılı Vergi Usul Kanununun 413. maddesi",
        }
    ]
    annotations = [
        {"document_id": "one", "current_references": references},
    ]
    pool = [
        {
            "evrakOid": "one",
            "pdfText": "Metin 213 sayılı Vergi Usul Kanununun 413. maddesi ile ilgilidir.",
        }
    ]
    if two_docs:
        annotations.append({"document_id": "two", "current_references": []})
        pool.append({"evrakOid": "two", "pdfText": "Referans içermeyen güvenli metin."})
    annotation_zip, pool_zip = tmp_path / "a.zip", tmp_path / "p.zip"
    zip_payload(annotation_zip, {"annotations": annotations})
    zip_payload(pool_zip, pool)
    key = tmp_path / "key"
    key.write_bytes(b"x" * 32)
    prepare_batch(
        config=config,
        annotation_zip=annotation_zip,
        document_pool_zip=pool_zip,
        batch_id="batch",
        hmac_key_file=key,
    )
    return config


def test_process_atomically_routes_every_document_and_resumes(tmp_path) -> None:
    config = prepared_fixture(tmp_path)
    first = process_batch(
        config=config,
        batch_id="batch",
        generation="G0",
        resume=False,
        fake_backend=True,
    )
    assert first["prediction_count"] == 2
    assert first["router_counts"] == {
        "GREEN": 2,
        "YELLOW": 0,
        "RED": 0,
        "QUARANTINE": 0,
    }
    outputs = sorted(
        (config.sensitive_root / "batches" / "batch" / "predictions" / "G0").glob("*.json")
    )
    assert len(outputs) == 2

    resumed = process_batch(
        config=config,
        batch_id="batch",
        generation="G0",
        resume=True,
        fake_backend=True,
    )
    assert resumed["prediction_count"] == 2
    assert resumed["router_counts"]["GREEN"] == 2


class CountingBackend:
    model_fingerprint = fingerprint_json({"backend": "counting"})

    def __init__(self):
        self.calls = 0

    def predict(self, document):
        self.calls += 1
        refs = document["human_references"]
        return PredictionResult(
            status="success",
            references=refs,
            raw_output=json.dumps(refs),
            operational={"truncated": False, "latency_seconds": 0.0},
        )


def test_resume_recovers_atomically_written_orphan_without_regeneration(
    monkeypatch, tmp_path
) -> None:
    config = prepared_fixture(tmp_path, two_docs=False)
    backend = CountingBackend()
    original = Store.persist_prediction
    failed = {"once": False}

    def fail_after_file(self, **kwargs):
        if not failed["once"]:
            failed["once"] = True
            raise RuntimeError("simulated crash after file promotion")
        return original(self, **kwargs)

    monkeypatch.setattr(Store, "persist_prediction", fail_after_file)
    with pytest.raises(RuntimeError, match="simulated crash"):
        process_batch(
            config=config,
            batch_id="batch",
            generation="G0",
            resume=False,
            backend=backend,
        )
    assert backend.calls == 1

    summary = process_batch(
        config=config,
        batch_id="batch",
        generation="G0",
        resume=True,
        backend=backend,
    )
    assert summary["prediction_count"] == 1
    assert backend.calls == 1


class BoilerplateOnlyBackend:
    model_fingerprint = fingerprint_json({"backend": "boilerplate-only"})

    def predict(self, document):
        references = [
            {
                "kanun_no": "213",
                "kanun_ad": "Vergi Usul Kanunu",
                "madde": "413",
                "fikra": "",
                "bent": "",
                "source_text": "213 sayılı Vergi Usul Kanununun 413. maddesi",
            }
        ]
        return PredictionResult(
            status="success",
            references=references,
            raw_output=json.dumps(references, ensure_ascii=False),
            operational={"truncated": False, "latency_seconds": 0.0},
        )


def legacy_unfiltered_fixture(tmp_path: Path):
    config = config_for(tmp_path)
    annotations = [{"document_id": "one", "current_references": []}]
    pool = [
        {
            "evrakOid": "one",
            "pdfText": "213 sayılı Vergi Usul Kanununun 413. maddesi.",
        }
    ]
    annotation_zip, pool_zip = tmp_path / "legacy-a.zip", tmp_path / "legacy-p.zip"
    zip_payload(annotation_zip, {"annotations": annotations})
    zip_payload(pool_zip, pool)
    key = tmp_path / "legacy-key"
    key.write_bytes(b"z" * 32)
    prepare_batch(
        config=config,
        annotation_zip=annotation_zip,
        document_pool_zip=pool_zip,
        batch_id="legacy",
        hmac_key_file=key,
    )
    process_batch(
        config=config,
        batch_id="legacy",
        generation="G0",
        resume=False,
        backend=BoilerplateOnlyBackend(),
    )
    with Store(config.database_path) as store:
        document = store.list_documents("legacy")[0]
        store.set_router_bucket("legacy", document["internal_doc_id"], "RED")
    return config


def test_reroute_preflights_then_atomically_applies_without_touching_prediction(
    tmp_path,
) -> None:
    config = legacy_unfiltered_fixture(tmp_path)
    with Store(config.database_path) as store:
        prediction = store.list_predictions("legacy", "G0")[0]
        prediction_sha = prediction["response_sha256"]

    preflight = reroute_batch(config=config, batch_id="legacy", apply=False)
    assert preflight["status"] == "preflight_passed"
    assert preflight["unfiltered_router_counts"]["RED"] == 1
    assert preflight["filtered_router_counts"]["GREEN"] == 1
    with Store(config.database_path) as store:
        assert store.status_summary("legacy")["documents"] == {"RED": 1}

    result = reroute_batch(config=config, batch_id="legacy", apply=True)
    assert result["status"] == "applied_and_verified"
    assert result["raw_prediction_files_modified"] is False
    assert result["database_router_counts_after_apply"] == {
        "GREEN": 1,
        "YELLOW": 0,
        "RED": 0,
        "QUARANTINE": 0,
    }
    with Store(config.database_path) as store:
        assert store.status_summary("legacy")["documents"] == {"GREEN": 1}
        assert store.list_predictions("legacy", "G0")[0]["response_sha256"] == prediction_sha

    assert reroute_batch(config=config, batch_id="legacy", apply=True) == result


def test_reroute_refuses_to_change_queue_after_review_started(tmp_path) -> None:
    config = legacy_unfiltered_fixture(tmp_path)
    with Store(config.database_path) as store:
        review = store.list_reviews("legacy")[0]
        store.update_review(
            batch_id="legacy",
            internal_doc_id=review["internal_doc_id"],
            expected_version=review["row_version"],
            status="finalized",
            action="accept_human",
            final_references=[],
            reason=None,
            reviewer="test",
        )
    with pytest.raises(GateBlocked, match="all reviews to remain pending"):
        reroute_batch(config=config, batch_id="legacy", apply=True)


def test_mlx_backend_accepts_an_explicit_isolated_registry_path(monkeypatch, tmp_path) -> None:
    config = config_for(tmp_path)
    snapshot = tmp_path / "snapshot"
    adapter = tmp_path / "adapter"
    snapshot.mkdir()
    adapter.mkdir()
    (adapter / "adapters.safetensors").write_bytes(b"adapter")
    registry = tmp_path / "isolated" / "G0.json"
    registry.parent.mkdir()

    from data_quality_checker.constants import MODEL_ID, MODEL_REVISION
    from data_quality_checker.fingerprints import sha256_file

    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "model_snapshot_path": str(snapshot),
                "adapter_path": str(adapter),
                "adapter_sha256": sha256_file(adapter / "adapters.safetensors"),
                "max_sequence_length": 12288,
                "max_generation_tokens": 4096,
            }
        ),
        encoding="utf-8",
    )

    class FakeMlxLm:
        @staticmethod
        def load(model_path, adapter_path):
            assert model_path == str(snapshot)
            assert adapter_path == str(adapter)
            return object(), object()

    monkeypatch.setitem(__import__("sys").modules, "mlx_lm", FakeMlxLm)
    backend = MlxG0Backend(config, registry_path=registry)
    assert backend.max_input_tokens == 12288
    assert backend.max_generation_tokens == 4096


def test_mlx_backend_preserves_generation_limit_as_primary_parse_error(monkeypatch):
    captured = {}

    def fake_stream_generate(model, tokenizer, **kwargs):
        captured.update(kwargs)
        yield SimpleNamespace(
            text="not-json",
            generation_tokens=8,
            finish_reason="length",
            prompt_tps=10.0,
            generation_tps=5.0,
            peak_memory=1024,
        )

    class Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            return "prompt"

        def encode(self, _prompt):
            return [1, 2, 3]

    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        SimpleNamespace(stream_generate=fake_stream_generate),
    )
    backend = object.__new__(MlxG0Backend)
    backend.model = object()
    backend.tokenizer = Tokenizer()
    backend.max_input_tokens = 100
    backend.max_generation_tokens = 8

    result = backend.predict({"text": "body"})

    assert result.status == "error"
    assert result.operational["truncated"] is True
    assert result.error.startswith("model output reached generation limit;")
    assert captured == {"prompt": "prompt", "max_tokens": 8}
