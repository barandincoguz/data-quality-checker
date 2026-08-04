from __future__ import annotations

import json
from pathlib import Path

from data_quality_checker.fingerprints import sha256_text
from data_quality_checker.g0 import SYSTEM_PROMPT
from data_quality_checker.q36_p1 import (
    EXPECTED_GREEN_DISTRIBUTION,
    LOCKED_PROMPT_SHA256,
    Q36TrainingContract,
    choose_inference_contract,
    classify_green_pool,
    inspect_green_audit,
    measure_inference_contract,
    promotion_decision,
    reconstruct_vuk_213_article_413,
    refit_optimizer_updates,
    write_prediction_views,
    write_production_gold_view,
)


def _reference(**updates):
    row = {
        "kanun_no": "213",
        "kanun_ad": "Vergi Usul Kanunu",
        "madde": "3",
        "fikra": "",
        "bent": "",
        "source_text": "213 sayılı Vergi Usul Kanununun 3 üncü maddesi",
    }
    row.update(updates)
    return row


def test_q36_prompt_and_training_recipe_are_byte_locked() -> None:
    assert sha256_text(SYSTEM_PROMPT) == LOCKED_PROMPT_SHA256
    contract = Q36TrainingContract()
    assert contract.optimizer_updates(3399) == 850
    config = contract.stateful_config(3399)
    assert config.total_updates == 850
    assert config.warmup_updates == 42
    assert config.peak_learning_rate == 2.5e-5
    assert config.end_learning_rate == 1e-5
    assert config.rank == 8
    assert config.num_layers == 16
    assert config.max_sequence_length == 1536
    assert config.checkpoint_every_updates == 25


def test_green_pool_classification_reproduces_locked_six_way_distribution(
    tmp_path: Path,
) -> None:
    historical = tmp_path / "artifacts/qwen3_14b_neon_external_eval_2026_07_16"
    prepared = tmp_path / "data/sensitive/neon_external_eval/prepared"
    historical.mkdir(parents=True)
    for relative in (
        "validation/docs.json",
        "validation/gold.json",
        "sealed/docs.json",
        "sealed/gold.json",
    ):
        path = prepared / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]\n", encoding="utf-8")

    documents = []
    green = []
    overlap = []
    quarantine = []
    index = 0
    split_counts = {"train": 156, "validation": 34, "sealed_test": 119}
    for split, count in split_counts.items():
        for _ in range(count):
            index += 1
            text_sha = f"{index:064x}"
            documents.append(
                {
                    "doc_id": 100000 + index,
                    "source": "external",
                    "split": split,
                    "text_sha256": text_sha,
                }
            )
            green.append(
                {
                    "internal_doc_id": f"d{index:06d}",
                    "public_doc_id": f"public-{index}",
                    "raw_document_id": f"raw-{index}",
                    "text_sha256": text_sha,
                    "metadata_json": '{"annotation_completed":true}',
                }
            )
    for category, count in (
        ("overlap", 11),
        ("quarantine", 9),
        ("incomplete", 13),
    ):
        for _ in range(count):
            index += 1
            raw_id = f"raw-{index}"
            document_hash = sha256_text(f"neon-document:{raw_id}")
            if category == "overlap":
                overlap.append(
                    {"document_id_hash": document_hash, "canonical_doc_ids": [1]}
                )
            elif category == "quarantine":
                quarantine.append(
                    {
                        "document_id_hash": document_hash,
                        "reason": "reference_validation_failed",
                    }
                )
            green.append(
                {
                    "internal_doc_id": f"d{index:06d}",
                    "public_doc_id": f"public-{index}",
                    "raw_document_id": raw_id,
                    "text_sha256": f"{index + 10000:064x}",
                    "metadata_json": json.dumps(
                        {"annotation_completed": category != "incomplete"}
                    ),
                }
            )
    (historical / "split_manifest.json").write_text(
        json.dumps({"documents": documents}), encoding="utf-8"
    )
    (historical / "canonical_overlap.json").write_text(
        json.dumps(overlap), encoding="utf-8"
    )
    (historical / "quarantine.json").write_text(
        json.dumps(quarantine), encoding="utf-8"
    )

    result = classify_green_pool(green, repo_root=tmp_path)
    assert result["document_count"] == 342
    assert result["counts"] == EXPECTED_GREEN_DISTRIBUTION


def test_green_audit_blocks_pending_and_escalates_membership_changes() -> None:
    ids = [f"d{index:06d}" for index in range(35)]
    human = [_reference()]
    documents = [
        {
            "internal_doc_id": doc_id,
            "human_references_json": json.dumps(human),
        }
        for doc_id in ids
    ]
    reviews = [
        {
            "internal_doc_id": doc_id,
            "status": "pending",
            "final_references_json": None,
        }
        for doc_id in ids
    ]
    audit = {
        "green_count": 342,
        "sample_internal_doc_ids": ids,
    }
    pending = inspect_green_audit(
        audit=audit,
        green_documents=documents,
        reviews=reviews,
        escalation_exists=False,
    )
    assert pending["status"] == "pending"
    assert pending["finalized_count"] == 0

    for review in reviews:
        review["status"] = "finalized"
        review["final_references_json"] = json.dumps(human)
    passed = inspect_green_audit(
        audit=audit,
        green_documents=documents,
        reviews=reviews,
        escalation_exists=False,
    )
    assert passed["status"] == "passed"

    reviews[0]["final_references_json"] = json.dumps(
        [_reference(madde="4", source_text="4 üncü madde")]
    )
    escalated = inspect_green_audit(
        audit=audit,
        green_documents=documents,
        reviews=reviews,
        escalation_exists=False,
    )
    assert escalated["status"] == "escalation_required"
    assert escalated["legal_membership_change_count"] == 1


def test_213_413_reconstruction_requires_exact_evidence_and_is_idempotent() -> None:
    footer = (
        "Bu Özelge 213 sayılı Vergi Usul Kanununun 413.maddesine "
        "dayanılarak verilmiştir."
    )
    references, audit = reconstruct_vuk_213_article_413(
        [_reference()], document_text=f"Başlangıç. {footer} Son."
    )
    assert audit["added"] is True
    assert audit["provenance"] == "derived_policy_reconstruction"
    assert references[-1]["source_text"] == footer
    assert references[-1]["kanun_no"] == "213"
    assert references[-1]["madde"] == "413"

    repeated, repeated_audit = reconstruct_vuk_213_article_413(
        references, document_text=f"Başlangıç. {footer} Son."
    )
    assert repeated == references
    assert repeated_audit["reason"] == "already_present"

    untouched, missing = reconstruct_vuk_213_article_413(
        [_reference()], document_text="Footer içermeyen özelge."
    )
    assert untouched == [_reference()]
    assert missing == {
        "policy": "derived_policy_reconstruction_v1",
        "added": False,
        "reason": "no_exact_evidence",
        "evidence_match_count": 0,
    }


def test_prediction_views_preserve_raw_bytes_and_filter_exact_identity(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "predictions.json"
    rows = [
        {
            "doc_id": 1,
            "status": "success",
            "references": [
                _reference(madde="413 üncü maddesi"),
                _reference(madde="413/A"),
                _reference(kanun_no="3065", madde="413"),
            ],
        }
    ]
    raw.write_text(
        json.dumps(rows, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    before = raw.read_bytes()
    first = write_prediction_views(raw_predictions=raw, output_dir=tmp_path / "views")
    second = write_prediction_views(raw_predictions=raw, output_dir=tmp_path / "views")
    assert first == second
    assert raw.read_bytes() == before
    raw_view = Path(first["raw_full_extraction"]["path"])
    assert raw_view.read_bytes() == before
    production = json.loads(
        Path(first["production_filtered_213_413"]["path"]).read_text(encoding="utf-8")
    )
    assert len(production[0]["references"]) == 2
    assert first["production_filtered_213_413"]["removed_reference_count"] == 1


def test_production_gold_view_applies_the_same_policy_without_source_mutation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "gold.json"
    source.write_text(
        json.dumps(
            [
                {
                    "doc_id": 1,
                    "references": [
                        _reference(madde="413"),
                        _reference(madde="3"),
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    before = source.read_bytes()
    output = tmp_path / "production_gold.json"
    first = write_production_gold_view(source=source, output=output)
    second = write_production_gold_view(source=source, output=output)
    assert first == second
    assert source.read_bytes() == before
    assert first["removed_reference_count"] == 1
    assert len(json.loads(output.read_text(encoding="utf-8"))[0]["references"]) == 1


class _Tokenizer:
    def apply_chat_template(self, messages, **_):
        return list(range(sum(len(row["content"].split()) for row in messages)))

    def encode(self, text):
        return list(range(len(text.split())))


def test_common_inference_budget_only_escalates_when_measurements_require_it() -> None:
    assert (
        choose_inference_contract(
            input_token_counts=[100, 16384], serialized_gold_token_counts=[100, 3276]
        )["output_tokens"]
        == 4096
    )
    escalated = choose_inference_contract(
        input_token_counts=[16385], serialized_gold_token_counts=[3277]
    )
    assert escalated["input_tokens"] == 32768
    assert escalated["output_tokens"] == 8192

    measured = measure_inference_contract(
        tokenizer=_Tokenizer(),
        universes={
            "canonical_test_50": [{"text": "bir iki üç", "references": [_reference()]}]
        },
    )
    assert measured["selected"]["input_tokens"] == 16384
    assert measured["selected"]["output_tokens"] == 4096


def test_refit_scaling_and_promotion_gates_are_explicit() -> None:
    assert refit_optimizer_updates(800) == round(800 * 803 / 550)
    assert refit_optimizer_updates(100, refit_training_view_rows=5000) == 1250
    decision = promotion_decision(
        base={"core_f1": 0.80, "recall": 0.80, "docwise": 0.30},
        canonical_only={"core_f1": 0.82, "recall": 0.81, "docwise": 0.32},
        augmented={"core_f1": 0.83, "recall": 0.81, "docwise": 0.34},
        g0={"core_f1": 0.79, "recall": 0.70, "docwise": 0.31},
        paired_ci_lower=0.0,
        coverage_count=400,
        parse_count=400,
        truncation_count=0,
        runaway_count=0,
    )
    assert decision["promote"] is True

    failed = promotion_decision(
        base={"core_f1": 0.80, "recall": 0.80, "docwise": 0.30},
        canonical_only={"core_f1": 0.84, "recall": 0.84, "docwise": 0.34},
        augmented={"core_f1": 0.83, "recall": 0.83, "docwise": 0.34},
        g0={"core_f1": 0.79, "recall": 0.70, "docwise": 0.31},
        paired_ci_lower=-0.001,
        coverage_count=400,
        parse_count=400,
        truncation_count=0,
        runaway_count=0,
    )
    assert failed["promote"] is False
    assert "paired_ci_lower_nonnegative" in failed["failed_gates"]
    assert "f1_drop_vs_canonical_at_most_0_005" in failed["failed_gates"]
