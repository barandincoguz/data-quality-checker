from __future__ import annotations

from data_quality_checker.normalization import compact_references, normalize_reference
from data_quality_checker.router import route_document


def ref(**updates):
    payload = {
        "kanun_no": "213",
        "kanun_ad": "Vergi Usul Kanunu",
        "madde": "mükerrer 20",
        "fikra": "1",
        "bent": "a",
        "source_text": "213 sayılı Vergi Usul Kanununun mükerrer 20 inci maddesi",
    }
    payload.update(updates)
    return payload


def test_alias_article_and_extension_normalization() -> None:
    normalized = normalize_reference(
        ref(
            kanun_no="",
            kanun_ad="VUK",
            madde="Mükerrer 20 inci maddesi",
            fikra="(Birinci)",
            bent="(A)",
        )
    )
    assert normalized["kanun_no"] == "213"
    assert normalized["kanun_ad"] == "Vergi Usul Kanunu"
    assert normalized["madde"] == "mükerrer 20"
    assert normalized["fikra"] == "1"
    assert normalized["bent"] == "a"


def test_tuple_dedup_and_generic_law_compaction() -> None:
    rows = compact_references(
        [
            ref(madde="", fikra="", bent="", source_text="generic"),
            ref(),
            ref(source_text="same tuple elsewhere"),
        ]
    )
    assert len(rows) == 1
    assert rows[0]["madde"] == "mükerrer 20"


def test_green_allows_aliases_and_evidence_format_or_length_only() -> None:
    decision = route_document(
        human_references=[ref(kanun_no="", kanun_ad="VUK")],
        model_references=[
            ref(source_text="Vergi Usul Kanununun mükerrer 20 inci maddesi")
        ],
    )
    assert decision.bucket == "GREEN"
    assert decision.similarity == 1.0


def test_yellow_for_extension_or_real_evidence_mismatch() -> None:
    extension = route_document(
        human_references=[ref(fikra="1")],
        model_references=[ref(fikra="2")],
    )
    assert extension.bucket == "YELLOW"
    assert extension.reasons == ("extension_mismatch",)

    evidence = route_document(
        human_references=[ref(source_text="first unrelated evidence")],
        model_references=[ref(source_text="second different passage")],
    )
    assert evidence.bucket == "YELLOW"
    assert evidence.reasons == ("evidence_mismatch",)


def test_red_for_missing_or_extra_core_and_conflicting_identity() -> None:
    missing = route_document(human_references=[ref()], model_references=[])
    assert missing.bucket == "RED"
    assert "missing_core_reference" in missing.reasons

    conflict = route_document(
        human_references=[
            ref(kanun_no="999", kanun_ad="Birinci Kanun"),
            ref(kanun_no="999", kanun_ad="İkinci Kanun", madde="2"),
        ],
        model_references=[],
    )
    assert conflict.bucket == "RED"
    assert conflict.reasons == ("conflicting_law_identity",)


def test_quarantine_precedes_similarity() -> None:
    decision = route_document(
        human_references=[],
        model_references=[],
        model_status="error",
        model_truncated=True,
    )
    assert decision.similarity == 1.0
    assert decision.bucket == "QUARANTINE"
