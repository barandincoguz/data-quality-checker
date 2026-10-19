from __future__ import annotations

from data_quality_checker.normalization import compact_references, normalize_reference
from data_quality_checker.reference_policy import NO_REFERENCE_FILTER_POLICY_ID
from data_quality_checker.router import route_document


def ref(**updates):
    payload = {
        "kanun_no": "213",
        "kanun_ad": "Vergi Usul Kanunu",
        "madde": "mükerrer 20",
        "source_text": "213 sayılı Vergi Usul Kanununun mükerrer 20 inci maddesi",
    }
    payload.update(updates)
    return payload


def test_alias_and_article_normalization() -> None:
    normalized = normalize_reference(
        ref(kanun_no="", kanun_ad="VUK", madde="Mükerrer 20 inci maddesi")
    )
    assert normalized["kanun_no"] == "213"
    assert normalized["kanun_ad"] == "Vergi Usul Kanunu"
    assert normalized["madde"] == "mükerrer 20"
    assert "fikra" not in normalized and "bent" not in normalized


def test_tuple_dedup_and_generic_law_compaction() -> None:
    rows = compact_references(
        [
            ref(madde="", source_text="generic"),
            ref(),
            ref(source_text="same tuple elsewhere"),
        ]
    )
    assert len(rows) == 1
    assert rows[0]["madde"] == "mükerrer 20"


def test_green_allows_aliases_and_ignores_source_text() -> None:
    decision = route_document(
        human_references=[ref(kanun_no="", kanun_ad="VUK")],
        model_references=[ref(source_text="Vergi Usul Kanununun mükerrer 20 inci maddesi")],
    )
    assert decision.bucket == "GREEN"
    assert decision.similarity == 1.0


def test_no_flag_is_raised_for_a_paragraph_difference() -> None:
    """1 Oct 2026: there is no YELLOW. Lists stored before that day still carry
    paragraph fields; the same law and article is GREEN whatever they say."""
    decision = route_document(
        human_references=[{**ref(), "fikra": "1", "bent": "a"}],
        model_references=[{**ref(), "fikra": "2", "bent": ""}, ref(madde="Mükerrer 20/b")],
    )
    assert decision.bucket == "GREEN"
    assert decision.reasons == ("normalized_law_article_set_equal",)


def test_green_when_law_and_article_match_despite_different_source_text() -> None:
    decision = route_document(
        human_references=[ref(source_text="first unrelated evidence")],
        model_references=[ref(source_text="second, completely different passage")],
    )
    assert decision.bucket == "GREEN"
    assert decision.reasons == ("normalized_law_article_set_equal",)


def test_one_character_source_text_still_routes_green() -> None:
    decision = route_document(
        human_references=[ref(source_text="a")],
        model_references=[ref()],
    )
    assert decision.bucket == "GREEN"
    assert decision.reasons == ("normalized_law_article_set_equal",)


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


def test_vuk_213_article_413_is_ignored_symmetrically_by_default() -> None:
    boilerplate = ref(madde="413")
    human_only = route_document(
        human_references=[boilerplate],
        model_references=[],
    )
    model_only = route_document(
        human_references=[],
        model_references=[boilerplate],
    )
    assert human_only.bucket == "GREEN"
    assert model_only.bucket == "GREEN"


def test_vuk_213_article_413_filter_can_be_disabled_for_impact_audit() -> None:
    decision = route_document(
        human_references=[],
        model_references=[ref(madde="413")],
        reference_policy_id=NO_REFERENCE_FILTER_POLICY_ID,
    )
    assert decision.bucket == "RED"
    assert decision.reasons == ("extra_or_different_core_reference",)


def test_vuk_filter_is_exact_and_keeps_other_articles_and_laws() -> None:
    decision = route_document(
        human_references=[],
        model_references=[
            ref(madde="413/A"),
            ref(kanun_no="3065", kanun_ad="KDV Kanunu", madde="413"),
        ],
    )
    assert decision.bucket == "RED"
