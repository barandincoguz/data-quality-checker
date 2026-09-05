"""Tests for `audit_reference_set` and `triage_disagreement`.

The single most important case here is the pair of tests pinning
`article_number_absent_from_span`: on the real ground truth this check fires
172 times and every one inspected was legitimate (anaphora, a bent reference,
an amending law, or a number the source PDF glued to the previous word). A
naive implementation that treats any of these as a *mismatch* or an *error*
would be a regression this suite must catch, so
`test_article_number_absent_from_span_classifies_anaphoric_span_without_error`
and `test_article_number_absent_from_span_classifies_glued_number_without_error`
both assert the finding is sub-classified for a human, and both assert the
result carries no key that could read as an assertion of wrongness (no
`error`, `is_error`, `valid`, `mismatch` key anywhere in the finding).
"""

from __future__ import annotations

from data_quality_checker.reference_audit import audit_reference_set, triage_disagreement

_NO_JUDGMENT_KEYS = {"error", "is_error", "valid", "is_valid", "mismatch", "wrong", "incorrect"}


def row(**updates: str) -> dict[str, str]:
    payload = {
        "kanun_no": "213",
        "kanun_ad": "Vergi Usul Kanunu",
        "madde": "94",
        "fikra": "",
        "bent": "",
        "source_text": "Vergi Usul Kanununun 94 üncü maddesi",
    }
    payload.update(updates)
    return payload


def assert_no_judgment_keys(finding: dict) -> None:
    assert not (_NO_JUDGMENT_KEYS & finding.keys()), finding
    assert not (_NO_JUDGMENT_KEYS & finding["reference"].keys()), finding


# ---------------------------------------------------------------------------
# audit_reference_set: ungrounded_span
# ---------------------------------------------------------------------------


def test_ungrounded_span_fires_on_empty_source_text() -> None:
    result = audit_reference_set([row(source_text="")], document_text="irrelevant document body")
    assert result["ungrounded_span_count"] == 1
    assert result["findings"]["ungrounded_span"][0]["reference"]["source_text"] == ""


def test_ungrounded_span_fires_when_span_absent_from_document() -> None:
    result = audit_reference_set(
        [row(source_text="Vergi Usul Kanununun 94 üncü maddesi")],
        document_text="Bu belgede baska bir sey yaziyor.",
    )
    assert result["ungrounded_span_count"] == 1


def test_ungrounded_span_collapses_whitespace_in_both_span_and_document() -> None:
    # The raw reference's own span carries a newline + double space; the
    # document has it as ordinary single spaces. Neither should defeat
    # grounding once both sides collapse whitespace runs.
    result = audit_reference_set(
        [row(source_text="Vergi Usul Kanununun\n94  üncü maddesi")],
        document_text="... Vergi Usul Kanununun 94 üncü maddesi ...",
    )
    assert result["ungrounded_span_count"] == 0


# ---------------------------------------------------------------------------
# audit_reference_set: law_number_name_conflict
# ---------------------------------------------------------------------------


def test_law_number_name_conflict_fires_on_a_genuine_mismatch() -> None:
    result = audit_reference_set(
        [row(kanun_no="213", kanun_ad="Gelir Vergisi Kanunu")],
        document_text="Gelir Vergisi Kanununun 94 üncü maddesi",
    )
    assert result["law_number_name_conflict_count"] == 1
    finding = result["findings"]["law_number_name_conflict"][0]
    assert finding["kanun_no"] == "213"
    assert finding["declared_law_name"] == "Gelir Vergisi Kanunu"
    assert finding["canonical_law_name"] == "Vergi Usul Kanunu"


def test_law_number_name_conflict_does_not_fire_for_an_alias_of_the_canonical_name() -> None:
    result = audit_reference_set(
        [row(kanun_no="213", kanun_ad="VUK")], document_text=row()["source_text"]
    )
    assert result["law_number_name_conflict_count"] == 0


def test_law_number_name_conflict_does_not_fire_when_name_is_blank() -> None:
    result = audit_reference_set(
        [row(kanun_no="213", kanun_ad="")], document_text=row()["source_text"]
    )
    assert result["law_number_name_conflict_count"] == 0


def test_law_number_name_conflict_does_not_fire_for_an_uncanonical_law_number() -> None:
    result = audit_reference_set(
        [row(kanun_no="9999", kanun_ad="Herhangi Bir Kanun", source_text="Herhangi Bir Kanun 5")],
        document_text="Herhangi Bir Kanun 5",
    )
    assert result["law_number_name_conflict_count"] == 0


# ---------------------------------------------------------------------------
# audit_reference_set: duplicate_row
# ---------------------------------------------------------------------------


def test_duplicate_row_fires_for_two_rows_sharing_identity_and_span() -> None:
    span = "Vergi Usul Kanununun 94 üncü maddesi"
    result = audit_reference_set([row(source_text=span), row(source_text=span)], document_text=span)
    assert result["duplicate_row_count"] == 1
    assert result["findings"]["duplicate_row"][0]["duplicate_count"] == 2


def test_duplicate_row_does_not_fire_for_two_mentions_with_different_spans() -> None:
    # Same provision named twice in one document, in two different places --
    # this is two mentions, not a defect, and must not be reported.
    document_text = (
        "Vergi Usul Kanununun 94 üncü maddesi uyarınca ... ve yine aynı "
        "Kanunun 94 üncü maddesine göre ..."
    )
    result = audit_reference_set(
        [
            row(source_text="Vergi Usul Kanununun 94 üncü maddesi"),
            row(source_text="aynı Kanunun 94 üncü maddesine"),
        ],
        document_text=document_text,
    )
    assert result["duplicate_row_count"] == 0


# ---------------------------------------------------------------------------
# audit_reference_set: article_number_absent_from_span (the hard requirement)
# ---------------------------------------------------------------------------


def test_article_number_absent_from_span_does_not_fire_when_the_number_is_properly_present() -> (
    None
):
    result = audit_reference_set([row()], document_text=row()["source_text"])
    assert result["article_number_absent_from_span_count"] == 0


def test_article_number_absent_from_span_classifies_anaphoric_span_without_error() -> None:
    anaphoric_span = "ikinci fıkrasında ise"
    document_text = (
        "Vergi Usul Kanununun 94 üncü maddesinin ikinci fıkrasında ise "
        "asagidaki hukum yer almaktadir."
    )
    result = audit_reference_set(
        [row(madde="94", source_text=anaphoric_span)], document_text=document_text
    )
    assert result["article_number_absent_from_span_count"] == 1
    finding = result["findings"]["article_number_absent_from_span"][0]
    assert finding["classification"] == "anaphoric"
    assert finding["numbers_in_span"] == []
    assert_no_judgment_keys(finding)
    # It must never also read as ungrounded -- the span really is in the text.
    assert result["ungrounded_span_count"] == 0


def test_article_number_absent_from_span_classifies_glued_number_without_error() -> None:
    glued_span = "Kanununun267 inci maddesi"
    document_text = f"... {glued_span} ..."
    result = audit_reference_set(
        [row(madde="267", source_text=glued_span)], document_text=document_text
    )
    assert result["article_number_absent_from_span_count"] == 1
    finding = result["findings"]["article_number_absent_from_span"][0]
    assert finding["classification"] == "other_number_present"
    assert finding["numbers_in_span"] == ["267"]
    assert_no_judgment_keys(finding)
    assert result["ungrounded_span_count"] == 0


def test_article_number_absent_from_span_classifies_a_bent_reference_as_other_number_present() -> (
    None
):
    bent_span = "söz konusu fıkranın (2/b) bendinde"
    document_text = f"... {bent_span} ..."
    result = audit_reference_set(
        [row(madde="94", source_text=bent_span)], document_text=document_text
    )
    finding = result["findings"]["article_number_absent_from_span"][0]
    assert finding["classification"] == "other_number_present"
    assert finding["numbers_in_span"] == ["2"]
    assert_no_judgment_keys(finding)


def test_article_number_absent_from_span_skips_a_non_plain_article_value() -> None:
    # "mükerrer 5" is not a plain number; the check only applies when the
    # whole madde field is digits.
    result = audit_reference_set(
        [row(madde="mükerrer 5", source_text="herhangi bir metin")],
        document_text="herhangi bir metin",
    )
    assert result["article_number_absent_from_span_count"] == 0


def test_audit_reference_set_reference_count_matches_compacted_input() -> None:
    result = audit_reference_set(
        [row(), row(kanun_no="193", kanun_ad="Gelir Vergisi Kanunu")], document_text="x"
    )
    assert result["reference_count"] == 2
    assert result["raw_reference_count"] == 2


# ---------------------------------------------------------------------------
# triage_disagreement
# ---------------------------------------------------------------------------


def test_triage_ungrounded_candidate() -> None:
    result = triage_disagreement(expert=[], candidate=[row(source_text="")], document_text="x")
    assert result["ungrounded_count"] == 1
    assert result["candidate_addition_count"] == 0
    assert result["label_inconsistent_with_span_count"] == 0


def test_triage_label_inconsistent_wrong_law_number() -> None:
    # Real example: span quotes "193 sayılı Gelir Vergisi Kanununun 37 nci
    # maddesinde" but is labelled law 197.
    document_text = "193 sayılı Gelir Vergisi Kanununun 37 nci maddesinde ..."
    result = triage_disagreement(
        expert=[],
        candidate=[
            row(
                kanun_no="197",
                kanun_ad="",
                madde="37",
                source_text="193 sayılı Gelir Vergisi Kanununun 37 nci maddesinde",
            )
        ],
        document_text=document_text,
    )
    assert result["label_inconsistent_with_span_count"] == 1
    finding = result["findings"]["label_inconsistent_with_span"][0]
    assert finding["law_number_absent"] is True
    assert finding["refers_back_to_a_law"] is False


def test_triage_label_inconsistent_wrong_article_number() -> None:
    # Real example: span reads "- 9/1 inci maddesinde," but is labelled
    # article 19.
    document_text = "213 sayılı Vergi Usul Kanununun - 9/1 inci maddesinde, ..."
    result = triage_disagreement(
        expert=[],
        candidate=[
            row(
                kanun_no="213",
                kanun_ad="Vergi Usul Kanunu",
                madde="19",
                source_text="213 sayılı Vergi Usul Kanununun - 9/1 inci maddesinde,",
            )
        ],
        document_text=document_text,
    )
    assert result["label_inconsistent_with_span_count"] == 1
    finding = result["findings"]["label_inconsistent_with_span"][0]
    assert finding["article_number_absent"] is True


def test_triage_label_inconsistent_transposed_law_number() -> None:
    # Real example: "4734 sayılı Kamu İhale Kanununun ... 4 üncü maddesi"
    # labelled law 4743.
    document_text = "4734 sayılı Kamu İhale Kanununun ilgili hükümleri uyarınca 4 üncü maddesi ..."
    result = triage_disagreement(
        expert=[],
        candidate=[
            row(
                kanun_no="4743",
                kanun_ad="Kamu İhale Kanunu",
                madde="4",
                source_text=(
                    "4734 sayılı Kamu İhale Kanununun ilgili hükümleri uyarınca 4 üncü maddesi"
                ),
            )
        ],
        document_text=document_text,
    )
    assert result["label_inconsistent_with_span_count"] == 1


def test_triage_candidate_addition_is_never_a_truth_error() -> None:
    document_text = "3065 sayılı Katma Değer Vergisi Kanununun 2 nci maddesi"
    result = triage_disagreement(
        expert=[],
        candidate=[
            row(
                kanun_no="3065",
                kanun_ad="Katma Değer Vergisi Kanunu",
                madde="2",
                source_text="3065 sayılı Katma Değer Vergisi Kanununun 2 nci maddesi",
            )
        ],
        document_text=document_text,
    )
    assert result["candidate_addition_count"] == 1
    assert result["label_inconsistent_with_span_count"] == 0
    assert result["ungrounded_count"] == 0
    for finding in result["findings"]["candidate_addition"]:
        assert_no_judgment_keys(finding)


def test_triage_anaphoric_law_reference_is_not_penalized_as_inconsistent() -> None:
    # The law is named earlier in the document and referred back to here;
    # the stated law number need not be restated in this particular span.
    document_text = "... Mezkûr Kanunun 5 inci maddesinde ..."
    result = triage_disagreement(
        expert=[],
        candidate=[
            row(
                kanun_no="213",
                kanun_ad="Vergi Usul Kanunu",
                madde="5",
                source_text="Mezkûr Kanunun 5 inci maddesinde",
            )
        ],
        document_text=document_text,
    )
    assert result["label_inconsistent_with_span_count"] == 0
    assert result["candidate_addition_count"] == 1


def test_triage_mirror_direction_counts_expert_only_provisions() -> None:
    expert_only_row = row(kanun_no="488", kanun_ad="Damga Vergisi Kanunu", madde="14")
    result = triage_disagreement(expert=[expert_only_row], candidate=[], document_text="irrelevant")
    assert result["expert_only_count"] == 1
    assert result["candidate_only_count"] == 0
    assert len(result["expert_only"]) == 1


def test_triage_matched_provision_is_not_double_reported() -> None:
    shared = row()
    result = triage_disagreement(
        expert=[shared], candidate=[shared], document_text=shared["source_text"]
    )
    assert result["candidate_matched_count"] == 1
    assert result["candidate_only_count"] == 0
    assert result["expert_only_count"] == 0
