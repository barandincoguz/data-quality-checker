"""Tests for `attribute_errors`, `rater_agreement` and expert workload.

The single most important case here is `test_joint_miss_count_counts_a_reference_all_three_raters_miss`:
`joint_miss_count` must be counted from the expert's full reference set, never
from where the raters merely disagree with each other. A metric derived only
from rater disagreement would report 0 on that fixture (all three raters
agree perfectly -- they all found the same reference and all missed the
same one), so it is the fixture that discriminates a correct implementation
from a disagreement-based one. This was confirmed by hand: temporarily
swapping `rater_agreement`'s `joint_miss_count` computation for a
disagreement-based one (counting references where the raters' sets differ
from each other) and rerunning this test made it fail with
`joint_miss_count == 0`, exactly as expected; the swap was then reverted.
"""

from __future__ import annotations

from typing import Any

import pytest

from data_quality_checker.errors import ContractError
from data_quality_checker.loop_metrics import (
    attribute_errors,
    documents_with_finalized_truth,
    edit_distance_from_best_candidate,
    expert_workload,
    rater_agreement,
)


def ref(**updates: str) -> dict[str, str]:
    payload = {
        "kanun_no": "213",
        "kanun_ad": "Vergi Usul Kanunu",
        "madde": "5",
        "fikra": "",
        "bent": "",
        "source_text": "madde 5",
    }
    payload.update(updates)
    return payload


REF_A = ref(kanun_no="213", kanun_ad="Vergi Usul Kanunu", madde="5", source_text="A")
REF_B = ref(kanun_no="193", kanun_ad="Gelir Vergisi Kanunu", madde="10", source_text="B")
REF_C = ref(kanun_no="5520", kanun_ad="Kurumlar Vergisi Kanunu", madde="1", source_text="C")
REF_D = ref(kanun_no="3065", kanun_ad="Katma Değer Vergisi Kanunu", madde="2", source_text="D")
VUK_413 = ref(kanun_no="213", kanun_ad="Vergi Usul Kanunu", madde="413", source_text="413")


# ---------------------------------------------------------------------------
# attribute_errors
# ---------------------------------------------------------------------------


def test_attribute_errors_scores_each_rater_against_the_expert() -> None:
    result = attribute_errors(
        expert=[REF_A, REF_B],
        annotator=[REF_A],
        model=[REF_A, REF_B],
        judge=[REF_A],
    )
    assert result["annotator"]["available"] is True
    assert len(result["annotator"]["correct"]) == 1
    assert len(result["annotator"]["missed"]) == 1
    assert result["annotator"]["spurious"] == []

    assert result["model"]["correct"] and len(result["model"]["correct"]) == 2
    assert result["model"]["missed"] == []
    assert result["model"]["spurious"] == []

    assert len(result["judge"]["correct"]) == 1
    assert len(result["judge"]["missed"]) == 1


def test_attribute_errors_reports_spurious_references_not_in_expert() -> None:
    result = attribute_errors(expert=[REF_A], annotator=[REF_A, REF_B], model=[], judge=[REF_A])
    assert result["annotator"]["missed"] == []
    assert len(result["annotator"]["spurious"]) == 1
    assert result["model"]["correct"] == []
    assert len(result["model"]["missed"]) == 1
    assert result["model"]["spurious"] == []


def test_attribute_errors_marks_a_missing_rater_unavailable_not_empty() -> None:
    """`None` (no usable output) must not be silently treated as `[]` (found nothing)."""
    result = attribute_errors(expert=[REF_A], annotator=None, model=[REF_A], judge=[REF_A])
    assert result["annotator"] == {
        "available": False,
        "correct": None,
        "missed": None,
        "spurious": None,
    }
    assert result["model"]["available"] is True


def test_attribute_errors_requires_the_expert() -> None:
    with pytest.raises(ContractError, match="expert"):
        attribute_errors(expert=None, annotator=[], model=[], judge=[])


def test_attribute_errors_applies_the_reference_policy_uniformly() -> None:
    """VUK 213/413 boilerplate must be filtered from every rater alike, so a rater
    that merely repeats it is not falsely charged with a spurious reference."""
    result = attribute_errors(
        expert=[REF_A],
        annotator=[REF_A, VUK_413],
        model=[REF_A],
        judge=[REF_A],
    )
    assert result["annotator"]["spurious"] == []
    assert result["annotator"]["missed"] == []
    assert len(result["annotator"]["correct"]) == 1


# ---------------------------------------------------------------------------
# rater_agreement: the joint-miss ceiling (the core requirement)
# ---------------------------------------------------------------------------


def test_joint_miss_count_counts_a_reference_all_three_raters_miss() -> None:
    """All three raters agree perfectly here (each finds REF_A, each misses REF_B).

    A metric derived only from where raters *disagree* would see zero
    disagreement in this fixture and report `joint_miss_count == 0`; the
    honest ceiling still has to be 1, because the expert kept a reference
    that not one rater ever produced.
    """
    records = [
        {
            "expert": [REF_A, REF_B],
            "annotator": [REF_A],
            "model": [REF_A],
            "judge": [REF_A],
        }
    ]
    metrics = rater_agreement(records)
    assert metrics["joint_miss_count"] == 1
    assert metrics["total_expert_reference_count"] == 2
    assert metrics["joint_miss_rate"] == pytest.approx(0.5)


def test_joint_miss_count_is_zero_when_at_least_one_rater_finds_it() -> None:
    records = [
        {
            "expert": [REF_A, REF_B],
            "annotator": [REF_A],
            "model": [REF_A, REF_B],
            "judge": [REF_A],
        }
    ]
    metrics = rater_agreement(records)
    assert metrics["joint_miss_count"] == 0


def test_joint_miss_count_excludes_documents_missing_a_rater() -> None:
    """A document where a rater simply has no data cannot honestly be called a
    joint miss -- that would conflate "unknown" with "actively missed"."""
    records = [
        {
            "expert": [REF_A, REF_B],
            "annotator": [REF_A],
            "model": [REF_A],
            "judge": None,
        }
    ]
    metrics = rater_agreement(records)
    assert metrics["joint_miss_count"] == 0
    assert metrics["joint_miss_eligible_document_count"] == 0
    assert metrics["total_expert_reference_count"] == 0
    assert metrics["joint_miss_rate"] is None


# ---------------------------------------------------------------------------
# rater_agreement: per-rater precision/recall/F1/exact-set-agreement
# ---------------------------------------------------------------------------


def test_per_rater_precision_recall_f1_and_exact_set_agreement_are_aggregated() -> None:
    records = [
        {
            "expert": [REF_A, REF_B],
            "annotator": [REF_A],
            "model": [REF_A, REF_B],
            "judge": [REF_A],
        },
        {
            "expert": [REF_A],
            "annotator": [REF_A],
            "model": [],
            "judge": [REF_A, REF_B],
        },
    ]
    metrics = rater_agreement(records)

    annotator = metrics["per_rater"]["annotator"]
    assert (annotator["tp"], annotator["fp"], annotator["fn"]) == (2, 0, 1)
    assert annotator["precision"] == pytest.approx(1.0)
    assert annotator["recall"] == pytest.approx(2 / 3)
    assert annotator["f1"] == pytest.approx(0.8)
    assert annotator["exact_set_matches"] == 1
    assert annotator["exact_set_documents"] == 2
    assert annotator["exact_set_agreement"] == pytest.approx(0.5)

    model = metrics["per_rater"]["model"]
    assert (model["tp"], model["fp"], model["fn"]) == (2, 0, 1)
    assert model["precision"] == pytest.approx(1.0)
    assert model["recall"] == pytest.approx(2 / 3)
    assert model["exact_set_agreement"] == pytest.approx(0.5)

    judge = metrics["per_rater"]["judge"]
    assert (judge["tp"], judge["fp"], judge["fn"]) == (2, 1, 1)
    assert judge["precision"] == pytest.approx(2 / 3)
    assert judge["recall"] == pytest.approx(2 / 3)
    assert judge["exact_set_agreement"] == pytest.approx(0.0)

    assert metrics["model_only_correct"] == 1
    assert metrics["annotator_only_correct"] == 0
    assert metrics["judge_only_correct"] == 0
    assert metrics["document_count"] == 2


def test_rater_never_available_reports_none_not_a_fabricated_perfect_score() -> None:
    """Zero data is not the same as flawless performance; a rater that never ran
    this round must not be reported as precision=1.0/recall=1.0 by default."""
    records = [
        {"expert": [REF_A], "annotator": [REF_A], "model": [REF_A], "judge": None},
        {"expert": [REF_B], "annotator": [REF_B], "model": [REF_B], "judge": None},
    ]
    metrics = rater_agreement(records)
    judge = metrics["per_rater"]["judge"]
    assert judge["available_document_count"] == 0
    assert judge["precision"] is None
    assert judge["recall"] is None
    assert judge["f1"] is None
    assert judge["exact_set_agreement"] is None


# ---------------------------------------------------------------------------
# rater_agreement: Cohen's kappa (pairwise) and Fleiss' kappa
# ---------------------------------------------------------------------------


def test_cohens_kappa_at_chance_level_agreement() -> None:
    # universe = {A, B, C, D}; annotator = {A, B}; model = {A, C}
    records = [
        {
            "expert": [REF_A, REF_B, REF_C, REF_D],
            "annotator": [REF_A, REF_B],
            "model": [REF_A, REF_C],
            "judge": None,
        }
    ]
    metrics = rater_agreement(records)
    pair = metrics["pairwise_agreement"]["annotator_model"]
    assert pair["items"] == 4
    assert pair["agreement_matches"] == 2
    assert pair["agreement"] == pytest.approx(0.5)
    assert pair["cohens_kappa"] == pytest.approx(0.0)


def test_cohens_kappa_perfect_agreement() -> None:
    records = [
        {
            "expert": [REF_A, REF_B, REF_C, REF_D],
            "annotator": [REF_A, REF_B],
            "model": [REF_A, REF_B],
            "judge": None,
        }
    ]
    metrics = rater_agreement(records)
    pair = metrics["pairwise_agreement"]["annotator_model"]
    assert pair["items"] == 4
    assert pair["agreement"] == pytest.approx(1.0)
    assert pair["cohens_kappa"] == pytest.approx(1.0)


def test_cohens_kappa_is_none_when_the_pair_shares_no_documents() -> None:
    """A missing measurement must be reported as `None`, never a fabricated `0.0`."""
    records = [
        {"expert": [REF_A, REF_B], "annotator": [REF_A], "model": None, "judge": [REF_A, REF_B]},
    ]
    metrics = rater_agreement(records)
    pair = metrics["pairwise_agreement"]["annotator_model"]
    assert pair["items"] == 0
    assert pair["agreement"] is None
    assert pair["cohens_kappa"] is None
    # the unaffected pair still gets a real measurement, over the same document
    other = metrics["pairwise_agreement"]["annotator_judge"]
    assert other["items"] == 2
    assert other["agreement_matches"] == 1
    assert other["agreement"] == pytest.approx(0.5)
    assert other["cohens_kappa"] == pytest.approx(0.0)


def test_fleiss_kappa_perfect_agreement_across_three_raters() -> None:
    records = [
        {
            "expert": [REF_A, REF_B, REF_C, REF_D],
            "annotator": [REF_A, REF_B],
            "model": [REF_A, REF_B],
            "judge": [REF_A, REF_B],
        }
    ]
    metrics = rater_agreement(records)
    assert metrics["fleiss_kappa_items"] == 4
    assert metrics["fleiss_kappa_document_count"] == 1
    assert metrics["fleiss_kappa"] == pytest.approx(1.0)


def test_fleiss_kappa_is_none_when_no_document_has_all_three_raters() -> None:
    records = [
        {"expert": [REF_A], "annotator": [REF_A], "model": None, "judge": [REF_A]},
    ]
    metrics = rater_agreement(records)
    assert metrics["fleiss_kappa_items"] == 0
    assert metrics["fleiss_kappa_document_count"] == 0
    assert metrics["fleiss_kappa"] is None


# ---------------------------------------------------------------------------
# rater_agreement: input validation
# ---------------------------------------------------------------------------


def test_rater_agreement_requires_an_expert_entry_for_every_record() -> None:
    with pytest.raises(ContractError, match="expert"):
        rater_agreement([{"annotator": [REF_A], "model": [REF_A], "judge": [REF_A]}])


def test_rater_agreement_over_no_records_is_empty_but_well_formed() -> None:
    metrics: dict[str, Any] = rater_agreement([])
    assert metrics["document_count"] == 0
    assert metrics["joint_miss_count"] == 0
    assert metrics["joint_miss_rate"] is None
    assert metrics["fleiss_kappa"] is None
    for name in ("annotator", "model", "judge"):
        assert metrics["per_rater"][name]["precision"] is None


# ---------------------------------------------------------------------------
# edit_distance_from_best_candidate
# ---------------------------------------------------------------------------


def test_edit_distance_from_best_candidate_picks_the_closer_candidate() -> None:
    result = edit_distance_from_best_candidate(
        final=[REF_A, REF_B], annotator=[REF_A], model=[REF_A, REF_B]
    )
    assert result == {"available": True, "closest_candidate": "model", "edit_distance": 0}


def test_edit_distance_from_best_candidate_counts_added_and_removed_references() -> None:
    """An "altered" reference costs two units: one for the old identity the
    expert removed (spurious relative to `final`) and one for the new
    identity the expert added (missed relative to `final`)."""
    result = edit_distance_from_best_candidate(
        final=[REF_A, REF_B], annotator=[REF_A, REF_C], model=None
    )
    assert result == {"available": True, "closest_candidate": "annotator", "edit_distance": 2}


def test_edit_distance_from_best_candidate_breaks_ties_toward_annotator() -> None:
    # annotator misses REF_B (distance 1); model misses REF_A (distance 1) -- a genuine tie.
    result = edit_distance_from_best_candidate(
        final=[REF_A, REF_B], annotator=[REF_A], model=[REF_B]
    )
    assert result == {"available": True, "closest_candidate": "annotator", "edit_distance": 1}


def test_edit_distance_from_best_candidate_is_unavailable_with_no_candidates() -> None:
    result = edit_distance_from_best_candidate(final=[REF_A], annotator=None, model=None)
    assert result == {"available": False, "closest_candidate": None, "edit_distance": None}


# ---------------------------------------------------------------------------
# documents_with_finalized_truth
# ---------------------------------------------------------------------------


def test_documents_with_finalized_truth_excludes_deferred_and_pending() -> None:
    finalized = {"status": "finalized", "final": [REF_A]}
    deferred = {"status": "deferred", "final": None}
    pending = {"status": "pending", "final": None}
    result = documents_with_finalized_truth([finalized, deferred, pending])
    assert result == [finalized]


# ---------------------------------------------------------------------------
# expert_workload
# ---------------------------------------------------------------------------


def _review_row(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "router_bucket": "RED",
        "escalated": False,
        "status": "finalized",
        "action": "accept_human",
        "created_at_epoch": 0.0,
        "updated_at_epoch": 10.0,
        "annotator": [REF_A],
        "model": [REF_A],
        "final": [REF_A],
    }
    payload.update(updates)
    return payload


def test_expert_workload_counts_documents_requiring_review() -> None:
    """Router RED/YELLOW always require review; GREEN only under an active
    escalation, exactly as `hitl._review_requirements` treats it."""
    records = [
        _review_row(router_bucket="RED"),
        _review_row(router_bucket="YELLOW"),
        _review_row(
            router_bucket="GREEN",
            escalated=False,
            status="pending",
            action=None,
            final=None,
        ),
        _review_row(router_bucket="GREEN", escalated=True),
    ]
    metrics = expert_workload(records)
    assert metrics["document_count"] == 4
    assert metrics["documents_requiring_review_count"] == 3
    assert metrics["documents_requiring_review_rate"] == pytest.approx(0.75)


def test_expert_workload_action_distribution_counts_each_action_and_the_total() -> None:
    records = [
        _review_row(action="accept_human"),
        _review_row(action="accept_model"),
        _review_row(action="revise", final=[REF_A, REF_B]),
        _review_row(action="defer", status="deferred", final=None),
        _review_row(action="judge_override"),
        _review_row(status="pending", action=None, final=None),
    ]
    metrics = expert_workload(records)
    assert metrics["document_count"] == 6
    assert metrics["action_distribution"] == {
        "accept_human": 1,
        "accept_model": 1,
        "revise": 1,
        "defer": 1,
        "judge_override": 1,
        "total": 5,
    }


def test_expert_workload_seconds_to_decision_p50_and_p95_exclude_pending_rows() -> None:
    records = [
        _review_row(created_at_epoch=0.0, updated_at_epoch=10.0),
        _review_row(created_at_epoch=0.0, updated_at_epoch=20.0),
        _review_row(created_at_epoch=0.0, updated_at_epoch=30.0),
        _review_row(
            status="pending",
            action=None,
            final=None,
            created_at_epoch=0.0,
            updated_at_epoch=0.0,
        ),
    ]
    metrics = expert_workload(records)
    assert metrics["seconds_to_decision_sample_count"] == 3
    assert metrics["seconds_to_decision_p50"] == pytest.approx(20.0)
    assert metrics["seconds_to_decision_p95"] == pytest.approx(30.0)


def test_expert_workload_edit_distance_only_measures_revise_rows() -> None:
    records = [
        _review_row(
            action="revise",
            final=[REF_A, REF_B],
            annotator=[REF_A],
            model=[REF_A, REF_B, REF_C],
        ),
        _review_row(action="accept_human"),
    ]
    metrics = expert_workload(records)
    edit = metrics["edit_distance_from_best_candidate"]
    assert edit["revise_count"] == 1
    assert edit["measured_count"] == 1
    assert edit["edit_distances"] == [1]
    assert edit["mean_edit_distance"] == pytest.approx(1.0)
    # annotator misses REF_B (distance 1); model has spurious REF_C (distance 1) -- a tie,
    # broken toward annotator.
    assert edit["closest_candidate_counts"] == {"annotator": 1, "model": 0}


def test_expert_workload_edit_distance_unmeasured_when_no_candidate_available() -> None:
    records = [
        _review_row(action="revise", final=[REF_A], annotator=None, model=None),
    ]
    metrics = expert_workload(records)
    edit = metrics["edit_distance_from_best_candidate"]
    assert edit["revise_count"] == 1
    assert edit["measured_count"] == 0
    assert edit["edit_distances"] == []
    assert edit["mean_edit_distance"] is None
    assert edit["total_edit_distance"] is None


def test_expert_workload_requires_final_references_when_finalized() -> None:
    row = _review_row(final=None)
    with pytest.raises(ContractError, match="final"):
        expert_workload([row])


def test_expert_workload_rejects_a_deferred_status_with_a_non_defer_action() -> None:
    row = _review_row(status="deferred", action="accept_human")
    with pytest.raises(ContractError):
        expert_workload([row])


def test_expert_workload_over_no_records_is_empty_but_well_formed() -> None:
    metrics = expert_workload([])
    assert metrics["document_count"] == 0
    assert metrics["documents_requiring_review_count"] == 0
    assert metrics["documents_requiring_review_rate"] is None
    assert metrics["seconds_to_decision_sample_count"] == 0
    assert metrics["seconds_to_decision_p50"] is None
    assert metrics["seconds_to_decision_p95"] is None
    assert metrics["edit_distance_from_best_candidate"]["mean_edit_distance"] is None


# ---------------------------------------------------------------------------
# The discriminating case: defer is workload, never truth
# ---------------------------------------------------------------------------


def test_deferred_row_counts_as_workload_but_is_excluded_from_truth() -> None:
    """A `defer` row is real work performed -- it must show up in every
    workload count -- but `final_references_json` is stored `null` for it,
    so it must never be treated as expert truth.

    This was confirmed by hand: `expert_workload`'s edit-distance gate was
    temporarily widened from `if action == "revise":` to
    `if action in {"revise", "defer"}:` (i.e. including deferred rows in
    the truth-bearing set fed to `edit_distance_from_best_candidate`), and
    this test was rerun. It failed -- not with a wrong number but with an
    unhandled `ContractError` from `attribute_errors` ("requires the
    expert's finalized references"), because a deferred row's `final` is
    `None`. The widening was then reverted and the test passes again.
    """
    deferred = _review_row(action="defer", status="deferred", final=None, reason="second opinion")
    finalized = _review_row(action="accept_human")

    metrics = expert_workload([deferred, finalized])
    assert metrics["document_count"] == 2
    assert metrics["deferred_count"] == 1
    assert metrics["action_distribution"]["defer"] == 1
    assert metrics["edit_distance_from_best_candidate"]["revise_count"] == 0

    truth_bearing = documents_with_finalized_truth([deferred, finalized])
    assert deferred not in truth_bearing
    assert truth_bearing == [finalized]


def test_green_audit_sample_counts_as_required_review() -> None:
    """A GREEN document in the audit sample is mandatory review even unescalated.

    hitl._review_requirements seeds the required set with the audit sample
    before it looks at any bucket. Omitting it here would understate expert
    workload exactly where the paper's claim is strongest -- late rounds, when
    most documents route GREEN and the audit sample is most of what is left.
    """
    records = [
        {
            "internal_doc_id": "d1",
            "router_bucket": "GREEN",
            "escalated": False,
            "in_green_audit_sample": True,
            "status": "pending",
        },
        {
            "internal_doc_id": "d2",
            "router_bucket": "GREEN",
            "escalated": False,
            "in_green_audit_sample": False,
            "status": "pending",
        },
    ]
    result = expert_workload(records)
    assert result["documents_requiring_review_count"] == 1
    assert result["document_count"] == 2
