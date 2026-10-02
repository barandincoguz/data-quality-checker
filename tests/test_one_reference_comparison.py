"""One reference comparison for the whole project (user decision, 2 Oct 2026).

"Döngü motoru ve değerlendiriciyi artık farklı iki şey gibi görmeyelim." Pinned:
the user's two rules -- a law without its number is matched by its name and
article; a law-only row beside an article of the same law *in the same list* is
extra, not wrong -- and that every module counting references answers with the
same numbers: the evaluator (validation F1), the router, the error attribution,
the journal table, the reference audit, the screen.
"""

from __future__ import annotations

import json

import pytest

from benchmark.reference.evaluate import evaluate_prediction_file
from data_quality_checker.hitl import ab_diff
from data_quality_checker.loop_metrics import attribute_errors, reference_table
from data_quality_checker.reference_audit import triage_disagreement
from data_quality_checker.reference_matching import compare_reference_lists
from data_quality_checker.router import route_document


def ref(kanun_no: str, kanun_ad: str = "", madde: str = "") -> dict[str, str]:
    return {"kanun_no": kanun_no, "kanun_ad": kanun_ad, "madde": madde, "source_text": "s"}


def counts(reference, candidate) -> tuple[int, int, int]:
    comparison = compare_reference_lists(reference, candidate)
    return comparison.tp, comparison.fp, comparison.fn


# -- the name fallback ---------------------------------------------------------

def test_a_law_without_its_number_matches_by_name_and_article() -> None:
    assert counts([ref("492", "Harçlar Kanunu", "57")], [ref("", "Harçlar Kanunu", "57")]) == (1, 0, 0)
    assert counts([ref("", "Harçlar Kanunu", "57")], [ref("492", "Harçlar Kanunu", "57")]) == (1, 0, 0)
    assert counts([ref("", "İmar Kanunu", "18")], [ref("", "İMAR KANUNU", "18")]) == (1, 0, 0)
    assert counts([ref("193", "", "94")], [ref("", "Gelir Vergisi Kanunu", "94")]) == (1, 0, 0)


def test_the_name_fallback_needs_the_same_article_and_the_same_law() -> None:
    assert counts([ref("492", "Harçlar Kanunu", "57")], [ref("", "Harçlar Kanunu", "58")]) == (0, 1, 1)
    assert counts([ref("492", "Harçlar Kanunu", "57")], [ref("", "Gelir Vergisi Kanunu", "57")]) == (0, 1, 1)
    # A number with no name says nothing about a name-only row.
    assert counts([ref("6770", "", "1")], [ref("", "Harçlar Kanunu", "1")]) == (0, 1, 1)


def test_a_name_two_laws_share_is_never_resolved_to_either() -> None:
    """5422 and its successor 5520 are both the Kurumlar Vergisi Kanunu."""
    gold = [ref("5422", "Kurumlar Vergisi Kanunu", ""), ref("5520", "Kurumlar Vergisi Kanunu", "14")]
    assert counts(gold, [ref("5520", "", "14"), ref("5422", "", "")]) == (2, 0, 0)
    tp, fp, fn = counts(gold, [ref("", "Kurumlar Vergisi Kanunu", "14"), ref("5422", "", "")])
    assert (tp, fp, fn) == (1, 1, 1)  # the name-only row is not guessed to be 5520
    assert counts([ref("", "Kurumlar Vergisi Kanunu", "14")], [ref("", "Kurumlar Vergisi Kanunu", "14")]) == (1, 0, 0)


def test_two_written_numbers_decide_alone() -> None:
    assert counts([ref("213", "Vergi Usul Kanunu", "5")], [ref("213", "Gelir Vergisi Kanunu", "5")]) == (1, 0, 0)
    assert counts([ref("213", "Vergi Usul Kanunu", "5")], [ref("193", "Vergi Usul Kanunu", "5")]) == (0, 1, 1)


# -- a law-only row beside an article of its law -------------------------------

def test_a_law_only_row_beside_its_article_in_the_same_list_is_extra_not_wrong() -> None:
    gold = [ref("193", "", "94")]
    comparison = compare_reference_lists(gold, [ref("193"), ref("193", "", "94")])
    assert (comparison.tp, comparison.fp, comparison.fn) == (1, 0, 0)
    assert len(comparison.excused_candidate) == 1
    # Either side, either order.
    assert counts([ref("193"), ref("193", "", "94")], [ref("193", "", "94")]) == (1, 0, 0)
    assert counts(gold, [ref("193", "", "94"), ref("", "Gelir Vergisi Kanunu", "")]) == (1, 0, 0)


def test_a_law_only_row_beside_its_article_still_scores_when_the_other_list_has_it() -> None:
    gold = [ref("6770")]
    assert counts(gold, [ref("6770"), ref("6770", "", "5")]) == (1, 1, 0)


def test_a_law_only_row_without_an_article_of_its_law_in_its_own_list_is_a_claim() -> None:
    """User decision (2 Oct 2026): only the same list excuses it."""
    assert counts([ref("193", "", "94")], [ref("193")]) == (0, 1, 1)
    assert counts([ref("7338")], [ref("7338", "", "23")]) == (0, 1, 1)


# -- every module answers alike ------------------------------------------------

PAIRS = [
    ([ref("492", "Harçlar Kanunu", "57")], [ref("", "Harçlar Kanunu", "57")]),
    ([ref("193", "", "94")], [ref("193"), ref("193", "", "94")]),
    ([ref("193"), ref("193", "", "94")], [ref("193", "", "94"), ref("213", "", "5")]),
    ([ref("193", "", "94")], [ref("193")]),
    ([ref("3065", "", "17"), ref("3065", "", "1")], [ref("3065", "", "(17/4-r)"), ref("3065", "", "-1/1")]),
    ([ref("5520", "", "32/A")], [ref("5520", "", "32/a")]),
    ([ref("213", "", "413"), ref("193", "", "94")], [ref("193", "", "94")]),
    ([ref("", "Kurumlar Vergisi Kanunu", "14"), ref("5422", "Kurumlar Vergisi Kanunu", "")],
     [ref("5520", "Kurumlar Vergisi Kanunu", "14"), ref("5422", "", "")]),
]


@pytest.mark.parametrize(("truth", "candidate"), PAIRS)
def test_every_module_counts_a_document_the_same(truth, candidate, tmp_path) -> None:
    from data_quality_checker.reference_matching import compare_with_policy

    expected = compare_with_policy(truth, candidate)
    tp, fp, fn = expected.tp, expected.fp, expected.fn

    route = route_document(human_references=truth, model_references=candidate)
    assert (route.bucket == "GREEN") == expected.agree

    errors = attribute_errors(expert=truth, annotator=candidate, model=None, judge=None)["annotator"]
    assert (len(errors["correct"]), len(errors["spurious"]), len(errors["missed"])) == (tp, fp, fn)

    table = reference_table([{"doc_id": 1, "truth": truth, "truth_source": "expert",
                              "annotator": candidate, "model": None, "judge": None}], samples=10)
    core = table["sources"]["annotator"]["core"]
    assert (core["tp"], core["fp"], core["fn"]) == (tp, fp, fn)

    audit = triage_disagreement(expert=truth, candidate=candidate, document_text="")
    assert (audit["candidate_matched_count"], audit["candidate_only_count"], audit["expert_only_count"]) == (tp, fp, fn)

    rows = ab_diff(truth, candidate)
    assert (not any(row["status"] != "same" for row in rows)) == compare_reference_lists(truth, candidate).agree

    gold = {1: [dict(r, type="kanun_madde_referansi") for r in truth]}
    predictions = tmp_path / "p.json"
    predictions.write_text(json.dumps([{"doc_id": 1, "status": "success", "references": candidate}]))
    from benchmark.reference.evaluate import parse_ignored_law_articles

    result = evaluate_prediction_file(predictions, gold, None, ignore_law_articles=parse_ignored_law_articles(["213:413"]))
    scored = result["core_law_article_strict"]
    assert (scored["tp"], scored["fp"], scored["fn"]) == (tp, fp, fn)


def test_a_law_only_row_beside_its_article_in_the_truth_is_no_reference_to_find() -> None:
    """tp + fn is the truth's number of references, always: the truth's own extra
    law-only row is neither found nor missed."""
    truth = [ref("193"), ref("193", "", "94")]
    assert counts(truth, [ref("193", "", "94")]) == (1, 0, 0)
    assert counts(truth, [ref("193"), ref("193", "", "94")]) == (1, 0, 0)
    assert counts(truth, [ref("193")]) == (0, 1, 1)  # alone in its list: a claim (user decision)


@pytest.mark.parametrize(("truth", "candidate"), PAIRS)
def test_found_plus_missed_is_the_truths_reference_count(truth, candidate) -> None:
    from data_quality_checker.reference_matching import reference_identities

    comparison = compare_reference_lists(truth, candidate)
    assert comparison.tp + comparison.fn == len(reference_identities(truth, context_lists=[candidate]))


def test_the_journal_tables_columns_add_up() -> None:
    truth = [ref("193", "", "94"), ref("6770")]
    candidate = [ref("193"), ref("193", "", "94"), ref("6770"), ref("6770", "", "5"), ref("213", "", "5"), ref("213", "", "5")]
    table = reference_table([{"doc_id": 1, "truth": truth, "truth_source": "expert",
                              "annotator": candidate, "model": None, "judge": None}], samples=10)
    a = table["sources"]["annotator"]
    core = a["core"]
    assert (core["tp"], core["fp"], core["fn"]) == (2, 2, 0)
    assert a["written_references"] == core["tp"] + core["fp"]
    assert a["truth_references"] == core["tp"] + core["fn"]
    assert a["subsumed_rows"] == 1 and a["duplicate_rows"] == 1
    assert a["written_rows"] == a["written_references"] + a["subsumed_rows"] + a["duplicate_rows"]
