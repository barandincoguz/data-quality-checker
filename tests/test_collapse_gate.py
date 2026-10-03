"""The empty-answer gate stops a collapsing model, not one missed document (3 Oct 2026)."""

from data_quality_checker.g0_training import COLLAPSE_EMPTY_ANSWERS, _collapsed, empty_answers_on_referenced_documents


def evaluation(per_doc):
    return {"results": [{"core_per_doc_metrics": [{"doc_id": i, "tp": tp, "fp": fp, "fn": fn} for i, (tp, fp, fn) in enumerate(per_doc)]}]}


def test_a_correct_empty_answer_on_a_document_without_references_is_not_counted():
    assert empty_answers_on_referenced_documents(evaluation([(0, 0, 0), (3, 0, 0)])) == 0


def test_one_missed_document_does_not_stop_a_run_but_a_collapse_does():
    one = {"predicted_reference_count": 200, "empty_answers_on_referenced_documents":
           empty_answers_on_referenced_documents(evaluation([(0, 0, 1), (3, 0, 0)]))}
    assert one["empty_answers_on_referenced_documents"] == 1 and not _collapsed(one)
    many = {"predicted_reference_count": 200, "empty_answers_on_referenced_documents": COLLAPSE_EMPTY_ANSWERS}
    assert _collapsed(many)
    assert _collapsed({"predicted_reference_count": 0, "empty_answers_on_referenced_documents": 0})


def test_a_summary_written_before_the_change_keeps_the_old_rule():
    assert _collapsed({"predicted_reference_count": 10, "parsed_zero_reference_output_count": 1,
                       "coverage_count": 50, "parse_count": 50, "zero_reference_output_count": 1})
