"""How the annotator and the model complement each other against the expert's truth.

User (7 Oct 2026): when the expert decides "neither is right" and writes the
references, what is measured? The error report scored each source against the
expert, and the workload counted revisions and their edit distance; it never
counted the references only the expert brought (no source had them), the ones
both sources wrote wrongly, or how much of the truth the two sources together
already held. This block does, on the same comparison (`attribute_errors`).
"""

from __future__ import annotations

from data_quality_checker.loop_metrics import complementarity


def ref(no: str, md: str) -> dict[str, str]:
    return {"kanun_no": no, "kanun_ad": "", "madde": md, "source_text": ""}


def record(final, annotator, model, judge=None, action="revise", status="finalized") -> dict:
    return {
        "status": status,
        "action": action,
        "final": final,
        "annotator": annotator,
        "model": model,
        "judge": judge,
    }


def test_each_truth_reference_is_counted_by_who_found_it() -> None:
    records = [
        record(
            final=[ref("193", "94"), ref("193", "61"), ref("213", "230"), ref("5520", "10")],
            annotator=[ref("193", "94"), ref("193", "61"), ref("3065", "1")],
            model=[ref("193", "94"), ref("213", "230"), ref("3065", "1")],
            judge=[ref("5520", "10")],
        )
    ]
    c = complementarity(records)["all_finalized"]
    assert c["truth_references"] == 4
    assert (c["found_by_annotator"], c["found_by_model"]) == (2, 2)
    assert (c["found_by_both"], c["found_by_either"], c["found_by_neither"]) == (1, 3, 1)
    assert c["found_by_none_of_three"] == 0
    assert c["union_recall"] == 0.75
    assert c["wrong_in_both"] == 1


def test_revisions_are_reported_on_their_own() -> None:
    records = [
        record([ref("193", "94")], [ref("193", "94")], [ref("193", "94")], action="accept_human"),
        record([ref("193", "61")], [], []),
    ]
    c = complementarity(records)
    assert c["all_finalized"]["documents"] == 2 and c["revise"]["documents"] == 1
    assert c["revise"]["found_by_neither"] == 1


def test_deferred_and_pending_rows_are_not_truth() -> None:
    records = [
        record(None, [ref("193", "94")], [], status="deferred", action="defer"),
        record(None, [ref("193", "94")], [], status="pending", action=None),
    ]
    assert complementarity(records)["all_finalized"]["documents"] == 0


def test_a_document_without_a_model_output_is_left_out_not_counted_as_a_miss() -> None:
    records = [record([ref("193", "94")], [ref("193", "94")], None)]
    c = complementarity(records)["all_finalized"]
    assert c["documents"] == 0 and c["documents_without_both_sources"] == 1


def test_decisions_carried_from_the_earlier_loop_are_reported_apart() -> None:
    """R01 carried 23 decisions from the fıkra/bent loop; their 'accept_model' took
    that loop's model, so against today's sources they look like expert additions."""
    new = record([ref("193", "94")], [ref("193", "94")], [ref("193", "94")], action="accept_human")
    carried = {**record([ref("213", "230")], [], [], action="accept_model"), "carried": True}
    c = complementarity([new, carried])
    assert c["all_finalized"]["found_by_neither"] == 1
    assert c["new_decisions"]["documents"] == 1 and c["new_decisions"]["found_by_neither"] == 0
    assert c["carried_documents"] == 1


def test_workload_timing_leaves_carried_decisions_out() -> None:
    from data_quality_checker.loop_metrics import expert_workload

    def row(doc, seconds, carried=False):
        return {
            "internal_doc_id": doc,
            "router_bucket": "RED",
            "in_green_audit_sample": False,
            "status": "finalized",
            "action": "accept_human",
            "created_at_epoch": 0.0,
            "updated_at_epoch": 1.0,
            "seconds_on_page": seconds,
            "annotator": [],
            "model": [],
            "final": [],
            "carried": carried,
        }

    w = expert_workload([row("d1", 60.0), row("d2", 120.0), row("d3", 5000.0, carried=True)])
    assert w["carried_decision_count"] == 1
    assert w["seconds_to_decision_new_p50"] == 60.0  # the project's nearest-rank percentile
    assert w["expert_seconds_new_total"] == 180.0


def test_the_error_report_reads_which_decisions_were_carried() -> None:
    import inspect

    from data_quality_checker import loop_bindings

    source = inspect.getsource(loop_bindings.round_records)
    assert "carried_decisions.json" in source and '"carried":' in source
