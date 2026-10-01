"""The journal's per-batch table: how many references each source found, how many
were wrong, what each missed -- against the expert -- and how the model does
batch after batch (user story, 1 Oct 2026):

  "For 50 rulings: the model found 150 references, the annotator 180; of the
  annotator's 180, 12 are wrong; of the model's 150, 40 are wrong; expert vs
  model errors, expert vs annotator errors; then for every 250-document batch,
  how good the model is and how much it improved."

Two levels: provision (law + article) and reference (law + article + fikra +
bent). 213/413 drop from both sides; the annotator's `source_text` is never
read.
"""

from __future__ import annotations

import pytest

from data_quality_checker.loop_metrics import (
    reference_table,
    scoped_truth_records,
    unpaired_f1_delta,
)


def r(no: str, madde: str, fikra: str = "", bent: str = "") -> dict[str, str]:
    return {"kanun_no": no, "kanun_ad": "", "madde": madde, "fikra": fikra, "bent": bent, "source_text": ""}


def _story():
    expert = [r("193", "94", "1"), r("193", "94", "2"), r("213", "30")]
    annotator = [r("193", "94", "1"), r("193", "94", "2"), r("213", "30"), r("213", "31"), r("213", "413")]
    model = [r("193", "94", "1"), r("213", "30", "2")]
    return [
        {"doc_id": "d1", "truth": expert, "truth_source": "expert", "annotator": annotator, "model": model, "judge": expert},
        {"doc_id": "d2", "truth": [], "truth_source": "expert", "annotator": [], "model": [r("5520", "10")], "judge": None},
    ]


def test_each_source_counts_what_it_found_and_what_it_got_wrong() -> None:
    table = reference_table(_story(), samples=200)
    assert table["documents"] == 2 and table["documents_by_truth_source"] == {"expert": 2}
    human = table["sources"]["annotator"]
    assert human["documents"] == 2
    assert (human["written_rows"], human["policy_dropped"]) == (4, 1)  # 213/413 never counts
    assert (human["written_references"], human["written_provisions"]) == (4, 3)
    assert (human["truth_references"], human["truth_provisions"]) == (3, 2)
    assert (human["full"]["tp"], human["full"]["fp"], human["full"]["fn"]) == (3, 1, 0)
    assert (human["core"]["tp"], human["core"]["fp"], human["core"]["fn"]) == (2, 1, 0)
    assert (human["wrong_references"], human["missed_references"], human["detail_errors"]) == (1, 0, 0)

    model = table["sources"]["model"]
    assert (model["written_references"], model["written_provisions"]) == (3, 3)
    # Right article, wrong paragraph twice: right at the provision level, wrong as references.
    assert (model["core"]["tp"], model["core"]["fp"], model["core"]["fn"]) == (2, 1, 0)
    assert (model["full"]["tp"], model["full"]["fp"], model["full"]["fn"]) == (1, 2, 2)
    assert (model["wrong_references"], model["missed_references"], model["detail_errors"]) == (2, 2, 2)
    assert model["full"]["precision"] == pytest.approx(1 / 3)
    assert model["full"]["recall"] == pytest.approx(1 / 3)

    judge = table["sources"]["judge"]
    assert judge["documents"] == 1 and judge["full"]["fp"] == 0 and judge["full"]["fn"] == 0


def test_every_f1_carries_a_document_bootstrap_interval() -> None:
    table = reference_table(_story(), samples=500, seed=42)
    for source in ("annotator", "model"):
        for level in ("core", "full"):
            low, high = table["sources"][source][level]["f1_ci95"]
            assert low <= table["sources"][source][level]["f1"] <= high
    assert table == reference_table(_story(), samples=500, seed=42)  # deterministic


def _record(doc_id, bucket, status, *, final=None, audit=False):
    refs = [r("193", "94")]
    return {"internal_doc_id": doc_id, "router_bucket": bucket, "status": status, "escalated": False,
            "in_green_audit_sample": audit, "final": final, "annotator": refs, "model": refs, "judge": refs}


def test_the_batch_scope_takes_the_agreed_list_only_where_the_green_audit_passed() -> None:
    records = [
        _record("red", "RED", "finalized", final=[r("193", "94")]),
        _record("audit", "GREEN", "finalized", final=[r("193", "94")], audit=True),
        _record("green", "GREEN", "pending"),
        _record("pending-red", "RED", "pending"),
    ]
    reviewed = scoped_truth_records(records, scope="reviewed")
    assert [x["doc_id"] for x in reviewed["records"]] == ["red", "audit"]
    batch = scoped_truth_records(records, scope="batch")
    assert {x["doc_id"]: x["truth_source"] for x in batch["records"]} == {
        "red": "expert", "audit": "expert", "green": "consensus"}
    assert batch["documents_without_truth"] == ["pending-red"]

    unfinished = [*records[:1], _record("audit", "GREEN", "pending", audit=True), *records[2:]]
    batch = scoped_truth_records(unfinished, scope="batch")
    # Until the audit sample is decided, the agreed GREEN lists are not truth yet.
    assert "green" in batch["documents_without_truth"]
    escalated = [{**x, "escalated": True} for x in records]
    assert "green" in scoped_truth_records(escalated, scope="batch")["documents_without_truth"]


def test_batch_to_batch_change_is_an_unpaired_bootstrap() -> None:
    worse = {f"a{i}": {"tp": 1, "fp": 1, "fn": 0} for i in range(30)}
    better = {f"b{i}": {"tp": 2, "fp": 0, "fn": 0} for i in range(30)}
    delta = unpaired_f1_delta(worse, better, samples=500, seed=42)
    assert delta["delta"] == pytest.approx(1.0 - 2 / 3)
    assert delta["lower_2_5"] > 0 and delta["verdict"] == "improved"
    same = unpaired_f1_delta(worse, worse, samples=500, seed=42)
    assert same["verdict"] == "inconclusive"


def test_the_round_error_report_carries_the_table_in_both_scopes(tmp_path) -> None:
    import json

    from test_judges import prepared_processed_fixture

    from data_quality_checker.constants import JUDGE_MODEL_KEY
    from data_quality_checker.loop_bindings import write_error_report
    from data_quality_checker.storage import Store

    config = prepared_processed_fixture(tmp_path, count=3, references=[r("193", "94")])
    with Store(config.database_path) as store:
        ids = sorted(row["internal_doc_id"] for row in store.list_documents("batch"))
        store.set_router_bucket("batch", ids[0], "RED")
        for doc_id in ids[1:]:
            store.set_router_bucket("batch", doc_id, "GREEN")
        review = store.get_review("batch", ids[0])
        store.update_review(batch_id="batch", internal_doc_id=ids[0], expected_version=review["row_version"],
                            status="finalized", action="revise", final_references=[r("193", "94"), r("213", "30")],
                            reason=None, reviewer="fixture")
    report = write_error_report(config, round_index=1, batch_id="batch", generation="G0",
                                judge_model=JUDGE_MODEL_KEY, path=tmp_path / "e.json")
    table = report["reference_table"]
    assert table["reviewed"]["documents"] == 1
    human = table["reviewed"]["sources"]["annotator"]
    assert (human["core"]["tp"], human["core"]["fn"]) == (1, 1)
    batch = table["batch"]
    # Two GREEN documents, neither in a (complete) audit sample here: the audit
    # plan decides whether they count; the table says which truth each row used.
    assert batch["documents"] + len(batch["documents_without_truth"]) == 3
    assert json.loads((tmp_path / "e.json").read_text())["reference_table"]["reviewed"]["documents"] == 1


def test_the_batch_trend_reads_every_round_and_compares_each_with_the_last(tmp_path) -> None:
    import json
    from types import SimpleNamespace

    from data_quality_checker import loop_ops

    rounds = tmp_path / "public" / "loop" / "rounds"
    rounds.mkdir(parents=True)

    def table(f1_tp, docs):
        per_document = {f"d{i}": {"model": {"core": [f1_tp, 2 - f1_tp, 0], "full": [f1_tp, 2 - f1_tp, 0]},
                                  "annotator": {"core": [2, 0, 0], "full": [2, 0, 0]}} for i in range(docs)}
        return {"documents": docs, "documents_without_truth": [], "documents_by_truth_source": {"expert": docs},
                "sources": {"model": {"core": {"f1": 0.5}, "full": {"f1": 0.5}}, "annotator": {"core": {"f1": 1.0}, "full": {"f1": 1.0}}},
                "per_document": per_document}

    for k, tp in ((1, 1), (2, 2)):
        (rounds / f"round_{k:03d}_errors.json").write_text(json.dumps(
            {"round": k, "batch_id": f"R0{k}", "generation": f"M00{k - 1}",
             "reference_table": {"reviewed": table(tp, 30), "batch": table(tp, 30)}}))
    for k, f1 in ((0, 0.90), (1, 0.93)):
        (rounds / f"round_{k:03d}_model.json").write_text(json.dumps({"selection": {"core_f1": f1, "selected_update": 1}}))
    config = SimpleNamespace(public_root=tmp_path / "public", sensitive_root=tmp_path / "s", raw={"loop": {}})
    trend = loop_ops.batch_trend(config)
    first, second = trend["batches"]
    assert first["batch_id"] == "R01" and first["predicted_by"] == "M000" and "vs_previous_batch" not in first["reviewed"]
    change = second["reviewed"]["vs_previous_batch"]["model"]["core"]
    assert change["verdict"] == "improved" and change["delta"] > 0
    assert trend["validation"] == [{"round": 0, "model": "M000", "windows_core_f1": 0.90},
                                   {"round": 1, "model": "M001", "windows_core_f1": 0.93}]
    assert json.loads((rounds / "batch_trend.json").read_text()) == trend
