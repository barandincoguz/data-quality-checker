"""Corrections to a sealed release's truth: decided by the user, applied on reading.

Four R01 decisions (5 and 7 Oct 2026) accepted an annotator list that held only
the endpoints of a range the text writes ("6098 sayılı Kanunun 620 ila 645 inci
maddeleri" -> 620 and 645) before the user's rule: "473 ila 475" means 473, 474
and 475, "HER MADDE AYRI SATIR OLARAK TUTULUR". A release is immutable, so the
user chose (7 Oct 2026, "Kuralla düzelt"): a dated corrections file beside the
release adds the articles in between; the training data and the error report
read it, the release stays as it was.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_loop_training import _Cfg, write_sealed_release

from data_quality_checker.errors import ContractError
from data_quality_checker.loop_training import round_training_documents
from data_quality_checker.truth_corrections import (
    apply_corrections,
    corrections_summary,
    endpoint_only_ranges,
    load_corrections,
    write_range_corrections,
    write_row_fixes,
)

TEXT = "Söz konusu ortaklık, 6098 sayılı Türk Borçlar Kanununun 620 ila 645 inci maddelerinde düzenlenmiştir."


def ref(no: str, md: str, name: str = "") -> dict[str, str]:
    return {"kanun_no": no, "kanun_ad": name, "madde": md, "source_text": ""}


def test_a_range_held_by_its_endpoints_only_is_found() -> None:
    found = endpoint_only_ranges(TEXT, [ref("6098", "620", "Türk Borçlar Kanunu"), ref("6098", "645")])
    assert found == [{"kanun_no": "6098", "low": 620, "high": 645, "text_writes": "620 ila 645"}]


def test_a_range_held_otherwise_is_left_alone() -> None:
    assert endpoint_only_ranges(TEXT, [ref("6098", "620 ila 645")]) == []
    assert endpoint_only_ranges(TEXT, [ref("6098", str(n)) for n in range(620, 646)]) == []
    assert endpoint_only_ranges(TEXT, [ref("6098", "620"), ref("6098", "630"), ref("6098", "645")]) == []
    assert endpoint_only_ranges(TEXT, [ref("6098", "620")]) == []
    assert endpoint_only_ranges("6098 sayılı Kanunun 620 nci maddesi", [ref("6098", "620"), ref("6098", "645")]) == []


def test_applying_adds_the_articles_in_between_after_the_first_endpoint() -> None:
    rows = [ref("193", "94"), ref("6098", "638", "Türk Borçlar Kanunu"), ref("6098", "642")]
    out = apply_corrections(rows, [{"kanun_no": "6098", "low": 638, "high": 642}])
    assert [(r["kanun_no"], r["madde"]) for r in out] == [
        ("193", "94"), ("6098", "638"), ("6098", "639"), ("6098", "640"), ("6098", "641"), ("6098", "642")]
    assert out[2] == {"kanun_no": "6098", "kanun_ad": "Türk Borçlar Kanunu", "madde": "639", "source_text": ""}
    assert len(rows) == 3


def test_a_correction_whose_endpoints_are_gone_refuses() -> None:
    with pytest.raises(ContractError):
        apply_corrections([ref("6098", "620")], [{"kanun_no": "6098", "low": 620, "high": 645}])


def _seal(root: Path) -> None:
    write_sealed_release(
        root,
        "R01",
        expert=[{"internal_doc_id": "d1", "text": TEXT, "references": [ref("6098", "620"), ref("6098", "645")]}],
        consensus=[{"internal_doc_id": "d2", "text": "193 sayılı Kanunun 94 üncü maddesi.", "references": [ref("193", "94")]}],
    )


def test_writing_records_the_decision_and_the_release_it_corrects(tmp_path: Path) -> None:
    _seal(tmp_path)
    payload = write_range_corrections(_Cfg(tmp_path), batch_id="R01", decision="Kuralla düzelt (Önerilen)")
    on_disk = json.loads((tmp_path / "releases/R01/truth_corrections.json").read_text(encoding="utf-8"))
    assert on_disk == payload
    [decision] = payload["decisions"]
    assert payload["release_id"] == "release_fixture" and decision["quote"] == "Kuralla düzelt (Önerilen)"
    assert [(c["internal_doc_id"], c["low"], c["high"], c["rows_added"]) for c in decision["corrections"]] == [
        ("d1", 620, 645, 24)]
    # Writing again with the same findings changes nothing; a different decision text is refused.
    assert write_range_corrections(_Cfg(tmp_path), batch_id="R01", decision="Kuralla düzelt (Önerilen)") == payload
    with pytest.raises(ContractError):
        write_range_corrections(_Cfg(tmp_path), batch_id="R01", decision="başka")


def test_a_consensus_document_is_listed_not_corrected(tmp_path: Path) -> None:
    """The decision is about the expert's lists; a consensus truth is the annotator's list."""
    write_sealed_release(
        tmp_path,
        "R01",
        expert=[],
        consensus=[{"internal_doc_id": "d3", "text": TEXT, "references": [ref("6098", "620"), ref("6098", "645")]}],
    )
    payload = write_range_corrections(_Cfg(tmp_path), batch_id="R01", decision="Kuralla düzelt (Önerilen)")
    [decision] = payload["decisions"]
    assert decision["corrections"] == []
    assert [c["internal_doc_id"] for c in decision["consensus_documents_not_corrected"]] == ["d3"]


def test_the_training_data_reads_the_corrected_truth(tmp_path: Path) -> None:
    _seal(tmp_path)
    before = round_training_documents(_Cfg(tmp_path), batch_ids=["R01"])
    assert {r["madde"] for r in before["d1"]["references"]} == {"620", "645"}
    write_range_corrections(_Cfg(tmp_path), batch_id="R01", decision="Kuralla düzelt (Önerilen)")
    after = round_training_documents(_Cfg(tmp_path), batch_ids=["R01"])
    assert {r["madde"] for r in after["d1"]["references"]} == {str(n) for n in range(620, 646)}
    assert after["d2"] == before["d2"]


def test_a_corrections_file_for_another_release_is_refused(tmp_path: Path) -> None:
    _seal(tmp_path)
    write_range_corrections(_Cfg(tmp_path), batch_id="R01", decision="Kuralla düzelt (Önerilen)")
    path = tmp_path / "releases/R01/truth_corrections.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**payload, "release_id": "release_other"}), encoding="utf-8")
    with pytest.raises(ContractError):
        load_corrections(_Cfg(tmp_path), "R01")


def test_no_file_means_no_corrections(tmp_path: Path) -> None:
    _seal(tmp_path)
    assert load_corrections(_Cfg(tmp_path), "R01") == {}


# --- rows the expert replaced or removed after the release (7 Oct 2026, "Önerileri uygula") ---------


def test_a_replace_puts_its_row_where_every_repeat_of_the_old_one_stood() -> None:
    rows = [ref("3065", "1"), ref("3065", "41"), ref("3065", "10"), ref("3065", "41")]
    fix = {"kind": "replace", "row": {"kanun_no": "3065", "madde": "41"},
           "with": {"kanun_no": "3065", "kanun_ad": "Katma Değer Vergisi Kanunu", "madde": "4"}}
    out = apply_corrections(rows, [fix])
    assert [r["madde"] for r in out] == ["1", "4", "10"]
    assert out[1] == {"kanun_no": "3065", "kanun_ad": "Katma Değer Vergisi Kanunu", "madde": "4", "source_text": ""}


def test_a_replace_by_a_row_already_held_only_removes() -> None:
    fix = {"kind": "replace", "row": {"kanun_no": "3065", "madde": "41"}, "with": {"kanun_no": "3065", "madde": "4"}}
    assert [r["madde"] for r in apply_corrections([ref("3065", "4"), ref("3065", "41")], [fix])] == ["4"]


def test_a_remove_drops_the_row_and_a_missing_row_refuses() -> None:
    fix = {"kind": "remove", "row": {"kanun_no": "488", "madde": "3"}}
    assert apply_corrections([ref("488", "3"), ref("488", "9")], [fix]) == [ref("488", "9")]
    with pytest.raises(ContractError):
        apply_corrections([ref("488", "9")], [fix])


def _fix(doc: str, kind: str, no: str, md: str, **with_: str) -> dict:
    fix = {"internal_doc_id": doc, "kind": kind, "row": {"kanun_no": no, "madde": md}, "reason": "test"}
    return {**fix, "with": with_} if with_ else fix


def test_row_fixes_are_a_second_decision_read_after_the_ranges(tmp_path: Path) -> None:
    _seal(tmp_path)
    write_range_corrections(_Cfg(tmp_path), batch_id="R01", decision="Kuralla düzelt (Önerilen)")
    fixes = [_fix("d1", "replace", "6098", "620", kanun_no="6098", madde="mükerrer 620")]
    payload = write_row_fixes(_Cfg(tmp_path), batch_id="R01", decision="Önerileri uygula (Önerilen)", fixes=fixes)
    assert [d["rule"] for d in payload["decisions"]] == ["article_range_endpoints_v1", "expert_row_fix_v1"]
    [out] = [apply_corrections([ref("6098", "620"), ref("6098", "645")], load_corrections(_Cfg(tmp_path), "R01")["d1"])]
    assert [r["madde"] for r in out][:2] == ["mükerrer 620", "621"] and len(out) == 26
    summary = corrections_summary(_Cfg(tmp_path), "R01")
    assert [(d["rule"], d["kinds"]) for d in summary["decisions"]] == [
        ("article_range_endpoints_v1", {"range": 1}), ("expert_row_fix_v1", {"replace": 1})]
    # Same decision again: nothing changes; another one under the same rule: refused.
    assert write_row_fixes(_Cfg(tmp_path), batch_id="R01", decision="Önerileri uygula (Önerilen)", fixes=fixes) == payload
    with pytest.raises(ContractError):
        write_row_fixes(_Cfg(tmp_path), batch_id="R01", decision="başka", fixes=fixes)


def test_a_row_fix_that_does_not_apply_or_touches_a_consensus_document_is_refused(tmp_path: Path) -> None:
    _seal(tmp_path)
    for fixes in ([_fix("d1", "remove", "6098", "999")],
                  [_fix("d2", "remove", "193", "94")],
                  [{**_fix("d1", "remove", "6098", "620"), "reason": ""}],
                  [_fix("d1", "replace", "6098", "620")]):
        with pytest.raises(ContractError):
            write_row_fixes(_Cfg(tmp_path), batch_id="R01", decision="Önerileri uygula (Önerilen)", fixes=fixes)
    assert not (tmp_path / "releases/R01/truth_corrections.json").exists()
