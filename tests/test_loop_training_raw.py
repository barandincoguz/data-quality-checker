"""The raw-label arm of the noisy-vs-clean ablation (`M_k_raw`).

Pre-registered 4 Oct 2026 (ner `Journal/experiments/dq_loop_gt750/ablation_noisy_vs_clean/`):
the same documents as round k's release, the annotator's raw lists in place of
the cleaned ones, the same reference policy. Placement rule (user decision,
5 Oct 2026): the locator first, exactly as the clean arm; a reference it cannot
place is placed by the annotator's own quote when that quote occurs verbatim in
the text, and otherwise dropped and counted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_loop_train import canonical_fixture
from test_loop_training import write_sealed_release

from data_quality_checker.errors import ContractError
from data_quality_checker.loop_train import train_round
from data_quality_checker.loop_training import round_raw_training_documents
from data_quality_checker.storage import Store

LOCATABLE = "193 sayılı Gelir Vergisi Kanununun 94 üncü maddesi uygulanır."


class _Cfg:
    def __init__(self, root: Path) -> None:
        self.sensitive_root = root

    @property
    def database_path(self) -> Path:
        return self.sensitive_root / "dqcheck.sqlite3"


def _reference(kanun_no: str, madde: str, quote: str, kanun_ad: str = "") -> dict:
    return {"kanun_no": kanun_no, "kanun_ad": kanun_ad, "madde": madde, "source_text": quote}


def _prepare(root: Path, batch_id: str, documents: dict[str, tuple[str, list[dict]]]) -> None:
    """The loop DB as batch preparation leaves it: the annotator's list per document."""
    with Store(root / "dqcheck.sqlite3") as store:
        store.create_batch(batch_id=batch_id, input_fingerprint="in", config_fingerprint="cfg")
        for index, (doc_id, (text, references)) in enumerate(documents.items()):
            store.add_document(
                batch_id,
                {
                    "internal_doc_id": doc_id,
                    "public_doc_id": f"p{index}",
                    "raw_document_id": f"r{index}",
                    "selected_channel": "html",
                    "pdf_coverage": 1.0,
                    "html_coverage": 1.0,
                    "text": text,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "human_references": references,
                },
            )


def test_the_raw_arm_reads_the_released_documents_with_the_annotator_s_lists(tmp_path: Path) -> None:
    _prepare(
        tmp_path,
        "R01",
        {
            "d1": (LOCATABLE, [_reference("193", "94", "94 üncü maddesi")]),
            "d2": ("iki", []),
            "dQ": ("uc", []),
        },
    )
    # The expert removed the annotator's reference from d1; the raw arm keeps it.
    write_sealed_release(
        tmp_path,
        "R01",
        expert=[{"internal_doc_id": "d1", "text": LOCATABLE, "references": []}],
        consensus=[{"internal_doc_id": "d2", "text": "iki", "references": []}],
        quarantine=[{"internal_doc_id": "dQ", "text": "uc", "references": []}],
    )
    documents, audit = round_raw_training_documents(_Cfg(tmp_path), batch_ids=["R01"])
    assert set(documents) == {"d1", "d2"}
    [reference] = documents["d1"]["references"]
    assert (reference["kanun_no"], reference["madde"]) == ("193", "94")
    assert audit["label_source"] == "raw"
    assert audit["batches"]["R01"]["documents"] == 2


def test_the_raw_arm_applies_the_release_s_reference_policy(tmp_path: Path) -> None:
    text = "213 sayılı Vergi Usul Kanununun 413 üncü maddesi uyarınca. " + LOCATABLE
    _prepare(
        tmp_path,
        "R01",
        {"d1": (text, [_reference("213", "413", "413 üncü maddesi"), _reference("193", "94", "x")])},
    )
    write_sealed_release(
        tmp_path, "R01", expert=[{"internal_doc_id": "d1", "text": text, "references": []}], consensus=[]
    )
    documents, _ = round_raw_training_documents(_Cfg(tmp_path), batch_ids=["R01"])
    assert [(r["kanun_no"], r["madde"]) for r in documents["d1"]["references"]] == [("193", "94")]


def test_the_locator_places_a_raw_reference_before_the_annotator_s_quote(tmp_path: Path) -> None:
    """Both arms place a locatable reference the same way."""
    _prepare(tmp_path, "R01", {"d1": (LOCATABLE, [_reference("193", "94", "sayılı")])})
    write_sealed_release(
        tmp_path, "R01", expert=[{"internal_doc_id": "d1", "text": LOCATABLE, "references": []}], consensus=[]
    )
    documents, audit = round_raw_training_documents(_Cfg(tmp_path), batch_ids=["R01"])
    placed = documents["d1"]["references"][0]["source_text"]
    assert placed != "sayılı" and "94 üncü maddesi" in placed
    assert audit["batches"]["R01"]["placed_by_locator"] == 1
    assert audit["batches"]["R01"]["placed_by_annotator_quote"] == 0


def test_an_unlocatable_reference_is_placed_by_the_annotator_s_verbatim_quote(tmp_path: Path) -> None:
    text = "Ödeme, ilgili düzenlemenin onuncu hükmü gereğince yapılır."
    quote = "onuncu hükmü gereğince"
    _prepare(tmp_path, "R01", {"d1": (text, [_reference("5520", "10", f"  {quote} ")])})
    write_sealed_release(
        tmp_path, "R01", expert=[{"internal_doc_id": "d1", "text": text, "references": []}], consensus=[]
    )
    documents, audit = round_raw_training_documents(_Cfg(tmp_path), batch_ids=["R01"])
    [reference] = documents["d1"]["references"]
    assert reference["source_text"] == quote
    assert audit["batches"]["R01"]["placed_by_annotator_quote"] == 1
    assert audit["dropped"] == []


def test_a_reference_neither_the_locator_nor_the_quote_can_place_is_dropped_and_named(
    tmp_path: Path,
) -> None:
    text = "Ödeme, ilgili düzenlemenin onuncu hükmü gereğince yapılır. " + LOCATABLE
    _prepare(
        tmp_path,
        "R01",
        {
            "d1": (
                text,
                [_reference("5520", "10", "metinde olmayan alıntı", "Kurumlar Vergisi Kanunu"),
                 _reference("193", "94", "")],
            )
        },
    )
    write_sealed_release(
        tmp_path, "R01", expert=[{"internal_doc_id": "d1", "text": text, "references": []}], consensus=[]
    )
    documents, audit = round_raw_training_documents(_Cfg(tmp_path), batch_ids=["R01"])
    assert [r["kanun_no"] for r in documents["d1"]["references"]] == ["193"]
    assert audit["batches"]["R01"] == {
        "documents_from": "release",
        "documents": 1,
        "references": 2,
        "repeated_rows": 0,
        "placed_by_locator": 1,
        "placed_by_annotator_quote": 0,
        "dropped_unplaceable": 1,
    }
    assert audit["dropped"] == [
        {"batch_id": "R01", "doc_id": "d1", "kanun_no": "5520",
         "kanun_ad": "Kurumlar Vergisi Kanunu", "madde": "10"}
    ]


def test_a_released_text_that_differs_from_the_prepared_text_is_refused(tmp_path: Path) -> None:
    _prepare(tmp_path, "R01", {"d1": (LOCATABLE, [])})
    write_sealed_release(
        tmp_path, "R01", expert=[{"internal_doc_id": "d1", "text": "başka", "references": []}], consensus=[]
    )
    with pytest.raises(ContractError, match="d1"):
        round_raw_training_documents(_Cfg(tmp_path), batch_ids=["R01"])


def test_a_released_document_missing_from_the_batch_is_refused(tmp_path: Path) -> None:
    _prepare(tmp_path, "R01", {"d1": (LOCATABLE, [])})
    write_sealed_release(
        tmp_path,
        "R01",
        expert=[{"internal_doc_id": "d1", "text": LOCATABLE, "references": []},
                {"internal_doc_id": "d9", "text": "dokuz", "references": []}],
        consensus=[],
    )
    with pytest.raises(ContractError, match="d9"):
        round_raw_training_documents(_Cfg(tmp_path), batch_ids=["R01"])


def test_reading_the_raw_labels_never_writes_the_loop_database(tmp_path: Path) -> None:
    _prepare(tmp_path, "R01", {"d1": (LOCATABLE, [_reference("193", "94", "")])})
    write_sealed_release(
        tmp_path, "R01", expert=[{"internal_doc_id": "d1", "text": LOCATABLE, "references": []}], consensus=[]
    )
    database = tmp_path / "dqcheck.sqlite3"
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    round_raw_training_documents(_Cfg(tmp_path), batch_ids=["R01"])
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    assert not (tmp_path / "dqcheck.sqlite3-wal").exists() or (tmp_path / "dqcheck.sqlite3-wal").stat().st_size == 0


def test_the_raw_arm_never_reads_the_cleaned_references(tmp_path: Path) -> None:
    """A reference only the expert added stays out of the raw arm."""
    _prepare(tmp_path, "R01", {"d1": (LOCATABLE, [])})
    write_sealed_release(
        tmp_path,
        "R01",
        expert=[{"internal_doc_id": "d1", "text": LOCATABLE,
                 "references": [_reference("193", "94", "")]}],
        consensus=[],
    )
    documents, _ = round_raw_training_documents(_Cfg(tmp_path), batch_ids=["R01"])
    assert documents["d1"]["references"] == []
    assert json.dumps(documents, ensure_ascii=False).count("193") == 1  # the text only


# --- train_round: the raw arm is its own run, never the clean one -------------------------

SPLIT = {"train": [1, 2, 3], "valid": [10, 11], "test": [20, 21]}


def _round1(config) -> None:
    root = Path(config.sensitive_root)
    _prepare(root, "round1", {"d1": (LOCATABLE, [_reference("193", "94", "")]), "d2": ("iki", [])})
    write_sealed_release(
        root,
        "round1",
        [{"internal_doc_id": "d1", "text": LOCATABLE, "references": []}],
        [{"internal_doc_id": "d2", "text": "iki", "references": []}],
    )


def test_the_raw_arm_is_a_different_run_on_the_same_documents(tmp_path: Path) -> None:
    config = canonical_fixture(tmp_path)
    _round1(config)
    clean = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"])
    raw = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"], labels="raw")
    assert raw["run_id"] != clean["run_id"]
    assert raw["training_documents"] == clean["training_documents"] == 5
    assert raw["label_source"] == "raw" and clean["label_source"] == "clean"
    assert raw["raw_label_audit"]["round1"]["placed_by_locator"] == 1


def test_the_raw_run_records_its_label_source_and_audit(tmp_path: Path) -> None:
    config = canonical_fixture(tmp_path)
    _round1(config)
    raw = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"], labels="raw")
    run_config = json.loads(Path(raw["run_config_path"]).read_text(encoding="utf-8"))
    assert run_config["label_source"] == "raw"
    audit = json.loads((Path(raw["data_dir"]) / "raw_label_audit.json").read_text(encoding="utf-8"))
    assert audit["label_source"] == "raw" and audit["batches"]["round1"]["documents"] == 2
    train_rows = (Path(raw["data_dir"]) / "train.jsonl").read_text(encoding="utf-8")
    assert '\\"kanun_no\\":\\"193\\"' in train_rows


def test_the_clean_run_carries_no_label_source(tmp_path: Path) -> None:
    """Existing run ids (M000, R01's M001) must not move: nothing new enters a clean run."""
    config = canonical_fixture(tmp_path)
    _round1(config)
    default = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"])
    explicit = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"], labels="clean")
    assert default["run_id"] == explicit["run_id"]
    run_config = json.loads(Path(default["run_config_path"]).read_text(encoding="utf-8"))
    assert "label_source" not in run_config


def test_the_raw_arm_needs_a_cleaned_round(tmp_path: Path) -> None:
    config = canonical_fixture(tmp_path)
    with pytest.raises(ContractError, match="raw"):
        train_round(config, generation="M000", split=SPLIT, cleaned_batch_ids=[], labels="raw")


def test_an_unknown_label_source_is_refused(tmp_path: Path) -> None:
    config = canonical_fixture(tmp_path)
    with pytest.raises(ContractError):
        train_round(config, generation="M000", split=SPLIT, cleaned_batch_ids=[], labels="noisy")


# --- before the release: the raw arm needs no expert decision --------------------------------
# The loop's release trains on every document of its batch: RED and QUARANTINE documents are
# required reviews (hitl._review_requirements), a deferred one blocks the release, and GREEN
# ones are either reviewed or released as consensus. So the raw arm can train before the
# release on the whole batch; `check_raw_candidate_against_release` compares afterwards.


def test_before_the_release_the_raw_arm_reads_every_document_of_the_batch(tmp_path: Path) -> None:
    _prepare(
        tmp_path,
        "R01",
        {"d2": ("iki", []), "d1": (LOCATABLE, [_reference("193", "94", "")]), "dQ": ("uc", [])},
    )
    documents, audit = round_raw_training_documents(_Cfg(tmp_path), batch_ids=["R01"])
    assert list(documents) == ["d1", "d2", "dQ"]
    assert audit["batches"]["R01"]["documents_from"] == "batch"
    assert audit["documents_by_batch"] == {"R01": ["d1", "d2", "dQ"]}


def test_with_a_release_the_raw_arm_reads_the_release_set_in_id_order(tmp_path: Path) -> None:
    _prepare(tmp_path, "R01", {"d1": ("bir", []), "d2": ("iki", [])})
    write_sealed_release(
        tmp_path,
        "R01",
        expert=[{"internal_doc_id": "d2", "text": "iki", "references": []}],
        consensus=[{"internal_doc_id": "d1", "text": "bir", "references": []}],
    )
    documents, audit = round_raw_training_documents(_Cfg(tmp_path), batch_ids=["R01"])
    assert list(documents) == ["d1", "d2"]
    assert audit["batches"]["R01"]["documents_from"] == "release"


def test_an_empty_batch_is_refused(tmp_path: Path) -> None:
    _prepare(tmp_path, "R01", {"d1": ("bir", [])})
    with pytest.raises(ContractError, match="R02"):
        round_raw_training_documents(_Cfg(tmp_path), batch_ids=["R02"])


def test_a_raw_run_before_the_release_is_the_run_after_it_when_the_documents_agree(tmp_path: Path) -> None:
    config = canonical_fixture(tmp_path)
    root = Path(config.sensitive_root)
    _prepare(root, "round1", {"d1": (LOCATABLE, [_reference("193", "94", "")]), "d2": ("iki", [])})
    before = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"], labels="raw")
    write_sealed_release(
        root,
        "round1",
        [{"internal_doc_id": "d2", "text": "iki", "references": []}],
        [{"internal_doc_id": "d1", "text": LOCATABLE, "references": []}],
    )
    after = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"], labels="raw")
    assert before["run_id"] == after["run_id"]
    assert before["training_documents"] == 5


def test_a_release_with_other_documents_is_another_raw_run(tmp_path: Path) -> None:
    config = canonical_fixture(tmp_path)
    root = Path(config.sensitive_root)
    _prepare(root, "round1", {"d1": (LOCATABLE, []), "d2": ("iki", []), "d3": ("uc", [])})
    before = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"], labels="raw")
    write_sealed_release(
        root,
        "round1",
        [{"internal_doc_id": "d2", "text": "iki", "references": []}],
        [{"internal_doc_id": "d3", "text": "uc", "references": []},
         {"internal_doc_id": "d4", "text": "dort", "references": []}],
    )
    with Store(root / "dqcheck.sqlite3") as store:
        store.add_document("round1", {"internal_doc_id": "d4", "public_doc_id": "p9", "raw_document_id": "r9",
                                      "selected_channel": "html", "pdf_coverage": 1.0, "html_coverage": 1.0,
                                      "text": "dort", "text_sha256": "x", "human_references": []})
    after = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"], labels="raw")
    assert before["training_documents"] == 6 and after["training_documents"] == 6
    assert before["run_id"] != after["run_id"]


# --- each reference once (5 Oct 2026) ---------------------------------------------------------


def test_the_raw_arm_trains_each_reference_once(tmp_path: Path) -> None:
    _prepare(tmp_path, "R01", {"d1": (LOCATABLE, [_reference("193", "94", ""), _reference("193", "94", ""),
                                                  _reference("193", "", "")])})
    documents, audit = round_raw_training_documents(_Cfg(tmp_path), batch_ids=["R01"])
    assert [(r["kanun_no"], r["madde"]) for r in documents["d1"]["references"]] == [("193", "94")]
    assert audit["batches"]["R01"]["references"] == 3
    assert audit["batches"]["R01"]["repeated_rows"] == 2
    assert audit["batches"]["R01"]["placed_by_locator"] == 1


def test_any_quote_of_a_repeated_reference_may_place_it(tmp_path: Path) -> None:
    text = "Ödeme, ilgili düzenlemenin onuncu hükmü gereğince yapılır."
    _prepare(tmp_path, "R01", {"d1": (text, [_reference("5520", "10", "metinde yok"),
                                             _reference("5520", "10", "onuncu hükmü")])})
    documents, audit = round_raw_training_documents(_Cfg(tmp_path), batch_ids=["R01"])
    [reference] = documents["d1"]["references"]
    assert reference["source_text"] == "onuncu hükmü"
    assert audit["batches"]["R01"]["placed_by_annotator_quote"] == 1


def test_a_raw_reference_carries_every_mention_as_a_quote(tmp_path: Path) -> None:
    text = ("193 sayılı Gelir Vergisi Kanununun 94 üncü maddesinde tevkifat düzenlenmiştir. "
            "Ödemeler bakımından aynı Kanunun 94 üncü maddesi uygulanır.")
    _prepare(tmp_path, "R01", {"d1": (text, [_reference("193", "94", "")])})
    documents, audit = round_raw_training_documents(_Cfg(tmp_path), batch_ids=["R01"])
    [reference] = documents["d1"]["references"]
    assert len(reference["source_texts"]) == 2
    assert audit["placement_rule"] == "locator_mentions_then_annotator_verbatim_quote_v2"


def test_a_raw_range_is_trained_as_its_articles(tmp_path: Path) -> None:
    text = "6102 sayılı Türk Ticaret Kanununun 473 ila 475 inci maddeleri uygulanır."
    _prepare(tmp_path, "R01", {"d1": (text, [_reference("6102", "473 ila 475", "")])})
    documents, audit = round_raw_training_documents(_Cfg(tmp_path), batch_ids=["R01"])
    assert [r["madde"] for r in documents["d1"]["references"]] == ["473", "474", "475"]


def test_gold_ranges_are_expanded_from_round_one_but_not_in_m000(tmp_path: Path) -> None:
    """M000 is sealed and keeps its run id; M001 on trains gold ranges as articles."""
    config = canonical_fixture(tmp_path)
    m000 = train_round(config, generation="M000", split=SPLIT, cleaned_batch_ids=[])
    assert "article_ranges" not in json.loads(Path(m000["run_config_path"]).read_text(encoding="utf-8"))
    _round1(config)
    m001 = train_round(config, generation="M001", split=SPLIT, cleaned_batch_ids=["round1"])
    assert json.loads(Path(m001["run_config_path"]).read_text(encoding="utf-8"))["article_ranges"] == "expanded_v1"
