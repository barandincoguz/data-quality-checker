from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_quality_checker.errors import ContractError
from data_quality_checker.loop_training import (
    compose_round_training_ids,
    round_training_documents,
    write_round_training_manifest,
)

SPLIT = {"train": [1, 2, 3], "valid": [10, 11], "test": [20, 21, 22]}


def test_round_zero_is_canonical_train_only() -> None:
    composition = compose_round_training_ids(SPLIT, [])
    assert composition["train"] == [1, 2, 3]
    assert composition["valid"] == [10, 11]
    assert composition["test"] == [20, 21, 22]


def test_each_round_appends_its_cleaned_batch() -> None:
    composition = compose_round_training_ids(SPLIT, [["dA", "dB"], ["dC"]])
    assert composition["train"] == [1, 2, 3, "dA", "dB", "dC"]


def test_validation_and_test_never_grow() -> None:
    base = compose_round_training_ids(SPLIT, [])
    grown = compose_round_training_ids(SPLIT, [["dA"], ["dB"], ["dC"]])
    assert grown["valid"] == base["valid"]
    assert grown["test"] == base["test"]


def test_a_cleaned_document_may_not_enter_twice() -> None:
    with pytest.raises(ContractError):
        compose_round_training_ids(SPLIT, [["dA"], ["dA"]])


def test_a_cleaned_document_may_not_read_as_a_canonical_id() -> None:
    """A canonical document is identified by its integer id, so a cleaned
    document whose id is a bare number would be indistinguishable from one --
    and would send the evaluator to the wrong ground-truth file.
    """
    for offender in ("1", "0007", 1, " 42 ", "-3"):
        with pytest.raises(ContractError):
            compose_round_training_ids(SPLIT, [[offender]])


def test_a_canonical_document_stays_an_integer_so_ground_truth_resolves() -> None:
    """Pin the identity itself: the evaluator reads `doc_<id>.json` and matches
    predictions by this key, so any decoration here would silently evaluate the
    round against nothing.
    """
    composition = compose_round_training_ids(SPLIT, [["dA"]])
    assert all(isinstance(doc_id, int) for doc_id in composition["valid"])
    assert all(isinstance(doc_id, int) for doc_id in composition["test"])
    assert composition["train"][:3] == [1, 2, 3]


def test_manifest_records_the_round_and_the_split_seal(tmp_path: Path) -> None:
    composition = compose_round_training_ids(SPLIT, [["dA", "dB"]])
    result = write_round_training_manifest(tmp_path, 1, composition, split_manifest_sha256="abc123")
    payload = json.loads((tmp_path / "round_001_training.json").read_text(encoding="utf-8"))
    assert payload["round"] == 1
    assert payload["split_manifest_sha256"] == "abc123"
    assert payload["batch_manifest_sha256"] is None
    assert payload["counts"] == {"train": 5, "valid": 2, "test": 3}
    assert result["manifest_sha256"]


def test_manifest_schema_version_is_2(tmp_path: Path) -> None:
    """The payload gained `batch_manifest_sha256` without a version bump once;
    pin the corrected version so a future shape change is caught the same way.
    """
    composition = compose_round_training_ids(SPLIT, [["dA", "dB"]])
    write_round_training_manifest(tmp_path, 1, composition, split_manifest_sha256="abc123")
    payload = json.loads((tmp_path / "round_001_training.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2


def test_manifest_records_the_batch_manifest_seal_when_given(tmp_path: Path) -> None:
    composition = compose_round_training_ids(SPLIT, [["dA", "dB"]])
    write_round_training_manifest(
        tmp_path,
        1,
        composition,
        split_manifest_sha256="abc123",
        batch_manifest_sha256="def456",
    )
    payload = json.loads((tmp_path / "round_001_training.json").read_text(encoding="utf-8"))
    assert payload["batch_manifest_sha256"] == "def456"


def write_sealed_release(
    root: Path,
    batch_id: str,
    expert: list[dict],
    consensus: list[dict],
    quarantine: list[dict] = (),
    *,
    raw_expert: str | None = None,
) -> Path:
    """A release in the layout `release_batch` writes: sealed, one directory.

    `root` is the sensitive root. `raw_expert` replaces the expert file's text
    (still sealed), to put a malformed line inside an otherwise valid seal.
    """
    from data_quality_checker.fingerprints import sha256_file
    from data_quality_checker.release import LAYER_FILES

    directory = root / "releases" / batch_id / "release_fixture"
    directory.mkdir(parents=True, exist_ok=True)
    as_text = lambda rows: "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)  # noqa: E731
    texts = {
        "expert_adjudicated.jsonl": as_text(expert) if raw_expert is None else raw_expert,
        "consensus_clean.jsonl": as_text(consensus),
        "quarantine.jsonl": as_text(quarantine),
        "training_export.jsonl": as_text([*expert, *consensus]),
    }
    for name, text in texts.items():
        (directory / name).write_text(text, encoding="utf-8")
    files = [
        {
            "path": name,
            "size": (directory / name).stat().st_size,
            "sha256": sha256_file(directory / name),
        }
        for name in LAYER_FILES
    ]
    documents = len(expert) + len(consensus) + len(quarantine)
    manifest = {"batch_id": batch_id, "counts": {"documents": documents}, "files": files}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(directory / n)}  {n}\n" for n in (*LAYER_FILES, "manifest.json")),
        encoding="utf-8",
    )
    return directory


def _release(root: Path, batch_id: str, expert: list[dict], consensus: list[dict]) -> None:
    write_sealed_release(root, batch_id, expert, consensus)


class _Cfg:
    def __init__(self, root: Path) -> None:
        self.sensitive_root = root


def test_reads_both_trust_levels(tmp_path: Path) -> None:
    # The annotator's quote ("bir") is replaced by the located citation: it
    # decides nothing (27 Sep 2026); the locator is tested in test_reference_locator.py.
    text = "193 sayılı Gelir Vergisi Kanununun 94 üncü maddesi uygulanır."
    _release(
        tmp_path,
        "round1",
        expert=[
            {
                "internal_doc_id": "d1",
                "text": text,
                "references": [{"kanun_no": "193", "madde": "94", "source_text": "bir"}],
            }
        ],
        consensus=[{"internal_doc_id": "d2", "text": "iki", "references": []}],
    )
    documents = round_training_documents(_Cfg(tmp_path), batch_ids=["round1"])
    assert set(documents) == {"d1", "d2"}
    assert documents["d1"]["text"] == text
    [reference] = documents["d1"]["references"]
    assert reference["madde"] == "94" and reference["source_text"] in text
    assert reference["source_text"] != "bir"


def test_reads_several_rounds(tmp_path: Path) -> None:
    _release(
        tmp_path,
        "round1",
        expert=[{"internal_doc_id": "d1", "text": "bir", "references": []}],
        consensus=[],
    )
    _release(
        tmp_path,
        "round2",
        expert=[{"internal_doc_id": "d2", "text": "iki", "references": []}],
        consensus=[],
    )
    documents = round_training_documents(_Cfg(tmp_path), batch_ids=["round1", "round2"])
    assert set(documents) == {'["round1","d1"]', '["round2","d2"]'}


def test_a_missing_release_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ContractError):
        round_training_documents(_Cfg(tmp_path), batch_ids=["never-released"])


def test_the_same_document_in_two_rounds_is_rejected(tmp_path: Path) -> None:
    _release(
        tmp_path,
        "round1",
        expert=[{"internal_doc_id": "d1", "text": "bir", "references": []}],
        consensus=[],
    )
    _release(
        tmp_path,
        "round2",
        expert=[{"internal_doc_id": "d1", "text": "bir", "references": []}],
        consensus=[],
    )
    with pytest.raises(ContractError):
        round_training_documents(_Cfg(tmp_path), batch_ids=["round1", "round2"])


def test_a_row_missing_text_or_references_is_rejected(tmp_path: Path) -> None:
    _release(tmp_path, "round1", expert=[{"internal_doc_id": "d1", "references": []}], consensus=[])
    with pytest.raises(ContractError):
        round_training_documents(_Cfg(tmp_path), batch_ids=["round1"])


def test_quarantined_documents_are_never_read(tmp_path: Path) -> None:
    """The reader names its two trustworthy files. A quarantined document sitting
    in the same release directory must never surface in the returned mapping --
    even a later refactor toward something like ``glob("*.jsonl")`` must not be
    able to sneak it in silently.
    """
    write_sealed_release(
        tmp_path,
        "round1",
        expert=[{"internal_doc_id": "d1", "text": "bir", "references": []}],
        consensus=[{"internal_doc_id": "d2", "text": "iki", "references": []}],
        quarantine=[{"internal_doc_id": "dQ", "text": "uc", "references": []}],
    )
    documents = round_training_documents(_Cfg(tmp_path), batch_ids=["round1"])
    assert set(documents) == {"d1", "d2"}
    assert "dQ" not in documents


def test_a_malformed_line_is_rejected_naming_file_and_line_number(tmp_path: Path) -> None:
    from data_quality_checker.errors import IntegrityError

    write_sealed_release(
        tmp_path,
        "round1",
        expert=[{"internal_doc_id": "d1", "text": "bir", "references": []}],
        consensus=[],
        raw_expert=json.dumps({"internal_doc_id": "d1", "text": "bir", "references": []})
        + "\n{not json\n",
    )
    with pytest.raises(IntegrityError) as excinfo:
        round_training_documents(_Cfg(tmp_path), batch_ids=["round1"])
    message = str(excinfo.value)
    assert "expert_adjudicated.jsonl" in message
    assert ":2" in message


def test_an_empty_release_is_rejected_naming_the_batch(tmp_path: Path) -> None:
    _release(tmp_path, "round1", expert=[], consensus=[])
    with pytest.raises(ContractError) as excinfo:
        round_training_documents(_Cfg(tmp_path), batch_ids=["round1"])
    assert "round1" in str(excinfo.value)


def test_an_annotator_quote_never_places_a_released_reference(tmp_path: Path) -> None:
    """User rule (27 Sep 2026): the annotator's source_text decides nothing -- not
    even where a reference sits in a training window. The locator places every
    released reference; the release itself keeps what was written."""
    text = "5520 sayılı Kurumlar Vergisi Kanununun 10 uncu maddesinde indirimler sayılmıştır."
    write_sealed_release(
        tmp_path,
        "round1",
        expert=[
            {
                "internal_doc_id": "d1",
                "text": text,
                "references": [
                    {
                        "kanun_no": "5520",
                        "kanun_ad": "Kurumlar Vergisi Kanunu",
                        "madde": "10",
                        "fikra": "",
                        "bent": "",
                        "source_text": "bursiyerin metinde olmayan alıntısı",
                    }
                ],
            }
        ],
        consensus=[],
    )
    documents = round_training_documents(_Cfg(tmp_path), batch_ids=["round1"])
    placed = documents["d1"]["references"][0]["source_text"]
    assert placed in text and "10 uncu maddesinde" in placed


def test_a_repeated_released_reference_is_trained_once(tmp_path: Path) -> None:
    """5 Oct 2026: a released list may repeat a reference (an annotator's rows that
    differed only in fıkra or bent). The target says it once, as the gold training
    data does; the release on disk keeps what was written."""
    text = "193 sayılı Gelir Vergisi Kanununun 94 üncü maddesi uygulanır."
    rows = [
        {"kanun_no": "193", "kanun_ad": "", "madde": "94", "source_text": ""},
        {"kanun_no": "193", "kanun_ad": "Gelir Vergisi Kanunu", "madde": "94", "source_text": ""},
        {"kanun_no": "193", "kanun_ad": "", "madde": "", "source_text": ""},
    ]
    _release(
        tmp_path,
        "round1",
        expert=[{"internal_doc_id": "d1", "text": text, "references": rows}],
        consensus=[],
    )
    documents = round_training_documents(_Cfg(tmp_path), batch_ids=["round1"])
    assert [(r["kanun_no"], r["madde"]) for r in documents["d1"]["references"]] == [("193", "94")]


def test_a_released_reference_carries_every_mention_as_a_quote(tmp_path: Path) -> None:
    """6 Oct 2026 (user: round documents only): a window citing the reference a
    second time lists it too, as gold documents' every-quote placement does."""
    text = (
        "193 sayılı Gelir Vergisi Kanununun 94 üncü maddesinde tevkifat düzenlenmiştir. "
        "Ödemeler bakımından aynı Kanunun 94 üncü maddesi uygulanır."
    )
    _release(
        tmp_path,
        "round1",
        expert=[
            {
                "internal_doc_id": "d1",
                "text": text,
                "references": [
                    {"kanun_no": "193", "kanun_ad": "", "madde": "94", "source_text": ""}
                ],
            }
        ],
        consensus=[],
    )
    [reference] = round_training_documents(_Cfg(tmp_path), batch_ids=["round1"])["d1"]["references"]
    assert (
        len(reference["source_texts"]) == 2
        and reference["source_text"] == reference["source_texts"][0]
    )


def test_a_released_range_is_trained_as_its_articles(tmp_path: Path) -> None:
    """User decision, 7 Oct 2026: "473 ila 475" is 473, 474 and 475, each its own row."""
    text = "6102 sayılı Türk Ticaret Kanununun 473 ila 475 inci maddeleri uygulanır."
    _release(
        tmp_path,
        "round1",
        expert=[
            {
                "internal_doc_id": "d1",
                "text": text,
                "references": [
                    {"kanun_no": "6102", "kanun_ad": "", "madde": "473 ila 475", "source_text": ""}
                ],
            }
        ],
        consensus=[],
    )
    references = round_training_documents(_Cfg(tmp_path), batch_ids=["round1"])["d1"]["references"]
    assert [r["madde"] for r in references] == ["473", "474", "475"]
    assert all("expanded_from" not in r for r in references)
