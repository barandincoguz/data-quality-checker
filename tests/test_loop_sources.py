import json
from pathlib import Path

import pytest

from data_quality_checker import loop_sources
from data_quality_checker.config import load_config
from data_quality_checker.constants import EXEMPLAR_DOC_IDS
from data_quality_checker.errors import ContractError, IntegrityError
from data_quality_checker.fingerprints import sha256_file
from data_quality_checker.g0 import repository_manifest_sha256

REPO = Path(__file__).resolve().parents[1]
PRESET = REPO / "configs" / "presets" / "gt750_loop.json"


N_DOCS = 80  # every exemplar id (max 77) must exist in the fixture universe


def _write_universe(root: Path, n: int = N_DOCS) -> tuple[Path, Path]:
    gt = root / "validated"
    gt.mkdir(parents=True)
    ids = list(range(1, n + 1))
    for doc_id in ids:
        (gt / f"doc_{doc_id}.json").write_text(json.dumps({
            "doc_id": doc_id, "ozelge_no": "", "text": f"metin {doc_id}",
            "references": []}, ensure_ascii=False))
    held = [i for i in ids if i not in EXEMPLAR_DOC_IDS][:20]
    split = {"train": [i for i in ids if i not in held], "test": held[:10], "val": held[10:]}
    split_path = root / "SPLIT.json"
    split_path.write_text(json.dumps(split))
    return gt, split_path


def _config(tmp_path: Path, gt: Path, split_path: Path, **loop_overrides):
    raw = json.loads(PRESET.read_text())
    raw["canonical_gt_dir"] = str(gt)
    raw["example_bank_path"] = str(REPO / "benchmark/reference/example_sets/few_shot_cot_v3_en_examples.json")
    raw["sensitive_root"] = str(tmp_path / "sensitive")
    raw["public_root"] = str(tmp_path / "public")
    raw["training_runs_root"] = str(tmp_path / "runs")
    split = json.loads(split_path.read_text())
    raw["loop"] = {
        "split_path": str(split_path),
        "expected_documents": N_DOCS,
        "sizes": {"train": len(split["train"]), "valid": len(split["val"]), "test": len(split["test"])},
        "gt_manifest_sha256": repository_manifest_sha256(gt),
        "split_sha256": sha256_file(split_path),
        **loop_overrides,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    return load_config(path)


def test_sources_read_the_sealed_split(tmp_path):
    gt, split_path = _write_universe(tmp_path)
    config = _config(tmp_path, gt, split_path)

    source = loop_sources.validate_loop_sources(config)

    assert set(source["documents"]) == set(range(1, N_DOCS + 1))
    assert source["split"]["valid"] == json.loads(split_path.read_text())["val"]
    assert source["summary"]["split_manifest_sha256"] == sha256_file(split_path)
    assert loop_sources.has_loop_settings(config)


def test_an_edited_ground_truth_is_refused(tmp_path):
    gt, split_path = _write_universe(tmp_path)
    config = _config(tmp_path, gt, split_path)
    (gt / "doc_3.json").write_text(json.dumps({"doc_id": 3, "ozelge_no": "", "text": "değişti", "references": []}))

    with pytest.raises(IntegrityError):
        loop_sources.validate_loop_sources(config)


def test_an_edited_split_is_refused(tmp_path):
    gt, split_path = _write_universe(tmp_path)
    config = _config(tmp_path, gt, split_path)
    split = json.loads(split_path.read_text())
    split["test"], split["val"] = split["val"], split["test"]
    split_path.write_text(json.dumps(split))

    with pytest.raises(IntegrityError):
        loop_sources.validate_loop_sources(config)


def test_an_exemplar_outside_train_is_refused(tmp_path):
    gt, split_path = _write_universe(tmp_path)
    split = json.loads(split_path.read_text())
    split["train"].remove(10)
    split["test"].append(10)
    split_path.write_text(json.dumps(split))
    config = _config(tmp_path, gt, split_path)

    with pytest.raises(ContractError, match="exemplar"):
        loop_sources.validate_loop_sources(config)


def test_the_preset_carries_the_real_seals():
    config = load_config(PRESET)

    source = loop_sources.validate_loop_sources(config)

    assert source["summary"]["split_counts"] == {"train": 500, "valid": 50, "test": 200}
    assert loop_sources.loop_split_policy(config) == "loop-gt750-500-50-200"


LAW_ARTICLE_PRESET = REPO / "configs" / "presets" / "gt750_law_article_loop.json"


def test_the_law_article_loop_reads_gt_v7_with_every_quote_and_no_paragraph_field():
    """The loop since 1 Oct 2026: GT v7, sealed; the same SPLIT.json as v6;
    every reference a law and an article, each carrying all its quotes."""
    config = load_config(LAW_ARTICLE_PRESET)
    old = json.loads(PRESET.read_text())["loop"]
    assert config.raw["loop"]["split_sha256"] == old["split_sha256"]
    source = loop_sources.validate_loop_sources(config)
    references = [r for doc in source["documents"].values() for r in doc["references"]]
    assert source["summary"]["canonical_reference_count"] == 3561 == len(references)
    assert all(set(r) <= {"kanun_no", "kanun_ad", "madde", "source_text", "source_texts"} for r in references)
    assert all(r["source_texts"][0] == r["source_text"] for r in references if "source_texts" in r)
    assert sum(len(r.get("source_texts", [])) > 1 for r in references) > 0


def test_a_malformed_quote_list_is_refused(tmp_path):
    gt, split_path = _write_universe(tmp_path)
    path = gt / "doc_1.json"
    payload = json.loads(path.read_text())
    payload["references"] = [{"kanun_no": "193", "madde": "94", "source_text": "a", "source_texts": [""]}]
    path.write_text(json.dumps(payload))
    with pytest.raises(ContractError, match="source_texts"):
        loop_sources.validate_loop_sources(_config(tmp_path, gt, split_path))
