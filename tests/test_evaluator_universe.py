"""The evaluator scores exactly the documents it is given (2 Oct 2026).

`--docs` defaulted to the old 500-document `data/test_data.json`, so a call that
named its documents with `--doc-ids-file` silently lost every id above 500: the
prompt screen's 100 training documents came back as 70. Validation (ids 1-500)
was never affected; the GT750 test split (100 of its 200 above 500) would have
been. The canonical call names its documents and must get all of them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "data/ground_truth/gt_v7_gt750_law_article_2026-10-01/validated"


@pytest.mark.skipif(not GT.is_dir(), reason="GT v7 is not in this checkout")
def test_documents_above_500_are_scored(tmp_path) -> None:
    from data_quality_checker.config import load_config
    from data_quality_checker.g0_training import canonical_evaluate

    split = json.loads((GT.parent / "SPLIT.json").read_text())
    ids = sorted(int(d) for d in split["train"] if 500 < int(d))[:3] + sorted(int(d) for d in split["train"] if int(d) <= 500)[:3]
    predictions = []
    for doc_id in ids:
        refs = json.loads((GT / f"doc_{doc_id}.json").read_text())["references"]
        predictions.append({"doc_id": doc_id, "status": "success",
                            "references": [{k: r[k] for k in ("kanun_no", "kanun_ad", "madde")} for r in refs]})
    (tmp_path / "p.json").write_text(json.dumps(predictions, ensure_ascii=False))
    (tmp_path / "ids.json").write_text(json.dumps(ids))
    config = load_config(ROOT / "configs/presets/gt750_law_article_loop.json")
    report = canonical_evaluate(config=config, predictions_path=tmp_path / "p.json", doc_ids_path=tmp_path / "ids.json",
                                output_dir=tmp_path / "out", expected_doc_count=len(ids))
    result = report["results"][0]
    assert result["evaluated_doc_count"] == 6
    assert result["core_law_article_strict"]["f1"] == 1.0
