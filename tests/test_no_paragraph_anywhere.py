"""Nothing below the article, anywhere (user decision, 1 Oct 2026).

"fıkra ve bent işimizden tamamen kalkıyor ... kesinlikle ve kesinlikle artık fıkra
ve benti ne LLM'lerden ne LLM-as-judge'dan ne de evaluator'dan ya da herhangi bir
eşleme motorundan geçmeyecek." A reference is a law and an article of it.

This file is the contamination guard: every surface a model reads, every field a
decision rests on, and every engine that compares references is checked here, so
the next change that brings a paragraph or sub-item back fails a test instead of
quietly reaching a model or a metric. The locator still *reads* the words in a
ruling's text ("94 üncü maddesinin 1 inci fıkrası" cites article 94) -- that is
parsing the document, not a field -- and is not scanned for them.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_IN_PROMPTS = re.compile(
    r"f[iı]kra|\bbent|paragraph|sub-?part|sub-?item|clause|alt\s*bent", re.IGNORECASE
)
DECISION_FIELDS = ("kanun_no", "kanun_ad", "madde")


def test_no_prompt_a_model_reads_names_a_unit_below_the_article() -> None:
    from data_quality_checker.g0 import SYSTEM_PROMPT
    from data_quality_checker.judges import JUDGE_PROMPT

    for name, prompt in (("SYSTEM_PROMPT", SYSTEM_PROMPT), ("JUDGE_PROMPT", JUDGE_PROMPT)):
        assert not FORBIDDEN_IN_PROMPTS.search(prompt), (name, FORBIDDEN_IN_PROMPTS.search(prompt))
    assert '"madde":"3"}' in SYSTEM_PROMPT  # the demonstration has the three fields only


def test_every_decision_field_tuple_is_the_law_and_the_article() -> None:
    from annotator.reference.common import CORE_FIELDS, REFERENCE_FIELDS
    from benchmark.reference.evaluate import PRIMARY_METRIC_FIELDS
    from data_quality_checker.constants import REFERENCE_DECISION_FIELDS
    from data_quality_checker.constants import REFERENCE_FIELDS as DQ_FIELDS
    from data_quality_checker.processing import LEGAL_TUPLE_FIELDS

    assert (
        REFERENCE_DECISION_FIELDS
        == DECISION_FIELDS
        == LEGAL_TUPLE_FIELDS
        == CORE_FIELDS
        == PRIMARY_METRIC_FIELDS
    )
    assert DQ_FIELDS == (*DECISION_FIELDS, "source_text")
    assert REFERENCE_FIELDS == ("type", *DECISION_FIELDS, "source_text")


def _code_files() -> list[Path]:
    return [
        *sorted((ROOT / "src" / "data_quality_checker").glob("*.py")),
        *sorted((ROOT / "benchmark" / "reference").glob("*.py")),
        *sorted((ROOT / "annotator" / "reference").glob("*.py")),
    ]


@pytest.mark.parametrize("path", _code_files(), ids=lambda p: p.name)
def test_no_code_reads_or_writes_a_paragraph_field(path: Path) -> None:
    """No string constant `"fikra"` or `"bent"` -- the field names -- in any
    module the loop runs: no key read, written, compared or displayed."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    hits = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value in {"fikra", "bent", "fıkra"}
    ]
    assert not hits, f"{path.name}: paragraph field named at lines {hits}"


def test_the_prompt_and_extractor_copies_that_asked_for_paragraphs_are_gone() -> None:
    present = {p.name for p in (ROOT / "annotator" / "reference").glob("*.py")}
    assert present == {"__init__.py", "common.py"}
    assert not (ROOT / "benchmark" / "reference" / "run.py").exists()


def test_a_stored_paragraph_field_is_projected_away_on_every_path() -> None:
    from annotator.reference.common import normalize_reference as evaluator_normalize
    from benchmark.reference.matching import canonicalize_reference
    from data_quality_checker.contracts import validate_reference_list
    from data_quality_checker.judges import blind_candidates
    from data_quality_checker.normalization import compact_references, normalize_reference

    stored = {
        "kanun_no": "193",
        "kanun_ad": "",
        "madde": "94",
        "fikra": "1",
        "bent": "a",
        "source_text": "x",
    }
    assert set(validate_reference_list([stored])[0]) == {*DECISION_FIELDS, "source_text"}
    assert set(normalize_reference(stored)) == {*DECISION_FIELDS, "source_text"}
    assert set(compact_references([stored])[0]) == {*DECISION_FIELDS, "source_text"}
    assert not {"fikra", "bent"} & set(canonicalize_reference(evaluator_normalize(stored)))
    _, a, b = blind_candidates(
        batch_id="b",
        internal_doc_id="d",
        model="m",
        human_references=[stored],
        model_references=[stored],
    )
    assert all(set(reference) == set(DECISION_FIELDS) for reference in [*a, *b])


def test_the_evaluator_reports_no_paragraph_level(tmp_path: Path) -> None:
    from benchmark.reference.evaluate import evaluate_prediction_file

    gold = {
        1: [
            {
                "kanun_no": "193",
                "kanun_ad": "Gelir Vergisi Kanunu",
                "madde": "94",
                "source_text": "",
            }
        ]
    }
    predictions = tmp_path / "p.json"
    predictions.write_text(
        json.dumps(
            [
                {
                    "doc_id": 1,
                    "status": "success",
                    "references": [
                        {
                            "kanun_no": "193",
                            "kanun_ad": "",
                            "madde": "94/1",
                            "fikra": "2",
                            "bent": "b",
                        }
                    ],
                }
            ]
        )
    )
    result = evaluate_prediction_file(predictions, gold, None, include_per_doc=True)
    assert result["core_law_article_strict"]["f1"] == 1.0
    flat = json.dumps(result)
    for word in ("fikra", "bent", "overall", "per_entity", "soft_extension", "error_bucket"):
        assert word not in flat, word


GT_LOOP = ROOT / "data" / "ground_truth" / "gt_v8_gt750_law_article_2026-10-02" / "validated"


@pytest.mark.skipif(not GT_LOOP.is_dir(), reason="GT v8 is not in this checkout")
def test_the_ground_truth_the_loop_reads_has_no_paragraph_field() -> None:
    preset = json.loads((ROOT / "configs" / "presets" / "gt750_law_article_loop.json").read_text())
    assert "gt_v8_gt750_law_article" in preset["canonical_gt_dir"]
    for path in GT_LOOP.glob("doc_*.json"):
        for reference in json.loads(path.read_text(encoding="utf-8"))["references"]:
            assert not {"fikra", "bent"} & set(reference), path.name
