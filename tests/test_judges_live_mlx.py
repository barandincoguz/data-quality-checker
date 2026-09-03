"""Live adjudication against the pinned local judge.

Opt-in: this is the only test in the suite that loads model weights (~27 GB
peak) and it takes on the order of two minutes. It exists because every layer
above the provider kept reporting success while the model, with its thinking
channel left on, spent its whole budget reasoning and returned no JSON at all.
Only an end-to-end run catches that class of failure.

Run with: DQCHECK_LIVE_MLX_JUDGE=1 .venv/bin/python -m pytest tests/test_judges_live_mlx.py -q
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from data_quality_checker.constants import JUDGE_MODEL_KEY

pytestmark = pytest.mark.skipif(
    os.environ.get("DQCHECK_LIVE_MLX_JUDGE") != "1",
    reason="set DQCHECK_LIVE_MLX_JUDGE=1 to run the live judge test",
)

GROUND_TRUTH = (
    Path(__file__).resolve().parents[1]
    / "data/ground_truth/gt_v3_triangulated_2026-05-15/validated/doc_266.json"
)


def test_local_judge_returns_a_contract_valid_verdict() -> None:
    from data_quality_checker.judges import MlxJudgeProvider, _validate_judge_result

    document = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    references = document["references"]
    corrupted = [dict(reference) for reference in references[:-1]]
    corrupted[0]["madde"] = "5"

    provider = MlxJudgeProvider()
    content, meta = provider.judge(
        model=JUDGE_MODEL_KEY,
        payload={
            "document_text": document["text"],
            "candidate_a": references,
            "candidate_b": corrupted,
        },
    )
    assert meta["provider"] == "mlx"
    assert meta["finish_reason"] == "stop", "the model ran out of budget mid-verdict"

    result = _validate_judge_result(content, document["text"])
    assert result["verdict"] == "A", "the unperturbed candidate should win"
