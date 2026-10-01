"""The default matcher must be `optimal`.

This branch's history is the reason. The DQ-Loop called the evaluator without
`--matching-engine`, so every loop metric was produced by `greedy` -- a matcher
the project itself superseded on 2026-08-03. The caller was fixed (`dbd4ccf`),
but the default stayed, so the next caller could fall into the same trap
silently. These tests pin the default itself.

`greedy` is not deleted. The frozen Run 015-021 pairing and Run 023's
paper-facing greedy half must stay reproducible, so it remains selectable by
name -- it just stops being what you get by saying nothing.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from benchmark.reference import evaluate as evaluate_module


def test_library_default_is_optimal() -> None:
    """`evaluate_prediction_file` is the library entry point every caller
    reaches; its default is what a caller gets by saying nothing."""
    signature = inspect.signature(evaluate_module.evaluate_prediction_file)
    assert signature.parameters["matching_engine"].default == "optimal"


def test_an_unspecified_engine_normalises_to_optimal() -> None:
    """A third silent path: normalize_matching_engine fell back to greedy for
    an empty or None engine, so "unspecified" meant the archived matcher."""
    assert evaluate_module.normalize_matching_engine("") == "optimal"
    assert evaluate_module.normalize_matching_engine(None) == "optimal"


def test_a_report_without_an_engine_stamp_is_still_read_back_as_greedy() -> None:
    """Deliberately NOT flipped. A report that records no engine is an old
    report, and an old report really was produced by greedy; saying otherwise
    would relabel history."""
    import re

    source = inspect.getsource(evaluate_module)
    # Both sites, not "somewhere": a plain substring check would pass while one
    # of the two had been flipped. The two spell the key differently -- one is
    # inside an f-string -- so match on the shape, not on one quoting style.
    fallbacks = re.findall(
        r"""result\.get\(\s*['"]matching_engine['"]\s*,\s*MATCHING_ENGINE_GREEDY\s*\)""",
        source,
    )
    assert len(fallbacks) == 2, fallbacks


def test_the_cli_default_is_optimal() -> None:
    # `benchmark/reference/run.py` (the extraction-method runner) left this
    # repository on 1 Oct 2026 with the prompts it drove; it lives on in
    # ner-project. The evaluator's own CLI is what the loop calls.
    for module in (evaluate_module,):
        source = inspect.getsource(module.parse_args)
        index = source.index("--matching-engine")
        window = source[index : index + 400]
        assert "MATCHING_ENGINE_OPTIMAL" in window, module.__name__
        assert "default=MATCHING_ENGINE_GREEDY" not in window, module.__name__


def test_greedy_stays_reachable_by_name() -> None:
    """Archiving is not deleting: a frozen run must stay reproducible."""
    assert "greedy" in evaluate_module.SUPPORTED_MATCHING_ENGINES
    assert evaluate_module.normalize_matching_engine("greedy") == "greedy"


def test_the_controlled_q36_track_is_scored_with_optimal_too() -> None:
    """q36_p1 pinned "greedy" as a bare literal with no recorded reason, while
    the loop's own choice carries a twelve-line justification. The track is a
    new experiment, not a frozen run, and it had produced no artefact, so the
    pin was an inherited default rather than a decision."""
    from data_quality_checker import q36_p1

    source = inspect.getsource(q36_p1)
    assert '"matching_engine": "greedy"' not in source
    assert '"matching_engine": "optimal"' in source
