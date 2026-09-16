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

from benchmark.reference import evaluate as evaluate_module
from benchmark.reference import run as run_module


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
    source = inspect.getsource(evaluate_module)
    assert 'result.get("matching_engine", MATCHING_ENGINE_GREEDY)' in source


def test_both_cli_defaults_are_optimal() -> None:
    for module in (evaluate_module, run_module):
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
