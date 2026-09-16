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


def test_every_evaluator_call_in_run_py_passes_the_requested_engine() -> None:
    """A call that omits `matching_engine` silently takes the library default.

    `run.py`'s hard-case block did exactly that: the main-report call passed the
    engine while the hard-case call a hundred lines below did not, so
    `--matching-engine greedy` produced a greedy main report and an optimal
    hard-case report with no error -- and before the default flipped, the bug
    was invisible because the default happened to match the CLI's.

    Parsed rather than grepped, so a future call site is caught too.
    """
    import ast

    source = Path(run_module.__file__).read_text(encoding="utf-8")
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "evaluate_prediction_file"
    ]
    assert calls, "no evaluate_prediction_file call found -- has it been renamed?"
    for call in calls:
        keywords = {keyword.arg for keyword in call.keywords}
        assert "matching_engine" in keywords, f"run.py:{call.lineno} omits matching_engine"
