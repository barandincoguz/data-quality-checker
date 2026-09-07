from __future__ import annotations

import pytest

from data_quality_checker.cli import build_parser


@pytest.mark.parametrize(
    ("command", "argv"),
    [
        ("process", ["process", "--prepared-batch", "M01", "--generation", "M000"]),
        ("reroute", ["reroute", "--batch-id", "M01", "--generation", "M000"]),
        ("serve", ["serve", "--batch-id", "M01", "--generation", "M000"]),
    ],
)
def test_a_round_model_can_be_named_on_the_command_line(command: str, argv: list[str]) -> None:
    """Every DQ-Loop round predicts, reroutes and is adjudicated with its own
    model. `process_batch` and `reroute_batch` have always accepted a round
    label; only the parser refused one, which would have forced round 1 to be
    cleaned by G0 -- the design violation this loop exists to avoid.
    """
    args = build_parser().parse_args(argv)
    assert args.generation == "M000"


@pytest.mark.parametrize(
    "argv",
    [
        ["process", "--prepared-batch", "M01", "--generation", "M1"],
        ["process", "--prepared-batch", "M01", "--generation", "g0"],
        ["reroute", "--batch-id", "M01", "--generation", "M0001"],
        ["reroute", "--batch-id", "M01", "--generation", "round1"],
    ],
)
def test_a_generation_that_is_neither_g0_nor_a_round_label_is_refused(argv: list[str]) -> None:
    """A typo here silently reads another model's predictions -- or none --
    so it must fail at the parser rather than halfway through a batch.
    """
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


@pytest.mark.parametrize("bad", ["round1", "M1", "m000", ""])
def test_serve_refuses_a_generation_the_expert_could_not_be_shown(bad: str) -> None:
    """`serve` decides which model's output the expert adjudicates. It used to
    take any string, so a typo would have shown an empty review queue -- or
    silently the wrong generation -- with no error anywhere.
    """
    with pytest.raises(SystemExit):
        build_parser().parse_args(["serve", "--batch-id", "M01", "--generation", bad])


def test_the_judge_can_be_pointed_at_a_round_s_predictions() -> None:
    """`run_judge_pilot` has always taken a generation; the parser offered none,
    so a round's judge would have compared the annotator against G0's output.
    """
    args = build_parser().parse_args(["pilot-judges", "--batch-id", "M01", "--generation", "M000"])
    assert args.generation == "M000"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["pilot-judges", "--batch-id", "M01", "--generation", "M1"])


LOOP_COMMANDS = {
    "process": ["process", "--prepared-batch", "M01"],
    "reroute": ["reroute", "--batch-id", "M01"],
    "serve": ["serve", "--batch-id", "M01"],
    "pilot-judges": ["pilot-judges", "--batch-id", "M01"],
}


@pytest.mark.parametrize("command", sorted(LOOP_COMMANDS))
def test_a_loop_command_will_not_guess_the_generation(command: str) -> None:
    """No default. Which model a round uses is a fact only the caller knows.

    `train_step` states the rule in its own docstring -- "every default guessed
    on this line of work produced a defect elsewhere in the pipeline: a round
    predicting with the wrong model, a judge reading the wrong generation, an
    evaluator gating the wrong split size" -- and all three of those were live
    bugs. A "G0" default is the same hazard in its quietest form: omit the flag
    on `serve` and the expert adjudicates the bootstrap model's output, and
    their decisions become the next round's training data. Nothing errors.
    """
    with pytest.raises(SystemExit):
        build_parser().parse_args(LOOP_COMMANDS[command])


@pytest.mark.parametrize("command", sorted(LOOP_COMMANDS))
def test_a_loop_command_takes_the_round_it_is_told(command: str) -> None:
    args = build_parser().parse_args([*LOOP_COMMANDS[command], "--generation", "M000"])
    assert args.generation == "M000"


@pytest.mark.parametrize("command", sorted(LOOP_COMMANDS))
def test_g0_may_still_be_named_explicitly(command: str) -> None:
    """The bootstrap model stays reachable -- it just cannot be reached by
    forgetting to say which model you meant.
    """
    args = build_parser().parse_args([*LOOP_COMMANDS[command], "--generation", "G0"])
    assert args.generation == "G0"


def test_train_bootstrap_trains_only_g0() -> None:
    """`train_bootstrap` refuses anything but G0, so the flag says so."""
    assert build_parser().parse_args(["train-bootstrap"]).generation == "G0"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train-bootstrap", "--generation", "M000"])
