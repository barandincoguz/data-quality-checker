"""`dqcheck` command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import commands
from .config import default_config_path, load_config
from .errors import DQCheckError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dqcheck",
        description="Crash-resilient data-quality checker and HITL workflow",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="JSON config path (default: %(default)s)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="securely prepare an annotation batch")
    prepare.add_argument("--annotation-zip", type=Path, required=True)
    prepare.add_argument("--document-pool-zip", type=Path, required=True)
    prepare.add_argument("--batch-id")
    prepare.add_argument("--hmac-key-file", type=Path)
    prepare.set_defaults(handler=commands.prepare)

    train = subparsers.add_parser("train-bootstrap", help="build or run canonical-only G0")
    train.add_argument("--generation", choices=["G0"], default="G0")
    train.add_argument(
        "--execute",
        action="store_true",
        help="execute compute after every preflight gate; otherwise emit the locked plan",
    )
    train.set_defaults(handler=commands.train_bootstrap)

    develop = subparsers.add_parser(
        "train-g0", help="run a segmented G0 development candidate"
    )
    develop.add_argument("--run-id", required=True)
    develop.add_argument(
        "--candidate",
        choices=["lr2.5e-5", "lr5e-5", "lr1e-4"],
        required=True,
        help="locked learning-rate candidate",
    )
    develop.add_argument("--target-updates", type=int, default=50)
    develop.add_argument(
        "--execute",
        action="store_true",
        help="start real MLX compute; without this flag only print the locked plan",
    )
    develop.add_argument("--resume", action="store_true")
    develop.set_defaults(handler=commands.train_g0)

    process = subparsers.add_parser("process", help="run resumable G0 inference and routing")
    process.add_argument("--prepared-batch", required=True)
    process.add_argument("--generation", choices=["G0"], default="G0")
    process.add_argument("--resume", action="store_true")
    process.add_argument("--fake-backend", action="store_true", help=argparse.SUPPRESS)
    process.set_defaults(handler=commands.process)

    pilot = subparsers.add_parser("pilot-judges", help="run the bounded blind judge pilot")
    pilot.add_argument("--batch-id", required=True)
    pilot.add_argument("--allow-external-judge", action="store_true")
    pilot.add_argument("--fake-backend", action="store_true", help=argparse.SUPPRESS)
    pilot.set_defaults(handler=commands.pilot_judges)

    lock = subparsers.add_parser("judge-lock", help="explicitly select a production judge")
    lock.add_argument("--batch-id", required=True)
    lock.add_argument("--model", required=True)
    lock.add_argument("--reason", required=True)
    lock.set_defaults(handler=commands.judge_lock)

    serve = subparsers.add_parser("serve", help="serve the local-only HITL UI")
    serve.add_argument("--batch-id", required=True)
    serve.add_argument("--port", type=int, default=5055)
    serve.set_defaults(handler=commands.serve)

    release = subparsers.add_parser("release", help="create an immutable atomic release")
    release.add_argument("--batch-id", required=True)
    release.set_defaults(handler=commands.release)

    status = subparsers.add_parser("status", help="show batch coverage and lifecycle state")
    status.add_argument("--batch-id", required=True)
    status.set_defaults(handler=commands.status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        return int(args.handler(args, config))
    except (DQCheckError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"dqcheck: error: {exc}", file=sys.stderr)
        return 2
