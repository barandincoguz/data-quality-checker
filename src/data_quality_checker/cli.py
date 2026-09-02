"""`dqcheck` command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import commands
from .config import default_config_path, load_config
from .errors import DQCheckError
from .reference_policy import DEFAULT_REFERENCE_POLICY_ID


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

    attribution = subparsers.add_parser(
        "import-attribution",
        help="privately import human annotator identities for an existing batch",
    )
    attribution.add_argument("--annotation-zip", type=Path, required=True)
    attribution.add_argument("--batch-id", required=True)
    attribution.set_defaults(handler=commands.import_attribution)

    train = subparsers.add_parser("train-bootstrap", help="build or run canonical-only G0")
    train.add_argument("--generation", choices=["G0"], default="G0")
    train.add_argument(
        "--execute",
        action="store_true",
        help="execute compute after every preflight gate; otherwise emit the locked plan",
    )
    train.add_argument(
        "--refit",
        action="store_true",
        help="final refit run: train on all 494 canonical documents (nominal validation "
        "only, no held-out test); derives its own run id",
    )
    train.set_defaults(handler=commands.train_bootstrap)

    develop = subparsers.add_parser("train-g0", help="run a segmented G0 development candidate")
    develop.add_argument("--run-id", required=True)
    develop.add_argument(
        "--candidate",
        choices=[
            "lr1e-5",
            "lr2.5e-5",
            "lr2.5e-5-cos200",
            "lr2.5e-5-warm42-cos150",
            "lr2.5e-5-warm42-cos850",
            "refit-cos1003",
            "lr5e-5",
            "lr1e-4",
        ],
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

    q36 = subparsers.add_parser(
        "q36-p1", help="operate the isolated controlled Qwen3.6-27B SFT track"
    )
    q36_actions = q36.add_subparsers(dest="q36_action", required=True)
    q36_preflight = q36_actions.add_parser(
        "preflight", help="seal Q36-P1 inputs and inspect all fail-closed gates"
    )
    q36_preflight.add_argument("--batch-id", default="neon_wl_v1")
    q36_preflight.add_argument(
        "--build-data",
        action="store_true",
        help="write sensitive training sources only after the GREEN audit passes",
    )
    q36_preflight.add_argument(
        "--skip-token-measurement",
        action="store_true",
        help="software-test aid; leaves the common inference-contract gate closed",
    )
    q36_preflight.set_defaults(handler=commands.q36_p1_preflight)
    q36_views = q36_actions.add_parser(
        "views", help="seal raw full-extraction and derive the production 213/413 view"
    )
    q36_views.add_argument("--predictions", type=Path, required=True)
    q36_views.add_argument("--output-dir", type=Path, required=True)
    q36_views.set_defaults(handler=commands.q36_p1_views)
    q36_gold = q36_actions.add_parser(
        "gold-view", help="derive the symmetric production 213/413 gold view"
    )
    q36_gold.add_argument("--source", type=Path, required=True)
    q36_gold.add_argument("--output", type=Path, required=True)
    q36_gold.set_defaults(handler=commands.q36_p1_gold_view)

    process = subparsers.add_parser("process", help="run resumable G0 inference and routing")
    process.add_argument("--prepared-batch", required=True)
    process.add_argument("--generation", choices=["G0"], default="G0")
    process.add_argument("--resume", action="store_true")
    process.add_argument("--fake-backend", action="store_true", help=argparse.SUPPRESS)
    process.set_defaults(handler=commands.process)

    reroute = subparsers.add_parser(
        "reroute", help="preflight or atomically apply a reference-policy reroute"
    )
    reroute.add_argument("--batch-id", required=True)
    reroute.add_argument("--generation", choices=["G0"], default="G0")
    reroute.add_argument(
        "--reference-policy",
        default=DEFAULT_REFERENCE_POLICY_ID,
        choices=[DEFAULT_REFERENCE_POLICY_ID],
    )
    reroute.add_argument(
        "--apply",
        action="store_true",
        help="apply verified bucket changes after backup; default is preflight only",
    )
    reroute.set_defaults(handler=commands.reroute)

    pilot = subparsers.add_parser("pilot-judges", help="run the bounded blind judge pilot")
    pilot.add_argument("--batch-id", required=True)
    pilot.add_argument("--allow-external-judge", action="store_true")
    pilot.add_argument(
        "--judge-models",
        default=None,
        help="comma-separated judge model ids; defaults to the two-model pilot pair",
    )
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

    agent = subparsers.add_parser(
        "predict-agent",
        help="serve G0 predictions to a remote annotation platform",
    )
    agent.add_argument("--space-url", required=True)
    agent.add_argument("--token-env", default="DQCHECK_INGEST_TOKEN")
    agent.add_argument("--batch-size", type=int, default=4)
    agent.add_argument("--poll-seconds", type=float, default=30.0)
    agent.add_argument("--once", action="store_true")
    agent.set_defaults(handler=commands.predict_agent)

    review_backup = subparsers.add_parser(
        "review-backup", help="inspect or operate verified HITL review backups"
    )
    review_backup_actions = review_backup.add_subparsers(dest="review_backup_action", required=True)
    review_backup_status = review_backup_actions.add_parser(
        "status", help="verify the latest review backup against live SQLite state"
    )
    review_backup_status.add_argument("--batch-id", required=True)
    review_backup_status.set_defaults(handler=commands.review_backup)
    review_backup_create = review_backup_actions.add_parser(
        "create", help="create and verify a snapshot of the live review state"
    )
    review_backup_create.add_argument("--batch-id", required=True)
    review_backup_create.set_defaults(handler=commands.review_backup)
    review_backup_verify = review_backup_actions.add_parser(
        "verify", help="verify the latest snapshot against the live review state"
    )
    review_backup_verify.add_argument("--batch-id", required=True)
    review_backup_verify.set_defaults(handler=commands.review_backup)
    review_backup_restore_smoke = review_backup_actions.add_parser(
        "restore-smoke",
        help="restore the latest snapshot into a temporary database and verify it",
    )
    review_backup_restore_smoke.add_argument("--batch-id", required=True)
    review_backup_restore_smoke.set_defaults(handler=commands.review_backup)

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
