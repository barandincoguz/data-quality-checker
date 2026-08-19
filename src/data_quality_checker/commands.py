"""Command handlers; implementations are imported here to keep CLI help cheap."""

from __future__ import annotations

import json
from argparse import Namespace
from typing import Any

from .config import AppConfig
from .errors import GateBlocked
from .storage import Store


def prepare(args: Namespace, config: AppConfig) -> int:
    from .preparation import prepare_batch

    result = prepare_batch(
        config=config,
        annotation_zip=args.annotation_zip,
        document_pool_zip=args.document_pool_zip,
        batch_id=args.batch_id,
        hmac_key_file=args.hmac_key_file,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def import_attribution(args: Namespace, config: AppConfig) -> int:
    from .preparation import import_annotation_attribution

    result = import_annotation_attribution(
        config=config,
        batch_id=args.batch_id,
        annotation_zip=args.annotation_zip,
    )
    summary = {key: value for key, value in result.items() if key != "attributions"}
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def train_bootstrap(args: Namespace, config: AppConfig) -> int:
    from .g0 import train_bootstrap

    result = train_bootstrap(
        config=config,
        generation=args.generation,
        execute=args.execute,
        refit=bool(getattr(args, "refit", False)),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def train_g0(args: Namespace, config: AppConfig) -> int:
    from .g0_training import run_development

    result = run_development(
        config=config,
        run_id=args.run_id,
        candidate_id=args.candidate,
        target_updates=args.target_updates,
        execute=args.execute,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def q36_p1_preflight(args: Namespace, config: AppConfig) -> int:
    from .q36_p1 import bootstrap_q36_p1

    result = bootstrap_q36_p1(
        config=config,
        batch_id=args.batch_id,
        build_data=args.build_data,
        measure_tokens=not args.skip_token_measurement,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def q36_p1_views(args: Namespace, config: AppConfig) -> int:
    del config
    from .q36_p1 import write_prediction_views

    result = write_prediction_views(
        raw_predictions=args.predictions,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def q36_p1_gold_view(args: Namespace, config: AppConfig) -> int:
    del config
    from .q36_p1 import write_production_gold_view

    result = write_production_gold_view(source=args.source, output=args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def process(args: Namespace, config: AppConfig) -> int:
    from .processing import process_batch

    result = process_batch(
        config=config,
        batch_id=args.prepared_batch,
        generation=args.generation,
        resume=args.resume,
        fake_backend=args.fake_backend,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def reroute(args: Namespace, config: AppConfig) -> int:
    from .rerouting import reroute_batch

    result = reroute_batch(
        config=config,
        batch_id=args.batch_id,
        generation=args.generation,
        policy_id=args.reference_policy,
        apply=args.apply,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def pilot_judges(args: Namespace, config: AppConfig) -> int:
    from .judges import run_judge_pilot

    result = run_judge_pilot(
        config=config,
        batch_id=args.batch_id,
        allow_external_judge=args.allow_external_judge,
        fake_backend=args.fake_backend,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def judge_lock(args: Namespace, config: AppConfig) -> int:
    from .judges import lock_judge

    result = lock_judge(
        config=config,
        batch_id=args.batch_id,
        model=args.model,
        reason=args.reason,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def serve(args: Namespace, config: AppConfig) -> int:
    from .hitl import create_hitl_app

    app = create_hitl_app(config=config, batch_id=args.batch_id)
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
    return 0


def review_backup(args: Namespace, config: AppConfig) -> int:
    from .review_backup import (
        create_review_backup,
        restore_review_backup_smoke,
        review_backup_lock,
        review_backup_status,
    )

    with (
        review_backup_lock(config=config, batch_id=args.batch_id),
        Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store,
    ):
        if args.review_backup_action == "create":
            result = create_review_backup(config=config, store=store, batch_id=args.batch_id)
        elif args.review_backup_action == "restore-smoke":
            result = restore_review_backup_smoke(config=config, store=store, batch_id=args.batch_id)
        else:
            result = review_backup_status(config=config, store=store, batch_id=args.batch_id)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def release(args: Namespace, config: AppConfig) -> int:
    from .release import release_batch

    result = release_batch(config=config, batch_id=args.batch_id)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def status(args: Namespace, config: AppConfig) -> int:
    if not config.database_path.exists():
        raise GateBlocked(f"database does not exist: {config.database_path}")
    with Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store:
        result: dict[str, Any] = store.status_summary(args.batch_id)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0

def predict_agent(args: Namespace, config: AppConfig) -> int:
    import os

    from .predict_agent import HttpTransport, build_backend, resolve_token, run_agent

    token = resolve_token(args.token_env, os.environ)
    transport = HttpTransport(base_url=args.space_url, token=token)
    backend = build_backend(config, fake=args.fake_backend)
    stats = run_agent(
        transport=transport,
        backend=backend,
        batch_size=args.batch_size,
        poll_seconds=args.poll_seconds,
        once=args.once,
    )
    print(
        json.dumps(
            {
                "pending": stats.pending,
                "predicted": stats.predicted,
                "upserted": stats.upserted,
                "failed": stats.failed,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0
