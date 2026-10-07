"""Command handlers; implementations are imported here to keep CLI help cheap."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

from .config import AppConfig
from .errors import ConfigurationError, ContractError, GateBlocked
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
    from .loop_ops import assert_round_generation

    raw_judge_models = getattr(args, "judge_models", None)
    judge_models: tuple[str, ...] | None = None
    if raw_judge_models is not None:
        parsed = [part.strip() for part in raw_judge_models.split(",") if part.strip()]
        if not parsed:
            raise ContractError(
                f"--judge-models {raw_judge_models!r} did not contain any model ids"
            )
        duplicates = sorted({model for model in parsed if parsed.count(model) > 1})
        if duplicates:
            raise ContractError(f"--judge-models contains duplicate model ids: {duplicates}")
        judge_models = tuple(parsed)
    assert_round_generation(config, args.batch_id, args.generation)
    result = run_judge_pilot(
        config=config,
        batch_id=args.batch_id,
        generation=args.generation,
        allow_external_judge=args.allow_external_judge,
        fake_backend=args.fake_backend,
        judge_models=judge_models,
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
    from .loop_ops import assert_round_generation

    assert_round_generation(config, args.batch_id, args.generation)

    app = create_hitl_app(config=config, batch_id=args.batch_id, generation=args.generation)
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
    from .loop_ops import assert_round_generation
    from .release import release_batch

    assert_round_generation(config, args.batch_id, args.generation)

    result = release_batch(config=config, batch_id=args.batch_id, generation=args.generation)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def status(args: Namespace, config: AppConfig) -> int:
    if not config.database_path.exists():
        raise GateBlocked(f"database does not exist: {config.database_path}")
    with Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store:
        result: dict[str, Any] = store.status_summary(args.batch_id)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def loop_status(args: Namespace, config: AppConfig) -> int:
    from .loop_rounds import rounds_state_dir
    from .loop_runner import round_status

    result = round_status(rounds_state_dir(config), args.round_index)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def predict_agent(args: Namespace, config: AppConfig) -> int:
    import os

    from .predict_agent import HttpTransport, build_backend, resolve_token, run_agent

    if not 1 <= args.batch_size <= 16:
        raise ConfigurationError("--batch-size must be between 1 and 16")
    if args.poll_seconds <= 0:
        raise ConfigurationError("--poll-seconds must be greater than zero")
    token = resolve_token(args.token_env, os.environ)
    transport = HttpTransport(base_url=args.space_url, token=token)
    backend = build_backend(config)
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


def _print_json(result: Any) -> int:
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


def loop_train(args: Namespace, config: AppConfig) -> int:
    from .loop_ops import train_loop_candidate, train_loop_round

    view = getattr(args, "view", None)
    # The whole-document line is never sealed: `--view full_documents` always
    # trains a candidate, whatever else is passed.
    stop_at = getattr(args, "stop_at", None)
    # The raw-label arm (`M_k_raw`, noisy-vs-clean ablation) is never sealed either.
    labels = getattr(args, "labels", "clean")
    if getattr(args, "no_seal", False) or view == "full_documents" or stop_at is not None or labels != "clean":
        # A run stopped short is a candidate too: it is never sealed.
        kwargs = {"round_index": args.round, "resume": args.resume, "view": view, "stop_at": stop_at}
        if labels != "clean":
            kwargs["labels"] = labels
        return _print_json(train_loop_candidate(config, **kwargs))
    return _print_json(train_loop_round(config, round_index=args.round, resume=args.resume))


def loop_raw_check(args: Namespace, config: AppConfig) -> int:
    from .loop_ops import check_raw_candidate_against_release

    report = check_raw_candidate_against_release(config, round_index=args.round)
    _print_json(report)
    # Nonzero when the run is not the release's M_k_raw: it must be trained again.
    return 0 if report["same_documents"] else 1


def loop_correct_ranges(args: Namespace, config: AppConfig) -> int:
    from .loop_ops import batch_id_for
    from .truth_corrections import propose_range_corrections, write_range_corrections

    batch_id = batch_id_for(args.round)
    if not args.write:
        return _print_json({"written": False, **propose_range_corrections(config, batch_id)})
    if not args.decision:
        raise ContractError("--write records a user decision: quote it with --decision")
    payload = write_range_corrections(config, batch_id=batch_id, decision=args.decision)
    return _print_json({"written": True, **payload})


def loop_correct_rows(args: Namespace, config: AppConfig) -> int:
    from .loop_ops import batch_id_for
    from .truth_corrections import write_row_fixes

    if args.write and not args.decision:
        raise ContractError("--write records a user decision: quote it with --decision")
    fixes = json.loads(Path(args.fixes).read_text(encoding="utf-8"))
    payload = write_row_fixes(
        config, batch_id=batch_id_for(args.round), decision=args.decision or "", fixes=fixes, dry_run=not args.write
    )
    return _print_json({"written": args.write, **payload})


def loop_select_rescored(args: Namespace, config: AppConfig) -> int:
    from . import rescored_selection

    selection = rescored_selection.select_rescored_checkpoint(config, run_id=args.run_id, label=args.label)
    return _print_json({key: value for key, value in selection.items() if key != "points"})


def loop_compare(args: Namespace, config: AppConfig) -> int:
    from .loop_ops import compare_views

    report = compare_views(config, round_index=args.round)
    return _print_json(
        {
            "round": report["round"],
            "selected": {view: line["core"] for view, line in report["selected"].items()},
            "full_minus_windows": report["paired_bootstrap_full_minus_windows"],
            "vs_previous_round": report.get("paired_bootstrap_vs_previous_round"),
        }
    )


def loop_predict(args: Namespace, config: AppConfig) -> int:
    from .loop_ops import predict_loop_round

    return _print_json(
        predict_loop_round(config, round_index=args.round, hmac_key_file=args.hmac_key_file)
    )


def loop_judge(args: Namespace, config: AppConfig) -> int:
    from .loop_ops import judge_round

    return _print_json(judge_round(config, round_index=args.round))


def loop_carry_decisions(args: Namespace, config: AppConfig) -> int:
    from .config import load_config
    from .loop_carry import carry_decisions
    from .loop_ops import batch_id_for

    source = load_config(args.from_config)
    return _print_json(
        carry_decisions(
            config,
            batch_id=batch_id_for(args.round),
            source_database=source.database_path,
            source_label=args.from_config.stem,
        )
    )


def loop_rescore(args: Namespace, config: AppConfig) -> int:
    from .rescore import rescore_run

    summary = rescore_run(config, run_id=args.run_id, label=args.label, reason=args.reason)
    return _print_json({k: v for k, v in summary.items() if k != "rows"}
                       | {"rows": [(r["checkpoint"], round(r["old_f1"], 4), round(r["new_f1"], 4)) for r in summary["rows"]]})


def loop_prompt_screen(args: Namespace, config: AppConfig) -> int:
    from .prompt_screen import stage_one, stage_two, write_stage_two_configs

    if args.stage == 2:
        import json as _json
        from pathlib import Path as _Path

        from .config import load_config
        from .g0 import PROMPT_VARIANT, loop_prompt
        from .loop_ops import LOOP_CANDIDATE

        runs = {PROMPT_VARIANT: _Path(config.training_runs_root) / args.base_run_id / "development" / LOOP_CANDIDATE}
        for path in args.candidate_config or []:
            candidate = load_config(path)
            record = _json.loads((_Path(candidate.public_root) / "loop" / "candidates" / "round_000_windows.json").read_text())
            runs[loop_prompt(candidate)[0]] = _Path(candidate.training_runs_root) / record["run_id"] / "development" / LOOP_CANDIDATE
        output = _Path(config.public_root) / "loop" / "prompt_screen" / "stage2" / "selection.json"
        result = stage_two(runs, output=output)
        return _print_json({"winner": result["winner"],
                            "ranking": [(r["variant"], round(r["f1"], 4), round(r["best_f1_up_to"], 4)) for r in result["ranking"]],
                            "output": str(output)})
    selection = stage_one(config, base_run_id=args.base_run_id)
    configs = write_stage_two_configs(args.config, selection)
    return _print_json({"ranking": [(r["variant"], round(r["f1"], 4)) for r in selection["ranking"]],
                        "to_stage_two": selection["to_stage_two"], "configs": [str(p) for p in configs]})


def loop_report(args: Namespace, config: AppConfig) -> int:
    from .loop_ops import batch_trend

    trend = batch_trend(config)
    return _print_json(
        {
            "batches": [
                {
                    "batch_id": batch["batch_id"],
                    "predicted_by": batch["predicted_by"],
                    **{
                        scope: {
                            "documents": batch[scope]["documents"],
                            "model_core_f1": batch[scope]["sources"]["model"]["core"]["f1"],
                            "annotator_core_f1": batch[scope]["sources"]["annotator"]["core"]["f1"],
                        }
                        for scope in ("reviewed", "batch")
                        if scope in batch
                    },
                }
                for batch in trend["batches"]
            ],
            "validation": trend["validation"],
        }
    )


def loop_mark(args: Namespace, config: AppConfig) -> int:
    from .loop_ops import mark_stage

    return _print_json(mark_stage(config, round_index=args.round, stage=args.stage))


def loop_errors(args: Namespace, config: AppConfig) -> int:
    from .loop_ops import error_report

    report = error_report(config, round_index=args.round)
    summary = {
        "round": report["round"],
        "reviews": report["reviews"],
        "per_rater": {
            name: {key: rater[key] for key in ("available_document_count", "tp", "fp", "fn", "f1")}
            for name, rater in report["all_finalized"]["per_rater"].items()
        },
    }
    return _print_json(summary)


def loop_open_test(args: Namespace, config: AppConfig) -> int:
    from .loop_ops import open_test

    return _print_json(open_test(config, confirm=args.confirm_open_test))
