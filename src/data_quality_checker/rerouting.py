"""Auditable, atomic reference-policy rerouting of sealed predictions."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic, write_jsonl_atomic, write_text_atomic
from .config import AppConfig
from .contracts import validate_reference_list
from .errors import GateBlocked, IntegrityError, VersionConflict
from .fingerprints import fingerprint_json, sha256_file
from .heartbeat import RunLease
from .reference_policy import (
    DEFAULT_REFERENCE_POLICY_ID,
    NO_REFERENCE_FILTER_POLICY_ID,
    apply_reference_policy,
    reference_policy_spec,
)
from .router import route_document
from .sqlite_backup import backup_database
from .storage import Store

BUCKETS = ("GREEN", "YELLOW", "RED", "QUARANTINE")


def _counts(values: list[str]) -> dict[str, int]:
    counted = Counter(values)
    return {bucket: counted[bucket] for bucket in BUCKETS}


def _prediction_set_fingerprint(rows: list[dict[str, Any]]) -> str:
    return fingerprint_json(
        [
            {
                "internal_doc_id": row["internal_doc_id"],
                "response_sha256": row["response_sha256"],
                "model_fingerprint": row["model_fingerprint"],
            }
            for row in rows
        ]
    )


def _snapshot(
    *,
    config: AppConfig,
    store: Store,
    batch_id: str,
    generation: str,
    policy_id: str,
) -> dict[str, Any]:
    batch = store.get_batch(batch_id)
    if batch is None or batch["status"] != "processed" or not batch["ready"]:
        raise GateBlocked("reference-policy reroute requires a processed READY batch")
    documents = store.list_documents(batch_id)
    predictions = store.list_predictions(batch_id, generation)
    if not documents or len(predictions) != len(documents):
        raise IntegrityError(f"prediction coverage incomplete: {len(predictions)}/{len(documents)}")
    prediction_by_id = {row["internal_doc_id"]: row for row in predictions}
    if len(prediction_by_id) != len(predictions):
        raise IntegrityError("duplicate prediction internal_doc_id")

    decisions: list[dict[str, Any]] = []
    removed_human = removed_model = affected_human = affected_model = 0
    for document in documents:
        internal_doc_id = str(document["internal_doc_id"])
        prediction = prediction_by_id.get(internal_doc_id)
        if prediction is None:
            raise IntegrityError(f"missing prediction for {internal_doc_id}")
        response_path = Path(str(prediction["response_path"]))
        if (
            not response_path.is_file()
            or sha256_file(response_path) != prediction["response_sha256"]
        ):
            raise IntegrityError(f"prediction seal failure: {response_path}")
        try:
            response = json.loads(response_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IntegrityError(f"prediction response cannot be parsed: {response_path}") from exc
        stored_model = validate_reference_list(json.loads(prediction["references_json"]))
        response_model = validate_reference_list(response.get("references"))
        if stored_model != response_model:
            raise IntegrityError(f"SQLite/file prediction drift: {internal_doc_id}")
        human = validate_reference_list(json.loads(document["human_references_json"]))
        _, human_audit = apply_reference_policy(human, policy_id=policy_id)
        _, model_audit = apply_reference_policy(stored_model, policy_id=policy_id)
        removed_human += int(human_audit["removed_reference_count"])
        removed_model += int(model_audit["removed_reference_count"])
        affected_human += int(human_audit["removed_reference_count"] > 0)
        affected_model += int(model_audit["removed_reference_count"] > 0)
        common = {
            "human_references": human,
            "model_references": stored_model,
            "preparation_status": str(document["preparation_status"]),
            "model_status": str(prediction["status"]),
            "model_truncated": bool(json.loads(prediction["operational_json"]).get("truncated")),
            "has_safe_text": bool(document["text"]),
        }
        before = route_document(**common, reference_policy_id=NO_REFERENCE_FILTER_POLICY_ID)
        after = route_document(**common, reference_policy_id=policy_id)
        decisions.append(
            {
                "internal_doc_id": internal_doc_id,
                "public_doc_id": document["public_doc_id"],
                "document_row_version": int(document["row_version"]),
                "database_bucket": document["router_bucket"],
                "unfiltered_bucket": before.bucket,
                "filtered_bucket": after.bucket,
                "removed_human_reference_count": human_audit["removed_reference_count"],
                "removed_model_reference_count": model_audit["removed_reference_count"],
                "prediction_sha256": prediction["response_sha256"],
            }
        )

    transition_counts: Counter[str] = Counter(
        f"{row['unfiltered_bucket']}->{row['filtered_bucket']}" for row in decisions
    )
    database_counts = _counts([str(row["database_bucket"]) for row in decisions])
    unfiltered_counts = _counts([str(row["unfiltered_bucket"]) for row in decisions])
    filtered_counts = _counts([str(row["filtered_bucket"]) for row in decisions])
    prediction_fingerprint = _prediction_set_fingerprint(predictions)
    source_fingerprint = fingerprint_json(
        {
            "batch_id": batch_id,
            "batch_input_fingerprint": batch["input_fingerprint"],
            "config_fingerprint": batch["config_fingerprint"],
            "generation": generation,
            "prediction_set_fingerprint": prediction_fingerprint,
            "policy": reference_policy_spec(policy_id),
            "decisions": [
                {
                    key: row[key]
                    for key in (
                        "internal_doc_id",
                        "unfiltered_bucket",
                        "filtered_bucket",
                        "removed_human_reference_count",
                        "removed_model_reference_count",
                        "prediction_sha256",
                    )
                }
                for row in decisions
            ],
        }
    )
    return {
        "batch": batch,
        "decisions": decisions,
        "document_count": len(documents),
        "prediction_count": len(predictions),
        "prediction_set_fingerprint": prediction_fingerprint,
        "source_fingerprint": source_fingerprint,
        "database_router_counts": database_counts,
        "unfiltered_router_counts": unfiltered_counts,
        "filtered_router_counts": filtered_counts,
        "transition_counts": dict(sorted(transition_counts.items())),
        "removed_human_reference_count": removed_human,
        "removed_model_reference_count": removed_model,
        "affected_human_document_count": affected_human,
        "affected_model_document_count": affected_model,
    }


def _public_payload(
    *, batch_id: str, generation: str, policy_id: str, snapshot: dict[str, Any]
) -> dict[str, Any]:
    before = snapshot["unfiltered_router_counts"]
    after = snapshot["filtered_router_counts"]
    return {
        "schema_version": 1,
        "batch_id": batch_id,
        "generation": generation,
        "policy": reference_policy_spec(policy_id),
        "source_fingerprint": snapshot["source_fingerprint"],
        "prediction_set_fingerprint": snapshot["prediction_set_fingerprint"],
        "document_count": snapshot["document_count"],
        "prediction_count": snapshot["prediction_count"],
        "database_router_counts_before_apply": snapshot["database_router_counts"],
        "unfiltered_router_counts": before,
        "filtered_router_counts": after,
        "absolute_router_deltas_filtered_minus_unfiltered": {
            bucket: after[bucket] - before[bucket] for bucket in BUCKETS
        },
        "transition_counts": snapshot["transition_counts"],
        "removed_human_reference_count": snapshot["removed_human_reference_count"],
        "removed_model_reference_count": snapshot["removed_model_reference_count"],
        "affected_human_document_count": snapshot["affected_human_document_count"],
        "affected_model_document_count": snapshot["affected_model_document_count"],
        "review_load_red_plus_yellow": {
            "unfiltered": before["RED"] + before["YELLOW"],
            "filtered": after["RED"] + after["YELLOW"],
            "absolute_delta": (after["RED"] + after["YELLOW"] - before["RED"] - before["YELLOW"]),
        },
        "raw_prediction_files_modified": False,
    }


def _write_checksums(public_dir: Path, names: list[str]) -> None:
    write_text_atomic(
        public_dir / "SHA256SUMS.txt",
        "".join(f"{sha256_file(public_dir / name)}  {name}\n" for name in names),
        mode=0o644,
    )


def reroute_batch(
    *,
    config: AppConfig,
    batch_id: str,
    generation: str = "G0",
    policy_id: str = DEFAULT_REFERENCE_POLICY_ID,
    apply: bool = False,
) -> dict[str, Any]:
    """Preflight or atomically apply a versioned policy reroute."""

    if generation != "G0":
        raise GateBlocked("only G0 rerouting is supported")
    policy = reference_policy_spec(policy_id)
    run_id = f"{batch_id}_{generation}_{policy_id}_{policy['fingerprint'][:12]}"
    sensitive_dir = config.sensitive_root / "batches" / batch_id / "reroutes" / run_id
    public_dir = config.public_root / "batches" / batch_id / "reroutes" / run_id
    lease = RunLease(
        lock_path=config.sensitive_root / "locks" / f"reroute_{batch_id}_{generation}.lock",
        heartbeat_path=sensitive_dir / "heartbeat.json",
        purpose="reference-policy-reroute",
        run_id=run_id,
        input_fingerprint=fingerprint_json({"batch_id": batch_id, "policy": policy}),
        config_fingerprint=config.fingerprint,
    ).start(stage="preflight")
    try:
        with Store(config.database_path, busy_timeout_ms=config.runtime.busy_timeout_ms) as store:
            snapshot = _snapshot(
                config=config,
                store=store,
                batch_id=batch_id,
                generation=generation,
                policy_id=policy_id,
            )
            public = _public_payload(
                batch_id=batch_id,
                generation=generation,
                policy_id=policy_id,
                snapshot=snapshot,
            )
            result_path = public_dir / "RESULT.json"
            if result_path.is_file():
                try:
                    existing_result = json.loads(result_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise IntegrityError(
                        f"existing reroute result is invalid: {result_path}"
                    ) from exc
                if (
                    existing_result.get("status") != "applied_and_verified"
                    or existing_result.get("prediction_set_fingerprint")
                    != snapshot["prediction_set_fingerprint"]
                    or existing_result.get("policy", {}).get("fingerprint") != policy["fingerprint"]
                    or snapshot["database_router_counts"] != snapshot["filtered_router_counts"]
                ):
                    raise IntegrityError("existing reroute result does not match live sealed state")
                lease.finish(status="completed")
                return existing_result
            write_jsonl_atomic(sensitive_dir / "DECISIONS.jsonl", snapshot["decisions"])
            preflight = {
                **public,
                "run_id": run_id,
                "status": "preflight_passed",
                "applied": False,
            }
            write_json_atomic(public_dir / "PREFLIGHT.json", preflight, mode=0o644)
            _write_checksums(public_dir, ["PREFLIGHT.json"])
            lease.beat(
                stage="preflight-complete",
                completed_units=snapshot["document_count"],
                expected_units=snapshot["document_count"],
                last_successful_unit=snapshot["decisions"][-1]["internal_doc_id"],
            )
            if not apply:
                lease.finish(status="completed")
                return preflight

            reviews = store.list_reviews(batch_id)
            changed_reviews = [row for row in reviews if row["status"] != "pending"]
            if changed_reviews:
                raise GateBlocked(
                    "reroute apply requires all reviews to remain pending; "
                    f"observed {len(changed_reviews)} changed reviews"
                )
            database_matches_unfiltered = all(
                row["database_bucket"] == row["unfiltered_bucket"] for row in snapshot["decisions"]
            )
            database_matches_filtered = all(
                row["database_bucket"] == row["filtered_bucket"] for row in snapshot["decisions"]
            )
            event = store.connection.execute(
                """SELECT payload_json FROM batch_events
                     WHERE batch_id=? AND event_type='reference_policy_rerouted'
                     ORDER BY event_id DESC""",
                (batch_id,),
            ).fetchone()
            recovered_after_commit = bool(
                database_matches_filtered
                and event
                and json.loads(event["payload_json"]).get("run_id") == run_id
            )
            if not database_matches_unfiltered and not recovered_after_commit:
                raise GateBlocked(
                    "database router buckets match neither the sealed unfiltered baseline "
                    "nor a prior committed reroute"
                )

            backup = backup_database(
                store.connection,
                config.sensitive_root / "backups" / f"pre_{run_id}.sqlite3",
            )
            write_json_atomic(
                sensitive_dir / "BACKUP.json",
                {"schema_version": 1, **backup, "run_id": run_id},
            )
            if not recovered_after_commit:
                lease.beat(stage="apply-transaction")
                with store.transaction():
                    for row in snapshot["decisions"]:
                        cursor = store.connection.execute(
                            """UPDATE documents
                                  SET router_bucket=?, row_version=row_version+1,
                                      updated_at_epoch=strftime('%s','now')
                                WHERE batch_id=? AND internal_doc_id=?
                                  AND row_version=? AND router_bucket=?""",
                            (
                                row["filtered_bucket"],
                                batch_id,
                                row["internal_doc_id"],
                                row["document_row_version"],
                                row["database_bucket"],
                            ),
                        )
                        if cursor.rowcount != 1:
                            raise VersionConflict(
                                f"document changed during reroute: {row['internal_doc_id']}"
                            )
                    store.connection.execute(
                        """INSERT INTO batch_events(
                               batch_id,event_type,payload_json,created_at_epoch
                             ) VALUES (?,?,?,strftime('%s','now'))""",
                        (
                            batch_id,
                            "reference_policy_rerouted",
                            json.dumps(
                                {
                                    "run_id": run_id,
                                    "policy_id": policy_id,
                                    "source_fingerprint": snapshot["source_fingerprint"],
                                    "filtered_router_counts": snapshot["filtered_router_counts"],
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        ),
                    )

            observed_raw = store.status_summary(batch_id)["documents"]
            observed = {bucket: int(observed_raw.get(bucket, 0)) for bucket in BUCKETS}
            if observed != snapshot["filtered_router_counts"]:
                raise IntegrityError(f"post-reroute router coverage mismatch: {observed}")
            result = {
                **public,
                "run_id": run_id,
                "status": "applied_and_verified",
                "applied": True,
                "recovered_after_committed_transaction": recovered_after_commit,
                "database_backup": {
                    "sha256": backup["sha256"],
                    "integrity_check": "ok",
                },
                "database_router_counts_after_apply": observed,
            }
            write_json_atomic(public_dir / "RESULT.json", result, mode=0o644)
            _write_checksums(public_dir, ["PREFLIGHT.json", "RESULT.json"])
            lease.finish(status="completed")
            return result
    except BaseException as exc:
        lease.finish(status="failed", error=str(exc))
        raise
