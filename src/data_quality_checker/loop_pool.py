"""The GT750 loop's pool: balance strata and a deal that leaves no document out.

`build_round_batches` deals exactly `rounds * size` documents and samples the
rest away. B-3 (2026-09-25) keeps every pool document: four full rounds of
250, then whatever the sample left becomes a fifth, smaller round.
"""

from __future__ import annotations

import ast
import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import AppConfig
from .errors import ContractError
from .loop_batches import build_round_batches
from .preparation import _pool_id, _record_document_id, _select_channel, annotation_attribution
from .safezip import read_json_records
from .text import html_to_text, normalize_text


def human_references(payload: dict[str, Any]) -> list[dict[str, Any]]:
    value = payload.get("current_references")
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ContractError("current_references is not a list of objects")
    return value


def pool_strata(
    *, config: AppConfig, annotation_zip: Path, document_pool_zip: Path, doc_ids: Sequence[str]
) -> list[dict[str, Any]]:
    wanted = set(doc_ids)
    _, annotation_records = read_json_records(annotation_zip, config.security)
    _, pool_records = read_json_records(document_pool_zip, config.security)
    annotations: dict[str, dict[str, Any]] = {}
    for record in annotation_records:
        try:
            raw_id = _record_document_id(record.payload)
        except ContractError:
            continue
        if raw_id in wanted:
            annotations[raw_id] = record.payload
    pool = {
        _pool_id(record.payload): record.payload
        for record in pool_records
        if "evrakOid" in record.payload and _pool_id(record.payload) in wanted
    }
    missing = sorted((wanted - set(annotations)) | (wanted - set(pool)))
    if missing:
        raise ContractError(f"{len(missing)} pool document(s) absent from the archives")

    lengths = {
        oid: len(
            html_to_text(pool[oid].get("htmlText")) or normalize_text(pool[oid].get("pdfText"))
        )
        for oid in wanted
    }
    cuts = statistics.quantiles(sorted(lengths.values()), n=4)
    strata = []
    for oid in sorted(wanted):
        references = human_references(annotations[oid])
        channel = _select_channel(pool[oid], references)[1]
        attribution = annotation_attribution(annotations[oid])
        annotator = (attribution.get("completed_by") or attribution.get("last_editor") or {}).get(
            "username", "unknown"
        )
        quartile = 1 + sum(lengths[oid] > cut for cut in cuts)
        strata.append(
            {
                "doc_id": oid,
                "length_quartile": f"Q{quartile}",
                "text_channel": channel,
                "annotator": annotator or "unknown",
                "reference_count": len(references),
            }
        )
    return strata


def deal_with_remainder(
    documents: Sequence[dict[str, Any]], *, full_rounds: int, size: int, seed: int
) -> list[list[str]]:
    batches = build_round_batches(documents, rounds=full_rounds, size=size, seed=seed)
    dealt = {doc_id for batch in batches for doc_id in batch}
    remainder = sorted(str(d["doc_id"]) for d in documents if str(d["doc_id"]) not in dealt)
    return batches + ([remainder] if remainder else [])
