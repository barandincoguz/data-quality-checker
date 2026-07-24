"""Lossless, fingerprinted training windows for Metal-bounded G0 LoRA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .atomic import write_json_atomic, write_jsonl_atomic
from .errors import ContractError, FingerprintMismatch, IntegrityError
from .fingerprints import fingerprint_json, sha256_file


PRIMARY_RESERVED_TOKENS = 512
MAX_PRIMARY_OVERLAP_TOKENS = 256
FALLBACK_WINDOW_TOKENS = 128
FALLBACK_OVERLAP_TOKENS = 48


def _primary_window_geometry(max_sequence_length: int) -> tuple[int, int]:
    """Use as much real document context as the accepted sequence permits."""

    window_tokens = max(
        FALLBACK_WINDOW_TOKENS,
        max_sequence_length - PRIMARY_RESERVED_TOKENS,
    )
    overlap_tokens = min(MAX_PRIMARY_OVERLAP_TOKENS, window_tokens // 4)
    return window_tokens, overlap_tokens


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ContractError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append(row)
    return rows


def _token_intervals(
    *, start: int, end: int, window_tokens: int, overlap_tokens: int
) -> list[tuple[int, int]]:
    if not 0 <= start < end:
        raise ValueError("token interval must be non-empty")
    if not 0 <= overlap_tokens < window_tokens:
        raise ValueError("overlap must be smaller than the window")
    intervals: list[tuple[int, int]] = []
    cursor = start
    while True:
        stop = min(cursor + window_tokens, end)
        intervals.append((cursor, stop))
        if stop == end:
            return intervals
        cursor = stop - overlap_tokens


def _compact_references(value: str) -> list[dict[str, Any]]:
    try:
        references = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ContractError(f"assistant target is not JSON: {exc}") from exc
    if not isinstance(references, list) or any(
        not isinstance(reference, dict) for reference in references
    ):
        raise ContractError("assistant target must be a JSON list of objects")
    return references


def _rendered_token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_dict=False,
    )
    if not isinstance(tokens, list):
        raise ContractError("tokenizer chat template did not return a token list")
    return len(tokens)


def _window_row(
    *,
    tokenizer: Any,
    system_message: dict[str, str],
    body: str,
    offsets: list[tuple[int, int]],
    interval: tuple[int, int],
    references: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], set[int]]:
    start, end = interval
    char_start = int(offsets[start][0])
    char_end = int(offsets[end - 1][1])
    text = body[char_start:char_end]
    visible_indices = {
        index
        for index, reference in enumerate(references)
        if isinstance(reference.get("source_text"), str)
        and reference["source_text"]
        and reference["source_text"] in text
    }
    visible = [references[index] for index in sorted(visible_indices)]
    messages = [
        system_message,
        {"role": "user", "content": text.rstrip()},
        {
            "role": "assistant",
            "content": json.dumps(
                visible, ensure_ascii=False, separators=(",", ":")
            ),
        },
    ]
    row = {"messages": messages}
    metadata = {
        "token_start": start,
        "token_end": end,
        "char_start": char_start,
        "char_end": char_end,
        "reference_count": len(visible),
        "tokens": _rendered_token_count(tokenizer, messages),
    }
    return row, metadata, visible_indices


def _validate_cached_view(
    *, manifest_path: Path, output_path: Path, output_doc_ids_path: Path, request: dict[str, Any]
) -> dict[str, Any] | None:
    if not manifest_path.exists():
        if output_path.exists() or output_doc_ids_path.exists():
            raise IntegrityError("orphaned context-window output exists without a manifest")
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"context-window manifest cannot be parsed: {exc}") from exc
    if manifest.get("request") != request:
        raise FingerprintMismatch("context-window request differs from the sealed manifest")
    if not output_path.is_file() or not output_doc_ids_path.is_file():
        raise IntegrityError("sealed context-window output is incomplete")
    if manifest.get("output_jsonl_sha256") != sha256_file(output_path):
        raise IntegrityError("context-window JSONL checksum mismatch")
    if manifest.get("output_doc_ids_sha256") != sha256_file(output_doc_ids_path):
        raise IntegrityError("context-window doc-id checksum mismatch")
    return manifest


def build_context_window_view(
    *,
    source_path: Path,
    source_doc_ids_path: Path,
    output_path: Path,
    output_doc_ids_path: Path,
    manifest_path: Path,
    tokenizer: Any,
    model_fingerprint: str,
    max_sequence_length: int,
) -> dict[str, Any]:
    """Build an idempotent view with complete text and reference coverage."""

    source_path = source_path.resolve()
    source_doc_ids_path = source_doc_ids_path.resolve()
    primary_window_tokens, primary_overlap_tokens = _primary_window_geometry(
        max_sequence_length
    )
    request = {
        "schema_version": 1,
        "source_jsonl_sha256": sha256_file(source_path),
        "source_doc_ids_sha256": sha256_file(source_doc_ids_path),
        "model_fingerprint": model_fingerprint,
        "max_sequence_length": max_sequence_length,
        "primary_window_tokens": primary_window_tokens,
        "primary_overlap_tokens": primary_overlap_tokens,
        "fallback_window_tokens": FALLBACK_WINDOW_TOKENS,
        "fallback_overlap_tokens": FALLBACK_OVERLAP_TOKENS,
        "reference_rescue": "source_text_v1",
        "empty_chunk_sampling": "max_one_per_source_document_v1",
        "thinking_control": "chat_template_enable_thinking_false",
    }
    request["fingerprint"] = fingerprint_json(request)
    if cached := _validate_cached_view(
        manifest_path=manifest_path,
        output_path=output_path,
        output_doc_ids_path=output_doc_ids_path,
        request=request,
    ):
        return cached

    rows = _read_jsonl(source_path)
    try:
        doc_ids = json.loads(source_doc_ids_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"source doc IDs cannot be parsed: {exc}") from exc
    if not isinstance(doc_ids, list) or len(doc_ids) != len(rows):
        raise ContractError("source doc IDs do not align with training rows")
    offset_tokenizer = getattr(tokenizer, "_tokenizer", None)
    if offset_tokenizer is None:
        raise ContractError("tokenizer exposes no offset-capable backend")

    output_rows: list[dict[str, Any]] = []
    output_doc_ids: list[int] = []
    chunks: list[dict[str, Any]] = []
    native_documents = 0
    windowed_documents = 0
    original_reference_count = 0
    chunk_reference_count = 0
    empty_chunk_count = 0
    candidate_empty_chunk_count = 0
    dropped_empty_chunk_count = 0
    reference_rescue_count = 0
    maximum_tokens = 0

    for row_index, (row, raw_doc_id) in enumerate(zip(rows, doc_ids, strict=True)):
        messages = row.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 3
            or [message.get("role") for message in messages]
            != ["system", "user", "assistant"]
        ):
            raise ContractError(f"unexpected chat contract at source row {row_index}")
        doc_id = int(raw_doc_id)
        references = _compact_references(messages[2]["content"])
        original_reference_count += len(references)
        full_tokens = _rendered_token_count(tokenizer, messages)
        if full_tokens <= max_sequence_length:
            native_documents += 1
            output_rows.append(row)
            output_doc_ids.append(doc_id)
            chunk_reference_count += len(references)
            maximum_tokens = max(maximum_tokens, full_tokens)
            chunks.append(
                {
                    "output_row_index": len(output_rows) - 1,
                    "source_row_index": row_index,
                    "doc_id": doc_id,
                    "mode": "native",
                    "tokens": full_tokens,
                    "reference_count": len(references),
                }
            )
            continue

        windowed_documents += 1
        user_content = messages[1].get("content")
        if not isinstance(user_content, str) or not user_content.strip():
            raise ContractError(f"user text missing at source row {row_index}")
        if user_content.rstrip().endswith("/no_think"):
            raise ContractError(
                f"unsupported literal /no_think suffix at source row {row_index}"
            )
        body = user_content.rstrip()
        encoded = offset_tokenizer(
            body, add_special_tokens=False, return_offsets_mapping=True
        )
        offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
        if not offsets:
            raise ContractError(f"empty user body at source row {row_index}")
        accepted: list[tuple[dict[str, Any], dict[str, Any], set[int], str]] = []
        for interval in _token_intervals(
            start=0,
            end=len(offsets),
            window_tokens=primary_window_tokens,
            overlap_tokens=primary_overlap_tokens,
        ):
            chunk_row, metadata, visible = _window_row(
                tokenizer=tokenizer,
                system_message=messages[0],
                body=body,
                offsets=offsets,
                interval=interval,
                references=references,
            )
            if metadata["tokens"] <= max_sequence_length:
                accepted.append((chunk_row, metadata, visible, "primary"))
                continue
            for fallback_interval in _token_intervals(
                start=interval[0],
                end=interval[1],
                window_tokens=FALLBACK_WINDOW_TOKENS,
                overlap_tokens=FALLBACK_OVERLAP_TOKENS,
            ):
                fallback_row, fallback_metadata, fallback_visible = _window_row(
                    tokenizer=tokenizer,
                    system_message=messages[0],
                    body=body,
                    offsets=offsets,
                    interval=fallback_interval,
                    references=references,
                )
                if fallback_metadata["tokens"] > max_sequence_length:
                    raise ContractError(
                        f"fallback chunk tokens={fallback_metadata['tokens']} exceed "
                        f"max_sequence_length={max_sequence_length} at row {row_index}"
                    )
                accepted.append(
                    (fallback_row, fallback_metadata, fallback_visible, "fallback")
                )

        covered_tokens = [False] * len(offsets)
        visible_union: set[int] = set()
        for _, metadata, visible, _ in accepted:
            for token_index in range(metadata["token_start"], metadata["token_end"]):
                covered_tokens[token_index] = True
            visible_union.update(visible)
        empty_indices = [
            index
            for index, (_, metadata, _, _) in enumerate(accepted)
            if metadata["reference_count"] == 0
        ]
        candidate_empty_chunk_count += len(empty_indices)
        kept_empty_index = (
            max(empty_indices, key=lambda index: accepted[index][1]["tokens"])
            if empty_indices
            else None
        )
        kept_indices = {
            index
            for index, (_, metadata, _, _) in enumerate(accepted)
            if metadata["reference_count"] > 0 or index == kept_empty_index
        }
        dropped_empty_chunk_count += len(empty_indices) - int(kept_empty_index is not None)
        for accepted_index, (chunk_row, metadata, _, mode) in enumerate(accepted):
            if accepted_index not in kept_indices:
                continue
            output_rows.append(chunk_row)
            output_doc_ids.append(doc_id)
            chunk_reference_count += metadata["reference_count"]
            empty_chunk_count += metadata["reference_count"] == 0
            maximum_tokens = max(maximum_tokens, metadata["tokens"])
            chunks.append(
                {
                    "output_row_index": len(output_rows) - 1,
                    "source_row_index": row_index,
                    "doc_id": doc_id,
                    "mode": mode,
                    **metadata,
                }
            )
        if not all(covered_tokens):
            raise IntegrityError(f"text coverage gap at source row {row_index}")
        missing_references = sorted(set(range(len(references))) - visible_union)
        for reference_index in missing_references:
            reference = references[reference_index]
            source_text = reference.get("source_text")
            if not isinstance(source_text, str) or not source_text:
                raise IntegrityError(
                    f"reference rescue has no source_text at source row {row_index}"
                )
            char_start = body.find(source_text)
            if char_start < 0:
                raise IntegrityError(
                    f"reference rescue span is absent at source row {row_index}"
                )
            rescue_messages = [
                messages[0],
                {"role": "user", "content": source_text},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        [reference], ensure_ascii=False, separators=(",", ":")
                    ),
                },
            ]
            rescue_tokens = _rendered_token_count(tokenizer, rescue_messages)
            if rescue_tokens > max_sequence_length:
                raise ContractError(
                    f"reference rescue tokens={rescue_tokens} exceed "
                    f"max_sequence_length={max_sequence_length} at row {row_index}"
                )
            output_rows.append({"messages": rescue_messages})
            output_doc_ids.append(doc_id)
            chunk_reference_count += 1
            reference_rescue_count += 1
            maximum_tokens = max(maximum_tokens, rescue_tokens)
            chunks.append(
                {
                    "output_row_index": len(output_rows) - 1,
                    "source_row_index": row_index,
                    "doc_id": doc_id,
                    "mode": "reference_rescue",
                    "char_start": char_start,
                    "char_end": char_start + len(source_text),
                    "reference_count": 1,
                    "tokens": rescue_tokens,
                }
            )
            visible_union.add(reference_index)
        if visible_union != set(range(len(references))):
            missing = sorted(set(range(len(references))) - visible_union)
            raise IntegrityError(
                f"reference coverage gap at source row {row_index}: {missing}"
            )

    if len(output_rows) != len(output_doc_ids) or maximum_tokens > max_sequence_length:
        raise IntegrityError("context-window output failed final coverage bounds")
    write_jsonl_atomic(output_path, output_rows)
    write_json_atomic(output_doc_ids_path, output_doc_ids)
    manifest = {
        "schema_version": 1,
        "request": request,
        "source_document_count": len(rows),
        "native_document_count": native_documents,
        "windowed_document_count": windowed_documents,
        "output_row_count": len(output_rows),
        "original_reference_count": original_reference_count,
        "chunk_reference_count": chunk_reference_count,
        "empty_chunk_count": empty_chunk_count,
        "candidate_empty_chunk_count": candidate_empty_chunk_count,
        "dropped_empty_chunk_count": dropped_empty_chunk_count,
        "reference_rescue_count": reference_rescue_count,
        "maximum_tokens": maximum_tokens,
        "candidate_text_coverage": "complete_before_negative_sampling",
        "training_text_policy": "all_positive_plus_max_one_empty_per_source_document",
        "reference_coverage": "complete",
        "output_jsonl_path": str(output_path.resolve()),
        "output_jsonl_sha256": sha256_file(output_path),
        "output_doc_ids_path": str(output_doc_ids_path.resolve()),
        "output_doc_ids_sha256": sha256_file(output_doc_ids_path),
        "chunks": chunks,
    }
    write_json_atomic(manifest_path, manifest)
    return manifest
