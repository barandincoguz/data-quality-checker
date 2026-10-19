"""Lossless, fingerprinted training windows for Metal-bounded G0 LoRA."""

from __future__ import annotations

import hashlib
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
DENSE_REPLAY_MAX_REFERENCES = 10
# Keep the namespace frozen so larger dose probes are strict supersets of the
# half-dose document set instead of drawing a new random-looking sample.
DENSE_REPLAY_SAMPLE_NAMESPACE = "dense-replay-half-v1"
DENSE_REPLAY_SAMPLE_THRESHOLD = 256


def _include_dense_replay(doc_id: int | str) -> bool:
    """Select a stable, order-independent replay subset.

    The id is hashed exactly as written. A canonical document is an integer
    here and in G0's own view, so its membership is identical across
    generations; a round-cleaned document carries its own string id and joins
    the sample on the same rule.
    """

    digest = hashlib.sha256(f"{DENSE_REPLAY_SAMPLE_NAMESPACE}:{doc_id}".encode()).digest()
    return digest[0] < DENSE_REPLAY_SAMPLE_THRESHOLD


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


_LOCATING_FIELDS = frozenset({"source_text", "source_texts"})


def _without_source_text(reference: dict[str, Any]) -> dict[str, Any]:
    """Drop annotator source_text before it reaches the assistant target.

    source_text still locates a reference inside the ruling while a window
    is being built; it must never appear in what the model is trained to
    emit, so every assistant-content JSON dump strips it here first.
    `source_texts` is the same evidence for a reference cited in several
    places (GT v7 keeps every quote of the rows it merged into one law and
    article); it is stripped the same way.
    """

    return {field: value for field, value in reference.items() if field not in _LOCATING_FIELDS}


def _quotes(reference: dict[str, Any]) -> list[str]:
    """Every passage that locates a reference: all of `source_texts`, else `source_text`.

    One law-and-article reference is often cited at several places in a ruling
    (once per paragraph it names, before 1 Oct 2026 a row each). A window that
    contains any of them contains the reference; reading only one would teach
    the model that the other windows do not cite it.
    """

    many = reference.get("source_texts")
    quotes = many if isinstance(many, list) else [reference.get("source_text")]
    return [quote.strip() for quote in quotes if isinstance(quote, str) and quote.strip()]


def _first_located(body: str, reference: dict[str, Any]) -> tuple[int, str] | None:
    """The earliest place in `body` any quote of the reference occurs."""

    found = [(body.find(quote), quote) for quote in _quotes(reference) if quote in body]
    return min(found) if found else None


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


def _token_span_for_chars(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int, int]:
    """Smallest token interval fully covering a character range."""

    start = next(
        (index for index, (_, end) in enumerate(offsets) if end > char_start),
        max(0, len(offsets) - 1),
    )
    end = next(
        (index + 1 for index in range(len(offsets) - 1, -1, -1) if offsets[index][0] < char_end),
        start + 1,
    )
    return start, max(end, start + 1)


def _expanded_interval(start: int, end: int, total: int, minimum_tokens: int) -> tuple[int, int]:
    """Grow a token interval to a minimum width, inside the body, still
    containing the interval it was given."""

    width = min(total, max(minimum_tokens, end - start))
    pad = width - (end - start)
    begin = max(0, min(start - pad // 2, total - width))
    return begin, begin + width


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
        if any(quote in text for quote in _quotes(reference))
    }
    visible = [references[index] for index in sorted(visible_indices)]
    messages = [
        system_message,
        {"role": "user", "content": text.rstrip()},
        {
            "role": "assistant",
            "content": json.dumps(
                [_without_source_text(reference) for reference in visible],
                ensure_ascii=False,
                separators=(",", ":"),
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


def _dense_replay_rows(
    *,
    tokenizer: Any,
    system_message: dict[str, str],
    body: str,
    references: list[dict[str, Any]],
    max_sequence_length: int,
) -> list[tuple[dict[str, Any], int, int]]:
    """Replay a document's references over the real passages that carry them.

    The pack used to be served as the annotator's `source_text` spans joined
    by newlines. That input never occurs at inference -- the model is given
    continuous ruling prose, not a list of clipped phrases -- and the
    annotators clip from several places in a document, so the joined blob was
    synthetic text carrying a many-reference answer. On the sealed
    300-document view it was 324 of 2022 training rows.

    `source_text` now only locates each reference; the row serves the
    shortest real passage that spans its pack. Measured on the same 300
    documents: 2041 reference instances survive, across 506 rows with a
    median pack of 4 and a median passage of 1834 characters. That is 34
    fewer reference instances than an earlier measurement of 2075: the
    native/windowed sizing fix above now keeps seven documents (ids 9, 109,
    166, 213, 363, 443, 477) whole, and a native document is served entire
    and never replayed. Reference coverage stays complete regardless --
    those 34 are taught by their native row, not lost.
    """

    located: list[tuple[int, int, dict[str, Any]]] = []
    for reference_index, reference in enumerate(references):
        if not _quotes(reference):
            raise IntegrityError(f"dense replay reference {reference_index} has no source_text")
        place = _first_located(body, reference)
        if place is None:
            raise IntegrityError(
                f"dense replay reference {reference_index} is absent from source text"
            )
        char_start, source_text = place
        located.append((char_start, char_start + len(source_text), reference))
    located.sort(key=lambda item: (item[0], item[1]))

    def render(
        pack: list[tuple[int, int, dict[str, Any]]],
    ) -> tuple[dict[str, Any], int]:
        passage = body[min(start for start, _, _ in pack) : max(end for _, end, _ in pack)]
        messages = [
            system_message,
            {"role": "user", "content": passage},
            {
                "role": "assistant",
                "content": json.dumps(
                    [_without_source_text(reference) for _, _, reference in pack],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
        return {"messages": messages}, _rendered_token_count(tokenizer, messages)

    packed: list[tuple[dict[str, Any], int, int]] = []
    index = 0
    while index < len(located):
        pack = [located[index]]
        row, tokens = render(pack)
        if tokens > max_sequence_length:
            raise ContractError(
                f"dense replay row tokens={tokens} exceed max_sequence_length={max_sequence_length}"
            )
        cursor = index + 1
        while cursor < len(located) and len(pack) < DENSE_REPLAY_MAX_REFERENCES:
            candidate_pack = [*pack, located[cursor]]
            candidate_row, candidate_tokens = render(candidate_pack)
            if candidate_tokens > max_sequence_length:
                break
            pack, row, tokens = candidate_pack, candidate_row, candidate_tokens
            cursor += 1
        packed.append((row, tokens, len(pack)))
        index = cursor
    return packed


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
    primary_window_tokens, primary_overlap_tokens = _primary_window_geometry(max_sequence_length)
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
        "reference_rescue": "real_window_v2",
        "empty_chunk_sampling": "true_negative_documents_only_v2",
        "dense_replay": "real_passage_span_pack_max10_full_v6",
        "dense_replay_sampling": {
            "policy": "sha256_doc_id_first_byte_threshold_v1",
            "namespace": DENSE_REPLAY_SAMPLE_NAMESPACE,
            "threshold": DENSE_REPLAY_SAMPLE_THRESHOLD,
            "possible_values": 256,
        },
        "thinking_control": "chat_template_enable_thinking_false",
        "reference_location": "any_quote_of_source_texts_v1",
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
    output_doc_ids: list[int | str] = []
    chunks: list[dict[str, Any]] = []
    native_documents = 0
    windowed_documents = 0
    original_reference_count = 0
    chunk_reference_count = 0
    empty_chunk_count = 0
    candidate_empty_chunk_count = 0
    dropped_empty_chunk_count = 0
    reference_rescue_count = 0
    dense_replay_eligible_document_count = 0
    dense_replay_document_count = 0
    dense_replay_skipped_document_count = 0
    dense_replay_row_count = 0
    dense_replay_reference_count = 0
    dense_replay_max_reference_count = 0
    maximum_tokens = 0

    for row_index, (row, raw_doc_id) in enumerate(zip(rows, doc_ids, strict=True)):
        messages = row.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 3
            or [message.get("role") for message in messages] != ["system", "user", "assistant"]
        ):
            raise ContractError(f"unexpected chat contract at source row {row_index}")
        # The id is a key, not a number. Canonical documents are integers
        # because the ground-truth directory resolves them that way
        # (`doc_<id>.json`); documents cleaned by a DQ-Loop round arrive as
        # strings like `d000001`. Coercing either into the other would break
        # one of the two, so it is carried through exactly as given.
        if not isinstance(raw_doc_id, int | str) or (
            isinstance(raw_doc_id, str) and not raw_doc_id
        ):
            raise ContractError(f"source row {row_index} has an unusable document id")
        doc_id = raw_doc_id
        references = _compact_references(messages[2]["content"])
        original_reference_count += len(references)
        # Size the row on exactly what the model is served. `source_text` is
        # stripped from every assistant target below, so counting it here
        # spent budget on a field the model never sees -- a median 132 tokens
        # per document on the sealed 300-document split, which pushed 7 of
        # them out of a whole-document row and into windows.
        served_messages = [
            messages[0],
            messages[1],
            {
                "role": "assistant",
                "content": json.dumps(
                    [_without_source_text(reference) for reference in references],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
        full_tokens = _rendered_token_count(tokenizer, served_messages)
        if full_tokens <= max_sequence_length:
            native_documents += 1
            output_rows.append({"messages": served_messages})
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
            raise ContractError(f"unsupported literal /no_think suffix at source row {row_index}")
        body = user_content.rstrip()
        encoded = offset_tokenizer(body, add_special_tokens=False, return_offsets_mapping=True)
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
                accepted.append((fallback_row, fallback_metadata, fallback_visible, "fallback"))

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
        # Canonical rows are document-level reference sets, not exhaustive
        # mention annotations.  A window from a positive document can contain
        # an explicit repeated citation whose chosen canonical source_text is
        # elsewhere; labelling that window [] is therefore a false negative.
        # Preserve a negative window only when the whole source document is a
        # genuine negative.
        kept_empty_index = (
            max(empty_indices, key=lambda index: accepted[index][1]["tokens"])
            if empty_indices and not references
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
            if reference_index in visible_union:
                continue
            reference = references[reference_index]
            if not _quotes(reference):
                raise IntegrityError(
                    f"reference rescue has no source_text at source row {row_index}"
                )
            place = _first_located(body, reference)
            if place is None:
                raise IntegrityError(f"reference rescue span is absent at source row {row_index}")
            char_start, source_text = place
            # The span locates the passage; the passage is what is served.
            # A rescue row used to put the annotator's bare span in the user
            # turn -- text that never occurs at inference, from a field the
            # system prompt does not name.
            span_start, span_end = _token_span_for_chars(
                offsets, char_start, char_start + len(source_text)
            )
            interval = _expanded_interval(
                span_start, span_end, len(offsets), FALLBACK_WINDOW_TOKENS
            )
            rescue_row, rescue_metadata, rescue_visible = _window_row(
                tokenizer=tokenizer,
                system_message=messages[0],
                body=body,
                offsets=offsets,
                interval=interval,
                references=references,
            )
            if rescue_metadata["tokens"] > max_sequence_length:
                raise ContractError(
                    f"reference rescue tokens={rescue_metadata['tokens']} exceed "
                    f"max_sequence_length={max_sequence_length} at row {row_index}"
                )
            if reference_index not in rescue_visible:
                raise IntegrityError(
                    f"reference rescue window does not contain reference "
                    f"{reference_index} at source row {row_index}"
                )
            output_rows.append(rescue_row)
            output_doc_ids.append(doc_id)
            chunk_reference_count += rescue_metadata["reference_count"]
            reference_rescue_count += 1
            maximum_tokens = max(maximum_tokens, rescue_metadata["tokens"])
            chunks.append(
                {
                    "output_row_index": len(output_rows) - 1,
                    "source_row_index": row_index,
                    "doc_id": doc_id,
                    "mode": "reference_rescue",
                    **rescue_metadata,
                }
            )
            visible_union.update(rescue_visible)
        if visible_union != set(range(len(references))):
            missing = sorted(set(range(len(references))) - visible_union)
            raise IntegrityError(f"reference coverage gap at source row {row_index}: {missing}")
        dense_replay_eligible_document_count += 1
        if not _include_dense_replay(doc_id):
            dense_replay_skipped_document_count += 1
            continue
        dense_replay_document_count += 1
        replay_rows = _dense_replay_rows(
            tokenizer=tokenizer,
            system_message=messages[0],
            body=body,
            references=references,
            max_sequence_length=max_sequence_length,
        )
        for replay_row, replay_tokens, replay_reference_count in replay_rows:
            output_rows.append(replay_row)
            output_doc_ids.append(doc_id)
            dense_replay_row_count += 1
            dense_replay_reference_count += replay_reference_count
            dense_replay_max_reference_count = max(
                dense_replay_max_reference_count, replay_reference_count
            )
            maximum_tokens = max(maximum_tokens, replay_tokens)
            chunks.append(
                {
                    "output_row_index": len(output_rows) - 1,
                    "source_row_index": row_index,
                    "doc_id": doc_id,
                    "mode": "reference_dense_replay",
                    "tokens": replay_tokens,
                    "reference_count": replay_reference_count,
                }
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
        "dense_replay_eligible_document_count": (dense_replay_eligible_document_count),
        "dense_replay_document_count": dense_replay_document_count,
        "dense_replay_skipped_document_count": (dense_replay_skipped_document_count),
        "dense_replay_row_count": dense_replay_row_count,
        "dense_replay_reference_count": dense_replay_reference_count,
        "dense_replay_max_reference_count": dense_replay_max_reference_count,
        "training_reference_count": (chunk_reference_count + dense_replay_reference_count),
        "maximum_tokens": maximum_tokens,
        "candidate_text_coverage": "complete_before_negative_sampling",
        "training_text_policy": ("all_positive_plus_true_negative_only_plus_full_dense_replay"),
        "reference_coverage": "complete",
        "output_jsonl_path": str(output_path.resolve()),
        "output_jsonl_sha256": sha256_file(output_path),
        "output_doc_ids_path": str(output_doc_ids_path.resolve()),
        "output_doc_ids_sha256": sha256_file(output_doc_ids_path),
        "chunks": chunks,
    }
    write_json_atomic(manifest_path, manifest)
    return manifest
