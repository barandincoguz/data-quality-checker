"""Durable same-filesystem writes with validation before atomic promotion."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

from .fingerprints import canonical_json_bytes

PathValidator = Callable[[Path], None]


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_bytes_atomic(
    path: Path,
    payload: bytes,
    *,
    validator: PathValidator | None = None,
    mode: int = 0o600,
) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if validator is not None:
            validator(temporary)
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _validate_json(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        json.load(handle)


def write_json_atomic(
    path: Path,
    payload: Any,
    *,
    validator: PathValidator | None = None,
    mode: int = 0o600,
    pretty: bool = True,
) -> None:
    if pretty:
        serialized = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    else:
        serialized = canonical_json_bytes(payload, newline=True)

    def combined(candidate: Path) -> None:
        _validate_json(candidate)
        if validator is not None:
            validator(candidate)

    write_bytes_atomic(path, serialized, validator=combined, mode=mode)


def write_text_atomic(path: Path, payload: str, *, mode: int = 0o600) -> None:
    write_bytes_atomic(path, payload.encode("utf-8"), mode=mode)


def write_jsonl_atomic(
    path: Path,
    rows: Iterable[Any],
    *,
    validator: PathValidator | None = None,
    mode: int = 0o600,
) -> None:
    serialized_rows = [canonical_json_bytes(row) for row in rows]
    payload = b"\n".join(serialized_rows) + (b"\n" if serialized_rows else b"")

    def validate_jsonl(candidate: Path) -> None:
        with candidate.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL line {line_number}: {exc}") from exc
        if validator is not None:
            validator(candidate)

    write_bytes_atomic(path, payload, validator=validate_jsonl, mode=mode)
