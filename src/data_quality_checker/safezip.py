"""Bounded, non-extracting ZIP reader for untrusted inputs."""

from __future__ import annotations

import json
import re
import stat
import unicodedata
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .config import SecurityConfig
from .errors import UnsafeArchive

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")


@dataclass(frozen=True)
class ZipRecord:
    entry_name: str
    record_index: int
    payload: dict[str, Any]


@dataclass(frozen=True)
class ZipAudit:
    path: Path
    entry_count: int
    file_count: int
    total_uncompressed_bytes: int
    json_entry_count: int


def normalized_entry_name(name: str) -> str:
    replaced = unicodedata.normalize("NFC", name.replace("\\", "/"))
    if replaced.startswith("/") or _WINDOWS_DRIVE_RE.match(replaced):
        raise UnsafeArchive(f"absolute ZIP path is forbidden: {name!r}")
    path = PurePosixPath(replaced)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise UnsafeArchive(f"unsafe ZIP path is forbidden: {name!r}")
    return path.as_posix().casefold()


def validate_zip_info(info: zipfile.ZipInfo, security: SecurityConfig) -> str:
    normalized = normalized_entry_name(info.filename.rstrip("/"))
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    if unix_mode and stat.S_ISLNK(unix_mode):
        raise UnsafeArchive(f"symlink ZIP entry is forbidden: {info.filename!r}")
    if info.flag_bits & 0x1:
        raise UnsafeArchive(f"encrypted ZIP entry is forbidden: {info.filename!r}")
    if info.file_size > security.max_zip_entry_bytes:
        raise UnsafeArchive(
            f"ZIP entry exceeds {security.max_zip_entry_bytes} bytes: {info.filename!r}"
        )
    if info.file_size:
        if info.compress_size <= 0:
            raise UnsafeArchive(f"invalid compressed size for {info.filename!r}")
        ratio = info.file_size / info.compress_size
        if ratio > security.max_zip_compression_ratio:
            raise UnsafeArchive(
                f"ZIP entry compression ratio {ratio:.2f} exceeds "
                f"{security.max_zip_compression_ratio}: {info.filename!r}"
            )
    return normalized


def inspect_archive(path: Path, security: SecurityConfig) -> ZipAudit:
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise UnsafeArchive(f"invalid ZIP archive {path}: {exc}") from exc
    with archive:
        infos = archive.infolist()
        if len(infos) > security.max_zip_entries:
            raise UnsafeArchive(
                f"ZIP has {len(infos)} entries; maximum is {security.max_zip_entries}"
            )
        names: set[str] = set()
        total = 0
        file_count = 0
        json_count = 0
        for info in infos:
            normalized = validate_zip_info(info, security)
            if normalized in names:
                raise UnsafeArchive(f"duplicate normalized ZIP path: {info.filename!r}")
            names.add(normalized)
            if info.is_dir():
                continue
            file_count += 1
            total += info.file_size
            if total > security.max_zip_total_bytes:
                raise UnsafeArchive(
                    f"ZIP total exceeds {security.max_zip_total_bytes} uncompressed bytes"
                )
            if normalized.endswith((".json", ".jsonl")):
                json_count += 1
        if not json_count:
            raise UnsafeArchive("ZIP contains no JSON or JSONL entries")
        return ZipAudit(path.resolve(), len(infos), file_count, total, json_count)


def _objects_from_payload(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                raise UnsafeArchive("JSON record arrays may contain only objects")
            yield item
        return
    if not isinstance(payload, dict):
        raise UnsafeArchive("JSON root must be an object or array of objects")
    for key in ("annotations", "documents", "records", "items", "data"):
        nested = payload.get(key)
        if isinstance(nested, list):
            for item in nested:
                if not isinstance(item, dict):
                    raise UnsafeArchive(f"{key} may contain only objects")
                yield item
            return
    yield payload


def read_json_records(path: Path, security: SecurityConfig) -> tuple[ZipAudit, list[ZipRecord]]:
    audit = inspect_archive(path, security)
    records: list[ZipRecord] = []
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise UnsafeArchive(f"invalid ZIP archive {path}: {exc}") from exc
    with archive:
        for info in sorted(
            archive.infolist(), key=lambda item: normalized_entry_name(item.filename.rstrip("/"))
        ):
            if info.is_dir():
                continue
            normalized = normalized_entry_name(info.filename)
            if not normalized.endswith((".json", ".jsonl")):
                continue
            try:
                with archive.open(info, "r") as handle:
                    raw = handle.read(security.max_zip_entry_bytes + 1)
            except (RuntimeError, OSError, zipfile.BadZipFile) as exc:
                raise UnsafeArchive(f"cannot read ZIP entry {info.filename!r}: {exc}") from exc
            if len(raw) > security.max_zip_entry_bytes:
                raise UnsafeArchive(f"ZIP entry grew past limit: {info.filename!r}")
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise UnsafeArchive(f"ZIP JSON is not UTF-8: {info.filename!r}") from exc
            if normalized.endswith(".jsonl"):
                payloads: list[Any] = []
                for line_number, line in enumerate(text.splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        payloads.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise UnsafeArchive(
                            f"invalid JSONL {info.filename!r}:{line_number}: {exc}"
                        ) from exc
            else:
                try:
                    payloads = [json.loads(text)]
                except json.JSONDecodeError as exc:
                    raise UnsafeArchive(f"invalid JSON {info.filename!r}: {exc}") from exc
            index = 0
            for payload in payloads:
                for item in _objects_from_payload(payload):
                    records.append(ZipRecord(info.filename, index, item))
                    index += 1
    return audit, records
