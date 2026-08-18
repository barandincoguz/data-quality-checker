"""Content fingerprints used to reject unsafe resume attempts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(payload: str) -> str:
    return sha256_bytes(payload.encode("utf-8"))


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any, *, newline: bool = False) -> bytes:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return serialized + (b"\n" if newline else b"")


def fingerprint_json(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def directory_manifest(paths: Iterable[Path], *, root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted((item.resolve() for item in paths), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        rows.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def manifest_fingerprint(rows: Iterable[dict[str, Any]]) -> str:
    """Match the repository's `shasum files | shasum` directory convention.

    The richer JSON fingerprint is returned for application contracts. The
    manifest rows themselves are always retained beside it for auditability.
    """

    return fingerprint_json(list(rows))
