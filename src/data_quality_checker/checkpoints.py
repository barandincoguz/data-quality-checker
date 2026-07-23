"""Atomic, checksummed stateful checkpoint directories."""

from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .atomic import fsync_directory, write_json_atomic
from .errors import FingerprintMismatch, IntegrityError
from .fingerprints import fingerprint_json, sha256_file

CheckpointWriter = Callable[[Path], list[str]]

REQUIRED_STATEFUL_COMPONENTS = frozenset(
    {
        "model_or_adapter",
        "optimizer",
        "scheduler",
        "global_step",
        "python_rng",
        "mlx_rng",
        "data_cursor",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable_random_state(state: tuple[Any, ...]) -> list[Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        return value

    return convert(state)


def restore_random_state(payload: list[Any]) -> tuple[Any, ...]:
    def convert(value: Any) -> Any:
        if isinstance(value, list):
            return tuple(convert(item) for item in value)
        return value

    restored = convert(payload)
    if not isinstance(restored, tuple):
        raise IntegrityError("Python RNG state is malformed")
    return restored


class CheckpointManager:
    def __init__(
        self,
        root: Path,
        *,
        input_fingerprint: str,
        config_fingerprint: str,
        model_fingerprint: str,
    ) -> None:
        self.root = root.resolve()
        self.input_fingerprint = input_fingerprint
        self.config_fingerprint = config_fingerprint
        self.model_fingerprint = model_fingerprint
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, global_update: int) -> Path:
        if global_update <= 0:
            raise ValueError("global_update must be positive")
        return self.root / f"update_{global_update:07d}"

    def save(
        self,
        *,
        global_update: int,
        trainer_state: dict[str, Any],
        writer: CheckpointWriter,
    ) -> Path:
        target = self.path_for(global_update)
        if target.exists():
            manifest = self.verify(target)
            if int(manifest["global_update"]) != global_update:
                raise IntegrityError(f"checkpoint update mismatch at {target}")
            return target

        self.root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(dir=self.root, prefix=f".{target.name}.", suffix=".tmp"))
        try:
            state_payload = {
                **trainer_state,
                "schema_version": 1,
                "global_update": global_update,
                "input_fingerprint": self.input_fingerprint,
                "config_fingerprint": self.config_fingerprint,
                "model_fingerprint": self.model_fingerprint,
            }
            write_json_atomic(temporary / "trainer_state.json", state_payload)
            components = set(writer(temporary))
            components.update({"global_step", "python_rng", "data_cursor"})
            missing = REQUIRED_STATEFUL_COMPONENTS - components
            if missing:
                raise IntegrityError(
                    "checkpoint writer did not provide full state: " + ", ".join(sorted(missing))
                )
            files = []
            for path in sorted(temporary.rglob("*")):
                if not path.is_file() or path.name == "manifest.json":
                    continue
                size = path.stat().st_size
                if size <= 0:
                    raise IntegrityError(f"empty checkpoint component: {path}")
                files.append(
                    {
                        "path": path.relative_to(temporary).as_posix(),
                        "size": size,
                        "sha256": sha256_file(path),
                    }
                )
            manifest = {
                "schema_version": 1,
                "checkpoint_kind": "stateful",
                "created_at": _utc_now(),
                "global_update": global_update,
                "input_fingerprint": self.input_fingerprint,
                "config_fingerprint": self.config_fingerprint,
                "model_fingerprint": self.model_fingerprint,
                "components": sorted(components),
                "files": files,
                "trainer_state_fingerprint": fingerprint_json(state_payload),
            }
            write_json_atomic(temporary / "manifest.json", manifest)
            self._verify_directory(temporary, expected_target=False)
            os.rename(temporary, target)
            fsync_directory(self.root)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        self.verify(target)
        return target

    def _verify_directory(self, path: Path, *, expected_target: bool) -> dict[str, Any]:
        try:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            trainer_state = json.loads(
                (path / "trainer_state.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise IntegrityError(f"checkpoint cannot be parsed at {path}: {exc}") from exc
        if manifest.get("checkpoint_kind") != "stateful":
            raise IntegrityError(f"checkpoint is not labelled stateful: {path}")
        for field, expected in (
            ("input_fingerprint", self.input_fingerprint),
            ("config_fingerprint", self.config_fingerprint),
            ("model_fingerprint", self.model_fingerprint),
        ):
            if manifest.get(field) != expected or trainer_state.get(field) != expected:
                raise FingerprintMismatch(f"checkpoint {field} mismatch at {path}")
        components = set(manifest.get("components", []))
        if not REQUIRED_STATEFUL_COMPONENTS <= components:
            raise IntegrityError(f"checkpoint component manifest is incomplete: {path}")
        if fingerprint_json(trainer_state) != manifest.get("trainer_state_fingerprint"):
            raise IntegrityError(f"trainer state fingerprint mismatch: {path}")
        rows = manifest.get("files")
        if not isinstance(rows, list) or not rows:
            raise IntegrityError(f"checkpoint file manifest is empty: {path}")
        for row in rows:
            candidate = path / str(row.get("path", ""))
            if not candidate.is_file():
                raise IntegrityError(f"checkpoint file missing: {candidate}")
            if candidate.stat().st_size != row.get("size"):
                raise IntegrityError(f"checkpoint file size mismatch: {candidate}")
            if sha256_file(candidate) != row.get("sha256"):
                raise IntegrityError(f"checkpoint SHA256 mismatch: {candidate}")
        return manifest

    def verify(self, path: Path) -> dict[str, Any]:
        return self._verify_directory(path.resolve(), expected_target=True)

    def load_trainer_state(self, path: Path) -> dict[str, Any]:
        self.verify(path)
        return json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))


def make_python_rng_state(rng: random.Random) -> list[Any]:
    return jsonable_random_state(rng.getstate())
