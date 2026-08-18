"""JSON configuration loading with explicit path and model validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import MINIMUM_MLX_LM, MODEL_ID, MODEL_REVISION, SCHEMA_VERSION
from .errors import ConfigurationError
from .fingerprints import fingerprint_json


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    revision: str
    minimum_mlx_lm: str
    enable_thinking: bool
    prompt_suffix: str


@dataclass(frozen=True)
class SecurityConfig:
    max_zip_entries: int
    max_zip_entry_bytes: int
    max_zip_total_bytes: int
    max_zip_compression_ratio: float


@dataclass(frozen=True)
class RuntimeConfig:
    busy_timeout_ms: int
    heartbeat_stale_seconds: int
    prediction_flush_every: int


@dataclass(frozen=True)
class AppConfig:
    source_path: Path
    schema_version: int
    canonical_gt_dir: Path
    example_bank_path: Path
    reference_split_manifest_path: Path
    sensitive_root: Path
    public_root: Path
    training_runs_root: Path
    model: ModelConfig
    security: SecurityConfig
    runtime: RuntimeConfig
    raw: dict[str, Any]
    fingerprint: str

    @property
    def database_path(self) -> Path:
        return self.sensitive_root / "dqcheck.sqlite3"


def default_config_path() -> Path:
    candidate = Path(__file__).resolve().parents[2] / "configs" / "default.json"
    if candidate.is_file():
        return candidate
    cwd_candidate = Path.cwd() / "configs" / "default.json"
    if cwd_candidate.is_file():
        return cwd_candidate
    return candidate


def _positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"{key} must be a positive integer")
    return value


def _resolved(base: Path, value: Any, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{key} must be a non-empty path string")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path: Path | str | None = None) -> AppConfig:
    source = Path(path).resolve() if path else default_config_path().resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot load config {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError("config root must be an object")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ConfigurationError(
            f"unsupported schema_version={raw.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )

    model = raw.get("model")
    security = raw.get("security")
    runtime = raw.get("runtime")
    if (
        not isinstance(model, dict)
        or not isinstance(security, dict)
        or not isinstance(runtime, dict)
    ):
        raise ConfigurationError("model, security, and runtime must be objects")
    if model.get("id") != MODEL_ID or model.get("revision") != MODEL_REVISION:
        raise ConfigurationError(
            "model id/revision drift is forbidden; start a new reviewed config version"
        )
    if model.get("minimum_mlx_lm") != MINIMUM_MLX_LM:
        raise ConfigurationError("minimum_mlx_lm does not match the v1 contract")
    if model.get("enable_thinking") is not False or model.get("prompt_suffix") != "":
        raise ConfigurationError(
            "Qwen3.5 thinking must be disabled by the chat template without a literal suffix"
        )

    base = source.parent
    security_config = SecurityConfig(
        max_zip_entries=_positive_int(security, "max_zip_entries"),
        max_zip_entry_bytes=_positive_int(security, "max_zip_entry_bytes"),
        max_zip_total_bytes=_positive_int(security, "max_zip_total_bytes"),
        max_zip_compression_ratio=float(security.get("max_zip_compression_ratio", 0)),
    )
    if security_config.max_zip_compression_ratio <= 0:
        raise ConfigurationError("max_zip_compression_ratio must be positive")
    runtime_config = RuntimeConfig(
        busy_timeout_ms=_positive_int(runtime, "busy_timeout_ms"),
        heartbeat_stale_seconds=_positive_int(runtime, "heartbeat_stale_seconds"),
        prediction_flush_every=_positive_int(runtime, "prediction_flush_every"),
    )
    return AppConfig(
        source_path=source,
        schema_version=SCHEMA_VERSION,
        canonical_gt_dir=_resolved(base, raw.get("canonical_gt_dir"), "canonical_gt_dir"),
        example_bank_path=_resolved(base, raw.get("example_bank_path"), "example_bank_path"),
        reference_split_manifest_path=_resolved(
            base,
            raw.get("reference_split_manifest_path"),
            "reference_split_manifest_path",
        ),
        sensitive_root=_resolved(base, raw.get("sensitive_root"), "sensitive_root"),
        public_root=_resolved(base, raw.get("public_root"), "public_root"),
        training_runs_root=_resolved(base, raw.get("training_runs_root"), "training_runs_root"),
        model=ModelConfig(
            model_id=str(model["id"]),
            revision=str(model["revision"]),
            minimum_mlx_lm=str(model["minimum_mlx_lm"]),
            enable_thinking=False,
            prompt_suffix="",
        ),
        security=security_config,
        runtime=runtime_config,
        raw=raw,
        fingerprint=fingerprint_json(raw),
    )
