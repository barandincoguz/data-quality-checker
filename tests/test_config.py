from __future__ import annotations

import json

import pytest

from data_quality_checker.config import default_config_path, load_config
from data_quality_checker.constants import MODEL_ID, MODEL_REVISION
from data_quality_checker.errors import ConfigurationError


def test_default_config_resolves_repo_paths_and_pins_model() -> None:
    config = load_config()

    assert config.canonical_gt_dir.is_dir()
    assert config.example_bank_path.is_file()
    assert config.model.model_id == MODEL_ID
    assert config.model.revision == MODEL_REVISION
    assert config.model.enable_thinking is False
    assert config.model.prompt_suffix == "/no_think"
    assert len(config.fingerprint) == 64


def test_model_revision_drift_is_rejected(tmp_path) -> None:
    payload = json.loads(default_config_path().read_text(encoding="utf-8"))
    payload["model"]["revision"] = "moving-main"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="drift"):
        load_config(path)
