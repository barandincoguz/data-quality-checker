from __future__ import annotations

import pytest

from data_quality_checker.cli import build_parser
from data_quality_checker.config import load_config
from data_quality_checker.errors import ConfigurationError
from data_quality_checker.web import create_app


def test_help_lists_every_public_command() -> None:
    help_text = build_parser().format_help()
    for command in (
        "prepare",
        "train-bootstrap",
        "process",
        "pilot-judges",
        "judge-lock",
        "serve",
        "release",
        "status",
    ):
        assert command in help_text


def test_app_factory_requires_session_secret_outside_tests() -> None:
    with pytest.raises(ConfigurationError, match="SESSION_SECRET"):
        create_app(load_config(), session_secret=None)


def test_health_route(tmp_path) -> None:
    app = create_app(load_config(), testing=True, batch_id="fixture")
    response = app.test_client().get("/healthz")
    assert response.status_code == 200
    assert response.get_json()["batch_id"] == "fixture"
