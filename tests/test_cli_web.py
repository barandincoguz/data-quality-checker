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
        "import-attribution",
        "train-bootstrap",
        "train-g0",
        "process",
        "pilot-judges",
        "judge-lock",
        "review-backup",
        "serve",
        "release",
        "status",
    ):
        assert command in help_text


def test_app_factory_requires_session_secret_outside_tests() -> None:
    with pytest.raises(ConfigurationError, match="SESSION_SECRET"):
        create_app(load_config(), session_secret=None)


@pytest.mark.parametrize(
    "secret",
    [
        "too-short",
        "demo_session_secret_0123456789abcdef0123456789",
        "demo_access_token_0123456789abcdef0123456789",
    ],
)
def test_app_factory_rejects_weak_or_published_secrets(secret: str) -> None:
    with pytest.raises(ConfigurationError):
        create_app(load_config(), session_secret=secret)


def test_health_route(tmp_path) -> None:
    app = create_app(load_config(), testing=True, batch_id="fixture")
    response = app.test_client().get("/healthz")
    assert response.status_code == 200
    assert response.get_json()["batch_id"] == "fixture"
