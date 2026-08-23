"""Flask application factory; review routes are registered lazily."""

from __future__ import annotations

import os
from typing import Any

from flask import Flask, jsonify

from .config import AppConfig
from .errors import ConfigurationError
from .storage import Store

PUBLISHED_DEMO_SECRETS = frozenset(
    {
        "demo_session_secret_0123456789abcdef0123456789",
        "demo_access_token_0123456789abcdef0123456789",
    }
)


def validate_runtime_secret(value: str | None, *, env_name: str) -> str:
    candidate = value.strip() if value else ""
    if len(candidate) < 32:
        raise ConfigurationError(f"{env_name} must contain at least 32 characters")
    folded = candidate.casefold()
    if any(secret.casefold() in folded for secret in PUBLISHED_DEMO_SECRETS):
        raise ConfigurationError(f"{env_name} must not use the published demo value")
    return candidate


def create_app(
    config: AppConfig,
    *,
    store: Store | None = None,
    batch_id: str | None = None,
    testing: bool = False,
    session_secret: str | None = None,
) -> Flask:
    secret = session_secret or os.environ.get("DQCHECK_SESSION_SECRET")
    if secret is not None or not testing:
        secret = validate_runtime_secret(secret, env_name="DQCHECK_SESSION_SECRET")

    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=secret or "test-only-secret",
        TESTING=testing,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
        SESSION_COOKIE_SECURE=False,
        DQCHECK_BATCH_ID=batch_id,
        DQCHECK_DATABASE=str(config.database_path),
    )
    app.extensions["dqcheck_store"] = store

    @app.get("/healthz")
    def healthz() -> Any:
        return jsonify(
            {
                "status": "ok",
                "batch_id": app.config["DQCHECK_BATCH_ID"],
                "schema_version": config.schema_version,
            }
        )

    return app
