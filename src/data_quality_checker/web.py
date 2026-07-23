"""Flask application factory; review routes are registered lazily."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify

from .config import AppConfig
from .errors import ConfigurationError
from .storage import Store


def create_app(
    config: AppConfig,
    *,
    store: Store | None = None,
    batch_id: str | None = None,
    testing: bool = False,
    session_secret: str | None = None,
) -> Flask:
    secret = session_secret or os.environ.get("DQCHECK_SESSION_SECRET")
    if not secret and not testing:
        raise ConfigurationError("DQCHECK_SESSION_SECRET is required for the HITL server")

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
