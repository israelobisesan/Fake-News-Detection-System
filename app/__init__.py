"""
Flask application factory for the fake news detection web app.

Structure
---------
    app/__init__.py       create_app() - builds and configures the Flask app
    app/routes.py         the URL rules (/ and /predict)
    app/detector.py       loads the saved model and turns text into a prediction
    app/preprocessing.py  shared text cleaning (also used during training)

Flask is imported inside ``create_app()`` rather than at module level. That way
``import app.preprocessing`` - which the training script does - never requires
Flask to be installed.

Run with::

    python run.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Reject oversized submissions before they are read into memory. A news article
# is normally well under 10 KB; 256 KB leaves generous headroom.
MAX_CONTENT_LENGTH = 256 * 1024


def create_app(config: dict | None = None) -> "Flask":  # noqa: F821
    """Create and configure the Flask application."""
    from flask import Flask

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(name)s: %(message)s",
    )

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    # In production this MUST come from the environment. The fallback is a fixed
    # development value so local runs work out of the box; there are no accounts,
    # sessions or credentials in this project, so nothing sensitive depends on it.
    app.config.update(
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me"),
        MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
        # Never let a traceback reach the browser. The error handlers in
        # routes.py log the detail server-side and show a friendly page.
        PROPAGATE_EXCEPTIONS=False,
        JSON_SORT_KEYS=False,
        MODEL_DIR=str(Path(__file__).resolve().parent.parent / "models"),
    )

    if config:
        app.config.update(config)

    from .routes import main as routes_blueprint

    app.register_blueprint(routes_blueprint)

    # Surface a missing model immediately at start-up instead of failing on the
    # first form submission, so the problem is obvious.
    from .detector import ModelNotReadyError, get_detector

    try:
        detector = get_detector()
        app.logger.info("Model ready: %s", detector.describe())
    except ModelNotReadyError as exc:
        app.logger.error("Model unavailable: %s", exc)
        app.logger.error("Run 'python training/train.py' to create the model artifacts.")

    return app