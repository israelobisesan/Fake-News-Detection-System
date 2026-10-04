"""
URL routes.

    GET  /          the detection form
    POST /predict   validate, classify, show the result
    GET  /health    machine-readable status check

Input safety
------------
Form values arrive as strings and are inserted into the page only through
Jinja templates, which escape HTML automatically. Submitted text is never
evaluated, never used to build a file path, and never logged in full.

Error handling
--------------
Missing model artifacts, malformed input and unexpected failures all produce a
friendly page. The underlying exception is written to the server log via
``app.logger.exception`` so the detail is available during development without
ever reaching the browser.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from flask import Blueprint, current_app, render_template, request

from .detector import ModelNotReadyError, PredictionError, get_detector

logger = logging.getLogger(__name__)

main = Blueprint("main", __name__)

# Validation limits for the form.
TITLE_MIN_LENGTH = 3
ARTICLE_MIN_LENGTH = 20
MAX_TITLE_LENGTH = 300


def _read_field(name: str) -> str:
    """Return a trimmed form field, treating None as an empty string."""
    return (request.form.get(name) or "").strip()


@main.route("/")
def index():
    """Display the detection form."""
    return render_template("index.html", form={}, errors={})


@main.route("/predict", methods=["POST"])
def predict():
    """Classify a submitted article."""
    title = _read_field("title")
    article = _read_field("article")

    # --- validate ---------------------------------------------------------
    errors: dict[str, str] = {}

    if not title and not article:
        errors["form"] = "Please enter a headline and the news article before analysing."
    else:
        if not title:
            errors["title"] = "Please enter a headline."
        elif len(title) < TITLE_MIN_LENGTH:
            errors["title"] = f"The headline must be at least {TITLE_MIN_LENGTH} characters long."

        if not article:
            errors["article"] = "Please paste the news article text."
        elif len(article) < ARTICLE_MIN_LENGTH:
            errors["article"] = (
                f"The article must be at least {ARTICLE_MIN_LENGTH} characters long "
                f"(currently {len(article)})."
            )

    if len(title) > MAX_TITLE_LENGTH:
        errors["title"] = f"The headline must be under {MAX_TITLE_LENGTH} characters."

    if errors:
        return render_template("index.html", form={"title": title, "article": article},
                               errors=errors), 400

    # --- classify ---------------------------------------------------------
    try:
        detector = get_detector(current_app.config.get("MODEL_DIR"))
        result = detector.predict(title, article)
    except ModelNotReadyError:
        current_app.logger.exception("Model artifacts are unavailable")
        return render_template(
            "error.html",
            title="Model not available",
            message=(
                "The trained model could not be loaded, so analysis is temporarily "
                "unavailable. If you are running this locally, create the model "
                "artifacts first with: python training/train.py"
            ),
        ), 503
    except PredictionError:
        current_app.logger.exception("Prediction failed")
        return render_template(
            "error.html",
            title="Could not analyse this article",
            message=(
                "Something went wrong while analysing the text you submitted. "
                "Please check the article and try again."
            ),
        ), 500
    except Exception:  # noqa: BLE001 - last line of defence, never leak a traceback
        current_app.logger.exception("Unexpected error while predicting")
        return render_template(
            "error.html",
            title="Unexpected error",
            message="An unexpected error occurred. The problem has been logged for review.",
        ), 500

    return render_template(
        "result.html",
        result=result,
        title_text=title,
        article_text=article,
        model_name=detector.describe(),
        analysed_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )


@main.route("/health")
def health():
    """Simple status endpoint - handy for checking the app is alive."""
    try:
        detector = get_detector(current_app.config.get("MODEL_DIR"))
    except ModelNotReadyError as exc:
        return {"status": "unavailable", "detail": str(exc)}, 503
    return {
        "status": "ok",
        "model": detector.metadata.get("model_name", "unknown"),
        "labels": detector.classes_,
        "test_accuracy": detector.metadata.get("test_accuracy"),
    }


@main.app_errorhandler(413)
def too_large(_error):
    """Fired when a submission exceeds MAX_CONTENT_LENGTH."""
    return render_template(
        "error.html",
        title="Submission too large",
        message=(
            "The text you submitted is too large to process. Please shorten it and "
            "try again - a typical news article is far smaller than the limit."
        ),
    ), 413


@main.app_errorhandler(404)
def not_found(_error):
    return render_template(
        "error.html",
        title="Page not found",
        message="The page you asked for does not exist.",
    ), 404


@main.app_errorhandler(500)
def server_error(_error):
    return render_template(
        "error.html",
        title="Server error",
        message="Something went wrong on our side. The problem has been logged.",
    ), 500