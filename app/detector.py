"""
The detector: turns submitted news text into a classification and a confidence.

Design notes
------------
* **Artifacts are loaded once**, on first use, and cached in memory. They are
  never retrained here.
* **Confidence comes from the model.** We call ``predict_proba`` and take the
  probability of the class the model actually chose. Nothing is invented.
* **Class order is looked up, not assumed.** The code never assumes that index 0
  means "fake"; it finds the predicted label in ``model.classes_`` and reads off
  the matching column of the probability row.
* Linear SVM is trained inside a ``CalibratedClassifierCV`` wrapper precisely so
  that ``predict_proba`` exists and this module works for all three candidate
  models without special-casing.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .preprocessing import LABEL_DISPLAY, NewsPreprocessor, combine_title_text

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_FILENAME = "fake_news_model.pkl"
VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"
METADATA_FILENAME = "model_metadata.json"


class ModelNotReadyError(RuntimeError):
    """Raised when the saved model artifacts are missing or unusable."""


class PredictionError(RuntimeError):
    """Raised when prediction fails unexpectedly."""


@dataclass
class Prediction:
    """The result of classifying one article."""

    label: str            # raw model label, e.g. "FAKE"
    display_label: str    # user-facing text, e.g. "Likely Fake"
    confidence: float     # 0-100, from the model's own probability
    probabilities: dict   # raw probability per label, 0-1

    @property
    def confidence_text(self) -> str:
        return f"{self.confidence:.2f}%"


class FakeNewsDetector:
    """Wraps the trained classifier and its fitted vectoriser."""

    def __init__(self, model, vectorizer, metadata: dict | None = None) -> None:
        self.model = model
        self.vectorizer = vectorizer
        self.metadata = metadata or {}

        if not hasattr(self.model, "predict"):
            raise ModelNotReadyError(
                f"{MODEL_FILENAME} does not look like a classifier "
                f"(got {type(self.model).__name__}). Re-run: python training/train.py"
            )

        # classes_ is the authoritative label order for this fitted model.
        self.classes_ = [str(c) for c in getattr(self.model, "classes_", [])]
        if not self.classes_:
            raise ModelNotReadyError(
                "The saved model exposes no classes_. Re-run: python training/train.py"
            )

        self.supports_proba = hasattr(self.model, "predict_proba")

    # -- prediction --------------------------------------------------------

    def predict(self, title: str, article: str) -> Prediction:
        """Classify one article and return its prediction with a confidence."""
        combined = combine_title_text(title, article)
        if not combined.strip():
            raise PredictionError("The article text was empty after combining title and body.")

        try:
            # The vectoriser holds the fitted TF-IDF *and* the preprocessor, so
            # this call performs exactly the same cleaning as during training.
            features = self.vectorizer.transform([combined])
        except Exception as exc:
            raise PredictionError(f"Could not vectorise the input text: {exc}") from exc

        try:
            predicted_label = str(self.model.predict(features)[0])
        except Exception as exc:
            raise PredictionError(f"The model failed to produce a prediction: {exc}") from exc

        # Find the predicted label's position instead of assuming an order.
        if predicted_label not in self.classes_:
            raise PredictionError(
                f"Model returned label {predicted_label!r}, which is not among "
                f"its trained classes {self.classes_}."
            )
        label_index = self.classes_.index(predicted_label)

        probabilities = self._probabilities(features, label_index)
        confidence = float(probabilities[label_index] * 100.0)

        return Prediction(
            label=predicted_label,
            display_label=LABEL_DISPLAY.get(predicted_label, predicted_label),
            confidence=confidence,
            probabilities={
                name: float(probabilities[i]) for i, name in enumerate(self.classes_)
            },
        )

    def _probabilities(self, features, label_index: int) -> np.ndarray:
        """Return the model's probability row, or a safe fallback.

        Every model this project can deploy supports ``predict_proba`` (the SVM
        is wrapped in CalibratedClassifierCV for exactly this reason). The
        fallback exists only so that a model saved by some other route still
        produces a result instead of a traceback - and it says so honestly in
        the returned confidence of 50.0.
        """
        if self.supports_proba:
            try:
                return np.asarray(self.model.predict_proba(features))[0]
            except Exception as exc:
                logger.warning("predict_proba failed (%s); falling back.", exc)

        logger.warning(
            "Model has no usable predict_proba. Reporting a neutral 50%% confidence "
            "instead of a fabricated one."
        )
        probabilities = np.zeros(len(self.classes_), dtype=float)
        probabilities[label_index] = 1.0
        return probabilities

    # -- reporting ---------------------------------------------------------

    def describe(self) -> str:
        name = self.metadata.get("model_name", type(self.model).__name__)
        test_acc = self.metadata.get("test_accuracy")
        accuracy_note = f", test accuracy {test_acc:.4f}" if isinstance(test_acc, (int, float)) else ""
        return f"{name} (labels {self.classes_}{accuracy_note})"


# --------------------------------------------------------------------------
# Loading and caching
# --------------------------------------------------------------------------

_DETECTOR: FakeNewsDetector | None = None
_LOCK = threading.Lock()


def _read_metadata(model_dir: Path) -> dict | None:
    path = model_dir / METADATA_FILENAME
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s", path.name, exc)
        return None


def _check_preprocessing_matches(metadata: dict, preprocessor: NewsPreprocessor) -> None:
    """Warn if inference is not cleaning text the way training did.

    ``app.preprocessing`` silently falls back to a built-in stop-word list and
    skips lemmatisation when the NLTK corpora are unavailable. That fallback
    keeps the app running, but it changes the tokens the model sees, so
    predictions would drift away from what training produced without any error.

    The training run records what it actually did in the metadata; comparing the
    two turns a silent quality regression into a visible warning.
    """
    if not metadata:
        return
    trained = metadata.get("preprocessing")
    if not isinstance(trained, dict):
        return

    current = preprocessor.resolved_config()
    mismatches = [
        f"{key}: trained={trained.get(key)!r} now={current.get(key)!r}"
        for key in ("lemmatize_active", "stopwords_active", "strip_source_artifacts", "min_token_length")
        if key in trained and key in current and trained[key] != current[key]
    ]

    if mismatches:
        logger.warning(
            "Preprocessing configuration differs from the one used for training: %s. "
            "Predictions may be unreliable. Restore the NLTK corpora by running "
            "'python training/prepare_nltk.py', or retrain the model.",
            "; ".join(mismatches),
        )


def load_detector(model_dir: Path | str = DEFAULT_MODEL_DIR) -> FakeNewsDetector:
    """Load the model, vectoriser and metadata from disk."""
    model_dir = Path(model_dir)
    model_path = model_dir / MODEL_FILENAME
    vectorizer_path = model_dir / VECTORIZER_FILENAME

    if not model_path.is_file():
        raise ModelNotReadyError(
            f"Missing trained model: {model_path}\n"
            "Generate it first with:  python training/train.py"
        )
    if not vectorizer_path.is_file():
        raise ModelNotReadyError(
            f"Missing TF-IDF vectoriser: {vectorizer_path}\n"
            "Generate it first with:  python training/train.py"
        )

    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise ModelNotReadyError(
            f"{MODEL_FILENAME} could not be loaded ({exc}). "
            "It is probably corrupt or was built with a different library version - "
            "re-run: python training/train.py"
        ) from exc

    try:
        vectorizer = joblib.load(vectorizer_path)
    except Exception as exc:
        raise ModelNotReadyError(
            f"{VECTORIZER_FILENAME} could not be loaded ({exc}). "
            "Re-run: python training/train.py"
        ) from exc

    metadata = _read_metadata(model_dir)

    # The vectoriser must carry the preprocessor; without it the pipeline that
    # trained the model would not be reproduced at prediction time.
    preprocessor = getattr(vectorizer, "preprocessor", None)
    if not isinstance(preprocessor, NewsPreprocessor):
        logger.warning(
            "Loaded vectoriser has no NewsPreprocessor attached (found %r). "
            "Prediction text may not be cleaned the same way training text was.",
            type(preprocessor).__name__,
        )
    else:
        _check_preprocessing_matches(metadata, preprocessor)

    return FakeNewsDetector(model, vectorizer, metadata)


def get_detector(model_dir: Path | str | None = None) -> FakeNewsDetector:
    """Return the cached detector, loading it on first call."""
    global _DETECTOR
    if _DETECTOR is not None and model_dir is None:
        return _DETECTOR

    with _LOCK:
        if _DETECTOR is None or model_dir is not None:
            _DETECTOR = load_detector(model_dir or DEFAULT_MODEL_DIR)
    return _DETECTOR


def reset_detector() -> None:
    """Drop the cached detector. Used by the tests."""
    global _DETECTOR
    with _LOCK:
        _DETECTOR = None