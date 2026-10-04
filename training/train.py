"""
Model training and evaluation for the fake news detection system.

Run with::

    python training/train.py

What this script does
---------------------
1.  Load and clean the ISOT dataset (44,898 raw records).
2.  Build each article's input string as ``title + " " + text``.
3.  Split into 70% train / 15% validation / 15% test, stratified, with a fixed
    ``random_state`` so every run is identical.
4.  Run the whole experiment TWICE:

      * **naive baseline**   - publisher markers left in the text
      * **de-leaked (main)** - Reuters datelines, outlet names, URLs and web
                               boilerplate removed

    Both runs use exactly the same split, so the comparison is fair. The
    reason for this is in ``app/preprocessing``: every genuine ISOT article is
    Reuters copy carrying a ``(Reuters)`` marker that no fake article has, which
    lets a classifier score very high without learning anything about deception.

5.  Fit TF-IDF on the TRAINING portion only, then transform validation/test.
6.  Train three models - Multinomial Naive Bayes, calibrated Linear SVM and
    Random Forest - and score all three on the validation set.
7.  Select the best model by validation F1 for the FAKE class.
8.  Evaluate ONLY that model on the held-out test set.
9.  Save the model, the vectoriser, the preprocessing configuration and the
    measured metrics.

Every number printed and stored here is measured by this script. Nothing is
hard-coded.

Artefacts written to models/
----------------------------
    fake_news_model.pkl     the selected classifier
    tfidf_vectorizer.pkl    the fitted TF-IDF vectoriser (with the preprocessor inside)
    model_metadata.json     measured metrics + preprocessing configuration
    reports/model_comparison.txt   the full comparison table
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from app.preprocessing import FAKE_LABELS, NewsPreprocessor  # noqa: E402
from training.data_loader import DatasetError, FAKE, TRUE, load_dataset, print_summary  # noqa: E402

# ==========================================================================
# Configuration - every setting in one place
# ==========================================================================

RANDOM_STATE = 42          # fixed so the split never changes between runs
TEST_SIZE = 0.15           # 15% test
VALID_SIZE = 0.15          # 15% validation
VALID_FRACTION_OF_TEMP = 0.5   # 30% temp split in half -> 15% + 15%

TFIDF_KWARGS = dict(
    ngram_range=(1, 2),    # unigrams + bigrams; bigrams capture phrases like "fake news"
    min_df=2,              # ignore words appearing in fewer than 2 documents
    max_df=0.90,           # ignore words in >90% of documents (pure noise)
    max_features=50_000,   # cap vocabulary for memory and runtime
    sublinear_tf=True,     # log(1 + tf): damps very frequent words
    token_pattern=r"(?u)\b[a-z][a-z]+\b",
)

MODEL_PARAMS = {
    "Multinomial Naive Bayes": dict(alpha=0.1),
    "Calibrated Linear SVM": dict(C=1.0, cv=5, method="sigmoid"),
    "Random Forest": dict(
        n_estimators=100,
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
}

MODEL_ORDER = ["Multinomial Naive Bayes", "Calibrated Linear SVM", "Random Forest"]

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODELS_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
REPORT_PATH = REPORTS_DIR / "model_comparison.txt"

# The two experiments. "de_leaked" is the one that gets deployed.
EXPERIMENTS = [
    {"name": "naive_baseline", "strip_source_artifacts": False,
     "description": "Publisher markers left in (Reuters datelines, outlet names, URLs)"},
    {"name": "de_leaked", "strip_source_artifacts": True,
     "description": "Publisher markers removed - reflects writing style, not publisher"},
]
DEPLOYED_EXPERIMENT = "de_leaked"


# ==========================================================================
# Helpers
# ==========================================================================

def build_model(name: str):
    """Instantiate one classifier by name, calibrated so probabilities exist.

    LinearSVC is a margin-based classifier and has no ``predict_proba``. We wrap
    it in ``CalibratedClassifierCV`` (sigmoid / Platt scaling), which fits an
    internal cross-validated model mapping the SVM margin onto a probability.
    That is what lets us report a genuine confidence figure for the SVM too.
    """
    params = dict(MODEL_PARAMS[name])
    if name == "Calibrated Linear SVM":
        return CalibratedClassifierCV(LinearSVC(C=params["C"], random_state=RANDOM_STATE),
                                      cv=params["cv"], method=params["method"])
    if name == "Multinomial Naive Bayes":
        return MultinomialNB(alpha=params["alpha"])
    return RandomForestClassifier(**params)


def score_model(model, X, y_true: pd.Series) -> dict:
    """Accuracy, precision, recall and F1, treating FAKE as the positive class."""
    y_pred = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=FAKE, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=FAKE, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=FAKE, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def print_confusion(model, X, y_true: pd.Series, heading: str) -> np.ndarray:
    """Print a labelled confusion matrix and return it."""
    cm = confusion_matrix(y_true, model.predict(X), labels=[FAKE, TRUE])
    rows = [(FAKE, "actually FAKE"), (TRUE, "actually TRUE")]
    print(f"\n  Confusion matrix - {heading}")
    print(f"  {'':16} predicted FAKE   predicted TRUE")
    for (label, desc), row in zip(rows, cm):
        print(f"  {desc:16} {row[0]:>13,} {row[1]:>15,}")
    print(f"  Interpretation: cells on the diagonal are correct predictions;")
    print(f"  FAKE predicted as TRUE = {cm[0][1]:,} false negatives (missed fake news).")
    return cm


def split_indices(y: pd.Series, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 70/15/15 split of positional indices.

    Returns indices into the original frame so the same rows can be reused by
    every experiment.
    """
    positions = np.arange(len(y))
    idx_train, idx_temp = train_test_split(
        positions, test_size=TEST_SIZE + VALID_SIZE,
        stratify=y, random_state=random_state,
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=VALID_FRACTION_OF_TEMP,
        stratify=y.iloc[idx_temp], random_state=random_state,
    )
    return np.sort(idx_train), np.sort(idx_val), np.sort(idx_test)


def preprocess_frame(df: pd.DataFrame, strip_source_artifacts: bool) -> tuple[pd.Series, NewsPreprocessor]:
    """Apply the shared preprocessor to every document.

    This is stateless cleaning, so applying it before the split does not leak
    anything. The *fitted* part (TF-IDF) is only ever fitted on training rows.
    """
    preprocessor = NewsPreprocessor(strip_source_artifacts=strip_source_artifacts)
    combined = (df["title"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str))
    start = time.time()
    cleaned = combined.map(preprocessor)
    print(f"  preprocessed {len(cleaned):,} documents in {time.time() - start:.1f}s "
          f"({preprocessor.describe()})")
    return cleaned, preprocessor


def select_best(results: dict[str, dict]) -> tuple[str, str]:
    """Pick the model with the best validation F1 (FAKE class); accuracy breaks ties."""
    best_name = max(results, key=lambda n: (results[n]["f1"], results[n]["accuracy"]))
    return best_name, (
        f"highest validation F1 for the FAKE class "
        f"({results[best_name]['f1']:.4f}, accuracy {results[best_name]['accuracy']:.4f})"
    )


# ==========================================================================
# The experiment
# ==========================================================================

def run_experiment(df: pd.DataFrame, experiment: dict, splits: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict:
    """Train and evaluate all three models for one preprocessing configuration."""
    idx_train, idx_val, idx_test = splits
    y = df["label"]

    print("\n" + "=" * 74)
    print(f"EXPERIMENT: {experiment['name']}")
    print(f"  {experiment['description']}")
    print("=" * 74)

    cleaned, preprocessor = preprocess_frame(df, experiment["strip_source_artifacts"])

    # Guard against a split producing empty documents (very short articles).
    for split_name, idx in (("train", idx_train), ("validation", idx_val), ("test", idx_test)):
        empty = int((cleaned.iloc[idx] == "").sum())
        if empty:
            print(f"  WARNING: {empty} {split_name} document(s) cleaned to empty string and were dropped.")
    mask = cleaned.str.strip() != ""
    idx_train = idx_train[mask.iloc[idx_train].to_numpy()]
    idx_val = idx_val[mask.iloc[idx_val].to_numpy()]
    idx_test = idx_test[mask.iloc[idx_test].to_numpy()]

    X_train_raw = cleaned.iloc[idx_train]
    X_val_raw = cleaned.iloc[idx_val]
    X_test_raw = cleaned.iloc[idx_test]
    y_train, y_val, y_test = y.iloc[idx_train], y.iloc[idx_val], y.iloc[idx_test]

    # --- TF-IDF: fitted on TRAINING DATA ONLY ---
    print(f"\n  Split sizes -> train {len(idx_train):,} | validation {len(idx_val):,} | test {len(idx_test):,}")
    vectorizer = TfidfVectorizer(preprocessor=preprocessor, **TFIDF_KWARGS)
    start = time.time()
    X_train = vectorizer.fit_transform(X_train_raw)   # fit + transform
    X_val = vectorizer.transform(X_val_raw)          # transform only
    X_test = vectorizer.transform(X_test_raw)        # transform only
    print(f"  TF-IDF vocabulary: {len(vectorizer.vocabulary_):,} terms "
          f"(fit on train in {time.time() - start:.1f}s)")
    print(f"  train matrix: {X_train.shape[0]:,} docs x {X_train.shape[1]:,} features, "
          f"{X_train.nnz:,} non-zero")

    # --- Train and score all three models on the VALIDATION set ---
    print("\n  VALIDATION RESULTS")
    print("  " + "-" * 70)
    print(f"  {'Model':<26}{'Accuracy':>11}{'Precision':>11}{'Recall':>10}{'F1':>9}")
    print("  " + "-" * 70)

    fitted: dict[str, object] = {}
    results: dict[str, dict] = {}
    matrices: dict[str, np.ndarray] = {}

    for name in MODEL_ORDER:
        model = build_model(name)
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        metrics = score_model(model, X_val, y_val)
        fitted[name] = model
        results[name] = metrics
        matrices[name] = print_confusion(model, X_val, y_val, name)
        print(f"  {name:<26}{metrics['accuracy']:>11.4f}{metrics['precision']:>11.4f}"
              f"{metrics['recall']:>10.4f}{metrics['f1']:>9.4f}   ({elapsed:.1f}s to train)")
        if hasattr(model, "predict_proba"):
            mean_conf = float(model.predict_proba(X_val).max(axis=1).mean() * 100)
            print(f"  {'':<26}mean confidence on validation: {mean_conf:.2f}%")

    print("  " + "-" * 70)

    # --- Model selection, based only on validation performance ---
    best_name, reason = select_best(results)
    print(f"\n  SELECTED MODEL: {best_name}")
    print(f"  Reason: {reason}")

    # --- Final evaluation: the selected model only, on the unseen test set ---
    test_metrics = score_model(fitted[best_name], X_test, y_test)
    test_matrix = print_confusion(fitted[best_name], X_test, y_test, f"{best_name} on TEST")

    print("\n  FINAL TEST RESULTS (selected model, test set never used for tuning)")
    print("  " + "-" * 70)
    print(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"  Precision : {test_metrics['precision']:.4f}")
    print(f"  Recall    : {test_metrics['recall']:.4f}")
    print(f"  F1-score  : {test_metrics['f1']:.4f}")
    print("  " + "-" * 70)
    print("\n  Per-class detail:")
    print(classification_report(y_test, fitted[best_name].predict(X_test),
                                labels=[FAKE, TRUE], digits=4, zero_division=0))

    return {
        "name": experiment["name"],
        "description": experiment["description"],
        "preprocessing": preprocessor.resolved_config(),
        "split_sizes": {
            "train": int(len(idx_train)),
            "validation": int(len(idx_val)),
            "test": int(len(idx_test)),
        },
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "validation": results,
        "validation_confusion_matrices": {
            name: matrix.tolist() for name, matrix in matrices.items()
        },
        "selected_model": best_name,
        "selection_reason": reason,
        "test": test_metrics,
        "test_confusion_matrix": test_matrix.tolist(),
        "class_order": [FAKE, TRUE],
        "_model": fitted[best_name],
        "_vectorizer": vectorizer,
        "_preprocessor": preprocessor,
    }


# ==========================================================================
# Reporting
# ==========================================================================

def build_report(payload: dict, experiments: list[dict]) -> str:
    """Render the human-readable comparison file."""
    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add("FAKE NEWS DETECTION - MODEL TRAINING AND COMPARISON REPORT")
    add("=" * 78)
    add(f"Generated          : {payload['generated_at']}")
    add(f"Python             : {payload['python_version']}")
    add(f"scikit-learn       : {payload['sklearn_version']}")
    add(f"Random state       : {RANDOM_STATE}")
    add("")
    add(f"Raw records read   : {payload['dataset']['rows_read']:,}")
    add(f"Usable records     : {payload['dataset']['rows_usable']:,} "
        f"(FAKE {payload['dataset']['fake_rows']:,} / TRUE {payload['dataset']['true_rows']:,})")
    add(f"Records dropped    : {payload['dataset']['rows_dropped_duplicate']:,} duplicate, "
        f"{payload['dataset']['rows_dropped_empty']:,} empty")
    split = experiments[0]["split_sizes"]
    add(f"Split              : train {split['train']:,} (70%) / validation {split['validation']:,} (15%) "
        f"/ test {split['test']:,} (15%), stratified")
    add(f"TF-IDF             : ngrams {TFIDF_KWARGS['ngram_range']}, min_df {TFIDF_KWARGS['min_df']}, "
        f"max_features {TFIDF_KWARGS['max_features']:,}, sublinear_tf {TFIDF_KWARGS['sublinear_tf']}")
    add("")
    add("Precision, recall and F1 are computed for the FAKE class (positive class).")
    add("")

    for experiment in experiments:
        add("=" * 78)
        add(f"EXPERIMENT: {experiment['name']}")
        add(f"  {experiment['description']}")
        add("=" * 78)
        add("  VALIDATION SET - all three models")
        add(f"  {'Model':<26}{'Accuracy':>11}{'Precision':>11}{'Recall':>10}{'F1':>9}")
        add("  " + "-" * 72)
        for name in MODEL_ORDER:
            m = experiment["validation"][name]
            marker = "  <-- selected" if name == experiment["selected_model"] else ""
            add(f"  {name:<26}{m['accuracy']:>11.4f}{m['precision']:>11.4f}"
                f"{m['recall']:>10.4f}{m['f1']:>9.4f}{marker}")
        add("")
        add("  Selected model : " + experiment["selected_model"])
        add("  Reason         : " + experiment["selection_reason"])
        add("")
        add("  CONFUSION MATRICES - validation set (rows = true, cols = predicted)")
        add(f"  {'Model':<26}{'true FAKE/pred FAKE':>21}{'true FAKE/pred TRUE':>21}")
        for name in MODEL_ORDER:
            cm = experiment["validation_confusion_matrices"][name]
            add(f"  {name:<26}{cm[0][0]:>21,}{cm[0][1]:>21,}")
        add(f"  {'':<26}{'true TRUE/pred FAKE':>21}{'true TRUE/pred TRUE':>21}")
        for name in MODEL_ORDER:
            cm = experiment["validation_confusion_matrices"][name]
            add(f"  {name:<26}{cm[1][0]:>21,}{cm[1][1]:>21,}")
        add("")
        add("  TEST SET - selected model only")
        add("  " + "-" * 72)
        for key in ("accuracy", "precision", "recall", "f1"):
            add(f"  {key.capitalize():<12}: {experiment['test'][key]:.4f}")
        add("")

    naive = next((e for e in experiments if e["name"] == "naive_baseline"), None)
    deleaked = next((e for e in experiments if e["name"] == DEPLOYED_EXPERIMENT), None)
    if naive and deleaked:
        add("=" * 78)
        add("WHY TWO EXPERIMENTS?")
        add("=" * 78)
        add("Every genuine article in ISOT is Reuters wire copy and carries a")
        add("'(Reuters)' marker; almost no fake article does. A model can therefore score")
        add("very highly by recognising the publisher instead of the writing style, which")
        add("would not generalise to real-world news. The de-leaked run removes those")
        add("markers so the reported score reflects language rather than provenance.")
        add("")
        add(f"  {'Model':<26}{'Naive F1':>11}{'De-leaked F1':>14}{'Difference':>12}")
        add("  " + "-" * 72)
        for name in MODEL_ORDER:
            a = naive["validation"][name]["f1"]
            b = deleaked["validation"][name]["f1"]
            add(f"  {name:<26}{a:>11.4f}{b:>14.4f}{b - a:>+12.4f}")
        add("")
        add("These are validation-set figures. The deployed model is the de-leaked run.")

    add("")
    add("=" * 78)
    add("INTERPRETATION")
    add("=" * 78)
    add("A high score on ISOT does NOT mean the system is accurate on news from other")
    add("countries, other publishers or other eras. ISOT's fake half comes from a small")
    add("group of unreliable outlets and its genuine half is homogeneous Reuters copy, so")
    add("the benchmark is far easier than real-world misinformation detection. Treat the")
    add("figures above as a measurement on THIS dataset, not as a general capability.")
    return "\n".join(lines)


# ==========================================================================
# Entry point
# ==========================================================================

def main() -> int:
    print("=" * 74)
    print("FAKE NEWS DETECTION - TRAINING")
    print("=" * 74)
    np.random.seed(RANDOM_STATE)

    print("\n[1/4] Loading dataset")
    try:
        df = load_dataset()
    except DatasetError as exc:
        print(f"\nERROR: {exc}")
        return 1
    print_summary(df)
    summary = df.attrs["load_summary"]

    if len(df) < 100:
        print("\nERROR: fewer than 100 usable records - check the dataset files.")
        return 1

    # One split, shared by every experiment, so comparisons are like-for-like.
    splits = split_indices(df["label"], RANDOM_STATE)
    print(f"  label values in use: {FAKE_LABELS}")

    print("\n[2/4] Running experiments (naive baseline, then de-leaked)")
    experiments = [run_experiment(df, experiment, splits) for experiment in EXPERIMENTS]

    deployed = next(e for e in experiments if e["name"] == DEPLOYED_EXPERIMENT)

    print("\n[3/4] Saving artefacts")
    import joblib

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(deployed["_model"], MODEL_PATH, compress=3)
    joblib.dump(deployed["_vectorizer"], VECTORIZER_PATH, compress=3)

    metadata = {
        "model_name": deployed["selected_model"],
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,

        # Validation metrics of the SELECTED model
        "validation_accuracy": deployed["validation"][deployed["selected_model"]]["accuracy"],
        "validation_precision": deployed["validation"][deployed["selected_model"]]["precision"],
        "validation_recall": deployed["validation"][deployed["selected_model"]]["recall"],
        "validation_f1": deployed["validation"][deployed["selected_model"]]["f1"],

        # Test metrics of the SELECTED model
        "test_accuracy": deployed["test"]["accuracy"],
        "test_precision": deployed["test"]["precision"],
        "test_recall": deployed["test"]["recall"],
        "test_f1": deployed["test"]["f1"],

        "selection_reason": deployed["selection_reason"],
        "random_state": RANDOM_STATE,
        "label_mapping": {"FAKE": "Likely Fake", "TRUE": "Likely Genuine"},
        "class_order": [FAKE, TRUE],
        "dataset": {
            "name": "ISOT Fake News Dataset",
            "rows_read": summary["rows_before_cleaning"],
            "rows_usable": summary["rows_final"],
            "rows_dropped_duplicate": summary["rows_dropped_duplicate"],
            "rows_dropped_empty": summary["rows_dropped_empty"],
            "fake_rows": summary["fake_rows"],
            "true_rows": summary["true_rows"],
        },
        "split_sizes": deployed["split_sizes"],
        "split_ratio": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "tfidf": {
            **{k: (list(v) if isinstance(v, tuple) else v) for k, v in TFIDF_KWARGS.items()},
            "vocabulary_size": deployed["vocabulary_size"],
            "fitted_on": "training split only",
        },
        "preprocessing": deployed["preprocessing"],
        "model_parameters": {
            name: {k: (v if not isinstance(v, tuple) else list(v))
                   for k, v in MODEL_PARAMS[name].items()}
            for name in MODEL_ORDER
        },
        "test_confusion_matrix": deployed["test_confusion_matrix"],
        "experiments": {
            e["name"]: {
                "description": e["description"],
                "validation": e["validation"],
                "selected_model": e["selected_model"],
                "test": e["test"],
            }
            for e in experiments
        },
        "disclaimer": (
            "Measured on the ISOT dataset only. Confidence is the model's confidence "
            "in its own classification, not the probability that the article is true "
            "or false. Not trained on Nigerian news."
        ),
    }

    with METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    report = build_report(metadata, experiments)
    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        handle.write(report + "\n")

    print(f"  model      -> {MODEL_PATH.relative_to(PROJECT_ROOT)}  "
          f"({MODEL_PATH.stat().st_size / 1_048_576:.1f} MB)")
    print(f"  vectorizer -> {VECTORIZER_PATH.relative_to(PROJECT_ROOT)}  "
          f"({VECTORIZER_PATH.stat().st_size / 1_048_576:.1f} MB)")
    print(f"  metadata   -> {METADATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  report     -> {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    print("\n[4/4] Done. Selected model: " + deployed["selected_model"])
    print("  (de-leaked configuration - publisher markers removed)")
    print("  Start the app with:  python run.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())