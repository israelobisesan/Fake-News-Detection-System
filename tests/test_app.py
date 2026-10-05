"""
End-to-end checks for the fake news detection system.

Run with::

    python tests/test_app.py

Deliberately uses plain ``assert`` statements and Flask's built-in test client,
so it needs no test-runner dependency. Exit status 0 means everything passed.

Coverage:
    1.  preprocessing pipeline behaviour
    2.  preprocessing survives a pickle round-trip (used by the saved model)
    3.  dataset loading
    4.  model and vectoriser load from disk
    5.  detector classifies a real test-set article
    6.  detector classifies a fake test-set article
    7.  confidence comes from predict_proba and respects class order
    8.  Flask: GET / renders the form
    9.  Flask: POST /predict returns a result
    10. Flask: empty input is rejected with a friendly message
    11. Flask: missing model artifacts produce a friendly error, not a traceback
    12. security: HTML in user input is escaped, never executed
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str):
    """Decorator registering a test function."""
    def decorator(func):
        def wrapper():
            try:
                func()
                PASSED.append(name)
                print(f"  PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                FAILED.append((name, f"{exc}\n{traceback.format_exc()}"))
                print(f"  FAIL  {name}")
                print(f"        {exc}")
        wrapper.__name__ = func.__name__
        TESTS.append(wrapper)
        return wrapper
    return decorator


TESTS: list = []


# ==========================================================================
# 1-2. Preprocessing
# ==========================================================================

@check("preprocessing: removes publisher markers, punctuation and numbers")
def test_preprocessing():
    from app.preprocessing import NewsPreprocessor, combine_title_text

    pre = NewsPreprocessor(strip_source_artifacts=True)
    raw = "WASHINGTON (Reuters) - The 5 GOP senators said 21 bills will pass in 2017."
    cleaned = pre(raw)

    assert "reuters" not in cleaned.lower(), f"Reuters not removed: {cleaned!r}"
    assert "washington" not in cleaned.lower(), f"dateline not removed: {cleaned!r}"
    assert "21" not in cleaned and "5" not in cleaned, f"numbers not removed: {cleaned!r}"
    assert "," not in cleaned and "." not in cleaned, f"punctuation not removed: {cleaned!r}"
    assert "senator" in cleaned, f"lemmatisation did not run: {cleaned!r}"

    # The naive configuration must retain the publisher marker.
    naive = NewsPreprocessor(strip_source_artifacts=False)
    assert "reuters" in naive(raw).lower(), "naive mode should keep the source marker"

    # Safe on missing input.
    assert pre(None) == ""
    assert pre(float("nan")) == ""
    assert combine_title_text(None, "Body only") == "Body only"
    assert combine_title_text(float("nan"), "Body only") == "Body only"


@check("preprocessing: identical output after a pickle round-trip")
def test_preprocessing_pickle():
    import pickle
    from app.preprocessing import NewsPreprocessor

    pre = NewsPreprocessor(strip_source_artifacts=True)
    sample = "Donald Trump criticized the fake news media in a 2017 speech."
    revived = pickle.loads(pickle.dumps(pre))
    assert revived(sample) == pre(sample), "pickled preprocessor behaves differently"


# ==========================================================================
# 3. Dataset loading
# ==========================================================================

@check("dataset: loads, labels and de-duplicates")
def test_dataset():
    from training.data_loader import DatasetError, FAKE, TRUE, load_dataset

    try:
        df = load_dataset()
    except DatasetError as exc:
        raise AssertionError(f"dataset could not be loaded: {exc}") from exc

    assert len(df) > 1000, f"only {len(df)} rows loaded"
    assert set(df["label"].unique()) <= {FAKE, TRUE}, f"unexpected labels: {df['label'].unique()}"
    assert not df.duplicated(subset=["title", "text"]).any(), "duplicates remain after cleaning"
    assert (df["text"].str.strip() != "").all(), "empty article bodies remain"
    assert not df["text"].isna().any(), "missing values remain"

    counts = df["label"].value_counts()
    print(f"        {len(df):,} usable rows | FAKE {counts.get(FAKE, 0):,} | TRUE {counts.get(TRUE, 0):,}")


# ==========================================================================
# 4. Model artifacts
# ==========================================================================

@check("artifacts: model, vectoriser and metadata load from disk")
def test_artifacts():
    import json
    import joblib

    from app.detector import (METADATA_FILENAME, MODEL_FILENAME, VECTORIZER_FILENAME,
                              get_detector)
    from app.preprocessing import NewsPreprocessor

    models_dir = PROJECT_ROOT / "models"
    for filename in (MODEL_FILENAME, VECTORIZER_FILENAME, METADATA_FILENAME):
        assert (models_dir / filename).is_file(), f"missing {filename} - run python training/train.py"

    vectorizer = joblib.load(models_dir / VECTORIZER_FILENAME)
    assert hasattr(vectorizer, "transform"), "vectoriser cannot transform"
    assert isinstance(getattr(vectorizer, "preprocessor", None), NewsPreprocessor), \
        "vectoriser does not carry the shared preprocessor"

    detector = get_detector()
    assert detector.classes_, "model exposes no classes"
    assert detector.supports_proba, "model cannot produce probabilities"

    with (models_dir / METADATA_FILENAME).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    for key in ("model_name", "validation_accuracy", "validation_f1",
                "test_accuracy", "test_precision", "test_recall", "test_f1"):
        assert key in metadata, f"metadata missing '{key}'"
    for key in ("test_accuracy", "test_f1"):
        value = metadata[key]
        assert isinstance(value, (int, float)) and 0.0 <= value <= 1.0, \
            f"metadata['{key}'] = {value!r} is not a valid measured score"


# ==========================================================================
# 5-7. Prediction behaviour
# ==========================================================================

def _test_article(label: str):
    """Grab one article from the HELD-OUT test split.

    The split is reproduced exactly as training/train.py does it, so these
    articles are ones the model never saw during fitting. Using the training
    rows instead would let a memorising model score 100% and prove nothing.
    """
    from sklearn.model_selection import train_test_split
    from training.data_loader import load_dataset
    from training.train import RANDOM_STATE, TEST_SIZE, VALID_SIZE, VALID_FRACTION_OF_TEMP

    df = load_dataset()

    positions = np.arange(len(df))
    _, idx_temp = train_test_split(
        positions, test_size=TEST_SIZE + VALID_SIZE,
        stratify=df["label"], random_state=RANDOM_STATE,
    )
    _, idx_test = train_test_split(
        idx_temp, test_size=VALID_FRACTION_OF_TEMP,
        stratify=df["label"].iloc[idx_temp], random_state=RANDOM_STATE,
    )

    subset = df.iloc[np.sort(idx_test)]
    subset = subset[subset["label"] == label]
    assert len(subset) > 0, f"no test-split articles labelled {label}"

    # Pick the one closest to the middle of the length range so the sample is
    # a typical article rather than a two-word outlier.
    lengths = subset["text"].str.len()
    target = lengths.median()
    row = subset.iloc[(lengths - target).abs().values.argmin()]
    return str(row["title"]), str(row["text"])


@check("prediction: classifies a GENUINE article from the dataset")
def test_predict_genuine():
    from app.detector import get_detector
    from app.preprocessing import TRUE

    title, text = _test_article(TRUE)
    result = get_detector().predict(title, text)
    assert result.label in ("FAKE", "TRUE"), f"unexpected label {result.label!r}"
    assert result.display_label in ("Likely Fake", "Likely Genuine")
    assert 0.0 <= result.confidence <= 100.0, f"confidence out of range: {result.confidence}"
    print(f"        '{title[:58]}...' -> {result.display_label} ({result.confidence_text})")


@check("prediction: classifies a FAKE article from the dataset")
def test_predict_fake():
    from app.detector import get_detector
    from app.preprocessing import FAKE

    title, text = _test_article(FAKE)
    result = get_detector().predict(title, text)
    assert result.label in ("FAKE", "TRUE")
    assert 0.0 <= result.confidence <= 100.0
    print(f"        '{title[:58]}...' -> {result.display_label} ({result.confidence_text})")


@check("confidence: derived from predict_proba and respects class order")
def test_confidence():
    from app.detector import get_detector

    detector = get_detector()
    title, text = _test_article("FAKE")

    features = detector.vectorizer.transform([f"{title} {text}"])
    probabilities = detector.model.predict_proba(features)[0]
    predicted_label = str(detector.model.predict(features)[0])

    # Resolve the label's column the same way the detector does.
    index = detector.classes_.index(predicted_label)

    result = detector.predict(title, text)
    assert result.label == predicted_label, "detector and model disagree on the label"
    assert abs(result.confidence - probabilities[index] * 100) < 1e-6, \
        "confidence does not match the model's probability"
    assert result.confidence >= 50.0, \
        "confidence should be the highest class probability, so at least 50%"
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6, "probabilities do not sum to 1"

    print(f"        classes_ order {detector.classes_}; "
          f"predicted {predicted_label} at index {index} -> {result.confidence_text}")


# ==========================================================================
# 8-12. Flask routes
# ==========================================================================

@check("flask: GET / returns the form")
def test_index():
    from app import create_app

    client = create_app({"TESTING": True}).test_client()
    response = client.get("/")
    assert response.status_code == 200, f"status {response.status_code}"
    body = response.get_data(as_text=True)
    assert "Fake News Detection System" in body
    assert 'name="title"' in body and 'name="article"' in body, "form fields missing"
    assert "Detect News" in body


@check("flask: POST /predict returns a labelled result with a confidence")
def test_predict_route():
    from app import create_app
    from training.data_loader import FAKE

    title, text = _test_article(FAKE)
    client = create_app({"TESTING": True}).test_client()
    response = client.post("/predict", data={"title": title, "article": text})
    body = response.get_data(as_text=True)

    assert response.status_code == 200, f"status {response.status_code}"
    assert ("Likely Fake" in body) or ("Likely Genuine" in body), "no verdict rendered"
    assert "Confidence" in body, "confidence not rendered"
    assert "%" in body, "no confidence percentage rendered"
    assert "not</strong> the probability" in body or "not the probability" in body, \
        "the confidence disclaimer is missing"


@check("flask: empty input is rejected with a friendly message")
def test_empty_input():
    from app import create_app

    client = create_app({"TESTING": True}).test_client()

    both_empty = client.post("/predict", data={"title": "", "article": ""})
    assert both_empty.status_code == 400, f"status {both_empty.status_code}"
    body = both_empty.get_data(as_text=True)
    assert "Please enter a headline" in body, "no message for empty submission"
    assert "Traceback" not in body, "a traceback leaked to the user"

    missing_title = client.post("/predict", data={"title": "", "article": "A" * 200})
    assert missing_title.status_code == 400
    assert "Please enter a headline" in missing_title.get_data(as_text=True)

    missing_article = client.post("/predict", data={"title": "Some headline", "article": ""})
    assert missing_article.status_code == 400
    assert "Please paste the news article" in missing_article.get_data(as_text=True)

    too_short = client.post("/predict", data={"title": "Breaking", "article": "too short"})
    assert too_short.status_code == 400
    assert "at least 20 characters" in too_short.get_data(as_text=True)


@check("flask: missing model artifacts give a friendly error, not a traceback")
def test_missing_model():
    from app import create_app
    from app import detector as detector_module

    detector_module.reset_detector()
    try:
        client = create_app({"TESTING": True, "MODEL_DIR": str(PROJECT_ROOT / "no_such_dir")}).test_client()
        response = client.post("/predict", data={"title": "A headline here",
                                                 "article": "Some article body text goes here."})
        body = response.get_data(as_text=True)

        assert response.status_code == 503, f"status {response.status_code}"
        assert "Model not available" in body, "no friendly error page"
        assert "Traceback" not in body, "a traceback leaked to the user"
        assert "python training/train.py" in body, "no guidance on how to fix it"
    finally:
        detector_module.reset_detector()


@check("security: HTML in submitted text is escaped, never executed")
def test_xss_escaping():
    from app import create_app
    from app.detector import get_detector

    payload = "<script>alert('xss')</script>"
    article = payload + " " + ("Ordinary news body text. " * 12)

    client = create_app({"TESTING": True}).test_client()
    response = client.post("/predict", data={"title": payload, "article": article})
    body = response.get_data(as_text=True)

    assert response.status_code == 200, f"status {response.status_code}"
    assert "<script>alert" not in body, "unescaped script tag rendered into the page"
    assert "&lt;script&gt;" in body, "payload was not HTML-escaped"


# ==========================================================================
# 13-16. Optional article-URL input
# ==========================================================================

@check("url input: refuses private, internal and non-http addresses (SSRF guard)")
def test_ssrf_guard():
    from app.article_fetcher import BlockedAddressError, InvalidURLError, validate_public_url

    # These are the targets an SSRF attack would aim for.
    must_be_private = [
        "http://127.0.0.1/",
        "http://localhost/",
        "http://localhost:5000/health",
        "http://169.254.169.254/latest/meta-data/",   # cloud instance metadata
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://172.16.0.1/",
        "http://[::1]/",
        "http://100.64.0.1/",                          # carrier-grade NAT
    ]
    for url in must_be_private:
        try:
            validate_public_url(url)
            raise AssertionError(f"NOT BLOCKED (security hole): {url}")
        except (BlockedAddressError, InvalidURLError):
            pass

    # Dangerous schemes must never be requested. Note that "data:" and
    # "javascript:" have no "//", so a naive bare-host fallback would turn them
    # into https URLs and quietly bypass the scheme check.
    for url in ("file:///etc/passwd", "file:///C:/Windows/win.ini",
                "ftp://example.com/x", "gopher://example.com/",
                "data:text/html,hi", "data:text/html;base64,PGgxPmhpPC9oMT4=",
                "javascript:alert(1)", "mailto:someone@example.com",
                "http://example.com:notaport/x"):
        try:
            validate_public_url(url)
            raise AssertionError(f"scheme NOT BLOCKED: {url}")
        except (BlockedAddressError, InvalidURLError):
            pass

    # Ordinary public addresses must still work, including a bare hostname.
    assert validate_public_url("https://www.bbc.co.uk/news").startswith("https://")
    assert validate_public_url("www.example.com/story").startswith("https://")


@check("url input: refuses pages with too little text instead of guessing")
def test_extraction_length_guard():
    from app import article_fetcher as af

    # An error page or a news index still contains enough boilerplate for the
    # classifier to return a confident, meaningless verdict. Extraction must
    # refuse these rather than pass them on.
    stub_pages = [
        "<html><body><h1>Not found</h1><p>Sorry, that page does not exist.</p></body></html>",
        "<html><head><title>Home</title></head><body><nav>Menu</nav><p>News</p></body></html>",
        "<html><body></body></html>",
    ]
    for html in stub_pages:
        _, body = af.extract_article(html)
        assert len(body) < af.MIN_BODY_CHARACTERS, (
            f"stub page yielded {len(body)} characters, which is at or above the "
            f"{af.MIN_BODY_CHARACTERS} minimum and would be classified"
        )

    # A realistic article must comfortably clear the bar.
    article_html = (
        "<html><head><title>T</title></head><body><nav>Menu</nav><h1>Real headline</h1>"
        + "<div>" + "".join(
            f"<p>This is sentence number {i} of a genuine news report about an event "
            f"that a reporter observed and wrote down carefully.</p>" for i in range(12)
        ) + "</div><footer>Copyright</footer></body></html>"
    )
    title, body = af.extract_article(article_html)
    assert len(body) >= af.MIN_BODY_CHARACTERS, "a real article was rejected"
    assert title == "Real headline", f"title extraction failed: {title!r}"
    assert "Copyright" not in body and "Menu" not in body, "boilerplate leaked into the body"


@check("url input: a blocked or invalid link gives a friendly message, not a traceback")
def test_url_failure_handling():
    from app import create_app

    client = create_app({"TESTING": True}).test_client()

    # A private address must be refused at the route, with an explanation.
    private = client.post("/predict", data={
        "title": "Some headline", "article": "",
        "url": "http://169.254.169.254/latest/meta-data/",
    })
    assert private.status_code == 400, f"status {private.status_code}"
    body = private.get_data(as_text=True)
    assert "Traceback" not in body, "a traceback leaked to the user"
    assert "private or internal network" in body, "no explanation for the refused address"

    # Nonsense input must also be handled gracefully.
    rubbish = client.post("/predict", data={
        "title": "Some headline", "article": "", "url": "not a url at all !!!!",
    })
    assert rubbish.status_code == 400, f"status {rubbish.status_code}"
    assert "Traceback" not in rubbish.get_data(as_text=True)

    # An unreachable host should not hang the form or leak internals.
    unreachable = client.post("/predict", data={
        "title": "Some headline", "article": "",
        "url": "https://this-host-should-not-resolve-9f8e7d.invalid/article",
    })
    assert unreachable.status_code in (400, 502), f"status {unreachable.status_code}"
    assert "Traceback" not in unreachable.get_data(as_text=True)


@check("url input: pasted text takes priority and the route is unchanged without a URL")
def test_url_priority_and_regression():
    from app import create_app
    from training.data_loader import FAKE

    title, text = _test_article(FAKE)
    client = create_app({"TESTING": True}).test_client()

    # With text present the URL is ignored entirely, so this must still work
    # exactly as it did before the feature existed.
    with_text = client.post("/predict", data={"title": title, "article": text, "url": ""})
    assert with_text.status_code == 200, f"status {with_text.status_code}"
    body = with_text.get_data(as_text=True)
    assert ("Likely Fake" in body) or ("Likely Genuine" in body)
    assert "Retrieved from" not in body, "no URL was supplied, so none should be reported"

    # A URL plus pasted text must classify the pasted text, not fetch anything.
    both = client.post("/predict", data={
        "title": title, "article": text,
        "url": "http://169.254.169.254/should-not-be-fetched",
    })
    assert both.status_code == 200, f"status {both.status_code}"
    assert "should-not-be-fetched" not in both.get_data(as_text=True), \
        "the URL was fetched even though text was pasted"


# ==========================================================================
# Runner
# ==========================================================================

def main() -> int:
    print("=" * 74)
    print("Fake News Detection System - test suite")
    print("=" * 74)
    print()

    for test in TESTS:
        test()

    print()
    print("=" * 74)
    print(f"passed: {len(PASSED)}    failed: {len(FAILED)}")
    print("=" * 74)

    if FAILED:
        print("\nFailures:\n")
        for name, detail in FAILED:
            print(f"--- {name} ---\n{detail}\n")
        return 1

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())