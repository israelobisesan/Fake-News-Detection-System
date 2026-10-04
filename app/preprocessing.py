"""
Text preprocessing for fake-news detection.

This module is the SINGLE SOURCE OF TRUTH for text cleaning.

Why it is built as a class
--------------------------
The most common way this kind of project goes wrong is having two copies of the
cleaning code: one inside the training script and one inside the web app. They
drift apart, and the model silently ends up scoring text it was never trained
on.

Here we avoid that entirely. ``NewsPreprocessor`` is handed to scikit-learn's
``TfidfVectorizer(preprocessor=...)``. From then on scikit-learn itself calls
this object every time it sees text - during ``fit_transform`` on the training
data AND during ``transform`` on new text. There is only ever one cleaning
implementation, so training and prediction cannot disagree.

It must stay a module-level class (not a lambda or closure) because the fitted
vectorizer is pickled to disk with joblib; unpickling needs to import this class
back from this module.

The pipeline
------------
    1. lowercase
    2. optionally strip source artifacts (wire-service datelines, outlet names,
       URLs, social handles, CMS boilerplate)  <- see "Source leakage" below
    3. strip accents, then remove punctuation and other special characters
    4. remove numbers
    5. tokenise
    6. remove stop words
    7. lemmatise to word roots

Source leakage - an important detail about this dataset
-------------------------------------------------------
Every genuine article in the ISOT dataset is Reuters wire copy and begins with a
dateline like ``WASHINGTON (Reuters) -``. No fake article contains that marker.
A plain TF-IDF model can therefore score ~99% accuracy just by memorising the
publisher, learning nothing about deception. Outlets are also named *inside*
articles ("told Reuters"), and the fake-news sites leak web-page furniture such
as "FEATURED IMAGE" and "GETTY IMAGES" into their body text.

Setting ``strip_source_artifacts=True`` removes those markers so the reported
score reflects writing style rather than publisher identity. The training script
runs the whole experiment twice - with and without - so the difference can be
reported honestly.
"""

from __future__ import annotations

import functools
import logging
import re
import unicodedata
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# Canonical label values. Labels are kept as strings everywhere so that the
# mapping from a numeric class index to a meaning is never implicit.
FAKE = "FAKE"
TRUE = "TRUE"
FAKE_LABELS = (FAKE, TRUE)

# How a raw model label is shown to the user.
LABEL_DISPLAY = {FAKE: "Likely Fake", TRUE: "Likely Genuine"}

# --------------------------------------------------------------------------
# Source-artifact patterns
# --------------------------------------------------------------------------

# Wire services and mainstream outlets. Genuine ISOT articles are Reuters.
_WIRE_SERVICES = r"reuters|associated press|\bap\b|\bafp\b|press association|bloomberg|\bbbc\b|\bcnn\b"

# Unreliable outlets that dominate the fake half of ISOT. Deliberately limited
# to unambiguous outlet names - generic words such as "propaganda" are ordinary
# news vocabulary and stripping them would destroy real signal.
_FAKE_OUTLETS = (
    r"breitbart|infowars|natural news|activist press"
    r"|21st century wire|world news daily report|news reporting network"
    r"|daily sbwire|\bwfb\b|presstv|newswire press"
)

_ALL_OUTLETS = rf"{_WIRE_SERVICES}|{_FAKE_OUTLETS}"

# Web-page furniture that the crawler captured along with the article body.
# Only distinctive multi-word phrases are listed. Bare words like "watch",
# "photo" or "video" are ordinary news vocabulary and are left alone.
_CMS_BOILERPLATE = (
    r"featured image|featured photo|getty images?|image credit|photograph credit"
    r"|read more|read next|click here|all rights reserved|share this"
    r"|related stories|follow us|subscribe to our|image caption"
)

# "WASHINGTON (Reuters) -" style dateline at the very start of an article.
# The city part is 1-40 characters of capitalised words.
_DATELINE = re.compile(
    rf"^\s*[A-Z][A-Za-z .,'\-]{{0,40}}\(\s*(?:{_ALL_OUTLETS})\s*\)\s*[-\u2013\u2014]\s*",
    re.IGNORECASE,
)

# "(Reuters)" appearing anywhere, e.g. "... told (Reuters) on Tuesday".
_PAREN_AGENCY = re.compile(rf"\(\s*(?:{_ALL_OUTLETS})\s*\)", re.IGNORECASE)

# Bare outlet names in running text: "told Reuters", "according to Breitbart".
_BARE_OUTLET = re.compile(rf"\b(?:{_ALL_OUTLETS})\b", re.IGNORECASE)

_BOILERPLATE = re.compile(rf"\b(?:{_CMS_BOILERPLATE})\b", re.IGNORECASE)

_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_HANDLE = re.compile(r"[@#][A-Za-z_][A-Za-z0-9_]*")
_EMAIL = re.compile(r"\S+@\S+\.\S+")

# Anything that is not a latin letter becomes a space, which also removes digits.
_NON_LETTER = re.compile(r"[^a-z]+")

# A "word": letters, optionally with an internal apostrophe (don't, it's).
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")

# Backstop stop-word list if the NLTK corpus is unavailable. sklearn ships one,
# and sklearn is a hard dependency anyway.
_FALLBACK_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "s", "same", "she", "should", "so", "some", "such", "t", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "you", "your",
    "yours", "yourself", "yourselves",
}


# --------------------------------------------------------------------------
# NLTK resources (loaded once, cached at module level)
# --------------------------------------------------------------------------

def _load_nltk() -> tuple[set[str], Any]:
    """Return (stopword set, lemmatizer).

    Tries NLTK first. If the corpora are missing - for example the app runs on a
    machine where ``training/prepare_nltk.py`` was never executed - it degrades
    gracefully instead of crashing: sklearn's built-in stop-word list is used
    and the lemmatizer is ``None``.
    """
    try:
        from nltk.corpus import stopwords as nltk_stopwords
        from nltk.stem import WordNetLemmatizer

        words = nltk_stopwords.words("english")
        lemma = WordNetLemmatizer()
        # Touch both resources so a missing corpus raises here, not mid-pipeline.
        lemma.lemmatize("articles")
        return set(words), lemma
    except Exception as exc:  # LookupError, or nltk not installed at all
        logger.warning(
            "NLTK resources unavailable (%s). Falling back to a built-in "
            "stop-word list and skipping lemmatisation. Run "
            "'python training/prepare_nltk.py' for full-quality preprocessing.",
            exc,
        )
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        return set(ENGLISH_STOP_WORDS), None


_STOPWORDS, _LEMMATIZER = _load_nltk()


@functools.lru_cache(maxsize=200_000)
def _lemmatise_token(token: str) -> str:
    """Reduce a single token to its root form.

    ``lru_cache`` matters for speed: the corpus has roughly 8.5 million tokens
    but only tens of thousands of distinct words, so almost every call becomes
    a dictionary lookup instead of a WordNet traversal.
    """
    if _LEMMATIZER is None:
        return token
    # Noun then verb: catches plurals and standard verb endings.
    token = _LEMMATIZER.lemmatize(token, pos="n")
    return _LEMMATIZER.lemmatize(token, pos="v")


# --------------------------------------------------------------------------
# The preprocessor
# --------------------------------------------------------------------------

class NewsPreprocessor:
    """Cleans raw news text into a bag of normalised words.

    Parameters
    ----------
    strip_source_artifacts:
        Remove publisher markers (Reuters datelines, outlet names, URLs,
        handles, CMS boilerplate). Set ``False`` to reproduce the "naive"
        baseline that ignores this problem.
    remove_stopwords:
        Drop common English function words ("the", "is", "said").
    lemmatize:
        Reduce words to their dictionary root ("articles" -> "article").
    remove_numbers:
        Drop digits. In this corpus digits are mostly dateline noise
        ("21st Century Wire", years, vote counts).
    min_token_length:
        Discard tokens shorter than this after cleaning.
    """

    def __init__(
        self,
        strip_source_artifacts: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        remove_numbers: bool = True,
        min_token_length: int = 3,
    ) -> None:
        self.strip_source_artifacts = bool(strip_source_artifacts)
        self.remove_stopwords = bool(remove_stopwords)
        self.lemmatize = bool(lemmatize)
        self.remove_numbers = bool(remove_numbers)
        self.min_token_length = int(min_token_length)

    # -- individual stages -------------------------------------------------

    def _remove_sources(self, text: str) -> str:
        """Remove publisher identifiers and web-page furniture."""
        text = _DATELINE.sub(" ", text)          # leading "CITY (Reuters) -"
        text = _PAREN_AGENCY.sub(" ", text)      # "(Reuters)" anywhere
        text = _URL.sub(" ", text)               # http://... and www....
        text = _EMAIL.sub(" ", text)
        text = _HANDLE.sub(" ", text)            # @user / #tag
        text = _BARE_OUTLET.sub(" ", text)       # "told Reuters"
        text = _BOILERPLATE.sub(" ", text)       # "FEATURED IMAGE"
        return text

    @staticmethod
    def _strip_accents(text: str) -> str:
        """Turn "café" into "cafe" so accented letters are not deleted."""
        decomposed = unicodedata.normalize("NFKD", text)
        return "".join(ch for ch in decomposed if not unicodedata.combining(ch))

    # -- public API --------------------------------------------------------

    def __call__(self, text: Any) -> str:
        """Clean one document. Safe for ``None``/``NaN``/non-string input."""
        if text is None:
            return ""
        if not isinstance(text, str):
            # float('nan') is the value pandas gives for a missing cell.
            try:
                if text != text:  # NaN check without importing numpy
                    return ""
            except Exception:
                pass
            text = str(text)

        text = text.lower()

        if self.strip_source_artifacts:
            text = self._remove_sources(text)

        text = self._strip_accents(text)

        # This single substitution removes punctuation, digits (if enabled),
        # and every other non-alphabetic character.
        text = _NON_LETTER.sub(" ", text)

        tokens: Iterable[str] = _WORD.findall(text)

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in _STOPWORDS]
        if self.lemmatize:
            tokens = [_lemmatise_token(t) for t in tokens]

        tokens = [t for t in tokens if len(t) >= self.min_token_length]
        return " ".join(tokens)

    def transform(self, texts: Iterable[Any]) -> list[str]:
        """Clean a sequence of documents. Vectorisers call us per-string."""
        return [self(t) for t in texts]

    # -- reporting ---------------------------------------------------------

    def resolved_config(self) -> dict[str, Any]:
        """What this instance will ACTUALLY do, after fallbacks.

        Saved into the model metadata so that inference can warn if it is not
        running the same configuration that produced the model.
        """
        return {
            "strip_source_artifacts": self.strip_source_artifacts,
            "remove_stopwords": self.remove_stopwords,
            "remove_numbers": self.remove_numbers,
            "lemmatize_requested": self.lemmatize,
            "lemmatize_active": bool(self.lemmatize and _LEMMATIZER is not None),
            "stopwords_active": bool(self.remove_stopwords and _STOPWORDS),
            "nltk_stopword_count": len(_STOPWORDS),
            "min_token_length": self.min_token_length,
        }

    def describe(self) -> str:
        c = self.resolved_config()
        return (
            f"source_artifacts={'REMOVED' if c['strip_source_artifacts'] else 'KEPT'}, "
            f"stopwords={'yes' if c['stopwords_active'] else 'no'}, "
            f"lemmatize={'yes' if c['lemmatize_active'] else 'no'}, "
            f"min_token_length={c['min_token_length']}"
        )


def combine_title_text(title: Any, text: Any) -> str:
    """Build the single string the model expects: headline + article body.

    Missing titles are tolerated because a user may submit only a body, and
    because real datasets contain blank headline cells.
    """
    parts = []
    for value in (title, text):
        if value is None:
            continue
        if not isinstance(value, str):
            try:
                if value != value:  # NaN
                    continue
            except Exception:
                pass
            value = str(value)
        value = value.strip()
        if value:
            parts.append(value)
    return " ".join(parts)