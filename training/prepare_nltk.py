"""
One-time NLTK resource preparation (run this BEFORE training).

Why this is a separate script
-----------------------------
The classifier needs two NLTK resources:

  * ``stopwords``  - the English stop-word list used during preprocessing
  * ``wordnet``    - required by ``WordNetLemmatizer`` to reduce words to a root form

Downloading corpora is slow (the wordnet corpus is roughly 10 MB) and requires
network access. We do it ONCE here, ahead of time, so that:

  * ``training/train.py`` can assume the resources already exist, and
  * the Flask application never tries to download anything at start-up.

The resources land in NLTK's default data directory (``%APPDATA%\\nltk_data``
on Windows), so they persist across sessions.

Two failure modes are handled explicitly, because both occur in practice:

1.  ``nltk.data.find()`` raises ``LookupError`` when a resource is absent - it does
    not return a falsy value. Every check here catches that.

2.  ``nltk.download()`` sometimes reports success while leaving the corpus
    *un-extracted*: the ``.zip`` lands on disk but the directory NLTK actually
    loads never appears. The resource then still raises ``LookupError``. This
    script detects that state and unzips the archive itself.

If the corpus genuinely cannot be fetched the script still exits cleanly: the
project keeps working, falling back to a built-in stop-word list and skipping
lemmatisation (see ``app/preprocessing.py``).

Usage
-----
    python training/prepare_nltk.py
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

# Allow `python training/prepare_nltk.py` to be run from any directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nltk  # noqa: E402

REQUIRED_RESOURCES = ["stopwords", "wordnet"]


# --------------------------------------------------------------------------
# Checking
# --------------------------------------------------------------------------

def _find_corpus(resource: str) -> str | None:
    """Locate an extracted corpus, or None. Never raises.

    ``nltk.data.find`` raises ``LookupError`` rather than returning None, which
    is easy to mistake for a successful check.
    """
    try:
        return str(nltk.data.find(f"corpora/{resource}"))
    except LookupError:
        return None


def _verify_usable() -> tuple[bool, str]:
    """Actually use both resources, because presence on disk is not enough.

    Returns (ok, detail).
    """
    try:
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        count = len(stopwords.words("english"))
        lemma = WordNetLemmatizer().lemmatize("articles")
    except Exception as exc:  # LookupError, or a partially extracted corpus
        return False, str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
    return True, f"stopwords={count}, lemmatize('articles') -> '{lemma}'"


# --------------------------------------------------------------------------
# Repairing
# --------------------------------------------------------------------------

def _candidate_data_dirs() -> list[Path]:
    """Directories NLTK may have downloaded into, most specific first."""
    dirs = [Path(p) for p in nltk.data.path]
    # nltk.data.path can contain odd entries such as "C:\\Users\\Delice/nltk_data".
    extra = [Path(os.environ.get("APPDATA", "")) / "nltk_data",
             Path.home() / "nltk_data",
             Path(os.environ.get("PROGRAMDATA", "")) / "nltk_data"]
    for path in extra:
        if str(path) not in [str(d) for d in dirs]:
            dirs.append(path)
    return [d / "corpora" for d in dirs]


def _extract_stuck_archive(resource: str) -> str | None:
    """Unzip ``<resource>.zip`` if the archive exists but was never extracted.

    This is the state ``nltk.download()`` leaves behind on some Windows
    machines: it prints success and writes the zip, but the directory that
    ``WordNetLemmatizer`` reads never materialises.
    """
    for corpora_dir in _candidate_data_dirs():
        archive = corpora_dir / f"{resource}.zip"
        if not archive.is_file():
            continue
        target = corpora_dir / resource
        if target.is_dir():
            continue
        try:
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(corpora_dir)
        except (zipfile.BadZipFile, OSError) as exc:
            print(f"  [warn]    {archive.name} is corrupt ({exc}); delete it and re-run")
            continue
        if target.is_dir():
            print(f"  [fixed]   {resource} was downloaded but not extracted - unzipped it now")
            return str(target)
    return None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def prepare() -> bool:
    """Make every required resource available. Returns True if all are usable."""
    for resource in REQUIRED_RESOURCES:
        if _find_corpus(resource):
            print(f"  [ok]      {resource} already available")
            continue

        print(f"  [missing] {resource} - downloading ...")
        try:
            nltk.download(resource, quiet=True, force=True)
        except Exception as exc:
            print(f"  [warn]    download failed: {exc}")

        # The download may have produced only a .zip; unpack it if so.
        if not _find_corpus(resource):
            _extract_stuck_archive(resource)

        if _find_corpus(resource):
            print(f"  [ok]      {resource} ready")
        else:
            print(f"  [FAILED]  {resource} is still unavailable")

    ok, detail = _verify_usable()
    return ok, detail


def main() -> int:
    print("=" * 66)
    print("NLTK resource preparation")
    print("=" * 66)

    ok, detail = prepare()

    print("-" * 66)
    if ok:
        print(f"Verified working: {detail}")
        print("Next:  python training/train.py")
        return 0

    print(f"Resources are NOT fully usable: {detail}")
    print()
    print("The project will still run, but with degraded preprocessing:")
    print("  * a built-in English stop-word list replaces NLTK's")
    print("  * lemmatisation is skipped")
    print()
    print("Try these, in order:")
    print("  1. Check your internet connection, then re-run this script.")
    print("  2. Delete the half-downloaded archives and re-run:")
    print('       Remove-Item "$env:APPDATA\\nltk_data\\corpora\\wordnet.zip" -ErrorAction SilentlyContinue')
    print("  3. As a last resort, download wordnet.zip manually from")
    print("       https://github.com/nltk/nltk_data/raw/gh-pages/packages/corpora/wordnet.zip")
    print('     and unzip it into "$env:APPDATA\\nltk_data\\corpora\\".')
    return 1


if __name__ == "__main__":
    raise SystemExit(main())