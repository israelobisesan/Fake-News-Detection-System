"""
Loading and cleaning the ISOT Fake News dataset.

What this script is responsible for
-----------------------------------
1. Finding the dataset files on disk.
2. Working out which columns hold the headline and the article body, rather
   than assuming fixed names.
3. Cleaning the records (missing values, empty rows, duplicates).
4. Building the single input string the model will see: ``title + " " + text``.

It does NOT do any token-level cleaning - that lives in
``app.preprocessing`` so that training and prediction share one implementation.

Supported layouts
-----------------
The ISOT dataset ships as two files whose filenames encode the label. That is
the expected layout, but a single combined file with a label column is also
accepted so the project keeps working if the data is supplied differently::

    dataset/
      Fake.csv     -> every row labelled FAKE
      True.csv     -> every row labelled TRUE

    dataset/
      news.csv     -> single file containing a label column (0/1, fake/true,
                      or FAKE/TRUE)
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

# Labels come from the shared preprocessing module so there is one definition.
from app.preprocessing import FAKE, TRUE  # noqa: E402

DATASET_DIR = PROJECT_ROOT / "dataset"

# Candidate filenames, in priority order, for the two-file ISOT layout.
_FAKE_FILE_CANDIDATES = ["Fake.csv", "fake.csv", "Fake.CSV", "fakes.csv"]
_TRUE_FILE_CANDIDATES = ["True.csv", "true.csv", "True.CSV", "real.csv", "genuine.csv"]

# Candidate names for the headline and body columns, matched case-insensitively.
_TITLE_COLUMNS = ["title", "headline", "head_line", "heading"]
_TEXT_COLUMNS = ["text", "article", "body", "content", "news", "article_text"]
_LABEL_COLUMNS = ["label", "class", "target", "is_fake", "fake", "category"]

# Accepted spellings of each label in a combined file.
_LABEL_MAP = {
    "fake": FAKE, "0": FAKE, "false": FAKE, "1": FAKE, "misinformation": FAKE,
    "true": TRUE, "real": TRUE, "genuine": TRUE, "2": TRUE, "fact": TRUE,
}


class DatasetError(RuntimeError):
    """Raised when the dataset is missing or cannot be understood."""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _resolve(columns: list[str], candidates: list[str]) -> str | None:
    """Return the first column matching one of ``candidates``, ignoring case."""
    lowered = {c.lower().strip(): c for c in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _first_existing(directory: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        path = directory / name
        if path.is_file():
            return path
    return None


def _read_csv(path: Path) -> pd.DataFrame:
    """Read one CSV as text, tolerating encoding quirks in the raw files."""
    try:
        return pd.read_csv(path, encoding="utf-8", dtype=str,
                           keep_default_na=False, na_values=[""])
    except UnicodeDecodeError:
        # The ISOT files are valid UTF-8, but be forgiving if a copy was saved
        # with a Windows code page instead.
        return pd.read_csv(path, encoding="cp1252", dtype=str,
                           keep_default_na=False, na_values=[""])


def _normalise_label(value: object) -> str | None:
    """Map a raw label cell onto FAKE / TRUE, or None if unrecognised."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    key = str(value).strip().lower()
    return _LABEL_MAP.get(key)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def load_dataset(dataset_dir: Path | str = DATASET_DIR) -> pd.DataFrame:
    """Load, clean and label the dataset.

    Returns a DataFrame with columns ``text`` (title + body, cleaned),
    ``label`` (``"FAKE"`` or ``"TRUE"``) and ``title``.

    Raises
    ------
    DatasetError
        If no usable dataset can be found or the columns cannot be identified.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise DatasetError(
            f"Dataset folder not found: {dataset_dir}\n"
            "Create it and add Fake.csv and True.csv (see README, 'Preparing the dataset')."
        )

    frames: list[pd.DataFrame] = []

    fake_path = _first_existing(dataset_dir, _FAKE_FILE_CANDIDATES)
    true_path = _first_existing(dataset_dir, _TRUE_FILE_CANDIDATES)

    if fake_path and true_path:
        frames.append(_read_csv(fake_path).assign(label=FAKE, source_file=fake_path.name))
        frames.append(_read_csv(true_path).assign(label=TRUE, source_file=true_path.name))
        layout = f"two-file ISOT layout ({fake_path.name} + {true_path.name})"
    else:
        # Fall back to a single combined CSV that carries its own label column.
        combined = _first_existing(dataset_dir, ["news.csv", "dataset.csv", "combined.csv", "all.csv"])
        if combined is None:
            csvs = sorted(p.name for p in dataset_dir.glob("*.csv"))
            raise DatasetError(
                f"No dataset CSV found in {dataset_dir}.\n"
                f"Expected Fake.csv and True.csv. Files present: {csvs or 'none'}\n"
                "See README, 'Preparing the dataset' for download instructions."
            )
        raw = _read_csv(combined)
        label_col = _resolve(list(raw.columns), _LABEL_COLUMNS)
        if label_col is None:
            raise DatasetError(
                f"{combined.name} has no recognisable label column. "
                f"Columns found: {list(raw.columns)}. Expected one of {_LABEL_COLUMNS}."
            )
        raw["label"] = raw[label_col].map(_normalise_label)
        frames.append(raw.assign(source_file=combined.name))
        layout = f"combined file ({combined.name}, label column '{label_col}')"

    df = pd.concat(frames, ignore_index=True)

    title_col = _resolve(list(df.columns), _TITLE_COLUMNS)
    text_col = _resolve(list(df.columns), _TEXT_COLUMNS)
    if text_col is None:
        raise DatasetError(
            f"Could not identify the article text column. Columns found: {list(df.columns)}"
        )

    # Missing values become empty strings so downstream string ops never see NaN.
    df["text"] = df[text_col].fillna("").astype(str)
    df["title"] = df[title_col].fillna("").astype(str) if title_col else ""
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    df = df[df["label"].isin([FAKE, TRUE])]

    # Drop rows with no body text at all (a headline on its own is not enough).
    before = len(df)
    df = df[df["text"].str.strip() != ""]

    # Remove duplicate articles. Verified against this dataset: no article is
    # labelled both FAKE and TRUE, so keeping the first is safe.
    before_dupes = len(df)
    df = df.drop_duplicates(subset=["title", "text"], keep="first").reset_index(drop=True)

    summary = {
        "layout": layout,
        "title_column": title_col or "(none)",
        "text_column": text_col,
        "rows_before_cleaning": before,
        "rows_dropped_empty": before - before_dupes,
        "rows_dropped_duplicate": before_dupes - len(df),
        "rows_final": len(df),
        "fake_rows": int((df["label"] == FAKE).sum()),
        "true_rows": int((df["label"] == TRUE).sum()),
    }
    df.attrs["load_summary"] = summary
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Log what the loader found. Called by the training script."""
    s = df.attrs.get("load_summary", {})
    print("Dataset loaded")
    print("-" * 62)
    print(f"  layout                     : {s.get('layout', 'n/a')}")
    print(f"  title column               : {s.get('title_column', 'n/a')}")
    print(f"  text column                : {s.get('text_column', 'n/a')}")
    print(f"  rows read                  : {s.get('rows_before_cleaning', len(df)):,}")
    print(f"  dropped (empty body)       : {s.get('rows_dropped_empty', 0):,}")
    print(f"  dropped (duplicates)       : {s.get('rows_dropped_duplicate', 0):,}")
    print(f"  usable records             : {s.get('rows_final', len(df)):,}")
    print(f"      labelled FAKE          : {s.get('fake_rows', 0):,}")
    print(f"      labelled TRUE          : {s.get('true_rows', 0):,}")
    total = max(s.get("rows_final", len(df)), 1)
    fake = s.get("fake_rows", 0)
    print(f"      fake share             : {fake / total * 100:.1f}%")
    print("-" * 62)


if __name__ == "__main__":
    # `python training/data_loader.py` prints the dataset summary on its own,
    # which is handy for confirming the loader reads your files correctly.
    try:
        data = load_dataset()
    except DatasetError as exc:
        print(f"\nERROR: {exc}\n")
        raise SystemExit(1)
    print_summary(data)
    print("\nExample combined record (truncated):\n")
    for i in range(min(2, len(data))):
        row = data.iloc[i]
        combined = (str(row["title"]) + " " + str(row["text"])).strip()
        print(f"[{row['label']}] {combined[:220]}...\n")