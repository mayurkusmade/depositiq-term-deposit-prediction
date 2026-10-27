"""
Central configuration for the DepositIQ project.

Why this file exists
--------------------
Every script and notebook needs the same paths and the same constants. Hard-coding them in
lots of places is how projects break when they are moved or graded on another machine.

The trick used here: `ROOT` is computed from the location of THIS file, not typed in by hand.

    src/config.py  ->  parents[0] = .../src  ->  parents[1] = .../depositiq-term-deposit-prediction

So if you clone the project into a different folder, every path below still works with no
edits. There is deliberately no absolute path, drive letter (C:, D:) or username anywhere.

Usage
-----
    from src import config

    df = pd.read_csv(config.BANK_FULL_CSV, sep=config.CSV_SEPARATOR)
    config.ensure_dirs()          # create output folders if a fresh clone is missing them
    config.summary()              # print every path, for debugging
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root and folders (all derived, never hard-coded)
# ---------------------------------------------------------------------------
# Path(__file__) is .../src/config.py; .resolve() makes it a full absolute path;
# parents[1] walks up one level from src/ to the project root.
ROOT: Path = Path(__file__).resolve().parents[1]

# Top-level folders
DATA_DIR: Path = ROOT / "data"
NOTEBOOKS_DIR: Path = ROOT / "notebooks"
SRC_DIR: Path = ROOT / "src"
MODELS_DIR: Path = ROOT / "models"
REPORTS_DIR: Path = ROOT / "reports"
DOCS_DIR: Path = ROOT / "docs"
APP_DIR: Path = ROOT / "app"

# Second-level folders
DATA_RAW: Path = DATA_DIR / "raw"          # original, never edited
DATA_PROCESSED: Path = DATA_DIR / "processed"  # cleaned data + train/test splits
FIGURES_DIR: Path = REPORTS_DIR / "figures"    # every saved chart goes here

# ---------------------------------------------------------------------------
# Dataset files
# ---------------------------------------------------------------------------
# Primary dataset for the project: 45,211 rows x 16 features + target `y`.
BANK_FULL_CSV: Path = DATA_RAW / "bank-full.csv"

# 10% random sample of the above (4,521 rows) - handy for fast experiments.
BANK_CSV: Path = DATA_RAW / "bank.csv"

# Data dictionary / citation for the primary dataset.
BANK_NAMES_TXT: Path = DATA_RAW / "bank-names.txt"

# Optional richer variant: 41,188 rows x 20 features. Adds the five social and
# economic indicators (emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m,
# nr.employed) and uses day_of_week instead of day/month.
BANK_ADDITIONAL_DIR: Path = DATA_RAW / "bank-additional"
BANK_ADDITIONAL_FULL_CSV: Path = BANK_ADDITIONAL_DIR / "bank-additional-full.csv"
BANK_ADDITIONAL_CSV: Path = BANK_ADDITIONAL_DIR / "bank-additional.csv"
BANK_ADDITIONAL_NAMES_TXT: Path = BANK_ADDITIONAL_DIR / "bank-additional-names.txt"

# ---------------------------------------------------------------------------
# Output files produced by later tasks
# ---------------------------------------------------------------------------
TRAIN_CSV: Path = DATA_PROCESSED / "train.csv"   # written in T15 (classic variant, the primary one)
TEST_CSV: Path = DATA_PROCESSED / "test.csv"     # written in T15

# The additional variant is a second, separate pipeline (D19/D20), so it gets its own split rather
# than sharing the classic rows. Keeping the names here means split.py is the only file that knows
# where the halves live.
TRAIN_ADDITIONAL_CSV: Path = DATA_PROCESSED / "train_additional.csv"
TEST_ADDITIONAL_CSV: Path = DATA_PROCESSED / "test_additional.csv"

# ---------------------------------------------------------------------------
# Run constants
# ---------------------------------------------------------------------------
# One seed for the whole project so every split, model and sample is reproducible.
RANDOM_STATE: int = 42

# Fraction of rows held out for the final test set.
TEST_SIZE: float = 0.20

# Name of the target column and its two values.
TARGET: str = "y"
POSITIVE_LABEL: str = "yes"   # client subscribed to a term deposit
NEGATIVE_LABEL: str = "no"

# The UCI files are semicolon-separated and every text value is quoted.
CSV_SEPARATOR: str = ";"

# Fraction of the dataset that is the positive class (used in reports and sanity checks).
KNOWN_POSITIVE_RATE: float = 0.117

# Every folder the project writes into. `ensure_dirs()` creates whichever are missing.
OUTPUT_DIRS: tuple[Path, ...] = (
    DATA_PROCESSED,
    MODELS_DIR,
    FIGURES_DIR,
    DOCS_DIR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_dirs() -> list[Path]:
    """
    Create the output folders if they do not exist yet, and return the ones created.

    Safe to call at the top of any script: it never deletes or overwrites anything.
    Raw data is intentionally excluded - `data/raw/` is treated as read-only so that
    cleaning and feature engineering can always be re-run from the original files.
    """
    created: list[Path] = []
    for folder in OUTPUT_DIRS:
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            created.append(folder)
    return created


def summary() -> None:
    """Print every configured path and whether it currently exists (debug helper)."""
    print(f"ROOT = {ROOT}")
    print("-" * 68)
    for name, path in sorted(globals().items()):
        if isinstance(path, Path):
            print(f"  {name:<26} {'OK ' if path.exists() else '   '} {path}")


# Running `python -m src.config` prints the path check above.
if __name__ == "__main__":
    summary()
