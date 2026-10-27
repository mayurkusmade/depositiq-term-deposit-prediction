"""
The single data entry point for the DepositIQ project.

Why this file exists
--------------------
Every notebook and script should read the raw data the same way. If each notebook calls
`pd.read_csv(...)` with its own arguments, it is very easy to forget `sep=";"`, forget to
encode the target, or accidentally read the 10% sample instead of the full dataset. Those
mistakes silently produce wrong numbers that look plausible.

So: notebooks never call `read_csv` on the raw files themselves. They call this module.

Variants
--------
    "full"        data/raw/bank-full.csv                   45,211 x 17   <- primary dataset
    "sample"      data/raw/bank.csv                         4,521 x 17   <- 10% sample of full
    "additional"  data/raw/bank-additional/...full.csv     41,188 x 21   <- richer variant

`load_raw` returns the untouched frame. `load_xy` returns `(X, y)` ready for scikit-learn,
with the target encoded as 1 = subscribed ("yes") and 0 = did not ("no").

How to run
----------
From the repository root, inside the project virtual environment:

    python -m src.data_loader          # loads every variant and prints a summary

Note: `data/raw/` is treated as read-only. Cleaning happens in src/clean.py (T14), and
`duration` (the leakage column) is deliberately NOT dropped here - Phase 4 needs it for the
leakage demonstration, and Phase 5 (T22) handles removing it for the main model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src import config

# ---------------------------------------------------------------------------
# Which file each variant name points at
# ---------------------------------------------------------------------------
VARIANTS: dict[str, Path] = {
    "full": config.BANK_FULL_CSV,
    "sample": config.BANK_CSV,
    "additional": config.BANK_ADDITIONAL_FULL_CSV,
}

# The five social/economic columns that only exist in the "additional" variant.
# Recorded here so the rest of the project can tell the two feature sets apart.
ECONOMIC_COLUMNS: tuple[str, ...] = (
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
)


def _resolve(variant: str) -> Path:
    """
    Turn a variant name into a real file path, failing loudly if it cannot.

    A wrong variant name must never quietly fall back to a default: reading the sample
    when you meant the full dataset is exactly the kind of silent error this module exists
    to prevent.
    """
    if variant not in VARIANTS:
        raise ValueError(
            f"Unknown variant {variant!r}. "
            f"Valid options are: {', '.join(sorted(VARIANTS))}."
        )
    path = VARIANTS[variant]
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset for variant {variant!r} not found at:\n  {path}\n"
            "Check that the raw data is present under data/raw/ (see PROGRESS.md task T02)."
        )
    return path


def load_raw(variant: str = "full", verbose: bool = True) -> pd.DataFrame:
    """
    Load one raw dataset variant exactly as shipped, with no cleaning.

    Parameters
    ----------
    variant : {"full", "sample", "additional"}
        Which file to read. See the module docstring for the sizes.
    verbose : bool, default True
        Print the shape and class balance. Helpful while learning; set to False in loops.

    Returns
    -------
    pandas.DataFrame
        The raw data, target column `y` still included and still as text ("yes"/"no").
    """
    path = _resolve(variant)

    # sep=";" because the UCI files are semicolon-separated. The values are quoted,
    # but pandas' default quotechar already handles that, so no extra argument is needed.
    df = pd.read_csv(path, sep=config.CSV_SEPARATOR)

    if verbose:
        print(f"load_raw({variant!r}) -> {df.shape[0]:,} rows x {df.shape[1]} columns")
        print(f"  source   : {path.name}")
        if config.TARGET in df.columns:
            balance = df[config.TARGET].value_counts(normalize=True).sort_index()
            pretty = ", ".join(f"{k}={v:.1%}" for k, v in balance.items())
            print(f"  target y : {pretty}")

    return df


def load_xy(
    variant: str = "full",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load a variant and split it into features `X` and encoded target `y`.

    `y` is 1 for the positive class (config.POSITIVE_LABEL, "yes" = client subscribed a term
    deposit) and 0 otherwise. This 1/0 encoding is what scikit-learn, the metrics and the
    confusion matrix all expect, so it is done once here instead of in every notebook.

    Returns
    -------
    (X, y) : tuple[pandas.DataFrame, pandas.Series]
        `X` keeps every column except the target. Nothing else is dropped or transformed
        yet - encoding, scaling and leakage removal belong to later tasks.
    """
    df = load_raw(variant, verbose=False)
    path = _resolve(variant)

    if config.TARGET not in df.columns:
        raise KeyError(f"Expected a target column named {config.TARGET!r} in {path}.")

    # Map the two text labels onto 0/1, and check that no unexpected label sneaked in.
    y = df[config.TARGET].map({config.POSITIVE_LABEL: 1, config.NEGATIVE_LABEL: 0})
    if y.isna().any():
        unexpected = sorted(df.loc[y.isna(), config.TARGET].unique())
        raise ValueError(
            f"Target column contains unexpected labels: {unexpected}. "
            f"Expected only {config.NEGATIVE_LABEL!r} and {config.POSITIVE_LABEL!r}."
        )
    y = y.astype("int8")
    y.name = config.TARGET

    # drop() with axis=1 (columns) to build the feature frame.
    X = df.drop(columns=[config.TARGET])

    if verbose:
        positive_rate = y.mean()
        print(f"load_xy({variant!r}) from {path.name}")
        print(f"  X : {X.shape[0]:,} rows x {X.shape[1]} features")
        print(f"  y : {y.shape[0]:,} values -> {int((y == 0).sum()):,} no (0) / "
              f"{int((y == 1).sum()):,} yes (1) = {positive_rate:.1%} positive")
        print(f"  dropped from X : {config.TARGET!r} only (no encoding or scaling yet)")

    return X, y


def variant_report() -> pd.DataFrame:
    """
    Load every variant and return a one-row-per-variant summary table.

    Used by the data-understanding notebook (T11) and handy as a sanity check: it proves all
    three files parse and reports their shapes and class balance side by side.
    """
    rows = []
    for name in sorted(VARIANTS):
        df = load_raw(name, verbose=False)
        y = df[config.TARGET].map({config.POSITIVE_LABEL: 1, config.NEGATIVE_LABEL: 0})
        rows.append(
            {
                "variant": name,
                "file": VARIANTS[name].name,
                "rows": len(df),
                "columns": df.shape[1],
                "features": df.shape[1] - 1,
                "positive_rate": round(y.mean(), 4),
                "has_economic_cols": all(c in df.columns for c in ECONOMIC_COLUMNS),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # `python -m src.data_loader` - loads everything and prints a summary.
    print("=" * 72)
    print("DepositIQ data loader self-check")
    print("=" * 72)
    for variant_name in sorted(VARIANTS):
        load_xy(variant_name)
        print()
    print("=" * 72)
    print(variant_report().to_string(index=False))
    sys.exit(0)
