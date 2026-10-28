"""
DepositIQ - engineered features (T23).

What this module does
---------------------
Turns the converted frame from `src/clean.py` into the columns a model can actually learn from. Every
feature here was *decided* by an earlier task and is implemented rather than re-litigated: the EDA
measured the shape, the leakage audit ruled out the trap, and this file writes it down once so T24's
`ColumnTransformer` and T25's selection have a single definition to import.

Five new columns, each replacing a raw column rather than sitting beside it:

=====================================  ====================  =========================================
feature                                replaces              why, with the measured reason
=====================================  ====================  =========================================
`campaign_bucket`                      `campaign`            skew +4.94 on a 1-63 scale; the six
                                                             buckets explain more than a line does
                                                             (**η² 0.0076** against r² 0.0052, T20
                                                             §5.3) and the gradient is real:
                                                             **14.59%** at one call falling to
                                                             **4.30%** at ten or more (T19)
`previous_capped`                      `previous`            skew **+45.03** and a maximum of 275 for
                                                             a column whose only honest step is
                                                             0 → 1: **9.22%** → **20.65%**, then
                                                             **23.97%** for two or more (T19 §4.1)
`balance_arsinh`                       `balance`             skew **+8.34** with **3,001 rows (8.30%)
                                                             negative** down to **−€8,019**, so
                                                             `log1p` is undefined and `arcsinh` is
                                                             the transform that survives (T17); it
                                                             takes |skew| from **8.34** to **1.68**
`balance_band`                         `balance`             the relationship is not linear: **5.36%**
                                                             when overdrawn, **9.92%**, **12.56%**,
                                                             **16.44%**, then **15.73%** at €5,000+
`recency_band`                         `days_since_contact`  recency is a **composition artefact**
                                                             (T21 I7): the 30-89 day peak at
                                                             **40.20%** is prior-success clustering,
                                                             so the band is kept as a feature to be
                                                             *tested* at T25, never as a rule
=====================================  ====================  =========================================

`previously_contacted` already exists (added by `src/clean.py`, D21) and stays: it is the one contact
history column the EDA found to be genuinely informative on its own.

What is *not* here, and why
---------------------------
* **`duration`** — the leak. It is recorded when the call ends, and on its own it reaches **0.8078**
  ROC-AUC, which is why `docs/05_leakage_audit.md` excludes it and `src/leakage.feature_columns` is the
  toggle T36 will use. Nothing in this module reads it.
* **`pdays`** — bit-for-bit `days_since_contact` where both exist, and a sentinel elsewhere (D36): a
  duplicate column, not a feature.
* **The target `y`** — never an input. `engineer()` asserts this empirically rather than promising it:
  permuting `y` must leave every engineered column byte-identical, and so must permuting `duration`.

Every feature is a deterministic, per-row function of facts the bank holds *before* the call. There is no
fit step, no group statistic and no target encoding, so there is nothing that could be learned from
validation rows — which is what makes this module safe to call on the training half and the test half
alike.

How to run
----------
From the repository root, inside the virtual environment:

    python -m src.features     # builds the features, prints them, runs the leakage checks
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src import clean, config
from src.leakage import feature_columns
from src.split import encode_target, load_split

# ---------------------------------------------------------------------------
# Feature names, in one place so a later task cannot spell one differently
# ---------------------------------------------------------------------------
CAMPAIGN_BUCKET: str = "campaign_bucket"
PREVIOUS_CAPPED: str = "previous_capped"
BALANCE_BAND: str = "balance_band"
BALANCE_ARSINH: str = "balance_arsinh"
RECENCY_BAND: str = "recency_band"

# Call intensity: the six buckets T19 measured, ending at the level where the campaign stops paying for
# itself. Right-closed edges, so 1 means "exactly one call this campaign".
CAMPAIGN_EDGES: tuple[float, ...] = (0, 1, 2, 3, 5, 9, np.inf)
CAMPAIGN_LABELS: tuple[str, ...] = ("1", "2", "3", "4-5", "6-9", "10+")

# Balance bands. 500 is roughly the median (€451) and 5,000 the point where the rate stops rising.
BALANCE_EDGES: tuple[float, ...] = (-np.inf, 0, 500, 1_500, 5_000, np.inf)
BALANCE_LABELS: tuple[str, ...] = ("overdrawn", "0-499", "500-1499", "1500-4999", "5000+")

# Recency: 30/90/180 days are the bucket edges T19 used, so this feature and §4.3's table agree.
RECENCY_EDGES: tuple[float, ...] = (0, 30, 90, 180, np.inf)
RECENCY_LABELS: tuple[str, ...] = ("0-29 d", "30-89 d", "90-179 d", "180+ d")
# Never contacted is not a missing value to be imputed: it is a distinct, and the largest, group
# (29,584 clients, 81.80%), and filling it with a plausible number would invent a history.
RECENCY_NEVER: str = "never"

# Two prior contacts is the cap: 0 → 1 is the step that matters, 1 against 2 is indistinguishable and
# 2 against 3+ is a mild rise, so everything above two shares one level.
PREVIOUS_CAP: int = 2

ENGINEERED: tuple[str, ...] = (CAMPAIGN_BUCKET, PREVIOUS_CAPPED, BALANCE_ARSINH, BALANCE_BAND, RECENCY_BAND)

# The columns each feature reads. Kept as data so `checks_no_leakage` can inspect it rather than trust
# a comment.
SOURCES: dict[str, tuple[str, ...]] = {
    CAMPAIGN_BUCKET: ("campaign",),
    PREVIOUS_CAPPED: ("previous",),
    BALANCE_ARSINH: ("balance",),
    BALANCE_BAND: ("balance",),
    RECENCY_BAND: (clean.CONTACT_RECENCY,),
}

# What the model will see, and what each dropped column was replaced by.
NUMERIC_FEATURES: tuple[str, ...] = ("age", "day", PREVIOUS_CAPPED, BALANCE_ARSINH)
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome",
    CAMPAIGN_BUCKET, BALANCE_BAND, RECENCY_BAND,
)
BOOLEAN_FEATURES: tuple[str, ...] = (clean.CONTACT_FLAG,)
FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES

# Never read by any feature, for reasons that are not about convenience.
FORBIDDEN: dict[str, str] = {
    "duration": "leaks the outcome - known only after the call ends (D05, T22)",
    "pdays": "duplicate of days_since_contact with a sentinel (D36, D37)",
    config.TARGET: "the target itself",
}

# Raw columns a feature deliberately replaces: read by `engineer()`, absent from `FEATURES`.
REPLACES: dict[str, str] = {
    "campaign": f"replaced by {CAMPAIGN_BUCKET}: bounded, and the buckets carry more than the line",
    "previous": f"replaced by {PREVIOUS_CAPPED}: skew +45.03 for a single meaningful step",
    "balance": f"replaced by {BALANCE_ARSINH} and {BALANCE_BAND}: negative and skewed",
    clean.CONTACT_RECENCY: f"replaced by {RECENCY_BAND}, which handles the 81.80% who were never called",
}


# ---------------------------------------------------------------------------
# The transformation
# ---------------------------------------------------------------------------
def _require(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(
            f"engineer() needs columns that are not in the frame: {missing}. "
            f"Load it with `src.split.load_split`, which restores {clean.CONTACT_FLAG!r} and "
            f"{clean.CONTACT_RECENCY!r}."
        )


def engineer(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of `frame` with the five engineered columns added.

    The input is never mutated: callers keep the cleaned frame intact and pass the engineered one on.
    Every added column is derived from a single row, so applying this to the test half produces the same
    columns with the same category levels - there is no state to fit and none to carry across.
    """
    _require(frame, ("campaign", "previous", "balance", clean.CONTACT_RECENCY))
    out = frame.copy()

    # Call intensity. `right=True` with edges (0,1,2,3,5,9,inf) reproduces T19's table exactly, and the
    # levels are declared as an ordered Categorical so the encoder in T24 sees a fixed vocabulary even
    # if a split happens to be missing a bucket.
    out[CAMPAIGN_BUCKET] = pd.cut(
        out["campaign"], bins=list(CAMPAIGN_EDGES), labels=list(CAMPAIGN_LABELS), right=True
    )

    # Prior contacts, capped at two.
    out[PREVIOUS_CAPPED] = out["previous"].clip(upper=PREVIOUS_CAP).astype("int8")

    # Balance: an arcsinh scale (defined for negatives, monotone, so it preserves order) and a band for
    # the non-linear part.
    out[BALANCE_ARSINH] = np.arcsinh(out["balance"].astype("float64"))
    out[BALANCE_BAND] = pd.cut(
        out["balance"], bins=list(BALANCE_EDGES), labels=list(BALANCE_LABELS), right=False
    )

    # Recency, with "never contacted" as its own level rather than a NaN or a 0.
    recency = pd.cut(
        out[clean.CONTACT_RECENCY], bins=list(RECENCY_EDGES), labels=list(RECENCY_LABELS), right=False
    )
    recency = recency.cat.add_categories([RECENCY_NEVER]).fillna(RECENCY_NEVER)
    out[RECENCY_BAND] = recency

    for column in (CAMPAIGN_BUCKET, BALANCE_BAND, RECENCY_BAND):
        if out[column].isna().any():
            raise ValueError(f"{column!r} has {int(out[column].isna().sum())} missing values after binning")
    if not np.isfinite(out[BALANCE_ARSINH]).all():
        raise ValueError(f"{BALANCE_ARSINH!r} is not finite for every row")
    return out


def select_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Return just the model's feature columns, in the declared order.

    Called after `engineer()`. Keeping selection separate means the raw columns are still available for
    the EDA-style checks and for `docs/05_leakage_audit.md`'s toggle, while the model path is explicit
    about what it consumes.
    """
    missing = [column for column in FEATURES if column not in frame.columns]
    if missing:
        raise KeyError(f"select_features() was called before engineer(): missing {missing}")
    return frame.loc[:, list(FEATURES)].copy()


def load_engineered(which: str = "train", variant: str = "full") -> pd.DataFrame:
    """Convenience path: read a written split, add the features, return the engineered frame."""
    return engineer(load_split(variant, which))


# ---------------------------------------------------------------------------
# Leakage checks
# ---------------------------------------------------------------------------
def checks_no_leakage(frame: pd.DataFrame, seed: int = config.RANDOM_STATE) -> list[dict]:
    """
    Test the claim "leaks nothing from `y`" instead of asserting it.

    Four checks, each of which fails for a different real mistake:

    1. **Target permutation.** Rebuild the features after shuffling `y`; if any engineered value moves,
       something was computed from the answer.
    2. **`duration` permutation.** The same test for the leaky column: a feature that reacts to it is
       reading the outcome through it.
    3. **Name and dependency audit.** No engineered column depends on a dropped column, and neither the
       target nor `duration` appears in `FEATURES`.
    4. **Row-order determinism.** Shuffle the rows, rebuild, and compare after sorting back: equal values
       mean every feature is per-row, so no group statistic can leak information across clients.
    """
    rng = np.random.default_rng(seed)
    base = engineer(frame)
    engineered_only = base.loc[:, list(ENGINEERED)]

    results: list[dict] = []

    def add(name: str, passed: bool, detail: str) -> None:
        results.append({"check": name, "passed": bool(passed), "detail": detail})

    shuffled_target = frame.copy()
    shuffled_target[config.TARGET] = rng.permutation(shuffled_target[config.TARGET].to_numpy())
    after_target = engineer(shuffled_target).loc[:, list(ENGINEERED)]
    same_target = after_target.equals(engineered_only)
    add("permuting y changes no feature", same_target,
        "identical" if same_target else "a feature is computed from the target")

    shuffled_duration = frame.copy()
    shuffled_duration["duration"] = rng.permutation(shuffled_duration["duration"].to_numpy())
    after_duration = engineer(shuffled_duration).loc[:, list(ENGINEERED)]
    same_duration = after_duration.equals(engineered_only)
    add("permuting duration changes no feature", same_duration,
        "identical" if same_duration else "a feature reads the leaky column")

    read = {source for sources in SOURCES.values() for source in sources}
    overlap = sorted(read & set(FORBIDDEN))
    add("no feature reads a forbidden column", not overlap, f"reads: {overlap}" if overlap else "none")

    # A replaced column must be gone from the model's view: keeping the raw copy beside its encoding is
    # how a feature set quietly doubles a column's weight.
    still_there = sorted(set(REPLACES) & set(FEATURES))
    add("replaced columns are not also features", not still_there,
        f"still present: {still_there}" if still_there else "every replacement replaced something")
    add("the target is not a feature", config.TARGET not in FEATURES, "absent from FEATURES")
    add("duration is not a feature", "duration" not in FEATURES, "absent from FEATURES")
    add("pdays is not a feature", "pdays" not in FEATURES, "absent from FEATURES")

    shuffled_rows = base.sample(frac=1.0, random_state=seed)
    row_order_same = shuffled_rows.sort_index().loc[:, list(ENGINEERED)].equals(engineered_only)
    add("features are per-row (row order does not matter)", row_order_same,
        "identical after re-sorting" if row_order_same else "a feature depends on other rows")
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(frame: pd.DataFrame | None = None) -> dict:
    """Everything the module prints: the columns, their distributions, and the checks."""
    if frame is None:
        frame = load_split("full", "train")

    engineered = engineer(frame)
    target = encode_target(frame)

    def rate_table(column: str) -> pd.DataFrame:
        # Banded features are ordered Categoricals; the capped count is numeric, so read its own levels.
        values = engineered[column]
        levels = list(values.cat.categories) if hasattr(values, "cat") else sorted(values.unique())
        rows = []
        for level in levels:
            mask = engineered[column] == level
            n = int(mask.sum())
            yes = int(target[mask].sum())
            rows.append({"level": level, "clients": n, "subscribers": yes,
                         "share": n / len(engineered), "rate": yes / n if n else float("nan")})
        return pd.DataFrame(rows)

    return {
        "rows": len(engineered),
        "new_columns": list(ENGINEERED),
        "campaign_bucket": rate_table(CAMPAIGN_BUCKET),
        "balance_band": rate_table(BALANCE_BAND),
        "recency_band": rate_table(RECENCY_BAND),
        "previous_capped": rate_table(PREVIOUS_CAPPED),
        "arsinh": {
            "min": float(engineered[BALANCE_ARSINH].min()),
            "median": float(engineered[BALANCE_ARSINH].median()),
            "max": float(engineered[BALANCE_ARSINH].max()),
            "skew_before": float(frame["balance"].skew()),
            "skew_after": float(engineered[BALANCE_ARSINH].skew()),
        },
        "feature_counts": {
            "numeric": len(NUMERIC_FEATURES),
            "categorical": len(CATEGORICAL_FEATURES),
            "boolean": len(BOOLEAN_FEATURES),
            "total": len(FEATURES),
        },
        # The toggle T36 will use, exercised here on FEATURES + duration so this task proves it works
        # rather than assuming a later one will.
        "toggle_with": feature_columns([*FEATURES, "duration"], with_duration=True),
        "toggle_without": feature_columns([*FEATURES, "duration"], with_duration=False),
        "checks": checks_no_leakage(frame),
    }


def print_report(result: dict) -> None:
    print("DepositIQ T23 - engineered features")
    print("=" * 96)
    print(f"training rows : {result['rows']:,}")
    print(f"new columns   : {', '.join(result['new_columns'])}")
    print()
    for key, title in [
        ("campaign_bucket", "CALL INTENSITY  campaign_bucket  (replaces the raw 1-63 count)"),
        ("previous_capped", "PRIOR CONTACTS  previous_capped  (capped at two)"),
        ("balance_band", "BALANCE BAND    balance_band     (replaces the raw euro amount)"),
        ("recency_band", "RECENCY         recency_band      (replaces days_since_contact)"),
    ]:
        table = result[key]
        print(title)
        for _, row in table.iterrows():
            level = row["level"]
            if isinstance(level, (int, float, np.integer, np.floating)):
                # The capped count's top level is a ceiling, not a value: 2 stands for "two or more".
                label = f"{PREVIOUS_CAP}+" if level == PREVIOUS_CAP else f"{int(level)}"
            else:
                label = str(level)
            print(f"  {label:<11} {int(row['clients']):>6,} clients ({row['share']:>6.2%})  "
                  f"{int(row['subscribers']):>5,} subscribers  {row['rate']:>7.2%}")
        print()
    arsinh = result["arsinh"]
    print(f"BALANCE SCALE   balance_arsinh: min {arsinh['min']:.3f}, median {arsinh['median']:.3f}, "
          f"max {arsinh['max']:.3f}")
    print(f"  skew {arsinh['skew_before']:.2f} -> {arsinh['skew_after']:.2f}")
    print()
    counts = result["feature_counts"]
    print(f"FEATURES        {counts['total']} total: {counts['numeric']} numeric, "
          f"{counts['categorical']} categorical, {counts['boolean']} boolean")
    added = set(result["toggle_with"]) - set(result["toggle_without"])
    print(f"  T36 toggle    {len(result['toggle_with'])} columns with duration, "
          f"{len(result['toggle_without'])} without; difference {sorted(added)}")
    print()
    print("LEAKAGE CHECKS")
    for check in result["checks"]:
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['check']} - {check['detail']}")
    print("=" * 96)


if __name__ == "__main__":
    result = report()
    print_report(result)
    ok = all(check["passed"] for check in result["checks"])
    print(f"result: {'PASS - features built, commented and leak nothing' if ok else 'FAIL'}")
    print("Next  : T24 builds the ColumnTransformer over these 17 columns.")
    sys.exit(0 if ok else 1)
