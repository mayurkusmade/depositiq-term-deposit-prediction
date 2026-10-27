"""
DepositIQ cleaning decisions: the 'unknown' policy and the 'pdays' sentinel (task T12).

This module implements the two decisions that Phase 3 has to make BEFORE any scaling, encoding
or modelling happens. Both are about values that look ordinary but are not.

Decision 1 - 'unknown' is kept as an explicit CATEGORY (no imputation, no row deletion)
======================================================================================
The data has ZERO true NaN cells, in either variant. That is not the same as having no missing
values: the missingness is stored as the literal string "unknown", and the dataset's own
documentation says so - "There are several missing values in some categorical attributes, all
coded with the 'unknown' label."

Three options were considered, and the measurements below are what decided it (all re-measured
by T11, notebooks/01_data_understanding.ipynb):

  Option A - convert 'unknown' to NaN and impute the mode/median.
      Rejected. It would fabricate data on a large scale: 'poutcome' is 'unknown' for 36,959 of
      the 45,211 classic rows (81.75%). Imputing there would invent a previous-campaign history
      for clients who simply had no previous campaign, and it would do so for 82% of the primary
      dataset. The same applies to 'default' in the additional variant (8,597 rows, 20.9%).

  Option B - delete every row containing an 'unknown'.
      Rejected outright. Listwise deletion would remove 37,369 classic rows (82.65%) and 10,700
      additional rows (25.98%). In the primary dataset that throws away more than four clients
      in five, and - because subscription rate varies sharply by segment - it would bias what
      remains rather than just shrink it.

  Option C - keep 'unknown' as its own level.  <-- CHOSEN
      It preserves every row, it matches how the data custodian encoded the data, and it lets
      the model test whether the missingness itself is informative. That last point is not a
      claim, it is a possibility left open: "contact = unknown" may mean an unrecorded channel,
      and "poutcome = unknown" definitely means "no previous campaign", which is real
      information rather than a gap. The models chosen in D08 (tree ensembles) can isolate such
      a level; for the Logistic Regression baseline in T26 it becomes its own one-hot column,
      which is visible and reviewable in the coefficient table.

The implementation goes one step past "leave it alone". A column that mixes real NaN with the
string "unknown" has TWO representations of the same idea, and every later step (encoders,
groupby, value_counts) would handle them differently. `unify_unknown()` therefore rewrites any
NaN in a categorical column to the literal "unknown", so from here on there is exactly one
marker. On the current files this converts nothing - there are no NaNs to convert - and the
report says so rather than implying work was done. It exists so that a future data refresh
cannot silently introduce a second representation of missingness.

Stated limitation: this module does not model WHY values are missing (it is not a MAR/MNAR
analysis), and keeping a level is not evidence that the level matters. If 'unknown' turns out to
be noise, it will show up as near-zero importance or a near-zero coefficient in T25/T26.

Decision 2 - 'pdays': the sentinel is removed, and the flag comes from 'previous' (D21)
======================================================================================
'pdays' is not a numeric column, it is a pair of things wearing a numeric coat: a sentinel
meaning "never contacted before", plus a real day count for everyone else. In the classic set
the sentinel is -1; in the additional set it is 999. Two consequences:

  1. The raw mean is meaningless - it averages a flag with day counts (40.2 classic, 962.5
     additional, while both medians are the sentinel itself).
  2. The sentinel is NOT safe to use as the "was contacted before" flag, because it disagrees
     with the independent witness 'previous' (contacts made before this campaign):

        classic    : pdays != -1  and  previous > 0  select the SAME 8,257 rows - 0 contradictions
        additional : 4,110 of the 39,673 sentinel rows have previous in 1-6 (every one of them
                     with poutcome = 'failure'), so 999 there also covers clients whose previous
                     campaign failed

So the flag is derived from 'previous > 0' (decision D21). It is the portable choice: it agrees
EXACTLY with 'poutcome != nonexistent' in both variants, and it needs no per-variant sentinel
constant. Using the sentinel instead would mislabel those 4,110 clients as never contacted and
hand them a recency value that means nothing.

Two derived columns are added, and nothing is dropped:

    previously_contacted : bool,  'previous > 0'
    days_since_contact   : float,  the real pdays value, NaN where pdays is the sentinel

NaN is used deliberately rather than an imputed number: a day count we do not have cannot be
invented, and XGBoost (D08) treats missing values as a learned branch direction rather than
requiring them to be filled. Raw 'pdays' is retained alongside them for traceability; the main
model is expected to use the flag and the cleaned recency, and T24 decides that exclusion in the
ColumnTransformer rather than here.

Scope
=====
This module makes those two decisions, applies them, and CHECKS the result. It does not scale,
encode, select features or touch data/raw/, and it is not a second reader of the raw CSVs (decision
D13): everything loads through src/data_loader.py.

The converted frame it returns is what T15 splits and persists to data/processed/ (train.csv /
test.csv), so this module deliberately writes no data files of its own.

How to run
----------
From the repository root, inside the project virtual environment:

    python -m src.clean        # converts both variants, prints before/after, runs the checks

Exit status is 0 only when every Phase 3 check passes, so this doubles as the pipeline's guard rail.
"""

from __future__ import annotations

import sys

import pandas as pd

from src import config
from src import data_loader as load

# ---------------------------------------------------------------------------
# Constants - the policy, stated once so the code and the report cannot drift apart
# ---------------------------------------------------------------------------
UNKNOWN: str = "unknown"

POLICY: str = (
    "'unknown' is kept as an explicit category: no imputation (it would fabricate up to 81.75% "
    "of 'poutcome'), no row deletion (it would remove 82.65% of the primary dataset), and any "
    "NaN in a categorical column is normalised to the same literal marker"
)

# The sentinel is a property of the VARIANT, not of the column (T11 section 7). Hard-coding -1
# would silently corrupt the additional set.
SENTINEL_BY_VARIANT: dict[str, int] = {"full": -1, "additional": 999}

# Measured in T11, recorded here so the report can flag an unexpected change rather than let a
# new value pass unnoticed.
UNKNOWN_COLUMNS_BY_VARIANT: dict[str, tuple[str, ...]] = {
    "full": ("contact", "education", "job", "poutcome"),
    "additional": ("default", "education", "housing", "job", "loan", "marital"),
}

RAW_PDAYS: str = "pdays"
CONTACT_FLAG: str = "previously_contacted"
CONTACT_RECENCY: str = "days_since_contact"
DERIVED_COLUMNS: tuple[str, str] = (CONTACT_FLAG, CONTACT_RECENCY)


def categorical_columns(df: pd.DataFrame) -> list[str]:
    """Text columns, which are the only place 'unknown' can appear."""
    return [column for column in df.columns if df[column].dtype == object]


# ---------------------------------------------------------------------------
# Decision 1 - the 'unknown' policy
# ---------------------------------------------------------------------------
def unify_unknown(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply the chosen 'unknown' policy: keep the level, and make it the ONLY missing marker.

    Returns the cleaned frame and a per-column report. The frame is copied, so the caller's data
    is never modified in place.
    """
    cleaned = df.copy()
    rows = []
    for column in categorical_columns(cleaned):
        series = cleaned[column]
        nan_cells = int(series.isna().sum())
        if nan_cells:
            # Normalise, do not impute: a NaN becomes the documented missing marker, not a guess
            # at the real category.
            cleaned[column] = series.fillna(UNKNOWN)
        unknown_cells = int((cleaned[column] == UNKNOWN).sum())
        rows.append(
            {
                "column": column,
                "nan_to_unknown": nan_cells,
                "unknown_cells": unknown_cells,
                "share_of_rows": unknown_cells / len(cleaned) if len(cleaned) else 0.0,
                "levels": int(cleaned[column].nunique()),
            }
        )
    report = pd.DataFrame(rows).sort_values("unknown_cells", ascending=False).reset_index(drop=True)
    return cleaned, report


# ---------------------------------------------------------------------------
# Decision 2 - the sentinel, and the contact-history columns
# ---------------------------------------------------------------------------
def add_contact_history(df: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, dict]:
    """
    Replace the 'pdays' sentinel with a flag plus a real (or genuinely missing) day count.

    The flag comes from 'previous > 0', not from the sentinel - see decision D21 and the module
    docstring. Returns the frame with two added columns and a report dict.
    """
    if variant not in SENTINEL_BY_VARIANT:
        raise ValueError(
            f"Unknown variant {variant!r}. Valid options are: "
            f"{', '.join(sorted(SENTINEL_BY_VARIANT))}."
        )
    for required in (RAW_PDAYS, "previous"):
        if required not in df.columns:
            raise KeyError(f"Expected a {required!r} column to build the contact history from.")

    sentinel = SENTINEL_BY_VARIANT[variant]
    cleaned = df.copy()

    # The independent witness: contacts performed before this campaign. bool, not int, because it
    # is a fact about the client rather than a measurement.
    cleaned[CONTACT_FLAG] = cleaned["previous"] > 0

    # The real day count, and NaN exactly where the sentinel sat. NaN stays NaN: the day count is
    # genuinely unknown there, and inventing one would be inventing history.
    cleaned[CONTACT_RECENCY] = cleaned[RAW_PDAYS].where(cleaned[RAW_PDAYS] != sentinel)

    sentinel_rows = int((cleaned[RAW_PDAYS] == sentinel).sum())
    contradictions = int((cleaned[CONTACT_FLAG] & (cleaned[RAW_PDAYS] == sentinel)).sum())
    orphan_days = int((~cleaned[CONTACT_FLAG] & (cleaned[RAW_PDAYS] != sentinel)).sum())

    report: dict = {
        "variant": variant,
        "sentinel": sentinel,
        "rows": len(cleaned),
        "sentinel_rows": sentinel_rows,
        "previously_contacted": int(cleaned[CONTACT_FLAG].sum()),
        "previously_contacted_share": float(cleaned[CONTACT_FLAG].mean()),
        "days_since_contact_available": int(cleaned[CONTACT_RECENCY].notna().sum()),
        "days_since_contact_missing": int(cleaned[CONTACT_RECENCY].isna().sum()),
        # Reported, not hidden: these are the rows where the sentinel would have given a different
        # answer from 'previous', and they are the reason D21 exists.
        "sentinel_rows_with_previous_contacts": contradictions,
        "day_count_without_previous_contacts": orphan_days,
        "warnings": [],
    }

    if contradictions:
        report["warnings"].append(
            f"{contradictions:,} rows carry the {sentinel} sentinel while 'previous' > 0, so the "
            f"sentinel alone would mislabel them as never contacted. The flag is derived from "
            f"'previous > 0' (D21) and these rows correctly get no day count."
        )
    if orphan_days:
        report["warnings"].append(
            f"{orphan_days:,} rows have a real day count but 'previous' == 0, which should not "
            f"happen; treated as an inconsistency to resolve before modelling."
        )
    return cleaned, report


# ---------------------------------------------------------------------------
# Compose both decisions
# ---------------------------------------------------------------------------
def dtype_summary(df: pd.DataFrame) -> dict[str, list[str]]:
    """Group column names by dtype, so a before/after comparison is readable at a glance."""
    summary: dict[str, list[str]] = {}
    for column in df.columns:
        summary.setdefault(str(df[column].dtype), []).append(column)
    return summary


def class_balance(df: pd.DataFrame) -> dict:
    """Target counts and the positive rate, for the before/after comparison."""
    counts = df[config.TARGET].value_counts()
    return {
        "rows": len(df),
        "no": int(counts.get(config.NEGATIVE_LABEL, 0)),
        "yes": int(counts.get(config.POSITIVE_LABEL, 0)),
        "positive_rate": float((df[config.TARGET] == config.POSITIVE_LABEL).mean()),
    }


def clean_frame(df: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, dict]:
    """
    Apply both T12 decisions to an already-loaded frame.

    Order matters only in that the sentinel work does not depend on the 'unknown' work; both run
    on every column so neither can be skipped by accident.
    """
    unified, unknown_report = unify_unknown(df)
    cleaned, sentinel_report = add_contact_history(unified, variant)

    target_before = float((df[config.TARGET] == config.POSITIVE_LABEL).mean())
    target_after = float((cleaned[config.TARGET] == config.POSITIVE_LABEL).mean())

    report = {
        "variant": variant,
        "policy": POLICY,
        "rows_before": len(df),
        "rows_after": len(cleaned),
        "columns_before": df.shape[1],
        "columns_after": cleaned.shape[1],
        "new_columns": list(DERIVED_COLUMNS),
        "target_positive_rate_before": target_before,
        "target_positive_rate_after": target_after,
        "rows_lost_if_unknown_were_deleted": int(
            df[categorical_columns(df)].eq(UNKNOWN).any(axis=1).sum()
        ),
        "unknown_report": unknown_report,
        "sentinel_report": sentinel_report,
        "dtypes_before": dtype_summary(df),
        "dtypes_after": dtype_summary(cleaned),
        "class_balance_before": class_balance(df),
        "class_balance_after": class_balance(cleaned),
    }
    # Cleaning must not touch the rows or the target - these are invariants, not statistics.
    report["invariants_hold"] = (
        report["rows_before"] == report["rows_after"]
        and abs(target_before - target_after) < 1e-12
    )
    return cleaned, report


def clean(variant: str = "full") -> tuple[pd.DataFrame, dict]:
    """
    Load a variant through src/data_loader.py and apply both T12 decisions.

    Deliberately goes through the loader rather than reading a CSV here (decision D13), so there
    is still exactly one place in the project that knows the separator and the file paths.
    """
    return clean_frame(load.load_raw(variant, verbose=False), variant)


# ---------------------------------------------------------------------------
# T14 - the conversion check
# ---------------------------------------------------------------------------
def validate(raw: pd.DataFrame, cleaned: pd.DataFrame, report: dict) -> list[dict]:
    """
    Check the converted frame against every property later tasks are entitled to assume.

    Returned as data (name / passed / detail) rather than only printed, so the CLI can exit
    non-zero on a failure instead of relying on someone reading the output carefully.
    """
    sentinel_report = report["sentinel_report"]
    sentinel = sentinel_report["sentinel"]
    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    def compare(name: str, evaluate) -> None:
        """
        Run a comparison and record the result - including when the comparison itself cannot run.

        A malformed frame makes an elementwise pandas comparison RAISE ("Can only compare
        identically-labeled Series objects") rather than return False, which would turn the guard
        rail into a traceback exactly when a diagnosis is most useful. A check that cannot execute
        is a FAILED check, not a crash.
        """
        try:
            passed, detail = evaluate()
        except Exception as error:  # noqa: BLE001 - any failure here is a failed check
            passed, detail = False, f"check could not run: {type(error).__name__}: {error}"
        add(name, passed, detail)

    # --- nothing lost, nothing silently altered ---------------------------------------------
    # Comparisons go through .to_numpy() so they compare by POSITION. Comparing indexed pandas
    # objects would either raise on a misaligned frame or silently align it, and the whole point
    # of this function is to catch a frame that is not what it claims to be.
    compare("row count preserved", lambda: (
        len(cleaned) == len(raw), f"{len(raw):,} -> {len(cleaned):,}",
    ))
    compare("every original column bit-identical", lambda: (
        not (changed := [column for column in raw.columns
                         if column in cleaned.columns and not raw[column].equals(cleaned[column])]),
        f"changed: {changed}",
    ))
    compare("target values untouched", lambda: (
        raw[config.TARGET].to_numpy().tolist() == cleaned[config.TARGET].to_numpy().tolist(), "",
    ))
    balance_before, balance_after = report["class_balance_before"], report["class_balance_after"]
    compare("class balance unchanged", lambda: (
        abs(balance_before["positive_rate"] - balance_after["positive_rate"]) < 1e-12,
        f"{balance_before['positive_rate']:.4%} -> {balance_after['positive_rate']:.4%}",
    ))

    # --- exactly the intended columns were added --------------------------------------------
    compare("only the two derived columns added", lambda: (
        [column for column in cleaned.columns if column not in raw.columns] == list(DERIVED_COLUMNS),
        f"added: {[c for c in cleaned.columns if c not in raw.columns]}",
    ))
    add("raw 'pdays' retained for traceability", RAW_PDAYS in cleaned.columns)

    # --- the 'unknown' policy held ----------------------------------------------------------
    compare("no NaN left in categorical columns", lambda: (
        int(cleaned[categorical_columns(cleaned)].isna().sum().sum()) == 0, "",
    ))
    compare("'unknown' counts preserved", lambda: (
        not (drifted := [column for column in categorical_columns(raw)
                         if int((raw[column] == UNKNOWN).sum())
                         != int((cleaned[column] == UNKNOWN).sum())]),
        f"drifted: {drifted}",
    ))

    # --- the sentinel handling held ---------------------------------------------------------
    compare("'previously_contacted' is boolean", lambda: (
        cleaned[CONTACT_FLAG].dtype == bool, str(cleaned[CONTACT_FLAG].dtype),
    ))
    compare("flag equals the independent witness 'previous > 0'", lambda: (
        cleaned[CONTACT_FLAG].to_numpy().tolist() == (raw["previous"] > 0).to_numpy().tolist(), "",
    ))
    compare("recency is NaN exactly where the sentinel was", lambda: (
        bool((cleaned[CONTACT_RECENCY].isna().to_numpy()
              == (raw[RAW_PDAYS] == sentinel).to_numpy()).all()), "",
    ))
    def sentinel_survives() -> tuple[bool, str]:
        present = sentinel in set(cleaned[CONTACT_RECENCY].dropna().unique())
        return not present, f"sentinel {sentinel} present" if present else ""

    compare("no sentinel value survives in the recency column", sentinel_survives)
    compare("recency equals raw 'pdays' wherever a real value exists", lambda: (
        bool((cleaned[CONTACT_RECENCY].to_numpy()[(raw[RAW_PDAYS] != sentinel).to_numpy()]
              == raw[RAW_PDAYS].to_numpy()[(raw[RAW_PDAYS] != sentinel).to_numpy()]).all()), "",
    ))
    compare("reported sentinel count matches the data", lambda: (
        sentinel_report["sentinel_rows"] == int((raw[RAW_PDAYS] == sentinel).sum()), "",
    ))
    compare("expected number of columns after conversion", lambda: (
        cleaned.shape[1] == raw.shape[1] + len(DERIVED_COLUMNS),
        f"{raw.shape[1]} -> {cleaned.shape[1]}",
    ))
    return checks


def process(variant: str = "full", *, check: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Convert one variant from raw to processed, and check the result.

    This is the single entry point for Phase 3: T15 calls it to get the frame it splits and writes.
    """
    raw = load.load_raw(variant, verbose=False)
    cleaned, report = clean_frame(raw, variant)
    if check:
        checks = validate(raw, cleaned, report)
        report["checks"] = checks
        report["checks_passed"] = sum(1 for item in checks if item["passed"])
        report["checks_total"] = len(checks)
        report["checks_ok"] = report["checks_passed"] == report["checks_total"]
    return cleaned, report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_conversion_report(report: dict) -> None:
    """Print the before/after conversion summary: shapes, dtypes, class balance, checks."""
    before, after = report["class_balance_before"], report["class_balance_after"]
    print("=" * 100)
    print(f"VARIANT {report['variant']} - raw to processed")
    print("=" * 100)
    print(f"shape   : {report['rows_before']:,} x {report['columns_before']}  ->  "
          f"{report['rows_after']:,} x {report['columns_after']}")
    print(f"rows    : lost {report['rows_before'] - report['rows_after']} "
          f"(the policy deletes none)")
    print(f"target  : {config.TARGET} == {config.POSITIVE_LABEL!r} "
          f"{before['no']:,} no / {before['yes']:,} yes -> "
          f"{after['no']:,} no / {after['yes']:,} yes  "
          f"({before['positive_rate']:.4%} -> {after['positive_rate']:.4%})")
    print()
    print("dtypes before:")
    for dtype, columns in sorted(report["dtypes_before"].items()):
        print(f"  {dtype:<9} {len(columns):>2} columns  {columns}")
    print("dtypes after:")
    for dtype, columns in sorted(report["dtypes_after"].items()):
        print(f"  {dtype:<9} {len(columns):>2} columns  {'' if dtype in report['dtypes_before'] else '<- new dtype'} {columns}")
    print()
    if "checks" in report:
        print(f"Phase 3 checks: {report['checks_passed']}/{report['checks_total']} passed")
        for item in report["checks"]:
            marker = "PASS" if item["passed"] else "FAIL"
            detail = f"  ({item['detail']})" if item["detail"] else ""
            print(f"  [{marker}] {item['check']}{detail}")
        print()


def print_report(report: dict) -> None:
    """Print what the two decisions changed, with the evidence a reviewer needs."""
    sentinel_report = report["sentinel_report"]
    print("=" * 100)
    print(f"VARIANT {report['variant']}")
    print("=" * 100)
    print(f"rows      : {report['rows_before']:,} -> {report['rows_after']:,} "
          f"(deleted: {report['rows_before'] - report['rows_after']})")
    print(f"columns   : {report['columns_before']} -> {report['columns_after']} "
          f"(added: {', '.join(report['new_columns'])})")
    print(f"target    : {config.TARGET} == {config.POSITIVE_LABEL!r} is "
          f"{report['target_positive_rate_before']:.4%} before and "
          f"{report['target_positive_rate_after']:.4%} after")
    print(f"invariants: {'OK - no row lost, target untouched' if report['invariants_hold'] else 'FAILED'}")
    print()
    print(f"POLICY (unknown): {report['policy']}")
    print(f"  rows listwise deletion would have removed: "
          f"{report['rows_lost_if_unknown_were_deleted']:,} "
          f"({report['rows_lost_if_unknown_were_deleted'] / report['rows_before']:.2%})")
    unknown_report = report["unknown_report"]
    with_unknown = unknown_report[unknown_report["unknown_cells"] > 0]
    print(f"  columns carrying the level: {len(with_unknown)}")
    if len(with_unknown):
        print(with_unknown.to_string(index=False, formatters={"share_of_rows": "{:.1%}".format}))
    print(f"  NaN cells rewritten to 'unknown': {int(unknown_report['nan_to_unknown'].sum())} "
          f"(no NaNs exist in these files; the rule exists so a refresh cannot add a second "
          f"missing marker)")
    print()
    print(f"SENTINEL (pdays = {sentinel_report['sentinel']} for this variant): "
          f"{sentinel_report['sentinel_rows']:,} rows")
    print(f"  previously_contacted (from 'previous' > 0, D21): "
          f"{sentinel_report['previously_contacted']:,} "
          f"({sentinel_report['previously_contacted_share']:.2%})")
    print(f"  days_since_contact available                   : "
          f"{sentinel_report['days_since_contact_available']:,}")
    print(f"  days_since_contact missing (sentinel removed)  : "
          f"{sentinel_report['days_since_contact_missing']:,}")
    print(f"  sentinel rows that DO have previous contacts   : "
          f"{sentinel_report['sentinel_rows_with_previous_contacts']:,}")
    for warning in sentinel_report["warnings"]:
        print(f"  WARNING: {warning}")
    print()


if __name__ == "__main__":
    # `python -m src.clean` - converts each variant from raw to processed, prints the before/after
    # comparison, then runs the Phase 3 checks. Exit status is 0 only if every check passes, so this
    # is a guard rail rather than a report: a regression cannot pass silently.
    print("DepositIQ Phase 3 - the cleaning module's own check")
    print()
    reports: dict[str, dict] = {}
    all_ok = True
    for variant_name in ("full", "additional"):
        _, variant_report = process(variant_name)
        reports[variant_name] = variant_report
        print_conversion_report(variant_report)
        print_report(variant_report)
        all_ok = all_ok and bool(variant_report["checks_ok"])

    print("=" * 100)
    print(f"variants converted : {', '.join(reports)}")
    print("Phase 3 checks     : " + ", ".join(
        f"{name} {report['checks_passed']}/{report['checks_total']}"
        for name, report in reports.items()
    ))
    print(f"result             : "
          f"{'PASS - raw to processed converts and checks cleanly' if all_ok else 'FAIL'}")
    print("Next               : T15 splits the processed frame and writes data/processed/train.csv")
    print("                     and test.csv; this module writes no data files of its own.")
    sys.exit(0 if all_ok else 1)
