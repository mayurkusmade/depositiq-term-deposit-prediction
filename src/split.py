"""
The train/test split for DepositIQ - the one place the evaluation halves are cut.

Why this file exists
--------------------
T15's job: cut the converted frame into a training half and a held-out test half, once, with one
seed, and write both to CSV so every later task trains and evaluates on exactly the same rows.

Three properties below are deliberate, and each one prevents a specific silent failure. The second
one is not a stylistic preference: it was measured, and the measurement is in the module.

1. The split is taken on the CONVERTED frame (`src/clean.py`), never on the raw CSV.
   D22's `unknown` level, D23's `previously_contacted` / `days_since_contact` columns and the retained
   raw `pdays` all have to exist *before* the halves are cut. Splitting first and cleaning each half
   separately is how the two halves drift apart - and then the test set is measuring a different
   pipeline from the one that was trained.

2. Stratified on the target, and shuffled. The positive class is 11.70% of the classic set and 11.27%
   of the additional set, and the ROWS ARE NOT IN A RANDOM ORDER: the file runs from `may 5` to
   `nov 17`, and the positive rate climbs from **2.99%** in the first row-order decile to **47.09%**
   in the last. Taking the first 80% as training - the obvious thing to do without looking - would
   put **31.58%** positives in the test half against 11.70% overall. Stratifying keeps the two halves
   comparable; shuffling is what stops the split from being a date cut in disguise.

3. Nothing is fitted, scaled, clipped or encoded here (D24). The `RobustScaler`, the
   `ColumnTransformer` and any resampling live inside the CV pipeline (T24/T29), fitted on the
   training half only. A scaler fitted before the split leaks test statistics into training.

Three limitations stated rather than hidden.

* **The evaluation is within one time period.** Because the campaign's timing is confounded with the
  outcome, a random split tests a model on clients the campaign happened to reach during the same
  months. It does not demonstrate that the model would hold up in a future campaign, and no result
  from this split should be presented as if it did. A time-based split would answer that question
  better and is deliberately not used here, because T15 asks for a stratified random split.
* **CSV stores text, not types.** pandas' parser maps exactly `True`/`False` back to `bool`, so
  `previously_contacted` does in practice round-trip as a boolean - but that is a property of the
  parser, not of the format, so `load_split` restores the declared dtypes explicitly and raises if the
  column holds anything that is not a boolean. `validate_round_trip` then proves the reloaded frame is
  row-for-row identical to the in-memory one, dtypes included.
* **Row identity is positional.** The files carry no index (`index=False`), so identity after a reload
  is the row number. Identical feature rows that already exist in the data - the additional variant
  has 12 - are **reported, not de-duplicated**: they are different clients, and dropping them is a data
  decision nobody asked for here. `split_frame` reports how many of those duplicate rows straddle the
  halves.

How to run
----------
From the repository root, inside the virtual environment:

    python -m src.split      # splits both variants, writes 4 files, exits non-zero on any FAIL
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import clean, config

# Where each variant's two halves are written: train path, test path.
SPLITS: dict[str, tuple[Path, Path]] = {
    "full": (config.TRAIN_CSV, config.TEST_CSV),
    "additional": (config.TRAIN_ADDITIONAL_CSV, config.TEST_ADDITIONAL_CSV),
}

# Columns whose dtype the loader declares explicitly instead of inheriting from whatever the CSV
# parser decides. pandas currently infers `bool` for exactly-True/False text, so this is a guard
# rather than a conversion on files this module wrote - and `load_split` raises if a column here holds
# anything that is not a boolean, which is what makes the guard worth keeping.
BOOL_COLUMNS: tuple[str, ...] = (clean.CONTACT_FLAG,)


def encode_target(df: pd.DataFrame) -> pd.Series:
    """
    Map the two text labels onto 0/1 - the encoding `stratify=` needs.

    Done here rather than written into the CSVs: the stored frame keeps the target exactly as the
    loader produces it ("yes"/"no"), so there is only ever one representation of `y` on disk and no
    chance of a later task training on the wrong column.
    """
    y = df[config.TARGET].map({config.POSITIVE_LABEL: 1, config.NEGATIVE_LABEL: 0})
    if y.isna().any():
        raise ValueError(
            f"Target column {config.TARGET!r} holds unexpected labels: "
            f"{sorted(df.loc[y.isna(), config.TARGET].unique())}. "
            f"Expected only {config.NEGATIVE_LABEL!r} and {config.POSITIVE_LABEL!r}."
        )
    y.name = config.TARGET
    return y.astype("int8")


def split_frame(
    df: pd.DataFrame,
    *,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Cut a converted frame into `(train, test, report)` with a stratified, seeded split.

    The split is taken on the row index, so both halves keep **every** column of the converted frame,
    including the target and the derived contact-history columns. Rows are returned in their original
    relative order (the index is sorted after shuffling) so the written files are reproducible and
    readable; the *set* of rows is what the seed decides, not the order.
    """
    y = encode_target(df)
    counts = y.value_counts()

    if len(counts) < 2:
        raise ValueError(
            f"Cannot split: the target has only one class ({dict(counts)}). "
            "An evaluation half needs both."
        )
    if int(counts.min()) < 2:
        raise ValueError(
            f"Cannot split: the rarest class has {int(counts.min())} row(s), so it cannot appear in "
            "both halves. "
        )

    train_index, test_index = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=y,          # keeps the 11.7% positive rate in both halves
        shuffle=True,
    )

    train = df.loc[sorted(train_index)]
    test = df.loc[sorted(test_index)]

    report = {
        "rows": len(df),
        "variant": None,   # filled in by write_split; kept here so this dict is self-describing
        "columns": df.shape[1],
        "test_size": test_size,
        "random_state": random_state,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_share": len(train) / len(df),
        "positive_rate_all": float((y == 1).mean()),
        "positive_rate_train": float((encode_target(train) == 1).mean()),
        "positive_rate_test": float((encode_target(test) == 1).mean()),
        "positives_train": int((encode_target(train) == 1).sum()),
        "positives_test": int((encode_target(test) == 1).sum()),
        "nan_cells_all": int(df.isna().sum().sum()),
        "nan_cells_train": int(train.isna().sum().sum()),
        "nan_cells_test": int(test.isna().sum().sum()),
        "duplicate_rows_in_source": int(df.duplicated(keep=False).sum()),
        "duplicate_rows_spanning_halves": _duplicate_rows_spanning(df, train, test),
    }
    return train, test, report


def _duplicate_rows_spanning(
    df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame
) -> int:
    """
    Count feature rows that exist in BOTH halves.

    Not an error: the source data contains genuine duplicate rows (12 in the additional variant), and
    a client whose details match another client's is still a separate client. It is reported because
    it is the one way a test row can carry information a training row already gave the model.
    """
    duplicated = df[df.duplicated(keep=False)]
    if duplicated.empty:
        return 0
    features = [column for column in df.columns if column != config.TARGET]
    train_index, test_index = set(train.index), set(test.index)
    rows = duplicated[features].to_numpy().tolist()
    train_keys = {tuple(row) for row, index in zip(rows, duplicated.index) if index in train_index}
    test_keys = {tuple(row) for row, index in zip(rows, duplicated.index) if index in test_index}
    return len(train_keys & test_keys)


def load_split(variant: str = "full", which: str = "train") -> pd.DataFrame:
    """
    Read one written half back and restore the dtypes CSV cannot carry.

    This is the counterpart to `write_split`: anything downstream that wants the processed rows goes
    through here rather than calling `read_csv` with its own arguments, so the two halves cannot be
    loaded with different settings.
    """
    if variant not in SPLITS:
        raise ValueError(f"Unknown variant {variant!r}. Valid options are: {', '.join(sorted(SPLITS))}.")
    if which not in ("train", "test"):
        raise ValueError(f"Unknown half {which!r}. Valid options are: 'train', 'test'.")

    train_path, test_path = SPLITS[variant]
    path = train_path if which == "train" else test_path
    if not path.exists():
        raise FileNotFoundError(
            f"{which} split for variant {variant!r} not found at:\n  {path}\n"
            "Run `python -m src.split` first."
        )

    df = pd.read_csv(path)

    for column in BOOL_COLUMNS:
        if column in df.columns:
            # Restore the declared dtype rather than trusting the parser (`True`/`False` text today,
            # but that is the parser's choice). A value that is neither means the file was written by
            # something else, so it raises instead of guessing.
            restored = df[column].map({True: True, False: False, "True": True, "False": False})
            if restored.isna().any():
                raise ValueError(
                    f"Column {column!r} in {path.name} holds values that are not booleans: "
                    f"{sorted(df.loc[restored.isna(), column].unique())}"
                )
            df[column] = restored.astype(bool)

    return df


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def validate_split(
    df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, report: dict
) -> list[dict]:
    """
    Check the two halves against every property later tasks are entitled to assume.

    Same contract as `src/clean.py`'s checker (D25), and the `compare` helper is a deliberate local
    copy of the one there: a check that cannot execute is recorded as FAILED rather than raised, and
    comparisons are positional, so a malformed half is diagnosed instead of crashing the guard rail.
    """
    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    def compare(name: str, evaluate) -> None:
        try:
            passed, detail = evaluate()
        except Exception as error:  # noqa: BLE001 - any failure here is a failed check
            passed, detail = False, f"check could not run: {type(error).__name__}: {error}"
        add(name, passed, detail)

    tolerance = 0.01  # the +/-1% the task is graded on, as percentage points of the rate

    # Every figure below is MEASURED FROM THE FRAMES HANDED IN, never read out of `report`.
    # That distinction is the whole point of the checks: fault injection showed an earlier version
    # reading the report happily pass a test half with every positive stripped out, because the
    # report still described the honest split. A checker that validates a *description* rather than
    # the data cannot see the two disagree - and then nothing stands between a wrong split and a
    # green run. `report` is used for the parameters it was configured with (test size, seed) and
    # for nothing else; the final check below is what holds the two representations together.
    def rate(half: pd.DataFrame) -> float:
        """Positive rate measured from the frame itself."""
        return float((encode_target(half) == 1).mean())

    def nan_cells(half: pd.DataFrame) -> int:
        return int(half.isna().sum().sum())

    def share(half: pd.DataFrame) -> float:
        return len(half) / len(df)

    # --- every row lands in exactly one half ----------------------------------------------------
    compare("no row lost or duplicated", lambda: (
        len(train) + len(test) == len(df), f"{len(train):,} + {len(test):,} vs {len(df):,}",
    ))
    compare("halves are disjoint", lambda: (
        not (shared := set(train.index) & set(test.index)), f"shared rows: {len(shared)}",
    ))
    compare("halves cover every row", lambda: (
        set(train.index) | set(test.index) == set(df.index),
        f"covered {len(set(train.index) | set(test.index)):,} of {len(df):,}",
    ))

    # --- the split proportions are the ones asked for --------------------------------------------
    expected_train_share = 1.0 - report["test_size"]
    compare(f"train share is {expected_train_share:.0%} (+/-1pp)", lambda: (
        abs(share(train) - expected_train_share) <= tolerance,
        f"{share(train):.4%} vs {expected_train_share:.4%}",
    ))
    compare("positive rate kept in train (+/-1pp)", lambda: (
        abs(rate(train) - rate(df)) <= tolerance, f"{rate(train):.4%} vs {rate(df):.4%}",
    ))
    compare("positive rate kept in test (+/-1pp)", lambda: (
        abs(rate(test) - rate(df)) <= tolerance, f"{rate(test):.4%} vs {rate(df):.4%}",
    ))
    compare("both halves contain both classes", lambda: (
        set(train[config.TARGET].unique()) == set(df[config.TARGET].unique())
        and set(test[config.TARGET].unique()) == set(df[config.TARGET].unique()),
        f"train {sorted(train[config.TARGET].unique())} / test {sorted(test[config.TARGET].unique())}",
    ))

    # --- the converted frame survived the cut intact ----------------------------------------------
    compare("both halves keep the exact column set", lambda: (
        list(train.columns) == list(df.columns) == list(test.columns),
        f"{train.shape[1]} / {df.shape[1]} / {test.shape[1]} columns",
    ))
    compare("no dtype was altered by splitting", lambda: (
        train.dtypes.equals(df.dtypes) and test.dtypes.equals(df.dtypes),
        f"train differs: {[c for c in df.columns if train[c].dtype != df[c].dtype]}, "
        f"test differs: {[c for c in df.columns if test[c].dtype != df[c].dtype]}",
    ))
    compare("no missing value invented or lost", lambda: (
        nan_cells(train) + nan_cells(test) == nan_cells(df),
        f"{nan_cells(train)} + {nan_cells(test)} vs {nan_cells(df)}",
    ))
    compare("'unknown' counts add back up to the whole frame", lambda: (
        not (drifted := [
            column for column in clean.categorical_columns(df)
            if int((train[column] == clean.UNKNOWN).sum()) + int((test[column] == clean.UNKNOWN).sum())
            != int((df[column] == clean.UNKNOWN).sum())
        ]),
        f"drifted: {drifted}",
    ))
    compare("derived contact-history columns present in both halves", lambda: (
        all(
            column in train.columns and column in test.columns
            for column in clean.DERIVED_COLUMNS
        ),
        f"{list(clean.DERIVED_COLUMNS)}",
    ))
    compare("sentinel still absent from the recency column in both halves", lambda: (
        all(
            report["sentinel"] not in set(half[clean.CONTACT_RECENCY].dropna().unique())
            for half in (train, test)
        ),
        f"sentinel {report['sentinel']}",
    ))

    # --- the reported figures describe the frames they came from ----------------------------------
    compare("the report's figures match the frames", lambda: (
        not (mismatched := [
            f"{key}={report[key]!r} but measured {measured!r}"
            for key, measured in (
                ("train_rows", len(train)),
                ("test_rows", len(test)),
                ("train_share", share(train)),
                ("positive_rate_all", rate(df)),
                ("positive_rate_train", rate(train)),
                ("positive_rate_test", rate(test)),
                ("nan_cells_all", nan_cells(df)),
                ("nan_cells_train", nan_cells(train)),
                ("nan_cells_test", nan_cells(test)),
            )
            if not _close(report[key], measured)
        ]),
        f"mismatched: {mismatched}",
    ))

    # --- the split is reproducible ---------------------------------------------------------------
    compare("the same seed reproduces the split exactly", lambda: (
        _same_split(df, train, test, report),
        "",
    ))
    return checks


def _close(stored, measured) -> bool:
    """Compare a stored figure against a freshly measured one, tolerating float representation."""
    return abs(float(stored) - float(measured)) < 1e-12


def _same_split(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, report: dict) -> bool:
    """Re-run the split with the recorded seed and require it to land on the same rows."""
    again_train, again_test, _ = split_frame(
        df, test_size=report["test_size"], random_state=report["random_state"]
    )
    return (
        again_train.index.tolist() == train.index.tolist()
        and again_test.index.tolist() == test.index.tolist()
    )


def validate_round_trip(written: pd.DataFrame, reloaded: pd.DataFrame, path: Path) -> list[dict]:
    """
    Check that a written CSV reads back as the frame that was written.

    The comparison is against `written.reset_index(drop=True)` on purpose. The files are written with
    `index=False`, so what a CSV preserves is the **row order and the values**; the index labels of an
    in-memory frame are not part of the artifact and come back as a fresh `RangeIndex`. Comparing the
    labelled frame directly would compare index labels as well and report every column as different,
    which is a false alarm that is far too easy to silence by deleting the check.

    `equals` treats NaN in the same position as equal and compares dtypes strictly, so a reordered
    file, a column that lost precision and a flag that returned as text all still fail.
    """
    expected = written.reset_index(drop=True)
    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    def compare(name: str, evaluate) -> None:
        try:
            passed, detail = evaluate()
        except Exception as error:  # noqa: BLE001
            passed, detail = False, f"check could not run: {type(error).__name__}: {error}"
        add(name, passed, detail)

    def differ() -> list[str]:
        """Which columns are not identical, so a failure names the culprit instead of the frame."""
        return [
            column for column in expected.columns
            if not expected[column].equals(reloaded[column])
            or expected[column].dtype != reloaded[column].dtype
        ]

    compare(f"{path.name} reloads with the same shape", lambda: (
        expected.shape == reloaded.shape, f"{expected.shape} vs {reloaded.shape}",
    ))
    compare(f"{path.name} reloads with the same columns", lambda: (
        list(expected.columns) == list(reloaded.columns), "",
    ))
    compare(f"{path.name} reloads row-for-row identical", lambda: (
        not (bad := differ()), f"differing columns: {bad}",
    ))
    return checks


# ---------------------------------------------------------------------------
# The task's entry point
# ---------------------------------------------------------------------------
def write_split(variant: str = "full", *, check: bool = True) -> dict:
    """
    Convert a variant, split it, check it, write both halves, and check the files read back.

    The conversion runs through `clean.process`, so the split inherits T14's 15 conversion checks -
    the halves are never cut from a frame that failed its own guard rail.
    """
    if variant not in SPLITS:
        raise ValueError(f"Unknown variant {variant!r}. Valid options are: {', '.join(sorted(SPLITS))}.")

    config.ensure_dirs()
    frame, clean_report = clean.process(variant)
    train, test, report = split_frame(frame)

    # The sentinel convention is per-variant (D19), so it is read off the conversion, not hard-coded.
    report["variant"] = variant
    report["sentinel"] = clean_report["sentinel_report"]["sentinel"]
    report["conversion_checks"] = (
        f"{clean_report['checks_passed']}/{clean_report['checks_total']}"
    )

    train_path, test_path = SPLITS[variant]
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    report["train_path"] = train_path
    report["test_path"] = test_path

    checks: list[dict] = []
    if check:
        checks = validate_split(frame, train, test, report)
        for which, written, path in (
            ("train", train, train_path),
            ("test", test, test_path),
        ):
            checks.extend(validate_round_trip(written, load_split(variant, which), path))

    report["checks"] = checks
    report["checks_passed"] = sum(1 for item in checks if item["passed"])
    report["checks_total"] = len(checks)
    report["checks_ok"] = bool(checks) and report["checks_passed"] == report["checks_total"]
    report["conversion_ok"] = bool(clean_report["checks_ok"])
    return report


def print_split_report(report: dict) -> None:
    """Print the shapes, the proportions and the checks - the evidence the task asks for."""
    label = report.get("variant_label") or report["variant"]
    print("=" * 100)
    print(f"{label} - stratified {1 - report['test_size']:.0%}/"
          f"{report['test_size']:.0%} split (random_state={report['random_state']})")
    print("=" * 100)
    print(f"source     : {report['rows']:,} rows x {report['columns']} columns "
          f"(conversion checks {report['conversion_checks']})")
    print(f"train      : {report['train_rows']:,} rows  ({report['train_share']:.4%} of the frame)")
    print(f"test       : {report['test_rows']:,} rows  ({1 - report['train_share']:.4%})")
    print(f"target     : {config.TARGET} == {config.POSITIVE_LABEL!r} "
          f"{report['positive_rate_all']:.4%} overall -> "
          f"{report['positive_rate_train']:.4%} train / {report['positive_rate_test']:.4%} test")
    print(f"positives  : {report['positives_train']:,} train / {report['positives_test']:,} test")
    print(f"missing    : days_since_contact NaN = {report['nan_cells_all']:,} "
          f"-> {report['nan_cells_train']:,} train / {report['nan_cells_test']:,} test "
          f"(source rows at the sentinel {report['sentinel']})")
    print(f"written    : {report['train_path'].name}, {report['test_path'].name}")
    if report["duplicate_rows_in_source"]:
        print(f"duplicates : {report['duplicate_rows_in_source']} identical feature rows in the source, "
              f"{report['duplicate_rows_spanning_halves']} of them land in both halves "
              f"(reported, not de-duplicated)")
    print()
    print(f"Phase 3 checks: {report['checks_passed']}/{report['checks_total']} passed")
    for item in report["checks"]:
        marker = "PASS" if item["passed"] else "FAIL"
        detail = f"  ({item['detail']})" if item["detail"] else ""
        print(f"  [{marker}] {item['check']}{detail}")
    print()


if __name__ == "__main__":
    # `python -m src.split` - splits both variants and writes four files. Exit status is 0 only if
    # every check on both variants passes, so a regression cannot pass silently.
    print("DepositIQ Phase 3 - the train/test split")
    print()
    ok = True
    for variant_name, label in (
        ("full", "Classic variant (primary, D20)"),
        ("additional", "Additional variant (comparison pipeline, D19)"),
    ):
        split_report = write_split(variant_name)
        split_report["variant_label"] = label
        print_split_report(split_report)
        ok = ok and split_report["checks_ok"] and split_report["conversion_ok"]

    print("=" * 100)
    print(f"result : {'PASS - both variants split, checked and written' if ok else 'FAIL'}")
    print("Next   : T16 starts Phase 4 (EDA). Nothing here is fitted, so no model artifact was written;")
    print("         scaling and encoding belong in the CV pipeline (T24/T29).")
    sys.exit(0 if ok else 1)
