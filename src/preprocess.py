"""
DepositIQ - the preprocessing `ColumnTransformer` (T24).

What this module does
---------------------
Takes the **17 columns** `src/features.py` declares and turns them into one numeric matrix: three
columns scaled, one count passed through, twelve columns one-hot encoded, one boolean passed through.
It fits on the training half and transforms the held-out half, and it reports the shapes on both sides.

The column list is **imported, never restated**. `features.FEATURES` is the single definition, so a
column cannot be spelled one way in the feature module and another way here, and the four groups below
are declared as data so `validate()` can inspect them instead of trusting a comment.

Four groups, because the 17 columns are not homogeneous
-------------------------------------------------------
=============================  ====  ===============================================================
group                          n     treatment and the measured reason
=============================  ====  ===============================================================
`NUMERIC_SCALED`               3     `age`, `day`, `balance_arsinh` -> `StandardScaler`. These are the
                                     columns with **no tails left to protect against**: `day` runs
                                     1-31, `age` 18-95 (skew **0.694**), and `balance_arsinh` is
                                     already an arcsinh scale (skew **-1.679**, T23). T17's
                                     median/IQR rule exists for *untransformed* skew, so the
                                     mean/std pair is the right estimator here
`NUMERIC_PASSTHROUGH`          1     `previous_capped` -> untouched (cast to float64, no value
                                     changed). Median **0**, Q1 **0**, Q3 **0**: the IQR is **0**, so
                                     an IQR-based scale is undefined for it (sklearn silently
                                     substitutes `1.0`), and the same is true of raw `previous` at
                                     **81.80% zeros** - so this is a property of the column, not of
                                     the cap. A z-score of a three-value count is a relabelling of
                                     {0, 1, 2}, not a units fix. Nothing to rescale
`CATEGORICAL`                 12     one-hot, `handle_unknown="ignore"`, **all** levels kept
`BOOLEAN`                      1     `previously_contacted` -> passed through as a single 0/1
                                     column. One-hot encoding a boolean would produce two columns
                                     that sum to 1, i.e. the same column twice
=============================  ====  ===============================================================

Result: **3 + 1 + 60 + 1 = 65 columns** from 17, where 60 is the measured one-hot width
(`job` 12 + `marital` 3 + `education` 4 + `default` 2 + `housing` 2 + `loan` 2 + `contact` 3 +
`month` 12 + `poutcome` 4 + `campaign_bucket` 6 + `balance_band` 5 + `recency_band` 5).

Three decisions that are easy to get wrong, and what is done instead
--------------------------------------------------------------------
* **`remainder="drop"`.** The frame handed in still holds `duration`, `pdays` and `y` - keeping them is
  what makes the leakage audit runnable (D14). `remainder="drop"` means the transformer cannot
  nevertheless *use* them, and `validate()` proves it by transforming a frame containing them and
  checking the matrix is bit-for-bit the one built from `FEATURES` alone.
* **`handle_unknown="ignore"`.** The encoder is fitted on one half and applied to another, so a level it
  has never seen must not be an error. Measured: **no test-only level exists** on this split (all 12
  categorical columns have identical vocabularies in both halves), so the net never fires here - and
  that is exactly why it is stated rather than relied on. `validate()` injects an unseen level and
  proves the row encodes to zeros instead of raising, and that `handle_unknown="error"` would raise.
* **`drop=None`.** Every level gets a column. Dropping the first level makes a coefficient interpretable
  only for the linear baseline and asymmetric across columns with different level counts; with L2
  regularisation the collinearity is harmless, and one encoding is shared by T26-T28 and by SHAP in
  T33.

Fitted inside the fold, never before it
---------------------------------------
D24 and T22 §4.2 both require it: nothing is scaled or encoded before the split. `build_preprocessor()`
therefore returns a **fresh, unfitted** transformer every call - the same factory, instantiated inside
each fold - so a scaler's statistics can only ever be those of its own fold's training rows.
`validate()` checks the fitted mean against the training mean to **1e-12** and reports how far the
pooled train+test mean would have been, so "fitted on train only" is a measurement rather than a claim.

How to run
----------
From the repository root, inside the virtual environment:

    python -m src.preprocess     # fits on train, transforms both halves, runs the checks
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config, features
from src.leakage import feature_columns
from src.split import encode_target, load_split

# ---------------------------------------------------------------------------
# The four groups, taken from the feature module rather than re-typed
# ---------------------------------------------------------------------------
NUMERIC_SCALED: tuple[str, ...] = ("age", "day", features.BALANCE_ARSINH)
NUMERIC_PASSTHROUGH: tuple[str, ...] = (features.PREVIOUS_CAPPED,)
CATEGORICAL: tuple[str, ...] = features.CATEGORICAL_FEATURES
BOOLEAN: tuple[str, ...] = features.BOOLEAN_FEATURES

# Every model column, in the order the matrix is built. Checked against `features.FEATURES` at import
# time by `validate()`, so a group that forgets a column fails loudly instead of dropping it silently.
MODEL_COLUMNS: tuple[str, ...] = NUMERIC_SCALED + NUMERIC_PASSTHROUGH + CATEGORICAL + BOOLEAN

# Names the ColumnTransformer uses internally; also what `named_steps`/`named_transformers_` is keyed by.
STEP_SCALED: str = "scaled"
STEP_COUNT: str = "count"
STEP_CATEGORICAL: str = "categorical"
STEP_BOOLEAN: str = "boolean"

# Columns the transformer must never see. Kept here as well as in `features.FORBIDDEN` because this is
# the module that would be at fault if one of them arrived in the matrix.
NEVER: tuple[str, ...] = ("duration", "pdays", config.TARGET)


def as_float(frame):
    """
    Cast a boolean passthrough to float64, changing no value.

    `"passthrough"` on a bool column leaves the assembled frame **heterogeneous** - the other blocks are
    float64, so `X.to_numpy()` comes back as `object` and `np.isfinite` refuses it. A model given that
    would either raise or silently cast somewhere less visible, so the cast is done here and named.
    Module-level because T31 pickles the pipeline and a lambda cannot be pickled.
    """
    return frame.astype(np.float64)


def feature_name(feature: str, category) -> str:
    """
    Name a one-hot column `job=admin.` instead of sklearn's default `job_admin.`.

    Module-level rather than a lambda: `joblib.dump` of the final pipeline (T31) cannot pickle a lambda,
    and the names are read by T25's importance table and T33's SHAP plots, so they are worth being
    legible.
    """
    return f"{feature}={category}"


def build_preprocessor(
    columns: list[str] | tuple[str, ...] | None = None,
    *,
    extra_numeric: tuple[str, ...] = (),
):
    """
    Return a **fresh, unfitted** `ColumnTransformer` over `columns`.

    This is the factory T22 §4.2 fixed as a precondition of T36: called once per fold, it returns an
    independent object with no state, so nothing can be carried between folds or between the two arms
    of the leakage comparison.

    `columns` defaults to `features.FEATURES`. `extra_numeric` exists for one caller: T36's diagnostic
    arm passes `("duration",)`, which is numeric seconds and belongs in the scaled group, so the two
    arms differ by exactly one column rather than by one code path. Any column that is not in one of the
    four declared groups raises, which is what stops a typo becoming a silently dropped feature.
    """
    if columns is None:
        columns = features.FEATURES
    columns = [column for column in columns]

    # Checked before the group lookup, so a mistyped `duration` is diagnosed as the leak it is rather
    # than as an unknown column. The guard has one documented escape hatch: `extra_numeric` is how T36's
    # diagnostic arm declares the leak on purpose. The target never gets one.
    forbidden = [
        column for column in columns
        if column == config.TARGET or (column in NEVER and column not in extra_numeric)
    ]
    if forbidden:
        raise ValueError(
            f"build_preprocessor() was handed a column that must never enter: {forbidden}. "
            f"The only way to include the leak is extra_numeric=(\"duration\",), which is T36's arm B."
        )

    stray = [column for column in columns if column not in MODEL_COLUMNS and column not in extra_numeric]
    if stray:
        raise KeyError(
            f"build_preprocessor() does not know how to treat {stray}. Declare it in "
            f"src.features.FEATURES (with its group) or pass it as extra_numeric."
        )

    scaled = [column for column in NUMERIC_SCALED if column in columns]
    scaled += [column for column in extra_numeric if column in columns]
    count = [column for column in NUMERIC_PASSTHROUGH if column in columns]
    categorical = [column for column in CATEGORICAL if column in columns]
    boolean = [column for column in BOOLEAN if column in columns]
    if len(scaled) + len(count) + len(categorical) + len(boolean) != len(columns):
        raise ValueError("a column was assigned to no group; the partition must cover every column")

    transformers = [
        (STEP_SCALED, StandardScaler(), scaled),
        # No scaler: `previous_capped` is a bounded count whose IQR is 0 (see the module docstring). The
        # cast is dtype only - the values that come out are the values that went in.
        (STEP_COUNT, FunctionTransformer(as_float, feature_names_out="one-to-one", validate=False), count),
        (
            STEP_CATEGORICAL,
            OneHotEncoder(
                handle_unknown="ignore",
                drop=None,
                # Dense: 65 columns x 36,168 rows is 18.8 MB of float64, and T33's SHAP values and T25's
                # coefficient reads consume dense arrays. Sparse would only pay for itself far wider.
                sparse_output=False,
                dtype=np.float64,
                feature_name_combiner=feature_name,
            ),
            categorical,
        ),
        # A boolean one-hot would be two columns summing to 1; a bool passed through raw would make the
        # whole matrix `object`. Cast, do not encode and do not scale.
        (STEP_BOOLEAN, FunctionTransformer(as_float, feature_names_out="one-to-one", validate=False), boolean),
    ]

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # `duration`, `pdays`, `y` and anything else stay out (D05, D36)
        verbose_feature_names_out=False,  # names come out as `job=admin.`, not `categorical__job=admin.`
        n_jobs=1,  # one process: T36 compares two arms in the same run and threads buy nothing here
    ).set_output(transform="pandas")


# ---------------------------------------------------------------------------
# Fit on train, transform both halves
# ---------------------------------------------------------------------------
def fit_train_transform_both(
    variant: str = "full",
    *,
    columns: list[str] | tuple[str, ...] | None = None,
    extra_numeric: tuple[str, ...] = (),
) -> dict:
    """
    Fit the preprocessor on the training half and apply it to both halves.

    The test half is transformed, never fitted: `fit_transform` is called on train only, and `transform`
    on test, which is the exact sequence a fold inside `cross_val_score` performs.
    """
    if columns is None:
        columns = features.FEATURES

    train_frame = features.load_engineered("train", variant)
    test_frame = features.load_engineered("test", variant)

    transformer = build_preprocessor(columns, extra_numeric=extra_numeric)
    X_train = transformer.fit_transform(train_frame)
    X_test = transformer.transform(test_frame)

    return {
        "transformer": transformer,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": encode_target(train_frame),
        "y_test": encode_target(test_frame),
        "train_frame": train_frame,
        "test_frame": test_frame,
        "columns": list(columns),
        "names": list(transformer.get_feature_names_out()),
    }


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def _block(names: list[str], prefix: str) -> list[int]:
    """Indices of the one-hot columns belonging to one categorical feature."""
    return [index for index, name in enumerate(names) if name.startswith(prefix + "=")]


def validate(fitted: dict) -> list[dict]:
    """
    Eight checks, each of which fails for a different real mistake.

    They are run on the *fitted* objects and the *returned matrices*, not on the report, because a
    description that agrees with itself proves nothing (D27).
    """
    from sklearn.base import clone

    transformer = fitted["transformer"]
    X_train, X_test = fitted["X_train"], fitted["X_test"]
    train_frame, test_frame = fitted["train_frame"], fitted["test_frame"]
    names = fitted["names"]
    results: list[dict] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        results.append({"check": name, "passed": bool(passed), "detail": detail})

    # 1. The partition covers exactly the feature module's list - every column once. Compared as a
    #    multiset rather than a sequence because grouping by treatment reorders two numeric columns:
    #    `features.FEATURES` lists age, day, previous_capped, balance_arsinh, and the matrix orders the
    #    scaled block before the count block. Nothing is added and nothing is dropped.
    add("groups partition features.FEATURES exactly",
        sorted(MODEL_COLUMNS) == sorted(features.FEATURES) and len(set(MODEL_COLUMNS)) == len(MODEL_COLUMNS),
        f"{len(MODEL_COLUMNS)} columns in 4 groups, no duplicates, none missing")

    # 2. The factory returns an unfitted object every time: nothing is shared between folds.
    fresh_a, fresh_b = build_preprocessor(), build_preprocessor()
    add("factory returns a fresh unfitted transformer each call",
        fresh_a is not fresh_b and not hasattr(fresh_a, "transformers_"),
        "two independent instances, neither fitted")

    try:
        build_preprocessor().transform(train_frame)
    except NotFittedError:
        add("transform before fit raises NotFittedError", True, "refuses to guess statistics")
    else:
        add("transform before fit raises NotFittedError", False, "an unfitted transformer transformed data")

    # 3. Shapes on both sides, and the widths agree.
    same_width = X_train.shape[1] == X_test.shape[1]
    add("train and test have the same width", same_width,
        f"train {X_train.shape}, test {X_test.shape}")

    # 4. The width equals what the fitted encoder actually learnt, not a literal written by hand.
    encoder = transformer.named_transformers_[STEP_CATEGORICAL]
    onehot_width = int(sum(len(levels) for levels in encoder.categories_))
    expected_width = len(NUMERIC_SCALED) + len(NUMERIC_PASSTHROUGH) + onehot_width + len(BOOLEAN)
    add("output width matches the fitted vocabulary",
        X_train.shape[1] == expected_width,
        f"{X_train.shape[1]} = {len(NUMERIC_SCALED)} scaled + {len(NUMERIC_PASSTHROUGH)} count + "
        f"{onehot_width} one-hot + {len(BOOLEAN)} boolean")

    # 5. Every row's one-hot block has exactly one 1 per categorical column: no row lost, no row doubled.
    onehot_sum = X_test[np.array([n for n in names if any(n.startswith(c + "=") for c in CATEGORICAL)])].to_numpy().sum(axis=1)
    add("every test row has one active level per categorical column",
        bool(np.all(onehot_sum == len(CATEGORICAL))),
        f"row sums are all {len(CATEGORICAL)}")

    # 6. `remainder="drop"`: a frame that still carries duration, pdays and y produces the same matrix.
    widened = test_frame.copy()
    widened[config.TARGET] = np.where(encode_target(test_frame) == 1, "yes", "no")
    widened["duration"] = widened["duration"] * 0 + 999  # a different value on purpose
    X_widened = transformer.transform(widened)
    add('remainder="drop" keeps duration, pdays and y out',
        np.array_equal(X_widened.to_numpy(), X_test.to_numpy()),
        "a frame carrying all three transforms to the identical matrix")

    # 7. `handle_unknown="ignore"`: an unseen level encodes to zeros instead of raising - and it would
    #    raise without the setting, so the check has teeth.
    unseen = test_frame.iloc[:1].copy()
    unseen["job"] = "astronaut"
    X_unseen = transformer.transform(unseen)
    job_block = X_unseen[np.array([n for n in names if n.startswith("job=")])].to_numpy()
    encoded_to_zeros = bool(np.all(job_block == 0.0)) and bool(X_unseen.notna().all().all())
    strict = clone(encoder).set_params(handle_unknown="error")
    strict.fit(train_frame[list(CATEGORICAL)])
    try:
        strict.transform(unseen[list(CATEGORICAL)])
    except ValueError:
        teeth = True
    else:
        teeth = False
    add('handle_unknown="ignore" encodes an unseen level to zeros',
        encoded_to_zeros and teeth,
        "zeros here, ValueError without the setting" if encoded_to_zeros and teeth
        else f"zeros={encoded_to_zeros}, strict-raising={teeth}")

    # 8. Nothing was fitted before the split: the scaler's statistics are the training half's, not the
    #    pooled frame's. The pooled gap is reported to show the difference is measurable, not epsilon.
    scaler = transformer.named_transformers_[STEP_SCALED]
    train_mean = train_frame[list(NUMERIC_SCALED)].to_numpy().mean(axis=0)
    pooled = pd.concat([train_frame[list(NUMERIC_SCALED)], test_frame[list(NUMERIC_SCALED)]], ignore_index=True)
    pooled_mean = pooled.to_numpy().mean(axis=0)
    train_gap = float(np.max(np.abs(scaler.mean_ - train_mean)))
    pooled_gap = float(np.max(np.abs(scaler.mean_ - pooled_mean)))
    add("scaler statistics come from the training half only",
        train_gap < 1e-12 and pooled_gap > train_gap,
        f"|fit - train| = {train_gap:.2e}, |fit - train+test| = {pooled_gap:.2e}")

    # 9. The timeout-free invariants: no NaN or inf anywhere, unique names, and no forbidden name.
    finite = bool(np.isfinite(X_train.to_numpy()).all() and np.isfinite(X_test.to_numpy()).all())
    add("every value is finite", finite, f"train {X_train.shape}, test {X_test.shape}")
    dtypes = set(X_train.dtypes.astype(str)) | set(X_test.dtypes.astype(str))
    add("the matrix is homogeneous float64", dtypes == {"float64"},
        f"dtypes present: {sorted(dtypes)}")
    flag = BOOLEAN[0]
    add("the boolean passes through unchanged, unscaled",
        np.array_equal(X_test[flag].to_numpy(), test_frame[flag].to_numpy().astype(np.float64)),
        f"{flag} == 0/1 values of the input flag, not a z-score")
    count_column = NUMERIC_PASSTHROUGH[0]
    add("the count is cast, not scaled",
        np.array_equal(X_test[count_column].to_numpy(), test_frame[count_column].to_numpy().astype(np.float64))
        and set(np.unique(X_test[count_column].to_numpy())) == {0.0, 1.0, 2.0},
        f"{count_column} still holds exactly {{0, 1, 2}}")
    add("column names are unique", len(set(names)) == len(names), f"{len(names)} names")
    prefixes = tuple(f"{column}=" for column in NEVER) + (config.TARGET,)
    leaked = [name for name in names if name in NEVER or name.startswith(prefixes)]
    add("no forbidden column reached the matrix", not leaked, f"found {leaked}" if leaked else "none")

    # 10. Deterministic: a second fit gives the identical matrix.
    again = build_preprocessor(fitted["columns"])
    add("refitting gives an identical matrix",
        np.array_equal(again.fit_transform(train_frame).to_numpy(), X_train.to_numpy()),
        "bit-for-bit identical")

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(variant: str = "full") -> dict:
    """Everything the module prints: the groups, the shapes, the vocabulary and the checks."""
    fitted = fit_train_transform_both(variant)
    transformer = fitted["transformer"]
    names = fitted["names"]
    train_frame, test_frame = fitted["train_frame"], fitted["test_frame"]
    scaler = transformer.named_transformers_[STEP_SCALED]
    encoder = transformer.named_transformers_[STEP_CATEGORICAL]

    levels = []
    for column, row, categories in zip(CATEGORICAL, encoder.categories_, encoder.categories_):
        test_levels = set(test_frame[column].astype(str).unique())
        levels.append({
            "column": column,
            "levels": len(categories),
            "vocabulary": [str(value) for value in row],
            "test_only": sorted(test_levels - {str(value) for value in row}),
        })

    # The measured reason `previous_capped` is passed through instead of scaled.
    from sklearn.preprocessing import RobustScaler

    capped = train_frame[features.PREVIOUS_CAPPED].to_numpy().reshape(-1, 1)
    robust = RobustScaler().fit(capped)
    capped_stats = {
        "median": float(np.median(capped)),
        "q1": float(np.percentile(capped, 25)),
        "q3": float(np.percentile(capped, 75)),
        "iqr": float(np.percentile(capped, 75) - np.percentile(capped, 25)),
        "robust_scale": float(robust.scale_[0]),
        "mean": float(capped.mean()),
        "std": float(capped.std()),
    }

    # The T36 toggle, exercised here so the diagnostic arm is one argument rather than a rewrite.
    arm_a = build_preprocessor(feature_columns([*features.FEATURES, "duration"], with_duration=False))
    arm_b = build_preprocessor(
        feature_columns([*features.FEATURES, "duration"], with_duration=True), extra_numeric=("duration",)
    )
    width_a = len(arm_a.fit_transform(train_frame).columns)
    width_b = len(arm_b.fit_transform(train_frame).columns)

    return {
        "variant": variant,
        "rows": {"train": len(train_frame), "test": len(test_frame)},
        "shapes": {"train": fitted["X_train"].shape, "test": fitted["X_test"].shape},
        "groups": {
            "scaled": list(NUMERIC_SCALED),
            "count": list(NUMERIC_PASSTHROUGH),
            "categorical": list(CATEGORICAL),
            "boolean": list(BOOLEAN),
        },
        "n_features": len(features.FEATURES),
        "n_columns": len(names),
        "onehot_width": int(sum(len(c) for c in encoder.categories_)),
        "levels": levels,
        "test_only_total": int(sum(len(entry["test_only"]) for entry in levels)),
        "scaler": {
            "columns": list(NUMERIC_SCALED),
            "mean": [float(value) for value in scaler.mean_],
            "scale": [float(value) for value in scaler.scale_],
        },
        "capped": capped_stats,
        "names_sample": names[:6] + ["..."] + names[-2:],
        "toggle": {"arm_a_columns": width_a, "arm_b_columns": width_b},
        "checks": validate(fitted),
    }


def print_report(result: dict) -> None:
    print("DepositIQ T24 - the preprocessing ColumnTransformer")
    print("=" * 96)
    print(f"variant       : {result['variant']}")
    print(f"rows          : {result['rows']['train']:,} train / {result['rows']['test']:,} test")
    print(f"features in   : {result['n_features']} columns from src.features.FEATURES")
    print(f"columns out   : {result['n_columns']} = {len(result['groups']['scaled'])} scaled + "
          f"{len(result['groups']['count'])} count + {result['onehot_width']} one-hot + "
          f"{len(result['groups']['boolean'])} boolean")
    print(f"shapes        : train {result['shapes']['train']}, test {result['shapes']['test']}")
    print()
    print("GROUPS")
    for key, label in [("scaled", "StandardScaler"), ("count", "passed through"),
                       ("categorical", "one-hot"), ("boolean", "passed through")]:
        print(f"  {label:<15} {len(result['groups'][key]):>2}  {', '.join(result['groups'][key])}")
    print()
    scaler = result["scaler"]
    print("SCALER STATISTICS (fitted on the training half)")
    for column, mean, scale in zip(scaler["columns"], scaler["mean"], scaler["scale"]):
        print(f"  {column:<16} mean {mean:>9.4f}   scale {scale:>8.4f}")
    capped = result["capped"]
    print(f"  {features.PREVIOUS_CAPPED:<16} NOT scaled: median {capped['median']:.0f}, "
          f"Q1 {capped['q1']:.0f}, Q3 {capped['q3']:.0f}, IQR {capped['iqr']:.0f}")
    print(f"                   a RobustScaler would store scale = {capped['robust_scale']:.1f} "
          f"(sklearn substitutes 1.0 for a zero IQR)")
    print()
    print("ONE-HOT VOCABULARY (fitted on the training half)")
    for entry in result["levels"]:
        extra = f"  TEST-ONLY: {entry['test_only']}" if entry["test_only"] else ""
        print(f"  {entry['column']:<16} {entry['levels']:>2} levels  {', '.join(entry['vocabulary'])}{extra}")
    print(f"  {'':<16} {result['onehot_width']:>2} columns, {result['test_only_total']} test-only levels")
    print()
    print(f"COLUMN NAMES   {result['names_sample']}")
    print(f"T36 TOGGLE     arm A {result['toggle']['arm_a_columns']} columns, "
          f"arm B {result['toggle']['arm_b_columns']} columns")
    print()
    print("CHECKS")
    for check in result["checks"]:
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['check']} - {check['detail']}")
    print("=" * 96)


if __name__ == "__main__":
    result = report()
    print_report(result)
    ok = all(check["passed"] for check in result["checks"])
    print(f"result: {'PASS - fits on train, transforms both halves, 65 columns' if ok else 'FAIL'}")
    print("Next  : T25 selects features from this matrix by mutual information and model importance.")
    sys.exit(0 if ok else 1)
