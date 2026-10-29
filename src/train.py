"""
DepositIQ - the cross-validation harness and every candidate model (T26-T28).

Why this file exists
--------------------
T26 asks for two baselines, and every model after them is judged against them. More usefully, this is
where the *harness* lives: one fold split, one preprocessor factory, one scorer, one metric function.
**T27 adds a Random Forest and T28 an XGBoost model** below, and T29 an imbalance variant, T30 tuned
parameters - each a dictionary entry plus a line in the report, not a second experiment with its own
conventions. The usual way a model comparison goes wrong is not a bad model; it is two models
evaluated slightly differently.

What the two baselines are for
------------------------------
**`dummy_prior`** is not decoration. It fixes the two floors T15's EDA derived (D28) so that they are
*measured* in the same folds as everything else rather than quoted from an earlier session:

* `ROC-AUC = 0.5` - every row gets the same score, so no pair can be ranked correctly.
* `PR-AUC = the positive rate` - a constant score's precision-recall curve sits at the base rate,
  which is why 0.1170 and not 0.5 is the number a PR-AUC has to beat.
* `accuracy = 88.30%` - predicting `no` for everybody is right 88.3% of the time, which is exactly
  why accuracy is not a headline metric here (D04).

**`logreg_balanced`** is the first real model: a linear model over the 12 features / 40 columns T25
kept (D44), with `class_weight="balanced"` and the `StandardScaler` that T24's transformer already
supplies. Its coefficients are also the cheapest available check on T25's decision - if dropping
`job` and `education` cost the linear model something, this is where it would show.

**`random_forest`** (T27) is 300 bagged trees at `min_samples_leaf=10`, `max_features="sqrt"` and
`class_weight="balanced_subsample"` - the weights are recomputed from each bootstrap sample rather
than applied as one global ratio, which is the documented variant for forests and why it is not
simply `"balanced"`. Its settings are deliberately close to T25's instrument so the reduction check
below re-runs that measurement with a model that is actually a candidate.

Why there is no F1 or precision column here
-------------------------------------------
Both are threshold metrics, and a `class_weight="balanced"` model is **deliberately miscalibrated**:
the weights push predicted probabilities up so that the 0.5 cut means something closer to "above the
odds of the positive class" than "above even odds". Reporting F1 at 0.5, or even at D17's 7.50%
break-even, would measure the weighting rather than the model. T26 therefore logs the **threshold-free**
ranking metrics (ROC-AUC, PR-AUC) plus two calibration diagnostics (Brier score, mean predicted
probability) that make the miscalibration visible instead of hiding it. Choosing an operating point is
T34's job, and T29 compares imbalance strategies on the metrics that can carry the comparison.

What is held fixed, and how the module proves it
------------------------------------------------
| element | how |
|---|---|
| rows | the 36,168 training rows of `data/processed/train.csv`; the 9,043 held-out rows are **never loaded** (they are T32's) |
| folds | one `StratifiedKFold(5, shuffle=True, random_state=42)`, materialised **once** and replayed into every estimator |
| preprocessing | `preprocess.build_preprocessor(select.shipped_features())` per fold - the same factory T24 defines, so nothing can be fitted on validation rows |
| metrics | one function, applied to every estimator's out-of-fold predictions, in the same order |
| environment | one process, one run |

Eleven checks run in the module and each is written so it can fail: fold identity, target identity,
structural identity of the two pipelines, per-fold fitting (a fold's scaler must match **that fold's**
training mean and differ from the pooled one), matrix width, prediction range, the dummy's three floors
re-derived arithmetically, the logistic model beating those floors in **every** fold, artifact
round-trip, no leaky column, and the shipped feature set matching T25's frozen decision.

How to run
----------
From the repository root, inside the virtual environment:

    python -m src.train      # ~30 s: 2 estimators x 5 folds, then 2 full fits and the artifact reload

Writes `models/baseline_dummy.joblib`, `models/baseline_logreg.joblib`, `models/model_random_forest.joblib`
and `models/cv_results.json` (renamed from `baselines_cv.json` in T27, when the file stopped holding only
baselines). The JSON is the machine-readable record, so no number in a report has to be re-typed by hand
(D06).

**Runtime:** ~3.8 s for the two baselines, ~156 s with the forest, most of it the forest's 16 fits - 5 CV
folds, 1 full fit and 10 for its two-arm ablation.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src import config, features, preprocess, select
from src.split import encode_target

# ---------------------------------------------------------------------------
# Fixed settings
# ---------------------------------------------------------------------------
N_FOLDS: int = 5                       # D28: 5 folds put 846-847 positives in every validation fold
FOLD_SEED: int = config.RANDOM_STATE   # one seed for the project (D03)
STEP_PREPROCESS: str = "preprocess"
STEP_MODEL: str = "model"

# `mean +/- sd over 5 folds` - the sd is the sample sd, stated here because ddof=0 and ddof=1 differ by
# 12% on five folds and nobody should have to guess which one a table means.
SD_DDOF: int = 1

# The registered estimators. A model is an entry here, not a second script: T27 and T28 add their
# factories to this dict and inherit the folds, the preprocessing and the scorer unchanged.
ESTIMATORS: dict[str, dict] = {
    "dummy_prior": {
        "make": lambda: DummyClassifier(strategy="prior"),
        "role": "floor",
        "note": "predicts the training fold's positive rate for every row (D28's two floors)",
    },
    "logreg_balanced": {
        "make": lambda: LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=FOLD_SEED,
        ),
        "role": "baseline",
        "note": "the task's required baseline; scaling comes from the T24 preprocessor",
    },
    # T27. Four settings decided one at a time, and one that is a correction rather than a choice:
    #
    # * `class_weight="balanced_subsample"`, not `"balanced"`. A forest bootstraps, so every tree sees a
    #   different positive share; sklearn's subsample variant recomputes the weights from each bootstrap
    #   sample instead of applying one global ratio to all of them. It is the documented choice for
    #   forests, and it keeps the imbalance treatment comparable to the logistic baseline's.
    # * `min_samples_leaf=10` - with 36k rows and 40 one-hot columns, leaves of one row fit noise; this is
    #   also what T25's instrument used, so the ablation below is a fair re-run of it.
    # * `max_features="sqrt"` - 6 of 40 columns per split, the standard forest default and what keeps 300
    #   correlated one-hot levels from swamping every tree.
    # * `n_jobs=1` - one process. It makes the runtime comparable across estimators and it is what T25's
    #   instrument used, so the reduction numbers can be read against each other.
    "random_forest": {
        "make": lambda: RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=FOLD_SEED,
            n_jobs=1,
        ),
        "role": "candidate",
        "note": "T27's bagged trees; 300x min_samples_leaf=10, sqrt(40) features per split",
    },
    # T28, and D08's chosen family. Settings, and where they came from:
    #
    # * `max_depth=3` - measured, on fold 1 only, across depths 2/3/4/6: PR-AUC 0.42948 / 0.43760 /
    #   0.43604 / 0.42985. Depth 6 overfits 40 sparse one-hot columns and depth 2 underfits; the two best
    #   are within 0.002, so the shallower one wins as the cheaper explanation. That is a one-fold sanity
    #   check, **not** tuning - T30 tunes properly on all five folds, and this setting is what it starts
    #   from rather than what it concludes.
    # * `learning_rate=0.05` with `n_estimators=400` - 1.8 s a fold measured, so the pair is affordable
    #   sixteen times over. The rate is deliberately not the default 0.3, which at 400 rounds would be
    #   badly over-trained.
    # * `scale_pos_weight` is XGBoost's native imbalance handle (D08) and stands in for the other models'
    #   `class_weight`; see `SCALE_POS_WEIGHT` for where the number comes from and how it is checked.
    # * `tree_method="hist"` - the only exact-enough method that is fast on 36k x 40 dense floats.
    # * `n_jobs=1` - as everywhere else in this table, so the runtimes are comparable.
    "xgboost_gbdt": {
        "make": lambda: xgboost.XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=5,
            subsample=0.9,
            colsample_bytree=0.8,
            scale_pos_weight=SCALE_POS_WEIGHT,
            tree_method="hist",
            eval_metric="aucpr",
            random_state=FOLD_SEED,
            n_jobs=1,
        ),
        "role": "candidate",
        "note": "T28's boosted trees; 400x depth 3, lr 0.05, scale_pos_weight from the training half",
    },
}

# Artifacts. The fits are the objects T31 replaces; the JSON is the record of every estimator's CV.
MODEL_PATHS: dict[str, Path] = {
    "dummy_prior": config.MODELS_DIR / "baseline_dummy.joblib",
    "logreg_balanced": config.MODELS_DIR / "baseline_logreg.joblib",
    "random_forest": config.MODELS_DIR / "model_random_forest.joblib",
    "xgboost_gbdt": config.MODELS_DIR / "model_xgboost.joblib",
}

# XGBoost's imbalance handle, as a number computed from the training half: 31,937 negatives against 4,231
# positives, so `neg / pos`. It is a **training-only** ratio - nothing here has seen the test half - and a
# check below requires this literal to match the measured ratio, because a hard-coded constant that nobody
# re-derives is how a weight quietly stops meaning what its comment says. It is fixed here rather than    # recomputed per fold because the folds are stratified: measured, the per-fold **training** ratio deviates
    # by at most 1.9e-3 (**0.025%** relative, in the one fold that holds back 847 positives rather than 846) -
    # which cannot matter, while a per-fold weight would make the model's own configuration depend on which
    # fold it is in.
SCALE_POS_WEIGHT: float = 7.548334

# Renamed from `baselines_cv.json` in T27: the file now holds a candidate model, not only baselines, and
# a record named for its contents is worth more than one that needs a footnote.
CV_PATH: Path = config.MODELS_DIR / "cv_results.json"

# The estimators T25's decision is re-measured on: the linear model that questioned it, and the forest
# that made it. XGBoost joins this tuple in T28, and D46's revisit trigger is read off that entry.
REDUCTION_ESTIMATORS: tuple[str, ...] = ("logreg_balanced", "random_forest", "xgboost_gbdt")

# Columns that must never reach the matrix (D05, D36). Checked on the fitted transformer's output names.
FORBIDDEN: tuple[str, ...] = ("duration", "pdays", config.TARGET)


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------
def make_pipeline(estimator, columns: list[str] | tuple[str, ...] | None = None) -> Pipeline:
    """
    The one place a model is assembled: T24's preprocessor, then the estimator.

    `columns` defaults to T25's shipped feature set (12 features -> 40 matrix columns), so a caller
    cannot accidentally train on the 65-column set T25 measured as unnecessary (D44).
    """
    if columns is None:
        columns = select.shipped_features()
    return Pipeline(
        steps=[
            (STEP_PREPROCESS, preprocess.build_preprocessor(columns)),
            (STEP_MODEL, estimator),
        ]
    )


def materialise_folds(frame: pd.DataFrame, y: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    The fold split, produced **once** per run.

    Materialised rather than re-created per estimator: `StratifiedKFold` would return the same arrays
    for the same seed, but "the same because the seed coincides" is a weaker guarantee than "the same
    object", and T36 makes this a hard requirement for its two arms (D40). Every model in a run gets
    this exact list.
    """
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    return [(train_idx, valid_idx) for train_idx, valid_idx in splitter.split(frame, y)]


def fold_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Every metric the project quotes, from one function, in one order."""
    return {
        "rows": int(len(y_true)),
        "positives": int(np.asarray(y_true).sum()),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
        "mean_pred": float(np.mean(scores)),
        "accuracy_at_half": float(accuracy_score(y_true, (scores >= 0.5).astype(int))),
    }


def cross_validate_factory(
    name: str,
    make,
    role: str,
    note: str,
    frame: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    columns: list[str],
) -> dict:
    """
    Fit one estimator factory once per fold and score its out-of-fold predictions.

    The factory is a parameter rather than a registry lookup because T30's search scores candidate
    parameter sets that are not registered estimators, and it has to do that through **this** loop: a
    candidate scored by a second, parallel implementation would make "tuned beats untuned" two
    experiments rather than a comparison (D45).

    `frame.iloc[...]` rather than `.loc[...]` on purpose: the splitter returns **positions**, and the
    engineered frame's index is whatever `load_split` produced, so positional indexing is the only
    correct reading of those arrays.
    """
    records: list[dict] = []
    fits: list[Pipeline] = []

    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        pipeline = make_pipeline(make(), columns)
        pipeline.fit(frame.iloc[train_idx], y.iloc[train_idx])
        scores = pipeline.predict_proba(frame.iloc[valid_idx])[:, 1]
        estimator = pipeline.named_steps[STEP_MODEL]
        records.append({
            "fold": fold,
            **fold_metrics(y.iloc[valid_idx].to_numpy(), scores),
            # Recorded because a model that stopped at `max_iter` has not converged, and its coefficients
            # (the check on T25) would then be a snapshot of an optimisation rather than a solution.
            "n_iter": int(getattr(estimator, "n_iter_", [-1])[0]),
            # The two denominators are different and the difference is not noise: the *validation* rate
            # drives PR-AUC and accuracy (it is `y_true`), while the *training* rate is what the dummy
            # predicts. Comparing one against the other is how a floor check reports a 4e-5 "failure"
            # that is actually the check describing the wrong fold.
            "train_rows": int(len(train_idx)),
            "train_positives": int(y.iloc[train_idx].sum()),
            "train_rate": float(y.iloc[train_idx].mean()),
        })
        fits.append((pipeline, train_idx, valid_idx))

    return {
        "estimator": name,
        "role": role,
        "note": note,
        "folds": records,
        "summary": summarise(records),
        "fits": fits,
    }


def cross_validate(
    name: str,
    frame: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    columns: list[str],
) -> dict:
    """Score a **registered** estimator on the run's folds. See `cross_validate_factory`."""
    entry = ESTIMATORS[name]
    return cross_validate_factory(
        name, entry["make"], entry["role"], entry["note"], frame, y, folds, columns
    )


def reduction_check(
    frame: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    names: tuple[str, ...] = REDUCTION_ESTIMATORS,
) -> dict:
    """
    Re-measure T25's reduction, per estimator, on the same folds.

    D44 dropped five features because a 200-tree forest could not tell their contribution from zero,
    and recorded that the decision was **instrument-specific** - which is an invitation to check it
    with a different instrument, not a claim that needs no checking.

    * **`logreg_balanced`** is the estimator most likely to *feel* the loss: each one-hot level gets its
      own coefficient, so a column carrying signal cannot hide behind a correlated one. T26 measured
      **-0.0039 +/- 0.0023** ROC-AUC there, with no fold improving.
    * **`random_forest`** is the instrument the decision was made with, at production settings: 300 trees
      rather than T25's 200, and `balanced_subsample` weights. T25 reported **-0.00014 +/- 0.00195** for
      the same toggle on its importance-only forest. The two forest numbers describe the same question
      with different weights, so they are **related but not identical** and are never quoted against each
      other; what matters here is whether the forest family still shows no loss when the trees are
      actually being fitted as candidates.

    Each estimator's shipped arm is built from that estimator's registered factory and the shipped column
    list, so its five folds must reproduce the corresponding run **bit-identically**. A check asserts it,
    and that assertion is what makes this an ablation (two column lists) rather than a comparison (two
    experiments that happen to look similar).
    """
    arms = {"all17": list(features.FEATURES), "shipped12": select.shipped_features()}
    arms_info = {"features": {label: len(columns) for label, columns in arms.items()}, "columns": {}}
    results: dict[str, dict] = {}

    for name in names:
        records: list[dict] = []
        for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
            row: dict = {"fold": fold}
            for label, columns in arms.items():
                pipeline = make_pipeline(ESTIMATORS[name]["make"](), columns)
                pipeline.fit(frame.iloc[train_idx], y.iloc[train_idx])
                scores = pipeline.predict_proba(frame.iloc[valid_idx])[:, 1]
                row[label] = {
                    **fold_metrics(y.iloc[valid_idx].to_numpy(), scores),
                    "columns": int(len(pipeline.named_steps[STEP_PREPROCESS].get_feature_names_out())),
                    # -1 where the estimator has no iteration count (a forest has no `n_iter_`).
                    "n_iter": int(getattr(pipeline.named_steps[STEP_MODEL], "n_iter_", [-1])[0]),
                }
            records.append(row)

        summary: dict = {}
        for metric in ("roc_auc", "pr_auc"):
            deltas = np.array([row["shipped12"][metric] - row["all17"][metric] for row in records])
            summary[metric] = {
                "all17_mean": float(np.mean([row["all17"][metric] for row in records])),
                "shipped12_mean": float(np.mean([row["shipped12"][metric] for row in records])),
                "delta_mean": float(deltas.mean()),
                "delta_sd": float(deltas.std(ddof=SD_DDOF)),
                "folds_improving": int((deltas > 0).sum()),
                # The paired sign test: with five folds and no ties, "all five went the same way" is
                # p = 1/32. Quoted because a small mean over five folds is otherwise hard to weigh.
                "sign_test_p": float(0.5 ** len(deltas)) if (deltas > 0).all() or (deltas < 0).all() else None,
            }
        results[name] = {"folds": records, "summary": summary}
        arms_info["columns"] = {label: records[0][label]["columns"] for label in arms}

    return {"arms": arms_info, "estimators": results}


def summarise(records: list[dict]) -> dict:
    """Mean and sample sd over the folds, for every metric that is a number."""
    metrics = ("roc_auc", "pr_auc", "brier", "mean_pred", "accuracy_at_half")
    summary = {}
    for metric in metrics:
        values = np.array([record[metric] for record in records], dtype=float)
        summary[metric] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=SD_DDOF)),
        }
    return summary


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def validate(
    frame: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    runs: dict[str, dict],
    finals: dict[str, Pipeline],
    columns: list[str],
    reduction: dict,
) -> list[dict]:
    """
    Check the properties a comparison is entitled to assume, and derive the floors rather than trust them.

    Same contract as the earlier phases: a check that cannot execute is recorded as FAILED, never raised,
    because a guard rail that crashes is a guard rail that is absent exactly when it is needed (D25).
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

    base = float(y.mean())

    # --- the two estimators were evaluated on the same thing ---------------------------------------
    def same_folds() -> tuple[bool, str]:
        again = materialise_folds(frame, y)
        same = len(again) == len(folds) and all(
            np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(again, folds)
        )
        sizes = [(len(v), int(y.iloc[v].sum())) for _, v in folds]
        return (
            same and all(len(run["folds"]) == len(folds) for run in runs.values())
            and len({tuple(sizes) for _ in runs.values()}) == 1,
            f"folds {len(folds)}, validation sizes/positives {sizes[0]} ... {sizes[-1]}",
        )

    compare("the fold split is deterministic and shared by every estimator", same_folds)

    def same_target() -> tuple[bool, str]:
        sums = {
            name: [record["positives"] for record in run["folds"]]
            for name, run in runs.items()
        }
        rows = {name: [record["rows"] for record in run["folds"]] for name, run in runs.items()}
        return (
            len({tuple(v) for v in sums.values()}) == 1 and len({tuple(v) for v in rows.values()}) == 1,
            f"{len(folds)} folds x {sum(rows['dummy_prior'])} validation rows, "
            f"{sum(sums['dummy_prior'])} positives",
        )

    compare("every estimator scored the same target rows", same_target)

    def same_pipeline_shape() -> tuple[bool, str]:
        """The pipelines must differ in the estimator and in nothing else."""
        signatures = {}
        for name in runs:
            pipeline = make_pipeline(ESTIMATORS[name]["make"]())
            transformer = pipeline.named_steps[STEP_PREPROCESS]
            signatures[name] = (
                tuple(
                    (step_name, type(step).__name__, tuple(step_columns))
                    for step_name, step, step_columns in transformer.transformers
                ),
                tuple(transformer.feature_names_out if hasattr(transformer, "feature_names_out") else ()),
                type(pipeline.named_steps[STEP_MODEL]).__name__,
            )
        bodies = {signature[:2] for signature in signatures.values()}
        estimators = {signature[2] for signature in signatures.values()}
        return (
            len(bodies) == 1 and len(estimators) == len(signatures),
            f"identical preprocessing across {len(signatures)} estimators, differing estimators: "
            f"{sorted(estimators)}",
        )

    compare("the estimators differ only in the estimator", same_pipeline_shape)

    # --- the preprocessor never saw a validation row ----------------------------------------------
    def fitted_per_fold() -> tuple[bool, str]:
        """
        Each fold's scaler must match that fold's training rows, and differ from the pooled frame.

        Two halves, both needed. Matching the fold's own rows proves the transformer was fitted on the
        training side; *differing* from the pooled mean proves it was not fitted on all 36,168 rows,
        which is the leak a single well-meaning `fit` before the loop would create while still looking
        exact.
        """
        worst_fold, worst_pooled, pooled_delta = 0.0, 0.0, 0.0
        for name, run in runs.items():
            for pipeline, train_idx, _ in run["fits"]:
                scaler = pipeline.named_steps[STEP_PREPROCESS].named_transformers_["scaled"]
                own = frame.iloc[train_idx][list(preprocess.NUMERIC_SCALED)].to_numpy(dtype=float)
                pooled = frame[list(preprocess.NUMERIC_SCALED)].to_numpy(dtype=float)
                worst_fold = max(worst_fold, float(np.max(np.abs(scaler.mean_ - own.mean(axis=0)))))
                worst_pooled = max(worst_pooled, float(np.max(np.abs(scaler.mean_ - pooled.mean(axis=0)))))
                pooled_delta = max(pooled_delta, float(np.max(np.abs(own.mean(axis=0) - pooled.mean(axis=0)))))
        return (
            worst_fold < 1e-12 and worst_pooled > 1e-9,
            f"fold mean matched to {worst_fold:.1e}; pooled mean is {worst_pooled:.3e} away "
            f"(folds themselves differ by up to {pooled_delta:.3e})",
        )

    compare("the preprocessor is fitted per fold, never on the pooled frame", fitted_per_fold)

    # --- the matrix is the one T25 and T24 describe -----------------------------------------------
    def matrix_width() -> tuple[bool, str]:
        widths, names = set(), None
        for name, run in runs.items():
            for pipeline, train_idx, _ in run["fits"]:
                out = pipeline.named_steps[STEP_PREPROCESS].get_feature_names_out()
                widths.add(len(out))
                names = names or list(out)
        full = preprocess.build_preprocessor(columns).fit_transform(frame)
        blocked = [column for column in full.columns if column.split("=", 1)[0] in FORBIDDEN]
        return (
            widths == {len(full.columns)} and not blocked and len(columns) == 12,
            f"{sorted(widths)} columns over {len(columns)} features, {len(full):,} rows, "
            f"forbidden columns present: {blocked}",
        )

    compare("every fold built the 40-column matrix from the 12 kept features", matrix_width)

    def prediction_range() -> tuple[bool, str]:
        worst = (1.0, 0.0)
        for name, run in runs.items():
            for pipeline, _, valid_idx in run["fits"]:
                scores = pipeline.predict_proba(frame.iloc[valid_idx])[:, 1]
                worst = (min(worst[0], float(scores.min())), max(worst[1], float(scores.max())))
        return (np.isfinite(worst).all() and 0.0 <= worst[0] and worst[1] <= 1.0,
                f"probabilities span [{worst[0]:.4f}, {worst[1]:.4f}]")

    compare("no prediction is missing or outside [0, 1]", prediction_range)

    # --- the dummy sits exactly on the floors, and they are derived, not assumed -------------------
    def dummy_floors() -> tuple[bool, str]:
        """
        Re-derive the three floors arithmetically instead of quoting D28.

        A constant score `p` cannot rank, so ROC-AUC is 0.5 exactly; its average precision is `p`,
        because precision equals the base rate at every recall level; and its Brier score is
        `mean((y - p)^2) = p(1 - p)` when the fold's positive rate is `p`. If any of the three drifts,
        the "floor" being compared against is not the floor.
        """
        run = runs["dummy_prior"]
        problems = []
        for record in run["folds"]:
            # q = the rate in the rows being scored (`y_true`), p = the constant score predicted, which
            # is the rate in the rows the dummy was fitted on.
            q = record["positives"] / record["rows"]
            p = record["train_rate"]
            if abs(record["roc_auc"] - 0.5) > 1e-12:
                problems.append(f"fold {record['fold']}: ROC-AUC {record['roc_auc']:.6f} != 0.5")
            if abs(record["pr_auc"] - q) > 1e-12:
                problems.append(f"fold {record['fold']}: PR-AUC {record['pr_auc']:.6f} != scored rate {q:.6f}")
            if abs(record["mean_pred"] - p) > 1e-12:
                problems.append(f"fold {record['fold']}: mean p {record['mean_pred']:.6f} != fitted rate {p:.6f}")
            # Brier of a constant score, expanded: mean((y-p)^2) = q - 2pq + p^2 because y is 0/1.
            expected_brier = q - 2 * p * q + p * p
            if abs(record["brier"] - expected_brier) > 1e-12:
                problems.append(f"fold {record['fold']}: Brier {record['brier']:.6f} != "
                                f"q - 2pq + p^2 {expected_brier:.6f}")
        accuracy = run["summary"]["accuracy_at_half"]["mean"]
        return (
            not problems and abs(accuracy - (1 - base)) < 0.002,
            f"ROC-AUC 0.5000, PR-AUC = the base rate {base:.4f}, Brier = p(1-p), "
            f"accuracy {accuracy:.4%} vs always-'no' {1 - base:.4%}; problems: {problems or 'none'}",
        )

    compare("the dummy reproduces D28's three floors exactly", dummy_floors)

    def every_model_beats_floors() -> tuple[bool, str]:
        """Every non-floor estimator must clear PR-AUC *and* ROC-AUC in every fold, not on average."""
        dummy = runs["dummy_prior"]
        notes, failures = [], []
        for name, run in runs.items():
            if run["role"] == "floor":
                continue
            pr_fail = [
                (record["fold"], round(record["pr_auc"], 4))
                for record in run["folds"]
                if record["pr_auc"] <= dummy["folds"][record["fold"] - 1]["pr_auc"]
            ]
            auc_fail = [(record["fold"], round(record["roc_auc"], 4))
                        for record in run["folds"] if record["roc_auc"] <= 0.5]
            notes.append(f"{name} {run['summary']['pr_auc']['mean']:.4f} "
                         f"(worst fold {min(r['pr_auc'] for r in run['folds']):.4f})")
            failures.extend((name, pr_fail, auc_fail) for _ in [0] if pr_fail or auc_fail)
        return (
            not failures,
            f"all beat the {dummy['summary']['pr_auc']['mean']:.4f} floor in every fold: "
            f"{'; '.join(notes)}; failures: {failures or 'none'}",
        )

    compare("every model beats both floors in every fold", every_model_beats_floors)

    def no_leak_in_matrix() -> tuple[bool, str]:
        """The fitted pipelines must have been handed a frame that still carried the leak."""
        carried = all(
            column in frame.columns for column in FORBIDDEN
        )
        out = finals["logreg_balanced"].named_steps[STEP_PREPROCESS].get_feature_names_out()
        present = [column for column in out if column.split("=", 1)[0] in FORBIDDEN]
        return (
            carried and not present,
            f"{list(FORBIDDEN)} were in the frame handed to the model and are absent from its "
            f"{len(out)} columns",
        )

    compare("no leaky or target column reached the matrix", no_leak_in_matrix)

    # --- the artifacts are what they claim --------------------------------------------------------
    def artifacts_reload() -> tuple[bool, str]:
        missing = [path.name for path in MODEL_PATHS.values() if not path.exists()]
        if missing:
            return False, f"missing: {missing}"
        sample = frame.iloc[:500]
        notes = []
        for name, path in MODEL_PATHS.items():
            reloaded = joblib.load(path)
            fresh = reloaded.predict_proba(sample)[:, 1]
            original = finals[name].predict_proba(sample)[:, 1]
            width = len(reloaded.named_steps[STEP_PREPROCESS].get_feature_names_out())
            ok = (
                np.array_equal(fresh, original)
                and width == 40
                and [step for step, _ in reloaded.steps] == [STEP_PREPROCESS, STEP_MODEL]
            )
            notes.append(f"{path.name} {'ok' if ok else 'DRIFTED'} "
                         f"(max delta {float(np.max(np.abs(fresh - original))):.1e}, {width} columns)")
            if not ok:
                return False, "; ".join(notes)
        return True, "; ".join(notes)

    compare("every saved artifact reloads and predicts identically", artifacts_reload)

    def imbalance_weight_is_current() -> tuple[bool, str]:
        """`SCALE_POS_WEIGHT` must equal the ratio measured on the training half, to 1e-5."""
        negatives = int((y == 0).sum())
        positives = int((y == 1).sum())
        measured = negatives / positives
        # The comparison is against each fold's **training** rows, because that is the side a per-fold weight
        # would be computed from. Using the validation fold's ratio instead measures the wrong thing - and
        # reports a 2.5e-3 "drift" that is really just the held-out share, the same wrong-denominator
        # mistake the dummy's floor check made in T26.
        deviations = []
        for train_idx, _ in folds:
            tr = y.iloc[train_idx]
            deviations.append(abs((len(tr) - int(tr.sum())) / max(int(tr.sum()), 1) - SCALE_POS_WEIGHT))
        return (
            abs(measured - SCALE_POS_WEIGHT) < 1e-5 and max(deviations) < 5e-3,
            f"SCALE_POS_WEIGHT={SCALE_POS_WEIGHT:.6f}, training half {negatives:,}/{positives:,} = "
            f"{measured:.6f}, worst fold-train deviation {max(deviations):.2e}",
        )

    compare("the imbalance weight matches the training half", imbalance_weight_is_current)

    add("the shipped feature set is T25's frozen decision",
        columns == select.shipped_features() and len(columns) == 12,
        f"{len(columns)} features; frozen drops {list(select.SHIPPED_DROPPED)}")

    # --- a fit that stopped at its iteration cap is a snapshot, not a solution -----------------------
    def converged() -> tuple[bool, str]:
        notes, problems = [], []
        for name in runs:
            limit = getattr(ESTIMATORS[name]["make"](), "max_iter", None)
            iterations = [record["n_iter"] for record in runs[name]["folds"]]
            if limit is None or max(iterations) < 0:
                notes.append(f"{name}: no iteration cap")
                continue
            notes.append(f"{name}: {min(iterations)}-{max(iterations)} of {limit}")
            if max(iterations) >= limit:
                problems.append(name)
        return not problems, "; ".join(notes)

    compare("no model stopped at its iteration cap", converged)

    def reduction_is_an_ablation() -> tuple[bool, str]:
        """
        Each reduction arm must be the same experiment twice over, not two experiments.

        Every estimator's shipped arm is built from that estimator's registered factory and the shipped
        column list, so all five folds' two ranking metrics have to come back bit-identical to its normal
        CV run. If they do not, the ablation has acquired a second difference and its delta means nothing.
        """
        drift, notes = [], []
        for name, block in reduction["estimators"].items():
            pairs = list(zip(runs[name]["folds"], block["folds"]))
            local = [
                (baseline["fold"], metric)
                for baseline, row in pairs
                for metric in ("roc_auc", "pr_auc")
                if abs(baseline[metric] - row["shipped12"][metric]) > 0.0
            ]
            drift.extend((name, *item) for item in local)
            notes.append(f"{name}: {len(pairs)} folds")
        columns = reduction["arms"]["columns"]
        return (
            not drift
            and all(len(block["folds"]) == N_FOLDS for block in reduction["estimators"].values())
            and columns["all17"] > columns["shipped12"],
            f"{'; '.join(notes)}, shipped arm identical to each run (drift: {drift or 'none'}), "
            f"{columns['all17']} -> {columns['shipped12']} columns",
        )

    compare("every reduction arm re-runs its own estimator, not a second experiment",
            reduction_is_an_ablation)

    return checks


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(variant: str = "full") -> dict:
    """Train every registered estimator, persist the fits, and return everything the module prints."""
    config.ensure_dirs()
    started = time.perf_counter()

    frame = features.load_engineered("train", variant)
    y = encode_target(frame)
    columns = select.shipped_features()
    folds = materialise_folds(frame, y)

    runs = {
        name: cross_validate(name, frame, y, folds, columns)
        for name in ESTIMATORS
    }

    # The persisted artifacts are fitted on the whole training half. Their CV score is what is
    # comparable; their in-sample score is not reported anywhere, because a trained-on-everything
    # number next to a cross-validated one invites exactly the wrong comparison.
    finals: dict[str, Pipeline] = {}
    for name, path in MODEL_PATHS.items():
        pipeline = make_pipeline(ESTIMATORS[name]["make"](), columns)
        pipeline.fit(frame, y)
        # `compress=3`: a 300-tree forest stores every leaf it grew and lands at 41 MB uncompressed,
        # which is 10x the size of everything else this project commits. Compressed it is a few MB and
        # reloads to the same arrays (checked below), so the size is paid once rather than per clone.
        joblib.dump(pipeline, path, compress=3)
        finals[name] = pipeline

    # D44's decision was measured with a forest; this is the same question asked of a linear model.
    reduction = reduction_check(frame, y, folds)

    checks = validate(frame, y, folds, runs, finals, columns, reduction)

    payload = {
        "task": "T26-T27",
        "variant": variant,
        "training_rows": int(len(frame)),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "features": list(columns),
        "dropped_features": list(select.SHIPPED_DROPPED),
        "matrix_columns": int(len(
            preprocess.build_preprocessor(columns).fit_transform(frame).columns)),
        "folds": N_FOLDS,
        "fold_seed": FOLD_SEED,
        "sd_ddof": SD_DDOF,
        "estimators": {
            name: {
                "role": run["role"],
                "note": run["note"],
                "params": ESTIMATORS[name]["make"]().get_params(),
                "folds": run["folds"],
                "summary": run["summary"],
                "artifact": str(MODEL_PATHS[name].relative_to(config.ROOT)),
            }
            for name, run in runs.items()
        },
        "reduction_check": reduction,
        "library": {"xgboost": xgboost.__version__, "scikit-learn": sklearn.__version__},
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }

    # The record is written **before** the last check and rewritten once at the end. That ordering is
    # deliberate: a failing check must not cost the run its numbers, and the final check is precisely
    # "the file on disk says what the run measured" - which can only be tested against a file that
    # exists.
    CV_PATH.write_text(json.dumps(payload, indent=2, sort_keys=False))

    checks.extend(validate_record(json.loads(CV_PATH.read_text()), payload))
    payload["checks"] = checks
    payload["checks_passed"] = sum(1 for check in checks if check["passed"])
    payload["checks_total"] = len(checks)
    payload["checks_ok"] = payload["checks_passed"] == payload["checks_total"]
    CV_PATH.write_text(json.dumps(payload, indent=2, sort_keys=False))

    return {**payload, "runs": runs, "folds_used": folds}


def validate_record(reading: dict, measured: dict) -> list[dict]:
    """
    Check the written record against the run that produced it, and against the shapes downstream expects.

    `reading` is the file parsed back from disk, not the dict that was written, so a value JSON could not
    represent (a numpy scalar, a NaN, a Path) has already had its chance to fail or drift.
    """
    checks: list[dict] = []

    def compare(name: str, evaluate) -> None:
        try:
            passed, detail = evaluate()
        except Exception as error:  # noqa: BLE001
            passed, detail = False, f"check could not run: {type(error).__name__}: {error}"
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    compare("the CV record on disk matches the run", lambda: (
        reading["estimators"]["logreg_balanced"]["summary"]["pr_auc"]["mean"]
        == measured["estimators"]["logreg_balanced"]["summary"]["pr_auc"]["mean"]
        and reading["estimators"]["dummy_prior"]["summary"]["roc_auc"]["mean"] == 0.5
        and reading["training_rows"] == measured["training_rows"]
        and reading["matrix_columns"] == 40
        and len(reading["estimators"]) == len(ESTIMATORS)
        and all(len(block["folds"]) == N_FOLDS for block in reading["estimators"].values()),
        f"{CV_PATH.name}: {reading['training_rows']:,} rows, {reading['matrix_columns']} columns, "
        f"{len(reading['estimators'])} estimators x {N_FOLDS} folds",
    ))
    return checks


def print_report(result: dict) -> None:
    """The evidence the task asks for: CV ROC-AUC and PR-AUC, per estimator and per fold."""
    base = result["base_rate"]
    print("DepositIQ T26 - baselines")
    print("=" * 112)
    print(f"training rows : {result['training_rows']:,}  ({result['positives']:,} positives, "
          f"{base:.4%})")
    print(f"features      : {len(result['features'])} of 17 (T25/D44, built by "
          f"src.select.shipped_features) -> {result['matrix_columns']} matrix columns")
    print(f"folds         : {result['folds']} stratified, seed {result['fold_seed']}, "
          f"sd over folds uses ddof={result['sd_ddof']}")
    print(f"columns       : ROC-AUC and PR-AUC are the headline metrics (D04); Brier and mean p are "
          f"diagnostics")
    print()
    header = (f"{'estimator':<20}{'ROC-AUC':>18}{'PR-AUC':>18}{'PR-AUC x base':>15}"
              f"{'Brier':>12}{'mean p':>10}{'acc@0.5':>10}")
    print(header)
    print("-" * len(header))
    for name, run in result["runs"].items():
        line = f"{name:<20}"
        for metric in ("roc_auc", "pr_auc"):
            stats = run["summary"][metric]
            line += f"{stats['mean']:>12.4f} +/- {stats['sd']:.4f}".rjust(18)
        line += f"{run['summary']['pr_auc']['mean'] / base:>15.2f}x"
        line += f"{run['summary']['brier']['mean']:>12.4f}"
        line += f"{run['summary']['mean_pred']['mean']:>10.4f}"
        line += f"{run['summary']['accuracy_at_half']['mean']:>10.4%}"
        print(line)
        print(f"  {run['role']}: {run['note']}")
    print()
    print(f"FLOORS (D28, re-derived here rather than quoted): ROC-AUC 0.5000, PR-AUC = the base rate "
          f"{base:.4f}, accuracy {1 - base:.4%}")
    print("  A PR-AUC of 0.60 is 5.1x the base rate - the multiplier is the honest reading, not the "
          "raw number.")
    print()
    print("PER-FOLD PR-AUC (the fold is the unit of evidence; 5 folds, 1,084-1,085 held-out rows each)")
    print(f"  {'fold':<6}{'positives':>10}" + "".join(f"{name:>20}" for name in result["runs"]))
    for index in range(result["folds"]):
        line = f"  {index + 1:<6}{result['runs']['dummy_prior']['folds'][index]['positives']:>10}"
        for run in result["runs"].values():
            line += f"{run['folds'][index]['pr_auc']:>20.4f}"
        print(line)
    print()
    reduction = result["reduction_check"]
    print("DOES T25's REDUCTION COST ANYTHING?  (same folds, same estimator, columns toggled)")
    for name, block in reduction["estimators"].items():
        print(f"  {name}")
        print(f"    {'arm':<12}{'features':>10}{'columns':>9}" +
              "".join(f"{m:>20}" for m in ("ROC-AUC", "PR-AUC")))
        for arm, features_count in reduction["arms"]["features"].items():
            line = f"    {arm:<12}{features_count:>10}{reduction['arms']['columns'][arm]:>9}"
            for metric in ("roc_auc", "pr_auc"):
                line += f"{block['summary'][metric][arm + '_mean']:>20.4f}"
            print(line)
        for metric, label in (("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")):
            stats = block["summary"][metric]
            sign = f", sign test p = {stats['sign_test_p']:.3f}" if stats["sign_test_p"] else ""
            print(f"    delta {label:<8}{stats['delta_mean']:+.4f} +/- {stats['delta_sd']:.4f}, "
                  f"{stats['folds_improving']}/{result['folds']} folds improve on 12 features{sign}")
        iterations = [row["shipped12"]["n_iter"] for row in block["folds"]]
        if max(iterations) >= 0:
            print(f"    n_iter    {iterations} (shipped arm)")
    print()
    artifacts = ", ".join(
        [Path(item["artifact"]).name for item in result["estimators"].values()] + [CV_PATH.name]
    )
    print(f"artifacts: {artifacts}")
    print(f"elapsed  : {result['elapsed_seconds']} s")
    print()
    print(f"Phase 6 checks: {result['checks_passed']}/{result['checks_total']} passed")
    for check in result["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        detail = f"  ({check['detail']})" if check["detail"] else ""
        print(f"  [{marker}] {check['check']}{detail}")
    print()


if __name__ == "__main__":
    print("DepositIQ Phase 6 - baseline models")
    print()
    outcome = report()
    print_report(outcome)
    print("=" * 112)
    print(f"result : {'PASS - every model trained in one Pipeline and scored on the same folds' if outcome['checks_ok'] else 'FAIL'}")
    print("Next   : T28 adds XGBoost as one more entry in ESTIMATORS, inheriting this harness.")
    sys.exit(0 if outcome["checks_ok"] else 1)
