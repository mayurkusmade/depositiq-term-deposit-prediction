"""
DepositIQ - hyperparameter tuning (T30).

Why this file exists
--------------------
Every setting in `src/train.py`'s XGBoost entry was chosen one at a time, by hand, mostly from a
**one-fold** sweep: `max_depth=3` came from a fold-1 comparison recorded in T28 as "a starting point for
T30, explicitly not tuning". This module does the tuning properly - all five folds, a stated space, a
stated budget - and it has to do so without becoming a second harness.

Two things are therefore inherited rather than re-implemented
-------------------------------------------------------------
* **The folds.** `train.materialise_folds` produces them once and they are passed to the search as an
  explicit list of `(train_positions, valid_positions)`, so `RandomizedSearchCV` replays *these* arrays
  instead of constructing its own splitter that happens to share a seed.
* **The scoring loop.** Every arm is scored by `train.cross_validate_factory`, the same function (refactored
  out of `cross_validate` for this purpose) that scores T26-T28's models: same `train.make_pipeline`, same
  `train.fold_metrics`, same `train.summarise`. A candidate scored by a parallel implementation would make
  "tuned beats untuned" two experiments, which is exactly the failure D45 exists to prevent.

The treatment is not tuned
--------------------------
T29 measured that `scale_pos_weight` buys no detectable ranking (+0.0023 PR-AUC at 1.11 SE *against* it) and
costs half the Brier score, so the **untreated** configuration is the one being tuned. The weight is kept
out of the search space - a search is free to trade calibration for ranking, and letting it choose the
imbalance treatment would quietly reopen a question that was already answered with evidence. It is
evaluated afterwards as a third arm, so "does the weight still cost nothing at a better point in the space"
has an answer rather than a shrug. SMOTE is closed by T29 and does not appear anywhere in this file.

What the search's own score means, and how it is reported
--------------------------------------------------------
The winner is selected on the **same five folds** that T32 will report, so its CV score is **optimistic**:
some of its edge over the untuned configuration is the edge of being the best of 40 candidates on the data
it was chosen on. Nothing here hides that - `search.best_score_` is never quoted as a result, the arms
table is rebuilt by refitting the chosen parameters through the ordinary loop, and the record carries the
**distribution of all 40 candidates** plus how many of them sit within one standard error of the winner.
If the tuned-vs-untuned gain is small next to that spread, the doc says so. T32's held-out half is what
disciplines the number; this module only produces the candidate.

Search space
------------
Six axes, each with a reason to be there and a bound that is not arbitrary:

| axis | values | why these |
|---|---|---|
| `n_estimators` | 200, 300, 400, 600, 800 | round count and learning rate trade off; the shipped 400/0.05 is one point in this grid, not a privileged one |
| `learning_rate` | 0.02, 0.03, 0.05, 0.08, 0.1 | 0.02 at 800 rounds was already the best fold-1 timing probe (4.0 s), so the slow end is affordable; 0.1 is kept because 400 rounds at 0.1 must be allowed to lose |
| `max_depth` | 2, 3, 4, 5 | T28's probe covered 2-6 on one fold and depth 6 overfit the 40 sparse one-hot columns; 6 is excluded on that measured evidence, and depth 1 is excluded because a stump per round is a different model family |
| `min_child_weight` | 1, 3, 5, 10 | the shipped 5 is the middle; 1 is the default and 10 is the regularised end |
| `subsample` | 0.7, 0.8, 0.9, 1.0 | row bagging, from none to strong |
| `colsample_bytree` | 0.6, 0.8, 1.0 | 40 correlated one-hot columns make column bagging plausible |

4,800 combinations, `n_iter=40` draws - 0.8% of the space. That is a search, not an optimum: the doc
states the budget so that a better configuration found later is a bigger budget rather than a mistake.

How to run
----------
From the repository root, inside the virtual environment:

    python -m src.tune    # ~3 minutes: 40 candidates x 5 folds, then three arms, then the rebuild check

Writes `models/best_params.json`; `docs/06_modeling.md` §9 quotes only numbers from that file.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.model_selection import RandomizedSearchCV

from src import config, features, preprocess, select, train
from src.split import encode_target

# ---------------------------------------------------------------------------
# Fixed settings
# ---------------------------------------------------------------------------
SEARCH_SPACE: dict[str, list] = {
    "n_estimators": [200, 300, 400, 600, 800],
    "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
    "max_depth": [2, 3, 4, 5],
    "min_child_weight": [1, 3, 5, 10],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

N_ITER: int = 40                       # 0.8% of the 4,800-point space; the budget, stated as a budget
SEED: int = config.RANDOM_STATE        # D03: one seed for the project
SCORING: str = "average_precision"    # D04's metric - PR-AUC, the one that responds to the positive class
SEARCH_JOBS: int = -1                  # parallelise the **search**; each fit stays n_jobs=1 (below)

# The treatment, held fixed at "none" for the search. T29's `none` arm; not a searchable axis.
TREATMENT: float = 1.0

# Every axis the search varies. `scale_pos_weight` is deliberately absent and a check asserts it.
TUNABLE: tuple[str, ...] = tuple(SEARCH_SPACE)

RECORD_PATH: Path = config.MODELS_DIR / "best_params.json"
IMBALANCE_RECORD: Path = config.MODELS_DIR / "imbalance_results.json"


# ---------------------------------------------------------------------------
# Building a candidate
# ---------------------------------------------------------------------------
def make_estimator(params: dict | None = None, scale_pos_weight: float = TREATMENT):
    """
    One XGBoost candidate, with every setting that is **not** being searched pinned in one place.

    The pinned ones are pinned for reasons that outlive a search: `tree_method="hist"` is the only
    exact-enough method that is fast on 36k x 40 dense floats, `eval_metric="aucpr"` matches the metric
    being selected on, `random_state` fixes the tie-breaking, and `n_jobs=1` keeps a single fit's runtime
    comparable across candidates - the parallelism belongs in the search, not inside each model, or two
    candidates' costs are being measured on different numbers of cores.
    """
    settings = {
        "tree_method": "hist",
        "eval_metric": "aucpr",
        "random_state": train.FOLD_SEED,
        "n_jobs": 1,
        "scale_pos_weight": float(scale_pos_weight),
    }
    settings.update(params or {})
    if "scale_pos_weight" in (params or {}):
        raise ValueError("the imbalance treatment is not searchable in T30; T29 decided it (D49)")
    return xgboost.XGBClassifier(**settings)


def pipeline_params(params: dict) -> dict:
    """`n_estimators` -> `model__n_estimators`: the pipeline is searched, and its last step is the model."""
    return {f"{train.STEP_MODEL}__{axis}": value for axis, value in params.items()}


def registered_params() -> dict:
    """
    The configuration T28 shipped, read from the registry rather than retyped.

    Retyping it would make "tuned beats untuned" depend on this file's copy being current; reading it
    means the comparison is against whatever `src/train.py` actually fits.
    """
    settings = train.ESTIMATORS["xgboost_gbdt"]["make"]().get_params()
    return {axis: settings[axis] for axis in TUNABLE}


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------
def run_search(
    frame: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    columns: list[str],
) -> tuple[RandomizedSearchCV, float]:
    """
    Draw `N_ITER` candidates and score each one on the project's five folds.

    `cv=folds` is the load-bearing argument: the list of positional index arrays is passed straight to
    `RandomizedSearchCV`, so its folds are *these* arrays rather than an identically-seeded splitter's.
    `refit=True` leaves `best_estimator_` fitted on the whole training half - the object T31 can persist
    without a second definition of "the tuned model".
    """
    search = RandomizedSearchCV(
        estimator=train.make_pipeline(make_estimator(), columns),
        param_distributions=pipeline_params(SEARCH_SPACE),
        n_iter=N_ITER,
        scoring=SCORING,
        cv=folds,
        refit=True,
        random_state=SEED,
        n_jobs=SEARCH_JOBS,
        return_train_score=False,
        error_score="raise",
    )
    started = time.perf_counter()
    search.fit(frame, y)
    return search, round(time.perf_counter() - started, 1)


def best_params(search: RandomizedSearchCV) -> dict:
    """The winning settings with the pipeline prefix stripped, so they are an estimator's parameters."""
    return {
        key.split(f"{train.STEP_MODEL}__", 1)[1]: value
        for key, value in search.best_params_.items()
    }


def fold_sd(search: RandomizedSearchCV, index: int) -> float:
    """
    The **sample** sd of one candidate's five fold scores, recomputed from the split scores.

    Not `cv_results_["std_test_score"]`: scikit-learn takes that with `ddof=0`, and this project states
    `ddof=1` everywhere else (T26's `SD_DDOF`, chosen because on five folds the two differ by 12%).
    Quoting sklearn's number next to the arms' sds would put two different quantities in one column and
    nobody would be able to see it - on the winner here that is 0.0099 against 0.0110.
    """
    splits = [
        search.cv_results_[f"split{fold}_test_score"][index] for fold in range(train.N_FOLDS)
    ]
    return float(np.std(splits, ddof=train.SD_DDOF))


def top_candidates(search: RandomizedSearchCV, count: int = 5) -> list[dict]:
    """The leaderboard, because a winner quoted without its neighbours is a number without a scale."""
    results = search.cv_results_
    order = np.argsort(results["mean_test_score"])[::-1][:count]
    return [
        {
            "params": {
                key.split(f"{train.STEP_MODEL}__", 1)[1]: results["params"][index][key]
                for key in results["params"][index]
            },
            "mean_test_score": float(results["mean_test_score"][index]),
            "fold_sd": fold_sd(search, index),
            "sklearn_std_test_score_ddof0": float(results["std_test_score"][index]),
        }
        for index in order
    ]


def evaluate(
    name: str,
    params: dict,
    scale_pos_weight: float,
    note: str,
    frame: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    columns: list[str],
) -> dict:
    """
    Score one arm through the project's ordinary loop.

    Binds the parameters as defaults rather than closing over the loop variable: a late-binding lambda
    inside a dict comprehension would give every arm the last arm's settings, and the arms would then
    differ by their labels rather than their parameters.
    """
    return train.cross_validate_factory(
        name,
        lambda p=dict(params), w=float(scale_pos_weight): make_estimator(p, w),
        "candidate",
        note,
        frame,
        y,
        folds,
        columns,
    )


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def validate(
    frame: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    search: RandomizedSearchCV,
    arms: dict[str, dict],
    columns: list[str],
    record: dict,
) -> list[dict]:
    """Same contract as every other phase: a check that cannot run is FAILED, never raised (D25)."""
    checks: list[dict] = []
    chosen = best_params(search)

    def compare(name: str, evaluate_check) -> None:
        try:
            passed, detail = evaluate_check()
        except Exception as error:  # noqa: BLE001 - any failure here is a failed check
            passed, detail = False, f"check could not run: {type(error).__name__}: {error}"
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    # --- it searched on the project's folds, not a look-alike --------------------------------
    def same_folds() -> tuple[bool, str]:
        again = train.materialise_folds(frame, y)
        same = len(search.cv) == len(folds) == train.N_FOLDS and all(
            np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
            for a, b in zip(search.cv, again)
        )
        return (
            same and int(search.n_splits_) == train.N_FOLDS,
            f"search replayed the {len(folds)} materialised splits "
            f"({[len(v) for _, v in folds]} held-out rows, n_splits={int(search.n_splits_)})",
        )

    compare("the search ran on the project's five folds", same_folds)

    # --- the winner is the parameters that were actually fitted ------------------------------
    def winning_params_applied() -> tuple[bool, str]:
        fitted = search.best_estimator_.named_steps[train.STEP_MODEL].get_params()
        wrong = {axis: (value, fitted[axis]) for axis, value in chosen.items() if fitted[axis] != value}
        return (
            not wrong and set(chosen) == set(TUNABLE),
            f"{len(chosen)} chosen parameters reappear in the refitted best estimator; wrong: {wrong or 'none'}",
        )

    compare("every chosen parameter reappears in the fitted estimator", winning_params_applied)

    # --- the treatment was not searched, and did not move -----------------------------------
    def treatment_held() -> tuple[bool, str]:
        space_clean = not any("scale_pos_weight" in key for key in search.param_distributions)
        expected = {"untuned_untreated": TREATMENT, "tuned_untreated": TREATMENT,
                    "tuned_weighted": train.SCALE_POS_WEIGHT}
        applied = {
            name: float(arm["fits"][0][0].named_steps[train.STEP_MODEL].get_params()["scale_pos_weight"])
            for name, arm in arms.items()
        }
        return (
            space_clean and applied == expected and set(applied.values()) == set(expected.values()),
            f"space axes: {sorted(TUNABLE)}; weights applied: {applied} (T29's literal "
            f"{train.SCALE_POS_WEIGHT} for the comparison arm)",
        )

    compare("the imbalance treatment was fixed, not tuned", treatment_held)

    def parallelism_inside_the_fit() -> tuple[bool, str]:
        jobs = {
            name: int(arm["fits"][0][0].named_steps[train.STEP_MODEL].get_params()["n_jobs"])
            for name, arm in arms.items()
        }
        search_jobs = int(search.best_estimator_.named_steps[train.STEP_MODEL].get_params()["n_jobs"])
        return (
            set(jobs.values()) == {1} and search_jobs == 1,
            f"n_jobs=1 inside every fit ({jobs}); the search itself used n_jobs={SEARCH_JOBS}",
        )

    compare("each fit ran on one core, so candidate costs are comparable", parallelism_inside_the_fit)

    def search_settings_recorded() -> tuple[bool, str]:
        distinct = len({json.dumps(params, sort_keys=True) for params in search.cv_results_["params"]})
        return (
            search.scoring == SCORING
            and int(search.n_iter) == N_ITER
            and int(search.random_state) == SEED
            and distinct == N_ITER,
            f"scoring {search.scoring!r}, n_iter {int(search.n_iter)}, seed {int(search.random_state)}, "
            f"{distinct} distinct candidates of {N_ITER} drawn",
        )

    compare("the search records its own budget, metric and seed", search_settings_recorded)

    # --- nothing else changed ---------------------------------------------------------------
    def feature_set_unchanged() -> tuple[bool, str]:
        matrix = len(preprocess.build_preprocessor(columns).fit_transform(frame).columns)
        leaked = [column for column in columns if column in train.FORBIDDEN]
        return (
            list(columns) == list(select.shipped_features())
            and matrix == 40
            and not leaked,
            f"{len(columns)} features -> {matrix} matrix columns, identical to T25's shipped set; "
            f"forbidden columns present: {leaked or 'none'}",
        )

    compare("the feature set is still T25's twelve columns", feature_set_unchanged)

    def smote_absent() -> tuple[bool, str]:
        steps = {name: list(arm["fits"][0][0].named_steps) for name, arm in arms.items()}
        return (
            "smote" not in json.dumps(SEARCH_SPACE).lower()
            and all("smote" not in value for value in steps.values())
            and all(value == [train.STEP_PREPROCESS, train.STEP_MODEL] for value in steps.values()),
            f"pipeline steps {sorted({tuple(v) for v in steps.values()})}; no resampling arm",
        )

    compare("SMOTE is absent, as T29 closed it", smote_absent)

    # --- the untuned arm is the model that already shipped ---------------------------------
    def reproduces_t29_control() -> tuple[bool, str]:
        """
        `untuned_untreated` must be T29's `none` arm to the last digit.

        It is the same estimator, the same weight and the same folds, so anything but bit-identity would
        mean this module changed something it did not intend to - which is the only way a "tuned beats
        untuned" claim can be manufactured rather than measured.
        """
        reference = json.loads(IMBALANCE_RECORD.read_text())["arms"]["none"]["folds"]
        mine = arms["untuned_untreated"]["folds"]
        worst = max(
            abs(float(mine[index][metric]) - float(fold[metric]))
            for index, fold in enumerate(reference)
            for metric in ("pr_auc", "roc_auc", "brier")
        )
        return (
            worst == 0.0 and len(mine) == len(reference),
            f"{len(mine)} folds compared against models/{IMBALANCE_RECORD.name} (arms.none): "
            f"max drift {worst:.1e}",
        )

    compare("the untuned arm reproduces T29's control bit-identically", reproduces_t29_control)

    # --- the arm scores are refits, not the search's own optimism ---------------------------
    def scores_come_from_refits() -> tuple[bool, str]:
        """
        The tuned arm's per-fold PR-AUC must equal the search's own per-split scores for the winner.

        This is what entitles the doc to quote the arms table instead of `search.best_score_`: the numbers
        are independently re-derived through `train.cross_validate_factory`, and they agree because both
        paths score the same model on the same rows - not because one was copied from the other.
        """
        split = np.array(
            [search.cv_results_[f"split{index}_test_score"][search.best_index_]
             for index in range(train.N_FOLDS)],
            dtype=float,
        )
        mine = np.array([fold["pr_auc"] for fold in arms["tuned_untreated"]["folds"]], dtype=float)
        drift = float(np.max(np.abs(split - mine)))
        return (
            drift <= 1e-12 and abs(float(split.mean()) - float(search.best_score_)) <= 1e-12,
            f"{len(mine)} folds re-derived; max drift from the search's own split scores {drift:.1e}; "
            f"refit mean {mine.mean():.6f} vs best_score_ {float(search.best_score_):.6f}",
        )

    compare("the tuned arm's scores are refits, not copied from the search", scores_come_from_refits)

    # --- the done-condition, as a check rather than a claim --------------------------------
    def tuned_beats_untuned() -> tuple[bool, str]:
        tuned = arms["tuned_untreated"]["summary"]["pr_auc"]
        base = arms["untuned_untreated"]["summary"]["pr_auc"]
        deltas = np.array([fold["pr_auc"] for fold in arms["tuned_untreated"]["folds"]]) - np.array(
            [fold["pr_auc"] for fold in arms["untuned_untreated"]["folds"]])
        return (
            tuned["mean"] > base["mean"],
            f"tuned {tuned['mean']:.4f} +/- {tuned['sd']:.4f} vs untuned {base['mean']:.4f} +/- "
            f"{base['sd']:.4f}; delta {deltas.mean():+.5f} +/- {deltas.std(ddof=train.SD_DDOF):.5f}, "
            f"{int((deltas > 0).sum())}/{train.N_FOLDS} folds improve",
        )

    compare("the tuned model beats the untuned one on CV PR-AUC", tuned_beats_untuned)

    # --- T31 can rebuild it from the file alone --------------------------------------------
    def rebuild_from_record() -> tuple[bool, str]:
        reading = json.loads(RECORD_PATH.read_text())
        rebuilt = evaluate(
            "rebuilt_from_record",
            reading["best_params"]["params"],
            TREATMENT,
            "rebuilt from models/best_params.json",
            frame,
            y,
            folds,
            columns,
        )
        mine = [fold["pr_auc"] for fold in rebuilt["folds"]]
        theirs = [fold["pr_auc"] for fold in reading["arms"]["tuned_untreated"]["folds"]]
        drift = max(abs(a - b) for a, b in zip(mine, theirs))
        params_same = rebuilt["fits"][0][0].named_steps[train.STEP_MODEL].get_params()
        applied = {axis: params_same[axis] for axis in TUNABLE} == reading["best_params"]["params"]
        return (
            drift == 0.0 and applied,
            f"rebuilt from the file reproduces {len(mine)} folds with max drift {drift:.1e}; "
            f"parameters applied: {applied}",
        )

    compare("the record alone rebuilds the tuned model bit-identically", rebuild_from_record)

    def record_matches_run() -> tuple[bool, str]:
        best = record["best_params"]
        return (
            best["params"] == chosen
            and record["search"]["n_iter"] == N_ITER
            and record["search"]["scoring"] == SCORING
            and record["search"]["seed"] == SEED
            and abs(record["arms"]["tuned_untreated"]["summary"]["pr_auc"]["mean"]
                    - arms["tuned_untreated"]["summary"]["pr_auc"]["mean"]) <= 1e-12,
            f"best params, budget, metric and seed on disk match the run; "
            f"tuned PR-AUC on disk {record['arms']['tuned_untreated']['summary']['pr_auc']['mean']:.6f}",
        )

    compare("the written record matches the run that produced it", record_matches_run)

    return checks


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(variant: str = "full") -> dict:
    """Search, evaluate three arms through the ordinary loop, check them, write the record."""
    config.ensure_dirs()
    started = time.perf_counter()

    frame = features.load_engineered("train", variant)
    y = encode_target(frame)
    columns = select.shipped_features()
    folds = train.materialise_folds(frame, y)

    search, search_seconds = run_search(frame, y, folds, columns)
    chosen = best_params(search)

    arms: dict[str, dict] = {}
    arms["untuned_untreated"] = evaluate(
        "untuned_untreated", registered_params(), TREATMENT,
        "T28's shipped settings with no imbalance weight - the model to beat (T29's `none` arm)",
        frame, y, folds, columns,
    )
    arms["tuned_untreated"] = evaluate(
        "tuned_untreated", chosen, TREATMENT,
        "the search's winner, refitted through the ordinary loop; the configuration T31 persists",
        frame, y, folds, columns,
    )
    arms["tuned_weighted"] = evaluate(
        "tuned_weighted", chosen, train.SCALE_POS_WEIGHT,
        "the same tuned settings with T29's weight restored, to test whether T29's finding survives tuning",
        frame, y, folds, columns,
    )

    scores = np.array(search.cv_results_["mean_test_score"], dtype=float)
    best_score = float(scores[search.best_index_])
    best_sd = fold_sd(search, search.best_index_)
    # The fold-to-fold sd is **not** the standard error of the mean score: on five folds they differ by
    # a factor of 2.24, so a "within one SE" count that used the sd would be claiming a band more than
    # twice as wide as the one it names. Both are recorded, and both counts are reported.
    best_se = best_sd / np.sqrt(train.N_FOLDS)

    payload = {
        "task": "T30",
        "variant": variant,
        "training_rows": int(len(frame)),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "features": list(columns),
        "folds": train.N_FOLDS,
        "fold_seed": train.FOLD_SEED,
        "sd_ddof": train.SD_DDOF,
        "search": {
            "method": "RandomizedSearchCV",
            "space": {axis: list(values) for axis, values in SEARCH_SPACE.items()},
            "space_size": int(np.prod([len(values) for values in SEARCH_SPACE.values()])),
            "n_iter": N_ITER,
            "scoring": SCORING,
            "seed": SEED,
            "search_n_jobs": SEARCH_JOBS,
            "estimator_n_jobs": 1,
            "cv": f"the {train.N_FOLDS} materialised folds from src.train.materialise_folds "
                  f"(StratifiedKFold, shuffle=True, random_state={train.FOLD_SEED})",
            "treatment": f"scale_pos_weight={TREATMENT} (fixed; T29, D49)",
            "seconds": search_seconds,
            # Kept as a diagnostic, not as a result: the winner is chosen on the folds it is scored on.
            "best_score_optimistic": best_score,
            "best_score_sd": best_sd,
            "candidate_scores": {
                "min": float(scores.min()),
                "median": float(np.median(scores)),
                "max": best_score,
                "all": [float(value) for value in scores],
                "fold_sd_of_best": best_sd,
                "fold_sd_ddof": train.SD_DDOF,
                "sklearn_std_test_score_ddof0": float(
                    search.cv_results_["std_test_score"][search.best_index_]),
                "standard_error_of_best": float(best_se),
                "within_one_se_of_best": int((scores >= best_score - best_se).sum()),
                "within_one_fold_sd_of_best": int((scores >= best_score - best_sd).sum()),
            },
            "top_candidates": top_candidates(search),
        },
        "best_params": {
            "params": chosen,
            "note": "estimator parameters with the pipeline prefix stripped; T31 rebuilds the model from "
                    "these through src.tune.make_estimator, not from prose",
        },
        "arms": {
            name: {
                "note": arm["note"],
                "summary": arm["summary"],
                "params": {
                    axis: value
                    for axis, value in arm["fits"][0][0].named_steps[train.STEP_MODEL].get_params().items()
                    if axis in TUNABLE
                },
                "scale_pos_weight": float(
                    arm["fits"][0][0].named_steps[train.STEP_MODEL].get_params()["scale_pos_weight"]
                ),
                "folds": [
                    {key: value for key, value in fold.items() if key not in ("scores", "truth")}
                    for fold in arm["folds"]
                ],
            }
            for name, arm in arms.items()
        },
        "library": {"xgboost": xgboost.__version__, "scikit-learn": sklearn.__version__},
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }

    # Written before the last check and rewritten after, as in `train.report`: a failing check must not
    # cost the run its numbers, and "the file says what the run measured" can only be tested on a file.
    RECORD_PATH.write_text(json.dumps(payload, indent=2, sort_keys=False))
    checks = validate(frame, y, folds, search, arms, columns, json.loads(RECORD_PATH.read_text()))
    payload["checks"] = checks
    payload["checks_passed"] = sum(1 for check in checks if check["passed"])
    payload["checks_total"] = len(checks)
    payload["checks_ok"] = payload["checks_passed"] == payload["checks_total"]
    RECORD_PATH.write_text(json.dumps(payload, indent=2, sort_keys=False))

    return {**payload, "arms_live": arms, "search_live": search}


def print_report(result: dict) -> None:
    print("DepositIQ T30 - hyperparameter tuning")
    print("=" * 112)
    print(f"rows     : {result['training_rows']:,} ({result['positives']:,} positives, "
          f"{result['base_rate']:.4%}) x {len(result['features'])} features")
    print(f"folds    : {result['folds']} materialised splits, seed {result['fold_seed']}; "
          f"sd over folds uses ddof={result['sd_ddof']}")
    search = result["search"]
    print(f"search   : {search['method']}, {search['n_iter']} of {search['space_size']:,} points, "
          f"scoring {search['scoring']!r}, seed {search['seed']}, {search['seconds']} s")
    scores = search["candidate_scores"]
    print(f"candidates: min {scores['min']:.4f}, median {scores['median']:.4f}, max {scores['max']:.4f}")
    print(f"            {scores['within_one_se_of_best']} of {search['n_iter']} within one standard error "
          f"({scores['standard_error_of_best']:.4f}) of the winner; "
          f"{scores['within_one_fold_sd_of_best']} within one fold-sd "
          f"({scores['fold_sd_of_best']:.4f})")
    print()
    print("BEST PARAMETERS (estimator settings, prefix stripped)")
    print("  " + ", ".join(f"{axis}={value}" for axis, value in result["best_params"]["params"].items()))
    print()
    print("TOP 5 OF THE SEARCH (mean_test_score = PR-AUC on the same five folds, so optimistic; "
          f"sd is ddof={train.SD_DDOF}, recomputed, not scikit-learn's ddof=0)")
    for rank, candidate in enumerate(search["top_candidates"], start=1):
        params = ", ".join(f"{axis}={value}" for axis, value in candidate["params"].items())
        print(f"  {rank}. {candidate['mean_test_score']:.4f} +/- {candidate['fold_sd']:.4f}  {params}")
    print()
    header = (f"{'arm':<20}{'PR-AUC':>18}{'ROC-AUC':>18}{'Brier':>10}{'mean p':>10}"
              f"{'acc @ 0.5':>12}{'weight':>9}")
    print(header)
    print("-" * len(header))
    for name, arm in result["arms_live"].items():
        s = arm["summary"]
        print(f"{name:<20}"
              f"{s['pr_auc']['mean']:>12.4f} +/- {s['pr_auc']['sd']:.4f}"
              f"{s['roc_auc']['mean']:>12.4f} +/- {s['roc_auc']['sd']:.4f}"
              f"{s['brier']['mean']:>10.4f}{s['mean_pred']['mean']:>10.4f}"
              f"{s['accuracy_at_half']['mean']:>11.4%}{arm['fits'][0][0].named_steps[train.STEP_MODEL].get_params()['scale_pos_weight']:>9.3f}")
    base = result["base_rate"]
    print()
    print(f"PR-AUC relative to the {base:.4f} base rate: " + ", ".join(
        f"{name} {arm['summary']['pr_auc']['mean'] / base:.2f}x"
        for name, arm in result["arms_live"].items()))
    print()
    print("DELTA against the untuned configuration, paired per fold")
    reference = result["arms_live"]["untuned_untreated"]["folds"]
    for name, arm in result["arms_live"].items():
        if name == "untuned_untreated":
            continue
        for metric in ("pr_auc", "roc_auc", "brier"):
            deltas = np.array([f[metric] for f in arm["folds"]]) - np.array([f[metric] for f in reference])
            # Brier is an error, so a **negative** delta is the improvement. Counting `> 0` for it would
            # report the weighted arm's 5-of-5 regression as 5-of-5 progress.
            improving = int((deltas < 0).sum()) if metric == "brier" else int((deltas > 0).sum())
            print(f"  {name:<17} {metric:<8} {deltas.mean():+.5f} +/- {deltas.std(ddof=train.SD_DDOF):.5f}, "
                  f"{improving}/{train.N_FOLDS} folds improve")
    print("  (Brier is an error: a negative delta is the improvement)")
    print()
    print(f"total    : {result['elapsed_seconds']} s -> {RECORD_PATH.name}")
    print()
    print(f"T30 checks: {result['checks_passed']}/{result['checks_total']} passed")
    for check in result["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        detail = f"  ({check['detail']})" if check["detail"] else ""
        print(f"  [{marker}] {check['check']}{detail}")
    print()


if __name__ == "__main__":
    print("DepositIQ Phase 6 - hyperparameter tuning")
    print()
    outcome = report()
    print_report(outcome)
    print("=" * 112)
    print(f"result : {'PASS - one search, one loop, and a tuned model that beats the one it replaces' if outcome['checks_ok'] else 'FAIL'}")
    print("Next   : T31 persists the selected model and documents why it was chosen.")
    sys.exit(0 if outcome["checks_ok"] else 1)
