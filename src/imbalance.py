"""
DepositIQ - imbalance handling: weights, nothing, or SMOTE (T29).

Why this file exists
--------------------
Every model in `src/train.py` handles the 11.70% positive rate with **weights** - `class_weight`,
`balanced_subsample`, `scale_pos_weight` - and none of them was tested against the alternative the task
asks about. This module runs that test: **one estimator, one preprocessor, one fold split**, and the
imbalance treatment toggled. Anything that differs between the three rows below other than the treatment
would be a second difference, and a comparison with two differences is two experiments.

The estimator is XGBoost (D08, and the best model in `models/cv_results.json`), because the question is
about the *treatment* and changing the estimator at the same time would answer a different one.

The three arms
--------------
| arm | treatment | what it does |
|---|---|---|
| `weights` | `scale_pos_weight=7.548334` | reweights the loss; the shipped configuration |
| `none` | `scale_pos_weight=1` | no treatment at all - the control |
| `smote` | `SMOTE(random_state=42)` inside the pipeline | synthesises minority rows in the **fitted fold only** |

`none` is not padding. Without it, a difference between `weights` and `smote` cannot be attributed to
either one - it could be the difference between *treating* and *not treating*, which are three arms, not
two.

Where the resampling is allowed to happen - and how that is proved
------------------------------------------------------------------
SMOTE sits **between** the preprocessor and the estimator in an `imblearn.pipeline.Pipeline`, so it is
fitted on the fold's training rows and applied to nothing else. The alternatives are all wrong in ways that
do not announce themselves:

* resampling the frame before `StratifiedKFold` would put synthetic rows in the validation half, and the
  scores would be inflated by construction;
* resampling after the split but outside the pipeline would still let a *different fold's* synthetic rows
  enter the fit;
* resampling before the preprocessor would fit the scaler and the one-hot vocabulary on synthetic rows,
  which changes the matrix every other model was scored on.

The module checks the arithmetic rather than the intention: on the fold-1 training rows, SMOTE must add
exactly `negatives - positives` synthetic rows, the fitted preprocessor's statistics must still equal the
**fold's own** training means, and every arm must score exactly the held-out fold's rows.

Why F1 is not read at 0.5
-------------------------
SMOTE changes the calibration by construction, and so do the weights (D45: the models score the average
client at 0.34-0.42 against a 0.1170 base rate). An F1 at a fixed cut would then decide the comparison by
construction rather than by signal - the arm whose probabilities happen to land nearest 0.5 would win.
So each fold's threshold is chosen **without that fold**: the F1-maximising cut is computed on the other
four folds' out-of-fold predictions and applied to the held-out one. That is a leave-one-fold-out
selection, it costs no extra fits, and it is checked by recomputation below. F1 at 0.5 is still reported,
next to it, as the number that this reasoning rejects.

How to run
----------
From the repository root, inside the virtual environment:

    python -m src.imbalance    # ~2 minutes: 3 arms x 5 folds, then the reproducibility refits

Writes `models/imbalance_results.json`; `docs/06_imbalance.md` quotes only numbers from that file.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.pipeline import Pipeline

from src import config, features, preprocess, select, train
from src.split import encode_target

N_FOLDS: int = train.N_FOLDS
SD_DDOF: int = train.SD_DDOF
RECORD_PATH: Path = config.MODELS_DIR / "imbalance_results.json"

STEP_SMOTE: str = "smote"

# F1 score reported at the conventional 0.5 cut, for contrast only - see the docstring.
CONVENTIONAL_CUT: float = 0.5

ARMS: dict[str, dict] = {
    "weights": {
        "treatment": "scale_pos_weight=7.548334",
        "note": "the shipped configuration: reweight the positive class in the loss",
        "weighted": True,
        "smote": False,
    },
    "none": {
        "treatment": "scale_pos_weight=1",
        "note": "no treatment at all: the control that makes the other two interpretable",
        "weighted": False,
        "smote": False,
    },
    "smote": {
        "treatment": "SMOTE(random_state=42) inside the pipeline",
        "note": "synthetic minority rows, synthesised from the fitted fold's training rows only",
        "weighted": False,
        "smote": True,
    },
}


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
def build_pipeline(arm: str, columns: list[str] | None = None) -> Pipeline:
    """
    One estimator, one preprocessor, one treatment toggled.

    The estimator is taken from `src.train`'s registered XGBoost factory and only `scale_pos_weight` is
    changed, so the arms cannot drift in depth, rounds or learning rate. The SMOTE arm gets an
    `imblearn` pipeline because a scikit-learn `Pipeline` refuses a resampler as a step; the other two use
    `train.make_pipeline`, so the treatment really is the only difference in the steps list.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}. Valid options are: {', '.join(ARMS)}.")
    if columns is None:
        columns = select.shipped_features()

    estimator = train.ESTIMATORS["xgboost_gbdt"]["make"]()
    estimator.set_params(
        scale_pos_weight=train.SCALE_POS_WEIGHT if ARMS[arm]["weighted"] else 1.0
    )

    if not ARMS[arm]["smote"]:
        return train.make_pipeline(estimator, columns)

    return ImbPipeline(
        steps=[
            (train.STEP_PREPROCESS, preprocess.build_preprocessor(columns)),
            (STEP_SMOTE, SMOTE(random_state=config.RANDOM_STATE)),
            (train.STEP_MODEL, estimator),
        ]
    )


def choose_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """
    The F1-maximising cut on the rows it is given.

    `precision_recall_curve` returns thresholds one shorter than precision/recall, so the last point -
    which is the trivial "predict nothing" corner - is dropped from the search rather than allowed to win
    on a zero denominator.
    """
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return float(thresholds[int(np.argmax(f1[:-1]))])


def run_arm(
    arm: str,
    frame: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    columns: list[str],
) -> dict:
    """Fit one arm once per fold and keep the out-of-fold scores, which the thresholds need."""
    records: list[dict] = []
    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        pipeline = build_pipeline(arm, columns)
        pipeline.fit(frame.iloc[train_idx], y.iloc[train_idx])
        scores = pipeline.predict_proba(frame.iloc[valid_idx])[:, 1]
        records.append({
            "fold": fold,
            "scores": scores,
            "truth": y.iloc[valid_idx].to_numpy(),
            "train_rows": int(len(train_idx)),
            "train_positives": int(y.iloc[train_idx].sum()),
            "valid_rows": int(len(valid_idx)),
            **train.fold_metrics(y.iloc[valid_idx].to_numpy(), scores),
        })
    return {"arm": arm, **ARMS[arm], "folds": records}


def apply_nested_thresholds(arm_result: dict) -> dict:
    """
    Give each fold the threshold chosen from the **other** folds' out-of-fold predictions.

    This is what keeps an F1 comparable across arms whose probabilities sit on different scales: the cut is
    selected on data the fold being scored never contributed to, so it cannot be fitted to the fold's own
    labels. Recorded per fold so the choice can be audited rather than trusted.
    """
    folds = arm_result["folds"]
    for index, fold in enumerate(folds):
        others_scores = np.concatenate([f["scores"] for j, f in enumerate(folds) if j != index])
        others_truth = np.concatenate([f["truth"] for j, f in enumerate(folds) if j != index])
        threshold = choose_threshold(others_scores, others_truth)
        predicted = (fold["scores"] >= threshold).astype(int)
        fold["threshold"] = threshold
        fold["f1_nested"] = float(f1_score(fold["truth"], predicted, zero_division=0))
        fold["f1_at_half"] = float(
            f1_score(fold["truth"], (fold["scores"] >= CONVENTIONAL_CUT).astype(int), zero_division=0)
        )
    return arm_result


def summarise(arm_result: dict) -> dict:
    metrics = ("roc_auc", "pr_auc", "brier", "mean_pred", "f1_nested", "f1_at_half")
    summary = {}
    for metric in metrics:
        values = np.array([fold[metric] for fold in arm_result["folds"]], dtype=float)
        summary[metric] = {"mean": float(values.mean()), "sd": float(values.std(ddof=SD_DDOF))}
    thresholds = [fold["threshold"] for fold in arm_result["folds"]]
    summary["threshold"] = {"mean": float(np.mean(thresholds)), "sd": float(np.std(thresholds, ddof=SD_DDOF))}
    return summary


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def validate(
    frame: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    results: dict[str, dict],
    columns: list[str],
) -> list[dict]:
    """Same contract as every other phase: an unexecutable check is FAILED, never raised (D25)."""
    checks: list[dict] = []

    def compare(name: str, evaluate) -> None:
        try:
            passed, detail = evaluate()
        except Exception as error:  # noqa: BLE001
            passed, detail = False, f"check could not run: {type(error).__name__}: {error}"
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    # --- the arms were compared on the same thing ------------------------------------------------
    def same_folds() -> tuple[bool, str]:
        again = train.materialise_folds(frame, y)
        same = len(again) == len(folds) and all(
            np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(again, folds)
        )
        per_fold = [len(arm["folds"]) for arm in results.values()]
        # Deliberately **not** `len(distinct (rows, positives)) == 5`: five stratified folds produce only
        # three distinct combinations (7234/846, 7234/847, 7233/846), so that assertion was arithmetic on
        # the wrong quantity and failed on a correct run. Agreement *per fold index* is the property that
        # matters, and the next check tests it directly.
        return (
            same and set(per_fold) == {N_FOLDS},
            f"{len(folds)} folds replayed into all {len(results)} arms, each with "
            f"{sorted(set(per_fold))[0]} folds",
        )

    compare("every arm ran the same five folds", same_folds)

    def same_validation_rows() -> tuple[bool, str]:
        by_fold = {}
        for arm, result in results.items():
            for fold in result["folds"]:
                by_fold.setdefault(fold["fold"], set()).add(
                    (fold["valid_rows"], fold["positives"], float(fold["truth"].sum()))
                )
        bad = {k: v for k, v in by_fold.items() if len(v) != 1}
        return not bad, f"identical validation rows in every arm: {not bad}; disagreements: {bad or 'none'}"

    compare("every arm scored the identical validation rows", same_validation_rows)

    def scores_are_only_the_held_out_rows() -> tuple[bool, str]:
        """SMOTE must not add a single row at predict time - the scores are the fold, nothing more."""
        wrong = [
            (arm, fold["fold"], len(fold["scores"]), fold["valid_rows"])
            for arm, result in results.items()
            for fold in result["folds"]
            if len(fold["scores"]) != fold["valid_rows"]
        ]
        return not wrong, f"every fold returns {results['none']['folds'][0]['valid_rows']} scores; wrong: {wrong or 'none'}"

    compare("no arm added rows at prediction time", scores_are_only_the_held_out_rows)

    # --- SMOTE inflated the training side by exactly the arithmetic amount ------------------------
    def smote_row_identity() -> tuple[bool, str]:
        """
        On the fold-1 training rows, SMOTE must add exactly `negatives - positives` synthetic rows.

        Not a stylistic check: it is the difference between "the resampler ran" and "the resampler ran on
        the right rows". Re-fitted here through the same preprocessor the arm used, so the matrix it sees
        is the matrix the arm saw.
        """
        train_idx, _ = folds[0]
        matrix = preprocess.build_preprocessor(columns).fit_transform(frame.iloc[train_idx])
        truth = y.iloc[train_idx].to_numpy()
        negatives = int((truth == 0).sum())
        positives = int((truth == 1).sum())
        resampled_x, resampled_y = SMOTE(random_state=config.RANDOM_STATE).fit_resample(matrix, truth)
        added = len(resampled_y) - len(truth)
        return (
            len(resampled_x) == 2 * negatives
            and added == negatives - positives
            and int(resampled_y.sum()) == negatives,
            f"{len(truth):,} rows ({positives:,} pos / {negatives:,} neg) -> {len(resampled_y):,} "
            f"({int(resampled_y.sum()):,} pos), added {added:,} = {negatives:,} - {positives:,}",
        )

    compare("SMOTE adds exactly the rows balancing requires, on the fold's training rows", smote_row_identity)

    def preprocessing_still_fold_local() -> tuple[bool, str]:
        """
        The SMOTE arm's scaler must match the **fold's own** training means, not the resampled matrix's.

        If SMOTE ran before the preprocessor, the scaler would have been fitted on synthetic rows and its
        means would drift away from the fold's - which is the one way resampling can corrupt the matrix
        silently, because the shape would still look right.
        """
        worst_fold, worst_resampled = 0.0, 1e9
        pipeline = build_pipeline("smote", columns)
        train_idx, _ = folds[0]
        pipeline.fit(frame.iloc[train_idx], y.iloc[train_idx])
        scaler = pipeline.named_steps[train.STEP_PREPROCESS].named_transformers_["scaled"]
        own = frame.iloc[train_idx][list(preprocess.NUMERIC_SCALED)].to_numpy(dtype=float)
        worst_fold = float(np.max(np.abs(scaler.mean_ - own.mean(axis=0))))

        matrix = preprocess.build_preprocessor(columns).fit_transform(frame.iloc[train_idx])
        resampled, _ = SMOTE(random_state=config.RANDOM_STATE).fit_resample(matrix, y.iloc[train_idx])
        resampled_means = np.asarray(resampled)[:, [0, 1, 2]].mean(axis=0)
        worst_resampled = float(np.max(np.abs(scaler.mean_ - resampled_means)))
        return (
            worst_fold < 1e-12 and worst_resampled > 1e-6,
            f"scaler mean matches the fold's training rows to {worst_fold:.1e}; the resampled matrix's "
            f"means are {worst_resampled:.3e} away, which is what a leak would have produced",
        )

    compare("the preprocessor was fitted on real training rows, not on synthetic ones",
            preprocessing_still_fold_local)

    def steps_differ_only_in_the_treatment() -> tuple[bool, str]:
        signatures = {}
        for arm in ARMS:
            pipeline = build_pipeline(arm, columns)
            names = [name for name, _ in pipeline.steps]
            estimator = pipeline.named_steps[train.STEP_MODEL]
            preprocessor_signature = tuple(
                (step_name, type(step).__name__, tuple(step_columns))
                for step_name, step, step_columns in pipeline.named_steps[train.STEP_PREPROCESS].transformers
            )
            signatures[arm] = (
                names,
                preprocessor_signature,
                float(estimator.get_params()["scale_pos_weight"]),
                type(pipeline).__name__,
            )
        bodies = {sig[1] for sig in signatures.values()}
        weights = {arm: sig[2] for arm, sig in signatures.items()}
        return (
            len(bodies) == 1
            and weights == {"weights": train.SCALE_POS_WEIGHT, "none": 1.0, "smote": 1.0}
            and signatures["smote"][0] == [train.STEP_PREPROCESS, STEP_SMOTE, train.STEP_MODEL]
            and signatures["weights"][0] == [train.STEP_PREPROCESS, train.STEP_MODEL],
            f"one preprocessing definition across 3 arms; scale_pos_weight {weights}; "
            f"steps {signatures['smote'][0]} vs {signatures['weights'][0]}",
        )

    compare("the arms differ in the treatment and in nothing else", steps_differ_only_in_the_treatment)

    # --- the thresholds were chosen without the fold they score ----------------------------------
    def thresholds_are_leave_one_fold_out() -> tuple[bool, str]:
        drift = []
        for arm, result in results.items():
            folds_here = result["folds"]
            for index, fold in enumerate(folds_here):
                others_scores = np.concatenate(
                    [f["scores"] for j, f in enumerate(folds_here) if j != index])
                others_truth = np.concatenate([f["truth"] for j, f in enumerate(folds_here) if j != index])
                if abs(choose_threshold(others_scores, others_truth) - fold["threshold"]) > 1e-12:
                    drift.append((arm, fold["fold"]))
        return not drift, f"re-derived from the other four folds for all 15 folds; drift: {drift or 'none'}"

    compare("each fold's F1 threshold came from the other folds only", thresholds_are_leave_one_fold_out)

    # --- reproducibility -------------------------------------------------------------------------
    def reproducible() -> tuple[bool, str]:
        train_idx, valid_idx = folds[0]
        drift = []
        for arm, result in results.items():
            pipeline = build_pipeline(arm, columns)
            pipeline.fit(frame.iloc[train_idx], y.iloc[train_idx])
            again = pipeline.predict_proba(frame.iloc[valid_idx])[:, 1]
            if not np.array_equal(again, result["folds"][0]["scores"]):
                drift.append((arm, float(np.max(np.abs(again - result["folds"][0]["scores"])))))
        return not drift, f"fold 1 of all 3 arms reproduces bit-identically; drift: {drift or 'none'}"

    compare("re-running fold 1 reproduces every arm exactly", reproducible)

    def no_degenerate_result() -> tuple[bool, str]:
        bad = [
            (arm, fold["fold"], fold["pr_auc"])
            for arm, result in results.items()
            for fold in result["folds"]
            if not 0.0 < fold["pr_auc"] < 1.0 or not 0.0 <= fold["mean_pred"] <= 1.0
        ]
        return not bad, f"every PR-AUC strictly inside (0, 1) and every mean p inside [0, 1]; bad: {bad or 'none'}"

    compare("no arm produced a degenerate fold", no_degenerate_result)

    checks.append({
        "check": "the shipped feature set is T25's frozen decision",
        "passed": columns == select.shipped_features(),
        "detail": f"{len(columns)} features",
    })
    return checks


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(variant: str = "full") -> dict:
    """Run the three arms on one fold split, check them, and write the record."""
    config.ensure_dirs()
    started = time.perf_counter()

    frame = features.load_engineered("train", variant)
    y = encode_target(frame)
    columns = select.shipped_features()
    folds = train.materialise_folds(frame, y)

    results = {}
    for arm in ARMS:
        arm_started = time.perf_counter()
        results[arm] = apply_nested_thresholds(run_arm(arm, frame, y, folds, columns))
        results[arm]["seconds"] = round(time.perf_counter() - arm_started, 1)
        results[arm]["summary"] = summarise(results[arm])

    checks = validate(frame, y, folds, results, columns)

    payload = {
        "task": "T29",
        "variant": variant,
        "estimator": "xgboost_gbdt (src.train.ESTIMATORS)",
        "training_rows": int(len(frame)),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "features": list(columns),
        "folds": N_FOLDS,
        "fold_seed": train.FOLD_SEED,
        "sd_ddof": SD_DDOF,
        "threshold_rule": "leave-one-fold-out: F1-maximising cut on the other four folds' out-of-fold "
                          "predictions, applied to the held-out fold",
        "arms": {
            arm: {
                "treatment": result["treatment"],
                "note": result["note"],
                "seconds": result["seconds"],
                "summary": result["summary"],
                "folds": [
                    {key: value for key, value in fold.items() if key not in ("scores", "truth")}
                    for fold in result["folds"]
                ],
            }
            for arm, result in results.items()
        },
        "checks": checks,
        "checks_passed": sum(1 for c in checks if c["passed"]),
        "checks_total": len(checks),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    payload["checks_ok"] = payload["checks_passed"] == payload["checks_total"]
    RECORD_PATH.write_text(json.dumps(payload, indent=2, sort_keys=False))

    return {**payload, "results": results}


def print_report(result: dict) -> None:
    print("DepositIQ T29 - imbalance handling")
    print("=" * 112)
    print(f"estimator  : {result['estimator']}")
    print(f"rows       : {result['training_rows']:,} ({result['positives']:,} positives, "
          f"{result['base_rate']:.4%}) x {len(result['features'])} features")
    print(f"folds      : {result['folds']} stratified, seed {result['fold_seed']}; "
          f"sd over folds uses ddof={result['sd_ddof']}")
    print("threshold  : leave-one-fold-out - each fold is scored at the F1-maximising cut of the OTHER")
    print("             four folds, so no arm is judged at a cut fitted to the fold it is scored on")
    print()
    header = (f"{'arm':<10}{'PR-AUC':>18}{'ROC-AUC':>18}{'F1 (nested)':>18}"
              f"{'F1 @ 0.5':>12}{'cut':>9}{'mean p':>10}{'Brier':>10}")
    print(header)
    print("-" * len(header))
    for arm, block in result["results"].items():
        s = block["summary"]
        print(f"{arm:<10}"
              f"{s['pr_auc']['mean']:>12.4f} +/- {s['pr_auc']['sd']:.4f}"
              f"{s['roc_auc']['mean']:>12.4f} +/- {s['roc_auc']['sd']:.4f}"
              f"{s['f1_nested']['mean']:>12.4f} +/- {s['f1_nested']['sd']:.4f}"
              f"{s['f1_at_half']['mean']:>12.4f}"
              f"{s['threshold']['mean']:>9.4f}"
              f"{s['mean_pred']['mean']:>10.4f}"
              f"{s['brier']['mean']:>10.4f}")
        print(f"  {block['treatment']}")
        print(f"  {block['note']}")
    base = result["base_rate"]
    print()
    print(f"PR-AUC relative to the {base:.4f} base rate: " + ", ".join(
        f"{arm} {block['summary']['pr_auc']['mean'] / base:.2f}x"
        for arm, block in result["results"].items()))
    print()
    print("DELTA against the shipped treatment (`weights`), paired per fold")
    reference = result["results"]["weights"]["folds"]
    for arm, block in result["results"].items():
        if arm == "weights":
            continue
        for metric in ("pr_auc", "roc_auc", "f1_nested"):
            deltas = np.array([f[metric] for f in block["folds"]]) - np.array(
                [f[metric] for f in reference])
            print(f"  {arm:<8} {metric:<9} {deltas.mean():+.4f} +/- {deltas.std(ddof=SD_DDOF):.4f}, "
                  f"{int((deltas > 0).sum())}/5 folds improve")
    print()
    print(f"time per arm: " + ", ".join(f"{arm} {block['seconds']}s" for arm, block in result["results"].items()))
    print(f"total       : {result['elapsed_seconds']} s -> {RECORD_PATH.name}")
    print()
    print(f"T29 checks: {result['checks_passed']}/{result['checks_total']} passed")
    for check in result["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        detail = f"  ({check['detail']})" if check["detail"] else ""
        print(f"  [{marker}] {check['check']}{detail}")
    print()


if __name__ == "__main__":
    print("DepositIQ Phase 6 - imbalance handling")
    print()
    outcome = report()
    print_report(outcome)
    print("=" * 112)
    print(f"result : {'PASS - three arms compared on one fold split, resampling confined to the folds' if outcome['checks_ok'] else 'FAIL'}")
    print("Next   : T30 tunes the winning configuration on all five folds.")
    sys.exit(0 if outcome["checks_ok"] else 1)
