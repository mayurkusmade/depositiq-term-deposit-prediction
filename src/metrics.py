"""
DepositIQ - the cross-validated metric table (T32).

Why this file exists
--------------------
Everything before this reported **threshold-free** metrics (ROC-AUC, PR-AUC) plus calibration diagnostics,
because the models were uncalibrated by construction (D45). The task now asks for precision, recall and F1
as well, and those three only exist **at a cut**. This module produces that table without letting the cut
decide the comparison.

The threshold rule, and why 0.5 is not it
----------------------------------------
T29 measured what a fixed 0.5 cut does: the weighted model looked **11.9 points better** on F1 at 0.5 and
**lost** at a properly chosen cut, because weighting moves the mean predicted probability from 0.117 to
0.39 and 0.5 happens to land near the weighted model's own optimum by accident. A table read at 0.5 would
therefore rank models by where their calibration sits, not by what they predict.

So each fold is scored at a cut chosen **without that fold** (`imbalance.choose_threshold` on the other
four folds' out-of-fold scores). That is a leave-one-fold-out selection: it costs no extra fits, no fold is
judged at a threshold fitted to itself, and the rule is checked by recomputation below.

Three cuts are reported, and they mean different things:

| cut | what it is | why it is in the table |
|---|---|---|
| **nested** | F1-maximising cut from the other four folds | the honest operating point for a model whose cut must be chosen from data |
| **break-even 7.50%** | D17's cost model: a call pays for itself above a 7.50% subscription rate | the business-relevant point, and the only one that is *not* fitted to the data |
| **0.5** | the conventional cut | reported **as the number this reasoning rejects**, never as the headline |

At 0.075 the shipped model flags **more** rows than the base rate (its mean prediction is 0.1166), so the
break-even cut is a genuine decision boundary rather than a truncation of a badly placed score.

What is not in this table
-------------------------
* **No held-out rows.** The test half is never loaded; T33 scores the artifact once, out of sample, and
  keeping the two files separate is what makes the eventual gap between them computable.
* **No recalibration.** The shipped model's probabilities already land on the base rate (mean p 0.1166 vs
  0.1170), so the break-even cut can be applied as a probability; T34 tests that formally and owns the
  final operating point.

How to run
----------
From the repository root, inside the virtual environment:

    python -m src.metrics    # ~4 minutes: 4 models x 5 folds, the forest being the slow one

Writes `reports/metrics_cv.csv`; `docs/06_modeling.md` §10 quotes only numbers from that file.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src import config, features, select, train, tune
from src.imbalance import choose_threshold
from src.split import encode_target

CV_PATH: Path = config.REPORTS_DIR / "metrics_cv.csv"
PARAMS_PATH: Path = config.MODELS_DIR / "best_params.json"
RECORD_PATH: Path = config.MODELS_DIR / "metrics_cv.json"

# D17's operating point: cost per call over expected value of a subscription.
BREAK_EVEN: float = 0.075
CONVENTIONAL: float = 0.5

SD_DDOF: int = train.SD_DDOF


def builds() -> dict[str, dict]:
    """
    The four models in the table, each with the factory that produced its published CV numbers.

    The shipped model is rebuilt from `models/best_params.json` through `tune.make_estimator` - the same
    path T31 persisted - so this table cannot describe a model that differs from the artifact.
    """
    chosen = json.loads(PARAMS_PATH.read_text())["best_params"]["params"]
    return {
        "dummy_prior": {
            "make": train.ESTIMATORS["dummy_prior"]["make"],
            "role": "floor",
            "note": "predicts the training fold's positive rate (D28's floor)",
        },
        "logreg_balanced": {
            "make": train.ESTIMATORS["logreg_balanced"]["make"],
            "role": "baseline",
            "note": "the task's required baseline, class_weight='balanced'",
        },
        "random_forest": {
            "make": train.ESTIMATORS["random_forest"]["make"],
            "role": "candidate",
            "note": "300 trees, balanced_subsample (T27)",
        },
        "xgboost_tuned": {
            "make": lambda: tune.make_estimator(chosen),
            "role": "shipped",
            "note": "T30's tuned point, untreated; the configuration T31 persisted",
        },
    }


def at_threshold(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Precision, recall, F1 and the flag counts at one cut, from counts rather than from a label array."""
    flagged = scores >= threshold
    true_positive = int((flagged & (y == 1)).sum())
    false_positive = int((flagged & (y == 0)).sum())
    false_negative = int((~flagged & (y == 1)).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "flagged": int(flagged.sum()),
        "flagged_share": float(flagged.mean()),
    }


def run_model(name: str, spec: dict, frame: pd.DataFrame, y: pd.Series,
              folds: list[tuple[np.ndarray, np.ndarray]], columns: list[str]) -> dict:
    """Fit one model on the five folds, then score every fold at three cuts."""
    started = time.perf_counter()
    run = train.cross_validate_factory(
        name, spec["make"], spec["role"], spec["note"], frame, y, folds, columns
    )
    seconds = round(time.perf_counter() - started, 1)

    truth = [y.iloc[valid].to_numpy() for _, valid in folds]
    scores = [pipeline.predict_proba(frame.iloc[valid])[:, 1] for pipeline, _, valid in run["fits"]]

    records = []
    for index in range(len(folds)):
        outside = np.concatenate([scores[other] for other in range(len(folds)) if other != index])
        outside_y = np.concatenate([truth[other] for other in range(len(folds)) if other != index])
        nested = choose_threshold(outside, outside_y)
        y_here, scores_here = truth[index], scores[index]
        fold = {
            "fold": index + 1,
            "rows": int(len(y_here)),
            "positives": int(y_here.sum()),
            "pr_auc": float(run["folds"][index]["pr_auc"]),
            "roc_auc": float(run["folds"][index]["roc_auc"]),
            **{f"nested_{key}": value for key, value in at_threshold(y_here, scores_here, nested).items()},
            **{f"be_{key}": value for key, value in at_threshold(y_here, scores_here, BREAK_EVEN).items()},
            **{f"half_{key}": value for key, value in at_threshold(y_here, scores_here, CONVENTIONAL).items()},
        }
        records.append(fold)

    return {
        "estimator": name,
        "role": spec["role"],
        "note": spec["note"],
        "seconds": seconds,
        "folds": records,
        "scores": scores,
        "truth": truth,
        **{"summary": summarise(records)},
    }


def summarise(records: list[dict]) -> dict:
    """Mean and **sample** sd over the five folds for every numeric metric (D50: ddof=1 everywhere)."""
    numeric = (
        "pr_auc", "roc_auc",
        "nested_threshold", "nested_precision", "nested_recall", "nested_f1", "nested_flagged",
        "nested_flagged_share",
        "be_precision", "be_recall", "be_f1", "be_flagged", "be_flagged_share",
        "half_f1", "half_precision", "half_recall", "half_flagged", "half_flagged_share",
    )
    return {
        metric: {
            "mean": float(np.mean([record[metric] for record in records])),
            "sd": float(np.std([record[metric] for record in records], ddof=SD_DDOF)),
        }
        for metric in numeric
    }


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def validate(
    frame: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    runs: dict[str, dict],
    table: pd.DataFrame,
    reference: dict,
) -> list[dict]:
    """Same contract as every other phase: a check that cannot run is FAILED, never raised (D25)."""
    checks: list[dict] = []

    def compare(name: str, evaluate) -> None:
        try:
            passed, detail = evaluate()
        except Exception as error:  # noqa: BLE001 - any failure here is a failed check
            passed, detail = False, f"check could not run: {type(error).__name__}: {error}"
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    shipped = runs["xgboost_tuned"]

    # --- every model saw the same rows ----------------------------------------------------
    def same_folds() -> tuple[bool, str]:
        shapes = {
            (record["rows"], record["positives"])
            for run in runs.values() for record in run["folds"]
        }
        sizes = {len(run["folds"]) for run in runs.values()}
        return (
            sizes == {train.N_FOLDS} and len(shapes) == 3,
            f"{len(runs)} models x {sorted(sizes)[0]} folds; {len(shapes)} distinct (rows, positives) "
            f"combinations: {sorted(shapes)} — the same three that five stratified folds can produce",
        )

    compare("every model ran the identical five folds", same_folds)

    def shipped_matches_the_selected_record() -> tuple[bool, str]:
        """The table's shipped row must be T30's tuned arm, or the table describes a different model."""
        reference_folds = reference["arms"]["tuned_untreated"]["folds"]
        worst = max(
            abs(record[metric] - float(reference_folds[index][metric]))
            for index, record in enumerate(shipped["folds"])
            for metric in ("pr_auc", "roc_auc")
        )
        return worst == 0.0, (
            f"5 folds compared against models/{PARAMS_PATH.name} (arms.tuned_untreated): "
            f"max drift {worst:.1e}"
        )

    compare("the shipped row reproduces T30's tuned arm bit-identically",
            shipped_matches_the_selected_record)

    # --- the cut was chosen without the fold it scores ------------------------------------
    def thresholds_are_leave_one_out() -> tuple[bool, str]:
        """
        Recompute every fold's cut from the **other four** folds and compare.

        The second half of the check is the one that matters: it confirms the recomputation would *notice*
        a leak. If the stored cut equalled the all-five-fold cut in every fold, the two procedures would be
        indistinguishable on this data and the first assertion would prove nothing.
        """
        mismatched, leaked = [], []
        for index, record in enumerate(shipped["folds"]):
            outside = np.concatenate([shipped["scores"][other]
                                      for other in range(len(folds)) if other != index])
            outside_y = np.concatenate([shipped["truth"][other]
                                        for other in range(len(folds)) if other != index])
            recomputed = choose_threshold(outside, outside_y)
            if recomputed != record["nested_threshold"]:
                mismatched.append((record["fold"], recomputed, record["nested_threshold"]))
            inside = choose_threshold(np.concatenate(shipped["scores"]), np.concatenate(shipped["truth"]))
            if inside == record["nested_threshold"]:
                leaked.append(record["fold"])
        return (
            not mismatched and not all(fold in leaked for fold in range(1, len(folds) + 1)),
            f"5 cuts re-derived from the other four folds, mismatches: {mismatched or 'none'}; folds where "
            f"the all-folds cut would have coincided: {leaked or 'none'}",
        )

    compare("each fold is scored at a cut computed from the other four", thresholds_are_leave_one_out)

    def nested_cut_is_not_the_conventional_one() -> tuple[bool, str]:
        cuts = [record["nested_threshold"] for record in shipped["folds"]]
        gaps = [abs(record["nested_f1"] - record["half_f1"]) for record in shipped["folds"]]
        return (
            all(cut != CONVENTIONAL for cut in cuts) and min(gaps) > 0,
            f"nested cuts {[round(cut, 4) for cut in cuts]}; F1 at 0.5 differs from F1 at the nested cut "
            f"by {min(gaps):.4f}-{max(gaps):.4f} across folds",
        )

    compare("the reported cut is never 0.5, and 0.5 would give different numbers",
            nested_cut_is_not_the_conventional_one)

    # --- the numbers are the arithmetic they claim to be ----------------------------------
    def f1_is_the_harmonic_mean() -> tuple[bool, str]:
        wrong = []
        for run in runs.values():
            for record in run["folds"]:
                for prefix in ("nested", "be", "half"):
                    precision, recall = record[f"{prefix}_precision"], record[f"{prefix}_recall"]
                    expected = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                    if abs(expected - record[f"{prefix}_f1"]) > 1e-12:
                        wrong.append((run["estimator"], record["fold"], prefix))
        return not wrong, f"3 cuts x 5 folds x {len(runs)} models checked; wrong: {wrong or 'none'}"

    compare("every F1 equals the harmonic mean of its precision and recall", f1_is_the_harmonic_mean)

    def counts_reproduce_the_sklearn_metrics() -> tuple[bool, str]:
        """Re-derive precision, recall and F1 from the stored counts, then against scikit-learn itself."""
        worst = 0.0
        for run in runs.values():
            for index, record in enumerate(run["folds"]):
                truth, scores = run["truth"][index], run["scores"][index]
                for prefix in ("nested", "be"):
                    cut = record[f"{prefix}_threshold"]
                    predicted = (scores >= cut).astype(int)
                    worst = max(
                        worst,
                        abs(precision_score(truth, predicted, zero_division=0) - record[f"{prefix}_precision"]),
                        abs(recall_score(truth, predicted, zero_division=0) - record[f"{prefix}_recall"]),
                        abs(f1_score(truth, predicted, zero_division=0) - record[f"{prefix}_f1"]),
                    )
        return worst <= 1e-12, f"max |from counts - scikit-learn| over every row = {worst:.1e}"

    compare("precision, recall and F1 match scikit-learn on the same predictions",
            counts_reproduce_the_sklearn_metrics)

    def break_even_is_a_real_boundary() -> tuple[bool, str]:
        shares = [record["be_flagged_share"] for record in shipped["folds"]]
        means = [record["positives"] / record["rows"] for record in shipped["folds"]]
        return (
            all(share > rate for share, rate in zip(shares, means))
            and all(record["be_threshold"] == BREAK_EVEN for run in runs.values() for record in run["folds"]),
            f"at {BREAK_EVEN:.2%} the shipped model flags {min(shares):.2%}-{max(shares):.2%} of rows "
            f"against a {min(means):.2%} positive rate, so the cut is below its average score and "
            f"selects rather than truncates",
        )

    compare("the 7.50% break-even is a decision boundary, not a truncation", break_even_is_a_real_boundary)

    def floors_behave() -> tuple[bool, str]:
        dummy = runs["dummy_prior"]["summary"]["nested_f1"]["mean"]
        shipped_f1 = shipped["summary"]["nested_f1"]["mean"]
        return (
            shipped_f1 > dummy and runs["dummy_prior"]["summary"]["roc_auc"]["mean"] == 0.5,
            f"shipped nested F1 {shipped_f1:.4f} against the dummy's {dummy:.4f}; dummy ROC-AUC "
            f"{runs['dummy_prior']['summary']['roc_auc']['mean']:.4f}",
        )

    compare("the shipped model beats the floor at the same threshold rule", floors_behave)

    def summary_is_mean_and_sample_sd() -> tuple[bool, str]:
        worst = 0.0
        for run in runs.values():
            for metric in ("pr_auc", "nested_f1", "be_f1"):
                values = np.array([record[metric] for record in run["folds"]], dtype=float)
                worst = max(
                    worst,
                    abs(values.mean() - run["summary"][metric]["mean"]),
                    abs(values.std(ddof=SD_DDOF) - run["summary"][metric]["sd"]),
                )
        return worst <= 1e-12, f"mean and ddof={SD_DDOF} sd recomputed for every metric; max drift {worst:.1e}"

    compare("the summary rows are the mean and sample sd of the folds", summary_is_mean_and_sample_sd)

    def no_test_rows() -> tuple[bool, str]:
        source = Path(__file__).read_text()
        clean = re.search(r"load_engineered\(\s*['\"]test", source) is None
        return (
            clean and all(record["rows"] < len(frame) for run in runs.values() for record in run["folds"]),
            f"every fold scores a held-out slice ({shipped['folds'][0]['rows']:,} rows) of the training "
            f"half; test-half loads in this module: {'found' if not clean else 'none'}",
        )

    compare("the table is cross-validation only: no test-half rows were scored", no_test_rows)

    def csv_matches_the_run() -> tuple[bool, str]:
        reading = pd.read_csv(CV_PATH)
        same_rows = len(reading) == len(table)
        numeric = [column for column in table.columns
                   if column not in ("model", "statistic") and pd.api.types.is_numeric_dtype(table[column])]
        written = reading[numeric].to_numpy(dtype=float)
        measured = table[numeric].to_numpy(dtype=float)
        # Comparing with `max` over the whole block would return NaN if any cell were missing - and the CSV
        # genuinely cannot represent a missing value from a stub row, so the comparison is masked explicitly
        # rather than silently passing on `nan`.
        missing_agree = np.array_equal(np.isnan(written), np.isnan(measured))
        drift = float(np.nanmax(np.abs(written - measured)))
        return (
            same_rows and missing_agree and drift <= 5e-7 and list(reading["model"]) == list(table["model"]),
            f"{len(reading)} rows on disk, {len(numeric)} numeric columns, max drift from the run "
            f"{drift:.1e} (CSV stores 6 decimals), missing cells agree: {missing_agree}",
        )

    compare("the CSV on disk is the table this run produced", csv_matches_the_run)

    return checks


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_table(runs: dict[str, dict]) -> pd.DataFrame:
    """One row per model per fold, then a `mean` and a `sd` row per model, in a fixed column order."""
    columns = [
        "model", "statistic", "rows", "positives", "pr_auc", "roc_auc",
        "nested_threshold", "nested_precision", "nested_recall", "nested_f1", "nested_flagged",
        "nested_flagged_share", "be_threshold", "be_precision", "be_recall", "be_f1", "be_flagged",
        "be_flagged_share", "half_threshold", "half_precision", "half_recall", "half_f1",
        "half_flagged", "half_flagged_share",
    ]
    rows = []
    for name, run in runs.items():
        for record in run["folds"]:
            rows.append({
                "model": name,
                "statistic": f"fold_{record['fold']}",
                "rows": record["rows"],
                "positives": record["positives"],
                **{key: value for key, value in record.items()
                   if key in columns and key not in ("model", "statistic", "rows", "positives")},
            })
        for statistic in ("mean", "sd"):
            summary = run["summary"]
            rows.append({
                "model": name,
                "statistic": statistic,
                "rows": int(np.mean([record["rows"] for record in run["folds"]])) if statistic == "mean" else "",
                "positives": int(np.mean([record["positives"] for record in run["folds"]])) if statistic == "mean" else "",
                **{key: summary[key][statistic] for key in summary},
            })
    return pd.DataFrame(rows, columns=columns)


def report(variant: str = "full") -> dict:
    """Fit the four models, score every fold at three cuts, write the CSV and the JSON record."""
    config.ensure_dirs()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    frame = features.load_engineered("train", variant)
    y = encode_target(frame)
    columns = select.shipped_features()
    folds = train.materialise_folds(frame, y)
    reference = json.loads(PARAMS_PATH.read_text())

    runs = {name: run_model(name, spec, frame, y, folds, columns)
            for name, spec in builds().items()}

    table = build_table(runs)
    table.to_csv(CV_PATH, index=False, float_format="%.6f")

    payload = {
        "task": "T32",
        "variant": variant,
        "training_rows": int(len(frame)),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "folds": train.N_FOLDS,
        "fold_seed": train.FOLD_SEED,
        "sd_ddof": SD_DDOF,
        "threshold_rule": {
            "nested": "F1-maximising cut chosen on the other four folds' out-of-fold scores "
                      "(imbalance.choose_threshold), applied to the held-out fold",
            "break_even": f"{BREAK_EVEN} - D17's cost model, not fitted to the data",
            "conventional": f"{CONVENTIONAL} - reported as the rejected alternative (T29, D49)",
        },
        "models": {
            name: {
                "role": run["role"],
                "note": run["note"],
                "seconds": run["seconds"],
                "summary": run["summary"],
                "folds": run["folds"],
            }
            for name, run in runs.items()
        },
        "csv": str(CV_PATH.relative_to(config.ROOT)),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }

    checks = validate(frame, y, folds, runs, pd.read_csv(CV_PATH), reference)
    payload["checks"] = checks
    payload["checks_passed"] = sum(1 for check in checks if check["passed"])
    payload["checks_total"] = len(checks)
    payload["checks_ok"] = payload["checks_passed"] == payload["checks_total"]
    RECORD_PATH.write_text(json.dumps(payload, indent=2, sort_keys=False))

    return {**payload, "runs": runs, "table": table}


def print_report(result: dict) -> None:
    print("DepositIQ T32 - cross-validated metric table")
    print("=" * 112)
    print(f"rows     : {result['training_rows']:,} ({result['positives']:,} positives, "
          f"{result['base_rate']:.4%}) x {train.N_FOLDS} stratified folds, seed {result['fold_seed']}; "
          f"sd uses ddof={SD_DDOF}")
    print("cuts     : nested = F1-maximising on the other four folds; break-even = D17's 7.50%; "
          "0.5 = rejected")
    print()
    header = (f"{'model':<18}{'PR-AUC':>17}{'ROC-AUC':>17}{'cut':>9}{'precision':>17}"
              f"{'recall':>17}{'F1':>17}")
    print(header)
    print("-" * len(header))
    for name, run in result["runs"].items():
        s = run["summary"]
        print(f"{name:<18}"
              f"{s['pr_auc']['mean']:>11.4f} +/-{s['pr_auc']['sd']:.4f}"
              f"{s['roc_auc']['mean']:>11.4f} +/-{s['roc_auc']['sd']:.4f}"
              f"{s['nested_threshold']['mean']:>9.4f}"
              f"{s['nested_precision']['mean']:>11.4f} +/-{s['nested_precision']['sd']:.4f}"
              f"{s['nested_recall']['mean']:>11.4f} +/-{s['nested_recall']['sd']:.4f}"
              f"{s['nested_f1']['mean']:>11.4f} +/-{s['nested_f1']['sd']:.4f}")
    print()
    print("AT D17's 7.50% BREAK-EVEN (the business cut, not fitted)")
    header = (f"{'model':<18}{'precision':>17}{'recall':>17}{'F1':>17}{'flagged':>10}")
    print(header)
    print("-" * len(header))
    for name, run in result["runs"].items():
        s = run["summary"]
        print(f"{name:<18}"
              f"{s['be_precision']['mean']:>11.4f} +/-{s['be_precision']['sd']:.4f}"
              f"{s['be_recall']['mean']:>11.4f} +/-{s['be_recall']['sd']:.4f}"
              f"{s['be_f1']['mean']:>11.4f} +/-{s['be_f1']['sd']:.4f}"
              f"{s['be_flagged_share']['mean']:>10.2%}")
    print()
    print("AT 0.5 - THE CUT THIS TABLE REJECTS (T29: a fixed 0.5 ranks models by calibration)")
    for name, run in result["runs"].items():
        s = run["summary"]
        print(f"  {name:<18}F1 {s['half_f1']['mean']:.4f} +/- {s['half_f1']['sd']:.4f}, "
              f"flags {s['half_flagged_share']['mean']:.2%} of rows")
    print()
    shipped = result["runs"]["xgboost_tuned"]["summary"]
    print(f"shipped  : nested F1 {shipped['nested_f1']['mean']:.4f} at a mean cut of "
          f"{shipped['nested_threshold']['mean']:.4f}; at 7.50% F1 {shipped['be_f1']['mean']:.4f}, "
          f"precision {shipped['be_precision']['mean']:.4f}, recall {shipped['be_recall']['mean']:.4f}")
    print(f"wrote    : {result['csv']} ({len(result['table'])} rows, "
          f"{len(result['models'])} models x 5 folds + mean + sd)")
    print(f"total    : {result['elapsed_seconds']} s")
    print()
    print(f"T32 checks: {result['checks_passed']}/{result['checks_total']} passed")
    for check in result["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        detail = f"  ({check['detail']})" if check["detail"] else ""
        print(f"  [{marker}] {check['check']}{detail}")
    print()


if __name__ == "__main__":
    print("DepositIQ Phase 7 - cross-validated metrics")
    print()
    outcome = report()
    print_report(outcome)
    print("=" * 112)
    print(f"result : {'PASS - the table is CV-only, thresholded at a cut chosen without the fold it scores' if outcome['checks_ok'] else 'FAIL'}")
    print("Next   : T33 scores the persisted artifact once on the held-out half.")
    sys.exit(0 if outcome["checks_ok"] else 1)
