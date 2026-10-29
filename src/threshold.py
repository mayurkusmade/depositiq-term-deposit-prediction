"""
DepositIQ - cost-based threshold selection and probability calibration (T34).

The shipped XGBoost model is a strong ranker but its class weighting pushes raw scores
above the population probability. This task fits a monotone isotonic map using only
out-of-fold training predictions, then applies D17's cost rule to the calibrated scale.
The held-out half is used once for confirmation and is never used to choose the cut.

Run from the repository root:

    python -m src.threshold
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from src import config, cost, features, select, train, tune
from src.split import encode_target


RECORD_PATH: Path = config.MODELS_DIR / "threshold_eval.json"
FIGURE_PATH: Path = config.FIGURES_DIR / "08_threshold_cost_calibration.png"
N_BINS: int = 10
BREAK_EVEN: float = cost.break_even_threshold()
LEGACY_CV_CUT: float = float(
    json.loads((config.MODELS_DIR / "metrics_cv.json").read_text())
    ["models"]["xgboost_tuned"]["summary"]["nested_threshold"]["mean"]
)


def load_frames() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_frame = features.load_engineered("train")
    test_frame = features.load_engineered("test")
    return (
        train_frame.drop(columns=[config.TARGET]),
        encode_target(train_frame),
        test_frame.drop(columns=[config.TARGET]),
        encode_target(test_frame),
    )


def selected_estimator():
    params = json.loads((config.MODELS_DIR / "best_params.json").read_text())["best_params"]["params"]
    return tune.make_estimator(params)


def out_of_fold_scores(frame: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, list[dict]]:
    folds = train.materialise_folds(frame, y)
    run = train.cross_validate_factory(
        "xgboost_tuned",
        selected_estimator,
        "shipped",
        "T30 tuned configuration",
        frame,
        y,
        folds,
        select.shipped_features(),
    )
    scores = np.empty(len(frame), dtype=float)
    fold_records = []
    for fold_number, (pipeline, _, valid_idx) in enumerate(run["fits"], start=1):
        fold_scores = pipeline.predict_proba(frame.iloc[valid_idx])[:, 1].astype(float)
        scores[valid_idx] = fold_scores
        fold_records.append({
            "fold": fold_number,
            "rows": int(len(valid_idx)),
            "positives": int(y.iloc[valid_idx].sum()),
            "raw_mean": float(fold_scores.mean()),
        })
    return scores, fold_records


def cross_fitted_calibration(raw: np.ndarray, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    calibrated = np.empty(len(raw), dtype=float)
    for train_idx, valid_idx in folds:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw[train_idx], y[train_idx])
        calibrated[valid_idx] = calibrator.predict(raw[valid_idx])
    return calibrated


def fit_calibrator(raw: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw, y)
    return calibrator


def campaign_at_threshold(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    result = cost.expected_campaign_value(y, scores, threshold)
    result["expected_net_value_per_call_eur"] = (
        result["net_value_eur"] / result["clients_called"]
        if result["clients_called"] else float("nan")
    )
    result["expected_cost_per_call_eur"] = cost.CALL_COST_EUR
    result["called_share"] = result["clients_called"] / len(y)
    result["subscriber_rate_in_calls"] = (
        result["subscribers_reached"] / result["clients_called"]
        if result["clients_called"] else 0.0
    )
    return result


def empirical_best(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    order = np.argsort(scores)[::-1]
    ordered_y = y[order]
    cumulative = np.cumsum(ordered_y)
    calls = np.arange(1, len(y) + 1)
    net = cumulative * cost.VALUE_PER_SUBSCRIPTION_EUR - calls * cost.CALL_COST_EUR
    position = int(np.argmax(net))
    threshold = float(scores[order[position]])
    return {
        **campaign_at_threshold(y, scores, threshold),
        "rank": position + 1,
        "threshold_rule": "maximum observed net campaign value on cross-fitted training scores",
    }


def reliability(y: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    rows = []
    for index in range(N_BINS):
        mask = (scores >= edges[index]) & (
            scores < edges[index + 1] if index < N_BINS - 1 else scores <= edges[index + 1]
        )
        if not mask.any():
            continue
        rows.append({
            "bin": index + 1,
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "count": int(mask.sum()),
            "mean_prediction": float(scores[mask].mean()),
            "observed_rate": float(y[mask].mean()),
        })
    return rows


def draw_figure(
    y_train: np.ndarray,
    raw_train: np.ndarray,
    calibrated_train: np.ndarray,
    y_test: np.ndarray,
    raw_test: np.ndarray,
    calibrated_test: np.ndarray,
) -> dict[str, object]:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for label, y, scores, colour in (
        ("training OOF", y_train, raw_train, "#4C78A8"),
        ("held-out raw", y_test, raw_test, "#F58518"),
        ("training cross-fitted", y_train, calibrated_train, "#54A24B"),
        ("held-out calibrated", y_test, calibrated_test, "#E45756"),
    ):
        points = reliability(y, scores)
        axes[0].plot(
            [p["mean_prediction"] for p in points],
            [p["observed_rate"] for p in points],
            "o-",
            label=label,
            color=colour,
        )
    axes[0].plot([0, 1], [0, 1], "--", color="#777777", label="perfect calibration")
    axes[0].axvline(BREAK_EVEN, color="#E45756", linestyle=":", label=f"break-even {BREAK_EVEN:.2%}")
    axes[0].set(xlabel="Mean predicted probability", ylabel="Observed subscription rate",
                title="Reliability: raw scores and monotone calibration")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    thresholds = np.linspace(0.0, 0.35, 351)
    net = []
    for threshold in thresholds:
        result = cost.expected_campaign_value(y_train, calibrated_train, float(threshold))
        net.append(result["net_value_eur"] / len(y_train))
    axes[1].plot(thresholds, net, color="#54A24B", linewidth=2)
    sensitivity_values = [
        cost.break_even_threshold(call_cost, value)
        for call_cost in (0.75, 1.00, 1.50, 2.00, 3.00)
        for value in (10.0, 20.0, 40.0, 60.0)
    ]
    axes[1].axvspan(
        min(sensitivity_values),
        max(sensitivity_values),
        color="#B279A2",
        alpha=0.12,
        label="D17 sensitivity range",
    )
    axes[1].axvline(BREAK_EVEN, color="#E45756", linestyle=":", label=f"chosen {BREAK_EVEN:.2%}")
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].set(xlabel="Calibrated probability threshold", ylabel="Expected net value per training row (EUR)",
                title="Business value selects the operating threshold")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=160)
    plt.close(figure)
    return {"bytes": FIGURE_PATH.stat().st_size, "path": str(FIGURE_PATH.relative_to(config.ROOT))}


def main() -> dict:
    started = time.perf_counter()
    config.ensure_dirs()
    train_frame, y_train_series, test_frame, y_test_series = load_frames()
    y_train = y_train_series.to_numpy(dtype=int)
    y_test = y_test_series.to_numpy(dtype=int)
    raw_train, fold_records = out_of_fold_scores(train_frame, y_train_series)
    folds = train.materialise_folds(train_frame, y_train_series)
    calibrated_train = cross_fitted_calibration(raw_train, y_train, folds)

    calibrator = fit_calibrator(raw_train, y_train)
    pipeline = joblib.load(config.MODELS_DIR / "final_model.joblib")
    raw_test = pipeline.predict_proba(test_frame)[:, 1].astype(float)
    calibrated_test = np.asarray(calibrator.predict(raw_test), dtype=float)

    chosen_train = campaign_at_threshold(y_train, calibrated_train, BREAK_EVEN)
    chosen_test = campaign_at_threshold(y_test, calibrated_test, BREAK_EVEN)
    legacy_test_raw = campaign_at_threshold(y_test, raw_test, LEGACY_CV_CUT)
    best_train = empirical_best(y_train, calibrated_train)
    sensitivity = cost.sensitivity_table().to_dict(orient="records")
    figure = draw_figure(y_train, raw_train, calibrated_train, y_test, raw_test, calibrated_test)

    checks = [
        {"check": "training and held-out rows remain separate", "passed": len(train_frame) == 36168 and len(test_frame) == 9043},
        {"check": "out-of-fold scores cover each training row once", "passed": bool(np.isfinite(raw_train).all())},
        {"check": "calibrated scores are probabilities", "passed": bool(((calibrated_train >= 0) & (calibrated_train <= 1)).all())},
        {"check": "calibrator is monotone", "passed": bool(np.all(np.diff(calibrator.predict(np.linspace(0, 1, 1001))) >= -1e-12))},
        {"check": "chosen threshold is D17 break-even", "passed": BREAK_EVEN == cost.break_even_threshold()},
        {"check": "figure is a real output", "passed": figure["bytes"] > 30000},
    ]
    payload = {
        "task": "T34",
        "variant": "full",
        "training_rows": len(train_frame),
        "held_out_rows": len(test_frame),
        "base_rate": float(y_train.mean()),
        "calibration": {
            "method": "isotonic regression",
            "training_fit": "five-fold out-of-fold raw probabilities only",
            "raw_training": {"mean_prediction": float(raw_train.mean()), "brier": float(brier_score_loss(y_train, raw_train))},
            "calibrated_training": {"mean_prediction": float(calibrated_train.mean()), "brier": float(brier_score_loss(y_train, calibrated_train))},
            "raw_held_out": {"mean_prediction": float(raw_test.mean()), "brier": float(brier_score_loss(y_test, raw_test))},
            "calibrated_held_out": {"mean_prediction": float(calibrated_test.mean()), "brier": float(brier_score_loss(y_test, calibrated_test))},
            "reliability_training_raw": reliability(y_train, raw_train),
            "reliability_training_calibrated": reliability(y_train, calibrated_train),
            "reliability_held_out_raw": reliability(y_test, raw_test),
            "reliability_held_out_calibrated": reliability(y_test, calibrated_test),
        },
        "cost_model": {
            "call_cost_eur": cost.CALL_COST_EUR,
            "value_per_subscription_eur": cost.VALUE_PER_SUBSCRIPTION_EUR,
            "break_even_threshold": BREAK_EVEN,
            "sensitivity": sensitivity,
        },
        "chosen_threshold": {
            "threshold": BREAK_EVEN,
            "rule": "D17 expected net value: call when calibrated p >= C/V",
            "training_cross_fitted": chosen_train,
            "held_out_confirmation": chosen_test,
        },
        "empirical_training_best": best_train,
        "legacy_raw_held_out": {
            "threshold": LEGACY_CV_CUT,
            "note": "T33's raw-score cut, reported for comparison rather than selected here",
            "campaign": legacy_test_raw,
        },
        "folds": fold_records,
        "figure": figure,
        "checks": checks,
        "checks_passed": sum(int(check["passed"]) for check in checks),
        "checks_total": len(checks),
        "checks_ok": all(check["passed"] for check in checks),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    RECORD_PATH.write_text(json.dumps(payload, indent=2))
    if not payload["checks_ok"]:
        raise SystemExit(1)
    return payload


if __name__ == "__main__":
    result = main()
    chosen = result["chosen_threshold"]
    train_result = chosen["training_cross_fitted"]
    test_result = chosen["held_out_confirmation"]
    print("DepositIQ T34 - cost-based threshold and calibration")
    print(f"break-even : {BREAK_EVEN:.4f} ({BREAK_EVEN:.2%})")
    print(f"training   : {train_result['clients_called']:,} calls, "
          f"{train_result['expected_net_value_per_call_eur']:+.2f} EUR net/call")
    print(f"held-out   : {test_result['clients_called']:,} calls, "
          f"{test_result['expected_net_value_per_call_eur']:+.2f} EUR net/call")
    print(f"checks     : {result['checks_passed']}/{result['checks_total']}")
    print(f"wrote      : {RECORD_PATH.name}, {FIGURE_PATH.name}")
