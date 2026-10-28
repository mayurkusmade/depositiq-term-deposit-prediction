"""
DepositIQ - the `duration` leakage audit (T22).

`duration` is the length of the marketing call in seconds. It is recorded when the call *ends*, and the
call's outcome is what the model is trying to predict, so a model given this column is reading the
answer off the back of the exam paper. D05 drops it from the main model; this module is the evidence for
that decision and the single place that defines how the column is toggled, so T36's comparison can run
one pipeline twice with nothing else changing.

Two things live here:

  * `feature_columns()` - the toggle. Given any list of engineered columns, it returns the list with or
    without the leaky ones. T24's `ColumnTransformer` and T36's comparison both call this rather than
    filtering `duration` by hand, so "identical pipeline, one toggle" is enforced by code.
  * `audit()` - the measured leak signature: how the subscription rate moves with the call length, how
    a ranking built on `duration` alone would score, and how much of the positive class it would
    capture. Diagnostic only - no model is fitted, and nothing here is a result to report as skill.

Run `python -m src.leakage` for the audit document's numbers.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config
from src.split import encode_target, load_split

# The column that leaks. Named in one place so T36 cannot toggle a different one by accident.
LEAKY_COLUMNS: tuple[str, ...] = ("duration",)

# Band edges in seconds: 0-60, 60-120, 120-180, 180-300, 300-600, 600+.
# The bands are coarse on purpose: the rate is monotone enough that finer splits only add noise, and the
# point is the shape, not a bucket table nobody will act on.
BAND_EDGES: tuple[int, ...] = (60, 120, 180, 300, 600)


def feature_columns(columns, *, with_duration: bool = False) -> list[str]:
    """
    Return `columns` with the leaky ones removed unless `with_duration=True`.

    This is the whole of the T36 toggle: both arms pass the same engineered list through this function,
    so the feature sets differ by exactly `LEAKY_COLUMNS` and by nothing else.
    """
    kept = [column for column in columns if with_duration or column not in LEAKY_COLUMNS]
    dropped = [column for column in columns if column not in kept]
    assert not with_duration or not dropped, "with_duration=True must keep every column"
    assert with_duration or not any(column in kept for column in LEAKY_COLUMNS), "leak survived the filter"
    return kept


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval - the same method T18/T19 used for every rate in the EDA."""
    if total == 0:
        return (float("nan"), float("nan"))
    phat = successes / total
    denominator = 1 + z**2 / total
    centre = (phat + z**2 / (2 * total)) / denominator
    half = z * np.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def band_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Subscription rate by call length, with counts, share of rows and a Wilson interval."""
    target = encode_target(frame)
    edges = [-0.5, *BAND_EDGES, float("inf")]
    labels = ["0-59s", "60-119s", "120-179s", "180-299s", "300-599s", "600s+"]
    band = pd.cut(frame["duration"], bins=edges, labels=labels, right=False)

    rows = []
    for label in labels:
        mask = band == label
        n = int(mask.sum())
        yes = int(target[mask].sum())
        low, high = wilson_interval(yes, n)
        rows.append(
            {
                "band": label,
                "calls": n,
                "share_of_calls": n / len(frame),
                "subscribers": yes,
                "rate": yes / n if n else float("nan"),
                "wilson_low": low,
                "wilson_high": high,
            }
        )
    return pd.DataFrame(rows)


def single_column_ranking(frame: pd.DataFrame) -> dict:
    """
    How well `duration` alone orders the call list.

    A ranking metric rather than a model: the column is used directly as the score, so this measures
    what the leak is worth, not what a classifier can do. A random scorer would score 0.5 ROC-AUC and
    a PR-AUC equal to the base rate, so both numbers are lifts over that baseline.
    """
    target = encode_target(frame)
    score = frame["duration"].to_numpy(dtype=float)
    order = np.argsort(-score)
    ranked = target.to_numpy()[order]

    positives = int(ranked.sum())
    decile = max(1, len(ranked) // 10)
    top_decile = ranked[:decile]

    pr_auc = float(average_precision_score(target, score))
    base = float(target.mean())
    top_precision = float(top_decile.mean())

    return {
        "roc_auc": float(roc_auc_score(target, score)),
        "pr_auc": pr_auc,
        "pr_auc_lift": pr_auc / base,
        "base_rate": base,
        "top_decile_precision": top_precision,
        "top_decile_lift": top_precision / base,
        "top_decile_recall": float(top_decile.sum() / positives),
        "positives": positives,
        "rows": len(frame),
    }


def audit(frame: pd.DataFrame | None = None) -> dict:
    """Everything the leakage document quotes, measured on the training half."""
    if frame is None:
        frame = load_split("full", "train")

    target = encode_target(frame)
    bands = band_table(frame)
    ranking = single_column_ranking(frame)

    shortest = bands.iloc[0]
    longest = bands.iloc[-1]

    return {
        "rows": len(frame),
        "duration_min": int(frame["duration"].min()),
        "duration_q1": float(frame["duration"].quantile(0.25)),
        "duration_median": float(frame["duration"].median()),
        "duration_q3": float(frame["duration"].quantile(0.75)),
        "duration_max": int(frame["duration"].max()),
        "median_yes": float(frame.loc[target == 1, "duration"].median()),
        "median_no": float(frame.loc[target == 0, "duration"].median()),
        "bands": bands,
        "ranking": ranking,
        "shortest_band_rate": float(shortest["rate"]),
        "shortest_band_calls": int(shortest["calls"]),
        "shortest_band_share_of_positives": float(shortest["subscribers"] / target.sum()),
        "longest_band_rate": float(longest["rate"]),
        "longest_band_share_of_positives": float(longest["subscribers"] / target.sum()),
        "monotone": bool((bands["rate"].diff().dropna() >= -0.005).all()),
        "with_duration_columns": feature_columns(["age", "duration", "campaign"], with_duration=True),
        "without_duration_columns": feature_columns(["age", "duration", "campaign"]),
        "test_rows": len(load_split("full", "test")),
    }


def print_audit(report: dict) -> None:
    print("DepositIQ T22 - the `duration` leakage audit")
    print("=" * 96)
    print(f"training half            : {report['rows']:,} calls")
    print(f"duration (seconds)       : min {report['duration_min']}, q1 {report['duration_q1']:.0f}, "
          f"median {report['duration_median']:.0f}, q3 {report['duration_q3']:.0f}, "
          f"max {report['duration_max']:,}")
    print(f"median duration, yes / no: {report['median_yes']:.0f}s / {report['median_no']:.0f}s "
          f"({report['median_yes'] / report['median_no']:.2f}x)")
    print()
    print("SUBSCRIPTION RATE BY CALL LENGTH")
    bands = report["bands"]
    for _, row in bands.iterrows():
        print(f"  {row['band']:<9} {row['calls']:>6,} calls ({row['share_of_calls']:>6.2%})  "
              f"{row['subscribers']:>5,} subscribers  {row['rate']:>7.2%} "
              f"[{row['wilson_low']:.2%}, {row['wilson_high']:.2%}]")
    print(f"  rate is non-decreasing across the six bands: {report['monotone']}")
    print(f"  shortest band supplies {report['shortest_band_share_of_positives']:.2%} of all subscribers; "
          f"longest band supplies {report['longest_band_share_of_positives']:.2%}")
    print()
    ranking = report["ranking"]
    print("A RANKING BUILT ON `duration` ALONE (diagnostic - not a model)")
    print(f"  ROC-AUC             : {ranking['roc_auc']:.4f}  (random = 0.5000)")
    print(f"  PR-AUC              : {ranking['pr_auc']:.4f}  (random = {ranking['base_rate']:.4f} base rate, "
          f"lift {ranking['pr_auc_lift']:.2f}x)")
    print(f"  top-decile precision: {ranking['top_decile_precision']:.4f}  "
          f"(lift {ranking['top_decile_lift']:.2f}x)")
    print(f"  top-decile recall   : {ranking['top_decile_recall']:.4f}  "
          f"({int(ranking['top_decile_recall'] * ranking['positives']):,} of {ranking['positives']:,})")
    print()
    print("THE TOGGLE T36 WILL USE")
    print(f"  with    : {report['with_duration_columns']}")
    print(f"  without : {report['without_duration_columns']}")
    print(f"  held-out rows that stay unread: {report['test_rows']:,}")
    print("=" * 96)
    print("No classifier is fitted here: these numbers describe the column, not a model's skill.")


if __name__ == "__main__":
    report = audit()
    print_audit(report)
    ok = (
        report["monotone"]
        and report["shortest_band_rate"] < report["ranking"]["base_rate"]
        and report["without_duration_columns"] == ["age", "campaign"]
        and "duration" not in report["without_duration_columns"]
    )
    sys.exit(0 if ok else 1)
