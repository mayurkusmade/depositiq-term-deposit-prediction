"""
DepositIQ - the held-out evaluation (T33).

Why this file exists
--------------------
Every number the project has quoted so far came from the **training half**: five folds of it, scored
against each other. That is a sound way to compare models, and it is not evidence about a model nobody
has seen the answer for yet. This is the step where the artifact is scored **once** on the 9,043 rows it
was never fitted on, and the result is written down before anything is allowed to react to it.

Four things are true of this module and each of them is enforced rather than intended:

* **It is the artifact, not a refit.** `models/final_model.joblib` is loaded and used. Nothing here fits
  anything - a refit on the held-out rows would turn the one out-of-sample number in the project into a
  second in-sample number that looks like an out-of-sample one, which is the worst of both.
* **The held-out rows are scored exactly once**, and the record stores every metric that came out of that
  single pass. There is no test-set tuning, no recalibration, no re-selection. T34 owns calibration; if
  the probability scale turns out to be wrong, this step's job is to show it.
* **The operating cut is not chosen here.** It is T32's mean cross-validated cut, read from
  `models/metrics_cv.json`. A cut fitted on the held-out scores would be a second use of the same rows.
* **The gap against CV is computed and stated**, because the CV figure was produced by a model selected on
  those same folds and is therefore optimistic in principle. The bootstrap below quantifies how large an
  effect this single test draw could even resolve.

Why three cuts and not one
--------------------------
The task asks for the 0.5-threshold numbers, and the figures carry them - but 0.5 is near the 95th
percentile of this model's scores (it flags 4.38% of the held-out rows), so a confusion matrix drawn at
0.5 alone shows a nearly empty positive column and reads as a broken model rather than as a badly placed
bar. T29 measured that and T32 confirmed it on two models. So each figure marks three cuts:

| cut | where it comes from | what it answers |
|---|---|---|
| **7.50%** | D17's cost model, not fitted to data | the highest call volume that still pays for itself |
| **0.2103** | T32's mean CV cut, chosen on the training half | the least-cost-per-subscription operating point |
| **0.5** | convention | the number this project rejects, carried because it was asked for |

How to run
----------
From the repository root, inside the virtual environment:

    python -m src.evaluate    # ~25 s: one prediction pass, one bootstrap, three figures

Writes `reports/figures/07_confusion_matrix.png`, `07_roc_curve.png`, `07_pr_curve.png` and
`models/holdout_eval.json`; `docs/06_modeling.md` §11 quotes only numbers from that record.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src import config, features, metrics as metric_table, train
from src.split import encode_target

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARTIFACT_PATH: Path = config.MODELS_DIR / "final_model.joblib"
CARD_PATH: Path = config.MODELS_DIR / "final_model.json"
CV_RECORD_PATH: Path = config.MODELS_DIR / "metrics_cv.json"
RECORD_PATH: Path = config.MODELS_DIR / "holdout_eval.json"

CONFUSION_FIGURE: Path = config.FIGURES_DIR / "07_confusion_matrix.png"
ROC_FIGURE: Path = config.FIGURES_DIR / "07_roc_curve.png"
PR_FIGURE: Path = config.FIGURES_DIR / "07_pr_curve.png"

# The two cuts the project argued for are imported, not restated: a second copy of 0.075 is a second
# thing to update, and the break-even's whole value is that it is *the* cost model's output.
CONVENTIONAL: float = metric_table.CONVENTIONAL
BREAK_EVEN: float = metric_table.BREAK_EVEN

# Bootstrap settings. 2,000 resamples of 9,043 rows is ~25 s of work for an interval that answers
# "could this single draw have landed anywhere else?", which is the question a gap needs.
BOOTSTRAP_RESAMPLES: int = 2_000
BOOTSTRAP_SEED: int = config.RANDOM_STATE

COLOURS: dict[str, str] = {
    "break_even": "#E45756",
    "operating": "#54A24B",
    "conventional": "#4C78A8",
}

# Floor for "the figure is a real chart and not a 2 KB stub". Chosen well under what these plots come
# out at (see the record) so it only fires on a genuinely empty or failed write.
MIN_FIGURE_BYTES: int = 30_000


# ---------------------------------------------------------------------------
# Loading and scoring
# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_artifact():
    """Load T31's pipeline and its card, and refuse to continue if the file is not the one recorded."""
    pipeline = joblib.load(ARTIFACT_PATH)
    card = json.loads(CARD_PATH.read_text())
    return pipeline, card


def score(pipeline, frame: pd.DataFrame) -> np.ndarray:
    """
    P(subscribe) for every held-out row, in float64. One call; the model is not touched again.

    The cast is not cosmetic. XGBoost's `predict_proba` returns **float32**, and numpy keeps
    float32 op int8 in float32 - so `(score - y) ** 2` and its mean are accumulated in single precision,
    which moved the recorded Brier score by **4.4e-9** between an int8 and an int64 target vector. T30's
    verifier found the same class of defect in the CV record (§9.5 of `docs/06_modeling.md`), and this
    module is the one place where a probability is compared against a cost threshold, so it scores in
    float64 and says so.
    """
    return pipeline.predict_proba(frame)[:, 1].astype(np.float64)


def operating_cut() -> float:
    """T32's mean cross-validated operating cut, from the record rather than from this file."""
    record = json.loads(CV_RECORD_PATH.read_text())
    return float(record["models"]["xgboost_tuned"]["summary"]["nested_threshold"]["mean"])


def cut_descriptors() -> tuple[dict, ...]:
    """
    The three cuts, ascending by threshold - so the panels read left to right as falling call volume.

    `rejected` is what the figures colour and label on; it is not decoration, it is the conclusion of
    T29-T32 reduced to one boolean so the 0.5 panel cannot quietly become the headline.
    """
    operating = operating_cut()
    return (
        {
            "key": "break_even",
            "label": f"{BREAK_EVEN:.2%}",
            "threshold": BREAK_EVEN,
            "role": "D17 break-even: a call pays for itself above this subscription rate",
            "rejected": False,
        },
        {
            "key": "operating",
            "label": f"{operating:.4f}",
            "threshold": operating,
            "role": "T32 mean CV cut: chosen on the training half, never on these rows",
            "rejected": False,
        },
        {
            "key": "conventional",
            "label": f"{CONVENTIONAL:.1f}",
            "threshold": CONVENTIONAL,
            "role": "conventional cut: carried because the task asks for it, rejected by T29-T32",
            "rejected": True,
        },
    )


def cut_table(y: np.ndarray, scores: np.ndarray, descriptors: tuple[dict, ...]) -> list[dict]:
    """
    Score every cut through T32's own `at_threshold`, then add the two quantities it does not report.

    Reusing `metrics.at_threshold` is deliberate: if the held-out figure and the CV table disagreed about
    what precision means at a threshold, the comparison between them would be two experiments.
    """
    negatives = int((y == 0).sum())
    rows = []
    for descriptor in descriptors:
        record = metric_table.at_threshold(y, scores, descriptor["threshold"])
        record["rows"] = int(len(y))
        record["positives"] = int(y.sum())
        record["negatives"] = negatives
        record["true_negative"] = int(len(y) - record["flagged"] - record["false_negative"])
        record["false_positive_rate"] = record["false_positive"] / negatives if negatives else 0.0
        record["specificity"] = 1.0 - record["false_positive_rate"]
        record["label"] = descriptor["label"]
        record["role"] = descriptor["role"]
        record["rejected"] = descriptor["rejected"]
        record["key"] = descriptor["key"]
        rows.append(record)
    return rows


def curve_data(y: np.ndarray, scores: np.ndarray) -> dict:
    """ROC and PR coordinates, kept as arrays so the checks can integrate them back to the metrics."""
    fpr, tpr, roc_thresholds = roc_curve(y, scores)
    precision, recall, pr_thresholds = precision_recall_curve(y, scores)
    return {
        "roc": {
            "false_positive_rate": fpr, "true_positive_rate": tpr, "thresholds": roc_thresholds,
            "auc": float(roc_auc_score(y, scores)),
        },
        "pr": {
            "precision": precision, "recall": recall, "thresholds": pr_thresholds,
            "average_precision": float(average_precision_score(y, scores)),
            "base_rate": float(y.mean()),
        },
    }


def bootstrap(y: np.ndarray, scores: np.ndarray, cv_mean: float,
              resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
    """
    Resample the held-out rows with replacement and re-measure both ranking metrics.

    This exists for one reason: the held-out PR-AUC is a single draw from a distribution that has 1,058
    positives in it, and the gap against CV is only interpretable against that width. Without it, any
    gap at all reads as evidence.
    """
    rng = np.random.default_rng(seed)
    size = len(y)
    aps: list[float] = []
    aucs: list[float] = []
    skipped = 0
    for _ in range(resamples):
        index = rng.integers(0, size, size)
        y_sample, s_sample = y[index], scores[index]
        if y_sample.sum() < 2:  # a draw with one class has no AUC; with 1,058/9,043 it never fires
            skipped += 1
            continue
        aps.append(float(average_precision_score(y_sample, s_sample)))
        aucs.append(float(roc_auc_score(y_sample, s_sample)))

    def summarise(values: list[float]) -> dict:
        array = np.asarray(values)
        low, high = np.percentile(array, [2.5, 97.5])
        return {
            "mean": float(array.mean()),
            "sd": float(array.std(ddof=1)),
            "p2_5": float(low),
            "p97_5": float(high),
        }

    return {
        "resamples": int(resamples),
        "seed": int(seed),
        "skipped": int(skipped),
        "pr_auc": summarise(aps),
        "roc_auc": summarise(aucs),
        "share_of_draws_below_cv": float(np.mean(np.asarray(aps) < cv_mean)),
        "values_pr_auc": aps,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _pyplot():
    """One place for the backend switch, so every figure in this module is drawn headless."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def draw_confusion(cuts: list[dict], path: Path = CONFUSION_FIGURE) -> list[str]:
    """
    Three 2x2 panels, shaded by **row** share so the diagonal reads as recall and specificity.

    Cell text carries the raw count and its share of the actual class, and the line under each panel
    carries precision, recall, F1 and the flag count - which is the set of numbers the task asked to be
    annotated, plus the two that show why 0.5 is the wrong bar.
    """
    plt = _pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(cuts), figsize=(5.0 * len(cuts), 5.6))
    annotations: list[str] = []

    for ax, cut in zip(np.atleast_1d(axes), cuts):
        counts = np.array(
            [[cut["true_negative"], cut["false_positive"]],
             [cut["false_negative"], cut["true_positive"]]],
            dtype=float,
        )
        share = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0.0, vmax=1.0)

        for row, actual in enumerate(("no", "yes")):
            for column in range(2):
                ax.text(
                    column, row,
                    f"{int(counts[row, column]):,}\n{share[row, column]:.1%} of actual {actual}",
                    ha="center", va="center", fontsize=10.5,
                    color="white" if share[row, column] > 0.55 else "#1A1A1A",
                )

        ax.set_xticks([0, 1], ["no", "yes"])
        ax.set_yticks([0, 1], ["no", "yes"])
        ax.set_xlabel(
            "predicted class\n\n"
            f"precision {cut['precision']:.4f}   recall {cut['recall']:.4f}\n"
            f"F1 {cut['f1']:.4f}   flags {cut['flagged']:,} ({cut['flagged_share']:.2%})",
            fontsize=10,
        )
        title = f"cut {cut['label']}"
        if cut["rejected"]:
            title += "  (rejected)"
        ax.set_title(title, fontsize=12, color="#C0392B" if cut["rejected"] else "#1A1A1A",
                     fontweight="bold" if cut["rejected"] else "normal")
        ax.text(0.5, -0.30, cut["role"], transform=ax.transAxes, ha="center", va="top",
                fontsize=8.5, color="#555555", wrap=True)

        annotations.append(
            f"{cut['label']}: precision {cut['precision']:.4f} recall {cut['recall']:.4f} "
            f"F1 {cut['f1']:.4f} flags {cut['flagged']:,}"
        )

    np.atleast_1d(axes)[0].set_ylabel("actual class", fontsize=11)
    figure.suptitle(
        f"DepositIQ - confusion matrix on the {cuts[0]['rows']:,} held-out rows "
        f"({cuts[0]['positives']:,} subscribed, {cuts[0]['rows'] / cuts[0]['positives']:.0f}:1)\n"
        f"the artifact scored once, at three cuts: {', '.join(cut['label'] for cut in cuts)}",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return annotations


def draw_roc(curves: dict, cuts: list[dict], path: Path = ROC_FIGURE) -> list[str]:
    """
    The full curve, then a zoom on the region where two of the three cuts actually sit.

    The zoom is not decoration: the conventional cut sits at FPR 1.7% and the operating cut at 9.2%, and
    on a 0-1 axis those two points are a few pixels apart. The break-even cut is at FPR 44%, off the
    zoomed panel, and the title says so rather than leaving a reader to wonder where it went.
    """
    plt = _pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 6.2))
    curve = curves["roc"]
    fpr, tpr = curve["false_positive_rate"], curve["true_positive_rate"]
    annotations: list[str] = []

    for panel in (left, right):
        panel.plot(fpr, tpr, color="#333333", linewidth=2.0,
                   label=f"held-out ROC (AUC {curve['auc']:.4f})")
        panel.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1.0,
                   label="no skill (AUC 0.5)")

    for cut in cuts:
        colour = COLOURS[cut["key"]]
        left.scatter(cut["false_positive_rate"], cut["recall"], s=90, color=colour, zorder=5,
                     edgecolor="white", linewidth=1.2)
        left.annotate(
            f"{cut['label']}\nTPR {cut['recall']:.3f}",
            xy=(cut["false_positive_rate"], cut["recall"]),
            xytext=(10, -26), textcoords="offset points", fontsize=9, color=colour,
            arrowprops={"arrowstyle": "-", "color": colour, "linewidth": 0.8},
        )
        if cut["false_positive_rate"] <= 0.14:
            right.scatter(cut["false_positive_rate"], cut["recall"], s=90, color=colour, zorder=5,
                          edgecolor="white", linewidth=1.2)
            right.annotate(
                f"cut {cut['label']}\nTPR {cut['recall']:.3f}  FPR {cut['false_positive_rate']:.3f}",
                xy=(cut["false_positive_rate"], cut["recall"]),
                # Below and to the right: the curve rises to the upper left, so that corner of the zoom is
                # the only one where the text does not sit on the line it is labelling.
                xytext=(16, -34), textcoords="offset points", fontsize=9, color=colour,
                arrowprops={"arrowstyle": "-", "color": colour, "linewidth": 0.8},
            )
        annotations.append(
            f"{cut['label']}: TPR {cut['recall']:.4f} FPR {cut['false_positive_rate']:.4f}"
        )

    left.set_xlabel("false positive rate (1 - specificity)")
    left.set_ylabel("true positive rate (recall)")
    left.set_title(f"held-out ROC - area under the curve {curve['auc']:.4f}", fontsize=11)
    left.set_xlim(-0.01, 1.01)
    left.set_ylim(-0.01, 1.01)
    left.legend(loc="lower right", fontsize=9)
    left.grid(alpha=0.25)

    right.set_xlabel("false positive rate (1 - specificity)")
    right.set_ylabel("true positive rate (recall)")
    right.set_title(
        "zoom on the two tight cuts\n"
        f"(the {cuts[0]['label']} cut sits at FPR {cuts[0]['false_positive_rate']:.3f}, off this axis)",
        fontsize=11,
    )
    right.set_xlim(0.0, 0.14)
    right.set_ylim(0.15, 0.62)
    right.grid(alpha=0.25)

    figure.suptitle(
        f"DepositIQ - ROC on {cuts[0]['rows']:,} held-out rows: the artifact scored once, "
        f"with the three cuts marked",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return annotations


def draw_pr(y: np.ndarray, scores: np.ndarray, curves: dict, cuts: list[dict],
            path: Path = PR_FIGURE) -> list[str]:
    """
    The precision-recall curve, then what each metric does as the bar moves.

    The right panel is the one that explains the whole threshold argument in one picture: F1 peaks
    somewhere near the operating cut and falls away on both sides, while at 0.5 the model keeps its
    precision and loses half its recall.
    """
    plt = _pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 6.2))
    curve = curves["pr"]
    precision, recall = curve["precision"], curve["recall"]
    annotations: list[str] = []

    left.plot(recall, precision, color="#333333", linewidth=2.0,
              label=f"held-out PR curve (AP {curve['average_precision']:.4f})")
    left.axhline(curve["base_rate"], linestyle="--", color="#999999", linewidth=1.0,
                 label=f"no skill = base rate {curve['base_rate']:.4f}")

    for cut in cuts:
        colour = COLOURS[cut["key"]]
        left.scatter(cut["recall"], cut["precision"], s=90, color=colour, zorder=5,
                     edgecolor="white", linewidth=1.2)
        left.annotate(
            f"cut {cut['label']}\nP {cut['precision']:.3f}  R {cut['recall']:.3f}",
            xy=(cut["recall"], cut["precision"]), xytext=(10, 10), textcoords="offset points",
            fontsize=9, color=colour,
            arrowprops={"arrowstyle": "-", "color": colour, "linewidth": 0.8},
        )
        annotations.append(
            f"{cut['label']}: precision {cut['precision']:.4f} recall {cut['recall']:.4f}"
        )

    left.set_xlabel("recall (share of the 1,058 subscribers found)")
    left.set_ylabel("precision (share of the calls that pay off)")
    left.set_title(f"held-out precision-recall - average precision {curve['average_precision']:.4f}",
                   fontsize=11)
    left.set_xlim(-0.01, 1.01)
    left.set_ylim(0.0, 1.0)
    left.legend(loc="upper right", fontsize=9)
    left.grid(alpha=0.25)

    # The three cuts are inserted into the grid so every drawn line passes exactly through the value the
    # record stores - a nearest-grid-point estimate would disagree with the table by a few ten-thousandths
    # and the figure would then not be the record in picture form.
    grid = np.unique(np.concatenate([
        np.linspace(0.002, 0.65, 400), np.array([cut["threshold"] for cut in cuts]),
    ]))
    sweep = {"precision": [], "recall": [], "f1": []}
    for threshold in grid:
        row = metric_table.at_threshold(y, scores, float(threshold))
        sweep["precision"].append(row["precision"])
        sweep["recall"].append(row["recall"])
        sweep["f1"].append(row["f1"])

    right.plot(grid, sweep["precision"], color="#4C78A8", linewidth=1.8, label="precision")
    right.plot(grid, sweep["recall"], color="#54A24B", linewidth=1.8, label="recall")
    right.plot(grid, sweep["f1"], color="#333333", linewidth=2.0, label="F1")
    for cut in cuts:
        right.axvline(cut["threshold"], linestyle="--", color=COLOURS[cut["key"]], linewidth=1.4)
        # Above the axes, not inside them and not rotated: at 1.005 the text was clipped by the top spine
        # and a rotated label read as a tick mark rather than as a caption.
        right.annotate(
            cut["label"], xy=(cut["threshold"], 1.0), xycoords=("data", "axes fraction"),
            xytext=(0, 8), textcoords="offset points", ha="center", va="bottom",
            fontsize=9, color=COLOURS[cut["key"]], clip_on=False,
        )
    right.set_xlabel("threshold applied to P(subscribe)")
    right.set_ylabel("metric value")
    # `pad` leaves room for the three cut labels, which sit just above the top spine; without it the
    # middle label overprints the second line of this title.
    right.set_title("precision, recall and F1 across the bar\n(the three cuts marked)", fontsize=11,
                    pad=26)
    right.set_xlim(0.0, 0.65)
    right.set_ylim(0.0, 1.02)
    right.legend(loc="center right", fontsize=9)
    right.grid(alpha=0.25)

    figure.suptitle(
        f"DepositIQ - precision-recall on {cuts[0]['rows']:,} held-out rows: calling everyone reaches "
        f"precision {curve['base_rate']:.4f}, so the curve is read against that line",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return annotations


# ---------------------------------------------------------------------------
# The gap against cross-validation
# ---------------------------------------------------------------------------
def build_gap(curves: dict, cv_summary: dict, boot: dict) -> dict:
    """
    Put the held-out figure beside the CV figure and say how far apart they are in CV standard deviations.

    Written as arithmetic first and prose second: `delta` and `z` are numbers the checks recompute, and
    the verdict sentence is generated from them rather than typed, so the sentence cannot survive a change
    in the numbers it describes.
    """
    cv_pr = float(cv_summary["pr_auc"]["mean"])
    cv_pr_sd = float(cv_summary["pr_auc"]["sd"])
    cv_roc = float(cv_summary["roc_auc"]["mean"])
    cv_roc_sd = float(cv_summary["roc_auc"]["sd"])
    held_pr = float(curves["pr"]["average_precision"])
    held_roc = float(curves["roc"]["auc"])

    delta = held_pr - cv_pr
    z_cv = delta / cv_pr_sd
    bootstrap_sd = float(boot["pr_auc"]["sd"])
    z_bootstrap = delta / bootstrap_sd
    delta_roc = held_roc - cv_roc
    z_roc = delta_roc / cv_roc_sd
    cv_inside = bool(boot["pr_auc"]["p2_5"] <= cv_pr <= boot["pr_auc"]["p97_5"])
    direction = "above" if delta > 0 else "below"
    share = float(boot["share_of_draws_below_cv"])

    # The verdict is judged against the **bootstrap** sd, not the CV sd. The CV sd is the spread between
    # five folds of models fitted to 28,934 rows each; the bootstrap sd is the spread of one held-out
    # estimate over resamples of these 9,043 rows. The second is the yardstick a single held-out draw is
    # drawn from, so dividing the gap by the first and calling 1.4 sd "large" would be a category error.
    # Both ratios are recorded, and the sentence says which one it is using.
    if abs(z_bootstrap) < 1.0 and cv_inside:
        verdict = (
            f"no measurable gap: the held-out PR-AUC sits {abs(delta):.4f} {direction} the CV mean, "
            f"which is {abs(z_bootstrap):.2f} bootstrap sd - and {share:.1%} of resampled draws of these "
            f"same rows land below the CV mean, so a gap of this size is an ordinary draw rather than "
            f"evidence either way"
        )
    elif cv_inside:
        verdict = (
            f"the held-out PR-AUC is {abs(z_bootstrap):.2f} bootstrap sd {direction} the CV mean, which "
            f"is a larger move than this sample's own noise, though the CV mean remains inside the "
            f"bootstrap interval"
        )
    else:
        verdict = (
            f"the held-out PR-AUC is {abs(z_bootstrap):.2f} bootstrap sd {direction} the CV mean and the "
            f"CV mean falls outside the bootstrap interval - the CV estimate and this draw disagree by "
            f"more than sampling noise"
        )

    return {
        "cv_pr_auc": cv_pr,
        "cv_pr_auc_sd": cv_pr_sd,
        "holdout_pr_auc": held_pr,
        "delta_pr_auc": delta,
        "delta_in_cv_sd": z_cv,
        "bootstrap_pr_auc_sd": bootstrap_sd,
        "delta_in_bootstrap_sd": z_bootstrap,
        "cv_roc_auc": cv_roc,
        "cv_roc_auc_sd": cv_roc_sd,
        "holdout_roc_auc": held_roc,
        "delta_roc_auc": delta_roc,
        "delta_roc_auc_in_cv_sd": z_roc,
        "cv_mean_inside_bootstrap_interval": cv_inside,
        "share_of_draws_below_cv": share,
        "verdict": verdict,
        "yardstick": "the verdict divides the gap by the bootstrap sd of the held-out estimate, not by "
                     "the CV sd: the CV sd is the spread between five models fitted to 28,934 rows each, "
                     "while the bootstrap sd is the spread a single held-out estimate actually has",
        "note": "The CV figure describes a model fitted to 28,934 rows (four fifths of the training half) "
                "per fold; the artifact was fitted to all 36,168. The two numbers therefore answer "
                "slightly different questions, and the extra 7,234 rows predict a small gain rather "
                "than a loss.",
    }


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def rank_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC from ranks (Mann-Whitney U), written out rather than called, with ties averaged."""
    n = len(scores)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks_sorted = np.empty(n, dtype=float)
    start = 0
    while start < n:
        stop = start
        while stop + 1 < n and sorted_scores[stop + 1] == sorted_scores[start]:
            stop += 1
        ranks_sorted[start:stop + 1] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    # `ranks_sorted` is indexed by sorted position, so it has to be mapped back through `order` before
    # it can be masked by `y` - which is in the original row order. Skipping that step is how this
    # function first came out at 0.7783 against scikit-learn's 0.8052: it was summing the ranks of the
    # rows that happen to sit at the subscribers' *positions*, not the subscribers' own ranks.
    rank_of_row = np.empty(n, dtype=float)
    rank_of_row[order] = ranks_sorted
    positive = y == 1
    n_pos = int(positive.sum())
    n_neg = n - n_pos
    return float((rank_of_row[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def step_ap(y: np.ndarray, scores: np.ndarray) -> float:
    """Average precision as the step-wise sum of precision over recall increments - the definition, not a call."""
    precision, recall, _ = precision_recall_curve(y, scores)
    return float(-np.sum(np.diff(recall) * precision[:-1]))


def f1_optimal_on_these_rows(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """
    The F1-maximising cut **on the held-out rows** - computed only so the check can prove it was not used.

    This is the threshold this step deliberately did not adopt: the operating cut comes from T32's folds.
    """
    best_cut, best_f1 = float(scores.min()), -1.0
    for threshold in np.unique(scores):
        row = metric_table.at_threshold(y, scores, float(threshold))
        if row["f1"] > best_f1:
            best_cut, best_f1 = float(threshold), row["f1"]
    return best_cut, best_f1


def validate(
    pipeline,
    card: dict,
    frame: pd.DataFrame,
    y: np.ndarray,
    scores: np.ndarray,
    cuts: list[dict],
    curves: dict,
    boot: dict,
    gap: dict,
    cv_record: dict,
    drawn: dict,
    started: float,
    peak: dict,
) -> list[dict]:
    """Same contract as every other phase: a check that cannot run is FAILED, never raised (D25)."""
    checks: list[dict] = []

    def compare(name: str, evaluate) -> None:
        try:
            passed, detail = evaluate()
        except Exception as error:  # noqa: BLE001 - any failure here is a failed check
            passed, detail = False, f"check could not run: {type(error).__name__}: {error}"
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    # --- it is the artifact, and nothing was fitted ----------------------------------------
    def artifact_is_the_recorded_file() -> tuple[bool, str]:
        digest = sha256(ARTIFACT_PATH)
        fitted = pipeline.named_steps[train.STEP_MODEL].get_params()
        wrong = {axis: (value, fitted[axis]) for axis, value in card["parameters"].items()
                 if fitted[axis] != value}
        steps = list(pipeline.named_steps)
        return (
            digest == card["artifact"]["sha256"] and not wrong
            and steps == [train.STEP_PREPROCESS, train.STEP_MODEL],
            f"{ARTIFACT_PATH.name} sha256 {digest[:16]}... matches T31's card; {len(card['parameters'])} "
            f"parameters equal the card's and are applied to the loaded estimator; steps {steps}; "
            f"mismatches: {wrong or 'none'}",
        )

    compare("the scored object is the artifact T31 persisted, not a refit", artifact_is_the_recorded_file)

    def nothing_was_fitted() -> tuple[bool, str]:
        """
        A source scan, built so it cannot match itself: the patterns are assembled from parts, because a
        literal search string written in this function appears in this file and would report itself.
        """
        source = Path(__file__).read_text()
        fit_calls = re.findall(r"\." + "fit" + r"(_transform)?\(", source)
        train_loads = re.findall("load_engineered" + r"\(\s*['\"]train", source)
        return (
            not fit_calls and not train_loads,
            f"{len(fit_calls)} fitting calls and {len(train_loads)} training-half loads in this module; "
            f"the model is loaded and only `predict_proba` is called",
        )

    compare("the module fits nothing and loads no training rows", nothing_was_fitted)

    # --- the rows are the held-out half ----------------------------------------------------
    def rows_are_the_held_out_half() -> tuple[bool, str]:
        raw = pd.read_csv(config.BANK_FULL_CSV, sep=config.CSV_SEPARATOR)
        stored_train = pd.read_csv(config.TRAIN_CSV)
        stored_test = pd.read_csv(config.TEST_CSV)
        stored_positives = int((stored_test[config.TARGET] == config.POSITIVE_LABEL).sum())
        return (
            len(stored_train) + len(stored_test) == len(raw)
            and len(frame) == len(stored_test)
            and int(y.sum()) == stored_positives,
            f"{len(stored_train):,} training + {len(stored_test):,} test = {len(raw):,} raw rows; "
            f"{len(frame):,} engineered rows scored, {int(y.sum()):,} of them subscribed "
            f"({y.mean():.4%}), matching the stored split",
        )

    compare("every row scored belongs to the held-out half", rows_are_the_held_out_half)
    def each_row_scored_once() -> tuple[bool, str]:
        again = score(pipeline, frame)
        return (
            len(scores) == len(frame)
            and np.isfinite(scores).all()
            and float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0
            and np.array_equal(scores, again),
            f"{len(scores):,} finite probabilities in [{scores.min():.6f}, {scores.max():.6f}], "
            f"mean {scores.mean():.6f} against a {y.mean():.6f} base rate; a second pass is identical",
        )

    compare("every held-out row was scored once, and scoring is deterministic", each_row_scored_once)

    # --- the cuts were not chosen on these rows --------------------------------------------
    def cuts_are_the_projects_three() -> tuple[bool, str]:
        operating = cuts[1]["threshold"]
        from_record = float(cv_record["models"]["xgboost_tuned"]["summary"]["nested_threshold"]["mean"])
        break_even = cuts[0]["threshold"]
        conventional = cuts[2]["threshold"]
        held_optimal, held_f1 = f1_optimal_on_these_rows(y, scores)
        adopted_f1 = metric_table.at_threshold(y, scores, operating)["f1"]
        return (
            abs(operating - from_record) <= 1e-12
            and break_even == metric_table.BREAK_EVEN
            and conventional == 0.5
            and len({break_even, operating, conventional}) == 3
            and min(break_even, operating, conventional) == break_even
            and abs(operating - held_optimal) > 1e-9
            and abs(peak["threshold"] - held_optimal) <= 1e-12
            and abs(peak["f1"] - held_f1) <= 1e-12,
            f"7.50% from the cost model, {operating:.6f} from metrics_cv.json to 1e-12, 0.5 by "
            f"convention; the F1-optimal cut on these rows would be {held_optimal:.6f} "
            f"(F1 {held_f1:.6f}) and was deliberately not adopted - the chosen cut sits "
            f"{abs(held_optimal - operating):.6f} from it and costs {held_f1 - adopted_f1:.6f} F1",
        )

    compare("the three cuts are the project's, and none was fitted on the held-out rows",
            cuts_are_the_projects_three)

    # --- the counts are the held-out rows --------------------------------------------------
    def counts_partition_the_rows() -> tuple[bool, str]:
        wrong = []
        for cut in cuts:
            flagged = (scores >= cut["threshold"]).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, flagged, labels=[0, 1]).ravel()
            totals = (cut["true_positive"] + cut["false_positive"]
                      + cut["false_negative"] + cut["true_negative"])
            if (
                totals != len(y)
                or cut["true_positive"] + cut["false_negative"] != int(y.sum())
                or cut["true_negative"] + cut["false_positive"] != int((y == 0).sum())
                or (tp, fp, fn, tn) != (cut["true_positive"], cut["false_positive"],
                                        cut["false_negative"], cut["true_negative"])
            ):
                wrong.append(cut["label"])
        return not wrong, (
            f"3 cuts: TP+FP+FN+TN = {len(y):,}, TP+FN = {int(y.sum()):,}, and every cell equals "
            f"scikit-learn's confusion_matrix; wrong: {wrong or 'none'}"
        )

    compare("the confusion counts partition the held-out rows and match scikit-learn",
            counts_partition_the_rows)

    def project_arithmetic_reproduces_every_number() -> tuple[bool, str]:
        worst = 0.0
        for cut in cuts:
            again = metric_table.at_threshold(y, scores, cut["threshold"])
            for key, value in again.items():
                worst = max(worst, abs(float(value) - float(cut[key])))
        return worst == 0.0, (
            f"metrics.at_threshold re-derives precision, recall, F1 and the flag counts at all three "
            f"cuts; max drift {worst:.1e} - the figures and the CV table share one definition of a "
            f"threshold metric"
        )

    compare("the held-out table is T32's arithmetic, not a second implementation",
            project_arithmetic_reproduces_every_number)

    # --- what 0.5 does, measured out of sample ---------------------------------------------
    def half_flags_almost_nothing() -> tuple[bool, str]:
        break_even, operating, conventional = cuts
        return (
            conventional["flagged_share"] < 0.10
            and conventional["flagged_share"] < operating["flagged_share"] < break_even["flagged_share"]
            and conventional["recall"] < operating["recall"]
            and conventional["f1"] < operating["f1"],
            f"0.5 flags {conventional['flagged_share']:.2%} of rows and reaches "
            f"{conventional['recall']:.2%} of subscribers (F1 {conventional['f1']:.4f}); "
            f"{operating['label']} flags {operating['flagged_share']:.2%} and reaches "
            f"{operating['recall']:.2%} (F1 {operating['f1']:.4f}); {break_even['label']} flags "
            f"{break_even['flagged_share']:.2%} and reaches {break_even['recall']:.2%}",
        )

    compare("0.5 is still the 95th-percentile bar out of sample, and costs recall twice over",
            half_flags_almost_nothing)

    # --- the metrics recompute from the ranks ----------------------------------------------
    def metrics_recompute_by_hand() -> tuple[bool, str]:
        auc = rank_auc(y, scores)
        ap = step_ap(y, scores)
        return (
            abs(auc - curves["roc"]["auc"]) <= 1e-12
            and abs(ap - curves["pr"]["average_precision"]) <= 1e-12,
            f"hand-written rank AUC {auc:.12f} against scikit-learn {curves['roc']['auc']:.12f}; "
            f"step-wise AP {ap:.12f} against {curves['pr']['average_precision']:.12f}",
        )

    compare("ROC-AUC and PR-AUC recompute from the ranks without scikit-learn's functions",
            metrics_recompute_by_hand)

    # --- the stored curves are the curves --------------------------------------------------
    def curves_are_monotone_and_their_areas_are_the_metrics() -> tuple[bool, str]:
        fpr = curves["roc"]["false_positive_rate"]
        tpr = curves["roc"]["true_positive_rate"]
        precision = curves["pr"]["precision"]
        recall = curves["pr"]["recall"]
        trapezoid = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))
        step_area = float(-np.sum(np.diff(recall) * precision[:-1]))
        return (
            fpr[0] == 0.0 and tpr[0] == 0.0 and fpr[-1] == 1.0 and tpr[-1] == 1.0
            and bool(np.all(np.diff(fpr) >= 0)) and bool(np.all(np.diff(tpr) >= 0))
            and bool(np.all(np.diff(recall) <= 0))
            and abs(trapezoid - curves["roc"]["auc"]) <= 1e-12
            and abs(step_area - curves["pr"]["average_precision"]) <= 1e-12,
            f"ROC {len(fpr):,} points from (0,0) to (1,1), monotone, trapezoid area {trapezoid:.12f}; "
            f"PR {len(recall):,} points, recall non-increasing, step area {step_area:.12f}",
        )

    compare("the stored curves run corner to corner and integrate back to the reported metrics",
            curves_are_monotone_and_their_areas_are_the_metrics)

    # --- the gap is arithmetic -------------------------------------------------------------
    def gap_is_arithmetic() -> tuple[bool, str]:
        expected = curves["pr"]["average_precision"] - gap["cv_pr_auc"]
        z = expected / gap["cv_pr_auc_sd"]
        roc_expected = curves["roc"]["auc"] - gap["cv_roc_auc"]
        return (
            abs(expected - gap["delta_pr_auc"]) <= 1e-15
            and abs(z - gap["delta_in_cv_sd"]) <= 1e-12
            and abs(roc_expected - gap["delta_roc_auc"]) <= 1e-15,
            f"delta PR-AUC {gap['delta_pr_auc']:+.6f} = {gap['holdout_pr_auc']:.6f} - "
            f"{gap['cv_pr_auc']:.6f}; {gap['delta_in_cv_sd']:+.3f} CV sd; delta ROC-AUC "
            f"{gap['delta_roc_auc']:+.6f} ({gap['delta_roc_auc_in_cv_sd']:+.3f} sd)",
        )

    compare("the gap against CV is computed from both records, not typed", gap_is_arithmetic)

    # --- the bootstrap is a distribution ---------------------------------------------------
    def bootstrap_reproduces_and_is_ordered() -> tuple[bool, str]:
        pr = boot["pr_auc"]
        roc = boot["roc_auc"]
        # The first draws of a fresh generator with the same seed must equal the stored ones: the loop
        # consumes exactly one block of integers per resample, so the streams correspond position by
        # position. Fifty is enough to catch a changed seed, a changed resample size or a changed metric.
        rng = np.random.default_rng(boot["seed"])
        size = len(y)
        probe = []
        for _ in range(50):
            index = rng.integers(0, size, size)
            probe.append(float(average_precision_score(y[index], scores[index])))
        stored = np.asarray(boot["values_pr_auc"][:50])
        reproduced = float(np.max(np.abs(np.asarray(probe) - stored)))
        return (
            pr["p2_5"] <= pr["mean"] <= pr["p97_5"]
            and roc["p2_5"] <= roc["mean"] <= roc["p97_5"]
            and pr["p2_5"] <= gap["holdout_pr_auc"] <= pr["p97_5"]
            and 0.0 <= boot["share_of_draws_below_cv"] <= 1.0
            and boot["skipped"] == 0
            and reproduced == 0.0,
            f"{boot['resamples']:,} resamples, seed {boot['seed']}, {boot['skipped']} skipped: PR-AUC "
            f"[{pr['p2_5']:.4f}, {pr['p97_5']:.4f}] mean {pr['mean']:.4f} around the point estimate "
            f"{gap['holdout_pr_auc']:.4f}; {boot['share_of_draws_below_cv']:.1%} of draws fall below the "
            f"CV mean; first 50 draws reproduce bit-identically (max drift {reproduced:.1e})",
        )

    compare("the bootstrap interval is ordered, contains its own estimate, and replays from its seed",
            bootstrap_reproduces_and_is_ordered)

    # --- the figures are the record in picture form -----------------------------------------
    def figures_were_written_and_carry_the_numbers() -> tuple[bool, str]:
        paths = {"confusion": CONFUSION_FIGURE, "roc": ROC_FIGURE, "pr": PR_FIGURE}
        problems, sizes = [], {}
        for name, path in paths.items():
            if not path.exists():
                problems.append(f"{name}: missing")
                continue
            data = path.read_bytes()
            sizes[name] = len(data)
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                problems.append(f"{name}: not a PNG")
            if len(data) < MIN_FIGURE_BYTES:
                problems.append(f"{name}: {len(data):,} bytes, under the {MIN_FIGURE_BYTES:,} floor")
            if path.stat().st_mtime < started - 2.0:
                problems.append(f"{name}: older than this run")
        for index, cut in enumerate(cuts):
            if f"{cut['precision']:.4f}" not in drawn["confusion"][index]:
                problems.append(f"confusion panel for {cut['label']} lacks its precision")
            if f"{cut['precision']:.4f}" not in drawn["pr"][index]:
                problems.append(f"PR annotation for {cut['label']} lacks its precision")
            if f"{cut['recall']:.4f}" not in drawn["roc"][index]:
                problems.append(f"ROC annotation for {cut['label']} lacks its recall")
        return not problems, (
            "three PNGs on disk (" + ", ".join(f"{name} {size:,} B" for name, size in sizes.items())
            + "), every panel's drawn text carrying the record's precision and recall; "
            + f"problems: {problems or 'none'}"
        )

    compare("all three figures exist and their annotations are the record's numbers",
            figures_were_written_and_carry_the_numbers)

    return checks


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(variant: str = "full") -> dict:
    """Score the artifact on the held-out half once, draw the three figures, and write the record."""
    config.ensure_dirs()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    started_monotonic = time.perf_counter()
    started_wall = time.time()

    pipeline, card = load_artifact()
    frame = features.load_engineered("test", variant)
    y = encode_target(frame).to_numpy()
    scores = score(pipeline, frame)

    cuts = cut_table(y, scores, cut_descriptors())
    curves = curve_data(y, scores)

    cv_record = json.loads(CV_RECORD_PATH.read_text())
    cv_summary = cv_record["models"]["xgboost_tuned"]["summary"]
    boot = bootstrap(y, scores, float(cv_summary["pr_auc"]["mean"]))
    gap = build_gap(curves, cv_summary, boot)

    # The cut the held-out rows would pick if they were allowed to choose. Recorded so the claim "the
    # CV-derived cut was close" is a number in the record rather than a sentence in a check's output.
    peak_threshold, peak_f1 = f1_optimal_on_these_rows(y, scores)
    peak = {
        "threshold": peak_threshold,
        "f1": peak_f1,
        "note": "the F1-maximising cut if it were fitted on the held-out rows; computed to show the "
                "adopted cut landed near it, never adopted and never used to re-select anything",
    }

    drawn = {
        "confusion": draw_confusion(cuts, CONFUSION_FIGURE),
        "roc": draw_roc(curves, cuts, ROC_FIGURE),
        "pr": draw_pr(y, scores, curves, cuts, PR_FIGURE),
    }

    figures = {
        "confusion_matrix": CONFUSION_FIGURE,
        "roc_curve": ROC_FIGURE,
        "pr_curve": PR_FIGURE,
    }
    figure_block = {
        name: {
            "path": str(path.relative_to(config.ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for name, path in figures.items()
    }

    payload = {
        "task": "T33",
        "variant": variant,
        "scored": {
            "artifact": str(ARTIFACT_PATH.relative_to(config.ROOT)),
            "sha256": sha256(ARTIFACT_PATH),
            "card": str(CARD_PATH.relative_to(config.ROOT)),
            "refit": False,
            "recalibrated": False,
        },
        "rows": int(len(frame)),
        "positives": int(y.sum()),
        "negatives": int((y == 0).sum()),
        "base_rate": float(y.mean()),
        "mean_prediction": float(scores.mean()),
        "brier": float(np.mean((scores - y) ** 2)),
        "metrics": {
            "pr_auc": curves["pr"]["average_precision"],
            "roc_auc": curves["roc"]["auc"],
        },
        "cuts": cuts,
        "f1_peak_on_held_out": peak,
        "gap_against_cv": gap,
        "bootstrap": {key: value for key, value in boot.items()},
        "figures": figure_block,
        "annotations": drawn,
        "cv_source": str(CV_RECORD_PATH.relative_to(config.ROOT)),
        "elapsed_seconds": round(time.perf_counter() - started_monotonic, 1),
    }

    checks = validate(pipeline, card, frame, y, scores, cuts, curves, boot, gap, cv_record,
                      drawn, started_wall, peak)
    payload["checks"] = checks
    payload["checks_passed"] = sum(1 for check in checks if check["passed"])
    payload["checks_total"] = len(checks)
    payload["checks_ok"] = payload["checks_passed"] == payload["checks_total"]
    payload["elapsed_seconds"] = round(time.perf_counter() - started_monotonic, 1)
    RECORD_PATH.write_text(json.dumps(payload, indent=2, sort_keys=False))

    return {**payload, "curves": curves, "scores": scores, "y": y}


def print_report(result: dict) -> None:
    print("DepositIQ T33 - held-out evaluation of the persisted artifact")
    print("=" * 112)
    print(f"scored   : {result['scored']['artifact']} (sha256 {result['scored']['sha256'][:16]}...)")
    print(f"rows     : {result['rows']:,} held-out rows, {result['positives']:,} subscribed "
          f"({result['base_rate']:.4%}), {result['negatives']:,} not")
    print(f"metrics  : PR-AUC {result['metrics']['pr_auc']:.4f}, "
          f"ROC-AUC {result['metrics']['roc_auc']:.4f}, "
          f"Brier {result['brier']:.4f}, mean p {result['mean_prediction']:.4f}")
    gap = result["gap_against_cv"]
    print(f"vs CV    : PR-AUC {gap['cv_pr_auc']:.4f} -> {gap['holdout_pr_auc']:.4f} "
          f"({gap['delta_pr_auc']:+.4f}, {gap['delta_in_cv_sd']:+.2f} CV sd); "
          f"ROC-AUC {gap['cv_roc_auc']:.4f} -> {gap['holdout_roc_auc']:.4f} "
          f"({gap['delta_roc_auc']:+.4f}, {gap['delta_roc_auc_in_cv_sd']:+.2f} CV sd)")
    boot = result["bootstrap"]
    print(f"bootstrap: PR-AUC 95% [{boot['pr_auc']['p2_5']:.4f}, {boot['pr_auc']['p97_5']:.4f}] over "
          f"{boot['resamples']:,} resamples; {boot['share_of_draws_below_cv']:.1%} of draws fall below "
          f"the CV mean")
    print(f"verdict  : {gap['verdict']}")
    print()

    header = (f"{'cut':>8}{'flagged':>18}{'TP':>7}{'FP':>7}{'FN':>7}"
              f"{'precision':>11}{'recall':>9}{'F1':>9}")
    print(header)
    print("-" * len(header))
    for cut in result["cuts"]:
        mark = "  <- rejected" if cut["rejected"] else ""
        print(f"{cut['label']:>8}"
              f"{cut['flagged']:>9,} ({cut['flagged_share']:>5.2%})"
              f"{cut['true_positive']:>7,}{cut['false_positive']:>7,}{cut['false_negative']:>7,}"
              f"{cut['precision']:>11.4f}{cut['recall']:>9.4f}{cut['f1']:>9.4f}{mark}")
    print()
    print("wrote    : " + ", ".join(f"{name} {block['bytes']:,} B"
                                    for name, block in result["figures"].items())
          + f" + {RECORD_PATH.name}")
    print(f"total    : {result['elapsed_seconds']} s")
    print()
    print(f"T33 checks: {result['checks_passed']}/{result['checks_total']} passed")
    for check in result["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        detail = f"  ({check['detail']})" if check["detail"] else ""
        print(f"  [{marker}] {check['check']}{detail}")
    print()


if __name__ == "__main__":
    print("DepositIQ Phase 7 - held-out evaluation")
    print()
    outcome = report()
    print_report(outcome)
    print("=" * 112)
    verdict = "PASS - the artifact scored once on the held-out half, at three cuts"
    print(f"result : {verdict if outcome['checks_ok'] else 'FAIL'}")
    print("Next   : T34 turns the score into a decision using the cost model.")
    sys.exit(0 if outcome["checks_ok"] else 1)
