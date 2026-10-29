"""
DepositIQ - lift, gain and deciles: what the ranking is worth per hundred calls (T35).

Why this file exists
--------------------
T34 fixed the operating threshold at the price of a call, **0.0750**, and reported it as money:
43,621 EUR on the training half, 10,988.50 EUR held out. A bank's campaign manager does not spend euros,
though - they spend **calls**, and they are asked how much better the list is than working down the
customer file in any order. That question has two standard answers, and this module produces both:

* **lift** - in the best 10% of the list, how many times the base subscription rate do we find?
* **gain** - what share of all subscribers do we reach by calling the best X% of the list?

Neither needs a threshold, which is what makes this the one Phase 7 step that can report on the held-out
rows without spending them (D53, D54). The threshold enters only at the end, where the adopted cut is
translated into the quantity a bank can book: **calls saved at equal recall**.

The two halves, and why they are never pooled
---------------------------------------------
| half | rows | subscribers | base rate |
|---|---|---|---|
| training (out-of-fold) | 36,168 | 4,231 | 11.6982% |
| held out (artifact) | 9,043 | 1,058 | 11.6997% |

The rates are 1.5 basis points apart, which is exactly why they must not be mixed: lift is a ratio to a
base rate, and a pooled denominator would belong to neither half. Every table below states its half and
divides by that half's own rate.

Deciles are of the **ranking**, not of the probability
-----------------------------------------------------
Rows are sorted by score and cut into ten equal-count blocks by position, so a decile is "the best 3,616
clients" rather than "clients whose p exceeds some number". Positional blocks are the convention because
they survive ties and a skewed score distribution unchanged; the consequence is that a one-row difference
in bin size is normal (10 blocks over 36,168 rows cannot be equal to the row).

How to run
----------
    python -m src.lift      # ~6 s: replays T34's five fold fits, scores the artifact, draws the figure

Writes `reports/lift_table.csv`, `reports/figures/08_lift_gain.png` and `models/lift_results.json`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config, cost, features, select, threshold, train
from src.split import encode_target

RECORD_PATH: Path = config.MODELS_DIR / "lift_results.json"
TABLE_PATH: Path = config.REPORTS_DIR / "lift_table.csv"
FIGURE_PATH: Path = config.FIGURES_DIR / "08_lift_gain.png"
ARTIFACT_PATH: Path = config.MODELS_DIR / "final_model.joblib"
THRESHOLD_PATH: Path = config.MODELS_DIR / "threshold_choice.json"
CV_PATH: Path = config.MODELS_DIR / "metrics_cv.json"

DECILES: int = 10

# D15's declared KPI: the top decile must find more than twice the base rate.
D15_LIFT_TARGET: float = 2.0

# T32's two published multipliers and the coverage each was measured at. This module has to land on them
# or explain why not, rather than introduce a third number for the same thing.
T32_POINTS: tuple[tuple[str, float, float, float], ...] = (
    # label, coverage, published lift, tolerance
    ("nested cut (13.26% coverage)", 0.132604, 3.9247, 0.15),
    ("business cut (41.10% coverage)", 0.411027, 1.8961, 0.05),
)

# Recall levels the calls-saved table is reported at: the adopted cut's, plus the decile boundaries.
SAVENOTE_RECALLS: tuple[float, ...] = (0.25, 0.50, 0.75, 0.90)

SEED: int = config.RANDOM_STATE


# ---------------------------------------------------------------------------
# Scores for each half
# ---------------------------------------------------------------------------
def half_scores() -> dict[str, dict]:
    """
    Out-of-fold scores for the training half and artifact scores for the held-out half.

    The training half goes through **T34's** `threshold.oof_scores` rather than a second copy of the fold
    loop, so the ranking behind every lift number here is bit-identical to the ranking behind T34's money
    (D45: one harness, or the two steps are separate experiments).
    """
    frame = features.load_engineered("train")
    y = encode_target(frame)
    folds = train.materialise_folds(frame, y)
    columns = select.shipped_features()
    scores, info = threshold.oof_scores(frame, y, folds, columns)

    pipeline = joblib.load(ARTIFACT_PATH)
    test = features.load_engineered("test")
    y_test = encode_target(test).to_numpy()
    p_test = pipeline.predict_proba(test)[:, 1].astype(np.float64)

    return {
        "train_oof": {"y": y.to_numpy(), "scores": scores, "folds": info, "label": "training (out-of-fold)"},
        "held_out": {"y": y_test, "scores": p_test, "folds": None, "label": "held out (artifact)"},
    }


# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------
def decile_table(y: np.ndarray, scores: np.ndarray, half: str) -> list[dict]:
    """
    Ten positional blocks of the descending ranking, with per-block and running columns.

    `lift = precision / base_rate` is the only definition used anywhere in this module; `cumulative_lift`
    is the same ratio for the top-k list, which is what a campaign that calls the best k clients earns.
    """
    y = np.asarray(y).astype(int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    order = np.argsort(-scores, kind="mergesort")     # stable: ties keep their original order
    ranked = y[order]

    total = len(y)
    positives = int(y.sum())
    base = positives / total
    edges = [round(total * index / DECILES) for index in range(DECILES + 1)]

    rows: list[dict] = []
    run_rows = 0
    run_hits = 0
    for block in range(DECILES):
        slab = ranked[edges[block]:edges[block + 1]]
        block_rows = int(len(slab))
        hits = int(slab.sum())
        run_rows += block_rows
        run_hits += hits
        rows.append({
            "half": half,
            "decile": block + 1,
            "rows": block_rows,
            "subscribers": hits,
            "precision": hits / block_rows if block_rows else float("nan"),
            "lift": (hits / block_rows) / base if block_rows else float("nan"),
            "cumulative_rows": run_rows,
            "cumulative_subscribers": run_hits,
            "coverage": run_rows / total,
            "cumulative_share": run_hits / positives,
            "cumulative_precision": run_hits / run_rows,
            "cumulative_lift": (run_hits / run_rows) / base,
            # The business quantity: a random campaign reaching the same share of subscribers would have
            # to call `share x total` clients, so this is how many calls the ranking saves at that recall.
            "calls_saved_at_equal_recall": run_hits / positives * total - run_rows,
        })
    return rows


def at_coverage(table: list[dict], coverage: float) -> dict:
    """
    Interpolate the cumulative curve at an arbitrary coverage, so T32's measured points can be compared.

    Linear between the two bracketing decile edges. Interpolation is stated rather than pretending the
    published coverage was one of the ten block edges - T32's 13.26% sits inside decile 2.
    """
    point = {"coverage": float(coverage)}
    previous = {"coverage": 0.0, "cumulative_share": 0.0, "cumulative_precision": 0.0,
                "cumulative_lift": 1.0, "calls_saved_at_equal_recall": 0.0}
    for row in table:
        if coverage <= row["coverage"]:
            span = row["coverage"] - previous["coverage"]
            weight = (coverage - previous["coverage"]) / span if span else 0.0
            for key in ("cumulative_share", "cumulative_precision", "cumulative_lift"):
                point[key] = previous[key] + weight * (row[key] - previous[key])
            return point
        previous = row
    for key in ("cumulative_share", "cumulative_precision", "cumulative_lift"):
        point[key] = table[-1][key]
    return point


def random_self_test(y: np.ndarray, seed: int = SEED) -> dict:
    """
    Run the decile machinery on a **shuffled** score vector.

    A lift table is an easy thing to get subtly wrong, and the failure is silent: an off-by-one edge or a
    descending sort mistake still produces plausible-looking numbers. With the scores removed, every lift
    must be 1.0 within sampling error and the cumulative gain must equal the diagonal. This tests the
    code, not the model, and it runs on every invocation.
    """
    rng = np.random.default_rng(seed)
    noise = rng.random(len(y))
    table = decile_table(y, noise, "random")
    lifts = [row["lift"] for row in table]
    worst_gain = max(abs(row["cumulative_share"] - row["coverage"]) for row in table)
    worst_saved = max(abs(row["calls_saved_at_equal_recall"]) for row in table)
    return {
        "lifts": lifts,
        "min_lift": float(min(lifts)),
        "max_lift": float(max(lifts)),
        "mean_lift": float(np.mean(lifts)),
        "worst_gain_deviation": float(worst_gain),
        "worst_calls_saved": float(worst_saved),
    }


# ---------------------------------------------------------------------------
# The adopted cut, in calls
# ---------------------------------------------------------------------------
def cut_translation(halves: dict[str, dict], tables: dict[str, list[dict]]) -> dict:
    """
    What T34's 0.0750 buys, expressed as calls rather than euros.

    Two readings, both about the same list: at the cut the campaign reaches R% of subscribers using C% of
    the list, so a random campaign reaching the same R% would need `R x N` calls - the difference is the
    saving, and `saving x C_call` is what it is worth before any subscriber is credited to it.
    """
    out: dict[str, dict] = {}
    for half, data in halves.items():
        y, scores = data["y"], data["scores"]
        total = len(y)
        called = int((scores >= threshold.CHOSEN).sum())
        reached = int(y[scores >= threshold.CHOSEN].sum())
        recall = reached / int(y.sum())
        precision = reached / called
        base = y.mean()
        saved = recall * total - called
        out[half] = {
            "threshold": threshold.CHOSEN,
            "rows": int(total),
            "positives": int(y.sum()),
            "calls": called,
            "coverage": called / total,
            "subscribers_reached": reached,
            "recall": recall,
            "precision": precision,
            "lift": precision / base,
            "calls_a_fixed_recall_would_need": recall * total,
            "calls_saved": saved,
            "calls_saved_share": saved / total,
            "euros_not_spent": saved * cost.CALL_COST_EUR,
            "subscribers_foregone": int(y.sum()) - reached,
        }
    return out


def recall_to_calls(y: np.ndarray, scores: np.ndarray, recalls: tuple[float, ...] = SAVENOTE_RECALLS) -> list[dict]:
    """For each target recall, the calls the ranking needs against the calls a random order needs."""
    order = np.argsort(-scores, kind="mergesort")
    ranked = np.asarray(y).astype(int).ravel()[order]
    cumulative = np.cumsum(ranked)
    positives = int(ranked.sum())
    total = len(ranked)
    rows = []
    for recall in recalls:
        needed = int(np.searchsorted(cumulative, recall * positives, side="left")) + 1
        rows.append({
            "target_recall": float(recall),
            "calls": needed,
            "coverage": needed / total,
            "calls_random_would_need": recall * total,
            "calls_saved": recall * total - needed,
            "euros_not_spent": (recall * total - needed) * cost.CALL_COST_EUR,
        })
    return rows


def savings_peak(y: np.ndarray, scores: np.ndarray) -> dict:
    """
    The coverage at which **calls saved at equal recall** peaks, found exactly rather than interpolated.

    Writing `saved(k) = gain(k) x N - k` for the best k clients makes the marginal step visible:

        saved(k) - saved(k-1) = (N / positives) x y[k] - 1

    so each call adds a saving when the client subscribes and costs one call when they do not. The saving
    therefore peaks where the **local** subscription rate crosses the list average (`N / positives`), not
    where it crosses the price. That is the difference between this curve's peak and T34's: efficiency is
    measured against the average client, value against the marginal one.
    """
    ranked = np.asarray(y).astype(int).ravel()[np.argsort(-np.asarray(scores), kind="mergesort")]
    total = len(ranked)
    positives = int(ranked.sum())
    running = np.cumsum(ranked)
    saved = running / positives * total - np.arange(1, total + 1)
    peak = int(np.argmax(saved)) + 1
    return {
        "calls": peak,
        "coverage": peak / total,
        "subscribers_reached": int(running[peak - 1]),
        "gain": float(running[peak - 1] / positives),
        "precision_at_peak": float(running[peak - 1] / peak),
        "lift_at_peak": float((running[peak - 1] / peak) / (positives / total)),
        "calls_saved": float(saved[peak - 1]),
        "euros_not_spent": float(saved[peak - 1] * cost.CALL_COST_EUR),
        "base_rate": positives / total,
        "is_local_max": bool(
            saved[peak - 1] >= (saved[peak - 2] if peak >= 2 else -np.inf)
            and saved[peak - 1] >= (saved[peak] if peak < total else -np.inf)
        ),
        "per_call_at_peak": float(saved[peak - 1] / peak),
    }


def marginal_block(peak: dict, cut: dict) -> dict:
    """
    The clients between the savings peak and the adopted cut, priced at the marginal rate.

    This is the argument that keeps the two curves from contradicting each other. The savings peak is where
    efficiency against the *average* client stops improving; it is not where the *next* client stops being
    worth calling. The block between the two points converts at

        rate = (reached at cut - reached at peak) / (calls at cut - calls at peak)

    and the campaign is still worth running through it iff that rate exceeds `C/V`.
    """
    calls = cut["calls"] - peak["calls"]
    hits = cut["subscribers_reached"] - peak["subscribers_reached"]
    rate = hits / calls if calls else float("nan")
    return {
        "calls": calls,
        "subscribers": hits,
        "marginal_rate": rate,
        "price": threshold.CHOSEN,
        "above_price": bool(rate > threshold.CHOSEN),
        "net_per_call_eur": rate * cost.VALUE_PER_SUBSCRIPTION_EUR - cost.CALL_COST_EUR,
        "net_for_the_block_eur": hits * cost.VALUE_PER_SUBSCRIPTION_EUR - calls * cost.CALL_COST_EUR,
    }


def reconciliation(tables: dict[str, list[dict]]) -> list[dict]:
    """T32's two published multipliers, re-derived from this module's ranking at its coverage."""
    rows = []
    for label, coverage, published, tolerance in T32_POINTS:
        measured = at_coverage(tables["train_oof"], coverage)
        rows.append({
            "label": label,
            "coverage": coverage,
            "published_lift": published,
            "measured_lift": measured["cumulative_lift"],
            "measured_share": measured["cumulative_share"],
            "difference": measured["cumulative_lift"] - published,
            "tolerance": tolerance,
            "agrees": abs(measured["cumulative_lift"] - published) <= tolerance,
        })
    return rows


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------
def draw(tables: dict[str, list[dict]], translation: dict[str, dict], peaks: dict[str, dict],
         path: Path = FIGURE_PATH) -> Path:
    """Lift by decile, cumulative gain, and the calls a campaign saves - the same list, three views."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, (left, middle, right) = plt.subplots(1, 3, figsize=(17.5, 6.0))
    colours = {"train_oof": "#4C78A8", "held_out": "#F58518"}
    names = {"train_oof": "training (out-of-fold)", "held_out": "held out (artifact)"}

    deciles = np.arange(1, DECILES + 1)
    width = 0.38
    for offset, half in ((-width / 2, "train_oof"), (width / 2, "held_out")):
        lifts = [row["lift"] for row in tables[half]]
        left.bar(deciles + offset, lifts, width=width, color=colours[half], alpha=0.9,
                 label=names[half])
        for x, value in zip(deciles, lifts):
            left.text(x + offset, value + 0.06, f"{value:.2f}", ha="center", fontsize=7.2,
                      color=colours[half])
    left.axhline(D15_LIFT_TARGET, color="#E45756", linestyle="--", linewidth=1.3,
                 label=f"D15 target {D15_LIFT_TARGET:.0f}x")
    left.axhline(1.0, color="#666666", linestyle=":", linewidth=1.2, label="no better than random")
    left.set_xlabel("decile of the ranking (1 = highest scores)")
    left.set_ylabel("lift = decile's subscription rate / base rate")
    left.set_title("lift by decile\nwhere the value is concentrated", fontsize=11)
    left.set_xticks(deciles)
    left.legend(fontsize=8.5)
    left.grid(axis="y", alpha=0.25)

    middle.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1.2,
                label="a random list (the diagonal)")
    for half in ("train_oof", "held_out"):
        coverage = [0.0] + [row["coverage"] for row in tables[half]]
        share = [0.0] + [row["cumulative_share"] for row in tables[half]]
        middle.plot(coverage, share, marker="o", markersize=5, linewidth=1.8, color=colours[half],
                    label=names[half])
    cut = translation["train_oof"]
    middle.scatter([cut["coverage"]], [cut["recall"]], s=140, marker="*", color="#54A24B", zorder=6,
                   edgecolor="white", linewidth=0.8,
                   label=f"adopted cut {cut['threshold']:.4f}")
    middle.annotate(f"{cut['recall']:.1%} of subscribers\nfor {cut['coverage']:.1%} of the list",
                    xy=(cut["coverage"], cut["recall"]), xytext=(52, -128),
                    textcoords="offset points", fontsize=8.5, color="#2E7D32",
                    arrowprops={"arrowstyle": "->", "color": "#2E7D32", "linewidth": 1.0})
    middle.set_xlabel("share of the list called")
    middle.set_ylabel("share of all subscribers reached")
    middle.set_title("cumulative gain\nthe same ranking, read as a curve", fontsize=11)
    middle.set_xlim(0, 1)
    middle.set_ylim(0, 1.02)
    middle.legend(loc="lower right", fontsize=8.5)
    middle.grid(alpha=0.25)

    # Normalised by each half's own list size: raw counts would put 36,168 rows and 9,043 on the same axis
    # and make the held-out curve look flat for the wrong reason.
    for half in ("train_oof", "held_out"):
        rows = sum(row["rows"] for row in tables[half])
        coverage = [row["coverage"] for row in tables[half]]
        saved = [row["calls_saved_at_equal_recall"] / rows for row in tables[half]]
        right.plot(coverage, saved, marker="o", markersize=5, linewidth=1.8, color=colours[half],
                   label=names[half])
    peak = peaks["train_oof"]
    cut_share = cut["calls_saved"] / cut["rows"]
    top = max(cut_share, peak["calls_saved"] / cut["rows"]) * 1.25
    right.axvline(peak["coverage"], color="#B279A2", linestyle=":", linewidth=1.5)
    right.text(peak["coverage"] + 0.01, top * 0.97, f"savings peak\n{peak['coverage']:.1%}",
               fontsize=8.5, color="#8E5A82", va="top", ha="left")
    right.axvline(cut["coverage"], color="#54A24B", linestyle="--", linewidth=1.3)
    right.scatter([cut["coverage"]], [cut_share], s=110, marker="D", color="#54A24B", zorder=6,
                  edgecolor="white", linewidth=0.9)
    right.annotate(f"adopted cut {cut['threshold']:.4f}\n{cut['calls_saved']:,.0f} calls saved",
                   xy=(cut["coverage"], cut_share), xytext=(0.56, 0.16), textcoords="data",
                   fontsize=8.5, color="#2E7D32",
                   arrowprops={"arrowstyle": "->", "color": "#2E7D32", "linewidth": 1.0})
    right.set_xlabel("share of the list called")
    right.set_ylabel("share of the list saved at equal recall")
    right.set_title("calls saved at equal recall\nnormalised, so both halves are comparable", fontsize=11)
    right.set_xlim(0, 1)
    right.set_ylim(0, top)
    right.legend(loc="lower left", fontsize=8.5)
    right.grid(alpha=0.25)

    figure.suptitle(
        f"DepositIQ - the ranking is worth the calls it saves: top decile lift "
        f"{tables['train_oof'][0]['lift']:.2f}x (training) / {tables['held_out'][0]['lift']:.2f}x "
        f"(held out), and the adopted cut reaches {cut['recall']:.1%} of subscribers with "
        f"{cut['coverage']:.1%} of the list",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def validate(halves: dict[str, dict], tables: dict[str, list[dict]], translation: dict[str, dict],
             random_test: dict, recon: list[dict], recalls: list[dict], frame_rows: dict,
             peaks: dict[str, dict], blocks: dict[str, dict]) -> list[dict]:
    """Same contract as every other phase: a check that cannot run is FAILED, never raised (D25)."""
    checks: list[dict] = []

    def compare(name: str, evaluate) -> None:
        try:
            passed, detail = evaluate()
        except Exception as error:  # noqa: BLE001
            passed, detail = False, f"check could not run: {type(error).__name__}: {error}"
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    def base_rates_are_each_halves_own() -> tuple[bool, str]:
        recorded = json.loads(THRESHOLD_PATH.read_text())["training"]
        train_rate = float(halves["train_oof"]["y"].mean())
        held_rate = float(halves["held_out"]["y"].mean())
        return (
            abs(train_rate - recorded["base_rate"]) < 1e-12
            and abs(held_rate - 1058 / 9043) < 1e-12
            and abs(train_rate - held_rate) > 1e-6,
            f"training {train_rate:.6f} ({int(halves['train_oof']['y'].sum()):,}/{len(halves['train_oof']['y']):,}) "
            f"vs held out {held_rate:.6f} ({int(halves['held_out']['y'].sum()):,}/"
            f"{len(halves['held_out']['y']):,}) - two rates, {abs(train_rate - held_rate) * 100:.4f} "
            f"points apart, never pooled",
        )

    compare("each half is measured against its own base rate, not a pooled one",
            base_rates_are_each_halves_own)

    def deciles_partition_the_list() -> tuple[bool, str]:
        details = []
        ok = True
        for half, table in tables.items():
            total = len(halves[half]["y"])
            rows = sum(row["rows"] for row in table)
            sizes = [row["rows"] for row in table]
            monotone_scores = True
            order = np.argsort(-halves[half]["scores"], kind="mergesort")
            ranked = halves[half]["scores"][order]
            edges = [round(total * index / DECILES) for index in range(DECILES + 1)]
            for block in range(DECILES - 1):
                if ranked[edges[block + 1] - 1] < ranked[edges[block + 1]]:
                    monotone_scores = False
            ok = ok and rows == total and max(sizes) - min(sizes) <= 1 and monotone_scores
            details.append(f"{half}: {rows:,} rows over 10 blocks of {min(sizes):,}-{max(sizes):,}, "
                           f"descending")
        return ok, "; ".join(details)

    compare("the ten blocks cover every row once, are equal to a row, and descend", deciles_partition_the_list)

    def precision_and_lift_are_arithmetic() -> tuple[bool, str]:
        worst = 0.0
        for half, table in tables.items():
            base = float(halves[half]["y"].mean())
            for row in table:
                worst = max(
                    worst,
                    abs(row["precision"] - row["subscribers"] / row["rows"]),
                    abs(row["lift"] - row["precision"] / base),
                    float(row["subscribers"] > row["rows"]),
                )
        return worst <= 1e-12, f"precision = subscribers/rows and lift = precision/base at all 20 rows; " \
                               f"max drift {worst:.1e}"

    compare("every published precision and lift is the ratio it claims to be",
            precision_and_lift_are_arithmetic)

    def cumulative_columns_are_running_sums() -> tuple[bool, str]:
        worst = 0.0
        ends = []
        for half, table in tables.items():
            positives = int(halves[half]["y"].sum())
            total = len(halves[half]["y"])
            run_rows = run_hits = 0
            for row in table:
                run_rows += row["rows"]
                run_hits += row["subscribers"]
                worst = max(
                    worst,
                    float(row["cumulative_rows"] != run_rows),
                    float(row["cumulative_subscribers"] != run_hits),
                    abs(row["coverage"] - run_rows / total),
                    abs(row["cumulative_share"] - run_hits / positives),
                )
            ends.append(f"{half} ends at {table[-1]['cumulative_share']:.6f} of subscribers")
        return worst <= 1e-12, "; ".join(ends) + f"; max drift {worst:.1e}"

    compare("the cumulative columns are running sums that end at 100% of subscribers",
            cumulative_columns_are_running_sums)

    def top_decile_clears_d15() -> tuple[bool, str]:
        train_lift = tables["train_oof"][0]["lift"]
        held_lift = tables["held_out"][0]["lift"]
        return (
            min(train_lift, held_lift) > D15_LIFT_TARGET,
            f"top decile lift {train_lift:.4f}x on training ({tables['train_oof'][0]['precision']:.4f} "
            f"precision) and {held_lift:.4f}x held out ({tables['held_out'][0]['precision']:.4f}) against "
            f"D15's >{D15_LIFT_TARGET:.0f}x",
        )

    compare(f"the top decile clears D15's {D15_LIFT_TARGET:.0f}x target on both halves",
            top_decile_clears_d15)

    def lift_declines_down_the_list() -> tuple[bool, str]:
        worst_descent = 0.0
        details = []
        for half, table in tables.items():
            lifts = [row["lift"] for row in table]
            descents = [lifts[i] - lifts[i + 1] for i in range(len(lifts) - 1)]
            worst_descent = max(worst_descent, max(descents))
            details.append(f"{half}: decile 1 {lifts[0]:.2f}x, decile 10 {lifts[-1]:.2f}x, "
                           f"{sum(1 for d in descents if d < 0)} of 9 steps not descending")
        return (
            tables["train_oof"][0]["lift"] == max(row["lift"] for row in tables["train_oof"])
            and tables["held_out"][0]["lift"] == max(row["lift"] for row in tables["held_out"])
            and tables["train_oof"][0]["lift"] > tables["train_oof"][-1]["lift"]
            and tables["held_out"][0]["lift"] > tables["held_out"][-1]["lift"],
            "; ".join(details) + f"; largest non-descending step {worst_descent:.3f}x",
        )

    compare("the highest decile holds the most lift and the list ends worse than it starts",
            lift_declines_down_the_list)

    def random_ranking_gives_unit_lift() -> tuple[bool, str]:
        return (
            abs(random_test["mean_lift"] - 1.0) < 0.05
            and random_test["worst_gain_deviation"] < 0.05
            and abs(random_test["worst_calls_saved"]) < 0.05 * 36168,
            f"with the scores shuffled, the ten lifts run {random_test['min_lift']:.3f}-"
            f"{random_test['max_lift']:.3f} (mean {random_test['mean_lift']:.4f}); the gain curve departs "
            f"from the diagonal by at most {random_test['worst_gain_deviation']:.4f} and the calls-saved "
            f"column by at most {random_test['worst_calls_saved']:,.0f}",
        )

    compare("the table reports no lift when the ranking is random", random_ranking_gives_unit_lift)

    def reconciles_with_t32() -> tuple[bool, str]:
        return all(row["agrees"] for row in recon), "; ".join(
            f"{row['label']}: measured {row['measured_lift']:.4f} vs published "
            f"{row['published_lift']:.4f} ({row['difference']:+.4f})" for row in recon
        )

    compare("the cumulative curve lands on T32's two published multipliers", reconciles_with_t32)

    def cut_matches_t34() -> tuple[bool, str]:
        recorded = json.loads(THRESHOLD_PATH.read_text())["at_chosen"]
        cut = translation["train_oof"]
        return (
            cut["calls"] == recorded["clients_called"]
            and cut["subscribers_reached"] == recorded["subscribers_reached"]
            and abs(cut["recall"] - recorded["subscribers_reached"] / 4231) < 1e-12,
            f"the adopted cut calls {cut['calls']:,} and reaches {cut['subscribers_reached']:,} on the "
            f"training half - identical to T34's record ({recorded['clients_called']:,} / "
            f"{recorded['subscribers_reached']:,})",
        )

    compare("this module's reading of the adopted cut is T34's reading of it", cut_matches_t34)

    def calls_saved_is_the_recall_difference() -> tuple[bool, str]:
        worst = 0.0
        details = []
        for half, cut in translation.items():
            expected = cut["recall"] * cut["rows"] - cut["calls"]
            worst = max(worst, abs(cut["calls_saved"] - expected),
                        abs(cut["euros_not_spent"] - cut["calls_saved"] * cost.CALL_COST_EUR))
            details.append(f"{half}: {cut['calls_saved']:,.0f} calls "
                           f"({cut['calls_saved_share']:.1%} of the list, "
                           f"{cut['euros_not_spent']:,.0f} EUR)")
        return worst <= 1e-9 and all(cut["calls_saved"] > 0 for cut in translation.values()), \
            "; ".join(details) + f"; max drift {worst:.1e}"

    compare("calls saved equals the recall a fixed list would have needed minus the calls made",
            calls_saved_is_the_recall_difference)

    def recall_table_is_monotone() -> tuple[bool, str]:
        calls = [row["calls"] for row in recalls]
        total = len(halves["train_oof"]["y"])
        anchored = max(abs(row["calls_random_would_need"] - row["target_recall"] * total)
                       for row in recalls)
        return (
            calls == sorted(calls)
            and len(set(calls)) == len(calls)
            and all(row["calls_saved"] > 0 for row in recalls)
            and anchored <= 1e-9,
            "; ".join(f"{row['target_recall']:.0%} recall -> {row['calls']:,} calls "
                      f"(random {row['calls_random_would_need']:,.0f}, saved "
                      f"{row['calls_saved']:,.0f})" for row in recalls)
            + f"; the random baseline is recall x {total:,} rows to {anchored:.1e}",
        )

    compare("reaching a higher recall always costs more calls", recall_table_is_monotone)

    def savings_peak_is_before_the_cut() -> tuple[bool, str]:
        details = []
        ok = True
        for half, peak in peaks.items():
            cut = translation[half]
            ok = ok and peak["is_local_max"] and peak["coverage"] < cut["coverage"] \
                and peak["calls_saved"] > cut["calls_saved"] and peak["lift_at_peak"] > 1.0
            details.append(
                f"{half}: peak at {peak['coverage']:.2%} ({peak['calls']:,} calls, "
                f"lift {peak['lift_at_peak']:.2f}x) saves {peak['calls_saved']:,.0f} vs the cut's "
                f"{cut['calls_saved']:,.0f} at {cut['coverage']:.2%}"
            )
        return bool(ok), "; ".join(details)

    compare("the calls-saved peak lies before the adopted cut, where efficiency crosses the base rate",
            savings_peak_is_before_the_cut)

    def the_block_after_the_peak_still_pays() -> tuple[bool, str]:
        details = []
        for half, block in blocks.items():
            details.append(
                f"{half}: the {block['calls']:,} calls between the peak and the cut convert at "
                f"{block['marginal_rate']:.2%} against a {block['price']:.2%} price "
                f"({block['net_for_the_block_eur']:+,.0f} EUR)"
            )
        return all(block["above_price"] and block["net_for_the_block_eur"] > 0
                   for block in blocks.values()), "; ".join(details)

    compare("the calls between the savings peak and the cut still convert above the price",
            the_block_after_the_peak_still_pays)

    def held_out_uses_the_artifact() -> tuple[bool, str]:
        artifact = joblib.load(ARTIFACT_PATH)
        scores = halves["held_out"]["scores"]
        return (
            len(scores) == frame_rows["test"] == 9043
            and float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0
            and artifact is not None and len(scores) == len(halves["held_out"]["y"]),
            f"{len(scores):,} held-out scores from {ARTIFACT_PATH.name} in "
            f"[{scores.min():.6f}, {scores.max():.6f}], no refit",
        )

    compare("the held-out ranking comes from the persisted artifact", held_out_uses_the_artifact)

    def table_file_is_written() -> tuple[bool, str]:
        frame = pd.read_csv(TABLE_PATH)
        return (
            len(frame) == 2 * DECILES
            and set(frame["half"]) == {"train_oof", "held_out"}
            and {"decile", "rows", "subscribers", "precision", "lift", "coverage", "cumulative_share",
                 "cumulative_lift", "calls_saved_at_equal_recall"} <= set(frame.columns),
            f"{TABLE_PATH.name}: {len(frame)} rows x {len(frame.columns)} columns, halves "
            f"{sorted(set(frame['half']))}",
        )

    compare("the lift table was written with both halves and its columns",
            table_file_is_written)

    def figure_is_written() -> tuple[bool, str]:
        data = FIGURE_PATH.read_bytes()
        return (
            data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) > 30_000,
            f"{FIGURE_PATH.name} is {len(data):,} bytes",
        )

    compare("the lift/gain figure was written", figure_is_written)

    return checks


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(variant: str = "full") -> dict:
    config.ensure_dirs()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    halves = half_scores()
    tables = {half: decile_table(data["y"], data["scores"], half) for half, data in halves.items()}
    translation = cut_translation(halves, tables)
    recalls = recall_to_calls(halves["train_oof"]["y"], halves["train_oof"]["scores"])
    recon = reconciliation(tables)
    random_test = random_self_test(halves["train_oof"]["y"])
    peaks = {half: savings_peak(data["y"], data["scores"]) for half, data in halves.items()}
    blocks = {half: marginal_block(peaks[half], translation[half]) for half in halves}

    frame = pd.concat([pd.DataFrame(table) for table in tables.values()], ignore_index=True)
    frame = frame[["half", "decile", "rows", "subscribers", "precision", "lift", "cumulative_rows",
                   "cumulative_subscribers", "cumulative_share", "coverage", "cumulative_precision",
                   "cumulative_lift", "calls_saved_at_equal_recall"]]
    frame.to_csv(TABLE_PATH, index=False, float_format="%.6f")

    draw(tables, translation, peaks)

    payload = {
        "task": "T35",
        "variant": variant,
        "deciles": DECILES,
        "decile_rule": "ten equal-count positional blocks of the descending score, stable for ties",
        "halves": {
            half: {
                "label": data["label"],
                "rows": int(len(data["y"])),
                "subscribers": int(data["y"].sum()),
                "base_rate": float(data["y"].mean()),
            }
            for half, data in halves.items()
        },
        "tables": tables,
        "translation": translation,
        "savings_peak": peaks,
        "block_after_peak": blocks,
        "recall_table": recalls,
        "reconciliation": recon,
        "d15": {
            "target": D15_LIFT_TARGET,
            "train_top_decile_lift": tables["train_oof"][0]["lift"],
            "held_top_decile_lift": tables["held_out"][0]["lift"],
            "met": bool(min(tables["train_oof"][0]["lift"], tables["held_out"][0]["lift"])
                        > D15_LIFT_TARGET),
        },
        "random_self_test": random_test,
        "csv": str(TABLE_PATH.relative_to(config.ROOT)),
        "figure": {"path": str(FIGURE_PATH.relative_to(config.ROOT)),
                   "bytes": FIGURE_PATH.stat().st_size},
    }

    checks = validate(halves, tables, translation, random_test, recon, recalls,
                      {"train": len(halves["train_oof"]["y"]), "test": len(halves["held_out"]["y"])},
                      peaks, blocks)
    payload["checks"] = checks
    payload["checks_passed"] = sum(1 for check in checks if check["passed"])
    payload["checks_total"] = len(checks)
    payload["checks_ok"] = payload["checks_passed"] == payload["checks_total"]
    payload["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    RECORD_PATH.write_text(json.dumps(payload, indent=2, sort_keys=False))

    on_disk = json.loads(RECORD_PATH.read_text())
    same = (
        len(on_disk["tables"]["train_oof"]) == DECILES
        and len(on_disk["tables"]["held_out"]) == DECILES
        and on_disk["halves"]["train_oof"]["base_rate"] == payload["halves"]["train_oof"]["base_rate"]
        and on_disk["translation"]["held_out"]["calls_saved"]
        == payload["translation"]["held_out"]["calls_saved"]
        and on_disk["checks_total"] == len(checks)
        and on_disk["figure"]["bytes"] == FIGURE_PATH.stat().st_size
    )
    payload["checks"].append({
        "check": "the record on disk is this run's answers, not a transcript of another",
        "passed": bool(same),
        "detail": f"{RECORD_PATH.name}: 10+10 deciles, held-out calls saved "
                  f"{on_disk['translation']['held_out']['calls_saved']:,.0f}, "
                  f"{on_disk['checks_total']} checks before this one, "
                  f"figure {on_disk['figure']['bytes']:,} B",
    })
    payload["checks_passed"] = sum(1 for check in payload["checks"] if check["passed"])
    payload["checks_total"] = len(payload["checks"])
    payload["checks_ok"] = payload["checks_passed"] == payload["checks_total"]
    RECORD_PATH.write_text(json.dumps(payload, indent=2, sort_keys=False))

    return payload


def print_report(result: dict) -> None:
    print("DepositIQ T35 - lift, gain and deciles")
    print("=" * 108)
    for half, info in result["halves"].items():
        print(f"{info['label']:<24} {info['rows']:>7,} rows  {info['subscribers']:>5,} subscribers  "
              f"base {info['base_rate']:.4%}")
    print()

    for half in ("train_oof", "held_out"):
        print(f"DECILES - {result['halves'][half]['label']}")
        header = (f"{'dec':>4}{'rows':>8}{'subs':>7}{'prec':>9}{'lift':>8}"
                  f"{'cum subs':>10}{'coverage':>10}{'gain':>8}{'cum lift':>10}{'saved':>9}")
        print(header)
        print("-" * len(header))
        for row in result["tables"][half]:
            print(f"{row['decile']:>4}{row['rows']:>8,}{row['subscribers']:>7,}"
                  f"{row['precision']:>9.4f}{row['lift']:>8.2f}"
                  f"{row['cumulative_subscribers']:>10,}{row['coverage']:>9.2%}"
                  f"{row['cumulative_share']:>8.2%}{row['cumulative_lift']:>10.2f}"
                  f"{row['calls_saved_at_equal_recall']:>9,.0f}")
        print()

    print("THE ADOPTED CUT, IN CALLS")
    for half, cut in result["translation"].items():
        print(f"  {result['halves'][half]['label']:<24} calls {cut['calls']:>6,} "
              f"({cut['coverage']:>6.2%})  recall {cut['recall']:>6.2%}  lift {cut['lift']:.2f}x  "
              f"saved {cut['calls_saved']:>6,.0f} calls ({cut['euros_not_spent']:>8,.0f} EUR)")
    print()

    print("WHERE CALLS SAVED PEAKS, AND WHERE THE CUT SITS")
    for half, peak in result["savings_peak"].items():
        cut = result["translation"][half]
        print(f"  {result['halves'][half]['label']:<24} peak {peak['coverage']:>6.2%} coverage "
              f"({peak['calls']:>6,} calls, lift {peak['lift_at_peak']:.2f}x) saves "
              f"{peak['calls_saved']:>7,.0f} | cut {cut['coverage']:>6.2%} saves "
              f"{cut['calls_saved']:>7,.0f} | difference {peak['calls_saved'] - cut['calls_saved']:>6,.0f}")
        block = result["block_after_peak"][half]
        print(f"  {'':<24} the {block['calls']:>6,} calls between them convert at "
              f"{block['marginal_rate']:>6.2%} vs the {block['price']:.2%} price -> "
              f"{block['net_for_the_block_eur']:>+9,.0f} EUR")
    print()

    print("CALLS NEEDED FOR A TARGET RECALL (training half)")
    for row in result["recall_table"]:
        print(f"  {row['target_recall']:>5.0%} recall: {row['calls']:>6,} calls "
              f"({row['coverage']:>6.2%} of the list) vs {row['calls_random_would_need']:>8,.0f} for a "
              f"random list -> {row['calls_saved']:>7,.0f} saved ({row['euros_not_spent']:>8,.0f} EUR)")
    print()

    print("RECONCILIATION WITH T32")
    for row in result["reconciliation"]:
        print(f"  {row['label']:<32} published {row['published_lift']:.4f}  measured "
              f"{row['measured_lift']:.4f}  ({row['difference']:+.4f})  "
              f"{'agrees' if row['agrees'] else 'DISAGREES'}")
    print()
    test = result["random_self_test"]
    print(f"RANDOM SELF-TEST: shuffled scores give lifts {test['min_lift']:.3f}-{test['max_lift']:.3f} "
          f"(mean {test['mean_lift']:.4f}), gain off the diagonal by <= {test['worst_gain_deviation']:.4f}")
    print()
    print(f"wrote : {result['csv']} + {result['figure']['path']} "
          f"({result['figure']['bytes']:,} B) + {RECORD_PATH.name}")
    print(f"total : {result['elapsed_seconds']} s")
    print()
    print(f"T35 checks: {result['checks_passed']}/{result['checks_total']} passed")
    for check in result["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        print(f"  [{marker}] {check['check']}  ({check['detail']})")
    print()


if __name__ == "__main__":
    print("DepositIQ Phase 7 - lift, gain and deciles")
    print()
    outcome = report()
    print_report(outcome)
    print("=" * 108)
    print(f"result : {'PASS' if outcome['checks_ok'] else 'FAIL'}")
    print("Next   : T36 runs the with-vs-without `duration` comparison at this operating point.")
    sys.exit(0 if outcome["checks_ok"] else 1)
