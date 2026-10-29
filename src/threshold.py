"""
DepositIQ - the operating threshold, chosen with money (T34).

Why this file exists
--------------------
Every earlier phase produced a ranking. A ranking is not a campaign: something has to decide who gets
called, and that decision has a price. `src/cost.py` prices it — a call costs **C = EUR 1.50**, a captured
subscription is worth **V = EUR 20.00** — and the arithmetic is one line:

    EV(call a client with probability p) = p*V - C,  which is positive exactly when p > C/V = 0.075

So the threshold is not a hyperparameter to be tuned for F1 or accuracy. It is a **pricing decision**, and
0.075 is the price the bank's own assumptions produce. This module measures whether the data agrees, says
how flat the optimum is, shows what a capacity limit does to the answer, and reports what the decision is
worth — and it does the choosing on the **training half** so the held-out half stays a check.

What is chosen, and where
-------------------------
**Chosen on the training half.** The thresholds are swept over the 36,168 training rows' **out-of-fold**
scores: five folds, each row scored by a model that did not see it. That is the same fold split T26-T32
reused, so the numbers here sit beside theirs rather than beside a second experiment.

**Checked once on the held-out half**, in a separate function called *after* the choice is fixed. The
record states which half the decision came from, and a check verifies that the chosen cut is **not** the
held-out optimum — if it were, the choice would have been contaminated by the rows used to test it.

Why the analytic answer and the empirical one are both reported
--------------------------------------------------------------
`p* = C/V` is exact *if the model's output is a probability*. T33 measured that it nearly is (mean p 0.1151
against a 0.1170 base rate), so the two should agree, and the gap between them is a calibration statement
rather than a tuning result. The empirical sweep is therefore used two ways: to show where the money
actually peaks, and to measure **how flat** the peak is. A curve that is within 1% of its maximum across a
wide band means the threshold is not a delicate quantity, which is the honest version of "the threshold was
tuned".

What a capacity limit does
--------------------------
If the bank can only make N calls, the answer is not `p*` at all — it is the N highest scores, whatever
their probabilities are. Both regimes are run, because a report that quotes only the unconstrained optimum
is advising a campaign the bank may not be able to staff.

How to run
----------
From the repository root, inside the virtual environment:

    python -m src.threshold    # ~30 s: five fold fits, one sweep, one figure

Writes `models/threshold_choice.json` and `reports/figures/07_threshold_curve.png`;
`docs/07_threshold.md` quotes only numbers from that record.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config, cost, features, metrics as metric_table, select, train, tune
from src.split import encode_target

RECORD_PATH: Path = config.MODELS_DIR / "threshold_choice.json"
FIGURE_PATH: Path = config.FIGURES_DIR / "07_threshold_curve.png"
ARTIFACT_PATH: Path = config.MODELS_DIR / "final_model.joblib"
PARAMS_PATH: Path = config.MODELS_DIR / "best_params.json"

# The chosen operating point **is** the cost model's answer, imported rather than typed: if `cost.py`'s
# assumptions move, this file's decision moves with them and the doc's claim "the threshold is C/V" stays
# true by construction rather than by memory.
CHOSEN: float = cost.break_even_threshold()

# Comparisons the phase has already argued for, kept so the money argument is read against them.
F1_CUT: float = 0.210325          # T32's mean cross-validated cut
CONVENTIONAL: float = 0.5

# Sweep resolution. The score distribution is dense below 0.3, so a uniform grid at 0.0015 is fine for the
# curve; the *chosen* thresholds are inserted exactly so the curve passes through the recorded values.
GRID_POINTS: int = 400
GRID_TOP: float = 0.6

# "Within 1% of the best net value" - the definition of the plateau the doc quotes.
PLATEAU_TOLERANCE: float = 0.01

# Capacity regimes: the share of the list a campaign can call if agents are the binding constraint.
CAPACITY_SHARES: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.40)


# ---------------------------------------------------------------------------
# Out-of-fold scores for the training half
# ---------------------------------------------------------------------------
def chosen_params() -> dict:
    """The six tuned settings, from T30's record rather than from this file."""
    return dict(json.loads(PARAMS_PATH.read_text())["best_params"]["params"])


def oof_scores(frame: pd.DataFrame, y: pd.Series, folds: list[tuple[np.ndarray, np.ndarray]],
               columns: list[str]) -> tuple[np.ndarray, dict]:
    """
    One out-of-fold probability per training row, from the shipped configuration.

    This is a **fit** - five of them - and that is correct here: the threshold is being chosen on the
    training half, and choosing it from in-fold predictions would be choosing it from rows the model had
    already memorised. The artifact itself is not refitted; the folds are, which is what cross-validation
    means.
    """
    params = chosen_params()
    run = train.cross_validate_factory(
        "xgboost_tuned", lambda: tune.make_estimator(params), "shipped",
        "T30's tuned point, the configuration T31 persisted", frame, y, folds, columns,
    )
    scores = np.full(len(frame), np.nan)
    covered = np.zeros(len(frame), dtype=int)
    for pipeline, _, valid in run["fits"]:
        scores[valid] = pipeline.predict_proba(frame.iloc[valid])[:, 1]
        covered[valid] += 1
    return scores, {"folds": run["folds"], "covered": covered}


def sweep(y: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> list[dict]:
    """Expected campaign value at each threshold, through `cost.py` rather than a local formula."""
    return [cost.expected_campaign_value(y, scores, float(t)) for t in thresholds]


# ---------------------------------------------------------------------------
# The choice
# ---------------------------------------------------------------------------
def grid() -> np.ndarray:
    """The sweep grid, with the thresholds that get quoted inserted exactly."""
    base = np.linspace(0.001, GRID_TOP, GRID_POINTS)
    return np.unique(np.concatenate([base, np.array([CHOSEN, F1_CUT, CONVENTIONAL])]))


def choose(y: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> dict:
    """
    Pick the money-optimal threshold from **training-half** out-of-fold scores only.

    The signature is the guarantee: nothing here can see the held-out half. The analytic break-even is
    reported beside the empirical argmax because they answer different questions - one is the pricing rule,
    the other is where this sample's money peaked.
    """
    curve = sweep(y, scores, thresholds)
    best = max(curve, key=lambda row: row["net_value_eur"])
    ceiling = best["net_value_eur"]
    floor = (1.0 - PLATEAU_TOLERANCE) * ceiling
    plateau = [row["threshold"] for row in curve if row["net_value_eur"] >= floor]

    at_choice = cost.expected_campaign_value(y, scores, CHOSEN)
    return {
        "curve": curve,
        "empirical_best": best,
        "plateau": {
            "tolerance": PLATEAU_TOLERANCE,
            "low": float(min(plateau)),
            "high": float(max(plateau)),
            "share_of_grid": float(len(plateau) / len(curve)),
            "net_floor_eur": float(floor),
        },
        "at_chosen": at_choice,
        "chosen_within_plateau": bool(at_choice["net_value_eur"] >= floor),
        "shortfall_vs_best_eur": float(ceiling - at_choice["net_value_eur"]),
        "shortfall_pct": float(100.0 * (ceiling - at_choice["net_value_eur"]) / ceiling)
        if ceiling else 0.0,
    }


def capacity_table(y: np.ndarray, scores: np.ndarray) -> list[dict]:
    """
    What the answer becomes when the bank can only make a fixed number of calls.

    The cut is the (100*share)-th percentile of the score, i.e. "the best N clients", which is what a
    capacity-constrained campaign actually is. Reported beside the unconstrained optimum so the two are not
    confused in the report.
    """
    rows = []
    for share in CAPACITY_SHARES:
        cut = float(np.quantile(scores, 1.0 - share))
        value = cost.expected_campaign_value(y, scores, cut)
        value["capacity_share"] = share
        rows.append(value)
    return rows


def sensitivity() -> list[dict]:
    """Every cell of `cost.sensitivity_table`, as records rather than as a printed frame."""
    frame = cost.sensitivity_table()
    rows = []
    for _, record in frame.iterrows():
        for column, value in record.items():
            if column == "call_cost_eur":
                continue
            rows.append({
                "call_cost_eur": float(record["call_cost_eur"]),
                "value_eur": float(column.split("_", 1)[1]),
                "threshold": float(value),
            })
    return rows


# ---------------------------------------------------------------------------
# The check on the held-out half
# ---------------------------------------------------------------------------
def held_out_check(chosen: float) -> dict:
    """
    Score the **persisted artifact** on the held-out rows at the threshold already fixed above.

    Called after the choice, and it cannot influence it: the value returned here is written into the record
    and reported, and no branch in this module reads it back. The held-out optimum is measured as well, for
    one reason only - to show that the chosen cut is not it.
    """
    pipeline = joblib.load(ARTIFACT_PATH)
    frame = features.load_engineered("test")
    y = encode_target(frame).to_numpy()
    scores = pipeline.predict_proba(frame)[:, 1].astype(np.float64)

    at_chosen = cost.expected_campaign_value(y, scores, chosen)
    at_f1 = cost.expected_campaign_value(y, scores, F1_CUT)
    at_half = cost.expected_campaign_value(y, scores, CONVENTIONAL)
    at_all = cost.expected_campaign_value(y, scores, 0.0)

    # The peak over rank cut-offs: every 7th rank locates it, then the neighbourhood is scanned exactly.
    order = np.argsort(-scores, kind="mergesort")
    ranked = scores[order]
    best = {"net_value_eur": -np.inf, "threshold": 0.0}
    for index in range(0, len(ranked), 7):
        value = cost.expected_campaign_value(y, scores, float(ranked[index]))
        if value["net_value_eur"] > best["net_value_eur"]:
            best = value
    coarse = int(np.argmin([abs(ranked[i] - best["threshold"])
                            for i in range(0, len(ranked), 7)])) * 7
    for index in range(max(0, coarse - 7), min(len(ranked), coarse + 8)):
        value = cost.expected_campaign_value(y, scores, float(ranked[index]))
        if value["net_value_eur"] > best["net_value_eur"]:
            best = value

    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "at_chosen": at_chosen,
        "at_f1_cut": at_f1,
        "at_conventional": at_half,
        "at_call_everyone": at_all,
        "held_out_best": best,
        "chosen_is_not_the_held_out_best": abs(float(best["threshold"]) - chosen) > 1e-9,
    }


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def validate(frame: pd.DataFrame, y: pd.Series, folds: list[tuple[np.ndarray, np.ndarray]],
             scores: np.ndarray, info: dict, choice: dict, capacity: list[dict],
             sens: list[dict], holdout: dict) -> list[dict]:
    """Same contract as every other phase: a check that cannot run is FAILED, never raised (D25)."""
    checks: list[dict] = []

    def compare(name: str, evaluate) -> None:
        try:
            passed, detail = evaluate()
        except Exception as error:  # noqa: BLE001 - any failure here is a failed check
            passed, detail = False, f"check could not run: {type(error).__name__}: {error}"
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    # --- the price is the cost model's ------------------------------------------------------
    def threshold_is_the_price() -> tuple[bool, str]:
        analytic = cost.break_even_threshold()
        ratio = cost.CALL_COST_EUR / cost.VALUE_PER_SUBSCRIPTION_EUR
        return (
            CHOSEN == analytic == ratio,
            f"chosen {CHOSEN:.6f} = cost.break_even_threshold() = C/V = "
            f"{cost.CALL_COST_EUR:.2f}/{cost.VALUE_PER_SUBSCRIPTION_EUR:.2f}",
        )

    compare("the chosen threshold is the cost model's break-even, not a searched value",
            threshold_is_the_price)

    # --- every training row contributed exactly one out-of-fold score ------------------------
    def oof_is_complete() -> tuple[bool, str]:
        covered = info["covered"]
        fold_rows = sum(len(valid) for _, valid in folds)
        return (
            not np.isnan(scores).any()
            and bool((covered == 1).all())
            and fold_rows == len(frame)
            and all(row["rows"] == len(valid) for row, (_, valid) in zip(info["folds"], folds)),
            f"{len(frame):,} training rows, each scored once out-of-fold; {fold_rows:,} fold "
            f"predictions over {len(folds)} folds, {int(np.isnan(scores).sum())} NaN",
        )

    compare("every training row contributed exactly one out-of-fold score", oof_is_complete)

    # --- and those folds are the project's folds, not a parallel set -------------------------
    def folds_reproduce_t30() -> tuple[bool, str]:
        recorded = json.loads(PARAMS_PATH.read_text())["arms"]["tuned_untreated"]["folds"]
        mine = [row["pr_auc"] for row in info["folds"]]
        theirs = [row["pr_auc"] for row in recorded]
        drift = max(abs(a - b) for a, b in zip(mine, theirs))
        return (
            len(mine) == len(theirs) and drift <= 1e-9,
            f"the five fold PR-AUCs reproduce T30's tuned arm to {drift:.1e} "
            f"({', '.join(f'{v:.4f}' for v in mine)})",
        )

    compare("the out-of-fold scores come from T30's folds and T30's tuned parameters",
            folds_reproduce_t30)

    # --- the money is the arithmetic it claims to be ----------------------------------------
    def money_identity() -> tuple[bool, str]:
        worst = 0.0
        for row in choice["curve"]:
            worst = max(
                worst,
                abs(row["net_value_eur"] - (row["revenue_eur"] - row["call_cost_eur"])),
                abs(row["revenue_eur"] - row["subscribers_reached"] * cost.VALUE_PER_SUBSCRIPTION_EUR),
                abs(row["call_cost_eur"] - row["clients_called"] * cost.CALL_COST_EUR),
                float(row["clients_called"] != row["subscribers_reached"] + row["wasted_calls"]),
                float(row["subscribers_reached"] + row["subscribers_missed"] != int(y.sum())),
            )
        return worst <= 1e-9, (
            f"{len(choice['curve'])} thresholds: net = revenue - cost, cost = calls x C, "
            f"revenue = hits x V, calls = hits + wasted, hits + missed = {int(y.sum()):,}; "
            f"max drift {worst:.1e}"
        )

    compare("every point on the curve obeys net = revenue - cost and the call arithmetic",
            money_identity)

    # --- the empirical peak is where the sweep says it is -----------------------------------
    def empirical_peak_is_the_max() -> tuple[bool, str]:
        best = choice["empirical_best"]
        highest = max(row["net_value_eur"] for row in choice["curve"])
        return (
            abs(best["net_value_eur"] - highest) <= 1e-9
            and all(row["net_value_eur"] <= highest + 1e-9 for row in choice["curve"]),
            f"peak {best['net_value_eur']:,.2f} EUR at {best['threshold']:.4f}, which is the maximum of "
            f"the {len(choice['curve'])} swept points",
        )

    compare("the recorded empirical optimum is the maximum of the recorded curve",
            empirical_peak_is_the_max)

    # --- and the chosen price is close enough to it to be defensible ------------------------
    def chosen_is_within_the_plateau() -> tuple[bool, str]:
        plateau = choice["plateau"]
        at_choice = choice["at_chosen"]
        return (
            choice["chosen_within_plateau"]
            and plateau["low"] <= CHOSEN <= plateau["high"],
            f"the cut at {CHOSEN:.4f} earns {at_choice['net_value_eur']:,.2f} EUR against a peak of "
            f"{choice['empirical_best']['net_value_eur']:,.2f} EUR at "
            f"{choice['empirical_best']['threshold']:.4f} - a shortfall of "
            f"{choice['shortfall_vs_best_eur']:,.2f} EUR ({choice['shortfall_pct']:.3f}%); the "
            f"within-{PLATEAU_TOLERANCE:.0%} band runs {plateau['low']:.4f}-{plateau['high']:.4f}",
        )

    compare("the cost model's cut is inside the empirical plateau of near-optimal thresholds",
            chosen_is_within_the_plateau)

    # --- money beats the alternatives the phase rejected ------------------------------------
    def beats_calling_everyone() -> tuple[bool, str]:
        everyone = cost.expected_campaign_value(y.to_numpy(), scores, 0.0)
        everyone["capacity_share"] = 1.0
        chosen_net = choice["at_chosen"]["net_value_eur"]
        return (
            chosen_net > everyone["net_value_eur"],
            f"calling all {len(frame):,} clients nets {everyone['net_value_eur']:,.2f} EUR; the cut nets "
            f"{chosen_net:,.2f} EUR, so the marginal client below the price loses money even though the "
            f"average one does not",
        )

    compare("the chosen cut outearns calling everyone", beats_calling_everyone)

    def beats_the_rejected_cuts() -> tuple[bool, str]:
        at_f1 = cost.expected_campaign_value(y.to_numpy(), scores, F1_CUT)
        at_half = cost.expected_campaign_value(y.to_numpy(), scores, CONVENTIONAL)
        chosen_net = choice["at_chosen"]["net_value_eur"]
        return (
            chosen_net > at_f1["net_value_eur"] and chosen_net > at_half["net_value_eur"],
            f"chosen {chosen_net:,.2f} EUR; T32's F1 cut {at_f1['net_value_eur']:,.2f} EUR; "
            f"0.5 {at_half['net_value_eur']:,.2f} EUR",
        )

    compare("the chosen cut outearns both the F1 cut and 0.5", beats_the_rejected_cuts)

    def subscriptions_pay_for_themselves() -> tuple[bool, str]:
        at_choice = choice["at_chosen"]
        return (
            0 < at_choice["cost_per_subscription_eur"] < cost.VALUE_PER_SUBSCRIPTION_EUR
            and at_choice["roi"] > 0,
            f"{at_choice['cost_per_subscription_eur']:.2f} EUR per captured subscription against "
            f"{cost.VALUE_PER_SUBSCRIPTION_EUR:.2f} EUR of value; ROI {at_choice['roi']:+.1%}",
        )

    compare("a captured subscription costs less than it is worth at the chosen cut",
            subscriptions_pay_for_themselves)

    # --- capacity is a different question ---------------------------------------------------
    def capacity_fills_each_share() -> tuple[bool, str]:
        worst = 0.0
        for row in capacity:
            target = len(frame) * row["capacity_share"]
            worst = max(worst, abs(row["clients_called"] - target) / target)
        return (
            worst <= 0.01 and max(row["capacity_share"] for row in capacity) < 1.0,
            f"each cut calls its capacity share to within {worst:.2%}: "
            + ", ".join(f"{row['capacity_share']:.0%}->{row['clients_called']:,}"
                         for row in sorted(capacity, key=lambda r: r["capacity_share"])),
        )

    compare("every capacity row calls the share of the list it claims to", capacity_fills_each_share)

    # --- the price moves with the assumptions ----------------------------------------------
    def sensitivity_is_the_price() -> tuple[bool, str]:
        wrong = [
            row for row in sens
            if abs(row["threshold"] - row["call_cost_eur"] / row["value_eur"]) > 1e-4
        ]
        costs = sorted({row["call_cost_eur"] for row in sens})
        values = sorted({row["value_eur"] for row in sens})
        # a dearer call must raise the bar; a more valuable subscription must lower it
        cost_rises = all(
            [row["threshold"] for row in sens if row["value_eur"] == v] == sorted(
                [row["threshold"] for row in sens if row["value_eur"] == v])
            for v in values
        )
        value_rises = all(
            [row["threshold"] for row in sens if row["call_cost_eur"] == c] == sorted(
                [row["threshold"] for row in sens if row["call_cost_eur"] == c], reverse=True)
            for c in costs
        )
        return (
            not wrong and cost_rises and value_rises and len(sens) == len(costs) * len(values),
            f"{len(sens)} cells over C in {{{', '.join(f'{c:g}' for c in costs)}}} x V in "
            f"{{{', '.join(f'{v:g}' for v in values)}}}, each exactly C/V, monotone in both directions; "
            f"at V=20 the bar runs "
            f"{min(r['threshold'] for r in sens if r['value_eur'] == 20):.4f} as C rises",
        )

    compare("every sensitivity cell is C/V, so a costlier call raises the bar",
            sensitivity_is_the_price)

    # --- the held-out half checked the choice, and did not make it --------------------------
    def the_choice_was_made_on_the_training_half() -> tuple[bool, str]:
        best = holdout["held_out_best"]
        return (
            holdout["chosen_is_not_the_held_out_best"],
            f"the held-out half would have chosen {float(best['threshold']):.4f} "
            f"({best['net_value_eur']:,.2f} EUR); the recorded choice is {CHOSEN:.4f}, made before the "
            f"held-out scores existed in this run",
        )

    compare("the choice was made on training folds, and the held-out optimum is not it",
            the_choice_was_made_on_the_training_half)

    def held_out_agrees_on_the_money() -> tuple[bool, str]:
        at_chosen = holdout["at_chosen"]
        at_half = holdout["at_conventional"]
        per_client = at_chosen["net_value_eur"] / holdout["rows"]
        return (
            at_chosen["net_value_eur"] > 0
            and at_chosen["net_value_eur"] > at_half["net_value_eur"],
            f"on {holdout['rows']:,} unseen rows the cut calls {at_chosen['clients_called']:,} and nets "
            f"{at_chosen['net_value_eur']:,.2f} EUR ({per_client:.3f} EUR per client), against "
            f"{at_half['net_value_eur']:,.2f} EUR at 0.5",
        )

    compare("the decision still earns money on the held-out rows", held_out_agrees_on_the_money)

    def figure_exists_and_carries_the_choice() -> tuple[bool, str]:
        data = FIGURE_PATH.read_bytes()
        return (
            data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) > 30_000,
            f"{FIGURE_PATH.name} is {len(data):,} bytes and the record carries the threshold it marks "
            f"({CHOSEN:.4f})",
        )

    compare("the threshold figure was written", figure_exists_and_carries_the_choice)

    return checks


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------
def draw(curve: list[dict], choice: dict, sens: list[dict], holdout: dict,
         path: Path = FIGURE_PATH) -> Path:
    """Net value against the threshold, the calibration that justifies reading it as money, and the price."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, (left, middle, right) = plt.subplots(1, 3, figsize=(17.5, 6.0))

    thresholds = [row["threshold"] for row in curve]
    nets = [row["net_value_eur"] for row in curve]

    left.plot(thresholds, nets, color="#333333", linewidth=2.0)
    left.axhline(0, color="#999999", linestyle=":", linewidth=1.0)
    plateau = choice["plateau"]
    top = max(nets)
    left.axhspan(plateau["net_floor_eur"], top * 1.06, color="#54A24B", alpha=0.13, zorder=0,
                 label=f"within {PLATEAU_TOLERANCE:.0%} of the peak")
    left.axvspan(plateau["low"], plateau["high"], color="#54A24B", alpha=0.22, zorder=0)
    # Labels sit in the empty lower band, left of each line, so they cannot collide with the curve or title.
    for value, label, colour in (
        (CHOSEN, f"chosen {CHOSEN:.4f}\n(C/V, adopted)", "#2E7D32"),
        (F1_CUT, f"T32 F1 cut\n{F1_CUT:.4f}", "#C62828"),
        (CONVENTIONAL, "conventional\n0.5", "#1565C0"),
    ):
        point = min(curve, key=lambda row: abs(row["threshold"] - value))
        left.axvline(value, color=colour, linestyle="--", linewidth=1.4)
        left.scatter([point["threshold"]], [point["net_value_eur"]], color=colour, zorder=5, s=90,
                     edgecolor="white", linewidth=1.2)
        left.annotate(f"{label}\n{point['net_value_eur']:,.0f} EUR",
                      xy=(value, 0.03 * top), xytext=(4, 0), textcoords="offset points",
                      fontsize=8.5, color=colour, ha="left", va="bottom", linespacing=1.35)
    left.set_xlabel("threshold applied to P(subscribe)")
    left.set_ylabel("net campaign value (EUR) over the training half")
    left.set_title("money against the threshold\n(out-of-fold, 36,168 rows)", fontsize=11)
    left.set_xlim(0, GRID_TOP)
    left.set_ylim(-0.02 * top, top * 1.10)
    left.legend(loc="upper right", fontsize=8.5)
    left.grid(alpha=0.25)

    # Reliability: the reason a threshold on the score can be read as a price at all.
    pipeline = joblib.load(ARTIFACT_PATH)
    frame = features.load_engineered("test")
    y_test = encode_target(frame).to_numpy()
    p_test = pipeline.predict_proba(frame)[:, 1].astype(np.float64)
    bins = np.quantile(p_test, np.linspace(0, 1, 11))
    bins[-1] = bins[-1] + 1e-12
    index = np.clip(np.digitize(p_test, bins) - 1, 0, 9)
    observed = [y_test[index == b].mean() if (index == b).any() else np.nan for b in range(10)]
    predicted = [p_test[index == b].mean() if (index == b).any() else np.nan for b in range(10)]
    middle.plot([0, 0.75], [0, 0.75], linestyle="--", color="#999999", linewidth=1.0,
                label="perfect calibration")
    middle.plot(predicted, observed, marker="o", color="#4C78A8", linewidth=1.6, markersize=6,
                label="decile of the score (held-out)")
    middle.axvline(CHOSEN, color="#54A24B", linestyle="--", linewidth=1.2)
    middle.annotate(f"chosen cut {CHOSEN:.4f}", xy=(CHOSEN, 0.02), xytext=(6, 0),
                    textcoords="offset points", fontsize=8.5, color="#54A24B")
    middle.set_xlabel("mean predicted probability in the decile")
    middle.set_ylabel("observed subscription rate")
    middle.set_title("why the score can be priced\n(reliability on 9,043 held-out rows)", fontsize=11)
    middle.legend(loc="upper left", fontsize=8.5)
    middle.grid(alpha=0.25)

    costs = sorted({row["call_cost_eur"] for row in sens})
    values = sorted({row["value_eur"] for row in sens})
    matrix = np.array([
        [next(r["threshold"] for r in sens if r["call_cost_eur"] == c and r["value_eur"] == v)
         for v in values]
        for c in costs
    ])
    image = right.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0.0, vmax=0.3)
    for row_index, cost_value in enumerate(costs):
        for col_index, value in enumerate(values):
            cell = matrix[row_index, col_index]
            right.text(col_index, row_index, f"{cell:.3f}", ha="center", va="center", fontsize=9,
                       color="white" if cell > 0.2 else "#1A1A1A")
    right.set_xticks(range(len(values)), [f"{v:g}" for v in values])
    right.set_yticks(range(len(costs)), [f"{c:g}" for c in costs])
    right.set_xlabel("value per subscription V (EUR)")
    right.set_ylabel("cost per call C (EUR)")
    right.set_title("the price moves with the assumptions\n(cell = break-even C/V)", fontsize=11)
    figure.colorbar(image, ax=right, fraction=0.046, pad=0.04, label="threshold")

    figure.suptitle(
        f"DepositIQ - the operating threshold is a price: C = EUR {cost.CALL_COST_EUR:.2f}, "
        f"V = EUR {cost.VALUE_PER_SUBSCRIPTION_EUR:.2f}, so call when P(subscribe) > "
        f"{CHOSEN:.4f}\nThe chosen cut nets {choice['at_chosen']['net_value_eur']:,.0f} EUR on the "
        f"training half and {holdout['at_chosen']['net_value_eur']:,.0f} EUR on the held-out rows",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(variant: str = "full") -> dict:
    """Choose on the training folds, check on the held-out rows, draw the figure, write the record."""
    config.ensure_dirs()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    frame = features.load_engineered("train", variant)
    y = encode_target(frame)
    columns = select.shipped_features()
    folds = train.materialise_folds(frame, y)

    scores, info = oof_scores(frame, y, folds, columns)
    chosen = choose(y.to_numpy(), scores, grid())
    capacity = capacity_table(y.to_numpy(), scores)
    sens = sensitivity()

    # Only now, with the choice fixed, is the held-out half touched.
    holdout = held_out_check(CHOSEN)
    draw(chosen["curve"], chosen, sens, holdout)

    payload = {
        "task": "T34",
        "variant": variant,
        "decision": {
            "threshold": CHOSEN,
            "rule": "call a client when P(subscribe) exceeds C/V",
            "chosen_on": "training half, out-of-fold scores from the shipped configuration",
            "checked_on": "held-out half, persisted artifact, after the choice was fixed",
            "why_not_f1": "the F1 cut optimises a count of errors; the bank pays for calls and "
                          "subscriptions, which is a price, not a rate",
            "why_not_accuracy": "accuracy peaks at 0.5 on this class balance and would call almost nobody "
                                "(D53, and T33's 801 missed subscribers)",
        },
        "prices": {
            "call_cost_eur": cost.CALL_COST_EUR,
            "value_per_subscription_eur": cost.VALUE_PER_SUBSCRIPTION_EUR,
            "break_even": CHOSEN,
            "assumptions": cost.ASSUMPTIONS,
        },
        "training": {
            "rows": int(len(frame)),
            "positives": int(y.sum()),
            "base_rate": float(y.mean()),
            "folds": train.N_FOLDS,
            "fold_seed": train.FOLD_SEED,
        },
        "at_chosen": chosen["at_chosen"],
        "empirical_best": chosen["empirical_best"],
        # The cuts the phase argues against, measured rather than remembered: the report must not quote a
        # euro figure that exists only in a check's detail string.
        "comparisons": [
            cost.expected_campaign_value(y.to_numpy(), scores, value)
            for value in (0.0, CHOSEN, F1_CUT, CONVENTIONAL, chosen["empirical_best"]["threshold"])
        ],
        "plateau": chosen["plateau"],
        "chosen_within_plateau": chosen["chosen_within_plateau"],
        "shortfall_vs_best_eur": chosen["shortfall_vs_best_eur"],
        "shortfall_pct": chosen["shortfall_pct"],
        "curve_points": len(chosen["curve"]),
        # The four numbers a reader needs to redraw the left panel: everything else in a curve point follows
        # from them (revenue = reached x V, cost = calls x C, missed = positives - reached).
        "curve": [
            {key: row[key] for key in ("threshold", "clients_called", "subscribers_reached",
                                      "net_value_eur")}
            for row in chosen["curve"]
        ],
        "capacity": capacity,
        "sensitivity": sens,
        "held_out": holdout,
        "figure": {"path": str(FIGURE_PATH.relative_to(config.ROOT)),
                   "bytes": FIGURE_PATH.stat().st_size},
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }

    checks = validate(frame, y, folds, scores, info, chosen, capacity, sens, holdout)
    payload["checks"] = checks
    payload["checks_passed"] = sum(1 for check in checks if check["passed"])
    payload["checks_total"] = len(checks)
    payload["checks_ok"] = payload["checks_passed"] == payload["checks_total"]
    payload["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    RECORD_PATH.write_text(json.dumps(payload, indent=2, sort_keys=False))

    # Read the file back and require it to be this run's answers: a record written from one set of arrays
    # and quoted from memory is exactly the failure mode D06 exists to prevent.
    on_disk = json.loads(RECORD_PATH.read_text())
    same = (
        on_disk["decision"]["threshold"] == CHOSEN
        and len(on_disk["curve"]) == payload["curve_points"]
        and on_disk["held_out"]["at_chosen"]["net_value_eur"] == holdout["at_chosen"]["net_value_eur"]
        and len(on_disk["comparisons"]) == 5
        and on_disk["checks_total"] == len(checks)
        and on_disk["figure"]["bytes"] == FIGURE_PATH.stat().st_size
    )
    disk = {
        "check": "the record on disk is this run's answers, not a transcript of another",
        "passed": bool(same),
        "detail": f"{RECORD_PATH.name}: threshold {on_disk['decision']['threshold']:.4f}, "
                  f"{len(on_disk['curve'])} curve points, held-out net "
                  f"{on_disk['held_out']['at_chosen']['net_value_eur']:,.2f} EUR, "
                  f"{on_disk['checks_total']} checks before this one, "
                  f"figure {on_disk['figure']['bytes']:,} B",
    }
    payload["checks"].append(disk)
    payload["checks_passed"] = sum(1 for check in payload["checks"] if check["passed"])
    payload["checks_total"] = len(payload["checks"])
    payload["checks_ok"] = payload["checks_passed"] == payload["checks_total"]
    RECORD_PATH.write_text(json.dumps(payload, indent=2, sort_keys=False))

    return {**payload, "choice": chosen}


def print_report(result: dict) -> None:
    print("DepositIQ T34 - the operating threshold from the cost model")
    print("=" * 112)
    prices = result["prices"]
    print(f"price    : C = {prices['call_cost_eur']:.2f} EUR per call, V = "
          f"{prices['value_per_subscription_eur']:.2f} EUR per subscription -> call when "
          f"P(subscribe) > {prices['break_even']:.4f}")
    print(f"chosen   : {result['decision']['threshold']:.4f} ({result['decision']['chosen_on']})")
    print(f"training : {result['training']['rows']:,} rows, {result['training']['positives']:,} "
          f"subscribers, {result['training']['folds']}-fold out-of-fold")
    print()

    header = (f"{'threshold':>10}{'calls':>8}{'reached':>8}{'missed':>8}{'net EUR':>13}"
              f"{'ROI':>9}{'EUR/sub':>9}")
    print(header)
    print("-" * len(header))
    labels = ("call all", "CHOSEN C/V", "F1 cut", "0.5", "peak*")
    for label, row in zip(labels, result["comparisons"]):
        print(f"{row['threshold']:>10.4f}{row['clients_called']:>8,}{row['subscribers_reached']:>8,}"
              f"{row['subscribers_missed']:>8,}{row['net_value_eur']:>13,.0f}"
              f"{row['roi']:>8.1%}{row['cost_per_subscription_eur']:>9.2f}   {label}")
    print("  * the empirical peak on the training folds; not the adopted rule")
    print()
    plateau = result["plateau"]
    print(f"plateau  : every threshold in {plateau['low']:.4f}-{plateau['high']:.4f} is within "
          f"{plateau['tolerance']:.0%} of the peak ({plateau['net_floor_eur']:,.0f} EUR); the chosen cut "
          f"gives up {result['shortfall_vs_best_eur']:,.0f} EUR ({result['shortfall_pct']:.3f}%)")
    print()
    print("capacity : what a fixed number of calls buys")
    for row in sorted(result["capacity"], key=lambda r: r["capacity_share"]):
        print(f"  top {row['capacity_share']:>4.0%}  threshold {row['threshold']:.4f}  "
              f"{row['clients_called']:>6,} calls  {row['subscribers_reached']:>5,} reached  "
              f"{row['net_value_eur']:>10,.0f} EUR")
    print()
    held = result["held_out"]
    print(f"held-out : {held['rows']:,} rows, {held['positives']:,} subscribers")
    for label, row in (("chosen", held["at_chosen"]), ("F1 cut", held["at_f1_cut"]),
                       ("0.5", held["at_conventional"]), ("call all", held["at_call_everyone"])):
        print(f"  {label:<9} calls {row['clients_called']:>6,}  reached {row['subscribers_reached']:>5,}  "
              f"net {row['net_value_eur']:>10,.0f} EUR  ROI {row['roi']:>7.1%}")
    print(f"  the held-out rows would have chosen {float(held['held_out_best']['threshold']):.4f} "
          f"({held['held_out_best']['net_value_eur']:,.0f} EUR) - not adopted")
    print()
    print(f"wrote    : {result['figure']['path']} ({result['figure']['bytes']:,} B) + "
          f"{RECORD_PATH.name}")
    print(f"total    : {result['elapsed_seconds']} s")
    print()
    print(f"T34 checks: {result['checks_passed']}/{result['checks_total']} passed")
    for check in result["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        detail = f"  ({check['detail']})" if check["detail"] else ""
        print(f"  [{marker}] {check['check']}{detail}")
    print()


if __name__ == "__main__":
    print("DepositIQ Phase 7 - the operating threshold")
    print()
    outcome = report()
    print_report(outcome)
    print("=" * 112)
    verdict = "PASS - the threshold is the price, and the held-out rows agree with it"
    print(f"result : {verdict if outcome['checks_ok'] else 'FAIL'}")
    print("Next   : T35 turns the ranking into a lift table and a call budget.")
    sys.exit(0 if outcome["checks_ok"] else 1)
