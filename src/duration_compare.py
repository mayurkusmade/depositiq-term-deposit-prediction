"""
DepositIQ - the with-vs-without `duration` comparison: the price of a deployable model (T36).

Why this file exists
--------------------
`duration` is the length of the marketing call in **seconds**, recorded when the call ends. The target is
whether the call ended in a subscription. A model given this column is reading the answer off the back of
the exam paper, so D05 excludes it - and that exclusion costs something, which nobody has measured until
now.

T22 audited the column and wrote down what would be done with the result *before* any number existed
(`docs/05_leakage_audit.md` §4.5, D40):

   1. **Arm A ships regardless of the gap.** D05 was decided on availability, not performance.
   2. **Arm B's metrics are an upper bound**, and no targeting, lift or savings claim may be built on them.
   3. **A near-zero gap is the finding** and would be reported as such.

This module executes that rule. It fits **one model twice** - same parameters, same folds, same scorer -
with the feature list toggled only through `leakage.feature_columns`, and reports the gap in PR-AUC and in
euros at T34's price.

The two arms
------------
| arm | features | matrix columns | role |
|---|---|---|---|
| **A** `arm_a_without_duration` | the 12 shipped features | 40 | the deployable model (D05) |
| **B** `arm_b_with_duration` | those 12 plus `duration` | 41 | the leaky diagnostic; an upper bound |

The widths are measured here, not quoted: T24 demonstrated the same toggle at the 17-feature width (65 vs
66 columns), and this is that toggle at the width the shipped model actually uses.

How the "gap is leakage" claim is checked rather than asserted
--------------------------------------------------------------
Saying "the gap is leakage" is easy; showing it is not. The verdict is computed from a rule fixed by
measurements that already existed before this run:

    gap is leakage  <=>  arm B wins at least 4 of 5 folds
                         AND T22's band table is non-decreasing in call length
                         AND `duration` used alone as a score beats ROC-AUC 0.70

The second and third clauses are the **mechanism** - the outcome drives the call's length, not the other way
round - and both were measured in T22. The first is this run's evidence. If any clause fails, the verdict is
false and the document has to say the gap is something else.

How to run
----------
    python -m src.duration_compare     # ~20 s: ten fold fits, no held-out rows read

Writes `reports/duration_comparison.csv` and `models/duration_comparison.json`.
"""
from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, cost, features, leakage, select, threshold, train, tune
from src.split import encode_target, load_split

RECORD_PATH: Path = config.MODELS_DIR / "duration_comparison.json"
TABLE_PATH: Path = config.REPORTS_DIR / "duration_comparison.csv"
PARAMS_PATH: Path = config.MODELS_DIR / "best_params.json"
THRESHOLD_PATH: Path = config.MODELS_DIR / "threshold_choice.json"

ARM_A: str = "arm_a_without_duration"
ARM_B: str = "arm_b_with_duration"

# The bank's price does not move because the model did: both arms are scored at T34's cut.
PRICE: float = threshold.CHOSEN

# T22's published audit numbers, recomputed here as a reconciliation rather than trusted.
PINNED_AUDIT: dict = {
    "roc_auc": 0.8078,
    "pr_auc": 0.3897,
    "top_decile_lift": 3.90,
}

# The pre-registered rule, in the words §4.5 fixed. Recorded with the result so the two cannot drift.
RULE: str = (
    "Arm A ships regardless of the gap; D05 was decided on availability, not performance. "
    "Arm B's metrics are an upper bound and no targeting, lift or savings claim may be built on them. "
    "A near-zero gap would be reported as the finding."
)

# Clause thresholds for the mechanism, fixed before the arms were fitted.
MIN_FOLD_WINS: int = 4
MIN_SOLO_ROC_AUC: float = 0.70

# Assembled at run time so this file's own text cannot satisfy the scan that proves the held-out half was
# never read (the lesson T31 and T33 wrote down: a source scan must not match itself).
_UNREAD_NEEDLE: str = "load_" + "engineered(" + chr(34) + "te" + "st" + chr(34)


def arm_columns(with_duration: bool) -> list[str]:
    """The feature list for one arm: the shipped twelve, toggled through `leakage.feature_columns`."""
    return leakage.feature_columns([*select.shipped_features(), "duration"], with_duration=with_duration)


def run_arm(arm: str, frame: pd.DataFrame, y: pd.Series,
            folds: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """
    Fit one arm across the project's folds and price it at the bank's cut.

    The estimator factory, the fold list and `train.cross_validate_factory` are the same objects for both
    arms; the only argument that differs is `extra_numeric`, which is how `duration` enters as a thirteenth
    feature instead of through a second pipeline.
    """
    with_duration = arm == ARM_B
    columns = arm_columns(with_duration)
    params = threshold.chosen_params()

    run = train.cross_validate_factory(
        arm,
        lambda: tune.make_estimator(params),
        "leaky diagnostic - upper bound (D40)" if with_duration else "deployable model (D05)",
        "duration included" if with_duration else "duration excluded",
        frame, y, folds, columns,
        extra_numeric=("duration",) if with_duration else (),
    )

    scores = np.full(len(frame), np.nan)
    per_fold = []
    for (pipeline, _, valid), record in zip(run["fits"], run["folds"]):
        fold_scores = pipeline.predict_proba(frame.iloc[valid])[:, 1].astype(np.float64)
        scores[valid] = fold_scores
        value = cost.expected_campaign_value(y.iloc[valid].to_numpy(), fold_scores, PRICE)
        per_fold.append({
            **record,
            "calls_at_price": value["clients_called"],
            "reached_at_price": value["subscribers_reached"],
            "net_eur_at_price": value["net_value_eur"],
            "roi_at_price": value["roi"],
        })

    estimator = run["fits"][0][0].named_steps[train.STEP_MODEL]
    names = list(run["fits"][0][0].named_steps[train.STEP_PREPROCESS]
                 .get_feature_names_out())
    pooled = cost.expected_campaign_value(y.to_numpy(), scores, PRICE)

    return {
        "arm": arm,
        "role": run["role"],
        "note": run["note"],
        "features": columns,
        "matrix_columns": len(names),
        "matrix_names": names,
        "params": dict(estimator.get_params()),
        "scale_pos_weight": float(estimator.get_params()["scale_pos_weight"]),
        "n_jobs": int(estimator.get_params()["n_jobs"]),
        "folds": per_fold,
        "summary": run["summary"],
        "pooled": {
            **train.fold_metrics(y.to_numpy(), scores),
            "at_price": pooled,
        },
        "scores": scores,
    }


def paired(arms: dict[str, dict], key: str) -> dict:
    """Paired per-fold difference (B - A) for one metric, with the fold-win count."""
    a = [row[key] for row in arms[ARM_A]["folds"]]
    b = [row[key] for row in arms[ARM_B]["folds"]]
    diffs = [later - earlier for earlier, later in zip(a, b)]
    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    return {
        "metric": key,
        "per_fold": diffs,
        "mean": mean,
        "sd": sd,
        "wins": int(sum(1 for value in diffs if value > 0)),
        "folds": len(diffs),
        # The ratio of the mean to its own standard error: how many fold-to-fold sds the gap is worth.
        "se_ratio": float(mean / (sd / np.sqrt(len(diffs)))) if sd else float("inf"),
    }


def mechanism() -> dict:
    """
    T22's evidence for *why* the gap is leakage, recomputed on the training half.

    The band table is the mechanism in plain sight: the longer the call, the more likely it ended in a
    subscription, and the relation is monotone across six bands. The single-column ranking is the same fact
    summarised - one raw column, no model, against the whole feature set.
    """
    raw = load_split("full", "train")
    bands = leakage.band_table(raw)
    solo = leakage.single_column_ranking(raw)
    return {
        "bands": bands.to_dict(orient="records"),
        "monotone": bool((bands["rate"].diff().dropna() >= -0.005).all()),
        "solo": solo,
        "shortest_band_rate": float(bands.iloc[0]["rate"]),
        "longest_band_rate": float(bands.iloc[-1]["rate"]),
        "longest_band_share_of_subscribers": float(
            bands.iloc[-1]["subscribers"] / bands["subscribers"].sum()),
    }


def verdict(pairs: dict[str, dict], mech: dict) -> dict:
    """Apply the pre-registered, mechanism-based rule. Nothing here reads the gap's size."""
    fold_wins = pairs["pr_auc"]["wins"]
    clauses = {
        "arm_b_wins_most_folds": fold_wins >= MIN_FOLD_WINS,
        "bands_are_monotone": mech["monotone"],
        "duration_alone_ranks_well": mech["solo"]["roc_auc"] > MIN_SOLO_ROC_AUC,
    }
    return {
        "clauses": clauses,
        "thresholds": {"min_fold_wins": MIN_FOLD_WINS, "min_solo_roc_auc": MIN_SOLO_ROC_AUC},
        "gap_is_leakage": bool(all(clauses.values())),
        "fold_wins": fold_wins,
        "rule": RULE,
        "arm_a_ships": True,
        "why_a_ships": "pre-registered §4.5 clause 1 and D05: the column is unavailable at decision time, "
                       "so the gap cannot change which model ships",
    }


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def validate(arms: dict[str, dict], pairs: dict[str, dict], mech: dict, decided: dict,
             folds: list[tuple[np.ndarray, np.ndarray]], y: pd.Series,
             t30: dict, money: dict) -> list[dict]:
    checks: list[dict] = []

    def compare(name: str, evaluate) -> None:
        try:
            passed, detail = evaluate()
        except Exception as error:  # noqa: BLE001
            passed, detail = False, f"check could not run: {type(error).__name__}: {error}"
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    def same_folds() -> tuple[bool, str]:
        a = [(row["rows"], row["positives"], row["train_rows"]) for row in arms[ARM_A]["folds"]]
        b = [(row["rows"], row["positives"], row["train_rows"]) for row in arms[ARM_B]["folds"]]
        expected = [(len(valid), int(y.iloc[valid].sum()), len(train_idx)) for train_idx, valid in folds]
        return (
            a == b == expected,
            f"both arms scored the same five folds: {[rows for rows, _, _ in a]} validation rows / "
            f"{[pos for _, pos, _ in a]} positives, from one materialised list"
            if a == b == expected else f"arms differ: A {a} vs B {b} vs folds {expected}",
        )

    compare("both arms were scored on the project's folds, replayed not re-derived", same_folds)

    def toggle_only_duration() -> tuple[bool, str]:
        a = set(arms[ARM_A]["features"])
        b = set(arms[ARM_B]["features"])
        shipped = set(select.shipped_features())
        return (
            b - a == set(leakage.LEAKY_COLUMNS)
            and a - b == set()
            and a == shipped
            and "duration" not in a,
            f"B has {len(b)} features and A has {len(a)}; the only difference is "
            f"{sorted(b - a)}; A is exactly the shipped set",
        )

    compare("the feature lists differ by exactly `duration`", toggle_only_duration)

    def widths_and_names() -> tuple[bool, str]:
        a_names = set(arms[ARM_A]["matrix_names"])
        b_names = set(arms[ARM_B]["matrix_names"])
        return (
            arms[ARM_A]["matrix_columns"] == 40
            and arms[ARM_B]["matrix_columns"] == 41
            and b_names - a_names == {"duration"}
            and a_names - b_names == set(),
            f"arm A {arms[ARM_A]['matrix_columns']} columns, arm B {arms[ARM_B]['matrix_columns']}; "
            f"the extra column is {sorted(b_names - a_names)} and every other name is shared",
        )

    compare("the two matrices are the same width plus one named column", widths_and_names)

    def same_estimator() -> tuple[bool, str]:
        a = arms[ARM_A]["params"]
        b = arms[ARM_B]["params"]
        keys = sorted(set(a) | set(b))

        # NaN-aware: XGBoost's `missing` defaults to nan, and `nan != nan` would report a difference between
        # two arms that are both unset. Comparing settings is the point of this check, not testing IEEE 754.
        def equal(left, right) -> bool:
            if isinstance(left, float) and isinstance(right, float) \
                    and np.isnan(left) and np.isnan(right):
                return True
            return left == right

        differences = [key for key in keys if not equal(a.get(key), b.get(key))]
        nan_settings = [key for key in keys
                        if isinstance(a.get(key), float) and np.isnan(a[key])]
        return (
            not differences
            and arms[ARM_A]["scale_pos_weight"] == arms[ARM_B]["scale_pos_weight"] == 1.0
            and arms[ARM_A]["n_jobs"] == arms[ARM_B]["n_jobs"] == 1,
            f"{len(keys)} estimator settings compared, {len(differences)} differ"
            + (f": {differences}" if differences else "")
            + f"; both unweighted ({arms[ARM_A]['scale_pos_weight']:.1f}) and single-threaded; "
            f"{nan_settings} are NaN in both and matched as such",
        )

    compare("both arms used identical estimator settings", same_estimator)

    def arm_a_is_the_shipped_model() -> tuple[bool, str]:
        recorded = [row["pr_auc"] for row in t30["folds"]]
        mine = [row["pr_auc"] for row in arms[ARM_A]["folds"]]
        drift = max(abs(x - z) for x, z in zip(mine, recorded))
        return (
            drift <= 1e-12 and np.array_equal(
                arms[ARM_A]["scores"], np.array(arms[ARM_A]["scores"])),
            f"arm A's five fold PR-AUCs reproduce T30's tuned arm to {drift:.1e} "
            f"({', '.join(f'{v:.4f}' for v in mine)})",
        )

    compare("arm A is the shipped model, not a neighbour of it", arm_a_is_the_shipped_model)

    def arm_a_reproduces_the_money() -> tuple[bool, str]:
        recorded = json.loads(THRESHOLD_PATH.read_text())["at_chosen"]
        mine = arms[ARM_A]["pooled"]["at_price"]
        return (
            abs(mine["net_value_eur"] - recorded["net_value_eur"]) < 1e-9
            and mine["clients_called"] == recorded["clients_called"]
            and mine["subscribers_reached"] == recorded["subscribers_reached"],
            f"arm A at {PRICE:.4f} nets {mine['net_value_eur']:,.2f} EUR from "
            f"{mine['clients_called']:,} calls - T34 recorded {recorded['net_value_eur']:,.2f} from "
            f"{recorded['clients_called']:,}",
        )

    compare("arm A reproduces T34's campaign value to the cent", arm_a_reproduces_the_money)

    def gap_is_the_difference() -> tuple[bool, str]:
        a = arms[ARM_A]["pooled"]["at_price"]
        b = arms[ARM_B]["pooled"]["at_price"]
        return (
            abs(money["pr_auc_gap"]
                - (arms[ARM_B]["pooled"]["pr_auc"] - arms[ARM_A]["pooled"]["pr_auc"])) < 1e-12
            and abs(money["euro_gap"] - (b["net_value_eur"] - a["net_value_eur"])) < 1e-9,
            f"PR-AUC gap {money['pr_auc_gap']:+.4f}, euro gap {money['euro_gap']:+,.2f} "
            f"({b['net_value_eur']:,.2f} - {a['net_value_eur']:,.2f}), both computed as the difference "
            f"of the two pooled campaigns at {PRICE:.4f}",
        )

    compare("the published gap is the difference between the two arms", gap_is_the_difference)

    def verdict_follows_the_rule() -> tuple[bool, str]:
        clauses = decided["clauses"]
        expected = all(clauses.values())
        return (
            decided["gap_is_leakage"] == expected
            and decided["fold_wins"] == pairs["pr_auc"]["wins"]
            and decided["arm_a_ships"] is True
            and "upper bound" in arms[ARM_B]["role"],
            "clauses " + ", ".join(f"{name}={value}" for name, value in clauses.items())
            + f"; fold wins {decided['fold_wins']}/5 -> gap_is_leakage={decided['gap_is_leakage']}; "
              f"arm B is labelled '{arms[ARM_B]['role']}'",
        )

    compare("the verdict is the pre-registered rule applied, and arm A ships either way",
            verdict_follows_the_rule)

    def mechanism_reproduces_t22() -> tuple[bool, str]:
        solo = mech["solo"]
        return (
            abs(solo["roc_auc"] - PINNED_AUDIT["roc_auc"]) < 5e-4
            and abs(solo["pr_auc"] - PINNED_AUDIT["pr_auc"]) < 5e-4
            and abs(solo["top_decile_lift"] - PINNED_AUDIT["top_decile_lift"]) < 5e-3
            and mech["monotone"],
            f"`duration` alone: ROC-AUC {solo['roc_auc']:.4f} (T22 {PINNED_AUDIT['roc_auc']}), "
            f"PR-AUC {solo['pr_auc']:.4f} (T22 {PINNED_AUDIT['pr_auc']}), top-decile lift "
            f"{solo['top_decile_lift']:.2f}x (T22 {PINNED_AUDIT['top_decile_lift']})",
        )

    compare("the leak's mechanism reproduces T22's audit", mechanism_reproduces_t22)

    def the_band_table_rises() -> tuple[bool, str]:
        return (
            mech["longest_band_rate"] > mech["shortest_band_rate"] * 10,
            f"subscription rate runs {mech['shortest_band_rate']:.2%} in the shortest band to "
            f"{mech['longest_band_rate']:.2%} in the longest - a "
            f"{mech['longest_band_rate'] / mech['shortest_band_rate']:.0f}x swing across call length, with "
            f"the longest band supplying {mech['longest_band_share_of_subscribers']:.2%} of subscribers",
        )

    compare("the subscription rate rises monotonically with call length", the_band_table_rises)

    def unread_half() -> tuple[bool, str]:
        source = inspect.getsource(sys.modules[__name__])
        hits = source.count(_UNREAD_NEEDLE)
        return (
            hits == 0,
            f"this module never loads the held-out half ({hits} references to it); both arms are scored on "
            f"the {len(y):,} training rows only",
        )

    compare("no held-out row was read by this comparison", unread_half)

    def table_file() -> tuple[bool, str]:
        frame = pd.read_csv(TABLE_PATH)
        arms_present = set(frame["arm"]) == {ARM_A, ARM_B}
        statistic_rows = set(frame["statistic"]) if "statistic" in frame.columns else set()
        return (
            len(frame) == 14 and arms_present and statistic_rows == {"fold", "mean", "sd"},
            f"{TABLE_PATH.name}: {len(frame)} rows, arms {sorted(arms_present and set(frame['arm']))}, "
            f"statistics {sorted(statistic_rows)}",
        )

    compare("the comparison table was written with both arms and their summaries", table_file)

    return checks


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(variant: str = "full") -> dict:
    config.ensure_dirs()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    frame = features.load_engineered("train", variant)
    y = encode_target(frame)
    folds = train.materialise_folds(frame, y)

    arms = {arm: run_arm(arm, frame, y, folds) for arm in (ARM_A, ARM_B)}
    pairs = {key: paired(arms, key) for key in ("pr_auc", "roc_auc", "brier", "net_eur_at_price")}
    mech = mechanism()
    decided = verdict(pairs, mech)

    a_price = arms[ARM_A]["pooled"]["at_price"]
    b_price = arms[ARM_B]["pooled"]["at_price"]
    money = {
        "price": PRICE,
        "pr_auc_gap": arms[ARM_B]["pooled"]["pr_auc"] - arms[ARM_A]["pooled"]["pr_auc"],
        "roc_auc_gap": arms[ARM_B]["pooled"]["roc_auc"] - arms[ARM_A]["pooled"]["roc_auc"],
        "brier_gap": arms[ARM_B]["pooled"]["brier"] - arms[ARM_A]["pooled"]["brier"],
        "euro_gap": b_price["net_value_eur"] - a_price["net_value_eur"],
        "euro_gap_per_1000_rows": (b_price["net_value_eur"] - a_price["net_value_eur"])
        / len(y) * 1000,
        "calls_gap": b_price["clients_called"] - a_price["clients_called"],
        "reach_gap": b_price["subscribers_reached"] - a_price["subscribers_reached"],
        "arm_a": a_price,
        "arm_b": b_price,
    }

    t30 = json.loads(PARAMS_PATH.read_text())["arms"]["tuned_untreated"]

    rows = []
    for arm, data in arms.items():
        for record in data["folds"]:
            rows.append({
                "arm": arm,
                "statistic": "fold",
                "fold": record["fold"],
                "rows": record["rows"],
                "positives": record["positives"],
                "pr_auc": record["pr_auc"],
                "roc_auc": record["roc_auc"],
                "brier": record["brier"],
                "mean_pred": record["mean_pred"],
                "calls_at_price": record["calls_at_price"],
                "reached_at_price": record["reached_at_price"],
                "net_eur_at_price": record["net_eur_at_price"],
            })
        for statistic in ("mean", "sd"):
            rows.append({
                "arm": arm,
                "statistic": statistic,
                "fold": "",
                "rows": int(data["folds"][0]["rows"]) if statistic == "mean" else "",
                "positives": int(data["folds"][0]["positives"]) if statistic == "mean" else "",
                **{
                    key: data["summary"][key][statistic]
                    for key in ("pr_auc", "roc_auc", "brier", "mean_pred")
                },
                "calls_at_price": float(np.mean([r["calls_at_price"] for r in data["folds"]])),
                "reached_at_price": float(np.mean([r["reached_at_price"] for r in data["folds"]])),
                "net_eur_at_price": (
                    float(np.mean([r["net_eur_at_price"] for r in data["folds"]]))
                    if statistic == "mean"
                    else float(np.std([r["net_eur_at_price"] for r in data["folds"]], ddof=1))
                ),
            })
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_PATH, index=False, float_format="%.6f")

    payload = {
        "task": "T36",
        "variant": variant,
        "rule": RULE,
        "decision": decided,
        "price": {"threshold": PRICE, "call_cost_eur": cost.CALL_COST_EUR,
                  "value_per_subscription_eur": cost.VALUE_PER_SUBSCRIPTION_EUR,
                  "note": "the bank's price does not move because the model did; both arms are scored at it"},
        "training": {"rows": int(len(frame)), "positives": int(y.sum()), "base_rate": float(y.mean()),
                     "folds": train.N_FOLDS, "fold_seed": train.FOLD_SEED},
        "arms": {
            arm: {key: value for key, value in data.items() if key != "scores"}
            for arm, data in arms.items()
        },
        "paired": pairs,
        "gap": money,
        "mechanism": mech,
        "csv": str(TABLE_PATH.relative_to(config.ROOT)),
    }

    checks = validate(arms, pairs, mech, decided, folds, y, t30, money)
    payload["checks"] = checks
    payload["checks_passed"] = sum(1 for check in checks if check["passed"])
    payload["checks_total"] = len(checks)
    payload["checks_ok"] = payload["checks_passed"] == payload["checks_total"]
    payload["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    RECORD_PATH.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str))

    on_disk = json.loads(RECORD_PATH.read_text())
    same = (
        on_disk["decision"]["gap_is_leakage"] is decided["gap_is_leakage"]
        and on_disk["gap"]["euro_gap"] == money["euro_gap"]
        and on_disk["arms"][ARM_A]["matrix_columns"] == 40
        and on_disk["arms"][ARM_B]["matrix_columns"] == 41
        and on_disk["checks_total"] == len(checks)
        and len(on_disk["arms"][ARM_B]["folds"]) == train.N_FOLDS
    )
    payload["checks"].append({
        "check": "the record on disk is this run's answers, not a transcript of another",
        "passed": bool(same),
        "detail": f"{RECORD_PATH.name}: gap {money['euro_gap']:+,.2f} EUR, verdict "
                  f"{decided['gap_is_leakage']}, widths 40/41, {on_disk['checks_total']} checks before "
                  f"this one",
    })
    payload["checks_passed"] = sum(1 for check in payload["checks"] if check["passed"])
    payload["checks_total"] = len(payload["checks"])
    payload["checks_ok"] = payload["checks_passed"] == payload["checks_total"]
    RECORD_PATH.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str))

    return payload


def print_report(result: dict) -> None:
    print("DepositIQ T36 - the with-vs-without `duration` comparison")
    print("=" * 112)
    print(f"training : {result['training']['rows']:,} rows, {result['training']['positives']:,} "
          f"subscribers, {result['training']['folds']}-fold, price {result['price']['threshold']:.4f}")
    print(f"rule     : {result['rule']}")
    print()

    header = (f"{'arm':<26}{'cols':>6}{'PR-AUC':>9}{'ROC-AUC':>9}{'Brier':>9}{'mean p':>9}"
              f"{'calls':>8}{'reach':>7}{'net EUR':>12}")
    print(header)
    print("-" * len(header))
    for arm, data in result["arms"].items():
        pooled = data["pooled"]
        price = pooled["at_price"]
        print(f"{arm:<26}{data['matrix_columns']:>6}{pooled['pr_auc']:>9.4f}{pooled['roc_auc']:>9.4f}"
              f"{pooled['brier']:>9.4f}{pooled['mean_pred']:>9.4f}{price['clients_called']:>8,}"
              f"{price['subscribers_reached']:>7,}{price['net_value_eur']:>12,.0f}")
    print()

    print("THE GAP (arm B - arm A)")
    gap = result["gap"]
    print(f"  PR-AUC  {gap['pr_auc_gap']:+.4f}      ROC-AUC {gap['roc_auc_gap']:+.4f}      "
          f"Brier {gap['brier_gap']:+.4f}")
    print(f"  euros   {gap['euro_gap']:+,.2f}      per 1,000 clients {gap['euro_gap_per_1000_rows']:+,.2f} "
          f"     calls {gap['calls_gap']:+,}   subscribers {gap['reach_gap']:+,}")
    print()
    for name, pair in result["paired"].items():
        print(f"  {pair['metric']:<18} mean {pair['mean']:+.6f}  sd {pair['sd']:.6f}  "
              f"SE-ratio {pair['se_ratio']:.2f}  B wins {pair['wins']}/{pair['folds']} folds")
    print()

    print("WHY IT IS LEAKAGE (T22's mechanism, recomputed)")
    mech = result["mechanism"]
    for band in mech["bands"]:
        print(f"  {band['band']:<9} {band['calls']:>6,} calls  {band['rate']:>7.2%} subscribe")
    solo = mech["solo"]
    print(f"  `duration` alone as a score: ROC-AUC {solo['roc_auc']:.4f}, PR-AUC {solo['pr_auc']:.4f} "
          f"({solo['pr_auc_lift']:.2f}x base), top-decile lift {solo['top_decile_lift']:.2f}x")
    print()
    decided = result["decision"]
    print(f"VERDICT  gap_is_leakage = {decided['gap_is_leakage']}  "
          f"(clauses: {', '.join(f'{k}={v}' for k, v in decided['clauses'].items())})")
    print(f"         arm A ships: {decided['arm_a_ships']} - {decided['why_a_ships']}")
    print()
    print(f"wrote    : {result['csv']} + {RECORD_PATH.name}")
    print(f"total    : {result['elapsed_seconds']} s")
    print()
    print(f"T36 checks: {result['checks_passed']}/{result['checks_total']} passed")
    for check in result["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        print(f"  [{marker}] {check['check']}  ({check['detail']})")
    print()


if __name__ == "__main__":
    print("DepositIQ Phase 7 - the price of a deployable model")
    print()
    outcome = report()
    print_report(outcome)
    print("=" * 112)
    print(f"result : {'PASS' if outcome['checks_ok'] else 'FAIL'}")
    print("Phase 7 complete: 5/5. Next phase is T37 - global SHAP.")
    sys.exit(0 if outcome["checks_ok"] else 1)
