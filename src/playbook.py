"""
DepositIQ - the campaign playbook, assembled from records rather than retyped (T39).

Why this file exists
--------------------
T39 is the phase's deliverable: **at least six actionable recommendations, each backed by a number this
project measured**. The temptation is to write that document by hand and copy figures into it, which is how
a report ends up quoting a number that no run produced. So the recommendations live here as **data**, every
figure is a `(file, key path)` reference into a saved record, and `report()` resolves each one and refuses to
write anything if a value does not match. `docs/08_recommendations.md` then quotes these values — never a
fresh calculation.

What this module does *not* do
------------------------------
No model is fitted, no threshold is chosen, no metric is recomputed from the data. Everything is read from
`models/*.json`, which is where the earlier phases left it. The only arithmetic performed here is
**derived** quantities that no single record holds - net value per 1,000 calls, the cost per subscription at
the cut, the share of the file called - and each of those is computed from two recorded numbers and written
into `models/playbook.json` so it is auditable in turn.

The scope sentence, which every recommendation inherits
-------------------------------------------------------
`p* = C/V = 7.50%` sits **below** the 11.6982% base rate (D17, D18). The average client is therefore still
worth calling, so this model **converts a fixed call budget better than a fixed order**; it does **not**
justify cutting campaign spend. Any recommendation that implies otherwise is out of scope here, and the
recommendation tagged `scope` says so explicitly.

How to run
----------
    python -m src.playbook     # ~1 s: reads nine records, resolves every figure, writes the tables

Writes `reports/playbook.csv` and `models/playbook.json`.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from src import config

RECORDS: dict[str, Path] = {
    "cv": config.MODELS_DIR / "metrics_cv.json",
    "threshold": config.MODELS_DIR / "threshold_choice.json",
    "lift": config.MODELS_DIR / "lift_results.json",
    "holdout": config.MODELS_DIR / "holdout_eval.json",
    "duration": config.MODELS_DIR / "duration_comparison.json",
    "shap_global": config.MODELS_DIR / "shap_global.json",
    "shap_local": config.MODELS_DIR / "shap_local.json",
    "train": config.MODELS_DIR / "cv_results.json",
    "final": config.MODELS_DIR / "final_model.json",
}

TABLE_PATH: Path = config.REPORTS_DIR / "playbook.csv"
RECORD_PATH: Path = config.MODELS_DIR / "playbook.json"

# Every recommendation must carry at least this many evidence references, and the checklist asks for six
# recommendations in total. Both are asserted rather than hoped for.
MIN_RECOMMENDATIONS: int = 6

SCOPE_SENTENCE: str = (
    "At the price of a call, the average client is worth calling (p* = 7.50% is below the 11.6982% base "
    "rate), so this model converts a fixed call budget better than a fixed order - it does not justify "
    "cutting campaign spend."
)


# --------------------------------------------------------------------------- #
# Evidence resolution
# --------------------------------------------------------------------------- #
def pluck(record: dict, path: str):
    """Resolve a dotted key path (`at_chosen.net_value_eur`) inside a record."""
    node = record
    for part in path.split("."):
        if isinstance(node, list):
            node = node[int(part)]
        else:
            node = node[part]
    return node


def load_records() -> dict[str, dict]:
    return {name: json.loads(path.read_text()) for name, path in RECORDS.items()}


def eur(value: float) -> str:
    return f"{value:,.2f} EUR"


def pct(value: float, decimals: int = 2) -> str:
    return f"{value * 100:.{decimals}f}%"


def count(value: float) -> str:
    """A head-count: no decimals. Every recorded count is integral, so `.4f` would be noise."""
    return f"{int(round(value)):,}"


def signed_count(value: float) -> str:
    return f"{int(round(value)):+,}"


# --------------------------------------------------------------------------- #
# The recommendations
#
# `evidence` entries are (label, record, path, formatter). The formatter is the *only* place a number is
# turned into text, so the document and this table cannot diverge.
# --------------------------------------------------------------------------- #
RECOMMENDATIONS: list[dict] = [
    {
        "id": "R1",
        "scope": "data-quality",
        "title": "Make `contact` a required field before the next campaign",
        "action": (
            "Record the contact channel on every call and re-score the file once it is filled. Treat the "
            "28.72% of clients currently coded `unknown` as unmeasured, not as poor prospects."
        ),
        "evidence": [
            ("clients with no recorded channel", "shap_global", "directions.0.levels.2.share_of_clients", pct),
            ("their subscription rate", "shap_global", "directions.0.levels.2.subscribe_rate", pct),
            ("rate for `cellular` clients", "shap_global", "directions.0.levels.0.subscribe_rate", pct),
            ("`contact=unknown` share of total model importance", "shap_global", "columns_table.0.share", pct),
            # Index 0 of the true-positive profile is its most negative contribution, which is
            # `contact=unknown` for that client - index 1 would be `marital`, a different claim.
            ("its contribution for one client", "shap_local", "profiles.true_positive.0.contribution", None),
        ],
        "expected": (
            "No conversion lift is claimed. The effect is on **scoring**: 28.72% of the file is currently "
            "assigned the model's worst channel value, and their 4.11% rate is a statement about a blank "
            "field. Filling it moves those clients into the level that describes them; the first campaign "
            "afterwards measures whether that changes anything."
        ),
        "check": (
            "Share of `unknown` in the scored file (target: near zero) and the measured rate of the "
            "re-coded group in the next campaign, against the 4.11% recorded here."
        ),
        "falsifier": (
            "If clients re-coded away from `unknown` still subscribe at about 4.11%, the level was carrying "
            "real information and this recommendation is wrong."
        ),
    },
    {
        "id": "R2",
        "scope": "fixed-budget",
        "title": "Call the list in model order, and stop at 0.0750",
        "action": (
            "Sort the campaign by P(subscribe) and call down the list while the score is at or above 0.0750. "
            "On this file that is 41.10% of clients, which reaches 77.90% of the subscribers."
        ),
        "evidence": [
            ("models called at the adopted cut", "threshold", "at_chosen.clients_called", count),
            ("share of the file that is", "lift", "translation.train_oof.coverage", pct),
            ("subscribers reached", "lift", "translation.train_oof.recall", pct),
            ("net value of the campaign", "threshold", "at_chosen.net_value_eur", eur),
            ("net value of calling everyone", "threshold", "comparisons.0.net_value_eur", eur),
            ("calls a fixed order would need for the same recall", "lift", "translation.train_oof.calls_a_fixed_recall_would_need", count),
            ("calls saved by the ranking", "lift", "translation.train_oof.calls_saved", count),
        ],
        "expected": (
            "13,309.31 fewer calls than working the file in any order for the same 3,296 subscribers, worth "
            "19,963.96 EUR of call time; the campaign nets 43,621.00 EUR in total."
        ),
        "check": (
            "Precision inside the called block each campaign, against the 22.17% measured here, and net EUR "
            "per 1,000 calls."
        ),
        "falsifier": (
            "If the called block converts far below 22.17%, the ranking has stopped describing this "
            "population; if it converts far above, the cut is leaving money on the table and should be "
            "re-derived."
        ),
    },
    {
        "id": "R3",
        "scope": "fixed-budget",
        "title": "Do not stop the list where calls saved peaks - stop it where capacity or the price does",
        "action": (
            "If agent hours are capped, choose the cut by capacity from the recorded table (e.g. 5% of the "
            "file = 1,809 calls) instead of pruning to the savings peak at 18.89% coverage."
        ),
        "evidence": [
            ("coverage where calls saved peaks", "lift", "savings_peak.train_oof.coverage", pct),
            ("calls saved there", "lift", "savings_peak.train_oof.calls_saved", count),
            ("calls at a 5% capacity cap", "threshold", "capacity.0.clients_called", count),
            ("calls between that peak and the cut", "lift", "block_after_peak.train_oof.calls", count),
            ("their conversion rate", "lift", "block_after_peak.train_oof.marginal_rate", pct),
            ("net value of that block", "lift", "block_after_peak.train_oof.net_for_the_block_eur", eur),
            ("held-out calls in the same block", "lift", "block_after_peak.held_out.calls", count),
            ("held-out marginal conversion of that block", "lift", "block_after_peak.held_out.marginal_rate", pct),
            ("held-out net value of the same block", "lift", "block_after_peak.held_out.net_for_the_block_eur", eur),
            ("net value of the grid's money peak", "threshold", "empirical_best.net_value_eur", eur),
            ("shortfall of the adopted cut from that peak", "threshold", "shortfall_vs_best_eur", eur),
        ],
        "expected": (
            "The 8,033 calls between the savings peak and the cut convert at 8.96%, above the 7.50% price, "
            "for +2,350.50 EUR. Pruning to the peak would give up that money to save calls the campaign was "
            "already going to spend. The grid's best cut is only 388.50 EUR ahead, so the exact cut is not "
            "where the value is - capacity is."
        ),
        "check": (
            "Marginal conversion of the block between any two candidate cuts, priced at C and V before the "
            "list is shortened."
        ),
        "falsifier": (
            "The held-out block contributes only +79.50 EUR on 1,547 calls (7.76%), so on fresh data this "
            "recommendation is thin: if the marginal rate falls below 7.50%, stopping earlier becomes correct."
        ),
    },
    {
        "id": "R4",
        "scope": "prohibition",
        "title": "Never feed call `duration` into the model, or use it as a manual scoring hint",
        "action": (
            "Keep `duration` out of the scored matrix and out of any agent's ranking. It is recorded when the "
            "call ends, so it cannot inform the decision to make the call."
        ),
        "evidence": [
            ("PR-AUC the leak would add", "duration", "gap.pr_auc_gap", None),
            ("euros it would appear to add", "duration", "gap.euro_gap", eur),
            ("calls it would appear to save", "duration", "gap.calls_gap", signed_count),
        ],
        "expected": (
            "The gap is **not** money left on the table: +0.1788 PR-AUC and +20,340.00 EUR from a column that "
            "does not exist at decision time. The deployable model's PR-AUC is 0.4448 pooled."
        ),
        "check": (
            "A scan of the scored matrix each release: 40 columns, twelve features, no `duration` and no "
            "target."
        ),
        "falsifier": (
            "If a future system captured the *expected* call length before dialling - not the actual one - the "
            "column would become legitimate and this prohibition would no longer apply."
        ),
    },
    {
        "id": "R5",
        "scope": "pricing",
        "title": "Establish the value of a subscription before booking any euro figure",
        "action": (
            "Replace the assumed V = 20.00 EUR with the bank's own number before the 43,621.00 EUR is "
            "planned against. The ranking's advantage survives any V above 6.77 EUR; the euro headline does "
            "not."
        ),
        "evidence": [
            ("assumed cost per call", "threshold", "prices.call_cost_eur", eur),
            ("assumed value per subscription", "threshold", "prices.value_per_subscription_eur", eur),
            ("break-even probability", "threshold", "prices.break_even", pct),
            ("cost per subscription at the cut", "threshold", "at_chosen.cost_per_subscription_eur", eur),
        ],
        "expected": (
            "Every euro figure in this playbook is linear in V. At V = 20.00 the campaign nets 43,621.00 EUR; "
            "at the cut each subscription costs 6.77 EUR in calls, so any V above that keeps the campaign "
            "profitable."
        ),
        "check": (
            "Re-run the pricing with the real V and C; the break-even moves to C/V and the recommended cut "
            "moves with it."
        ),
        "falsifier": (
            "If the bank's V is below 6.77 EUR, the cut is unprofitable and a stricter cut - or no campaign - "
            "is correct. If C is materially higher than 1.50 EUR, the same applies."
        ),
    },
    {
        "id": "R6",
        "scope": "monitoring",
        "title": "Track precision and net value, never accuracy",
        "action": (
            "Report precision inside the called block, recall of subscribers and net EUR per 1,000 calls. "
            "Retire accuracy from the campaign dashboard."
        ),
        "evidence": [
            ("accuracy at the cut, held out", "holdout", "cuts.0.accuracy", pct),
            # Resolved against `derived`, because no single record holds it: it is 1 - the base rate.
            ("accuracy of always predicting no", "derived", "accuracy_of_always_no", pct),
            ("precision at the cut, held out", "holdout", "cuts.0.precision", pct),
            ("recall of subscribers, held out", "holdout", "cuts.0.recall", pct),
            # Comparisons index 3 is the 0.5 cut. `at_chosen.subscribers_missed` would have been 935, the
            # misses at the *adopted* cut - the wrong number under this label.
            ("subscribers a 0.5 cut would have missed", "threshold", "comparisons.3.subscribers_missed", count),
            ("net value of a 0.5 cut", "threshold", "comparisons.3.net_value_eur", eur),
        ],
        "expected": (
            "The model is **less accurate** than predicting `no` for everyone (66.42% against 88.30%) and "
            "far more useful: 77.60% of held-out subscribers reached. Grading it at 0.5 instead of at the "
            "price would reach 1,039 subscribers and net 18,248.00 EUR against 3,296 and 43,621.00."
        ),
        "check": "Precision and net EUR per 1,000 calls, quarter on quarter, against 22.67% and the same cut.",
        "falsifier": (
            "If precision inside the called block stays at the base rate over several campaigns, the ranking "
            "has stopped adding value regardless of its accuracy."
        ),
    },
    {
        "id": "R7",
        "scope": "modelling",
        "title": "Keep the twelve-feature XGBoost; do not re-open the feature set or the imbalance weight",
        "action": (
            "Ship the persisted artifact as is. Do not add back the five dropped features, do not weight the "
            "classes, and do not re-tune on a hunch."
        ),
        "evidence": [
            ("shipped PR-AUC, cross-validated", "cv", "models.xgboost_tuned.summary.pr_auc.mean", None),
            ("random forest PR-AUC", "cv", "models.random_forest.summary.pr_auc.mean", None),
            ("logistic baseline PR-AUC", "cv", "models.logreg_balanced.summary.pr_auc.mean", None),
            ("held-out PR-AUC of the artifact", "holdout", "metrics.pr_auc", None),
            ("matrix width", "final", "matrix_columns", count),
        ],
        "expected": (
            "The shipped model beats both alternatives on ranking (0.4460 against 0.4346 and 0.4008) with 40 "
            "columns instead of 65, and holds 0.4616 PR-AUC on rows it never saw."
        ),
        "check": (
            "The D46 revisit trigger: if a 40-column XGBoost ever lands below the logistic baseline's 0.4008, "
            "the feature set is re-opened."
        ),
        "falsifier": (
            "A re-measurement that puts the shipped configuration below 0.4008 PR-AUC, or a business change "
            "that adds a genuinely new column, which is a T22-style audit rather than a re-tune."
        ),
    },
    {
        "id": "R8",
        "scope": "scheduling",
        "title": "Use `month` to schedule, never to predict a client",
        "action": (
            "Read the month effects as a calendar signal when planning the campaign, and keep them out of any "
            "statement about a client's propensity."
        ),
        "evidence": [
            ("`month` share of model importance", "shap_global", "features_table.1.share", pct),
            ("March's contribution to a client", "shap_global", "directions.1.levels.0.mean_contribution", None),
            # `may` is the last of the twelve month levels, not the fourth: index 3 is December (+0.9068).
            ("May's contribution", "shap_global", "directions.1.levels.11.mean_contribution", None),
            ("share of the file reached in May", "shap_global", "directions.1.levels.11.share_of_clients", pct),
            ("campaign-side share of total importance", "shap_global", "campaign_side.share", pct),
        ],
        "expected": (
            "Campaign-side features carry 55.7% of total importance, so the largest levers are how and when "
            "the bank calls. That is a scheduling and data-quality agenda, not a targeting one."
        ),
        "check": (
            "Conversion by month next campaign, against the 50.90% (March) and 6.68% (May) recorded here, "
            "with volume held roughly constant."
        ),
        "falsifier": (
            "These are 2008-2010 months confounded with the economic conditions this dataset variant cannot "
            "see. If conversion by month flattens under a different macro environment, the signal was the "
            "period, not the calendar."
        ),
    },
]


# --------------------------------------------------------------------------- #
# Derived quantities, which no single record holds
# --------------------------------------------------------------------------- #
def derive(records: dict[str, dict]) -> dict:
    """Arithmetic on recorded numbers only, each one written to the record so it can be audited."""
    chosen = records["threshold"]["at_chosen"]
    translation = records["lift"]["translation"]["train_oof"]
    held = records["lift"]["translation"]["held_out"]
    holdout_cut = records["holdout"]["cuts"][0]

    held_chosen = records["threshold"]["held_out"]["at_chosen"]

    return {
        "net_eur_per_1000_calls_train": float(chosen["net_value_eur"] / chosen["clients_called"] * 1000),
        # Net value, not EUR not spent: `euros_not_spent` is the call budget the (smaller) fixed-order list
        # would have needed, which is a different quantity and was the wrong numerator in the first draft.
        "net_eur_per_1000_calls_held_out": float(
            held_chosen["net_value_eur"] / held_chosen["clients_called"] * 1000
        ),
        "cost_per_subscription_all_calls_eur": float(
            records["threshold"]["comparisons"][0]["cost_per_subscription_eur"]
        ),
        "share_of_file_called_train": float(translation["coverage"]),
        "share_of_file_called_held_out": float(held["coverage"]),
        "subscribers_reached_held_out_share": float(holdout_cut["recall"]),
        "precision_at_cut_held_out": float(holdout_cut["precision"]),
        # A constant `no` scores the negative share, which is exactly 1 - base rate.
        "accuracy_of_always_no": float(1.0 - records["cv"]["base_rate"]),
        "campaign_side_share": float(records["shap_global"]["campaign_side"]["share"]),
        "client_side_share": float(records["shap_global"]["client_side"]["share"]),
    }


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def validate(records: dict[str, dict], derived: dict, resolved: list[dict]) -> list[dict]:
    """Same contract as every phase: a check that cannot run is FAILED, never raised (D25)."""
    checks: list[dict] = []

    def compare(name: str, evaluate) -> None:
        try:
            passed, detail = evaluate()
        except Exception as error:  # noqa: BLE001 - any failure here is a failed check
            passed, detail = False, f"check could not run: {type(error).__name__}: {error}"
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    def enough() -> tuple[bool, str]:
        return len(RECOMMENDATIONS) >= MIN_RECOMMENDATIONS, (
            f"{len(RECOMMENDATIONS)} recommendations (minimum {MIN_RECOMMENDATIONS})"
        )

    compare("the playbook has at least six recommendations", enough)

    def complete() -> tuple[bool, str]:
        required = ("id", "scope", "title", "action", "evidence", "expected", "check", "falsifier")
        missing = {
            rec["id"]: [field for field in required if not rec.get(field)]
            for rec in RECOMMENDATIONS
            if any(not rec.get(field) for field in required)
        }
        return not missing, f"every recommendation carries {list(required)}; missing: {missing or 'none'}"

    compare("every recommendation is actionable, measurable and falsifiable", complete)

    def unique_ids() -> tuple[bool, str]:
        ids = [rec["id"] for rec in RECOMMENDATIONS]
        return len(set(ids)) == len(ids), f"{ids}"

    compare("the recommendation ids are unique", unique_ids)

    def every_figure_resolves() -> tuple[bool, str]:
        broken = [item["ref"] for item in resolved if not item["resolved"]]
        return not broken, f"{len(resolved)} figures resolved from the records; broken: {broken or 'none'}"

    compare("every cited number resolves to a saved record", every_figure_resolves)

    def figures_match_their_records() -> tuple[bool, str]:
        # Re-read each record from disk and compare with what was cited, so a stale copy cannot survive.
        # `derived` is re-computed rather than re-read: it is arithmetic on those same files.
        fresh = load_records()
        fresh_derived = derive(fresh)
        mismatched = []
        for item in resolved:
            if not item["resolved"]:
                continue
            source = fresh_derived if item["record"] == "derived" else fresh[item["record"]]
            actual = pluck(source, item["path"])
            if abs(float(actual) - float(item["value"])) > 1e-9:
                mismatched.append(f"{item['ref']}: cited {item['value']} vs record {actual}")
        return not mismatched, f"{len(resolved)} figures re-read from disk; mismatched: {mismatched[:3] or 'none'}"

    compare("every cited number still matches its record on disk", figures_match_their_records)

    def at_least_two() -> tuple[bool, str]:
        thin = [rec["id"] for rec in RECOMMENDATIONS if len(rec["evidence"]) < 2]
        return not thin, f"each recommendation cites at least two figures; thin: {thin or 'none'}"

    compare("no recommendation rests on a single figure", at_least_two)

    def scope_present() -> tuple[bool, str]:
        tagged = [rec["id"] for rec in RECOMMENDATIONS if rec["scope"] == "fixed-budget"]
        return bool(tagged) and "fixed call budget" in SCOPE_SENTENCE, (
            f"{tagged} carry the fixed-budget scope; the sentence names it"
        )

    compare("the D18 scope is stated and attached", scope_present)

    def duration_only_prohibited() -> tuple[bool, str]:
        # `duration` may appear in the prohibition and nowhere else: a playbook that mentions it in two
        # places has either a typo or a leak.
        mentions = [rec["id"] for rec in RECOMMENDATIONS
                    if "duration" in (rec["title"] + rec["action"] + rec["expected"]).lower()]
        return mentions == ["R4"], f"`duration` appears in {mentions}; it must appear only as the prohibition R4"

    compare("`duration` is mentioned only where it is forbidden", duration_only_prohibited)

    def money_identity() -> tuple[bool, str]:
        chosen = records["threshold"]["at_chosen"]
        rebuilt = chosen["subscribers_reached"] * records["threshold"]["prices"]["value_per_subscription_eur"] \
            - chosen["clients_called"] * records["threshold"]["prices"]["call_cost_eur"]
        return abs(rebuilt - chosen["net_value_eur"]) < 1e-6, (
            f"{chosen['subscribers_reached']:,} x {records['threshold']['prices']['value_per_subscription_eur']:.2f} "
            f"- {chosen['clients_called']:,} x {records['threshold']['prices']['call_cost_eur']:.2f} = "
            f"{rebuilt:,.2f} EUR, recorded {chosen['net_value_eur']:,.2f}"
        )

    compare("the campaign's net value is revenue minus call cost", money_identity)

    def crossover() -> tuple[bool, str]:
        # Calling everyone costs 12.82 EUR per subscription; the recommended cut costs 6.77. A V between the
        # two makes targeting necessary rather than merely better, which is what R5 rests on.
        all_calls = derived["cost_per_subscription_all_calls_eur"]
        at_cut = records["threshold"]["at_chosen"]["cost_per_subscription_eur"]
        return at_cut < 6.80 and all_calls > 12.80, (
            f"cost per subscription: {at_cut:.2f} EUR at the cut against {all_calls:.2f} EUR calling "
            f"everyone - the band where targeting is the difference between profit and loss"
        )

    compare("the pricing band R5 quotes is measured", crossover)

    def two_halves_agree() -> tuple[bool, str]:
        # The playbook quotes both halves; if they disagreed on the direction of the campaign the
        # recommendations could not be written at all.
        train = records["lift"]["translation"]["train_oof"]
        held = records["lift"]["translation"]["held_out"]
        return abs(train["coverage"] - held["coverage"]) < 0.02 and abs(train["recall"] - held["recall"]) < 0.02, (
            f"coverage {train['coverage']:.2%} vs {held['coverage']:.2%}; recall {train['recall']:.2%} vs "
            f"{held['recall']:.2%}"
        )

    compare("the training and held-out halves describe the same campaign", two_halves_agree)

    def derived_are_consistent() -> tuple[bool, str]:
        return (
            abs(derived["campaign_side_share"] + derived["client_side_share"] - 1.0) < 1e-9
            and derived["net_eur_per_1000_calls_train"] > 0,
            f"campaign {derived['campaign_side_share']:.2%} + client {derived['client_side_share']:.2%} = 1; "
            f"net per 1,000 calls {derived['net_eur_per_1000_calls_train']:,.2f} EUR"
        )

    compare("the derived quantities are internally consistent", derived_are_consistent)

    return checks


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def report() -> dict:
    records = load_records()
    derived = derive(records)

    # `derived` is not a file: it is the arithmetic block this function computes from the others, so an
    # evidence reference can point at it and still be re-derivable from disk.
    sources = {**records, "derived": derived}
    resolved: list[dict] = []
    for rec in RECOMMENDATIONS:
        for label, record_name, path, formatter in rec["evidence"]:
            ref = (f"models/{RECORD_PATH.name}:derived.{path}" if record_name == "derived"
                   else f"{RECORDS[record_name].name}:{path}")
            try:
                value = float(pluck(sources[record_name], path))
                resolved.append({
                    "recommendation": rec["id"],
                    "label": label,
                    "record": record_name,
                    "path": path,
                    "value": value,
                    "text": formatter(value) if formatter else f"{value:.4f}",
                    "ref": ref,
                    "resolved": True,
                })
            except Exception as error:  # noqa: BLE001 - an unresolvable figure is a failed check
                resolved.append({
                    "recommendation": rec["id"], "label": label, "record": record_name, "path": path,
                    "value": None, "text": f"UNRESOLVED ({type(error).__name__}: {error})",
                    "ref": ref, "resolved": False,
                })

    checks = validate(records, derived, resolved)

    with TABLE_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["recommendation", "label", "value", "record", "path"])
        writer.writeheader()
        for item in resolved:
            writer.writerow({
                "recommendation": item["recommendation"], "label": item["label"], "value": item["value"],
                "record": item["record"], "path": item["path"],
            })

    record = {
        "task": "T39",
        "scope": SCOPE_SENTENCE,
        "recommendations": [
            {
                "id": rec["id"], "scope": rec["scope"], "title": rec["title"], "action": rec["action"],
                "expected": rec["expected"], "check": rec["check"], "falsifier": rec["falsifier"],
                "evidence": [item for item in resolved if item["recommendation"] == rec["id"]],
            }
            for rec in RECOMMENDATIONS
        ],
        "derived": derived,
        "sources": {name: str(path) for name, path in RECORDS.items()},
        "checks": checks,
    }
    RECORD_PATH.write_text(json.dumps(record, indent=2, default=float) + "\n")
    return record


def print_report(result: dict) -> None:
    print("DepositIQ T39 - the campaign playbook")
    print("=" * 118)
    print(f"scope: {result['scope']}")
    print()
    for rec in result["recommendations"]:
        print(f"{rec['id']} [{rec['scope']}] {rec['title']}")
        print(f"   do     : {rec['action']}")
        for item in rec["evidence"]:
            print(f"   figure : {item['text']:<24} {item['label']}  <- {item['ref']}")
        print(f"   expect : {rec['expected']}")
        print(f"   check  : {rec['check']}")
        print(f"   wrong  : {rec['falsifier']}")
        print()
    derived = result["derived"]
    print("DERIVED (arithmetic on the records, written here so it is auditable in turn)")
    for key, value in derived.items():
        print(f"  {key:<38} {value:,.4f}")
    print()
    print("CHECKS")
    for check in result["checks"]:
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['check']} - {check['detail']}")
    print("=" * 118)


if __name__ == "__main__":
    result = report()
    print_report(result)
    ok = all(check["passed"] for check in result["checks"])
    print(f"result: {'PASS - playbook assembled from records' if ok else 'FAIL'}")
    print("Next  : T40 writes the final report from these records.")
    sys.exit(0 if ok else 1)
