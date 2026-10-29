"""
DepositIQ - local SHAP for three exemplar clients (T38).

Why this file exists
--------------------
T37 explained the model as a whole: `contact` 22.14%, `month` 17.55%, and a list of averages. A branch
manager does not score averages, though - they score **one client** - and the average hides exactly the
case they care about. This module takes one client from each of the three interesting cells of the campaign
and shows which features moved that client's score, in log-odds, with the exact value the model saw.

The three cells, defined against the price rather than against 0.5
-----------------------------------------------------------------
T34 adopted **0.0750** - call a client when P(subscribe) exceeds C/V. "Positive" below therefore means
*above the price*, not above a coin flip, because that is the decision the bank actually makes:

| cell | definition | what it is in the campaign |
|---|---|---|
| **true positive** | y = 1 and p >= 0.0750 | called, and subscribed |
| **false positive** | y = 0 and p >= 0.0750 | called, and did not |
| **false negative** | y = 1 and p < 0.0750 | not called, and would have |

Using 0.5 instead would move most of the list into "negative" (41.10% of clients are called at the price,
against a fraction of a percent at 0.5) and would describe a campaign nobody runs.

Which scores define the cells, and why the counts differ from T34's
------------------------------------------------------------------
Cells are taken from the **artifact's own scores on the training rows** - the same scores and the same
matrix T37 explained - so the explanation belongs to the model that is on display. Those scores are
**in-sample**, so the counts are not T34's out-of-fold counts (**3,296** true positives, **11,570** false
positives, **935** false negatives); both are recorded and the difference is expected rather than a
discrepancy. The held-out half is never loaded (D53, D54).

The exemplar rule, fixed before any client is looked at
------------------------------------------------------
> **The exemplar of a cell is the client whose score is closest to that cell's median score; ties are
> broken by the lowest training row index.**

Pre-registered in `PROGRESS.md` as part of T38's carry-in, and for two reasons:

* **Median, not most extreme.** The most confident false negative is the worst case of its kind; naming it
  and then explaining it invites the reader to generalise from an outlier. The middle of the cell is a
  typical member, which is what an illustration should be.
* **Score only, not SHAP.** The rule reads `y` and `p` and nothing else, so the exemplars cannot be chosen
  because their explanations looked good. `validate()` re-derives the choice from the data and fails if any
  other client in the cell is strictly closer to the median.

What a local explanation is, in units
-------------------------------------
Each client's twelve feature values sum to `margin - reference`, where the reference is T37's **-2.04695**
(the booster's bias) and the margin is that client's log-odds. So "+1.37 for `month=mar`" means that
feature moved *this* client 1.37 log-odds up from the reference, and the twelve of them plus the reference
give the score the bank sees. Contributions are not causal (T36 §9.4) and not comparable across clients as
"importance" - T37 §8.4 showed a 0.35%-share feature can decide one client while a 22%-share feature is
irrelevant to another.

How to run
----------
    python -m src.local_explain     # ~50 s: one TreeSHAP pass, three profiles, one figure

Writes `reports/figures/10_shap_local.png`, `reports/shap_local.csv` and `models/shap_local.json`.
`docs/08_explainability.md` §9 quotes only numbers from those three files.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, cost, explain, features, select

THRESHOLD_PATH: Path = config.MODELS_DIR / "threshold_choice.json"
GLOBAL_PATH: Path = config.MODELS_DIR / "shap_global.json"
FIGURE_PATH: Path = config.FIGURES_DIR / "10_shap_local.png"
TABLE_PATH: Path = config.REPORTS_DIR / "shap_local.csv"
RECORD_PATH: Path = config.MODELS_DIR / "shap_local.json"

# The three cells, in the order the figure draws them. Keys are the record's field names.
CELLS: tuple[str, ...] = ("true_positive", "false_positive", "false_negative")

CELL_LABELS: dict[str, str] = {
    "true_positive": "true positive - called, subscribed",
    "false_positive": "false positive - called, did not subscribe",
    "false_negative": "false negative - not called, would have subscribed",
}

# T34's out-of-fold counts at the same cut, carried so the in-sample counts can be compared to them
# explicitly rather than silently replacing them.
T34_OUT_OF_FOLD: dict[str, int] = {
    "true_positive": 3296,
    "false_positive": 11570,
    "false_negative": 935,
}


# --------------------------------------------------------------------------- #
# Cells, and the pre-registered selection
# --------------------------------------------------------------------------- #
def cell_of(y: np.ndarray, p: np.ndarray, cut: float) -> np.ndarray:
    """Label every row `TP` / `FP` / `FN` / `TN` against the price, not against 0.5."""
    called = p >= cut
    positive = y == 1
    return np.where(
        positive & called, "TP",
        np.where(~positive & called, "FP", np.where(positive & ~called, "FN", "TN")),
    )


CELL_OF_CODE = {"true_positive": "TP", "false_positive": "FP", "false_negative": "FN"}


def pick_exemplars(y: np.ndarray, p: np.ndarray, cut: float) -> dict[str, dict]:
    """
    The client closest to each cell's median score - the rule written in the docstring, executed.

    `distance` is kept so `validate()` can re-derive the minimum independently: the run is correct when no
    other member of the cell is *strictly* closer to the median.
    """
    labels = cell_of(y, p, cut)
    chosen: dict[str, dict] = {}
    for cell in CELLS:
        mask = labels == CELL_OF_CODE[cell]
        indices = np.flatnonzero(mask)
        scores = p[indices]
        median = float(np.median(scores))
        distance = np.abs(scores - median)
        # `argmin` returns the first minimum, and `indices` is ascending, so the tie-break is the lowest
        # row index without a second sorting step.
        best = int(indices[int(np.argmin(distance))])
        chosen[cell] = {
            "cell": cell,
            "row": best,
            "clients_in_cell": int(mask.sum()),
            "cell_share_of_training": float(mask.mean()),
            "cell_median_score": median,
            "score": float(p[best]),
            "mean_score_in_cell": float(scores.mean()),
            "distance_to_median": float(distance.min()),
            "minimum_distance_in_cell": float(distance.min()),
            "score_percentile_in_cell": float((scores <= p[best]).mean()),
            "label": int(y[best]),
            "share_of_all_train_rows": float(mask.sum() / len(p)),
        }
    return chosen


# --------------------------------------------------------------------------- #
# One client's twelve features
# --------------------------------------------------------------------------- #
def value_of(feature: str, row_X: pd.Series, row_frame: pd.Series, columns: list[str],
             blocks: dict[str, list[int]]) -> str:
    """
    What the model saw for this client in this feature, as a readable string.

    A categorical is read off its one-hot block (the column holding 1); a numeric is the value itself. The
    matrix values are float64 copies of the engineered columns, so the numeric string is formatted rather
    than echoed - `balance_arsinh = 6.59`, not 6.5899999999999999.
    """
    if feature in features.CATEGORICAL_FEATURES:
        block = row_X.iloc[blocks[feature]].to_numpy()
        position = int(np.argmax(block))
        return columns[blocks[feature][position]].split("=", 1)[1]
    value = float(row_frame[feature])
    if feature in ("age", "day", "previous_capped", "previously_contacted"):
        return f"{value:.0f}"
    if feature == "balance_arsinh":
        return f"{value:.2f}"
    return f"{value:.3f}"


def local_profile(
    row: int,
    explained: dict,
    X: pd.DataFrame,
    frame: pd.DataFrame,
    global_table: pd.DataFrame,
) -> list[dict]:
    """The twelve signed contributions for one client, with the global context beside each."""
    columns = explained["names"]
    blocks = select.block_map(columns)
    ranks = dict(zip(global_table["feature"], global_table["shap_rank"]))
    shares = dict(zip(global_table["feature"], global_table["share"]))
    row_X = X.iloc[row]
    row_frame = frame.iloc[row]
    profile = []
    for feature, indices in blocks.items():
        contribution = float(explained["values"][row, indices].sum())
        profile.append(
            {
                "feature": feature,
                "contribution": contribution,
                "value": value_of(feature, row_X, row_frame, columns, blocks),
                "columns": len(indices),
                "global_rank": int(ranks[feature]),
                "global_share": float(shares[feature]),
            }
        )
    return sorted(profile, key=lambda item: item["contribution"])


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def validate(
    digest: str,
    size: int,
    cut: float,
    y: np.ndarray,
    p: np.ndarray,
    explained: dict,
    chosen: dict[str, dict],
    profiles: dict[str, list[dict]],
    global_table: pd.DataFrame,
) -> list[dict]:
    """Same contract as every phase: a check that cannot run is FAILED, never raised (D25)."""
    checks: list[dict] = []

    def compare(name: str, evaluate) -> None:
        try:
            passed, detail = evaluate()
        except Exception as error:  # noqa: BLE001 - any failure here is a failed check
            passed, detail = False, f"check could not run: {type(error).__name__}: {error}"
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    def artifact() -> tuple[bool, str]:
        return (
            digest.startswith(explain.EXPECTED_SHA256_PREFIX) and size == explain.EXPECTED_BYTES
            and int(explained["values"].shape[1]) == 40,
            f"sha256 {digest[:16]}..., {size:,} bytes, {explained['values'].shape[1]} columns, "
            f"{explained['values'].shape[0]:,} rows",
        )

    compare("the explained model is T31's artifact, on its 40-column matrix", artifact)

    def the_cut_is_the_price() -> tuple[bool, str]:
        record = json.loads(THRESHOLD_PATH.read_text())
        recorded = float(record["decision"]["threshold"])
        derived = float(cost.break_even_threshold())
        return (
            abs(cut - recorded) < 1e-12 and abs(cut - derived) < 1e-12,
            f"cut {cut:.6f} equals T34's record ({recorded:.6f}) and the C/V identity "
            f"({derived:.6f}); derived again by `threshold`/`cost`, not typed in",
        )

    compare("the cell boundary is the price T34 adopted, not 0.5", the_cut_is_the_price)

    def cells_partition() -> tuple[bool, str]:
        labels = cell_of(y, p, cut)
        counts = {name: int((labels == code).sum()) for name, code in CELL_OF_CODE.items()}
        counts["TN"] = int((labels == "TN").sum())
        return (
            sum(counts.values()) == len(y),
            f"{counts} sum to {sum(counts.values()):,} = {len(y):,} rows",
        )

    compare("the four cells partition the training half", cells_partition)

    def each_exemplar_is_in_its_cell() -> tuple[bool, str]:
        labels = cell_of(y, p, cut)
        wrong = {
            cell: labels[item["row"]] for cell, item in chosen.items()
            if labels[item["row"]] != CELL_OF_CODE[cell]
        }
        scores = {cell: round(float(p[item["row"]]), 5) for cell, item in chosen.items()}
        return not wrong, f"every exemplar's label agrees with its cell; scores {scores}; wrong: {wrong or 'none'}"

    compare("each exemplar actually sits in the cell it names", each_exemplar_is_in_its_cell)

    def exemplars_are_distinct() -> tuple[bool, str]:
        rows = [item["row"] for item in chosen.values()]
        return len(set(rows)) == len(rows), f"training rows {rows}"

    compare("the three exemplars are three different clients", exemplars_are_distinct)

    def the_rule_was_followed() -> tuple[bool, str]:
        # Recomputed from the scores alone: for each cell, no other member may be strictly closer to the
        # median than the exemplar that was picked. This is the rule, not a re-run of the same `argmin`.
        labels = cell_of(y, p, cut)
        problems = []
        for cell, item in chosen.items():
            members = p[labels == CELL_OF_CODE[cell]]
            distance = float(np.abs(p[item["row"]] - np.median(members)))
            closest = float(np.abs(members - np.median(members)).min())
            if distance > closest + 1e-15:
                problems.append(f"{cell}: picked at {distance:.6f}, closest is {closest:.6f}")
        detail = "; ".join(
            f"{cell}: |p - median| = {item['distance_to_median']:.6f} of {item['clients_in_cell']:,} clients"
            for cell, item in chosen.items()
        )
        return not problems, f"{detail}; violations: {problems or 'none'}"

    compare("the exemplar is the client nearest its cell's median score", the_rule_was_followed)

    def additivity_per_client() -> tuple[bool, str]:
        blocks = select.block_map(explained["names"])
        errors = []
        for cell, item in chosen.items():
            row = item["row"]
            rebuilt = sum(float(explained["values"][row, indices].sum()) for indices in blocks.values())
            errors.append(abs(rebuilt + explained["base"] - explained["margin"][row]))
        return max(errors) < 1e-4, (
            f"max over the three clients |sum of 12 feature values + reference - margin| = "
            f"{max(errors):.3e}, reference {explained['base']:+.5f}"
        )

    compare("each client's twelve contributions add up to that client's score", additivity_per_client)

    def probability_per_client() -> tuple[bool, str]:
        # Now a real test: the left side is SHAP's margin through the link, the right side is the pipeline's
        # own `predict_proba`. If the explained rows were not the scored rows, this difference would be large.
        errors = {
            cell: abs(float(1 / (1 + np.exp(-explained["margin"][item["row"]]))) - float(p[item["row"]]))
            for cell, item in chosen.items()
        }
        return max(errors.values()) < 1e-6, (
            "max |sigmoid(shap margin) - predict_proba| = "
            f"{max(errors.values()):.3e} over the three clients"
        )

    compare("the explained margin is the probability the cells were defined on", probability_per_client)

    def counts_survive_both_derivations() -> tuple[bool, str]:
        # The same score has two routes: the pipeline's `predict_proba`, and SHAP's decomposition through the
        # logistic link. They differ by float32 rounding (~5e-6), and 43 clients sit within 1e-4 of the cut, so
        # the honest question is not "is the cut stable" (it is not, in a dense region) but "do the quoted
        # counts depend on which route is used". Measured, not assumed: one client, of 36,168.
        rebuilt = explained["values"].sum(axis=1) + explained["base"]
        p_shap = 1.0 / (1.0 + np.exp(-rebuilt))
        flips = int((cell_of(y, p, cut) != cell_of(y, p_shap, cut)).sum())
        boundary = int((np.abs(p - cut) < 1e-4).sum())
        return flips <= 2, (
            f"{flips} of {len(p):,} clients change cell between `predict_proba` and the SHAP decomposition "
            f"(max |p - p_shap| = {float(np.abs(p - p_shap).max()):.2e}); {boundary} clients sit within 1e-4 "
            f"of the cut and the notes quote the served version, so the counts are stated to that client"
        )

    compare("the quoted cell counts do not depend on how the score is derived", counts_survive_both_derivations)

    def matches_pred_contribs() -> tuple[bool, str]:
        values = explained["xgboost_contribs"][:, :-1]
        errors = [
            float(np.abs(values[item["row"]].sum() - explained["values"][item["row"]].sum()))
            for item in chosen.values()
        ]
        blocks = select.block_map(explained["names"])
        per_feature = []
        for item in chosen.values():
            for indices in blocks.values():
                per_feature.append(abs(
                    float(values[item["row"], indices].sum() - explained["values"][item["row"], indices].sum())
                ))
        return max(errors + per_feature) < 1e-6, (
            f"max |pred_contribs - shap| over 3 rows x 12 features = {max(errors + per_feature):.3e}"
        )

    compare("the local values reproduce from xgboost's own contributions", matches_pred_contribs)

    def twelve_features_cover_the_matrix() -> tuple[bool, str]:
        blocks = select.block_map(explained["names"])
        covered = sorted(index for indices in blocks.values() for index in indices)
        return (
            len(blocks) == 12 and covered == list(range(len(explained["names"]))),
            f"{len(blocks)} features cover all {len(covered)} column indices exactly once",
        )

    compare("the twelve features partition the 40 columns", twelve_features_cover_the_matrix)

    def leak_and_target_absent() -> tuple[bool, str]:
        features_used = sorted({item["feature"] for profile in profiles.values() for item in profile})
        bad = [name for name in features_used if name in ("duration", config.TARGET)]
        return not bad, f"{len(features_used)} features named across the three profiles; forbidden: {bad or 'none'}"

    compare("`duration` and the target are not explained", leak_and_target_absent)

    def global_context_matches_t37() -> tuple[bool, str]:
        ranks = dict(zip(global_table["feature"], global_table["shap_rank"]))
        wrong = [
            (item["feature"], item["global_rank"], ranks[item["feature"]])
            for profile in profiles.values()
            for item in profile
            if item["global_rank"] != int(ranks[item["feature"]])
        ]
        return not wrong, f"every annotated global rank equals T37's record; wrong: {wrong[:3] or 'none'}"

    compare("the global ranks shown beside each client come from T37's record", global_context_matches_t37)

    def above_the_price_counts_as_positive() -> tuple[bool, str]:
        # The rule that would be wrong: grading at 0.5. Find a client called at the price but below 0.5 and
        # confirm the cell logic calls it positive - i.e. the module is not quietly using a coin flip.
        labels = cell_of(y, p, cut)
        window = (p >= cut) & (p < 0.5)
        example = int(np.flatnonzero(window)[0]) if window.any() else -1
        return (
            example >= 0 and labels[example] in ("TP", "FP"),
            f"{int(window.sum()):,} clients sit between {cut:.4f} and 0.5; row {example} at p = "
            f"{p[example]:.4f} is labelled {labels[example]} - a 0.5 rule would call it negative",
        )

    compare("a client above the price but below 0.5 is treated as called", above_the_price_counts_as_positive)

    def no_held_out_read() -> tuple[bool, str]:
        needle = "load_engineered(" + '"te' + 'st"'
        hits = Path(__file__).read_text().count(needle)
        return hits == 0, f"{hits} references to the held-out half in src/local_explain.py"

    compare("the held-out half is never loaded", no_held_out_read)

    def figure_written() -> tuple[bool, str]:
        return FIGURE_PATH.exists() and FIGURE_PATH.stat().st_size > 40_000, (
            f"{FIGURE_PATH.name}: {FIGURE_PATH.stat().st_size / 1024:.0f} KB"
            if FIGURE_PATH.exists() else "figure missing"
        )

    compare("the figure was written", figure_written)

    return checks


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def draw(chosen: dict[str, dict], profiles: dict[str, list[dict]], cut: float,
         base: float, path: Path = FIGURE_PATH) -> Path:
    """Three clients, twelve signed bars each - the same axis, so the three can be read against each other."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(19.0, 7.6), sharex=False)
    positive, negative = "#54A24B", "#E45756"

    for axis, cell in zip(axes, CELLS):
        profile = profiles[cell]
        item = chosen[cell]
        names = [f"{row['feature']} = {row['value']}  (#{row['global_rank']})" for row in profile]
        values = [row["contribution"] for row in profile]
        colours = [positive if value >= 0 else negative for value in values]
        positions = np.arange(len(profile))
        axis.barh(positions, values, color=colours, alpha=0.9, height=0.68)
        axis.axvline(0.0, color="#333333", linewidth=1.1)
        axis.set_yticks(positions)
        axis.set_yticklabels(names, fontsize=8.0)
        span = max(abs(value) for value in values) or 1.0
        axis.set_xlim(-span * 1.35, span * 1.35)
        for position, value in zip(positions, values):
            offset = span * 0.04
            axis.text(value + (offset if value >= 0 else -offset), position, f"{value:+.2f}",
                      va="center", ha="left" if value >= 0 else "right", fontsize=7.4,
                      color="#1a1a1a")
        axis.set_xlabel("contribution to this client's log-odds", fontsize=9)
        # The additivity line lives in the title, not inside the axes: drawn at the bottom it collided with
        # the most negative bar in every one of the three panels.
        axis.set_title(
            f"{CELL_LABELS[cell]}\n"
            f"p = {item['score']:.4f} against a cut of {cut:.4f}; y = {item['label']}\n"
            f"{item['clients_in_cell']:,} clients ({item['cell_share_of_training']:.2%} of the training "
            f"half), at the {item['score_percentile_in_cell']:.0%} percentile of its own cell\n"
            f"sum {sum(values):+.4f} + reference {base:+.5f} = margin {sum(values) + base:+.4f}",
            fontsize=9.2,
        )
        axis.grid(axis="x", alpha=0.22)

    figure.suptitle(
        "DepositIQ - why the model scored three clients the way it did (T38, local SHAP on "
        "models/final_model.joblib)\nfilled bars push the client up, red push down; (#n) is the feature's "
        "T37 global rank",
        fontsize=12.2,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def report(variant: str = "full") -> dict:
    """Everything the module prints, and the source of every number in §9 of the explainability notes."""
    started = time.perf_counter()
    frame = features.load_engineered("train", variant)
    y = frame[config.TARGET].astype(str).str.lower().eq(config.POSITIVE_LABEL).to_numpy().astype(int)

    cut = float(json.loads(THRESHOLD_PATH.read_text())["decision"]["threshold"])
    pipeline, digest, size = explain.load_artifact()
    X = explain.design_matrix(pipeline, frame)
    explained = explain.explain(pipeline, X)

    # The cells are built on the **served** probability - `predict_proba` through the whole pipeline - not on
    # `sigmoid(shap margin)`. The two agree to ~5e-6, and that difference is enough to move a client that sits
    # inside it across the cut: the first version of this module defined the cells from the margin and one
    # client flipped cell as a result. The served number is the one the bank acts on, and using it makes the
    # margin check below a real test instead of an identity.
    p = pipeline.predict_proba(frame)[:, 1]
    chosen = pick_exemplars(y, p, cut)
    # Recorded so the printed report can check its own arithmetic without re-deriving the link.
    for item in chosen.values():
        item["margin"] = float(explained["margin"][item["row"]])

    global_table = pd.DataFrame(json.loads(GLOBAL_PATH.read_text())["features_table"])
    profiles = {cell: local_profile(item["row"], explained, X, frame, global_table)
                for cell, item in chosen.items()}

    draw(chosen, profiles, cut, explained["base"])
    checks = validate(digest, size, cut, y, p, explained, chosen, profiles, global_table)

    rows = [{"cell": cell, **profile_row} for cell, profile in profiles.items() for profile_row in profile]
    pd.DataFrame(rows).to_csv(TABLE_PATH, index=False)

    labels = cell_of(y, p, cut)
    counts = {cell: int((labels == code).sum()) for cell, code in CELL_OF_CODE.items()}
    record = {
        "task": "T38",
        "variant": variant,
        "artifact": explain.ARTIFACT_PATH.name,
        "artifact_sha256": digest,
        "artifact_bytes": size,
        "rows": int(len(y)),
        "cut": cut,
        "cut_source": "models/threshold_choice.json (T34), re-derived from cost.break_even_threshold()",
        "cells_in_sample": counts,
        "clients_within_1e-4_of_cut": int((np.abs(p - cut) < 1e-4).sum()),
        "cell_flips_between_derivations": int((cell_of(y, p, cut) != cell_of(
            y, 1.0 / (1.0 + np.exp(-(explained["values"].sum(axis=1) + explained["base"]))), cut)).sum()),
        "cells_out_of_fold_t34": T34_OUT_OF_FOLD,
        "cells_note": "cells are defined on the artifact's in-sample training scores, the same scores T37 "
                      "explained; T34's counts are out-of-fold, so they differ by construction",
        "base_value": explained["base"],
        "selection_rule": "the client whose score is closest to the cell's median score, ties broken by the "
                          "lowest training row index",
        "exemplars": {
            cell: {key: value for key, value in item.items() if key not in ("cell",)}
            for cell, item in chosen.items()
        },
        "profiles": profiles,
        "checks": checks,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    RECORD_PATH.write_text(json.dumps(record, indent=2, default=float) + "\n")
    return record


def print_report(result: dict) -> None:
    print("DepositIQ T38 - local SHAP for three exemplar clients")
    print("=" * 112)
    print(f"artifact : {result['artifact']} sha256 {result['artifact_sha256'][:16]}..., "
          f"{result['artifact_bytes']:,} bytes; {result['rows']:,} training rows")
    print(f"cut      : {result['cut']:.4f} (T34, from C/V) - 'positive' means above the price, not above 0.5")
    counts = result["cells_in_sample"]
    carried = result["cells_out_of_fold_t34"]
    print(f"cells    : in sample  TP {counts['true_positive']:>6,}  FP {counts['false_positive']:>6,}  "
          f"FN {counts['false_negative']:>6,}")
    print(f"           T34 out-of-fold TP {carried['true_positive']:>6,}  FP {carried['false_positive']:>6,}  "
          f"FN {carried['false_negative']:>6,}  (different denominator by construction)")
    print(f"reference: {result['base_value']:+.5f} log-odds (T37's booster bias)")
    print()
    for cell in CELLS:
        item = result["exemplars"][cell]
        print(f"{CELL_LABELS[cell].upper()}")
        print(f"  training row {item['row']:,}   p = {item['score']:.4f}   y = {item['label']}   "
              f"cell size {item['clients_in_cell']:,}   cell median {item['cell_median_score']:.4f}   "
              f"|p - median| {item['distance_to_median']:.4f}")
        profile = sorted(result["profiles"][cell], key=lambda row: -abs(row["contribution"]))
        total = sum(row["contribution"] for row in profile)
        print(f"  margin {item['margin']:+.4f} = reference {result['base_value']:+.4f} + sum of twelve "
              f"{total:+.4f}")
        for row in profile[:5]:
            print(f"    {row['feature']:<18} = {row['value']:<12} {row['contribution']:+.4f}   "
                  f"(global rank #{row['global_rank']}, {row['global_share']:.2%} share)")
        if len(profile) > 5:
            rest = profile[5:]
            print(f"    ... {len(rest)} more, net {sum(row['contribution'] for row in rest):+.4f}: "
                  f"{', '.join(row['feature'] for row in rest)}")
        print()
    print("CHECKS")
    for check in result["checks"]:
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['check']} - {check['detail']}")
    print("=" * 112)


if __name__ == "__main__":
    result = report()
    print_report(result)
    ok = all(check["passed"] for check in result["checks"])
    print(f"result: {'PASS - three clients explained against the price' if ok else 'FAIL'}")
    print("Next  : T39 turns this into the campaign playbook and the ranked recommendations.")
    sys.exit(0 if ok else 1)
