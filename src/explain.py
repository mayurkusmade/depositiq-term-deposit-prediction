"""
DepositIQ - global SHAP on the model that ships (T37).

Why this file exists
--------------------
Everything before this step answers "how good is the ranking". This one answers the question a branch
manager actually asks, which the ranking cannot: **what is the model looking at, and in which direction?**
The instrument is SHAP - per-client, per-feature contributions from cooperative game theory that add up
exactly to the model's own score - and because they add up, every statement below is arithmetic rather
than interpretation.

Which model is explained, and why that is the whole point
-------------------------------------------------------
`models/final_model.joblib` (sha256 **7a40752f75a264a0...**, 180,520 bytes), loaded from disk. Not a
fresh fit, not the 200-tree forest of T25, not the untuned configuration. A SHAP plot of a model that is
not the shipped one is a plot of a different model, and the check `artifact_digest_matches_T31` is there
to make that mistake loud. The pipeline carries its own preprocessing, so the caller hands it the twelve
engineered columns and the explanation is taken on the **40-column matrix the estimator actually saw**.

Training half only
------------------
The explanation is computed on `features.load_engineered("train")` - 36,168 rows. The held-out half has
been spent twice already (T32's metrics, T35's artifact table) and D53/D54 make it report-only; an
importance table read off it would be a third look taken to describe rather than to test. It is also the
half the model was fitted on, which is the correct denominator for a *description* of that model's
behaviour: T37 does not claim the attributions generalise, it claims they are what the artifact computes.

Units, and the one sentence that keeps this honest
--------------------------------------------------
SHAP values here are in **log-odds** of P(subscribe): a contribution of +0.30 means that feature moved
that client's log-odds up by 0.30 relative to the model's **reference value, -2.04695** (= 11.4361% in
probability). That reference is the bias term xgboost itself reports in `pred_contribs` - checked equal
to 0.0 - and it is deliberately *not* described as the average prediction, because it is not one: the
mean margin over the 36,168 rows is **-2.49678**, and the mean probability is **11.6990%**. The three
numbers differ for two different reasons (the bias is not the mean output; the mean of a convex function
is not the function of the mean), and the average feature contribution is therefore **-0.4499** rather
than zero. They are not causal effects either - a feature can earn credit because it is correlated with
the thing that really drives the call. Both caveats are stated again where the numbers are, not only here.

The aggregation rule, which is the technical crux
-------------------------------------------------
Twelve features become forty columns, because each categorical is one-hot expanded. Per-column
attribution therefore answers "what did the level `month=may` do", not "what did `month` do", and the
brief asks for the second. The rule is to **sum the signed contributions within a feature's block**,
which:

* keeps additivity exact - the twelve feature-level values still sum to `margin - base`, so the table can
  be checked against the model rather than merely computed from it (`feature_level_contributions_...`);
* never double counts. The naive alternative - summing the absolute values of the one-hot columns -
  inflates a feature by its internal offsets, because a client is `month=may` *and* not `month=jun`, and
  both columns carry a nonzero contribution. Measured: on `month` the naive sum is **0.5680** against an
  aggregated **0.3583**, a 58.5% overstatement. The direction of that bias is the opposite of the usual
  "one-hot understates importance" warning, which is why the comparison is printed instead of assumed.

The importance of a feature is then `mean over rows of |block sum|`, and its share is that divided by the
total across features.

What this step can and cannot say about T25
-------------------------------------------
T25 ranked the same features with a **permutation** test on a **200-tree forest**, measuring ROC-AUC
lost when a feature's columns are shuffled. T37 ranks them by **SHAP on the shipped XGBoost**, measuring
log-odds contributed. Different instruments, different units, different questions - so the two rankings
agreeing would be a coincidence and disagreeing is not an error. The comparison is reported as a
dumbbell chart with the gaps named, never as an average of two ranks, and the largest gap is explained
by what the two measures actually do.

How to run
----------
    python -m src.explain     # ~35 s: one TreeSHAP pass over 36,168 rows, one figure

Writes `reports/figures/09_shap_summary.png`, `reports/shap_importance.csv` and
`models/shap_global.json`. `docs/08_explainability.md` quotes only numbers from those three files.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost

from src import config, features, select, train

ARTIFACT_PATH: Path = config.MODELS_DIR / "final_model.joblib"
FIGURE_PATH: Path = config.FIGURES_DIR / "09_shap_summary.png"
TABLE_PATH: Path = config.REPORTS_DIR / "shap_importance.csv"
RECORD_PATH: Path = config.MODELS_DIR / "shap_global.json"

# T31's artifact. The digest is pinned so a re-fitted or re-tuned model cannot be explained silently:
# if this fails, the explanation is of some other model and the rest of the output is not comparable.
EXPECTED_SHA256_PREFIX: str = "7a40752f"
EXPECTED_BYTES: int = 180_520

# Rows drawn in the beeswarm. The *table* uses all 36,168 rows; 6,000 dots is a plotting decision, because
# 36k overlapping points is slower to draw and no more legible. Fixed seed, stated in the figure subtitle.
PLOT_ROWS: int = 6_000
PLOT_SEED: int = config.RANDOM_STATE

# T25's permutation ranking for the twelve shipped features, from `docs/05_leakage_audit.md` §8.1: the
# ROC-AUC lost when a feature's columns are shuffled, on the whole held-out fold of a 200-tree forest
# whose out-of-fold ROC-AUC is 0.7936 +/- 0.0063. A **pin with provenance, not a second source of truth** -
# T37 must compare against T25 as T25 recorded it, not against a fresh forest run that would be a
# different instrument with its own sampling noise. `validate()` checks the pin covers exactly the shipped
# twelve, and a scratch verifier outside the module re-ran T25's instrument and reproduced these values.
T25_PERMUTATION: tuple[tuple[str, float], ...] = (
    ("month", 0.04971),
    ("contact", 0.03270),
    ("poutcome", 0.01905),
    ("day", 0.01498),
    ("age", 0.01000),
    ("housing", 0.00732),
    ("recency_band", 0.00614),
    ("marital", 0.00453),
    ("balance_arsinh", 0.00390),
    ("campaign_bucket", 0.00271),
    ("previously_contacted", 0.00241),
    ("previous_capped", 0.00233),
)

# Gaps at least this wide are called disagreements rather than ordering noise. T25 used 5 places over 17
# features; over 12 the same idea is 3, and it is declared here rather than chosen after seeing the table.
GAP_NOTABLE: int = 3

# A reading, not a measurement: which features describe **the campaign** (how and when the bank calls) as
# opposed to **the client**. The shares below are measured; the grouping is a judgement, declared here so
# the sentence "most of this model is about delivery" can be checked against the list it rests on.
CAMPAIGN_SIDE: tuple[str, ...] = ("contact", "month", "day", "campaign_bucket")


# --------------------------------------------------------------------------- #
# The artifact, the matrix, and the explanation
# --------------------------------------------------------------------------- #
def load_artifact(path: Path = ARTIFACT_PATH) -> tuple[object, str, int]:
    """Load the shipped pipeline, and return it with its digest and size."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return joblib.load(path), digest, path.stat().st_size


def design_matrix(pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    """The 40-column matrix the estimator saw, as a named frame (`job=admin.`, not `job_admin.`)."""
    transformer = pipeline.named_steps[train.STEP_PREPROCESS]
    return transformer.transform(frame)


def explain(pipeline, X: pd.DataFrame) -> dict:
    """
    TreeSHAP on the artifact's own estimator, plus the margin it has to reconstruct.

    `output_margin=True` is the quantity SHAP explains: tree contributions are additive in log-odds, not
    in probability, so the check is `sum(shap) + base == margin` and the probability check goes through
    the logistic link afterwards. XGBoost's sklearn wrapper in this version rejects a `DMatrix` argument
    to `predict`, so the array is passed and the DMatrix is built only for `pred_contribs`.
    """
    model = pipeline.named_steps[train.STEP_MODEL]
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(X), dtype=np.float64)
    base = float(np.ravel(explainer.expected_value)[0])

    array = X.to_numpy()
    margin = np.asarray(model.predict(array, output_margin=True), dtype=np.float64)
    contributions = model.get_booster().predict(
        xgboost.DMatrix(array, feature_names=list(X.columns)), pred_contribs=True
    )
    return {
        "values": values,
        "base": base,
        "margin": margin,
        "xgboost_contribs": np.asarray(contributions, dtype=np.float64),
        "names": list(X.columns),
    }


# --------------------------------------------------------------------------- #
# Aggregation: forty columns back to twelve features
# --------------------------------------------------------------------------- #
def feature_table(values: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """
    Per **source feature**: mean |summed contribution|, its share, and the naive per-column alternative.

    `mean_abs` is the headline and is what the figures use. `mean_abs_columns` is
    `mean(sum over the feature's columns of |contribution|)` - the number a per-column table would
    aggregate to - and it is carried alongside purely so the inflation can be measured instead of argued.
    """
    blocks = select.block_map(columns)
    rows = []
    for feature, indices in blocks.items():
        block = values[:, indices]
        aggregated = block.sum(axis=1)
        rows.append(
            {
                "feature": feature,
                "columns": len(indices),
                "mean_abs": float(np.abs(aggregated).mean()),
                "mean_abs_columns": float(np.abs(block).sum(axis=1).mean()),
                "mean_signed": float(aggregated.mean()),
            }
        )
    table = pd.DataFrame(rows)
    table["share"] = table["mean_abs"] / table["mean_abs"].sum()
    table["inflation"] = table["mean_abs_columns"] / table["mean_abs"] - 1.0
    table["shap_rank"] = table["mean_abs"].rank(ascending=False, method="min").astype(int)
    order = table.sort_values(["shap_rank", "feature"]).reset_index(drop=True)
    return order


def column_table(values: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """The un-aggregated view, so the figure can show what a per-level reader would have seen."""
    table = pd.DataFrame(
        {
            "column": columns,
            "feature": [select.source_of(column) for column in columns],
            "mean_abs": np.abs(values).mean(axis=0),
            "mean_signed": values.mean(axis=0),
        }
    )
    table["share"] = table["mean_abs"] / table["mean_abs"].sum()
    return table.sort_values("mean_abs", ascending=False).reset_index(drop=True)


def compare_with_t25(table: pd.DataFrame) -> pd.DataFrame:
    """
    Both ranks side by side, restricted to the twelve shipped features.

    T25's pin is ranked **within the twelve**, because T37 explains twelve columns and a comparison that
    included the five dropped features would grade a ranking on features the model cannot use.
    """
    pin = pd.DataFrame(T25_PERMUTATION, columns=["feature", "t25_perm_mean"])
    pin["t25_rank"] = pin["t25_perm_mean"].rank(ascending=False, method="min").astype(int)

    merged = table.merge(pin, on="feature", how="inner")
    if len(merged) != len(table) or len(merged) != len(pin):
        raise ValueError("the SHAP table and T25's pin do not cover the same features")

    merged["rank_gap"] = merged["shap_rank"] - merged["t25_rank"]
    merged["notable"] = merged["rank_gap"].abs() >= GAP_NOTABLE
    merged["spearman"] = float(merged["mean_abs"].corr(merged["t25_perm_mean"], method="spearman"))
    return merged.sort_values("shap_rank").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Direction: what the contribution means for a client
# --------------------------------------------------------------------------- #
def directions(
    values: np.ndarray, table: pd.DataFrame, X: pd.DataFrame, frame: pd.DataFrame,
    columns: list[str], top: int = 6,
) -> list[dict]:
    """
    For the strongest features, which levels (or which end of the range) push a client up or down.

    This is what turns a bar chart into a sentence. For a categorical the level is read off the one-hot
    block (the column holding 1); for a numeric the direction is a Spearman correlation between the value
    and the aggregated contribution, plus the mean contribution in the bottom and top quintile, which is
    robust to a badly behaved middle.
    """
    blocks = select.block_map(columns)
    out: list[dict] = []
    for feature in table["feature"].head(top):
        indices = blocks[feature]
        aggregated = values[:, indices].sum(axis=1)
        entry: dict = {
            "feature": feature,
            "mean_abs": float(table.loc[table["feature"] == feature, "mean_abs"].iloc[0]),
            "share": float(table.loc[table["feature"] == feature, "share"].iloc[0]),
        }
        if feature in features.CATEGORICAL_FEATURES:
            block = X.iloc[:, indices].to_numpy()
            chosen = block.argmax(axis=1)
            labels = [columns[index].split("=", 1)[1] for index in indices]
            levels = []
            for position, label in enumerate(labels):
                mask = chosen == position
                if not mask.any():
                    continue
                # Client counts here index the one-hot block, which is row-aligned with `frame`; the
                # subscription rate is filled in by `level_rates()`, which owns the target.
                levels.append(
                    {
                        "level": label,
                        "clients": int(mask.sum()),
                        "share_of_clients": float(mask.mean()),
                        "mean_contribution": float(aggregated[mask].mean()),
                        "subscribe_rate": float("nan"),
                    }
                )
            entry["kind"] = "categorical"
            entry["levels"] = sorted(levels, key=lambda row: -row["mean_contribution"])
        else:
            raw = frame[feature].to_numpy(dtype=float)
            finite = np.isfinite(raw)
            quintiles = np.quantile(raw[finite], [0.2, 0.8])
            low = finite & (raw <= quintiles[0])
            high = finite & (raw >= quintiles[1])
            entry["kind"] = "numeric"
            entry["spearman_with_value"] = float(
                pd.Series(raw[finite]).corr(pd.Series(aggregated[finite]), method="spearman")
            )
            entry["low_quintile"] = {
                "value_at_most": float(quintiles[0]),
                "clients": int(low.sum()),
                "mean_contribution": float(aggregated[low].mean()),
            }
            entry["high_quintile"] = {
                "value_at_least": float(quintiles[1]),
                "clients": int(high.sum()),
                "mean_contribution": float(aggregated[high].mean()),
            }
        out.append(entry)
    return out


def level_rates(directions_list: list[dict], frame: pd.DataFrame) -> list[dict]:
    """Fill in each level's measured subscription rate - the observable the sentence can be checked on."""
    target = frame[config.TARGET].astype(str).str.lower().eq(config.POSITIVE_LABEL)
    for entry in directions_list:
        if entry["kind"] != "categorical":
            continue
        for level in entry["levels"]:
            mask = frame[entry["feature"]].astype(str).str.lower().eq(level["level"].lower())
            level["subscribe_rate"] = float(target[mask].mean()) if mask.any() else float("nan")
            level["clients"] = int(mask.sum())
            level["share_of_clients"] = float(mask.mean())
    return directions_list


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def validate(
    digest: str,
    size: int,
    pipeline,
    frame: pd.DataFrame,
    X: pd.DataFrame,
    explained: dict,
    table: pd.DataFrame,
    columns_table: pd.DataFrame,
    comparison: pd.DataFrame,
) -> list[dict]:
    """Same contract as every phase: a check that cannot run is FAILED, never raised (D25)."""
    checks: list[dict] = []

    def compare(name: str, evaluate) -> None:
        try:
            passed, detail = evaluate()
        except Exception as error:  # noqa: BLE001 - any failure here is a failed check
            passed, detail = False, f"check could not run: {type(error).__name__}: {error}"
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    def artifact_digest_matches_T31() -> tuple[bool, str]:
        return (
            digest.startswith(EXPECTED_SHA256_PREFIX) and size == EXPECTED_BYTES,
            f"sha256 {digest[:16]}..., {size:,} bytes (expected prefix {EXPECTED_SHA256_PREFIX}, "
            f"{EXPECTED_BYTES:,} bytes)",
        )

    compare("the explained model is the artifact T31 persisted", artifact_digest_matches_T31)

    def structure() -> tuple[bool, str]:
        steps = list(pipeline.named_steps)
        model = pipeline.named_steps[train.STEP_MODEL]
        rounds = int(model.get_booster().num_boosted_rounds())
        return (
            steps == [train.STEP_PREPROCESS, train.STEP_MODEL]
            and isinstance(model, xgboost.XGBClassifier)
            and rounds == 300,
            f"steps {steps}; {type(model).__name__}, {rounds} rounds, matrix {X.shape[1]} columns",
        )

    compare("the artifact is the two-step XGBoost pipeline with 300 rounds", structure)

    def leak_is_absent() -> tuple[bool, str]:
        bad = [column for column in X.columns if column.split("=", 1)[0] in ("duration", config.TARGET)]
        return not bad, f"no `duration` or target column in the {len(X.columns)} explained columns; stray: {bad or 'none'}"

    compare("`duration` and the target are not explained, because they are not in the matrix", leak_is_absent)

    def twelve_features_cover_forty_columns() -> tuple[bool, str]:
        blocks = select.block_map(list(X.columns))
        sizes = {feature: len(indices) for feature, indices in blocks.items()}
        return (
            len(blocks) == 12 and sum(sizes.values()) == 40 and len(set(X.columns)) == 40,
            f"{len(blocks)} source features over {sum(sizes.values())} unique columns: {sizes}",
        )

    compare("the 12 features partition the 40 columns exactly", twelve_features_cover_forty_columns)

    def the_twelve_are_t25_kept() -> tuple[bool, str]:
        shipped = sorted(select.shipped_features())
        explained = sorted(select.block_map(list(X.columns)))
        return (
            shipped == explained,
            f"explained set equals select.shipped_features(); missing "
            f"{sorted(set(shipped) - set(explained)) or 'none'}, extra "
            f"{sorted(set(explained) - set(shipped)) or 'none'}",
        )

    compare("the explained features are exactly the twelve T25 kept", the_twelve_are_t25_kept)

    def shap_reconstructs_the_margin() -> tuple[bool, str]:
        error = float(np.abs(explained["values"].sum(axis=1) + explained["base"] - explained["margin"]).max())
        return error < 1e-4, f"max |sum(SHAP) + base - margin| = {error:.3e} over {len(X):,} rows (tolerance 1e-4)"

    compare("the SHAP values add up to the model's own margin", shap_reconstructs_the_margin)

    def margin_reproduces_the_probability() -> tuple[bool, str]:
        # Through `frame`, the raw twelve engineered columns - the interface the artifact actually serves.
        # Handing the pipeline `X` (the already-transformed matrix) raises "columns are missing", which is
        # how the first version of this check failed: it measured the model against a shape it never sees.
        probability = 1.0 / (1.0 + np.exp(-explained["margin"]))
        error = float(np.abs(probability - pipeline.predict_proba(frame)[:, 1]).max())
        return error < 1e-6, (
            f"max |sigmoid(margin) - pipeline.predict_proba(12 raw columns)| = {error:.3e} (tolerance 1e-6)"
        )

    compare("the explained margin is the probability the model serves", margin_reproduces_the_probability)

    def shap_matches_xgboost_contribs() -> tuple[bool, str]:
        contributions = explained["xgboost_contribs"]
        error = float(np.abs(contributions[:, :-1] - explained["values"]).max())
        bias = float(np.abs(contributions[:, -1] - explained["base"]).max())
        return (
            error < 1e-6 and bias < 1e-6,
            f"max |pred_contribs - shap| = {error:.3e} over the 40 columns, |bias - base| = {bias:.3e}; "
            f"shap delegates to this path for XGBoost, so this confirms the wiring (model, matrix, column "
            f"order) rather than a second algorithm",
        )

    compare("SHAP agrees with xgboost's own pred_contribs", shap_matches_xgboost_contribs)

    def feature_levels_stay_additive() -> tuple[bool, str]:
        blocks = select.block_map(list(X.columns))
        rebuilt = np.zeros(len(X))
        for indices in blocks.values():
            rebuilt += explained["values"][:, indices].sum(axis=1)
        error = float(np.abs(rebuilt + explained["base"] - explained["margin"]).max())
        return error < 1e-4, f"max |sum of 12 feature-level values + base - margin| = {error:.3e}"

    compare("aggregating to feature level does not break additivity", feature_levels_stay_additive)

    def aggregation_only_shrinks() -> tuple[bool, str]:
        worst = table.loc[table["inflation"].idxmax()]
        negative = table.loc[table["inflation"] < -1e-12, "feature"].tolist()
        return (
            not negative,
            f"mean |sum| <= mean sum |column| for all 12 (triangle inequality); largest inflation "
            f"{worst['feature']} at {worst['inflation']:+.1%}; features where it did not hold: {negative or 'none'}",
        )

    compare("the aggregation never inflates a feature's importance", aggregation_only_shrinks)

    def reference_is_the_bias_not_the_mean_output() -> tuple[bool, str]:
        # The average contribution is `mean margin - base`, which is the sign of `sum(SHAP)` per row
        # averaged; writing it as `base - mean` would report the right magnitude with the wrong sign.
        contribution = float(explained["margin"].mean() - explained["base"])
        return abs(contribution) > 0.1, (
            f"reference {explained['base']:+.5f} vs mean margin {explained['margin'].mean():+.5f}: the "
            f"average feature contribution is {contribution:+.5f}, not zero - so the reference is not the "
            f"mean prediction and the docs must not call it one"
        )

    compare("the SHAP reference is the model's bias, not its average output", reference_is_the_bias_not_the_mean_output)

    def shares_sum_to_one() -> tuple[bool, str]:
        return abs(table["share"].sum() - 1.0) < 1e-12, f"shares sum to {table['share'].sum():.15f}"

    compare("importance shares sum to one", shares_sum_to_one)

    def ranks_are_a_permutation() -> tuple[bool, str]:
        ranks = sorted(table["shap_rank"].tolist())
        perms = sorted(comparison["t25_rank"].tolist())
        return (
            ranks == list(range(1, len(table) + 1)) and perms == list(range(1, len(table) + 1)),
            f"SHAP ranks {ranks}; T25 ranks within the twelve {perms}",
        )

    compare("both rankings are permutations of 1..12", ranks_are_a_permutation)

    def pin_covers_the_twelve() -> tuple[bool, str]:
        pinned = [feature for feature, _ in T25_PERMUTATION]
        return (
            sorted(pinned) == sorted(select.shipped_features()) and len(set(pinned)) == 12,
            f"{len(pinned)} pinned features, all shipped, no duplicates; missing "
            f"{sorted(set(select.shipped_features()) - set(pinned)) or 'none'}",
        )

    compare("T25's pinned ranking covers exactly the twelve shipped features", pin_covers_the_twelve)

    def columns_table_covers_the_matrix() -> tuple[bool, str]:
        return (
            len(columns_table) == X.shape[1]
            and set(columns_table["column"]) == set(X.columns)
            and abs(columns_table["share"].sum() - 1.0) < 1e-12,
            f"{len(columns_table)} rows for {X.shape[1]} columns, shares sum to "
            f"{columns_table['share'].sum():.6f}",
        )

    compare("the per-column table covers the matrix once", columns_table_covers_the_matrix)

    def no_held_out_read() -> tuple[bool, str]:
        # Assembled at run time so this file does not match its own search text.
        needle = "load_engineered(" + '"te' + 'st"'
        source = Path(__file__).read_text()
        hits = source.count(needle)
        return hits == 0, f"{hits} references to the held-out half in src/explain.py (needle assembled, not literal)"

    compare("the held-out half is never loaded", no_held_out_read)

    def figure_exists() -> tuple[bool, str]:
        return FIGURE_PATH.exists() and FIGURE_PATH.stat().st_size > 60_000, (
            f"{FIGURE_PATH.name}: {FIGURE_PATH.stat().st_size / 1024:.0f} KB" if FIGURE_PATH.exists()
            else "figure missing"
        )

    compare("the figure was written", figure_exists)

    return checks


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def draw(
    explained: dict,
    table: pd.DataFrame,
    comparison: pd.DataFrame,
    X: pd.DataFrame,
    path: Path = FIGURE_PATH,
) -> Path:
    """Beeswarm on the 40 columns, importance bar on the 12 features, and the rank comparison."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(19.0, 10.5))
    grid = figure.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 0.78],
                              wspace=0.30, hspace=0.42)
    left = figure.add_subplot(grid[:, 0])
    right_top = figure.add_subplot(grid[0, 1])
    right_bottom = figure.add_subplot(grid[1, 1])

    # --- A: the beeswarm, as the model sees it -------------------------------------------
    rng = np.random.default_rng(PLOT_SEED)
    chosen = np.sort(rng.choice(len(X), size=min(PLOT_ROWS, len(X)), replace=False))
    plt.sca(left)
    shap.summary_plot(
        explained["values"][chosen],
        X.iloc[chosen],
        max_display=40,
        show=False,
        plot_size=None,
        alpha=0.55,
        # A colour bar is meaningless for a 0/1 one-hot column - the scale would read "value 0 to 1".
        # The subtitle states the mapping instead, and the panel keeps the width the bar was taking.
        color_bar=False,
    )
    left.set_title(
        f"every column the model sees (40 one-hot levels)\n{len(chosen):,} of {len(X):,} training clients; "
        f"red = the client has this level, blue = they do not",
        fontsize=11.0,
    )
    left.set_xlabel("SHAP value (log-odds of subscribing, shifted from the reference "
                    f"{explained['base']:+.4f})", fontsize=9.5)
    left.tick_params(axis="y", labelsize=7.8)
    left.grid(axis="x", alpha=0.20)

    # --- B: the same values, summed back into twelve features ---------------------------
    ordered = table.iloc[::-1].reset_index(drop=True)
    positions = np.arange(len(ordered))
    right_top.barh(positions, ordered["mean_abs"], height=0.66, color="#4C78A8", alpha=0.92,
                   label="mean |SHAP| of the feature's summed contribution")
    right_top.scatter(ordered["mean_abs_columns"], positions, marker="|", s=260, color="#E45756",
                      linewidths=2.0, zorder=5, label="the same, summed per column (overstates)")
    for position, (_, row) in zip(positions, ordered.iterrows()):
        # Past the *longer* of the two marks, so the share never sits under the naive-sum tick.
        right_top.text(max(row["mean_abs"], row["mean_abs_columns"]) + 0.010, position,
                       f"{row['share']:.1%}", va="center", fontsize=8.2, color="#333333")
    right_top.set_yticks(positions)
    right_top.set_yticklabels([f"{row['feature']} ({row['columns']})" for _, row in ordered.iterrows()],
                              fontsize=9)
    right_top.set_xlim(0, max(ordered["mean_abs_columns"].max(), ordered["mean_abs"].max()) * 1.22)
    right_top.set_xlabel("mean |SHAP| in log-odds (feature = the signed sum of its one-hot columns)",
                        fontsize=9.5)
    right_top.set_title("the same contributions, aggregated to 12 features\n"
                        "(n) = how many one-hot columns the feature owns; % = share of total importance",
                        fontsize=11)
    right_top.legend(fontsize=7.8, loc="center right", framealpha=0.92)
    right_top.grid(axis="x", alpha=0.22)

    # --- C: where the two rankings disagree --------------------------------------------
    ranked_by_shap = comparison.sort_values("shap_rank").reset_index(drop=True)
    y = np.arange(len(ranked_by_shap))
    for position, (_, row) in zip(y, ranked_by_shap.iterrows()):
        right_bottom.plot([row["shap_rank"], row["t25_rank"]], [position, position],
                          color="#E45756" if row["notable"] else "#BBBBBB",
                          linewidth=2.4 if row["notable"] else 1.3, zorder=2)
    right_bottom.scatter(ranked_by_shap["shap_rank"], y, s=58, color="#4C78A8", zorder=4,
                         label="SHAP on the shipped model")
    right_bottom.scatter(ranked_by_shap["t25_rank"], y, s=58, color="#F58518", zorder=4,
                         marker="D", label="T25 permutation, 200-tree forest")
    right_bottom.set_yticks(y)
    right_bottom.set_yticklabels(ranked_by_shap["feature"], fontsize=8.6)
    right_bottom.invert_yaxis()
    right_bottom.invert_xaxis()
    right_bottom.set_xticks(range(1, len(ranked_by_shap) + 1))
    right_bottom.set_xlabel("rank within the twelve (1 = most important)", fontsize=9.5)
    right_bottom.set_title(f"SHAP versus T25's permutation ranking\nSpearman "
                           f"{comparison['spearman'].iloc[0]:+.3f}; red joins a gap of "
                           f"{GAP_NOTABLE}+ places", fontsize=11)
    right_bottom.legend(fontsize=8, loc="lower left")
    right_bottom.grid(axis="x", alpha=0.22)

    figure.suptitle("DepositIQ - what the shipped model looks at (T37, global SHAP on "
                    f"models/final_model.joblib, 36,168 training clients)", fontsize=13.5)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def report(variant: str = "full") -> dict:
    """Everything the module prints, and the source of every number in `docs/08_explainability.md`."""
    started = time.perf_counter()
    frame = features.load_engineered("train", variant)
    pipeline, digest, size = load_artifact()
    X = design_matrix(pipeline, frame)
    explained = explain(pipeline, X)

    table = feature_table(explained["values"], list(X.columns))
    columns_table = column_table(explained["values"], list(X.columns))
    comparison = compare_with_t25(table)
    direction_rows = level_rates(directions(explained["values"], table, X, frame, list(X.columns)), frame)

    draw(explained, table, comparison, X)
    checks = validate(digest, size, pipeline, frame, X, explained, table, columns_table, comparison)

    # The **merged** table, so the CSV a reader opens carries both ranks and the gap; a CSV with only
    # SHAP's column would make §8.5's comparison unreproducible from the file it cites.
    comparison.drop(columns=["spearman"]).to_csv(TABLE_PATH, index=False)

    record = {
        "task": "T37",
        "variant": variant,
        "artifact": ARTIFACT_PATH.name,
        "artifact_sha256": digest,
        "artifact_bytes": size,
        "rows": len(X),
        "matrix_columns": int(X.shape[1]),
        "features": int(len(table)),
        "base_value": explained["base"],
        "base_probability": float(1.0 / (1.0 + np.exp(-explained["base"]))),
        "base_score": float(json.loads(
            pipeline.named_steps[train.STEP_MODEL].get_booster().save_config()
        )["learner"]["learner_model_param"]["base_score"].strip("[]")),
        "mean_margin": float(explained["margin"].mean()),
        "mean_contribution": float(explained["margin"].mean() - explained["base"]),
        "mean_probability": float(np.mean(1.0 / (1.0 + np.exp(-explained["margin"])))),
        "median_probability": float(np.median(1.0 / (1.0 + np.exp(-explained["margin"])))),
        "base_rate": float(frame[config.TARGET].astype(str).str.lower().eq(config.POSITIVE_LABEL).mean()),
        "plot_rows": int(min(PLOT_ROWS, len(X))),
        "plot_seed": PLOT_SEED,
        "features_table": table.to_dict(orient="records"),
        "columns_table": columns_table.to_dict(orient="records"),
        "comparison": comparison.drop(columns=["spearman"]).to_dict(orient="records"),
        "spearman_vs_t25": float(comparison["spearman"].iloc[0]),
        "campaign_side": {
            "features": list(CAMPAIGN_SIDE),
            "share": float(table.loc[table["feature"].isin(CAMPAIGN_SIDE), "share"].sum()),
        },
        "client_side": {
            "features": [feature for feature in table["feature"] if feature not in CAMPAIGN_SIDE],
            "share": float(table.loc[~table["feature"].isin(CAMPAIGN_SIDE), "share"].sum()),
        },
        "notable_gaps": comparison.loc[comparison["notable"], ["feature", "shap_rank", "t25_rank", "rank_gap"]]
        .to_dict(orient="records"),
        "directions": direction_rows,
        "checks": checks,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    RECORD_PATH.write_text(json.dumps(record, indent=2, default=float) + "\n")
    return record


def print_report(result: dict) -> None:
    table = pd.DataFrame(result["features_table"]).sort_values("shap_rank")
    comparison = pd.DataFrame(result["comparison"]).sort_values("shap_rank")
    print("DepositIQ T37 - global SHAP on the shipped model")
    print("=" * 118)
    print(f"artifact      : {result['artifact']} sha256 {result['artifact_sha256'][:16]}..., "
          f"{result['artifact_bytes']:,} bytes")
    print(f"rows explained: {result['rows']:,} (training half only) x {result['matrix_columns']} columns "
          f"over {result['features']} features")
    print(f"reference     : base value {result['base_value']:+.5f} log-odds (= P {result['base_probability']:.4%}), "
          f"the booster's bias (base_score {result['base_score']:.5f}); NOT the mean prediction")
    print(f"                mean margin {result['mean_margin']:+.5f} (-> P {1 / (1 + np.exp(-result['mean_margin'])):.4%}), "
          f"mean P {result['mean_probability']:.4%}, median P {result['median_probability']:.4%}, "
          f"base rate {result['base_rate']:.4%}")
    print(f"                average feature contribution {result['mean_contribution']:+.5f} log-odds, not zero")
    print(f"beeswarm      : {result['plot_rows']:,} rows drawn of {result['rows']:,} (seed "
          f"{result['plot_seed']}); the table below uses every row")
    print()
    print(f"{'#':>2}  {'feature':<21}{'cols':>5}{'mean|SHAP|':>12}{'share':>8}{'per-col':>10}"
          f"{'inflation':>11}{'T25 rank':>10}{'gap':>5}")
    print("-" * 118)
    for _, row in comparison.iterrows():
        print(f"{row['shap_rank']:>2}  {row['feature']:<21}{row['columns']:>5}{row['mean_abs']:>12.5f}"
              f"{row['share']:>8.1%}{row['mean_abs_columns']:>10.5f}{row['inflation']:>+11.1%}"
              f"{row['t25_rank']:>10}{row['rank_gap']:>+5}")
    print(f"  Spearman between the two rankings: {result['spearman_vs_t25']:+.3f}")
    print(f"  campaign-side features {result['campaign_side']['features']} carry "
          f"{result['campaign_side']['share']:.1%} of total importance; the client's own attributes "
          f"carry {result['client_side']['share']:.1%}")
    if result["notable_gaps"]:
        print(f"  gaps of {GAP_NOTABLE}+ places:")
        for gap in result["notable_gaps"]:
            print(f"    {gap['feature']:<21} SHAP {gap['shap_rank']:>2} vs T25 {gap['t25_rank']:>2} "
                  f"({gap['rank_gap']:+d})")
    print()
    print("DIRECTION, strongest six (contribution is in log-odds; rates are measured on the same rows)")
    for entry in result["directions"]:
        print(f"  {entry['feature']} ({entry['kind']}, {entry['share']:.1%} of importance, "
              f"mean |SHAP| {entry['mean_abs']:.5f})")
        if entry["kind"] == "categorical":
            for level in entry["levels"][:3]:
                print(f"    {level['level']:<14} {level['clients']:>6,} clients "
                      f"({level['share_of_clients']:>5.1%})  mean contribution "
                      f"{level['mean_contribution']:+.4f}  subscribe rate {level['subscribe_rate']:.2%}")
            if len(entry["levels"]) > 3:
                worst = entry["levels"][-1]
                print(f"    {worst['level']:<14} {worst['clients']:>6,} clients "
                      f"({worst['share_of_clients']:>5.1%})  mean contribution "
                      f"{worst['mean_contribution']:+.4f}  subscribe rate {worst['subscribe_rate']:.2%}")
        else:
            low, high = entry["low_quintile"], entry["high_quintile"]
            print(f"    value <= {low['value_at_most']:.3f}: {low['clients']:>6,} clients, mean contribution "
                  f"{low['mean_contribution']:+.4f}")
            print(f"    value >= {high['value_at_least']:.3f}: {high['clients']:>6,} clients, mean contribution "
                  f"{high['mean_contribution']:+.4f}")
            print(f"    Spearman(value, contribution) = {entry['spearman_with_value']:+.3f}")
    print()
    print("CHECKS")
    for check in result["checks"]:
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['check']} - {check['detail']}")
    print("=" * 118)


if __name__ == "__main__":
    result = report()
    print_report(result)
    ok = all(check["passed"] for check in result["checks"])
    print(f"result: {'PASS - global SHAP explained the shipped artifact' if ok else 'FAIL'}")
    print("Next  : T38 explains three individual clients (a true positive, a false positive, a false negative).")
    sys.exit(0 if ok else 1)
