"""
DepositIQ - feature selection and importance (T25).

What this module does
---------------------
Ranks the **17 features** behind `src/preprocess.py`'s 65 columns with **two independent methods**,
compares the two rankings, and then decides what to keep with a **third measurement** rather than with a
threshold. It writes `reports/figures/06_feature_importance.png`, and every number in
`docs/05_leakage_audit.md` §8 is printed by this module.

Two methods, then one decision
------------------------------
**M1 - mutual information** (`mutual_info_classif`) on the **17 source features**, not the 65 columns: a
categorical feature enters as its integer codes with `discrete_features=True`, so `job` is one variable
with 12 levels rather than twelve binaries. Model-free, and it sees non-monotone dependence - which is why
it is here at all, since a linear correlation scores `duration` highest and `poutcome` invisible (D38).

MI is biased upward by cardinality: a 12-level column out-scores a 2-level one at random. Rather than
quoting that caveat, the module **measures** it. `N_PERMUTATIONS` label shuffles give every feature its own
null, and `z` says how far the real MI sits above *that feature's* noise floor.

**M2 - permutation importance** from a random forest: the ROC-AUC lost when a feature's columns are
shuffled, on held-out folds. Model-based, so it is subject to masking (two columns carrying the same
information split the credit) and blind to anything the forest failed to learn. A useful counterweight,
not a better method.

The forest is an **instrument, not a candidate**. D08 selects XGBoost for the shipped model and T26-T30
pick and tune it; this forest exists only to produce an importance vector, its settings are frozen in
`INSTRUMENT`, and it is never reported as a model result. Its out-of-fold ROC-AUC is printed anyway,
because an importance vector from an instrument that cannot rank is worthless.

**M3 - group ablation** is what actually decides. Screen candidates are features that *either* method
cannot distinguish from nothing; those candidates are then removed **as a group** and the same folds are
re-measured. This is D43's second stage, and it exists because the first stage turned out to be nearly
vacuous - see "What the first stage could not do" below.

Attribution back to the 17 features
-----------------------------------
Permutation importance is computed per one-hot column and **summed within each feature block**, per fold,
then averaged over folds. The sum is taken per fold rather than over averaged columns because one-hot
levels of one feature are not independent, and a variance-summed interval would pretend they are. The
block figure therefore **overstates** a block whose levels overlap in what they predict - stated rather
than corrected, since joint shuffling would no longer attribute anything to individual levels.

What the first stage could not do, measured
-------------------------------------------
The original rule (D43) was "drop only where both methods fail: `z < 1` **and** `perm_mean - perm_sd <= 0`".
Run on 36,168 rows that first condition never fires: **no feature has `z < 1`**, the lowest is `day` at
**7.4**, **16 of 17** clear `z = 20`, and the smallest MI in the whole table (`default`, 0.0004 nats) sits
at `z = 27.5`. The null is that tight because 10 label shuffles of 36,168 rows leave almost no room for
spurious association. So the MI half of the screen nominated **nothing at all**, and every nominee comes
from the model half. A rule whose first condition cannot fail is a one-condition rule wearing two
conditions' clothes, so it is not allowed to decide on its own: the screen only nominates, the ablation
decides, and the nominee list is reported with the evidence that put each feature on it.

The decision rule, in full (D43, with the second stage disclosed)
-----------------------------------------------------------------
1. **Screen.** A feature becomes a candidate if `mi <= null_q95` **or** `perm_mean - perm_sd <= 0`. The
   `or` is deliberate: a feature is nominated when *either* independent view cannot see it.
2. **Guard.** A candidate is not dropped if it is the model's **only** representation of a finding the
   project relies on. One such carve-out is declared in `GUARDED` with its reason, and the module prints
   it rather than burying it.
3. **Ablate.** The remaining candidates are removed as a group, the same 5 folds are re-run, and the
   paired per-fold AUC differences decide by the **one-standard-error form**: the group goes if
   `mean ΔAUC >= -sd`, i.e. if the loss cannot be distinguished from noise, and stays if it can. The
   first version compared the mean against exactly zero; the run then produced `-0.00014 ± 0.00195`, which
   would have made the verdict turn on a quantity **14× smaller than its own noise**. A threshold that
   sits inside the noise it is judging is a coin flip wearing a decimal point, so the standard is the same
   one stage 1 already uses: is the effect distinguishable from zero?

`duration`, `pdays` and `y` are not in the matrix and cannot be ranked (D05, D36). Nothing here says the
leak would be useful - only what is left once it is gone.

How to run
----------
From the repository root, inside the virtual environment:

    python -m src.select     # ~5 minutes: 2 x 5 folds, 3 permutation repeats each, then the figure
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src import config, features, preprocess
from src.split import encode_target

# ---------------------------------------------------------------------------
# Fixed settings
# ---------------------------------------------------------------------------
N_FOLDS: int = 5
N_PERMUTATIONS: int = 10  # label shuffles behind each feature's own MI null
N_REPEATS: int = 3  # permutation repeats per fold
SCORING: str = "roc_auc"  # D04/D15: a ranking metric, so a drop in AUC is a drop in ranking skill

# The instrument. Fixed so the importance vector is reproducible; not a model claim (see docstring).
INSTRUMENT: dict = {
    "n_estimators": 200,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "random_state": config.RANDOM_STATE,
    "n_jobs": 1,
}

# Stage 1 floors, in each method's own units.
MI_Z_FLOOR: float = 1.0
PERMUTATION_FLOOR: float = 0.0

# Stage 2 uses the **one-standard-error form**: the group goes if the held-out loss is within one sd of
# zero (`mean_delta >= -sd`), i.e. if the loss cannot be told from noise. Comparing a mean against exactly
# zero instead is not a rule - on the first run the mean was -0.00014 against an sd of 0.00195, so the
# verdict would have turned on a quantity 14x smaller than the noise it sits in.
ABLATION_TOLERANCE: str = "one_se"

# The carve-out, declared with its reason. A feature here can be nominated by the screen and is still kept.
GUARDED: dict[str, str] = {
    features.CAMPAIGN_BUCKET:
        "the model's only representation of the T19/D33 call-budget finding; one instrument failing to "
        "use it is not evidence that the finding is representable another way, and T37 discusses that "
        "finding while the shipped model is on display",
}

# Measured as strongly associated in the EDA (D38, T18-T21); the rule must never remove one of these.
ANCHORS: tuple[str, ...] = ("poutcome", "contact", "housing")

FIGURE_PATH: Path = config.FIGURES_DIR / "06_feature_importance.png"


def source_of(column: str) -> str:
    """Map a matrix column back to the 17-feature name it came from: `job=admin.` -> `job`."""
    return column.split("=", 1)[0]


def block_map(columns: list[str] | tuple[str, ...]) -> dict[str, list[int]]:
    """The indices of every matrix column belonging to each source feature."""
    blocks: dict[str, list[int]] = {}
    for index, column in enumerate(columns):
        blocks.setdefault(source_of(column), []).append(index)
    return blocks


def columns_for(kept: list[str] | tuple[str, ...]) -> list[str]:
    """The preprocessor's column list for a kept subset, in `features.FEATURES` order."""
    chosen = set(kept)
    return [column for column in features.FEATURES if column in chosen]


# ---------------------------------------------------------------------------
# M1 - mutual information, with each feature's own null
# ---------------------------------------------------------------------------
def _codes(frame: pd.DataFrame) -> tuple[np.ndarray, list[bool]]:
    """
    The 17 features as a numeric matrix plus the `discrete_features` mask.

    Categorical features enter as integer codes so MI sees one 12-level variable rather than twelve dummy
    columns; `previous_capped` and the boolean are discrete too. `age`, `day` and `balance_arsinh` are
    continuous, which is what they are.
    """
    matrix = []
    discrete = []
    for column in features.FEATURES:
        values = frame[column]
        if column in features.CATEGORICAL_FEATURES:
            codes = pd.Categorical(values).codes
            if int((codes < 0).sum()):
                raise ValueError(f"{column!r} has missing values; the T24 matrix has none")
            matrix.append(codes.astype(np.int64))
            discrete.append(True)
        else:
            matrix.append(values.to_numpy(dtype=np.float64))
            discrete.append(column not in ("age", "day", features.BALANCE_ARSINH))
    return np.vstack(matrix).T, discrete


def mutual_information(frame: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """MI per source feature, plus the null distribution each one is judged against."""
    X, discrete = _codes(frame)
    target = y.to_numpy()
    values = mutual_info_classif(X, target, discrete_features=discrete,
                                 random_state=config.RANDOM_STATE)

    rng = np.random.default_rng(config.RANDOM_STATE)
    null = np.empty((N_PERMUTATIONS, len(features.FEATURES)))
    for index in range(N_PERMUTATIONS):
        null[index] = mutual_info_classif(X, rng.permutation(target), discrete_features=discrete,
                                          random_state=config.RANDOM_STATE)

    rows = []
    for position, column in enumerate(features.FEATURES):
        null_values = null[:, position]
        spread = float(null_values.std(ddof=0))
        rows.append({
            "feature": column,
            "mi": float(values[position]),
            "null_mean": float(null_values.mean()),
            "null_q95": float(np.percentile(null_values, 95)),
            "null_sd": spread,
            # Excess over this feature's own noise floor, in units of that floor's spread. A high
            # cardinality column with no signal has a large `mi` and a `z` near 0; that is the point.
            "z": float((values[position] - null_values.mean()) / spread) if spread > 0 else 0.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# M2 / M3 - the forest, measured on held-out folds
# ---------------------------------------------------------------------------
def _folds(frame: pd.DataFrame, y: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    """One splitter, materialised once, so every experiment below replays the same membership."""
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    return list(splitter.split(frame, y))


def _fit_and_score(frame: pd.DataFrame, y: pd.Series, columns: list[str],
                   folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[list[dict], list[float], list[str]]:
    """Fit the pipeline inside each fold and return the fitted models, the fold AUCs and the columns."""
    models = []
    aucs = []
    names: list[str] = []
    for train_index, valid_index in folds:
        pipeline = Pipeline([
            ("prep", preprocess.build_preprocessor(columns)),
            ("model", RandomForestClassifier(**INSTRUMENT)),
        ])
        pipeline.fit(frame.iloc[train_index], y.iloc[train_index])
        prep = pipeline.named_steps["prep"]
        model = pipeline.named_steps["model"]
        X_valid = prep.transform(frame.iloc[valid_index])
        names = list(X_valid.columns)
        models.append((model, X_valid, y.iloc[valid_index]))
        aucs.append(float(roc_auc_score(y.iloc[valid_index], model.predict_proba(X_valid)[:, 1])))
    return models, aucs, names


def permutation_importance_by_feature(
    frame: pd.DataFrame, y: pd.Series, folds: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[pd.DataFrame, dict]:
    """The ROC-AUC each feature's columns are worth, shuffling them on held-out rows."""
    models, aucs, names = _fit_and_score(frame, y, list(features.FEATURES), folds)
    blocks = block_map(names)

    # The **whole** held-out fold, never a head slice. `StratifiedKFold` returns indices in ascending
    # original order and the training file is ordered by campaign date (D26), so `X_valid.iloc[:3000]` is
    # not a sample - it is the campaign's first weeks. Measured the hard way: that slice contained **no**
    # `poutcome=success` rows at all, so 20 columns were constant in it and their permutation importance
    # came back as exactly 0.00000 +/- 0.00000. A cheaper run is not worth an estimate of the wrong thing.
    per_fold: list[dict[str, float]] = []
    for model, X_valid, y_valid in models:
        result = permutation_importance(
            model, X_valid, y_valid, scoring=SCORING, n_repeats=N_REPEATS,
            random_state=config.RANDOM_STATE, n_jobs=1,
        )
        per_fold.append({
            feature: float(np.sum(result.importances_mean[indices]))
            for feature, indices in blocks.items()
        })

    table = pd.DataFrame(per_fold)  # rows = folds, columns = features
    summary = pd.DataFrame({
        "feature": list(table.columns),
        "perm_mean": table.mean(axis=0).to_numpy(),
        "perm_sd": table.std(axis=0, ddof=0).to_numpy(),
    })
    return summary, {
        "fold_auc": aucs,
        "oof_auc": float(np.mean(aucs)),
        "cd_auc": float(np.std(aucs, ddof=0)),
        "n_columns": len(names),
        "blocks": {feature: len(indices) for feature, indices in blocks.items()},
        "permutation_rows": len(models[0][1]),
        "n_names": names,
    }


def compare(mi_table: pd.DataFrame, perm_table: pd.DataFrame) -> pd.DataFrame:
    """
    Put the two rankings side by side: both ranks, the gap, and the correlation between them.

    Ranks are `1 = most important` in each method's own units. The gap is `mi_rank - perm_rank`, so a
    positive gap means the filter rates the feature higher than the forest does.
    """
    table = mi_table.merge(perm_table, on="feature", how="inner")
    if len(table) != len(mi_table) or len(table) != len(perm_table):
        raise ValueError("the two methods did not score the same features")

    table["mi_rank"] = table["mi"].rank(ascending=False, method="min").astype(int)
    table["perm_rank"] = table["perm_mean"].rank(ascending=False, method="min").astype(int)
    table["rank_gap"] = table["mi_rank"] - table["perm_rank"]
    table["spearman"] = float(table["mi"].corr(table["perm_mean"], method="spearman"))
    return table.sort_values(["perm_rank", "mi_rank"]).reset_index(drop=True)


def screen(table: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 1: nominate every feature that **either** method cannot distinguish from nothing.

    Three states are recorded per feature, because the interesting rows are the disagreements: both
    methods see it, only the filter does, or only the model does.
    """
    out = table.copy()
    mi_weak = out["mi"] <= out["null_q95"]
    perm_weak = (out["perm_mean"] - out["perm_sd"]) <= PERMUTATION_FLOOR
    out["mi_weak"] = mi_weak
    out["perm_weak"] = perm_weak
    out["candidate"] = mi_weak | perm_weak
    out["guarded"] = out["feature"].map(lambda name: name in GUARDED)

    def state(row) -> str:
        if row["mi_weak"] and row["perm_weak"]:
            return "both methods blind"
        if row["mi_weak"]:
            return "filter blind, model uses it"
        if row["perm_weak"]:
            return "model blind, filter sees it"
        return "both methods see it"

    out["evidence"] = out.apply(state, axis=1)
    return out


def nominate(table: pd.DataFrame) -> tuple[list[str], list[str]]:
    """The candidates the ablation will test, and the guarded ones excused from it, with reasons."""
    candidates = table.loc[table["candidate"] & ~table["guarded"], "feature"].tolist()
    excused = table.loc[table["candidate"] & table["guarded"], "feature"].tolist()
    return candidates, excused


def ablate(frame: pd.DataFrame, y: pd.Series, candidates: list[str],
           folds: list[tuple[np.ndarray, np.ndarray]], full_auc: list[float]) -> dict:
    """
    Stage 2: remove the candidates **as a group** and re-measure the same folds.

    Paired by fold, because the folds are shared - the same reason T22 §4.4 pairs T36's arms. The
    question asked is only "does the group carry measurable skill", which is why the group is ablated in
    one run rather than searched one feature at a time: a greedy search over 17 features would find a
    subset that happens to suit this instrument, and that is not a finding about the data.
    """
    kept = [column for column in features.FEATURES if column not in set(candidates)]
    if not candidates:
        return {"candidates": [], "kept": kept, "delta": 0.0, "delta_sd": 0.0,
                "dropped": False, "reduced_auc": full_auc, "columns_after": None}

    reduced_columns = columns_for(kept)
    _, reduced_auc, names = _fit_and_score(frame, y, reduced_columns, folds)
    deltas = np.array(reduced_auc) - np.array(full_auc)
    mean_delta = float(deltas.mean())
    spread = float(deltas.std(ddof=0))
    return {
        "candidates": candidates,
        "kept": kept,
        "reduced_auc": reduced_auc,
        "delta_folds": deltas.tolist(),
        "delta": mean_delta,
        "delta_sd": spread,
        "dropped": bool(mean_delta >= -spread),  # one-standard-error form (ABLATION_TOLERANCE)
        "columns_after": len(names),
        "reduced_columns": reduced_columns,
    }


def decide(table: pd.DataFrame, ablation: dict) -> pd.DataFrame:
    """Write the decision and its reason onto every row, from the two stages' numbers."""
    out = table.copy()
    dropped = set(ablation["candidates"]) if ablation["dropped"] else set()
    decisions, reasons = [], []
    for _, row in out.iterrows():
        feature = row["feature"]
        if feature in dropped:
            decisions.append("drop")
            reasons.append(
                f"nominated ({row['evidence']}: MI {row['mi']:.4f} vs its {row['null_q95']:.4f} null, "
                f"AUC lost {row['perm_mean']:+.5f} +/- {row['perm_sd']:.5f}) and the group ablation -"
                f"{len(ablation['candidates'])} features, {ablation['columns_after']} columns - moved "
                f"held-out ROC-AUC by {ablation['delta']:+.5f} +/- {ablation['delta_sd']:.5f}, so the "
                f"loss is not measurable"
            )
        else:
            decisions.append("keep")
            if row["guarded"]:
                reasons.append(f"guarded: {GUARDED[feature]}")
            elif row["candidate"]:
                reasons.append(
                    f"nominated ({row['evidence']}) but the group ablation cost "
                    f"{ablation['delta']:+.5f} +/- {ablation['delta_sd']:.5f} ROC-AUC, so the group "
                    f"carries skill collectively"
                )
            else:
                reasons.append(
                    f"both methods see it: MI z={row['z']:.0f} over its null, and shuffling it costs "
                    f"{row['perm_mean']:+.5f} +/- {row['perm_sd']:.5f} AUC"
                )
    out["decision"] = decisions
    out["reason"] = reasons
    return out


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------
def draw(table: pd.DataFrame, ablation: dict, oof_auc: float, cd_auc: float,
         path: Path = FIGURE_PATH) -> Path:
    """
    Two panels over the same 17 features, ordered by the forest's ranking.

    The left panel draws each feature's MI against its **own** 95th-percentile null, which is the only way
    the raw numbers are comparable: without that marker a reader would rank a 12-level column above a
    2-level one on bias alone. Dropped features are hatched in both panels.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = table.sort_values("perm_rank", ascending=True).reset_index(drop=True)
    labels = ordered["feature"].tolist()
    y_pos = np.arange(len(labels))
    is_drop = (ordered["decision"] == "drop").to_numpy()

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 7.2), sharey=True)

    left.barh(y_pos - 0.19, ordered["mi"], height=0.36, color="#4C78A8", label="MI (measured)",
              hatch=["//" if flag else "" for flag in is_drop], edgecolor="white")
    left.scatter(ordered["null_q95"], y_pos - 0.19, marker="|", s=150, color="#E45756",
                 label="each feature's own null 95th pct", zorder=3)
    left.set_yticks(y_pos)
    left.set_yticklabels(labels)
    left.invert_yaxis()
    left.set_xlabel("mutual information with the target (nats)")
    left.set_title("M1 - filter: mutual information against\neach feature's own permutation null",
                   fontsize=11)
    left.legend(loc="lower right", fontsize=8)
    left.grid(axis="x", alpha=0.25)

    right.barh(y_pos, ordered["perm_mean"], height=0.55, color="#54A24B",
               hatch=["//" if flag else "" for flag in is_drop], edgecolor="white",
               xerr=ordered["perm_sd"],
               error_kw={"elinewidth": 1, "ecolor": "#333333", "capsize": 2})
    right.axvline(0, color="#333333", linewidth=0.8)
    right.set_xlabel(f"ROC-AUC lost when shuffled (mean of {N_FOLDS} folds, +/- sd)")
    right.set_title(f"M2 - model: permutation importance, {INSTRUMENT['n_estimators']}-tree "
                    f"forest\n(out-of-fold ROC-AUC {oof_auc:.3f} +/- {cd_auc:.3f})", fontsize=11)
    right.grid(axis="x", alpha=0.25)

    if ablation["candidates"]:
        message = (f"M3 ablation: the {len(ablation['candidates'])} nominated features removed together "
                   f"cost {ablation['delta']:+.4f} +/- {ablation['delta_sd']:.4f} ROC-AUC")
        verdict = message + (" -> within one sd of zero, dropped" if ablation["dropped"]
                             else " -> outside the noise, kept")
    else:
        verdict = "M3 ablation: nothing was nominated"
    dropped_names = ordered.loc[is_drop, "feature"].tolist()
    second = f"Kept {int((~is_drop).sum())} of {len(ordered)}"
    if dropped_names:
        second += f"; dropped {', '.join(dropped_names)}"
    figure.suptitle(f"DepositIQ - two rankings and one measured decision over 65 columns, 17 features\n"
                    f"{verdict}\n{second}", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def validate(table: pd.DataFrame, mi_table: pd.DataFrame, perm_table: pd.DataFrame,
             info: dict, ablation: dict) -> list[dict]:
    """Checks that fail for different real mistakes, recomputed from the tables rather than from prose."""
    results: list[dict] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        results.append({"check": name, "passed": bool(passed), "detail": detail})

    matrix_columns = list(preprocess.build_preprocessor().fit_transform(
        features.load_engineered("train")).columns)
    index_blocks = block_map(matrix_columns)
    others = [source_of(column) for column in matrix_columns if column.split("=", 1)[0] not in features.FEATURES]
    add("every matrix column maps to a source feature",
        not others and sum(len(v) for v in index_blocks.values()) == info["n_columns"],
        f"{info['n_columns']} columns over {len(index_blocks)} features")

    add("both methods score the same 17 features",
        set(mi_table["feature"]) == set(perm_table["feature"]) == set(features.FEATURES),
        f"{len(table)} rows")

    add("mutual information is never negative", bool((mi_table["mi"] >= 0).all()),
        f"min {mi_table['mi'].min():.6f}")
    add("every feature has a null wider than a point", bool((mi_table["null_sd"] > 0).all()),
        f"narrowest null sd {mi_table['null_sd'].min():.2e} over {N_PERMUTATIONS} shuffles")

    # Ranks use `method="min"`, so genuine ties share a position and the ranks are not necessarily 1..17.
    # What must hold is that the ranking is consistent with the values: no feature with a larger value may
    # carry a worse rank, and the best feature must occupy rank 1.
    def ranks_consistent(values: pd.Series, ranks: pd.Series) -> bool:
        best = values.idxmax()
        if int(ranks.loc[best]) != 1:
            return False
        pairs = list(zip(values.tolist(), ranks.tolist()))
        for value_a, rank_a in pairs:
            for value_b, rank_b in pairs:
                if value_a > value_b and rank_a > rank_b:
                    return False
        return True

    add("both rankings agree with their values",
        ranks_consistent(table["mi"], table["mi_rank"])
        and ranks_consistent(table["perm_mean"], table["perm_rank"]),
        f"ranks 1..{int(table['mi_rank'].max())} (ties share a position)")

    # The screen, re-derived from its two declared conditions.
    recomputed = [(row["mi"] <= row["null_q95"]) or (row["perm_mean"] - row["perm_sd"] <= PERMUTATION_FLOOR)
                  for _, row in table.iterrows()]
    add("the candidate column follows stage 1 exactly", recomputed == table["candidate"].tolist(),
        f"{int(table['candidate'].sum())} nominated, {int(table['guarded'].sum())} guarded")

    # The decision, re-derived from the ablation.
    expected = set(ablation["candidates"]) if ablation["dropped"] else set()
    add("the decision follows stage 2 exactly",
        set(table.loc[table["decision"] == "drop", "feature"]) == expected,
        f"ablation says {'drop' if ablation['dropped'] else 'keep'} for "
        f"{len(ablation['candidates'])} candidates")

    add("no dropped feature is an EDA anchor", not set(table.loc[table["decision"] == "drop", "feature"])
        & set(ANCHORS), f"anchors {list(ANCHORS)} kept")

    add("guarded features were never dropped",
        not set(table.loc[table["decision"] == "drop", "feature"]) & set(GUARDED),
        f"{len(GUARDED)} carve-out(s) survived")

    add("the kept set is the 17 minus the drops",
        sorted(table.loc[table["decision"] == "keep", "feature"]) ==
        sorted(set(features.FEATURES) - expected),
        f"{int((table['decision'] == 'keep').sum())} kept")

    # The hand-off to T26: the kept subset must build a working matrix with exactly the width the
    # ablation measured, or the shipped column list and the experiment describe different things.
    kept_columns = columns_for(table.loc[table["decision"] == "keep", "feature"].tolist())
    shipped = preprocess.build_preprocessor(kept_columns).fit_transform(features.load_engineered("train"))
    add("the shipped matrix builds from the kept set",
        len(shipped.columns) == ablation["columns_after"]
        and len(shipped) == len(features.load_engineered("train")),
        f"{len(shipped.columns)} columns over {len(shipped):,} rows, fitted on train")

    if ablation["candidates"]:
        deltas = np.array(ablation["reduced_auc"]) - np.array(info["fold_auc"])
        add("the ablation replayed the same folds",
            len(ablation["reduced_auc"]) == len(info["fold_auc"]) == N_FOLDS,
            f"{N_FOLDS} paired folds")
        add("the ablation delta matches its fold differences",
            abs(float(deltas.mean()) - ablation["delta"]) < 1e-12
            and abs(float(deltas.std(ddof=0)) - ablation["delta_sd"]) < 1e-12,
            f"{ablation['delta']:+.5f} +/- {ablation['delta_sd']:.5f}")
        add("the ablation measured a real feature reduction",
            ablation["columns_after"] < info["n_columns"],
            f"{info['n_columns']} -> {ablation['columns_after']} columns")

    add("no forbidden feature was ranked",
        not set(table["feature"]) & {"duration", "pdays", config.TARGET}, "none present")
    add("the instrument scored above chance", info["oof_auc"] > 0.5,
        f"out-of-fold ROC-AUC {info['oof_auc']:.4f} +/- {info['cd_auc']:.4f}")
    add("the figure was written", FIGURE_PATH.exists() and FIGURE_PATH.stat().st_size > 20_000,
        f"{FIGURE_PATH.name}, {FIGURE_PATH.stat().st_size // 1024 if FIGURE_PATH.exists() else 0} KB")
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(variant: str = "full") -> dict:
    """Everything the module prints, and the source of every number in the notes."""
    frame = features.load_engineered("train", variant)
    y = encode_target(frame)
    folds = _folds(frame, y)

    mi_table = mutual_information(frame, y)
    perm_table, info = permutation_importance_by_feature(frame, y, folds)
    table = screen(compare(mi_table, perm_table))

    candidates, excused = nominate(table)
    ablation = ablate(frame, y, candidates, folds, info["fold_auc"])
    table = decide(table, ablation)
    draw(table, ablation, info["oof_auc"], info["cd_auc"])

    dropped = table.loc[table["decision"] == "drop", "feature"].tolist()
    kept = table.loc[table["decision"] == "keep", "feature"].tolist()

    return {
        "variant": variant,
        "rows": len(frame),
        "positives": int(y.sum()),
        "table": table,
        "ablation": ablation,
        "candidates": candidates,
        "excused": excused,
        "dropped": dropped,
        "kept": kept,
        # The matrix width after selection, measured by building it - not the 12-feature count.
        "kept_columns": len(preprocess.build_preprocessor(columns_for(kept)).fit_transform(frame).columns),
        "dropped_columns": info["n_columns"] - len(
            preprocess.build_preprocessor(columns_for(kept)).fit_transform(frame).columns),
        "oof_auc": info["oof_auc"],
        "cd_auc": info["cd_auc"],
        "fold_auc": info["fold_auc"],
        "n_columns": info["n_columns"],
        "blocks": info["blocks"],
        "permutation_rows": info["permutation_rows"],
        "spearman": float(table["spearman"].iloc[0]),
        "checks": validate(table, mi_table, perm_table, info, ablation),
    }


def print_report(result: dict) -> None:
    table = result["table"]
    ablation = result["ablation"]
    print("DepositIQ T25 - feature selection and importance")
    print("=" * 116)
    print(f"training rows : {result['rows']:,}  ({result['positives']:,} positives)")
    print(f"matrix        : {result['n_columns']} columns over {len(table)} source features")
    print(f"instrument    : {INSTRUMENT['n_estimators']}-tree forest, {N_FOLDS}-fold out-of-fold "
          f"ROC-AUC {result['oof_auc']:.4f} +/- {result['cd_auc']:.4f}")
    print(f"methods       : MI ({N_PERMUTATIONS} shuffles) x permutation ({N_REPEATS}x on the whole "
          f"{result['permutation_rows']:,}-row held-out fold), Spearman {result['spearman']:+.3f}")
    print()
    print(f"{'feature':<18}{'MI':>8}{'null95':>8}{'z':>7}{'MIrk':>5}{'perm':>9}{'sd':>8}"
          f"{'Prk':>5}{'gap':>5}  evidence")
    print("-" * 116)
    for _, row in table.iterrows():
        flag = "*" if row["guarded"] else " "
        print(f"{row['feature']:<18}{row['mi']:>8.4f}{row['null_q95']:>8.4f}{row['z']:>7.1f}"
              f"{row['mi_rank']:>5}{row['perm_mean']:>+9.5f}{row['perm_sd']:>8.5f}"
              f"{row['perm_rank']:>5}{row['rank_gap']:>5}  {row['evidence']}{flag}")
    print("  * = nominated but guarded (see GUARDED)")
    print()
    print("DISAGREEMENTS (|rank gap| >= 5)")
    gaps = table[table["rank_gap"].abs() >= 5].sort_values("rank_gap", key=lambda s: s.abs(), ascending=False)
    if gaps.empty:
        print("  none")
    for _, row in gaps.iterrows():
        print(f"  {row['feature']:<18} filter rank {row['mi_rank']:>2} vs model rank {row['perm_rank']:>2}"
              f"   {row['evidence']}")
    print()
    print("STAGE 1 - screen")
    print(f"  nominated {len(result['candidates'])}: {', '.join(result['candidates']) or 'none'}")
    for feature in result["excused"]:
        print(f"  excused {feature}: {GUARDED[feature]}")
    print()
    print("STAGE 2 - group ablation")
    if ablation["candidates"]:
        print(f"  removed   : {', '.join(ablation['candidates'])}")
        print(f"  columns   : {result['n_columns']} -> {ablation['columns_after']}")
        for fold, (full, reduced) in enumerate(zip(result["fold_auc"], ablation["reduced_auc"])):
            print(f"  fold {fold + 1}      : full {full:.5f}   reduced {reduced:.5f}   "
                  f"delta {reduced - full:+.5f}")
        print(f"  delta     : {ablation['delta']:+.5f} +/- {ablation['delta_sd']:.5f} ROC-AUC")
        print(f"  verdict   : {'DROP the group - the loss is within one sd of zero' if ablation['dropped'] else 'KEEP - the loss is outside the noise'}")
    else:
        print("  nothing nominated")
    print()
    print(f"DECISION  drop {len(result['dropped'])}: {', '.join(result['dropped']) or 'none'}")
    for feature in result["dropped"]:
        print(f"  {feature:<18} {table.loc[table['feature'] == feature, 'reason'].iloc[0]}")
    print(f"          keep {len(result['kept'])}: {', '.join(result['kept'])}")
    print(f"          shipped matrix: {result['kept_columns']} columns "
          f"(from {result['n_columns']})")
    print()
    print("CHECKS")
    for check in result["checks"]:
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['check']} - {check['detail']}")
    print("=" * 116)


if __name__ == "__main__":
    result = report()
    print_report(result)
    ok = all(check["passed"] for check in result["checks"])
    print(f"result: {'PASS - two rankings compared, decision measured, figure written' if ok else 'FAIL'}")
    print("Next  : T26 trains the Dummy and Logistic baselines on the kept feature set.")
    sys.exit(0 if ok else 1)
