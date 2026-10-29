"""
DepositIQ - the final model: rebuild, persist, prove (T31).

Why this file exists
--------------------
Phase 6 produced five candidate builds and one selected configuration. This module turns the selection
into an **artifact**: it rebuilds the model from `models/best_params.json` (not from numbers retyped here),
fits it on the whole training half, writes `models/final_model.joblib`, and then tries to break it -
because "the artifact reloads" is a claim that has to survive a fresh interpreter, not a `joblib.load`
in the process that just created the object.

What is deliberately *not* done here
------------------------------------
* **No held-out scoring.** The test half is never loaded. T32 owns that number, and a held-out metric
  computed inside the step that chooses the model would make the choice look better than it is.
* **No re-tuning.** The parameters come from `models/best_params.json`; a check confirms the file's
  parameters equal the ones actually applied to the fitted estimator, so the artifact cannot silently
  drift from the record that justifies it.
* **No threshold.** The model predicts probabilities. D17's 7.50% break-even gets applied in T34, after
  recalibration is settled (D45), and baking a cut into the artifact now would freeze a decision that has
  not been made yet.

The model that ships, and why it is this one
--------------------------------------------
**XGBoost, untreated, at T30's tuned point.** The three reasons are recorded with their evidence in
`docs/06_model_comparison.md`; in one line each: it has the best PR-AUC and ROC-AUC of any build in this
phase; it is the only one whose probabilities land on the base rate without recalibration (mean p 0.1166
against a 0.1170 base rate), which is what makes a cost-per-call threshold applicable; and T29 measured
the imbalance weight buying no ranking while *halving* the Brier score, so the untreated form is the one
to keep.

What the artifact accepts
-------------------------
A `pandas.DataFrame` carrying the twelve shipped feature columns, as produced by
`features.engineer(...)` followed by `select.select_features(...)` - or simply
`features.load_engineered("train")`, which the reload check uses from a **separate interpreter**. The
pipeline contains its own preprocessing, so the caller passes raw engineered columns, not a matrix. It
returns `predict_proba` with two columns; column 1 is P(subscribe).

How to run
----------
From the repository root, inside the virtual environment:

    python -m src.final    # ~10 s: one refit, one reload, one cross-process reload

Writes `models/final_model.joblib` and `models/final_model.json`; `docs/06_model_comparison.md` quotes only
numbers from the card or from the phase's earlier records.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from src import config, features, preprocess, select, train, tune
from src.split import encode_target

FINAL_PATH: Path = config.MODELS_DIR / "final_model.joblib"
CARD_PATH: Path = config.MODELS_DIR / "final_model.json"
PARAMS_PATH: Path = config.MODELS_DIR / "best_params.json"

# `compress=3` for the reason T27 gave: this file is cloned by anyone who runs the project, and an
# uncompressed pipeline is several megabytes of leaf values.
COMPRESS: int = 3

# Rows used for the reload comparison. Fixed and taken from the **start** of the training half so the
# check is reproducible and obviously not a held-out sample.
PROBE_ROWS: int = 1_000


# ---------------------------------------------------------------------------
# Build and persist
# ---------------------------------------------------------------------------
def chosen_params() -> dict:
    """The six tuned settings, read from T30's record rather than restated."""
    record = json.loads(PARAMS_PATH.read_text())
    return dict(record["best_params"]["params"])


def build(params: dict, columns: list[str]):
    """
    The final pipeline: T24's preprocessor, then the winner rebuilt through `tune.make_estimator`.

    `tune.make_estimator` is used rather than a fresh `XGBClassifier(...)` so the pinned settings that the
    search never varied - `tree_method="hist"`, `eval_metric="aucpr"`, the seed, `n_jobs=1` - are the same
    object-level defaults that produced the CV numbers being quoted. A model assembled twice from the same
    parameters is still two models if one of them got a different `tree_method`.
    """
    return train.make_pipeline(tune.make_estimator(params), columns)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cross_process_predictions() -> np.ndarray:
    """
    Score the probe rows in a **fresh interpreter** and return its predictions.

    This is the check that distinguishes "the object is in memory" from "the file works". A `joblib.load`
    in the same process can succeed while the artifact still depends on something the current process
    happens to have imported; a new process has to import whatever the pickle names, or fail.
    """
    # `-c` rather than a script file in the temp directory: with a script, Python puts the *script's*
    # directory on `sys.path`, not the working directory, so `from src import ...` fails for the wrong
    # reason. With `-c`, `sys.path[0]` is the working directory, which is the project root - the same
    # import path a user has.
    code = (
        "import sys\n"
        "import numpy as np, joblib\n"
        "from src import features\n"
        "pipe = joblib.load(sys.argv[1])\n"
        "frame = features.load_engineered('train')\n"
        f"rows = frame.iloc[:{PROBE_ROWS}]\n"
        "np.save(sys.argv[2], pipe.predict_proba(rows)[:, 1])\n"
    )
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "predictions.npy"
        completed = subprocess.run(
            [sys.executable, "-c", code, str(FINAL_PATH), str(target)],
            cwd=config.ROOT, capture_output=True, text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"reload in a fresh interpreter failed with exit {completed.returncode}: "
                f"{completed.stderr.strip()[-600:]}"
            )
        return np.load(target)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def validate(
    frame: pd.DataFrame,
    y: pd.Series,
    pipeline,
    reloaded,
    params: dict,
    card: dict,
    reference: dict,
) -> list[dict]:
    """Same contract as every other phase: a check that cannot run is FAILED, never raised (D25)."""
    checks: list[dict] = []
    probe = frame.iloc[:PROBE_ROWS]

    def compare(name: str, evaluate) -> None:
        try:
            passed, detail = evaluate()
        except Exception as error:  # noqa: BLE001 - any failure here is a failed check
            passed, detail = False, f"check could not run: {type(error).__name__}: {error}"
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    # --- the artifact is the model that was selected --------------------------------------
    def params_came_from_the_record() -> tuple[bool, str]:
        fitted = reloaded.named_steps[train.STEP_MODEL].get_params()
        wrong = {axis: (value, fitted[axis]) for axis, value in params.items() if fitted[axis] != value}
        return (
            not wrong and params == reference["best_params"]["params"],
            f"{len(params)} parameters match models/{PARAMS_PATH.name} and are applied to the persisted "
            f"estimator; wrong: {wrong or 'none'}",
        )

    compare("the artifact carries exactly the parameters T30 selected", params_came_from_the_record)

    def pinned_settings_survived() -> tuple[bool, str]:
        fitted = reloaded.named_steps[train.STEP_MODEL].get_params()
        expected = {"tree_method": "hist", "eval_metric": "aucpr", "random_state": train.FOLD_SEED,
                    "n_jobs": 1, "scale_pos_weight": tune.TREATMENT}
        wrong = {key: (value, fitted[key]) for key, value in expected.items() if fitted[key] != value}
        return (
            not wrong,
            f"tree_method/eval_metric/seed/n_jobs/treatment survived the pickle: "
            f"{ {key: fitted[key] for key in expected} }; wrong: {wrong or 'none'}",
        )

    compare("the settings the search never varied survived persistence", pinned_settings_survived)

    def structure() -> tuple[bool, str]:
        steps = list(reloaded.named_steps)
        estimator = reloaded.named_steps[train.STEP_MODEL]
        rounds = int(estimator.get_booster().num_boosted_rounds())
        return (
            steps == [train.STEP_PREPROCESS, train.STEP_MODEL]
            and isinstance(estimator, xgboost.XGBClassifier)
            and rounds == params["n_estimators"],
            f"steps {steps}; {type(estimator).__name__} with {rounds} boosted rounds "
            f"(selected {params['n_estimators']})",
        )

    compare("the artifact is a two-step pipeline with the selected number of rounds", structure)

    def matrix_width() -> tuple[bool, str]:
        transformer = reloaded.named_steps[train.STEP_PREPROCESS]
        names = transformer.get_feature_names_out()
        leaked = [name.split("=", 1)[0] for name in names if name.split("=", 1)[0] in train.FORBIDDEN]
        return (
            len(names) == 40 and not leaked,
            f"{len(names)} matrix columns from {len(select.shipped_features())} features; "
            f"forbidden columns: {leaked or 'none'}",
        )

    compare("the fitted preprocessing is still the 40-column T24 matrix", matrix_width)

    # --- it reloads, and it predicts ------------------------------------------------------
    def reload_is_identical() -> tuple[bool, str]:
        before = pipeline.predict_proba(probe)[:, 1]
        after = reloaded.predict_proba(probe)[:, 1]
        drift = float(np.max(np.abs(before - after)))
        return drift == 0.0, f"max |in-memory - reloaded| over {PROBE_ROWS} rows = {drift:.1e}"

    compare("the reloaded pipeline reproduces the fitted one bit-identically", reload_is_identical)

    def reloads_in_a_fresh_interpreter() -> tuple[bool, str]:
        outside = cross_process_predictions()
        inside = pipeline.predict_proba(frame.iloc[:PROBE_ROWS])[:, 1]
        drift = float(np.max(np.abs(outside - inside)))
        return (
            drift == 0.0 and len(outside) == PROBE_ROWS,
            f"a new Python process loaded the file and scored {len(outside)} rows, max drift "
            f"{drift:.1e}",
        )

    compare("the file loads and predicts in a separate process", reloads_in_a_fresh_interpreter)

    def probabilities_are_probabilities() -> tuple[bool, str]:
        scores = reloaded.predict_proba(probe)
        one = reloaded.predict_proba(probe.iloc[:1])
        return (
            scores.shape == (PROBE_ROWS, 2)
            and np.isfinite(scores).all()
            and scores.min() >= 0.0 and scores.max() <= 1.0
            and np.allclose(scores.sum(axis=1), 1.0, atol=1e-12)
            and one.shape == (1, 2),
            f"{scores.shape} with rows in [{scores.min():.6f}, {scores.max():.6f}], "
            f"columns summing to 1 within {np.abs(scores.sum(axis=1) - 1).max():.1e}; "
            f"single row -> {one.shape}",
        )

    compare("predict_proba returns finite probabilities for one row or many", probabilities_are_probabilities)

    def deterministic() -> tuple[bool, str]:
        first = reloaded.predict_proba(probe)[:, 1]
        second = reloaded.predict_proba(probe)[:, 1]
        return np.array_equal(first, second), "two consecutive calls return identical arrays"

    compare("the artifact predicts deterministically", deterministic)

    def trained_on_everything() -> tuple[bool, str]:
        """
        The artifact must be fitted on the whole training half - not on one fold, and not on the test half.

        Both errors are easy and neither announces itself: a fold-fitted artifact looks fine until someone
        notices its predictions are worse than the CV table, and a test-fitted one cannot be detected from
        the artifact at all, which is why the source scan below exists as a second, independent check.
        """
        # A regex, not a substring search: the earlier version searched the source for a literal that is
        # itself written in this file - as part of the check - and reported itself. Nothing below quotes
        # the call site it is looking for, so the pattern cannot match its own text.
        source = Path(__file__).read_text()
        clean = re.search(r"load_engineered\(\s*['\"]test", source) is None
        return (
            card["training_rows"] == int(len(frame))
            and card["positives"] == int(y.sum())
            and clean,
            f"fitted on {card['training_rows']:,} rows / {card['positives']:,} positives, all of the "
            f"training half; test-half loads in this module: {'found' if not clean else 'none'}",
        )

    compare("the artifact was fitted on the training half and never saw the test half", trained_on_everything)

    # --- the card describes the artifact ---------------------------------------------------
    def card_matches_file() -> tuple[bool, str]:
        return (
            card["artifact"]["sha256"] == sha256(FINAL_PATH)
            and card["artifact"]["bytes"] == FINAL_PATH.stat().st_size,
            f"{FINAL_PATH.name} is {card['artifact']['bytes']:,} bytes, sha256 {card['artifact']['sha256'][:16]}...",
        )

    compare("the card's checksum and size describe the file on disk", card_matches_file)

    def card_quotes_the_cv_record() -> tuple[bool, str]:
        tuned = reference["arms"]["tuned_untreated"]["summary"]
        card_cv = card["cross_validation"]
        return (
            abs(card_cv["pr_auc"]["mean"] - tuned["pr_auc"]["mean"]) <= 1e-12
            and abs(card_cv["pr_auc"]["sd"] - tuned["pr_auc"]["sd"]) <= 1e-12
            and abs(card_cv["roc_auc"]["mean"] - tuned["roc_auc"]["mean"]) <= 1e-12,
            f"card PR-AUC {card_cv['pr_auc']['mean']:.6f} +/- {card_cv['pr_auc']['sd']:.6f} equals "
            f"T30's tuned arm; ROC-AUC {card_cv['roc_auc']['mean']:.6f}",
        )

    compare("the card's CV numbers are T30's, not retyped", card_quotes_the_cv_record)

    return checks


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(variant: str = "full") -> dict:
    """Rebuild the winner, fit it on the training half, persist it, and check the file."""
    config.ensure_dirs()
    started = time.perf_counter()

    frame = features.load_engineered("train", variant)
    y = encode_target(frame)
    columns = select.shipped_features()
    params = chosen_params()
    reference = json.loads(PARAMS_PATH.read_text())

    fit_started = time.perf_counter()
    pipeline = build(params, columns)
    pipeline.fit(frame, y)
    fit_seconds = round(time.perf_counter() - fit_started, 1)

    joblib.dump(pipeline, FINAL_PATH, compress=COMPRESS)
    reloaded = joblib.load(FINAL_PATH)

    estimator = pipeline.named_steps[train.STEP_MODEL]
    tuned = reference["arms"]["tuned_untreated"]["summary"]

    card = {
        "task": "T31",
        "variant": variant,
        "chosen": "XGBoost (xgboost_gbdt), untreated, at T30's tuned point",
        "rationale": "see docs/06_model_comparison.md; summaries: best PR-AUC/ROC-AUC of the phase, "
                     "mean prediction on the base rate without recalibration, and the untreated form "
                     "chosen by T29's measurement",
        "artifact": {
            "path": str(FINAL_PATH.relative_to(config.ROOT)),
            "bytes": FINAL_PATH.stat().st_size,
            "sha256": sha256(FINAL_PATH),
            "compress": COMPRESS,
            "format": "joblib: sklearn Pipeline(preprocess=ColumnTransformer, model=XGBClassifier)",
        },
        "parameters": params,
        "pinned": {
            "tree_method": estimator.get_params()["tree_method"],
            "eval_metric": estimator.get_params()["eval_metric"],
            "random_state": estimator.get_params()["random_state"],
            "n_jobs": estimator.get_params()["n_jobs"],
            "scale_pos_weight": float(estimator.get_params()["scale_pos_weight"]),
            "boosted_rounds": int(estimator.get_booster().num_boosted_rounds()),
        },
        "features": list(columns),
        "matrix_columns": int(len(
            pipeline.named_steps[train.STEP_PREPROCESS].get_feature_names_out())),
        "training_rows": int(len(frame)),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "fold_seed": train.FOLD_SEED,
        "cross_validation": {
            "source": str(PARAMS_PATH.relative_to(config.ROOT)),
            "arm": "tuned_untreated",
            "pr_auc": {"mean": tuned["pr_auc"]["mean"], "sd": tuned["pr_auc"]["sd"]},
            "roc_auc": {"mean": tuned["roc_auc"]["mean"], "sd": tuned["roc_auc"]["sd"]},
            "brier": {"mean": tuned["brier"]["mean"], "sd": tuned["brier"]["sd"]},
            "mean_pred": {"mean": tuned["mean_pred"]["mean"], "sd": tuned["mean_pred"]["sd"]},
            "note": "5-fold stratified CV on the training half only. The test half has not been scored; "
                    "T32 owns that number, and the CV figure here is optimistic because T30 chose the "
                    "configuration on these same folds.",
        },
        "usage": {
            "input": "DataFrame with the twelve `features` columns (features.engineer -> select.select_features)",
            "output": "predict_proba(...)[:, 1] = P(subscribe); the pipeline carries its own preprocessing",
            "threshold": "none baked in: T34 applies D17's 7.50% break-even after recalibration",
        },
        "library": {"xgboost": xgboost.__version__, "scikit-learn": sklearn.__version__,
                    "pandas": pd.__version__, "numpy": np.__version__},
        "fit_seconds": fit_seconds,
    }

    CARD_PATH.write_text(json.dumps(card, indent=2, sort_keys=False))
    checks = validate(frame, y, pipeline, reloaded, params, json.loads(CARD_PATH.read_text()), reference)
    card["checks"] = checks
    card["checks_passed"] = sum(1 for check in checks if check["passed"])
    card["checks_total"] = len(checks)
    card["checks_ok"] = card["checks_passed"] == card["checks_total"]
    card["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    CARD_PATH.write_text(json.dumps(card, indent=2, sort_keys=False))

    return {**card, "pipeline": reloaded}


def print_report(result: dict) -> None:
    print("DepositIQ T31 - the final model")
    print("=" * 112)
    print(f"chosen   : {result['chosen']}")
    print(f"artifact : {result['artifact']['path']} ({result['artifact']['bytes']:,} bytes, "
          f"sha256 {result['artifact']['sha256'][:16]}...)")
    print(f"parameters: " + ", ".join(f"{axis}={value}" for axis, value in result["parameters"].items()))
    print(f"pinned   : " + ", ".join(f"{key}={value}" for key, value in result["pinned"].items()))
    print(f"trained  : {result['training_rows']:,} rows ({result['positives']:,} positives, "
          f"{result['base_rate']:.4%}) x {len(result['features'])} features -> "
          f"{result['matrix_columns']} matrix columns; fit {result['fit_seconds']} s")
    cv = result["cross_validation"]
    print(f"CV       : PR-AUC {cv['pr_auc']['mean']:.4f} +/- {cv['pr_auc']['sd']:.4f}, "
          f"ROC-AUC {cv['roc_auc']['mean']:.4f} +/- {cv['roc_auc']['sd']:.4f}, "
          f"Brier {cv['brier']['mean']:.4f}, mean p {cv['mean_pred']['mean']:.4f}")
    print(f"           ({cv['source']}, arm {cv['arm']} - training half only, optimistic: see note)")
    print()
    print(f"T31 checks: {result['checks_passed']}/{result['checks_total']} passed")
    for check in result["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        detail = f"  ({check['detail']})" if check["detail"] else ""
        print(f"  [{marker}] {check['check']}{detail}")
    print()


if __name__ == "__main__":
    print("DepositIQ Phase 6 - persisting the selected model")
    print()
    outcome = report()
    print_report(outcome)
    print("=" * 112)
    print(f"result : {'PASS - the selected model is persisted and the file reloads and predicts' if outcome['checks_ok'] else 'FAIL'}")
    print("Next   : T32 scores this artifact once on the held-out half.")
    sys.exit(0 if outcome["checks_ok"] else 1)
