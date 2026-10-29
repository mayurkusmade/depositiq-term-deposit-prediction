"""
DepositIQ - the interactive demo core (T43).

Why this file exists
--------------------
The repository has a report, a deck and a README. This is the one deliverable a marker can click: a
Streamlit app that takes a client's attributes and returns **P(subscribe)** with the SHAP drivers behind
it. The UI is `app/app.py`; everything it displays is computed here, so the logic can be checked without
a browser and the app cannot drift into a second implementation of the model.

Five constraints, and where each one is enforced
------------------------------------------------
1. **The app scores the shipped artifact, not a fresh fit.** `load_model()` loads
   `models/final_model.joblib` and refuses it unless its sha256 and size equal `models/final_model.json`.
   The probability the app shows is the same number T33 recorded for the same row.
2. **The drivers are D58's aggregation.** TreeSHAP contributions are summed **with their sign** inside
   each one-hot block, in log-odds. The twelve feature values still add to `margin - reference`; the
   naive per-column alternative inflates `month` by **58.52%** and is not used.
3. **`duration` is not an input.** It is absent from `INPUT_FIELDS`, from the frame handed to the
   pipeline and from the 40-column matrix. The check is mechanical, not a promise in a comment.
4. **The output is a probability against the price, not a verdict.** The cut is T34's **0.0750 = C/V**,
   re-derived from `src.cost`; the app never treats 0.5 as a decision rule. A client at 0.5 is called,
   because 0.5 is far above the price.
5. **The euro figures are labelled assumptions.** **1.50 EUR** per call and **20.00 EUR** per
   subscription come from `src.cost.ASSUMPTIONS`; the app shows `EV = p*V - C` and the 20-cell
   sensitivity table rather than presenting them as a bank's own numbers.

What the form asks for, and what it derives
-------------------------------------------
The shipped model uses **12 features**, but the form asks for **11 business facts** and derives the
rest exactly as `src.features.engineer` does:

| form field | derived feature |
|---|---|
| `previous` | `previous_capped` (cap at 2) and `previously_contacted` (`previous > 0`, D21) |
| `balance` | `balance_arsinh` |
| `campaign` | `campaign_bucket` |
| `days_since_contact` | `recency_band` (`never` when `previous == 0`) |

`job`, `education`, `default` and `loan` are not asked for because T25 dropped them (D44). `pdays` is
not asked for because D36 replaced it with `days_since_contact`.

Three exemplar clients, from the record rather than retyped
-----------------------------------------------------------
`example_clients()` reads `models/shap_local.json` for the three T38 rows (true positive, false
positive, false negative) and pulls their raw values from `data/processed/train.csv`. The app can start
from any of them, which makes the demo usable without typing eleven fields and keeps the illustration
consistent with the report. Their recorded scores are checked against the artifact to **1e-9**.

How to run
----------
    python -m src.demo          # ~2 s: headless checks, writes models/demo.json
    streamlit run app/app.py    # the interactive UI, from the repository root

`models/demo.json` records the checks, the default client's score and the three exemplar scores. It is a
record, not a second source of truth: every number in the app is recomputed from the artifact on each
rerun.
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

from src import clean, config, cost, explain, features, select, split, train

ARTIFACT_PATH: Path = config.MODELS_DIR / "final_model.joblib"
CARD_PATH: Path = config.MODELS_DIR / "final_model.json"
LOCAL_PATH: Path = config.MODELS_DIR / "shap_local.json"
THRESHOLD_PATH: Path = config.MODELS_DIR / "threshold_choice.json"
RECORD_PATH: Path = config.MODELS_DIR / "demo.json"
APP_PATH: Path = config.APP_DIR / "app.py"

# The price is C/V, not a fitted number. `float()` is deliberate: the record stores a Python float and
# every displayed comparison goes through the same value.
PRICE: float = float(cost.break_even_threshold())
CALL_COST: float = float(cost.CALL_COST_EUR)
VALUE: float = float(cost.VALUE_PER_SUBSCRIPTION_EUR)

# The 11 form fields. `previously_contacted` is derived from `previous`; it is not a separate question.
INPUT_FIELDS: tuple[str, ...] = (
    "age",
    "day",
    "balance",
    "campaign",
    "previous",
    "days_since_contact",
    "marital",
    "housing",
    "contact",
    "month",
    "poutcome",
)

CATEGORICAL_INPUTS: tuple[str, ...] = ("marital", "housing", "contact", "month", "poutcome")

# Never accepted, never built, never explained. `train.FORBIDDEN` is the same list the model path uses.
FORBIDDEN_INPUTS: tuple[str, ...] = ("duration", "pdays", config.TARGET)

# The three T38 cells, in the order the app lists them.
EXEMPLAR_CELLS: tuple[str, ...] = ("true_positive", "false_positive", "false_negative")


# --------------------------------------------------------------------------- #
# The artifact
# --------------------------------------------------------------------------- #
def load_model(path: Path = ARTIFACT_PATH) -> dict:
    """
    Load the shipped pipeline, and refuse it if the card does not describe the bytes on disk.

    A demo that silently scores a re-fitted model is worse than a demo that fails: the probability would
    look plausible while no longer being the number the report quotes. The digest is therefore checked
    before the object is returned, not after the first prediction.
    """
    card = json.loads(CARD_PATH.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    size = path.stat().st_size
    expected = card["artifact"]
    if digest != expected["sha256"] or size != expected["bytes"]:
        raise RuntimeError(
            f"{path.name} does not match models/{CARD_PATH.name}: "
            f"{size:,} bytes / sha256 {digest[:16]}... against "
            f"{expected['bytes']:,} bytes / sha256 {expected['sha256'][:16]}..."
        )
    return {"pipeline": joblib.load(path), "digest": digest, "bytes": size, "card": card}


# --------------------------------------------------------------------------- #
# The form's vocabulary and defaults
# --------------------------------------------------------------------------- #
def training_profile() -> dict:
    """
    The categorical levels and sensible defaults, read from the training half.

    Deriving them from the data rather than typing a list means a level cannot be offered that the
    encoder has never seen, and the default client is a median/mode client rather than an invention.
    """
    # The processed split is comma-separated; `config.CSV_SEPARATOR` is for the raw UCI files. Going
    # through `split.load_split` keeps the dtype restoration (the boolean flag) in one place.
    frame = split.load_split("full", "train")
    options = {
        column: [str(value) for value in frame[column].astype(str).value_counts().index]
        for column in CATEGORICAL_INPUTS
    }
    previous = int(round(float(frame["previous"].median())))
    contacted = frame.loc[frame["previous"] > 0, clean.CONTACT_RECENCY].dropna()
    defaults = {
        "age": int(round(float(frame["age"].median()))),
        "day": int(round(float(frame["day"].median()))),
        "balance": float(frame["balance"].median()),
        "campaign": int(round(float(frame["campaign"].median()))),
        "previous": previous,
        "days_since_contact": float(contacted.median()) if previous > 0 and len(contacted) else None,
        "marital": str(frame["marital"].mode().iloc[0]),
        "housing": str(frame["housing"].mode().iloc[0]),
        "contact": str(frame["contact"].mode().iloc[0]),
        "month": str(frame["month"].mode().iloc[0]),
        "poutcome": str(frame["poutcome"].mode().iloc[0]),
    }
    return {"options": options, "defaults": defaults, "rows": int(len(frame))}


# --------------------------------------------------------------------------- #
# One client, and one explanation
# --------------------------------------------------------------------------- #
def client_frame(values: dict) -> pd.DataFrame:
    """
    Turn the form's 11 facts into the engineered frame `features.engineer` would produce for one row.

    The derivation is the project's, not the app's: `previous > 0` is D21's contact flag, the cap and
    the bands come from `src.features`, and the raw balance is kept so the driver table can show the
    euro amount beside the arcsinh value the model saw.
    """
    missing = [field for field in INPUT_FIELDS if field not in values]
    if missing:
        raise KeyError(f"client_frame() is missing {missing}")
    forbidden = [field for field in FORBIDDEN_INPUTS if field in values]
    if forbidden:
        raise ValueError(f"client_frame() was handed a column the model must never see: {forbidden}")

    previous = int(values["previous"])
    days = values.get("days_since_contact")
    if previous > 0:
        if days is None or not np.isfinite(float(days)) or float(days) < 0:
            raise ValueError(
                "days_since_contact must be a non-negative number when previous > 0; "
                "use previous = 0 for a client who was never contacted"
            )
        days_value = float(days)
    else:
        # Never contacted: the recency band is the explicit `never` level, not a zero-day recency.
        days_value = np.nan

    row = {
        "age": int(values["age"]),
        "day": int(values["day"]),
        "balance": float(values["balance"]),
        "campaign": int(values["campaign"]),
        "previous": previous,
        clean.CONTACT_FLAG: previous > 0,
        clean.CONTACT_RECENCY: days_value,
        "marital": str(values["marital"]),
        "housing": str(values["housing"]),
        "contact": str(values["contact"]),
        "month": str(values["month"]),
        "poutcome": str(values["poutcome"]),
    }
    return features.engineer(pd.DataFrame([row]))


def shipped_frame(client: pd.DataFrame) -> pd.DataFrame:
    """The twelve columns the artifact was fitted on, in `select.shipped_features()` order."""
    return client.loc[:, select.shipped_features()].copy()


def display_value(feature: str, row_x: pd.Series, row_frame: pd.Series,
                  columns: list[str], blocks: dict[str, list[int]]) -> str:
    """
    What the model saw for this feature, as a readable string.

    A categorical is read off its one-hot block (the column holding 1); `balance_arsinh` shows both the
    transformed value and the euro balance behind it, so a reader is not asked to invert an arcsinh in
    their head.
    """
    if feature in features.CATEGORICAL_FEATURES:
        block = row_x.iloc[blocks[feature]].to_numpy()
        position = int(np.argmax(block))
        return columns[blocks[feature][position]].split("=", 1)[1]
    if feature == features.BALANCE_ARSINH:
        return f"{float(row_frame[feature]):.2f} (balance {float(row_frame['balance']):,.0f} EUR)"
    if feature == clean.CONTACT_FLAG:
        return "yes" if bool(row_frame[feature]) else "no"
    value = float(row_frame[feature])
    if feature in ("age", "day", features.PREVIOUS_CAPPED):
        return f"{value:.0f}"
    return f"{value:.3f}"


def explain_client(pipeline, client: pd.DataFrame) -> dict:
    """
    Score one client and return the twelve signed SHAP contributions behind the score.

    The served probability is `predict_proba`, not `sigmoid(shap margin)`: T38 measured the two routes
    differing by up to **4.49e-07**, and **43** clients sit within **1e-4** of the cut, so the route is
    named rather than assumed. The margin check below proves the explanation is of the scored row.
    """
    frame = shipped_frame(client)
    matrix = explain.design_matrix(pipeline, frame)
    explained = explain.explain(pipeline, matrix)
    columns = explained["names"]
    blocks = select.block_map(columns)
    values = np.asarray(explained["values"], dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    contributions = [
        {
            "feature": feature,
            "value": display_value(feature, matrix.iloc[0], client.iloc[0], columns, blocks),
            "contribution": float(values[0, indices].sum()),
            "columns": len(indices),
        }
        for feature, indices in blocks.items()
    ]
    # `top drivers` means largest absolute contribution; the sign is kept in the printed value.
    contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)

    probability = float(pipeline.predict_proba(frame)[:, 1][0])
    margin = float(np.ravel(explained["margin"])[0])
    base = float(np.ravel(explained["base"])[0])
    total = float(sum(item["contribution"] for item in contributions))
    return {
        "probability": probability,
        "base": base,
        "margin": margin,
        "total": total,
        "shap_probability": float(1.0 / (1.0 + np.exp(-margin))),
        "contributions": contributions,
        "features": len(blocks),
        "matrix_columns": int(matrix.shape[1]),
    }


# --------------------------------------------------------------------------- #
# The decision, in euros
# --------------------------------------------------------------------------- #
def should_call(probability: float) -> bool:
    """True when the expected net value of the call is positive, using the stated C and V."""
    return bool(cost.should_call(float(probability), CALL_COST, VALUE))


def expected_value(probability: float) -> float:
    """`EV = p*V - C` in euro, with C and V the assumed figures from `src.cost`."""
    return float(cost.expected_net_value_per_call(float(probability), CALL_COST, VALUE))


# --------------------------------------------------------------------------- #
# The three T38 exemplars
# --------------------------------------------------------------------------- #
def example_clients() -> dict[str, dict]:
    """
    The three T38 rows, rebuilt from `data/processed/train.csv` and the local-SHAP record.

    The record holds the row index, the label and the score; the data holds the attributes. Neither is
    retyped, so an example cannot drift away from the illustration in `docs/08_explainability.md`.
    """
    record = json.loads(LOCAL_PATH.read_text(encoding="utf-8"))
    frame = features.load_engineered("train")
    examples: dict[str, dict] = {}
    for cell in EXEMPLAR_CELLS:
        item = record["exemplars"][cell]
        row = frame.iloc[int(item["row"])]
        examples[cell] = {
            "values": {
                "age": int(row["age"]),
                "day": int(row["day"]),
                "balance": float(row["balance"]),
                "campaign": int(row["campaign"]),
                "previous": int(row["previous"]),
                "days_since_contact": (
                    None if pd.isna(row[clean.CONTACT_RECENCY]) else float(row[clean.CONTACT_RECENCY])
                ),
                "marital": str(row["marital"]),
                "housing": str(row["housing"]),
                "contact": str(row["contact"]),
                "month": str(row["month"]),
                "poutcome": str(row["poutcome"]),
            },
            "row": int(item["row"]),
            "label": int(item["label"]),
            "score": float(item["score"]),
            "clients_in_cell": int(item["clients_in_cell"]),
        }
    return examples


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def validate(bundle: dict, profile: dict, examples: dict[str, dict],
             default_client: pd.DataFrame, default_result: dict) -> list[dict]:
    """Same contract as every other phase: a check that cannot run is FAILED, never raised (D25)."""
    checks: list[dict] = []
    pipeline = bundle["pipeline"]
    card = bundle["card"]

    def compare(name: str, evaluate) -> None:
        try:
            passed, detail = evaluate()
        except Exception as error:  # noqa: BLE001 - any failure here is a failed check
            passed, detail = False, f"check could not run: {type(error).__name__}: {error}"
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    # --- the artifact -----------------------------------------------------------------
    def artifact_is_t31() -> tuple[bool, str]:
        return (
            bundle["digest"] == card["artifact"]["sha256"]
            and bundle["bytes"] == card["artifact"]["bytes"],
            f"sha256 {bundle['digest'][:16]}..., {bundle['bytes']:,} bytes = models/final_model.json",
        )

    compare("the demo scores T31's artifact, not a fresh fit", artifact_is_t31)

    def pipeline_shape() -> tuple[bool, str]:
        steps = list(pipeline.named_steps)
        names = list(pipeline.named_steps[train.STEP_PREPROCESS].get_feature_names_out())
        return (
            steps == [train.STEP_PREPROCESS, train.STEP_MODEL]
            and len(names) == 40
            and len(select.shipped_features()) == 12,
            f"steps {steps}; {len(names)} matrix columns from {len(select.shipped_features())} features",
        )

    compare("the pipeline is the 12-feature / 40-column artifact", pipeline_shape)

    # --- the price, not 0.5 ------------------------------------------------------------
    def price_is_the_record() -> tuple[bool, str]:
        record = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
        recorded = float(record["decision"]["threshold"])
        derived = float(cost.break_even_threshold())
        return (
            abs(PRICE - recorded) < 1e-12 and abs(PRICE - derived) < 1e-12,
            f"{PRICE:.6f} equals T34's record and C/V = {derived:.6f}",
        )

    compare("the cut is T34's price, re-derived from C/V", price_is_the_record)

    def decision_uses_the_price() -> tuple[bool, str]:
        return (
            abs(PRICE - 0.5) > 0.1
            and should_call(0.5)
            and not should_call(PRICE)
            and should_call(PRICE + 0.001),
            f"0.5 -> {should_call(0.5)}, price -> {should_call(PRICE)}, "
            f"price+0.001 -> {should_call(PRICE + 0.001)}",
        )

    compare("a client at 0.5 is called, and the price is the boundary", decision_uses_the_price)

    def euro_arithmetic() -> tuple[bool, str]:
        at_half = expected_value(0.5)
        at_price = expected_value(PRICE)
        return (
            abs(at_half - (0.5 * VALUE - CALL_COST)) < 1e-12
            and abs(at_price) < 1e-12,
            f"EV(0.5) = {at_half:+.2f} EUR, EV(price) = {at_price:+.6f} EUR, "
            f"EV = p x {VALUE:.2f} - {CALL_COST:.2f}",
        )

    compare("expected value is p*V - C, and the price is break-even", euro_arithmetic)

    def assumptions_are_labelled() -> tuple[bool, str]:
        statuses = {item["name"]: item["status"] for item in cost.ASSUMPTIONS}
        table = cost.sensitivity_table()
        cell = table.loc[table["call_cost_eur"] == CALL_COST, "value_20"].iloc[0]
        return (
            abs(CALL_COST - 1.50) < 1e-12
            and abs(VALUE - 20.00) < 1e-12
            and statuses.get("AGENT_COST_PER_HOUR_EUR") == "ASSUMED"
            and statuses.get("AVERAGE_DEPOSIT_EUR") == "ASSUMED"
            and abs(float(cell) - PRICE) < 1e-12,
            f"C {CALL_COST:.2f} EUR (ASSUMED), V {VALUE:.2f} EUR (ASSUMED); "
            f"the sensitivity grid returns {float(cell):.4f} at C=1.50, V=20.00",
        )

    compare("the euro inputs are labelled assumptions and the grid agrees", assumptions_are_labelled)

    # --- the leak is absent ------------------------------------------------------------
    def duration_is_not_an_input() -> tuple[bool, str]:
        forbidden = [field for field in FORBIDDEN_INPUTS if field in INPUT_FIELDS]
        columns = list(default_client.columns)
        matrix = list(explain.design_matrix(pipeline, shipped_frame(default_client)).columns)
        bad = [name for name in ("duration", "pdays", config.TARGET)
               if name in columns or any(column.split("=", 1)[0] == name for column in matrix)]
        return (
            not forbidden and not bad and "duration" not in INPUT_FIELDS,
            f"{len(INPUT_FIELDS)} form fields; forbidden in the form: {forbidden or 'none'}; "
            f"forbidden in the frame/matrix: {bad or 'none'}",
        )

    compare("duration, pdays and the target are absent from the form and the matrix", duration_is_not_an_input)

    # --- the form ----------------------------------------------------------------------
    def defaults_are_in_the_vocabulary() -> tuple[bool, str]:
        wrong = [
            (field, profile["defaults"][field], profile["options"][field])
            for field in CATEGORICAL_INPUTS
            if profile["defaults"][field] not in profile["options"][field]
        ]
        empty = [field for field, values in profile["options"].items() if not values]
        return (
            not wrong and not empty and profile["rows"] == card["training_rows"],
            f"{len(CATEGORICAL_INPUTS)} categorical fields, {profile['rows']:,} training rows; "
            f"wrong defaults: {wrong or 'none'}",
        )

    compare("every categorical default is a level the encoder knows", defaults_are_in_the_vocabulary)

    def defaults_are_the_median_client() -> tuple[bool, str]:
        defaults = profile["defaults"]
        return (
            18 <= int(defaults["age"]) <= 100
            and 1 <= int(defaults["day"]) <= 31
            and int(defaults["campaign"]) >= 1
            and int(defaults["previous"]) >= 0,
            f"age {defaults['age']}, day {defaults['day']}, balance {defaults['balance']:.0f}, "
            f"campaign {defaults['campaign']}, previous {defaults['previous']}",
        )

    compare("the default client is inside the form's ranges", defaults_are_the_median_client)

    def never_contacted_is_the_explicit_level() -> tuple[bool, str]:
        previous = int(default_client["previous"].iloc[0])
        band = str(default_client[features.RECENCY_BAND].iloc[0])
        return (
            (previous > 0) or band == features.RECENCY_NEVER,
            f"previous {previous}, recency_band {band!r}",
        )

    compare("a client with no prior contact gets recency 'never', not a zero-day value",
            never_contacted_is_the_explicit_level)

    def shipped_columns_are_the_model_columns() -> tuple[bool, str]:
        columns = list(shipped_frame(default_client).columns)
        return (
            columns == select.shipped_features(),
            f"{len(columns)} columns in the shipped order: {columns}",
        )

    compare("the frame handed to the pipeline is the shipped twelve, in order",
            shipped_columns_are_the_model_columns)

    # --- the explanation ---------------------------------------------------------------
    def twelve_features_cover_the_matrix() -> tuple[bool, str]:
        columns = list(explain.design_matrix(pipeline, shipped_frame(default_client)).columns)
        blocks = select.block_map(columns)
        covered = sorted(index for indices in blocks.values() for index in indices)
        return (
            len(blocks) == 12 and covered == list(range(len(columns))) and default_result["matrix_columns"] == 40,
            f"{len(blocks)} features cover all {len(covered)} columns exactly once",
        )

    compare("the twelve feature contributions partition the 40 columns", twelve_features_cover_the_matrix)

    def additivity() -> tuple[bool, str]:
        error = abs(default_result["total"] + default_result["base"] - default_result["margin"])
        return (
            error < 1e-4,
            f"|sum of twelve + reference - margin| = {error:.3e}; reference "
            f"{default_result['base']:+.5f}",
        )

    compare("the twelve contributions add up to the score", additivity)

    def probability_matches_the_margin() -> tuple[bool, str]:
        error = abs(default_result["shap_probability"] - default_result["probability"])
        return (
            error < 1e-6,
            f"|sigmoid(margin) - predict_proba| = {error:.3e} on the default client",
        )

    compare("the explained margin is the probability the app displays", probability_matches_the_margin)

    def drivers_are_sorted() -> tuple[bool, str]:
        magnitudes = [abs(item["contribution"]) for item in default_result["contributions"]]
        return (
            magnitudes == sorted(magnitudes, reverse=True)
            and all(item["value"] for item in default_result["contributions"]),
            f"{len(magnitudes)} drivers, largest |contribution| {magnitudes[0]:.4f}, "
            f"smallest {magnitudes[-1]:.4f}",
        )

    compare("the driver table is sorted by absolute contribution and every value is shown",
            drivers_are_sorted)

    def score_is_a_probability() -> tuple[bool, str]:
        p = default_result["probability"]
        return (
            np.isfinite(p) and 0.0 <= p <= 1.0,
            f"P(subscribe) = {p:.6f} for the default client",
        )

    compare("the displayed score is a finite probability", score_is_a_probability)

    # --- the examples ------------------------------------------------------------------
    def examples_are_the_three_cells() -> tuple[bool, str]:
        wrong = [cell for cell in EXEMPLAR_CELLS if cell not in examples]
        return (
            not wrong and len(examples) == 3,
            f"cells {sorted(examples)}; missing {wrong or 'none'}",
        )

    compare("the three T38 exemplars load from the record", examples_are_the_three_cells)

    def examples_reproduce_their_scores() -> tuple[bool, str]:
        errors = {}
        for cell, item in examples.items():
            frame = client_frame(item["values"])
            scored = float(pipeline.predict_proba(shipped_frame(frame))[:, 1][0])
            errors[cell] = abs(scored - item["score"])
        return (
            max(errors.values()) < 1e-9,
            f"max |rebuilt score - T38 record| over 3 clients = {max(errors.values()):.3e}",
        )

    compare("each exemplar rebuilds to the score T38 recorded", examples_reproduce_their_scores)

    def examples_sit_on_the_right_side_of_the_price() -> tuple[bool, str]:
        sides = {
            "true_positive": examples["true_positive"]["score"] >= PRICE,
            "false_positive": examples["false_positive"]["score"] >= PRICE,
            "false_negative": examples["false_negative"]["score"] < PRICE,
        }
        labels = {cell: examples[cell]["label"] for cell in EXEMPLAR_CELLS}
        return (
            all(sides.values()) and labels == {"true_positive": 1, "false_positive": 0, "false_negative": 1},
            f"scores { {cell: round(examples[cell]['score'], 4) for cell in EXEMPLAR_CELLS} }; "
            f"labels {labels}; all on the expected side: {all(sides.values())}",
        )

    compare("the three exemplars sit in the cells their names claim",
            examples_sit_on_the_right_side_of_the_price)

    def examples_explain_additively() -> tuple[bool, str]:
        errors = {}
        for cell, item in examples.items():
            frame = client_frame(item["values"])
            result = explain_client(pipeline, frame)
            errors[cell] = abs(result["total"] + result["base"] - result["margin"])
        return (
            max(errors.values()) < 1e-4,
            f"max |sum + reference - margin| over 3 exemplars = {max(errors.values()):.3e}",
        )

    compare("each exemplar's twelve contributions add up to its score", examples_explain_additively)

    # --- the app is a UI over this core ------------------------------------------------
    def app_exists_and_compiles() -> tuple[bool, str]:
        text = APP_PATH.read_text(encoding="utf-8")
        compile(text, str(APP_PATH), "exec")
        return (
            "streamlit" in text and "demo" in text and "duration" not in re.findall(
                r"st\.(?:number_input|selectbox|slider|text_input)\(\s*[\"']([^\"']*)[\"']", text
            ),
            f"{APP_PATH.name}: {len(text.splitlines())} lines, compiles, imports the core",
        )

    compare("the Streamlit app exists, compiles and is a UI over this module",
            app_exists_and_compiles)

    def no_held_out_read() -> tuple[bool, str]:
        pattern = re.compile(r"load_engineered\(\s*['\"]test")
        hits = {
            "src/demo.py": len(pattern.findall(Path(__file__).read_text(encoding="utf-8"))),
            "app/app.py": len(pattern.findall(APP_PATH.read_text(encoding="utf-8"))),
        }
        return (
            not any(hits.values()),
            f"held-out loads in the demo path: {hits}",
        )

    compare("the demo never loads the held-out half", no_held_out_read)

    return checks


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def report() -> dict:
    """Score the default client, check the demo, and write `models/demo.json`."""
    config.ensure_dirs()
    started = time.perf_counter()

    bundle = load_model()
    profile = training_profile()
    examples = example_clients()
    default_client = client_frame(profile["defaults"])
    default_result = explain_client(bundle["pipeline"], default_client)
    checks = validate(bundle, profile, examples, default_client, default_result)

    record = {
        "task": "T43",
        "app": str(APP_PATH.relative_to(config.ROOT)),
        "core": str(Path(__file__).relative_to(config.ROOT)),
        "artifact": {
            "path": str(ARTIFACT_PATH.relative_to(config.ROOT)),
            "bytes": bundle["bytes"],
            "sha256": bundle["digest"],
        },
        "price": PRICE,
        "price_source": "cost.break_even_threshold() = C/V; equals models/threshold_choice.json",
        "call_cost_eur": CALL_COST,
        "value_per_subscription_eur": VALUE,
        "assumptions": "C and V are ASSUMED figures from src.cost, not bank data; euro totals are linear in V",
        "inputs": list(INPUT_FIELDS),
        "forbidden_inputs": list(FORBIDDEN_INPUTS),
        "derived": {
            "previous_capped": "min(previous, 2)",
            "previously_contacted": "previous > 0 (D21)",
            "balance_arsinh": "arcsinh(balance)",
            "campaign_bucket": "features.CAMPAIGN_EDGES / LABELS",
            "recency_band": "features.RECENCY_EDGES / LABELS, 'never' when previous == 0",
        },
        "default_client": profile["defaults"],
        "default_score": default_result["probability"],
        "default_contributions": default_result["contributions"],
        "examples": {
            cell: {
                "row": examples[cell]["row"],
                "label": examples[cell]["label"],
                "score": examples[cell]["score"],
                "clients_in_cell": examples[cell]["clients_in_cell"],
            }
            for cell in EXEMPLAR_CELLS
        },
        "checks": checks,
        "checks_passed": sum(1 for check in checks if check["passed"]),
        "checks_total": len(checks),
        "checks_ok": all(check["passed"] for check in checks),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    RECORD_PATH.write_text(json.dumps(record, indent=2, default=float) + "\n", encoding="utf-8")
    return record


def print_report(result: dict) -> None:
    print("DepositIQ T43 - the interactive demo")
    print("=" * 112)
    print(f"artifact : {result['artifact']['path']} sha256 {result['artifact']['sha256'][:16]}..., "
          f"{result['artifact']['bytes']:,} bytes")
    print(f"price    : {result['price']:.4f} = C/V = {result['call_cost_eur']:.2f} / "
          f"{result['value_per_subscription_eur']:.2f} (assumptions, not bank data)")
    print(f"inputs   : {len(result['inputs'])} form fields -> 12 shipped features; "
          f"forbidden: {', '.join(result['forbidden_inputs'])}")
    print(f"default  : P(subscribe) = {result['default_score']:.4f}; "
          f"call = {should_call(result['default_score'])}")
    print()
    print("top drivers for the default client (log-odds, signed sum within each one-hot block):")
    for item in result["default_contributions"][:5]:
        print(f"  {item['feature']:<18} = {item['value']:<24} {item['contribution']:+.4f}")
    print()
    print("checks")
    for check in result["checks"]:
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['check']} - {check['detail']}")
    print("=" * 112)
    print(f"result: {'PASS - the demo scores the artifact and explains the score' if result['checks_ok'] else 'FAIL'}")
    print("Next  : T44 - the end-to-end rerun from a clean clone.")


if __name__ == "__main__":
    outcome = report()
    print_report(outcome)
    sys.exit(0 if outcome["checks_ok"] else 1)
