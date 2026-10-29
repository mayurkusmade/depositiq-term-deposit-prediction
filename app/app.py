"""
DepositIQ - the interactive demo (T43).

Run it from the repository root:

    streamlit run app/app.py

The app is deliberately thin: every number it shows comes from `src/demo.py`, which is the same module
`python -m src.demo` checks without a browser. The form asks for the eleven business facts the shipped
model needs, derives the other features exactly as `src.features.engineer` does, scores
`models/final_model.joblib`, and shows the twelve SHAP contributions behind the score.

Three things the interface refuses to do:

* **Ask for `duration`.** It is recorded after the call ends and is never an input (D05/T22). The form
  has no field for it, and `src/demo.validate()` checks that mechanically.
* **Treat 0.5 as the cut.** The decision boundary is the price, `C/V = 0.0750` (T34), so a client at
  0.5 is called. The app says so rather than letting a reader assume a coin flip.
* **Present the euro figures as bank data.** `C = 1.50 EUR` and `V = 20.00 EUR` are assumptions from
  `src/cost.py`, and the app labels them that way beside every euro total.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# `streamlit run app/app.py` puts `app/` on the path, not the repository root. Adding the root here is
# what makes `from src import ...` work from any clone location, the same path-independence `src/config`
# provides for the pipeline.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import cost, demo  # noqa: E402 - the path insert has to happen first

st.set_page_config(
    page_title="DepositIQ - which clients to call today",
    page_icon="📞",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def model_bundle() -> dict:
    return demo.load_model()


@st.cache_data(show_spinner=False)
def training_profile() -> dict:
    return demo.training_profile()


@st.cache_data(show_spinner=False)
def exemplars() -> dict:
    return demo.example_clients()


BUNDLE = model_bundle()
PROFILE = training_profile()
EXAMPLES = exemplars()
CARD = BUNDLE["card"]

EXAMPLE_LABELS = {
    "Default client (training medians)": None,
    "True positive - called, subscribed (T38)": "true_positive",
    "False positive - called, did not subscribe (T38)": "false_positive",
    "False negative - not called, would have subscribed (T38)": "false_negative",
}


def load_example() -> None:
    """Copy an exemplar's values into the widget state before the widgets are created."""
    choice = st.session_state.get("example_choice", next(iter(EXAMPLE_LABELS)))
    cell = EXAMPLE_LABELS[choice]
    values = EXAMPLES[cell]["values"] if cell else PROFILE["defaults"]
    for field in demo.INPUT_FIELDS:
        value = values.get(field)
        if field == "days_since_contact" and value is None:
            # The widget only renders when previous > 0; a numeric placeholder keeps its state valid.
            value = 30
        st.session_state[f"in_{field}"] = value


# Seed every widget from the median/mode client. `setdefault` means a user's edit survives a rerun.
for field, value in PROFILE["defaults"].items():
    if field == "days_since_contact" and value is None:
        value = 30
    st.session_state.setdefault(f"in_{field}", value)

with st.sidebar:
    st.header("Client attributes")
    st.caption(
        "The eleven business facts the model needs. `duration` is not here: it is only known after the "
        "call ends. `job`, `education`, `default` and `loan` are not here because T25 dropped them."
    )
    st.selectbox(
        "Start from an example",
        list(EXAMPLE_LABELS),
        key="example_choice",
        on_change=load_example,
    )
    st.divider()

    age = st.number_input("Age", min_value=18, max_value=100, step=1, key="in_age")
    day = st.number_input("Day of month", min_value=1, max_value=31, step=1, key="in_day")
    balance = st.number_input(
        "Balance (EUR)", min_value=-20_000, max_value=200_000, step=100, key="in_balance"
    )
    campaign = st.number_input(
        "Calls this campaign", min_value=1, max_value=100, step=1, key="in_campaign"
    )
    previous = st.number_input(
        "Calls before this campaign", min_value=0, max_value=300, step=1, key="in_previous"
    )
    if previous > 0:
        days = st.number_input(
            "Days since last contact", min_value=0, max_value=1_000, step=1,
            key="in_days_since_contact",
        )
    else:
        days = None
        st.caption("No prior contact: the model sees `recency_band = never`, not a zero-day recency.")

    marital = st.selectbox("Marital status", PROFILE["options"]["marital"], key="in_marital")
    housing = st.selectbox("Housing loan", PROFILE["options"]["housing"], key="in_housing")
    contact = st.selectbox("Contact channel", PROFILE["options"]["contact"], key="in_contact")
    month = st.selectbox("Last contact month", PROFILE["options"]["month"], key="in_month")
    poutcome = st.selectbox("Previous outcome", PROFILE["options"]["poutcome"], key="in_poutcome")

values = {
    "age": int(age),
    "day": int(day),
    "balance": float(balance),
    "campaign": int(campaign),
    "previous": int(previous),
    "days_since_contact": None if days is None else float(days),
    "marital": marital,
    "housing": housing,
    "contact": contact,
    "month": month,
    "poutcome": poutcome,
}

st.title("DepositIQ - which clients to call today")
st.markdown(
    "A bank has fewer agents than clients. This model ranks clients by **P(term-deposit subscription)** "
    "so a fixed call budget reaches more subscribers. It **converts a fixed call budget** better; it is "
    "**not** evidence that the bank should cut campaign spend."
)
st.caption(
    f"Scoring `models/final_model.joblib` ({BUNDLE['bytes']:,} bytes, sha256 "
    f"`{BUNDLE['digest'][:16]}...`). The cut is **{demo.PRICE:.2%} = C/V**, not 0.5."
)

try:
    client = demo.client_frame(values)
    result = demo.explain_client(BUNDLE["pipeline"], client)
except Exception as error:  # noqa: BLE001 - a form error is shown, not raised into the UI
    st.error(f"The demo could not score this client: {type(error).__name__}: {error}")
    st.stop()

probability = result["probability"]
value = demo.expected_value(probability)

left, right = st.columns([1, 1])
with left:
    st.metric("P(subscribe)", f"{probability:.2%}")
    st.metric(
        "Expected value per call",
        f"{value:+.2f} EUR",
        help="EV = p x V - C, with C and V the assumed figures from src/cost.py",
    )
with right:
    if demo.should_call(probability):
        st.success(
            f"**Call this client.** {probability:.2%} is above the {demo.PRICE:.2%} price; "
            f"the call is worth {value:+.2f} EUR on average."
        )
    else:
        st.info(
            f"**Do not call.** {probability:.2%} is below the {demo.PRICE:.2%} price; "
            f"the call is worth {value:+.2f} EUR on average."
        )
    st.caption(
        f"Break-even price p* = C/V = {demo.CALL_COST:.2f} / {demo.VALUE:.2f} = {demo.PRICE:.2%}. "
        "C and V are assumptions, not bank figures."
    )

st.progress(
    min(max(probability, 0.0), 1.0),
    text=f"P(subscribe) = {probability:.2%}   |   price = {demo.PRICE:.2%}",
)

st.subheader("Why this score")
st.caption(
    "SHAP contributions in log-odds, aggregated by signed sum within each one-hot block (D58). "
    "Positive raises the score, negative lowers it. These are not causal effects."
)

top = pd.DataFrame(result["contributions"][:5])[["feature", "value", "contribution"]].copy()
top["contribution"] = top["contribution"].map(lambda number: f"{number:+.3f}")
st.markdown("**Top five drivers**")
st.dataframe(top, hide_index=True, width="stretch")

chart = (
    pd.DataFrame(result["contributions"])
    .set_index("feature")["contribution"]
    .sort_values()
)
st.bar_chart(chart, height=320)

with st.expander("All twelve contributions, and the arithmetic"):
    every = pd.DataFrame(result["contributions"])[["feature", "value", "contribution"]].copy()
    every["contribution"] = every["contribution"].map(lambda number: f"{number:+.3f}")
    st.dataframe(every, hide_index=True, width="stretch")
    st.caption(
        f"Reference {result['base']:+.4f} + twelve contributions {result['total']:+.4f} = "
        f"margin {result['margin']:+.4f}; sigmoid(margin) = {result['shap_probability']:.4%} against "
        f"the served probability {probability:.4%}."
    )

with st.expander("Model, data and the three prohibitions"):
    st.markdown(
        f"- **Artifact**: `models/final_model.joblib`, {BUNDLE['bytes']:,} bytes, sha256 "
        f"`{BUNDLE['digest'][:16]}...`; loaded, never re-fitted."
    )
    st.markdown(
        f"- **Cross-validated PR-AUC**: {CARD['cross_validation']['pr_auc']['mean']:.4f} +/- "
        f"{CARD['cross_validation']['pr_auc']['sd']:.4f} on the training half; base rate "
        f"{CARD['base_rate']:.4%}."
    )
    st.markdown(
        f"- **Trained on**: {CARD['training_rows']:,} rows, {CARD['positives']:,} subscribers. "
        "The held-out half is not loaded by this demo."
    )
    st.markdown(
        "- **`duration` is never an input**: it is recorded after the call ends. **Accuracy is not a "
        "headline**: always answering `no` already scores about 88%. **`month` is not a cause**: it "
        "schedules work, and it spans a recession."
    )
    st.markdown(
        "- **The score is a probability against a price, not a verdict**: 0.5 is far above the "
        f"{demo.PRICE:.2%} cut and is not used as a decision rule."
    )

with st.expander("What if the assumed prices are wrong?"):
    st.dataframe(
        cost.sensitivity_table().set_index("call_cost_eur"),
        width="stretch",
    )
    st.caption(
        "Each cell is p* = C/V. The demo uses 1.50 / 20.00 = 7.50%. Every euro total is linear in V, "
        "so a different V moves the cut and the money together."
    )

st.divider()
st.caption(
    "T43 - DepositIQ - scores `models/final_model.joblib` - SHAP in log-odds - "
    "not a causal explanation, and not a bank's own cost figures."
)
