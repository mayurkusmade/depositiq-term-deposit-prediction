"""
Cost-of-errors model for the DepositIQ project.

Why this file exists
--------------------
Accuracy says "how many predictions were right". The bank cares about a different question:
**is making this call worth it?** That question has a numeric answer, and it depends on two
business facts the dataset does not contain:

    C  = the cost of making one outbound call        (a wasted call = a false positive)
    V  = the value the bank gains from one subscription (a missed subscriber = a false negative)

Given those, calling a client with predicted probability `p` has an expected net value of:

    EV(call) = p * (V - C) + (1 - p) * (-C)
             = p*V - p*C - C + p*C
             = p*V - C

and not calling them is worth 0. So the bank should call exactly when `p*V > C`, which gives the
**break-even probability threshold**:

    p* = C / V

Anyone with a predicted probability above `p*` is worth a call; below it, the call loses money on
average. This is why the project does not use the default 0.5 cut-off: 0.5 is a convention, not a
business decision. Task T34 applies this threshold to real model output.

Every euro figure below is an **assumption**, stated and labelled as such - they are not real
bank data. That is exactly why `sensitivity_table()` exists: so the conclusion can be checked
against a range of plausible costs instead of resting on one guessed number.

Run it from the repository root:

    python -m src.cost
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Assumptions - cost side
# ---------------------------------------------------------------------------
CURRENCY: str = "EUR"

# Fully-loaded hourly cost of a telemarketing agent: wage plus employer
# contributions, supervision and telephony overhead. ASSUMED.
AGENT_COST_PER_HOUR_EUR: float = 18.00

# Minutes of agent time per call, including dialling, talk time and call notes.
# ASSUMED at 5.0 minutes. Supporting evidence from this dataset: the mean `duration`
# (talk time alone) is 258 seconds = 4.3 minutes, so 5.0 minutes total is plausible
# and deliberately not optimistic.
MINUTES_PER_CALL: float = 5.0

# Cost of one call. Derived, not assumed.
CALL_COST_EUR: float = AGENT_COST_PER_HOUR_EUR * MINUTES_PER_CALL / 60.0

# ---------------------------------------------------------------------------
# Assumptions - value side
# ---------------------------------------------------------------------------
# Size of a typical term deposit. ASSUMED at EUR 2,000. For context, the median
# `balance` in this dataset is EUR 448 and the mean is EUR 1,362, i.e. the client's
# average yearly balance, which is not the same thing as the deposit they open.
AVERAGE_DEPOSIT_EUR: float = 2000.00

# Term length in months. ASSUMED at 12.
TERM_MONTHS: int = 12

# Net interest margin the bank keeps on the deposit, annual. ASSUMED at 1.0%: the bank
# pays the client a rate and lends the money on at a higher rate, keeping the difference.
NET_INTEREST_MARGIN_ANNUAL: float = 0.01

# Value of one captured subscription. Derived, not assumed.
# Note this is a deliberately conservative "direct margin" view: it counts only interest
# margin for one term. It excludes cross-selling, long-term customer value and the value
# of the deposit as stable funding, all of which would raise V and therefore lower p*.
VALUE_PER_SUBSCRIPTION_EUR: float = (
    AVERAGE_DEPOSIT_EUR * NET_INTEREST_MARGIN_ANNUAL * (TERM_MONTHS / 12.0)
)

# Machine-readable list of every assumption, so the report and the code cannot drift apart
# and so each assumption can be labelled with its status and provenance.
ASSUMPTIONS: list[dict[str, object]] = [
    {
        "name": "AGENT_COST_PER_HOUR_EUR",
        "value": AGENT_COST_PER_HOUR_EUR,
        "unit": f"{CURRENCY}/hour",
        "status": "ASSUMED",
        "note": "Fully-loaded agent cost: wage, employer contributions, supervision, telephony. "
                "No bank figures are public, so this is a stated assumption.",
    },
    {
        "name": "MINUTES_PER_CALL",
        "value": MINUTES_PER_CALL,
        "unit": "minutes",
        "status": "ASSUMED",
        "note": "Dialling + talk + notes. Corroborated by this dataset: mean talk time "
                "(`duration`) is 258 s = 4.3 min, so 5.0 min total is not optimistic.",
    },
    {
        "name": "CALL_COST_EUR",
        "value": CALL_COST_EUR,
        "unit": CURRENCY,
        "status": "DERIVED",
        "note": "= AGENT_COST_PER_HOUR_EUR * MINUTES_PER_CALL / 60",
    },
    {
        "name": "AVERAGE_DEPOSIT_EUR",
        "value": AVERAGE_DEPOSIT_EUR,
        "unit": CURRENCY,
        "status": "ASSUMED",
        "note": "Typical term deposit size. Not in the data: `balance` is the client's "
                "average yearly balance (median EUR 448), not the amount deposited.",
    },
    {
        "name": "TERM_MONTHS",
        "value": TERM_MONTHS,
        "unit": "months",
        "status": "ASSUMED",
        "note": "Standard 12-month term used for the comparison.",
    },
    {
        "name": "NET_INTEREST_MARGIN_ANNUAL",
        "value": NET_INTEREST_MARGIN_ANNUAL,
        "unit": "fraction per year",
        "status": "ASSUMED",
        "note": "Margin the bank keeps on the deposited funds.",
    },
    {
        "name": "VALUE_PER_SUBSCRIPTION_EUR",
        "value": VALUE_PER_SUBSCRIPTION_EUR,
        "unit": CURRENCY,
        "status": "DERIVED",
        "note": "= AVERAGE_DEPOSIT_EUR * NET_INTEREST_MARGIN_ANNUAL * (TERM_MONTHS / 12). "
                "Conservative: excludes cross-sell and long-term customer value.",
    },
]


# ---------------------------------------------------------------------------
# Core cost model
# ---------------------------------------------------------------------------
def break_even_threshold(
    call_cost: float = CALL_COST_EUR,
    value_per_subscription: float = VALUE_PER_SUBSCRIPTION_EUR,
) -> float:
    """
    The predicted probability at which a call exactly pays for itself: `p* = C / V`.

    Call a client only if their predicted probability is **above** this value. Expressed as a
    probability in [0, 1]. If `p* > 1`, no client is worth calling under those assumptions;
    if `p* <= 0`, the call costs nothing and everyone should be called.
    """
    if value_per_subscription <= 0:
        raise ValueError("value_per_subscription must be positive to be meaningfully compared.")
    return call_cost / value_per_subscription


def expected_net_value_per_call(
    probability: float,
    call_cost: float = CALL_COST_EUR,
    value_per_subscription: float = VALUE_PER_SUBSCRIPTION_EUR,
) -> float:
    """
    Expected net value of calling one client with the given predicted probability.

    `EV = p * V - C`. Positive means the call is worth making on average.
    """
    return probability * value_per_subscription - call_cost


def should_call(
    probability: float,
    call_cost: float = CALL_COST_EUR,
    value_per_subscription: float = VALUE_PER_SUBSCRIPTION_EUR,
) -> bool:
    """True when calling this client has positive expected net value (ties count as no)."""
    return expected_net_value_per_call(probability, call_cost, value_per_subscription) > 0


def cost_of_errors(
    call_cost: float = CALL_COST_EUR,
    value_per_subscription: float = VALUE_PER_SUBSCRIPTION_EUR,
) -> dict[str, float]:
    """
    The two mistakes this model can make, priced.

    - **False positive** (we call, they were never going to subscribe): we lose the call cost C.
    - **False negative** (we skip someone who would have subscribed): we forgo the subscription
      value but *save* the call, so the net loss is `V - C`, not `V`. Both cases are consistent
      with `EV = p*V - C` above, and the ratio is what justifies caring about recall.

    Also returns `recall_weight`, the ratio of a missed subscriber's net cost to a wasted call's
    cost. A value of 12 means missing a subscriber costs twelve times as much as a wasted call.
    """
    fp_cost = call_cost
    fn_cost_net = value_per_subscription - call_cost
    return {
        "false_positive_cost": fp_cost,               # wasted call
        "false_negative_cost": fn_cost_net,           # missed subscriber, net of the call saved
        "value_per_subscription": value_per_subscription,
        "call_cost": call_cost,
        "recall_weight": fn_cost_net / fp_cost if fp_cost else float("nan"),
    }


def expected_campaign_value(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    call_cost: float = CALL_COST_EUR,
    value_per_subscription: float = VALUE_PER_SUBSCRIPTION_EUR,
) -> dict[str, float]:
    """
    Value of a whole campaign that calls every client whose probability is >= `threshold`.

    Used in T34 to choose the threshold by money rather than by convention. Returns counts and
    euros for the ranked call list, including how many subscribers it reached, how many it
    missed, and how many calls were wasted.

    `y_true` must be 0/1 and `y_prob` the predicted probability of class 1.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.shape != y_prob.shape:
        raise ValueError(f"y_true {y_true.shape} and y_prob {y_prob.shape} must have the same shape.")

    selected = y_prob >= threshold      # the clients we decide to call
    called = int(selected.sum())
    reached = int(y_true[selected].sum())          # true positives: subscribers we called
    missed = int(y_true.sum()) - reached           # false negatives: subscribers we skipped
    wasted = called - reached                      # false positives: calls that converted nothing

    revenue = reached * value_per_subscription
    cost = called * call_cost
    net = revenue - cost

    return {
        "threshold": float(threshold),
        "clients_called": called,
        "subscribers_reached": reached,
        "subscribers_missed": missed,
        "wasted_calls": wasted,
        "revenue_eur": revenue,
        "call_cost_eur": cost,
        "net_value_eur": net,
        "roi": (net / cost) if cost else float("nan"),
        "cost_per_subscription_eur": (cost / reached) if reached else float("nan"),
    }


def best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    call_cost: float = CALL_COST_EUR,
    value_per_subscription: float = VALUE_PER_SUBSCRIPTION_EUR,
) -> dict[str, float]:
    """
    Pick the money-optimal cut-off by sweeping candidate thresholds and keeping the best net value.

    This is the empirical counterpart to the analytic `p* = C / V`: it should land near `p*` on
    well-calibrated probabilities, and diverging from it is a signal worth investigating.
    """
    candidates = np.unique(np.round(np.linspace(0.0, 1.0, 101), 4))
    results = [
        expected_campaign_value(y_true, y_prob, t, call_cost, value_per_subscription)
        for t in candidates
    ]
    best = max(results, key=lambda r: r["net_value_eur"])
    best["analytic_break_even"] = break_even_threshold(call_cost, value_per_subscription)
    return best


def sensitivity_table(
    call_costs: tuple[float, ...] = (0.75, 1.00, 1.50, 2.00, 3.00),
    values: tuple[float, ...] = (10.0, 20.0, 40.0, 60.0),
) -> pd.DataFrame:
    """
    Break-even threshold for each combination of call cost and subscription value.

    This is the honesty check on the assumed figures: if the conclusion flips across plausible
    inputs, it is an assumption, not a finding. Cells are probabilities (call if `p` is above).
    """
    rows = []
    for cost in call_costs:
        row: dict[str, float] = {"call_cost_eur": cost}
        for value in values:
            row[f"value_{value:g}"] = round(break_even_threshold(cost, value), 4)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # `python -m src.cost` - prints the full cost model so the report can quote real output.
    line = "=" * 74
    print(line)
    print("DepositIQ cost-of-errors model")
    print(line)

    print("\nASSUMPTIONS")
    print("-" * 74)
    for a in ASSUMPTIONS:
        val = a["value"]
        val = f"{val:,.2f}" if isinstance(val, float) and abs(val) >= 1 else f"{val:g}"
        print(f"  {a['name']:<30} {val:>10} {a['unit']:<16} [{a['status']}]")

    print("\nDERIVED")
    print("-" * 74)
    p_star = break_even_threshold()
    print(f"  Cost per call C                 : {CALL_COST_EUR:.2f} {CURRENCY}")
    print(f"  Value per subscription V        : {VALUE_PER_SUBSCRIPTION_EUR:.2f} {CURRENCY}")
    print(f"  BREAK-EVEN THRESHOLD  p* = C/V  : {p_star:.4f}  ({p_star:.2%})")
    print(f"  Base rate in the dataset        : 0.1170  (11.70%)")

    err = cost_of_errors()
    print("\nCOST OF ERRORS")
    print("-" * 74)
    print(f"  Wasted call (false positive)    : {err['false_positive_cost']:.2f} {CURRENCY}")
    print(f"  Missed subscriber (false neg.)  : {err['false_negative_cost']:.2f} {CURRENCY} (net of the call saved)")
    print(f"  Ratio missed : wasted           : {err['recall_weight']:.1f} : 1")

    print("\nFALSE-POSITIVE RATE CHECK (measured from the raw data)")
    print("-" * 74)
    p_zero = expected_net_value_per_call(0.0)
    p_one = expected_net_value_per_call(1.0)
    print(f"  EV of calling a client with p=0 : {p_zero:+.2f} {CURRENCY}")
    print(f"  EV of calling a client with p=1 : {p_one:+.2f} {CURRENCY}")
    print(f"  should_call(0.00)={should_call(0.0)}  should_call(p*)={should_call(p_star)}  should_call(1.0)={should_call(1.0)}")

    print("\nSENSITIVITY - break-even p* across plausible costs and values")
    print("-" * 74)
    print(sensitivity_table().to_string(index=False))
    print("\n  Reading it: values are the probability above which a call pays for itself.")
    print("  Compare each cell with the 11.70% base rate to see how aggressive targeting should be.")

    sys.exit(0)
