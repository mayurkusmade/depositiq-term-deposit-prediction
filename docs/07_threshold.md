# DepositIQ — The Operating Threshold

**Rubric:** Criterion 7 — Evaluation & Business Translation (15 marks)
**Task:** T34 | **Status:** complete
**Date:** 2026-10-30
**Source of every number here:** the run of `python -m src.threshold`, recorded in
`models/threshold_choice.json` and drawn in `reports/figures/07_threshold_curve.png`. Nothing below was
computed by hand (D06).

---

## 1. What was decided, and what was not

A ranking is not a campaign. T33 ended with a probability per client and three candidate cut-offs, none of
which had been chosen for the reason a bank actually chooses one: **money**.

This step decides one number — the threshold at which a client gets called — and it decides it from
`src/cost.py`, the cost model written in **T09** and never changed since:

| symbol | value | derivation / status in `cost.py` |
|---|---|---|
| `AGENT_COST_PER_HOUR_EUR` | 18.00 EUR per hour | ASSUMED: wage + employer costs + supervision + telephony |
| `MINUTES_PER_CALL` | 5.0 minutes | ASSUMED, corroborated by the data's mean talk time of 258 s |
| `CALL_COST_EUR` **C** | **1.50 EUR** per call | DERIVED: `18.00 x 5.0 / 60` |
| `AVERAGE_DEPOSIT_EUR` | 2,000 EUR | ASSUMED: a term deposit, not the 448 EUR median `balance` |
| `NET_INTEREST_MARGIN_ANNUAL` | 1.0% per year | ASSUMED |
| `TERM_MONTHS` | 12 | ASSUMED |
| `VALUE_PER_SUBSCRIPTION_EUR` **V** | **20.00 EUR** | DERIVED: `2,000 x 1.0% x 12/12` |

T31 persisted the artifact with **no threshold baked in** precisely so that this decision would be made
here rather than arriving frozen inside a pickle (D52).

The rule is one line, and it is not a hyperparameter:

> **Call a client when `P(subscribe) > C / V = 1.50 / 20.00 = 0.0750`.**

Adopted: **0.0750**, imported as `cost.break_even_threshold()` rather than typed, so that if an assumption
moves the decision moves with it. A check requires the recorded threshold to equal `C/V` exactly, which
means the doc's claim "the threshold is a price" is true by construction and not by memory.

**What was deliberately not done:** no search over thresholds scored by F1, accuracy or PR-AUC. Those were
T29–T32's questions and their answers are already on record; re-asking them here would replace a priced
decision with a preferred one. The empirical sweep below is run **only** to measure how flat the optimum
is, and its argmax is reported as a diagnostic, not adopted.

## 2. Where the choice was made

On the **training half only**, from **out-of-fold** scores: five folds over **36,168 rows / 4,231
subscribers (11.6982%)**, each row scored by a model that did not see it. The model is T30's tuned
configuration (`max_depth=5`, `learning_rate=0.08`, `n_estimators=300`, `min_child_weight=10`,
`subsample=1.0`, `colsample_bytree=0.6`, unweighted), rebuilt from `models/best_params.json` through
`src.tune.make_estimator`.

`choose()` takes the training arrays and nothing else — the signature is the guarantee. The held-out half
is scored in a separate function, called afterwards, and its optimum is recorded so a reader can see it was
**not** the adopted answer:

* training folds would have chosen **0.0821**;
* the held-out rows would have chosen **0.0909**;
* the adopted cut is **0.0750**, and it was fixed before either empirical peak existed.

That these five folds are the project's folds is checked rather than asserted: the per-fold PR-AUCs
reproduce T30's tuned arm to **0.0e+00** (`0.4435, 0.4378, 0.4339, 0.4588, 0.4561`). A second fold set
would have made this section a parallel experiment instead of a continuation of the same one (D45).

## 3. Money at five cuts

Training half, out-of-fold. Every row is `cost.expected_campaign_value` — revenue = hits x 20.00 EUR,
cost = calls x 1.50 EUR, net = revenue - cost.

| cut | called | reached | missed | net EUR | ROI | EUR per captured subscription |
|---|---|---|---|---|---|---|
| call everyone (0.0000) | 36,168 | 4,231 | 0 | 30,368 | 56.0% | 12.82 |
| **0.0750 — adopted (C/V)** | **14,866** | **3,296** | **935** | **43,621** | **195.6%** | **6.77** |
| 0.0821 — empirical peak | 13,247 | 3,194 | 1,037 | 44,010 | 221.5% | 6.22 |
| 0.2103 — T32's F1 cut | 4,806 | 2,205 | 2,026 | 36,891 | 511.7% | 3.27 |
| 0.5000 — convention | 1,688 | 1,039 | 3,192 | 18,248 | 720.7% | 2.44 |

**The finding is the first two rows against the last two.** At 0.075 the campaign calls **41.1%** of the
list and reaches **77.9%** of all subscribers for **43,621 EUR** of net value. At T32's F1 cut it nets
**36,891 EUR**, at 0.5 **18,248 EUR** — and calling *everyone* nets **30,368 EUR**, which is less than the
priced cut because the marginal client below 0.075 loses money (each one is worth `p x 20.00 < 1.50`).

So the cut that maximises F1 leaves **6,730 EUR** on the table relative to the priced one, and the cut that
maximises accuracy leaves **25,373 EUR**. Neither is a criticism of T29–T32: F1 and accuracy were answered
for the question they were asked. They are simply not prices, and a bank cannot spend an F1.

## 4. The optimum is flat, and the price sits on its aggressive edge

The empirical peak is **0.0821** at **44,009.50 EUR**. The adopted cut earns **43,621.00 EUR** — a shortfall
of **388.50 EUR**, or **0.883%**. Every threshold within 1% of the peak (**43,569.41 EUR**) spans:

* **0.0746 – 0.1016** in probability;
* **14,973 – 9,970 calls**, i.e. the campaign can call anywhere between **27.6% and 41.4%** of the list and
  lose at most 1% of the money;
* **4.96%** of the swept grid.

The curve is **asymmetric**: it climbs steeply from 0.00 to the peak and falls shallowly towards 0.5,
because the score density is thick at the bottom (41% of the list is above 0.075) and thin at the top. That
is why both empirical argmaxes land *right* of the price — training at 0.0821, held-out at 0.0909 — and it
is also why neither is worth adopting: the argmax of a flat curve moves with the sample, and the price does
not.

Read the band's lower edge carefully. **0.0750 is 0.0004 above the bottom of the plateau.** Calling *more*
than the price starts losing money faster than calling fewer, so `C/V` is the most aggressive cut that is
still within 1% of the peak — which is what a break-even rule should be, and worth saying out loud rather
than letting a reader assume the price sits in the middle of the safe region.

![Money against the threshold, reliability, and the price under other assumptions](../reports/figures/07_threshold_curve.png)

The middle panel is why a threshold on this score can be read as a price at all: decile by decile the
held-out probabilities sit close to the observed rates, with the low deciles marginally **above** the
diagonal. Without that, 0.075 would be a coordinate, not a break-even.

## 5. When calls are capped

If agents, not budget, are the constraint, the answer is no longer `p*` — it is the N highest scores.

| calls made | threshold | reached | net EUR |
|---|---|---|---|
| top 5% (1,809) | 0.4842 | 1,098 | 19,246 |
| top 10% (3,617) | 0.2967 | 1,847 | 31,514 |
| top 20% (7,234) | 0.1322 | 2,613 | 41,409 |
| top 30% (10,851) | 0.0952 | 3,003 | 43,784 |
| top 40% (14,467) | 0.0767 | 3,265 | 43,600 |

Two readings, both worth having in a report. **A 5% campaign captures 26% of the subscribers for 44% of the
money** — if only a fraction of the list can be worked, most of the value is still reachable. And **the
unconstrained optimum sits near 30% of the list**, not 5%: the bank's own numbers say the campaign is worth
running at scale, and a report that quoted only the top-5% figure would be advising a cheaper campaign than
the model justifies.

Note the 30% row nets **43,784 EUR**, *above* the adopted 43,621 EUR — it is a coarser grid on a flat curve
(0.0952 is inside the plateau, 10,851 calls against 14,866). This is the plateau again, not a better rule:
the same 1% band contains both.

## 6. The check on rows no model had seen

Applied to the persisted artifact on the **9,043 held-out rows / 1,058 subscribers (11.6997%)**, once, after
the choice was fixed:

| cut | called | reached | missed | net EUR | ROI |
|---|---|---|---|---|---|
| **0.0750 — adopted** | 3,621 | 821 | 237 | **10,988.50** | 202.3% |
| 0.2103 — F1 cut | 1,159 | 561 | 497 | 9,481.50 | 545.4% |
| 0.5000 — convention | 396 | 257 | 801 | 4,546.00 | 765.3% |
| call everyone | 9,043 | 1,058 | 0 | 7,595.50 | 56.0% |

The priced cut is the best of the four by **1,507 EUR** over the F1 cut and **3,393 EUR** over calling
everyone, on rows that had no part in choosing it. The held-out optimum (**0.0909**, 11,198.00 EUR, 2,868
calls) beats the adopted cut by **209.50 EUR (1.9%)** — just outside the training plateau's 1% — and is
**recorded, not adopted**: adopting it would be choosing the answer with the sample that T33 and T34 exist
to test.

The consistency between halves is the result: **77.6%** of held-out subscribers reached by **40.0%** of the
list, against **77.9%** by **41.1%** on training. The cut transferred.

## 7. Why ROI and net value disagree

The table above has 0.5 at **765% ROI** and the adopted cut at **202%**, and 0.5 loses. This is not a
contradiction, it is a definition:

* **ROI = net / spend.** It is maximised by spending almost nothing on the few clients who are near-certain —
  a ratio with no denominator worth having. At 0.5 the bank spends **594 EUR** and misses **801 of 1,058**
  subscribers.
* **net value = revenue - spend.** It is maximised at the price, because the objective is the campaign, not
  the efficiency of the campaign.

The two metrics agree on ordering only when the budget, not the list, is the constraint. Since section 5
shows the value is still rising at 30% of the list, the campaign is **capacity-limited, not
capital-limited**, so net value is the objective and ROI is a diagnostic. The same argument applies, in
milder form, to precision: 0.5 has the best precision of any cut in the project (0.649 on held-out) and the
worst campaign value.

## 8. The assumption, priced

`p* = C/V` is only as good as **C** and **V**, so the module records the whole grid rather than a
sensitivity disclaimer:

| C \ V | 10 | 20 | 40 | 60 |
|---|---|---|---|---|
| **0.75** | 0.075 | 0.037 | 0.019 | 0.013 |
| **1.00** | 0.100 | 0.050 | 0.025 | 0.017 |
| **1.50** | 0.150 | **0.075** | 0.037 | 0.025 |
| **2.00** | 0.200 | 0.100 | 0.050 | 0.033 |
| **3.00** | 0.300 | 0.150 | 0.075 | 0.050 |

Twenty cells, each exactly `C/V`, monotone in both directions (a dearer call raises the bar, a more valuable
subscription lowers it) — checked, not eyeballed. The adopted cell is bold. The range is **0.013 to 0.300**,
which is the honest scale of the uncertainty: if a call costs 3.00 EUR and a subscription is worth 10.00 EUR,
the bar quadruples to **0.30** and the campaign becomes the top-decile list of section 5.

## 9. What this does not prove

* **The euros are the model's, not the bank's.** `V` excludes cross-sell, retention and lifetime value, and
  the campaign excludes fixed costs, contact fatigue and the dataset's own `unknown` outcomes. Section 8
  gives the scale of that exposure; it does not remove it.
* **No confidence interval on the euro figures.** T33 bootstrapped PR-AUC because a *metric* needed an
  interval; a difference of 388 EUR between two points on a flat curve does not need one, because the
  decision is a price rather than the argmax. If the recommendation ever turns into "call N clients exactly",
  a bootstrap over the net-value curve is the next measurement.
* **The held-out half has now been used twice** (T33, T34). Every further decision taken on it is a third
  look; T35 and T36 may **report** on those rows and may not **choose** on them (D53's rule, restated for
  this step).
* **The threshold is for this campaign.** Re-pricing for a different product, term or margin is a new row in
  section 8, not a re-run of this task.

## 10. Verification

`python -m src.threshold` exits **0** with **15/15** checks in **4.6 s**, and writes a **77,751-byte**
record whose `curve` array carries 403 `(threshold, calls, reached, net)` points, so the left panel can be
redrawn from the record alone. The runtime and the record's size vary run to run; the values quoted here are
this run's, read from the file rather than remembered (the lesson of T33's superseded timing).

| check | what it establishes |
|---|---|
| the threshold is `cost.break_even_threshold()` = `C/V` | the decision is a price, not a preference |
| every training row scored exactly once out-of-fold | 36,168 rows, 36,168 predictions, 0 NaN |
| the fold PR-AUCs reproduce T30's tuned arm | same folds, same parameters, drift `0.0e+00` |
| net = revenue - cost and the call arithmetic, at all 403 points | every euro in the tables above is bookkeeping, drift `0.0e+00` |
| the recorded empirical optimum is the maximum of the curve | the diagnostic is the max, not a rounded guess |
| the price is inside the 1% plateau | 388.50 EUR short of the peak, band 0.0746-0.1016 |
| the cut outearns calling everyone | 43,621 vs 30,368 EUR on training |
| the cut outearns the F1 cut and 0.5 | 43,621 vs 36,891 vs 18,248 EUR on training |
| a captured subscription costs less than it is worth | 6.77 EUR against 20.00 EUR, ROI +195.6% |
| every capacity row calls the share it claims | each within 0.03% of its target |
| every sensitivity cell is `C/V` and monotone | 20 cells, both directions |
| the held-out optimum is not the adopted cut | 0.0909 recorded, 0.0750 adopted |
| the decision still earns money held out | 10,988.50 EUR, 1.215 EUR per client |
| the figure was written | 221,145 B, opened and inspected |
| the record on disk is this run's answers | read back and compared field by field |

Two things this step got wrong before the checks caught them, recorded because they are the kind of error a
reader would not have seen:

* **a record that could not rebuild its own figure.** The first version stored the plateau, the peak and the
  capacity table but not the **curve**, so the three euro figures in §3 for the rejected cuts existed only
  inside a check's detail string — quoteable, unverifiable. The record now carries **403**
  `(threshold, calls, reached, net)` points and a `comparisons` array of the five cuts, and a final check
  reads the file back and compares it field by field against the run.
* **a dead branch left in `held_out_check`** that reassigned the peak's threshold to itself. It was harmless
  and it was still removed — a line that cannot be read is a liability in a file whose job is to be
  audited.

**Next:** T35 lifts this ranking into a decile/gain table and turns the cut into calls saved; T36 runs the
with-vs-without `duration` comparison, where this threshold supplies the operating point both arms are
scored at.
