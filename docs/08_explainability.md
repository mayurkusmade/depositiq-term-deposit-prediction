# 8. Explainability — what the shipped model actually looks at (T37)

Every phase up to this one answers *how good the ranking is*. This one answers the question a branch
manager asks, which none of those numbers can: **what is the model looking at, and in which direction?**
The instrument is SHAP — per-client, per-feature contributions that add up exactly to the model's own
score — and because they add up, the statements below are arithmetic rather than interpretation.

`src/explain.py`, 38.8 s, **17/17** checks, `python -m src.explain` exits **0**. The figure is
`reports/figures/09_shap_summary.png`; the numbers are in `models/shap_global.json` and
`reports/shap_importance.csv`.

Three panels: the beeswarm over all 40 columns the estimator sees, the same values summed back into 12
features, and the comparison with T25's permutation ranking. The beeswarm draws **6,000 of 36,168**
clients (seed 42) because 36k overlapping dots is slower and no more legible; every number in the tables
uses the full 36,168.

## 8.1 The instrument, and the reference value it is measured from

SHAP explains `models/final_model.joblib` — sha256 **7a40752f75a264a0…**, **180,520** bytes — on the
**training half**, 36,168 rows × **40** columns over **12** features. Four things about that sentence are
deliberate and each has a check:

- **It is the artifact, not a refit.** A SHAP plot of a re-fitted or re-tuned model is a plot of a
  different model; the digest is pinned in `EXPECTED_SHA256_PREFIX` and checked first.
- **Training half only.** The held-out half has been scored twice (T32, T35) and D53/D54 make it
  report-only. T37 describes the model, so it uses the rows the model was fitted on and claims nothing
  about generalisation. A source scan confirms `src/explain.py` never loads the held-out half.
- **`duration` is absent** because it was never in the matrix (D05, §3–4, §9). The check lists the
  explained columns and finds no `duration` and no target.
- **Units are log-odds**, not percentage points: a contribution of +0.30 is a 0.30 shift in log-odds of
  subscribing, relative to the model's reference value.

The reference value needs its own paragraph, because the obvious description of it is wrong. It is
**−2.04695** (= 11.4361% in probability). It is **not** the average prediction:

| quantity | log-odds | probability |
|---|---|---|
| SHAP reference (the booster's bias term, = `pred_contribs` bias to 0.0) | **−2.04695** | **11.4361%** |
| mean margin over the 36,168 rows | −2.49678 | 7.6084% |
| mean of the predicted probabilities | — | **11.6990%** |
| median predicted probability | — | 6.2522% |
| training base rate | — | 11.6982% |

So the **average feature contribution is −0.44983, not zero**, and a document that called −2.04695 "the
average log-odds" would be misdescribing the same model by 0.45. Two separate reasons: the bias term is
not the mean output, and the mean of a convex function is not the function of the mean. There is also a
number a reader may try to recompute: `base_score` is **0.116981864** — the training base rate to float32
precision, since the float64 mean is 0.116981862 — and its logit is **−2.0213**, while the reference is
−2.04695, i.e. **0.0256 lower**. Both implementations
(`shap.TreeExplainer` and xgboost's own `pred_contribs`) report the same −2.04695, so that difference is
in the booster's bookkeeping, not in this module, and it is recorded rather than smoothed over.

Additivity holds on every row: `max |sum(SHAP) + reference − margin| = **5.010e-06**` over 36,168 rows,
and `sigmoid(margin)` reproduces `pipeline.predict_proba` on the twelve raw columns to **7.521e-08**. The
5e-06 is float32 in the margins, which is why the tolerance is 1e-4 rather than exact.

## 8.2 Forty columns, twelve features, and why the aggregation rule matters

Twelve features become forty columns because each categorical is one-hot expanded. Per-column attribution
answers "what did the level `month=may` do", not "what did `month` do" — and the brief asks for the
second. The rule is to **sum the signed contributions within a feature's block**, which keeps additivity
exact (checked: 5.010e-06 at feature level too) and never double counts.

The obvious alternative — sum the *absolute* values of the one-hot columns — is wrong, and it is wrong in
the opposite direction to the usual warning. Because a client is `month=may` **and not** `month=jun`, both
columns carry a nonzero contribution and both get counted:

| feature | columns | mean \|summed\| | mean sum \|per column\| | inflation |
|---|---|---|---|---|
| `month` | 12 | 0.35829 | 0.56795 | **+58.52%** |
| `campaign_bucket` | 6 | 0.11133 | 0.16658 | **+49.63%** |
| `recency_band` | 5 | 0.02979 | 0.04401 | **+47.74%** |
| `poutcome` | 4 | 0.16278 | 0.22123 | **+35.91%** |
| `marital` | 3 | 0.15795 | 0.17126 | +8.42% |
| `contact` | 3 | 0.45197 | 0.47024 | +4.04% |
| `housing` | 2 | 0.16427 | 0.16455 | +0.17% |
| `day`, `balance_arsinh`, `age`, `previous_capped`, `previously_contacted` | 1 each | — | — | 0.00% |

The cost of leaving the columns separate is not the inflation, though — it is **fragmentation**. The
largest single `month` column is `month=jun` at 0.09687, which ranks **9th of 40**; a reader of per-level
bars would conclude `month` is a mid-field feature and miss that it is the second-most important thing in
the model. That is the disagreement T25 §8.2 flagged between one-hot fragmentation and feature-level
ranking, measured here instead of asserted.

## 8.3 The twelve features

Importance = `mean over 36,168 rows of |the feature's summed contribution|`, share = that divided by the
total. `T25 rank` is the permutation rank within the same twelve, for §8.5.

| # | feature | columns | mean \|SHAP\| | share | T25 rank |
|---|---|---|---|---|---|
| 1 | **`contact`** | 3 | 0.45197 | **22.14%** | 2 |
| 2 | **`month`** | 12 | 0.35829 | **17.55%** | 1 |
| 3 | `day` | 1 | 0.21641 | 10.60% | 4 |
| 4 | `balance_arsinh` | 1 | 0.17328 | 8.49% | 9 |
| 5 | `housing` | 2 | 0.16427 | 8.05% | 6 |
| 6 | `poutcome` | 4 | 0.16278 | 7.97% | 3 |
| 7 | `marital` | 3 | 0.15795 | 7.74% | 8 |
| 8 | `age` | 1 | 0.11449 | 5.61% | 5 |
| 9 | `campaign_bucket` | 6 | 0.11133 | 5.45% | 10 |
| 10 | `previous_capped` | 1 | 0.09350 | 4.58% | 12 |
| 11 | `recency_band` | 5 | 0.02979 | 1.46% | 7 |
| 12 | `previously_contacted` | 1 | 0.00724 | 0.35% | 11 |

## 8.4 In words a branch manager can use

The share column is what each feature is worth *across the whole list*. Each paragraph below pairs it with
the level-level contribution (how much that level moves one client's log-odds) and the measured
subscription rate — so every sentence can be checked against the data rather than taken on trust.

**`contact` — 22.14%, the largest single driver, and 28.72% of it is a missing value.** A client whose
recorded channel is `unknown` gets **−1.0150** log-odds, and those 10,386 clients subscribe at **4.11%**
against **14.83%** for the 23,465 on `cellular` (**+0.2318**). This is the model's biggest lever and it is
not about the client at all: it is a field the agent left blank. Two facts stop it being a simple win.
`unknown` is *not* a channel — it means "the system does not know", so the honest fix is process (record
the channel), not targeting. And the model can only use the value it is given: 28.72% of the file arrives
with the worst-scoring value, which is why scoring these clients at 4.11% is a statement about data
quality dressed as a prediction. `telephone` sits between the two (**−0.1443**, 13.94% on 2,317 clients),
so the channel effect is mostly `unknown` versus everything else.

**`month` — 17.55%, and it is a property of the campaign calendar, not of the client.** `mar`
(**+1.3736**, 50.90% of 391 clients), `oct` (**+1.0379**, 43.75%), `sep` (**+0.9792**, 45.23%), `dec`
(**+0.9068**, 47.19%) are the strong months, and `may` — **30.6% of the whole training file, 11,062
clients** — is negative (**−0.2999**, 6.68% conversion). The model is partly saying *when to call*. That
is usable as scheduling, and it is **not** evidence that calling in March causes subscriptions: these are
2008–2010 months, confounded with whatever else was happening to the bank and the economy, which this
dataset variant cannot control for (the additional variant has the macro columns; D19/D20). Read it as
"March campaigns converted, and the model learned that", never as "run the campaign in March".

**`day` — 10.60%, and it points nowhere on its own.** Bottom fifth of the month (≤ 7) contributes
**−0.1257**, top fifth (≥ 24) **−0.1230** — both negative, and Spearman(value, contribution) is
**−0.060**. There is no "best day" here. T25 measured the same thing from the other direction (permutation
rank 4 on a feature with a near-zero marginal), and D35 established why: `day`'s marginal band difference
**reverses inside `may`**, so its value is in the interaction with `month` — a scheduling coordinate, not a
client attribute. The actionable statement is a negative one, and worth saying plainly: **do not build a
"call on the 5th" rule from this**, because the average hides the sign flip.

**`balance_arsinh` — 8.49%, the one strong financial signal, and it is monotone.** Bottom fifth (≤ 3.83)
contributes **−0.3242**, top fifth (≥ 8.22) **+0.2020**, a **0.526** log-odds spread, Spearman **+0.773**.
Clients with money subscribe more, smoothly, with no cliff — which is what a savings product should look
like, and the only feature where a single sentence about direction is the whole story.

**`housing` — 8.05%, a competing commitment.** A mortgage is **−0.1678** (7.71% for 20,182 clients)
against **+0.1584** for no mortgage (16.73% for 15,986): a 9pp gap. A housing loan is a monthly obligation
that competes with a deposit instalment, and the model reads it that way.

**`poutcome` — 7.97%, the most decisive input in the model that almost nobody has.** `success` contributes
**+1.6841** — the largest single-level effect anywhere in the table — on **1,205 clients (3.33%)**, who
subscribe at **64.65%**. But `unknown` covers **29,589 clients (81.81%)** at −0.0909, and `failure`
(−0.2719, 12.29%) is barely below `unknown`. So the feature that best separates a subscriber from a
non-subscriber is available for one client in thirty. This is the clearest illustration of why §8.3's
share column must not be read as "how much this matters": mean |SHAP| averages over clients, so a feature
that decides everything for 3% of the list ranks below one that nudges everybody.

**And the one that is nearly redundant.** `previously_contacted` is a stored flag the project derived
itself (D21) and it scores **0.35%** — because `poutcome`, `recency_band` and `previous_capped` already
encode the same history. It is not that the flag is wrong; it is that the model has three other ways to
know it.

## 8.5 The two rankings disagree — and neither is corrected by the other

SHAP on the shipped model versus T25's permutation test on a 200-tree forest: **Spearman +0.755**, four
gaps of 3+ places. Different instruments (log-odds attributed vs ROC-AUC lost when shuffled), different
units, different questions. Agreement would be a coincidence.

| feature | SHAP rank | T25 rank | gap | why they differ |
|---|---|---|---|---|
| `balance_arsinh` | **4** | 9 | **−5** | SHAP credits information *used*; permutation measures information *unique*. T25 §8.2: the forest can reach the same signal through `age` and `day`, so shuffling `balance` costs only **0.0039** AUC even though it carries **0.1733** log-odds of credit. Notably MI ranked it **3rd** — SHAP sides with the marginal view here |
| `recency_band` | 11 | 7 | +4 | 5 one-hot columns on top of a history `poutcome`/`previous_capped` already carry; the tree's splits use it late and rarely, the forest's shuffles still lose AUC |
| `poutcome` | 6 | 3 | +3 | the §8.4 dilution: permutation grades the 3.33% it decides, SHAP averages in the 81.81% it cannot help |
| `age` | 8 | 5 | +3 | permutation's third-strongest client attribute (0.01000 AUC); SHAP gives it 5.61%, because the credit is partly shared with `marital` and `balance` |

The remaining eight features move by at most two places, which is ordering noise over twelve items, not a
finding. Nothing is averaged: an average of two ranks would be a third number belonging to neither method,
and would hide exactly the mechanism above.

## 8.6 The sentence this analysis supports

Grouping the twelve by *what they describe* — a reading, declared in `CAMPAIGN_SIDE`, not a measurement —
gives the headline:

**Campaign-side features — `contact`, `month`, `day`, `campaign_bucket` — carry 55.7% of total importance.
The client's own attributes carry 44.3%.**

The shipped model is, by this measure, **more a model of how the bank calls than of who the client is**.
Three consequences, in order of how much they matter:

1. **It supports the project's scoped claim, and only that claim.** D18: the model converts a *fixed call
   budget* better; it does not show that targeting reduces spend. A model whose importance is dominated by
   delivery variables is exactly a model for ordering a list — it is not a model of client demand.
2. **The largest single improvement available is a data-quality one.** `contact=unknown` is 10.96% of
   total importance by itself, the top column of forty, and it is a blank field on 28.72% of the file.
   That is cheaper than any modelling change in this project.
3. **Among client attributes the ranking is stable and sensible:** `balance_arsinh` (8.49%), `housing`
   (8.05%), `poutcome` (7.97%), `marital`, `age`, then the campaign-history counts. If the bank wanted one
   sentence about who subscribes, `balance` up and `housing` down is it.

## 8.7 What this does not claim

- **Not causal.** SHAP attributes the model's own computation. A feature can earn credit for being
  correlated with the real driver; `month` is the clearest case.
- **No generalisation claim.** Everything here is the training half (D53/D54). T32 already scored the
  held-out half twice; T37 adds no third reading.
- **The split among correlated features is arbitrary but consistent.** `contact`, `month` and `day` are
  intercorrelated (they describe the same campaign schedule), as are `poutcome`, `recency_band`,
  `previous_capped` and `previously_contacted`. Which one gets the credit is decided by the fitted trees,
  not by the world. This is precisely why §8.5's disagreements are reported instead of resolved.
- **`month`'s effect is confounded** with 2008–2010 conditions this variant cannot see, and `may` alone is
  30.6% of the rows.
- **Importance is not per-client influence.** A 0.35% feature can decide an individual call; a 22% feature
  can be irrelevant to a given client. T38 is the step that reads single clients.
- **The 6,000-dot sample** is a plotting decision; no table here uses it.

## 8.8 Verification

`python -m src.explain` → exit **0**, **17/17** checks, **38.8 s**. The checks that would catch a wrong
analysis rather than a crash:

| check | result |
|---|---|
| explained model is T31's artifact | sha256 `7a40752f75a264a0…`, 180,520 bytes |
| two-step XGBoost, 300 rounds, 40 columns | `['preprocess', 'model']`, `XGBClassifier`, 300 |
| `duration` and target absent | no stray column among the 40 |
| 12 features partition 40 columns | 1+1+1+1+3+2+3+12+4+6+5+1 |
| the twelve are T25's kept set | `select.shipped_features()` exact match |
| SHAP adds up to the margin | **5.010e-06** max error, 36,168 rows |
| margin is the served probability | **7.521e-08** vs `predict_proba` on raw columns |
| SHAP matches `pred_contribs` | **0.0** on both values and bias |
| feature-level additivity survives aggregation | **5.010e-06** |
| aggregation never inflates | holds for all 12; largest +58.52% (`month`) |
| shares sum to one | 1.000000000000000 |
| both rankings are permutations of 1..12 | yes |
| T25's pin covers the twelve | 12 of 12, no duplicates |
| per-column table covers the matrix once | 40 rows, shares sum to 1 |
| reference is the bias, not the mean output | −2.04695 vs −2.49678 → average contribution −0.44983 |
| held-out half never loaded | 0 references (needle assembled at run time) |
| figure written | 445 KB |

**Two defects were found by running it, both mine.** (1) The probability check compared
`sigmoid(margin)` against `pipeline.predict_proba(X)` with `X` the *transformed* matrix, and the pipeline
raised `columns are missing: {'marital', 'poutcome', ...}` — the check was measuring the model against a
shape it never sees; it now goes through the twelve raw columns, which is the interface that serves.
(2) `mean_probability` was computed as `1/(1+exp(-margin)).mean()`, which divides by the mean instead of
averaging the probabilities, and printed **4.96%** where the true mean prediction is **11.6990%** — the
precedence bug was caught only because 4.96% contradicted T31's recorded mean of 0.1166. The reference
value −2.04695 was then checked against `base_score` instead of being described from memory, which is how
§8.1's 0.0256 discrepancy became a recorded fact rather than a wrong sentence.

Recorded as **D58** in `PROGRESS.md`. Next: **T38** — local explanations for one true positive, one false
positive and one false negative, on the same artifact.
