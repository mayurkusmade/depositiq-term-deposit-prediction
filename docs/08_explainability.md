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

Recorded as **D58** in `PROGRESS.md`.

---

# 9. Three clients, explained (T38)

Averages do not get called. This section takes one client from each of the three interesting cells of the
campaign — a true positive, a false positive and a false negative — and shows which features moved **that**
client's score, with the exact values the model saw. `src/local_explain.py`, **16/16** checks, **76.7 s**
(the run varies 50–80 s with machine load), `python -m src.local_explain` exits **0**. Figure: `reports/figures/10_shap_local.png`; numbers in
`models/shap_local.json` and `reports/shap_local.csv`.

## 9.1 The cells are defined against the price, not against 0.5

T34 adopted **0.0750** (call when P exceeds C/V), so "positive" here means *above the price*. A 0.5 rule
would relabel **12,867** clients — every one of them currently called — as negative, which is a campaign
nobody runs.

| cell | definition | in-sample (this step) | share of training half | T34's out-of-fold |
|---|---|---|---|---|
| true positive | y = 1, p ≥ 0.0750 | **3,523** | 9.74% | 3,296 |
| false positive | y = 0, p ≥ 0.0750 | **11,059** | 30.58% | 11,570 |
| false negative | y = 1, p < 0.0750 | **708** | 1.96% | 935 |
| true negative | y = 0, p < 0.0750 | 20,878 | 57.73% | — |

The counts differ from T34's **on purpose and by construction**: T34's table is built on *out-of-fold*
scores, while these cells use the **artifact's own in-sample scores** — the same scores and the same matrix
T37 explained, because the explanation must belong to the model on display. In sample the model clears the
price for 15,290 clients against T34's 14,866 and misses 708 subscribers against 935, because it has seen
these labels. Both numbers are recorded; neither replaces the other.

One caveat on the counts, measured rather than assumed: the same score has two routes — the pipeline's
`predict_proba` and SHAP's decomposition through the logistic link — and they disagree by at most
**4.49e-07**. **43** clients sit within 1e-4 of the cut, and exactly **one** of them changes cell depending
on the route. The cells above use the **served** probability, and the check that measures the flip is what
makes "to the client" an honest claim rather than an implied exactness.

## 9.2 The selection rule, fixed before any client was looked at

> **The exemplar of a cell is the client whose score is closest to that cell's median score; ties are
> broken by the lowest training row index.**

The rule reads `y` and `p` and nothing else — no SHAP value, no intuition about which story would read
well. It exists because the alternative is the trap this section is most exposed to: pick the most
confident false negative, explain it, and let a reader conclude that the 708 are like that one. The middle
of the cell is a typical member; a check re-derives the choice from the scores alone and fails if any other
client in the cell is strictly closer to the median. All three exemplars sit at the **50th percentile of
their own cell** by construction — typical *of the cell*, which is not the same as typical of the file.

## 9.3 True positive — training row 19,263, p = 0.3819, y = 1

**margin −0.4815 = reference −2.04695 + sum of twelve +1.5654.**

| feature | value | contribution | T37 global rank |
|---|---|---|---|
| **`month`** | **oct** | **+1.8824** | 2 |
| **`contact`** | **unknown** | **−0.6989** | 1 |
| `day` | 27 | +0.3295 | 3 |
| `campaign_bucket` | 1 | +0.1626 | 9 |
| `housing` | yes | +0.0740 | 5 |
| the other six | — | net −0.0792 | — |

One feature did the work: `month=oct` contributed **+1.8824**, nearly four times the margin the client
finished with. `day=27` and a first call in this campaign (`campaign_bucket=1`) added the rest.

The striking part is the second row. **`contact=unknown` — the model's single most important feature
overall — took 0.6989 of it back**, and the client still converted. That is the local/global distinction in
one row: the top-ranked feature is not the top contributor for every client, and a 22.14% share of global
importance is perfectly consistent with a large *negative* push here.

**In business language:** this client was reached in October, and the month carried them past the missing
channel. It is also the honest counter-example to §8.4's data-quality story — an `unknown` channel did not
make this client un-subscribable, it made them look worse on paper.

## 9.4 False positive — training row 23,374, p = 0.1133, y = 0

**margin −2.0578 = reference −2.04695 + sum of twelve −0.0109.**

| feature | value | contribution | T37 global rank |
|---|---|---|---|
| **`day`** | **2** | **−0.5130** | 3 |
| `poutcome` | failure | −0.2790 | 6 |
| `month` | feb | −0.1042 | 2 |
| `marital` | married | −0.0765 | 7 |
| `balance_arsinh` | 9.37 | **+0.3471** | 4 |
| `contact` | cellular | +0.2434 | 1 |
| `housing` | no | +0.2168 | 5 |
| `previous_capped` | 2 | +0.1961 | 10 |
| the other four | — | net −0.0415 | — |

**The twelve contributions nearly cancel: −0.0109.** This client has no story. They have a good balance
(+0.3471), a mortgage-free record (+0.2168) and a cellular call (+0.2434) pulling them over the price, and
a call early in the month (−0.5130) with a failed previous campaign (−0.2790) pulling them back under it —
netting out to the model's reference value almost exactly.

**In business language, and this is the finding:** a false positive at a 7.5% price is not an error that
needs explaining, it is arithmetic. The reference value is **11.4361%**, the cut is **7.50%**, so a
completely average client clears the cut and roughly **30.58%** of the file is in this cell by construction.
The false positive is the **modal** cell, and its exemplar sits at the reference — the most ordinary client
in the list.

## 9.5 False negative — training row 10,760, p = 0.0550, y = 1

**margin −2.8446 = reference −2.04695 + sum of twelve −0.7976.**

| feature | value | contribution | T37 global rank |
|---|---|---|---|
| **`balance_arsinh`** | **0.00** | **−0.3502** | 4 |
| **`month`** | **jul** | **−0.3121** | 2 |
| `previous_capped` | 0 | −0.1579 | 10 |
| `age` | 49 | −0.1458 | 8 |
| `poutcome` | unknown | −0.1439 | 6 |
| `marital` | married | −0.1230 | 7 |
| `contact` | cellular | +0.2329 | 1 |
| `housing` | no | +0.1328 | 5 |
| the other four | — | net +0.0694 | — |

Four client-side drags and two scheduling ones outweighed everything positive. **The client had no balance
at all** (arcsinh 0.00, −0.3502) and was reached in **July** (−0.3121) on a **first** contact with **no
previous campaign** (−0.1579) — the exact profile the model has learned to discount. They subscribed anyway.

**In business language:** this is the miss that matters. The cut is at logit(0.0750) = **−2.5123** and this
client sits at −2.8446, so the model was **0.3323 log-odds** short — about two percentage points of
probability. It is a ranking miss of the kind T32's PR-AUC already quantified, not a category error: no
single feature made the client look impossible, and `contact=cellular` was actively arguing the other way.

## 9.6 What the three say together

1. **Campaign-side variables take two of the three largest slots.** The single biggest contribution is
   `month=oct` (**+1.8824**) for the true positive and `day=2` (**−0.5130**) for the false positive; in the
   false negative a client attribute finally leads (`balance_arsinh` 0.00, **−0.3502**) with `month=jul`
   second (**−0.3121**). That is T37's 55.7% campaign-side share showing up client by client — and the
   reminder that a share is an average: in the one client the campaign most wanted, the model's own
   financial signal led.
2. **`balance_arsinh` is the only client attribute that reached the top two of any client** — +0.3471 for
   the false positive, −0.3502 for the false negative, and −0.0069 for the true positive. A client's
   financial position is doing less per-client work than its 8.49% global share suggests, because the
   campaign variables are doing the deciding.
3. **The two errors are ordering misses of 0.3–0.5 log-odds, not absurd calls.** The false negative needed
   +0.3323 to cross; the false positive cleared the cut with contributions summing to −0.0109. Both look
   like the base rate being asked to do a job it cannot do at the margin, which is what a price below the
   average score implies.

## 9.7 What this section does not claim

- **Three clients are illustrations, not a sample.** Each is the median of its cell, so the three together
  say nothing about the *distribution* of explanations across the file — only what one typical member of
  each cell looks like.
- **The scores are in-sample** (9.1), so the confidence here is optimistic relative to the held-out half,
  which was not read.
- **Not causal.** `month=jul` did not make this client not subscribe; it is where the campaign happened to
  be, and it is confounded with 2008–2010 conditions (§8.7).
- **A local explanation explains the model, not the client.** It says why the score came out as it did —
  not why the person said yes or no. Nothing in this dataset observes that.
- **No client is identified.** Row indices are positional in `data/processed/train.csv`; the dataset carries
  no identifiers, and none are invented here.

## 9.8 Verification

`python -m src.local_explain` → exit **0**, **16/16** checks. The ones that would catch a wrong analysis:

| check | result |
|---|---|
| explained model is T31's artifact | sha256 `7a40752f75a264a0…`, 180,520 bytes, 40 columns, 36,168 rows |
| the cut is T34's, re-derived | 0.075000 = `threshold_choice.json` = `cost.break_even_threshold()` |
| the four cells partition the training half | 3,523 + 11,059 + 708 + 20,878 = 36,168 |
| each exemplar sits in the cell it names | scores 0.38189 / 0.11326 / 0.05496 agree with their labels |
| the three exemplars are distinct | rows 19,263 / 23,374 / 10,760 |
| **the rule was followed** | no other member of any cell is strictly closer to its median |
| per-client additivity | max \|sum of 12 + reference − margin\| = **6.853e-07** |
| the margin is the probability the cells were defined on | **8.598e-09** vs `predict_proba` |
| the counts do not depend on the score's route | **1** of 36,168 clients flips; max difference **4.49e-07** |
| local values match `pred_contribs` | **0.000e+00** over 3 rows × 12 features |
| twelve features partition the 40 columns | all 40 indices, once each |
| global ranks come from T37's record | no mismatch |
| above the price is treated as called | 12,867 clients sit in (0.0750, 0.5); row 158 at p = 0.0761 is labelled **FP** |
| `duration` and the target absent | none among the features named |
| held-out half never loaded | 0 references |
| figure written | 197 KB |

The rule check is the one worth keeping: it recomputes the median distance from `p` alone and would fail if
the exemplars had ever been chosen for their explanations rather than their scores.

Recorded as **D59** in `PROGRESS.md`. Next: **T39** — the campaign playbook and the ranked recommendations.
