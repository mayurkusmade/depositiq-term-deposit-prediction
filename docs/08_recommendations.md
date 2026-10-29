# The DepositIQ campaign playbook (T39)

Eight recommendations, in the order a bank should act on them. Every figure here is **read from a saved
record** by `src/playbook.py` — the recommendations live in that module as data, each number carries the
file and key path it came from, and the module refuses to write anything if a value does not match. Nothing
in this document was retyped from a run: `python -m src.playbook` resolves **46 figures** across nine
records with **12/12** checks passing, and writes `reports/playbook.csv` and `models/playbook.json`.

The five documents behind it: `docs/07_threshold.md` (the price and the cut), `docs/07_lift.md` (what the
ranking is worth in calls), `docs/06_model_comparison.md` (why this model), `docs/08_explainability.md`
(what it looks at, globally and per client), and `docs/05_leakage_audit.md` (what was deliberately left out).

---

## 1. The scope, before any number

> **At the price of a call, the average client is worth calling (p\* = 7.50% is below the 11.6982% base
> rate), so this model converts a fixed call budget better than a fixed order — it does not justify cutting
> campaign spend.**

This is D18 and it governs all eight recommendations. The arithmetic is blunt: calling **every** client nets
**30,368.00 EUR** at the assumed prices, and the model's cut nets **43,621.00 EUR**. The model is worth
**+13,253.00 EUR (+43.6%)** and **21,302 fewer calls** — an improvement on a campaign that already pays, not
a rescue of one that does not. Any sentence in this playbook that sounds like "targeting saves the bank
money" is out of scope and wrong.

| at the adopted cut (0.0750) | training (out-of-fold) | held out (artifact) |
|---|---|---|
| clients called | **14,866** | **3,621** |
| share of the file | **41.10%** | **40.04%** |
| subscribers reached | **3,296** (**77.90%**) | **821** (**77.60%**) |
| precision inside the block | **22.17%** | **22.67%** |
| net value | **43,621.00 EUR** | **10,988.50 EUR** |
| net per 1,000 calls | **2,934.28 EUR** | **3,034.66 EUR** |

---

## 2. The eight recommendations

| # | Recommendation | Headline figure | Would be wrong if |
|---|---|---|---|
| **R1** | Make `contact` a required field | **28.72%** of clients have no channel; they score **4.11%** against **14.83%** | the re-coded clients still convert at 4.11% |
| **R2** | Call in model order, stop at 0.0750 | **77.90%** of subscribers from **41.10%** of the list; **43,621.00 EUR** | the called block converts far below 22.17% |
| **R3** | Do not stop where calls saved peaks | the **8,033** calls past the peak still convert at **8.96%** | that marginal rate falls below 7.50% |
| **R4** | Never use call `duration` | it would add **+0.1788 PR-AUC** and **20,340.00 EUR** — and is unavailable | expected pre-dial length were captured |
| **R5** | Establish V before booking euros | cut costs **6.77 EUR**/subscription; calling everyone **12.82 EUR** | the bank's V is below 6.77 EUR |
| **R6** | Track precision and net value, never accuracy | **66.42%** accuracy against **88.30%** for always-`no` | precision sits at the base rate for campaigns |
| **R7** | Keep the twelve-feature XGBoost | **0.4460** PR-AUC against **0.4346** and **0.4008** | it lands below 0.4008 |
| **R8** | Use `month` to schedule, not to predict | **55.75%** of importance is campaign-side | month effects flatten in a new macro period |

---

## R1. Make `contact` a required field before the next campaign

**Do.** Record the contact channel on every call, and re-score the file once it is filled. Treat the clients
currently coded `unknown` as *unmeasured*, not as poor prospects.

**Why.** **28.72%** of the training file (10,386 clients) has no recorded channel. The model assigns that
level a contribution of **−1.0150** log-odds, and those clients subscribe at **4.11%** against **14.83%** for
`cellular`. `contact=unknown` is the single largest column of the forty (**10.96%** of total column
importance), and `contact` is the model's most important feature (**22.14%**). For one explained client it
took **−0.6989** back from an otherwise strong score (`docs/08_explainability.md` §8.4, §9.3).

**What to expect.** No conversion lift is claimed — that would need a campaign this dataset does not have.
The effect is on **scoring**: nearly three clients in ten receive the model's worst channel value because a
field was left blank, and their 4.11% rate is a measurement of the blank field, not of the person.

**How to check it.** Track the share of `unknown` in the scored file (target: near zero) and the measured
conversion of the re-coded group next campaign, against **4.11%**.

**What would make it wrong.** If clients re-coded away from `unknown` still convert at about 4.11%, the level
was carrying real information about them and the recommendation is wrong.

---

## R2. Call the list in model order, and stop at 0.0750

**Do.** Sort the campaign by P(subscribe), call down the list while the score is at or above **0.0750**, and
stop there. On this file that is **14,866** clients (**41.10%**), reaching **3,296** subscribers
(**77.90%** of them), for **43,621.00 EUR** net.

**Why.** A fixed order would need **28,175** calls to reach the same 3,296 subscribers; the ranking needs
14,866, so it saves **13,309** calls (**36.8%** of the list), worth **19,963.96 EUR** of call time. The
block converts at **22.17%** against the **11.6982%** base rate. On the held-out half the same rule calls
**3,621** clients, reaches **821** subscribers (**77.60%**) at **22.67%** precision.

**What to expect.** **2,934.28 EUR** of net value per 1,000 calls in the called block. The same campaign
run as a blanket call costs **12.82 EUR** per subscription; this cut costs **6.77 EUR** for the same
subscription value, which is where the improvement shows up in a budget line.

**How to check it.** Precision inside the called block each campaign, against **22.17%**, and net EUR per
1,000 calls.

**What would make it wrong.** If the called block converts far below 22.17%, the ranking has stopped
describing this population; if it converts far above, the cut is leaving money on the table and should be
re-derived from the current campaign.

---

## R3. Do not stop the list where calls saved peaks

**Do.** If agent hours are capped, choose the cut by **capacity** from the recorded table — 5% of the file
is **1,809** calls and nets **19,246.50 EUR** — not by minimising calls.

**Why.** Maximising calls saved peaks at **18.89%** coverage (**6,833** calls, **15,188** calls saved,
precision **37.70%**, lift **3.2227×**). That peak sits **before** the adopted cut, and the **8,033** calls
between them still convert at **8.96%**, above the **7.50%** price, for **+2,350.50 EUR**. Pruning to the
peak would give up that money to save calls the campaign was already going to make.

**What to expect.** The saving is a rate, not a decision: each call past the peak adds value while the local
conversion rate stays above the price. The grid's best cut is only **388.50 EUR** ahead of the adopted one, so
the exact cut is not where the value is — **capacity** is.

**How to check it.** Marginal conversion of the block between any two candidate cuts, priced at C and V
before the list is shortened.

**What would make it wrong.** On the held-out half that same block is **1,547** calls at **7.76%** for only
**+79.50 EUR**. This is the thinnest recommendation in the playbook: if the marginal rate drops below 7.50%
on fresh data, stopping earlier becomes correct.

---

## R4. Never feed call `duration` into the model

**Do.** Keep `duration` out of the scored matrix and out of any agent's ranking or manual scoring hint.

**Why.** The column is recorded when the call ends, so it cannot inform the decision to make the call.
Measured, it would add **+0.1788 PR-AUC** and **20,340.00 EUR**, on **4,320 fewer calls** that reach **693**
more subscribers — and it would be a model that has already seen the answer. Used alone, with no model at
all, it ranks the list at ROC-AUC 0.8078. The deployable model's pooled PR-AUC is **0.4448**.

**What to expect.** A lower headline and an honest one. The gap is the price of a model that can be used
before dialling, not money left on the table.

**How to check it.** A release-time scan of the scored matrix: **40** columns, twelve features, no `duration`
and no target.

**What would make it wrong.** If a future system captured the *expected* call length **before** dialling —
not the actual one — the column would become legitimate and this prohibition would no longer apply.

---

## R5. Establish the value of a subscription before booking any euro figure

**Do.** Replace the assumed **V = 20.00 EUR** with the bank's own number before the 43,621.00 EUR is planned
against. The ranking's advantage survives any V above **6.77 EUR**; the euro headline does not.

**Why.** Both prices are **ASSUMED** (D17): C = **1.50 EUR** per call, V = **20.00 EUR** per subscription,
which puts the break-even probability at **7.50%**. At the adopted cut each subscription costs **6.77 EUR**
in calls; calling everyone costs **12.82 EUR** per subscription. The band between those two numbers is where
targeting decides whether the campaign is profitable at all.

**What to expect.** Every euro figure in this playbook is linear in V. At V = 20.00 the campaign nets
43,621.00 EUR; at V = 6.77 it nets zero. Net value per 1,000 calls is **2,934.28 EUR** at the assumed V.

**How to check it.** Re-run the pricing with the real V and C: the break-even moves to C/V and the
recommended cut moves with it.

**What would make it wrong.** If V is below 6.77 EUR, the cut is unprofitable and a stricter cut — or no
campaign — is correct. The same applies if C is materially above 1.50 EUR.

---

## R6. Track precision and net value, never accuracy

**Do.** Report precision inside the called block, recall of subscribers, and net EUR per 1,000 calls.
Retire accuracy from the campaign dashboard.

**Why.** The model is **less accurate** than predicting `no` for everyone — **66.42%** against **88.30%** on
held-out rows — and far more useful: **77.60%** of held-out subscribers reached at **22.67%** precision. A
0.5 rule instead of the price would miss **3,192** subscribers and net **18,248.00 EUR** against 43,621.00.
Accuracy rewards predicting `no`; this problem punishes it.

**What to expect.** Precision around **22.67%** and net value around **3,000 EUR** per 1,000 calls. If those
hold, the campaign is working regardless of what accuracy does.

**How to check it.** Precision and net EUR per 1,000 calls, campaign on campaign, against 22.67% and the
same cut.

**What would make it wrong.** If precision inside the called block stays at the base rate over several
campaigns, the ranking has stopped adding value — whatever its accuracy.

---

## R7. Keep the twelve-feature XGBoost

**Do.** Ship the persisted artifact as it is. Do not add back the five dropped features, do not weight the
classes, and do not re-tune on a hunch.

**Why.** The shipped configuration leads both alternatives on ranking: **0.4460** PR-AUC against **0.4346**
(random forest) and **0.4008** (logistic baseline), on **40** columns instead of 65. On rows it never saw it
holds **0.4616**. Tuning moved cross-validated PR-AUC from **0.4444** to **0.4460**, and the imbalance weight
cost ranking while halving calibration — both measured, both recorded.

**What to expect.** No further gain from re-opening the feature set or the weight. The remaining gains are
in R1 (data quality), R5 (pricing) and R2/R3 (how the list is worked).

**How to check it.** The D46 revisit trigger: if a 40-column XGBoost ever lands below the logistic
baseline's **0.4008**, the feature set is re-opened.

**What would make it wrong.** A re-measurement putting the shipped configuration below 0.4008, or a business
change that adds a genuinely new column — which is a T22-style leakage audit, not a re-tune.

---

## R8. Use `month` to schedule, never to predict a client

**Do.** Read the month effects as a **calendar** signal when planning the campaign, and keep them out of any
statement about an individual client's propensity.

**Why.** `month` carries **17.55%** of model importance; March contributes **+1.3736** log-odds for a client
(50.90% conversion) and May **−0.2999** (6.68%), while May is **30.59%** of the file. Campaign-side features
together carry **55.75%** of importance against **44.25%** for the client's own attributes, so the largest
levers are *how and when* the bank calls, not *whom*.

**What to expect.** A scheduling and data-quality agenda (R1, R8) rather than a targeting one. That is the
honest reading of the model's own explanation.

**How to check it.** Conversion by month next campaign against **50.90%** (March) and **6.68%** (May), with
volume held roughly constant.

**What would make it wrong.** These are 2008–2010 months, confounded with economic conditions this dataset
variant cannot see. If conversion by month flattens under a different macro environment, the signal was the
period, not the calendar.

---

## 3. The order to act in

1. **R1** — cheapest and largest lever on scores; costs one required field.
2. **R5** — because it decides the cut. Everything euro-denominated depends on V.
3. **R2** — run the campaign at the cut.
4. **R3** — size the list by capacity, not by the savings peak.
5. **R6** — instrument precision and net value so the next campaign can disagree with this document.
6. **R4, R7** — standing prohibitions, checked every release.
7. **R8** — planning input for the next calendar, with its confound stated.

## 4. What this playbook does not promise

- **No spend reduction.** The average client is still worth calling; this is a fixed-budget improvement
  (D18, §1).
- **No lift from filling `contact`.** The 4.11% rate says the field is blank, not that filling it converts
  anyone. R1 is a measurement recommendation.
- **No causal month effect.** R8 is a calendar observation under 2008–2010 conditions.
- **No generalisation beyond this population.** Every recommendation rests on the one classic
  telemarketing campaign in `data/raw/bank-full.csv`; the held-out half is **9,043** rows from the same
  period, not a second campaign.
- **No precision beyond the scope.** The euro figures are linear in an **assumed** V = 20.00 EUR; the bank's
  own value replaces them (R5).
- **No claim that the exact cut is optimal.** It sits **388.50 EUR** below the grid's best point, inside a
  plateau that spans thresholds **0.0746** to **0.1016**.

## 5. Verification

`python -m src.playbook` → exit **0**, **12/12** checks, **8** recommendations, **46** figures resolved.

| check | result |
|---|---|
| at least six recommendations | **8** |
| each carries an action, a check and a falsifier | all 8 |
| ids unique | R1–R8 |
| every cited number resolves | **46** of 46 |
| every cited number still matches its record on disk | 46 re-read, 0 mismatched |
| no recommendation rests on one figure | at least 2 each |
| the D18 scope is stated and attached | R2, R3 carry the fixed-budget tag |
| `duration` appears only as the prohibition | R4 only |
| net value = revenue − call cost | 3,296 × 20.00 − 14,866 × 1.50 = **43,621.00** |
| the pricing band is measured | 6.77 at the cut vs 12.82 calling everyone |
| both halves describe the same campaign | coverage 41.10% / 40.04%; recall 77.90% / 77.60% |
| derived quantities are consistent | campaign 55.75% + client 44.25% = 1 |

The check that matters is the **fifth**: every figure is re-read from disk and compared with what the
document cites, so a stale record or a mistyped path fails the build instead of shipping. `duration`
appearing anywhere except R4 also fails — the prohibition is enforced mechanically, not by memory.

Recorded as **D60** in `PROGRESS.md`. Next: **T40** — the final report, assembled from these records.
