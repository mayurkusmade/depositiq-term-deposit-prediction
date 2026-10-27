# DepositIQ — Problem Definition

**Rubric:** Criterion 1 — Problem Definition & Business Understanding (10 marks)
**Tasks:** T07 (problem statement, objectives, scope) · T08 (stakeholders and KPIs) · T09 (cost of errors) | **Status:** all complete — Phase 1 done | **Date:** 2026-10-27

---

## 1. Problem Statement

A Portuguese retail bank runs outbound telemarketing campaigns to sell **term deposits**, a
low-risk savings product. Each campaign is expensive because every prospect costs an agent's
time on a phone call, yet the overwhelming majority of calls end in a refusal: in the dataset
used here, only **11.7% of contacted clients subscribed**. The bank therefore cannot simply call
more people — it must decide *which* people to call, using the information it already holds about
each client (age, job, marital status, education, account balance, existing loans, and the history
of previous campaign contacts). The problem is that this decision is currently made without a
systematic, evidence-based way of ranking clients by their likelihood of subscribing, so agent
time is spread across the whole list rather than concentrated on the clients most likely to say
yes. This project addresses that gap by building a supervised machine-learning classifier that
estimates each client's probability of subscribing to a term deposit, so the marketing team can
prioritise its call list and convert a limited number of calls into a larger number of
subscriptions.

---

## 2. Business Objective

**Increase the subscription rate per call made by ranking clients by predicted likelihood of
subscribing, so that a fixed calling budget produces more term deposits than calling the list in
its existing order.**

Restated as a machine-learning objective: *given a client's demographic, financial and
campaign-history attributes, predict the binary outcome `y` — whether that client subscribes to
a term deposit — with a well-calibrated probability that can be used to order a call list.*

---

## 3. Analytical Objectives

Each objective below is written so that it can be **measured or tested**, and each maps to a
later task so that progress is verifiable rather than assumed.

| # | Objective | How it is measured | Task |
|---|---|---|---|
| **O1** | Establish a reproducible, documented baseline of the data: structure, quality and known traps. | Data dictionary exists; quality report records row counts, missing/sentinel values and duplicate rows with evidence. | T10, T11 |
| **O2** | Quantify how strongly each client attribute relates to subscription, and identify which client segments are genuinely worth calling. | Documented EDA figures, each with a written insight; segment subscription rates reported as numbers. | T16–T21 |
| **O3** | Build a leakage-free predictive pipeline, explicitly removing `duration` and recording what its removal costs. | Identical pipelines run with and without `duration`; the difference is reported as a metric gap and explained. | T22, T36 |
| **O4** | Train and compare at least three classification models plus a naive baseline, and select a final model on evidence. | Cross-validated ROC-AUC and PR-AUC for each model; a comparison table; a stated reason for the final choice. | T26–T31 |
| **O5** | Evaluate the final model on business-relevant metrics rather than accuracy, and choose a decision threshold from cost rather than convention. | PR-AUC, precision, recall, F1, confusion matrix, lift and gain reported on held-out test data; threshold justified against the cost model. | T32–T35 |
| **O6** | Explain *why* the model makes its predictions and convert that into actions the marketing team can take. | SHAP global and local explanations; ≥6 actionable recommendations tied to measured results. | T37–T39 |

**Target performance to aim for (not a guarantee):** meaningfully better than the naive
"predict everyone as no" baseline on PR-AUC, and a top-decile lift that is large enough to
justify reordering the call list.

---

## 4. Scope

### 4.1 In scope

- Binary classification of `y` (subscribed / not subscribed) using the 16 client and
  campaign attributes in `data/raw/bank-full.csv`.
- Data quality assessment, cleaning, encoding and scaling of those attributes.
- Systematic handling of the five known traps: the `duration` leakage, `"unknown"` categories,
  the `pdays = -1` sentinel, skewed/negative numeric columns, and accuracy being misleading
  under a ~88/12 class split.
- Model training, tuning, cross-validated evaluation, threshold selection and explainability.
- Deriving business recommendations and reporting them for the marketing team.

### 4.2 Out of scope

The following are **deliberately excluded**; stating them prevents the project from drifting and
shows the boundary of the claim being made:

1. **No real clients, no deployment.** This is an offline, retrospective analysis of a historical
   public dataset. Models are evaluated on held-out historical data; nothing is deployed to
   production, no live client is scored, and no calls are placed.
2. **No profitability guarantee or revenue forecast.** The project reports *ranking* quality
   (lift, gain, precision/recall) and an explicitly-assumed cost comparison. The cost and deposit
   values used in T09 are stated assumptions, not bank-confidential figures, so no claim is made
   about actual currency profit.
3. **No causal claims.** A high-importance feature means the model uses it to *predict*
   subscription, not that changing it would *cause* a client to subscribe. Recommendations are
   framed as targeting guidance, not intervention effects.
4. **No customer-identity or fairness audit as a deliverable.** Attributes such as `age`,
   `marital` and `job` are used because they are in the dataset, but a formal legal or ethical
   fairness audit (e.g. disparate-impact testing) is not undertaken here and would be required
   before any real deployment.
5. **No time-series forecasting or campaign simulation.** `day` and `month` are used only as
   attributes associated with contact timing, not to forecast future demand or model
   campaign scheduling dynamics. The records are ordered by date but the split is not
   chronological, which is acknowledged as a limitation rather than solved here.
6. **No deep learning or unstructured data.** No text, audio, images or neural networks: the
   dataset is small (45,211 rows), tabular and structured, and gradient-boosted trees plus
   simple baselines are the appropriate tool.

### 4.3 Dataset and target — the terms used by the code

To keep the prose and the code describing the same thing, the canonical names are:

| Concept | Canonical name | Where |
|---|---|---|
| Primary dataset file | `bank-full.csv` | `data/raw/bank-full.csv` |
| Rows / columns | 45,211 rows × 17 columns (16 features + target) | verified in T06 |
| Target column | **`y`** (`"yes"` / `"no"`) | encoded 1/0 by `src/data_loader.py` |
| Positive class | `y = 1` = `"yes"` = client subscribed | 11.7% of rows |
| Secondary dataset | `bank-additional-full.csv` (41,188 × 21) | optional comparison variant |
| Source / citation | Moro, Laureano & Cortez (2011), UCI Bank Marketing | `data/raw/bank-names.txt` |

---

## 5. Stakeholders, Decisions and KPIs

**Task:** T08

A model is only useful if it changes somebody's decision. This section names who acts on the
output, what they actually decide differently, and the number they are personally judged on.
The final column separates metrics we can **measure from this dataset** from ones that rest on an
**assumed** business input, so nothing is claimed beyond what the data supports.

### 5.1 Stakeholder table

| # | Stakeholder | Decision they make with the output | KPI they are judged on | KPI type | Source of the KPI |
|---|---|---|---|---|---|
| S1 | **Head of Retail Marketing / Campaign Manager** | Which clients go on the call list, in what order, and how many calls to fund this campaign (e.g. "call only the top 20% by predicted probability"). | **Subscription rate per call** (conversions ÷ calls made), and total subscriptions per campaign | Business | **Measurable** — simulated on held-out test data via the gain/lift table (T35) |
| S2 | **Telemarketing Team Lead / Call-centre Manager** | How to allocate limited agent hours across the ranked list, and where to stop calling (the cut-off score). | **Calls per subscription** (cost-to-acquire efficiency) and agent hours saved versus calling the list in its existing order | Business | **Measurable** — calls-needed-to-reach-X%-of-subscribers comes from the cumulative-gain curve (T35) |
| S3 | **Bank Management / Budget Owner** | Whether to fund a model-driven campaign at all, versus the current untargeted approach; approves the campaign budget. | **Expected net value per campaign** = (extra subscriptions × value) − (call cost × calls made) | Business | **Assumed inputs** — the cost-per-call and deposit-value figures are stated assumptions in T09, not bank-confidential data |
| S4 | **Data Science / Analytics Team** | Whether the model is good enough to ship (accept/reject), which algorithm to deploy, and which probability threshold to use in production. | **PR-AUC** (primary), with ROC-AUC, precision and recall at the chosen threshold as secondary checks | Model | **Measurable** — cross-validated and held-out test results (T32–T34) |
| S5 | **Compliance / Risk Officer** | Whether targeted marketing is acceptable to sign off, and whether any attribute used creates a discriminatory pattern. | **Disparate-impact ratio** across `age` and `job` segments, plus auditability: can each decision be explained? | Model / governance | **Not measured in this project** — formally out of scope (§4.2 item 4) and flagged as a precondition for real deployment |
| S6 | **The customer (the person being called)** | Not a user of the output, but directly affected: decides whether to subscribe, and how many calls to tolerate. | **Unwanted contacts avoided** — fewer irrelevant calls made to clients unlikely to subscribe | Business (qualitative) | **Not measured in this project** — no complaints data exists in this dataset; included so the targeting benefit is not presented as cost-free |

### 5.2 Making the KPIs concrete

The headline KPIs, stated precisely enough to be checked later:

| KPI | Type | Definition used in this project | Numeric target |
|---|---|---|---|
| Subscription rate per call | Business | Subscribers reached ÷ calls made, on the top-ranked slice of clients | **To be beat:** the untargeted baseline rate of 11.7%. Target: materially higher on the top decile. |
| Top-decile lift | Business / model bridge | (subscription rate in the top 10% of predicted clients) ÷ (overall subscription rate) | **Target: > 2×** — i.e. the best 10% of clients should be at least twice as likely to subscribe as an average client |
| Calls per subscription | Business | How many calls must be made, in ranked order, to capture a given share of all subscribers | **To be beat:** calling in the dataset's existing order |
| PR-AUC | Model | Area under the precision–recall curve for the positive class | **To be beat:** the 11.7% positive-class base rate (the level a random ranker achieves) |
| ROC-AUC | Model | Probability a random subscriber is ranked above a random non-subscriber | **Target: ≥ 0.75** on held-out data |
| Recall at chosen threshold | Model | Share of true subscribers that the call list actually reaches | Reporting metric; the cut-off is chosen from cost in T34, not fixed in advance |

### 5.3 Why accuracy is not a KPI here

A model that predicts "no" for every single client scores **≈88.3% accuracy** and converts nobody.
Because of that, accuracy is explicitly excluded from the KPI set (decision D04): it cannot
distinguish a useful model from a useless one on this dataset, so the headline measures are PR-AUC
and top-decile lift instead.

### 5.4 Stakeholder tension worth naming

The campaign manager and the budget owner are not perfectly aligned: the manager wants the
**highest conversion rate**, while the budget owner wants the **highest total net value**. Those
can point at different thresholds — a very high cut-off gives a great conversion rate on few
calls, while a lower cut-off makes more calls with a worse rate but possibly more total
deposits. The threshold task (T34) resolves this explicitly rather than silently picking 0.5.

*Interpretation for the viva:* "The model's output changes one concrete decision — the order and
cut-off of the call list. The marketing manager is measured on conversion rate per call, the
budget owner on net value, and the data science team on PR-AUC because accuracy is meaningless
at an 88/12 class split. Threshold choice is where those three stakeholders disagree, which is
why it is decided from cost rather than convention."

---

## 6. Cost of Errors and the Break-Even Threshold

**Task:** T09 | **Implementation:** `src/cost.py` (every number below is printed by `python -m src.cost`)

Accuracy answers "how often was the model right?". The bank asks a different question: **is this
call worth making?** That has a numeric answer built from two business facts the dataset does not
contain. Because they are assumptions, they are labelled as such and stress-tested in §6.4.

### 6.1 Assumptions

| Input | Value | Status | Source / reason |
|---|---|---|---|
| Fully-loaded agent cost | €18.00 / hour | **ASSUMED** | Wage + employer contributions + supervision + telephony. No bank cost data is public. |
| Minutes of agent time per call | 5.0 min | **ASSUMED** | Dialling + talk + notes. **Corroborated by this dataset**: mean talk time alone (`duration`) is 258.2 s = 4.3 min, so 5.0 min total is plausible rather than optimistic. |
| **Cost of one call, C** | **€1.50** | DERIVED | = 18.00 × 5.0 ÷ 60 |
| Typical term deposit | €2,000 | **ASSUMED** | Not in the data: `balance` is the client's average yearly balance (median €448, mean €1,362), **not** the amount deposited. |
| Term length | 12 months | **ASSUMED** | Standard term used for the comparison. |
| Net interest margin | 1.0% / year | **ASSUMED** | Margin the bank keeps on the deposited funds. |
| **Value of one subscription, V** | **€20.00** | DERIVED | = 2,000 × 1.0% × 12/12. Deliberately conservative: excludes cross-selling, long-term customer value, and the value of deposits as stable funding — all of which would raise V and therefore call *more* clients. |

### 6.2 The break-even threshold

Calling a client with predicted probability `p` is worth it when expected value is positive:

```
EV(call) = p × (V − C) + (1 − p) × (−C)      # subscribe, or don't
         = p × V − C                          # the p·C terms cancel
EV(do not call) = 0

Call if   p × V > C        ⟹        p* = C / V
```

**Break-even threshold: p\* = €1.50 / €20.00 = 0.0750 — i.e. call a client when their predicted
probability of subscribing exceeds 7.5%.**

This is why the project does not use the default 0.5 cut-off: 0.5 is a software convention, not a
business decision. T34 applies `p*` to real model output.

### 6.3 Cost of the two mistakes

| Mistake | What it means | Cost |
|---|---|---|
| **False positive** | We call a client who was never going to subscribe | **€1.50** (the wasted call) |
| **False negative** | We skip a client who *would* have subscribed | **€18.50** — we forgo V but *save* the call, so the net loss is V − C, not V |
| **Ratio** | Missing a subscriber costs | **12.3 × as much as a wasted call** |

The 12.3 : 1 ratio is the quantitative reason accuracy is a bad metric here and recall matters: a
model that is "accurate" by predicting almost everyone as "no" is expensive precisely because
false negatives dominate. Note that both mistake costs are consistent with the same `EV = p·V − C`
formula — the saved call is already netted off, not double-counted.

### 6.4 Sensitivity — the honesty check on the assumptions

If the recommendation flips across plausible inputs, it is an assumption, not a finding. Each cell
is the break-even probability; compare it with the **11.7% base rate** to see how aggressively the
bank should target.

| Call cost C ↓ \ Value V → | €10 | €20 | €40 | €60 |
|---|---|---|---|---|
| **€0.75** | 7.5% | 3.8% | 1.9% | 1.3% |
| **€1.00** | 10.0% | 5.0% | 2.5% | 1.7% |
| **€1.50** (base) | 15.0% | **7.5%** | 3.8% | 2.5% |
| **€2.00** | 20.0% | 10.0% | 5.0% | 3.3% |
| **€3.00** | 30.0% | 15.0% | 7.5% | 5.0% |

### 6.5 The finding that matters — and it cuts against the obvious story

**Under the base assumptions, p\* = 7.5% is *below* the 11.7% base rate.** That means the *average*
client is worth calling (their expected value is +€0.84), so the model is **not** primarily a
cost-cutting tool that shrinks the call list. Its value is **ranking**: within a fixed calling
budget — limited agent hours — it decides *who* gets the calls, not how many should be made.

This is a real and awkward result, so it is stated rather than buried:

- **Supported by this cost model:** for a fixed number of calls, targeting raises the number of
  subscriptions captured (the lift/gain claim measured in T35).
- **Not supported:** the claim that the model *reduces* total call volume or campaign spend. Under
  base assumptions, calling more people is still profitable.

The crossover makes this precise. Calling the average client breaks even when `V = C ÷ base rate`:

```
V* = €1.50 ÷ 0.117 = €12.82
```

- If the bank values a subscription **above €12.82**, the average client pays for their call → call broadly.
- If it values one **below €12.82**, the average client does not → the model must *exclude* people.

Our assumed €20 sits above this line, so with these inputs targeting is a prioritisation exercise,
not a cost-trimming one. The €10 column of §6.4 is the case where that flips.

### 6.6 Sanity check on the arithmetic

Applying the model to all 45,211 clients (everyone called), using only the real labels:

| Quantity | Value |
|---|---|
| Calls made | 45,211 |
| Subscribers reached | 5,289 (the true count) |
| Revenue (× €20) | €105,780 |
| Call cost (× €1.50) | €67,816 |
| **Net value** | **+€37,964** |
| Cost per subscription | €12.82 |

The cost-per-subscription figure is €12.82, which equals `C ÷ base rate` (€1.50 ÷ 0.117) and matches
the crossover value V\* as the algebra requires. `src/cost.py` was also checked against a synthetic,
correctly-calibrated case: the empirical net-value-optimal sweep returned a threshold of **0.06**
against the analytic **p\* = 0.075**, and selected exactly the profitable group — close agreement,
with the small gap expected because any threshold in a flat region of the curve gives the same
selection.

### 6.7 What this is used for next

`src/cost.py` exposes `break_even_threshold()`, `expected_campaign_value()` and `best_threshold()`.
T34 uses them to pick the production cut-off by money rather than convention, T35 turns the ranking
into calls-saved, and S3's "expected net value" KPI in §5.1 becomes a number built from these
assumptions.

*Interpretation for the viva:* "Calls cost about €1.50 and a captured deposit is worth about €20 in
direct margin, so a call pays for itself above a 7.5% probability. Because that is below the 11.7%
base rate, targeting here is about picking the *right* clients for a fixed number of calls — it
saves calls only if the bank's true deposit value is under about €13, and I show that crossover
rather than hide it."

---

## 7. Self-check against the done-conditions

**T07 — problem statement, objectives and scope**

| Requirement | Where satisfied |
|---|---|
| Single-paragraph problem statement | §1 (one paragraph, business framing, no code) |
| ≥4 numbered objectives, each measurable/testable | §3 — six objectives, O1–O6, each with a measurement method and a task ID |
| Explicit out-of-scope section, ≥3 entries | §4.2 — six numbered exclusions |
| Names the dataset and target in the same terms as the code | §4.3 — `bank-full.csv`, target `y`, `data/raw/` path |

**T08 — stakeholders and KPIs**

| Requirement | Where satisfied |
|---|---|
| Stakeholder table with ≥4 stakeholders | §5.1 — six stakeholders, S1–S6 |
| Every row names the **decision** they make | §5.1, column 3 — each is a concrete action (reorder/cut off the list, allocate hours, fund the campaign, ship the model, sign off, tolerate calls) |
| Every row names the **KPI** they are judged on | §5.1, column 4 — no run of "improve performance"-style wording |
| Every KPI labelled business vs model | §5.1, column 5 |
| Each KPI either measurable here or explicitly assumed | §5.1, column 6 — four measurable, two explicitly *not measured* with the reason |

**T09 — cost of errors**

| Requirement | Where satisfied |
|---|---|
| FP and FN costs stated with assumptions | §6.1 (C = €1.50, V = €20.00) and §6.3 (FP €1.50, FN €18.50), every input labelled ASSUMED or DERIVED with its reason |
| Break-even threshold derived, as one explicit number | §6.2 — **p\* = 0.0750 (7.5%)**, derived from `EV = p·V − C`, not asserted |
| `src/cost.py` imports and runs from the repo root | `python -m src.cost` prints the whole model; imported as `from src import cost` |
| Figures reproducible from the script, none hand-computed | §6.1–§6.6 quote the output of `src/cost.py`; the sensitivity grid and the €37,964 sanity figure both come from script runs |

*Interpretation for the viva:* "The project's claim is a **ranking** claim, not a profit claim —
we predict who is likely to subscribe so a fixed number of calls converts better than calling the
list in its existing order, and we deliberately exclude deployment, causal claims and a formal
fairness audit from that claim."

*Note:* §6.5 sharpens this further. Under the assumed €1.50 call cost and €20 subscription value,
the average client is still worth calling, so the claim made here is specifically about converting
a **fixed call budget** better — not about reducing campaign spend.
