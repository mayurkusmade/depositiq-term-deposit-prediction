# DepositIQ — Viva Q&A Bank

**Purpose:** the questions a marker, supervisor or client is most likely to ask about this project, with
answers that can be defended from the repository rather than from memory.

**Coverage:** **37 questions**, at least four for each of the nine rubric criteria. Every answer names the
record or document that holds its evidence. The bank introduces no new experiment and no new figure; it is
a route through work that is already on disk.

**How to use it:** read the short answer first, then the evidence line. If a question is followed by
*If pressed*, that is the follow-up that separates a memorised answer from an understood one. The five
numbers worth knowing cold are in §10.

**Source policy:** a number in this bank is either in a `models/*.json` record, a `reports/*.csv` table, or
one of the eleven analysis documents under `docs/`. The validator `python -m src.viva` re-parses every
decimal and fails if one appears here and nowhere else. It also checks that all nine criteria are covered,
that every question has an answer and a source, and that every relative link resolves.

---

## Coverage map

| Criterion | Questions | Main sources |
|---|---|---|
| 1. Problem definition and business understanding | Q1–Q4 | `01_problem_definition.md`, `models/threshold_choice.json` |
| 2. Dataset and data understanding | Q5–Q8 | `02_data_dictionary.md`, `models/cv_results.json` |
| 3. Preprocessing and cleaning | Q9–Q12 | `src/clean.py`, `src/features.py`, `src/preprocess.py` |
| 4. Exploratory analysis | Q13–Q16 | `04_eda_insights.md`, figures `01`–`05` |
| 5. Feature engineering and selection | Q17–Q20 | `05_leakage_audit.md`, `src/select.py` |
| 6. Model selection and implementation | Q21–Q24 | `06_modeling.md`, `06_imbalance.md`, `models/cv_results.json` |
| 7. Evaluation and validation | Q25–Q29 | `07_threshold.md`, `07_lift.md`, `models/holdout_eval.json` |
| 8. Business insights and recommendations | Q30–Q33 | `08_explainability.md`, `08_recommendations.md` |
| 9. Documentation and reproducibility | Q34–Q37 | `report.md`, `README.md`, `PROGRESS.md` |

---

## 1. Problem definition and business understanding

### Q1. What decision does this model support, and why is it a ranking problem rather than a classification problem?

**Answer.** The bank runs an outbound telephone campaign and has more clients than agents, so every day it
must choose **which clients to call today**. The model estimates the probability that a client will
subscribe if called, and the campaign works down the list in that order until its call capacity or the
price cut is reached. That is a ranking problem because the constraint is the number of calls, not a
yes/no label: a classifier at 0.5 answers "is this client above a convention?" while the campaign needs
"which clients are most likely to pay for the call?" The unit of cost is one call, and the unit of value is
one captured deposit, so the ranking has to be priced rather than merely scored.

**Evidence.** [`01_problem_definition.md`](01_problem_definition.md) §1 states the decision and the six
objectives; [`report.md`](report.md) §1.1 restates it; `models/threshold_choice.json` records the price
that turns the ranking into a call list.

**If pressed.** A classifier is not wrong, it is answering a different question. The project does fit
classifiers, but it evaluates them on ranking metrics and chooses the operating cut from the cost model
(Q25–Q27), because a label at 0.5 would call 4.67% of the list and miss most subscribers.

### Q2. What is the cost model, and where does the 0.0750 cut come from?

**Answer.** A call costs **C = 1.50 EUR** and a captured subscription is assumed to be worth **V = 20.00
EUR**. Calling is worthwhile when the expected value is positive:

`V · P(subscribe) > C  ⇒  P(subscribe) > C / V = 1.50 / 20.00 = 0.0750`.

So the break-even probability is **7.50%**, and it is imported from `src/cost.py` rather than typed. Both
prices are **assumptions**, not bank data, and they are declared as such in the record. Every euro total in
the project is linear in V: if the bank's true deposit value is 40 EUR, the cut halves to 3.75% and every
total doubles. A 20-cell sensitivity grid over C ∈ {0.75, 1.50, 2.50, 4.00} × V ∈ {10, 20, 30, 50} EUR
confirms the arithmetic in each cell.

**Evidence.** `models/threshold_choice.json` holds the prices, the cut and the sensitivity grid;
[`report.md`](report.md) §1.4 derives it; `src/cost.py` is the function that computes it.

**If pressed.** The cut is not fitted to the data. It is chosen on the training half from out-of-fold
scores and checked once on the held-out half (Q27). The empirical optimum is 0.0821 and is deliberately not
adopted.

### Q3. What is the scope sentence, and why does it matter that the cut is below the base rate?

**Answer.** The break-even probability of **7.50%** sits **below** the modelling half's base rate of
**11.6982%**. The average client is therefore already worth calling, so the model **converts a fixed call
budget** better than a fixed order; it is an improvement to a campaign that already pays, and it is **not**
evidence that the bank should cut campaign spend. That sentence is placed before the first euro total in
the report (D18), because a reader who sees only "+13,253.00 EUR" could otherwise conclude that the campaign
was rescued or that its budget should shrink. Neither reading is supported.

**Evidence.** [`report.md`](report.md) §0 and §1.4; [`01_problem_definition.md`](01_problem_definition.md)
§2; `models/threshold_choice.json` records the base rate beside the cut.

**If pressed.** Calling everyone already nets **30,368.00 EUR** on the training half. The model nets
**43,621.00 EUR**, a gain of **+13,253.00 EUR** on **21,302 fewer calls**. The correct comparison is against
"call everyone", not against a dummy that calls nobody.

### Q4. Who are the stakeholders, and what does each one own?

**Answer.** Six stakeholders were mapped before modelling: the **campaign manager** owns subscribers
reached per 1,000 calls and decides whom to call today; the **head of retail** owns net value in euro and
decides whether the campaign is worth its budget; **contact-centre operations** owns calls per day at a
fixed coverage and decides how many agents to roster; **data governance** owns the leakage verdict and
decides whether the model may be used on live clients; **compliance** owns segment-level recall and
precision and decides whether the model treats client groups fairly; and the **analyst** owns monitoring of
PR-AUC against the **0.4460** benchmark and decides when to refit.

**Evidence.** [`01_problem_definition.md`](01_problem_definition.md) §3; [`report.md`](report.md) §1.3;
`models/metrics_cv.json` holds the benchmark.

**If pressed.** The compliance requirement is only **partly** met: segment-level metrics exist, but a full
fairness audit across age or job groups does not (Q36). The project says so rather than claiming a clean
bill of health.

---

## 2. Dataset and data understanding

### Q5. Why is `bank-full.csv` the modelling variant, and not `bank-additional-full.csv`?

**Answer.** `bank-full.csv` has **45,211 rows × 17 columns**, **5,289** subscribers (**11.70%**), no nulls
and no duplicate rows. It also carries the client's contact history — `pdays`, `previous`, `poutcome` —
which is the only part of the file that tells the model what the bank already knows about this person.
`bank-additional-full.csv` has **41,188 rows × 21 columns**, **4,640** subscribers (**11.27%**), **12**
exact duplicates, and five macro-economic columns that are identical for every client on a given day, so
they cannot change one client's score against another's. The two are not interchangeable: `education` is
coded differently, `pdays` uses **−1** in one and **999** in the other, and `poutcome = unknown` corresponds
to `poutcome = nonexistent`. The variant is therefore a parameter, not a default.

**Evidence.** [`02_data_dictionary.md`](02_data_dictionary.md) §1–§2; [`report.md`](report.md) §2.1;
`src/config.py` exposes `variant`.

**If pressed.** The additional set is kept because it is the richer economic variant and the brief names
it, but shipping it would require a separate encoder and a separate validation. Mixing the two encodings
would silently produce nonsense.

### Q6. There are no nulls in the file. So what is missing, and how is it handled?

**Answer.** Missingness is disguised as literal strings. On the modelling variant, `poutcome = unknown`
covers **36,959 rows (81.7%)**, `contact = unknown` covers **13,020 (28.8%)**, `education = unknown` covers
**1,857**, and `job = unknown` covers **288**. They are not the same kind of missing. `poutcome = unknown`
is a **fact** — there was no prior campaign — so it belongs as a level. `contact = unknown` is a **failure
to record**, and it turns out to be the model's most influential input, so it also stays as a level:
imputing it would invent the very field whose absence is informative. The model is told "unknown", not
given a guess.

**Evidence.** [`02_data_dictionary.md`](02_data_dictionary.md) §3; [`report.md`](report.md) §2.2;
`models/shap_global.json` holds the contribution of `contact = unknown`.

**If pressed.** `contact = unknown` contributes **−1.0150** log-odds and those clients subscribe at
**4.11%** against **14.83%** for `cellular`. The model is not saying they are bad prospects; it is saying it
cannot tell. That is recommendation R1: make the field required.

### Q7. What is wrong with `pdays`, and why is `days_since_contact` a duplicate rather than a correlated feature?

**Answer.** `pdays = −1` means "never contacted before", not "contacted minus one day ago". Read as a
number it produces a mean of **40.2** days, a figure no client has. Worse, `pdays` and
`days_since_contact` are the **same number on all 6,584 rows where both exist**. The complete-case
correlation matrix is **rank 7 of 8** with a condition number of **5.0×10¹⁶**, and both columns carry a VIF
of **9.0×10¹⁵**. That is not a correlated pair; it is one column stored twice. The project drops `pdays`,
derives a `previously_contacted` flag from `previous > 0`, and derives `recency_band` with an explicit
`never` level for the 81.80% of clients who were never contacted.

**Evidence.** [`05_leakage_audit.md`](05_leakage_audit.md) §7; [`report.md`](report.md) §2.3;
`src/features.py` implements both derivations.

**If pressed.** The flag is derived from `previous > 0`, not from `pdays`, because the additional variant
contains sentinel rows where `previous` is 1–6; the `previous` rule agrees with `poutcome != nonexistent`
in both variants, so it is the portable choice (D21).

### Q8. What is the base rate, and why does the project distinguish 11.70% from 11.6982%?

**Answer.** The file's base rate is **5,289 / 45,211 = 11.70%**. The modelling half is **4,231 / 36,168 =
11.6982%**, and the held-out half is **1,058 / 9,043 = 11.6997%**. The three are close but not identical,
and each is used in the place where it is the correct denominator: lift on the training half is measured
against **11.6982%**, lift on the held-out half against **11.6997%**, and the file-level description uses
**11.70%**. Pooling them would make a small but avoidable error, and the project's rule is that every
percentage names its denominator.

**Evidence.** `models/cv_results.json` records the training counts; `models/holdout_eval.json` records the
held-out counts; [`report.md`](report.md) §2.1 and §7.4 keep the three separate.

**If pressed.** The split differs from the population by about **0.0012 pp**, which is the check that
stratification worked. It is not a reason to treat the rates as interchangeable in a lift table.

---

## 3. Preprocessing and cleaning

### Q9. How were `unknown` values handled, and why not impute them?

**Answer.** They were **kept as their own category**, never imputed. The decision is not a refusal to
clean; it is a claim about what the values mean. `poutcome = unknown` is a fact ("no prior campaign"), and
`contact = unknown` is a recording failure whose absence is itself predictive. Imputing `contact` would
replace a real signal with a guess and would make the model look more confident than the data allows. The
cleaning module converts raw → processed for both variants with **0 rows lost**, prints before/after shape,
dtypes and class balance, and exits 0 with **15 checks per variant**. The checker was then proven able to
fail: **8 injected faults** were all reported, none raised.

**Evidence.** `src/clean.py`; [`02_data_dictionary.md`](02_data_dictionary.md) §3; [`report.md`](report.md)
§3.

**If pressed.** A missing-indicator plus imputation is a legitimate alternative, but it would still need
the `unknown` level to preserve the signal, and there is no evidence in this project that the imputed value
adds anything. The simpler model is the one that ships.

### Q10. Why keep outliers instead of winsorising or clipping them?

**Answer.** The tails are **signal, not error**. **3,001 rows (8.30%)** carry a negative balance, down to
**−8,019 EUR**, and the subscription rate varies by more than ten percentage points across balance deciles
and campaign bands. Clipping would merge groups that differ materially. Winsorising at 5–95 would rewrite
**13,054 rows** and destroy **3,366 distinct values**, and it is the only non-invertible option considered.
The project therefore keeps every value and lets the tree ensemble find the split points; the linear
baseline gets a `RobustScaler`, fitted inside each fold.

**Evidence.** [`04_eda_insights.md`](04_eda_insights.md) §2–§3; [`report.md`](report.md) §3;
`PROGRESS.md` D24 records the measurement.

**If pressed.** Keeping outliers does not mean ignoring skew. `balance` is transformed with `arsinh`,
`campaign` is bucketed, and `previous` is capped, so the model sees a usable scale without deleting rows.

### Q11. Why `arsinh` for balance, and why buckets and caps for campaign and previous?

**Answer.** `log1p` is undefined on a negative balance, and the file has thousands of negative balances, so
the transform has to be signed. `arsinh` is signed, defined on the whole real line, and takes the absolute
skew of `balance` from **8.34** to **1.68**. `campaign` is bucketed into six call-count bands because the
relationship is not linear: one call converts at **14.59%**, ten or more at **4.30%**. `previous` is capped
at 2 because the useful distinction is none / one / two-or-more, and the rates climb **9.22% → 20.65% →
23.97%** across those groups. `balance_band` was engineered and then dropped: its mutual information was
**0.0050**, but its ablation cost **−0.00023 ± 0.00085**, so it added nothing.

**Evidence.** `src/features.py`; [`report.md`](report.md) §5.2; `models/cv_results.json` records the
feature list and `reports/figures/06_feature_importance.png` shows the ablation.

**If pressed.** A median/IQR scaler is not a safe default here: two columns have an IQR of exactly zero, so
the scaler would divide by zero. The choice of transform is driven by the data, not by convention.

### Q12. How was the train/test split made, and how is leakage prevented in preprocessing?

**Answer.** The split is **stratified 80/20 with seed 42**, producing **36,168** training rows and **9,043**
held-out rows, with the two rates within **0.0012 pp** of the population. It is written to
`data/processed/train.csv` and `test.csv` and reloaded by every later script through `split.load_split`, so
no step can quietly re-draw it. The fold arrays are materialised **once** and replayed, and every
transformer is fitted **inside the fold**, not on the pooled data. A pooled fit was measured to sit
**4.32e-02** away from the training mean, which is exactly the kind of drift the fold-local fit prevents.
The 9,043 held-out rows are not loaded by any modelling script, and a source scan enforces that.

**Evidence.** `src/split.py`, `src/preprocess.py`; [`report.md`](report.md) §3 and §7.5;
`models/final_model.json` records the source-scan check.

**If pressed.** The split is stratified-random, **not temporal**. A client in the test half may be a
neighbour of one in train, and a time-based split would be the honest test of generalisation to the next
campaign. That is stated as a limitation, not hidden (Q36).

---

## 4. Exploratory analysis

### Q13. What did the target distribution show, and why is accuracy misleading here?

**Answer.** The training half has **4,231** subscribers against **31,937** non-subscribers, a **7.55 : 1**
split, or one deposit per **8.55** calls. An always-"no" model therefore scores **88.3003%** accuracy and
finds nobody. The shipped model scores **66.42%** at the price cut — **22 points worse** — while finding
**77.60%** of subscribers. Accuracy ranks the useless model above the useful one, which is why it is
reported once, in §7.6, and labelled as a trap rather than used to select anything.

**Evidence.** `reports/figures/01_target_distribution.png`; `models/holdout_eval.json`;
[`report.md`](report.md) §4 and §7.6.

**If pressed.** Balanced accuracy is better but not sufficient: the shipped model reaches **71.27%** at the
price cut against **61.28%** at 0.5, and it still misses **935** subscribers on the training half. The
campaign is capacity-limited, so recall and net value are the decision metrics.

### Q14. Which categorical segment is strongest, and is it usable?

**Answer.** `poutcome = success` converts at **64.65%**, against **4.11%** for `contact = unknown`, a
**15.7×** spread across the 44 levels measured. It covers **779 of 1,205** clients (Wilson 95% interval
**61.91%–67.30%**), is **3.33%** of the training half, and supplies **18.41%** of all **4,231**
subscriptions. It is usable because it is **history**: the outcome of a previous campaign is known before
the next call. The encoder is fitted on fold-train only, so the level does not leak the current outcome.

**Evidence.** [`04_eda_insights.md`](04_eda_insights.md) §4; [`report.md`](report.md) §4;
`models/shap_global.json` records the `poutcome = success` contribution.

**If pressed.** `poutcome = success` is the largest single-**level** SHAP effect at **+1.6841**, but the
feature ranks **6th** by mean absolute contribution. Mean |SHAP| is not per-client influence; a rare level
can have a large effect on the few clients who have it.

### Q15. What did the campaign analysis find about repeated calls?

**Answer.** Clients needing **4 or more calls** hold **52.68%** of the **99,966** calls for **13.47%** of
the successes. The first three calls convert at **7.7386%**, just above the **7.50%** price; the extra calls
convert at **1.9298%**, or **77.73 EUR** per extra subscription against a **20.00 EUR** deposit — a net
**−32,906 EUR** of agent time in the tail. The finding is not "cap everyone at three calls": **93.28%** of
the tail sits in the no-known-outcome group, while `poutcome = success` still returns **22.55%** a call. The
cap has to be **targeted**, which is what `campaign_bucket` lets the model do.

**Evidence.** [`04_eda_insights.md`](04_eda_insights.md) §5; `reports/figures/04_campaign_vs_target.png`;
[`report.md`](report.md) §4.

**If pressed.** This is a marginal analysis, not a causal one. The clients who need four calls are
different from those who need one, and the file cannot tell us what would happen if a four-call client were
stopped at three.

### Q16. What did the correlation heatmap show, and what is the trap in reading it?

**Answer.** It showed that **no client attribute carries much linear signal** with the target. The
strongest-looking column is `duration`, which is recorded **after** the call and is therefore not available
at decision time. The heatmap also exposed the `pdays` / `days_since_contact` duplicate: the two columns are
the same number on every complete case, which is redundancy rather than a useful correlation. The trap is
to treat a correlation matrix as a feature-selection tool: it cannot see non-linearities or interactions,
and it will happily rank a post-outcome column first.

**Evidence.** [`04_eda_insights.md`](04_eda_insights.md) §6; `reports/figures/05_correlation_heatmap.png`;
[`report.md`](report.md) §4 and §2.3.

**If pressed.** That is why selection uses **mutual information with a permutation null** and
**permutation importance**, not correlation. The two methods rank the 17 features at Spearman **+0.647**,
which is agreement without redundancy (Q19).

---

## 5. Feature engineering and selection

### Q17. Why is `duration` excluded, and what would it have bought?

**Answer.** `duration` is the length of the call, so it does not exist until the call has ended. A model
given it learns "was this call long?", which is answered by the outcome it is meant to predict. Alone, as a
score, `duration` reaches **ROC-AUC 0.8078** and **PR-AUC 0.3897** against the **0.1170** base rate —
**3.33×**, better than the final model's ROC-AUC. Its bands climb from **0.16%** under a minute to
**48.56%** at 600 seconds or more, a **302×** swing, and the longest tenth of calls supplies **35.03%** of
all subscribers. Including it would raise PR-AUC to **0.6254 ± 0.0155**, ROC-AUC to **0.9360**, and net
value to **63,961.00 EUR**, a gap of **+0.1788** PR-AUC and **+20,340.00 EUR**. It is a **diagnosis, not a
gain**: the mean predicted probability barely moves (**0.1166** vs **0.1162**) while Brier improves from
**0.0818** to **0.0621**, the signature of a model that has seen the answer.

**Evidence.** `models/duration_comparison.json`; [`05_leakage_audit.md`](05_leakage_audit.md) §5–§6;
[`report.md`](report.md) §5.4.

**If pressed.** The verdict was pre-registered: arm B must win at least **4/5** folds, the duration bands
must be monotone, and `duration` alone must reach ROC-AUC above **0.7**. All three are true, so
`gap_is_leakage = True`; arm A ships anyway because the column does not exist at decision time.

### Q18. What five features were engineered, and what does each replace?

**Answer.** Five, each **replacing** its raw column rather than sitting beside it:

| Feature | Replaces | What it encodes |
|---|---|---|
| `campaign_bucket` | `campaign` | six call-count buckets; 14.59% at one call → 4.30% at ten or more |
| `previous_capped` | `previous` | previous contacts capped at 2; 9.22% → 20.65% → 23.97% |
| `balance_arsinh` | `balance` | signed log-like transform; skew 8.34 → 1.68 |
| `balance_band` | `balance` | non-linear balance; tested and dropped (−0.00023 ± 0.00085) |
| `recency_band` | `days_since_contact` | explicit `never` level; 81.80% of clients |

The result is **17 features** (4 numeric, 12 categorical, 1 boolean) with `duration`, `pdays` and `y`
excluded. Eight leakage checks run inside the module: permuting `y` and permuting `duration` must leave
every engineered value byte-identical, no feature may read a forbidden column, and features must be
per-row. The permutation tests were proven able to fail by planting a target-derived and a
duration-derived column.

**Evidence.** `src/features.py`; [`report.md`](report.md) §5.2; `models/cv_results.json` records the
feature list.

**If pressed.** `balance_band` is the honest failure: mutual information **0.0050** looked promising, but
the ablation said it added nothing, so it was dropped rather than kept because it was engineered.

### Q19. How did 17 features become 12, and why were those five dropped?

**Answer.** Two independent methods ranked the same 17 features: mutual information with a **per-feature
permutation null**, and permutation importance from a 200-tree forest with out-of-fold ROC-AUC **0.7936 ±
0.0063**. They correlated at Spearman **+0.647**. A third measurement decided: the five nominated features
removed as a group cost **−0.00014 ± 0.00195** ROC-AUC with **3 of 5** folds improving, which is inside one
standard error, so the one-standard-error rule dropped them. The five are `job`, `education`, `default`,
`loan` and `balance_band`. The twelve kept are `contact`, `month`, `day`, `balance_arsinh`, `housing`,
`poutcome`, `marital`, `age`, `campaign_bucket`, `previous_capped`, `recency_band` and
`previously_contacted`.

**Evidence.** `src/select.py`; `reports/figures/06_feature_importance.png`; [`report.md`](report.md) §5.3.

**If pressed.** The reduction was re-measured per estimator rather than trusted: the forest loses
**−0.0014 ± 0.0024** ROC-AUC and the gradient booster **−0.0001 ± 0.0008**, both inside one sd, while the
linear model loses **−0.0039 ± 0.0023** with **0/5** folds improving (sign test p = **0.031**). That is a
linear-model effect, not a property of the feature set.

### Q20. How do you know the feature selection did not leak or overfit the held-out half?

**Answer.** The selection uses the **training half only**. Mutual information is compared with a
**per-feature permutation null**, permutation importance is measured on **out-of-fold** predictions, and
the group ablation is evaluated on the project's five folds. The held-out 9,043 rows are not loaded by
`src.select`, and the final model's source scan proves no held-out load. The one-standard-error rule was
fixed before the ablation was read, so the decision rule is not chosen after seeing the answer. The
12-feature / 40-column partition is re-derived and checked, not remembered.

**Evidence.** `src/select.py`; [`report.md`](report.md) §5.3 and §9.2; `models/final_model.json` records
the source-scan check.

**If pressed.** Any data-driven selection carries some selection bias; the controls are the permutation
null, the fold-based ablation, the pre-registered one-SE rule, and the fact that the held-out half is spent
once at the very end (Q26).

---

## 6. Model selection and implementation

### Q21. Why XGBoost, and what were the alternatives?

**Answer.** Seven builds ran through one harness — the same `ColumnTransformer`, the same five folds, the
same 36,168 rows and 4,231 positives. On cross-validated PR-AUC: dummy **0.1170**, logistic regression
**0.4008**, random forest **0.4346**, XGBoost weighted **0.4420**, XGBoost untreated **0.4444**, XGBoost
with SMOTE **0.4152**, and the tuned untreated XGBoost **0.4460**. XGBoost wins on the ranking metric, has
native `scale_pos_weight` for the imbalance experiment, and its trees work directly with TreeSHAP. The
random forest is close enough that the choice is not a landslide; the logistic model is kept as the
interpretable baseline and as the calibration warning.

**Evidence.** `models/cv_results.json`; [`06_modeling.md`](06_modeling.md) §11;
[`report.md`](report.md) §6.1.

**If pressed.** The margin over the forest is **0.0114** PR-AUC, and the tuned gain over the untuned
XGBoost is **+0.00165 ± 0.00286** — smaller than its own error bar. The model is chosen because it is the
best ranking model and not worse on calibration, not because tuning produced a dramatic win.

### Q22. Why did the imbalance treatment make things worse?

**Answer.** Three arms plus a control, one estimator and one fold split: `weights`
(`scale_pos_weight = 7.548334`) scored **0.4420 ± 0.0134** PR-AUC, the untreated control **0.4444 ±
0.0131**, and SMOTE **0.4152 ± 0.0234**. SMOTE is worse than doing nothing in **0/5** folds on all three
metrics — **−0.0268** PR-AUC (**−6.07%**), **−0.0194** ROC-AUC (**6.8 SE**) — and about three times
slower. The class weight buys nothing on ranking either (**+0.0023** in the control's favour, **5/5**
folds, identical ROC-AUC) and **halves** calibration: Brier **0.1637** against **0.0821**, mean predicted
probability **0.3892** against a **0.1170** base rate. The untreated model is the control, and it wins.

**Evidence.** `models/imbalance_results.json`; [`06_imbalance.md`](06_imbalance.md);
[`report.md`](report.md) §6.2.

**If pressed.** F1 at a fixed 0.5 cut **reverses** the ordering: the weighted arm looks **11.9 points**
better at 0.5 and is behind at the honest leave-one-fold-out cut. That is the same trap as accuracy, and
it is why every threshold metric in §7 is reported at an out-of-fold cut.

### Q23. What did tuning buy, and is it worth it?

**Answer.** `RandomizedSearchCV` searched six axes, **40 of 4,800** points, scored on average precision
over the project's own folds. The winner was **300 rounds, learning rate 0.08, depth 5,
`min_child_weight=10`, `subsample=1.0`, `colsample_bytree=0.6`**. Untuned PR-AUC was **0.4444 ± 0.0131**;
tuned **0.4460 ± 0.0110**; the gain is **+0.00165 ± 0.00286**, **4/5** folds, or **0.58 SE**. **14 of 40**
candidates sit within one SE (**0.0049**), and the configuration that had been shipping before tuning was
**4th of 40** at **0.4449**. The honest reading is that hyperparameters are not where this problem's
performance lives; the tuned point is kept because it is not *worse*, not because it is meaningfully
better.

**Evidence.** `models/best_params.json`; [`report.md`](report.md) §6.3; [`06_modeling.md`](06_modeling.md)
§10.

**If pressed.** `best_score_` is **never quoted** anywhere in the project, because it is a selection-biased
number. The table is a **refit**, and its per-fold scores equal the search's own split scores to
**0.0e+00**. The feature set and the imbalance treatment were held out of the search on purpose.

### Q24. How is the final artifact pinned and checked?

**Answer.** `models/final_model.joblib` is **180,520 bytes**, sha256 `7a40752f…`, a two-step pipeline
(preprocessor + `XGBClassifier`) fitted in **0.8 s** on all 36,168 rows, with **no threshold baked in**. It
is rebuilt from `models/best_params.json` through `src.tune.make_estimator` rather than retyped, so the
parameters cannot drift from the record that justified them. Eleven checks pass, including a reload in a
**separate Python process** (drift **0.0e+00** over 1,000 rows), a bit-identical in-memory reload, finite
probabilities in **[0.005045, 0.994955]**, determinism, **300** boosted rounds confirmed, and a source scan
proving no held-out load.

**Evidence.** `models/final_model.json`; [`report.md`](report.md) §6.4; `src/final.py`.

**If pressed.** The demo does not trust the path: `src/demo.py` refuses to load the artifact unless its
sha256 and size match the record, so a re-fitted model cannot be demonstrated under T31's name.

---

## 7. Evaluation and validation

### Q25. Why is PR-AUC the headline rather than ROC-AUC or accuracy?

**Answer.** The positive class is **11.70%** of the file, and the campaign's job is to find the rare
positives. PR-AUC concentrates on that class; ROC-AUC can look strong while the precision at the operating
cut is poor, and accuracy is dominated by the negative class. The shipped model reaches **0.4460** PR-AUC
in cross-validation and **0.4616** held out, against a **0.1170** base rate; the dummy floor is **0.1170**,
the logistic baseline **0.4008**, and the random forest **0.4346**. ROC-AUC is reported too — **0.8052**
held out — but it is not the metric that decides the cut.

**Evidence.** `models/metrics_cv.json`, `models/holdout_eval.json`;
[`report.md`](report.md) §7.1; [`07_threshold.md`](07_threshold.md) §2.

**If pressed.** PR-AUC is threshold-free but not assumption-free: it still treats the positive class as the
one that matters. For a capacity-limited campaign, that is the right assumption; for a fairness audit, it
would not be enough on its own.

### Q26. What are the held-out results, and did the model generalise?

**Answer.** The artifact was **loaded, not refitted** — **0** fitting calls, asserted by a source scan —
and scored once on the **9,043** rows (**1,058** subscribers, **11.6997%**). Held-out PR-AUC is **0.4616**
against a cross-validated **0.4460 ± 0.0110**; ROC-AUC **0.8052** against **0.7990 ± 0.0086**; Brier
**0.0804** against **0.0818**; mean prediction **0.1151** against **0.1166**. The PR-AUC is **+0.0155**
above the CV mean, or **1.41** CV standard deviations, but a bootstrap over these same 9,043 rows puts the
estimate in **[0.4304, 0.4969]** with a standard deviation of **0.0169**, which is **1.53×** the CV sd.
Divided by the bootstrap sd, the gap is **0.92** standard deviations, and **15.4%** of resampled draws land
below the CV mean. The verdict is **no measurable gap**: the model generalises, and the held-out number is
the better estimate of the same quantity, not evidence of improvement.

**Evidence.** `models/holdout_eval.json`; [`report.md`](report.md) §7.2;
[`06_modeling.md`](06_modeling.md) §11.

**If pressed.** "Generalises" here means "to unseen clients of this campaign", not "to next year". The
held-out half is from the same 2008–2010 run, and the split is not temporal (Q36).

### Q27. Why is the threshold 0.0750 rather than the empirical peak?

**Answer.** The cut is **not fitted**. It is `C / V = 0.0750`, imported from the cost model, chosen on the
training half from out-of-fold scores, and only then checked on the held-out half. The empirical argmax is
**0.0821**, worth **388.50 EUR (0.883%)** more on the training half — and it is a number chosen by looking
at the data it then scores. The price gives up **0.9%** of the money to avoid fitting the cut to the
answer, and it sits only **0.0004** above the bottom edge of the **1%-of-peak plateau**
(**0.0746–0.1016**), which makes it the most aggressive cut still within 1% of the best. Both empirical
argmaxes land to its right — training **0.0821**, held out **0.0909** — because the curve rises steeply and
falls shallowly. Neither is adopted.

**Evidence.** `models/threshold_choice.json`; [`07_threshold.md`](07_threshold.md) §3–§4;
[`report.md`](report.md) §7.3.

**If pressed.** Held out, the price nets **10,988.50 EUR** from **3,621** calls, against **9,481.50** at
the F1 cut, **4,546.00** at 0.5, and **7,595.50** calling everyone. The 0.5 cut posts a **765%** ROI and
still loses money in absolute terms; ROI favours small, precise campaigns, and the objective here is net
value under a call-capacity constraint.

### Q28. What does the lift analysis show, and where does the model stop working?

**Answer.** Each half is measured against its own base rate. The top decile lifts **4.3652×** on training
and **4.5100×** held out, and supplies **43.65%** / **45.09%** of subscribers; the top two deciles supply
**61.76%** / **62.29%**. D15's ">2×" target is cleared by more than a factor of two. But **decile 3 is
already below random** — **0.92×** on training and **0.88×** held out — and decile 10 is **0.18×**. Past
the best fifth of the list, the model's message is mainly "do not call". A shuffled-score self-test reruns
the whole table on noise and finds lifts of **0.950–1.108** (mean **1.000001**), gain within **0.0060** of
the diagonal, and savings within **218 calls** of zero, so the table is measuring the model, not the
arithmetic.

**Evidence.** `models/lift_results.json`; `reports/lift_table.csv`; [`07_lift.md`](07_lift.md);
[`report.md`](report.md) §7.4.

**If pressed.** **Calls saved is not the objective.** It peaks at **18.89%** coverage (**15,188** calls
saved) *before* the cut, because its marginal step crosses the **list average** rather than the price. The
**8,033** calls between that peak and the cut still convert at **8.96%** against the **7.50%** price, worth
**+2,350.50 EUR**. Pruning to the peak is an optimisation that loses money.

### Q29. Why is accuracy reported at all, and what does it hide?

**Answer.** It is reported **once**, because it is the metric everyone asks for, and it is labelled as a
trap. Held out: always-"no" scores **88.3003%** and finds nobody; 0.5 scores **89.6052%** and misses **801**
of the **1,058** subscribers; the price cut scores **66.42%** and finds **77.60%**. The most accurate cut on
these rows is **89.76%** at **0.478284**, which is why 0.5 is carried as a convention rather than a target.
Accuracy hides the only thing the campaign cares about: how many subscribers are reached for the calls
spent. A metric that ranks "call nobody" above "call the right 40%" is not a metric to lead with.

**Evidence.** `models/holdout_eval.json`; [`report.md`](report.md) §7.6;
[`07_threshold.md`](07_threshold.md) §5.

**If pressed.** Balanced accuracy is the fairer summary — **71.27%** at the price cut against **61.28%** at
0.5 — but it is still a threshold metric, and the project's decision is made on net value and recall, not
on either accuracy.

---

## 8. Business insights and recommendations

### Q30. What are the top SHAP drivers, and why is `contact = unknown` first?

**Answer.** TreeSHAP on the shipped artifact has additivity **5.010e-06** and contributions equal to
xgboost's own `pred_contribs` to **0.0**. Contributions are aggregated by **signed sum within each one-hot
block**, so the twelve feature values stay additive; the naive per-column alternative inflates `month` by
**+58.52%**. The top features by share of importance are `contact` **22.14%**, `month` **17.55%**, `day`
**10.60%**, `balance_arsinh` **8.49%**, `housing` **8.05%**, and `poutcome` **7.97%**. Campaign-side
features carry **55.75%** of the importance against **44.25%** for the client's own, which is why most of
the playbook is about *how* the bank calls, not *whom* it calls.

**Evidence.** `models/shap_global.json`; [`08_explainability.md`](08_explainability.md);
[`report.md`](report.md) §8.1.

**If pressed.** `contact = unknown` covers **28.72%** of clients and contributes **−1.0150** log-odds, the
largest single-column effect of the forty; those clients subscribe at **4.11%** against **14.83%** for
`cellular`. The model is not saying they are bad prospects; it is saying it **cannot tell**. That is a data
quality finding, not a customer judgement.

### Q31. What do the three exemplar clients show?

**Answer.** The cells are defined against the **price**, not 0.5: in-sample **TP 3,523 / FP 11,059 / FN
708 / TN 20,878**. The exemplar rule was pre-registered — the client closest to its cell's **median**
score, ties by lowest row index — so the illustrations are typical members, not worst cases. **A true
positive** (row 19,263, p = **0.3819**) shows the tension in one row: `month = oct` **+1.8824**, nearly four
times the margin the client finished with, against `contact = unknown` **−0.6989** — the model's most
important feature taking most of it back. The client subscribed anyway. **A false positive** (row 23,374,
p = **0.1133**) is the finding: its twelve contributions sum to **−0.0109**, a completely average client.
The reference is **11.4361%** and the cut is **7.50%**, so an average client clears the cut; the false
positive is the **modal** cell (**30.58%** of the file) by construction. **A false negative** (row 10,760,
p = **0.0550**) misses by **0.3323** log-odds: `balance_arsinh = 0.00` (**−0.3502**) and `month = jul`
(**−0.3121**) outweigh `contact = cellular` (**+0.2329**). It is a ranking miss, not a category error.

**Evidence.** `models/shap_local.json`; `reports/figures/10_shap_local.png`;
[`report.md`](report.md) §8.2.

**If pressed.** Being called and being wrong needs no story at this price: the cut is below the base rate,
so the modal client is expected to clear it. The false positive is arithmetic, not a model failure.

### Q32. What are the eight recommendations, and which ones are prohibitions?

**Answer.** The playbook holds eight recommendations as records, each with `action`, `evidence`,
`expected`, `check` and `falsifier`:

| # | Recommendation | Headline |
|---|---|---|
| R1 | Make `contact` a required field | **28.72%** of clients at **−1.0150** |
| R2 | Call the list in model order, stop at **0.0750** | **+13,253.00 EUR** on **21,302** fewer calls |
| R3 | Do not stop where calls saved peaks | **8,033** calls at **8.96%** = **+2,350.50 EUR** |
| R4 | Never feed call `duration` into the model | **+0.1788** PR-AUC, **20,340.00 EUR**, dead at dial time |
| R5 | Establish the value of a subscription | every euro is linear in **V = 20.00 EUR** |
| R6 | Track precision and net value, never accuracy | **88.3003%** from a model that finds nobody |
| R7 | Keep the twelve-feature XGBoost | **0.4460** vs **0.4346** and **0.4008** |
| R8 | Use `month` to schedule, never to predict a client | **17.55%** of importance, 2008–2010 confound |

**R4 and R7 are prohibitions**, and a prohibition is only credible next to the number it declines: R4
carries the **20,340.00 EUR** the bank would bank by using a column that does not exist at decision time.
The order is by leverage, not importance: R1 first because one required field changes how nearly three
clients in ten are scored; R5 second because it decides the cut, and every euro depends on it.

**Evidence.** `models/playbook.json`; `reports/playbook.csv`;
[`08_recommendations.md`](08_recommendations.md); [`report.md`](report.md) §8.3.

**If pressed.** R3 is written as a **refusal**, not an action: calls saved peaks at **18.89%** coverage,
but the **8,033** calls between that peak and the cut still convert at **8.96%** for **+2,350.50 EUR**.
Held out, the same block is only **1,547** calls at **7.76%** for **+79.50 EUR**, so the recommendation
states that it may flip at scale rather than pretending the training number replicates.

### Q33. What is the model worth, and what does that number not mean?

**Answer.** On the training half at the price cut, the model calls **14,866** clients (**41.10%** of the
list), reaches **3,296** subscribers (**77.90%**), misses **935**, and nets **43,621.00 EUR**, against
**30,368.00 EUR** for calling everyone — a gain of **+13,253.00 EUR** on **21,302 fewer calls**. Held out,
the same policy calls **3,621** clients (**40.04%**), reaches **821** subscribers (**77.60%**), and nets
**10,988.50 EUR** against **7,595.50 EUR** for calling everyone. The top decile alone converts at
**4.5100×** the base rate.

It does **not** mean the bank should cut campaign spend: the cut is below the base rate, so the campaign
already pays, and the model improves the allocation of a fixed budget. It does **not** mean the euro totals
are the bank's: C and V are assumptions, and every total is linear in V. It does **not** mean the gain
replicates next year: the held-out half is the same campaign.

**Evidence.** `models/threshold_choice.json`, `models/holdout_eval.json`,
`models/lift_results.json`; [`report.md`](report.md) §0, §7.3 and §8.4.

**If pressed.** The correct comparison is "call everyone", not the dummy. Calling everyone already nets
**30,368.00 EUR** on training; the model's **43,621.00 EUR** is an improvement on a paying campaign, not a
rescue of a failing one.

---

## 9. Documentation and reproducibility

### Q34. How do you know the numbers are real?

**Answer.** Three layers, applied to every task. **In-module checks:** each module validates its own output
before writing it, prints its checks, and exits non-zero on disagreement — 12 to 17 checks per module.
**Independent verifiers:** throwaway scripts that import neither the module nor its library recompute every
figure by a different route — rank-based AUC, step-block average precision, `searchsorted` counts, and
xgboost's `pred_contribs` instead of `shap` — and trace every number in the prose to a record. **Numeric
traces:** every decimal in a document must exist in a `models/*.json` record, a `reports/*.csv` column, or a
supporting document. The report's **497** tokens, the deck's **157** and the playbook's **63** all pass.

The verifiers found real defects rather than confirming the work: a `rank_auc` that masked the wrong order
(**0.7783** against **0.8052**), a Brier score computed in `float32`, scikit-learn's `std_test_score` using
`ddof=0`, a settings check comparing `nan != nan`, a mislabelled `subscribers_missed`, and three prose
figures with no record behind them.

**Evidence.** [`report.md`](report.md) §9.3; `PROGRESS.md`; `src/report.py`, `src/deck.py`,
`src/playbook.py`.

**If pressed.** A token-level trace admits a number that appears *anywhere* in the documents, so a figure
lifted from the wrong column can still pass. That is why the deck additionally checks each slide against the
**specific** record it cites (D62), and why the report's provenance check is paired with a structure and
coverage check.

### Q35. Can someone reproduce this from a clean clone?

**Answer.** Yes, and it was tested rather than asserted. The pushed repository was cloned into a directory
outside the working tree, with **105 tracked files** and no `.venv` or bytecode; a fresh **Python 3.11.9**
virtual environment was created and all **14 pins** installed from `requirements.txt`; `src.config.ROOT`
resolved to the clone. The **22 documented commands** were run in README order, and all exit 0 after the
two order fixes the clone found. `src.report` passed **30/30**, `src.deck` passed, `src.demo` passed
**22/22**, and `src.readme` passed **34/34** in the clone. All **5** joblibs, all **4** processed CSVs, all
**6** report CSVs and **7 of 8** figures came back byte-identical; the eighth was the SHAP beeswarm, whose
jitter was unseeded and is now seeded. The full sequence took about a quarter of an hour on this machine;
the 12 records that store `elapsed_seconds` summed to under ten minutes.

**Evidence.** `PROGRESS.md` Session 46 (the checklist and the four defects the clone found); `README.md`
§Reproduce every number; `models/readme.json`.

**If pressed.** One clone on one Windows machine at one commit is evidence of determinism on that stack,
not a promise across every OS and BLAS build. The two order defects — `duration_compare` running before
`tune` and `threshold`, and the report quoting machine-specific seconds — were fixed in the repository, not
worked around in the clone.

### Q36. What are the limitations, and which one would you fix first?

**Answer.** Seven, stated plainly:

1. **The prices are assumptions** — C = 1.50 EUR and V = 20.00 EUR are not the bank's numbers; every euro
   total is linear in V and the break-even is C/V.
2. **The held-out half is the same campaign** — "generalises" means unseen clients of this campaign, not
   next year.
3. **The split is stratified-random, not temporal** — a test client may sit next to a train client.
4. **The tuned gain is smaller than its error bar** — **+0.00165 ± 0.00286**, **0.58 SE**; the shipped model
   is not meaningfully better than the untuned one, only not worse.
5. **`month` carries a confound** — **17.55%** of importance sits on a calendar spanning a recession; its
   marginal rates are measured, not causal.
6. **The counterfactual is unobserved** — nothing here answers "what would happen if we called the 935
   missed subscribers?".
7. **Fairness is not audited** — segment-level precision and recall exist; a proper audit across age or job
   groups does not.

I would fix the **temporal split** first, because it is the test that answers the operational question
("does this survive the next campaign?"), and it is a change to validation rather than to the model. The
**fairness audit** would be second, because compliance's requirement is only partly met.

**Evidence.** [`report.md`](report.md) §10; [`08_recommendations.md`](08_recommendations.md) §4;
[`01_problem_definition.md`](01_problem_definition.md) §5.

**If pressed.** The project also makes six explicit non-promises: no claim that filling `contact` raises
conversion; no claim that the campaign should shrink; no causal reading of `month`; no promise that the
**+2,350.50 EUR** block replicates (held out it is **79.50 EUR**); no fairness claim beyond segment-level
metrics; and no claim that the prices are the bank's.

### Q37. What does the interactive demo add, and what does it refuse to do?

**Answer.** `app/app.py` is a thin Streamlit UI over `src/demo.py`. It asks for **11 business facts**,
derives the **12 shipped features** through the project's own functions, scores `models/final_model.joblib`
only after its sha256 and size match the record, explains the score with signed-sum SHAP, and compares the
probability with **0.0750**. It refuses to ask for `duration`, `pdays` or `y`; it refuses to treat **0.5**
as the decision rule; and it labels **C = 1.50 EUR** and **V = 20.00 EUR** as assumptions. The default
client scores **7.22%**, the true-positive exemplar **38.19%**, and the false-negative **5.50%**; the core
passes **22** headless checks, and the app was rendered headlessly through Streamlit's `AppTest`.

**Evidence.** `models/demo.json`; `src/demo.py`; `app/app.py`; `README.md` §The interactive demo.

**If pressed.** The UI is deliberately thin so the logic can be tested without a browser. The demo also
does not refit: it loads T31's artifact by digest, so a new model cannot be demonstrated under the old
model's name.

---

## 10. Five numbers to know cold

| Number | What it is | Where it lives |
|---|---|---|
| **0.0750** | the price cut, `C / V = 1.50 / 20.00` | `models/threshold_choice.json` |
| **0.4460** | cross-validated PR-AUC of the shipped model | `models/metrics_cv.json` |
| **0.4616** | held-out PR-AUC, the better estimate of the same quantity | `models/holdout_eval.json` |
| **77.60%** | held-out recall at the price cut | `models/holdout_eval.json` |
| **10,988.50 EUR** | held-out net value at the price cut | `models/holdout_eval.json` |

**The three prohibitions, in one line each:** `duration` is never an input; accuracy is never a headline;
`month` is never read as a cause. **The scope sentence:** the model converts a fixed call budget, and is
not evidence that the bank should cut campaign spend.

---

## 11. Rehearsal notes

- **Open with the decision, not the model.** "The bank has more clients than agents; this ranks whom to
  call today, and prices the call at 7.50%."
- **Say what the model is worth, then what it is not.** The euro gain is an allocation gain on a campaign
  that already pays; the prices are assumptions.
- **Use the traps as proof of judgement.** 0.5 reverses the model ranking; accuracy prefers "call nobody";
  ROI prefers a tiny campaign; calls saved peaks before the cut. Each is a place where the naive metric
  loses money.
- **If you do not know a number, open the record.** The answer is always a file path away, and the project
  is designed so that the file wins the argument.
