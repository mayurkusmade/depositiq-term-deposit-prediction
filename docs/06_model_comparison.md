# DepositIQ — Model Comparison and Selection

**Rubric:** Criterion 6 — Model Selection & Implementation (15 marks)
**Task:** T31 | **Status:** complete | **Date:** 2026-10-29
**Artifact:** `models/final_model.joblib` | **Model card:** `models/final_model.json`
**Sources of every number here:** `models/cv_results.json` (T26–T28), `models/imbalance_results.json` (T29),
`models/best_params.json` (T30), `models/final_model.json` (T31). Nothing below was typed from a rendered
table.

---

## 1. What is being compared

Seven builds of one problem, all on the **same 36,168 training rows**, the **same five stratified folds**
(materialised once, seed 42), the **same twelve features** (40 matrix columns), and the **same scoring
function**. That is what makes the table a comparison rather than seven experiments that happen to look
alike.

| # | build | what it is | record |
|---|---|---|---|
| 1 | `dummy_prior` | predicts the training fold's positive rate for every row | T26 |
| 2 | `logreg_balanced` | the task's required baseline, `class_weight="balanced"` | T26 |
| 3 | `random_forest` | 300 trees, `min_samples_leaf=10`, `balanced_subsample` | T27 |
| 4 | `xgboost_weighted` | 400 rounds, depth 3, `scale_pos_weight=7.548334` | T28 |
| 5 | `xgboost_untreated` | the same settings with the weight removed — T29's control | T29 |
| 6 | `xgboost_smote` | SMOTE inside the pipeline, no weight | T29 |
| 7 | **`xgboost_tuned`** | **300 rounds, lr 0.08, depth 5, `min_child_weight=10`, `subsample=1.0`, `colsample_bytree=0.6`, untreated** | **T30 — ships** |

**No held-out number appears in this document.** The test half has not been scored by anything; T32 owns
that single measurement. Every figure below is 5-fold cross-validation on the training half.

---

## 2. The table

PR-AUC is the metric of record (D04): with an 11.70% positive rate, accuracy ranks the models by how well
they say "no", and the × column is the honest reading of a PR-AUC — the multiple of the base rate.

| build | PR-AUC | ROC-AUC | ×base rate | Brier | mean p | accuracy @ 0.5 |
|---|---|---|---|---|---|---|
| `dummy_prior` | 0.1170 ± 0.0001 | 0.5000 ± 0.0000 | 1.00× | 0.1033 | 0.1170 | 88.30% |
| `logreg_balanced` | 0.4008 ± 0.0134 | 0.7640 ± 0.0114 | 3.43× | 0.1852 | 0.4161 | 75.21% |
| `random_forest` | 0.4346 ± 0.0146 | 0.7906 ± 0.0089 | 3.72× | 0.1403 | 0.3404 | 83.91% |
| `xgboost_weighted` | 0.4420 ± 0.0134 | **0.7993** ± 0.0089 | 3.78× | 0.1637 | 0.3892 | 81.56% |
| `xgboost_untreated` | 0.4444 ± 0.0131 | **0.7993** ± 0.0079 | 3.80× | 0.0821 | 0.1168 | 89.43% |
| `xgboost_smote` | 0.4152 ± 0.0234 | 0.7799 ± 0.0105 | 3.55× | 0.1221 | 0.2967 | — |
| **`xgboost_tuned`** | **0.4460 ± 0.0110** | 0.7990 ± 0.0086 | **3.81×** | **0.0818** | **0.1166** | 89.38% |

Base rate **0.116982** (4,231 of 36,168). The accuracy column is reported because it is the number a
non-technical reader reaches for first, and it is included deliberately to be *read against* Brier and
mean p: the dummy wins nothing but is within seven points of the best accuracy, which is the whole reason
this project does not select on accuracy.

`xgboost_smote`'s accuracy is `—` because T29 scored F1 at a leave-one-fold-out cut rather than at 0.5, and
filling in a 0.5 number here would re-introduce the artefact that measurement rejected.

---

## 3. Reading the table

**Ranking.** The three tree ensembles are within **0.0114** of each other on PR-AUC and the linear model
sits 0.034 behind them; the tuned XGBoost's lead over the forest is **0.0114** with a narrower sd. T30
already reported the uncomfortable part: the tuned model beats T28's configuration by **+0.00165 at 0.58
standard errors**, so most of the gap between "the shipped model" and "the runner-up" is noise. What is
*not* noise is the gap between any of them and the alternatives: SMOTE loses to doing nothing in 0/5 folds
by 6.8 standard errors, and the linear model loses to the forest by more than two standard errors of the
paired difference.

**Calibration.** This column decides more than the ranking does. The three weighted models predict the
average client at **0.39, 0.34 and 0.42** when the true rate is **0.1170** — they are 2.9–3.6× too
optimistic, by construction, because weighting rewrites the loss. The two untreated XGBoost builds land at
**0.1168** and **0.1166** against a base rate of **0.1170**, which is why their Brier scores are
**0.0821** and **0.0818** instead of 0.16. T29 established this; T30 reproduced it at a better point in the
parameter space; this table is where the two findings become one number a reader can act on.

**Reproducibility.** The shipped build's parameters exist in `models/best_params.json`, its CV numbers in
the same file, and its fitted form in an artifact whose checksum is recorded. A model justified by a
sentence in a report cannot be rebuilt; this one can, and a check confirms the rebuilt file predicts
bit-identically.

---

## 4. The decision

**Ship `xgboost_tuned` — XGBoost, untreated, at T30's tuned point.** Three reasons, in the order they
should be argued:

1. **It is the best ranker of the seven on the metric of record** (PR-AUC **0.4460**, 3.81× the base rate)
   and ties the best ROC-AUC. The margin over T28's build is small and is reported as small, but the
   alternative is not better on any metric.
2. **It is the only build whose probabilities can be read as probabilities.** Mean prediction **0.1166**
   against a **0.1170** base rate, Brier **0.0818**. D17's **€1.50**-per-call, **€20**-per-deposit break-even
   is a statement about a probability — a call pays for itself above a 7.50% subscription rate — and it can
   only be applied to a score that means what it says. T34 will still test the calibration formally and
   choose the operating threshold; the point here is that the artifact does not *require* rescaling to be
   usable.
3. **The treatment it omits is the one T29 measured as costing more than it bought.** The weight gained
   **+0.0023** PR-AUC at **1.11 SE against it** and lost **half the Brier score**; at the tuned point the
   gap widened to **−0.00291 PR-AUC in 0/5 folds** with Brier **0.1535** against **0.0818**. Shipping the
   weighted build would be shipping the configuration that T29 and T30 both measured as worse.

Three things this document must not be read as saying:

* **"Tuning found a much better model."** It did not. The tuned model is **0.00165** better at **0.58 SE**,
  and T28's hand-picked configuration was **4th of 40** candidates. The justification for this build is
  that it is no worse, is better calibrated than the weighted alternative, and is reproducible — not that
  hyperparameters moved the needle.
* **"The forest's better calibration is a reason to ship the forest."** The forest's Brier **0.1403** beat
  the *weighted* XGBoost's **0.1637** in T27, which was a fact about that build. Both untreated XGBoost
  builds now beat it (**0.0821**, **0.0818**), so the comparison that mattered has been superseded, not
  ignored.
* **"0.4460 is the model's performance."** It is a 5-fold CV number on the data the configuration was
  chosen on. It is optimistic by construction (§6), and T32's held-out score is the number to quote.

---

## 5. The artifact

| property | value |
|---|---|
| file | `models/final_model.joblib` |
| size | **180,520 bytes** (joblib `compress=3`) |
| sha256 | `7a40752f75a264a0933bae2dc42feb28385b6ab75de36c32782c8a7ab8f2ed5f` |
| structure | `Pipeline(preprocess=ColumnTransformer, model=XGBClassifier)` — 2 steps, no resampler |
| fitted on | all **36,168** training rows (4,231 positives), `fit` **0.8 s** |
| rounds / matrix | **300** boosted rounds over **40** columns built from **12** features |
| input | a `DataFrame` with the twelve shipped feature columns |
| output | `predict_proba(...)[:, 1]` = P(subscribe) |
| threshold | **none baked in** — T34 applies D17's 7.50% break-even after recalibration |

The pipeline carries its own preprocessing, so a caller passes raw **engineered** columns and never builds
a matrix:

```python
import joblib
from src import features, select

pipeline = joblib.load("models/final_model.joblib")
frame = features.load_engineered("train")            # or any frame run through engineer()
scores = pipeline.predict_proba(select.select_features(frame))[:, 1]
```

`select.select_features` is not optional: the transformer was fitted on those twelve columns in that order,
and `handle_unknown="ignore"` protects against unseen *levels*, not against a missing *column*.

---

## 6. What this step does not claim, and what would change the choice

* **No held-out number.** The training half was used to choose the configuration and to fit the artifact;
  the test half has not been touched. T32's single held-out evaluation is the only number that can
  distinguish "0.4460 is real" from "0.4460 is what 40 candidates look like when you pick the best one".
* **The CV figure is optimistic, deliberately and visibly.** T30 selected on these five folds; 14 of 40
  candidates sit within one standard error of the winner. A held-out PR-AUC in the 0.43–0.45 range is the
  expected outcome, not a disappointment.
* **Calibration is measured, not guaranteed.** T29/T30 show the mean prediction lands on the base rate
  across five folds; a single training-half fit can drift. T34's job is to test that formally (reliability
  curve, Brier decomposition) and to fix the operating point.
* **The `duration` exclusion stands.** The model cannot see call length (D05, T36), so it can rank clients
  *before* a call but cannot be used to decide whether to continue one. T36 prices that gap.
* **What would reopen this choice:** a held-out PR-AUC below the forest's 0.4346, a reliability curve that
  fails on the held-out half, or a T32 finding that the tuned configuration's advantage does not survive
  out of sample. Any of those sends the decision back to §4 with evidence rather than re-opening it by
  preference.

---

## 7. Verification

`python -m src.final` exits **0** with **11/11** checks in **3.0 s**:

* the artifact carries **exactly** the six parameters in `models/best_params.json`, and they are the
  parameters **applied to the fitted estimator** — a file cannot drift from the record that justifies it;
* the settings the search never varied survived the pickle (`tree_method="hist"`,
  `eval_metric="aucpr"`, seed 42, `n_jobs=1`, `scale_pos_weight=1.0`) — a model reassembled from the same
  numbers with a different `tree_method` is a different model;
* the pipeline has exactly two steps and **300** boosted rounds, matching the selected `n_estimators`;
* the fitted preprocessing is still the 40-column T24 matrix with no `duration`, `pdays` or target;
* the reloaded pipeline reproduces the in-memory one **bit-identically** (max drift **0.0e+00** over 1,000
  rows);
* **a separate Python process** loads the file and scores 1,000 rows with max drift **0.0e+00** — the check
  that separates "the object is in memory" from "the file works", and the one that caught the first
  version's failure (a temp script cannot `import src`; the fix runs the probe with `-c` from the project
  root);
* `predict_proba` returns finite probabilities in **[0.005045, 0.994955]** whose rows sum to 1 within
  **0.0e+00**, for one row or a thousand;
* predictions are deterministic across calls;
* the fit used the training half only, and a source scan proves this module contains no test-half load —
  a scan that first **reported itself**, because the literal it searched for was written in the check;
* the card's sha256 and byte count describe the file on disk;
* the card's CV numbers equal T30's tuned arm to **1e-12**, so the document and the artifact quote one
  source.

*Interpretation for the viva:* "Seven builds, same rows, same folds, same features, same scoring function.
The shipped model is XGBoost at a tuned point: PR-AUC 0.446 — 3.8 times the base rate — with the best
ROC-AUC of the field and, unusually for this project, probabilities that land on the true base rate without
recalibration, which is what lets a cost-per-call threshold be applied at all. I did not ship the version
with the class weight, because I had measured that the weight buys no ranking and costs half the Brier
score. And I am careful about what the number is: it is cross-validated on the same folds I chose the
configuration on, so it is optimistic, and the honest test is the held-out evaluation I have deliberately
not run yet. The artifact itself is a 180 KB file with a recorded checksum that reloads in a fresh
interpreter and predicts bit-identically — so the model is a thing that exists, not a paragraph."
