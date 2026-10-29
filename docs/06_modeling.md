# DepositIQ — Model Selection Log

**Rubric:** Criterion 6 — Model Selection & Implementation (15 marks)
**Task:** T26 (this section); extended by T27–T29, consolidated in T31 | **Status:** T26 complete
**Date:** 2026-10-29
**Source of every number here:** the run of `python -m src.train`, recorded in
`models/baselines_cv.json` and printed by the module. Nothing below was computed by hand (D06).

---

## 1. What was trained, and what it had to beat

Two estimators, in **one** `Pipeline` whose first step is T24's `ColumnTransformer`, on the **12
features / 40 columns** T25 kept (D44):

| estimator | parameters | role |
|---|---|---|
| `dummy_prior` | `DummyClassifier(strategy="prior")` | the floor |
| `logreg_balanced` | `LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=42)` | first real model |

Training half only: **36,168 rows, 4,231 positives, 11.6982%**. The 9,043 held-out rows were never
loaded — they belong to T32, and a baseline that has seen them is not a baseline.

`logreg_balanced` uses `class_weight="balanced"` because the required baseline does, and because T29
exists to test that choice against SMOTE rather than to assume it. The scaling it needs comes from
T24's `StandardScaler` group, so no model file contains a preprocessing decision.

---

## 2. The floors are derived, not quoted

The `DummyClassifier` exists to fix the two floors from T15's EDA (D28) in the *same folds* as
everything else, rather than quoting a number from an earlier session:

| floor | value | why it is that value |
|---|---|---|
| ROC-AUC | **0.500000** | one constant score per row; no pair can be ranked correctly |
| PR-AUC | **0.116982** ± 0.000059 | a constant score's precision-recall curve sits at the base rate |
| accuracy | **88.3018%** | predicting `no` for everyone is right 88.3% of the time (D04) |
| Brier | **0.103297** | for a constant score `p`, `mean((y − p)²) = q − 2pq + p²` |

The module **re-derives** all four arithmetically in every fold rather than asserting them: ROC-AUC
must be exactly 0.5, PR-AUC must equal the scored fold's rate to 1e-12, the mean score must equal the
*fitting* fold's rate, and the Brier score must equal the expansion above. Two denominators are
involved and confusing them is easy — PR-AUC and accuracy depend on which rows are being **scored**,
while the constant score itself is the rate in the rows the dummy was **fitted** on. The first version
of this check compared both against the scored rate and reported four "failures" of 4.2 × 10⁻⁵ that
were the check describing the wrong fold.

---

## 3. The baselines, 5-fold stratified CV

| estimator | ROC-AUC | PR-AUC | PR-AUC ÷ base rate | Brier | mean p | accuracy @ 0.5 |
|---|---|---|---|---|---|---|
| `dummy_prior` | 0.5000 ± 0.0000 | 0.1170 ± 0.0001 | 1.00× | 0.1033 | 0.1170 | 88.3018% |
| `logreg_balanced` | **0.7640 ± 0.0114** | **0.4008 ± 0.0134** | **3.43×** | 0.1852 | 0.4161 | 75.2129% |

The sd is the **sample** sd over the five folds (`ddof=1`), stated because `ddof=0` and `ddof=1`
differ by 12% at n = 5 and a table should not leave that to be guessed.

Per fold, because the fold is the unit of evidence and a mean can hide a fold that failed:

| fold | positives | dummy PR-AUC | logreg PR-AUC | logreg ROC-AUC | iterations |
|---|---|---|---|---|---|
| 1 | 846 | 0.116948 | 0.393203 | 0.756884 | 42 |
| 2 | 846 | 0.116948 | 0.397146 | 0.779296 | 35 |
| 3 | 847 | 0.117086 | 0.385206 | 0.749395 | 39 |
| 4 | 846 | 0.116964 | 0.418816 | 0.768455 | 43 |
| 5 | 846 | 0.116964 | 0.409825 | 0.766155 | 40 |

The logistic baseline beats **both** floors in **every** fold — the spread across folds is 0.031
AUC on ROC and 0.034 on PR-AUC, and the worst fold (3) is still 3.29× the base rate. It also clears
D15's declared ROC-AUC target of **≥ 0.75** on the mean, which is a target met, not a result claimed:
D15's second target (>2× top-decile lift) is T35's and has not been measured.

---

## 4. Why there is no F1 column, and what the calibration diagnostics show

Both are threshold metrics, and `class_weight="balanced"` is **deliberately miscalibrated**. The
weights are 7.55 : 1 (0.8830 / 0.1170), so the fitted probabilities are pushed up: the model's mean
predicted probability is **0.4161** for a population whose base rate is **0.1170**, a factor of 3.56.

The consequence is visible in the Brier score, and it points the opposite way to every other column:

| | Brier |
|---|---|
| `dummy_prior` (a single constant) | **0.1033** |
| `logreg_balanced` (ROC-AUC 0.7640) | 0.1852 |

**The model that ranks far better is a worse probability forecaster than a constant.** That is not a
bug; it is what balancing the classes does to the calibration, and it is why:

* T26 reports **ROC-AUC and PR-AUC**, which are threshold-free, as the metrics the comparison uses.
* F1 and precision at 0.5 — or even at D17's 7.50% break-even — would measure the weighting rather
  than the model. At the 0.5 cut the balanced model flags a large share of the list; accuracy falls
  to 75.21% for exactly that reason.
* **T34 must recalibrate** (isotonic or Platt, fitted inside the fold) before D17's 7.50% threshold
  can be applied, or state that it is a *ranking* score being cut at a quantile instead of a
  probability being compared to a cost. Choosing the operating point is T34's job; this baseline does
  not pretend to have chosen one.

---

## 5. Does T25's reduction cost the linear model anything?

D44 dropped five features (`education`, `loan`, `default`, `balance_band`, `job`) because a 200-tree
forest could not distinguish their contribution from zero, and recorded that the decision was
**instrument-specific**. The obvious objection: a forest masks a correlated column, while a linear
model gives each one-hot level its own coefficient and would feel the loss. So it was measured — the
same estimator, the same five folds, the columns toggled:

| arm | features | columns | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| all 17 | 17 | 65 | 0.767926 | 0.402904 |
| shipped (D44) | 12 | 40 | 0.764037 | 0.400839 |
| **delta (shipped − all)** | −5 | −25 | **−0.003889 ± 0.002252** | **−0.002065 ± 0.002195** |
| folds improving | | | **0 / 5** | 1 / 5 |

Read against T25's own one-standard-error standard:

* **ROC-AUC: the loss is real.** −0.0039 against a fold sd of 0.0023 is 1.73 standard errors, and
  **0 of 5** folds improve — a one-sided sign test gives p = 1/32 = **0.031**. The reduction is not
  free for a linear model.
* **PR-AUC: the loss is inside the noise.** −0.0021 against a sd of 0.0022 is 0.94 standard errors,
  and one fold improves. This is the same "cannot be told from zero" verdict the forest reached.

So the honest summary is: **the forest said neutral, the linear model says a small but measurable
ROC-AUC loss (0.0039, or 0.5% relative) and no detectable PR-AUC loss.** D44 stands, for three
reasons that are stated rather than assumed:

1. The shipped model is **XGBoost** (D08), which is tree-based and therefore closer to the instrument
   that measured the decision than to the linear model that disputes it.
2. 25 fewer columns is **38.5%** of the matrix, and the price is 0.0039 AUC on a model that is not
   the shipped one.
3. The comparison is available to re-run in one command, on the final model, in T31. **If XGBoost on
   40 columns lands below this logistic baseline, that is the trigger to revisit** — and the trigger
   is written here, before the result exists, rather than after.

The shipped arm of this ablation *is* `logreg_balanced`, and a module check requires its five folds
to reproduce that run **bit-identically**. Without that, the ablation would be two experiments
compared and its delta would mean nothing.

---

## 6. What is held fixed, and how the module proves it

| element | how it is held fixed | check |
|---|---|---|
| rows | one load of `data/processed/train.csv`; the test half is never read | source scan + no `TEST_CSV` reference |
| folds | one `StratifiedKFold(5, shuffle=True, seed=42)`, materialised **once** and replayed into every estimator | re-derives the split from the seed and compares arrays |
| target | one `encode_target()` call, sliced per fold | every estimator's per-fold row and positive counts must agree |
| preprocessing | `preprocess.build_preprocessor(select.shipped_features())` per fold | step names, column lists and estimator class compared across arms; a fold's scaler must equal **that fold's** training mean (0.0e+00 error) and differ from the pooled one (4.74e-02) |
| metrics | one `fold_metrics()` function | both arms' numbers come from the same call path |
| environment | one process, one run | — |

The pooled-vs-fold scaler comparison is the one that matters most: a scaler fitted once before the
loop still *looks* exact, because it is exact — on the wrong rows. Matching the fold mean proves it
was fitted on the training side; differing from the pooled mean proves it was not fitted on
everything.

---

## 7. Verification

* `python -m src.train` exits **0** with **14/14** checks: fold determinism, target identity,
  pipeline structural identity, per-fold fitting, matrix width, prediction range, the dummy's four
  floors re-derived, the logistic model beating both floors in every fold, no leaky column, artifact
  round-trip, T25's frozen decision, convergence (`n_iter` 35–43 against `max_iter` 2000), and the
  reduction check reproducing the baseline exactly.
* **127 independent checks** in a separate script that refits everything from `train.csv` through a
  plain `read_csv` and recomputes every metric without importing `src.train`. It includes two
  hand-written implementations — a rank-based Mann-Whitney AUC and a threshold-step average precision
  — that are first checked **against scikit-learn to 1e-12** (including the all-tied constant-score
  case), because a checker that has never been shown to agree with the library is a checker of
  nothing.
* `models/baseline_logreg.joblib` reloads as a 2-step pipeline and reproduces a fresh full fit to
  1e-12; the dummy artifact predicts a constant.

**Runtime:** 3.8 s for both baselines, 5 folds each, plus the 10-fit reduction check. That is cheap
enough that T27–T31 can re-run the whole comparison on every change.

---

## 8. What T27–T31 inherit

* A model is an entry in `ESTIMATORS` plus a `MODEL_PATHS` line. It gets the folds, the preprocessing
  and the scorer unchanged.
* The floor to beat is **PR-AUC 0.1170 / ROC-AUC 0.5000**, not 0.5 PR-AUC.
* The number to beat so far is **PR-AUC 0.4008 (3.43×), ROC-AUC 0.7640**.
* Threshold metrics wait for T34's recalibration; imbalance variants are T29's to compare on
  **PR-AUC**, which is the metric that responds to the positive class.

*Interpretation for the viva:* "A linear model on twelve features ranks clients at ROC-AUC 0.764 and
finds the subscribers at 3.4× the base rate, against a floor of 0.5 and 1.0×. Its probabilities are
deliberately uncalibrated — it scores the average client at 0.42 when the true rate is 0.12 — so I
report ranking metrics here and recalibrate before applying the €1.50 break-even. And I checked the
one thing that could embarrass the feature selection: dropping five features costs a linear model
about four thousandths of an AUC on ROC, nothing measurable on PR-AUC, and the tree model that ships
didn't see the loss at all."
