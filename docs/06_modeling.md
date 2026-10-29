# DepositIQ — Model Selection Log

**Rubric:** Criterion 6 — Model Selection & Implementation (15 marks)
**Task:** T26–T27 (this section); extended by T28–T29, consolidated in T31 | **Status:** T26–T27 complete
**Date:** 2026-10-29
**Source of every number here:** the run of `python -m src.train`, recorded in
`models/cv_results.json` and printed by the module. Nothing below was computed by hand (D06).

---

## 1. What was trained, and what it had to beat

Three estimators, in **one** `Pipeline` whose first step is T24's `ColumnTransformer`, on the **12
features / 40 columns** T25 kept (D44):

| estimator | parameters | role |
|---|---|---|
| `dummy_prior` | `DummyClassifier(strategy="prior")` | the floor |
| `logreg_balanced` | `LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=42)` | first real model (T26) |
| `random_forest` | `RandomForestClassifier(n_estimators=300, min_samples_leaf=10, max_features="sqrt", class_weight="balanced_subsample", random_state=42, n_jobs=1)` | candidate (T27) |

The forest's four settings are decided, not defaulted:

* **`class_weight="balanced_subsample"`, not `"balanced"`.** A forest bootstraps, so each tree sees a
  different positive share; sklearn's subsample variant recomputes the weights per bootstrap sample
  instead of applying one global ratio to all of them. That is the documented choice for forests and it
  keeps the imbalance treatment comparable to the logistic baseline's, which is T29's question to answer
  properly rather than T27's to assume.
* **`min_samples_leaf=10`** — with 36k rows and 40 one-hot columns, single-row leaves fit noise, and it
  matches T25's instrument so the ablation in §5 is a fair re-run.
* **`max_features="sqrt"`** — 6 of 40 columns per split, which stops 300 correlated one-hot levels from
  swamping every tree.
* **`n_jobs=1`** — one process, so the runtime is comparable across estimators and so the numbers can be
  read against T25's instrument, which also ran single-threaded.

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
| `logreg_balanced` | 0.7640 ± 0.0114 | 0.4008 ± 0.0134 | 3.43× | 0.1852 | 0.4161 | 75.2129% |
| `random_forest` | **0.7906 ± 0.0089** | **0.4346 ± 0.0146** | **3.72×** | 0.1403 | 0.3404 | 83.9140% |

The forest leads on both headline metrics — **+0.0266 ROC-AUC** and **+0.0338 PR-AUC** over the linear
baseline, which is **+8.4% relative** on the metric that matters here. It is also *better calibrated*
than the balanced linear model (Brier **0.1403** against **0.1852**), though still worse than the
constant, and its sd is slightly **narrower** on ROC-AUC (0.0089 against 0.0114) as bagging should
make it.

The sd is the **sample** sd over the five folds (`ddof=1`), stated because `ddof=0` and `ddof=1`
differ by 12% at n = 5 and a table should not leave that to be guessed.

Per fold, because the fold is the unit of evidence and a mean can hide a fold that failed:

| fold | positives | dummy PR-AUC | logreg PR-AUC | logreg ROC-AUC | forest PR-AUC | forest ROC-AUC |
|---|---|---|---|---|---|---|
| 1 | 846 | 0.116948 | 0.393203 | 0.756884 | 0.433026 | 0.784995 |
| 2 | 846 | 0.116948 | 0.397146 | 0.779296 | 0.427353 | 0.800205 |
| 3 | 847 | 0.117086 | 0.385206 | 0.749395 | 0.415216 | 0.778149 |
| 4 | 846 | 0.116964 | 0.418816 | 0.768455 | 0.452317 | 0.794363 |
| 5 | 846 | 0.116964 | 0.409825 | 0.766155 | 0.445200 | 0.795320 |

The forest wins **every** fold against the linear model, on both metrics — no mean is carrying a
losing fold. Both clear **both** floors in every fold; for the forest the worst fold (3) is still
**3.55×** the base rate. The logistic baseline also clears D15's declared ROC-AUC target of **≥ 0.75**
on the mean, which is a target met, not a result claimed: D15's second target (>2× top-decile lift) is
T35's and has not been measured.

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
| `random_forest` (ROC-AUC 0.7906) | 0.1403 |

**Both models that rank far better are worse probability forecasters than a constant**, and the linear
one is the worse of the two. That is not a bug; it is what the weights do to the calibration, and it is
why:

* T26 reports **ROC-AUC and PR-AUC**, which are threshold-free, as the metrics the comparison uses.
* F1 and precision at 0.5 — or even at D17's 7.50% break-even — would measure the weighting rather
  than the model. At the 0.5 cut the balanced model flags a large share of the list; accuracy falls
  to 75.21% for exactly that reason.
* **T34 must recalibrate** (isotonic or Platt, fitted inside the fold) before D17's 7.50% threshold
  can be applied, or state that it is a *ranking* score being cut at a quantile instead of a
  probability being compared to a cost. Choosing the operating point is T34's job; this baseline does
  not pretend to have chosen one.

---

## 5. Does T25's reduction cost anything?

D44 dropped five features (`education`, `loan`, `default`, `balance_band`, `job`) because a 200-tree
forest could not distinguish their contribution from zero, and recorded that the decision was
**instrument-specific**. That invites two questions, so both were measured — the same five folds, the
columns toggled, once per estimator:

**The linear model — the instrument most likely to feel a loss,** because each one-hot level gets its
own coefficient and a column carrying signal cannot hide behind a correlated one:

| arm | features | columns | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| all 17 | 17 | 65 | 0.767926 | 0.402904 |
| shipped (D44) | 12 | 40 | 0.764037 | 0.400839 |
| **delta (shipped − all)** | −5 | −25 | **−0.003889 ± 0.002252** | **−0.002065 ± 0.002195** |
| folds improving | | | **0 / 5** | 1 / 5 |

**The forest — the instrument the decision was made with,** at production settings (300 trees rather
than T25's 200, and `balanced_subsample` weights):

| arm | features | columns | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| all 17 | 17 | 65 | 0.791999 | 0.433449 |
| shipped (D44) | 12 | 40 | 0.790606 | 0.434622 |
| **delta (shipped − all)** | −5 | −25 | **−0.001393 ± 0.002425** | **+0.001173 ± 0.002721** |
| folds improving | | | 1 / 5 | 3 / 5 |

Read against T25's own one-standard-error standard:

| estimator | Δ ROC-AUC | verdict | Δ PR-AUC | verdict |
|---|---|---|---|---|
| `logreg_balanced` | −0.003889 (1.73 SE) | **real loss**, 0/5 folds improve, sign test p = **0.031** | −0.002065 (0.94 SE) | inside the noise |
| `random_forest` | −0.001393 (0.57 SE) | inside the noise | +0.001173 (0.43 SE) | inside the noise, and the *sign* favours dropping |

The forest's verdict **reproduces T25** (−0.00014 ± 0.00195 on the 200-tree instrument) at production
settings, which is the strongest evidence the decision has: the family that made it still agrees with it
when its trees are actually fitted as candidates. Dropping the five features slightly *raises* the
forest's PR-AUC.

So the honest summary is: **for tree models, the reduction is free; for a linear model it costs 0.0039
ROC-AUC, which is statistically visible and practically negligible.** D44 stands, for three reasons that
are stated rather than assumed:

1. The shipped model is **XGBoost** (D08), which is tree-based, so both the instrument that chose the set
   and the two measurements above point the same way.
2. 25 fewer columns is **38.5%** of the matrix, and the price is 0.0039 AUC on a model that is not the
   shipped one — while the shipped family shows no price at all.
3. The comparison is available to re-run in one command, on the final model, in T31. **If XGBoost on
   40 columns lands below this logistic baseline's PR-AUC (0.4008), that is the trigger to revisit** —
   and the trigger is written here, before the result exists, rather than after.

Each estimator's shipped arm is built from that estimator's registered factory, and a module check
requires all five folds to reproduce its normal CV run **bit-identically**. Without that, the ablation
would be two experiments compared and its delta would mean nothing.

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
  floors re-derived, **every** model beating both floors in every fold, no leaky column, every artifact
  round-tripping, T25's frozen decision, no model stopping at its iteration cap (`n_iter` 35–43 against
  `max_iter` 2000; the trees have no cap), and each reduction arm reproducing its own estimator exactly.
* **196 independent checks** across two scripts that refit everything from `train.csv` through a plain
  `read_csv` and recompute every metric without importing `src.train` — **127** for the baselines and
  **69** for the forest. Both include two hand-written implementations — a rank-based Mann-Whitney AUC
  and a threshold-step average precision — that are first checked **against scikit-learn to 1e-12**
  (including the all-tied constant-score case), because a checker that has never been shown to agree
  with the library is a checker of nothing. The forest script also verifies that 300 trees were fitted,
  that `class_weight` was `balanced_subsample`, and that the all-17 arm of the ablation is reproduced.
* Every artifact reloads as a 2-step pipeline and reproduces a fresh full fit to 1e-12; the forest's
  300 trees survive the `joblib` round-trip; the dummy artifact predicts a constant.
* Artifacts are dumped with `compress=3`. A 300-tree forest stores every leaf it grew and is **41 MB**
  uncompressed against **15.8 MB** compressed — the reload check runs on the decompressed object, so the
  size is paid once rather than per clone.

**Runtime:** **156 s** — 3.8 s for the two baselines, and the rest is the forest: 5 CV folds, 1 full
fit, and 10 fits for the two-arm ablation. That is slow enough to be worth knowing before T28 adds
XGBoost to the same loop, and fast enough that re-running the whole comparison remains the default
rather than a ceremony.

---

## 8. What T28–T31 inherit

* A model is an entry in `ESTIMATORS` plus a `MODEL_PATHS` line. It gets the folds, the preprocessing
  and the scorer unchanged; add it to `REDUCTION_ESTIMATORS` and the §5 ablation covers it for free.
* The floor to beat is **PR-AUC 0.1170 / ROC-AUC 0.5000**, not 0.5 PR-AUC.
* The number to beat is now the forest: **PR-AUC 0.4346 (3.72×), ROC-AUC 0.7906**.
* Threshold metrics wait for T34's recalibration; imbalance variants are T29's to compare on
  **PR-AUC**, which is the metric that responds to the positive class.
* D46's revisit trigger is live: if XGBoost on 40 columns lands below the logistic baseline's **PR-AUC
  0.4008**, the feature set is reopened. Given the forest's result, that now looks unlikely — which is
  itself worth knowing.

*Interpretation for the viva:* "A linear model on twelve features ranks clients at ROC-AUC 0.764 and
finds the subscribers at 3.4× the base rate; a 300-tree forest does better on both — 0.791 and 3.7× —
against a floor of 0.5 and 1.0×. The linear model's probabilities are deliberately uncalibrated, so I
report ranking metrics and recalibrate before applying the €1.50 break-even. And I checked the one
thing that could embarrass the feature selection: dropping five features costs a linear model about
four thousandths of ROC-AUC, and the tree models — the one that chose the set and the one that will
ship — see no loss at all."
