# DepositIQ — Model Selection Log

**Rubric:** Criterion 6 — Model Selection & Implementation (15 marks)
**Task:** T26–T28 (this section); T29 is in `docs/06_imbalance.md`; consolidated in T31 | **Status:** T26–T28 complete
**Date:** 2026-10-29
**Source of every number here:** the run of `python -m src.train`, recorded in
`models/cv_results.json` and printed by the module. Nothing below was computed by hand (D06).

---

## 1. What was trained, and what it had to beat

Four estimators, in **one** `Pipeline` whose first step is T24's `ColumnTransformer`, on the **12
features / 40 columns** T25 kept (D44):

| estimator | parameters | role |
|---|---|---|
| `dummy_prior` | `DummyClassifier(strategy="prior")` | the floor |
| `logreg_balanced` | `LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=42)` | first real model (T26) |
| `random_forest` | `RandomForestClassifier(n_estimators=300, min_samples_leaf=10, max_features="sqrt", class_weight="balanced_subsample", random_state=42, n_jobs=1)` | candidate (T27) |
| `xgboost_gbdt` | `XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=3, min_child_weight=5, subsample=0.9, colsample_bytree=0.8, scale_pos_weight=7.548334, tree_method="hist", eval_metric="aucpr", random_state=42, n_jobs=1)` | candidate, D08's family (T28) |

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

XGBoost's settings, and where each came from:

* **`max_depth=3` was measured on one fold, not chosen by reputation.** Across depths 2 / 3 / 4 / 6 the
  fold-1 PR-AUC was **0.42948 / 0.43760 / 0.43604 / 0.42985** — depth 6 overfits 40 sparse one-hot columns
  and depth 2 underfits, while the two best are within 0.002 of each other, so the shallower one wins as the
  cheaper explanation. That is a **one-fold sanity check, not tuning**: T30 tunes on all five folds, and this
  is what it starts from rather than what it concludes.
* **`learning_rate=0.05` with 400 rounds**, measured at **1.8 s per fold** — affordable sixteen times over,
  and deliberately not the library default 0.3, which at 400 rounds would be badly over-trained.
* **`scale_pos_weight=7.548334`** is XGBoost's native imbalance handle (D08), standing in for the other
  models' `class_weight`. It is `31,937 / 4,231`, the ratio on the **training half only**, pinned as a
  literal and **checked against the measured ratio** — a hard-coded constant nobody re-derives is how a
  weight quietly stops meaning what its comment says. A per-fold weight was rejected: the folds are
  stratified, so the per-fold training ratio deviates by at most **1.9e-3** (**0.025%** relative), which
  cannot matter, while a per-fold weight would make the model's configuration depend on which fold it is in.
* **`tree_method="hist"`** — the only exact-enough method that is fast on 36,168 × 40 dense floats.

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
| `random_forest` | 0.7906 ± 0.0089 | 0.4346 ± 0.0146 | 3.72× | 0.1403 | 0.3404 | 83.9140% |
| `xgboost_gbdt` | **0.7993 ± 0.0089** | **0.4420 ± 0.0134** | **3.78×** | 0.1637 | 0.3892 | 81.5611% |

XGBoost leads on both headline metrics — **+0.0353 ROC-AUC over the linear baseline and +0.0412 PR-AUC**
(**+10.3% relative**), and **+0.0087 / +0.0074** over the forest, in **every one of the five folds** on
both metrics. The margin over the forest is small but consistent, which is what a boosted ensemble is
supposed to buy over a bagged one.

**It is the worst-calibrated model in the table** (Brier **0.1637** against the forest's **0.1403**) while
being the best ranker, because `scale_pos_weight` inflates its scores the way `class_weight` inflates the
linear model's — mean predicted probability **0.3892** against a **0.1170** base rate. Three models now
demonstrate the same point from different directions: **ranking quality and calibration are independent
properties here, and only T34's recalibration can turn one into the other.**

The sd is the **sample** sd over the five folds (`ddof=1`), stated because `ddof=0` and `ddof=1`
differ by 12% at n = 5 and a table should not leave that to be guessed.

Per fold, because the fold is the unit of evidence and a mean can hide a fold that failed:

| fold | positives | dummy PR-AUC | logreg PR-AUC | logreg ROC-AUC | forest PR-AUC | forest ROC-AUC | xgb PR-AUC | xgb ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| 1 | 846 | 0.116948 | 0.393203 | 0.756884 | 0.433026 | 0.784995 | 0.437596 | 0.792828 |
| 2 | 846 | 0.116948 | 0.397146 | 0.779296 | 0.427353 | 0.800205 | 0.435037 | 0.811285 |
| 3 | 847 | 0.117086 | 0.385206 | 0.749395 | 0.415216 | 0.778149 | 0.425812 | 0.791189 |
| 4 | 846 | 0.116964 | 0.418816 | 0.768455 | 0.452317 | 0.794363 | 0.457030 | 0.795176 |
| 5 | 846 | 0.116964 | 0.409825 | 0.766155 | 0.445200 | 0.795320 | 0.454688 | 0.806184 |

XGBoost wins **every** fold against **both** other models, on **both** metrics — no mean is carrying a
losing fold — and the narrowest margin (fold 4, ROC-AUC 0.795176 against 0.794363) is a real ordering, not
a tie, because the two models are scored on identical rows. Every model clears **both** floors in every
fold; for XGBoost the worst fold (3) is still **3.64×** the base rate. The logistic baseline also clears
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
| `random_forest` (ROC-AUC 0.7906) | 0.1403 |
| `xgboost_gbdt` (ROC-AUC 0.7993) | 0.1637 |

**All three models that rank far better are worse probability forecasters than a constant.** That is not
a bug; it is what the imbalance weights do to the calibration, and it is why:

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

**The model that would ship — XGBoost** (D08's family, T31's likely choice):

| arm | features | columns | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| all 17 | 17 | 65 | 0.799432 | 0.442776 |
| shipped (D44) | 12 | 40 | 0.799332 | 0.442032 |
| **delta (shipped − all)** | −5 | −25 | **−0.000100 ± 0.000839** | **−0.000743 ± 0.001534** |
| folds improving | | | 2 / 5 | 2 / 5 |

Read against T25's own one-standard-error standard:

| estimator | Δ ROC-AUC | verdict | Δ PR-AUC | verdict |
|---|---|---|---|---|
| `logreg_balanced` | −0.003889 (1.73 SE) | **real loss**, 0/5 folds improve, sign test p = **0.031** | −0.002065 (0.94 SE) | inside the noise |
| `random_forest` | −0.001393 (0.57 SE) | inside the noise | +0.001173 (0.43 SE) | inside the noise, and the *sign* favours dropping |
| `xgboost_gbdt` | −0.000100 (0.12 SE) | inside the noise, by a factor of six | −0.000743 (0.48 SE) | inside the noise |

The gradient-boosted model's loss is the smallest of the three — **a tenth of a thousandth of ROC-AUC** —
and both signs are split 2/5, which is what noise looks like. The forest's verdict **reproduces T25**
(−0.00014 ± 0.00195 on the 200-tree instrument) at production settings, which is the strongest evidence
the decision has: the family that made it still agrees with it when its trees are actually fitted as
candidates. Dropping the five features slightly *raises* the forest's PR-AUC.

So the honest summary is: **for tree models the reduction is free, and for the model that will ship it is
free by a factor of six; for a linear model it costs 0.0039 ROC-AUC, which is statistically visible and
practically negligible.** D44 stands, for three reasons that are stated rather than assumed:

1. The shipped model is **XGBoost** (D08), and its own ablation puts the loss at **−0.0001 ± 0.0008 ROC**
   and **−0.0007 ± 0.0015 PR** — smaller than either of the other two estimators.
2. 25 fewer columns is **38.5%** of the matrix, and the only measurable price is 0.0039 AUC on a model
   that is not the shipped one, while the shipped family shows no detectable price at all.
3. **D46's revisit trigger has now been evaluated and did not fire:** the trigger was a 40-column XGBoost
   landing below the logistic baseline's PR-AUC **0.4008**, and it landed at **0.4420**. The feature set
   stays as T25 left it, decided by a measurement that was written down before this number existed.

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

* `python -m src.train` exits **0** with **15/15** checks: fold determinism, target identity,
  pipeline structural identity, per-fold fitting, matrix width, prediction range, the dummy's four
  floors re-derived, **every** model beating both floors in every fold, no leaky column, every artifact
  round-tripping, T25's frozen decision, no model stopping at its iteration cap (`n_iter` 35–43 against
  `max_iter` 2000; the trees have no cap), each reduction arm reproducing its own estimator exactly, and
  **the pinned imbalance weight still matching the training half**.
* **278 independent checks** across three scripts that refit everything from `train.csv` through a plain
  `read_csv` and recompute every metric without importing `src.train` — **127** for the baselines, **69**
  for the forest, **82** for XGBoost. All three include two hand-written implementations — a rank-based
  Mann-Whitney AUC and a threshold-step average precision — that are first checked **against scikit-learn
  to 1e-12** (including the all-tied constant-score case), because a checker that has never been shown to
  agree with the library is a checker of nothing. The forest script verifies 300 trees and
  `balanced_subsample`; the XGBoost script verifies 400 rounds reached the booster, depth 3, the weight
  actually applied, and that XGBoost beats the forest in all five folds on both metrics.
* **A float32 trap, found by the verifier.** The XGBoost script first failed on 11 checks with differences
  of **1e-7**: it computed `scale_pos_weight` as `31,937 / 4,231 = 7.5483337` while the module pins
  **7.548334**, and those two values round to *different* float32 numbers. That is enough to shuffle rows
  that are near-tied and move AUC in the seventh decimal. The verifier now uses the pinned literal and
  **separately** asserts it is within 1e-6 of the measured ratio — which is exactly what the module's own
  check does. Reproduced bit-identically after the fix.
* Every artifact reloads as a 2-step pipeline and reproduces a fresh full fit to 1e-12; the forest's
  300 trees survive the `joblib` round-trip; the dummy artifact predicts a constant.
* Artifacts are dumped with `compress=3`. A 300-tree forest stores every leaf it grew and is **41 MB**
  uncompressed against **15.8 MB** compressed — the reload check runs on the decompressed object, so the
  size is paid once rather than per clone.

**Runtime:** **182 s**, of which the baselines are 3.8 s, the forest is ~150 s and XGBoost is ~30 s (1.8 s
per fit measured). Fast enough that re-running the whole comparison remains the default rather than a
ceremony, and the forest remains the bottleneck: it costs five times what XGBoost does for a worse result.

---

## 8. What T30–T31 inherit

* A model is an entry in `ESTIMATORS` plus a `MODEL_PATHS` line. It gets the folds, the preprocessing
  and the scorer unchanged; add it to `REDUCTION_ESTIMATORS` and the §5 ablation covers it for free.
* The floor to beat is **PR-AUC 0.1170 / ROC-AUC 0.5000**, not 0.5 PR-AUC.
* The number to beat is now XGBoost: **PR-AUC 0.4420 (3.78×), ROC-AUC 0.7993**.
* Threshold metrics wait for T34's recalibration.
* **T29 has since answered the imbalance question** (`docs/06_imbalance.md`, D49): SMOTE is worse than
  doing nothing in 0/5 folds, and dropping `scale_pos_weight` costs nothing measurable while **halving**
  the Brier score. So the number T30 should tune against is the **untreated** estimator's **PR-AUC
  0.4444**, not the weighted 0.4420 in the table above — the table records the models as they were built,
  and T29 is where that build was questioned.
* **D46's revisit trigger was evaluated and did not fire** — XGBoost on 40 columns reached PR-AUC
  **0.4420** against the 0.4008 that would have reopened the feature set.
* T30 now has a measured starting point rather than a guess: depth 3 / lr 0.05 / 400 rounds, with 1.8 s
  per fit, so a `RandomizedSearchCV` over that space is affordable.

*Interpretation for the viva:* "Four models on the same twelve features, same folds, same pipeline. The
constant scores 0.50 and 1.0× the base rate; the linear model 0.764 and 3.4×; the forest 0.791 and 3.7×;
XGBoost 0.799 and 3.8×, and it wins every fold against both. All three real models are deliberately
uncalibrated — they score the average client between 0.34 and 0.42 when the true rate is 0.12 — so I
report ranking metrics and recalibrate before applying the €1.50 break-even. And I checked the one thing
that could embarrass the feature selection: dropping five features costs a linear model about four
thousandths of ROC-AUC, and for the model that ships it costs a ten-thousandth, which is noise."

---

## 9. Tuning (T30): what the search found, and what it did not

T28 chose `max_depth=3` from a **fold-1** sweep and recorded it as "a starting point for T30, explicitly
not tuning". This section is the tuning: six axes, five folds, 40 draws, scored by the same loop as every
model above.

### 9.1 The space, and why each axis is in it

| axis | values | why these bounds |
|---|---|---|
| `n_estimators` | 200, 300, 400, 600, 800 | round count and learning rate trade off; 400/0.05 is one point in the grid, not a privileged one |
| `learning_rate` | 0.02, 0.03, 0.05, 0.08, 0.1 | 0.02 at 800 rounds measured **4.0 s** on one fold, so the slow end is affordable; 0.1 must be allowed to lose at 400 rounds |
| `max_depth` | 2, 3, 4, 5 | T28's one-fold probe already showed depth **6** overfits the 40 sparse one-hot columns, so 6 is excluded on measured evidence, not preference |
| `min_child_weight` | 1, 3, 5, 10 | 1 is the library default, 5 is the shipped value, 10 is the regularised end |
| `subsample` | 0.7, 0.8, 0.9, 1.0 | row bagging, from none to strong |
| `colsample_bytree` | 0.6, 0.8, 1.0 | 40 correlated one-hot columns make column bagging plausible |

**4,800 points, 40 drawn** — 0.83% of the space, `scoring="average_precision"` (D04), seed 42. The budget
is stated as a budget: this is a search, not an optimum, and a better configuration found later is a
larger `n_iter`, not a mistake here. The **search** runs `n_jobs=-1`; every individual fit stays
`n_jobs=1`, so two candidates' runtimes are measured on the same number of cores.

Two things are held out of the space rather than tuned: the **feature set** (T25's twelve columns, D44)
and the **imbalance treatment** (fixed at `scale_pos_weight=1`, T29's finding). A search is free to trade
calibration for ranking; letting it re-choose the treatment would reopen an answered question by the back
door. The weight is evaluated afterwards as a third arm instead.

### 9.2 The winner, and the three arms

| arm | PR-AUC | ROC-AUC | Brier | mean p | accuracy @ 0.5 | `scale_pos_weight` |
|---|---|---|---|---|---|---|
| `untuned_untreated` | 0.4444 ± 0.0131 | **0.7993 ± 0.0079** | 0.0821 | 0.1168 | 89.43% | 1 |
| `tuned_untreated` | **0.4460 ± 0.0110** | 0.7990 ± 0.0086 | **0.0818** | 0.1166 | 89.38% | 1 |
| `tuned_weighted` | 0.4415 ± 0.0153 | 0.7961 ± 0.0082 | 0.1535 | 0.3629 | 81.98% | 7.548334 |

Winner: **`subsample=1.0, n_estimators=300, min_child_weight=10, max_depth=5, learning_rate=0.08,
colsample_bytree=0.6`** — 300 rounds at 0.08 rather than 400 at 0.05, a deeper tree with heavier leaf
regularisation, and no row bagging at all.

Paired per fold against the untuned configuration:

| arm | Δ PR-AUC | Δ ROC-AUC | Δ Brier (negative = better) |
|---|---|---|---|
| `tuned_untreated` | **+0.00165 ± 0.00286**, 4/5 folds improve | −0.00028 ± 0.00332, 2/5 | −0.00024 ± 0.00026, 4/5 |
| `tuned_weighted` | −0.00291 ± 0.00300, **0/5** | −0.00319 ± 0.00190, **0/5** | +0.07147 ± 0.00165, **0/5** |

### 9.3 The finding is not the gain — it is the size of the gain

The done-condition is met: the tuned model beats the untuned one on CV PR-AUC, in four of five folds. The
honest reading is what that is worth:

* **+0.00165 is 0.58 standard errors.** It is the same order as the fold-to-fold noise it is measured
  against, and smaller than the 0.0039 ROC-AUC loss T26 called *statistically visible and practically
  negligible* for a different comparison.
* **40 candidates span 0.4080 to 0.4460, and 14 of them sit within one standard error (0.0049) of the
  winner.** The median candidate is 0.4389, so the search's whole useful range is about a
  hundredth of a point of PR-AUC — the same hundredth the arms already differed by.
* **The shipped configuration was 4th of 40**, at 0.4449 ± 0.0129 against the winner's 0.4460: inside the
  noise, 0.0012 behind. T28's hand-chosen settings were not wrong; they were near the top of what this
  space offers, which is the reason tuning bought so little.
* ROC-AUC moves the *other* way (−0.00028, 2/5 folds) and Brier is flat. So the gain is specific to the
  positive-class metric and is not a uniformly better model.

What the search genuinely changed is the **configuration and the record**: where T28 had one fold's
opinion that depth 3 beats depth 4 and 6, this run has five folds saying a deeper tree with
`min_child_weight=10` is at least as good — and a machine-readable `models/best_params.json` from which T31
can rebuild the model instead of reading a sentence. **A deeper tree with heavier leaf regularisation is a
different model from a shallower one with light regularisation; that they score the same is a finding, not
a null result.**

### 9.4 The selection bias is stated, not hidden

The winner was chosen on **these five folds**, and T32 reports on **the same five folds**. Its CV score is
therefore **optimistic by construction**, and `search.best_score_` (0.446015) is never quoted as a result:

* the arms table is rebuilt by refitting the chosen parameters through `train.cross_validate_factory`, and
  a check requires those per-fold numbers to equal the search's own per-split scores to **0.0e+00** — so
  the table is a refit that agrees, not a copy;
* the record carries all 40 candidate scores and both bands (14 within one SE, 28 within one fold-sd), so
  a reader can see how much of the winner's edge is the edge of being best-of-40 on this data;
* **T32's held-out half is the only thing that can discipline this number**, and §9.3 is written so that a
  held-out PR-AUC anywhere near 0.4460 is a confirmation rather than a surprise.

### 9.5 Two defects the verification caught

* **`std_test_score` is a population sd.** scikit-learn computes `cv_results_["std_test_score"]` with
  `ddof=0`; this project states `ddof=1` everywhere else (T26's `SD_DDOF`, chosen because on five folds
  they differ by 12%). The first version of the record quoted sklearn's number next to the arms' sample
  sds — **0.0099 against 0.0110** for the winner, an 11.8% error in a column nobody would have questioned.
  The module now recomputes the fold sd from the per-split scores with `ddof=1` and keeps sklearn's value
  alongside, labelled.
* **Brier is computed in float32, and average precision collapses tied scores.** Both surfaced in the
  independent verifier rather than in this module: scikit-learn converts the target to a **bool**
  indicator, and XGBoost's probabilities are float32, so `bool − float32` keeps the whole Brier
  computation in float32 and the recorded value carries float32 round-off (a float64 recomputation differs
  by up to **1.38e-10**). Average precision has the mirror-image issue: the arms' folds contain **18-115
  bit-identical predicted probabilities** each (float32 again — 18 of 7,233 on `tuned_weighted` fold 5,
  **115** on `untuned_untreated` fold 1), and at a tied score no threshold exists between the rows, so a
  staircase evaluated **per row** differs from scikit-learn's **tie-collapsed** one by up to **5.22e-07**
  with no model change. Collapsing the ties makes the two agree to **1.11e-16** — one float64 ulp. Both
  conventions are measured in the verifier rather than argued about, and both the Brier and the AP
  tolerances are stated explicitly. Neither affects a published figure at four decimals; both would have
  been invisible in a "checks pass" line.

### 9.6 Verification

* `python -m src.tune` exits **0** with **12/12** checks, **68.2 s** (search 50.1 s): the search replayed
  the project's materialised folds; every chosen parameter reappears in the fitted estimator; the treatment
  was fixed and every arm applied the weight it claims (`1.0 / 1.0 / 7.548334`); every fit ran on one core;
  the budget, metric and seed are recorded and the 40 candidates are distinct; the feature set is still
  twelve columns into 40; SMOTE appears nowhere; the untuned arm reproduces **T29's `none` arm
  bit-identically (max drift 0.0e+00)**; the tuned arm's scores equal the search's own split scores
  (0.0e+00); the tuned model beats the untuned one; the record alone rebuilds the model bit-identically;
  and the file on disk matches the run.
* **131 independent checks** in a separate script that refits all three arms and the top five candidates
  from `train.csv`, building its own folds, its own pipelines and its own metrics, without importing
  `src.tune` or `src.train`: every per-fold PR-AUC, ROC-AUC, Brier, mean-p and accuracy, every arm mean
  and sd, every paired delta, all five leaderboard entries, the 40-candidate distribution and both bands,
  the space size, and T29's verdict surviving tuning. Its AUC, AP and Brier functions are validated
  against scikit-learn **before** they are used to check anything.

*Interpretation for the viva:* "I tuned it properly rather than claiming the defaults were tuned: 40
random draws over six axes, five folds, scored on PR-AUC. The tuned model does win - 0.4460 against
0.4444, in four folds of five. But I am not going to sell that as a discovery: the gain is 0.0016 at 0.58
standard errors, and 14 of the 40 candidates are within one standard error of the winner. What the search
actually told me is that the configuration I had hand-picked was already the fourth best of forty, and
that hyperparameters are not where this problem's performance lives - the feature set and the model
family were. It also gave me the configuration as a file, so the model can be rebuilt rather than
described. And because the winner was chosen on the same folds I report, I say plainly that its CV score
is optimistic and that the held-out number is the one that counts."

