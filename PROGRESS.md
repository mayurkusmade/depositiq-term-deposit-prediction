# Project Status

**Title:** DepositIQ: Predicting Term Deposit Subscription to Optimize Bank Telemarketing Campaigns
**Last updated:** 2026-10-27 20:08
**Overall completion:** 35% (16 / 46 tasks)
**Working location:** `depositiq-term-deposit-prediction/` (GitHub: `mayurkusmade/depositiq-term-deposit-prediction`)

## Rubric Coverage

| # | Rubric criterion | Marks | Tasks done / total | Tasks |
|---|---|---|---|---|
| 1 | Problem Definition & Business Understanding | 10 | 3 / 3 ✅ | T07–T09 |
| 2 | Dataset & Data Understanding | 10 | 2 / 2 ✅ | T10–T11 |
| 3 | Data Preprocessing & Cleaning | 10 | 4 / 4 ✅ | T12–T15 |
| 4 | EDA | 10 | 1 / 6 | T16–T21 |
| 5 | Feature Engineering / Selection | 10 | 0 / 4 | T22–T25 |
| 6 | Model Selection & Implementation | 15 | 0 / 6 | T26–T31 |
| 7 | Model Evaluation & Validation | 15 | 0 / 5 | T32–T36 |
| 8 | Business Insights & Recommendations | 10 | 0 / 3 | T37–T39 |
| 9 | Documentation & Presentation | 10 | 6 / 13 | T01–T06, T40–T46 |
| | **Total** | **100** | **16 / 46** | |

> Note: the rubric has nine criteria, listed above. Phase 10 (T45–T46, viva preparation) is not a
> separate criterion — those two tasks are counted inside criterion 9 — so the per-criterion
> numbers sum to exactly the 46 tasks in the checklist.

---

## NEXT TASK

**T17 | Phase 4 – EDA | Numeric distributions and skew.**
Open: `notebooks/03_eda.ipynb` (§2), `reports/figures/02_numeric_distributions.png`.
Plot the distribution of every numeric column of the classic training half and write **one
interpretation line per variable** — the wording of an insight, not the shape of a histogram.
**Done when:** the figure is saved to `reports/figures/02_numeric_distributions.png` and one
interpretation line exists for each numeric variable.
**Carry in:** the numeric set is `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`
plus the derived `days_since_contact` (D23). Three traps are already measured and must not be
re-discovered: **`previous`'s IQR is 0** (81.7% of rows are zeros), so any median/IQR rescale divides
by zero; **`balance` is negative for 3,766 rows** (legitimate overdrafts), so `log1p` is undefined
and `arcsinh` is the correct transform if one is shown at all; and **`duration` is the post-outcome
leak (D05)** — it belongs in the figure, clearly labelled, but never in the model. Skews measured in
T13 from the full frame are `previous` +41.85, `balance` +8.36, `campaign` +4.90, `duration` +3.14;
T17 should report the training-half values rather than reuse those numbers (D28).

---

## Decisions Log

| # | Date | Decision | Reason |
|---|---|---|---|
| D01 | 2026-10-27 | Project lives in the existing repo `mayurkusmade/depositiq-term-deposit-prediction` | It already contains both dataset variants; avoids splitting the work across repos |
| D02 | 2026-10-27 | `bank-full.csv` (45,211 × 17) is the **primary** dataset; `bank-additional-full.csv` (41,188 × 21) kept as secondary | The brief names bank-full.csv first and it matches the 16-feature story; the additional set stays available for the richer economic-indicator variant |
| D03 | 2026-10-27 | Fixed `random_state = 42` globally, defined once in `src/config.py` | Reproducibility across sessions and machines |
| D04 | 2026-10-27 | Accuracy is **not** a headline metric | A "always predict no" model already scores ~88%; headline metrics are PR-AUC and ROC-AUC, with precision/recall/F1 and lift |
| D05 | 2026-10-27 | `duration` is dropped from the main model; retained only for a documented with-vs-without comparison | It is only known *after* the call ends, so it leaks the outcome and cannot be used for targeting |
| D06 | 2026-10-27 | Every metric reported must come from a saved script/notebook run, never re-typed by hand | Rubric requires verifiable, reproducible numbers |
| D07 | 2026-10-27 | Data files live under `data/raw/` after T02 | Keeps raw data immutable and separate from processed splits |
| D08 | 2026-10-27 | XGBoost chosen as the gradient-boosting model (not LightGBM) | Mainstream for graded work, native `scale_pos_weight` for imbalance, and its trees work directly with SHAP `TreeExplainer` |
| D09 | 2026-10-27 | `xgboost==3.2.0` and `shap==0.51.0` — deliberately **not** the newest releases | The current latest (xgboost 3.4.1, shap 0.54.1) require Python ≥ 3.12; this environment is Python 3.11.9, so these are the newest installable versions |
| D10 | 2026-10-27 | Deliverable tooling (PDF/PPT generation) is **not** pinned yet | Keeps `requirements.txt` honest about what the pipeline actually needs; the dependency gets added in a new task when T40/T41 are reached |
| D11 | 2026-10-27 | `ROOT` in `src/config.py` is derived from `Path(__file__).resolve().parents[1]` | Makes the project path-independent: it runs from any clone location with no edits, which was proven by copying `config.py` to another folder |
| D12 | 2026-10-27 | Modules are imported as `from src import config`, run from the repo root; no `src/__init__.py` | Python 3 namespace packages make `src` importable without an `__init__.py`, so the project stays one file lighter. Notebooks must either start at the repo root or add it to `sys.path` |
| D13 | 2026-10-27 | `src/data_loader.py` is the **only** place that reads the raw CSVs | Prevents the classic silent errors: forgetting `sep=";"`, or reading the 10% sample when the full dataset was intended |
| D14 | 2026-10-27 | The loader does **not** drop `duration`; it returns all features except the target | Phase 4 needs `duration` present to *demonstrate* the leakage, and removing it for the main model is designed deliberately in T22 rather than hidden in the loader |
| D15 | 2026-10-27 | Declared KPI targets: top-decile lift **> 2×**, ROC-AUC **≥ 0.75**, PR-AUC to beat the 0.117 base rate | Clear, testable targets make the evaluation honest; they are **targets, not results**, and must never be presented as achieved until measured in T32–T35 |
| D16 | 2026-10-27 | Stakeholder KPIs are split into *measurable from this dataset* vs *assumed / not measured* | Two KPIs (net value, fairness ratio) rest on inputs this dataset does not contain, so they are labelled instead of silently implied to be verified |
| D17 | 2026-10-27 | Cost model inputs: **C = €1.50 per call** (€18/h loaded agent × 5 min) and **V = €20.00 per subscription** (€2,000 deposit × 1% margin × 12 months) | Gives a break-even threshold of **p\* = 7.50%**. Both euro figures are labelled ASSUMED; V is deliberately conservative (excludes cross-sell and customer lifetime value), and the 5-minute call is corroborated by the measured mean talk time of 4.3 min |
| D18 | 2026-10-27 | The project claim is scoped to **converting a fixed call budget better**, NOT to reducing campaign spend | Because p\* = 7.50% sits *below* the 11.7% base rate, the average client is still worth calling (EV +€0.84), so "targeting cuts costs" is **not** supported. It is only supported for fixed call volume, or if the true subscription value is below the €12.82 crossover |
| D19 | 2026-10-27 | The two variants are treated as **complementary, not superset/subset**, and pipelines must be fitted separately per variant | Measured: the additional set has **no `balance` column** (it trades the client's own balance for five macro indicators), its `pdays` sentinel is **999** not −1, its `education`/`marital`/`default` coding differs, its `month` has only 10 of 12 levels, and it contains **12 exact duplicate rows**. Code that hard-codes the classic conventions would silently corrupt the additional variant |
| D20 | 2026-10-27 | The classic set is the **primary** dataset for all main modelling; the additional variant is used for the documented comparison | D02 restated after profiling: `bank-full.csv` carries `balance`, which is the stronger client-level signal, and it matches the brief; the additional set is the second configuration, not a replacement |
| D21 | 2026-10-27 | The `previously_contacted` flag is derived from **`previous > 0`**, *not* from the `pdays` sentinel | Measured in T11 §7.1: in the additional variant **4,110** of the 39,673 sentinel rows have `previous` 1–6 with `poutcome = failure`, so 999 does **not** reliably mean "never contacted" there. In the classic set it does — `pdays != −1` and `previous > 0` select exactly the same 8,257 rows — but `previous > 0` agrees exactly with `poutcome != nonexistent` in **both** variants, so it is the portable choice. Using the sentinel would mislabel 4,110 clients and give them a wrong recency value |
| D22 | 2026-10-27 | `unknown` is kept as an **explicit category** — no imputation, no row deletion — and any NaN in a categorical column is normalised to the same literal marker | Both alternatives were measured and rejected (T11 §6, restated in `src/clean.py`): imputing would fabricate up to **81.75%** of classic `poutcome` (36,959 rows, which really means "no previous campaign"); listwise deletion would remove **37,369** classic rows (**82.65%**) and 10,700 additional rows (25.98%). Keeping the level preserves every row, matches the custodian's own encoding, and leaves open the possibility that the missingness is informative rather than assuming it is noise. Applying it changed **0** cells, which is the honest outcome — the level was already there |
| D23 | 2026-10-27 | The sentinel is removed as a **value** and replaced by two columns: `previously_contacted` (bool, from `previous > 0`) and `days_since_contact` (float, the real `pdays` value, **NaN** at the sentinel). Raw `pdays` is retained | A day count that is genuinely unknown cannot be invented, and XGBoost (D08) learns a branch direction for missing values instead of requiring them to be filled. Raw `pdays` is kept for traceability only; excluding it from the model matrix is T24's decision in the `ColumnTransformer`, not a silent drop here |
| D24 | 2026-10-27 | **Keep every value** — no winsorising, no clipping, no removal. `RobustScaler` for the Logistic Regression baseline only; trees get raw values; **`previous` is excluded from scaling entirely**, and any log-like transform must be **`arcsinh`**, not `log1p` | Measured in T13: all four named columns are heavily skewed (`previous` +41.85, `balance` +8.36, `campaign` +4.90, `duration` +3.14), but the tails are **signal, not error** — the subscription rate runs 6.90% → 16.61% across `balance` deciles and 14.60% → 3.93% across `campaign` bands, so clipping merges groups that differ by more than 10 pp. Winsorising at 5–95 would rewrite **13,054 rows** and destroy **3,366 distinct values**, and it is the only non-invertible option |
| D25 | 2026-10-27 | The Phase 3 conversion check is **total**: comparisons are **positional** (`.to_numpy()`) and any check that cannot execute is recorded as **FAILED**, never raised; `validate()` returns 15 labelled checks as data and `process()` exits non-zero on any failure | Fault injection showed the first version **raised** `ValueError: Can only compare identically-labeled Series objects` on a damaged frame — the guard rail crashed *exactly* when a diagnosis was wanted, and a check that never ran was silently missing from the count, so "15/15 passed" could have meant "14 checks ran". **8/8** injected faults (dropped row, edited value, flipped target, surviving sentinel, flag taken from the sentinel instead of `previous`, invented day count, extra column, `unknown` replaced by a null) are now reported as failures with the reason attached |
| D26 | 2026-10-27 | The split is **stratified on the target and shuffled**, taken **after** conversion, fitted with **nothing**, and written **per variant** (`train.csv`/`test.csv` + `_additional` twins). Row identity in the files is positional, and identical source rows are **reported, not removed** | Measured, not stylistic. The classic file is **ordered by campaign date** (`may 5` → `nov 17`) and its positive rate climbs from **2.99%** in the first row-order decile to **47.09%** in the last, so the obvious "first 80% is train" split would put **31.58%** positives in test against 11.70% overall — a 19.9pp swing, 20× the ±1pp this task is graded on. Shuffling is what stops the split being a date cut in disguise. Stratifying gets the actual result to within **0.0012pp**. Splitting after conversion keeps D22/D23/D24 intact; fitting nothing before the split keeps test statistics out of training; separate per-variant splits keep D19's two pipelines from sharing rows |
| D27 | 2026-10-27 | The Phase 3 checkers must **measure the frames they are handed**, never read the numbers out of the report they are checking; a separate check holds report and data together | Found by fault injection, not review: an earlier `validate_split` read `report["positive_rate_test"]` and therefore **passed a test half with every positive stripped out** (0.00% positives), because the report still described the honest split. A checker that validates a *description* rather than the data cannot see the two disagree. Rates, shares and NaN counts are now recomputed from `train`/`test`/`df` on every check, and one new check fails if any reported figure drifts from its measurement by more than 1e-12 |
| D28 | 2026-10-27 | EDA is computed on the **training half only**, and the evaluation baselines are **fixed from it**: accuracy 88.3018% (always-`no`), PR-AUC random baseline 0.1170, ROC-AUC 0.50, and **5 folds** for CV | The test half must stay unseen until T32 or it stops being a held-out set, so every Phase 4 number comes from `data/processed/train.csv`. The baselines are *derived here rather than assumed*: the accuracy baseline is the majority share, and PR-AUC's random baseline is the positive rate (11.70%), not 0.5 — a distinction that decides whether a reported 0.6 is 5× lift or chance. The fold count is measured, not inherited: 5-fold stratified CV puts **846–847 positives in every validation fold**, while 10 folds would halve that denominator for no gain |

---

## Task Checklist

Legend: `- [ ] Txx | Rubric n | Task | Output: <artifact> | Done when: <verification>`
Tick only after the deliverable is verified to exist and run.

### Phase 0 — Setup
- [x] T01 | Rubric 9 | Create repo folder structure and .gitignore | Output: `data/raw/`, `data/processed/`, `notebooks/`, `src/`, `models/`, `reports/figures/`, `app/`, `docs/`, `.gitignore` | Done when: directories exist and `git status` stages no datasets or `.venv` — **done 2026-10-27**
- [x] T02 | Rubric 9 | Move raw datasets to `data/raw/` and verify integrity | Output: `data/raw/bank-full.csv`, `bank.csv`, `bank-names.txt`, `data/raw/bank-additional/*` | Done when: loader prints `(45211, 17)` and `(41188, 21)` from the new paths — **done 2026-10-27**
- [x] T03 | Rubric 9 | Write `requirements.txt` with pinned versions | Output: `requirements.txt` | Done when: pandas, scikit-learn, xgboost/lightgbm, imbalanced-learn, shap, matplotlib, seaborn, streamlit, jupyter all pinned — **done 2026-10-27** (all 13 pins resolve exactly on Python 3.11)
- [x] T04 | Rubric 9 | Create virtual environment and install dependencies | Output: `.venv/` | Done when: `import sklearn, xgboost, shap, seaborn, streamlit` succeeds inside the venv — **done 2026-10-27** (exit 0, 13/13 pins exact, `pip check` clean)
- [x] T05 | Rubric 9 | Add `src/config.py` (paths + constants) | Output: `src/config.py` | Done when: importing it resolves every data/report/models path with **no absolute path** hard-coded — **done 2026-10-27** (20/20 paths absolute, portability re-derivation proven)
- [x] T06 | Rubric 9 | Add `src/data_loader.py` used by every notebook | Output: `src/data_loader.py` | Done when: loads both variants with `sep=";"`, returns `X, y`, and prints shape + class balance — **done 2026-10-27** (15/15 assertions pass; unknown variant raises)

### Phase 1 — Problem Definition & Business Understanding
- [x] T07 | Rubric 1 | Write problem statement, objectives and scope | Output: `docs/01_problem_definition.md` | Done when: one-paragraph statement + numbered objectives + explicit out-of-scope list — **done 2026-10-27** (single 179-word paragraph, 6 traceable objectives, 6 out-of-scope exclusions)
- [x] T08 | Rubric 1 | Map stakeholders to decisions and KPIs | Output: `docs/01_problem_definition.md` | Done when: each stakeholder is linked to the decision they make and the metric they care about — **done 2026-10-27** (6 stakeholders S1–S6, all with decision + typed KPI + measurability)
- [x] T09 | Rubric 1 | Build cost-of-errors model (call cost vs deposit value) | Output: `docs/01_problem_definition.md`, `src/cost.py` | Done when: FP/FN costs stated with assumptions and a break-even threshold is derived — **done 2026-10-27** (C=€1.50, V=€20.00, p*=7.50%, 20/20 sensitivity cells verified)

### Phase 2 — Dataset & Data Understanding
- [x] T10 | Rubric 2 | Write the data dictionary from `bank-names.txt` | Output: `docs/02_data_dictionary.md` | Done when: all features documented with type, allowed values, meaning and leakage risk (16-feature and 20-feature sets) — **done 2026-10-27** (all 17+21 columns documented, 44 assertions pass)
- [x] T11 | Rubric 2 | Profile structure and data quality | Output: `notebooks/01_data_understanding.ipynb`, `reports/figures/` | Done when: dtypes, null counts, duplicate rows, constant columns, `unknown` counts and sentinel ranges are all reported with evidence — **done 2026-10-27** (14/14 code cells execute clean, 53-check dictionary cross-check with 0 disagreements, 3 figures)

### Phase 3 — Preprocessing & Cleaning
- [x] T12 | Rubric 3 | Decide and implement `unknown` + sentinel handling | Output: `src/clean.py` | Done when: the `unknown` policy (category vs NaN) is documented with justification and `pdays = -1` becomes a `previously_contacted` flag — **done 2026-10-27** (96/96 assertions; flag from `previous > 0` per D21; 0 rows lost, target untouched)
- [x] T13 | Rubric 3 | Analyse outliers and skewed numerics | Output: `notebooks/02_preprocessing.ipynb`, figures | Done when: skew of `balance`, `campaign`, `previous`, `duration` is documented and the treatment decision (keep + robust scale vs winsorise) is justified — **done 2026-10-27** (8/8 cells execute clean, all four skews measured, decision D24 justified from tail-vs-body rates, 3 figures)
- [x] T14 | Rubric 3 | Build and check the cleaning module | Output: `src/clean.py` | Done when: it converts raw → processed with no errors and prints before/after shapes, dtypes and class balance — **done 2026-10-27** (both variants convert with **0 rows lost**, `python -m src.clean` exits 0, **15/15 checks pass each**; the checker is made *total* and proven able to fail — 8/8 injected faults reported, none raised)
- [x] T15 | Rubric 3 | Create the stratified train/test split | Output: `data/processed/train.csv`, `data/processed/test.csv` | Done when: `y` proportion in each split is within ±1% of the original and files reload cleanly — **done 2026-10-27** (4 files written for both variants; **21/21 checks** each; split rates within **0.0012pp** of the original, independently re-verified **28/28** with plain `read_csv`; byte-identical on a re-run)

### Phase 4 — EDA
- [x] T16 | Rubric 4 | Target distribution and class imbalance chart | Output: `reports/figures/01_target_distribution.png` | Done when: figure saved and a one-line viva insight is written next to it — **done 2026-10-27** (figure 1707×677 costing 66 KB, one-sentence viva pull-quote immediately after it; 4,231 / 31,937 = 11.6982%, 7.55 : 1; re-verified by **39/39 independent checks** and **byte-identical across two fresh runs**)
- [ ] T17 | Rubric 4 | Numeric distributions and skew | Output: `reports/figures/02_numeric_distributions.png` | Done when: one interpretation line exists per numeric variable
- [ ] T18 | Rubric 4 | Categorical features vs subscription rate | Output: `reports/figures/03_categorical_vs_target.png` | Done when: best and worst segments are identified with actual rates
- [ ] T19 | Rubric 4 | Campaign/history features vs subscription rate | Output: `reports/figures/04_campaign_vs_target.png` | Done when: timing, contact-type and previous-outcome insights are written
- [ ] T20 | Rubric 4 | Correlation heatmap of numeric features | Output: `reports/figures/05_correlation_heatmap.png` | Done when: strongest correlations with `y` are listed and multicollinearity is flagged
- [ ] T21 | Rubric 4 | Consolidate EDA findings | Output: `docs/04_eda_insights.md` | Done when: every figure has its one-line business interpretation and at least 5 insights are listed

### Phase 5 — Feature Engineering & Selection
- [ ] T22 | Rubric 5 | Design the `duration` leakage with-vs-without experiment | Output: `docs/05_leakage_audit.md` | Done when: the experiment (identical pipeline, feature set toggled) is specified and the leakage risk is documented
- [ ] T23 | Rubric 5 | Engineer additional features | Output: `src/features.py` | Done when: contact recency, campaign intensity and balance-band features are created, commented, and leak nothing from `y`
- [ ] T24 | Rubric 5 | Build the preprocessing `ColumnTransformer` | Output: `src/preprocess.py` | Done when: the pipeline fits on train and transforms test with no errors and output shapes are reported
- [ ] T25 | Rubric 5 | Feature selection / importance with justification | Output: `reports/figures/06_feature_importance.png`, notes in `docs/05_leakage_audit.md` | Done when: at least two methods (mutual information + model importance) are compared and kept/dropped features are justified

### Phase 6 — Model Selection & Implementation
- [ ] T26 | Rubric 6 | Baseline models: `DummyClassifier` + Logistic Regression | Output: `src/train.py`, artifacts in `models/` | Done when: both are trained inside a Pipeline and their CV ROC-AUC/PR-AUC are logged
- [ ] T27 | Rubric 6 | Random Forest model | Output: `src/train.py`, `models/` | Done when: trained in the same pipeline and CV scores are logged for direct comparison
- [ ] T28 | Rubric 6 | Gradient boosting model (XGBoost or LightGBM) | Output: `src/train.py`, `models/` | Done when: trained in the same pipeline and CV scores are logged for direct comparison
- [ ] T29 | Rubric 6 | Compare imbalance handling (`class_weight` vs SMOTE) | Output: `docs/06_imbalance.md` | Done when: variants are compared on PR-AUC/F1 with resampling applied **only inside CV folds** (no leakage)
- [ ] T30 | Rubric 6 | Hyperparameter tuning | Output: `src/tune.py`, `models/best_params.json` | Done when: `RandomizedSearchCV` with 5-fold StratifiedKFold completes and tuned model beats untuned on CV PR-AUC
- [ ] T31 | Rubric 6 | Select and persist the final model | Output: `models/final_model.joblib`, `docs/06_model_comparison.md` | Done when: the comparison table is complete, the chosen model is justified, and the saved artifact reloads and predicts

### Phase 7 — Model Evaluation & Validation
- [ ] T32 | Rubric 7 | Cross-validated metric table | Output: `reports/metrics_cv.csv` | Done when: ROC-AUC, PR-AUC, precision, recall and F1 are reported as mean ± std over 5 stratified folds
- [ ] T33 | Rubric 7 | Confusion matrix, ROC curve and PR curve on held-out test data | Output: `reports/figures/07_*.png` | Done when: all three are saved with the 0.5-threshold numbers annotated
- [ ] T34 | Rubric 7 | Threshold tuning against the cost model | Output: `docs/07_threshold.md`, threshold figure | Done when: the chosen threshold is justified by expected business cost rather than by accuracy
- [ ] T35 | Rubric 7 | Lift, gain and decile analysis | Output: `reports/figures/08_lift_gain.png`, `reports/lift_table.csv` | Done when: top-decile lift is quantified and translated into calls saved
- [ ] T36 | Rubric 7 | With-vs-without `duration` comparison | Output: `reports/duration_comparison.csv`, notes in `docs/05_leakage_audit.md` | Done when: both runs use identical settings and the performance gap is explained as leakage

### Phase 8 — Explainability & Business Insights
- [ ] T37 | Rubric 8 | Global SHAP analysis | Output: `reports/figures/09_shap_summary.png` | Done when: beeswarm plus importance bar are saved and the top drivers are explained in plain language
- [ ] T38 | Rubric 8 | Local SHAP explanations for exemplar clients | Output: `reports/figures/10_shap_local.png` | Done when: a true positive, a false positive and a false negative are each explained
- [ ] T39 | Rubric 8 | Business recommendations and campaign playbook | Output: `docs/08_recommendations.md` | Done when: at least 6 actionable recommendations are backed by the actual measured metrics

### Phase 9 — Deliverables
- [ ] T40 | Rubric 9 | Write the final report | Output: `docs/report.md` (+ PDF) | Done when: all nine rubric criteria are covered using only real numbers from saved runs
- [ ] T41 | Rubric 9 | Build the slide deck | Output: `docs/presentation.pptx` (or outline + export) | Done when: ≤ 15 slides, business-first narrative, results table taken from actual runs
- [ ] T42 | Rubric 9 | Finalise README and repo hygiene | Output: `README.md` | Done when: a clean clone can reproduce the results by following the README alone
- [ ] T43 | Rubric 9 | Build the interactive demo | Output: `app/app.py` (Streamlit) | Done when: it runs locally, accepts client attributes, and returns a probability with the top SHAP drivers
- [ ] T44 | Rubric 9 | End-to-end rerun from a clean clone | Output: verification checklist recorded in `PROGRESS.md` | Done when: fresh clone + install + run reproduces the reported metrics

### Phase 10 — Viva Preparation
- [ ] T45 | Rubric 9 | Viva Q&A bank | Output: `docs/viva_qa.md` | Done when: 25+ likely questions are answered, covering every rubric criterion
- [ ] T46 | Rubric 9 | Demo script and rehearsal checklist | Output: `docs/demo_script.md` | Done when: a 5-minute walkthrough is scripted and every claim traces to a saved figure or metric

---

## Files Created

| File | Contents | Created |
|---|---|---|
| `PROGRESS.md` | This tracker: status, decisions, 46-task checklist, session log | 2026-10-27 |
| `.gitignore` | Excludes `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `*.log`, `.DS_Store`, `._*`, `__MACOSX/`, editor folders | 2026-10-27 |
| `requirements.txt` | 13 pinned dependencies grouped by purpose, with a header noting the Python 3.11 ceiling on xgboost/shap | 2026-10-27 |
| `.venv/` | Project virtual environment (Python 3.11.9, 132 packages). **Git-ignored** — recreate with `python -m venv .venv` + `pip install -r requirements.txt` | 2026-10-27 |
| `src/config.py` | Single source of truth for paths and constants: derived `ROOT`, 20 absolute path objects, `RANDOM_STATE=42`, `TEST_SIZE=0.20`, `TARGET="y"`, `CSV_SEPARATOR=";"`, plus `ensure_dirs()` and `summary()` helpers | 2026-10-27 |
| `src/data_loader.py` | The single data entry point: `load_raw(variant)`, `load_xy(variant)` returning `(X, y)` with `y` encoded 1/0, `variant_report()` summary table, `ECONOMIC_COLUMNS`, and a runnable `python -m src.data_loader` self-check | 2026-10-27 |
| `src/clean.py` | Phase 3 (rubric 3, **T12**): the two cleaning decisions, implemented and justified in the module docstring — `unify_unknown()` (D22: keep `unknown` as an explicit category, normalise any NaN to it, never impute or delete; returns a per-column report) and `add_contact_history()` (D23: adds `previously_contacted` from `previous > 0` per D21 and `days_since_contact` with the sentinel → NaN, keeps raw `pdays`, warns when the sentinel and `previous` disagree). Plus `clean_frame()` / `clean()` (loads via the loader, so D13 holds) with no-touch invariants, `print_report()` and a runnable `python -m src.clean` self-check. **T14** turned it into the checked raw → processed converter: `process(variant)` (the single Phase 3 entry point, D13 respected), `validate()` returning **15 labelled checks as data** that are *total* and proven able to fail (D25: positional comparisons, an unexecutable check counts as FAILED, never raised), `print_conversion_report()` printing before/after shapes, dtypes and class balance, and a CLI that exits non-zero on any failure | 2026-10-27 |
| `src/split.py` | Phase 3 (rubric 3, **T15**): the stratified train/test split. `split_frame()` (seeded, stratified, shuffled; returns train/test/report), `encode_target()`, `write_split()` (converts via `clean.process` so every half inherits T14's checks, then writes both files), `load_split()` with the declared-dtype guard, `validate_split()` (**15 checks that measure the frames, never the report** - D27) and `validate_round_trip()` (3 checks per file), plus `_duplicate_rows_spanning()` which reports duplicate source rows that straddle the halves instead of removing them (D26) | 2026-10-27 |
| `data/processed/train.csv`, `test.csv`, `train_additional.csv`, `test_additional.csv` | Phase 3 (**T15**) outputs: the fixed evaluation halves, 36,168 / 9,043 rows (classic) and 32,950 / 8,238 (additional), 19 and 23 columns, target `y` still as text and `previously_contacted` as a real boolean. Byte-identical on a re-run of `python -m src.split` | 2026-10-27 |
| `notebooks/03_eda.ipynb` | Phase 4 (rubric 4, **T16**): 17 cells (8 code) with stored outputs — a provenance-guarded load of the **train half only**, the class counts with share and cumulative share, the two-panel chart, the baselines the imbalance forces (always-`no` 88.3018% accuracy, PR-AUC 0.1170, ROC-AUC 0.50, 8.55 calls per subscription), the stratification sanity line against the population's 11.698480%, the 5-fold stratified positive counts (846/846/847/846/846), **13 in-notebook assertions** and a reproducibility record. Sections `2.`–`5.` are reserved for T17–T20 | 2026-10-27 |
| `reports/figures/01_target_distribution.png` | Phase 4 (**T16**) figure, 1707×677 px: class counts with counts and shares annotated beside the 100%-share bar carrying the **7.55 : 1** ratio | 2026-10-27 |
| `src/cost.py` | Cost-of-errors model: labelled `ASSUMPTIONS` (C = €1.50, V = €20.00), `break_even_threshold()` → 0.0750, `expected_net_value_per_call()`, `should_call()`, `cost_of_errors()` (12.3:1 ratio), `expected_campaign_value()`, `best_threshold()`, `sensitivity_table()`, and a runnable `python -m src.cost` report | 2026-10-27 |
| `docs/02_data_dictionary.md` | Phase 2 (rubric 2, **T10**): all 17 + 21 columns with measured type, values/range, meaning and leakage risk; variant difference summary; `unknown` and `pdays` sentinel tables; leakage assessment; self-check | 2026-10-27 |
| `notebooks/01_data_understanding.ipynb` | Phase 2 (rubric 2, **T11**): 30-cell quality profile of both variants with stored outputs — inventory, dtypes, nulls, duplicates, concentration, `unknown` profile, `pdays` sentinel **plus a §7.1 test of whether the sentinel can be trusted**, skew/outliers, negative-value checks, 9 written findings and a 53-check cross-check against the T10 dictionary | 2026-10-27 |
| `reports/figures/00_quality_pdays_sentinel.png`, `00_quality_numeric_skew.png`, `00_quality_unknown_profile.png` | Phase 2 (**T11**) quality figures, `00_`-prefixed so they cannot collide with the Phase 4 EDA figures (`01_`–`05_`) | 2026-10-27 |
| `notebooks/02_preprocessing.ipynb` | Phase 3 (rubric 3, **T13**): 18-cell outlier/skew analysis with stored outputs — T12 invariance re-proved, skew/kurtosis/quantile profile of all four named columns, tail-vs-body subscription rates, a five-option treatment comparison (keep, arcsinh, robust scale, winsorise 1–99, winsorise 5–95) with rows-rewritten / distinct-values-lost / order-preserved, the `log1p`-on-negatives trap quantified, the **IQR = 0** degenerate case detected, 7 written findings and the D24 decision | 2026-10-27 |
| `reports/figures/00_preprocessing_distributions.png`, `00_preprocessing_treatments.png`, `00_preprocessing_signal_in_tail.png` | Phase 3 (**T13**) figures: raw vs `arcsinh` distributions, skew remaining per treatment, and the tail signal (balance deciles, campaign bands) that decided D24 | 2026-10-27 |
| `docs/01_problem_definition.md` | Phase 1 (rubric 1): 179-word problem statement, business objective, six measurable objectives O1–O6 each traced to a task, six out-of-scope exclusions, canonical dataset/target naming table (**T07**), plus **T08**: a six-stakeholder table (S1–S6) with decision + typed KPI + measurability per row, concrete KPI definitions with numeric targets, why accuracy is excluded, and the named stakeholder tension over the threshold | 2026-10-27 |
| `data/raw/`, `data/processed/`, `notebooks/`, `src/`, `models/`, `reports/figures/`, `app/`, `docs/` | Project folder skeleton, each empty directory held open by a `.gitkeep` | 2026-10-27 |

**Pre-existing assets already in the repo (not created by me):**

| File | Contents |
|---|---|
| `data/raw/bank-full.csv` | UCI Bank Marketing, 45,211 × 17, semicolon-delimited (**primary dataset**) |
| `data/raw/bank.csv` | 10% random sample, 4,521 × 17 |
| `data/raw/bank-names.txt` | Data dictionary and citation (Moro et al., 2011) for the classic set |
| `data/raw/bank-additional/bank-additional-full.csv` | Richer variant, 41,188 × 21 (adds 5 economic indicators) |
| `data/raw/bank-additional/bank-additional.csv` | 10% sample of the above, 4,119 × 21 |
| `data/raw/bank-additional/bank-additional-names.txt` | Data dictionary for the additional set |

---

## Session Log

### 2026-10-27 — Session 1
**Completed:** Located and cloned the project repo; accessed and profiled both dataset variants;
added the `bank-additional/` dataset files to the repo and committed them as `8278fae`;
wrote this tracker (`PROGRESS.md`) with the full 46-task plan.
**Blocked:** The commit `8278fae` (bank-additional files) is still **unpushed** — GitHub
authentication is not available in this environment (`gh` is not logged in, no cached
credentials), so a Git Credential Manager sign-in is required. See Open Questions.
**Next:** T01 — create the repo folder structure and `.gitignore`.

### 2026-10-27 — Session 2
**Completed:** **T01** — created the eight-folder project skeleton (`data/raw`, `data/processed`,
`notebooks`, `src`, `models`, `reports/figures`, `app`, `docs`), each empty one held open with a
`.gitkeep`, and wrote `.gitignore`. Verified with `git check-ignore` that `.venv/`, `__pycache__/`,
`.ipynb_checkpoints/`, `*.log`, `.DS_Store` and `__MACOSX/` are ignored, and that no dataset path
is accidentally ignored; `git status` shows only the intended new paths.
**Blocked:** unchanged — the bank-additional commit `8278fae` is still unpushed pending GitHub sign-in.
**Next:** T02 — move the raw datasets under `data/raw/` behind a checksum check.

### 2026-10-27 — Session 3
**Completed:** **T02** — moved all six raw data files from the repo root into `data/raw/` using
`git mv` (so git records six clean renames and history is preserved): `bank-full.csv`, `bank.csv`,
`bank-names.txt` and the whole `bank-additional/` folder. Verified by MD5 checksum before and after
the move — all six files are byte-identical (`diff` reported no differences) — and by reloading all
four datasets from the **new** paths with pandas, which returned exactly `(45211, 17)`, `(4521, 17)`,
`(41188, 21)` and `(4119, 21)`. The obsolete `data/raw/.gitkeep` was removed now that the folder
has content. No file was lost or duplicated.
**Blocked:** unchanged — commit `8278fae` (bank-additional dataset) is still unpushed pending GitHub sign-in.
**Next:** T03 — pin the dependency list in `requirements.txt`.

### 2026-10-27 — Session 4
**Completed:** **T03** — wrote `requirements.txt` with 13 concretely pinned packages. Versions were
taken from the live environment and from PyPI, not from memory. Found and worked around a real
constraint: **the newest `xgboost` (3.4.1) and `shap` (0.54.1) require Python ≥ 3.12**, while this
machine runs **Python 3.11.9**, so the file pins the newest versions that actually support 3.11
(`xgboost==3.2.0`, `shap==0.51.0`) and the header documents the ceiling. Verified with
`pip install --dry-run --report` that all 13 pins resolve on Python 3.11 with **no version
substitution** (13/13 satisfied exactly); 5 still need fetching (xgboost, imbalanced-learn, shap,
jupyterlab, ipykernel) plus their transitive deps, and 8 are already present at the pinned version.
**Blocked:** unchanged — commit `8278fae` (bank-additional dataset) is still unpushed pending GitHub sign-in.
**Next:** T04 — build the project venv from `requirements.txt` and import-check it.

### 2026-10-27 — Session 5
**Completed:** **T04** — created `.venv` (Python 3.11.9) and installed all 13 pinned packages
(`INSTALL_EXIT=0`, 132 packages in the environment). Verified three ways: (1) all **13/13 pins are
installed at exactly the pinned versions** inside the venv, read back with `importlib.metadata`;
(2) the five packages named in the done-condition import successfully inside the venv —
`sklearn 1.7.2`, `xgboost 3.2.0`, `shap 0.51.0`, `seaborn 0.13.2`, `streamlit 1.52.2`, plus
`pandas 2.3.3`; (3) `pip check` reports **"No broken requirements found"**. An assertion also
confirmed the interpreter really is the venv (`sys.prefix` ends in `.venv`), not system Python.
Confirmed `.venv/` is caught by `.gitignore`.

**Activation command (every later task must run inside this venv):**
```bash
source .venv/Scripts/activate      # Git Bash / WSL, from the repo root
.venv\Scripts\activate.bat         # cmd.exe
.venv\Scripts\Activate.ps1         # PowerShell
```
Non-interactive runs can skip activation and call `.venv/Scripts/python` directly.

**Note:** the install was interrupted once by a dropped connection after pip had already finished
(exit 0); the state was re-read and re-verified rather than re-run, so no duplicate work was done.
**Blocked:** unchanged — commit `8278fae` (bank-additional dataset) is still unpushed pending GitHub sign-in.
**Next:** T05 — write `src/config.py` with relative-derived absolute paths and `RANDOM_STATE = 42`.

### 2026-10-27 — Session 6
**Completed:** **T05** — wrote `src/config.py`: `ROOT` derived from `Path(__file__).resolve().parents[1]`,
20 absolute path objects (folders, both dataset variants, and the T15 split files), and the run
constants `RANDOM_STATE=42`, `TEST_SIZE=0.20`, `TARGET="y"`, `POSITIVE_LABEL="yes"`,
`CSV_SEPARATOR=";"`. Added `ensure_dirs()` (idempotent, never touches `data/raw/`) and a `summary()`
self-check runnable as `python -m src.config`. Removed the now-redundant `src/.gitkeep`.
Verified: the done-condition command prints an existing absolute path; **all 20/20 exported paths are
absolute**; only `TRAIN_CSV`/`TEST_CSV` are absent, which is correct because T15 creates them;
`grep` confirms **no drive letter, username or absolute path appears anywhere in the source**; and
portability was proven by copying `config.py` alone into an unrelated folder on a different drive,
where `ROOT` re-derived to the new location instead of the original.
**Blocked:** unchanged — commit `8278fae` (bank-additional dataset) is still unpushed pending GitHub sign-in.
**Next:** T06 — add `src/data_loader.py` as the single data entry point for all notebooks.

### 2026-10-27 — Session 7  (Phase 0 complete)
**Completed:** **T06** — wrote `src/data_loader.py` with `load_raw(variant)`, `load_xy(variant)`,
`variant_report()`, the `ECONOMIC_COLUMNS` tuple and a `python -m src.data_loader` self-check.
Verified with **15/15 programmatic assertions**: `full` loads as exactly `(45211, 17)` with an
yes-rate of **11.698%** (matches the ~11.7% brief); `additional` loads as `(41188, 21)` and keeps all
five economic columns; `sample` loads as `(4521, 17)`; `X` has exactly one column fewer than the raw
frame and no target column; `y` is integer-encoded, contains only `{0, 1}`, and sums to the true count
of `yes` rows (cross-checked against the raw column). Also confirmed that an unknown variant name
raises `ValueError` with the valid options listed rather than silently returning a default file,
and that `duration` is still present by design (D14).

**Phase 0 (Setup) is now complete: T01–T06 all ticked.** The project has a reproducible folder
structure, pinned dependencies, a working virtual environment, path-independent config and a single
verified data entry point. Phase 0 accounts for 6 of the 13 tasks under rubric criterion 9.
**Blocked:** unchanged — commit `8278fae` (bank-additional dataset) is still unpushed pending GitHub sign-in.
**Next:** T07 — write the problem statement, objectives and scope (starts Phase 1).

### 2026-10-27 — Session 8  (Phase 1 started)
**Completed:** **T07** — wrote `docs/01_problem_definition.md` covering rubric criterion 1. It
contains a single 179-word problem statement, a one-sentence business objective, **six numbered
objectives O1–O6** (each with a stated measurement method *and* the task ID that delivers it), an
explicit out-of-scope section with **six numbered exclusions**, and a naming table that pins the
same terms the code uses (`bank-full.csv` at `data/raw/bank-full.csv`, target `` `y` ``, 11.7%
positive rate). Verified with **13/13 programmatic assertions**, including a structural check that
the problem statement really is one paragraph block, that no objective lacks a measurement method
or a task ID, and that the quoted class balance matches the value measured in T06. Removed the
now-redundant `docs/.gitkeep`.

**Scope discipline:** stakeholders/KPIs were deliberately left to T08 and the cost-of-errors model
to T09, so each produces its own artefact rather than being half-written here.
**Blocked:** unchanged — commit `8278fae` (bank-additional dataset) is still unpushed pending GitHub sign-in.
**Next:** T08 — map stakeholders to their decisions and KPIs.

### 2026-10-27 — Session 9
**Completed:** **T08** — added §5 (Stakeholders, Decisions and KPIs) to `docs/01_problem_definition.md`:
a **six-stakeholder table S1–S6** (marketing campaign manager, telemarketing team lead, bank
management/budget owner, data science team, compliance officer, and the affected customer), where
every row states the decision taken, the KPI they are judged on, whether it is a **business or a
model** metric, and whether it is **measurable from this dataset or explicitly assumed**. Added
§5.2 concrete KPI definitions with numeric targets, §5.3 the reason accuracy is excluded (an
all-"no" model scores ≈88.3%), and §5.4 the named stakeholder tension between conversion rate and
net value, which T34 must resolve. Renumbered the later sections so the document reads 1–7 in order.

Verified with **31/31 assertions**, including a per-row structural check on all six stakeholders
(decision present, KPI present and non-vague, typed business/model, measurability labelled) plus a
**T07 regression re-run** on the same file, since editing it could have damaged the earlier
done-condition. Both suites pass.

**Honesty note:** the KPI targets in §5.2 (lift > 2×, ROC-AUC ≥ 0.75) are *targets*, logged as D15 so
they cannot later be mistaken for measured results; two stakeholder KPIs are explicitly labelled
*not measured* rather than quietly implied (D16).
**Blocked:** unchanged — commit `8278fae` (bank-additional dataset) is still unpushed pending GitHub sign-in.
**Next:** T09 — build the cost-of-errors model in `src/cost.py` and derive the break-even threshold.

### 2026-10-27 — Session 10  (Phase 1 complete)
**Completed:** **T09** — wrote `src/cost.py` and added §6 (Cost of Errors and the Break-Even Threshold)
to `docs/01_problem_definition.md`. Derived the break-even threshold analytically from `EV = p·V − C`
rather than asserting a cut-off: with **C = €1.50 per call** and **V = €20.00 per subscription**,
**p\* = 0.0750 (7.50%)**. Priced both errors (FP €1.50, FN €18.50 net of the call saved, ratio
**12.3 : 1**), added a 5×4 sensitivity grid, and a sanity check evaluating the whole campaign.
Verified with **24 assertions** in two suites: exact arithmetic (EV at p\* equals zero to 1e-12,
`should_call` flips either side of p\*), boundary behaviour (threshold 0 calls everyone, >1 calls
nobody), a shape-mismatch guard, and a synthetic **correctly-calibrated** case where the empirical
net-value-optimal sweep returned **0.06** against the analytic **0.075** and selected exactly the
profitable group. Also confirmed every one of the **20 sensitivity cells matches `cost.sensitivity_table()`**
and that all four dataset figures quoted in the docstring are measured, not remembered (mean
`duration` 258.2 s, median balance €448, mean balance €1,362.3).

**The finding worth carrying forward (D18):** p\* = 7.50% is *below* the 11.7% base rate, so the
average client is still worth calling (+€0.84 expected). "Targeting reduces campaign cost" is
**not** supported by these assumptions; the supported claim is that targeting converts a *fixed* call
budget better. The crossover is V* = C ÷ base rate = **€12.82** — above that value, call broadly;
below it, the model must exclude people. This is documented in §6.5 rather than buried, and it is a
real constraint on how the results in Phase 8 may be described.

**Two verification failures were investigated and both were bugs in my test code, not in the
deliverable:** Python's `round()` uses banker's rounding (1.25 → 1.2) where the document used
conventional half-up (1.3), and my parser split a section on `---`, which also occurs inside a
markdown table separator. Both test defects were fixed and the checks re-run clean; no assertion
was weakened and no code was changed to make a check pass.

**Phase 1 is now complete (T07–T09), so rubric criterion 1 is fully covered: 3/3 tasks.**
**Blocked:** unchanged — commit `8278fae` (bank-additional dataset) is still unpushed pending GitHub sign-in.
**Next:** T10 — write the data dictionary for both dataset variants (starts Phase 2).

### 2026-10-27 — Session 11  (Phase 2 started)
**Completed:** **T10** — wrote `docs/02_data_dictionary.md`, documenting **all 17 columns** of the
classic set and **all 21** of the additional set with five fields each (name, measured type, measured
allowed values/range, plain-English meaning, leakage risk). Every number was measured from the files
with `src/data_loader.py` rather than copied from the supplied documentation.

**Verification caught a real content error in my own first draft, and it mattered.** I had listed
`balance` as a column shared by both variants and `month` as classic-only. The data says the
opposite: **the additional set has no `balance` column at all** (it replaced the client's own balance
with five macro-economic indicators), and it **does** carry `month` — it is `day` alone that is
replaced by `day_of_week`. The corrected split is 14 shared + 2 classic-only (`balance`, `day`) = 16,
and 14 shared + 6 additional-only = 20. I also corrected §5.1 and the §7 comparison table, and
turned the discovery into decision **D19** (variants are complementary, not superset/subset) because
it has real downstream consequences: any `balance`-based feature engineered in Phase 5 cannot be
transferred to the additional variant.

Other findings documented: the extra set's **12 exact duplicate rows** (the classic set has none),
the **different `pdays` sentinels** (−1 vs 999) with counts 36,954 and 39,673, the eight columns
carrying `unknown` (including `poutcome` at 81.7% and the additional set's `default` at 20.9%), the
classic set's `contact` being 28.8% `unknown`, and `duration`'s 🔴 post-outcome leak status.
Verified with **44 assertions** in two suites (all columns present for both variants; every quoted
count, range and percentage re-checked against the loaded data; sentinel counts; leakage flags;
five economic columns confirmed additional-only), plus a regression re-run after the correction.
**Blocked:** unchanged — commit `8278fae` (bank-additional dataset) is still unpushed pending GitHub sign-in.
**Next:** T11 — profile structure and data quality into `notebooks/01_data_understanding.ipynb`.

### 2026-10-27 — Session 12  (Phase 2 complete)
**Completed:** **T11** — built `notebooks/01_data_understanding.ipynb`, a 30-cell quality profile that
is executed with stored outputs via `jupyter nbconvert --execute` (14/14 code cells ran, zero error
outputs), plus three `00_quality_*.png` figures. It covers, for **both** full variants: inventory and
checked shapes, dtypes and cardinality, true nulls, exact duplicates, constant/near-constant columns,
an `unknown` profile, the `pdays` sentinel, skew/IQR outliers, and negative-value verdicts — each
followed by a written insight line rather than raw output alone.

**Verification caught a real content problem, and the mechanism is worth recording.** The §12 block
parses `docs/02_data_dictionary.md` and compares **53 documented figures** against values measured
here. On the first executed run it reported **six disagreements — all six of them defects in my own
parser**, not drift: a rate cell that captured `5,289` as `5.0`, a count cell that concatenated
`36,954` with `81.7` into `36954817`, and a tuple parser that dropped the minus sign from `−1`. I
fixed the parser (`first_int`, `first_pct`, sign-aware `every_int`) and re-ran: **49 MATCH, 0
disagreements**, so there is no real drift between the dictionary and the data. A parser bug and a
genuine data change look identical on screen, which is why each disagreement was classified before
being acted on rather than "fixed" by editing the dictionary.

**The substantive find of the session (new decision D21).** A new §7.1 tests the documented claim
that the `pdays` sentinel means "client was not previously contacted", using `previous` as an
**independent witness**. The primary set passes: `pdays != −1` and `previous > 0` select exactly the
same **8,257** rows, zero contradictions. The additional set **fails**: **4,110** of its 39,673
sentinel rows have `previous` 1–6, and in every one of those `poutcome` is `failure`. So 999 there
also covers clients whose previous campaign failed. Consequence: T12 must derive
`previously_contacted` from `previous > 0` (which agrees exactly with `poutcome != nonexistent` in
both variants), not from the sentinel — otherwise 4,110 clients are mislabelled and get a wrong
recency value. This is precisely the class of defect no null count, dtype check or duplicate test
can see: every value is a valid integer and nothing is missing.

Three further differences were reported as **NUANCE** rather than as errors — the number is right but
the documented *meaning* is incomplete — and the dictionary was deliberately **not** edited, since
T11 reports and T12 owns cleanup: (1) the additional sentinel count is presented as "not previously
contacted" when 4,110 of those rows had prior contacts; (2) that row's "other values" cell reads
`0 – 999`, repeating the sentinel inside the column meant to exclude it (measured non-sentinel range
**0 – 27**); (3) the classic row reads `0 – 871` where the measured non-sentinel minimum is **1**
(no classic row has `pdays = 0`). One documented figure is reported as **SKIPPED** rather than
guessed: the additional 10% sample's shape is not measurable here because it is not exposed as a
variant of `src/data_loader.py`.

**Verification:** 14/14 code cells executed with no error output; **24/24 required-evidence
assertions** present in the stored outputs (both shapes, 11.70% balance, zero NaNs, 12 duplicates,
both sentinels, the 4,110 contradiction, all three figures, the agreement line, and both the
`balance`-absent and `month`-present re-checks); the 53-check cross-check reconciles as
49/3/1; three figures on disk; and **zero absolute paths** in the notebook source, so it stays
portable. Removed the now-redundant `notebooks/.gitkeep` (as with `src/` and `docs/` earlier). The
throwaway generator used to build the notebook was deleted; the `.ipynb` is the self-contained
deliverable.

**Phase 2 is now complete (T10–T11), so rubric criterion 2 is fully covered: 2/2 tasks.**
**Blocked:** unchanged — commit `8278fae` (bank-additional dataset) is still unpushed pending GitHub sign-in.
**Next:** T12 — implement `unknown` and sentinel handling in `src/clean.py` (starts Phase 3), deriving
`previously_contacted` from `previous > 0` per D21.

### 2026-10-27 — Session 13  (Phase 3 started)
**Completed:** **T12** — wrote `src/clean.py`, implementing the two cleaning decisions that must be
settled before any encoding or scaling, with the justification recorded in the module docstring and
re-measured by a 96-assertion suite.

**Decision D22 — the `unknown` policy: keep it as an explicit category.** Both alternatives were
measured and rejected rather than dismissed on principle: imputing would **fabricate** up to
**81.75%** of classic `poutcome` (36,959 rows, where `unknown` genuinely means "no previous
campaign", not a random gap), and listwise deletion would remove **37,369** classic rows
(**82.65%** — more than four clients in five) and 10,700 additional rows (25.98%). Keeping the level
preserves every row and matches the custodian's own encoding, and it leaves open whether the
missingness is informative instead of assuming it is noise. `unify_unknown()` goes one step further
and normalises any NaN in a categorical column to the same literal marker, so there is exactly one
representation of missingness; on these files it converts **0** cells, and the report says so rather
than implying work was done. A stated limitation is recorded too: this is not a MAR/MNAR analysis,
and keeping a level is not evidence that it matters.

**Decision D23 — the sentinel is removed as a value, not just flagged.** `add_contact_history()` adds
`previously_contacted` (bool, from `previous > 0` per **D21**) and `days_since_contact` (the real
`pdays` value, **NaN** at the sentinel). NaN over an invented number, because a day count we do not
have cannot be estimated, and XGBoost (D08) treats missing values as a learned branch direction. Raw
`pdays` is retained for traceability; excluding it from the model matrix stays T24's decision rather
than a silent drop here.

**The D21 payoff shows up as a warning, not a silent fix.** For the classic set the sentinel and
`previous` agree perfectly — 36,954 sentinel rows, 8,257 previously contacted, **0 contradictions** —
so the report is clean. For the additional set it emits: *"4,110 rows carry the 999 sentinel while
`previous` > 0, so the sentinel alone would mislabel them as never contacted. The flag is derived
from `previous > 0` (D21) and these rows correctly get no day count."* Those 4,110 rows therefore
have `previously_contacted = 1` **and** `days_since_contact = NaN`, which is the honest description of
what the data actually supports.

**Verification:** `python -m src.clean` exits 0 and prints the full report for both variants; the
suite passed **96/96 assertions**, covering the no-touch invariants (row count unchanged, target
identical elementwise and to 1e-12 on the rate), every `unknown` count and level set preserved,
idempotency, the flag equalling `previous > 0` elementwise, NaN at exactly the sentinel positions,
no sentinel surviving in the recency column, per-variant sentinel use (confirming the additional set
contains no `-1` at all, so the classic convention is not hard-coded), the two error paths, and —
importantly — **re-measurement of every figure quoted in the justification** (37,369 / 82.65%, the
81.75% `poutcome` share, and the 4,110 contradiction count), so the prose cannot drift from the data.
Also confirmed `clean.py` contains no `read_csv` call, keeping D13 intact.

**Phase 3 has started: 1 of its 4 tasks is done.**
**Blocked:** unchanged — commit `8278fae` (bank-additional dataset) is still unpushed pending GitHub sign-in.
**Next:** T13 — analyse outliers and skewed numerics into `notebooks/02_preprocessing.ipynb`.

### 2026-10-27 — Session 14
**Completed:** **T13** — built and executed `notebooks/02_preprocessing.ipynb` (18 cells, 8 code, outputs
stored) plus three `00_preprocessing_*.png` figures, and recorded the treatment decision as **D24**.
It re-proves T12's invariance first (every original column bit-identical after cleaning), then
profiles the four named columns, tests whether their tails are signal, and compares five candidate
treatments on the same axes.

**The decision (D24): keep every value; no winsorising.** Measured skew is `previous` **+41.85**,
`balance` **+8.36**, `campaign` **+4.90**, `duration` **+3.14** — all well past the ±1 rule of thumb,
with excess kurtosis 4,507 and 141 on the first two. The reason not to clip is that the tails are
where the signal lives: the subscription rate climbs from **6.90%** in the lowest `balance` decile to
**16.61%** by the ninth, and falls from **14.60%** at one `campaign` contact to **3.93%** at 11+
(`previous` splits the same way, 9.16% to 25.63%). Winsorising at 5–95 would rewrite **13,054 rows**
and destroy **3,366 distinct values**, merging groups that differ by more than 10 pp. Trees need no
transform at all; the Logistic Regression baseline gets a `RobustScaler`, which is affine and
order-preserving, so it changes leverage without discarding a client.

**Executing the notebook caught three real defects, each found by reading the output rather than
trusting the code.** (1) The obvious log-like transform, `log1p`, is **undefined for negative
values**, and `balance` contains 3,766 of them — it produced **3,716 NaN and 50 -inf**, i.e. it would
have silently deleted exactly the legitimate overdrafts the project is required to keep. Switched to
`arcsinh`, which is defined on all reals, and the count is now printed as evidence. (2) `previous`
has an **IQR of 0** (81.74% of rows are exactly 0), so a median/IQR rescale **divides by zero**; it
was returning infinities and a `NaN` skew that read like a measurement. The notebook now checks the
IQR first and omits the option with a stated reason — a genuine constraint handed to T24. (3) pandas
prints a float `NaN` **without consulting** a `to_string` formatter, so the undefined row still
showed as `NaN`; the comparison table is now built from strings. All three were defects in this
notebook, not in the data.

**Two prose claims were corrected after measuring, not argued away.** I had written that `campaign`
and `duration` were skewed "much more moderately" (they measure +4.90 and +3.14 — heavily skewed by
the notebook's own threshold), and the decision's headline gap initially quoted `duration`'s +50.3 pp
— which is the **leakage** column and never reaches the model, so it is now reported separately as
evidence for D05 and the headline gap is quoted from the columns that do reach it.

**Verification:** 8/8 code cells executed with no error output; **20/20 required-evidence
assertions** present in the stored outputs; **13/13 numeric claims made in the prose re-checked
against the printed values** (the skews, the four rates, the three IQR-flag counts and the 81.74%
zero share), so the write-up cannot drift from the data; 3 figures on disk; 0 absolute paths. The
earlier log1p warnings are now deliberately silenced around the diagnostic so the saved output shows
the finding instead of numpy noise.

**Blocked:** unchanged — commit `8278fae` (bank-additional dataset) is still unpushed pending GitHub sign-in.
**Next:** T14 — turn `src/clean.py` into the checked raw → processed converter.

### 2026-10-27 — Session 15
**Completed:** **T14** — `src/clean.py` is now the checked raw → processed converter, not just the two
decision functions. `process(variant)` is the single Phase 3 entry point (T15 will call it), it goes
through the loader rather than reading a CSV (**D13**), and it prints what a marker needs to see: the
shape, the rows lost, the target before and after, the dtype inventory on both sides, and the class
balance — plus a pass/fail line per check.

**The measured result: nothing is lost and nothing is silently altered.** Classic **45,211 × 17 →
45,211 × 19**, additional **41,188 × 21 → 41,188 × 23**, with **0 rows lost** in both and the target
counts identical elementwise (39,922 no / 5,289 yes → 11.6985% unchanged; 36,548 / 4,640 → 11.2654%)
— only the two derived columns are added. `python -m src.clean` exits **0** with **15/15 checks
passing on each variant**.

**The part worth recording is that a *checked* module has to be shown able to fail, and doing so found
a real defect in the checker itself (new decision D25).** I injected eight faults — drop a row, edit an
original value, flip the target, leave a sentinel in the recency column, derive the flag from the
sentinel instead of `previous`, fill an unknown day count with 0, add an extra column, turn an
`unknown` into a null — and the **first** of them made `validate()` **raise**
`ValueError: Can only compare identically-labeled Series objects` instead of returning a failed check.
That is the worst possible behaviour for a guard rail: it crashes precisely when a diagnosis is being
asked for, and because the exception escaped, the run reported nothing at all. Worse, the same design
meant a check that never executed would simply be **absent** from the count, so a future "15/15 passed"
could quietly mean "14 checks ran". Comparisons are now **positional** (`.to_numpy()`), which also
removes a second, subtler bug: an indexed pandas comparison **silently aligns** on a misaligned frame
rather than complaining. Every check now runs inside a helper that records an unexecutable check as a
**FAILED** one, with the reason (`check could not run: ValueError: ...`), so the count is honest.
Re-running all eight faults: **8/8 reported as failures, none raised**.

**Verification:** baseline frame **15/15 checks, 0 crashes**; **8/8 injected faults reported** as
FAILED checks with reasons (the dropped-row case now fails 7 checks and names the shape mismatch
instead of aborting); the module CLI exits **0** for both variants with **15/15 each**; and the **T12
invariants were re-run after the edit — 68/68 assertions** (rows unchanged, target untouched, every
original column bit-identical, `unknown` counts and level sets preserved, flag ≡ `previous > 0`,
NaN exactly at the sentinel, sentinel not surviving as a value, real day counts preserved, idempotent
re-run, the per-variant sentinel conventions, `clean.py` containing no `read_csv`, and the **4,110**
D21 warning rows in the additional variant). Both throwaway scripts were deleted afterwards; the
module is the deliverable.

**Phase 3 is now 3 of its 4 tasks done**, so criterion 3 stands at 3/4 with only T15 outstanding.
**Blocked:** unchanged — commit `8278fae` (bank-additional dataset) is still unpushed pending GitHub sign-in.
**Next:** T15 — the stratified train/test split written to `data/processed/train.csv` and `test.csv`.

### 2026-10-27 — Session 16  (Phase 3 complete)
**Completed:** **T15** — wrote `src/split.py` and produced the four split files, so the evaluation
halves are now fixed on disk: `data/processed/train.csv` (**36,168 rows**) and `test.csv` (**9,043**)
for the classic variant, plus `train_additional.csv` (32,950) and `test_additional.csv` (8,238) for
the comparison pipeline. Two path constants were added to `src/config.py` so the module stays the
only file that knows where the halves live. **Phase 3 is complete: 4/4, criterion 3 fully covered.**

**The measured result beats the bar by three orders of magnitude.** The task is graded on the split's
`y` proportion landing within ±1% of the original; the deltas are **−0.000294pp** (train) and
**+0.001177pp** (test) for the classic variant and **+0.000137pp** / **−0.000547pp** for the
additional one — verified a second time from outside the module, by plain `pd.read_csv` reading the raw
files and the written files with no project code in the path (**28/28 independent assertions**).
Missing values are conserved exactly (36,954 sentinel rows → 29,584 + 7,370), no dtype changes, and a
second run rewrites all four files **byte-identically** (`sha256sum` compared before and after), so
the split is reproducible rather than merely seeded.

**The most valuable finding is about the row order (new decision D26).** The classic file is ordered by
campaign date — first row `may 5`, last row `nov 17` — and the positive rate is not remotely uniform
across it: **2.99%** in the first row-order decile rising to **47.09%** in the last. The obvious
"take the first 80%" split would therefore put **31.58%** positives into the test half against 11.70%
overall: a 19.9pp swing, twenty times the tolerance being graded, and it would have been invisible in
a shape check. Stratifying and shuffling is what prevents it. The same measurement is why the module
documents a limitation rather than claiming more than it can: evaluating on rows drawn at random from
the **same months** does not show that the model would hold up in a future campaign, and no result from
this split should be presented as if it did.

**Fault injection found a defect in the split checker itself (new decision D27), and it was the kind
that would have produced a confident green run.** Stripping every positive out of the test half — test
rate 0.00% against 11.70% overall — **passed**. The reason: `validate_split` was reading
`report["positive_rate_test"]`, i.e. checking a description of the split instead of the split. The
report still described the honest split, so the checker agreed with itself. Rates, shares and NaN
counts are now recomputed from the frames on every check, and a new check fails if any reported figure
disagrees with its measurement by more than 1e-12, so the two representations are held together.
Every injected fault is now caught: dropped row, a row appearing in both halves, the stripped test
half, a label-sorted split, a reordered column set, an altered dtype, a seed that disagrees with the
split, shuffled rows in the file, an edited value in the file, a boolean reloaded as an integer, and a
missing column — **12/12 reported as failures, none raised**.

**One claim of mine was wrong and the measurement corrected it.** I had written that "CSV cannot store a
boolean", so `previously_contacted` would need restoring. A probe showed pandas' parser infers `bool`
for exactly-`True`/`False` text, so the column round-trips as a boolean on its own. The comments and
the module docstring now say what was measured — the restoration is a **guard** (it raises on anything
that is not a boolean), not a conversion — and the round-trip check is what actually holds the dtype
contract. The same discipline caught a second self-inflicted bug earlier: the reload check compared
index-**labelled** frames, and since the CSVs are written `index=False` a reload carries a fresh
`RangeIndex`, so it reported all 19 columns as different. The fix was to make the contract explicit
(row order and values are preserved; index labels are not part of the artifact) rather than to delete a
check that was merely noisy.

**Verification:** `python -m src.split` exits **0** with **21/21 checks on each variant** (15 split + 3
per written file), each split inheriting T14's **15/15** conversion checks so no half is cut from a
frame that failed its own guard rail; **28/28** independent assertions via plain `read_csv`; four files
byte-identical across runs; **12/12 injected faults reported**; and `git status` shows no stray files
(the two throwaway scripts were deleted, and the now-redundant `data/processed/.gitkeep` went with
them, as `src/`, `docs/` and `notebooks/` did earlier).

**Blocked:** unchanged — commit `8278fae` (bank-additional dataset) is still unpushed pending GitHub sign-in.
**Next:** T16 — Phase 4 EDA begins with the target distribution and class-imbalance chart, read from the
**train** half only so the test half stays unseen.

### 2026-10-27 — Session 17  (Phase 4 started)
**Completed:** **T16** — built `notebooks/03_eda.ipynb` (17 cells, 8 code, executed with stored outputs)
and `reports/figures/01_target_distribution.png` (1707×677 px). The section measures the target
distribution on the **training half only** and, more usefully, derives the evaluation baselines the
distribution *forces* on every later task.

**The measured result.** The training half is **36,168 calls: 4,231 subscribed (11.6982%) and 31,937
did not (88.3018%) — a 7.55 : 1 imbalance.** Three conclusions are then derived from those counts
rather than asserted: an always-`no` classifier scores **88.3018% accuracy** while finding nobody
(the quantified version of D04, which is why accuracy is never reported alone); the random baseline for
**PR-AUC is 0.1170**, the positive rate itself, so 0.6 is roughly a 5× lift rather than barely-above-
chance while ROC-AUC keeps its 0.50 baseline (this is the reading D15's targets depend on); and at the
base rate one subscription costs about **8.55 calls**, which is the business denominator behind
p* = 7.50% (D17) and therefore why D18 scopes the claim to converting a fixed budget better rather
than to cutting spend. A stratification sanity line checks the training half against the source
population's 11.698480% and finds **−0.000294 pp** of drift, so the EDA describes the same population
as the raw file. Finally, 5-fold stratified CV is measured here rather than assumed: every fold carries
**846–847 positives**, which is the concrete justification for T30's 5-fold design.

**Re-analysed from the task definition, not from the previous session's summary.** The notebook was
re-executed twice from scratch and its outputs are **byte-identical across both runs** (61 output lines,
sha256 `2e2a462385a7547b`, **0 error outputs**), so there is no hidden state behind the stored results.
Every figure the notebook states was then recomputed by an **independent path** — plain `read_csv` and
`sklearn` with no project code in the loop — giving **39/39 checks**: the counts, rates, ratio,
baselines, population delta, the 8.55 calls, the per-fold positives and their 4,231 total, plus the
**done-condition itself** (the figure exists as a real 1707×677 PNG, and the markdown immediately after
the figure cell is a single-sentence, single-block pull-quote). The guard that the held-out half stays
unseen was tested too: neither the test row count (9,043) nor its positive count (1,058) appears
anywhere in the notebook's outputs, and nothing loads the test split.

**One real defect was found in the re-read and fixed.** The subsection numbering **jumped from 1.3 to
1.5 with no 1.2 or 1.4**, and the cross-validation cell had no heading at all — an artefact of building
the file by insertions. The headings were renumbered to a clean `1.1` (what the distribution forces us
to report), `1.2` (where the training half came from) and `1.3` (what the imbalance means for
validation). The cell inserted to carry `1.3` was also missing nbformat's required `id` field, which
`nbformat.validate()` flagged as a future hard error; the id was added and the notebook
re-serialised through nbformat, after which `validate()` reports clean. One check of mine was wrong
rather than the notebook — I had asserted the viva insight was "one line" by counting physical source
lines, when it is one *block* of wrapped text; the check now tests the right property (a single
sentence in a single blockquote), which is what "a one-line viva insight" actually means.

**Verification:** `jupyter nbconvert --execute` exits 0 with **0 error outputs** and outputs identical
across two independent runs; **39/39 independent checks** recomputed from the data; the notebook's own
**13 assertion lines** pass inside its output; `nbformat.validate()` clean; the figure is a valid PNG at
1707×677 px (66 KB); and the T16 done-condition was tested directly rather than assumed. The two
throwaway scripts were deleted; `git status` shows no strays.

**Blocked:** unchanged — commit `8278fae` (bank-additional dataset) is still unpushed pending GitHub sign-in.
**Next:** T17 — numeric distributions and skew for every numeric column into
`reports/figures/02_numeric_distributions.png`, one interpretation line per variable, carrying in the
three measured traps (`previous` IQR = 0, negative `balance`, `duration` as the labelled leak).

### 2026-10-27 — Session 18  (repository upload)
**Completed:** the whole project is on GitHub. Everything that existed only on this machine was committed
in three logically separate commits and pushed; `origin/main` is now `a734914` and
`git ls-tree -r origin/main` matches the working tree **file for file — 31 files, no differences.**

**What had never been uploaded.** The pushed history contained the datasets and nothing else. Three
commits closed the gap: `5ad4dd5` moves the six raw files into `data/raw/` — the `git mv` had been
staged since Session 3 but never committed — and adds the folder skeleton, `.gitignore` and
`requirements.txt`; `7c2d0e9` adds the five source modules, both documents, the three executed notebooks
and the seven figures; `a734914` adds the four stratified split files written by T15 (**86,403 lines**).
This tracker lands as the fourth commit. The splits are about 9.1 MB of the roughly 10 MB uploaded and
are committed deliberately rather than ignored: they are the exact inputs T17 onwards reads, and T15
proved them byte-identical across runs, so a marker can inspect them without re-running the pipeline.
They stay reproducible from `python -m src.split` under seed 42, and one line of `.gitignore` drops them
if a smaller upload is preferred.

**The stated blocker was stale, and said so for fifteen sessions.** The push had already worked at 18:44
— `push.log` records `e17f30f..8278fae  main -> main` with `EXIT=0`, and the `origin/main` ref file is
timestamped 18:44:09 — yet Sessions 2–17 each carried "Blocked: unchanged — still unpushed pending
GitHub sign-in". Nothing was lost, because the local branch never moved; but the tracker was wrong about
its own repository for most of the project. That is the same failure mode as a check that reports a
description instead of measuring the thing, and it is why Open Questions 1 is now written from evidence.

**The obstacle that was real.** A push needs an interactive Git Credential Manager sign-in and the
agent's shell has no TTY, so `git push` hangs at the credential prompt and GCM 2.9.1 will not release the
stored `mayurkusmade` account non-interactively (`GCM_INTERACTIVE=never` makes it refuse outright rather
than supply the account). The upload completed the moment the sign-in was finished in the user's desktop
session. Worth remembering: reads such as `git ls-remote` work anonymously and prove nothing about push
access, so "the remote is reachable" is not evidence that a push will succeed.

**Verification:** `git push` exited **0** with `8278fae..a734914`; `git ls-remote origin
refs/heads/main` returns `a7349147ed5e325b4d975965a8efc77c584aacb4`, equal to local `HEAD`;
`git ls-tree -r --name-only origin/main` diffed against `git ls-files` reports **no differences**
(31 files each, and no leftover root-level `bank-additional/` copies, since all six moves were recorded
as 100% renames); and no credential material reached the repository — `.venv/`, `src/__pycache__/` and
the `*.log` scratch files remain ignored.

**Blocked:** none — the push blocker is closed (see Open Questions 1).
**Next:** T17 — numeric distributions and skew for every numeric column into
`reports/figures/02_numeric_distributions.png`, one interpretation line per variable, carrying in the
three measured traps (`previous` IQR = 0, negative `balance`, `duration` as the labelled leak).

---

## Open Questions / Blockers

1. ~~**BLOCKER — GitHub push pending.**~~ **Resolved 2026-10-27 (Session 18).** The push of `8278fae`
   had already succeeded at 18:44 — `push.log` records `e17f30f..8278fae  main -> main` with `EXIT=0`
   and `.git/refs/remotes/origin/main` is timestamped 18:44:09 — so the "still unpushed" line repeated
   in Sessions 2–17 was stale from that moment on. The remaining files were then committed and pushed
   in Session 18; `origin/main` is now `a734914` and its tree matches the working tree file for file.
   The one genuine obstacle was that a push requires an interactive Git Credential Manager sign-in,
   which the agent's shell cannot drive (see Session 18).
2. **Primary dataset confirmation.** I have assumed `bank-full.csv` is primary and
   `bank-additional-full.csv` optional (D02). Confirm, or say if the richer variant should lead.
3. **Deliverable format.** Assumed a written report **and** a slide deck are both marked
   (T40, T41). If only one is required, say so and I will merge those tasks.
4. **Demo framework.** Assumed Streamlit (T43) over FastAPI. Confirm if the brief specifies one.
5. ~~**Data reorganisation.** T02 moves the CSVs into `data/raw/`.~~ **Resolved 2026-10-27:** executed on
the user's "go"; all six files moved with `git mv` and verified byte-identical by checksum, so no
data was altered. Any older path references elsewhere should be updated to `data/raw/`.
6. **Notebook vs script balance.** Plan assumes notebooks for exploration (Phase 2–4) and
   importable modules under `src/` for anything reused. Confirm if the marker expects notebooks only.
7. **Assumed cost figures.** T09 needs a cost per call and an average deposit value. These are
   assumptions, not real bank figures — I will state them explicitly. Tell me if your brief gives real numbers.
8. **Python version ceiling.** This machine runs Python 3.11.9, which caps `xgboost` at 3.2.0 and
   `shap` at 0.51.0 (see D09). If you install Python 3.12+ and want the newer releases, tell me and I
   will re-pin — but everything in the plan works on 3.11, so this is optional.
9. **Will the marker run our code?** If the project is graded by running it, the marker's Python
   version must be ≥ 3.11. Nothing in `requirements.txt` works on 3.9 or older.
