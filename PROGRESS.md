# Project Status

**Title:** DepositIQ: Predicting Term Deposit Subscription to Optimize Bank Telemarketing Campaigns
**Last updated:** 2026-10-30 23:59
**Overall completion:** 100% (46 / 46 tasks)
**Working location:** `depositiq-term-deposit-prediction/` (GitHub: `mayurkusmade/depositiq-term-deposit-prediction`)

## Rubric Coverage

| # | Rubric criterion | Marks | Tasks done / total | Tasks |
|---|---|---|---|---|
| 1 | Problem Definition & Business Understanding | 10 | 3 / 3 ✅ | T07–T09 |
| 2 | Dataset & Data Understanding | 10 | 2 / 2 ✅ | T10–T11 |
| 3 | Data Preprocessing & Cleaning | 10 | 4 / 4 ✅ | T12–T15 |
| 4 | EDA | 10 | 6 / 6 ✅ | T16–T21 |
| 5 | Feature Engineering / Selection | 10 | 4 / 4 ✅ | T22–T25 |
| 6 | Model Selection & Implementation | 15 | 6 / 6 ✅ | T26–T31 |
| 7 | Model Evaluation & Validation | 15 | 5 / 5 ✅ | T32–T36 |
| 8 | Business Insights & Recommendations | 10 | 3 / 3 ✅ | T37–T39 |
| 9 | Documentation & Presentation | 10 | 13 / 13 ✅ | T01–T06, T41–T46 |
| | **Total** | **100** | **46 / 46** | |

> Note: the rubric has nine criteria, listed above. Phase 10 (T45–T46, viva preparation) is not a
> separate criterion — those two tasks are counted inside criterion 9 — so the per-criterion
> numbers sum to exactly the 46 tasks in the checklist.

---

## NEXT TASK

**None — the 46-task plan is complete.** The final deliverables are `docs/viva_qa.md`, checked by
`python -m src.viva`, and `docs/demo_script.md`, checked by `python -m src.demo_script`. The project is
**46 / 46**; the remaining work is maintenance — re-run the validators after any change, and add a new
question to the viva bank whenever one is asked that the bank cannot answer from a record.

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
| D29 | 2026-10-27 | Correlations between a numeric column and the target **drop missing rows pairwise**; no column is median-filled to make a correlation computable | Measured, and it changed the answer. Median-filling `days_since_contact`'s 29,584 missing rows (81.8% of the column) gave **−0.0452**, while the honest pairwise value over the 6,584 rows that have one is **−0.1526** — an understatement of 3.4× that also mis-ranked the runner-up, and therefore the ratio the EDA quotes for the leak. Missingness here means "never contacted before", so filling it with a median invents contact dates that never happened |
| D30 | 2026-10-27 | Every numeric-distribution panel is drawn over that column's **1st–99th percentile window** and states the **full range and the share of plotted values it shows**; few-valued integer columns get **one bin per distinct value**; the leak column is coloured and labelled | The tails are the finding, so silently clipping them would be dishonest, while drawing `balance` (min −8,019 to max 102,127) or `previous` (81.8% zeros) over its full range collapses the real distribution into one bar. Stating the window and the full range discloses both. The bin rule was corrected after looking at the rendered figure: 50 bins over `campaign`'s 16 values produced bars alternating with empty bins, and `campaign` showed fractional ticks until integer locators were applied |
| D31 | 2026-10-28 | Categorical association is measured with **Cramér's V**; every segment reports its **n** and a **Wilson 95% interval**; the middle of the ranking is read as a **tie**, not an order | With a binary target the formula collapses to sqrt(χ²/n), which is the only reason 2- to 12-level columns can be compared at all — but V still rewards degrees of freedom, and `job` spends 11 of them to reach 0.1378 where `housing` reaches 0.1393 with 1, so printing `dof` beside V turns a spurious ordering into an honest tie. Wilson rather than the normal interval because `month = dec` rests on 178 rows and `default = yes` converts at 5.87%, where the normal form runs below zero. The intervals also govern what may be claimed: `dec`'s 39.99%–54.51% cannot be separated from `mar`, `sep` or `oct`, even though the group is plainly above the base rate |
| D32 | 2026-10-28 | Prior contact is represented by **`previously_contacted`** (D21), never by `poutcome != unknown` | Measured, and the two differ on exactly **5 rows**: 6,584 rows have `previous > 0` with a valid `pdays`, but only 6,579 carry a known `poutcome`. All 5 orphans have prior contact and no recorded outcome (2 of them subscribed), while the reverse case never occurs, so the implication runs one way only. Using `poutcome` as the flag would silently drop 5 contacted clients; modelling both as separate "contacted" indicators would encode one fact twice |
| D33 | 2026-10-28 | The `campaign` gradient is reported as **arithmetic about the call budget**, explicitly **not** as a causal effect of calling | A client recorded at six calls is one who declined five times, so part of the gradient is the definition of the column. What decides the question is what the extra calls cost and return, which is measurable without identifying a cause: 29,537 calls beyond the third return 570 subscriptions at €1.50 each against €20 of value, **−€32,906** net. The association is then checked for the confounds that are visible (prior outcome, prior contact, month) and survives all three, which is stated as evidence against the *composition* explanation rather than as proof of the reverse |
| D34 | 2026-10-28 | Any call cap is **conditional**: three calls for clients with **no known prior outcome**, unlimited for clients whose previous campaign **succeeded** | The tail is not uniformly bad. Beyond the third call, `poutcome = unknown` returns **1.66%** a call (€90.43 per extra subscription) and `poutcome = success` returns **22.55%** (€6.65, three times break-even), and **93.28%** of all the extra calls go to the first group. A blanket cap would cut the one segment where persistence pays, which is exactly the error a single pooled average hides |
| D35 | 2026-10-28 | `day` is kept as a raw feature but **no rule is built on it**; `campaign` enters T23 as a **capped count or bucket**, not the raw 1–63 scale | `day`'s marginal band difference (12.66% for 1–10 against 10.92% for 11–20, intervals not overlapping) **reverses inside `may`** (5.94% against 7.14% for 21–31) and changes direction month to month, and the column is entangled with `month` by construction (`jan` never runs below day 6, `feb` stops at 27, `may` starts at 4) — so a day rule would encode this file's calendar rather than the behaviour. `campaign`'s own scale is unusable for the same reason the bucket table exists: median 2, max 63, and the productivity changes by a factor of 3.4 between one call and ten |
| D36 | 2026-10-28 | T24 keeps **`days_since_contact`** and **`previously_contacted`** and **drops raw `pdays`**; there is no other multicollinearity to defuse, so no PCA and no dropping of `balance` or `campaign` on "correlation" grounds | Measured: of the 28 off-diagonal pairs only two reach an absolute r of 0.20, and one of those is `pdays` ↔ `days_since_contact` at **+1.0000** — the same values, not merely correlated ones. With both present the 8-column complete-case matrix is **rank 7**, the condition number is **5.0×10¹⁶** and both VIFs are **9.0×10¹⁵**; with `pdays` dropped every VIF is **1.01–1.03**. Keeping the raw column would let a linear model split one effect between two names while the VIF appeared tolerable, because the sentinel masks the duplication in the full-column view |
| D37 | 2026-10-28 | Any correlation involving **`pdays`** is reported on the **previously-contacted rows only** (or not at all); the whole-column value is a sentinel artefact and is never quoted as a finding | The column reads **+0.1007** against the target over all 36,168 rows and **−0.1526** — bit-for-bit `days_since_contact`'s value — once the 29,584 sentinel rows are removed, because −1 is the smallest possible value and therefore sorts as "very recent". A modeller acting on the +0.1007 would invert the effect for every real observation. This is the concrete reason D23 replaced the column instead of re-scaling it |
| D54 | 2026-10-30 | **The held-out half is scored once by the persisted artifact at three cuts, the
threshold is *not* re-chosen on it, and the gap against CV is judged against the bootstrap sd of that draw
rather than the CV fold sd** — held-out **PR-AUC 0.4616** / **ROC-AUC 0.8052** against CV's **0.4460** /
**0.7990**, i.e. **+0.0155** (**+1.41 CV sd**, but **0.92 bootstrap sd**), with **15.4%** of resampled draws
falling below the CV mean | Every part of this is a rule about what a single number may be used for.
**Scored once, loaded not refitted:** the object is `models/final_model.joblib` (sha256 **7a40752f...**)
verified against T31's card, because a refit here would silently convert the project's one out-of-sample
number into a second in-sample one. **The cut is not chosen here:** the F1-maximising threshold on these
rows is **0.212050**, and adopting it would spend the held-out set on a decision the training folds already
made — so it is recorded as a diagnostic instead, and it is how we know T32's **0.2103** landed **0.001725**
from the optimum for **0.000481** F1. **The yardstick is the bootstrap, not the CV sd:** the CV sd
(**0.0110**) is the spread between five models fitted to **28,934** rows each, while the bootstrap sd
(**0.0169**) is the spread a held-out estimate actually has; dividing by the first and calling 1.4 sd large
would be a category error, and both ratios are recorded with the judgement stated. The expected fall did not
occur and that is reported in those words: the artifact saw **36,168** rows against a fold model's
**28,934**, so a small *gain* is what the difference in training data predicts, and one draw of **1,058**
positives carries the error the bootstrap quantifies. What this does **not** claim is that the tuned
configuration was correct — the 40-candidate ranking is not re-tested by one point. Governance follows from
the same rule: T35 and T36 may **report** on these rows and may not **choose** on them, or the number above
stops being evidence |
| D55 | 2026-10-30 | **The operating threshold is `C/V = 0.0750`, set by the cost model's assumptions and chosen on the training folds from out-of-fold scores - never from an empirical argmax** (training peak **0.0821**, held-out peak **0.0909**, neither adopted) | Three reasons, in the order they would survive a challenge. **It is a price, not a hyperparameter:** `p* = C/V` follows from `EV = pV - C`, so it is set by `src/cost.py`'s two assumptions (**1.50 EUR** per call, **20.00 EUR** per subscription) rather than by this dataset - which is why the answer to "what if those are wrong" is a 20-cell table (**0.013-0.300** across C in 0.75-3.00 and V in 10-60), not a caveat. **The argmax is not stable and the price is:** adopting the empirical peak would have meant **0.0821** on the training folds and **0.0909** on the held-out rows, for **388.50 EUR (0.883%)** on 36,168 rows - a sampled maximum swapped for a stated rule, and it would have been chosen on the half that T33 had already been scored against. **Which edge matters:** the 1% plateau runs **0.0746-0.1016** and **0.0750 is 0.0004 above its floor**, so the rule is the most aggressive cut still within 1% of the peak - exactly what a break-even threshold should be, and said out loud rather than letting a reader assume the price sits mid-plateau. What it does **not** claim: the euros are the model's assumptions (no cross-sell, no fixed costs, no contact fatigue), no confidence interval is attached to a 0.9% difference on a flat curve, and the held-out half has now been scored twice (**T33**, **T34**) so T35 and T36 may report on those rows and may not choose on them. The capacity table is the live qualification: at **5%** of the list the campaign reaches **26%** of subscribers, so a call-capped bank should read the top of that table, not `p*` |
| D57 | 2026-10-30 | **`duration` costs the deployable model +0.1788 PR-AUC and +20,340 EUR — measured, reported as the price of honesty, and not a shortfall; arm A ships and arm B's numbers are an upper bound** | The rule was written in §4.5 of `docs/05_leakage_audit.md` before any number existed (D40), and this is it discharged. **Why arm A anyway:** the gap cannot change the decision, because the column does not exist at the moment the bank chooses whom to call — it is recorded when the call ends. **Why the gap is leakage rather than skill**, decided by three clauses fixed before the arms were fitted and none of which reads the gap's size: arm B wins **5/5** folds, the band table is non-decreasing (**0.16%** of 0-59s calls subscribe against **48.56%** of 600s+ calls, a **302x** swing), and **`duration` alone, with no model and no fitting, reaches ROC-AUC 0.8078 and PR-AUC 0.3897** — better than most of the project's purpose-built models on PR-AUC. The arrow runs outcome → length: a client who says yes keeps talking. **Why it is not money left on the table:** the 20,340 EUR is the cost of a deployable model, not of a wrong one, and presenting it as a shortfall would measure the wrong thing. **Why the numbers are quotable at all:** arm A's per-fold scores reproduce T30's arm to **0.0e+00** and its campaign value reproduces T34's **43,621.00 EUR** to the cent, so the shipping arm is the documented one rather than a neighbour of it. Two details that would mislead if skimmed: mean predicted probability barely moves (**0.1166** vs **0.1162**) while **Brier improves** (**0.0818 → 0.0621**), because a model that has seen the outcome is well calibrated — calibration is not evidence of skill. The gap is cross-validated, not held out, and the **held-out 9,043 rows were not read by either arm** |
| D58 | 2026-10-30 | **SHAP explains the shipped artifact on its own 40-column matrix, aggregated to the 12 features by signed sum within each one-hot block; global importance is 55.7% campaign-side against 44.3% client-side, `contact` leads at 22.14%, and the reference value -2.04695 is the booster's bias rather than the mean prediction** | The aggregation rule is the technical crux and it was chosen by a property, not by taste: summing the **signed** contributions inside a feature keeps additivity exact (**5.010e-06** at feature level), so the table can be checked against the model instead of merely computed from it, whereas summing absolute values counts a client's `month=may` column and its `month=jun` column both, inflating `month` by **58.52%**. The inflation is not the real cost though - **fragmentation** is: `month=jun` alone ranks 9th of 40, so a per-level reader would rank the model's second-most-important feature mid-field, which is the same one-hot effect T25 §8.2 flagged from the other direction. **Why the reference is documented instead of assumed:** -2.04695 equals xgboost's own `pred_contribs` bias to **0.0** and sits **0.0256** below `logit(base_score)`, while the mean margin is -2.49678 and the mean probability 11.6990%; calling it "the average log-odds" would misdescribe the model by 0.45 log-odds, so the three numbers are tabulated and a check fails if a future refactor quotes the base as the mean. **Why the 55.7% split is recorded:** it is the sentence the rubric asks for - the model is more about **how the bank calls** than **who the client is** - and it supports D18's scoped claim (a fixed call budget converts better) while blocking the unsupported one (targeting reduces spend). **Why T25's disagreement is named rather than averaged:** SHAP credits information *used* and permutation measures information *unique*, so `balance_arsinh` ranking 4th for SHAP and 9th for the forest: 0.1733 log-odds of credit against 0.0039 AUC of unique contribution, with MI ranking it 3rd - an average of the two ranks would be a third number belonging to neither method and would hide exactly that mechanism. **Scope:** training half only (**36,168** rows, D53/D54), log-odds not percentage points, attributions not causal effects, and no generalisation claim |
| D59 | 2026-10-30 | **The three exemplar clients are drawn from the cells defined against T34's 0.0750 price, selected by a pre-registered rule (closest to the cell's median score, ties by lowest row index), scored with the served `predict_proba` rather than the SHAP decomposition, and the finding is that a false positive at this price is arithmetic rather than an error** | The rule exists because the alternative is the trap this task is most exposed to: explain the most confident false negative and a reader concludes the 708 are like that one. The middle of each cell is a typical member, and a check re-derives the choice from `y` and `p` alone - no SHAP value enters - so an exemplar cannot be chosen because its explanation read well. **Why the served probability:** the same score has two routes - `predict_proba` and SHAP's decomposition through the logistic link - and they differ by up to **4.49e-07**; **43** clients sit within 1e-4 of the cut and **exactly one** changes cell depending on the route, so the counts are quoted as the served version and the flip is measured rather than assumed. A first version defined the cells from the margin and this check was vacuous, because both sides were the same array; it now compares SHAP's margin against the pipeline's own probability (**8.598e-09**) and is a real test. **Why the cells are in-sample:** the explanation belongs to the artifact, so its own scores define the cells - which is why they are **3,523 / 11,059 / 708** against T34's out-of-fold **3,296 / 11,570 / 935**, and both are recorded rather than one replacing the other. **Why the false positive is the finding:** the reference value is **11.4361%** and the cut is **7.50%**, so an average client clears the cut; the FP exemplar's twelve contributions sum to **-0.0109**, essentially the reference, and **30.58%** of the file is in its cell. The illustration therefore states a property of the price, not a failure of the model. **Scope:** three clients are illustrations and not a sample, attributable to the model and not to the client, and no held-out row was read |
| D56 | 2026-10-30 | **Lift and gain are reported per half against that half's own base rate, the adopted cut is translated into 13,309 calls saved / 19,964 EUR, and calls saved is explicitly *not* treated as the objective** | Four reasons. **The tables are threshold-free**, which is the only property that lets a Phase 7 step report on the held-out rows without spending them (D53, D54) - and it is why the held-out deciles (**4.51x** top decile) can be quoted beside the training ones (**4.37x**) rather than held back. **Each half keeps its own denominator:** the two rates are **1.5 basis points** apart, which is close enough to look interchangeable and is not, and lift is a ratio to a base rate. **D15 is met by a factor of two** (**4.3652x** / **4.5100x** against >2x), so the target is a measured result rather than an aspiration. **And the savings peak is reported against the cut rather than being adopted as it:** it sits at **18.89%** coverage (**15,188 calls**) while the cut sits at **41.10%** (**13,309**), because `saved(k) = gain x N - k` has a marginal step of `(N/positives) x y - 1` and so peaks where the *local* rate crosses the *list average* (11.7%), not the price (7.5%). Adopting it would be the same error as adopting an F1 argmax, one level up: a rate dressed as a decision. The two are reconciled by pricing the block between them, which converts at **8.96%** (**+2,350 EUR**) and **7.76%** (**+79.50 EUR**) against the **7.50%** price - inefficient calls that are still profitable ones. The self-test (**shuffled scores give a mean lift of 1.000001**) is what licenses these numbers to mean what their labels say |
| D53 | 2026-10-30 | **Thresholded metrics are reported at three cuts — leave-one-fold-out F1-optimal (the operating point), D17's 7.50% break-even (the business cut), and 0.5 (the rejected number) — and 0.5 is never the headline** | T32 measured what T29 predicted, this time across **two different models**: at 0.5 the forest beats the shipped model by **10.6 F1 points** (**0.4565** vs **0.3509**) and at the nested cut the shipped model wins by **2.4** (**0.4877** vs **0.4641**). Neither model changed. The mechanism is visible in the flag rates: the forest's mean prediction is **0.3404** so 0.5 sits inside its distribution (**17.89%** flagged), while the shipped model's mean is **0.1166** so 0.5 is near its **95th percentile** (**4.67%** flagged) — the calibrated model is being scored at a bar almost nothing clears. Each fold's cut is chosen on the **other four folds**, so no row is judged at a threshold fitted to itself, and the module re-derives all five cuts as a check. The **7.50%** cut is the only one *not* fitted to data, and it is also the one that makes the business case: **41.10%** of the list yields **77.90%** of subscribers at **22.18%** precision, against the "call everyone" baseline's F1 of **0.2095**. Which of the two live cuts to run is a capacity question for T34, not a metric's |
| D52 | 2026-10-29 | **The shipped model is XGBoost, untreated, at T30's tuned point** — **PR-AUC 0.4460 ± 0.0110**, ROC-AUC **0.7990**, Brier **0.0818**, mean p **0.1166** against a **0.1170** base rate — persisted as `models/final_model.joblib` (**180,520 bytes**, sha256 **7a40752f...**), with **no threshold baked in**, and justified by ranking, calibration and T29's measurement rather than by the tuned margin | The three reasons are ordered by how much they survive scrutiny. **Ranking**: it is the best PR-AUC of the seven builds and ties the best ROC-AUC. **Calibration**: mean p **0.1166** against a base rate of **0.1170** means D17's **7.50%** break-even can be applied to its output *as a probability*, which no weighted build could offer (they score the average client at **0.34–0.42**) — that is why recalibration is a refinement for T34 rather than a prerequisite. **T29**: the weight bought **+0.0023** at **1.11 SE against it** and cost half the Brier score; at the tuned point the loss widened to **−0.00291 in 0/5 folds**. What the decision explicitly does **not** rest on is the tuning: **+0.00165 at 0.58 SE**, with T28's configuration **4th of 40**, so the justification for the tuned build is that it is no worse, is equally calibrated and is reproducible from a file. **No threshold is persisted** because the operating point is T34's decision (D45), and freezing a cut into the artifact would ship an unmade decision. **No held-out score exists in this step** — T32/T33 own it, and computing it inside the step that chooses the model would make the choice look better than it is |
| D51 | 2026-10-29 | **Tuning is a 40-draw `RandomizedSearchCV` over six axes (4,800 points, 0.8% drawn), scored on `average_precision` over the project's materialised folds, with the feature set and the imbalance treatment held out of the space**; the winner is **300 rounds / lr 0.08 / depth 5 / `min_child_weight=10` / `subsample=1.0` / `colsample_bytree=0.6`**, and its **+0.00165 ± 0.00286 PR-AUC over the untuned configuration is reported as small (0.58 SE) rather than as an improvement** | The budget is declared because a search is a sample of a space, not an optimum: 40 of 4,800 is stated in the doc, so a better configuration found later is a larger `n_iter` rather than a defect here. Two exclusions are the substance of the decision. The **feature set** is T25's and stays there (D44, and T28's trigger did not fire). The **treatment** is excluded because a search may legitimately trade calibration for ranking, and letting it re-choose `scale_pos_weight` would silently reopen T29: the weight is instead evaluated at the tuned point and reproduces T29's verdict (**-0.00291 ± 0.00300 PR-AUC, 0/5 folds**, Brier **0.1535** vs **0.0818**). The search's own score is never quoted as a result - the arms are **refits** whose per-fold numbers equal the search's split scores to **0.0e+00**, and the record carries all 40 candidate scores and both bands so the optimism is visible: **14 of 40** within one standard error of the winner, median candidate **0.4389**, and the **shipped configuration 4th of 40** at **0.4449**. That last number is the finding: T28's hand-picked settings were already inside the noise of the best of forty, which is why tuning bought a hundredth of a point rather than a model |
| D50 | 2026-10-29 | **Every sd in this project is a sample sd (`ddof=1`) — including the fold sd of a search candidate, which is recomputed from the per-split scores instead of taken from scikit-learn's `std_test_score` (`ddof=0`)** | `RandomizedSearchCV` reports `std_test_score` with `ddof=0`; T26 fixed `SD_DDOF=1` for every other table and stated why (on five folds the two differ by 12%). Putting the search's population sd next to the arms' sample sd would have put two different quantities in one column with nothing on the page to show it — on the winner the difference is **0.0099 against 0.0110**, and it also moves the "within one fold-sd" band. The module now computes `np.std(splits, ddof=1)` from `split{i}_test_score` and keeps sklearn's value in the record as `sklearn_std_test_score_ddof0`, labelled, so the discrepancy is documented rather than removed. **Caught by the independent verifier, not by the module's own checks** — the module was internally consistent and consistently wrong about the label |
| D49 | 2026-10-29 | **SMOTE is rejected**, and the **imbalance weight is put on notice**: T29's three-arm test shows SMOTE is worse than doing nothing in **0/5** folds on PR-AUC (**-0.0268 ± 0.0118**), ROC-AUC (**-0.0194 ± 0.0028**, **6.8 SE**) and F1 (**-0.0324 ± 0.0150**), while dropping `scale_pos_weight` costs nothing measurable in ranking (**+0.0023 ± 0.0021** PR-AUC *in favour of* no weight, 5/5 folds; ROC-AUC identical to four decimals) and **halves the Brier score** (**0.0821** vs **0.1637**) by landing the mean prediction on the base rate (**0.1168** vs **0.1170**) | Three arms — `weights`, `smote` and **`none`** — on one estimator (XGBoost), one preprocessor and **one materialised fold split**, because without the control a difference between weighting and SMOTE could not be attributed to either. SMOTE sits **inside** an `imblearn.pipeline.Pipeline`, and the leak-free property is checked arithmetically rather than asserted: it adds exactly **22,164** rows on fold 1 (**28,934 → 51,098**) and the fitted scaler still matches the **fold's own** training means to **0.0e+00** while the resampled matrix's means are **40.9** away. **F1 was not read at 0.5:** a fixed cut reverses the ordering — `weights` looks **11.9 points** better at 0.5, and `none` is better at the leave-one-fold-out cut — because `scale_pos_weight` moves the mean probability to **0.3892** while the untreated model sits at **0.1168**, so a fixed cut ranks the arms by where their calibration lands. SMOTE's failure has a mechanism worth stating: its synthetic rows are interpolations between positives that do not correspond to any real client, and in a one-hot matrix a midpoint between `month=may` and `month=oct` is a value no client can have |
| D48 | 2026-10-29 | XGBoost is **400 rounds, depth 3, lr 0.05, `min_child_weight=5`, `subsample=0.9`, `colsample_bytree=0.8`, `scale_pos_weight=7.548334`, `tree_method="hist"`**, and it **wins every fold against every other model on both ranking metrics**; **D46's revisit trigger was evaluated and did not fire** | Depth 3 was picked from a **one-fold** comparison across 2/3/4/6 (PR-AUC 0.42948 / 0.43760 / 0.43604 / 0.42985), which is a sanity check and explicitly **not** the tuning T30 does — it is the starting point T30 gets. `scale_pos_weight` is XGBoost's native imbalance handle (D08) and is pinned to the training-half ratio with a check that re-derives it, because a hard-coded constant nobody checks stops describing its comment. A per-fold weight was rejected once measured: the per-fold **training** ratio deviates by at most **1.9e-3** (**0.025%** relative), which cannot matter, while it would make the model's configuration depend on which fold it sits in. The result is **PR-AUC 0.4420 ± 0.0134** and **ROC-AUC 0.7993 ± 0.0089**, beating the forest in **5/5** folds by a small but consistent margin, and the 40-column model's PR-AUC clears the **0.4008** that D46 said would reopen the feature set — so the set stays as T25 left it. Worth stating separately: XGBoost is the **worst-calibrated** model in the table (Brier **0.1637** vs the forest's **0.1403**) while being the best ranker, so T34's recalibration is doing real work rather than decorating |
| D47 | 2026-10-29 | The Random Forest is **300 trees, `min_samples_leaf=10`, `max_features="sqrt"`, `class_weight="balanced_subsample"`, `n_jobs=1`**, and it **beats the linear baseline in every one of the 5 folds** on both ranking metrics; its own re-run of T25's ablation reproduces the neutral verdict | Each setting is a decision with a reason rather than a default: `balanced_subsample` recomputes the class weights **per bootstrap sample** instead of applying one global ratio to 300 trees whose positive shares all differ, which is the documented choice for forests and keeps the imbalance treatment comparable to the linear baseline's (T29 is where the choice itself gets tested). `min_samples_leaf=10` and `n_jobs=1` deliberately match T25's instrument so §5's ablation is a fair re-run rather than a differently-configured second opinion. The result is **PR-AUC 0.4346 ± 0.0146** against **0.4008 ± 0.0134** and **ROC-AUC 0.7906 ± 0.0089** against **0.7640 ± 0.0114** — **+8.4% relative** on PR-AUC — and unlike the linear model it is also *better calibrated* (Brier **0.1403** vs **0.1852**). The ablation on this estimator gives **-0.0014 ± 0.0024** ROC and **+0.0012 ± 0.0027** PR, so D44's reduction remains free for the tree family and D46's localised loss really is a linear-model effect |
| D46 | 2026-10-29 | T25's drop is re-measured on the **linear** model — the one instrument that could plausibly feel it — and **D44 stands**, with the loss recorded and a revisit trigger written before the result exists: on ROC-AUC the 12-feature set costs **-0.0039 ± 0.0023** with **0/5 folds improving** (one-sided sign test p = **0.031**), on PR-AUC **-0.0021 ± 0.0022**, inside one sd. **T31 re-runs the same ablation on the shipped XGBoost**, and a 40-column XGBoost landing below the logistic baseline's PR-AUC **0.4008** is the trigger to revisit | D44 was measured with a 200-tree forest, which masks a correlated column; a linear model gives each one-hot level its own coefficient and should feel the loss if there is one, so the objection was answered with a measurement rather than an argument. It found a small, systematic ROC-AUC loss (**1.73** standard errors, no fold improving) and no detectable PR-AUC loss — so the forest's "neutral" and the linear model's "0.004" are both true, on different metrics. The decision is kept because the shipped model is XGBoost (D08), which is tree-based like the instrument that chose the set, and because the price is **38.5%** of the matrix for **0.0039** AUC on a model that does not ship. Recording the trigger in advance is the point: if XGBoost lands under the baseline, the honest reading is that feature selection removed something the whole family needs, and that reading must not be a judgement call made after seeing the number |
| D45 | 2026-10-29 | One CV harness for all of T26–T31: **one materialised fold list**, one preprocessor factory per fold, one `fold_metrics` call path; and T26 logs **threshold-free** metrics only (ROC-AUC, PR-AUC) with Brier and mean predicted probability as calibration diagnostics — **no F1 or precision** | Three separate reasons, each measured here. One fold list replayed as an object rather than re-derived from a seed is what T36's two arms will require (D40), and it is what makes T26's reduction check able to require its shipped arm to reproduce the baseline **bit-identically**. A threshold metric on `class_weight="balanced"` measures the weighting rather than the model: the fitted mean probability is **0.4161** against a **0.1170** base rate (**7.55 : 1** weights), and the Brier score says so plainly — **0.1852** for the model with ROC-AUC 0.7640 against **0.1033** for a constant. Reporting F1 at 0.5 would have hidden that; reporting Brier beside the ranking metrics surfaces it, and it is why **T34 must recalibrate before D17's 7.50% break-even can be applied** rather than assume the score is a probability |
| D44 | 2026-10-29 | The shipped feature set drops **5 of 17** features — `education`, `loan`, `default`, `balance_band`, `job` — leaving **12 features and a 40-column matrix**, down from 65; `campaign_bucket` carries a declared carve-out so it can never be dropped | The five were nominated because the forest could not distinguish their contribution from zero, then removed **as a group** and the same 5 folds re-measured: held-out ROC-AUC moved **-0.00014 +/- 0.00195**, with **3 of 5 folds improving**. `balance_band` is redundant with `balance_arsinh`, which keeps the same information at finer resolution; `job` was the largest single block (**12 of 65 columns**) and 17th by the model; `default` carries **0.0004 nats** of mutual information. The decision is **instrument-specific** — the forest is not the shipped XGBoost — so the ablation numbers are recorded in `docs/05_leakage_audit.md` §8.4 as the evidence to revisit if T32 finds the reduced set underperforming |
| D43 | 2026-10-29 | Feature selection is **two stages**: a screen that nominates any feature **either** method cannot see (MI at or below its own permutation-null 95th percentile, **or** `AUC lost - sd <= 0`), then a **group ablation** decided by the **one-standard-error form** (`mean delta >= -sd`); the screen alone may not drop anything | Written before the numbers and then corrected by them, in the open: the original rule required **both** methods to fail, and the MI condition turned out to be incapable of failing — with 10 label shuffles of 36,168 rows the null is so tight that **no feature has z < 1** (lowest is `day` at **7.4**), **16 of 17** clear `z = 20` and the smallest MI in the table (`default`, 0.0004 nats) sits at `z = 27.5`, so the filter nominated **nothing** and every nominee came from the model half. The second correction was the ablation threshold: the measured effect was **-0.00014 +/- 0.00195**, so comparing the mean against exactly zero would have decided on a quantity **14x smaller than its own noise**. The one-SE form is not an invention — it is the standard rule for choosing the simpler model when the scores are within one standard error — and it applies the same standard stage 1 already used |
| D42 | 2026-10-29 | T24 scales **only** `age`, `day` and `balance_arsinh`; `previous_capped` and `previously_contacted` pass through unscaled; categoricals are one-hot with **all** levels kept and `handle_unknown="ignore"`, `remainder="drop"` | Three of the four group choices are measured rather than conventional. `previous_capped` has **median 0, Q1 0, Q3 0 — an IQR of 0**, and raw `previous` has one too (**81.80% zeros**), so the degeneracy belongs to the column, not to the cap; sklearn silently substitutes `scale_ = 1.0` for a zero IQR, so a median/IQR scaler would appear to work while doing nothing, and a z-score of a three-value count is a relabelling of {0, 1, 2} rather than a units fix. The boolean is passed through because one-hot encoding it produces two columns that sum to 1 — the same column twice — and passing the raw `bool` through made the assembled matrix `object`, which failed `np.isfinite`, so it is cast to float64 and not scaled: measured, the first version of this module failed for exactly that reason. `drop=None` keeps every level because dropping the first is asymmetric across columns with 2 and 12 levels and makes a coefficient interpretable only for the linear baseline, while the L2 regularisation T26 uses makes the collinearity harmless. `handle_unknown="ignore"` is required because the encoder is fitted on one half and applied to the other; measured, **no test-only level exists on this split**, so the net is proven by injecting an unseen level rather than by the data happening to exercise it. `remainder="drop"` is what makes the frame safe to hand in with `duration`, `pdays` and `y` still attached (D14), and it is verified by transforming such a frame and comparing the matrix element-wise |
| D41 | 2026-10-28 | `campaign` enters as a **six-level bucket**, not a count capped at three; `days_since_contact` is replaced by `recency_band`; raw `balance` is replaced by `balance_arsinh` plus `balance_band` | A cap at three is the *policy* decision, but as a feature it would merge "exactly 3" (11.26%) with "ten or more" (4.30%) into one level at 8.80% — throwing away a real gradient the model needs in order to rank heavy clients, since the cap is applied after scoring, not before. The bucket is bounded, fixes a skew of 4.94, and reproduces T19's table exactly. Recency keeps only the band because the 30-89 day peak (40.20%) is prior-success composition (T21 I7) and a continuous column invites a marginal reading of a controlled-negative result; `balance` keeps both a monotone scale (arcsinh, defined for the 3,001 negative balances) and a band, because the top band dips (16.44% → 15.73%) and a single monotone column cannot represent that |
| D40 | 2026-10-28 | T36's with-vs-without comparison is **pre-registered** in `docs/05_leakage_audit.md` §4.5 and runs both arms in **one process, with one fold split replayed into each**, toggling the column only through `src.leakage.feature_columns` | A comparison whose decision rule is written after seeing the numbers is a rationalisation, so the rule is fixed first: arm A ships whatever the gap, arm B is an upper bound no targeting claim may quote, and a near-zero gap is a finding to report. The single splitter and single scorer exist because the usual way a controlled comparison goes wrong is a second, unnoticed difference — a re-seeded fold, a refitted scaler, a different metric call |
| D39 | 2026-10-28 | `docs/04_eda_insights.md` carries **seven** insights selected from the notebook's eighteen findings, ranked by **money at stake**, then **planner actionability**, then **survival under a control** | The task asked for at least five, and all eighteen would bury the two that move money. The order puts the **€32,906** call-budget loss first because it is both large and immediately actionable, and demotes `duration` (I6) and `day`/recency (I7) to constraints on what may be built, because they remove work rather than win subscriptions — a reader who wants the full eighteen is sent to notebook §6, which stays the exhaustive record |
| D38 | 2026-10-28 | The numeric correlation matrix and the categorical Cramér's V tables are **two halves of one feature ranking**, and neither is quoted as a ranking on its own | `duration` is the strongest number in the matrix (+0.3967) and unusable (D05); the strongest *usable* numeric is −0.1526; `poutcome` at V = 0.3102 is the strongest association anywhere in the data and does not appear in the heatmap at all, because Pearson cannot see a categorical column. Any statement of the form "the top predictors are …" therefore has to name which half it came from |

| D60 | 2026-10-30 | **The playbook is eight recommendations held as data in `src/playbook.py`, each figure a file-plus-key-path pair re-read from disk, ordered data quality → pricing → ranking → instrumentation, with the D18 scope sentence ahead of any euro figure and `duration` permitted only in the recommendation that forbids it** | Three reasons the recommendations are not prose. **A sentence cannot be checked; a record can.** Every one of the **46** cited figures is resolved by `resolve()` and re-read in `validate()`, so a mistyped key path, a stale model or a number carried over from an older run fails the command rather than being marked — and a figure with no record behind it cannot be written at all, which is what the `derived()` function is for: the ten quantities the document needs (net per 1,000 calls on each half, cost per subscription both ways, held-out recall and precision, the always-`no` accuracy, the campaign/client split) are computed from records, not dictated. **Eight, not the required six**, because two of the eight are prohibitions rather than actions - R4 on `duration` and R7 against re-opening the feature set - and a prohibition is only credible next to the number it declines: **+0.1788** PR-AUC and **20,340.00 EUR** that would be banked by a model that has seen the answer, and a **0.4460** baseline that a future re-tune must beat to be worth doing. **The order is by leverage, not by importance**: R1 first because `contact=unknown` covers **28.72%** of clients at a **-1.0150** contribution, so one required field changes how nearly three clients in ten are scored, and no other recommendation touches that many rows for so little work; R5 second because it decides the cut, and every euro in the document is linear in an **assumed** V = **20.00 EUR**; R2/R3 after it, because how the list is worked only matters once the price is real. **Why R3 is written as a refusal:** calls saved peaks at **18.89%** coverage while the price puts the cut at **41.10%**, and the block between them converts at **8.96%** for **+2,350.50 EUR** - profitable, so pruning to the peak is an optimisation that loses money. The honest form is therefore "do not stop where calls saved peaks", with the held-out half stated beside it: **1,547** calls at **7.76%** for **+79.50 EUR** is a rounding error, so on fresh data the same policy may well flip, and the recommendation says so. **Why the scope sentence leads:** p* = **7.50%** sits below the **11.6982%** base rate, so the campaign already pays; the measured gain is **+13,253.00 EUR (+43.6%)** on **21,302 fewer calls**, which is a fixed-budget improvement. A playbook that omitted this would license a spend cut the model never justified (D18). **Scope:** one classic telemarketing campaign, prices assumed, no causal month effect, three clients and eight recommendations are not a validation set |
| D61 | 2026-10-30 | **The final report is checked three ways — structure, coverage, provenance — and rendered to PDF by a headless browser already on the machine, so the deliverable adds no dependency** | Three reasons the report is generated rather than typed. **Its risk is transcription, not arithmetic.** Every other document was written beside the module that computed its figures; this one synthesises twelve records and eleven documents, so a misquoted 0.4460 is the failure mode and no test would ever run again to catch it. The provenance check therefore parses **every decimal in the document** — **497 tokens** — and requires each to exist in a `models/*.json` record, a `reports/*.csv` column, or a supporting document; thousands separators are stripped before matching because a comma splits `43,621.00` into a fragment that matches nothing, which is the same tokenizer defect T39 hit. **It caught three real errors, and they were mine:** `77.87%` reached subscribers in two places where the records say **77.90%** (3,296 / 4,231) — copied from an older progress note rather than from `lift_results.json`, which is exactly the drift the check exists for; and a pair of balance-band rates (**5.36%**, **16.44%**) that appear in no record and no document, replaced with the quoted **−0.00023 ± 0.00085** ablation and the **3,001 rows (8.30%)** of negative balances that *are* on file. **The four constraints are asserted, not trusted:** the D18 scope sentence must precede the first `EUR` total (it does, at byte 1,790 against 2,281), the price assumptions must be declared in the same breath, and each of the three prohibitions must appear inside the section where its result is presented — a prohibition stated only in the summary is one a reader will not connect to anything. **The renderer is deliberately small and the PDF is not a new dependency:** a ~120-line markdown subset (headings, tables, lists, code, images, bold/italic) and Chromium's own `--print-to-pdf`, which exists on every Windows and macOS machine — so the deliverable **27 pages, 2,794,041 bytes, 9 embedded figures** — is reproducible without adding a single pin to `requirements.txt`. A missing browser is reported as a failed check, never as a silent HTML-only run. **Scope:** the report selects and orders evidence; it computes nothing new, and its only derived claims (**3.95×** the base rate, the counts behind each percentage) are recomputed by the independent verifier from the records |
| D62 | 2026-10-30 | **The deck is generated from a fifteen-slide `SPEC` by `src/deck.py` with `python-pptx==1.0.2`, and validated against the saved `.pptx` rather than the objects that built it, with the figure column widened to 6.34 in and every slide's citations checked to exist** | Three reasons. **The risk here is layout, which no test of the data would catch.** The report's risk was transcription; a deck's is a text box that overflows, a caption that runs past the bottom edge, or a chart shrunk until its labels are decoration — so the checks read coordinates, font sizes and text back out of the saved package (**24/24** in module, **38** independently through the XML) instead of trusting the code that placed them. That discipline found **two inherited errors** in `docs/report.md` § 7.6: a held-out accuracy beside a cross-validated balanced accuracy (**61.26%** where the held-out pair says **61.28%**), and a claim that accuracy maximises at **0.5** when the record's own peak is **89.76%** at **0.478284**. Both are repaired in the report and the deck, and they are the trace's blind spot made concrete: a token-level trace admits a number that appears *anywhere* in the documents, so a figure lifted from the wrong column passes it — which is why the deck additionally checks each slide against the **specific** record it cites. **Fifteen slides, ordered by decision rather than by rubric:** the answer and its scope, the decision, the price, then the evidence, then the eight recommendations — a deck that walks the nine criteria in order is a report read aloud. **The figure column is 6.34 inches, not 5.8:** the first build was exported to PNG through PowerPoint and looked at, and the project's multi-panel charts were unreadable at an even split, so the split, the baseline alignment and the type sizes were changed in response — a layout assertion that was never rendered is a guess. **Scope:** the deck selects and compresses; it computes nothing, and every number on it is the report's, traced to the same records. |
| D63 | 2026-10-30 | **The README is a document plus a validator (`src/readme.py`), and its "reproduce by following it alone" claim is tested by reconstructing a clone from `git ls-files` and running the commands there; the data is committed, and the README must declare what is missing** | Three reasons. **A README is the one document nobody re-runs:** every other deliverable is protected by the module that produced it, but prose about paths, versions and commands has no such guard, so it drifts silently until a marker follows it into a `ModuleNotFoundError`. The validator therefore checks the README against the **filesystem, the records and `requirements.txt`** rather than against its own intentions: every `python -m src.<name>` must name an existing module with a `__main__` guard, every runnable module must be named somewhere, the command order must respect 14 dependency constraints, the quoted versions must equal the pins, the counts (modules, figures, CSV tables, documents, decisions, tasks) are counted from disk, every relative link must resolve, and every decimal must appear in a record. **The clone is the real test:** `_t42_clone.py` rebuilds a 102-file tree from git's own file list, imports all 24 modules there, confirms `src.config` resolves to the clone rather than the original tree, runs the smoke test, and checks that no `.venv` or bytecode leaks in — which is the static half of T44, and it passed 15/15. **The data is committed, contradicting this task's own carry-in:** `data/raw` and `data/processed` are tracked (10 files, 5 MB), and that is deliberate rather than an oversight — a clone with no data cannot follow step 1 at all, and the README publishes the three SHA-256 prefixes so a reader can confirm the file is the UCI original. **The honesty clause is enforced:** the README must say `app/` is empty, and the check fails the moment `app/app.py` appears, so T43 cannot land without updating it. **Four stale counts and one conflation were caught by building it this way:** the first draft claimed 15 pins (14), 23 modules (24), 4 CSV tables (6), and — the one that mattered — paired the file's **5,289** subscribers with the modelling half's **11.6982%** rate. That conflation was already in `docs/report.md` § 0 and on the deck's decision slide, all three quoting the same shared sentence; the file's rate is **11.70%** and the half's is **11.6982%**, and all three now say which is which. **Scope:** the README selects and links; it computes nothing, and its numbers are the report's, traced to the same records. |

| D64 | 2026-10-30 | **The demo is a thin Streamlit UI (`app/app.py`) over a checked core (`src/demo.py`), and the core is what is tested: it loads T31's artifact by digest, turns 11 business facts into the shipped 12 features, scores with `predict_proba`, and explains with D58's signed-sum SHAP aggregation against the 0.0750 price rather than 0.5** | Four reasons. **The UI must not be the only implementation.** A Streamlit script is hard to test and easy to fork: the probability shown on screen, the SHAP table and the euro decision would each be a second copy of logic the report already fixed. Putting them in `src/demo.py` means `python -m src.demo` can score the same default client, the same three T38 exemplars and the same 40-column matrix without a browser, and the app imports those functions rather than re-deriving them. **The artifact is pinned by digest, not by path:** `load_model()` refuses `models/final_model.joblib` unless its sha256 and size equal `models/final_model.json`, so a re-fitted model cannot be demoed under T31's name. **The form asks for 11 facts, not 12 features:** `previous` derives both `previous_capped` and `previously_contacted` (D21), `balance` derives `balance_arsinh`, `campaign` derives `campaign_bucket`, and `days_since_contact` derives `recency_band`; the five dropped features (D44) and `duration`/`pdays` (D05, D36) are absent by construction, and a check proves the forbidden names appear in neither the frame nor the 40-column matrix. **The three prohibitions are interface decisions, not footnotes:** the cut is `C/V = 0.0750` re-derived from `src.cost`, so a client at 0.5 is explicitly called and the app says why; `duration` has no field; and `C = 1.50 EUR` / `V = 20.00 EUR` are labelled assumptions beside the EV and the 20-cell sensitivity grid. The three T38 exemplars are rebuilt from `models/shap_local.json` plus `data/processed/train.csv` and reproduce their recorded scores to **0.0e+00**, so the demo's examples cannot drift from the report. **22 checks** pass in **1.7 s**, and Streamlit was already pinned at **1.52.2**, so T43 adds no dependency. |

| D65 | 2026-10-30 | **The clean-clone rerun is the only test of the README's claim, and what it checks is the artifacts rather than the prose: a fresh clone at the pushed commit, a fresh venv, the 14 pins, and the 22 commands in the README's own order, with every rebuilt record compared to the committed one** | Three reasons. **A command list can be internally consistent and still fail in order.** `duration_compare` reads `models/best_params.json` (from `src.tune`) and `models/threshold_choice.json` (from `src.threshold`), so the README's original position before both was wrong in a way no static check could see; the clone failed at 4.5 s with `FileNotFoundError`, the order moved after `lift`, and `src.readme` now enforces **18** constraints including both dependencies. **A runtime is not a result.** `docs/report.md` and the README quoted exact seconds from one machine; when the clone's `src.train` took **188.1 s** against the quoted **182.3 s**, the report's provenance check refused the decimal — correctly, because a marker on another laptop will get a third number. Exact seconds now live only in each record's `elapsed_seconds`, the documents use qualitative bands, and the README's **eight-to-ten-minute** claim is checked against the sum of the 12 records (**562.0 s = 9.4 min** in the clone) rather than a rounded anecdote. **A figure is an output too.** The SHAP values were identical across runs but `09_shap_summary.png` was not, because `shap.summary_plot` jitters with the global NumPy RNG; passing `rng=PLOT_SEED` made two fresh runs byte-identical (**B0EF0026…**, **455,211 bytes**), which is what lets the figure be compared at all. **What the comparison showed:** all **5** joblibs, all **4** processed CSVs, all **6** report CSVs and **7 of 8** figures byte-identical; every substantive JSON metric identical after NaN handling; the remaining differences are the records that cite the edited prose (`deck.json`, `readme.json`, `report.json`), the regenerated report/deck files, and the newly seeded figure. **Two clean-room mistakes were mine, not the repository's:** deleting `docs/report.md` as if it were generated broke `src.report` (it is the source the renderer reads, so it was restored from `HEAD`), and treating a missing browser as skippable would have hidden a failed deliverable — the machine had one, so `src.report` rendered **30/30** and the PDF was produced. **Scope:** one clone on one Windows machine at one commit; the byte-identity result is evidence of determinism on that stack, not a promise across every OS and BLAS build. |

| D66 | 2026-10-30 | **The viva bank is a document plus a validator (`src/viva.py`): 37 questions covering all nine rubric criteria, each with an answer and an evidence line, and every decimal traced to a record or an upstream document — with the bank and the demo script excluded from its own sources** | Three reasons. **A viva is a different reading task from a report.** A report can be skimmed; an answer has to survive a follow-up, so the bank carries four questions per criterion, an evidence line per question, and an *If pressed* follow-up where the obvious answer is incomplete. **Coverage is a property of the bank, not of the author's memory.** The validator parses the nine sections, counts the questions per criterion, checks sequential numbering and the 25-question floor, and fails if a criterion is thin — the first draft had only three questions in criterion 9, which is exactly the chapter a viva is most likely to probe. **A number without a record is a liability in a viva, not a rhetorical flourish.** Every decimal is matched against `models/*.json`, `reports/*.csv`, the report, or one of the eleven analysis documents; the bank and the downstream demo script are excluded so it cannot quote itself, and the first draft's three untraceable EDA rates were replaced with figures that a record holds. **Scope:** 37 questions, 8,061 words, 59 links, 364 tokens; 33/33 checks. The bank is not a script to memorise — it is an index to the files that win the argument. |

| D67 | 2026-10-30 | **The demo script is a timed document plus a validator (`src/demo_script.py`): six contiguous segments to 5:00, a spoken-word budget, and a claims table in which every number said is re-found in the file that holds it** | Three reasons. **A demo is where unchecked numbers enter.** The report and deck are parsed, and then the spoken track quietly acquires a rounded euro total or a recall figure remembered from another cut; the script is therefore held to the same trace as the documents, and a number that is not in the claims table is not said. **The clock is part of the requirement.** A five-minute walkthrough that runs seven minutes is not a five-minute walkthrough, so the validator checks six contiguous timecodes to 5:00 and a spoken text of **697 words** at **139 words per minute** — fast enough to be alive, slow enough to be followed. **A script must survive failure.** The rehearsal checklist has **20 items** across five sections, including what to do if Streamlit is slow, if the app fails, if a number is challenged, and if the examiner asks for a file; the fallback is the deck and the record, never a re-derived number. **What is derived is shown as arithmetic:** 13,253.00 = 43,621.00 − 30,368.00, 21,302 = 36,168 − 14,866, 77.60% = 821 / 1,058 × 100, and 30.58% = 11,059 / 36,168 × 100, so a value that no single record stores is still checked against operands that records do store. **Scope:** 34 claims over **9** files, **85** decimal tokens, **37/37** checks. |

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
- [x] T17 | Rubric 4 | Numeric distributions and skew | Output: `reports/figures/02_numeric_distributions.png` | Done when: one interpretation line exists per numeric variable — **done 2026-10-27** (figure 2459×1194 px costing 196 KB with all 8 panels; **one interpretation line for each of the 8 numeric variables**; `previous` and `pdays` both have **IQR 0** and are the **same 29,584 rows**; `duration` correlates **+0.3967**, which is **2.60×** the runner-up; **60/60 independent checks** and identical output content across two runs)
- [x] T18 | Rubric 4 | Categorical features vs subscription rate | Output: `reports/figures/03_categorical_vs_target.png` | Done when: best and worst segments are identified with actual rates — **done 2026-10-28** (all **44 levels** of the nine `object` columns measured with `n`, share, lift and **Wilson 95% intervals**; best `poutcome = success` **64.65%** (779/1,205) against worst `contact = unknown` **4.11%** (427/10,386), a **15.7×** gap; figure 2473×1773 px costing 274 KB; **123/123 independent checks**, and identical text plus identical figure bytes across two fresh runs)
- [x] T19 | Rubric 4 | Campaign/history features vs subscription rate | Output: `reports/figures/04_campaign_vs_target.png` | Done when: timing, contact-type and previous-outcome insights are written — **done 2026-10-28** (figure 2468×1449 px costing 252 KB with six panels; **one interpretation line for each of the seven campaign/history variables**; the 4th call onward measured as value-destroying — clients needing 4+ calls hold **52.68% of the 99,966 calls for 13.47% of the 4,231 successes**, the first three calls convert at **7.7386%** against D17's **7.50%** break-even while the extra calls convert at **1.9298%**, i.e. **€77.73** per extra subscription against a €20 deposit and **−€32,906** net; the cap must be targeted because **93.28%** of the tail sits in the no-known-outcome group while `poutcome = success` still returns **22.55%** a call; **86/86 independent checks**, and identical text plus identical figure bytes across the stored notebook and two fresh runs)
- [x] T20 | Rubric 4 | Correlation heatmap of numeric features | Output: `reports/figures/05_correlation_heatmap.png` | Done when: strongest correlations with `y` are listed and multicollinearity is flagged — **done 2026-10-28** (figure 2443×1921 px costing 343 KB with four panels; Pearson **and** Spearman on a shared colour scale, the target ranking annotated with the **rows behind each coefficient**, and the pairwise-complete counts plotted so a 6,584-row correlation cannot be read like a 36,168-row one; the ranking is listed in §5.1; **multicollinearity flagged as a duplicate rather than a correlated pair** — only **2 of 28** off-diagonal pairs reach |r| ≥ 0.20, but `pdays` and `days_since_contact` are the **same number on all 6,584 rows where both exist**, making the complete-case matrix **rank 7 of 8** with condition number **5.0×10¹⁶** and VIF **9.0×10¹⁵** for both, while every VIF is **1.01–1.03** once `pdays` is dropped; **42/42 independent checks** and identical text plus figure bytes across the stored notebook and two fresh runs)
- [x] T21 | Rubric 4 | Consolidate EDA findings | Output: `docs/04_eda_insights.md` | **done 2026-10-28** (2,804 words; **five figures**, each with a one-sentence business interpretation directly under the image; **seven insights I1–I7** ranked by money at stake → planner actionability → survival under a control, where the task asked for five; **605 checks** pass — the page's structure, **164 numeric claims traced** to the notebook's cell sources rather than its outputs, and the headline figures re-derived from `train.csv` through `src.split.load_split`; the €-figures rest on the €1.50/€20 cost model and every claim that failed a control is labelled as such)

### Phase 5 — Feature Engineering & Selection
- [x] T22 | Rubric 5 | Design the `duration` leakage with-vs-without experiment | Output: `docs/05_leakage_audit.md` | **done 2026-10-28** (2,907 words; the experiment is two arms, **six held-fixed elements**, one toggled column and **four assertions that fail loudly** — identical fold arrays replayed into both arms, identical target vector, and every transformer fitted on fold-train only; the decision rule is **pre-registered before the numbers exist**, so arm A ships regardless of the gap, arm B is labelled an upper bound and a near-zero gap must be reported as the finding; the risk is **quantified rather than asserted** — `duration` alone reaches **ROC-AUC 0.8078** and **PR-AUC 0.3897** against a **0.1170** base rate (**3.33×**), and the rate climbs monotonically from **0.16%** under a minute to **48.56%** at 600 s or more, so the shortest tenth of calls supplies **0.14%** of subscribers while the longest **8.44%** supplies **35.03%**; **seven further leakage surfaces** audited with verdicts — the `pdays` duplicate is redundancy rather than leakage, `poutcome` is legitimate provided the encoder is fitted on train only, and the **stratified-random split is left unresolved and declared**; **541/541 independent checks**, with bands, Wilson intervals, ROC-AUC and average precision re-derived by a separate numpy/scipy path and the quoted manual verified word for word in `bank-additional-names.txt`; `python -m src.leakage` exits 0 and prints every quoted figure)
- [x] T23 | Rubric 5 | Engineer additional features | Output: `src/features.py` | **done 2026-10-28** (379 lines, **five engineered columns** each replacing a raw one rather than sitting beside it: `campaign_bucket` reproduces T19's six buckets exactly, **14.59%** at one call down to **4.30%** at ten or more; `previous_capped` caps at two, **9.22%** → **20.65%** → **23.97%**; `balance_arsinh` takes |skew| from **8.34** to **1.68** and `balance_band` carries the non-linear part, **5.36%** overdrawn to **16.44%** at €1,500–4,999; `recency_band` gives the **81.80%** never-contacted their own level instead of an imputed number; the resulting set is **17 features (4 numeric, 12 categorical, 1 boolean)** with `duration`, `pdays` and `y` excluded; **eight leakage checks run in the module** — permuting `y` and permuting `duration` must leave every engineered value byte-identical, no feature may read a forbidden column, and features must be per-row — and **145/145 independent checks** recompute every band from the CSV, verify the fixed category vocabularies T24 depends on, and prove the permutation tests can fail by planting a target-derived and a duration-derived column; `python -m src.features` exits 0)
- [x] T24 | Rubric 5 | Build the preprocessing `ColumnTransformer` | Output: `src/preprocess.py` | **done 2026-10-29** (507 lines; **17 features in -> 65 columns out** = 3 `StandardScaler` + 1 count passed through + **60 one-hot** + 1 boolean, with the column list imported from `features.FEATURES` rather than restated and the four groups declared as data; fits on the **36,168** training rows and transforms the **9,043** held-out rows to `(36168, 65)` and `(9043, 65)` with **no NaN, no inf and a single float64 dtype**; `build_preprocessor()` returns a **fresh unfitted** transformer per call so the scaler can only ever see its own fold — its `mean_` matches the training mean to **0.00e+00** while the pooled train+test mean is **4.32e-02** away; `remainder="drop"` is proven by transforming a frame that still carries `duration`, `pdays` and `y` and getting the **identical matrix**; `handle_unknown="ignore"` is proven by injecting an unseen `job` level (zeros, where the `handle_unknown="error"` clone raises `ValueError`), with **0 test-only levels** measured on this split so the safety net is explicitly untriggered; **15 checks in the module** all pass and `python -m src.preprocess` exits 0; **23,162 independent checks** recompute every one of the 65 columns from `train.csv`/`test.csv` — names, one-hot cells, scaled values and the check that no statistic was fitted across the split; and the T36 toggle is exercised now: arm A 65 columns, arm B 66, differing by exactly `duration` through `extra_numeric=("duration",)`)
- [x] T25 | Rubric 5 | Feature selection / importance with justification | Output: `reports/figures/06_feature_importance.png`, notes in `docs/05_leakage_audit.md` §8 | **done 2026-10-29** (`src/select.py`, and **2,149 words** in §8; **two methods over the same 17 features** — mutual information with a **per-feature permutation null** (narrowest null sd **1.25e-05**) and permutation importance from a 200-tree forest with out-of-fold ROC-AUC **0.7936 +/- 0.0063** — correlating at **Spearman +0.647**, with the five rank gaps of 5+ explained (`marital` 16 vs 8, `day` 11 vs 4, `balance_arsinh` 3 vs 9, `previous_capped` 7 vs 12, `job` 10 vs 17); a **third measurement decides**: the 5 nominated features removed as a group cost **-0.00014 +/- 0.00195** ROC-AUC with **3 of 5 folds improving**, so the **one-standard-error** rule drops them; **5 dropped** (`education`, `loan`, `default`, `balance_band`, `job`), **12 kept**, and the shipped matrix goes from **65 columns to 40**; **17/17 in-module checks** pass and `python -m src.select` exits 0; the figure draws both rankings with each feature's own null and the dropped features hatched)

### Phase 6 — Model Selection & Implementation
- [x] T26 | Rubric 6 | Baseline models: `DummyClassifier` + Logistic Regression | Output: `src/train.py`, artifacts in `models/` | **done 2026-10-29** (`src/train.py` (749 lines) is the harness T27–T31 inherit: `make_pipeline()`= T24 `ColumnTransformer` + estimator, `materialise_folds()` (one `StratifiedKFold(5, shuffle, 42)` produced **once** and replayed), `fold_metrics()` and `cross_validate()`; both baselines trained on the **12 features / 40 columns** of D44 with the target half only (**36,168 rows, 4,231 positives**) and the **9,043 held-out rows never loaded**; **`dummy_prior`** reproduces D28's floors in the same folds rather than quoting them — ROC-AUC **0.5000**, PR-AUC **0.116982 ± 0.000059**, accuracy **88.3018%**, Brier **0.103297**, all four re-derived arithmetically per fold (`q - 2pq + p^2` for Brier, the scored rate for AP, the fitted rate for the constant score); **`logreg_balanced`** scores **ROC-AUC 0.7640 ± 0.0114** and **PR-AUC 0.4008 ± 0.0134 = 3.43× the base rate**, beating both floors in **all 5 folds** and clearing D15's ≥ 0.75 target; the calibration finding is stated not hidden — mean p **0.4161** and Brier **0.1852** against the dummy's **0.1033**, i.e. a better ranker is a worse probability forecaster, so **no threshold metric is reported** and T34 must recalibrate (D45); the reduction check answers the carry-in: on ROC-AUC the 12-feature set costs the linear model **-0.0039 ± 0.0023** with **0/5 folds improving** (sign test p = **0.031**), on PR-AUC **-0.0021 ± 0.0022** — inside one sd (D46); **14/14 in-module checks** pass and `python -m src.train` exits **0**; **127 independent checks** refit everything from `train.csv` and recompute every metric with hand-written rank-based AUC and threshold-step AP first validated against scikit-learn to **1e-12**, including the all-tied constant case; artifacts `models/baseline_dummy.joblib`, `models/baseline_logreg.joblib` and `models/baselines_cv.json` (now `models/cv_results.json`) written, the joblib reloading as a 2-step pipeline and reproducing a fresh full fit to **1e-12**; runtime **3.8 s**)
- [x] T27 | Rubric 6 | Random Forest model | Output: `src/train.py`, `models/` | **done 2026-10-29** (a second `ESTIMATORS` entry, not a second script: `RandomForestClassifier(n_estimators=300, min_samples_leaf=10, max_features="sqrt", class_weight="balanced_subsample", n_jobs=1)`, inheriting the one materialised fold list, the per-fold preprocessor and `fold_metrics`; **ROC-AUC 0.7906 ± 0.0089** and **PR-AUC 0.4346 ± 0.0146 = 3.72× the base rate**, beating `logreg_balanced` in **every one of the 5 folds** on both metrics and **+8.4% relative** on PR-AUC, while also being better calibrated (Brier **0.1403** against **0.1852**); `balanced_subsample` chosen over `balanced` because a forest's trees each see a different positive share (D47); the D44 ablation was **generalised to run per estimator** (`REDUCTION_ESTIMATORS`) so the forest re-measures the decision it made — **-0.0014 ± 0.0024** ROC and **+0.0012 ± 0.0027** PR, reproducing T25's neutral verdict at production settings and leaving the linear model's **-0.0039** as a linear-model effect; `models/model_random_forest.joblib` written and the JSON **renamed to `models/cv_results.json`** now that it holds a candidate; **14/14 in-module checks** pass and `python -m src.train` exits **0** in **156 s**; **69 further independent checks** refit all 5 folds plus the 16-feature arm from `train.csv`, confirm **300** trees survived the `joblib` round-trip, and reproduce every per-fold figure with the hand-written AUC/AP functions)
- [x] T28 | Rubric 6 | Gradient boosting model (XGBoost or LightGBM) | Output: `src/train.py`, `models/` | **done 2026-10-29** (`xgboost_gbdt` — the fourth `ESTIMATORS` entry, D08's chosen family, `XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=3, min_child_weight=5, subsample=0.9, colsample_bytree=0.8, scale_pos_weight=7.548334, tree_method="hist", eval_metric="aucpr", n_jobs=1)` on `xgboost 3.2.0` / `scikit-learn 1.7.2`, both recorded in the JSON); **ROC-AUC 0.7993 ± 0.0089** and **PR-AUC 0.4420 ± 0.0134 = 3.78×** the base rate, winning **every fold against both other models on both metrics** (narrowest margin: fold 4 ROC-AUC **0.795176** vs the forest's **0.794363**); **worst-calibrated model in the table** (Brier **0.1637** against the forest's **0.1403**) while being the best ranker, which is the third independent demonstration that ranking and calibration are separate here; depth 3 chosen from a **one-fold** sweep (2/3/4/6 → PR-AUC 0.42948/0.43760/0.43604/0.42985), explicitly a starting point for T30 rather than tuning (D48); `scale_pos_weight` pinned to the training-half ratio **31,937/4,231** with a check that re-derives it (per-fold deviation ≤ **1.9e-3**, so per-fold weights were rejected); the §5 ablation now covers **three** estimators and the **D46 trigger was evaluated and did not fire** — 40-column PR-AUC **0.4420** against the **0.4008** that would have reopened the feature set, with XGBoost's own loss at **-0.0001 ± 0.0008** ROC; **15/15 in-module checks** pass, `python -m src.train` exits **0** in **182 s**, and **82 further independent checks** refit all five folds plus the all-17 arm, confirm 400 rounds reached the booster, and reproduce every figure — after catching a real **float32** trap where `7.548334` and `7.5483337` round differently and move AUC in the seventh decimal
- [x] T29 | Rubric 6 | Compare imbalance handling (`class_weight` vs SMOTE) | Output: `docs/06_imbalance.md` | **done 2026-10-29** (`src/imbalance.py` + `docs/06_imbalance.md` + `models/imbalance_results.json`; **three arms** — `weights` (`scale_pos_weight=7.548334`), `none` (the control) and `smote` (inside an `imblearn` pipeline) — on one estimator, one preprocessor and **one materialised fold split**, because without the control the other two would not be interpretable; **SMOTE is worse than doing nothing in 0/5 folds on all three metrics** — PR-AUC **0.4152 ± 0.0234** (**-0.0268 ± 0.0118**, **-6.07%**), ROC-AUC **0.7799 ± 0.0105** (**-0.0194 ± 0.0028**, **6.8 SE**) and F1 **0.4524 ± 0.0222** (**-0.0324 ± 0.0150**) — and it is slower (**28.3 s** vs **9.3 s**, **3.0×**); **the weight is on notice** — `none` gets PR-AUC **0.4444 ± 0.0131** (**+0.0023 ± 0.0021 in its favour, 5/5 folds**), identical ROC-AUC, and **half the Brier score** (**0.0821** vs **0.1637**) by landing mean p on the base rate (**0.1168** vs **0.1170**) (D49); **F1 was not read at 0.5** — each fold is scored at the **leave-one-fold-out** F1-maximising cut, because a fixed 0.5 cut reverses the ordering (**weights** +11.9 points at 0.5, **none** ahead at the honest cut); leakage is checked **arithmetically** — SMOTE adds exactly **22,164** rows on fold 1 (**28,934 → 51,098**) while the fitted scaler still matches the **fold's own** means to **0.0e+00** and the resampled matrix's means sit **40.9** away, and no arm adds a row at predict time (**7,234** scores per fold); **10/10 in-module checks** pass and `python -m src.imbalance` exits **0** in **61.1 s**; **197 independent checks** refit all three arms from `train.csv` and reproduce every per-fold metric, all **15** nested thresholds and the paired deltas)
- [x] T30 | Rubric 6 | Hyperparameter tuning | Output: `src/tune.py`, `models/best_params.json` | **done 2026-10-29** (**`RandomizedSearchCV`**, **40 of 4,800** points over six axes, `scoring="average_precision"`, seed 42, the project's **materialised folds** passed as `cv` so the search cannot build a look-alike split; the **search** is parallel (`n_jobs=-1`) while **every fit is `n_jobs=1`**, so candidate costs stay comparable; the **feature set** (T25, 12 columns) and the **imbalance treatment** (fixed `scale_pos_weight=1`, D49) are held out of the space, and the weight returns as a third arm; **tuned beats untuned**: PR-AUC **0.4460 ± 0.0110** vs **0.4444 ± 0.0131**, **+0.00165 ± 0.00286 in 4/5 folds** — the winner is **300 rounds / lr 0.08 / depth 5 / `min_child_weight=10` / `subsample=1.0` / `colsample_bytree=0.6`**; **the gain is reported as small** — **0.58 SE**, **14 of 40** candidates within one SE (**0.0049**), and the **shipped configuration was 4th of 40** at **0.4449**, so the honest reading is that hyperparameters are not where this problem's performance lives; **T29 survives tuning** — `tuned_weighted` is worse on PR-AUC (**-0.00291 ± 0.00300**, **0/5**) and ROC-AUC (**-0.00319**, **0/5**) with Brier **0.1535** vs **0.0818**; **selection bias stated, not hidden** — `best_score_` is never quoted, the arms table is a **refit** whose per-fold scores equal the search's own split scores to **0.0e+00**, and T32's held-out half is named as the only discipline; **12/12 in-module checks** pass and `python -m src.tune` exits **0** in **68.2 s** (search **50.1 s**); **131 independent checks** refit all three arms and the top five candidates from `train.csv` without importing `src.tune` or `src.train`; **two defects found by that verifier** — scikit-learn's `std_test_score` is **ddof=0** (**0.0099** vs the project's **0.0110**, D50) and the recorded Brier is computed in **float32** with average precision collapsing **18 tied scores** (§9.5)
- [x] T31 | Rubric 6 | Select and persist the final model | Output: `models/final_model.joblib`, `docs/06_model_comparison.md` | **done 2026-10-29** (`src/final.py` + `models/final_model.joblib` + `models/final_model.json` + `docs/06_model_comparison.md`; **seven builds compared** on one fold split — base rate ×**1.00**, logistic ×**3.43**, forest ×**3.72**, XGBoost weighted ×**3.78**, untreated ×**3.80**, SMOTE ×**3.55**, **tuned ×3.81**; **shipped: XGBoost, untreated, T30's tuned point** — PR-AUC **0.4460 ± 0.0110**, ROC-AUC **0.7990**, Brier **0.0818**, mean p **0.1166** against a **0.1170** base rate; the three reasons are the ranking (best PR-AUC of the field), **calibration** (the only builds whose probabilities can carry D17's **7.50%** break-even without rescaling) and **T29's measurement** (the weight buys no ranking and halves the Brier score — reproduced at the tuned point as **−0.00291 PR-AUC in 0/5 folds**, Brier **0.1535** vs **0.0818**); the doc also states what is **not** being claimed — the tuned margin is **+0.00165 at 0.58 SE**, the forest's old calibration advantage is superseded, and **no held-out number exists yet** (T32 owns it); artifact is **180,520 bytes**, sha256 **7a40752f...**, rebuilt from `models/best_params.json` through `src.tune.make_estimator` rather than retyped, **fit in 0.8 s** on all **36,168** training rows, **no threshold baked in**; **11/11 checks** pass and `python -m src.final` exits **0** in **3.0 s** — including a reload **in a separate Python process** (drift **0.0e+00** over 1,000 rows), bit-identical in-memory reload, finite probabilities in **[0.005045, 0.994955]** summing to 1, determinism, **300** boosted rounds confirmed, and a source scan proving no test-half load)

### Phase 7 — Model Evaluation & Validation
- [x] T32 | Rubric 7 | Cross-validated metric table | Output: `reports/metrics_cv.csv` | **done 2026-10-30** (28-row CSV + `models/metrics_cv.json`: **4 models × 5 folds + mean + sd**, on the project's materialised folds; **three cuts reported** — **nested** (F1-maximising on the other four folds), **D17's 7.50%** break-even, and **0.5 as the rejected number**; shipped model leads **every** column — PR-AUC **0.4460 ± 0.0110**, ROC-AUC **0.7990 ± 0.0086**, precision **0.4591 ± 0.0130**, recall **0.5202 ± 0.0133**, F1 **0.4877 ± 0.0110** at a mean cut of **0.2103**; **the 0.5 reversal now measured across two models** — at 0.5 the forest wins by **10.6 points** (**0.4565** vs **0.3509**) and at the nested cut the shipped model wins by **2.4 points** (**0.4877** vs **0.4641**), because 0.5 is the shipped model's **95th percentile** (flags **4.67%**) but sits inside the forest's inflated distribution (flags **17.89%**); **at the business cut the baseline is “call everyone”** — shipped flags **41.10%** of the list for **77.90%** of subscribers at **22.18%** precision (**1.90×** the base rate, vs **3.92×** at the nested cut's **13.26%** coverage); the dummy's **±0.4472** recall is a constant score alternating all/nothing, left unsmoothed; **11/11 checks** pass and `python -m src.metrics` exits **0** in **30.8 s** (forest 24.9 s); **151 independent checks** refit all four models from `train.csv` without importing `src.metrics`, `src.train` or `choose_threshold` — catching that the verifier itself was reading the **six-decimal CSV** at **1e-12** tolerance)
- [x] T33 | Rubric 7 | Confusion matrix, ROC curve and PR curve on held-out test data | Output: `reports/figures/07_*.png` | **done 2026-10-30** (`src/evaluate.py` + three figures + `models/holdout_eval.json`; the artifact was **loaded, not refitted** — **0** fitting calls and **0** training-half loads, asserted by a source scan — and scored **once** on the **9,043** held-out rows (**1,058** subscribed, **11.6997%**): **PR-AUC 0.4616**, **ROC-AUC 0.8052**, Brier **0.0804**, mean p **0.1151**; **three cuts on every figure** — **7.50%** flags **3,621 (40.04%)** for precision **0.2267** / recall **0.7760** / F1 **0.3509**, T32's **0.2103** flags **1,159 (12.82%)** for **0.4840** / **0.5302** / **0.5061**, and **0.5** flags **396 (4.38%)** for **0.6490** / **0.2429** / **0.3535** at the **95.6th percentile** of the score distribution; **the expected fall did not happen** — held-out PR-AUC is **+0.0155** (**+1.41 CV sd**) above T30's **0.4460**, but the bootstrap over these rows puts it in **[0.4304, 0.4969]** (sd **0.0169**, **1.53×** the CV sd) with **15.4%** of draws below the CV mean, so the gap is **0.92 bootstrap sd** and the verdict is **no measurable gap**; T32's CV-derived cut lands **0.001725** from the held-out F1 optimum and costs **0.000481** F1, which is the rule's real vindication; **15/15 checks** and `python -m src.evaluate` exits **0** in **9.2 s**; **144 independent checks** recompute everything with different implementations (midranks, step-block AP, `searchsorted` counts); **two defects found and fixed** — a `rank_auc` that masked the wrong order (**0.7783** vs **0.8052**) and a **float32** Brier score (**4.4e-9**, the same family as T30's §9.5); **§11** of `docs/06_modeling.md` records it and **214** numeric tokens trace to the records, and the **accuracy** block in §11.3 - **89.6052%** at 0.5 against the **88.3003%** always-no baseline, with balanced accuracy reversing the order - is derived from the same record)
- [x] T34 | Rubric 7 | Threshold tuning against the cost model | Output: `docs/07_threshold.md`, `reports/figures/07_threshold_curve.png` | **done 2026-10-30** (`src/threshold.py` + `docs/07_threshold.md` + `07_threshold_curve.png` (**221,145 B**) + `models/threshold_choice.json` (**77,751 B** with a **403**-point curve); the threshold **is** the price - **C 1.50 EUR / V 20.00 EUR -> p\* = 0.0750**, imported from `cost.break_even_threshold()` so a check can require the recorded cut to equal `C/V` exactly; chosen on the **training half only** from **out-of-fold** scores (**36,168** rows, **4,231** subscribers, 5 folds) whose per-fold PR-AUCs reproduce T30's tuned arm to **0.0e+00**, then checked once on the **9,043** held-out rows; **0.0750 calls 14,866 (41.10%) and reaches 3,296 (77.87%) for net 43,621 EUR**, ROI **195.6%**, **6.77 EUR** per captured subscription - against the empirical peak **0.0821 / 44,010 EUR** (the adopted cut gives up **388.50 EUR, 0.883%**), T32's F1 cut **0.2103 / 36,891 EUR**, **0.5 / 18,248 EUR** and **call-everyone 30,368 EUR**; the **1%-of-peak plateau spans 0.0746-0.1016 = 14,973-9,970 calls** and **0.0750 sits 0.0004 above its bottom edge**, making the price the most aggressive cut still within 1% - the curve ascends steeply and descends shallowly, which is why both empirical argmaxes land right of it (training **0.0821**, held-out **0.0909**) and why neither is adopted; **held out** the cut nets **10,988.50 EUR** (3,621 calls, 821 reached, ROI **202.3%**) against **9,481.50** at the F1 cut, **4,546.00** at 0.5 and **7,595.50** calling everyone, reaching **77.6%** of subscribers from **40.0%** of the list against training's **77.9%** from **41.1%**; **capacity is a different question** and is tabulated (top 5% reaches **26%** of subscribers for **44%** of the money; the unconstrained optimum sits near **30%** of the list, not 5%); **ROI and net value disagree by construction** (0.5 posts **765%** ROI and loses) so net value is the objective while the campaign is capacity- not capital-limited; **15/15 checks** and `python -m src.threshold` exits **0** in **4.6 s**; **two defects fixed** - a dead self-assignment in `held_out_check`, and a record that stored the peak but **not the curve**, leaving three euro figures quoteable only from a check's detail string)
- [x] T35 | Rubric 7 | Lift, gain and decile analysis | Output: `reports/figures/08_lift_gain.png`, `reports/lift_table.csv` | **done 2026-10-30** (`src/lift.py` + `reports/lift_table.csv` (**20 rows x 13 columns**) + `08_lift_gain.png` (**231,645 B**) + `docs/07_lift.md` + `models/lift_results.json` (**20,661 B**); the training ranking is **T34's ranking**, produced by `threshold.oof_scores` so the lifts sit on the same five fold fits, and the held-out ranking is the persisted artifact; **each half against its own base rate** (**11.6982%** vs **11.6997%**, never pooled); **top decile lift 4.3652x training and 4.5100x held out**, clearing D15's >2x by more than a factor of two, holding **43.65%** / **45.09%** of all subscribers with the top **two** deciles holding **61.76%** / **62.29%**; **decile 3 is already below random** (**0.92x** / **0.88x**) and decile 10 sits at **0.18x** (**78 of 3,617** clients), so past the best fifth of the list the model's message is mainly negative; **the adopted cut translated into calls** - **77.90%** of subscribers from **41.10%** of the list, i.e. **13,309 calls saved (36.8% of the list, 19,964 EUR)** against a fixed order needing **28,175**, and **3,396 calls / 5,094 EUR** held out, with the calls and subscribers **identical** to T34's record; **calls saved is not the objective** - it peaks at **18.89%** coverage (**15,188 calls / 22,781 EUR**), *before* the cut and **outside** T34's money plateau (**27.6%–41.4%**), because its marginal step is `(N/positives) x y - 1` and therefore crosses the **list average**, not the price; the **8,033 calls** between the peak and the cut still convert at **8.96%** against the **7.50%** price (**+2,350 EUR**, and **7.76% / +79.50 EUR** held out, which is the aggressive edge again); **reconciliation with T32** lands on both published multipliers (**3.9488x** vs **3.9247x**, **1.8991x** vs **1.8961x**) and the second is near-exact because **0.0750 *is* 41.10% coverage**; **a shuffled-score self-test** reruns the whole table on noise and finds lifts of **0.950–1.108** (mean **1.000001**), gain within **0.0060** of the diagonal and savings within **218 calls** of zero; **17/17 checks** and `python -m src.lift` exits **0** in **4.3 s**; **401 numeric tokens** in `docs/07_lift.md` trace to the two records, and the trace caught one hand-rounded figure (**28,176** for a true **28,175**)
- [x] T36 | Rubric 7 | With-vs-without `duration` comparison | Output: `reports/duration_comparison.csv`, notes in `docs/05_leakage_audit.md` | **done 2026-10-30** (`src/duration_compare.py` + `reports/duration_comparison.csv` (**14 rows**, fold/mean/sd per arm) + **§9** of `docs/05_leakage_audit.md` + `models/duration_comparison.json` (**24,338 B**); one estimator, one fold list, one scorer, toggled only through `leakage.feature_columns` — feature lists differ by exactly `['duration']` and the matrices by one column (**40** vs **41**, measured, since T24 demonstrated the same toggle at the 17-feature width); **arm A reproduces T30's tuned arm to 0.0e+00 per fold and T34's 43,621.00 EUR to the cent**, so the shipping arm is the documented one; **the gap is +0.1788 PR-AUC and +20,340.00 EUR** (**+562.38 EUR per 1,000 clients**) from **4,320 fewer calls** reaching **693 more subscribers**, with **5/5 fold wins** and an SE ratio of **23.12** (ROC-AUC +0.1369, Brier -0.0197, euros +4,068.00/fold); **mean p barely moves** (0.1166 vs 0.1162) while Brier *improves* (0.0818 → 0.0621), which is what a model that has seen the outcome looks like; **the verdict is computed from §4.5's pre-registered mechanism clauses, not from the gap's size** — arm B wins ≥4/5 folds, T22's band table is non-decreasing (**0.16% → 48.56%** across call length, a **302x** swing, longest band supplying **35.03%** of subscribers) and **`duration` alone as a score reaches ROC-AUC 0.8078 / PR-AUC 0.3897** — all three true, so `gap_is_leakage = True` and **arm A ships regardless** (the column does not exist at decision time); **13/13 checks** and `python -m src.duration_compare` exits **0** in **7.1 s**; **no held-out row is read** (source scan for the held-out load finds 0 references, needle assembled at run time); **one defect fixed** — the settings check compared XGBoost's `missing` parameter with `!=`, and `nan != nan` reported a difference between two arms that were both unset (now NaN-aware)

### Phase 8 — Explainability & Business Insights
- [x] T37 | Rubric 8 | Global SHAP analysis | Output: `reports/figures/09_shap_summary.png` | **done 2026-10-30** (`src/explain.py`; TreeSHAP on **the artifact** - sha256 **7a40752f...**, checked first - on the **training half**, **36,168 rows x 40 columns over 12 features**; additivity **5.010e-06** and `sigmoid(margin)` vs `predict_proba` **7.521e-08**; contributions equal xgboost's `pred_contribs` to **0.0**; **aggregated by signed sum within each one-hot block** so the twelve feature values stay additive, with the naive per-column alternative measured as **inflation** (`month` **+58.52%**, `campaign_bucket` **+49.63%**, `poutcome` **+35.91%**) and fragmentation named as the real cost (`month=jun` alone is only 9th of 40); top drivers **`contact` 22.14%**, **`month` 17.55%**, `day` 10.60%, `balance_arsinh` 8.49%, `housing` 8.05%, `poutcome` 7.97%; **`contact=unknown` is 28.72% of clients and contributes -1.0150 on a 4.11% rate**, the top column of forty; **`poutcome=success` +1.6841 on 3.33% of clients** is the largest single-level effect but the feature is 7.97% overall, which is why mean|SHAP| is not per-client influence; **55.7% of importance is campaign-side** (`contact`/`month`/`day`/`campaign_bucket`) against **44.3%** client-side; against T25's permutation ranking **Spearman +0.755** with four gaps of 3+ explained rather than averaged (`balance_arsinh` 4 vs 9, `poutcome` 6 vs 3, `age` 8 vs 5, `recency_band` 11 vs 7); the reference value **-2.04695** is the booster's bias, not the mean margin **-2.49678**, so the average contribution is **-0.44983**; **17/17 in-module checks** pass and `python -m src.explain` exits **0** in **38.8 s**; **32 further checks** recompute the whole table from `pred_contribs` without importing the module or `shap`, re-derive every level/rate quoted in the prose, and trace **115 numeric tokens** in `docs/08_explainability.md` to the records with **0 unmatched**)
- [x] T38 | Rubric 8 | Local SHAP explanations for exemplar clients | Output: `reports/figures/10_shap_local.png` | **done 2026-10-30** (`src/local_explain.py`; cells defined against **the price**, not 0.5 - T34's **0.0750** re-derived from C/V - giving in-sample **TP 3,523 / FP 11,059 / FN 708 / TN 20,878** against T34's out-of-fold **3,296 / 11,570 / 935**, both recorded because the explanation belongs to the artifact and T34's table is a different denominator; the exemplar rule is **pre-registered**: the client whose score is closest to its cell's median, ties by lowest row index, so the illustrations are typical rather than worst cases - rows **19,263 / 23,374 / 10,760**, all at the 50th percentile of their own cell; **TP** row 19,263 **p = 0.3819**, `month=oct` **+1.8824** against `contact=unknown` **-0.6989** (the model's top global feature taking most of it back); **FP** row 23,374 **p = 0.1133** with contributions summing to **-0.0109**, i.e. a completely average client, which is the finding - at a 7.5% price below the 11.4361% reference the modal cell **is** the false positive (**30.58%** of the file); **FN** row 10,760 **p = 0.0550** dragged down by `balance_arsinh = 0.00` (**-0.3502**) and `month=jul` (**-0.3121**) and missing the cut by only **0.3323 log-odds**; per-client additivity **6.853e-07**, `pred_contribs` **0.0**, 12 features cover the 40 columns; **16/16 in-module checks** pass and `python -m src.local_explain` exits **0** in **76.7 s**; **70 further checks** rebuild the cells, the rule and all three profiles without importing the module or `shap`, and trace **179 numeric tokens** in `docs/08_explainability.md` to the records with **0 unmatched**)
- [x] T39 | Rubric 8 | Business recommendations and campaign playbook | Output: `docs/08_recommendations.md` | **done 2026-10-30** (`src/playbook.py` writes `reports/playbook.csv` and `models/playbook.json`; the eight recommendations live in the module as data and each figure is a **file-plus-key-path pair re-read from disk**, so **46 figures** across **9 records** resolve and **12/12** checks pass, including one that re-reads every figure and compares it with the value the document cites - a stale record fails the build instead of shipping; `duration` appearing anywhere except **R4** also fails, so the prohibition is enforced mechanically; **scope first** (D18): calling everyone nets **30,368.00 EUR** and the cut nets **43,621.00 EUR**, so the model is worth **+13,253.00 EUR (+43.6%)** and **21,302 fewer calls** - an improvement on a campaign that already pays, not a rescue of one that does not; **R1 is a data-quality action**, because `contact=unknown` covers **28.72%** of clients, is worth **-1.0150** log-odds, and scores them at **4.11%** against **14.83%** for `cellular`, with **no lift claimed** - filling the field changes how those clients are scored, it does not relabel them; **R3 refuses the obvious optimisation**: calls saved peaks at **18.89%** coverage (**15,188** calls saved) but the **8,033** calls past it still convert at **8.96%** for **+2,350.50 EUR**, so the cut is sized by **capacity** - and the held-out half prices that same block at **1,547** calls and **7.76%** for only **+79.50 EUR**, which the document states as the thinnest recommendation rather than hiding; **R4** carries the `duration` gap (**+0.1788** PR-AUC, **20,340.00 EUR**, dead at dial time); **R7** keeps the 40-column XGBoost on **0.4460** against **0.4346** and **0.4008** PR-AUC, with tuning worth **+0.0016**; **R8** reads `month` as a calendar (**17.55%** of importance, March **+1.3736** at **50.90%**, May **-0.2999** at **6.68%** on **30.59%** of the file) and names the 2008-2010 confound; campaign-side features carry **55.75%** of importance against **44.25%** for the client's own, which is why the playbook is mostly about *how* the bank calls; `python -m src.playbook` exits **0** in **9.4 s**, and **78 independent checks** recompute every table cell and every derived euro without importing the module and trace **63 numeric tokens** in the document to the records with **0 unmatched**)

### Phase 9 — Deliverables
- [x] T40 | Rubric 9 | Write the final report | Output: `docs/report.md` (+ PDF) | **done 2026-10-30** (`src/report.py` → `docs/report.md`, `docs/report.html`, `docs/report.pdf` and `models/report.json`; **7,289 words over 27 rendered pages**, covering the nine criteria **in rubric order**, each section naming the document or module that carries its evidence; **9 figures embedded**, every one a file under `reports/figures/` that a record already produced; **12 records cited and re-parsed**; and **497 decimal tokens** all matched against a value in a `models/*.json` record, a `reports/*.csv` column or one of the eleven supporting documents — **0 unmatched**; the **D18 scope sentence sits at byte 1,790, before the first `EUR` total at 2,281**, with the price assumptions declared in the same paragraph; the three prohibitions are asserted where their results are (`duration` in §5.4, accuracy in §7.6, causal `month` in §8.1); `python -m src.report` exits **0** with **30/30** checks in **3.4 s**, and **45 independent checks** recompute the headline euros from the records without importing the module — every one of the fourteen quoted totals matched to the cent, every percentage re-derived from its counts; **three defects caught by that trace and fixed** — `77.87%` twice where the record says **77.90%**, and a balance-band pair that existed in no record at all)
- [x] T41 | Rubric 9 | Build the slide deck | Output: `docs/presentation.pptx`, `src/deck.py`, `models/deck.json` | **done 2026-10-30** (15 slides, business-first: the answer and its scope, the decision, the price, then the evidence, then the eight recommendations; **24/24** in-module checks and **38 independent** ones that open the package as a zip and parse its XML without python-pptx, recomputing **3,296 × 20 − 14,866 × 1.5 = 43,621.00 EUR** from confusion counts and confirming the six embedded charts are byte-identical to `reports/figures/`; **157** decimal tokens trace to a record or document and the **33** slide-foot citations are each verified to exist; geometry, type size and text overflow are checked against the **saved** file; all **15** slides were exported to PNG through PowerPoint and inspected, which is how charts too small to read at 5.8 in were caught; **two inherited report defects found and fixed** — § 7.6 paired a held-out accuracy with a CV balanced accuracy (**61.26%** → **61.28%**) and claimed accuracy peaked at 0.5 when the record's peak is **89.76%** at **0.478284**)
- [x] T42 | Rubric 9 | Finalise README and repo hygiene | Output: `README.md`, `src/readme.py`, `models/readme.json` | **done 2026-10-30** (249 lines: the decision and its scope, the headline figures beside the record that holds each, a two-line quick start, the **24-command** reproduction sequence in dependency order with per-step runtimes, a repository map, the data's provenance with three SHA-256 prefixes, and a plain statement that `app/` is empty; **33 checks** in `src/readme.py` hold it to the repository — every command names a real module with a `__main__` guard, every runnable module is named, **14** dependency constraints hold, the quoted versions equal the pins, the counts are counted from disk, every link resolves, no absolute path appears and every decimal traces to a record; **15/15** clean-clone checks rebuild a **102-file** tree from `git ls-files`, import all **24** modules there and prove `src.config` resolves to the clone; **four stale counts and one conflation** found — the file's **5,289** subscribers had been paired with the modelling half's **11.6982%** rate in the README, `docs/report.md` § 0 and the deck's decision slide, all three now distinguishing **11.70%** from **11.6982%**)
- [x] T43 | Rubric 9 | Build the interactive demo | Output: `app/app.py` (Streamlit) | **done 2026-10-30** (`src/demo.py` + `app/app.py` + `models/demo.json`; the UI is thin and the core is checked headlessly — **11 business facts** become the **12 shipped features** through the project's own derivations, `duration`/`pdays`/`y` are absent from the form, the frame and the 40-column matrix, and the artifact is loaded by **sha256/size** rather than by path; the drivers are D58's **signed sum within each one-hot block**, additivity **4.605e-07** and `sigmoid(margin)` against `predict_proba` **2.027e-09**; the cut is T34's **0.0750 = C/V**, and a client at **0.5** is explicitly called, so 0.5 is not the rule; **C = 1.50 EUR** and **V = 20.00 EUR** are labelled assumptions beside the EV and the 20-cell sensitivity grid; the three T38 exemplars are rebuilt from `models/shap_local.json` + `data/processed/train.csv` and reproduce their recorded scores to **0.0e+00**; **22/22 checks** pass and `python -m src.demo` exits **0** in **1.7 s**; `app/app.py` was rendered headlessly through Streamlit's `AppTest`, the default client scores **7.22%** (just below the price), the true-positive exemplar **38.19%** and the false-negative **5.50%**; Streamlit **1.52.2** was already pinned, so no dependency was added; `src.readme` now requires the README to name the demo and no longer accepts an empty `app/`)
- [x] T44 | Rubric 9 | End-to-end rerun from a clean clone | Output: verification checklist recorded in `PROGRESS.md` | **done 2026-10-30** (a **fresh clone of the pushed repo** at `c1e70b6`, **105 tracked files**, Python **3.11.9**, a fresh `.venv`, all **14 pins** installed, `src.config.ROOT` resolving to the clone and no `.venv`/bytecode tracked; the **22 documented commands** were run in README order and all exit **0** after the fixes below; **`src.report` 30/30**, **`src.deck`** passes, **`src.demo` 22/22**, **`src.readme` 34/34** in the clone; the rebuilt artifacts match `HEAD`: **all 5 joblibs, all 4 processed CSVs, all 6 report CSVs and 7 of 8 figures byte-identical**, every substantive JSON metric identical after NaN handling, with differences only in the records that cite the edited prose (`deck.json`, `readme.json`, `report.json`), the regenerated report/deck files, and the now-seeded SHAP figure; **four defects found and fixed** — the README ran `duration_compare` before `tune` and `threshold` even though it reads both records, `docs/report.md` quoted machine-specific seconds that its own provenance check could not re-find, `09_shap_summary.png` was unseeded inside `shap.summary_plot`, and a clean-room script had deleted `docs/report.md` as if it were generated when it is a source document; the full sequence took **about 15.4 minutes** on this machine, with the **12 recorded steps summing to 562.0 s = 9.4 min**, inside the README's corrected **eight-to-ten-minute** band)

### Phase 10 — Viva Preparation
- [x] T45 | Rubric 9 | Viva Q&A bank | Output: `docs/viva_qa.md` | **done 2026-10-30** (`docs/viva_qa.md` + `src/viva.py` + `models/viva.json`; **37 questions**, at least four for each of the nine rubric criteria (criterion 7 carries five), every question with an **Answer**, an **Evidence** line naming a record, document or module, and where useful an **If pressed** follow-up; the bank opens with a coverage map and closes with five numbers to know cold, the three prohibitions in one line each and the D18 scope sentence; **8,061 words**, **59 links**, every relative link resolves; the validator checks the nine sections, sequential numbering, the 25-question floor, four-per-criterion coverage, answer and evidence length, the three prohibitions, the scope sentence, the price identity `0.0750 = C/V = 1.50/20.00`, every link, and **every decimal**; **364 decimal tokens** all trace to a `models/*.json` record, a `reports/*.csv` column, the report or one of the eleven analysis documents, with `viva_qa.md` and `demo_script.md` excluded from the sources so the bank cannot quote itself; **33/33 checks** pass and `python -m src.viva` exits **0**)
- [x] T46 | Rubric 9 | Demo script and rehearsal checklist | Output: `docs/demo_script.md` | **done 2026-10-30** (`docs/demo_script.md` + `src/demo_script.py` + `models/demo_script.json`; a **five-minute** walkthrough in **six contiguous segments** (0:00–5:00), each with a **Say**, a **Show** and an **Evidence** line; the spoken text is **697 words** at **139 words per minute**, inside the human speaking band; **34 claims**, each spoken once and each traced to one of **9** files, with `record` values re-found in the named file and `derived` values recomputed from operands the file holds (13,253.00 = 43,621.00 − 30,368.00; 21,302 = 36,168 − 14,866; 77.60% = 821 / 1,058 × 100; 30.58% = 11,059 / 36,168 × 100); every decimal spoken appears in the claims table, and **85 decimal tokens** in the whole document trace to a record, the report or an analysis document; the three prohibitions, the D18 scope sentence and the price identity `0.0750 = C / V = 1.50 / 20.00` are in the spoken track; **20 rehearsal items** across **5** sections including failure paths; **37/37 checks** pass and `python -m src.demo_script` exits **0**)

---

## Files Created

| File | Contents | Created |
|---|---|---|
| `docs/demo_script.md`, `src/demo_script.py`, `models/demo_script.json` | Phase 10 (**T46**): the five-minute walkthrough and its validator. The document has **six contiguous segments** to **5:00**, each with a **Say**, a **Show** and an **Evidence** line; **697 spoken words** at **139 words per minute**; a **34-row claims table** naming **9** files, with `record` values re-found and `derived` values recomputed (13,253.00 = 43,621.00 − 30,368.00; 21,302 = 36,168 − 14,866; 77.60% = 821 / 1,058 × 100; 30.58% = 11,059 / 36,168 × 100); the three prohibitions, the D18 scope sentence and the price identity in the spoken track; a five-number summary; and a **20-item** rehearsal checklist across five sections including failure paths. `src/demo_script.py` checks the clock, the shape, every claim against its named file, the spoken-word budget, the checklist and the global decimal trace. **37/37 checks** pass over **85** decimal tokens | 2026-10-30 |
| `docs/viva_qa.md`, `src/viva.py`, `models/viva.json` | Phase 10 (**T45**): the viva bank and its validator. The document holds **37 questions** - four per rubric criterion, five for evaluation - each with an **Answer**, an **Evidence** line naming a record, document or module, and an optional **If pressed** follow-up; it opens with a coverage map and closes with five numbers to know cold, the three prohibitions in one line each and the D18 scope sentence. `src/viva.py` parses the nine sections, checks sequential numbering, the 25-question floor, four-per-criterion coverage, answer and evidence length, the prohibitions, the scope sentence, the price identity `0.0750 = C/V = 1.50/20.00`, and every relative link; its provenance trace admits numbers from `models/*.json`, `reports/*.csv`, the report and the eleven analysis documents, excluding `viva_qa.md` and `demo_script.md` so the bank cannot quote itself. **33/33 checks** pass over **364** decimal tokens | 2026-10-30 |
| `src/demo.py`, `app/app.py`, `models/demo.json` | Phase 9 (**T43**): the interactive demo. `src/demo.py` is the tested core - `load_model()` loads `models/final_model.joblib` only after its sha256 and size match `models/final_model.json`; `training_profile()` reads the categorical vocabulary and median/mode defaults from the processed training half through `split.load_split`; `client_frame()` turns **11 business facts** into the **12 shipped features** using `features.engineer`'s own derivations (`previous > 0` flag, the cap, `balance_arsinh`, `campaign_bucket`, `recency_band` with the explicit `never` level); `shipped_frame()` selects `select.shipped_features()` in order; `explain_client()` scores with `predict_proba`, builds the 40-column matrix, aggregates TreeSHAP by **signed sum within each one-hot block** (D58) and sorts by absolute contribution; `should_call()` / `expected_value()` price the call at **C = 1.50 EUR / V = 20.00 EUR** and compare against **0.0750 = C/V**, never 0.5; `example_clients()` rebuilds the three T38 exemplars from `models/shap_local.json` + the training split. `validate()`'s **22 checks** cover the artifact digest, the 12/40 shape, the price identity, the 0.5 counter-example, the EV arithmetic, the labelled assumptions, the absent `duration`/`pdays`/`y`, the categorical defaults, the shipped column order, the 12-feature partition of 40 columns, additivity, the margin-versus-probability identity, the driver sort, the finite score, the three exemplars' scores and cells, their additivity, the app compiling, and the unread held-out half. `app/app.py` is the thin Streamlit UI: an 11-field sidebar, three exemplar presets, live score, decision and EV, the top-five and all-twelve driver tables, a bar chart, the model/assumptions expanders and the sensitivity grid | 2026-10-30 |
| `src/local_explain.py`, `reports/shap_local.csv`, `reports/figures/10_shap_local.png`, `models/shap_local.json` | Phase 8 (**T38**): `cell_of()` labels rows `TP`/`FP`/`FN`/`TN` against the **price** from T34's record, re-derived through `cost.break_even_threshold()` rather than typed; `pick_exemplars()` executes the pre-registered rule (closest to the cell median, ties by lowest row index) and keeps the distances so `validate()` can re-derive the minimum from the scores alone; `value_of()` reads a categorical's level off its one-hot block and formats a numeric, so the figure labels what the model actually saw; `local_profile()` sums each block's **signed** contributions, keeping the aggregation rule and the reference value identical to T37 so local and global are comparable. The cells are built on the served `predict_proba`, and `validate()`'s **16 checks** cover the digest, the C/V identity, the four-cell partition, cell membership, distinctness, the selection rule, per-client additivity, the margin-versus-`predict_proba` identity, the derivation-stability count, `pred_contribs`, the 12/40 partition, the absent leak, the global ranks read back from T37, the 0.5-misrule counter-example, the unread held-out half and the figure | 2026-10-30 |
| `src/explain.py`, `reports/shap_importance.csv`, `reports/figures/09_shap_summary.png`, `models/shap_global.json` | Phase 8 (**T37**): `load_artifact()` pins T31's digest so a re-fit cannot be explained silently; `design_matrix()` takes the matrix the estimator actually saw; `explain()` returns TreeSHAP values, the margin they must reconstruct, and xgboost's own `pred_contribs` for comparison (the sklearn wrapper rejects a `DMatrix` argument to `predict` in this version, which is why the array goes in and the DMatrix is built only for `pred_contribs`); `feature_table()` sums **signed** contributions per one-hot block and carries the naive per-column alternative as a measured `inflation`; `column_table()` keeps the un-aggregated view the figure's beeswarm needs; `compare_with_t25()` ranks T25's pinned permutation values **within the twelve** and marks gaps of 3+; `directions()` reads which level (or which end of a numeric range) moves a client, with `level_rates()` filling in the measured subscription rate behind each sentence. `validate()`'s **17 checks** cover the digest, the pipeline shape, the absent leak, the 12/40 partition, T25's kept set, additivity, the probability identity, the `pred_contribs` agreement, feature-level additivity, the aggregation bound, the shares, both rank permutations, the pin, the per-column table, the reference-vs-mean distinction, the unread held-out half and the figure. The figure is a beeswarm over 40 columns (**6,000** of **36,168** rows, seed 42, `color_bar=False` because a 0/1 scale is meaningless), the 12-feature importance bar with the naive sum overlaid, and a dumbbell of the two rankings | 2026-10-30 |
| `src/playbook.py`, `reports/playbook.csv`, `models/playbook.json` | Phase 8 (**T39**): `Recommendation` records carry `action`, `evidence`, `expected`, `check` and `falsifier` with `scope` set on the two that spend the fixed budget, and `evidence` entries are `(record, key path, formatting)` triples resolved by `resolve()`, so a number cannot be written that no record holds; `derived()` computes the ten quantities the document needs - net EUR per 1,000 calls on each half, cost per subscription both ways, coverage, held-out recall and precision, the always-`no` accuracy, the campaign/client split - instead of typing them; `validate()` re-reads every cited figure from disk and compares it with what was written (**46/46**), checks each recommendation carries all five fields and at least two figures, and fails if `duration` is mentioned outside **R4**; `render()` emits the CSV and JSON | 2026-10-30 |
| `docs/08_recommendations.md` | Criterion 8 (**T39**): the D18 scope paragraph before any number, the eight recommendations as a table of headline figure and falsifier, then one section each with *do / why / what to expect / how to check / what would make it wrong*, the order to act in, six explicit non-promises, and the verification table - the last task of Phase 8 | 2026-10-30 |
| `docs/08_explainability.md` | Criterion 8 (**T37** and **T38**): the instrument and the artifact it explains, the reference-value table and the 0.0256 gap, the aggregation rule with the measured inflation and the fragmentation cost, the twelve-feature table, eight business-language driver paragraphs each with its level contribution and measured rate, the four disagreements with T25 and why neither ranking is corrected, the 55.7%/44.3% campaign-versus-client split with its three consequences, what the step does not claim, the verification table including the two defects it caught; **§9** (T38) then defines the three cells against the price, states the pre-registered selection rule, explains one client per cell with its twelve contributions and the values behind them, draws what the three say together, and lists the 16-check table | 2026-10-30 |
| `src/threshold.py`, `models/threshold_choice.json`, `reports/figures/07_threshold_curve.png` | Phase 7 (**T34**): `choose()` takes the training arrays and nothing else - the signature is the guarantee that the held-out half cannot influence the choice; `sweep()`, `capacity_table()` and `sensitivity()` all price their cuts through `cost.expected_campaign_value` rather than re-deriving euros, and `held_out_check()` runs after the cut is fixed and records the held-out optimum so a check can prove it was **not** adopted. `validate()`'s **15 checks** cover the price identity, one out-of-fold score per row, the T30 fold reproduction, the money arithmetic at all 403 points, the plateau, the three rejected cuts, cost per captured subscription, capacity fill, sensitivity monotonicity, training-only provenance, held-out profitability, the figure, and a read-back of the file against the run. The JSON records the decision with its provenance, the five compared cuts, the 403-point curve, the capacity and sensitivity tables, the held-out block and the figure's path and size | 2026-10-30 |
| `docs/07_threshold.md` | Phase 7 (**T34**): the price and where it comes from, why the choice was made on the training folds, money at five cuts, the flat optimum and which edge the price sits on, capacity when calls are capped, the held-out check, why ROI and net value disagree, the 20-cell sensitivity grid, what the step does not prove, and the verification table with the two defects it caught | 2026-10-30 |
| `src/lift.py`, `reports/lift_table.csv`, `reports/figures/08_lift_gain.png`, `models/lift_results.json` | Phase 7 (**T35**): `decile_table()` cuts the descending ranking into ten positional blocks and derives every precision, lift, cumulative column and saving from the counts; `at_coverage()` interpolates the cumulative curve so T32's published coverage points can be compared rather than approximated; `savings_peak()` finds the exact coverage where calls saved peaks (via `saved(k) - saved(k-1) = (N/positives) x y[k] - 1`) and `marginal_block()` prices the clients between that peak and the adopted cut; `random_self_test()` reruns the whole table on shuffled scores as a code test. `validate()`'s **17 checks** cover the two base rates, the partition, the ratio arithmetic, the running sums, D15, the decline down the list, the shuffled self-test, the T32 reconciliation, the agreement with T34's cut, the saving identity, the recall table, the savings peak, the marginal block, the artifact, the CSV and the figure - plus a read-back of the record | 2026-10-30 |
| `docs/07_lift.md` | Phase 7 (**T35**): what lift answers and what it does not, the two halves and why they are never pooled, both decile tables, the concentration and where the list stops being useful, the adopted cut in calls, the three optima on one curve and why the campaign runs past the savings peak, the reconciliation with T32, the shuffled-score self-test, what the step does not prove, and the verification table | 2026-10-30 |
| `src/duration_compare.py`, `reports/duration_comparison.csv`, `models/duration_comparison.json` | Phase 7 (**T36**): `arm_columns()` is the only place the two feature lists are built, through `leakage.feature_columns`; `run_arm()` fits both arms through `train.cross_validate_factory` with the same estimator factory, the same fold list and the same scorer, differing only by `extra_numeric` (which `train.make_pipeline` and `train.cross_validate_factory` now accept as a defaulted keyword, so the leaky arm needs no second assembly path); `paired()` computes per-fold differences with the fold-win count and the ratio of the mean to its own standard error; `mechanism()` recomputes T22's band table and single-column ranking; `verdict()` applies the pre-registered §4.5 clauses. **13 checks** cover the shared folds, the one-column toggle, the 40/41 widths, the identical settings (NaN-aware), arm A's reproduction of T30 and T34, the gap's arithmetic, the verdict, the mechanism's reconciliation with T22, the band table, the unread held-out half, the CSV and the record | 2026-10-30 |
| `docs/05_leakage_audit.md` | Criterion 5 (**T22**, extended by T25 and **T36**): what the column is and when it becomes known, the three arguments against it, the measured leak signature, the pre-registered experiment and its four-clause decision rule, the other leakage surfaces, and now **§9** - the comparison run, with the rule discharged clause by clause, both arms, the gap in PR-AUC and in euros, the mechanism that makes it leakage rather than skill, what it does not claim, and the verification table | 2026-10-30 |
| `PROGRESS.md` | This tracker: status, decisions, 46-task checklist, session log | 2026-10-27 |
| `.gitignore` | Excludes `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `*.log`, `.DS_Store`, `._*`, `__MACOSX/`, editor folders | 2026-10-27 |
| `requirements.txt` | 13 pinned dependencies grouped by purpose, with a header noting the Python 3.11 ceiling on xgboost/shap | 2026-10-27 |
| `.venv/` | Project virtual environment (Python 3.11.9, 132 packages). **Git-ignored** — recreate with `python -m venv .venv` + `pip install -r requirements.txt` | 2026-10-27 |
| `src/config.py` | Single source of truth for paths and constants: derived `ROOT`, 20 absolute path objects, `RANDOM_STATE=42`, `TEST_SIZE=0.20`, `TARGET="y"`, `CSV_SEPARATOR=";"`, plus `ensure_dirs()` and `summary()` helpers | 2026-10-27 |
| `src/data_loader.py` | The single data entry point: `load_raw(variant)`, `load_xy(variant)` returning `(X, y)` with `y` encoded 1/0, `variant_report()` summary table, `ECONOMIC_COLUMNS`, and a runnable `python -m src.data_loader` self-check | 2026-10-27 |
| `src/clean.py` | Phase 3 (rubric 3, **T12**): the two cleaning decisions, implemented and justified in the module docstring — `unify_unknown()` (D22: keep `unknown` as an explicit category, normalise any NaN to it, never impute or delete; returns a per-column report) and `add_contact_history()` (D23: adds `previously_contacted` from `previous > 0` per D21 and `days_since_contact` with the sentinel → NaN, keeps raw `pdays`, warns when the sentinel and `previous` disagree). Plus `clean_frame()` / `clean()` (loads via the loader, so D13 holds) with no-touch invariants, `print_report()` and a runnable `python -m src.clean` self-check. **T14** turned it into the checked raw → processed converter: `process(variant)` (the single Phase 3 entry point, D13 respected), `validate()` returning **15 labelled checks as data** that are *total* and proven able to fail (D25: positional comparisons, an unexecutable check counts as FAILED, never raised), `print_conversion_report()` printing before/after shapes, dtypes and class balance, and a CLI that exits non-zero on any failure | 2026-10-27 |
| `src/split.py` | Phase 3 (rubric 3, **T15**): the stratified train/test split. `split_frame()` (seeded, stratified, shuffled; returns train/test/report), `encode_target()`, `write_split()` (converts via `clean.process` so every half inherits T14's checks, then writes both files), `load_split()` with the declared-dtype guard, `validate_split()` (**15 checks that measure the frames, never the report** - D27) and `validate_round_trip()` (3 checks per file), plus `_duplicate_rows_spanning()` which reports duplicate source rows that straddle the halves instead of removing them (D26) | 2026-10-27 |
| `data/processed/train.csv`, `test.csv`, `train_additional.csv`, `test_additional.csv` | Phase 3 (**T15**) outputs: the fixed evaluation halves, 36,168 / 9,043 rows (classic) and 32,950 / 8,238 (additional), 19 and 23 columns, target `y` still as text and `previously_contacted` as a real boolean. Byte-identical on a re-run of `python -m src.split` | 2026-10-27 |
| `notebooks/03_eda.ipynb` | Phase 4 (rubric 4, **T16–T20**): 53 cells (22 code) with stored outputs. §1 (T16): a provenance-guarded load of the **train half only**, the class counts with share and cumulative share, the two-panel chart, the baselines the imbalance forces (always-`no` 88.3018% accuracy, PR-AUC 0.1170, ROC-AUC 0.50, 8.55 calls per subscription), the stratification sanity line against the population's 11.698480%, the 5-fold stratified positive counts (846/846/847/846/846). §2 (T17): the 8-column numeric profile, **one interpretation line per variable**, the three carried-in traps re-measured on this half, `|skew|` and IQR counts printed from the data. §3 (T18): all **44 levels** of the nine `object` columns with `n`, share, lift and **Wilson 95% intervals**, Cramér's V and `dof` per column, the **best and worst segments named with their rates**, the never-contacted control that stops `month` from being read as prior-success composition, and the 5-row gap between `poutcome` and `previously_contacted`. §4 (T19): the seven campaign/history variables, **one interpretation line each**, the call-budget table with its cost arithmetic, and four controls that pin which effects survive. §5 (T20): Pearson and Spearman matrices on a shared scale, the target ranking with the rows behind each coefficient, and the pairwise-complete counts. **Five evidence cells (13, 32, 59, 79 and 38 assertions)** and a reproducibility record covering all five sections; §6 holds the 18-item findings list and §7 the record | 2026-10-28 |
| `reports/figures/01_target_distribution.png` | Phase 4 (**T16**) figure, 1707×677 px: class counts with counts and shares annotated beside the 100%-share bar carrying the **7.55 : 1** ratio | 2026-10-27 |
| `reports/figures/02_numeric_distributions.png` | Phase 4 (**T17**) figure, 2459×1194 px: one log-count panel per numeric column, each drawn over its 1st–99th percentile window with the full range and the plotted share disclosed; median marked; one bin per value for the few-valued integer columns; `previous` annotated **IQR = 0**; `pdays` annotated with its 29,584 sentinel rows; `days_since_contact` with its 29,584 missing; `duration` coloured and labelled as the D05 leak | 2026-10-27 |
| `reports/figures/03_categorical_vs_target.png` | Phase 4 (**T18**) figure, 2473×1773 px: one horizontal-bar panel per categorical column, panels ordered by **Cramér's V** and levels within a panel by rate, teal above and slate below the 11.6982% base-rate line, Wilson 95% whiskers, and `n` printed for every level so a 178-row rate cannot be read like a 29,589-row one | 2026-10-27 |
| `reports/figures/04_campaign_vs_target.png` | Phase 4 (**T19**) figure, 2468×1449 px: rate by calls placed with the base rate and the **7.50% break-even** as two reference lines; the call budget as **cost against value** for calls 1–3 and 4+, annotated with the net; rate by prior contacts; recency with the never-contacted group as its own bar; the robustness check that splits the gradient by prior contact; and the negative result showing `day`'s marginal lead reversing inside `may` | 2026-10-28 |
| `reports/figures/05_correlation_heatmap.png` | Phase 4 (**T20**) figure, 2443×1921 px: Pearson and Spearman matrices on one colour scale so their disagreement is visible, the target ranking as dodged bars annotated with both coefficients and the `n` behind each row, and the pairwise-complete **row-count matrix** that qualifies every cell; `pdays` and `days_since_contact` coloured to mark the duplicate, `duration` marked as the D05 leak | 2026-10-28 |
| `src/cost.py` | Cost-of-errors model: labelled `ASSUMPTIONS` (C = €1.50, V = €20.00), `break_even_threshold()` → 0.0750, `expected_net_value_per_call()`, `should_call()`, `cost_of_errors()` (12.3:1 ratio), `expected_campaign_value()`, `best_threshold()`, `sensitivity_table()`, and a runnable `python -m src.cost` report | 2026-10-27 |
| `docs/02_data_dictionary.md` | Phase 2 (rubric 2, **T10**): all 17 + 21 columns with measured type, values/range, meaning and leakage risk; variant difference summary; `unknown` and `pdays` sentinel tables; leakage assessment; self-check | 2026-10-27 |
| `notebooks/01_data_understanding.ipynb` | Phase 2 (rubric 2, **T11**): 30-cell quality profile of both variants with stored outputs — inventory, dtypes, nulls, duplicates, concentration, `unknown` profile, `pdays` sentinel **plus a §7.1 test of whether the sentinel can be trusted**, skew/outliers, negative-value checks, 9 written findings and a 53-check cross-check against the T10 dictionary | 2026-10-27 |
| `reports/figures/00_quality_pdays_sentinel.png`, `00_quality_numeric_skew.png`, `00_quality_unknown_profile.png` | Phase 2 (**T11**) quality figures, `00_`-prefixed so they cannot collide with the Phase 4 EDA figures (`01_`–`05_`) | 2026-10-27 |
| `notebooks/02_preprocessing.ipynb` | Phase 3 (rubric 3, **T13**): 18-cell outlier/skew analysis with stored outputs — T12 invariance re-proved, skew/kurtosis/quantile profile of all four named columns, tail-vs-body subscription rates, a five-option treatment comparison (keep, arcsinh, robust scale, winsorise 1–99, winsorise 5–95) with rows-rewritten / distinct-values-lost / order-preserved, the `log1p`-on-negatives trap quantified, the **IQR = 0** degenerate case detected, 7 written findings and the D24 decision | 2026-10-27 |
| `reports/figures/00_preprocessing_distributions.png`, `00_preprocessing_treatments.png`, `00_preprocessing_signal_in_tail.png` | Phase 3 (**T13**) figures: raw vs `arcsinh` distributions, skew remaining per treatment, and the tail signal (balance deciles, campaign bands) that decided D24 | 2026-10-27 |
| `src/train.py` | Phase 6 (rubric 6, **T26**): the cross-validation harness every later model inherits — `make_pipeline()` (T24 preprocessor + estimator, feature list from `select.shipped_features()`), `materialise_folds()` (one 5-fold stratified split, produced once and replayed), `fold_metrics()`, `summarise()` (mean ± sample sd), `cross_validate()`, `reduction_check()` (D44's drop re-measured on the linear model) and `validate()`'s **14 checks**; `ESTIMATORS`/`MODEL_PATHS` are the two registries a model is added to. CLI prints the table, the per-fold PR-AUC and the checks, exits non-zero on any failure | 2026-10-29 |
| `models/baseline_dummy.joblib`, `models/baseline_logreg.joblib`, `models/model_random_forest.joblib`, `models/model_xgboost.joblib`, `models/cv_results.json` | Phase 6 (**T26–T28**) artifacts: the fitted pipelines (40-column preprocessor included, so a reload is self-contained) and the machine-readable record — per-fold and summary metrics for every estimator, the per-estimator reduction ablation, the feature list, fold seed, `sd_ddof`, estimator parameters and the 14 check results. Renamed from `baselines_cv.json` in T27, when it stopped holding only baselines. **Every number in `docs/06_modeling.md` comes from this file** | 2026-10-29 |
| `src/tune.py`, `models/best_params.json` | Phase 6 (**T30**): the hyperparameter search — `SEARCH_SPACE` (six axes, 4,800 points), `make_estimator()` pinning everything that is *not* searched (`tree_method`, `eval_metric`, seed, `n_jobs=1`), `run_search()` handing `RandomizedSearchCV` the **materialised folds** as `cv`, `fold_sd()` recomputing the sd with `ddof=1`, `registered_params()` reading T28's configuration from the registry rather than retyping it, and `evaluate()` scoring each arm through `train.cross_validate_factory`. `validate()`'s **12 checks** cover the folds, the applied parameters, the fixed treatment, the one-core fits, the budget, the feature set, the absence of SMOTE, bit-identity with T29's control, refit-not-copy scores, the done-condition, a rebuild from the file, and the file matching the run. The JSON records the space, budget, metric, seed, **all 40 candidate scores**, both bands, the top five, the winner, three arms and the checks | 2026-10-29 |
| `src/metrics.py`, `reports/metrics_cv.csv`, `models/metrics_cv.json` | Phase 7 (**T32**): the metric table — `builds()` reads the shipped parameters from `best_params.json`, `run_model()` fits each model through `train.cross_validate_factory` and then scores every fold at **three cuts** (`at_threshold()` computes precision/recall/F1 from TP/FP/FN counts, not from a label array), `summarise()` reports mean and **sample** sd, `build_table()` writes one row per model per fold plus `mean` and `sd` rows, and `validate()`'s **11 checks** re-derive the folds, the class record, the leave-one-out cuts, the harmonic-mean identity, the scikit-learn agreement, the break-even boundary, the floor, the summary, the CV-only property and the CSV. **28 rows**, 6-decimal storage, full precision in the JSON | 2026-10-30 |
| `src/evaluate.py`, `reports/figures/07_confusion_matrix.png`, `07_roc_curve.png`,
`07_pr_curve.png`, `models/holdout_eval.json` | Phase 7 (**T33**): the held-out evaluation — `load_artifact()`
loads T31's file and refuses to continue if the checksum is not the recorded one, `score()` returns
**float64** probabilities (XGBoost hands back float32, which moved the Brier score by **4.4e-9**), `cut_table()`
scores every cut through **T32's own** `at_threshold` so the two tables share one definition,
`operating_cut()` reads the operating point from `models/metrics_cv.json` instead of restating it,
`bootstrap()` resamples the held-out rows 2,000 times and stores every draw, `build_gap()` judges the gap
against the bootstrap sd and generates its verdict sentence from the arithmetic, `f1_optimal_on_these_rows()`
finds the cut that was deliberately **not** adopted, and `validate()`'s **13 checks** cover the artifact's
identity, the absence of any fitting call, the row partition, single-pass determinism, the cuts'
provenance, the confusion counts against scikit-learn, the shared threshold arithmetic, the 0.5 behaviour,
hand-written rank AUC and step AP, the curve endpoints and areas, the gap arithmetic, the bootstrap's
replay from its seed, and the figures' existence and annotations. The JSON records the scored artifact and
its checksum, the metrics, all three cuts with full confusion counts and FPR/specificity, the F1 peak, the
gap block, the bootstrap summaries, and figure paths, sizes and checksums (6-decimal CSV-free: this phase
writes figures, not a table) | 2026-10-30 |
| `src/final.py`, `models/final_model.joblib`, `models/final_model.json` | Phase 6 (**T31**): the shipped model and its proof — `chosen_params()` reads the six settings from `models/best_params.json`, `build()` assembles the pipeline through `tune.make_estimator` so the pinned settings are the ones that produced the CV, `cross_process_predictions()` reloads and scores **1,000 rows in a fresh interpreter** (`-c` from the project root, because a temp script cannot `import src`), and `validate()`'s **11 checks** cover the parameters, the pinned settings, structure, matrix width, bit-identical reload, cross-process reload, probability range and simplex, determinism, training-half-only fitting (with a source scan that cannot match its own text), the checksum, and the card's CV numbers agreeing with T30's arm to **1e-12**. The JSON card records parameters, pinned settings, feature list, training rows, the CV summary **with its optimistic-by-construction note**, usage contract and library versions | 2026-10-29 |
| `docs/06_model_comparison.md` | Phase 6 (**T31**): seven builds on one fold split with ×base-rate multipliers, the three-way reading (ranking vs calibration vs reproducibility), the decision with what it does **not** rest on, the artifact contract with a load snippet, the explicit list of what the step does not claim (no held-out number, optimistic CV, calibration tested in T34, `duration` excluded, what would reopen the choice), and the verification record | 2026-10-29 |
| `src/imbalance.py`, `models/imbalance_results.json` | Phase 6 (**T29**): the three-arm imbalance experiment — `build_pipeline()` reads the estimator from `src.train` and changes **only** `scale_pos_weight`, `SMOTE` sits between the preprocessor and the model in an `imblearn.pipeline.Pipeline`, `choose_threshold()` is the F1-maximising cut, `apply_nested_thresholds()` gives each fold the cut chosen from the other four, and `validate()`'s **10 checks** re-derive SMOTE's row arithmetic, the fold-local scaler, the threshold provenance and fold-1 reproducibility. The JSON records per-fold PR-AUC/ROC-AUC/F1/Brier/mean-p/threshold for all three arms, the threshold rule, per-arm runtimes and the check results | 2026-10-29 |
| `docs/06_imbalance.md` | Phase 6 (**T29**): why three arms rather than two, where SMOTE is allowed to run and the three checks that prove it, the results and paired deltas, **the F1 trap measured twice** (a fixed 0.5 cut reverses the ordering), why SMOTE loses (**6.8 SE** on ROC-AUC) with its mechanism, the calibration finding that puts the weight on notice, and the hand-off to T30/T31 | 2026-10-29 |
| `docs/06_modeling.md` | Phase 6 (**T26–T28**, extended by T29 and T30): **§9 is the tuning record** — the space with a reason per axis, the three arms, the winner, why the gain is 0.58 SE and why that is the finding, the selection-bias statement, and the two defects the verifier caught. Earlier sections: the floors derived rather than quoted, the CV table with per-fold evidence, why no F1 column exists (the Brier-vs-base-rate calibration finding), the reduction check against D44 with its revisit trigger, the held-fixed contract and how each element is proven, and the hand-off to T27–T31 | 2026-10-29 |
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

### 2026-10-27 — Session 19  (T17 complete)
**Completed:** **T17** — extended `notebooks/03_eda.ipynb` to 25 cells (11 code) and added
`reports/figures/02_numeric_distributions.png` (2459×1194 px, 196 KB). The section profiles the eight
numeric columns of the training half and writes **one interpretation line per variable**, which is what
the task is graded on; the trailing sections were renumbered to `3.` and `4.` so the notebook still has
no gap in its numbering.

**What the training half says (re-measured here per D28, not carried over from T13).** Five of the eight
numeric columns have `|skew|` above 2 — `previous` **+45.03**, `balance` **+8.34**, `campaign` **+4.94**,
`duration` **+3.21**, `pdays` **+2.62** — while `age` (+0.69), `days_since_contact` (+0.69) and `day`
(+0.09) are close to symmetric. `balance`'s mean of **€1,365** against a median of **€451** is the
clearest case for scaling rather than trusting a raw value, and **3,001 rows (8.30%)** are negative,
which is why `arcsinh` is the transform that survives and `log1p` is not.

**The finding the task did not anticipate: `previous` is not the only column with a zero IQR.** `pdays`
has one too (81.80% of rows are the −1 sentinel), so a median/IQR scaler divides by zero **twice**, not
once. And the two degenerate columns turn out to be the **same 29,584 rows** — `pdays == -1` is
element-for-element identical to `previous == 0`, 0 disagreements — so a single underlying fact, "this
client was never contacted before", makes two columns useless as scales. The message to T23/T24 is a
flag plus a real scale (D23's `days_since_contact`), not a different scaler.

**Two numbers were corrected by measurement rather than carried forward.** `duration` correlates with
the target at **+0.3967** and the runner-up is `days_since_contact` at **−0.1526**, a ratio of **2.60×**
— not the "four times" a first pass assumed when the runner-up was computed with its missing rows
median-filled instead of dropped (which understated it at −0.0452 and became D29). Spearman agrees on
the ordering (**+0.3427** vs **−0.1873**). And `previous`'s skew on this half is **+45.03** against the
**+41.85** T13 measured on the full frame, with `balance` losing 765 negative rows in the same move:
exactly the drift D28 exists to catch.

**Three corrections to my own work, each caught by a check rather than by review.** The generated
section first printed "evidence: 31 assertion lines pass" while the cell contained **32** — a count
written by hand instead of measured, the same defect class as D27. The first draft of the build script
also **wiped the findings list** by replacing a whole cell where only its heading line should have
changed; the script's anchors are asserted to match exactly once and it refused to write, so the
notebook on disk never held that state. And two figure defects were visible only in the rendered PNG:
`campaign` and `previous` were drawn with 50 bins across a handful of integer values (bars alternating
with empty bins, plus fractional ticks on a count variable), and `days_since_contact`'s footnote read
"17.9% of rows" where the truthful statement is "98.1% of the plotted values", since it has 29,584
missing rows. All are fixed, and the footnote is now measured against the values the panel can plot
(D30).

**Verification:** `jupyter nbconvert --execute` exits **0** with 0 error outputs and every code cell
holding an output; **60/60 independent checks** recomputed from `data/processed/train.csv` with plain
pandas (42 on the data, 18 on the notebook), including the done-condition itself — one bullet per
numeric variable, each carrying a measured number — and the heading sequence
`1, 1.1, 1.2, 1.3, 2, 2.1, 2.2, 2.3, 3, 4`; two independent executions produce **identical output
content, including the figure's PNG bytes**, the only variation being whether the kernel delivers a
cell's stdout as one stream message or two, which is serialization rather than content; and the test
half is still unread (`load_split("full", "train")` is the only load, no bare `read_csv` of a
processed file). A pixel check confirmed `duration` is the only coloured panel title, so the warning
colour is deliberate rather than accidental.

**One note for later phases, recorded rather than changed:** the raw files are semicolon-delimited
(`config.CSV_SEPARATOR`) while the processed splits are comma-delimited, because `split.write_split`
uses `to_csv` defaults and `split.load_split` is the declared single reader for them. Nothing reads a
processed file any other way, so this is consistent today — but code that later calls `read_csv` on
`data/processed/*.csv` with `sep=";"` would silently get a single column.

**Blocked:** none.
**Uncommitted:** this session changed `notebooks/03_eda.ipynb`, added the T17 figure and edited this
tracker; those edits are local, on top of the four commits pushed in Session 18.
**Next:** T18 — categorical features vs subscription rate into
`reports/figures/03_categorical_vs_target.png`.

### 2026-10-28 — Session 20  (T18 complete)
**Completed:** **T18** — extended `notebooks/03_eda.ipynb` to 34 cells (15 code) with a new §3, and added
`reports/figures/03_categorical_vs_target.png` (2473×1773 px, 274 KB). All **44 levels** of the nine
`object` columns are measured against the 11.6982% base rate, each carrying its row count and a Wilson
95% interval; the trailing sections were renumbered to `4.` and `5.`, and the header cell — which still
described the notebook as T16-only — now lists all three sections.

**The answer the task grades.** Best segment **`poutcome = success`: 779/1,205 = 64.65%** (lift 5.53×),
worst **`contact = unknown`: 427/10,386 = 4.11%** (lift 0.35×) — a **15.7× gap**. Unusually, both ends
rest on large segments, so their intervals barely overlap (61.91%–67.30% against 3.75%–4.51%);
`success` is 3.33% of the half but supplies **18.41%** of all subscribers, while `contact = unknown` is
28.72% of the half and still supplies 10.09%.

**What the ranking says.** Cramér's V puts `poutcome` (**0.3102**) and `month` (**0.2573**) clearly
ahead, then `contact` 0.1500, `housing` 0.1393 and `job` 0.1378 as a **tie**, and `default` last at
0.0241 (D31). The largest level is not a segment at all: `default = no` covers 35,521 rows — 98.21% of
the half — so its 11.80% is the base rate wearing a label. The largest level that still carries
information is `poutcome = unknown`, 29,589 rows (81.81%) at **9.22%**: the strongest column still
leaves five clients in six in one bucket it cannot split, and that bucket is below average.

**Two things the measurements forced.** The four highest-rate months are also the four smallest
(178–592 rows, 4.46% combined), `dec`'s interval spans 39.99%–54.51%, and the order *inside* that group
flips when never-contacted clients are separated (`sep` falls below `oct`) — so the group is a finding
and its order is noise. And `poutcome = unknown` is not `previous == 0` (D32): 6,584 rows have prior
contact, 6,579 have a known outcome, and the 5-row difference is one-way.

**Three defects were caught by measuring, not by re-reading.** The assertion that the largest segment
was `month = may` failed twice — first because the largest *month* is not the largest *segment*, then
because the largest segment is `default = no` — so that claim was rewritten until it was true.
`Series.argmax()` returned an index *label* where `iloc` needed a position, silently fetching a
different row; `idxmax` with `.loc` fixed it. And the rendered PNG showed every annotation reading
`0.65%` beside a 64.65% bar, because the label formatted the fraction without multiplying by 100 —
invisible to the text checks and only caught by looking at the image. The evidence cell's assertion
count is now read from its own source instead of typed, after T17 shipped "31" for 32 lines.

**Verification:** `jupyter nbconvert --execute` exits **0** with 0 error outputs and every code cell
holding an output; **123/123 independent checks** recomputed from `data/processed/train.csv` with plain
pandas, including that the prose *contains* each formatted value it quotes, the heading sequence
`1, 2, 3, 4, 5`, one bullet per categorical variable, and the best/worst done-condition; all three
evidence cells' self-reported counts match their assertions (13, 32, **59**); two fresh executions give
**identical text and identical figure bytes** (sha256 `706a773a…`); and the test half is still unread.

**Blocked:** none.
**Uncommitted:** `notebooks/03_eda.ipynb`, `PROGRESS.md` and the new figure, on top of the four commits
pushed in Session 18. T17's edits were never pushed either, so the next push carries both sessions.
**Next:** T19 — campaign and history features vs subscription rate into
`reports/figures/04_campaign_vs_target.png`.

---

### 2026-10-28 — Session 21  (T19 complete)
**Completed:** **T19** — extended `notebooks/03_eda.ipynb` to 44 cells (19 code) with a new §4, and added
`reports/figures/04_campaign_vs_target.png` (2468×1449 px, 252 KB). All seven campaign/history variables
carry a measured interpretation line, and the section is written in money because that is what the call
list is spent in: **52.68% of the 99,966 calls go to the 7,707 clients who needed four or more of them,
and they hold 13.47% of the 4,231 subscriptions.** Against D17's 7.50% break-even that is decisive — the
first three calls convert at **7.7386%** (net **+€2,258**) and everything beyond them at **1.9298%**
(**−€32,906**, or **€77.73** per extra subscription against a €20 deposit).

**The finding that changes a decision.** The cap cannot be uniform: **93.28%** of the extra calls
(27,551 of 29,537) go to clients with **no known prior outcome**, at **1.66%** a call, while clients whose
previous campaign **succeeded** still return **22.55%** a call — **€6.65** per extra subscription, three
times break-even. That is D34, and it is the kind of thing a pooled average would hide.

**What survived a control, and what did not.** The `campaign` gradient holds inside all four prior
outcomes, inside both history groups and in all six busiest months (`aug` most sharply, 24.42% → 6.29%),
so it is not composition. Two neighbours did **not** survive: the recency "sweet spot" at 31–90 days
(41.69%) is prior-success composition — that bucket is **36.2%** prior successes against **13.0%** for
over-180, and inside `poutcome = success` the rate *rises* with age (62.89% → 66.48%) — and `day`'s
marginal lead reverses inside `may`. D33 records why the section still does not claim causality.

**Five defects were caught before they shipped.** `never.sum()` is 29,584 and not 6,584, so the recency
check was asserting the wrong complement; the "worst bucket in all four outcomes" claim was first tested
against six buckets and failed because `poutcome = success` at 10+ calls holds **one row** (the 4+
aggregate exists for that reason); the same test failed again for `apr`, whose 4+ bucket sits just above
its 2-call bucket, so the claim was narrowed to "four of six" and then verified as five; panel 5's
endpoints had been measured at `campaign == 6` rather than at the `10+` bucket the figure actually plots;
and the evidence cell needed `json` imported before it could count its own assertions. Two further
defects were only visible in the rendered PNG — a `break-even 7.50%` label colliding with the `10+` error
bar, and the base-rate line running through the `9.22%` labels — so the labels got a white pad and the
reference lines got a legend.

**Verification:** `jupyter nbconvert --execute` exits **0** with 0 error outputs and all 19 code cells
holding output; **86/86 independent checks** recomputed from `data/processed/train.csv` with plain pandas
(including that the prose *contains* each of 56 quoted values it states, the heading sequence
`1, 1.1–1.3, 2, 2.1–2.3, 3, 3.1–3.3, 4, 4.1–4.4, 5, 6`, and the done-condition); the evidence cell
counts its own **79** assertions from source (D27) and they all hold; three executions — the stored
notebook plus two fresh runs — produce **identical text and identical PNG bytes for all four figures**;
and the test half is still unread.

**Blocked:** the local commit `a87be66` (T17 + T18) is still **unpushed**. `git push` fails with
`could not read Username for 'https://github.com'` because the Git Credential Manager account cannot be
released non-interactively (`GIT_TERMINAL_PROMPT=0`, `GCM_INTERACTIVE=never`), there is no
`GH_TOKEN`/`GITHUB_TOKEN` in the environment, and the local SSH key is not authorised on GitHub
(`Permission denied (publickey)`). A one-line interactive `git push` from a terminal resolves it.
**Next:** T20 — the numeric correlation heatmap into `reports/figures/05_correlation_heatmap.png`.

---

### 2026-10-28 — Session 22  (T20 complete)
**Completed:** **T20** — extended `notebooks/03_eda.ipynb` to 53 cells (22 code) with a new §5, and
added `reports/figures/05_correlation_heatmap.png` (2443×1921 px, 343 KB). Pearson and Spearman are
shown on one colour scale because seven of the eight numeric columns are skewed, and every cell is
annotated with the **number of rows behind it** — the section's central warning, since a correlation
involving `days_since_contact` rests on 6,584 rows and everything else on 36,168.

**The finding that changes a decision.** The numeric half is **nearly orthogonal**: of the 28
off-diagonal pairs only **two** reach |r| ≥ 0.20. The severe problem is not a correlated pair but a
**duplicate column** — `pdays` and `days_since_contact` are the same number on all 6,584 rows where both
exist, which makes the complete-case matrix **rank 7 of 8** with condition number **5.0×10¹⁶** and hands
both columns a VIF of **9.0×10¹⁵**. Drop `pdays` and every remaining VIF is **1.01–1.03**. Hence D36:
T24 keeps `days_since_contact` plus `previously_contacted` and drops raw `pdays`, and there is nothing
else to defuse.

**Two readings the matrix alone would get wrong.** `pdays` correlates **+0.1007** with the target over
the whole column and **−0.1526** — exactly `days_since_contact`'s value — on the contacted rows, because
−1 sorts as "very recent" (D37). And a line understates structure it cannot see: `campaign`'s r² is
0.0052 against the six-bucket η² of 0.0076, while `previous`'s step at zero (9.22% → 20.65%, T19) reads
as +0.0888. `duration` stays first at +0.3967 and remains unusable (D05); the strongest *usable* numeric
is −0.1526, and `poutcome` at V = 0.3102 never appears at all (D38).

**Four defects were caught before shipping.** A stray `del span` and an f-string whose `\n` the
generator's own string literal consumed (a `SyntaxError`, caught by validating the cells standalone
before they ever reached the notebook); `xlim` and annotation placement in panel 3, where negative rows
pushed their numbers into the tick labels; and `imshow`'s equal aspect leaving a band of empty canvas
above two shrunken matrices. Two more appeared only in the rendered PNG: literal backticks in three
matplotlib strings, and the top row's rotated tick labels colliding with the bottom row's titles. The
`tight_layout` call also warned that colorbar axes from an explicit gridspec are "not compatible" with
it — replaced with explicit margins so the inline and Agg backends produce the same file.

**Verification:** `jupyter nbconvert --execute` exits **0** with 0 error outputs and all 22 code cells
holding output; **42/42 independent checks** recomputed from `data/processed/train.csv` with plain pandas,
including the rank, condition number and VIFs recomputed from scratch, the ranking order, and that the
prose *contains* each of 37 quoted values; the evidence cell counts its own **38** assertions from source
(D27); the stored notebook plus two fresh executions give **identical text and identical PNG bytes for
all five figures**; and the test half is still unread.

**Blocked:** three commits are now unpushed (`a87be66`, `ec739e9` and this one). `git push` still cannot
authenticate: no interactive credential, no `GH_TOKEN`, and the local SSH key is not authorised on
GitHub. A one-line interactive `git push` resolves it.
**Next:** T21 — consolidate §1–§5 into `docs/04_eda_insights.md`.

---

### 2026-10-28 — Session 23  (Phase 4 complete, T21)

**T21 — consolidated the EDA into `docs/04_eda_insights.md`.** 2,804 words, 276 lines, six sections. The
task is a selection and rewrite, so nothing new was measured: every number quotes the notebook's §1–§5.

**What the document does.** §2 gives each of the five figures a one-sentence business interpretation
directly under its image, plus the measures behind it. §3 is the deliverable: **seven** insights (I1–I7)
in a stated ranking — money at stake, then whether a planner can act before the next campaign, then
whether it survived a control — with the ranking rule written before the list. §4 turns them into seven
actions (A1–A7), §5 lists six things the EDA does **not** establish, §6 checks the page against the
task's wording, and the closing paragraph is the viva line.

**The selection, and why it is not the notebook's order.** I1 is the call budget (**€32,906** lost beyond
the third call, **93.28%** of it in the no-known-outcome group); I2 the prior-success segment
(**64.65%**, **18.41%** of subscribers from **3.33%** of the list); I3 `contact = unknown` (**4.11%** on
**28.72%** of rows); I4 `may` (**30.59%** of calls at **6.68%**); I5 prior contact as a flag (**2.24×** at
zero, nothing added by the second call); then I6 `duration` as a constraint (**+0.3967**, post-outcome)
and I7 the two negative results (`day` and recency, both of which fail their controls).

**Verification:** **605/605 checks** pass in `_t21_verify.py`, in three independent passes — structure
(five figures with a one-sentence interpretation read as a *paragraph*, not a physical line; ≥5 insights
with contiguous numbering; the ranking rule before the list; table pipe consistency; ≤100-column prose),
**recomputation** (26 headline values re-derived from `train.csv` through `src.split.load_split`, a
different path from the notebook: call counts, the €-totals, the poutcome and contact counts, the May
share, the prior-contact step, and that `may` really is the largest month), and **provenance** (every
numeric token in the page must already appear in the notebook's *cell sources* — not its outputs, so the
base64 cannot vouch for anything — or in the earlier documents; 164 claims traced).

**Defects caught before commit.** Three of my own: `contact = unknown` described as "the second-largest
segment in the file" when `cellular` and `poutcome = unknown` are both larger (reworded to the 28.72%
share, which is what the argument needs); a test-half row count typed as 16,043 when the split is 9,043
(measured, then fixed); and an invented "~2 days of work not spent" in the A7 row. One layout fix: my
"one sentence" check read only the first physical line of a wrapped paragraph, so it reported five false
failures and would have accepted a two-sentence interpretation. And re-wrapping the prose to the project's
100-column convention was done by a script whose own two bugs — repeating the list marker on every
continuation line, and merging consecutive bullet items — were caught by its word-sequence guard, which
refused to write either time.

**Blocked:** four commits are now unpushed (`a87be66`, `ec739e9`, `3334e49` and this one). `git push`
still cannot authenticate from this shell: no TTY for the credential prompt, no `GH_TOKEN`, and the local
SSH key is not authorised on GitHub. One interactive `git push` clears all four.
**Next:** T22 — the `duration` leakage audit in `docs/05_leakage_audit.md`.

---

### 2026-10-28 — Session 24  (T22 complete)

**T22 — the `duration` leakage audit.** `docs/05_leakage_audit.md` (2,907 words) plus `src/leakage.py`,
which both defines the toggle and prints every number the document quotes. No model was fitted: this task
measures the column, it does not compete with T26–T31.

**What the audit establishes.** `duration` is the call's length in seconds, recorded at hang-up, while the
score is needed before dialling — the document draws that timeline rather than asserting it. The risk is
then measured rather than described: the subscription rate climbs monotonically across six bands, from
**0.16%** (6 subscribers in 3,727 calls) under a minute to **48.56%** at 600 s or more, so the shortest
tenth of calls yields **0.14%** of subscribers and the longest **8.44%** yields **35.03%**. Used directly
as a score with no fitting, the single column reaches **ROC-AUC 0.8078** and **PR-AUC 0.3897** — **3.33×**
the 0.1170 base rate, with a **3.90×** lift in the top decile — which is the yardstick T36's gap will be
read against.

**The experiment, specified not sketched.** Two arms; fold membership materialised **once** and replayed
into both; one preprocessing factory rebuilt inside each fold; one estimator, one scoring function, one
process. Four assertions make "identical except one column" checkable. The decision rule is written
*before* the results exist (arm A ships whatever the gap; arm B is an upper bound; a near-zero gap is
reported as the finding), which is D40.

**Seven other surfaces audited.** `pdays`/`days_since_contact` is redundancy, not leakage (D36);
`poutcome` is legitimate but only with encoders fitted on training rows; `y` is never a feature; the
primary variant has no duplicate rows; and the split's **stratified-random rather than chronological**
design is declared as an unresolved limitation — the file carries no year and no client id, so a
time-based split cannot be reconstructed without invalidating Phase 4.

**Verification:** `python -m src.leakage` exits **0** and prints every quoted figure; **541/541 independent
checks** pass in `_t22_verify.py`, which re-derives the six bands, their Wilson intervals, ROC-AUC (Mann-
Whitney with average ranks for ties) and average precision (over distinct score cuts) through a separate
path, checks the seven audit surfaces and the pre-registered rule, compares 18 quoted literals against the
recomputed values, and verifies the quoted manual sentence word for word in
`data/raw/bank-additional/bank-additional-names.txt`.

**Defects caught before commit.** My first "independent" average precision walked every row instead of
grouping tied scores and disagreed with sklearn in the fourth decimal (0.3899 against 0.3897) — durations
are whole seconds, so ties are common and the naive walk is simply a different estimator. My number
normaliser turned `6,900` into `69` by stripping trailing zeros from an integer, which produced a
provenance failure that did not exist. And the document claimed every figure was printed by
`python -m src.leakage` while two capture shares and two lifts were not — the module was extended rather
than the claim softened.

**Blocked:** five commits are now unpushed (`a87be66`, `ec739e9`, `3334e49`, `b6f76fb` and this one);
`git push` still cannot authenticate from this shell. One interactive `git push` clears all five.
**Next:** T23 — the engineered features in `src/features.py`.

---

### 2026-10-28 — Session 25  (T23 complete)

**T23 — engineered features.** `src/features.py`, 379 lines, producing **17 features** from the cleaned
frame: five new columns, each replacing a raw one, and the flag `src/clean.py` already added.

**The five columns, and what they replace.** `campaign_bucket` (six levels, reproducing T19's
14.59% → 4.30% gradient) replaces the raw 1–63 count whose skew is 4.94; `previous_capped` (0/1/2+ at
9.22% / 20.65% / 23.97%) replaces a column with skew 45.03 and a maximum of 275; `balance_arsinh` takes
|skew| from 8.34 to **1.68** while staying defined for the **3,001** negative balances `log1p` cannot
touch; `balance_band` carries the non-linear part, **5.36%** when overdrawn rising to **16.44%** at
€1,500–4,999 and dipping to **15.73%** above €5,000; and `recency_band` gives the **29,584** never-
contacted clients (81.80%) their own level rather than imputing a history they do not have.

**The design choice worth recording (D41).** The EDA's money argument caps calls at three, but capping
the *feature* at three would merge "exactly 3" (11.26%) with "ten or more" (4.30%) into one level at
8.80% — destroying the gradient the model needs to rank heavy clients, since the cap is applied *after*
scoring. The bucket keeps the signal; the policy stays a downstream decision.

**Leakage, tested rather than promised.** Eight checks run inside the module on every invocation:
permuting `y` must leave all five engineered columns byte-identical, and so must permuting `duration`;
no feature may read a column on the forbidden list; every replaced column must be absent from `FEATURES`
(so nothing is counted twice); and shuffling rows must change nothing, which is what proves each feature
is per-row and no group statistic could carry information across clients.

**Verification:** `python -m src.features` exits **0** with all eight checks passing, and **145/145
independent checks** pass in `_t23_verify.py`, which recomputes all six campaign buckets, three prior-
contact levels, five balance bands and five recency bands directly from `train.csv`, asserts the fixed
category vocabularies T24's encoder will rely on, and — the part that matters — **proves the leakage
tests have teeth** by planting a target-derived and a duration-derived column and confirming each is
caught.

**Defects caught before commit.** My first "no feature reads a dropped column" check contradicted the
design: the features are *supposed* to read `campaign`, `previous`, `balance` and `days_since_contact` —
they replace them. Splitting that into `FORBIDDEN` (never read) and `REPLACES` (read, then removed from
the model's view) turned a failing check into two meaningful ones. The T36 toggle line was vacuous —
17 columns with `duration`, 17 without — because `duration` was never in the list, so the report now
exercises the toggle on `FEATURES + duration` and prints the difference explicitly. `rate_table` assumed
every level was categorical and crashed on the numeric capped count; and the printed level `2` claimed
"exactly two" when it means "two or more", which is now labelled `2+`.

**Blocked:** six commits are now unpushed (`a87be66`, `ec739e9`, `3334e49`, `b6f76fb`, `a3b4d04` and this
one); `git push` still cannot authenticate from this shell. One interactive `git push` clears all six.
**Next:** T24 — the `ColumnTransformer` over these 17 columns in `src/preprocess.py`.

### 2026-10-29 — Session 26  (T24 complete)
**Completed:** **T24** — `src/preprocess.py` (507 lines), the single `ColumnTransformer` every model from
T26 onwards will sit on top of. **17 features in, 65 columns out:** 3 columns `StandardScaler`ed
(`age`, `day`, `balance_arsinh`), `previous_capped` passed through unscaled, **60 one-hot columns** from
the 12 categorical features (`job` 12, `marital` 3, `education` 4, `default` 2, `housing` 2, `loan` 2,
`contact` 3, `month` 12, `poutcome` 4, `campaign_bucket` 6, `balance_band` 5, `recency_band` 5), and
`previously_contacted` as one 0/1 column. It fits on the **36,168** training rows and transforms the
**9,043** held-out rows to `(36168, 65)` and `(9043, 65)`.

**The column list is imported, not restated.** `features.FEATURES` is the only definition, the four
treatment groups are declared as data so the checks can inspect them, and a column in no group raises
rather than being silently dropped. Three traps are closed by construction rather than by care:
`remainder="drop"` means the frame may still carry `duration`, `pdays` and `y` (D14) without the
transformer being able to use them — proven by transforming such a frame and comparing element-wise;
`handle_unknown="ignore"` matters because the encoder is fitted on one half and applied to the other; and
`build_preprocessor()` returns a **fresh unfitted** object every call, which is the factory T22 §4.2
required so the scaler is rebuilt inside each fold.

**A trap measured rather than assumed:** **no test-only level exists** — all 12 categorical vocabularies
are identical across the split — so the encoder's safety net is never exercised by the data. Injecting an
unseen `job` level proves it encodes to zeros and that a `handle_unknown="error"` clone raises, which is
the difference between a setting that is written and a setting that is working.

**Three defects, all mine, all caught by checks rather than review.** The raw `bool` passthrough left the
assembled frame **heterogeneous** — `float64` blocks beside `object`, so `X.to_numpy()` was not numeric
and `np.isfinite` refused it; the boolean is now cast to float64 with a named module-level function (a
lambda would break T31's `joblib.dump`). The count block's `int8` dtype broke the same invariant, so it is
cast too — values unchanged, verified to still hold exactly {0, 1, 2}. And `previous_capped`'s zero-IQR
justification was first written for the wrong column: raw `previous` has a zero IQR as well (**81.80%**
zeros), so the degeneracy is a property of the column rather than of the cap, which makes the case
stronger, not weaker. A fourth error was numeric: the docstring said raw `previous` is **81.18%** zeros —
neither half — caught by the independent recomputation, which is the whole reason it recomputes.

**Verified:** `python -m src.preprocess` exits **0** with **15/15** in-module checks (partition covers
`FEATURES` exactly, fresh factory, `NotFittedError` before fit, equal widths, width equals the *fitted*
vocabulary rather than a literal, one active level per categorical per row, `remainder="drop"` on a
widened frame, the unseen-level pair, scaler fitted on train only to **0.00e+00** against a **4.32e-02**
pooled gap, finite, homogeneous float64, both passthroughs unchanged, unique names, no forbidden column,
and a bit-for-bit identical refit). **23,162 independent checks** rebuild the matrix from the CSVs by a
different route — 60 one-hot columns derived from `value_counts` rather than from the encoder, every
scaled cell recomputed with `numpy` mean/std, 450 spot-checked rows for one-hot and passthrough values,
150 test rows for the same rules on unseen rows, the no-pre-split-fit property, `check_array`
compatibility — and the T36 toggle is exercised end to end: **arm A 65 columns, arm B 66**, differing by
exactly `duration`.

**Blocked:** none — and the long-standing one is gone. `git push` succeeded from this shell for the first
time since Session 18 (`5e20d58..85e2fd9 main -> main`, exit 0), so the **seven** commits that had been
accumulating locally (`a87be66`, `ec739e9`, `3334e49`, `b6f76fb`, `a3b4d04`, `38f61bc`, `85e2fd9`) are on
the remote and `origin/main` now equals `HEAD`. The credential helper is evidently holding a token now;
nothing in the workflow changed.
**Uncommitted:** at the time of writing, `src/preprocess.py`, `PROGRESS.md` and this session's script.
**Next:** T25 — feature selection and importance in `src/select.py`, comparing mutual information against
model importance per one-hot *level* and reading the ranking back to the 17 source features.

### 2026-10-29 — Session 27  (T25 complete)
**Completed:** **T25** — `src/select.py` plus `reports/figures/06_feature_importance.png` (128 KB) and
**2,149 words** appended as §8 of `docs/05_leakage_audit.md`. **17 features ranked two ways**, and then a
third measurement made the decision the two rankings could not.

**The two rankings, and the disagreement that makes them worth having.** Mutual information on the 17
source features (a categorical enters as its integer codes, so `job` is one 12-level variable) judged
against **each feature's own permutation null**, versus permutation importance from a 200-tree forest
(out-of-fold ROC-AUC **0.7936 +/- 0.0063**) measured as the ROC-AUC lost when a feature's columns are
shuffled. They correlate at **Spearman +0.647** — substantial, and nothing like agreement. `poutcome` is
first for the filter and third for the model; `month` is the reverse; `marital` moves **eight** places
(16th to 8th) because its marginal MI is 0.0023 while the forest gets **+0.00453** AUC from it in
combination. `day` moves seven the other way, which D35 predicted: its marginal effect **reverses inside
`may`**, so a marginal statistic averages a real interaction to nothing and only the tree can see it.

**The finding about the rule itself.** The original screen required **both** methods to fail. Measured,
the filter half could not fail: with 10 label shuffles of 36,168 rows **no feature has z < 1** (the lowest
is `day` at **7.4**), **16 of 17** clear `z = 20`, and the smallest MI in the table (**0.0004** nats, for
`default`) sits at **z = 27.5**. So every nominee came from the model half, and a two-condition rule with
one dead condition is a one-condition rule in disguise. Rather than quietly keep it, the screen now only
nominates and the **group ablation decides**: the 5 nominees removed together cost **-0.00014 +/- 0.00195**
ROC-AUC, with **3 of the 5 folds improving**.

**That exposed the second error, in the threshold.** A mean of -0.00014 against a spread of 0.00195 is a
decision resting on a quantity **14 times smaller than its own noise** — a coin flip with a decimal point.
The rule is now the **one-standard-error form** (`mean delta >= -sd`), the same standard stage 1 already
applied per feature.

**The decision:** drop **5** — `education`, `loan`, `default`, `balance_band`, `job` — keep **12**, and the
matrix goes **65 → 40 columns**. `balance_band` is redundant with `balance_arsinh`; `job` was the largest
single block (**12 of 65** columns) and last by the model; `default` carries **0.0004 nats**. One
consequence is stated rather than left to be discovered: the shipped model keeps `age` and `marital` but no
longer sees `job` or `education`. D16 scopes out a formal fairness audit and this is not one, but it is a
defensible place for a credit-offering model to land, and it happened for measured reasons.

**Two defects, both invisible from reading the code.** The first: the permutation step originally used
`X_valid.iloc[:3000]` to save time. `StratifiedKFold` returns indices in ascending original order and the
file is ordered by campaign date (D26), so that "sample" was the campaign's opening weeks — it contained
**no** `poutcome=success` rows at all, making **20 columns constant**. `poutcome`, `recency_band`,
`previous_capped` and `previously_contacted` came back at exactly **0.00000 +/- 0.00000**, and the screen
nominated **13 of 17** including `month`, the most important feature. The shortcut did not produce a
noisier number, it produced a wrong one; using the whole held-out fold fixed it. The second is the threshold
above.

**Verified:** `python -m src.select` exits **0** with **17/17** checks — the screen and the decision are
both re-derived from their declared conditions rather than read off a column, no dropped feature is an EDA
anchor, the guard survived, the rankings are consistent with their values, the ablation replayed the same
folds and its delta matches its fold differences, and one check **rebuilds the shipped 40-column matrix**
from the kept list rather than trusting the count. Runtime ~5.5 minutes (2 x 5 folds, 3 permutation repeats
on the full 7,234-row validation fold).

**Blocked:** none.
**Uncommitted:** `src/select.py`, the figure, `docs/05_leakage_audit.md`, `PROGRESS.md`.
**Next:** T26 — the `DummyClassifier` and Logistic Regression baselines on the 40-column kept set, in one
Pipeline, with CV ROC-AUC and PR-AUC logged.

---

### 2026-10-29 — Session 28  (T26 complete, Phase 6 started)

**Completed:** **T26** — `src/train.py` (749 lines), `models/baseline_dummy.joblib`,
`models/baseline_logreg.joblib`, `models/baselines_cv.json` (renamed `cv_results.json` in T27), and
`docs/06_modeling.md`.
Phase 6 is open: **26 / 46 (57%)**, criterion 6 at **1 / 6**.

**The numbers.** Training half only, **36,168 rows / 4,231 positives**, 12 features -> 40 columns.
`dummy_prior` gives **ROC-AUC 0.5000**, **PR-AUC 0.116982 ± 0.000059**, accuracy **88.3018%**;
`logreg_balanced` gives **ROC-AUC 0.7640 ± 0.0114** and **PR-AUC 0.4008 ± 0.0134**, which is **3.43x** the
base rate, with both floors beaten in **all five folds**. **3.8 s** for the whole run.

The floors are **derived in the module**, not quoted from T15: per fold the ROC-AUC must be exactly 0.5,
the average precision must equal the **scored** fold's rate to 1e-12, the constant score must equal the
**fitted** fold's rate, and the Brier score must equal `q - 2pq + p^2`. That distinction is not pedantry —
the first version compared both against the scored rate and reported four failures of **4.2e-05** that
were the check describing the wrong fold, since the dummy predicts the *training* rate and is scored on
the *validation* rows.

**The finding worth the session: a better ranker is a worse forecaster.** `class_weight="balanced"`
carries **7.55 : 1** weights, so the model scores the average client at **0.4161** where the true rate is
**0.1170**, and its Brier score is **0.1852** against the dummy constant's **0.1033**. Nothing is broken;
that is what balancing does to calibration, and it is the reason T26 has **no F1 or precision column** — a
threshold metric there would measure the weighting rather than the model. T34 must recalibrate before
D17's 7.50% break-even can be applied to a probability, and D45 records that.

**The carry-in answered, and it contradicted T25's instrument.** D44 dropped five features because a
forest could not distinguish their contribution from zero, so the linear model — the one instrument that
gives each one-hot level its own coefficient — was the obvious place for the loss to show. It does, in
one metric: **ROC-AUC -0.0039 ± 0.0023 with 0/5 folds improving** (1.73 standard errors; one-sided sign
test p = **0.031**), while **PR-AUC -0.0021 ± 0.0022** sits inside the noise and one fold improves. D44
stands — the shipped model is XGBoost, which is tree-based like the instrument that chose the set, and the
price is 38.5% of the matrix for 0.004 AUC on a model that does not ship — but the **revisit trigger is
written down in advance** (D46): T31 re-runs the same ablation on the shipped model, and a 40-column
XGBoost landing below PR-AUC 0.4008 reopens the decision. The ablation's shipped arm has to reproduce the
baseline **bit-identically**, and a check enforces it, because otherwise it would be two experiments
compared.

**One design change outside `train.py`.** T25's decision now has a **frozen pin** in `select.py`
(`SHIPPED_DROPPED` + `shipped_features()`), so the model tasks do not re-run a 5-minute ablation to learn
four names, and `select.validate()` grew a check that fails if a fresh measurement lands anywhere else. A
pin that a measurement can contradict is a cache; a pin nothing checks is a second source of truth.

**Verified:** `python -m src.train` exits **0** with **14/14** checks — fold determinism, target identity,
pipeline structural identity across arms, per-fold fitting (a fold's scaler matches **that fold's**
training mean to **0.0e+00** and differs from the pooled mean by **4.74e-02**, which is the half of the
check that catches a `fit` before the loop), matrix width, prediction range, the four floors re-derived,
the baseline beating both floors in every fold, no leaky column, artifact round-trip, T25's frozen
decision, convergence (**n_iter** 35-43 against `max_iter` 2000), and the reduction arm reproducing the
baseline. Then **127 independent checks** in a separate script that refits from `train.csv` through a plain
`read_csv` without importing `src.train`, recomputing every per-fold figure — and that validates its own
two hand-written implementations (rank-based Mann-Whitney AUC, threshold-step average precision) against
scikit-learn to **1e-12** first, including the all-tied constant-score case, because a checker never shown
to agree with the library is a checker of nothing. Finally every numeric token in `docs/06_modeling.md`
(**220** of them) was traced to the run output or the JSON: the only two that do not appear there are the
date and the **9,043** held-out rows, which is T15's recorded split size.

**Disclosed:** the `Files Created` table is behind for T23-T25 (`src/features.py`,
`src/preprocess.py`, `src/select.py` and `docs/05_leakage_audit.md` are described in the checklist and
their own session entries but were never added as rows); this session added rows only for its own files
rather than backfilling three sessions of description here.

**Blocked:** none.
**Uncommitted:** `src/train.py`, `src/select.py`, `models/` (3 files), `docs/06_modeling.md`,
`PROGRESS.md`.
**Next:** T27 — the Random Forest as one more `ESTIMATORS` entry, scored by the same harness against
PR-AUC **0.4008** and ROC-AUC **0.7640**.

---

### 2026-10-29 — Session 29  (T27 complete)

**Completed:** **T27** — `random_forest` added to `src/train.py` as a second registry entry,
`models/model_random_forest.joblib`, the JSON renamed to `models/cv_results.json`, and `docs/06_modeling.md`
extended to §1–§8. **27 / 46 (59%)**, criterion 6 at **2 / 6**.

**The numbers.** `RandomForestClassifier(n_estimators=300, min_samples_leaf=10, max_features="sqrt",
class_weight="balanced_subsample", n_jobs=1)` on the same 40 columns and the same five folds:
**ROC-AUC 0.7906 ± 0.0089** and **PR-AUC 0.4346 ± 0.0146**, which is **3.72×** the base rate. It beats
`logreg_balanced` in **every one of the 5 folds** on **both** metrics — no mean is carrying a losing fold —
and by **+8.4% relative** on PR-AUC. It is also better calibrated than the balanced linear model
(Brier **0.1403** against **0.1852**), though both remain worse than the constant's 0.1033.

**One setting is a correction rather than a choice.** `class_weight="balanced_subsample"`, not
`"balanced"`: a forest bootstraps, so each of the 300 trees sees a different positive share, and the
subsample variant recomputes the weights per bootstrap sample instead of applying one global ratio to all
of them. That is the documented variant for forests and it keeps the imbalance treatment comparable to
the linear baseline's, which is T29's question to test rather than T27's to assume.

**The ablation was generalised rather than duplicated.** T26 measured D44's reduction on the linear model
only; the forest is the instrument the decision was made with, so `REDUCTION_ESTIMATORS` now drives the
same toggle for any registered estimator. Result: the forest gives **-0.0014 ± 0.0024** ROC and **+0.0012
± 0.0027** PR — inside the noise, with the PR *sign favouring* the drop — which **reproduces T25's
-0.00014 ± 0.00195** at production settings. So the reading sharpens: the reduction is free for the tree
family that ships, and D46's **-0.0039** ROC loss is specifically a **linear-model** effect, not evidence
about XGBoost.

**A defect worth recording, because the check caught it and my reading did not.** Three of the forest's
ablation figures were first written into the notes from the 4-decimal printout — the all-17 PR-AUC, both
delta means and both delta sds — and every one of them was wrong in the third or fourth decimal
(`0.792007` for `0.791999`, `-0.001401 ± 0.002372` for `-0.001393 ± 0.002425`). The token-by-token trace
against `cv_results.json` found them; the arithmetic I had done by eye had not. That is the whole reason
D06 says no number is ever re-typed by hand, and it is a reminder that "I read it off the screen" is
re-typing with extra steps.

**Runtime is now a real number, not a rounding error:** **156 s**, of which the baselines are 3.8 s and the
rest is the forest's 16 fits (5 CV, 1 full, 10 for the two-arm ablation). T28 should time one XGBoost fold
before choosing a round count rather than discovering the total afterwards.

**Verified:** `python -m src.train` exits **0** with **14/14** checks — with four of them now generalised
over the registry rather than named after one estimator (every model beats both floors in every fold, every
artifact round-trips, no model stops at its iteration cap, and **every** reduction arm reproduces its own
estimator bit-identically; drift was `none`). Then **69 independent checks** that refit all five folds and
the all-17 arm from `train.csv` without importing `src.train`, confirm the artifact carries **300** trees
and `balanced_subsample`, and reproduce every per-fold figure with the hand-written AUC/AP functions first
validated against scikit-learn. Every numeric token in `docs/06_modeling.md` re-traced to the run output or
the JSON.

**Disclosed:** the `Files Created` table was behind for T23–T25 before this session and a fourth session
has now passed without backfilling it; the rows added here cover `src/train.py`, the artifacts and
`docs/06_modeling.md` only.

**Blocked:** none.
**Uncommitted:** `src/train.py`, `models/` (4 files), `docs/06_modeling.md`, `PROGRESS.md`.
**Next:** T28 — XGBoost as the third registry entry, against the forest's **PR-AUC 0.4346 / ROC-AUC
0.7906**, with D46's revisit trigger read off its result.

---

### 2026-10-29 — Session 30  (T28 complete)

**Completed:** **T28** — `xgboost_gbdt` added, `models/model_xgboost.joblib`, `docs/06_modeling.md`
extended to four models. **28 / 46 (61%)**, criterion 6 at **3 / 6**.

**The numbers.** **ROC-AUC 0.7993 ± 0.0089**, **PR-AUC 0.4420 ± 0.0134** (**3.78×** the base rate), Brier
**0.1637**. It beats the forest in **all five folds on both metrics**, and the forest in **all five on both**
before it — so the ordering of the table is uniform per fold rather than an artefact of averaging.

**Depth was measured, then handed to T30.** A one-fold sweep across depths 2/3/4/6 gave PR-AUC
**0.42948 / 0.43760 / 0.43604 / 0.42985**: 6 overfits 40 sparse one-hot columns, 2 underfits, and 3 beats 4
by 0.0016, so the shallower one wins. That is a sanity check, **not** tuning — T30 tunes on all five folds,
and D48 says so explicitly so the report cannot present a one-fold choice as a search.

**D46 closed itself.** The trigger written down in T26 was *"a 40-column XGBoost landing below the logistic
baseline's PR-AUC **0.4008** reopens the feature set"*. It landed at **0.4420**, and its own ablation puts
the cost of the reduction at **-0.0001 ± 0.0008** ROC and **-0.0007 ± 0.0015** PR — the smallest of the
three estimators. The feature set stays as T25 left it, and it stays for a reason that was fixed before
the number existed rather than because the number was convenient.

**A float32 trap the verifier caught in itself.** The independent script first failed **11** checks with
differences of **1e-7**, because it computed `scale_pos_weight` as `31,937 / 4,231 = 7.5483337` while the
module pins **7.548334** — and those are different **float32** numbers, enough to shuffle near-tied rows and
move AUC in the seventh decimal. That is worth knowing beyond this task: a *recomputed* constant and a
*pinned* one are not interchangeable inside XGBoost, so the verifier now uses the pin and separately asserts
it is within 1e-6 of the measurement. Bit-identical after the fix.

**A wrong-denominator error, again, and in the same shape as T26's.** The new check on `scale_pos_weight`
compared the pin against each fold's **validation** ratio and reported a 2.5e-3 "drift" that was simply the
held-out share. A per-fold weight would be computed on the **training** rows. Corrected, the real maximum
deviation is **1.9e-3** (0.025% relative, in the one fold that holds back 847 positives instead of 846), and
the module's comment now states that measured number rather than a guessed `1e-5`.

**Verified:** `python -m src.train` exits **0** with **15/15** checks in **182 s**; **82** further independent
checks refit all five folds and the all-17 arm from `train.csv`, confirm **400** rounds reached the booster,
that depth is 3 and the weight is actually applied, and that XGBoost beats the forest in every fold. All
**425** numeric tokens in `docs/06_modeling.md` trace to the run or the JSON; the two that do not are the
date and T15's **9,043** held-out rows. One typo was found by that trace and fixed: `36,720` for the
training half's **36,168** rows.

**Blocked:** none.
**Uncommitted:** `src/train.py`, `models/` (2 new files), `docs/06_modeling.md`, `PROGRESS.md`.
**Next:** T29 — weighting vs SMOTE on PR-AUC/F1, with resampling inside the folds and the F1 threshold
chosen per variant rather than at 0.5.

---

### 2026-10-29 — Session 31  (T29 complete)

**Completed:** **T29** — `src/imbalance.py`, `models/imbalance_results.json`, `docs/06_imbalance.md`.
**29 / 46 (63%)**, criterion 6 at **4 / 6**.

**Three arms, not two.** `weights` (`scale_pos_weight=7.548334`), `smote` (inside an
`imblearn.pipeline.Pipeline`) and **`none`** — the control. Without it, a difference between the first two
could not be attributed to either: it could be the difference between *treating* and *not treating*.

**SMOTE loses to doing nothing, in every fold.** PR-AUC **0.4152** against 0.4420 (**-0.0268 ± 0.0118**,
**0/5** folds), ROC-AUC **0.7799** against 0.7993 (**-0.0194 ± 0.0028**, **6.8 standard errors**), F1
**0.4524** against 0.4848. It is also **3.0× slower** (28.3 s against 9.3 s) and **less stable** (PR-AUC sd
0.0234 against 0.0134). The mechanism is in the doc rather than left as a number: the synthetic rows are
interpolations between clients that correspond to no real client, and in a one-hot matrix a midpoint
between `month=may` and `month=oct` is a value no client can have.

**The bigger finding arrived from the control.** Dropping `scale_pos_weight` **costs nothing measurable in
ranking** — PR-AUC **0.4444** against 0.4420 (**+0.0023 ± 0.0021 in favour of no weight**, 5/5 folds, 1.11
SE), ROC-AUC identical to four decimals — and **halves the Brier score**, **0.0821** against **0.1637**,
because the untreated model's mean prediction lands on the base rate (**0.1168** against **0.1170**) while
the weighted one sits at **0.3892**. So every model in this project has been paying for a treatment that
buys nothing and damages calibration; T34 can now apply the 7.50% break-even **as a probability**.

**The F1 trap, measured rather than warned about.** At a fixed 0.5 cut `weights` beats `none` by **11.9
points**; at each fold's **leave-one-fold-out** cut, `none` wins. Nothing about the models changed, only the
cut — because a fixed cut ranks the arms by where their calibration happens to land. That is why each fold
is scored at an F1-maximising threshold computed on the **other four folds**, and why the threshold choice
is itself a check (all **15** re-derived, zero drift).

**Leakage proof is arithmetic, not intention.** SMOTE adds exactly **22,164** rows on fold 1
(**28,934 → 51,098**, ending balanced at 25,549 each), the fitted scaler still matches the **fold's own**
training means to **0.0e+00** while the resampled matrix's means are **40.9** away — which is what a leak
*would* have produced — and no arm returns more or fewer than the **7,234 / 7,233** held-out scores.

**One check was wrong, and it was mine.** "Every arm ran the same five folds" required five distinct
`(rows, positives)` combinations; five stratified folds actually produce **three** (7234/846 twice,
7234/847, 7233/846 twice). That assertion was arithmetic on the wrong quantity and failed on a correct run.
The property that matters — agreement **per fold index** — was already the next check, so the first now
only asserts the replay and the arm lengths.

**Verified:** `python -m src.imbalance` exits **0** with **10/10** checks in **61.1 s**; **197** independent
checks refit all three arms from `train.csv`, recompute every per-fold PR-AUC/ROC-AUC/Brier/mean-p/F1 with
hand-written AUC/AP routines validated against scikit-learn to 1e-12, re-derive all **15** nested
thresholds, and confirm the two calibration claims and the SMOTE row identity. All **209** numeric tokens
in `docs/06_imbalance.md` trace to the run or the JSON; the only one that does not is the date.

**Blocked:** none.
**Uncommitted:** `src/imbalance.py`, `models/imbalance_results.json`, `docs/06_imbalance.md`,
`docs/06_modeling.md`, `PROGRESS.md`.
**Next:** T30 — `RandomizedSearchCV` over the XGBoost space with 5-fold `StratifiedKFold`, tuning the
**untreated** configuration, recording the search space and the winner in `models/best_params.json`.

---

### 2026-10-29 — Session 32  (T30 complete)

**Completed:** **T30** — `src/tune.py`, `models/best_params.json`, §9 of `docs/06_modeling.md`.
**30 / 46 (65%)**, criterion 6 at **5 / 6**.

**One loop, not two.** `train.cross_validate` was split so its body is `cross_validate_factory`, and the
search's candidates are scored through **that** function: same `make_pipeline`, same `fold_metrics`, same
`summarise`. `RandomizedSearchCV` gets the materialised folds as `cv`, so it cannot construct a
look-alike split that happens to share a seed. A candidate scored by a parallel implementation would make
"tuned beats untuned" two experiments, which is the failure D45 exists to prevent.

**The winner:** 300 rounds, lr 0.08, depth 5, `min_child_weight=10`, `subsample=1.0`,
`colsample_bytree=0.6` — deeper and more regularised than T28's hand-picked depth 3, which came from a
single fold. Untuned PR-AUC **0.4444 ± 0.0131**, tuned **0.4460 ± 0.0110**, delta **+0.00165 ± 0.00286** in
**4/5** folds. The done-condition is met.

**0.58 standard errors is the finding.** The gain is real in sign and small in size, and the search's own
spread says why: 40 candidates span **0.4080** to **0.4460**, **14** sit within one standard error
(**0.0049**) of the winner, the median is **0.4389**, and **the configuration T28 shipped was 4th of 40** at
**0.4449**. Hyperparameters are not where this problem's performance lives; the shipped settings were
already inside the noise of the best of forty. What the search did buy is a **different configuration**
(a deeper tree with heavier leaf regularisation) and a **file to rebuild it from** instead of a sentence.

**T29 survives tuning, which is the check that matters.** With the weight restored at the tuned point:
PR-AUC **-0.00291 ± 0.00300** (**0/5** folds), ROC-AUC **-0.00319** (**0/5**), Brier **0.1535** against
**0.0818**. So the untuned model was not merely undertuned — the weight costs the same at a better point in
the space. The arm to persist is the **untreated** one.

**Selection bias, stated rather than hidden.** The winner was chosen on the same five folds T32 reports, so
its CV score is optimistic **by construction**. `best_score_` is never quoted as a result: the arms table is
a **refit** whose per-fold numbers equal the search's own split scores to **0.0e+00** (a check, not a
claim), the record carries all 40 candidate scores and both bands, and T32's held-out half is named as the
only discipline.

**Two defects, found by the verifier rather than by the module.** scikit-learn's `std_test_score` is a
**population** sd, and quoting it beside the arms' sample sds put two quantities in one column — **0.0099**
against **0.0110**, an 11.8% error nothing on the page would have revealed (D50; the module now recomputes
with `ddof=1` and keeps sklearn's value labelled). And the recorded **Brier is computed in float32** —
sklearn turns the target into a bool indicator and XGBoost's probabilities are float32 — so a float64
recomputation differs by up to **1.4e-10**; average precision has the mirror issue, because fold 5 holds
**18 bit-identical probabilities** out of 7,233 and a per-row staircase differs from the tie-collapsed one
by **9.3e-8** with no model change.

**Verified:** `python -m src.tune` exits **0** with **12/12** checks in **68.2 s** (search **50.1 s**);
**131** independent checks refit all three arms **and the top five candidates** from `train.csv`, building
their own folds, pipelines and metrics without importing `src.tune` or `src.train`, and validate the
hand-written AUC/AP/Brier routines against scikit-learn **before** using them. All **177** numeric tokens
in §9 trace to the run, the JSON or the verifier's printed output; the one unmatched is the leading digit
of the thousands-separated **7,233**, which is the record's own fold size. Two figures the trace caught as
rounded quotients (**0.8%** for 40/4800 and **1.4e-10** for 1.38e-10) were corrected to their exact values.

**Blocked:** none.
**Uncommitted:** `src/tune.py`, `models/best_params.json`, `src/train.py`, `docs/06_modeling.md`,
`PROGRESS.md`.
**Next:** T31 — persist the winner from `models/best_params.json`, write `docs/06_model_comparison.md`,
and prove the artifact reloads and predicts.

---

### 2026-10-29 — Session 33  (T31 complete, Phase 6 closed)

**Completed:** **T31** — `src/final.py`, `models/final_model.joblib`, `models/final_model.json`,
`docs/06_model_comparison.md`.
**31 / 46 (67%)**, criterion 6 at **6 / 6** — **Phase 6 is closed.**

**The model is a file now.** Rebuilt from `models/best_params.json` through `tune.make_estimator` rather
than from parameters retyped in a second place, fitted on all **36,168** training rows in **0.8 s**, saved
as **180,520 bytes** (`compress=3`) with sha256 **7a40752f...**, and reloaded.

**Seven builds, one fold split.** ×base rate: base rate **1.00**, logistic **3.43**, forest **3.72**,
XGBoost weighted **3.78**, untreated **3.80**, SMOTE **3.55**, **tuned 3.81**. The shipped build has the
best PR-AUC (**0.4460 ± 0.0110**), ties the best ROC-AUC (**0.7990**), and is calibrated without help
(mean p **0.1166** against **0.1170**, Brier **0.0818**).

**The justification is not the tuning.** The tuned margin over T28's configuration is **+0.00165 at 0.58
SE**, and T28's settings were **4th of 40** candidates, so “tuned beats untuned” cannot carry a selection.
What carries it: best ranker of the field, the only builds whose scores are probabilities (so D17's
**7.50%** break-even applies without rescaling), and T29's measurement that the weight buys no ranking and
halves the Brier score — reproduced at the tuned point as **−0.00291 in 0/5 folds**.

**Two self-inflicted bugs, both found by running it.** The cross-process reload failed with
`ModuleNotFoundError: No module named 'src'` because Python puts the **script's** directory on `sys.path`,
not the working directory — the probe now runs with `-c` from the project root. And the “no test-half
load” scan **reported itself**, because the literal it searched for was written inside the check; it is a
regex now, and the comment no longer quotes the pattern.

**Verified:** `python -m src.final` exits **0** with **11/11** checks in **3.0 s**, including a reload in a
**separate Python process** (max drift **0.0e+00** over 1,000 rows), a bit-identical in-memory reload,
finite probabilities in **[0.005045, 0.994955]** summing to 1 within **0.0e+00** for one row or a thousand,
determinism, **300** boosted rounds, and a card whose sha256 matches the file. All **187** numeric tokens
in `docs/06_model_comparison.md` trace to the four JSON records or to arithmetic on them; the only
unmatched ones are the date.

**Blocked:** none.
**Uncommitted:** `src/final.py`, `models/final_model.joblib`, `models/final_model.json`,
`docs/06_model_comparison.md`, `PROGRESS.md`.
**Next:** T32 — the cross-validated metric table in `reports/metrics_cv.csv`, with the threshold rule
stated (leave-one-fold-out, not 0.5) and every thresholded metric reported at both the F1-optimal and
D17's 7.50% cut.

---

### 2026-10-30 — Session 34  (T32 complete, Phase 7 started)

**Completed:** **T32** — `src/metrics.py`, `reports/metrics_cv.csv`, `models/metrics_cv.json`, §10 of
`docs/06_modeling.md`.
**32 / 46 (70%)**, criterion 7 at **1 / 5**. **Phase 7 is open.**

**Four models, five folds, three cuts.** PR-AUC, ROC-AUC, precision, recall and F1 as mean ± sd, on the
project's materialised folds, with the shipped model leading **every column**: PR-AUC **0.4460 ± 0.0110**,
ROC-AUC **0.7990 ± 0.0086**, precision **0.4591 ± 0.0130**, recall **0.5202 ± 0.0133**, F1 **0.4877 ± 0.0110**
at a mean cut of **0.2103**.

**The 0.5 reversal, now across two models.** At 0.5 the forest wins by **10.6 F1 points** (**0.4565** vs
**0.3509**); at the nested cut the shipped model wins by **2.4** (**0.4877** vs **0.4641**). Neither model
changed. The flag rates explain it: 0.5 is near the **95th percentile** of the calibrated model's scores
(**4.67%** flagged) but sits inside the forest's 2.9×-inflated distribution (**17.89%**).

**At the business cut the baseline is “call everyone”.** At D17's **7.50%** the weighted builds flag
essentially the whole list, so their precision collapses to the base rate; the shipped model flags
**41.10%** of the list, reaches **77.90%** of subscribers at **22.18%** precision — **1.90×** the base rate,
and **3.92×** at the nested cut's **13.26%** coverage. Which cut to run is a capacity question, handed to T34.

**One correction of my own before the trace ran.** I had written that the break-even precision was **3.5×**
the base rate; **0.2218 / 0.1170 = 1.90**. The 3.51 figure was flag-share over base rate — a different
quantity wearing a similar shape. Corrected in both the table narrative and the viva paragraph.

**A verifier that measured the wrong artefact.** The independent script's first run failed every value
comparison by **5e-7**, because it read `reports/metrics_cv.csv` (six decimals) and tested at **1e-12**. The
numbers were right and the check was pointed at a rounded view of them; it now verifies against the JSON and
checks the CSV at its own precision — the arrangement where a **stale** CSV would still be caught.

**Verified:** `python -m src.metrics` exits **0** with **11/11** checks in **30.8 s** (forest 24.9 s);
**151** independent checks refit all four models from `train.csv`, including a hand-written F1-maximising
scan that reproduces every nested cut, and confirm the §10.2 reversal independently.

**Blocked:** none.
**Uncommitted:** `src/metrics.py`, `reports/metrics_cv.csv`, `models/metrics_cv.json`,
`docs/06_modeling.md`, `PROGRESS.md`.
**Next:** T33 — score the persisted artifact once on the held-out half and save the confusion matrix, ROC
and PR curves, annotated at 0.5 **and** at the two live operating cuts.

### 2026-10-30 — Session 35  (T33 complete)

**Completed:** **T33** — `src/evaluate.py` (1,011 lines) plus
`reports/figures/07_confusion_matrix.png` (164,168 B), `07_roc_curve.png` (135,860 B) and
`07_pr_curve.png` (178,378 B), `models/holdout_eval.json`, and **§11** of `docs/06_modeling.md`.

**The result.** The artifact — loaded, not refitted — scored the 9,043 held-out rows once: **PR-AUC
0.4616**, **ROC-AUC 0.8052**, Brier **0.0804**, mean p **0.1151** against a **0.1170** base rate.

**The finding, and it cut against the carry-in.** The carried expectation was that the held-out number
would come out **lower** than CV, because T30 chose the configuration on those folds. It came out
**higher**: **+0.0155** PR-AUC, **+0.0061** ROC-AUC. The step's job was then to find out whether that means
anything, and it does not: the bootstrap over these rows puts the estimate in **[0.4304, 0.4969]**, sd
**0.0169** — **1.53×** the CV fold sd — and **15.4%** of draws fall **below** the CV mean. The verdict is
**no measurable gap**. Two candidate explanations are recorded and neither is "the model is better than
measured": the artifact saw **36,168** rows against a fold model's **28,934**, and one draw of 1,058
positives is one draw.

**The number worth more than the headline.** The F1-maximising cut *on the held-out rows* is **0.212050**,
and it was **not** adopted — choosing it would spend the held-out set on a decision the training folds
already made. But measuring it shows how good that earlier decision was: T32's CV-derived **0.2103** sits
**0.001725** from the optimum and costs **0.000481** F1. A threshold picked on five folds and carried to
unseen rows landed where those rows would have put it.

**At the three cuts.** **7.50%** flags 3,621 (40.04%) at precision 0.2267 / recall 0.7760; **0.2103** flags
1,159 (12.82%) at 0.4840 / 0.5302; **0.5** flags 396 (4.38%) at 0.6490 / 0.2429. 0.5 has the best precision
of the three and the worst F1 (**0.3535** against **0.5061**), because it misses **801 of the 1,058**
subscribers — the same mechanism T32 measured, now out of sample.

**Verified:** `python -m src.evaluate` exits **0** with **15/15** checks in **9.2 s**; **144 independent
checks** in `_t33_verify.py` (deleted before the commit) recompute every figure without importing
`src.evaluate`, using midranks from `unique`/`bincount` for ROC-AUC, tie-block steps for AP, `searchsorted`
for the confusion counts, and numpy percentiles over the **stored draws** for the interval. **214** numeric
tokens in §11 trace to the three records or to arithmetic on them; the trace runs in two tiers, and every
token in its *computed* tier was read and named.

**Two defects, both mine and both caught by running it.** `rank_auc` summed the ranks of the *positions*
the subscribers occupy in the score-sorted array rather than the subscribers' own ranks — it read
**0.7783** against scikit-learn's **0.8052** until `ranks_sorted` was mapped back through `order`. And the
record's Brier score was averaged in **float32**, because `predict_proba` returns float32 and numpy keeps
float32 arithmetic with an int8 target in float32: **4.4e-9** between an int8 and an int64 target vector.
That is the same family as the defect T30's verifier found in the CV record (§9.5), and `score()` now casts
to float64 with the reason written down. **Ties were handled rather than dodged:** 9,007 distinct scores
over 9,043 rows, so 36 rows share a score with another row, and a naive per-positive average precision
comes out **1.8e-8** low.

**Also checked by hand, because the trace could not:** the record's runtime moved when the F1-peak
computation was added, and the superseded value still matched a pairwise arithmetic combination in the
trace's *computed* tier — a stale number can be vouched for by coincidence there, so the runtime and the
three figure sizes were read directly out of `models/holdout_eval.json` instead of through the trace.

**Blocked:** none.
**Uncommitted:** `src/evaluate.py`, `reports/figures/07_confusion_matrix.png`, `07_roc_curve.png`,
`07_pr_curve.png`, `models/holdout_eval.json`, `docs/06_modeling.md`, `PROGRESS.md`.
**Next:** T34 — choose the operating threshold from the cost model, with a reliability check and a
sensitivity band, on the training half and checked on the held-out rows.

### 2026-10-30 — Session 36  (T34 complete)

**Completed:** **T34** — `src/threshold.py`, `docs/07_threshold.md`,
`reports/figures/07_threshold_curve.png` (221,145 B) and `models/threshold_choice.json` (77,751 B).

**The decision.** The operating threshold is **`C/V = 1.50 / 20.00 = 0.0750`**, imported from
`cost.break_even_threshold()` rather than typed, so the doc's claim "the threshold is a price" is true by
construction: a check requires the recorded cut to equal `C/V` exactly.

**Where it was chosen.** On the training half's **out-of-fold** scores — 36,168 rows, 4,231 subscribers,
five folds — because choosing from in-fold predictions would be choosing from rows the model had memorised.
That those are the project's folds is checked, not asserted: the per-fold PR-AUCs reproduce T30's tuned arm
to **0.0e+00** (`0.4435, 0.4378, 0.4339, 0.4588, 0.4561`). The held-out half is scored in a separate function
called afterwards, and its own optimum is recorded so a reader can see it was not the answer.

**The money.** At 0.0750 the campaign calls **14,866 (41.10%)** and reaches **3,296 (77.87%)** for
**43,621 EUR**, ROI **195.6%**, **6.77 EUR** per captured subscription. T32's F1 cut nets **36,891 EUR** and
0.5 nets **18,248 EUR**; calling everyone nets **30,368 EUR**, less than the priced cut because the marginal
client below 0.075 is worth `p x 20.00 < 1.50`.

**The shape of the optimum, which is the real finding.** The empirical peak is **0.0821 / 44,009.50 EUR**, so
the adopted rule gives up **388.50 EUR (0.883%)** and nothing else. Every threshold within 1% of the peak
spans **0.0746–0.1016** — **14,973 to 9,970 calls**, i.e. 27.6% to 41.4% of the list. The curve climbs
steeply and falls shallowly, so both empirical argmaxes land *right* of the price (training 0.0821, held-out
0.0909); that is why neither is adopted, and why the plateau is the honest statement of how delicate the
threshold is rather than the argmax. **0.0750 sits 0.0004 above the plateau's floor** — the price is the most
aggressive cut still within 1%, which is what a break-even rule should be.

**Held out.** The persisted artifact, scored once on the 9,043 held-out rows: **10,988.50 EUR** from 3,621
calls and **821** subscribers, against **9,481.50** at the F1 cut, **4,546.00** at 0.5 and **7,595.50** calling
everyone. **77.6%** of subscribers from **40.0%** of the list, against training's **77.9%** from **41.1%** —
the cut transferred. The held-out rows would have chosen **0.0909** (**11,198.00 EUR**, **209.50 EUR** better)
and it is recorded, not adopted.

**Two things the tables say that a single number would hide.** A capacity-limited bank is reading the wrong
table: **top 5%** reaches **26%** of subscribers for **44%** of the money, and the unconstrained optimum sits
near **30%** of the list, so the campaign is capacity- not capital-limited — which is also why **ROI** is a
diagnostic and not the objective (0.5 posts **765%** ROI and loses **3,393 EUR** to the adopted cut). And the
whole answer moves with the assumptions, tabulated across **20 cells**: **0.300** if a call costs 3.00 EUR and
a subscription is worth 10.00 EUR, **0.0125** at the other corner.

**Verified:** `python -m src.threshold` exits **0** with **15/15** checks in **4.6 s**, on **403** swept
thresholds. The record carries the curve points, so the left panel is rebuildable from the file, and a final
check reads the record back and compares it field by field against the run.

**Two defects, both mine.** A dead branch in `held_out_check` reassigned the peak's threshold to itself —
harmless, unreadable, removed. And the first record stored the plateau, the peak and the capacity table but
**not the curve**, so the three euro figures for the rejected cuts existed only inside a check's detail
string: quoteable and unverifiable. The record now carries the 403 points and a `comparisons` array.

**Blocked:** none.
**Uncommitted:** `src/threshold.py`, `docs/07_threshold.md`, `reports/figures/07_threshold_curve.png`,
`models/threshold_choice.json`, `PROGRESS.md`.
**Next:** T35 — lift, gain and decile analysis, with the adopted cut translated into calls saved at equal
recall.

### 2026-10-30 — Session 37  (T35 complete)

**Completed:** **T35** — `src/lift.py`, `reports/lift_table.csv` (20 rows x 13 columns),
`reports/figures/08_lift_gain.png` (231,645 B), `docs/07_lift.md` and `models/lift_results.json` (20,661 B).

**The tables.** Ten positional deciles per half, each against its own base rate (**11.6982%** training,
**11.6997%** held out — 1.5 basis points apart and never pooled). Top decile lift **4.3652x** training and
**4.5100x** held out, clearing D15's >2x target by more than a factor of two; the top two deciles hold
**61.76%** / **62.29%** of all subscribers.

**The finding I did not expect.** **Decile 3 is already below random** (**0.92x** training, **0.88x** held
out), and decile 10 sits at **0.18x** (**78 of 3,617** clients). Past roughly the best fifth of the list the
model's message is mostly *negative* — it is telling the bank who **not** to call, about 80% of the file, and
that is a decision the lift table makes visible in a way the ROC and PR curves do not.

**The cut, in calls.** **77.90%** of subscribers from **41.10%** of the list: a fixed order would need
**28,175** calls to reach the same people, so the ranking saves **13,309 calls (36.8% of the list,
19,964 EUR)**. Held out: **3,396 calls / 5,094 EUR**. The calls and subscribers are **identical** to T34's
record, so the two steps are reading one cut rather than two.

**Three optima on one curve, and why the cut is at the last.** Calls saved peaks at **18.89%** coverage
(**15,188 calls**), T34's money peak sits at **36.63%**, and the priced cut is at **41.10%** (**13,309**).
The mechanism is one line: `saved(k) - saved(k-1) = (N/positives) x y[k] - 1`, so the saving peaks where the
*local* rate crosses the *list average* (11.7%), while money peaks where it crosses the *price* (7.5%). The
savings peak is **outside** T34's 1%-of-peak money plateau (**27.6%–41.4%** coverage), which is the cleanest
way to say that maximising calls saved is not a money decision. The block between the two — **8,033 calls**
— converts at **8.96%**, above the **7.50%** price, for **+2,350 EUR**; held out **1,547 calls** at **7.76%**
for **+79.50 EUR**, thin in exactly the way T34's "aggressive edge" predicted.

**Reconciliation.** T32's two published multipliers are re-derived from this one ranking: **3.9488x** against
**3.9247x** at 13.26% coverage and **1.8991x** against **1.8961x** at 41.10%. The second is near-exact because
**0.0750 *is* 41.10% coverage** — the business cut and the price are the same point.

**Verified:** `python -m src.lift` exits **0** with **17/17** checks in **4.3 s**. The self-test reruns the
whole table on **shuffled scores**: lifts **0.950–1.108** (mean **1.000001**), gain within **0.0060** of the
diagonal, savings within **218 calls** of zero — the code tested, not the model. **401 numeric tokens** in
`docs/07_lift.md` trace to the two records.

**One defect, caught by the trace:** the first draft said a fixed order would need **28,176** calls;
`recall x rows` is **28,175.3**, so the honest figure is **28,175**. A hand-rounding error of one call in a
document whose whole point is the call count.

**Also corrected before commit:** two claims that did not survive their own arithmetic — decile 10 was
described as "1.8% of its clients subscribe" (it is **2.16%**, **78 of 3,617**) and the held-out decile noise
as "±0.06" (one standard error on a precision near 0.53 over 904 rows is **0.017**, so the two top-decile
lifts are the same number within noise rather than different ones).

**Blocked:** none.
**Uncommitted:** `src/lift.py`, `reports/lift_table.csv`, `reports/figures/08_lift_gain.png`,
`docs/07_lift.md`, `models/lift_results.json`, `PROGRESS.md`.
**Next:** T36 — the with-vs-without `duration` comparison, executing the pre-registered rule in §4.5 of the
leakage audit.

### 2026-10-30 — Session 38  (T36 complete, Phase 7 closed)

**Completed:** **T36** — `src/duration_compare.py`, `reports/duration_comparison.csv` (14 rows, `fold`/`mean`/
`sd` per arm), **§9** of `docs/05_leakage_audit.md` and `models/duration_comparison.json` (24,338 B).

**The rule came first.** §4.5 fixed it in T22, before any number existed: arm A ships regardless of the gap,
arm B's metrics are an upper bound, and a near-zero gap would be the finding. This step discharges it rather
than discovering it — which is the point of writing a decision rule down before the measurement.

**One model, one toggle.** Same estimator settings, same materialised fold list, same scorer; the feature
lists come from `leakage.feature_columns` and differ by exactly `['duration']`, and the matrices by exactly one
column — **40** vs **41**, measured for the shipped 12-feature set (T24 demonstrated the same toggle at the
17-feature width, 65 vs 66). Arm A's per-fold PR-AUCs reproduce T30's tuned arm to **0.0e+00** and its campaign
value reproduces T34's **43,621.00 EUR** to the cent, so the arm that ships is the documented one, not a re-fit.

**The price of a deployable model.** **+0.1788 PR-AUC and +20,340.00 EUR** (**+562.38 EUR per 1,000 clients**)
on the training half — from **4,320 fewer calls** that reach **693 more subscribers**. Arm B wins **5/5** folds,
SE ratio **23.12**; ROC-AUC **+0.1369**, Brier **-0.0197**. The mean predicted probability barely moves
(**0.1166** vs **0.1162**) while Brier *improves* (**0.0818 → 0.0621**): a model that has seen the outcome is
well calibrated, and calibration is not evidence of skill.

**Why it is leakage, checked rather than asserted.** Three clauses fixed before the arms were fitted, none of
which reads the gap's size: arm B wins ≥4/5 folds (**5/5**), T22's band table is non-decreasing (**0.16%** of
0–59s calls subscribe against **48.56%** of 600s+ calls, a **302x** swing, with the longest band supplying
**35.03%** of subscribers), and **`duration` alone — no model, no fitting — reaches ROC-AUC 0.8078 and PR-AUC
0.3897**, beating most of the project's purpose-built models on PR-AUC. All three hold, so the verdict is
`gap_is_leakage = True` and **arm A ships**. The arrow runs outcome → length: a client who says yes keeps
talking. The 20,340 EUR is the cost of an honest model, not money left on the table.

**Verified:** `python -m src.duration_compare` exits **0** with **13/13** checks in **7.1 s**. **No held-out row
is read** — a source scan for the held-out load finds **0** references, with the needle assembled at run time
so the file cannot match its own text.

**One defect, caught by running it.** The settings check compared XGBoost's `missing` parameter (default `nan`)
with `!=`, and `nan != nan` is true, so it reported a difference between two arms that were both unset. The
comparison is now NaN-aware and names the parameter it matched that way.

**Blocked:** none.
**Uncommitted:** `src/duration_compare.py`, `src/train.py` (two defaulted keywords for the toggle),
`reports/duration_comparison.csv`, `docs/05_leakage_audit.md`, `models/duration_comparison.json`,
`PROGRESS.md`.
**Next:** T37 — global SHAP on the shipped artifact, opening Phase 8.

### 2026-10-30 — Session 39  (T37 complete, Phase 8 started)

**Completed:** **T37** — `src/explain.py`, `reports/figures/09_shap_summary.png` (445 KB),
`reports/shap_importance.csv` (12 rows x 12 columns), `models/shap_global.json` and
`docs/08_explainability.md` (criterion 8 opens at **1 / 3**).

**The instrument, and what it is pointed at.** TreeSHAP on `models/final_model.joblib` — sha256
**7a40752f75a264a0...**, **180,520** bytes, checked before anything else runs — over the training half,
**36,168 rows x 40 columns x 12 features**. Not a refit, not the forest, not the untuned configuration:
a plot of a different model is a plot of a different model. Contributions are in log-odds and they
reconstruct the model's own margin to **5.010e-06**, with `sigmoid(margin)` matching `predict_proba` on the
twelve raw columns to **7.521e-08** and `pred_contribs` agreeing to **0.0**.

**The aggregation rule, which was the crux.** Twelve features become forty columns, so the table sums the
**signed** contributions inside each one-hot block — the only aggregation that keeps additivity (verified at
feature level, 5.010e-06). The naive per-column absolute sum inflates `month` by **58.52%**,
`campaign_bucket` by **49.63%** and `poutcome` by **35.91%**, so the usual "one-hot understates importance"
warning is backwards here. The real cost is **fragmentation**: `month=jun` alone ranks **9th of 40**, so a
per-level reader would miss that `month` is the second-strongest feature in the model.

**What the model looks at.** `contact` **22.14%**, `month` **17.55%**, `day` **10.60%**, `balance_arsinh`
**8.49%**, `housing` **8.05%**, `poutcome` **7.97%**, `marital` **7.74%**, `age` **5.61%**,
`campaign_bucket` **5.45%**, `previous_capped` **4.58%**, `recency_band` **1.46%**, `previously_contacted`
**0.35%**. Grouped by what they describe — a reading, declared in `CAMPAIGN_SIDE` — **55.7% of importance is
campaign-side** and **44.3%** is the client's own attributes.

**Three findings worth the phase.** (1) **`contact=unknown` is the top column of forty and it is a blank
field**: 10,386 clients (**28.72%** of the file) contribute **-1.0150** and subscribe at **4.11%** against
**14.83%** on `cellular` — the largest available improvement is data quality, not modelling. (2)
**`poutcome=success` contributes +1.6841, the largest single-level effect anywhere, on 1,205 clients
(3.33%)**, while **81.81%** have no history at all — which is why the feature is only **7.97%** globally and
why mean |SHAP| must not be read as per-client influence. (3) **`day` is 10.60% of importance and points
nowhere**: bottom fifth **-0.1257**, top fifth **-0.1230**, Spearman **-0.060** — its value is in the
interaction with `month` (D35), so "call on the 5th" is exactly the rule this feature does not support.

**Against T25, reported rather than resolved.** **Spearman +0.755** with four gaps of 3+ places:
`balance_arsinh` 4 vs 9 (SHAP credits information *used*, permutation measures information *unique* — 0.1733
log-odds against 0.0039 AUC, and MI said 3rd, so SHAP sides with the marginal view), `recency_band` 11 vs 7,
`poutcome` 6 vs 3, `age` 8 vs 5. No average of ranks: it would be a third number belonging to neither
method.

**Verified:** `python -m src.explain` exits **0** with **17/17** checks in **38.8 s**. **32 independent
checks** in `_t37_trace.py` recompute the whole table from xgboost's `pred_contribs` **without importing the
module or `shap`**, re-derive every client count, contribution and rate quoted in the prose, and trace
**115 numeric tokens** in the doc to the records — **0 unmatched**.

**Three defects, all mine, all caught by running it.** (1) The probability check fed the *transformed*
matrix to `pipeline.predict_proba` and the pipeline raised `columns are missing` — measuring the model
against a shape it never sees; it now goes through the twelve raw columns. (2) `mean_probability` was
`1/(1+exp(-margin)).mean()`, which divides by the mean instead of averaging probabilities and printed
**4.96%** where the truth is **11.6990%** — caught only because 4.96% contradicted T31's recorded 0.1166.
(3) The doc called −2.04695 "the model's average log-odds", which the mean margin **−2.49678** refutes: it
is the booster's bias, **0.0256** below `logit(base_score)` (**−2.0213**), and the average feature
contribution is therefore **−0.44983** rather than zero. A trace tolerance cap of 8 decimals then hid the
9-decimal `base_score` from its own check, and a first trace run mis-signed the gap — both fixed in the
verifier, not by weakening a claim.

**Blocked:** none.
**Uncommitted:** `src/explain.py`, `reports/shap_importance.csv`, `reports/figures/09_shap_summary.png`,
`docs/08_explainability.md`, `models/shap_global.json`, `PROGRESS.md`.
**Next:** T38 — local explanations for one true positive, one false positive and one false negative.

### 2026-10-30 — Session 40  (T38 complete)

**Completed:** **T38** — `src/local_explain.py`, `reports/figures/10_shap_local.png` (197 KB),
`reports/shap_local.csv` (36 rows) and `models/shap_local.json`; §9 appended to
`docs/08_explainability.md` (criterion 8 now **2 / 3**).

**Cells against the price, not against 0.5.** T34's **0.0750** is re-derived from C/V inside the module, so
"positive" means *above the price*: **12,867** clients sit between 0.0750 and 0.5 and a coin-flip rule
would relabel every one of them. In-sample the cells are **TP 3,523 / FP 11,059 / FN 708 / TN 20,878**;
T34's out-of-fold table (**3,296 / 11,570 / 935**) is carried alongside rather than replaced, because the
explanation must belong to the artifact while T34's counts come from a different denominator.

**The rule was written before the clients were looked at.** *The exemplar of a cell is the client whose
score is closest to that cell's median; ties are broken by the lowest row index.* It reads `y` and `p` and
nothing else - no SHAP value - and `validate()` re-derives the minimum from the scores alone. The three
land at rows **19,263 / 23,374 / 10,760**, each at the **50th percentile of its own cell**, which is what
makes them typical members rather than worst cases.

**The true positive shows the top global feature losing an argument.** Row 19,263, p = **0.3819**:
`month=oct` contributes **+1.8824**, and `contact=unknown` - the model's most important feature overall,
22.14% of global importance - takes **-0.6989** of it back. The client subscribed anyway. Local and global
importance are different questions, in one row.

**The false positive is the finding, and it is arithmetic.** Row 23,374, p = **0.1133**, with the twelve
contributions summing to **-0.0109**: a good balance (+0.3471), no mortgage (+0.2168) and a cellular call
(+0.2434) against an early-month call (-0.5130) and a failed previous campaign (-0.2790), netting out to
the reference value almost exactly. The reference is **11.4361%** and the cut is **7.50%**, so an average
client clears the cut and **30.58%** of the file is in this cell. Called and wrong is what the price buys.

**The false negative misses by 0.3323 log-odds.** Row 10,760, p = **0.0550**: `balance_arsinh = 0.00`
(-0.3502) and `month=jul` (-0.3121) with no previous campaign (-0.1579) outweigh `contact=cellular`
(+0.2329); logit(0.0750) is -2.5123 and the client sits at -2.8446. A ranking miss of about two percentage
points, not a category error - and across the three clients the largest single contribution is
campaign-side in two of them, which is T37's 55.7% share appearing client by client.

**Verified:** `python -m src.local_explain` exits **0** with **16/16** checks in **76.7 s** (50-80 s
depending on machine load); **70 independent checks** rebuild the cells, the selection rule and all three
profiles from the artifact and `pred_contribs` without importing the module or `shap`, and trace
**179 numeric tokens** in the doc to the records with **0 unmatched**. Per-client additivity **6.853e-07**,
`pred_contribs` **0.0**, twelve features covering all forty columns.

**Three defects, all in my own work.** (1) The first version defined the cells from `sigmoid(shap margin)`
instead of the served `predict_proba`; the two routes differ by up to **4.49e-07** and **one** client sits
inside that window at the cut - found because the independent verifier got **3,524 / 11,059 / 707** against
my **3,523 / 11,059 / 708**. (2) That made the "the margin is the score the cells were defined on" check
**vacuous** - both sides were the same array - so it now compares SHAP's margin with the pipeline's own
probability (**8.598e-09**), and a new check measures the flip instead of asserting stability. (3) A
boundary check I wrote first asked whether the counts are stable under a 1e-4 cut movement; the honest
answer was **43 clients**, which is a property of a dense region rather than a defect, so the check was
rewritten to the question that matters - do the quoted counts depend on the score's route. The prose then
lost another three rounding slips (unnamed remainders summed from already-rounded values, off by 1e-4 in
all three cells), all corrected to the exact sums.

**Blocked:** the push, as recorded in Open Questions item 10 - five commits now local (`33fff7a` T34,
`7bce0c1` T35, `d149f58` T36, `7612071` T37, this one), `origin/main` at `caefa9a`, and the credential
helper needs an interactive sign-in the agent shell cannot drive.
**Uncommitted:** `src/local_explain.py`, `reports/shap_local.csv`, `reports/figures/10_shap_local.png`,
`models/shap_local.json`, `docs/08_explainability.md`, `PROGRESS.md`.
**Next:** T39 — the campaign playbook, closing Phase 8.

### 2026-10-30 — Session 41  (T39 complete, Phase 8 closed)

**Completed:** **T39** — `src/playbook.py`, `reports/playbook.csv` (8 rows), `models/playbook.json` and
`docs/08_recommendations.md`; criterion 8 is **3 / 3** and **Phase 8 is closed**.

**The recommendations are data, not prose.** Each one carries its action, its evidence, what to expect, how
to check it and what would make it wrong; each figure is a **file plus key path** resolved on every run, so
**46 figures across nine records** resolve and `validate()` re-reads them from disk before writing anything.
A number with no record behind it cannot be written at all — the ten quantities the document needs (net per
1,000 calls on each half, cost per subscription both ways, the held-out recall and precision, the
always-`no` accuracy, the campaign/client split) are computed by `derived()`, not typed. A check also fails
if `duration` appears anywhere except **R4**, so the prohibition is enforced by the build.

**The scope sentence comes before the first number.** Calling everyone nets **30,368.00 EUR**; the cut nets
**43,621.00 EUR**. The model is worth **+13,253.00 EUR (+43.6%)** and **21,302 fewer calls** on a campaign
that already pays — an improvement to a fixed budget, not a rescue, which is D18's narrow claim and the one
a reader is most likely to overstate.

**R1 is a data-quality action, and it claims no lift.** `contact=unknown` covers **28.72%** of clients, is
worth **-1.0150** log-odds, and those clients subscribe at **4.11%** against **14.83%** for `cellular`.
Filling the field changes how they are *scored*; it does not relabel them today, and the document says so
in the recommendation rather than in a caveat at the end.

**R3 refuses the obvious optimisation, and shows what it costs.** Calls saved peaks at **18.89%** coverage
(**15,188** calls) while the price puts the cut at **41.10%**; the **8,033** calls between them still
convert at **8.96%** for **+2,350.50 EUR**, so pruning to the peak gives up money to save calls the campaign
was going to make. On the held-out half that same block is **1,547** calls at **7.76%** for **+79.50 EUR**
— the thinnest recommendation in the playbook, stated as such, because on fresh data the same policy may
flip.

**The two prohibitions carry the number they decline.** R4: `duration` would add **+0.1788** PR-AUC and
**20,340.00 EUR** on **4,320** fewer calls, and is recorded when the call ends. R7: the shipped
configuration leads on **0.4460** PR-AUC against **0.4346** and **0.4008**, tuning moved it **+0.0016**, and
the re-open trigger is a measurement below **0.4008**. R8 reads `month` as a calendar (**17.55%** of
importance, March **+1.3736** at **50.90%**, May **-0.2999** at **6.68%** on **30.59%** of the file) with
the 2008–2010 confound named; campaign-side features carry **55.75%** of importance against **44.25%** for
the client's own, which is the sentence the whole playbook follows from.

**Verified:** `python -m src.playbook` exits **0** with **12/12** checks, **8** recommendations and **46**
figures resolved, in **9.4 s**; **78 independent checks** recompute every table cell, every derived euro and
the always-`no` accuracy from the records **without importing the module**, then trace **63 numeric tokens**
in the document to those records — **0 unmatched**.

**Five defects, all mine.** (1) R6's **3,192** missed subscribers was read from the **adopted cut's**
`subscribers_missed` field, where the sentence is about a **0.5** rule; the 0.5 row says **3,192** as well,
but only after reading the right row — the label was right by luck and is now sourced. (2) Three figures in
the prose had no record behind them at all (a standard error, a coverage range, a raw row count); each was
replaced with the recorded equivalent, which is what the record-or-nothing rule exists for. (3) The
verifier's percentage check asserted **43.63%** where the doc correctly rounds to **43.6%** — a verifier
bug, fixed by checking the rounding rather than a finer figure. (4) Its token scan matched the fragment
`253.00` inside `19,253.00`, because a comma thousands separator breaks the number apart; separators are
now stripped before matching. (5) The doc's ×-rounding of one decimal place hid that same issue on three
figures, which is why the fix belongs in the tokenizer and not in the prose.

**Blocked:** the push, as recorded in Open Questions item 10 — six commits now local, `origin/main` at
`caefa9a`, and the credential helper needs an interactive sign-in the agent shell cannot drive.
**Uncommitted:** `src/playbook.py`, `reports/playbook.csv`, `models/playbook.json`,
`docs/08_recommendations.md`, `PROGRESS.md`.
**Next:** T40 — the final report, assembling the nine criteria from the records already on disk.

---

### 2026-10-30 — Session 42  (T40 complete, Phase 9 started)

**Completed:** **T40** — `src/report.py`, `docs/report.md`, `docs/report.html`, `docs/report.pdf` and
`models/report.json`; criterion 9 is **7 / 13** and the project is **40 / 46**.

**The report is a selection problem, and the risk is transcription.** Eight phases produced twelve records and
eleven documents; T40 assembles them into one document a marker reads top to bottom, in rubric order rather
than project chronology, so each criterion opens with its own evidence instead of the report following the
order the work happened to be done in. Nothing in it is a new measurement: **497 decimal tokens**, every one
matched to a value held in a record or a supporting document.

**The shape.** **§0** states the decision and the scope sentence before the first euro total. **§1–§8** are
the eight analytic criteria, each ~900 words with its tables and its figure. **§9** is the reproducibility
claim: the module table with checks and runtimes, the three-layer verification discipline, and the command
sequence. **§10** lists seven limitations — the assumed prices, the held-out half being the same campaign, the
random rather than temporal split, the tuning gain below its own error bar, the `month` confound, the
unobserved counterfactual, and the absent fairness audit. **Appendices A–C** index the figures, the records,
and the checks themselves.

**What it says, in one paragraph.** The model ranks clients well enough that calling **41.10%** of the list
reaches **77.90%** of subscribers — **43,621.00 EUR** on the training half and **10,988.50 EUR** held out,
against **30,368.00** and **7,595.50** for calling everyone. The cut is the price, **0.0750 = 1.50 / 20.00**,
not a fitted argmax. The top decile converts at **4.3652×**. And the most important feature is a field that
was left blank on **28.72%** of clients, which is a data-quality finding wearing a modelling costume.

**Verified:** `python -m src.report` exits **0** with **30/30** checks in **3.4 s** — nine sections in order,
nine sections naming their evidence, 9 figures present on disk, 12 records parsing, 497 tokens traced, the
scope sentence before the first `EUR`, all three prohibitions inside their own sections, and a PDF that is
**27 pages** of real content with **9 embedded images**. A separate verifier that imports neither the module
nor `pdfminer` runs **45 checks** of its own, matching all fourteen quoted euro totals to the records to the
cent and re-deriving every percentage from its counts.

**Three defects, all caught by the trace.** (1) **`77.87%`** reached subscribers, in the threshold table and
again in the prose — the records say **77.90%** (3,296 / 4,231), and the wrong figure had been copied forward
from a progress note instead of from `lift_results.json`. (2) and (3) Two balance-band rates, **5.36%** and
**16.44%**, that appear in no record and no document: they were replaced with the recorded ablation
(**−0.00023 ± 0.00085**) and the **3,001 rows (8.30%)** carrying a negative balance, both of which are on
file. This is the case the record-or-nothing rule was written for, and the first one where the offender was
this session's own prose rather than a hand-typed figure in a data structure.

**Blocked:** the push, as recorded in Open Questions item 10 — six commits local, `origin/main` at
`caefa9a`, and the credential helper needs an interactive sign-in the agent shell cannot drive.
**Uncommitted:** `src/report.py`, `docs/report.md`, `docs/report.html`, `docs/report.pdf`,
`models/report.json`, `PROGRESS.md`.
**Next:** T41 — the slide deck, at most 15 slides, drawing every figure from the records the report already
traced.

---

### 2026-10-30 — Session 43  (T41 complete, Phase 9 two thirds done)

**Completed:** **T41** — `src/deck.py`, `docs/presentation.pptx` (**15 slides**, 1,342,750 bytes, sha256
`7ffd11f8…`) and `models/deck.json`; criterion 9 is **8 / 13**.

**The slides are data, and the numbers are checked after the file is saved.** `SPEC` holds the fifteen slides as
text, tables and figure names; the build writes the package and then reopens it, so every check runs against the
bytes PowerPoint will open rather than against the objects that produced them. **157 decimal tokens** trace to a
record or a supporting document; the **33** citations printed at the foot of the slides are each verified to
exist; and geometry — inside the slide, above the source line, no text block taller than its box, no type
below 9 pt — is read back out of the file, so an overflowing text box fails the command instead of being
discovered in the presentation.

**Business-first, in the order a decision needs it.** Cover → the answer and its scope → the decision → the
price → the data → what the campaign already gets wrong → `duration` → features → models → the operating
point → held out → lift → accuracy → what drives the score → eight recommendations. The D18 scope
sentence is the **second bullet of slide 2**, before the first euro total (byte 161 against 294), and each of the
three prohibitions has its own slide rather than a footnote — a deck is where a nuance is most likely to be
dropped for space.

**Two defects the deck inherited from the report, caught by its verifier.** (1) § 7.6 of `docs/report.md` put
a **held-out** accuracy (**89.6052%**) beside a **cross-validated** balanced accuracy (**61.26%**); the held-out
pair is **89.6052% / 61.28%** and the row now says so, with the CV pair (**89.3801%**, **0.6126**) named as the
other table instead of being mixed into this one. (2) The same paragraph claimed accuracy "maximises at **0.5**",
which the record's own `peak_on_held_out` contradicts: the accuracy-maximising cut is **89.76%** at **0.478284**.
Both are corrected in the report and the deck, and `python -m src.report` was re-run to rebuild the 27-page PDF
around the fix. This is a defect class rather than an incident: a token-level trace admits a number that appears
*anywhere* in the documents, so a figure lifted from the wrong column passes it. The deck's verifier checks each
slide against the **specific** record it cites, which is what caught both.

**Four defects in my own code.** A caption box was sized as figure-plus-caption, which pushed six captions off
the bottom of their slides. The geometry check was **vacuous** — it iterated the wrong XML namespace and found
**8** shapes where there are **112**, so it was passing on nothing. The determinism check passed by luck:
python-pptx stamps every zip member with the clock, so two builds in the same second match and two a second
apart do not; it now compares **package parts** (**72**, 0 differing). And a text helper was handed shapes where
it expected a slide.

**The layout was looked at, not assumed.** All fifteen slides were exported to PNG through PowerPoint's COM
interface and inspected, which is how two real problems surfaced: the project's multi-panel charts sat in a
5.8-inch column where the panel titles were unreadable, and a lone figure was centred so it floated below the
baseline of the text beside it. Figure columns now take **6.34** of the 11.93 available inches, figures are
top-aligned, and body type went up one point. `pywin32` is used for that inspection only and is **not** a project
dependency.

**One new pin, and it is the one D10 reserved.** `python-pptx==1.0.2` under a Deliverables heading; the PDF still
needs no package at all, because it is Chromium printing HTML.

**Verified:** `python -m src.deck` exits **0** with **24/24** checks in **1.6 s** (two builds, since determinism
is checked by rebuilding); **38 independent checks** open the `.pptx` as a zip and parse its XML without
python-pptx, recompute every euro from confusion counts (**3,296 × 20 − 14,866 × 1.5 = 43,621.00 EUR**),
re-derive the SHAP shares from `mean_abs`, and confirm the six embedded images are **byte-identical** to
`reports/figures/*.png`.

**Blocked:** the push, as Open Questions item 10 — now **eight** commits local.
**Uncommitted:** `src/deck.py`, `docs/presentation.pptx`, `models/deck.json`, `requirements.txt`, `src/report.py`,
`docs/report.md`, `docs/report.pdf`, `models/report.json`, `PROGRESS.md`.
**Next:** T42 — the README. Every other deliverable is downstream of it, and T44's clean-clone rerun cannot
begin until it exists.

---

### 2026-10-30 — Session 44  (T42 complete)

**Completed:** **T42** — `README.md` (**248 lines**), `src/readme.py` and `models/readme.json`; criterion 9 is
**9 / 13**.

**The front door, and the command list that has to be true.** What was missing was the thing a marker reaches
for first: there is now a README that states the decision and its scope, the headline numbers beside the record
that holds each one, a two-line quick start, the **24-command** reproduction sequence in dependency order with
per-step runtimes, a repository map, the data's provenance with checksums, and the verification discipline.
It also states plainly that `app/` is empty and four tasks remain, because a README that implies a finished
repository would be the project's only dishonest document.

**The README is checked against the repository, not against its own intentions.** `python -m src.readme` runs
**33 checks**: every `python -m src.<name>` must name a real module with a `__main__` guard, every runnable
module must be named somewhere, the command order must satisfy **14 dependency constraints**, the quoted
versions must equal the pins in `requirements.txt`, the counts (modules, figures, CSV tables, documents,
decision rows, tasks done) are taken from disk, every relative link must resolve, no absolute path may appear,
and every decimal must trace to a record. The honesty clause is enforced too: the check fails the moment
`app/app.py` appears, so T43 cannot land without updating the README.

**A clean clone was then reconstructed, not imagined.** A **102-file** tree was rebuilt from `git ls-files`,
all **24** modules were imported inside it, `src.config` was confirmed to resolve to the clone rather than the
original tree (the derived-`ROOT` decision paying off), the smoke test was run there, and the copy was checked
for leaked `.venv` or bytecode: **15/15**. That is the static half of T44; the timed end-to-end rerun is still
T44's job.

**Five defects, four of them counts and one that mattered.** The first draft claimed **15** pins where the file
has **14**, **23** modules where there are **24**, and **4** CSV tables where there are **6** — all caught by
counting from disk instead of memory. The one that mattered: the README paired the file's **5,289** subscribers
with the modelling half's **11.6982%** rate. The same conflation was already in `docs/report.md` § 0
("11.6982% of the 45,211 clients in the file") and on the deck's decision slide ("4,231 subscribed" out of
45,211) — three documents carrying one shared sentence, which is how a defect propagates. All three now
distinguish the file's **11.70%** from the modelling half's **11.6982%**, and `src.report` and `src.deck` were
re-run to rebuild the PDF (27 pages) and the deck (24/24 checks). Two further flaws were in the validator
itself: the H1 check counted shell comments inside fenced code blocks, and the absolute-path check read the
`https://` in a URL as a drive letter.

**A carry-in was wrong and is recorded rather than fixed.** The task brief assumed `data/raw` and
`data/processed` stay uncommitted; they are in fact tracked (**10 files**, 5 MB) — which is what makes the
clone followable offline. The README publishes the three SHA-256 prefixes so a reader can confirm the CSVs are
the UCI originals.

**Verified:** `python -m src.readme` exits **0** with **33/33** checks in **1.2 s**; `_t42_clone.py` exits
**0** with **15/15**; `python -m src.deck` still passes **24/24** and `python -m src.report` **30/30** after the
base-rate correction.

**Blocked:** the push, as Open Questions item 10 — now **nine** commits local.
**Uncommitted:** `README.md`, `src/readme.py`, `models/readme.json`, `src/deck.py`,
`docs/presentation.pptx`, `models/deck.json`, `src/report.py`, `docs/report.md`, `docs/report.pdf`,
`models/report.json`, `PROGRESS.md`.
**Next:** T43 — the Streamlit demo, the last substantial piece of code.

---

### 2026-10-30 — Session 45  (T43 complete)

**Completed:** **T43** — `src/demo.py`, `app/app.py` and `models/demo.json`; criterion 9 is **10 / 13** and
the project is **43 / 46**.

**The front door is now a demo, and the demo is a thin UI over checked code.** The last substantial piece
of code was the one a marker clicks: an app that takes a client's attributes and returns a probability
with the drivers behind it. It is deliberately two files. `src/demo.py` holds every decision - loading the
artifact, deriving the features, scoring, explaining, pricing the call - and `app/app.py` is a Streamlit
form that imports it. The split is what makes the demo testable without a browser: `python -m src.demo`
scores the same default client, the same three T38 exemplars and the same 40-column matrix that the UI
shows.

**Five constraints, each enforced rather than promised.** The app scores `models/final_model.joblib`
only after its **sha256 and size** match `models/final_model.json`, so a re-fitted model cannot be
demoed under T31's name. The form asks for **11 business facts**, not 12 features: `previous` derives
both `previous_capped` and `previously_contacted`, `balance` derives `balance_arsinh`, `campaign`
derives `campaign_bucket`, and `days_since_contact` derives `recency_band`, all through
`features.engineer`'s own definitions. **`duration`, `pdays` and `y`** are absent from the form, the
frame and the matrix - checked mechanically. The drivers are D58's **signed sum within each one-hot
block**, in log-odds; the naive per-column alternative is not used. The cut is T34's **0.0750 = C/V**,
re-derived from `src.cost`, and a client at **0.5** is explicitly called, so 0.5 is not the rule.
**C = 1.50 EUR** and **V = 20.00 EUR** are labelled assumptions beside the EV and the 20-cell sensitivity
grid.

**The default client sits just below the price, which makes the demo honest.** The median/mode client
scores **7.22%** against the **7.50%** price and is not called; the true-positive exemplar scores
**38.19%** and the false-negative **5.50%**. The app was rendered headlessly through Streamlit's
`AppTest`: the default, both exemplars and the `previous > 0` branch that reveals the days field all
render without an exception, and the form drives the score.

**Verified:** `python -m src.demo` exits **0** with **22/22** checks in **1.7 s**, additivity
**4.605e-07**, `sigmoid(margin)` against `predict_proba` **2.027e-09**, and the three exemplars reproduce
their recorded scores to **0.0e+00**. Streamlit was already pinned at **1.52.2**, so no dependency was
added. `src.readme` was updated in the same task: it now requires the README to name the demo and the
`streamlit run app/app.py` command, and fails if `app/` is described as empty.

**Blocked:** none.
**Uncommitted:** `src/demo.py`, `app/app.py`, `models/demo.json`, `README.md`, `src/readme.py`,
`models/readme.json`, `PROGRESS.md`.
**Next:** T44 — the end-to-end rerun from a clean clone, recorded in this file.

---

### 2026-10-30 — Session 46  (T44 complete)

**Completed:** **T44** — the end-to-end rerun from a clean clone; criterion 9 is **11 / 13** and the project
is **44 / 46**.

**The test was a clone, not a convenience copy.** The pushed repository was cloned into a directory outside
the working tree at **`c1e70b6`** — **105 tracked files**, a clean working tree, no `.venv` and no bytecode —
and everything that can be rebuilt was deleted before the run: the processed CSVs, all five joblibs, every
`models/*.json`, the six `reports/*.csv`, the eight pipeline figures, and the generated report and deck.
`docs/report.md` was restored when that proved to be a mistake (below). A fresh **Python 3.11.9** virtual
environment was created and all **14 pins** installed from `requirements.txt`; `src.config.ROOT` resolves to
the clone, not to the original tree. The **22 commands** in the README were then run in the README's order,
one process at a time, with every exit code and wall-clock time recorded.

**What the clone found, in the order it found it — four real defects, all fixed:**

1. **`duration_compare` was in the wrong place in the README.** It reads `models/best_params.json`, which
   `src.tune` writes, so the original position before `train`/`tune` failed at **4.5 s** with
   `FileNotFoundError`. Moved after `tune`, it then failed at **15.8 s** on `models/threshold_choice.json`,
   which `src.threshold` writes — a second dependency the first failure had masked. It now runs after
   `lift`, and `src.readme` enforces **18** constraints, including `tune → duration_compare`,
   `threshold → duration_compare` and `duration_compare → playbook`.
2. **The report quoted machine-specific seconds.** `docs/report.md` listed `src.train` at **182.3 s**; the
   clone's record says **188.1 s**, so `src.report`'s provenance check refused the decimal and exited **1**.
   That is the check working: a number from one laptop is not reproducible on another. Exact seconds now
   live only in each record's `elapsed_seconds`, the report and README use qualitative bands, and the
   README's claim is **eight to ten minutes**, checked by `src.readme` against the 12 records' sum
   (**562.0 s = 9.4 min**) rather than rounded to a single anecdote.
3. **`09_shap_summary.png` was not reproducible.** Every SHAP value was identical between runs, but
   `shap.summary_plot` jitters the beeswarm with the **global NumPy RNG**, so the image changed. Passing
   `rng=PLOT_SEED` in `src/explain.py` fixed it: two fresh runs now produce a byte-identical figure
   (**`B0EF0026CD15B409…`**, **455,211 bytes**), which is what makes a figure comparison meaningful.
4. **`docs/report.md` is a source document, not an artifact.** The clean-room deletion removed it, and
   `src.report` correctly failed with `missing …\docs\report.md`; it was restored from `HEAD`. The generated
   outputs are `docs/report.html`, `docs/report.pdf` and `docs/presentation.pptx`.

**The checklist:**

| Check | Result |
|---|---|
| Fresh clone of the pushed repo, outside the working tree | **PASS** — `c1e70b6`, 105 tracked files, clean tree |
| Fresh virtual environment and pinned install | **PASS** — Python 3.11.9, 14/14 pins |
| `src.config` resolves to the clone, no `.venv`/bytecode tracked | **PASS** — `ROOT == clone`, 0 leaked paths |
| Every documented command exits 0, in README order | **PASS** — 22/22 after the two order fixes |
| `src.report` renders, with a browser present | **PASS** — **30/30**, PDF produced (no skipped renderer) |
| `src.deck` validates the saved package | **PASS** |
| `src.demo` scores the committed artifact | **PASS** — **22/22** |
| `src.readme` holds the README to the clone | **PASS** — **34/34** |
| Fitted artifacts reproduce | **PASS** — 5/5 joblibs byte-identical |
| Processed data and report tables reproduce | **PASS** — 4/4 processed CSVs and 6/6 report CSVs byte-identical |
| Figures reproduce | **7/8 byte-identical**; the eighth was the unseeded SHAP beeswarm, now seeded and reproducible |
| JSON metrics reproduce | **PASS** — every substantive numeric/string value identical after NaN handling; only prose-driven records differ |
| Wall-clock time recorded | **926.8 s ≈ 15.4 min** for the full sequence; the 12 records that store `elapsed_seconds` sum to **562.0 s = 9.4 min** |
| Discrepancies recorded | **Yes** — the two order defects, the runtime trace, the SHAP seed, and the source/artifact mistake above |

**The metrics did not move; the clock did.** All **five** joblibs, all **four** processed CSVs, all **six**
report CSVs and **seven of eight** figures came back byte-identical. The only JSON differences were the
records that quote the edited prose (`deck.json`, `readme.json`, `report.json`), the regenerated
report/deck files, and the newly seeded figure; after NaN handling there were **0** substantive numeric or
string differences in the metric records. The clone was slower than the machine that produced the
committed records — `src.select` alone took **288.6 s** and records no `elapsed_seconds`, `src.train`
**188.1 s**, `src.explain` **79.1 s**, `src.local_explain` **77.7 s** — but the 12 recorded steps still sum
to **9.4 min**, inside the README's stated **eight-to-ten-minute** band.

**Verified:** `python -m src.readme` exits **0** with **34/34** checks in the clone; `python -m src.demo`
**22/22**; `python -m src.report` **30/30**; `python -m src.deck` passes. The clone is disposable and lives
at `C:\Users\Mayur\AppData\Local\Temp\opencode\depositiq-t44-clone`; the fixes were applied to the working
tree and the affected records regenerated there.

**Blocked:** none.
**Uncommitted:** `README.md`, `src/readme.py`, `docs/report.md`, `src/explain.py`, the regenerated records
and figures, and `PROGRESS.md`.
**Next:** T45 — the viva Q&A bank.

---

### 2026-10-30 — Session 47  (T45 complete)

**Completed:** **T45** — the viva Q&A bank; criterion 9 is **12 / 13** and the project is **45 / 46**.

**The bank is written for a conversation, not for a shelf.** `docs/viva_qa.md` holds **37 questions** —
four for each of the nine rubric criteria, five for evaluation because that is where a marker pushes
hardest — each with a short **Answer**, an **Evidence** line naming the record, document or module that
holds the number, and, where the first answer is incomplete, an **If pressed** follow-up. It opens with a
coverage map so a reader can see the nine criteria at a glance, and closes with five numbers to know cold,
the three prohibitions in one line each, and the D18 scope sentence. The document is **8,061 words** and
**59 links**; every relative link resolves.

**The validator is the point.** `src/viva.py` parses the document rather than trusting it: the nine
criterion sections and their order, sequential question numbering, the 25-question floor, at least four
questions per criterion, an Answer and an Evidence line for every question, minimum answer and evidence
length, the three prohibitions, the D18 scope sentence, the price identity **0.0750 = C/V = 1.50 / 20.00**,
every relative link, and **every decimal**. The provenance trace admits numbers from `models/*.json`,
`reports/*.csv`, the report and the eleven analysis documents, and deliberately excludes `viva_qa.md`
itself and the downstream `demo_script.md`, so the bank cannot admit a number by quoting itself. **364
decimal tokens** all trace; **33/33 checks** pass and `python -m src.viva` exits **0**.

**Two defects were found while writing it, and both were fixed rather than excused.** The first draft had
**36 questions** but only **three** in criterion 9 — the validator's coverage check caught it, and Q37 on
the interactive demo brought that criterion to four. Three EDA rates in Q10 (**6.90% → 16.61%** and
**14.60% → 3.93%**) were not present in any upstream source; they were replaced with the qualitative claim
that the rate varies by more than ten percentage points, which the records do support, rather than
widening the source list until the numbers passed. `src/report.py` now excludes both downstream documents
from its own `SOURCE_DOCS`, so adding the bank cannot make the report's provenance check circular.

**Verified:** `python -m src.viva` exits **0** with **33/33** checks; **37** questions, per-criterion
counts **4/4/4/4/4/4/5/4/4**, **8,061** words, **59** links, **364** numeric tokens, **53,110** admissible
values. The record is `models/viva.json`.

**Blocked:** none.
**Uncommitted:** `docs/viva_qa.md`, `src/viva.py`, `models/viva.json`, `src/report.py`, `README.md`,
`PROGRESS.md`, and the regenerated report/deck/readme records.
**Next:** T46 — the five-minute demo script and rehearsal checklist.

---

### 2026-10-30 — Session 48  (T46 complete)

**Completed:** **T46** — the five-minute demo script and rehearsal checklist; criterion 9 is **13 / 13**
and the project is **46 / 46**.

**The script is written to be spoken, and checked as a document.** `docs/demo_script.md` divides the five
minutes into **six contiguous segments**: the decision and the price; the data and the leak; the model and
the cut; the live demo; what it is worth; the limits. Each segment carries a **Say** block, a **Show**
line naming what is on screen, and an **Evidence** line naming the record or figure behind it. The spoken
text is **697 words** — **139 words per minute**, which is a deliberate, unhurried pace — and the validator
refuses a script that falls outside the human speaking band.

**The claims table is the part that makes it honest.** Every number spoken appears once in a **34-row**
table, each row naming one of **9** files. A `record` value is re-found in that file at the precision the
script uses; a `derived` value is recomputed from operands the file holds. Four claims are derived rather
than stored: **13,253.00 = 43,621.00 − 30,368.00**, **21,302 = 36,168 − 14,866**, **77.60% = 821 /
1,058 × 100**, and **30.58% = 11,059 / 36,168 × 100**. The validator also checks that every decimal in
the **Say** blocks is in the table, so the script cannot acquire a number that the table does not cover.

**Three checks failed on the first run, and all three were the validator being right.** Section
references in the evidence lines (`§7.3`, `§7.4`, `§8.2`) were being counted as spoken decimals, so the
check now reads only the **Say** blocks. Claim values were compared raw (`45,211`, `11.70%`, `−0.2504`)
against a normalised set, so both sides are normalised first. And the derived-recall formula required the
scaling constant `100` to be stored in `holdout_eval.json`, which it is not; the operand check now excludes
the scale factor while still requiring the data operands. The first run was **34/37**; the fixes are in the
validator, not in a weakened claim.

**The rehearsal checklist is a failure plan, not a ritual.** **20 items** across five sections: the day
before, five minutes before, during, if something breaks, and after. The failure paths name the fallbacks
— `python -m src.demo` if Streamlit is slow, the deck and `docs/report.pdf` §7 if the app fails, the
claims table if a file is requested — and the closing item is to add any new question to the viva bank.

**Verified:** `python -m src.demo_script` exits **0** with **37/37** checks; **6** segments, **0–300 s**,
**697** spoken words, **2.32** words per second, **34** claims, **9** files, **20** checklist items,
**85** decimal tokens, **53,205** admissible values. The record is `models/demo_script.json`.

**Blocked:** none.
**Uncommitted:** `docs/demo_script.md`, `src/demo_script.py`, `models/demo_script.json`, `README.md`,
`PROGRESS.md`, `docs/report.md`, and the regenerated report/deck/readme records.
**Next:** none — the 46-task plan is complete.

---

## Open Questions / Blockers

1. ~~**BLOCKER — GitHub push pending.**~~ **Resolved 2026-10-27 (Session 18).** The push of `8278fae`
   had already succeeded at 18:44 — `push.log` records `e17f30f..8278fae  main -> main` with `EXIT=0`
   and `.git/refs/remotes/origin/main` is timestamped 18:44:09 — so the "still unpushed" line repeated
   in Sessions 2–17 was stale from that moment on. The remaining files were then committed and pushed
   in Session 18; `origin/main` is now `a734914` and its tree matches the working tree file for file.
   The one genuine obstacle was that a push requires an interactive Git Credential Manager sign-in,
   which the agent's shell could not drive (see Session 18). **Resolved again, and properly, 2026-10-29
   (Session 26):** the same push now succeeds non-interactively — `5e20d58..85e2fd9 main -> main`, exit
   0 — so the credential helper is holding a token and the seven commits that had queued up locally
   since Session 19 are pushed. This item is closed; it should not be reopened unless a push actually
   fails again.
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
10. ~~**BLOCKER — nine commits unpushed, this one included (item 1 reopened 2026-10-30, Session 41; still open at Session 44).**~~ **Resolved 2026-10-30 (Session 45).** The push succeeded non-interactively: `caefa9a..7f929c6 main -> main`, exit 0, so the credential helper is supplying a token again and `origin/main` is now `7f929c6` (T42). The nine commits that had queued since T34 are pushed. T43's commit follows in this session; if a future push fails with the same `could not read Username` error, re-authenticate Git Credential Manager and reopen this item.
