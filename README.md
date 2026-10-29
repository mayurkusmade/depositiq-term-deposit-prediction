# DepositIQ — which clients to call today

**Predicting term-deposit subscription to optimise a bank telemarketing campaign.**

A bank runs an outbound telephone campaign and has fewer agents than clients. This project builds a
ranking model that decides **which clients to call today**, prices that decision against a stated cost
model, and reports what the ranking is worth in euro — with every figure traceable to a saved record.

| | |
|---|---|
| **Dataset** | UCI Bank Marketing, `bank-full.csv` — 45,211 clients × 17 columns, committed in `data/raw/` |
| **Target** | `y` — did this client subscribe to a term deposit? (5,289 of 45,211 did — 11.70%) |
| **Model** | XGBoost, 12 features / 40 matrix columns, tuned, no imbalance treatment |
| **Operating cut** | **0.0750** — the price of a call (`C/V`), not a fitted number |
| **Status** | **44 of 46 tasks complete**; see [What is not here yet](#what-is-not-here-yet) |
| **Python** | **3.11** — required, not preferred (see [Quick start](#quick-start)) |

---

## The result, and what it does not mean

The break-even probability is **7.50%** — a call costs **1.50 EUR** and a captured deposit is worth
**20.00 EUR** — which sits *below* the **11.6982%** base rate of the **36,168**-row modelling half. The
average client is therefore already worth calling, so this model **converts a fixed call budget** better
than a fixed order. It is an improvement to a campaign that already pays, and it is **not** evidence that
the bank should cut campaign spend. That sentence governs every number below.

| | Call everyone | Ranked, cut at 0.0750 |
|---|---|---|
| Calls made (training half) | 36,168 | 14,866 |
| Subscribers reached | 4,231 | 3,296 |
| **Net value** | **30,368.00 EUR** | **43,621.00 EUR** |
| Calls made (held out) | 9,043 | 3,621 |
| **Net value held out** | **7,595.50 EUR** | **10,988.50 EUR** |

| Metric | Value | Where it is recorded |
|---|---|---|
| PR-AUC, cross-validated | **0.4460** ± 0.0110 | `models/metrics_cv.json` |
| PR-AUC, held out once | **0.4616** | `models/holdout_eval.json` |
| ROC-AUC, held out | **0.8052** | `models/holdout_eval.json` |
| Share of list called / subscribers reached | **41.10%** / **77.90%** | `models/threshold_choice.json` |
| Top-decile lift, held out | **4.5100×** | `models/lift_results.json` |
| Gain over calling everyone | **+13,253.00 EUR** on **21,302** fewer calls | `models/playbook.json` |

Three things this project deliberately does **not** do, each with the number that tempts it:

1. **`duration` is never an input.** It is the length of the call, recorded after the call ends.
   Including it would add **+0.1788** PR-AUC and **+20,340.00 EUR** — and is worthless at decision time.
2. **Accuracy is never a headline.** A model that answers "no" to everyone scores **88.3003%**.
3. **`month` is never read as a cause.** It carries **17.55%** of importance but spans a 2008–2010
   recession; it schedules work, it does not predict a person.

---

## Quick start

```bash
git clone https://github.com/mayurkusmade/depositiq-term-deposit-prediction
cd depositiq-term-deposit-prediction

python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate  on macOS/Linux
python -m pip install -r requirements.txt
```

`requirements.txt` carries **14 exact pins**. **Python 3.11 is required**: `xgboost==3.2.0` and
`shap==0.51.0` are the newest releases that still install on 3.11, and their newer versions need 3.12+.
Bumping them without moving the interpreter will fail at install time, not at run time.

Smoke test — no data or model needed, and it prints the number the whole project turns on:

```bash
python -m src.cost              # C = 1.50, V = 20.00 -> p* = 7.50%, with a 20-cell sensitivity grid
```

Nothing needs to be downloaded: the raw CSVs are committed under `data/raw/` (see [The data](#the-data)).

---

## The interactive demo

```bash
streamlit run app/app.py        # from the repository root, with the venv active
```

The app scores `models/final_model.joblib` — loaded, never re-fitted — from eleven client attributes and
returns **P(subscribe)** with the twelve SHAP contributions behind it. The cut it uses is the price,
**7.50% = C/V**, not 0.5; `C = 1.50 EUR` and `V = 20.00 EUR` are labelled as assumptions beside every
euro figure. It also loads the three T38 exemplar clients (true positive, false positive, false negative)
so the demo can be used without typing a client first.

The interface deliberately does not ask for `duration`, because it is recorded after the call ends. It
does not treat 0.5 as a decision rule, and it does not present the euro figures as a bank's own data.
Those three refusals are checked mechanically: `src.demo` runs **22 headless checks** over the same core
the app imports, including the artifact digest, the C/V identity, the absent leak, the additivity of the
twelve contributions and the three exemplars' recorded scores.

---

## Reproduce every number

Each module is runnable as `python -m src.<name>`, validates its own output, prints its checks and
exits non-zero on disagreement. Run from the repository root. The twelve steps that record their runtime
total **about eight to ten minutes** depending on the machine; `src.select`'s permutation null, `src.train`
and `src.local_explain` dominate, and the exact seconds are in each record's `elapsed_seconds` field.

```bash
# 1. data: raw -> processed, then the one split every later step replays
python -m src.clean             # both variants, 0 rows lost
python -m src.split             # train.csv / test.csv, 36,168 / 9,043

# 2. features and selection
python -m src.features          # 17 engineered features, 8 leakage checks
python -m src.preprocess        # 17 features -> 65 matrix columns
python -m src.select            # -> 12 features / 40 columns, 06_feature_importance.png
python -m src.leakage           # the duration audit
# 3. models
python -m src.train             # 4 candidate builds, 5-fold CV (~3 min)
python -m src.imbalance         # weights / none / SMOTE (~1 min)
python -m src.tune              # 40 of 4,800 points (~1 min)
python -m src.final             # models/final_model.joblib
# 4. evaluation - the held-out half is scored here, once
python -m src.metrics           # metrics_cv.csv (~1 min)
python -m src.evaluate          # held-out metrics + 3 figures (~10 s)
python -m src.threshold         # the cut at 0.0750 (~10 s)
python -m src.lift              # deciles, lift and gain (~10 s)
python -m src.duration_compare  # with-vs-without duration at the tuned point (~15 s)
# 5. explanation and recommendations
python -m src.explain           # global SHAP (~1 min)
python -m src.local_explain     # three exemplar clients (~1 min)
python -m src.playbook          # eight recommendations as records
# 6. deliverables
python -m src.report            # report.md -> report.pdf (needs Chrome, Edge or Chromium)
python -m src.deck              # presentation.pptx
python -m src.demo              # headless checks for the Streamlit demo core
python -m src.readme            # re-checks this README against the repository
```

Two further modules are utilities rather than pipeline steps: `python -m src.config` prints every resolved
path (and proves none is hard-coded), and `python -m src.data_loader` prints both variants' shapes and
class balance. `src.cost`, `src.leakage`, `src.features` and `src.preprocess` are libraries that also run
their own checks when invoked.

`src.report` prints its HTML through a browser already on the machine, so the PDF needs no Python
package; if no browser is found it fails loudly rather than writing an HTML-only report.

Figures are written to `reports/figures/`, records to `models/*.json`, tables to `reports/*.csv`.
Every module is deterministic under seed 42, and the modules that were re-run from scratch produced
byte-identical figures.

---

## What is in the repository

```
depositiq-term-deposit-prediction/
├── data/raw/              bank-full.csv, bank.csv, bank-additional/, bank-names.txt
├── data/processed/        train.csv, test.csv (+ the 20-feature variant)
├── src/                   25 modules, each runnable as `python -m src.<name>`
├── notebooks/             01_data_understanding, 02_preprocessing, 03_eda
├── models/                artifacts + the JSON record behind every claim
├── reports/               6 CSV tables + 19 figures
├── docs/                  the report, the deck, and 11 supporting documents
├── app/                   app.py - the Streamlit demo (T43)
└── PROGRESS.md            46-task checklist, 65-entry decisions log, session log
```

Start with the two deliverables, then follow whichever thread you need:

| Document | What it is |
|---|---|
| [docs/report.md](docs/report.md) · [docs/report.pdf](docs/report.pdf) | the full report — nine rubric criteria in order, 27 pages rendered |
| [docs/presentation.pptx](docs/presentation.pptx) | the 15-slide deck, business-first |
| [docs/01_problem_definition.md](docs/01_problem_definition.md) | the decision, six objectives, six stakeholders, the cost model |
| [docs/02_data_dictionary.md](docs/02_data_dictionary.md) | every column, its allowed values and its leakage risk |
| [docs/04_eda_insights.md](docs/04_eda_insights.md) | seven ranked insights over five figures |
| [docs/05_leakage_audit.md](docs/05_leakage_audit.md) | `duration` and seven further leakage surfaces, with verdicts |
| [docs/06_modeling.md](docs/06_modeling.md) | the model comparison, imbalance study, tuning and held-out evaluation |
| [docs/07_threshold.md](docs/07_threshold.md) · [docs/07_lift.md](docs/07_lift.md) | the operating cut and the ranking's value per hundred calls |
| [docs/08_explainability.md](docs/08_explainability.md) | SHAP globally and for three clients |
| [docs/08_recommendations.md](docs/08_recommendations.md) | eight recommendations, each with a falsifier |
| [PROGRESS.md](PROGRESS.md) | every task, every decision and every defect found, in order |

---

## The data

UCI Machine Learning Repository, **Bank Marketing** dataset (`bank-full.csv`), used unchanged.

> S. Moro, P. Cortez and P. Rita. *A data-driven approach to predict the success of bank telemarketing.*
> Decision Support Systems, 62:22–31, 2014.

It contains no missing values, which is the first trap: missingness arrives as the literal string
`unknown` — `poutcome = unknown` covers **81.7%** of rows and `contact = unknown` **28.8%**.

The CSVs are **committed** rather than downloaded at run time, so a clone reproduces offline. To confirm
you have the same file, compare its checksum:

| File | Size | SHA-256 (first 16) |
|---|---|---|
| `data/raw/bank-full.csv` | 4,655,560 B | `ada45d7c04ac65e4` |
| `data/raw/bank.csv` | 465,996 B | `809288f2221d024b` |
| `data/raw/bank-additional/bank-additional-full.csv` | 5,834,924 B | `74adfc578bf77a7f` |

The 45,211-row `bank-full.csv` is the modelling variant. The 41,188-row `bank-additional-full.csv` adds
five macro-economic columns that are identical for every client on a given day, so it cannot change one
client's score against another's; it is carried but not shipped. The two are **not** interchangeable —
`education` is coded differently, `pdays` uses `-1` in one and `999` in the other — which is why the
variant is a parameter and not a default.

---

## How the numbers were checked

Three layers, applied to every task, because a number that cannot be re-derived is a number that will
eventually be wrong:

1. **In-module checks** — each module validates its own output before writing it, and exits non-zero on
   disagreement (12 to 17 checks per module, listed in the session log).
2. **Independent verifiers** — a throwaway script that imports neither the module nor its library,
   recomputes every figure by a different route (rank-based AUC, step-block average precision,
   `searchsorted` counts, xgboost's own `pred_contribs` instead of `shap`), and traces every number in
   the prose to a record.
3. **Numeric traces** — documents are parsed for every decimal, and each one must exist in a
   `models/*.json` record, a `reports/*.csv` column or a supporting document. The report's 497 tokens,
   the deck's 157 and the playbook's 63 all pass; a number with no record behind it fails the build.
4. **A clean-clone rerun** — the pushed repository was cloned outside the working tree, given a fresh
   virtual environment and the pinned install, and run through the command list above in order. The
   fitted artifacts, processed data, report tables and seven of eight figures came back byte-identical;
   the eighth was a SHAP beeswarm whose jitter was unseeded, now seeded. The two defects the rerun found
   — a command in the wrong position and exact runtimes quoted in the report — are recorded with the fix.

The verifiers found real defects rather than confirming the work: a `rank_auc` that masked the wrong
order, a Brier score computed in `float32`, scikit-learn's `std_test_score` using `ddof=0`, a
misleading `subscribers_missed` field, and — in the report — a held-out accuracy quoted beside a
cross-validated balanced accuracy. All are recorded in `PROGRESS.md`, with the fix.

---

## What is not here yet

The repository is **44 of 46 tasks** into a 46-task plan, and the two open items are deliverables
rather than analysis:

| Task | Missing piece |
|---|---|
| T45 | `docs/viva_qa.md` — 25+ likely questions with answers |
| T46 | `docs/demo_script.md` — a five-minute walkthrough |

The Streamlit demo is present at `app/app.py` and is checked by `python -m src.demo`; no number above
depends on the two remaining tasks.

---

## Limits worth knowing before quoting it

1. **The prices are assumptions.** `C = 1.50 EUR` and `V = 20.00 EUR` are not the bank's figures. Every
   euro total is linear in `V` and the cut is `C/V`: at 40.00 EUR the cut halves and the totals double.
2. **The held-out half is the same campaign.** "Generalises" here means unseen clients of this
   campaign, not next year's.
3. **The split is stratified-random, not temporal.** A client in the test half may be a neighbour of one
   in train. A temporal split is the honest test and was not done.
4. **The tuning gain is smaller than its error bar** — **+0.00165 ± 0.00286** — so the tuned model is not
   meaningfully better than the untuned one, only not worse.
5. **Fairness is not audited** beyond segment-level precision and recall.

---

## Licence and attribution

Coursework for a machine-learning module; the dataset is redistributed under its original UCI terms and
cited above. The code is the author's. `PROGRESS.md` records the decisions, including the ones that were
reversed.
