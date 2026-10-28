# DepositIQ — `duration` Leakage Audit

**Rubric:** Criterion 5 — Feature Engineering / Selection (10 marks) **Tasks:** T22 (design the
with-vs-without experiment) · **Status:** complete · **Date:** 2026-10-28 **Depends on:** D05 (the
column is excluded from the main model), T16–T21 (the EDA that measured its correlation), and T15's
frozen split (D26). **Numbers:** every figure below is printed by `python -m src.leakage`, which
recomputes the whole audit from `data/processed/train.csv` on demand. No model is fitted by this
task, and the **held-out 9,043 rows are not read** by it.

---

## 1. What the column is, and when it becomes known

`duration` is the length of the marketing call in **seconds**: last contact duration, measured from
the moment the call connects to the moment it ends.

| | value |
|---|---|
| minimum | **0 s** |
| first quartile | **103 s** |
| median | **180 s** |
| third quartile | **318 s** |
| maximum | **4,918 s** (82 minutes) |
| median when the client subscribed | **426 s** |
| median when the client did not | **164 s** |

The timing is the whole problem, and it is worth drawing rather than describing:

```
   dial ──► conversation ──► outcome (y) ──► hang up ──► duration recorded
     ▲                                                        │
     └──────────── the model must decide HERE ────────────────┘
                   duration does not exist yet
```

A scoring model is used **before** the call, to decide whether to place it. `duration` is produced
**at the end** of the call whose outcome it is supposed to predict. There is no point in the
campaign at which a decision-maker has the column and still has the decision to make.

The dataset's own documentation says this, and says it more strongly than a project note can:

> “duration: last contact duration, in seconds (numeric). Important note: this attribute
> highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration
> is not known before a call is performed. Also, after the end of the call y is obviously
> known. Thus, this input should only be included for benchmark purposes and **should be
> discarded if the intention is to have a realistic predictive model**.”
> — `data/raw/bank-additional/bank-additional-names.txt`, line 56

D05 therefore excludes it from the main model and keeps it only for one documented comparison. This
document is the specification of that comparison, and the record of how much the exclusion costs.

---

## 2. Why the column cannot be used, in three arguments

**2.1 It is not available at decision time.** That is §1, and it is sufficient on its own: a feature
the bank cannot compute when it needs the score cannot be in the score.

**2.2 It is not a client attribute at all — it is a property of the call.** Two clients with
identical files can produce different durations, because the duration depends on how the
conversation went. A model that ranks by it is ranking *conversations that already happened*, which
is not a list of people to call.

**2.3 You cannot obtain it without doing the thing it is meant to prioritise.** The obvious
objection — "call everyone briefly, measure, then call the promising ones back" — fails on its own
arithmetic: the measurement call is the cost the model exists to avoid, it would be placed on every
client rather than on a ranked subset, and its result is not the outcome (a client who talks for
nine minutes and says no is exactly the client a duration ranker would promote).

**Why this is not merely a technicality.** The project's claim, stated in
`docs/01_problem_definition.md`, is that a **fixed call budget converts better** when the list is
ranked. A ranking built on `duration` cannot be applied to a client who has not been called, so it
cannot change which clients get called — the only decision the business has.

---

## 3. How much the leak is worth, measured

This section measures the column, not a model: the rate by call length, and what a ranking built on
`duration` **alone** would score. Both are diagnostics for the audit; neither is a result to report
as predictive skill.

### 3.1 The subscription rate rises monotonically with call length

| call length | calls | share of calls | subscribers | rate | Wilson 95% |
|---|---|---|---|---|---|
| **0–59 s** | 3,727 | 10.30% | **6** | **0.16%** | 0.07% – 0.35% |
| 60–119 s | 7,379 | 20.40% | 158 | 2.14% | 1.83% – 2.50% |
| 120–179 s | 6,900 | 19.08% | 400 | 5.80% | 5.27% – 6.37% |
| 180–299 s | 8,347 | 23.08% | 901 | 10.79% | 10.15% – 11.48% |
| 300–599 s | 6,763 | 18.70% | 1,284 | 18.99% | 18.07% – 19.94% |
| **600 s +** | 3,052 | 8.44% | **1,482** | **48.56%** | 46.79% – 50.33% |

The gradient is monotone across all six bands — every band's rate is above the one before it, and
the intervals of the first two and the last two do not overlap. Two summaries make the point sharper
than the table:

- **The shortest tenth of the calls produces almost nobody.** 3,727 calls (10.30%) under a minute
  yield **6 subscribers**, which is **0.14%** of the 4,231 in the half.
- **The longest 8.44% of calls produce more than a third of the subscriptions.** 3,052 calls at 600
  s or more yield **1,482** subscribers, **35.03%** of the total.

No client attribute in the EDA comes close to that. The strongest categorical segment
(`poutcome = success`, 64.65%) covers 3.33% of the list; the strongest numeric correlation is
+0.3967. `duration` does not merely correlate with the outcome — it is a *stochastic recording of
the outcome itself*.

### 3.2 A ranking built on `duration` alone scores like a model

| diagnostic | value | baseline |
|---|---|---|
| ROC-AUC | **0.8078** | 0.5000 |
| PR-AUC | **0.3897** | **0.1170** (the base rate) |
| top-decile precision | **0.4558** | 0.1170 |
| top-decile recall | 0.3895 (**1,648 of 4,231**) | — |

One column, used directly as a score with no fitting, beats the base rate by **3.33×** on PR-AUC and
reaches a **3.90×** lift in the top decile. Any model reported with this column inside it would look
excellent and be worthless, which is precisely the failure mode D05 exists to prevent.

**Read the numbers as a ceiling, not a result.** This is not a classifier and it has not been
cross-validated; it is the magnitude of the thing being excluded, measured so that T36's gap has a
yardstick rather than a shrug.

### 3.3 What this does *not* assert

The table in §3.1 is a description. Longer calls and subscriptions are jointly produced — a client
who is interested stays on the phone, and the conversation continues because they are interested —
so the causal arrow runs in both directions and no claim is made that extending calls would raise
the rate. That ambiguity is another reason the column is unusable for targeting: a predictor whose
mechanism cannot be separated from its target cannot guide an intervention.

---

## 4. The experiment T36 will run

The task's requirement is an **identical pipeline with the feature set toggled**. "Identical" is the
part that is easy to claim and hard to guarantee, so it is specified as **six held-fixed elements**
(§4.2) and **four assertions** (§4.3), not as an intention.

### 4.1 The two arms

| arm | feature set | purpose |
|---|---|---|
| **A — ships** | every engineered column **except** `duration` | the model whose metrics are reported in T32–T35 |
| **B — diagnostic** | the same columns **plus** `duration` | measured once, to price the exclusion; never deployed |

The toggle is code, not a convention: both arms pass their column list through
`src.leakage.feature_columns(columns, with_duration=...)`, which drops `LEAKY_COLUMNS` and asserts
that it did. A hand-filtered list in one arm and a hand-filtered list in the other is exactly how a
comparison acquires a second, unnoticed difference.

### 4.2 What is held fixed

| element | how it is held fixed |
|---|---|
| **rows** | the same training half, 36,168 calls, loaded once and passed to both arms; the 9,043 held-out rows are never loaded |
| **folds** | **one** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` splitter, whose `.split()` output is materialised **once** and replayed into both arms — fold membership is identical by construction, not by the seed coinciding |
| **preprocessing** | the same T24 `ColumnTransformer` **factory**, rebuilt inside each fold so nothing is fitted on validation rows; scaler and encoder choices are read from one definition, so a difference cannot come from a different transform |
| **model** | the same estimator class, hyperparameters and `random_state`, fixed by T26–T29 before this comparison runs |
| **evaluation** | one scoring function, applied to both arms, returning the same metrics in the same order |
| **environment** | one script, one run, both arms in the same process, so no library or machine difference can enter |

### 4.3 What differs

Exactly one thing: the presence of `duration`. Four assertions make that literal, and the comparison
script must fail loudly rather than proceed if any is false:

1. `set(arm_b_columns) - set(arm_a_columns) == {"duration"}`
2. the fold index arrays of the two arms are equal element-wise
3. both arms see the same target vector (`y` compared with `.equals()`)
4. every fitted transformer in every fold reports statistics computed from the fold's training rows
   only

### 4.4 What is reported

Not a single number and a verdict, but a table with the paired difference, because the folds are
shared and the arms are therefore paired samples:

| metric | A (ships) | B (diagnostic) | gap (B − A) |
|---|---|---|---|
| ROC-AUC | mean over 5 folds | mean over 5 folds | mean ± sd of the 5 **per-fold** differences |
| PR-AUC | ” | ” | ” |
| precision at the 7.50% break-even | ” | ” | ” |
| recall at the 7.50% break-even | ” | ” | ” |
| F1 at the 7.50% break-even | ” | ” | ” |
| top-decile lift | ” | ” | ” |
| Brier score | ” | ” | ” |

The threshold is D17's **7.50%** break-even, not 0.5, for the same reason T34 will use it: it is the
point at which a call pays for itself. The gap is also translated once into the project's money —
the extra subscriptions a fixed call budget would appear to buy — so the cost of honesty is stated
in the unit the business uses.

### 4.5 The decision rule, written before the numbers exist

This is stated now, so the result cannot be rationalised after the fact:

1. **Arm A ships regardless of the gap.** A gap is expected and is not a reason to reconsider; D05
   is decided on availability, not on performance.
2. **The gap is reported as the price of a deployable model**, in PR-AUC and in money, and it is
   quoted wherever `duration`'s correlation with the target is quoted. The EDA already does this
   from the other side (I6: +0.3967 is "never quoted as a good predictor").
3. **Arm B's metrics are labelled an upper bound** wherever they appear, and no targeting, lift or
   savings claim may be built on them.
4. **If the gap turns out to be near zero, that is reported as the finding.** It would mean the
   other features already carry what `duration` carries, and the audit would say so rather than
   insist on a loss the data does not show.

---

## 5. The other leakage surfaces, audited

`duration` is the leak the brief warns about, but "is anything else leaking?" deserves an answer
rather than an omission. Each surface below was checked against the EDA's measurements.

| surface | what it is | verdict | evidence |
|---|---|---|---|
| **`duration`** | this call's length, recorded at hang-up | **LEAK — excluded** (D05) | §3: 0.8078 ROC-AUC on its own; 0.16% → 48.56% across the bands |
| **`pdays` vs `days_since_contact`** | the same number twice, with a sentinel | **not leakage — redundancy.** Keep `days_since_contact`, drop raw `pdays` (D36) | identical on all 6,584 rows where both exist (D23); VIF 9.0×10¹⁵ for both, 1.01–1.03 once dropped |
| **`poutcome`** | the **previous** campaign's result | **legitimate at prediction time** — 64.65% against a 4.11% floor — provided any encoder is fitted on training rows only | T18 §3.1; T21 I2 |
| **`y`** | the label | never a feature; encoded 1/0 by `src/split.encode_target` and asserted to be binary | `src/split.py` |
| **temporal order** | the split is **stratified random** (D26), not chronological | **unresolved, and declared** | if the file's row order is chronological, a model can train on later calls than it is tested on; the file carries no year and no client id, so this cannot be repaired from the data |
| **duplicate rows** | 12 exact duplicates in the additional variant, **0 in the primary** | no cross-split duplicate risk in the model that ships | T10/T14 (`docs/02_data_dictionary.md` §1) |
| **test-half contact** | any statistic computed over train + test | **forbidden**; every transformation is fitted inside the training fold | §4.2, §4.3 assertion 4 |

Three of these are worth a sentence more than a table cell.

**The duplicate is not a leak.** `pdays` and `days_since_contact` being the same column is a
redundancy problem: both are available before the call, so neither reveals the outcome. Folding them
into one is D36, and conflating that with leakage would make the audit look thorough while
mis-stating the risk.

**`poutcome`'s legitimacy depends on the encoder.** As a column it is a fact about a *past* campaign
and is known before this call. It becomes a leak only if a target-encoding step is fitted over the
whole frame, including the held-out rows — which is why §4.3's fourth assertion checks that every
fitted transformer reports statistics from its own training fold.

**The unresolved one is temporal.** The split is stratified random, so the test half contains
earlier calls than some training rows. For a bank that runs campaigns in sequence, the honest
evaluation would be a time-based split, and this project does not have one: T15 froze the split
(D26), every EDA number is computed on it, and changing it now would invalidate the whole of Phase
4. It is recorded as a limitation of the *evaluation*, not of the model, and T35's discussion
carries it forward.

---

## 6. What would make this audit wrong

Stated so the audit can be argued with rather than merely believed:

1. **If the model were used after the call** — to decide whom to call *back*, say — `duration` would
   be available and legitimate. It is not: `docs/01_problem_definition.md` §3 (O4, O5) and §4.1
   define the task as ranking clients *before* dialling, and §4.2 excludes deployment-time scoring
   of anything but client attributes.
2. **If `duration` were the length of a *previous* call** it would be a history feature like
   `previous`. It is not; it is this call's.
3. **If the gap in §4 were near zero**, the exclusion would cost nothing and the argument would
   reduce to "no harm done". The audit would still exclude the column — availability, not accuracy,
   is the reason — but it would say so plainly.
4. **If the bands in §3.1 were an artefact of a few extreme calls**, the top band would not rest on
   3,052 calls with a 46.79%–50.33% interval. It does, so the gradient is not a tail effect.

---

## 7. Self-check against the done-condition

The task's condition is: *the experiment (identical pipeline, feature set toggled) is specified and
the leakage risk is documented.*

| Requirement | Where satisfied |
|---|---|
| Identical pipeline, feature set toggled | §4.1–§4.3 — two arms, six held-fixed elements, one difference, four assertions that fail loudly |
| The experiment is specified concretely enough to run | §4.4's reporting table and §4.5's decision rule, both fixed **before** the numbers exist |
| The leakage risk is documented | §1–§3 — what the column is, why it is unavailable at decision time, and what it is worth (0.8078 ROC-AUC alone; 0.16% → 48.56% across the bands) |
| The risk is quantified, not asserted | §3 — seven measured statistics, reproducible with `python -m src.leakage` |
| Other surfaces checked | §5 — seven surfaces with a verdict and its evidence, including one **unresolved** limitation |
| A runnable artifact, not a plan on paper | `src/leakage.py` — `feature_columns()` is the toggle T36 will call; `audit()` prints every number in this document |

*Interpretation for the viva:* "`duration` is the length of the call, so it is only known once the
call has ended — and it is worth more on its own, 0.8078 ROC-AUC, than any combination of client
attributes in the EDA. That is exactly why it is excluded: a model containing it would score the
conversations it has already had, not the clients it has to choose between. I measured the cost of
honesty as a specification rather than an apology — the same pipeline run twice with one column
toggled, same folds, same preprocessing, and the gap reported in PR-AUC and in money."

---

## 8. Feature selection: what the ranking does with the leak (T25)

T25 asks for two importance methods, compared, with every keep and every drop justified. This section is
the record: `src/select.py` prints every number below, and `reports/figures/06_feature_importance.png`
draws both rankings over the same 17 features. The 17 are the features behind T24's 65 columns, so the
leak is already gone from the matrix before anything here is measured - which is the point of ordering it
after §3-§4 rather than before.

### 8.1 The two rankings

**M1 - mutual information** on the 17 source features (a categorical enters as its integer codes, so `job`
is one 12-level variable), each judged against **its own permutation null** from 10 label shuffles.
**M2 - permutation importance**: the ROC-AUC lost when a feature's columns are shuffled, on the whole
held-out fold, from a 200-tree forest whose out-of-fold ROC-AUC is **0.7936 +/- 0.0063**.

| feature | MI | its null 95th | z | MI rank | AUC lost when shuffled | model rank |
|---|---|---|---|---|---|---|
| `month` | 0.0240 | 0.0003 | 271.9 | 2 | **+0.04971** +/- 0.00461 | **1** |
| `contact` | 0.0134 | 0.0001 | 400.7 | 4 | +0.03270 +/- 0.00851 | 2 |
| `poutcome` | **0.0291** | 0.0001 | 707.8 | **1** | +0.01905 +/- 0.00282 | 3 |
| `day` | 0.0082 | 0.0025 | 7.4 | 11 | +0.01498 +/- 0.00102 | 4 |
| `age` | 0.0127 | 0.0014 | 21.2 | 6 | +0.01000 +/- 0.00265 | 5 |
| `housing` | 0.0097 | 0.0001 | 444.9 | 9 | +0.00732 +/- 0.00220 | 6 |
| `recency_band` | 0.0132 | 0.0002 | 232.3 | 5 | +0.00614 +/- 0.00207 | 7 |
| `marital` | 0.0023 | 0.0001 | 125.6 | 16 | +0.00453 +/- 0.00277 | 8 |
| `balance_arsinh` | 0.0222 | 0.0026 | 20.1 | 3 | +0.00390 +/- 0.00068 | 9 |
| `campaign_bucket` | 0.0041 | 0.0002 | 74.9 | 13 | +0.00271 +/- 0.00256 | 10 |
| `previously_contacted` | 0.0115 | 0.0000 | 635.1 | 8 | +0.00241 +/- 0.00159 | 11 |
| `previous_capped` | 0.0116 | 0.0001 | 527.7 | 7 | +0.00233 +/- 0.00145 | 12 |
| `education` | 0.0024 | 0.0001 | 68.6 | 15 | +0.00069 +/- 0.00093 | 13 |
| `loan` | 0.0025 | 0.0001 | 124.4 | 14 | +0.00052 +/- 0.00131 | 14 |
| `default` | 0.0004 | 0.0000 | 27.5 | 17 | -0.00010 +/- 0.00015 | 15 |
| `balance_band` | 0.0050 | 0.0001 | 127.1 | 12 | -0.00023 +/- 0.00085 | 16 |
| `job` | 0.0085 | 0.0002 | 120.5 | 10 | -0.00036 +/- 0.00039 | 17 |

The two orderings correlate at **Spearman +0.647** - substantial, and far from agreement. `poutcome`
tops the filter and sits third for the model; `month` does the reverse; `marital` moves eight places.

The `null 95th` column is what makes the MI column readable. Raw MI is biased upward by cardinality, so
`month` (12 levels) would out-rank `contact` (3) even if neither carried information; dividing by each
feature's own null is what separates 0.0240 with signal from 0.0240 of bias. It also shows how sharp the
null is: the narrowest null standard deviation is **1.25e-05**, which is why the `z` column is enormous
for everything and is **not** used as an effect size anywhere below.

### 8.2 Where they disagree, and why neither wins

Five features sit 5 or more rank places apart:

- **`marital`** - filter 16, model 8. Marginal association is nearly nil (MI 0.0023) but the forest uses
  it for **+0.00453** AUC. This is the classic masked-signal case: alone it says little, in combination
  it separates.
- **`day`** - filter 11, model 4. D35 already established that `day`'s marginal band difference
  **reverses inside `may`**, so a marginal statistic averages a real interaction to nothing while a tree
  can split on `day` x `month`. The model is right here and the filter cannot be.
- **`balance_arsinh`** - filter 3, model 9. The filter sees the monotone relationship; the forest finds
  the same information reachable through `age` and `day`.
- **`previous_capped`** (7 vs 12) and **`job`** (10 vs 17) - the filter rates both on association the
  forest does not need, since `poutcome` and `previously_contacted` already carry the contact history and
  `job`'s levels are largely reproduced by `education`, which is itself weak.

Neither ranking is corrected by the other. The disagreement is the input to the next step, not a problem
to be smoothed away by averaging two ranks.

### 8.3 The screen, and the finding that it cannot decide

Nominating a feature required **either** view to be unable to see it: MI at or below its own 95th
percentile, **or** `AUC lost - sd <= 0`.

Measured, the first condition never fires. No feature has `z < 1`; the lowest is `day` at **7.4**,
**16 of 17** clear `z = 20`, and the smallest MI in the table (`default`, 0.0004 nats) still sits at
`z = 27.5`. With 10 shuffles of 36,168 rows there is almost no room for spurious association, so
**every nominee comes from the model half**. The rule I wrote had two conditions and only one of them was
capable of being false - which is precisely why it is not allowed to make the decision on its own.

That leaves **five nominees**, each labelled *model blind, filter sees it*: `education`, `loan`, `default`,
`balance_band`, `job`.

`campaign_bucket` carries a declared **carve-out** (it is the model's only representation of the T19/D33
call-budget finding, which §6 of the report discusses while the shipped model is on display). In the event
the carve-out was not needed: it cleared the model condition by **+0.00015** AUC, which is inside
rounding distance of the noise. It is recorded as a guard that exists rather than a decision that was
taken.

### 8.4 The ablation, which does decide

The five nominees were removed **as a group**, the same five folds were re-run, and the paired per-fold
differences decide:

| fold | all 17 features (65 cols) | 12 features (40 cols) | delta |
|---|---|---|---|
| 1 | 0.78816 | 0.78688 | -0.00129 |
| 2 | 0.80354 | 0.80429 | **+0.00075** |
| 3 | 0.78568 | 0.78229 | -0.00340 |
| 4 | 0.79410 | 0.79593 | **+0.00183** |
| 5 | 0.79653 | 0.79794 | **+0.00141** |
| **mean** | | | **-0.00014 +/- 0.00195** |

Three of five folds **improve** when the five features are removed, and the mean loss is 0.00014 against a
spread of 0.00195 - the loss is **14 times smaller than its own noise**.

My first version of this test compared the mean against exactly zero, which would have made the verdict
turn on that 0.00014. That is not a threshold, it is a coin flip with a decimal point. The rule is now the
**one-standard-error form** used in model selection: take the simpler model when its score is within one
standard error of the best. Concretely, the group goes if `mean delta >= -sd`. Stage 1 already judged
features by "distinguishable from zero"; stage 2 now uses the same standard.

### 8.5 The decision

**Dropped (5):** `education`, `loan`, `default`, `balance_band`, `job`.

- `education`, `loan`, `default` - the three weakest categoricals by both measures; `default` is
  **0.0004 nats** of mutual information and costs **-0.00010** AUC when shuffled.
- `balance_band` - redundant with `balance_arsinh`, which keeps the balance information at a finer
  resolution and is 9th by the model. The band's only distinctive content (the dip above EUR 5,000) is not
  something the forest uses.
- `job` - the largest single block in the matrix (**12 columns** of the 65) and the 17th by the model:
  shuffling all twelve costs **-0.00036** AUC. It is 10th by mutual information, which is the disagreement
  the ablation resolves against it.

**Kept (12):** `month`, `contact`, `poutcome`, `day`, `age`, `housing`, `recency_band`, `marital`,
`balance_arsinh`, `campaign_bucket`, `previously_contacted`, `previous_capped` - the shipped matrix is
**40 columns, down from 65**.

One consequence worth stating rather than letting a reader discover it: the shipped model keeps `age` and
`marital` but no longer sees `job` or `education`. D16 already scopes out a formal fairness audit, and
this decision is not one - but a model that has stopped using occupation and schooling is a defensible
place to be, and it happened for measured reasons rather than as a gesture.

### 8.6 What this says, and does not say, about the leak

- **No forbidden column was ranked.** `duration`, `pdays` and `y` are absent from the matrix, so the
  check that they never reached the ranking is a test, not a promise.
- **This cannot be used to argue the leak is expendable.** Everything here is computed on the feature set
  §3-§4 already stripped. `duration` is worth **0.8078** ROC-AUC by itself (§3.2); nothing in this ranking
  measures or implies what would be gained or lost by admitting it.
- **Selection is not itself a leakage surface here.** The ablation re-uses the folds materialised before
  it, and every fold refits the T24 preprocessor inside its own training rows (D24), so no fold's scaler
  or encoder ever saw another fold's rows.
- **The decision is instrument-specific, and that is a limitation.** The forest is not the shipped model
  (D08 selects XGBoost; T26-T30 choose and tune it). A feature can be redundant to one learner and useful
  to another - `job`'s twelve columns are a plausible case for the linear baseline. The ablation numbers
  are on record so that if T32 finds the reduced set underperforming, this decision is the first thing
  revisited, with evidence rather than hindsight.
- **Block attribution overstates overlapping levels.** Permutation importance is summed within a feature
  block, so a block whose one-hot levels predict the same thing has its contribution counted more than
  once. The direction of that bias is known (upward) and is stated rather than corrected.

### 8.7 Two defects the measurement caught

Both were mine, and neither was visible from reading the code.

1. **A head slice of a validation fold is a time slice.** The first version permuted on
   `X_valid.iloc[:3000]` to save time. `StratifiedKFold` returns indices in ascending original order and
   the file is ordered by campaign date (D26), so those 3,000 rows were the campaign's opening weeks.
   They contained **no** `poutcome=success` rows at all, which made 20 columns constant: `poutcome`,
   `recency_band`, `previous_capped` and `previously_contacted` came back at exactly
   **0.00000 +/- 0.00000**, `housing` at -0.00711, and the screen nominated **13 of 17** features
   including `month`, the most important one. Using the whole fold removed it. The cost of the shortcut
   was not a noisier number but a wrong one.
2. **A threshold inside the noise is not a threshold.** Discussed in §8.4: the mean ablation effect was
   14 times smaller than its spread, so comparing it against exactly zero decided the question by
   rounding.

### 8.8 Self-check against the done-condition

| Requirement | Where satisfied |
|---|---|
| At least two methods, compared | §8.1 - mutual information with a per-feature permutation null, and permutation importance from a fixed forest, both over the same 17 features, with **Spearman +0.647** reported and the five rank gaps of 5+ explained in §8.2 |
| Kept features justified | §8.5 - 12 kept, each with its measured contribution; the five that only one method saw are justified by the ablation rather than by preference |
| Dropped features justified | §8.5 - 5 dropped, each named with the evidence that nominated it and the group ablation that confirmed the loss is within one sd of zero |
| A third method, so the decision is measured | §8.4 - the group ablation, three of five folds improving and a mean loss 14x smaller than the noise |
| The figure exists | `reports/figures/06_feature_importance.png`, 128 KB, both rankings on one pair of axes with each feature's own null drawn and dropped features hatched |
| Numbers reproducible, not typed | `python -m src.select` prints every figure in this section; the module runs **17 checks** on its own output, and one of them rebuilds the shipped 40-column matrix rather than trusting the count |

*Interpretation for the viva:* "Two methods that disagree are more useful than one that agrees with
itself, so I reported both and then settled the argument by measurement rather than by preference: drop
the five features that either method could not see, then check whether the model actually gets worse.
Held-out ROC-AUC moved by minus 0.0001 with a spread of 0.002, so the five went and the matrix dropped
from 65 columns to 40. What I will not claim is that this transfers to the XGBoost model T32 will
evaluate - a forest failing to use `job` is not proof that `job` is useless, which is exactly why the
ablation numbers are in the record."
