# DepositIQ — Data Dictionary

**Rubric:** Criterion 2 — Dataset & Data Understanding (10 marks)
**Task:** T10 | **Status:** complete | **Date:** 2026-10-27
**Sources:** `data/raw/bank-names.txt` (Moro, Laureano & Cortez, 2011), and for the additional set
`data/raw/bank-additional/bank-additional-names.txt` (Moro, Cortez & Rita, 2014). Every **type,
count, range and value list below was measured** from the files with `src/data_loader.py` — the
supplied documentation describes the *intended* schema, and where the two disagree the measured
value is used and the difference is flagged.

---

## 1. Datasets at a glance

Measured via `python -m src.data_loader`:

| Variant | File | Rows | Columns | Features | `y = yes` | Nulls | Exact duplicate rows |
|---|---|---|---|---|---|---|---|
| **`full`** (primary) | `bank-full.csv` | **45,211** | **17** | 16 | 5,289 (**11.70%**) | 0 | **0** |
| `sample` | `bank.csv` | 4,521 | 17 | 16 | 521 (11.52%) | 0 | 0 |
| **`additional`** | `bank-additional-full.csv` | **41,188** | **21** | 20 | 4,640 (**11.27%**) | 0 | **12** ⚠ |
| `additional_sample` | `bank-additional.csv` | 4,119 | 21 | 20 | 449 (10.90%) | 0 | 0 |

Both files are semicolon-separated (`sep=";"`) with quoted text values. Neither has a single true
`NaN` — "missingness" is hidden inside the literal string `"unknown"` and inside `pdays` sentinels.

⚠ **The additional variant contains 12 exact duplicate rows.** The primary variant has none. This
is recorded here and must be handled and reported in Phase 3 (T14); it is 0.03% of rows, so the
effect is small, but silently dropping or keeping them without saying so is worse.

---

## 2. Target variable

| Column | Type (measured) | Allowed values | Meaning | Leakage risk |
|---|---|---|---|---|
| `y` | object | `"no"`, `"yes"` (2 levels) | Did the client subscribe a **term deposit**? This is the outcome we predict. | **It IS the target.** Never used as a feature; encoded 1 = `"yes"` / 0 = `"no"` by `src/data_loader.load_xy()`. |

Class imbalance: **11.70% positive** in the primary set. A model predicting "no" for everybody
scores ≈88.3% accuracy and converts nobody — which is why accuracy is excluded as a headline
metric (decision D04).

---

## 3. Columns present in both variants (14)

These 14 appear in the classic **and** the additional set, though some are **coded differently**
between them — see the notes column and §6.

| Column | Type (measured) | Allowed values / range (measured) | Meaning | Leakage risk |
|---|---|---|---|---|
| `age` | int64 | classic 18–95 (mean 40.9); additional 17–98 (mean 40.0) | Client's age in years. | **None** — known before the call. |
| `job` | object | 12 levels: `admin.`, `blue-collar`, `entrepreneur`, `housemaid`, `management`, `retired`, `self-employed`, `services`, `student`, `technician`, `unemployed`, `unknown` (classic 288 unknown, additional 330) | Type of job. | **None.** Contains `unknown` — see §5. |
| `marital` | object | classic 3: `divorced`, `married`, `single`. Additional **4**: same plus `unknown` (80) | Marital status; `divorced` means divorced *or widowed*. | **None.** Additional set alone has `unknown`. |
| `education` | object | classic 4: `primary`, `secondary`, `tertiary`, `unknown` (1,857). Additional **8**: `basic.4y`, `basic.6y`, `basic.9y`, `high.school`, `illiterate`, `professional.course`, `university.degree`, `unknown` (1,731) | Highest education attained. | **None.** ⚠ **Coded differently between variants** — the classic set has no `illiterate` level and collapses the three `basic.*` levels into `primary`. |
| `default` | object | classic 2: `no`, `yes`. Additional **3**: plus `unknown` (8,597) | Has credit in default? | **None.** Additional set alone has `unknown`, and it is the largest unknown block there (20.9% of rows). |
| `housing` | object | classic 2: `no`, `yes`. Additional 3: plus `unknown` (990) | Has a housing loan? | **None.** |
| `loan` | object | classic 2: `no`, `yes`. Additional 3: plus `unknown` (990) | Has a personal loan? | **None.** |
| `contact` | object | classic 3: `cellular`, `telephone`, `unknown` (13,020). Additional **2**: `cellular`, `telephone` only | Contact communication type. | **None.** ⚠ Classic carries 13,020 `unknown` (28.8% of rows); the additional set dropped the category entirely. |
| `month` | object | classic **12** levels: `jan`, `feb`, `mar`, `apr`, `may`, `jun`, `jul`, `aug`, `sep`, `oct`, `nov`, `dec`. Additional **10**: same minus **`jan` and `feb`** | Last contact **month of year**. | **None** — known at call time. Carries seasonality (campaign timing), which is genuinely predictive and partly explains the June/September spikes in §6-style booking behaviour. ⚠ The additional set is missing two levels, so the two variants' month features are not directly comparable. |
| `duration` | int64 | 0 – 4,918 seconds (classic mean 258.2, median 180; additional mean 258.3) | **Last contact duration, in seconds.** | 🔴 **HIGH — POST-OUTCOME LEAK.** Only known *after* the call ends, and at that point the outcome is already known. The supplied documentation states it "highly affects the output target (e.g. if duration=0 then y='no')". **Excluded from the main model**; kept only for the documented with/without comparison (T22, T36). |
| `campaign` | int64 | classic 1–63 (mean 2.76); additional 1–56 (mean 2.57) | Number of contacts performed **during this campaign** for this client, including the last contact. | **None, with a caveat** — it is knowable at call time, but it accumulates as the campaign proceeds, so it partly encodes "the bank kept chasing". |
| `pdays` | int64 | classic **−1** – 871; additional **0** – **999** | Days since the client was **last contacted in a previous campaign**. | **None** — history, known before the call. ⚠ **Sentinel value, and the sentinel differs by variant** — see §5.2. |
| `previous` | int64 | classic 0–275 (mean 0.58); additional **0–7** (mean 0.17) | Number of contacts performed **before this campaign** for this client. | **None** — historical. ⚠ The additional variant's range is narrower (max 7 vs 275). |
| `poutcome` | object | classic 4: `failure`, `other`, `success`, `unknown` (36,959). Additional **3**: `failure`, `nonexistent`, `success` | Outcome of the **previous** marketing campaign. | **None** — history. ⚠ Encoded differently: the classic `unknown` corresponds to the additional `nonexistent`, and the classic set has an extra `other` category. |

These 14 + the 2 classic-only columns = the **16 features** of the primary set.
These 14 + the 6 additional-only columns = the **20 features** of the additional set.

---

## 4. Classic-only columns (2)

| Column | Type (measured) | Allowed values / range (measured) | Meaning | Leakage risk |
|---|---|---|---|---|
| `balance` | int64 | −8,019 – 102,127 (mean 1,362.3, median 448) | Average **yearly account balance**, in euros. | **None** — a pre-existing balance, known before the call. ⚠ Strongly right-skewed and **negative balances are legitimate** (overdrafts), not data errors. |
| `day` | int64 | 1 – 31 (mean 15.8) | Last contact **day of the month**. | **None** — known at call time. ⚠ Not meaningful on its own; it is a calendar position, not a quantity. |

⚠ **This is the most consequential difference between the variants:** the additional set has **no
`balance` column at all**. It substituted five macro-economic indicators for the client's own
account balance. So the classic set knows *how much money this client has*, while the additional set
knows *what the economy was doing* — they are complementary views, not one being a superset of the
other. It also means `balance`-based features created in Phase 5 cannot be transferred to the
additional variant.

`day` exists only in the classic set; the additional set replaces **`day` alone** with
`day_of_week`. It **keeps `month`** (see §3 and §5.1).

---

## 5. Additional-only columns (6)

### 5.1 Contact timing (1)

| Column | Type (measured) | Allowed values / range (measured) | Meaning | Leakage risk |
|---|---|---|---|---|
| `day_of_week` | object | 5 levels: `mon`, `tue`, `wed`, `thu`, `fri` | Last contact **day of the week**. | **None** — known at call time. Replaces the classic `day` column **only** — `month` is *retained* in the additional set (§3), so this variant actually carries both month and weekday. |

### 5.2 Social / economic context (5) — the "economic indicators"

These five are published national indicators from Banco de Portugal and **exist only in the
additional variant**. They describe the *macroeconomic environment* on the contact date, not the
client — so they are shared by many rows at once.

| Column | Type (measured) | Range (measured) | Meaning | Leakage risk |
|---|---|---|---|---|
| `emp.var.rate` | float64 | −3.4 – 1.4 (10 distinct values) | Employment variation rate — **quarterly** indicator. | **None** — published before the contact date. |
| `cons.price.idx` | float64 | 92.201 – 94.767 (26 distinct) | Consumer price index — **monthly** indicator. | **None** — published before the contact date. |
| `cons.conf.idx` | float64 | −50.8 – −26.9 (26 distinct) | Consumer confidence index — **monthly** indicator. Note it is **negative throughout**: that is the index's construction, not an error. | **None** — published before the contact date. |
| `euribor3m` | float64 | 0.634 – 5.045 (316 distinct) | Euribor 3-month rate — **daily** indicator. | **None** — published same-day/before. ⚠ With 316 distinct values it is near-continuous and highly correlated with time, so it can act as a proxy for "when in the 2.5 years this was". |
| `nr.employed` | float64 | 4,963.6 – 5,228.1 (11 distinct) | Number of employees — **quarterly** indicator. | **None** — published before the contact date. |

The supplied documentation notes these five "lead to substantial improvement in the prediction of a
success, **even when the duration of the call is not included**" — i.e. they partly compensate for
removing the leak. That is a reason to evaluate the additional variant as a genuine second
configuration rather than a pure superset.

---

## 6. Value-level notes: `unknown`, and the `pdays` sentinel

These are the two places where the dataset hides information inside ordinary-looking values.

### 6.1 `"unknown"` is a category, not a null

`df.isna().sum()` returns **0 for every column in both variants**. Missingness is encoded as the
string `"unknown"`. Measured counts:

| Column | Classic `unknown` | % of 45,211 | Additional `unknown` | % of 41,188 |
|---|---|---|---|---|
| `poutcome` | **36,959** | 81.7% | — (uses `nonexistent`) | — |
| `contact` | **13,020** | 28.8% | — (category absent) | — |
| `education` | 1,857 | 4.1% | 1,731 | 4.2% |
| `job` | 288 | 0.6% | 330 | 0.8% |
| `default` | — (no unknown level) | — | **8,597** | **20.9%** |
| `housing` | — | — | 990 | 2.4% |
| `loan` | — | — | 990 | 2.4% |
| `marital` | — | — | 80 | 0.2% |

*(—) means the level genuinely does not exist in that variant, which is itself a difference worth
documenting rather than assuming.*

Two of these are large enough to change modelling decisions: in the classic set `poutcome` is
81.7% `unknown` (for most clients there was no previous campaign), and in the additional set
`default` is 20.9% `unknown`. The treatment policy is decided in **T12**, not here — this section
only establishes the facts, including that the supplied documentation confirms these are genuine
missing values: *"There are several missing values in some categorical attributes, all coded with
the 'unknown' label."*

### 6.2 `pdays` is a **sentinel-encoded** column, and the sentinel differs by variant

| Variant | Sentinel | Meaning of sentinel | Count | Other ("real") values |
|---|---|---|---|---|
| `full` (classic) | **−1** | Client was **not previously contacted** | **36,954** (81.7% of rows) | 0 – 871 |
| `additional` | **999** | Client was **not previously contacted** | **39,673** (96.3% of rows) | 0 – 999 |

Treating this as a number is a real trap: the classic mean of `pdays` is **40.2** and the additional
mean is **962.5**, yet the *median* of both is the sentinel itself (−1 and 999). Both distributions
are therefore bimodal — sentinel, then a genuine gap of days. The intended fix is a boolean
`previously_contacted` flag plus a cleaned days-since-contact value, which is implemented in
**T12**. Also note the two variants use *different sentinels*, so any code that hard-codes −1 would
silently corrupt the additional variant.

---

## 7. Variant difference summary

| Aspect | `full` (classic) | `additional` |
|---|---|---|
| Rows × columns | 45,211 × 17 | 41,188 × 21 |
| Date coverage | May 2008 – Nov 2010 | May 2008 – Nov 2010 |
| Economic indicators | ✗ none | ✓ **five** (§5.2) |
| **Client account balance** | ✓ `balance` (€ −8,019 – 102,127) | ✗ **absent** |
| Contact timing | `day` (1–31) + `month` (12 levels) | `month` (10 levels — no `jan`/`feb`) + `day_of_week` (5 levels) |
| `education` levels | 4 (`primary`/`secondary`/`tertiary`/`unknown`) | 8 (incl. `illiterate`, three `basic.*`) |
| `marital` levels | 3 | 4 (+`unknown`) |
| `default`/`housing`/`loan` | binary, no `unknown` | ternary, **+`unknown`** |
| `contact` | 3 incl. `unknown` | 2 |
| `poutcome` | 4 (`other`, `unknown`) | 3 (`nonexistent`) |
| `pdays` sentinel | **−1** | **999** |
| `previous` range | 0 – 275 | 0 – 7 |
| Duplicate rows | 0 | **12** |
| Positive rate | 11.70% | 11.27% |

**Consequence for modelling:** the two variants are *not* interchangeable and are not a strict
superset/subset pair. Any pipeline must be fitted separately per variant, and the classic `pdays = −1`
convention cannot be reused for the additional set.

---

## 8. Leakage assessment (summary)

| Risk | Columns | Verdict |
|---|---|---|
| 🔴 **Direct outcome leak** | `duration` | Known only *after* the call; excluded from the main model, retained only for the documented comparison (T22, T36). |
| 🟡 **Proxy-for-time risk** | `euribor3m`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `nr.employed`, `month`, `day`/`day_of_week` | Not outcome leaks — all are known before the call. But they encode **when** the contact happened, and the rows are ordered by date, so they can let a model exploit campaign-era patterns. Because the planned split is random rather than chronological, this is a limitation to state, not a leak to fix. |
| 🟢 Known at call time | all remaining columns | Safe as features. |

---

## 9. Self-check against the T10 done-condition

| Requirement | Where satisfied |
|---|---|
| Every column of both variants in a table with all five fields | §3 (14 shared) + §4 (2 classic-only) = **16 features** of the primary set; §3 (14) + §5 (6) = **20 features** of the additional set; §2 covers the target. Each row has name, type, allowed values/range, meaning and leakage risk. |
| Type, allowed values, meaning, leakage risk present for each | Yes — the five columns of every table. |
| Quoted counts match what the loader reports | §1 — 45,211 × 17 and 41,188 × 21, produced by `src/data_loader.py`, matching the T06 verification. |
| The five economic columns identified as additional-only | §5.2 — `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`, marked "additional variant only" in the section heading and again in §7. |
| `duration` explicitly flagged as a post-outcome leak | §3 (`duration` row, 🔴) and §8. |
| `unknown` values and the `pdays` sentinel recorded as values with meanings | §6.1 (per-column `unknown` counts with meanings) and §6.2 (both sentinels, −1 and 999, with counts and the reason they matter). |

*Interpretation for the viva:* "The data looks complete — zero NaNs — but the missingness is
disguised: 81.7% of `poutcome` is the literal string `unknown`, and `pdays` uses −1 (or 999 in the
other variant) to mean 'never contacted before'. And `duration` is a trap: it predicts the target
almost perfectly, but only because it is measured after the call, so I exclude it from the main
model and show what it was buying."
