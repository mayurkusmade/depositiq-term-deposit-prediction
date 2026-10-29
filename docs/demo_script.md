# DepositIQ — Five-Minute Demo Script

**Purpose:** a timed walkthrough of the decision, the model, the live demo and the limits, for a marker,
supervisor or client who has five minutes and will ask questions afterwards.

**Length:** **5:00**, six segments. The spoken text is about **750 words**, which is a comfortable
**150 words per minute**; the script is written to be spoken, not read.

**Audience:** someone who understands a confusion matrix but has not seen this repository. Assume they
will remember one number and one picture, so make the number the price and the picture the live score.

**Setup:** run from the repository root with the virtual environment active. The demo is
`streamlit run app/app.py`; the headless preflight is `python -m src.demo`; the validator for this script
is `python -m src.demo_script`.

**Rule:** every number spoken in the script appears once in the claims table, and every claims-table row
names a file that holds it. A number that is not in the table is not said. A claim that is derived shows
its arithmetic rather than pointing at a record that does not hold it.

---

## At a glance

| # | Time | Segment | On screen | Spoken words |
|---|---|---|---|---|
| 1 | 0:00–0:30 | Open with the decision | title slide, then the price | ~75 |
| 2 | 0:30–1:10 | The data and the leak | `01_target_distribution.png`, `05_leakage_audit.md` | ~100 |
| 3 | 1:10–2:00 | The model and the price | `07_threshold_curve.png`, `07_pr_curve.png` | ~120 |
| 4 | 2:00–3:20 | The live demo | `app/app.py`, `10_shap_local.png` | ~190 |
| 5 | 3:20–4:20 | What it is worth | `08_lift_gain.png`, `docs/report.pdf` §7 | ~150 |
| 6 | 4:20–5:00 | Limits and close | `docs/report.pdf` §10, `08_recommendations.md` | ~110 |
| | **5:00** | **Total** | | **~745** |

---

## Before you start

Run these two commands before the clock starts; neither is part of the five minutes.

```bash
python -m src.demo            # preflight: 22 checks, about 5 seconds
streamlit run app/app.py      # the live demo, from the repository root
```

- [ ] `python -m src.demo` exits **0** and prints **22/22** checks.
- [ ] The browser shows the default client, the score and the three exemplar presets.
- [ ] `docs/report.pdf`, `docs/presentation.pptx` and the figures named in the table are open in tabs.
- [ ] The timer is visible and set to **5:00**.

---

## The script

### 0:00–0:30 — Open with the decision

**Say.** Every day this bank has more clients than agents. The decision is not "will this client
subscribe?" in the abstract — it is "which clients do we call today?" This model ranks the file by the
probability of a subscription and stops at a price: one call costs 1.50 EUR, one captured deposit is
assumed to be worth 20.00 EUR, so the break-even probability is 7.50%. The campaign works down the list
until that price.

**Show.** The title slide, then the price line in `docs/report.pdf` §0.

**Evidence.** `models/threshold_choice.json`; `docs/report.md` §0; `docs/presentation.pptx` slide 1.

### 0:30–1:10 — The data and the leak

**Say.** The data is the UCI Bank Marketing file: 45,211 clients, 17 columns, 11.70% subscribers. There
are no nulls, which is the first trap: missingness arrives as the word "unknown". The second trap is
duration — the length of the call. It is recorded after the call ends, so it cannot be an input. If we
used it, PR-AUC would rise by 0.1788 and the campaign would appear to earn 20,340.00 EUR more, but that
model would be reading the answer off the call it is supposed to decide whether to make. So duration is
excluded, and the exclusion is measured, not asserted.

**Show.** `reports/figures/01_target_distribution.png`, then the duration table in
`docs/05_leakage_audit.md` §5.

**Evidence.** `models/duration_comparison.json`; `reports/figures/01_target_distribution.png`;
`docs/05_leakage_audit.md` §5–§6.

### 1:10–2:00 — The model and the price

**Say.** Seventeen features became twelve, chosen by mutual information with a permutation null and by
permutation importance; the five dropped features cost less than one standard error of ROC-AUC. The model
is an XGBoost with 300 rounds, depth 5, tuned over 40 of 4,800 points. Cross-validated PR-AUC is 0.4460
against a 0.1170 base rate. On clients it never saw, PR-AUC is 0.4616 and ROC-AUC is 0.8052, so the
ranking holds. The cut is not 0.5 and not the empirical peak; it is the price, 0.0750. At 0.5 the model
would call under 5% of the list and miss most subscribers.

**Show.** `reports/figures/07_threshold_curve.png`, then `reports/figures/07_pr_curve.png`.

**Evidence.** `models/metrics_cv.json`; `models/holdout_eval.json`;
`reports/figures/07_threshold_curve.png`; `docs/report.md` §6–§7.

### 2:00–3:20 — The live demo

**Say.** Let me show the artifact, not a slide. The form asks for eleven facts a bank knows before a
call. It does not ask for duration, because the call has not happened. The default client scores 7.22% —
just below the 7.50% price, so the app says do not call, and shows why: the top positive contribution is
contact = cellular at +0.2545, the top negative is month = may at −0.2504. Change one field and the score
moves.

This client scores 38.19% and is called; the model is right. This one scores 11.33% and is called; the
model is wrong, but at this price that is expected: an average client clears 7.50%, and this is the modal
cell — 30.58% of the file. This one scores 5.50% and is not called; the model misses a subscriber. The
point of showing all three is that the app exposes its errors, not just its wins.

**Show.** `streamlit run app/app.py`: the default client, then the true-positive, false-positive and
false-negative presets in that order. Keep `reports/figures/10_shap_local.png` as the fallback.

**Evidence.** `models/demo.json`; `reports/figures/10_shap_local.png`; `docs/report.md` §8.2.

### 3:20–4:20 — What it is worth

**Say.** What is it worth? On the training half, calling everyone nets 30,368.00 EUR; the ranked list at
7.50% nets 43,621.00 EUR, a gain of 13,253.00 EUR on 21,302 fewer calls. Held out, the same policy calls
3,621 clients, reaches 821 subscribers — 77.60% recall — and nets 10,988.50 EUR against 7,595.50 for
calling everyone. The top decile lifts 4.5100 times the base rate.

Three cautions. The prices are assumptions; every euro is linear in the 20.00 EUR deposit value. The cut
is below the 11.6982% base rate, so this converts a fixed call budget; it is not evidence to cut spend.
And accuracy is a trap: always saying no scores 88.3003%, while the model scores 66.42% and finds three
subscribers in four.

**Show.** `reports/figures/08_lift_gain.png`, then the net-value table in `docs/report.pdf` §7.3.

**Evidence.** `models/threshold_choice.json`; `models/lift_results.json`;
`reports/figures/08_lift_gain.png`; `docs/report.md` §7.3–§7.4.

### 4:20–5:00 — Limits and close

**Say.** Three things I would fix next. The split is stratified-random, not temporal, so "generalises"
means unseen clients of this campaign, not next year. Fairness is not audited beyond segment-level
precision and recall. And the value of a subscription is an assumption the bank should replace with its
own number before anyone quotes the euro totals.

The playbook has eight actions, each with the number that would prove it wrong; the first is to make
contact a required field, because 28.72% of clients are scored on a blank. The model is not a crystal
ball. It is a priced ranking, and every number in it is in a file.

**Show.** `docs/report.pdf` §10, then the eight-row table in `docs/08_recommendations.md` §1.

**Evidence.** `models/playbook.json`; `docs/report.md` §10; `docs/08_recommendations.md` §1 and §4.

---

## Claims and where they live

Every number spoken above appears once in this table. `Kind` is `record` when the value is stored in the
named file, and `derived` when the file holds the operands and the arithmetic is shown.

| # | Claim | Value | Source | Kind | Derivation |
|---|---|---|---|---|---|
| 1 | Clients in the modelling file | 45,211 | `data/raw/bank-full.csv` | record | — |
| 2 | Subscribers in the file | 11.70% | `models/threshold_choice.json` | record | — |
| 3 | Cost of one call | 1.50 | `models/threshold_choice.json` | record | — |
| 4 | Value of one deposit | 20.00 | `models/threshold_choice.json` | record | — |
| 5 | Break-even probability | 7.50% | `models/threshold_choice.json` | record | — |
| 6 | The cut as a probability | 0.0750 | `models/threshold_choice.json` | record | — |
| 7 | PR-AUC the leak would add | 0.1788 | `models/duration_comparison.json` | record | — |
| 8 | Euros the leak would add | 20,340.00 | `models/duration_comparison.json` | record | — |
| 9 | Cross-validated PR-AUC | 0.4460 | `models/metrics_cv.json` | record | — |
| 10 | Training base rate | 0.1170 | `models/metrics_cv.json` | record | — |
| 11 | Held-out PR-AUC | 0.4616 | `models/holdout_eval.json` | record | — |
| 12 | Held-out ROC-AUC | 0.8052 | `models/holdout_eval.json` | record | — |
| 13 | The conventional cut | 0.5 | `models/holdout_eval.json` | record | — |
| 14 | Default client score | 7.22% | `models/demo.json` | record | — |
| 15 | `contact = cellular` contribution | 0.2545 | `models/demo.json` | record | — |
| 16 | `month = may` contribution | −0.2504 | `models/demo.json` | record | — |
| 17 | True-positive exemplar | 38.19% | `models/demo.json` | record | — |
| 18 | False-positive exemplar | 11.33% | `models/demo.json` | record | — |
| 19 | False-negative exemplar | 5.50% | `models/demo.json` | record | — |
| 20 | False positives are the modal cell | 30.58% | `models/shap_local.json` | derived | 11,059 / 36,168 * 100 |
| 21 | Calling everyone, training | 30,368.00 | `models/threshold_choice.json` | record | — |
| 22 | Ranked campaign, training | 43,621.00 | `models/threshold_choice.json` | record | — |
| 23 | Gain over calling everyone | 13,253.00 | `models/threshold_choice.json` | derived | 43,621.00 - 30,368.00 |
| 24 | Fewer calls for that gain | 21,302 | `models/threshold_choice.json` | derived | 36,168 - 14,866 |
| 25 | Held-out calls at the price | 3,621 | `models/threshold_choice.json` | record | — |
| 26 | Held-out subscribers reached | 821 | `models/threshold_choice.json` | record | — |
| 27 | Held-out recall | 77.60% | `models/holdout_eval.json` | derived | 821 / 1,058 * 100 |
| 28 | Held-out net at the price | 10,988.50 | `models/threshold_choice.json` | record | — |
| 29 | Held-out net calling everyone | 7,595.50 | `models/threshold_choice.json` | record | — |
| 30 | Top-decile lift, held out | 4.5100 | `models/lift_results.json` | record | — |
| 31 | Training base rate, four decimals | 11.6982% | `models/threshold_choice.json` | record | — |
| 32 | Always-no accuracy | 88.3003% | `models/holdout_eval.json` | record | — |
| 33 | Accuracy at the price | 66.42% | `models/holdout_eval.json` | record | — |
| 34 | Clients with no recorded channel | 28.72% | `models/playbook.json` | record | — |

---

## Three refusals and one scope sentence

Say these in the spoken track, not only in the slides:

- **`duration` is never an input.** It is recorded after the call ends; the 20,340.00 EUR it would add is
  a diagnosis of leakage, not a gain.
- **Accuracy is never a headline.** Always saying no scores 88.3003% and finds nobody; the model scores
  66.42% at the price and finds **77.60%** of subscribers.
- **`month` is never read as a cause.** It is a calendar signal for scheduling, not a lever to move a
  client into a better month.
- **The scope sentence:** the model **converts a fixed call budget**, and is **not evidence that the bank
  should cut campaign spend**.
- **The price identity:** **0.0750 = C / V = 1.50 / 20.00**.

---

## If you only remember five numbers

| Number | What it is | Record |
|---|---|---|
| **0.0750** | the price cut, `C / V = 1.50 / 20.00` | `models/threshold_choice.json` |
| **0.4460** | cross-validated PR-AUC of the shipped model | `models/metrics_cv.json` |
| **0.4616** | held-out PR-AUC, the better estimate of the same quantity | `models/holdout_eval.json` |
| **77.60%** | held-out recall at the price cut | `models/holdout_eval.json` |
| **10,988.50 EUR** | held-out net value at the price cut | `models/threshold_choice.json` |

---

## Rehearsal checklist

### The day before

- [ ] Run `python -m src.demo` and confirm **22/22** checks.
- [ ] Run `python -m src.demo_script` and confirm every check passes.
- [ ] Open `docs/report.pdf`, `docs/presentation.pptx` and the figures named above in the order used.
- [ ] Rehearse once with a timer; target **4:45–5:00**, not 5:30.
- [ ] Confirm the Streamlit app loads and all three exemplar presets render.
- [ ] Confirm `models/final_model.joblib` matches the sha256 in `models/final_model.json`.

### Five minutes before

- [ ] Start `streamlit run app/app.py` and leave the browser on the default client.
- [ ] Close heavy applications; the demo is local and needs no network.
- [ ] Keep `models/demo.json`, `models/threshold_choice.json` and `docs/viva_qa.md` open in tabs.
- [ ] Set a visible timer at **5:00** and decide in advance which segment to cut if late.

### During the five minutes

- [ ] Say the price **before** the first euro figure.
- [ ] Say "assumption" whenever C or V is mentioned.
- [ ] Show the false positive and explain why it is expected at this price.
- [ ] If a number is challenged, open the record rather than re-deriving it from memory.

### If something breaks

- [ ] If Streamlit is slow, switch to `python -m src.demo` and narrate the same numbers.
- [ ] If the app fails, use `docs/presentation.pptx` slides 4–9 and `docs/report.pdf` §7.
- [ ] If a file is requested, give the path from the claims table, not a paraphrase.
- [ ] If time is short, drop the model detail in segment 3, not the demo or the limits.

### After

- [ ] Note which segment ran long and which question was hardest.
- [ ] Add any new question to `docs/viva_qa.md` before the next rehearsal.

---

## Hand-off to questions

The full bank is [`viva_qa.md`](viva_qa.md), with **37** questions across the nine rubric criteria.

| If they ask about | Go to |
|---|---|
| the decision, the price or the scope sentence | Q1–Q4 |
| the data, the variants or the missing values | Q5–Q8 |
| cleaning, outliers or the split | Q9–Q12 |
| the EDA or the campaign tail | Q13–Q16 |
| `duration`, feature selection or the 12 features | Q17–Q20 |
| the model choice, imbalance or tuning | Q21–Q24 |
| PR-AUC, the held-out half or the threshold | Q25–Q29 |
| SHAP, the exemplars or the eight recommendations | Q30–Q33 |
| reproducibility, the clean clone or the limitations | Q34–Q37 |
