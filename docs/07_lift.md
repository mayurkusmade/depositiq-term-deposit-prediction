# DepositIQ — Lift, Gain and Deciles

**Rubric:** Criterion 7 — Evaluation & Business Translation (15 marks)
**Task:** T35 | **Status:** complete
**Date:** 2026-10-30
**Source of every number here:** the run of `python -m src.lift`, recorded in `models/lift_results.json` and
tabulated in `reports/lift_table.csv`. Nothing below was computed by hand (D06).

---

## 1. What this step answers, and what it deliberately does not

T34 fixed the operating threshold at **0.0750** and reported it in euros. A campaign manager works in
**calls**, and is asked a different question: *is this list better than working down the customer file?*

Two standard answers, both produced here:

* **lift** — in the best 10% of the list, how many times the base subscription rate do we find?
* **gain** — what share of all subscribers do we reach by calling the best X%?

Neither needs a threshold, which makes this the one Phase 7 step that can report on the held-out rows
**without spending them** (D53, D54): the deciles are of the ranking, and a ranking is threshold-free by
construction. The threshold enters once, at the end, where 0.0750 is translated into calls saved.

**D15's declared KPI was top-decile lift > 2×**, and it is measured here rather than assumed.

## 2. Two halves, two base rates, never pooled

| half | rows | subscribers | base rate |
|---|---|---|---|
| training (out-of-fold) | 36,168 | 4,231 | **11.6982%** |
| held out (persisted artifact) | 9,043 | 1,058 | **11.6997%** |

The two rates differ by **1.5 basis points**. They are also not the same number, and lift is a ratio *to a
base rate*, so a pooled denominator would belong to neither half. Every row in the CSV carries its half and
divides by that half's own rate; a check requires the two to differ and each to match its record.

The training ranking is **T34's ranking**, not a second one: it is produced by `threshold.oof_scores`, the
same five fold fits, which reproduce T30's tuned arm to `0.0e+00` (D45's one-harness rule). The held-out
ranking is `final_model.joblib` scoring rows it has never seen.

## 3. The decile table

Ten positional blocks of the descending score — "the best 3,617 clients", not "clients above some
probability". Positional blocks survive ties and a skewed distribution unchanged; the cost is that block
sizes differ by a row (36,168 does not divide by ten).

**Training half, out-of-fold**

| decile | rows | subscribers | precision | lift | cumulative gain | cumulative lift | calls saved |
|---|---|---|---|---|---|---|---|
| 1 | 3,617 | 1,847 | 0.5106 | **4.37×** | 43.65% | 4.37× | 12,172 |
| 2 | 3,617 | 766 | 0.2118 | 1.81× | 61.76% | 3.09× | 15,103 |
| 3 | 3,616 | 390 | 0.1079 | 0.92× | 70.98% | 2.37× | 14,821 |
| 4 | 3,617 | 262 | 0.0724 | 0.62× | 77.17% | 1.93× | 13,443 |
| 5 | 3,617 | 238 | 0.0658 | 0.56× | 82.79% | 1.66× | 11,861 |
| 6 | 3,617 | 209 | 0.0578 | 0.49× | 87.73% | 1.46× | 10,030 |
| 7 | 3,617 | 163 | 0.0451 | 0.39× | 91.59% | 1.31× | 7,807 |
| 8 | 3,616 | 149 | 0.0412 | 0.35× | 95.11% | 1.19× | 5,464 |
| 9 | 3,617 | 129 | 0.0357 | 0.30× | 98.16% | 1.09× | 2,950 |
| 10 | 3,617 | 78 | 0.0216 | 0.18× | 100.00% | 1.00× | 0 |

**Held out, artifact**

| decile | rows | subscribers | precision | lift | cumulative gain | cumulative lift | calls saved |
|---|---|---|---|---|---|---|---|
| 1 | 904 | 477 | 0.5277 | **4.51×** | 45.09% | 4.51× | 3,173 |
| 2 | 905 | 182 | 0.2011 | 1.72× | 62.29% | 3.11× | 3,824 |
| 3 | 904 | 93 | 0.1029 | 0.88× | 71.08% | 2.37× | 3,715 |
| 4 | 904 | 69 | 0.0763 | 0.65× | 77.60% | 1.94× | 3,400 |
| 5 | 905 | 66 | 0.0729 | 0.62× | 83.84% | 1.68× | 3,059 |
| 6 | 904 | 42 | 0.0465 | 0.40× | 87.81% | 1.46× | 2,514 |
| 7 | 904 | 40 | 0.0442 | 0.38× | 91.59% | 1.31× | 1,952 |
| 8 | 904 | 37 | 0.0409 | 0.35× | 95.09% | 1.19× | 1,365 |
| 9 | 905 | 29 | 0.0320 | 0.27× | 97.83% | 1.09× | 707 |
| 10 | 904 | 23 | 0.0254 | 0.22× | 100.00% | 1.00× | 0 |

![Lift by decile, cumulative gain, and calls saved at equal recall](../reports/figures/08_lift_gain.png)

## 4. What the concentration means

**The top decile finds 4.37× the base rate on the training half and 4.51× held out** — D15's >2× target is
cleared by a factor of two, and the two halves agree to 0.14×, which is the transfer that T33 and T34 kept
insisting on. **The top two deciles hold 61.76% of all subscribers** (held out: 62.29%).

**The third decile already sits below random** — a lift of **0.92×** (held out: **0.88×**). This is the
finding a lift table exists to expose: past roughly the best fifth of the list, the model's message is
*mostly negative*. Clients in decile 3 are **less** likely to subscribe than a client picked at random from
the same data. The ranking is therefore doing two jobs, and the second one is the one a bank rarely
monetises: it says who **not** to call, and it says it about 80% of the file.

That is also why the 10th decile matters. It holds **0.18×** — only **78 of its 3,617 clients** subscribe
(2.16% against an 11.7% base rate): calling it last is not a tie-break, it is a different decision from
calling it at all.

## 5. The adopted cut, in calls

| half | calls | coverage | recall | lift | calls saved | EUR not spent |
|---|---|---|---|---|---|---|
| training (OOF) | 14,866 | 41.10% | **77.90%** | 1.90× | **13,309** | 19,964 |
| held out | 3,621 | 40.04% | **77.60%** | 1.94× | **3,396** | 5,094 |

Read the last two columns carefully, because they are the answer to "what does the model save you": a
campaign that reached **77.90%** of subscribers by calling down an arbitrarily ordered file would have to
call **77.90% of it** — 28,175 clients. The ranking reaches the same subscribers with **14,866**. The
difference, **13,309 calls (36.8% of the list, 19,964 EUR at 1.50 EUR per call)**, is what the model is worth
before a single subscriber is credited to it. Held out the same arithmetic gives **3,396 calls / 5,094 EUR**.

The check that this is T34's cut and not a re-reading of it: the calls and subscribers here
(14,866 / 3,296) are **identical** to T34's record.

### Calls needed for a target recall

| target recall | calls needed | coverage | a random list needs | calls saved |
|---|---|---|---|---|
| 25% | 1,738 | 4.81% | 9,042 | 7,304 |
| 50% | 4,472 | 12.36% | 18,084 | 13,612 |
| 75% | 13,061 | 36.11% | 27,126 | 14,065 |
| 90% | 24,125 | 66.70% | 32,551 | 8,426 |

## 6. Three different optima on the same curve — and why the cut is at the last of them

The savings column does not peak at the adopted cut. It peaks much earlier:

| what is being maximised | coverage | calls | value |
|---|---|---|---|
| **calls saved** (efficiency vs the average client) | **18.89%** | 6,833 | 15,188 calls / 22,781 EUR |
| money (T34's empirical peak, 0.0821) | 36.63% | 13,247 | 44,010 EUR |
| **the price, 0.0750 — adopted** | **41.10%** | 14,866 | 43,621 EUR / 13,309 calls saved |

Three points, one ranking, and each is the answer to a different question. The reason is visible in the
arithmetic: writing `saved(k) = gain(k) × N − k` gives a marginal step of `(N / positives) × y[k] − 1`, so
**each call adds a saving when the client subscribes and costs one call when they do not**. The saving
therefore peaks where the **local** subscription rate crosses the **list average** (11.7%) — which happens
inside decile 3. Money peaks where the local rate crosses the **price** (7.5%) — inside decile 4. Efficiency
against the average client and value against the marginal client are different crossings, and the curve is
steep enough between them to matter.

**So why is the campaign not stopped at 18.9%?** Because the clients between the savings peak and the cut
still pay:

| half | calls between the two | subscribers | marginal rate | vs the 7.50% price | net |
|---|---|---|---|---|---|
| training | 8,033 | 720 | **8.96%** | above | **+2,350 EUR** |
| held out | 1,547 | 120 | **7.76%** | above | +79.50 EUR |

Those calls are *inefficient* — they lower the average, which is why the savings curve falls — and they are
still *profitable*, which is why the campaign runs through them. Held out the margin is thin (**+79.50 EUR**,
7.76% against 7.50%), which is the same "the price sits at the aggressive edge" conclusion T34 reached from
the other direction.

The savings peak is also **outside** T34's 1%-of-peak money plateau (27.6%–41.4% coverage), which is the
cleanest statement of the point: maximising calls saved is not a money decision, and it is not presented as
one.

## 7. Reconciliation with T32

T32 published two multipliers for this model, measured on its cross-validated cut table. This module derives
both from one ranking:

| T32 point | coverage | published | measured here | difference |
|---|---|---|---|---|
| nested cut | 13.2604% | 3.9247× | **3.9488×** | +0.0241 |
| business cut | 41.1027% | 1.8961× | **1.8991×** | +0.0030 |

They agree to within 0.024× and 0.003×. The second is nearly exact because **0.0750 *is* 41.10% coverage**
on this list — the business cut and the price are the same point, so the two tasks are measuring the same
thing from opposite ends. The first is close rather than exact because T32 averaged a per-fold cut
(13.26% ± 0.39% coverage per fold) while this is one pooled ranking; the interpolation between decile edges
is stated in the module rather than hidden.

## 8. The self-test: no lift when there is no ranking

A lift table is easy to get subtly wrong — an off-by-one block edge or a reversed sort still produces
plausible numbers — so the module reruns its own machinery on **shuffled scores** on every invocation:

* the ten lifts run **0.950–1.108**, mean **1.000001**;
* the gain curve departs from the diagonal by at most **0.0060**;
* the calls-saved column departs from zero by at most **218** calls out of 36,168.

That is the code being tested, not the model, and it is the reason the numbers in §3 can be trusted to mean
what their labels say.

## 9. What this does not prove

* **Lift is not value.** A 4.37× decile is a statement about a *rate*; the euros are in T34 and they depend on
  `C` and `V`. This document prices calls saved at 1.50 EUR each only to make the count legible.
* **Deciles are coarse by design.** Ten blocks cannot locate the savings peak to the call; §6 finds it on the
  exact cumulative curve for that reason, and the decile table and the peak therefore differ by design
  (14,821 at 30% coverage against 15,188 at 18.89%).
* **The held-out table is one draw of 1,058 subscribers.** A decile of 904 rows carries a standard error of
  about **0.017** on a precision near 0.53 (`sqrt(0.5277 x 0.4723 / 904)`), so the 4.51× and 4.37× top
  deciles are the same number within noise - which is why the two halves are quoted side by side instead of
  one being called right and the other wrong.
* **No capacity constraint is applied.** §5 answers "what does targeting save"; if agents are the limit, T34's
  capacity table is the right table, not this one.

## 10. Verification

`python -m src.lift` exits **0** with **17/17** checks in **4.3 s**, writing
`reports/lift_table.csv` (20 rows × 13 columns), `reports/figures/08_lift_gain.png` (**231,645 B**) and a
**20,661-byte** record that carries both decile tables, the translation, the savings peak, the marginal
block, the recall table, the reconciliation and the self-test.

| check | what it establishes |
|---|---|
| each half against its own base rate | 11.6982% vs 11.6997%, never pooled |
| the ten blocks cover every row once, descend, differ by ≤1 | 36,168 and 9,043 both partition exactly |
| precision = subscribers/rows and lift = precision/base | all 20 rows, drift `0.0e+00` |
| cumulative columns are running sums ending at 100% | both halves end at 1.000000 |
| top decile clears D15's 2× | 4.3652× training, 4.5100× held out |
| the highest decile holds the most lift, the list ends worse than it starts | 4.37× → 0.18×, 0 of 9 steps not descending |
| a shuffled ranking reports no lift | mean 1.000001, gain within 0.0060 of the diagonal |
| the cumulative curve lands on T32's two multipliers | +0.0241× and +0.0030× |
| this reading of the cut is T34's | 14,866 calls / 3,296 subscribers, identical |
| calls saved = the recall a fixed list needs − the calls made | 13,309 and 3,396, drift `0.0e+00` |
| higher recall costs more calls | 1,738 → 24,125, strictly increasing |
| the savings peak precedes the cut and is a local maximum | 18.89% vs 41.10% |
| the block between them converts above the price | 8.96% and 7.76% against 7.50% |
| the held-out ranking is the artifact's | 9,043 scores in [0.004575, 0.940095], no refit |
| the CSV has both halves and its columns | 20 rows × 13 columns |
| the figure was written | 231,645 B, inspected |
| the record on disk is this run's | read back and compared |

**Next:** T36 runs the with-vs-without `duration` comparison at this operating point, and is the last task
in Phase 7.
