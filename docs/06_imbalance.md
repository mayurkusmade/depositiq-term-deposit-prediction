# DepositIQ — Imbalance Handling

**Rubric:** Criterion 6 — Model Selection & Implementation (15 marks)
**Task:** T29 | **Status:** complete | **Date:** 2026-10-29
**Source of every number here:** `python -m src.imbalance`, recorded in
`models/imbalance_results.json`. Nothing below was typed from a rendered table (D06).

---

## 1. The experiment

The positive class is **11.70%** of the training half (4,231 of 36,168). Every model so far handles that
with **weights** and none of them was tested against the alternative. This is that test: **one estimator,
one preprocessor, one fold split**, and the treatment toggled.

The estimator is XGBoost — D08's family and the best model in `models/cv_results.json` — because the
question is about the *treatment*, and changing the estimator at the same time would answer a different
question. Its parameters are read from `src.train` and only `scale_pos_weight` is changed, so the arms
cannot drift in depth, rounds or learning rate.

| arm | treatment | what it does |
|---|---|---|
| `weights` | `scale_pos_weight=7.548334` | reweights the positive class in the loss — the shipped configuration |
| `none` | `scale_pos_weight=1` | no treatment at all — the control |
| `smote` | `SMOTE(random_state=42)` inside the pipeline | synthesises minority rows in the **fitted fold only** |

**`none` is not padding.** Without it, a difference between `weights` and `smote` cannot be attributed to
either one — it could be the difference between *treating* and *not treating*, which is a third arm, not a
comparison of two.

---

## 2. Where SMOTE is allowed to run, and how that is proved

SMOTE sits **between** the preprocessor and the estimator in an `imblearn.pipeline.Pipeline`, so it is
fitted on the fold's training rows and applied to nothing else. The wrong versions do not announce
themselves:

* resampling the frame **before** `StratifiedKFold` would put synthetic rows in the validation half, and
  every score would be inflated by construction;
* resampling **after** the split but outside the pipeline would still let a different fold's synthetic rows
  enter the fit;
* resampling **before** the preprocessor would fit the scaler and the one-hot vocabulary on synthetic rows,
  changing the matrix that every other model in this project was scored on.

Three checks test the arithmetic rather than the intention:

| check | measured |
|---|---|
| SMOTE adds exactly `negatives − positives` rows, on the fold's training rows | **28,934 → 51,098** rows, added **22,164** = 25,549 − 3,385, ending at 25,549 positives |
| the fitted scaler still matches the **fold's own** training means | matches to **0.0e+00**; the *resampled* matrix's means are **4.09e+01** away — exactly what a leak would have produced |
| no arm adds rows at prediction time | every fold returns its **7,234 / 7,233** held-out scores in all three arms |

---

## 3. Results, 5-fold stratified CV, nested F1

The threshold rule is **leave-one-fold-out**: each fold is scored at the F1-maximising cut computed on the
**other four folds'** out-of-fold predictions. It costs no extra fits and no fold is judged at a cut fitted
to itself. §4 explains why this is not optional.

| arm | PR-AUC | ROC-AUC | F1 (nested) | F1 @ 0.5 | mean cut | mean p | Brier |
|---|---|---|---|---|---|---|---|
| `weights` | 0.4420 ± 0.0134 | **0.7993 ± 0.0089** | 0.4848 ± 0.0118 | 0.4470 | 0.6050 | 0.3892 | 0.1637 |
| `none` | **0.4444 ± 0.0131** | 0.7993 ± 0.0079 | **0.4878 ± 0.0118** | 0.3280 | 0.1974 | **0.1168** | **0.0821** |
| `smote` | 0.4152 ± 0.0234 | 0.7799 ± 0.0105 | 0.4524 ± 0.0222 | 0.4490 | 0.5279 | 0.2967 | 0.1221 |

Against the **0.1170** base rate: `weights` **3.78×**, `none` **3.80×**, `smote` **3.55×**.

Paired per fold against the shipped treatment:

| arm | Δ PR-AUC | Δ ROC-AUC | Δ F1 (nested) |
|---|---|---|---|
| `none` | **+0.0023 ± 0.0021**, 5/5 folds improve | −0.0000 ± 0.0011, 2/5 | **+0.0030 ± 0.0039**, 4/5 |
| `smote` | **−0.0268 ± 0.0118**, **0/5** folds improve | **−0.0194 ± 0.0028**, **0/5** | **−0.0324 ± 0.0150**, **0/5** |

---

## 4. The F1 trap, measured twice

Reading F1 at the conventional 0.5 cut produces the **opposite verdict** to the honest one:

* at **0.5**, `weights` scores **0.4470** and `none` scores **0.3280** — a **11.9-point** win for weighting;
* at each arm's **leave-one-fold-out** cut, `none` scores **0.4878** and `weights` **0.4848** — a small win
  for *no treatment*.

Neither arm changed; only the cut did. The mechanism is arithmetic: `scale_pos_weight` multiplies the odds
of the positive class by 7.55, so the weighted model's probabilities sit around **0.39**, and 0.5 falls near
its own F1-optimal region by accident. The untreated model's probabilities sit at **0.117**, so a 0.5 cut
flags almost nothing and its recall collapses. **An F1 at a fixed cut ranks the arms by where their
calibration happens to land, not by what they predict.** D45 predicted this from the T26 numbers; here it
is measured, with the arms unchanged and the verdict reversed.

This is also why `smote` looks respectable at 0.5 (**0.4490**, the highest of the three): its mean predicted
probability is **0.2967**, the closest of the three to 0.5, so the naive cut flatters it most.

---

## 5. SMOTE loses to doing nothing, in every fold

This is the clearest result in the phase. SMOTE is worse than the untreated control on **all three**
metrics, in **all five** folds:

| metric | `weights` → `smote` | standard errors | folds improving |
|---|---|---|---|
| PR-AUC | 0.4420 → 0.4152 (**−6.07% relative**) | 2.27 | **0 / 5** |
| ROC-AUC | 0.7993 → 0.7799 (−2.43%) | **6.82** | **0 / 5** |
| F1 (nested) | 0.4848 → 0.4524 | 2.16 | **0 / 5** |

The ROC-AUC loss is **6.8 standard errors** — there is no reading of this where SMOTE is a close call. It
is also **much slower** (28.3 s against 9.3 s for the same five folds — **3.0×**) and **less stable** (PR-AUC sd 0.0234
against 0.0134).

Why it fails here is worth stating, because the mechanism is not "SMOTE is bad in general":

* **The synthetic rows are interpolations between real ones, and this dataset's positives are not
  clustered.** With 3,385 positives spread across 40 one-hot levels and a strong `poutcome`/`contact`
  structure, a straight line between two positives frequently crosses a region where the true rate is near
  the base rate, so the synthesised label is not the label that region deserves.
* **The classifier does not need more positives to find them.** `none` at ROC-AUC 0.7993 shows the ranking
  signal was already available; SMOTE spends its information budget re-balancing a problem the tree
  ensembles could already separate.
* **The one-hot matrix makes the interpolation worse**, because a midpoint between `month=may` and
  `month=oct` is a vector that is half in each level — a value that corresponds to no client at all.

---

## 6. The weight is not earning its place either

The comparison that was not asked for but matters more:

| | `weights` | `none` | verdict |
|---|---|---|---|
| PR-AUC | 0.4420 | **0.4444** (+0.0023, 5/5 folds, 1.11 SE) | no detectable loss without the weight, and the sign favours dropping it |
| ROC-AUC | 0.7993 | 0.7993 (−0.00003) | **identical** |
| F1 (nested) | 0.4848 | **0.4878** | marginally better without weights |
| mean predicted | 0.3892 | **0.1168** | `none` lands on the base rate (**0.1170**) to within **0.0002** |
| Brier | 0.1637 | **0.0821** | **half the error** — and better than the constant from T26 (**0.1033**) |

So the treatment that every model in this project uses **buys nothing measurable in ranking** (ROC-AUC is
identical to four decimals; PR-AUC moves +0.0023 at 1.11 standard errors, with the sign against it) and
**costs half the Brier score**. The untreated XGBoost is both the better ranker and, without any
recalibration at all, a nearly perfectly calibrated probability model.

That matters for T34: if the shipped model carries no `scale_pos_weight`, D17's **7.50%** break-even can be
applied to its output **as a probability**, which is what the cost model always assumed and what every
weighted model in `docs/06_modeling.md` could not offer.

---

## 7. Verification

* `python -m src.imbalance` exits **0** with **10/10** checks — the fold split is replayed, the arms are
  checked to differ only in the treatment (`scale_pos_weight` **7.548334 / 1.0 / 1.0**, step lists
  `[preprocess, model]` vs `[preprocess, smote, model]`), SMOTE's row arithmetic is re-derived, the scaler
  is proved to have been fitted on real rows while the resampled matrix's means are 40.9 units away, every
  fold's threshold is re-derived from the other four folds (15 thresholds, zero drift), and fold 1 of all three
  arms reproduces bit-identically.
* **197 independent checks** in a separate script that refits all three arms from `train.csv` through a
  plain `read_csv`, without importing `src.imbalance`: every per-fold PR-AUC, ROC-AUC, Brier, mean predicted
  probability and F1 is recomputed — with hand-written rank-based AUC and threshold-step average precision
  first validated against scikit-learn to 1e-12 — plus the 15 nested thresholds, the paired deltas, the
  SMOTE row identity, and the two claims about calibration.
* **Total runtime 61.1 s** for 15 fits, as this run recorded it: `weights` 9.3 s, `none` 9.5 s,
  `smote` 28.3 s. These are wall-clock seconds on one machine, and they move by tens of percent between
  runs — repeated runs here ranged 52.6–61.1 s total — so they are quoted to show **scale**, not as a
  benchmark. The stable part is the ratio: SMOTE trains on roughly **twice** the rows and takes about
  **three times** as long.

---

## 8. What T30 and T31 inherit

* **SMOTE is rejected** — not "not better", but worse in every fold on every metric, by 6.8 standard errors
  on ROC-AUC. `docs/06_imbalance.md` is the record; no SMOTE variant needs to be tuned.
* **The imbalance weight is on notice.** T30 should tune the **untreated** estimator (and may keep the
  weighted one as a second configuration), because the weight costs half the Brier score and buys no
  detectable ranking, and T31 should persist whichever wins on PR-AUC — with a strong prior that it is the
  untreated one.
* **The F1 rule travels.** T32 reports CV metrics and T34 picks a threshold; both must state their cut and
  why it is comparable across whatever they compare, because §4 shows a fixed 0.5 can reverse two arms'
  ordering while changing nothing about the models.

*Interpretation for the viva:* "I tested the imbalance treatment rather than assuming it. SMOTE was worse
than doing nothing in all five folds — a 6.8-sigma drop in ROC-AUC — because the synthetic rows it invents
are interpolations between clients that do not correspond to any real client, and this dataset's positives
are not clustered. More interestingly, my class weighting was not buying anything either: without it the
model ranks identically and its probabilities are already calibrated to the base rate, taking the Brier
score from 0.164 to 0.082. And I did not compare F1 at 0.5, because that would have ranked the arms by where
their calibration lands — at a properly chosen cut the ordering reverses."
