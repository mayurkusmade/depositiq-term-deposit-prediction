"""
DepositIQ - the presentation deck, generated from the records (T41).

Why this file exists
--------------------
T41 asks for a slide deck of at most fifteen slides, business-first, whose results table comes from actual
runs. The failure mode for a deck is worse than for a report: a number is copied onto a slide, the slide is
exported to PDF, the PDF is emailed, and nobody can tell afterwards which of the forty figures on it were
measured and which were remembered. So the deck is built the same way the report and the playbook were:

* the slides are **data** in `SPEC` below - text, tables and figure paths, nothing else;
* every decimal in the generated file is matched against a value in `models/*.json`, `reports/*.csv` or a
  supporting document in `docs/`, and a number with no record behind it fails the build;
* the geometry is checked against the **saved** file, not the in-memory objects, so an overflowing text box
  is a build failure rather than something a reader discovers in the presentation.

The one thing this module does not do is compute. No model is fitted and no metric is recomputed: the
figures demonstrated in the previous phases are quoted here, and the trace proves each quote has a source.

Business-first, and the sentence that travels with the numbers
--------------------------------------------------------------
The narrative order is the order a decision-maker needs it: the answer and its scope, then the decision,
then the price, and only then the data, the models and the evaluation. The scope sentence (D18) is the
second bullet of the second slide, *before* the first euro total: at C = 1.50 EUR and V = 20.00 EUR the
break-even probability 7.50% sits below the 11.6982% base rate, so the model **converts a fixed call
budget** and is not a case for cutting campaign spend. A check enforces that ordering on the saved file.

The three prohibitions each get a slide, because a prohibition that appears only in a footnote is a claim
the presenter will not be asked about: `duration` never an input, accuracy never a headline, `month` never
a cause.

How to run
----------
    python -m src.deck        # ~2 s: writes docs/presentation.pptx, validates the saved file, records it

Writes `docs/presentation.pptx` and `models/deck.json`.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

from src import config, report

# --------------------------------------------------------------------------- #
# Outputs and rules

DECK_PATH: Path = config.DOCS_DIR / "presentation.pptx"
RECORD_PATH: Path = config.MODELS_DIR / "deck.json"

# The brief caps the deck at fifteen slides. The floor is this project's own: a deck that tried to cover
# nine rubric criteria in fewer than ten slides would be a summary of the report, not a presentation of it.
MAX_SLIDES: int = 15
MIN_SLIDES: int = 10

# Style rules that make a slide readable, enforced rather than intended.
MAX_BULLETS: int = 5
MAX_BULLET_CHARS: int = 165
MIN_BODY_PT: float = 9.0
MIN_SMALL_PT: float = 8.0
MIN_BYTES: int = 100_000

# D18, the sentence that must precede the first euro total on the headline slide.
SCOPE_PHRASE: str = "converts a fixed call budget"
SCOPE_REFUSAL: str = "not a case for cutting"

# The three prohibitions, and the slide kicker each must appear on.
PROHIBITIONS: tuple[tuple[str, str], ...] = (
    ("duration", "Leakage"),
    ("accuracy", "refuse"),
    ("month", "drives"),
)

FIGURE_HEAD: tuple[str, ...] = (
    "01_target_distribution.png",
    "04_campaign_vs_target.png",
    "06_feature_importance.png",
    "07_threshold_curve.png",
    "08_lift_gain.png",
    "09_shap_summary.png",
)

# --------------------------------------------------------------------------- #
# Layout (inches, 16:9) and palette

W_IN, H_IN = 13.333, 7.5
MARGIN = 0.7
CONTENT_W = W_IN - 2 * MARGIN
COL_GAP = 0.34
COL_W = (CONTENT_W - COL_GAP) / 2
WIDE_LEFT = CONTENT_W * 0.42
WIDE_RIGHT = CONTENT_W - WIDE_LEFT - COL_GAP
BODY_TOP = 1.99
BODY_BOTTOM = 6.42
BODY_H = BODY_BOTTOM - BODY_TOP
BLOCK_GAP = 0.24

INK = RGBColor(0x1B, 0x2A, 0x41)
TEAL = RGBColor(0x0F, 0x6E, 0x6E)
MUTED = RGBColor(0x6B, 0x72, 0x80)
RULE = RGBColor(0xD8, 0xDE, 0xE9)
BAND = RGBColor(0xF4, 0xF6, 0xFA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xA9, 0xB4, 0xC4)

FONT = "Calibri"

# Average glyph width as a fraction of the point size, used by the overflow estimator. Calibri sits near
# 0.50 em for mixed case; the estimator is deliberately generous so a box that *just* fits reads as a
# warning rather than passing on a technicality.
EM_WIDTH = 0.505
LINE_SPACING = 1.18


# --------------------------------------------------------------------------- #
# Slide spec

@dataclass(frozen=True)
class Block:
    """One element of a slide body: a bullet list, a table, or an image."""

    kind: str
    items: tuple[str, ...] = ()
    headers: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()
    widths: tuple[float, ...] = ()
    figure: str = ""
    caption: str = ""
    size: float = 15.0


@dataclass(frozen=True)
class Slide:
    kicker: str
    title: str
    source: str
    left: tuple[Block, ...] = ()
    right: tuple[Block, ...] = ()
    subtitle: str = ""
    meta: str = ""
    # A figure column gets the larger share of the width. The project's charts are multi-panel and their
    # axis labels only survive at six inches or more; an even split makes them decoration.
    wide_right: bool = False

    @property
    def is_title(self) -> bool:
        return not self.left and not self.right


def bullets(*items: str, size: float = 16.0) -> Block:
    return Block("bullets", items=tuple(items), size=size)


def table(headers: tuple[str, ...], rows: tuple[tuple[str, ...], ...],
          widths: tuple[float, ...], size: float = 10.5) -> Block:
    return Block("table", headers=headers, rows=rows, widths=widths, size=size)


def figure(name: str, caption: str) -> Block:
    return Block("figure", figure=name, caption=caption)


# --------------------------------------------------------------------------- #
# The deck

SPEC: tuple[Slide, ...] = (
    Slide(
        kicker="DepositIQ",
        title="Which clients to call today",
        source="mayurkusmade/depositiq-term-deposit-prediction",
        subtitle="A ranking model for a bank telemarketing campaign, and the price at which\n"
                 "calling stops paying for itself.",
        meta="45,211 clients   |   9,043 held out   |   XGBoost, 12 features   |   cut 0.0750",
    ),
    Slide(
        kicker="The answer",
        title="Spend the same calls better, not fewer",
        source="models/playbook.json, models/threshold_choice.json",
        left=(
            bullets(
                "At 1.50 EUR a call and a 20.00 EUR deposit, break-even is 7.50% - below the 11.6982% rate of "
                "the modelling half.",
                "So this model converts a fixed call budget. It is not a case for cutting campaign spend.",
                "Calling 41.10% of the list reaches 77.90% of subscribers: 43,621.00 EUR against 30,368.00 EUR "
                "for calling everyone.",
                "+13,253.00 EUR, on 21,302 fewer calls.",
            ),
        ),
        right=(
            table(
                ("Per campaign", "Call everyone", "Ranked, cut at 0.0750"),
                (("Calls made", "36,168", "14,866"),
                 ("Subscribers reached", "4,231", "3,296"),
                 ("Net value", "30,368.00 EUR", "43,621.00 EUR")),
                widths=(0.34, 0.28, 0.38),
                size=12.5,
            ),
            bullets(
                "Held out: 10,988.50 EUR against 7,595.50, on 3,621 calls.",
                size=13.0,
            ),
        ),
    ),
    Slide(
        kicker="1 - The decision",
        title="Seven calls in eight are wasted",
        source="docs/01_problem_definition.md, models/threshold_choice.json",
        wide_right=True,
        left=(
            bullets(
                "45,211 clients in the file; 5,289 subscribed - 11.70%, a 7.55 : 1 split.",
                "Each day the bank has more clients than agents, so its decision is which clients to call.",
                "The constraint is agent capacity, not budget: the unit of cost is one call.",
                "So the question is not 'will this client buy', but 'is this client worth a call today'.",
            ),
        ),
        right=(
            figure("01_target_distribution.png",
                   "One deposit per 8.55 calls, so an unordered list wastes seven calls in eight."),
        ),
    ),
    Slide(
        kicker="2 - The cost model",
        title="The cut is a price, not a parameter",
        source="src/cost.py, models/threshold_choice.json, docs/01_problem_definition.md",
        left=(
            bullets(
                "A call costs C = 1.50 EUR; a captured deposit is worth V = 20.00 EUR. Both are assumptions, and "
                "both are declared as such.",
                "Call when V x p > C, so the break-even probability is p* = C / V = 7.50%.",
                "Every euro here is linear in V: at 40.00 EUR the cut halves and the totals double.",
                "The ranking does not move, which is why the decisions rest on order and the euros stay labelled.",
            ),
        ),
        right=(
            table(
                ("Quantity", "Value", "Where it comes from"),
                (("C, cost of a call", "1.50 EUR", "assumption"),
                 ("V, value of a deposit", "20.00 EUR", "assumption"),
                 ("p*, break-even", "7.50%", "C / V"),
                 ("Base rate", "11.6982%", "measured: 4,231 in 45,211")),
                widths=(0.36, 0.22, 0.42),
                size=12.5,
            ),
        ),
    ),
    Slide(
        kicker="3 - The data",
        title="No missing values, and that is the problem",
        source="docs/02_data_dictionary.md, docs/05_leakage_audit.md",
        left=(
            bullets(
                "Zero NaN in 45,211 rows: missingness arrives as a literal string. poutcome = unknown covers "
                "81.7%; contact = unknown covers 28.8%.",
                "pdays = -1 means 'never contacted', not 'minus one day'. Read as a number it returns a mean of "
                "40.2 days that no client has.",
                "pdays and days_since_contact carry the same number on all 6,584 rows where both exist: a "
                "duplicate, not a correlation.",
                "The two unknowns are different objects. poutcome = unknown is a fact about the bank; contact = "
                "unknown is a failure to record.",
                "So one becomes a level and the other becomes recommendation R1.",
            ),
        ),
    ),
    Slide(
        kicker="4 - What the campaign already gets wrong",
        title="The fourth call costs more than it earns",
        source="docs/04_eda_insights.md, reports/figures/04_campaign_vs_target.png",
        wide_right=True,
        left=(
            bullets(
                "poutcome = success converts at 64.65% - 779 of 1,205 clients - and is known before the call.",
                "Clients needing 4 or more calls hold 52.68% of the calls for 13.47% of the successes.",
                "The first three calls convert at 7.7386%, just above the price; the extra calls at 1.9298%.",
                "That tail is 32,906 EUR of agent time against a 20.00 EUR deposit.",
                "The cap must be targeted: 93.28% of the tail has no known outcome, while poutcome = success "
                "still returns 22.55% a call.",
            ),
        ),
        right=(
            figure("04_campaign_vs_target.png",
                   "Call count, contact type and prior outcome, against the price line."),
        ),
    ),
    Slide(
        kicker="5 - Leakage: prohibition one",
        title="The column that doubles the money and must not be used",
        source="models/duration_comparison.json, docs/05_leakage_audit.md",
        left=(
            bullets(
                "duration is the length of the call. It is recorded after the call ends, so it does not exist "
                "when today's list is built.",
                "Alone it reaches ROC-AUC 0.8078, more than the shipped model. Its bands climb from 0.16% under "
                "a minute to 48.56% at ten minutes or more.",
                "Given to the model it adds 0.1788 PR-AUC and 20,340.00 EUR.",
                "The verdict came from a rule written before the numbers existed, and it says leakage.",
            ),
        ),
        right=(
            table(
                ("Arm", "PR-AUC", "ROC-AUC", "Net at 7.50%"),
                (("A - without duration (ships)", "0.4460", "0.7990", "43,621.00 EUR"),
                 ("B - with duration (diagnostic)", "0.6254", "0.9360", "63,961.00 EUR")),
                widths=(0.40, 0.18, 0.18, 0.24),
                size=12.0,
            ),
            bullets(
                "Arm A ships regardless: the column does not exist at the moment a decision is made.",
                size=13.0,
            ),
        ),
    ),
    Slide(
        kicker="6 - Features",
        title="Seventeen features became twelve, and the loss was measured",
        source="docs/05_leakage_audit.md, reports/figures/06_feature_importance.png",
        wide_right=True,
        left=(
            bullets(
                "Five engineered columns replace their raw ones rather than sit beside them; balance through "
                "arsinh takes absolute skew from 8.34 to 1.68.",
                "Two rankings over the same 17 features - mutual information and permutation importance - "
                "correlate at +0.647.",
                "Removing the five nominated features costs 0.00014 +/- 0.00195 ROC-AUC, with 3 of 5 folds "
                "improving, so the one-standard-error rule drops them.",
                "Re-measured per model: the forest loses 0.0014 +/- 0.0024 and the booster 0.0001 +/- 0.0008. "
                "The matrix goes from 65 columns to 40.",
            ),
        ),
        right=(
            figure("06_feature_importance.png",
                   "Two rankings, each feature's own permutation null, and the five dropped features hatched."),
        ),
    ),
    Slide(
        kicker="7 - Models",
        title="One harness, seven builds, and a fix that made it worse",
        source="models/cv_results.json, models/imbalance_results.json, models/final_model.json",
        left=(
            table(
                ("Build", "PR-AUC", "ROC-AUC", "Brier", "Role"),
                (("dummy_prior", "0.1170", "0.5000", "0.1033", "floor"),
                 ("logreg_balanced", "0.4008", "0.7640", "0.1852", "baseline"),
                 ("random_forest", "0.4346", "0.7906", "0.1403", "candidate"),
                 ("XGBoost, weighted", "0.4420", "0.7993", "0.1637", "candidate"),
                 ("XGBoost, untreated", "0.4444", "0.7993", "0.0821", "candidate"),
                 ("XGBoost, SMOTE", "0.4152", "0.7799", "0.1221", "rejected"),
                 ("XGBoost, tuned", "0.4460", "0.7990", "0.0818", "shipped")),
                widths=(0.34, 0.16, 0.17, 0.16, 0.17),
                size=11.5,
            ),
        ),
        right=(
            bullets(
                "Same folds, same rows, same scorer for every build - 36,168 training rows and 4,231 positives.",
                "Ranking and calibration are separate problems: the logistic baseline ranks at 0.7640 ROC-AUC "
                "while predicting 0.4161 on average against a 0.1170 base rate.",
                "SMOTE lost in 0 of 5 folds and ran three times slower; the class weight bought no ranking and "
                "halved the Brier score.",
                "Tuning added 0.00165 +/- 0.00286 - 0.58 SE - so the tuned point is kept for not being worse, not "
                "for being better.",
            ),
        ),
    ),
    Slide(
        kicker="8 - The operating point",
        title="The cut is the price the bank already pays",
        source="models/threshold_choice.json, reports/figures/07_threshold_curve.png",
        wide_right=True,
        left=(
            table(
                ("Cut", "Calls", "Reached", "Net value"),
                (("Call everyone", "36,168", "4,231", "30,368.00 EUR"),
                 ("0.5", "-", "-", "18,248.00 EUR"),
                 ("F1 cut 0.2103", "-", "-", "36,891.00 EUR"),
                 ("Price 0.0750", "14,866", "3,296", "43,621.00 EUR"),
                 ("Empirical peak 0.0821", "13,247", "3,194", "44,009.50 EUR")),
                widths=(0.38, 0.19, 0.19, 0.24),
                size=11.5,
            ),
            bullets(
                "The peak is worth 388.50 EUR more, and is chosen by looking at the data it then scores.",
                "0.0750 sits 0.0004 above the bottom of the 1%-of-peak plateau: the most aggressive cut still "
                "within 1% of the best.",
                "ROI is a trap of its own: 0.5 posts 765% and loses money in absolute terms.",
                size=13.0,
            ),
        ),
        right=(
            figure("07_threshold_curve.png",
                   "Net value against the cut, with the price, the 1%-of-peak plateau and both "
                   "empirical argmaxes marked."),
        ),
    ),
    Slide(
        kicker="9 - Held out, spent once",
        title="On clients it never saw, the ranking holds",
        source="models/holdout_eval.json, models/metrics_cv.json",
        left=(
            table(
                ("Metric", "Held out", "Cross-validated"),
                (("PR-AUC", "0.4616", "0.4460"),
                 ("ROC-AUC", "0.8052", "0.7990"),
                 ("Brier", "0.0804", "0.0818"),
                 ("Mean prediction", "0.1151", "0.1166")),
                widths=(0.42, 0.29, 0.29),
                size=12.5,
            ),
        ),
        right=(
            bullets(
                "The artifact was loaded, not refitted, and scored once on 9,043 rows holding 1,058 subscribers.",
                "Net value 10,988.50 EUR against 7,595.50 for calling everyone, at a recall of 77.60%.",
                "The gap of 0.0155 is not a gain: the bootstrap over these same rows spans 0.4304 to 0.4969, "
                "which is 0.92 standard deviations.",
                "What this does not say: the prices are assumptions, the split is random rather than temporal, "
                "and the held-out rows are the same campaign.",
            ),
        ),
    ),
    Slide(
        kicker="10 - Lift",
        title="The top tenth of the list converts 4.5 times the base rate",
        source="models/lift_results.json, reports/lift_table.csv, reports/figures/08_lift_gain.png",
        wide_right=True,
        left=(
            bullets(
                "Top decile lift 4.5100 held out and 4.3652 out of fold; that decile holds 45.09% of all "
                "subscribers.",
                "Decile 3 is already below random at 0.88 - past the best fifth, the model's message is 'do not "
                "call'.",
                "The adopted cut saves 13,309 calls, 19,964 EUR, against a fixed order needing 28,175.",
                "Calls saved peaks at 18.89% coverage, before the cut; the 8,033 calls between them still convert "
                "at 8.96%.",
                "A shuffled-score self-test reruns the table on noise and finds lifts of 0.950 to 1.108.",
            ),
        ),
        right=(
            figure("08_lift_gain.png",
                   "Lift and cumulative gain per decile on both halves, each against its own base rate."),
        ),
    ),
    Slide(
        kicker="11 - The metric we refuse: prohibition two",
        title="The best model by accuracy finds nobody",
        source="models/holdout_eval.json, models/metrics_cv.json",
        left=(
            table(
                ("Cut", "Accuracy", "Balanced accuracy"),
                (("Always 'no'", "88.3003%", "50%"),
                 ("0.5", "89.6052%", "61.28%"),
                 ("Price 0.0750", "66.42%", "71.27%")),
                widths=(0.36, 0.32, 0.32),
                size=12.5,
            ),
            bullets(
                "Every figure in that table is held out, from the same 9,043 rows.",
                "A model that answers 'no' to everyone is 88.3003% accurate and calls nobody.",
                "At its cut this model scores 66.42% - 22 points worse - while finding 77.60% of subscribers.",
                "The most accurate cut is 0.5 at 89.6052%, and it still misses 801 of the 1,058.",
                "A metric that ranks 'call nobody' above 'call the right 40%' is not a headline.",
                size=13.0,
            ),
        ),
    ),
    Slide(
        kicker="12 - What drives the score",
        title="The top driver is a blank field",
        source="models/shap_global.json, reports/shap_importance.csv, reports/figures/09_shap_summary.png",
        wide_right=True,
        left=(
            bullets(
                "contact carries 22.14% of importance, month 17.55% and day 10.60%: campaign-side features take "
                "55.75% against 44.25% for the client's own.",
                "contact = unknown contributes -1.0150 log-odds on 28.72% of clients, who subscribe at 4.11% "
                "against 14.83% for cellular.",
                "The model is not saying those clients are bad prospects. It is saying it cannot tell.",
                "Prohibition three: month is a calendar from a 2008-2010 recession. Schedule with it; never "
                "predict a person's answer from it.",
            ),
        ),
        right=(
            figure("09_shap_summary.png",
                   "The twelve features by share of importance, each contribution's sign, and the shared "
                   "one-hot blocks that keep the values additive."),
        ),
    ),
    Slide(
        kicker="13 - What to do",
        title="Eight actions, each with the number that would prove it wrong",
        source="models/playbook.json, reports/playbook.csv, docs/08_recommendations.md",
        left=(
            table(
                ("", "Action", "Headline", "Falsified if"),
                (("R1", "Make contact a required field", "28.72% of clients at -1.0150", "filled and the segment does not re-score"),
                 ("R2", "Call in model order, stop at 0.0750", "+13,253.00 EUR, 21,302 fewer calls", "a later campaign reaches under 60%"),
                 ("R3", "Do not stop where calls saved peaks", "8,033 calls at 8.96%, +2,350.50 EUR", "the held-out block replicates at scale"),
                 ("R4", "Never feed call duration to the model", "+0.1788 PR-AUC, dead at dial time", "a dialler can supply it before the call"),
                 ("R5", "Establish the value of a subscription", "every euro is linear in V = 20.00 EUR", "the bank publishes a true V"),
                 ("R6", "Track precision and net value, not accuracy", "88.3003% from a model that finds nobody", "a stakeholder accepts a cost-based cut"),
                 ("R7", "Keep the twelve-feature XGBoost", "0.4460 against 0.4346 and 0.4008", "a re-tune beats 0.4008 by over one SE"),
                 ("R8", "Use month to schedule, never to predict", "17.55% of importance, 2008-2010 confound", "a campaign outside the recession persists")),
                widths=(0.06, 0.32, 0.29, 0.33),
                size=10.5,
            ),
        ),
    ),
)


# --------------------------------------------------------------------------- #
# Measurement helpers (used for layout and for the overflow check)

def text_lines(text: str, width_in: float, size_pt: float) -> int:
    """Estimate wrapped line count for `text` in a box `width_in` wide at `size_pt`."""
    per_line = max(int(width_in * 72.0 / (EM_WIDTH * size_pt)), 8)
    lines, current = 1, 0
    for word in text.split():
        addition = len(word) + (1 if current else 0)
        if current + addition > per_line:
            lines += 1
            current = len(word)
        else:
            current += addition
    return lines


def caption_height(caption: str, width_in: float, size_pt: float = 10.0) -> float:
    return text_lines(caption, width_in, size_pt) * size_pt * LINE_SPACING / 72.0


def row_height(row: tuple[str, ...], widths: tuple[float, ...], width_in: float, size_pt: float) -> float:
    """Height of one table row: its tallest cell at the column width it was given."""
    cols = [width_in * fraction for fraction in widths]
    lines = max(text_lines(cell, cols[i] - 0.14, size_pt) for i, cell in enumerate(row))
    return max(lines * size_pt * LINE_SPACING / 72.0 + 0.11, 0.31)


def block_height(block: Block, width_in: float) -> float:
    """Height `block` needs at `width_in` wide, before any stretching to fill its column."""
    if block.kind == "bullets":
        line_h = block.size * LINE_SPACING / 72.0
        total = 0.0
        for index, item in enumerate(block.items):
            total += text_lines(item, width_in - 0.22, block.size) * line_h
            total += 0.14 if index < len(block.items) - 1 else 0.0
        return total
    if block.kind == "table":
        return sum(row_height(row, block.widths, width_in, block.size)
                   for row in (block.headers,) + block.rows)
    if block.kind == "figure":
        path = config.FIGURES_DIR / block.figure
        with Image.open(path) as image:
            aspect = image.width / image.height
        caption_h = caption_height(block.caption, width_in)
        return width_in / aspect + (caption_h + 0.12 if block.caption else 0.0)
    raise ValueError(block.kind)


# --------------------------------------------------------------------------- #
# Drawing primitives

def _textbox(slide, name: str, left: float, top: float, width: float, height: float,
             align: PP_ALIGN = PP_ALIGN.LEFT) -> object:
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    box.name = name
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = 0
    frame.margin_right = 0
    frame.margin_top = 0
    frame.margin_bottom = 0
    frame.vertical_anchor = MSO_ANCHOR.TOP
    frame.paragraphs[0].alignment = align
    return box


def _run(paragraph, text: str, size: float, color: RGBColor, bold: bool = False, italic: bool = False):
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = FONT
    run.font.color.rgb = color
    return run


def _set_bullet(paragraph, color: RGBColor, size: float) -> None:
    """Replace the default bullet with a teal dash, indented to match the text."""
    props = paragraph._p.get_or_add_pPr()
    props.set("marL", str(int(Inches(0.22))))
    props.set("indent", str(int(-Inches(0.22))))
    for tag, attrs in (("a:buClr", None), ("a:buFont", {"typeface": "Calibri"}), ("a:buChar", {"char": "\u2013"})):
        element = etree.SubElement(props, qn(tag))
        if tag == "a:buClr":
            fill = etree.SubElement(element, qn("a:srgbClr"))
            fill.set("val", f"{color}")
        if attrs:
            for key, value in attrs.items():
                element.set(key, value)


def _rule(slide, left: float, top: float, width: float, height: float = 0.018,
          color: RGBColor = RULE) -> None:
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.name = "rule"
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False


def _draw_bullets(slide, block: Block, left: float, top: float, width: float) -> None:
    # Measure of readability, not of room: a 12-inch line at 15 pt is about 120 characters, which is
    # past the point where a reader loses the start of the line. Long bullets are capped at 10.2 in and
    # the slide keeps the right margin as white space.
    width = min(width, 10.2)
    box = _textbox(slide, "body", left, top, width, block_height(block, width) + 0.05)
    frame = box.text_frame
    for index, item in enumerate(block.items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.space_before = Pt(0)
        paragraph.space_after = Pt(7 if index < len(block.items) - 1 else 0)
        paragraph.line_spacing = 1.02
        _set_bullet(paragraph, TEAL, block.size)
        _run(paragraph, item, block.size, INK)


def _draw_table(slide, block: Block, left: float, top: float, width: float) -> None:
    rows = (block.headers,) + block.rows
    heights = [row_height(row, block.widths, width, block.size) for row in rows]
    shape = slide.shapes.add_table(len(rows), len(block.headers),
                                   Inches(left), Inches(top), Inches(width), Inches(sum(heights)))
    shape.name = "body"
    grid = shape.table
    grid.first_row = True
    grid.horz_banding = False
    for index, fraction in enumerate(block.widths):
        grid.columns[index].width = Inches(width * fraction)
    for row_index, height in enumerate(heights):
        grid.rows[row_index].height = Inches(height)
    for row_index, row in enumerate(rows):
        header = row_index == 0
        for col_index, value in enumerate(row):
            cell = grid.cell(row_index, col_index)
            cell.fill.solid()
            cell.fill.fore_color.rgb = INK if header else (BAND if row_index % 2 == 0 else WHITE)
            cell.margin_left = Inches(0.07)
            cell.margin_right = Inches(0.05)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            frame = cell.text_frame
            frame.word_wrap = True
            paragraph = frame.paragraphs[0]
            paragraph.space_after = Pt(0)
            _run(paragraph, value, block.size, WHITE if header else INK, bold=header)


def _draw_figure(slide, block: Block, left: float, top: float, width: float) -> None:
    picture = slide.shapes.add_picture(str(config.FIGURES_DIR / block.figure),
                                       Inches(left), Inches(top), width=Inches(width))
    picture.name = "figure"
    if block.caption:
        caption_top = top + picture.height / 914400 + 0.10
        box = _textbox(slide, "caption", left, caption_top, width,
                       caption_height(block.caption, width) + 0.06)
        paragraph = box.text_frame.paragraphs[0]
        _run(paragraph, block.caption, 10.0, MUTED)
        paragraph.line_spacing = 1.02


DRAWERS = {"bullets": _draw_bullets, "table": _draw_table, "figure": _draw_figure}


def _place(slide, blocks: tuple[Block, ...], left: float, width: float,
           top: float = BODY_TOP, available: float = BODY_H) -> None:
    """Stack `blocks` in a column. A lone figure is centred; everything else starts at the top."""
    if not blocks:
        return
    # Every column starts at the same baseline. Centring a lone figure was tried and looked wrong: the
    # chart floated in the middle of the slide with its left edge level with nothing.
    del available
    cursor = top
    for block in blocks:
        DRAWERS[block.kind](slide, block, left, cursor, width)
        cursor += block_height(block, width) + BLOCK_GAP


# --------------------------------------------------------------------------- #
# Slide assembly

def build(title_slide: Slide, spec: tuple[Slide, ...]) -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(W_IN)
    prs.slide_height = Inches(H_IN)

    core = prs.core_properties
    core.title = title_slide.title
    core.subject = "DepositIQ - bank telemarketing decisioning"
    core.author = "DepositIQ"
    core.last_modified_by = "DepositIQ"
    core.keywords = "term deposit; uplift; XGBoost; cost-sensitive threshold"
    core.created = datetime(2026, 10, 30, 9, 0, 0)
    core.modified = datetime(2026, 10, 30, 9, 0, 0)
    core.revision = 1

    # Cover
    cover = prs.slides.add_slide(prs.slide_layouts[6])
    background = cover.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(W_IN), Inches(H_IN))
    background.name = "background"
    background.fill.solid()
    background.fill.fore_color.rgb = INK
    background.line.fill.background()
    background.shadow.inherit = False

    _rule(cover, MARGIN, 2.02, 1.55, 0.05, TEAL)
    kicker = _textbox(cover, "kicker", MARGIN, 1.55, CONTENT_W, 0.34)
    _run(kicker.text_frame.paragraphs[0], title_slide.kicker.upper(), 12.0, PALE, bold=True)

    title_box = _textbox(cover, "title", MARGIN, 2.30, CONTENT_W - 1.2, 1.45)
    paragraph = title_box.text_frame.paragraphs[0]
    _run(paragraph, title_slide.title, 44.0, WHITE, bold=True)

    subtitle = _textbox(cover, "subtitle", MARGIN, 3.85, CONTENT_W - 2.6, 1.15)
    frame = subtitle.text_frame
    for index, line in enumerate(title_slide.subtitle.split("\n")):
        para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        para.line_spacing = 1.1
        _run(para, line, 15.5, PALE)

    meta = _textbox(cover, "source", MARGIN, 5.55, CONTENT_W - 1.2, 0.4)
    _run(meta.text_frame.paragraphs[0], title_slide.meta, 11.5, RGBColor(0x8E, 0x9B, 0xAE))

    footer = _textbox(cover, "footer", MARGIN, 6.62, CONTENT_W - 1.2, 0.35)
    _run(footer.text_frame.paragraphs[0], title_slide.source + "   |   2026-10-30", 10.0, RGBColor(0x6E, 0x7C, 0x91))

    # Content slides
    for number, slide_spec in enumerate(spec, start=2):
        slide = prs.slides.add_slide(prs.slide_layouts[6])

        kicker = _textbox(slide, "kicker", MARGIN, 0.46, CONTENT_W, 0.30)
        _run(kicker.text_frame.paragraphs[0], slide_spec.kicker.upper(), 10.5, TEAL, bold=True)

        title = _textbox(slide, "title", MARGIN, 0.74, CONTENT_W, 0.92)
        _run(title.text_frame.paragraphs[0], slide_spec.title, 27.0, INK, bold=True)

        _rule(slide, MARGIN, 1.74, CONTENT_W)

        if slide_spec.right:
            left_w = WIDE_LEFT if slide_spec.wide_right else COL_W
            right_w = WIDE_RIGHT if slide_spec.wide_right else COL_W
            _place(slide, slide_spec.left, MARGIN, left_w)
            _place(slide, slide_spec.right, MARGIN + left_w + COL_GAP, right_w)
        else:
            _place(slide, slide_spec.left, MARGIN, CONTENT_W)

        source = _textbox(slide, "source", MARGIN, 6.70, CONTENT_W - 1.0, 0.44)
        source.text_frame.paragraphs[0].line_spacing = 1.0
        _run(source.text_frame.paragraphs[0], "Source: " + slide_spec.source, 8.5, MUTED)

        number_box = _textbox(slide, "footer", W_IN - 1.25, 6.70, 0.55, 0.30, PP_ALIGN.RIGHT)
        _run(number_box.text_frame.paragraphs[0], f"{number} / {len(spec) + 1}", 9.0, MUTED)

    return prs


# --------------------------------------------------------------------------- #
# Validation against the saved file

COUNT = 0
FAILURES: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> bool:
    global COUNT
    COUNT += 1
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not passed:
        FAILURES.append(name)
    return passed


def shape_text(shape) -> str:
    parts: list[str] = []
    if shape.has_text_frame:
        parts.append(shape.text_frame.text)
    if getattr(shape, "has_table", False) and shape.has_table:
        for row in shape.table.rows:
            for cell in row.cells:
                parts.append(cell.text)
    return "\n".join(parts)


def slide_text(slide) -> str:
    return "\n".join(shape_text(shape) for shape in slide.shapes)


def named(slide, name: str) -> list:
    return [shape for shape in slide.shapes if shape.name == name]


def package_parts(path: Path) -> dict[str, bytes]:
    """Every zip member of the .pptx by name, so two builds can be compared without their timestamps."""
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def validate(path: Path) -> dict:
    prs = Presentation(str(path))
    slides = list(prs.slides)
    texts = [slide_text(slide) for slide in slides]
    record: dict = {"slides": len(slides), "layout": [], "figures": [], "titles": []}

    print("-- structure -------------------------------------------------------------")
    check("slide count within the brief", MIN_SLIDES <= len(slides) <= MAX_SLIDES,
          f"{len(slides)} slides (max {MAX_SLIDES})")
    check("the saved file reopens with python-pptx", len(slides) > 0, f"{len(slides)} slides read back")

    titled = [bool(named(slide, "title") and shape_text(named(slide, "title")[0]).strip()) for slide in slides]
    check("every slide carries a non-empty title", all(titled), f"{sum(titled)}/{len(slides)}")
    record["titles"] = [shape_text(named(slide, "title")[0]).strip() for slide in slides]

    bodies = [bool(named(slide, "body")) or bool(named(slide, "subtitle")) for slide in slides]
    check("every slide carries body content", all(bodies), f"{sum(bodies)}/{len(slides)}")

    sourced = [bool(named(slide, "source")) for slide in slides[1:]]
    check("every content slide cites its source", all(sourced), f"{sum(sourced)}/{len(sourced)}")

    # Citations must point at files that exist, so a slide cannot cite evidence it invented.
    missing: list[str] = []
    seen_paths = 0
    for slide in slides[1:]:
        citation = shape_text(named(slide, "source")[0]).replace("Source:", "").strip()
        for part in citation.split(","):
            token = part.strip().split(" \u00a7")[0].strip()
            if not token:
                continue
            seen_paths += 1
            if not (config.ROOT / token).exists():
                missing.append(token)
    check("every cited path exists in the repository", not missing and seen_paths >= 12,
          f"{seen_paths} citations, {len(missing)} missing" + (f": {missing}" if missing else ""))

    print("-- narrative -------------------------------------------------------------")
    headline = texts[1]
    scope_at = headline.find(SCOPE_PHRASE)
    refusal_at = headline.find(SCOPE_REFUSAL)
    first_total = re.search(r"\d{1,3}(?:,\d{3})+\.\d{2}", headline)
    check("the scope sentence precedes the first euro total",
          scope_at != -1 and first_total is not None and scope_at < first_total.start(),
          f"scope at {scope_at}, first total at {first_total.start() if first_total else 'absent'}")
    check("the headline slide states the scope as a refusal", refusal_at != -1, f"found at {refusal_at}")

    for word, kicker in PROHIBITIONS:
        hits = [i for i, text in enumerate(texts) if kicker.lower() in text.lower() and word.lower() in text.lower()]
        check(f"prohibition '{word}' appears on its own slide", bool(hits), f"slides {[h + 1 for h in hits]}")

    last = texts[-1]
    present = [f"R{index}" for index in range(1, 9) if re.search(rf"\bR{index}\b", last)]
    check("all eight recommendations appear", len(present) == 8, f"{len(present)}: {','.join(present)}")

    check("the deck opens on a decision, not a dataset",
          "which clients to call today" in texts[0].lower(), texts[0].splitlines()[0][:60])

    print("-- numbers ---------------------------------------------------------------")
    allowed = report.record_numbers(exclude=RECORD_PATH) | report.document_numbers()
    total, unmatched = report.trace("\n".join(texts), allowed)
    check("every decimal on every slide traces to a record or document", not unmatched,
          f"{total} tokens, unmatched: {unmatched[:8]}")

    print("-- geometry --------------------------------------------------------------")
    limit_w, limit_h = prs.slide_width, prs.slide_height
    outside: list[str] = []
    for index, slide in enumerate(slides, start=1):
        for shape in slide.shapes:
            if shape.left is None or shape.top is None:
                continue
            if shape.left < 0 or shape.top < 0 or shape.left + shape.width > limit_w or shape.top + shape.height > limit_h:
                outside.append(f"slide {index}:{shape.name}")
    check("every shape sits inside the slide", not outside, f"{len(outside)} outside" + (f": {outside[:4]}" if outside else ""))

    # The source line sits at 6.70 in; a figure and its caption must finish above it, or the slide would
    # show a chart overlapping its own citation.
    crowding = [f"slide {index}:{shape.name} ends at {round((shape.top + shape.height) / 914400, 2)} in"
                for index, slide in enumerate(slides, start=1)
                for shape in slide.shapes
                if shape.name in ("figure", "caption") and (shape.top + shape.height) / 914400 > 6.60]
    check("no figure or caption runs into the source line", not crowding,
          f"{len(crowding)}" + (f": {crowding[:3]}" if crowding else ""))

    sizes: list[tuple[float, str, str]] = []
    for index, slide in enumerate(slides, start=1):
        for shape in slide.shapes:
            frames = []
            if shape.has_text_frame:
                frames.append(shape.text_frame)
            if getattr(shape, "has_table", False) and shape.has_table:
                frames.extend(cell.text_frame for row in shape.table.rows for cell in row.cells)
            for frame in frames:
                for paragraph in frame.paragraphs:
                    for run in paragraph.runs:
                        if run.font.size is not None:
                            sizes.append((run.font.size.pt, shape.name, f"slide {index}"))
    body_sizes = [size for size, name, _ in sizes if name == "body"]
    small_sizes = [size for size, name, _ in sizes if name in ("source", "caption", "footer")]
    check("body text is never below 9 pt", bool(body_sizes) and min(body_sizes) >= MIN_BODY_PT,
          f"min {min(body_sizes) if body_sizes else '-'} pt over {len(body_sizes)} runs")
    check("source and caption text is never below 8 pt",
          bool(small_sizes) and min(small_sizes) >= MIN_SMALL_PT,
          f"min {min(small_sizes) if small_sizes else '-'} pt")

    # Overflow: re-estimate each text block from the text actually stored in the file, against the frame
    # it was given. This is what catches a bullet that wrapped to one line more than the box allowed.
    overflows: list[str] = []
    for index, slide in enumerate(slides, start=1):
        for shape in named(slide, "body"):
            if not shape.has_text_frame:
                continue
            width_in = shape.width / 914400
            height_in = shape.height / 914400
            paragraphs = [p for p in shape.text_frame.paragraphs if p.text.strip()]
            if not paragraphs:
                continue
            size = paragraphs[0].runs[0].font.size.pt if paragraphs[0].runs else 12.0
            needed = sum(text_lines(p.text, width_in - 0.22, size) * size * LINE_SPACING / 72.0
                         for p in paragraphs) + 0.14 * max(len(paragraphs) - 1, 0)
            if needed > height_in + 0.03:
                overflows.append(f"slide {index}: needs {needed:.2f} in, box {height_in:.2f} in")
    check("no bullet block overflows its box", not overflows, f"{len(overflows)}" + (f": {overflows[:3]}" if overflows else ""))

    bullets_too_long = 0
    bullets_total = 0
    too_many = 0
    for slide in slides[1:]:
        for shape in named(slide, "body"):
            if not shape.has_text_frame:
                continue
            paragraphs = [p for p in shape.text_frame.paragraphs if p.text.strip()]
            if len(paragraphs) > 1:
                bullets_total += len(paragraphs)
                if len(paragraphs) > MAX_BULLETS:
                    too_many += 1
                bullets_too_long += sum(1 for p in paragraphs if len(p.text) > MAX_BULLET_CHARS)
    check("bullet lists stay within five items and 165 characters",
          bullets_too_long == 0 and too_many == 0,
          f"{bullets_total} bullets, {bullets_too_long} too long, {too_many} lists too long")

    print("-- assets ----------------------------------------------------------------")
    pictures: list[str] = []
    for index, slide in enumerate(slides, start=1):
        for shape in slide.shapes:
            if shape.shape_type == 13 or shape.name == "figure":
                pictures.append(f"slide {index}")
    record["figures"] = pictures
    check("the six planned figures are embedded", len(pictures) == len(FIGURE_HEAD),
          f"{len(pictures)} pictures: {pictures}")
    check("every embedded figure is also on disk",
          all((config.FIGURES_DIR / name).exists() for name in FIGURE_HEAD), f"{len(FIGURE_HEAD)} checked")

    size = path.stat().st_size
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    check("the package is a plausible size for seven embedded charts", size >= MIN_BYTES,
          f"{size:,} bytes, sha256 {digest[:8]}")
    record.update({"bytes": size, "sha256": digest})

    print("-- determinism -----------------------------------------------------------")
    # Byte equality is the wrong test here and passing it early was luck: python-pptx stamps each zip
    # member with the current clock, so two builds in the same second match and two a second apart do
    # not. The property that matters is that the *parts* are identical - same slides, same text, same
    # images - so that is what is compared.
    with tempfile.TemporaryDirectory() as folder:
        twin = Path(folder) / "twin.pptx"
        build(SPEC[0], SPEC[1:]).save(str(twin))
        original = package_parts(path)
        rebuilt = package_parts(twin)
        differing = sorted(name for name in original if original.get(name) != rebuilt.get(name))
        check("a second build produces identical package parts",
              original == rebuilt, f"{len(original)} parts, {len(differing)} differ"
              + (f": {differing[:4]}" if differing else ""))

    return record


# --------------------------------------------------------------------------- #
# Main

def main() -> int:
    config.ensure_dirs()
    title_slide, spec = SPEC[0], SPEC[1:]
    prs = build(title_slide, spec)
    prs.save(str(DECK_PATH))

    print("=" * 78)
    print("T41 - the presentation deck")
    print("=" * 78)
    print(f"  wrote {DECK_PATH.name}: {len(SPEC)} slides, {DECK_PATH.stat().st_size:,} bytes")
    print()

    record = validate(DECK_PATH)
    record["slide_count"] = len(SPEC)
    record["max_allowed"] = MAX_SLIDES
    record["figures_planned"] = list(FIGURE_HEAD)
    record["generator"] = "src/deck.py"
    record["checks"] = COUNT - len(FAILURES)
    record["failures"] = FAILURES

    print()
    print(f"{COUNT - len(FAILURES)}/{COUNT} checks passed")
    if FAILURES:
        for failure in FAILURES:
            print("  FAILED:", failure)
        return 1

    RECORD_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"  record: {RECORD_PATH.name} ({RECORD_PATH.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
