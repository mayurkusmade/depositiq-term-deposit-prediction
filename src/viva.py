"""
DepositIQ - the viva Q&A bank (T45).

Why this file exists
--------------------
The report is written for a reader; a viva is written for a conversation. The same facts have to survive a
follow-up question, and the failure mode is different: a report can be skimmed, but an answer has to be
defended in one breath and then traced to a file when the examiner pushes. This module holds the bank to
three things a memorised answer cannot fake:

* **Coverage.** Every one of the nine rubric criteria has at least four questions, and the question numbers
  are sequential. A bank that answers the modelling chapters and skips documentation is not a viva bank.
* **Shape.** Every question has an **Answer** and an **Evidence** line, the answer is substantial, and every
  relative link resolves. A question with no source is an opinion.
* **Provenance.** Every decimal in the bank is matched against a `models/*.json` record, a `reports/*.csv`
  column, the report, or one of the eleven analysis documents. The bank itself and the downstream demo
  script are excluded, so a number cannot be admitted by quoting the document that is being checked.

The bank introduces no new experiment. It is a route through work that is already on disk, and the trace is
what makes that claim true rather than aspirational.

What is checked
---------------
The nine criterion sections, the 25-question floor, the four-per-criterion coverage, sequential numbering,
the answer/evidence shape, answer length, the three prohibitions, the D18 scope sentence, the price
identity, every relative link, and the numeric trace. The module writes `models/viva.json` and exits
non-zero on any failure.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from src import config
from src.report import _NUMBER, _normalise, record_numbers, trace

VIVA_MD: Path = config.DOCS_DIR / "viva_qa.md"
RECORD_PATH: Path = config.MODELS_DIR / "viva.json"

MIN_QUESTIONS: int = 25
MIN_PER_CRITERION: int = 4
MIN_ANSWER_WORDS: int = 40
MIN_EVIDENCE_WORDS: int = 5
MIN_DOC_WORDS: int = 2500

CRITERIA: tuple[str, ...] = (
    "Problem definition and business understanding",
    "Dataset and data understanding",
    "Preprocessing and cleaning",
    "Exploratory analysis",
    "Feature engineering and selection",
    "Model selection and implementation",
    "Evaluation and validation",
    "Business insights and recommendations",
    "Documentation and reproducibility",
)

PROHIBITIONS: tuple[str, ...] = (
    "`duration` is never an input",
    "accuracy is never a headline",
    "`month` is never read as a cause",
)

SCOPE_PHRASE: str = "converts a fixed call budget"
PRICE_TOKENS: tuple[str, ...] = ("0.0750", "C / V", "1.50", "20.00")

COUNT = 0
FAILURES: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> bool:
    global COUNT
    COUNT += 1
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not passed:
        FAILURES.append(name)
    return passed


def words(text: str) -> int:
    return len(re.findall(r"\b[\w'\u2019-]+\b", text))


def parse_sections(doc: str) -> list[tuple[int, str, int]]:
    """Return (number, title, start offset) for each criterion section."""
    return [(int(match.group(1)), match.group(2).strip(), match.start())
            for match in re.finditer(r"^## (\d)\. (.+)$", doc, re.M)]


def parse_questions(doc: str) -> list[dict]:
    """Return each question with its number, title, block and owning criterion."""
    matches = list(re.finditer(r"^### Q(\d+)\. (.+)$", doc, re.M))
    sections = parse_sections(doc)
    questions: list[dict] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(doc)
        block = doc[match.end():end]
        criterion = 0
        for number, _title, start in sections:
            if start < match.start():
                criterion = number
        questions.append({
            "number": int(match.group(1)),
            "title": match.group(2).strip(),
            "block": block,
            "criterion": criterion,
        })
    return questions


def answer_text(block: str) -> str:
    match = re.search(r"\*\*Answer\.\*\*(.*?)(?=\*\*Evidence\.\*\*|\Z)", block, re.S)
    return match.group(1).strip() if match else ""


def evidence_text(block: str) -> str:
    match = re.search(r"\*\*Evidence\.\*\*(.*?)(?=\*\*If pressed\.\*\*|\Z)", block, re.S)
    return match.group(1).strip() if match else ""


def source_numbers() -> set[str]:
    """Numbers the bank may quote: records, tables, the report and the analysis documents.

    `viva_qa.md` is excluded because a document cannot be its own source. `demo_script.md` and
    `report_academic.md` are excluded because they are downstream of this bank, the same reason
    `src/report.py` excludes them from the report's sources.
    """
    allowed = record_numbers(exclude=RECORD_PATH)
    downstream = {VIVA_MD.name, "demo_script.md", "report_academic.md"}
    sources = [config.DOCS_DIR / "report.md"]
    sources.extend(path for path in sorted(config.DOCS_DIR.glob("*.md"))
                   if path.name not in downstream and path.name != "report.md")
    for path in sources:
        if path.exists():
            for token in _NUMBER.findall(path.read_text(encoding="utf-8")):
                allowed.add(_normalise(token))
    return allowed


def main() -> int:
    if not VIVA_MD.exists():
        print(f"missing {VIVA_MD}", file=sys.stderr)
        return 2
    doc = VIVA_MD.read_text(encoding="utf-8")
    print("=" * 78)
    print("T45 - the viva Q&A bank")
    print("=" * 78)
    print(f"  source {VIVA_MD.name}: {len(doc.splitlines())} lines, {words(doc):,} words")
    print()

    print("-- structure -------------------------------------------------------------")
    sections = parse_sections(doc)
    numbers = [number for number, _title, _start in sections]
    check("the nine criterion sections exist and are in order", numbers == list(range(1, 10)),
          f"{len(numbers)} sections: {numbers}")
    for number, title in enumerate(CRITERIA, start=1):
        matched = [entry for entry in sections if entry[0] == number and entry[1].lower() == title.lower()]
        check(f"criterion {number} has its own section", bool(matched), title)

    questions = parse_questions(doc)
    question_numbers = [question["number"] for question in questions]
    check(f"the bank has at least {MIN_QUESTIONS} questions", len(questions) >= MIN_QUESTIONS,
          f"{len(questions)} questions")
    check("question numbers are sequential and unique",
          question_numbers == list(range(1, len(questions) + 1)),
          f"{len(question_numbers)} numbered questions")

    print("-- coverage --------------------------------------------------------------")
    for number, title in enumerate(CRITERIA, start=1):
        owned = [question for question in questions if question["criterion"] == number]
        check(f"criterion {number} has at least {MIN_PER_CRITERION} questions",
              len(owned) >= MIN_PER_CRITERION,
              f"{len(owned)} questions: {[question['number'] for question in owned]}")

    print("-- answers ---------------------------------------------------------------")
    missing_answer = [question["number"] for question in questions if not answer_text(question["block"])]
    missing_evidence = [question["number"] for question in questions if not evidence_text(question["block"])]
    check("every question has an Answer", not missing_answer, f"missing: {missing_answer}" if missing_answer else "all")
    check("every question has an Evidence line", not missing_evidence,
          f"missing: {missing_evidence}" if missing_evidence else "all")

    short_answers = [question["number"] for question in questions
                     if words(answer_text(question["block"])) < MIN_ANSWER_WORDS]
    check(f"every answer is at least {MIN_ANSWER_WORDS} words", not short_answers,
          f"short: {short_answers}" if short_answers else f"minimum {min(words(answer_text(q['block'])) for q in questions)} words")

    thin_evidence = [question["number"] for question in questions
                     if words(evidence_text(question["block"])) < MIN_EVIDENCE_WORDS]
    check(f"every evidence line is at least {MIN_EVIDENCE_WORDS} words", not thin_evidence,
          f"thin: {thin_evidence}" if thin_evidence else "all")

    unsourced = [question["number"] for question in questions
                 if not re.search(r"`[^`]+\.(?:json|csv|md|py)`|\[[^\]]+\]\([^)]+\)|PROGRESS\.md",
                                  evidence_text(question["block"]))]
    check("every answer names a record, document or module", not unsourced,
          f"unsourced: {unsourced}" if unsourced else "all")

    print("-- the non-negotiables ---------------------------------------------------")
    lower = doc.lower()
    for phrase in PROHIBITIONS:
        check(f"the bank states the prohibition '{phrase}'", phrase.lower() in lower)
    check("the D18 scope sentence appears", SCOPE_PHRASE in doc)
    check("the price identity is stated", all(token in doc for token in PRICE_TOKENS),
          "0.0750, C / V, 1.50, 20.00")

    print("-- links -----------------------------------------------------------------")
    links = re.findall(r"\]\(([^)]+)\)", doc)
    broken: list[str] = []
    for target in links:
        if target.startswith(("http://", "https://", "#")):
            continue
        path = (VIVA_MD.parent / target).resolve()
        if not path.exists():
            broken.append(target)
    check("every relative link resolves", not broken,
          f"{len(links)} links, broken: {broken}" if broken else f"{len(links)} links")

    print("-- provenance ------------------------------------------------------------")
    allowed = source_numbers()
    total, unmatched = trace(doc, allowed)
    check("every decimal traces to a record, the report or an analysis document",
          not unmatched,
          f"{total} tokens, {len(allowed):,} admissible values"
          + (f"; unmatched: {unmatched[:12]}" if unmatched else ""))

    print()
    passed = COUNT - len(FAILURES)
    print(f"{passed}/{COUNT} checks passed")
    if FAILURES:
        for failure in FAILURES:
            print("  FAILED:", failure)
        return 1

    digest = hashlib.sha256(doc.encode("utf-8")).hexdigest()
    record = {
        "task": "T45",
        "document": "docs/viva_qa.md",
        "bytes": len(doc.encode("utf-8")),
        "sha256": digest,
        "lines": len(doc.splitlines()),
        "words": words(doc),
        "questions": len(questions),
        "per_criterion": {
            str(number): len([question for question in questions if question["criterion"] == number])
            for number in range(1, 10)
        },
        "criteria_covered": sorted({question["criterion"] for question in questions}),
        "prohibitions": list(PROHIBITIONS),
        "scope_phrase": SCOPE_PHRASE,
        "numeric_tokens": total,
        "admissible_values": len(allowed),
        "unmatched": unmatched,
        "links": len(links),
        "checks_passed": passed,
        "checks_total": COUNT,
        "checks_ok": not FAILURES,
    }
    RECORD_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"  record: {RECORD_PATH.name} ({RECORD_PATH.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
