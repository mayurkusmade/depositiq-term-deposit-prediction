"""
DepositIQ - the five-minute demo script (T46).

Why this file exists
--------------------
A demo script is where a project is most likely to lie. The slides are checked, the report is checked, and
then the spoken track quietly acquires a number that no record holds - a rounded euro total, a recall
figure remembered from a different cut, a runtime that sounded good. The script is therefore treated like
every other deliverable: it is parsed, timed, and traced.

What is checked
---------------
* **The clock.** Six segments, contiguous timecodes, a 5:00 total, and a spoken word count that fits the
  time at a human speaking rate.
* **The shape.** Every segment has a **Say**, a **Show** and an **Evidence** line, and every evidence line
  names a file that exists.
* **The claims.** Every number spoken appears once in the claims table; every claims-table row names an
  existing file; a `record` value is re-found in that file, and a `derived` value is recomputed from
  operands that the file holds.
* **The non-negotiables.** The three prohibitions, the D18 scope sentence, the price identity, the five
  numbers to know cold, and the commands that start the demo and this validator.
* **Provenance.** Every decimal in the whole document is matched against a `models/*.json` record, a
  `reports/*.csv` column, the report or an analysis document. `demo_script.md` and `viva_qa.md` are
  excluded from the sources so the script cannot admit a number by quoting itself.

The module writes `models/demo_script.json` and exits non-zero on any failure.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from src import config
from src.report import _NUMBER, _admit, _normalise, _walk_numbers, record_numbers, trace

SCRIPT_MD: Path = config.DOCS_DIR / "demo_script.md"
RECORD_PATH: Path = config.MODELS_DIR / "demo_script.json"

TOTAL_SECONDS: int = 300
MIN_SEGMENTS: int = 6
MIN_CLAIMS: int = 25
MIN_CHECKLIST_ITEMS: int = 15
MIN_CHECKLIST_SECTIONS: int = 4
MIN_SPOKEN_WORDS: int = 600
MAX_SPOKEN_WORDS: int = 900
MIN_WORDS_PER_SECOND: float = 1.8
MAX_WORDS_PER_SECOND: float = 3.0
MIN_DOC_WORDS: int = 2_000
MAX_DOC_WORDS: int = 6_000

PROHIBITIONS: tuple[str, ...] = (
    "`duration` is never an input",
    "accuracy is never a headline",
    "`month` is never read as a cause",
)
SCOPE_PHRASE: str = "converts a fixed call budget"
PRICE_TOKENS: tuple[str, ...] = ("0.0750", "C / V", "1.50", "20.00")

SEGMENT = re.compile(r"^### (\d+):(\d\d)[–-](\d+):(\d\d) — (.+)$", re.M)
CLAIM_ROW = re.compile(r"^\| (\d+) \| (.+?) \| (.+?) \| `([^`]+)` \| (record|derived) \| (.+?) \|$", re.M)
OPERAND = re.compile(r"[+\-\u2212]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")

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


def seconds(minutes: str, secs: str) -> int:
    return int(minutes) * 60 + int(secs)


def parse_segments(doc: str) -> list[dict]:
    matches = list(SEGMENT.finditer(doc))
    segments: list[dict] = []
    for index, match in enumerate(matches):
        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = doc.find("\n## ", match.end())
            if end < 0:
                end = len(doc)
        block = doc[match.end():end]
        segments.append({
            "start": seconds(match.group(1), match.group(2)),
            "end": seconds(match.group(3), match.group(4)),
            "title": match.group(5).strip(),
            "block": block,
        })
    return segments


def section(doc: str, heading: str, next_heading: str) -> str:
    start = doc.find(heading)
    if start < 0:
        return ""
    end = doc.find(next_heading, start + len(heading))
    return doc[start:end] if end >= 0 else doc[start:]


def block_after(block: str, marker: str, stops: tuple[str, ...]) -> str:
    start = block.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    end = len(block)
    for stop in stops:
        position = block.find(stop, start)
        if 0 <= position < end:
            end = position
    return block[start:end].strip()


def numbers_in(path: Path) -> set[str]:
    """Every number a single file holds, at every precision prose might round to."""
    allowed: set[str] = set()
    if path.suffix == ".json":
        _walk_numbers(json.loads(path.read_text(encoding="utf-8")), allowed)
    elif path.suffix == ".csv":
        import csv
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            rows = [row for row in reader if row]
        if rows:
            allowed.add(str(len(rows) - 1))
            allowed.add(str(len(rows[0])))
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for value in row.values():
                    text = str(value) or ""
                    for token in _NUMBER.findall(text):
                        allowed.add(_normalise(token))
                    try:
                        _admit(allowed, float(text))
                    except (TypeError, ValueError):
                        continue
    elif path.suffix == ".md":
        for token in _NUMBER.findall(path.read_text(encoding="utf-8")):
            allowed.add(_normalise(token))
    return allowed


def source_numbers() -> set[str]:
    allowed = record_numbers(exclude=RECORD_PATH)
    downstream = {SCRIPT_MD.name, "viva_qa.md"}
    sources = [config.DOCS_DIR / "report.md"]
    sources.extend(path for path in sorted(config.DOCS_DIR.glob("*.md"))
                   if path.name not in downstream and path.name != "report.md")
    for path in sources:
        if path.exists():
            for token in _NUMBER.findall(path.read_text(encoding="utf-8")):
                allowed.add(_normalise(token))
    return allowed


def evaluate(formula: str) -> float | None:
    tokens = re.findall(r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[+\-*/]", formula)
    if not tokens:
        return None
    values: list[float] = []
    operators: list[str] = []
    for token in tokens:
        if token in {"+", "-", "*", "/"}:
            operators.append(token)
        else:
            values.append(float(token.replace(",", "")))
    if not values or len(operators) != len(values) - 1:
        return None
    result = values[0]
    for operator, value in zip(operators, values[1:]):
        if operator == "+":
            result += value
        elif operator == "-":
            result -= value
        elif operator == "*":
            result *= value
        elif operator == "/":
            if value == 0:
                return None
            result /= value
    return result


def decimals_in(text: str) -> list[str]:
    return [_normalise(token) for token in _NUMBER.findall(text)]


def main() -> int:
    if not SCRIPT_MD.exists():
        print(f"missing {SCRIPT_MD}", file=sys.stderr)
        return 2
    doc = SCRIPT_MD.read_text(encoding="utf-8")
    print("=" * 78)
    print("T46 - the five-minute demo script")
    print("=" * 78)
    print(f"  source {SCRIPT_MD.name}: {len(doc.splitlines())} lines, {words(doc):,} words")
    print()

    print("-- the clock -------------------------------------------------------------")
    segments = parse_segments(doc)
    check(f"the script has {MIN_SEGMENTS} timed segments", len(segments) == MIN_SEGMENTS,
          f"{len(segments)} segments")
    check("the first segment starts at 0:00", bool(segments) and segments[0]["start"] == 0,
          f"{segments[0]['start']} s" if segments else "no segments")
    contiguous = all(
        segments[index]["end"] == segments[index + 1]["start"]
        for index in range(len(segments) - 1)
    )
    check("the segments are contiguous, with no gap or overlap", contiguous,
          " -> ".join(f"{segment['start']}-{segment['end']}" for segment in segments))
    check(f"the script ends at {TOTAL_SECONDS // 60}:{TOTAL_SECONDS % 60:02d}",
          bool(segments) and segments[-1]["end"] == TOTAL_SECONDS,
          f"{segments[-1]['end']} s" if segments else "no segments")

    print("-- the shape -------------------------------------------------------------")
    for index, segment in enumerate(segments, start=1):
        say = block_after(segment["block"], "**Say.**", ("**Show.**", "**Evidence.**"))
        show = block_after(segment["block"], "**Show.**", ("**Evidence.**",))
        evidence = block_after(segment["block"], "**Evidence.**", ())
        check(f"segment {index} has a Say, a Show and an Evidence line",
              bool(say) and bool(show) and bool(evidence),
              f"{segment['title']}: say {words(say)} words")
        named = re.findall(r"`([^`]+\.(?:json|csv|md|png|py|pdf|pptx))`", evidence)
        resolved = [name for name in named if (config.ROOT / name).exists()]
        check(f"segment {index} names an evidence file that exists", bool(resolved),
              f"{len(resolved)} of {len(named)} resolve: {resolved[:3]}")

    spoken = sum(
        words(block_after(segment["block"], "**Say.**", ("**Show.**", "**Evidence.**")))
        for segment in segments
    )
    rate = spoken / TOTAL_SECONDS
    check(f"the spoken text is {MIN_SPOKEN_WORDS}-{MAX_SPOKEN_WORDS} words", MIN_SPOKEN_WORDS <= spoken <= MAX_SPOKEN_WORDS,
          f"{spoken} words")
    check(f"the speaking rate is {MIN_WORDS_PER_SECOND}-{MAX_WORDS_PER_SECOND} words per second",
          MIN_WORDS_PER_SECOND <= rate <= MAX_WORDS_PER_SECOND,
          f"{rate:.2f} words/s = {rate * 60:.0f} words/min")

    print("-- the claims ------------------------------------------------------------")
    claims_text = section(doc, "## Claims and where they live", "## Three refusals")
    claims = [
        {"number": int(match.group(1)), "claim": match.group(2).strip(), "value": match.group(3).strip(),
         "source": match.group(4).strip(), "kind": match.group(5), "derivation": match.group(6).strip()}
        for match in CLAIM_ROW.finditer(claims_text)
    ]
    check(f"the claims table has at least {MIN_CLAIMS} rows", len(claims) >= MIN_CLAIMS,
          f"{len(claims)} claims")
    check("the claim numbers are sequential",
          [claim["number"] for claim in claims] == list(range(1, len(claims) + 1)),
          f"{len(claims)} rows")

    script_text = " ".join(
        block_after(segment["block"], "**Say.**", ("**Show.**", "**Evidence.**"))
        for segment in segments
    )
    script_decimals = set(decimals_in(script_text))
    script_numbers = {_normalise(token) for token in OPERAND.findall(script_text)}
    claim_decimals = {_normalise(token) for claim in claims for token in _NUMBER.findall(claim["value"])}
    missing_claims = sorted(script_decimals - claim_decimals)
    check("every decimal spoken appears in the claims table", not missing_claims,
          f"{len(script_decimals)} spoken decimals" + (f"; missing: {missing_claims}" if missing_claims else ""))

    def clean_value(value: str) -> str:
        return re.sub(r"[^0-9.\-]", "", value.replace(",", "").replace("\u2212", "-"))

    missing_source: list[str] = []
    missing_value: list[str] = []
    bad_derivation: list[str] = []
    for claim in claims:
        path = config.ROOT / claim["source"]
        if not path.exists():
            missing_source.append(claim["source"])
            continue
        if clean_value(claim["value"]) not in script_numbers:
            missing_value.append(f"{claim['number']}:{claim['value']}")
        allowed = numbers_in(path)
        if claim["kind"] == "record":
            tokens = decimals_in(claim["value"]) or [clean_value(claim["value"])]
            if any(token not in allowed for token in tokens):
                missing_value.append(f"{claim['number']}:{claim['value']} not in {claim['source']}")
        else:
            result = evaluate(claim["derivation"])
            if result is None:
                bad_derivation.append(f"{claim['number']}: unparsable '{claim['derivation']}'")
                continue
            target = float(re.sub(r"[^0-9.\-]", "", claim["value"]))
            places = len(claim["value"].split(".")[1].rstrip("%")) if "." in claim["value"] else 0
            tolerance = 0.5 * 10 ** (-places) + 1e-9
            if abs(result - target) > tolerance:
                bad_derivation.append(f"{claim['number']}: {claim['derivation']} = {result:.6f}, not {target}")
            operands = [_normalise(token) for token in OPERAND.findall(claim["derivation"])]
            if any(operand not in allowed for operand in operands if operand != "100"):
                bad_derivation.append(f"{claim['number']}: operands not in {claim['source']}")

    check("every claims row names a file that exists", not missing_source,
          f"{len({claim['source'] for claim in claims})} distinct files" if not missing_source else f"missing: {missing_source}")
    check("every claims value is spoken and traceable to its named file", not missing_value,
          "all" if not missing_value else f"unresolved: {missing_value[:8]}")
    check("every derived claim recomputes from operands the file holds", not bad_derivation,
          "all" if not bad_derivation else f"bad: {bad_derivation[:6]}")

    print("-- the non-negotiables ---------------------------------------------------")
    lower = doc.lower()
    for phrase in PROHIBITIONS:
        check(f"the script states the prohibition '{phrase}'", phrase.lower() in lower)
    check("the D18 scope sentence appears", SCOPE_PHRASE in doc)
    check("the price identity is stated", all(token in doc for token in PRICE_TOKENS),
          "0.0750, C / V, 1.50, 20.00")
    check("the commands to start the demo and this validator are present",
          "streamlit run app/app.py" in doc and "python -m src.demo" in doc and "python -m src.demo_script" in doc)

    print("-- five numbers and the checklist ----------------------------------------")
    five_text = section(doc, "## If you only remember five numbers", "## Rehearsal checklist")
    five_rows = [line for line in five_text.splitlines() if line.startswith("| **")]
    check("the five-numbers table has five rows", len(five_rows) == 5, f"{len(five_rows)} rows")
    five_ok = True
    for row in five_rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        number, source = cells[0].strip("* "), cells[-1].strip("` ")
        path = config.ROOT / source
        if not path.exists() or not decimals_in(number):
            five_ok = False
            continue
        if any(token not in numbers_in(path) for token in decimals_in(number)):
            five_ok = False
    check("every five-numbers row traces to its record", five_ok, f"{len(five_rows)} rows")

    checklist = section(doc, "## Rehearsal checklist", "## Hand-off to questions")
    items = re.findall(r"^- \[ \] .+$", checklist, re.M)
    subsections = re.findall(r"^### .+$", checklist, re.M)
    check(f"the rehearsal checklist has at least {MIN_CHECKLIST_ITEMS} items", len(items) >= MIN_CHECKLIST_ITEMS,
          f"{len(items)} items")
    check(f"the rehearsal checklist has at least {MIN_CHECKLIST_SECTIONS} sections",
          len(subsections) >= MIN_CHECKLIST_SECTIONS, f"{len(subsections)} sections")
    check("the checklist covers a failure path", "If something breaks" in checklist)

    print("-- links and provenance --------------------------------------------------")
    links = re.findall(r"\]\(([^)]+)\)", doc)
    broken: list[str] = []
    for target in links:
        if target.startswith(("http://", "https://", "#")):
            continue
        if not (SCRIPT_MD.parent / target).resolve().exists():
            broken.append(target)
    check("every relative link resolves", not broken,
          f"{len(links)} links" if not broken else f"broken: {broken}")

    allowed = source_numbers()
    total, unmatched = trace(doc, allowed)
    check("every decimal traces to a record, the report or an analysis document", not unmatched,
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
        "task": "T46",
        "document": "docs/demo_script.md",
        "bytes": len(doc.encode("utf-8")),
        "sha256": digest,
        "lines": len(doc.splitlines()),
        "words": words(doc),
        "spoken_words": spoken,
        "words_per_second": round(rate, 4),
        "segments": [
            {"start": segment["start"], "end": segment["end"], "title": segment["title"]}
            for segment in segments
        ],
        "total_seconds": TOTAL_SECONDS,
        "claims": len(claims),
        "claim_sources": sorted({claim["source"] for claim in claims}),
        "five_numbers": 5,
        "checklist_items": len(items),
        "checklist_sections": len(subsections),
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
