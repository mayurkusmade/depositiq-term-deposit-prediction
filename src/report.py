"""
DepositIQ - the final report, rendered to PDF and checked against the records (T40).

Why this file exists
--------------------
T40 is the deliverable a marker reads top to bottom, and the one document in the project that could
plausibly contain a number no run produced. Every other document was written next to a module that
computed its figures; this one is a synthesis of twelve records and eleven documents, so the risk is not
arithmetic but **transcription**. A report that misquotes 0.4460 as 0.4640 is worse than no report,
because it looks authoritative and no test will ever run again.

So the report is checked three ways, and all three are mechanical:

1. **Structure** - all nine rubric criteria are present, in order, and each names the document or module
   that carries its evidence. A criterion that exists only as a heading fails.
2. **Coverage** - every figure the markdown embeds exists on disk, and every record cited in the
   appendix loads and parses.
3. **Provenance** - **every decimal number in the document is matched against a value held in a
   `models/*.json` record, a `reports/*.csv` column, or one of the eleven analysis documents.** A number
   that appears in the report and nowhere else is a defect, and the check fails rather than printing a
   warning. This is the same discipline T37, T38 and T39 applied to their own prose.

The four constraints from the task are asserted too, because they are the ones a reader cannot check by
skimming: the D18 scope sentence must appear **before the first euro total**; and the three prohibitions
- `duration` as an input, accuracy as a headline, `month` as a cause - must each appear in the section
where the relevant result is presented.

Rendering
---------
No new dependency. The markdown is converted to HTML by a deliberately small renderer that supports
exactly the syntax this report uses (headings, paragraphs, tables, lists, code fences, images, bold,
italic, inline code, links), and the HTML is printed to PDF by a headless Chromium-class browser that
every Windows and macOS machine already has. If no browser is found the PDF step is reported as failed
and the run exits non-zero, because a missing PDF is a missing deliverable.

How to run
----------
    python -m src.report      # ~3 s: checks, renders docs/report.html, prints docs/report.pdf

Writes `docs/report.html`, `docs/report.pdf` and `models/report.json`.
"""
from __future__ import annotations

import csv
import html as html_lib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from src import config

REPORT_MD: Path = config.DOCS_DIR / "report.md"
REPORT_HTML: Path = config.DOCS_DIR / "report.html"
REPORT_PDF: Path = config.DOCS_DIR / "report.pdf"
RECORD_PATH: Path = config.MODELS_DIR / "report.json"

# The nine criteria, in the order the rubric lists them, with the artefact that carries each one's
# evidence. The report must present them in this order and name the source; T40 is a synthesis, so a
# section that cannot point at its own evidence is a section that was written from memory.
CRITERIA: tuple[tuple[int, str, str], ...] = (
    (1, "Problem definition and business understanding", "01_problem_definition.md"),
    (2, "Dataset and data understanding", "02_data_dictionary.md"),
    (3, "Preprocessing and cleaning", "src/clean.py"),
    (4, "Exploratory analysis", "04_eda_insights.md"),
    (5, "Feature engineering and selection", "05_leakage_audit.md"),
    (6, "Model selection and implementation", "06_modeling.md"),
    (7, "Evaluation and validation", "07_threshold.md"),
    (8, "Business insights and recommendations", "08_recommendations.md"),
    (9, "Documentation and reproducibility", ""),
)

# D18: the sentence that travels with every headline number. p* sits below the base rate, so this model
# improves a campaign that already pays; it does not license cutting spend.
SCOPE_PHRASE: str = "converts a fixed call budget"

# The three prohibitions and where each must be visible, because a prohibition stated only in the
# executive summary is a claim a reader will not connect to the result it constrains.
PROHIBITIONS: tuple[tuple[str, str, str], ...] = (
    ("duration", "5.4", "never an input"),
    ("accuracy", "7.6", "not the headline"),
    ("month", "8.1", "Prohibition three"),
)

# Browsers that can print to PDF on the platforms this project runs on. Never a hard-coded install path
# for one machine: the first one that exists wins, and the search order prefers the browser the user is
# most likely to have.
BROWSER_CANDIDATES: tuple[str, ...] = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)

MIN_WORDS: int = 4000
MIN_PDF_BYTES: int = 100_000
MIN_PDF_PAGES: int = 8

# Documents admitted as sources for the provenance trace. `report.md` is excluded: the trace exists to
# prove the report did not invent a number, so quoting the report against itself would be vacuous.
# Downstream deliverables (the viva bank and the demo script) are excluded too: they quote the report, so
# admitting them would let a number circulate between two documents with no record behind it.
DOWNSTREAM_DOCS: frozenset[str] = frozenset({"viva_qa.md", "demo_script.md"})
SOURCE_DOCS: tuple[Path, ...] = tuple(
    path for path in sorted(config.DOCS_DIR.glob("*.md"))
    if path.name != REPORT_MD.name and path.name not in DOWNSTREAM_DOCS
)


# --------------------------------------------------------------------------- #
# Checks

COUNT = 0
FAILURES: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> bool:
    global COUNT
    COUNT += 1
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))
    if not passed:
        FAILURES.append(name)
    return passed


# --------------------------------------------------------------------------- #
# Markdown -> HTML
#
# A small renderer, not a parser. Every construct it supports is one this report actually uses, and an
# unsupported line is emitted as a paragraph rather than silently dropped - a lost table in a report is
# worse than an ugly one.

_CODE_TOKEN = "\x00CODE{}\x00"
_ESCAPED_STAR = "\x00STAR\x00"


def _inline(text: str) -> str:
    """Escape, then apply inline markdown: code spans, images, links, bold, italic."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("\\*", _ESCAPED_STAR)

    spans: list[str] = []

    def stash(match: re.Match[str]) -> str:
        spans.append(match.group(1))
        return _CODE_TOKEN.format(len(spans) - 1)

    text = re.sub(r"`([^`]+)`", stash, text)
    text = re.sub(
        r"!\[([^\]]*)\]\(([^)]+)\)",
        lambda m: f'<img src="{m.group(2)}" alt="{m.group(1)}">',
        text,
    )
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>',
        text,
    )
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    text = text.replace(_ESCAPED_STAR, "*")
    for index, code in enumerate(spans):
        text = text.replace(_CODE_TOKEN.format(index), f"<code>{code}</code>")
    return text


def _split_row(line: str) -> list[str]:
    """Split a table row on unescaped pipes, keeping the pipes inside code spans."""
    cells, current, in_code = [], [], False
    for char in line.strip().strip("|"):
        if char == "`":
            in_code = not in_code
            current.append(char)
        elif char == "|" and not in_code:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    cells.append("".join(current).strip())
    return cells


def render_body(markdown: str) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            language = stripped[3:].strip()
            index += 1
            block: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            index += 1
            body = html_lib.escape("\n".join(block))
            out.append(f'<pre class="code"><code>{body}</code></pre>')
            continue

        if stripped.startswith("|") and index + 1 < len(lines) and re.match(
            r"^\|[\s:|-]+\|$", lines[index + 1].strip()
        ):
            header = _split_row(stripped)
            index += 2
            rows: list[list[str]] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(_split_row(lines[index]))
                index += 1
            head = "".join(f"<th>{_inline(cell)}</th>" for cell in header)
            body = "".join(
                "<tr>" + "".join(f"<td>{_inline(cell)}</td>" for cell in row) + "</tr>" for row in rows
            )
            out.append(f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>')
            continue

        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            level = len(heading.group(1))
            out.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            index += 1
            continue

        if re.match(r"^---+$", stripped):
            out.append('<hr class="rule">')
            index += 1
            continue

        if stripped.startswith("> "):
            quoted: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("> "):
                quoted.append(lines[index].strip()[2:])
                index += 1
            out.append(f"<blockquote>{_inline(' '.join(quoted))}</blockquote>")
            continue

        if re.match(r"^[-*]\s+", stripped):
            items: list[str] = []
            while index < len(lines) and re.match(r"^[-*]\s+", lines[index].strip()):
                items.append(re.sub(r"^[-*]\s+", "", lines[index].strip()))
                index += 1
            body = "".join(f"<li>{_inline(item)}</li>" for item in items)
            out.append(f"<ul>{body}</ul>")
            continue

        if re.match(r"^\d+\.\s+", stripped):
            items = []
            while index < len(lines) and re.match(r"^\d+\.\s+", lines[index].strip()):
                items.append(re.sub(r"^\d+\.\s+", "", lines[index].strip()))
                index += 1
            body = "".join(f"<li>{_inline(item)}</li>" for item in items)
            out.append(f"<ol>{body}</ol>")
            continue

        if not stripped:
            index += 1
            continue

        paragraph: list[str] = []
        while index < len(lines):
            current = lines[index].strip()
            if not current or current.startswith(("#", "|", ">", "```")) or re.match(r"^[-*]\s", current):
                break
            paragraph.append(current)
            index += 1
        text = " ".join(paragraph)
        if text.startswith("!["):
            out.append(f'<figure>{_inline(text)}</figure>')
        else:
            out.append(f"<p>{_inline(text)}</p>")
    return "\n".join(out)


CSS = """
@page { size: A4; margin: 17mm 16mm 18mm 16mm; }
html { -webkit-print-color-adjust: exact; }
body { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 10pt;
       line-height: 1.52; color: #1b1b1b; margin: 0; }
h1 { font-size: 25pt; line-height: 1.15; margin: 0 0 4mm 0; color: #0b3d63;
     border-bottom: 2.5pt solid #0b3d63; padding-bottom: 3mm; }
h2 { font-size: 15pt; color: #0b3d63; margin: 0 0 3mm 0; padding-top: 2mm;
     border-bottom: 0.8pt solid #c8d4de; page-break-before: always; page-break-after: avoid; }
h3 { font-size: 11.5pt; color: #14527f; margin: 6mm 0 2mm 0; page-break-after: avoid; }
h4 { font-size: 10.5pt; margin: 4mm 0 1.5mm 0; page-break-after: avoid; }
p { margin: 0 0 2.6mm 0; text-align: justify; }
ul, ol { margin: 0 0 3mm 0; padding-left: 6mm; }
li { margin-bottom: 1.2mm; }
strong { color: #0b3d63; }
code { font-family: "Cascadia Mono", Consolas, monospace; font-size: 8.6pt;
       background: #f2f5f8; padding: 0.4mm 1mm; border-radius: 1.2mm; }
pre.code { background: #f5f7f9; border-left: 2.5pt solid #9fb6c8; padding: 3mm 4mm;
           font-size: 8.4pt; line-height: 1.4; page-break-inside: avoid; white-space: pre-wrap; }
pre.code code { background: none; padding: 0; font-size: 8.4pt; }
table { width: 100%; border-collapse: collapse; margin: 0 0 4mm 0; font-size: 8.6pt;
        page-break-inside: avoid; }
thead { background: #0b3d63; color: #ffffff; }
th { text-align: left; padding: 1.8mm 2mm; font-weight: 600; }
td { padding: 1.5mm 2mm; border-bottom: 0.4pt solid #d8e0e6; vertical-align: top; }
tbody tr:nth-child(even) { background: #f6f8fa; }
figure { margin: 4mm 0 5mm 0; page-break-inside: avoid; text-align: center; }
img { max-width: 100%; max-height: 155mm; border: 0.4pt solid #d8e0e6; }
blockquote { margin: 3mm 0; padding: 2mm 4mm; border-left: 2.5pt solid #0b3d63;
             background: #f4f7fa; }
hr.rule { border: none; border-top: 0.6pt solid #c8d4de; margin: 5mm 0; }
a { color: #14527f; text-decoration: none; }
"""


def build_html(body: str, title: str) -> str:
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{html_lib.escape(title)}</title>\n<style>{CSS}</style>\n</head>\n<body>\n"
        f"{body}\n</body>\n</html>\n"
    )


# --------------------------------------------------------------------------- #
# PDF

def find_browser() -> str | None:
    for candidate in BROWSER_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    for name in ("chrome", "msedge", "chromium", "google-chrome"):
        found = shutil.which(name)
        if found:
            return found
    return None


def to_pdf(html_path: Path, pdf_path: Path) -> tuple[bool, str]:
    browser = find_browser()
    if browser is None:
        return False, "no Chromium-class browser found; the HTML is written but the PDF was not"
    if pdf_path.exists():
        pdf_path.unlink()
    with tempfile.TemporaryDirectory() as profile:
        command = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            "--virtual-time-budget=10000",
            f"--user-data-dir={profile}",
            f"--print-to-pdf={pdf_path}",
            html_path.resolve().as_uri(),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if not pdf_path.exists():
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        return False, f"browser exited {result.returncode}: {' '.join(detail[-3:])[:300]}"
    return True, f"{Path(browser).name}"[:60]


def pdf_stats(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    counts = [int(m) for m in re.findall(rb"/Count\s+(\d+)", data)]
    return len(data), (max(counts) if counts else 0)


# --------------------------------------------------------------------------- #
# Provenance

_NUMBER = re.compile(r"[+\-\u2212]?(?:\d{1,3}(?:,\d{3})+|\d+)\.\d+")


def _normalise(token: str) -> str:
    return token.replace(",", "").replace("\u2212", "-").lstrip("+")


def _admit(target: set[str], value: float) -> None:
    """Admit a value at every precision the prose might round it to, raw and as a percentage."""
    for decimals in range(0, 7):
        for scaled in (value, value * 100):
            for sign in (1, -1):
                target.add(_normalise(f"{sign * abs(scaled) if scaled < 0 else scaled:,.{decimals}f}"))
                target.add(_normalise(f"{abs(scaled):,.{decimals}f}"))
            target.add(_normalise(f"{abs(scaled):,.{decimals}f}"))


def _walk_numbers(node: object, target: set[str]) -> None:
    if isinstance(node, dict):
        for value in node.values():
            _walk_numbers(value, target)
    elif isinstance(node, list):
        for value in node:
            _walk_numbers(value, target)
    elif isinstance(node, bool):
        return
    elif isinstance(node, (int, float)):
        _admit(target, float(node))
    elif isinstance(node, str):
        for token in _NUMBER.findall(node):
            target.add(_normalise(token))


def record_numbers(exclude: Path | None = None) -> set[str]:
    """Every number the records hold.

    `exclude` names a second record to skip besides this module's own, which T41 uses so the deck cannot
    be validated against the numbers it just wrote: a trace that admits its own output proves nothing.
    """
    allowed: set[str] = set()
    skip = {RECORD_PATH.name} | ({exclude.name} if exclude is not None else set())
    for path in sorted(config.MODELS_DIR.glob("*.json")):
        if path.name in skip:
            continue
        _walk_numbers(json.loads(path.read_text(encoding="utf-8")), allowed)
    for path in sorted(config.REPORTS_DIR.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for value in row.values():
                    for token in _NUMBER.findall(str(value) or ""):
                        allowed.add(_normalise(token))
                    try:
                        _admit(allowed, float(str(value)))
                    except (TypeError, ValueError):
                        continue
    return allowed


def document_numbers() -> set[str]:
    allowed: set[str] = set()
    for path in SOURCE_DOCS:
        for token in _NUMBER.findall(path.read_text(encoding="utf-8")):
            allowed.add(_normalise(token))
    return allowed


def trace(doc: str, allowed: set[str]) -> tuple[int, list[str]]:
    tokens = [_normalise(token) for token in _NUMBER.findall(doc)]
    unmatched = sorted({token for token in tokens if token not in allowed})
    return len(tokens), unmatched


# --------------------------------------------------------------------------- #
# Main

def main() -> int:
    config.ensure_dirs()
    if not REPORT_MD.exists():
        print(f"missing {REPORT_MD}", file=sys.stderr)
        return 2

    doc = REPORT_MD.read_text(encoding="utf-8")
    words = len(re.findall(r"\b[\w'\u2019-]+\b", doc))
    print("=" * 78)
    print("T40 - the final report")
    print("=" * 78)
    print(f"  source {REPORT_MD.name}: {len(doc.splitlines())} lines, {words:,} words")
    print()

    print("-- structure -------------------------------------------------------------")
    positions: dict[int, int] = {}
    for number, title, source in CRITERIA:
        match = re.search(rf"^## {number}\. ", doc, flags=re.MULTILINE)
        positions[number] = match.start() if match else -1
        check(
            f"criterion {number} ({title}) has its own section",
            match is not None,
            match.group(0).strip() if match else "no heading '## {}. '".format(number),
        )
    check(
        "the nine criteria appear in rubric order",
        all(positions[a] < positions[b] for a, b in zip(sorted(positions), sorted(positions)[1:]))
        if all(p >= 0 for p in positions.values())
        else False,
        " -> ".join(str(positions[n]) for n in sorted(positions)),
    )
    for number, title, source in CRITERIA:
        if not source:
            continue
        section = doc[positions[number] :] if positions[number] >= 0 else ""
        section = section.split("\n## ")[0]
        check(f"criterion {number} names its evidence", source in section, source)

    print()
    print("-- coverage --------------------------------------------------------------")
    figures = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", doc)
    missing = [path for path in figures if not (REPORT_MD.parent / path).exists()]
    check(
        "every embedded figure exists on disk",
        not missing,
        f"{len(figures)} figures, {len(set(figures))} distinct"
        + (f"; missing: {missing}" if missing else ""),
    )
    records = sorted(set(re.findall(r"`models/([a-z_]+\.json)`", doc)))
    broken = [name for name in records if not (config.MODELS_DIR / name).exists()]
    unparsed: list[str] = []
    for name in records:
        path = config.MODELS_DIR / name
        if path.exists():
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                unparsed.append(name)
    check(
        "every cited record exists and parses",
        not broken and not unparsed,
        f"{len(records)} records cited" + (f"; broken: {broken + unparsed}" if broken + unparsed else ""),
    )
    check("the report is long enough to cover nine criteria", words >= MIN_WORDS, f"{words:,} words")

    print()
    print("-- provenance ------------------------------------------------------------")
    allowed = record_numbers() | document_numbers()
    total, unmatched = trace(doc, allowed)
    check(
        "every decimal in the report traces to a record or a source document",
        not unmatched,
        f"{total} tokens, {len(allowed):,} admissible values"
        + (f"; unmatched: {unmatched[:12]}" if unmatched else ""),
    )

    print()
    print("-- the four constraints --------------------------------------------------")
    scope_at = doc.find(SCOPE_PHRASE)
    euro_at = doc.find("EUR")
    check(
        "the D18 scope sentence precedes the first euro total",
        scope_at >= 0 and euro_at >= 0 and scope_at < euro_at,
        f"scope at {scope_at}, first EUR at {euro_at}",
    )
    for topic, section, phrase in PROHIBITIONS:
        block = doc.split(f"### {section} ")[-1].split("\n### ")[0] if f"### {section} " in doc else ""
        check(
            f"the {topic} prohibition is stated in section {section}",
            phrase in block,
            f"'{phrase}' in {len(block)} chars",
        )
    # "assumed"/"assumptions" both count: the requirement is that the reader is told the prices are not
    # the bank's, before the first total that depends on them, in whatever grammatical form fits.
    preamble = doc[: euro_at + 400].lower()
    check(
        "the prices are declared as assumptions next to the first euro figure",
        "assum" in preamble,
        "the price assumption precedes the first EUR total",
    )

    print()
    print("-- rendering -------------------------------------------------------------")
    body = render_body(doc)
    REPORT_HTML.write_text(build_html(body, "DepositIQ - Final Report"), encoding="utf-8")
    check(
        "the HTML was written",
        REPORT_HTML.exists() and REPORT_HTML.stat().st_size > 20_000,
        f"{REPORT_HTML.stat().st_size:,} bytes" if REPORT_HTML.exists() else "missing",
    )

    pdf_ok, pdf_detail = to_pdf(REPORT_HTML, REPORT_PDF)
    bytes_written, pages = pdf_stats(REPORT_PDF) if REPORT_PDF.exists() else (0, 0)
    check("the PDF was rendered", pdf_ok, pdf_detail)
    check(
        "the PDF is a real document, not an empty print",
        bytes_written >= MIN_PDF_BYTES and pages >= MIN_PDF_PAGES,
        f"{bytes_written:,} bytes, {pages} pages",
    )

    print()
    print(f"{COUNT - len(FAILURES)}/{COUNT} checks passed")
    if FAILURES:
        for failure in FAILURES:
            print("  FAILED:", failure)

    RECORD_PATH.write_text(
        json.dumps(
            {
                "task": "T40",
                "report": str(REPORT_MD.relative_to(config.ROOT)),
                "words": words,
                "lines": len(doc.splitlines()),
                "figures": len(figures),
                "distinct_figures": len(set(figures)),
                "records_cited": len(records),
                "numeric_tokens": total,
                "admissible_values": len(allowed),
                "unmatched": unmatched,
                "criteria": [
                    {"number": number, "title": title, "source": source} for number, title, source in CRITERIA
                ],
                "html_bytes": REPORT_HTML.stat().st_size if REPORT_HTML.exists() else 0,
                "pdf_bytes": bytes_written,
                "pdf_pages": pages,
                "pdf_renderer": pdf_detail,
                "checks_passed": COUNT - len(FAILURES),
                "checks_total": COUNT,
                "checks_ok": not FAILURES,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
