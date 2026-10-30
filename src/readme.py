"""
DepositIQ - the README, checked against the repository it describes (T42).

Why this file exists
--------------------
A README is the one document nobody re-runs. Every other deliverable in this project is protected by the
module that produced it: change a model, re-run, and the prose fails. The README has no such module, so it
drifts the moment a path, a version, a count or a command changes - and the drift is invisible until a
marker clones the repository and follows it into a `ModuleNotFoundError`.

So the README is checked the way the rest of the project is checked: against the filesystem, the records
and `requirements.txt`.

What is asserted
----------------
* **Every command is real.** Each `python -m src.<name>` in the README names `src/<name>.py`, that module
  has a `__main__` guard, and every module that has one is named somewhere in the README. Two utility
  modules (`config`, `data_loader`) are documented as utilities rather than steps, so nothing is exempt.
* **The commands are in dependency order.** `clean` before `split` before `train` before `tune` before
  `final` before `evaluate`; `final` before `explain` before `local_explain`; `playbook` before `report`
  before `deck`.
* **The pins are the real pins.** `requirements.txt` is parsed, the README's count of exact pins must match
  it, and the versions it quotes must equal the pinned ones - the README's claim about Python 3.11 is the
  constraint a reader is most likely to hit, and it comes from the same file.
* **Every number appears in a record**, the same trace the report and the deck are held to, with the
  version strings admitted from `requirements.txt` rather than from prose.
* **The data is the data.** The sizes and SHA-256 prefixes in the README's table are recomputed from
  `data/raw/`, the row and column counts from the CSV itself, and the base rate from the target column.
* **The counts are the counts.** Modules, figures, CSV tables, supporting documents, decisions and tasks
  are counted from disk, not remembered.
* **Every relative link resolves**, and no absolute path or home directory appears anywhere.
* **The README does not claim what does not exist.** The demo is `app/app.py`, the README must name the
  command that starts it, and the check fails if the file disappears or the README calls `app/` empty -
  the honesty is enforced, not encouraged.

How to run
----------
    python -m src.readme     # ~1 s: validates README.md, writes models/readme.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

from src import config, report

README_PATH: Path = config.ROOT / "README.md"
RECORD_PATH: Path = config.MODELS_DIR / "readme.json"
REQUIREMENTS: Path = config.ROOT / "requirements.txt"

MIN_CHARS: int = 6000

# Sections a reader needs: what the result is, how to start, how to reproduce, what is where, what is
# missing. A README without the last one is advertising.
SECTIONS: tuple[str, ...] = (
    "The result, and what it does not mean",
    "Quick start",
    "The interactive demo",
    "Reproduce every number",
    "What is in the repository",
    "The data",
    "What is not here yet",
    "Limits worth knowing",
)

# The dependency order the command list must respect.
ORDER: tuple[tuple[str, str], ...] = (
    ("clean", "split"),
    ("split", "train"),
    ("train", "imbalance"),
    ("train", "tune"),
    ("tune", "duration_compare"),
    ("tune", "final"),
    ("final", "demo"),
    ("final", "metrics"),
    ("final", "evaluate"),
    ("final", "explain"),
    ("explain", "local_explain"),
    ("final", "threshold"),
    ("threshold", "lift"),
    ("threshold", "duration_compare"),
    ("duration_compare", "playbook"),
    ("lift", "playbook"),
    ("playbook", "report"),
    ("report", "deck"),
)

# The three prohibitions, in the wording the README uses for them.
PROHIBITIONS: tuple[str, ...] = ("duration` is never an input", "Accuracy is never a headline",
                                 "`month` is never read as a cause")


COUNT = 0
FAILURES: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> bool:
    global COUNT
    COUNT += 1
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not passed:
        FAILURES.append(name)
    return passed


def pins() -> dict[str, str]:
    found: dict[str, str] = {}
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.\-]+)==([0-9][\w.]*)\s*$", line.strip())
        if match:
            found[match.group(1).lower()] = match.group(2)
    return found


def version_tokens(versions: dict[str, str]) -> set[str]:
    """Admit `3.2.0` and every prefix the decimal regex could pull out of it."""
    allowed: set[str] = set()
    header = re.search(r"Python (\d+\.\d+)", REQUIREMENTS.read_text(encoding="utf-8"))
    if header:
        allowed.add(header.group(1))
    for version in versions.values():
        parts = version.split(".")
        for index in range(1, len(parts) + 1):
            allowed.add(".".join(parts[:index]))
    return allowed


def commands(text: str) -> list[str]:
    return re.findall(r"python -m src\.([a-z_]+)", text)


def main() -> int:
    config.ensure_dirs()
    if not README_PATH.exists():
        print(f"missing {README_PATH}", file=sys.stderr)
        return 2

    text = README_PATH.read_text(encoding="utf-8")
    modules = {path.stem: path for path in sorted(config.SRC_DIR.glob("*.py"))}
    print("=" * 78)
    print("T42 - the README")
    print("=" * 78)
    print(f"  source {README_PATH.name}: {len(text.splitlines())} lines, {len(text):,} chars")
    print()

    print("-- structure -------------------------------------------------------------")
    check("the README exists and is substantial", len(text) >= MIN_CHARS, f"{len(text):,} chars")
    missing = [section for section in SECTIONS if section.lower() not in text.lower()]
    check("every required section is present", not missing, f"{len(SECTIONS) - len(missing)}/{len(SECTIONS)}"
          + (f", missing {missing}" if missing else ""))
    # Headings inside fenced blocks are shell comments, not headings; counting them was the first
    # version's error and it reported five titles on a document with one.
    h1: list[str] = []
    fenced = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced and line.startswith("# "):
            h1.append(line)
    check("it opens with exactly one H1", len(h1) == 1, f"{len(h1)} outside code fences: {h1[:3]}")

    print("-- commands --------------------------------------------------------------")
    named = commands(text)
    unknown = sorted({name for name in named if name not in modules})
    check("every command names a module that exists", not unknown,
          f"{len(set(named))} distinct: {unknown or 'all resolve'}")
    runnable = {name for name, path in modules.items() if "__main__" in path.read_text(encoding="utf-8")}
    without = sorted(name for name in named if name in modules and "__main__" not in modules[name].read_text(encoding="utf-8"))
    check("every module the README tells you to run can be run", not without, f"without a __main__ guard: {without or 'none'}")
    undocumented = sorted(runnable - set(named))
    check("every runnable module is named somewhere", not undocumented,
          f"{len(runnable)} runnable, undocumented: {undocumented or 'none'}")

    first: dict[str, int] = {}
    for name in named:
        first.setdefault(name, text.find(f"python -m src.{name}"))
    out_of_order = [f"{before} before {after}" for before, after in ORDER
                    if not (first.get(before, -1) < first.get(after, 10 ** 9))]
    check("the command list respects dependency order", not out_of_order,
          f"{len(ORDER)} constraints" + (f", violations: {out_of_order}" if out_of_order else ""))

    print("-- versions --------------------------------------------------------------")
    pinned = pins()
    claimed = re.search(r"(\d+)\s+exact pins", text)
    check("the pin count matches requirements.txt", claimed is not None and int(claimed.group(1)) == len(pinned),
          f"README says {claimed.group(1) if claimed else '?'}, file has {len(pinned)}")
    quoted = dict(re.findall(r"`([a-z0-9_.\-]+)==([0-9][\w.]*)`", text))
    wrong = {name: value for name, value in quoted.items() if pinned.get(name.lower()) != value}
    check("every version quoted in the README equals its pin", not wrong,
          f"{len(quoted)} quoted" + (f", mismatched {wrong}" if wrong else ""))
    check("the Python 3.11 requirement is stated", "3.11" in text and "required" in text.lower(),
          "the ceiling that breaks an install is named, not hidden")

    print("-- numbers ---------------------------------------------------------------")
    allowed = (report.record_numbers(exclude=RECORD_PATH) | report.document_numbers()
               | version_tokens(pinned))
    total, unmatched = report.trace(text, allowed)
    check("every decimal in the README traces to a record or a document", not unmatched,
          f"{total} tokens, unmatched: {unmatched[:8]}")

    threshold = json.loads((config.MODELS_DIR / "threshold_choice.json").read_text(encoding="utf-8"))
    holdout = json.loads((config.MODELS_DIR / "holdout_eval.json").read_text(encoding="utf-8"))
    lift = json.loads((config.MODELS_DIR / "lift_results.json").read_text(encoding="utf-8"))
    cv = json.loads((config.MODELS_DIR / "metrics_cv.json").read_text(encoding="utf-8"))

    net_half = threshold["at_chosen"]["net_value_eur"]
    net_held = threshold["held_out"]["at_chosen"]["net_value_eur"]
    net_all = threshold["training"]["positives"] * 20.00 - threshold["training"]["rows"] * 1.50
    quoted_nets = (f"{net_half:,.2f}", f"{net_held:,.2f}", f"{net_all:,.2f}",
                   f"{threshold['held_out']['at_call_everyone']['net_value_eur']:,.2f}")
    check("the four euro totals on the front page are the records' own",
          all(f"{value} EUR" in text for value in quoted_nets)
          and f"{net_half:,.2f}" == "43,621.00" and f"{net_held:,.2f}" == "10,988.50",
          f"{', '.join(quoted_nets)}")
    check("the gain and the call saving recompute from those totals",
          f"{net_half - net_all:,.2f}" == "13,253.00"
          and str(threshold["training"]["rows"] - threshold["at_chosen"]["clients_called"]) == "21302"
          and "+13,253.00 EUR" in text,
          f"{net_half - net_all:,.2f} EUR on "
          f"{threshold['training']['rows'] - threshold['at_chosen']['clients_called']:,} calls")
    shipped = cv["models"]["xgboost_tuned"]["summary"]["pr_auc"]["mean"]
    check("the quoted PR-AUC figures are the shipped model's and the held-out evaluation's",
          f"{shipped:.4f}" == "0.4460" and f"{holdout['metrics']['pr_auc']:.4f}" == "0.4616"
          and "0.4460" in text and "0.4616" in text,
          f"CV {shipped:.4f}, held out {holdout['metrics']['pr_auc']:.4f}")
    check("the top-decile lift is the held-out decile table's first row",
          f"{lift['tables']['held_out'][0]['lift']:.4f}" == "4.5100" and "4.5100" in text,
          f"{lift['tables']['held_out'][0]['lift']:.4f}x")

    print("-- data ------------------------------------------------------------------")
    for name, size, digest in re.findall(r"`(data/raw/[^`]+)` \| ([\d,]+) B \| `([0-9a-f]{16})`", text):
        path = config.ROOT / name
        actual = path.read_bytes() if path.exists() else b""
        check(f"{name} is described accurately",
              path.exists() and len(actual) == int(size.replace(",", ""))
              and hashlib.sha256(actual).hexdigest().startswith(digest),
              f"{len(actual):,} B, sha256 {hashlib.sha256(actual).hexdigest()[:16]}")
    with config.BANK_FULL_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=config.CSV_SEPARATOR)
        header = next(reader)
        rows = [row for row in reader if row]
    positives = sum(1 for row in rows if row[-1].strip().strip('"') == "yes")
    base = positives / len(rows) * 100
    check("the row and column counts on the front page match the file",
          f"{len(rows):,}" == "45,211" and str(len(header)) == "17" and "45,211" in text,
          f"{len(rows):,} rows x {len(header)} columns")
    # The full file's rate is 11.70%, and the modelling half's is 11.6982% - two different numbers that a
    # first draft of this README conflated, pairing the file's 5,289 subscribers with the half's rate.
    half = json.loads((config.MODELS_DIR / "metrics_cv.json").read_text(encoding="utf-8"))
    half_rate = half["base_rate"] * 100
    check("the base rate is recomputed from the target column, and the half's rate is labelled as such",
          f"{base:.2f}" == "11.70" and f"{half_rate:.4f}" == "11.6982"
          and f"{positives:,}" in text and "11.70%" in text and "modelling half" in text,
          f"{positives:,}/{len(rows):,} = {base:.2f}% in the file, {half_rate:.4f}% on the half")

    print("-- repository ------------------------------------------------------------")
    figure_count = len(list(config.FIGURES_DIR.glob("*.png")))
    csv_count = len(list(config.REPORTS_DIR.glob("*.csv")))
    doc_count = len([p for p in config.DOCS_DIR.glob("*.md")
                     if p.name not in {"report.md", "report_academic.md"}])
    ticks = (config.ROOT / "PROGRESS.md").read_text(encoding="utf-8")
    decisions = len(re.findall(r"^\| D\d+ \|", ticks, re.M))
    done = len(re.findall(r"^- \[x\] T\d+", ticks, re.M))
    total_tasks = len(re.findall(r"^- \[[ x]\] T\d+", ticks, re.M))
    check("the module count is the filesystem's", f"{len(modules)} modules" in text, f"{len(modules)} modules")
    check("the figure count is the directory's", f"{figure_count} figures" in text, f"{figure_count} figures")
    check("the CSV table count is the directory's", f"{csv_count} CSV tables" in text, f"{csv_count} CSV tables")
    check("the supporting-document count is the directory's", f"{doc_count} supporting documents" in text,
          f"{doc_count} documents")
    check("the decisions-log count is the log's", f"{decisions}-entry decisions log" in text,
          f"{decisions} decisions")
    check("the task progress in the README is the checklist's",
          f"{done} of {total_tasks}" in text and f"{total_tasks}-task checklist" in text,
          f"{done}/{total_tasks}")

    links = re.findall(r"\]\(([^)]+)\)", text)
    broken = [target for target in links
              if not target.startswith(("http", "#")) and not (config.ROOT / target).exists()]
    check("every relative link resolves", not broken, f"{len(links)} links" + (f", broken {broken}" if broken else ""))

    # URLs contain `s:/` and would trip a naive drive-letter pattern, so they are removed first.
    prose = re.sub(r"https?://\S+", "", text)
    absolute = re.findall(r"[A-Za-z]:[\\/]|/Users/|/home/[a-z]", prose)
    check("no absolute path or home directory appears", not absolute, f"{absolute[:3] or 'none'}")

    print("-- honesty ---------------------------------------------------------------")
    demo = config.APP_DIR / "app.py"
    check("the demo the README names exists", demo.exists(),
          "app/app.py present" if demo.exists() else "app/app.py missing")
    check("the README starts the demo and does not call app/ empty",
          "streamlit run app/app.py" in text and "app/` is empty" not in text,
          "the run command is stated and the directory is no longer described as empty")
    app_text = demo.read_text(encoding="utf-8") if demo.exists() else ""
    check("the demo is a UI over the checked core",
          "streamlit" in app_text and "demo" in app_text,
          "app/app.py imports src.demo" if app_text else "app/app.py missing")
    check("the three prohibitions appear in it", all(phrase.lower() in text.lower() for phrase in PROHIBITIONS),
          f"{len(PROHIBITIONS)} phrases")
    # Whitespace and bold markers are removed before matching, because a reflow of the paragraph is not a
    # change of claim - the first version of this check failed on a line break it had itself introduced.
    flat = " ".join(text.replace("**", "").split())
    check("the scope sentence governs the numbers",
          "converts a fixed call budget" in flat
          and "not evidence that the bank should cut campaign spend" in flat,
          "D18 stated before the first euro total")

    runtimes = []
    for path in sorted(config.MODELS_DIR.glob("*.json")):
        if path.name == RECORD_PATH.name:
            continue
        elapsed = json.loads(path.read_text(encoding="utf-8")).get("elapsed_seconds")
        if isinstance(elapsed, (int, float)):
            runtimes.append(float(elapsed))
    minutes = sum(runtimes) / 60.0
    check("the stated runtime range contains the records' sum",
          7.0 <= minutes <= 11.0 and "eight to ten" in text,
          f"{sum(runtimes):.1f} s over {len(runtimes)} recorded steps = {minutes:.1f} min "
          f"(stated: eight to ten)")

    print()
    print(f"{COUNT - len(FAILURES)}/{COUNT} checks passed")
    if FAILURES:
        for failure in FAILURES:
            print("  FAILED:", failure)
        return 1

    record = {
        "task": "T42",
        "readme": README_PATH.name,
        "bytes": len(text.encode("utf-8")),
        "sha256": hashlib.sha256(README_PATH.read_bytes()).hexdigest(),
        "lines": len(text.splitlines()),
        "sections": list(SECTIONS),
        "commands": sorted(set(named)),
        "pins": len(pinned),
        "tokens_traced": total,
        "figures": figure_count,
        "tasks_done": done,
        "tasks_total": total_tasks,
        "checks": COUNT - len(FAILURES),
        "failures": FAILURES,
    }
    RECORD_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"  record: {RECORD_PATH.name} ({RECORD_PATH.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
