"""Oneiros single-screen panel demonstration.

One command, about two seconds. Shows a defective program, the test that
catches it, both executions, the verdict, and the measured rate across the
full validation set.

Written for a mixed audience: no ML jargon in the output.

Usage:
    "%PY%" scripts\\demo_panel.py
    "%PY%" scripts\\demo_panel.py curated::bugsinpy_black_1
"""
import difflib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "data" / "corpus" / "v3_final_candidate"
HEADLINE = ROOT / "results" / "v3_full_sft_monitored_20260819_1" / "sft_validation_results.json"

DEFAULT_RECORD = "mutation::humaneval_HumanEval_121_mut_01039"
KNOWN = (
    "mutation::humaneval_HumanEval_121_mut_01039",
    "curated::bugsinpy_black_1",
    "curated::bugsinpy_pandas_1",
)

W = 78
RULE = "=" * W
THIN = "-" * W


def load_safe_exec():
    """Load the executor without importing baseline/__init__ (which needs torch)."""
    spec = importlib.util.spec_from_file_location(
        "oneiros_benchmark_runner", ROOT / "baseline" / "benchmark_runner.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.safe_exec


def load_record(record_id):
    with open(CORPUS / "records.json", encoding="utf-8") as handle:
        for item in json.load(handle):
            if item["id"] == record_id:
                return item
    return None


def heading(text):
    print()
    print("  " + text)
    print("  " + "-" * (W - 4))


def strip_docstring(code):
    """Drop the function docstring from the displayed code.

    The specification is already printed in section 1, and a long HumanEval
    docstring pushes the actual defect off the readable part of the screen.
    """
    lines = code.rstrip().splitlines()
    out, in_doc, dropped = [], False, False
    for line in lines:
        stripped = line.strip()
        if in_doc:
            if stripped.endswith('"""') or stripped.endswith("'''"):
                in_doc = False
            continue
        if stripped.startswith('"""') or stripped.startswith("'''"):
            dropped = True
            # single-line docstring?
            if not (len(stripped) > 3 and (stripped.endswith('"""') or stripped.endswith("'''"))):
                in_doc = True
            continue
        out.append(line)
    if dropped:
        indent = "    "
        for line in out:
            if line.strip() and not line.strip().startswith("def "):
                indent = line[: len(line) - len(line.lstrip())]
                break
        out.insert(1, indent + "# (description shown above)")
    return out


def show_diff(reference, defective):
    """Print the defective code, marking the lines that differ from the reference."""
    ref_lines = strip_docstring(reference)
    def_lines = strip_docstring(defective)
    matcher = difflib.SequenceMatcher(None, ref_lines, def_lines)
    changed, replaced = set(), []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "insert"):
            changed.update(range(j1, j2))
        if tag in ("replace", "delete"):
            replaced.append((tag, ref_lines[i1:i2]))
    for index, line in enumerate(def_lines):
        marker = ">>  " if index in changed else "    "
        for piece in _split_code(line, W - 4):
            print("%s%s" % (marker, piece))
            marker = "    "
    was_replaced = any(tag == "replace" for tag, _ in replaced)
    flat = [line for _, block in replaced for line in block]
    if flat:
        print()
        print("      the correct version instead has:"
              if was_replaced else "      the correct version also has:")
        # Dedent the block as a whole so nested structure survives; stripping
        # each line individually would flatten an if/elif into a false sequence.
        body = [line for line in flat if line.strip()]
        pad = min((len(l) - len(l.lstrip()) for l in body), default=0)
        for line in flat:
            marker = "        "
            for piece in _split_code(line[pad:].rstrip(), W - 10):
                print("%s%s" % (marker, piece))
                marker = "          "


def _split_code(line, width):
    """Soft-wrap an over-long code line so it never wraps unpredictably on screen."""
    if len(line) <= width:
        return [line]
    pieces, current = [], line
    while len(current) > width:
        cut = current.rfind(" ", 0, width)
        if cut < width // 2:
            cut = width
        pieces.append(current[:cut])
        current = current[cut:].lstrip()
    pieces.append(current)
    return pieces


def main():
    record_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RECORD
    safe_exec = load_safe_exec()
    record = load_record(record_id)
    if record is None:
        print("No corpus record with id %r." % record_id)
        print("Records you can use:")
        for known in KNOWN:
            print("   ", known)
        return 1

    test = record["tests"][0]["code"]
    spec = " ".join(str(record["specification"]).split())
    if len(spec) > 220:
        spec = spec[:217] + "..."

    print()
    print(RULE)
    print("  ONEIROS   automated generation of bug-revealing tests")
    print(RULE)

    heading("1. THE PROGRAM WE ARE TESTING")
    print("     function:  %s()" % record["entry_point"])
    print()
    print("     what it is supposed to do:")
    for line in _wrap(spec, W - 10):
        print("       %s" % line)

    heading("2. THE VERSION WITH A DEFECT   (>> marks what is wrong)")
    print()
    show_diff(record["reference_code"], record["code_under_test"])

    heading("3. THE TEST THAT CATCHES IT")
    print()
    for line in test.strip().splitlines():
        print("       %s" % line)

    heading("4. RUN THAT SAME TEST AGAINST BOTH VERSIONS")
    reference = safe_exec(record["reference_code"], test)
    defective = safe_exec(record["code_under_test"], test)
    ref_ok, def_ok = bool(reference[0]), bool(defective[0])
    print()
    print("       on the correct version .......  %s" % ("PASS" if ref_ok else "FAIL"))
    print("       on the defective version .....  %s   %s"
          % ("PASS" if def_ok else "FAIL", (defective[2] or "").strip()))

    print()
    if ref_ok and not def_ok:
        print("       VERDICT:  DEFECT CAUGHT")
        print()
        print("       It passes on correct code and fails on the defect,")
        print("       so the test genuinely detected the bug.")
    else:
        print("       VERDICT:  NOT CAUGHT")
        print()
        print("       A test only counts when it passes on the correct version")
        print("       and fails on the defective one.")

    print()
    print(THIN)
    _footer()
    return 0


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for word in words:
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        out.append(line)
    return out


def _footer():
    with open(CORPUS / "manifest.json", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with open(HEADLINE, encoding="utf-8") as handle:
        results = json.load(handle)
    killed = results.get("function_validation_killed")
    total = results.get("function_validation_records")
    rate = float(results.get("function_kill_rate", 0)) * 100

    print("  MEASURED ACROSS THE FULL VALIDATION SET")
    print()
    print("     The trained model produced tests for %s functions it had" % total)
    print("     never seen. %s of those defects were caught:  %.2f%%" % (killed, rate))
    print()
    print("     Corpus: %s behaviourally verified records." % f"{manifest.get('training_records', 8387):,}")
    print()
    print("     This screen replays one execution-verified corpus record to show")
    print("     the decision rule. The %.2f%% above is the trained model's own" % rate)
    print("     generated output, measured offline on GPU.")
    print(RULE)
    print()


if __name__ == "__main__":
    sys.exit(main())
