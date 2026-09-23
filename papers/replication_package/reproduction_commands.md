# Reproduction commands

How to regenerate every table and figure in the four drafts from the receipts
alone. **Nothing here runs a model, touches a GPU, or reads any split.**

## 0. Verify the package

These commands apply to the **assembled archive**, after the receipts listed in
`MANIFEST.md` have been copied beside this file and `SHA256SUMS` has been
generated.  This source-tree directory is the archive plan, so it deliberately
does not pretend that a not-yet-assembled checksum file exists.

    python verify_provenance.py

Expect `verified: 13/13` and every digest matching the full value in
`provenance.json`. To emit a conventional checksum file after assembly, run
`python verify_provenance.py --write-sha256sums`.

## 1. Read the table values back from the receipts

Every number in every table is a receipt field. Re-read them with:

    # Development arms  -> Table I (development block), Table II (development block)
    python -c "import json; d=json.load(open('v4_2_development_selection_receipt.json')); print(json.dumps({k: v['metrics'] for k, v in d['arms'].items()}, indent=2))"

    # Predeclared primary comparison (B@150 vs A@150)
    python -c "import json; d=json.load(open('v4_2_development_selection_receipt.json')); print(json.dumps(d['comparisons'], indent=2))"

    # Locked arms -> Table I (locked block), Table II (locked block)
    python -c "import json; d=json.load(open('v4_2_locked_validation_result_receipt.json')); print(json.dumps({k: v['metrics'] for k, v in d['arms'].items()}, indent=2))"

    # Paired comparison, criteria and decision
    python -c "import json; d=json.load(open('v4_2_locked_validation_result_receipt.json')); print(json.dumps(d['paired_comparison'], indent=2)); print(json.dumps(d['promotion_criteria'], indent=2)); print(d['decision'])"

    # Rehearsal -> Table III
    python -c "import json; d=json.load(open('v4_2_rehearsal_execution_receipt.json')); [print(k, json.dumps(d[k])) for k in ('scope','candidates','outcomes','failure_taxonomy','execution','operational_kill_at_k','reproduction_check')]"

**The promotion rule predates the result.** Confirm it by reading the rule from
the preflight, which was committed before either locked arm ran:

    python -c "import json; d=json.load(open('v4_2_locked_validation_preflight.json')); print(json.dumps(d, indent=2)[:4000])"

## 2. Re-derive every computed value

The drafts state a small number of derived quantities — differences, one ratio,
and rounded displays. Each is listed in `evidence_ledger.md` section I with its
formula, inputs and unrounded result. Recompute them all:

    python -c "
    from decimal import Decimal as D
    pts = lambda a, b: (D(a) - D(b)) * 100
    print('D1  dev gain A@431   ', pts('0.695572', '0.605166'))
    print('D2  dev gain A@150   ', pts('0.690037', '0.605166'))
    print('D3  dev gain B@150   ', pts('0.658672', '0.605166'))
    print('D4  locked gain      ', pts('0.628798', '0.594452'))
    print('D5  shrinkage ratio  ', D('9.0406') / D('3.4346'))
    print('D6  discordant pairs ', 114 + 88)
    print('D7  candidates       ', 542 * 8)
    print('D8  fns ref-valid    ', 566 - 675)
    print('D12 ref-validity pts ', pts('0.333388', '0.452114'))
    print('D12 parse pts        ', pts('0.988771', '0.987285'))
    print('D12 execution pts    ', pts('0.942206', '0.928336'))
    "

Expected, in order: `9.040600`, `8.487100`, `5.350600`, `3.434600`,
`2.6322...`, `202`, `4336`, `-109`, `-11.872600`, `0.148600`, `1.387000`.

Every printed value must match `evidence_ledger.md` section I. If one does not,
the paper is wrong, not the ledger.

## 3. Rebuild the figure coordinates

Figures are TikZ, drawn inline in each `main.tex` from coordinates mapped out of
`data/kill_at_8.csv`. The mapping is stated in a comment above each figure:

- ICST Fig. 2: `x = (Kill@8 - 0.50) * 25` cm
- SANER SP&P Fig. 1: `x = (Kill@8 - 0.54) * 22` cm

Re-derive them:

    python -c "
    import csv
    rows = csv.DictReader(l for l in open('data/kill_at_8.csv') if not l.startswith('#'))
    for r in rows:
        v, lo, hi = float(r['kill_at_8']), float(r['wilson_lo']), float(r['wilson_hi'])
        f = lambda x: (x - 0.50) * 25
        print('%-12s %-8s pt=%.3f lo=%.3f hi=%.3f' % (r['panel'], r['arm'], f(v), f(lo), f(hi)))
    "

**Do not trust the coordinates in the `.tex` without recomputing them.** An
earlier draft of the ICST figure carried an incorrect Wilson offset for one row;
it was caught by recomputation, not by reading.

## 4. Compile the papers

    cd <venue>/
    pdflatex main
    bibtex   main
    pdflatex main
    pdflatex main

Then **count the pages in the produced PDF** and check against the limit in that
directory's `submission_checklist.md`. Do not estimate.

## 5. Run the validator

From the repository's `papers/` directory:

    python common/validate_papers.py

From inside an assembled replication archive, copy the validator as prescribed
by `MANIFEST.md` and run:

    python validate_papers.py

It checks environment and brace balance, citation resolution against the venue's
`references.bib`, anonymity strings, prohibited claims over whitespace-collapsed
text **and** over the compiled PDFs, every numeric token against the evidence
ledger, and leaked TODO markers.

The whitespace-collapsed matching is not incidental: a line-based grep once
reported a prohibited phrase absent while three instances survived, wrapped
across line breaks.
