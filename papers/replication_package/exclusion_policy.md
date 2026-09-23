# Exclusion policy

What this package deliberately does **not** contain, and why. Each exclusion is
a decision, not an oversight.

## 1. Raw model outputs

The rehearsal retained **4,336** raw completions, and the development and locked
runs retained more. None is included.

**Why.** Volume, and redistribution of generated content at scale. Retention
itself is a feature of the pipeline — it is what makes a disputed score
re-examinable — but publishing the corpus of completions is a different act.

**What is published instead.** Every retained output has a SHA-256 recorded by
the pipeline, and the execution receipt publishes the aggregate integrity result:
4,336 present, 4,336 digests, 0 missing, 0 mismatched, and 0 mismatches on an
independent recomputation. A reader holding a retained output can verify it
against its digest without the package carrying the content.

## 2. Raw prompts

Not included. Prompt *construction* is fully specified by the frozen protocol in
the receipts — variants, budgets, schema version and the digests of the prompt
sources — so a prompt is reconstructible from the receipt without shipping
rendered text.

## 3. Candidate code

Not included, for the same reasons as (1).

## 4. Material from the consumed final split

**None exists and none is included.** The authorized attempt on that split
generated zero candidates and produced zero reportable metrics. Nothing was
rendered from its records, nothing generated, nothing scored.

Neither the split's size, its membership, its record counts, nor any performance
score appears anywhere in this package or in any draft. The only facts stated
about it are that the attempt happened and that it produced nothing.

## 5. Identity-bearing links

No repository host, account name, remote URL, personal page, institutional
address, funder, or acknowledgment. The source snapshot is recorded as a bare
commit identifier.

**On hosting.** When this package is archived for review it must go to an
anonymous-capable service. A personal repository link would break
double-anonymous review at every one of the four target venues, and ICST states
explicitly that the replication package itself must be anonymized. SANER's
guidance likewise says to avoid linking directly to code repositories that can
reveal identity.

## 6. Project source code

The package carries receipts, data and the scripts needed to regenerate tables
and figures — not the evaluation pipeline itself. The pipeline is identified by
per-file digests inside the receipts, which is what the papers' claims actually
depend on.

**If a venue requires the full source**, release it from the recorded commit,
anonymized, as a separate artifact. That decision is not made here.

## 7. What this means for a reviewer

A reviewer can verify every number in the papers, re-derive every computed value,
and confirm that the decision rule predates the measurement it governs — all from
this package. What a reviewer cannot do from this package alone is re-run the
generation, because that needs the model and the corpus. That limit is stated in
each paper's threats-to-validity section rather than left implicit.
