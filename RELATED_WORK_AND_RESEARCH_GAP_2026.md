# Related Work and Defensible Oneiros Research Gap

Verified against primary publication pages on 27 August 2026.

## Closest prior work

### MuTAP

[MuTAP](https://arxiv.org/abs/2308.16557) uses mutation testing to improve
LLM-generated tests. It repairs invalid tests and feeds surviving mutants back
into later prompts. It is therefore direct prior work for any broad claim that
Oneiros is the first system to combine LLM test generation and mutation
testing.

Oneiros must not make that broad claim. The defensible distinction to test is
that Oneiros uses an offline, execution-verified corpus to adapt a compact local
model with SFT, optionally followed by gated DPO, and evaluates whether the
adapted policy generalises to group-disjoint unseen defects without exposing
the fixed implementation or mutation diff at ordinary inference time.

### Mokav

[Mokav](https://arxiv.org/abs/2406.10375) iteratively supplies execution
feedback while searching for difference-exposing inputs. It is the primary
comparison for the Oneiros feedback ablation.

Mokav receives two program versions and execution information derived from
their comparison. The implemented Oneiros feedback profile is deliberately
stricter: the next prompt receives only bounded observations from executing
earlier assertions on the visible code under test. The hidden reference remains
oracle-only. The experiment must determine whether this reference-free
feedback improves Kill@k under the same eight-candidate budget; no superiority
is assumed before the experiment.

### SWE-Mutation

[SWE-Mutation](https://aclanthology.org/2026.findings-acl.1976/) evaluates
LLM-generated test suites with more realistic, agentically produced mutants and
shows that conventional mutations can make test effectiveness look stronger
than it is. This motivates keeping Oneiros function-level rule-based mutation
results separate from native real-repository results.

Oneiros has verified BugsInPy and SWE-bench-derived records and a locked
external task index, but it cannot claim real-project generalisation until
generated tests run in the original isolated project environments.

### CDBench

[CDBench](https://link.springer.com/article/10.1007/s10664-026-10901-8) uses a
Code Defenders zero-sum game: LLM attackers create mutants and LLM defenders
create targeted tests. Its dynamic setup addresses static-benchmark saturation
and contamination, and its evaluation explicitly distinguishes invalid and
equivalent mutants.

Oneiros currently evaluates a fixed, immutable corpus rather than a competitive
attacker/defender game. Its result should therefore be framed as reproducible
offline policy adaptation, not as a contamination-resistant dynamic benchmark.
Ordered failure accounting and the corpus-quality report address part of the
same validity problem but do not reproduce CDBench.

### LLM-generated mutants and real bugs

The 2026 [comprehensive mutation-testing
study](https://discovery.ucl.ac.uk/id/eprint/10223970/) reports that LLM-generated
mutants can be closer to real bugs while also increasing non-compiling,
duplicate, and equivalent-mutant problems. Oneiros consequently reports
behavioral-witness admission, semantic duplicate removal, context exclusions,
and the limitation that witness-based non-equivalence is only established for
accepted pairs.

[LLM vs. Human Unit Tests: Fault Detection on Real Python
Bugs](https://arxiv.org/abs/2606.08588) evaluates historical BugsInPy faults and
also finds that coverage alone is not an adequate proxy for fault detection.
This supports Oneiros reporting kill behavior, validity, diversity, and native
real-bug results separately instead of treating coverage as its main outcome.

## Research question

The central question is:

> Can execution-verified SFT, with optional gated preference optimisation,
> improve a compact model's ability to generate reference-valid,
> mutation-killing tests for group-disjoint unseen defects without revealing a
> fixed implementation or mutation diff at inference time?

Secondary questions are:

1. Does SFT outperform the pinned base model on the exact same functions,
   seeds, runtime, and candidate budget?
2. Does reference-free iterative execution feedback improve Kill@k without
   reducing reference validity?
3. Does diversity prioritisation improve early Kill@k or only reorder redundant
   candidates?
4. Does DPO add unique function coverage, or mainly add redundant killing
   candidates while regressing other functions?
5. How much performance survives when an entire mutation family is excluded
   from training and checkpoint selection?
6. Do improvements transfer from function-level mutants to untouched native
   real-repository bugs?

## Minimum controlled table

Every row must use the same locked validation scope and eight raw candidate
slots unless explicitly labelled otherwise.

| Policy | Feedback | Diversity | Seeds | Kill@1 | Kill@2 | Kill@4 | Kill@8 | Pass@8 | Invalid rate | Real bugs |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Pinned base Phi-3 | 0 | none | 5 | pending | pending | pending | pending | pending | pending | pending |
| SFT | 0 | none | 5 | pending | pending | pending | pending | pending | pending | pending |
| SFT | 1 | none | 5 | pending | pending | pending | pending | pending | pending | pending |
| SFT | 2 | none | 5 | pending | pending | pending | pending | pending | pending | pending |
| SFT | 0 | AST | 5 | pending | pending | pending | pending | pending | pending | pending |
| SFT + gated DPO | 0 | none | 5 | blocked until SFT gate | | | | | | |

The table must not be populated with the earlier 67/100 smoke result or the
partial seed-44 prefix. Results are populated only from complete compatible
research-schema JSON files.
