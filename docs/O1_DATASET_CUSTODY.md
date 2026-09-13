# O1 oracle-supervision dataset — custody and interpretation

Accepted as a **train-only oracle-supervision sidecar**. Built by
`scripts/build_oracle_dataset.py` at commit `d99842a`; manifest tracked at
`results/v4_2_oracle_dataset_full/manifest.json`.

## The two payload files are not in Git

`.gitignore:64` (`results/**`) excludes them. Only the manifest is tracked, by
`git add -f`. **A clone of this repository does not reconstruct the dataset.**
It exists on DESKTOP-A1EGVDN and nowhere else.

| file | bytes | sha256 |
|---|---:|---|
| `results/v4_2_oracle_dataset_full/candidates.json` | 95,226,370 | `b4a92fd7fa5c5d081204f41be348886501830c309bfe9bde869ff538fc152175` |
| `results/v4_2_oracle_dataset_full/positives.json` | 3,110,844 | `b2e79bf95bc474dc73682f01b920aa842310a64a3c4a6776855455836dbfd0e5` |

Verify before any use:

```bash
sha256sum results/v4_2_oracle_dataset_full/candidates.json \
          results/v4_2_oracle_dataset_full/positives.json
```

Regenerating them requires the two upstream artifacts, also ignored and also
only on this machine: the rescored derived artifact
`dae750f1d0061f4dba0f8b8e727285c597a81598c1b88b6ab676766cebf32631` and its
parent `142326ce9bd4b1f7752b204b7dcc852ef4716494d1d812d5d5ce19d3030d3181`.
Losing either makes the build unreproducible, because regeneration needs GPU
sampling at temperature 0.7 and would not return the same candidates.

## What it is, and what it is not

**It is** training data mined from train-split generations only. Every record
id in it is a subset of the hash-verified train shard; the canonical
`records.json` was never opened and the sealed final test was never touched.

**It is not a generalization result and not a model improvement.** No
validation number follows from it. The repaired successor Kill@8 of 0.632887
remains a **train-only whole-output diagnostic**, protocol-noncomparable with
the legacy `first_assertion` figure of 0.629133.

## Four standing limits on what may be claimed

1. **The 0.70 MBPP cap stays.** 26,166 candidates are usable as positives and
   1,316 were selected. That number is an arithmetic ceiling, not a balance
   achievement: non-MBPP supply after dedup and the lineage cap is 395 rows, so
   395/0.30 = 1,316.7. The cap must not be relaxed to grow the dataset.

2. **1,316 unique positives is an auxiliary component, not an SFT corpus.** Use
   it as a bounded replay/auxiliary mixture, never as the whole training set.

3. **No real-repository performance claim.** Only **12** selected positives are
   real-repository rows (0.91%). The other 99.09% are synthetic mutants. Any
   repository-level claim needs the expansion in
   [REPOSITORY_DATA_EXPANSION_PLAN.md](REPOSITORY_DATA_EXPANSION_PLAN.md).

4. **No whole-output benefit claim from this artifact.** Assertion counts
   across all 44,760 candidates are `{1: 43576, 2: 1, 3: 1, 4: 1, 5: 2}`. The
   base model writes one assertion, so `whole_output_preserved: true` is true
   and nearly vacuous here. The successor protocol could only demonstrate value
   on an SFT model's generations.

## Concentration

The 5,595 train records collapse to **670 distinct function lineages** (~8.4
mutants per underlying function); 658 carry a positive, 509 are represented in
the selection and 189 of those sit at the 4-per-lineage cap. Row counts in this
corpus overstate functional diversity by roughly 8x.
