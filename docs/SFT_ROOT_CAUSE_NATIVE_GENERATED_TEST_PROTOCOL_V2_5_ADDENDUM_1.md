# Protocol v2.5 — operational and statistical addendum 1 (additive)

The frozen v2.5 document (`c707d8ae…`) is unchanged. This addendum was frozen on 2026-10-01,
before any v2.5 GPU output, training run or confirmation outcome existed. Where the two
documents conflict on an operational or statistical point, this addendum governs.

## 0. Erratum

In v2.5 section F, the cross-reference "untouched confirmation panel (section F)" should read
**section D**. Section D is the confirmation-panel specification; section F is promotion.

## 1. Endpoints (confirmation panel only)

| Role | Endpoint |
|---|---|
| **Primary** | Target-level **Kill@8**: the fraction of the 3 seeds in which any of 8 candidates is a confirmed semantic kill, averaged over seeds and then over targets |
| Key secondary | Kill@1 |
| Key secondary | Fixed-validity rate (fixed revision passes and target reached), averaged at target level |
| Validity secondaries | Parse rate; collection rate; execution rate; reach rate |
| Reported | Crash kills, separately from semantic kills |

- **Unit:** the **target**, never the candidate. Candidates are never pooled across targets.
- **Normalisation:** both the raw and the R4-normalised (repair-transform v1) outcomes are
  reported. **The primary endpoint uses R4-normalised outcomes, identically for every arm.**

## 2. Inference

1. **Comparisons:** C − A (primary) and C − B (key secondary), paired by target.
2. **Repository-cluster-aware test:** an exact or Monte-Carlo sign-flip test on
   repository-mean paired differences (10,000 flips, seed 20261001).
3. **Intervals:** 95% intervals by repository-cluster bootstrap (10,000 resamples, seed
   20261001). Leave-one-repository-out estimates are reported.
4. **Multiplicity:** fixed-sequence testing.
   1. The primary C − A Kill@8 superiority test is at α = 0.05, two-sided.
   2. Only if it rejects, the fixed-validity non-inferiority test follows, then C − B Kill@8.
   3. Every other result is descriptive.
5. **Non-inferiority margins** (C relative to A, the lower bound of the 95% cluster
   interval):

   | Measure | Margin |
   |---|---|
   | Fixed validity | −5 points |
   | Execution rate | −5 points |
   | Diversity (unique canonical tests per target) | −10% relative |
   | Repository generalisation (leave-one-repository-out primary effect) | sign must stay ≥ 0 for at least 80% of repositories |

## 3. Confirmation-panel size (power analysis)

**Minimum panel:**
- **80 targets** from **at least 10 repositories**;
- **at most 8 targets per repository**;
- every complexity stratum present;
- at least 4 defect families.

**Power basis:**
- **Simulation:** repository-level logit heterogeneity (SD 1.1), 3 seeds per target, the
  sign-flip test above, α = 0.05.
- **At a base Kill@8 ≤ 5%:** ≥ 0.83 power to detect **+10 points**, and ≥ 0.95 to detect +15.
- **At a base rate of 10%:** power for +10 points falls to about 0.7. That is a documented
  limitation, not a reason to relax the gate.

**If fewer than 80 targets or 10 repositories qualify,** the panel is still frozen as built.
The study is then labelled *underpowered*, and a non-significant result is reported as
inconclusive, never as evidence of no effect.

## 4. Missing and infrastructure rows

- **Canary-proven infrastructure failures** are excluded symmetrically across all arms at
  target level (the v2.4 rule). They are never model failures and never Atheris non-kills.
- **Any other missing row** fails the run closed.
- **Excluded targets** are listed, together with their repositories.

## 5. Corpus rules

| Rule | Value |
|---|---|
| Sources | Only verified `pytest_module_v1` positives (fixed passes, buggy fails, rerun agrees, target reached, policy passes); never pending, failed or guessed rows |
| Unique-first selection | One row per canonical test and lineage first |
| Per function | ≤ 3 canonical tests |
| Per repository | ≤ 40 |
| Per bug lineage | ≤ 3 |
| Per bug family | ≤ 35% of the corpus |
| Per source dataset group | ≤ 60% (MBPP + HumanEval + curated mutations form **one** synthetic group) |
| Per canonical test | 1 |
| Repetition | None to reach a percentage |
| **Minimum unique verified repository yield** before training | ≥ 150 canonical repository tests from ≥ 8 repositories and ≥ 60 bug lineages; otherwise **the training gate fails** |
| Replay | Verified synthetic modules (an auxiliary group) at a ratio of **≤ 1 synthetic per repository example**, and no more than the synthetic cap |

## 6. Training exposure and held-constant factors

- **Exposure:** arm C is matched to the baseline adapter by **optimizer updates**: 431
  updates at 16 examples per update. When C's unique corpus is smaller, the remaining updates
  come from additional **epochs over the same unique set**, never from duplicated rows inside
  one epoch. The number of target tokens per arm is reported.
- **Held constant:**
  - base model and immutable revision `2e1fd397…`, and its tokenizer revision;
  - LoRA configuration and target modules;
  - effective batch size 16, learning rate 1e-5, the optimizer, constant-with-warmup scheduling
    with 25 warmup steps;
  - 3,072 maximum sequence, 1,024 prompt and 1,024 completion tokens;
  - seed 42, and the decoding and evaluation protocol.
- **Interpretation:** arm C changes a **package** of supervision properties: output-contract
  alignment, verifier filtering, deduplication, mixture and replay. Results are reported as a
  package-level effect. No single component is credited without a separately frozen ablation.
- **Checkpoint:** the final checkpoint (update 431) is used. The confirmation panel is never
  used for checkpoint selection. If a monitor checkpoint is used, it must be chosen on a frozen
  **train-split** monitor only.

## 7. Pilot (train-only rejection sampling)

| Item | Rule |
|---|---|
| Sample | 30 train targets, structurally selected before outcomes; 8 candidates each; the base model only; `pytest_module_v1` prompt with R4 normalisation |
| **Success** | ≥ 10% of targets yield at least one verified unique positive, **and** the projected full run reaches the §5 minimum repository yield within the §8 budget |
| **Futility** | < 5% of targets yield a positive. Stop rejection sampling; train only on converted positives, or report the training gate as failed |
| In between | One predeclared extension of 30 more targets, then the same rules |

## 8. GPU budget

| Run | Ceiling |
|---|---|
| Pilot | ≤ 1 GPU-hour |
| Full train-split rejection sampling | ≤ 10 GPU-hours |
| One SFT run | ≤ 3 GPU-hours |
| Confirmation generation, three arms | ≤ 8 GPU-hours |

Exceeding any ceiling stops the run and requires a new authorisation.
