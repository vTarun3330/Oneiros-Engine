# SANER 2027 — RENE Track submission checklist

**Retrieved from the official CFP on 2026-09-18. Re-check before submitting;
conference pages change.**

## Venue facts

| Item | Value |
|---|---|
| Official CFP URL | https://conf.researchr.org/track/saner-2027/saner-2027-reproducibility-studies-and-negative-results-rene-track |
| Track | Reproducibility Studies and Negative Results (RENE) |
| Abstract deadline | **Mon 19 Oct 2026**, AoE (UTC−12) |
| Paper deadline | **Fri 23 Oct 2026**, AoE (UTC−12) |
| Notification | Tue 8 Dec 2026 |
| Camera ready | Fri 8 Jan 2027 |
| Author registration deadline | Fri 8 Jan 2027 |
| Page limit | **10 pages** including figures and appendices, **plus up to 2 pages containing ONLY references**. (A shorter 5-page category exists for appendices/previous work; this submission uses the full 10-page category.) |
| Formatting | The RENE page says "The publication format should follow the SANER guidelines." The **SANER 2027 Research Track** states those guidelines explicitly: "LaTeX users must use `\documentclass[10pt,conference]{IEEEtran}` **without including the `compsoc` or `compsocconf` option**". This draft complies. |
| Anonymity | **Double-anonymous.** RENE: "author names and affiliations must be omitted". SANER's main guidance adds: references to one's own work in the third person; anonymize identifiable project names; avoid revealing institutional affiliations; "**Avoid linking directly to code repositories or tool deployments which can reveal your identity**". |
| Artifact expectation | Strong. "Availability of instruments and complementary artifacts" is an explicit RENE evaluation criterion; papers should "provide the links to all the artifacts in the submission", with an exception only for proprietary datasets that cannot be released. |
| **Data Availability section** | SANER's main guidance: authors should include a "**Data Availability**" section **after the Conclusions** explaining how artifacts are disclosed, or why disclosure is not feasible. Encouraged rather than strictly mandatory — but for a RENE submission, where artifact availability is a scoring criterion, treat it as required. **This draft includes one.** |
| Figures/tables formally required? | **No.** No figure or table mandate is stated. |
| Screencast required? | No. |
| Registration/presentation | "At least one author of each accepted paper must register (**full registration, not student registration**) and attend the conference in order to present their paper." |
| Generative-AI policy | **Applies via SANER's main guidance.** Submissions "must follow the latest '**IEEE Submission and Peer Review Policy**' and '**ACM Policy on Authorship**' (with associated FAQ, which includes a policy regarding the use of generative AI tools and technologies, such as ChatGPT)". Read both current policies and comply. |

## Planned tables, figures and page budget

| Item | Type | Est. share of 10 pages |
|---|---|---|
| Table I — main results | `booktabs` | ~0.45 |
| Table II — candidate validity | `booktabs` | ~0.35 |
| Table III — claim status | `booktabs` | ~0.40 |
| Figure 1 — split roles | TikZ, single column | ~0.30 |
| Figure 2 — incident → safeguard | TikZ, single column | ~0.30 |

Visual budget ≈ **1.8 of 10 pages**. Bibliography goes in the 2-page reference
allowance, not the body. **Cut if over:** Table III first (its content survives
as prose), then Figure 2.

## Pre-submission checks

- [ ] **Anonymity.** No author name, affiliation, acknowledgment, funder,
      institution, ORCID, email, repository host, account name, or URL that
      resolves to an identifiable account anywhere in `main.tex`, the bib, or
      the artifact. Self-citations in third person. **Identifiable project names
      anonymized**, and **no direct link to a code repository or tool
      deployment** — SANER names this explicitly.
- [ ] **Formatting exactly as SANER specifies** —
      `\documentclass[10pt,conference]{IEEEtran}`, **without** `compsoc` or
      `compsocconf`. Verify the compiled output, not just the source line.
- [ ] **Generative-AI policy.** Read the current **IEEE Submission and Peer
      Review Policy** and the **ACM Policy on Authorship** (including its FAQ,
      which covers generative-AI tools), and comply with whatever disclosure or
      authorship conditions they impose at submission time. SANER's main
      guidance requires both.
- [ ] **Data Availability section present, after the Conclusions**, giving an
      anonymized artifact link or a justified statement of why disclosure is not
      feasible. For RENE this is effectively required: artifact availability is
      a scoring criterion on this track.
- [ ] **PDF metadata scrubbed** — `pdfinfo` shows no author/producer identity.
- [ ] **Artifact anonymized** — uploaded to an anonymous-capable archive
      (e.g. Zenodo restricted, Anonymous GitHub). Not a personal repository.
- [ ] **Page count** — body ≤ 10 pages, references ≤ 2 additional pages, nothing
      but references on those pages.
- [ ] **No padding or margin/spacing manipulation** to fit. If it is over,
      cut content.
- [ ] **Bibliography verified** — every entry confirmed against a DOI,
      publisher page, or official proceedings page. Entries marked
      `% VERIFY-DOI` in `references.bib` must have their DOI added or their
      publisher page confirmed.
- [ ] **No citation to unpublished assistant transcripts or internal chat logs.**
- [ ] **Evidence check** — every number matches `../common/evidence_ledger.md`.
- [ ] **Prohibited-claim grep** — none of: "beats Atheris", "SFT generalizes",
      "better than the base model", any final-test metric, any repository-level
      performance figure, any value from the consumed split, or the rehearsal
      described as a model-quality result.
- [ ] **Non-significance wording** — the locked result is never described as
      "no effect" or "no difference".
- [ ] **TODO sweep** — no `TODO` marker survives into the submitted PDF.
- [ ] **Compiles cleanly** with `pdflatex → bibtex → pdflatex ×2`, no unresolved
      references, no overfull boxes past the margin.
- [ ] **Registration plan confirmed** — a full (non-student) registration is
      available for an author if accepted.

## Concurrent submission warning

> **This manuscript must not be submitted to SANER RENE, ICST, SANER SP&P or
> CAIN at the same time.** All four drafts in `papers/` describe the same study
> and overlap substantially. Simultaneous submission of the same or
> substantially similar work to more than one venue violates IEEE and ACM
> policy and the submission policies of every venue listed here, and is grounds
> for desk rejection at all of them.
>
> **Choose one venue, submit there, and wait for the decision before submitting
> a derived version elsewhere.** If a later version is submitted after a
> rejection, ensure it is substantially revised and disclose any prior
> submission where the venue requires it.
