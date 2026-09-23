# Reference verification

Every entry in `verified_references.bib`, with how it was verified, against what,
and on what date. **Verification date for all entries: 2026-09-18.**

Method:

- **Crossref** — the DOI was resolved through `api.crossref.org/works/<doi>` and
  the returned title, container title, year, volume, issue and pages compared
  against the entry.
- **arXiv** — the abstract page at `arxiv.org/abs/<id>` was read and title,
  authors and year compared. The `10.48550/arXiv.*` DOIs are DataCite-registered
  and therefore return HTTP 404 from Crossref; that is expected and is not a
  verification failure.
- **Proceedings / journal page** — for publishers that mint no DOI (NeurIPS,
  JMLR), the official page was read directly and its URL recorded in the entry.

**No DOI in the bibliography was guessed.** Where a publisher mints none, the
entry carries an official URL instead. No `% VERIFY-DOI` marker remains.

---

## Verified entries

| Key | Verified against | Identifier / URL | Result |
|---|---|---|---|
| `jia2011mutation` | Crossref | `10.1109/TSE.2010.62` | OK — IEEE TSE, 2011 |
| `papadakis2019mutation` | Crossref | `10.1016/bs.adcom.2018.03.015` | OK — Advances in Computers, 2019 |
| `just2014mutants` | Crossref | `10.1145/2635868.2635929` | OK — FSE 2014 |
| `papadakis2018mutationscores` | Crossref | `10.1145/3180155.3180183` | OK — ICSE 2018 |
| `just2014defects4j` | Crossref | `10.1145/2610384.2628055` | OK — ISSTA 2014 |
| `fraser2011evosuite` | Crossref | `10.1145/2025113.2025179` | OK — ESEC/FSE 2011 |
| `fraser2013wholesuite` | Crossref | `10.1109/TSE.2012.14` | OK — IEEE TSE, 2013 |
| `pacheco2007randoop` | Crossref | `10.1145/1297846.1297902` | OK — OOPSLA Companion 2007 |
| `lemieux2023codamosa` | Crossref | `10.1109/ICSE48619.2023.00085` | OK — ICSE 2023 |
| `schafer2024llmunit` | Crossref | `10.1109/TSE.2023.3334955` | OK — IEEE TSE, 2024 |
| `wang2024testingsurvey` | Crossref | `10.1109/TSE.2024.3368208` | OK — IEEE TSE, 2024 |
| `hou2024llmse` | Crossref | `10.1145/3695988` | OK — ACM TOSEM, 2024 |
| `tufano2020athenatest` | arXiv abstract page | `arxiv.org/abs/2009.05617` | OK — title and all five authors match; submitted 2020, revised 2021. **No peer-reviewed venue is stated on the arXiv page, so the entry remains `@misc`.** |
| `serebryany2016libfuzzer` | Crossref | `10.1109/SecDev.2016.043` | OK — IEEE SecDev 2016 |
| `bohme2016aflfast` | Crossref | `10.1145/2976749.2978428` | OK — CCS 2016 |
| `klees2018evaluating` | Crossref | `10.1145/3243734.3243804` | OK — CCS 2018 |
| `xia2024fuzz4all` | Crossref | `10.1145/3597503.3639121` | OK — ICSE 2024 |
| `deng2023titanfuzz` | Crossref | `10.1145/3597926.3598067` | OK — ISSTA 2023 |
| `chen2021codex` | arXiv abstract page | `arxiv.org/abs/2107.03374` | OK — title matches; Chen, Tworek, Jun, Yuan **and 51 further authors**, so `and others` is correct; 2021 |
| `austin2021mbpp` | arXiv abstract page | `arxiv.org/abs/2108.07732` | OK — title matches; Austin, Odena, Nye, Bosma **and 7 further authors**; 2021 |
| `liu2023evalplus` | NeurIPS 2023 proceedings page | [proceedings.neurips.cc …43e9d647…](https://proceedings.neurips.cc/paper_files/paper/2023/hash/43e9d647ccd3e4b7b5baab53f0368686-Abstract-Conference.html) | OK **after correction** — see C1 below. NeurIPS mints no DOI; official URL recorded in the entry. |
| `yang2023rephrased` | arXiv abstract page | `arxiv.org/abs/2311.04850` | OK — title and all five authors match; 2023; no peer-reviewed venue stated, so `@misc` is correct |
| `kapoor2023leakage` | Crossref | `10.1016/j.patter.2023.100804` | OK — Patterns, 2023 |
| `gundersen2018reproducibility` | Crossref | `10.1609/aaai.v32i1.11503` | OK — AAAI 2018 |
| `hutson2018reproducibility` | Crossref | `10.1126/science.359.6377.725` | OK — Science, 2018 |
| `pineau2021reproducibility` | JMLR article page | [jmlr.org/papers/v22/20-303.html](https://jmlr.org/papers/v22/20-303.html) | OK — JMLR 22(164), 1–20, 2021; all eight authors match. JMLR mints no DOI for this article; official URL recorded. |
| `amershi2019seforml` | Crossref | `10.1109/ICSE-SEIP.2019.00042` | OK — ICSE-SEIP 2019 |
| `sculley2015debt` | NeurIPS 2015 proceedings page | [papers.nips.cc …86df7dcf…](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) | OK **after correction** — see C3 below. NeurIPS mints no DOI; official URL recorded. |
| `sallou2024threats` | Crossref | `10.1145/3639476.3639764` | OK **after correction** — see C2 below |
| `mcnemar1947note` | Crossref | `10.1007/BF02295996` | OK — Psychometrika 12(2), 153–157, 1947 |
| `wilson1927probable` | Crossref | `10.1080/01621459.1927.10502953` | OK — JASA 22(158), 209–212, 1927 |
| `dietterich1998tests` | Crossref | `10.1162/089976698300017197` | OK — Neural Computation 10(7), 1998 |
| `arcuri2014hitchhiker` | Crossref | `10.1002/stvr.1486` | OK — STVR 24(3), 219–250. Online 2012, **print issue 2014**; the entry's `year = {2014}` matches the print issue, which is the citable one |
| `amrhein2019retire` | Crossref | `10.1038/d41586-019-00857-9` | OK — Nature 567(7748), 305–307, 2019 |
| `wasserstein2016asa` | Crossref | `10.1080/00031305.2016.1154108` | OK — The American Statistician 70(2), 129–133, 2016. An automated title comparison flagged a mismatch; manual inspection showed it was caused by the `{ASA}` brace protection in the entry, not by a metadata difference. |

**35 entries. 35 verified. 0 removed.**

---

## Corrections applied during verification

### C1 — `liu2023evalplus`: wrong title

The entry read *"…Rigorous Evaluation of Large Language Models for Code
**Synthesis**"*. The published NeurIPS 2023 title is *"…for Code
**Generation**"*. Corrected.

This is the kind of error that survives a plausibility read: the wrong word is
a synonym in context, and the paper is widely known by its tool name. It was
caught only because the proceedings page was opened.

### C2 — `sallou2024threats`: proceedings title

The entry read *"Proceedings of the 46th International Conference on Software
Engineering: New Ideas and Emerging Results"*. Crossref and the ACM Digital
Library both record the proceedings as *"Proceedings of the 2024 ACM/IEEE
**44th** International Conference on Software Engineering: New Ideas and
Emerging Results"*.

ICSE 2024 was the 46th ICSE, so the publisher's string is arguably wrong — but
the bibliography must match the record as published, not as it should have been.
Corrected to the publisher's string.

### C3 — `sculley2015debt`: unverified page range removed

The entry claimed `pages = {2503--2511}`. The NeurIPS 2015 proceedings page does
not publish a page range for this paper. Rather than assert a number that could
not be verified, the field was **removed** and the official proceedings URL added.

---

## Standing requirement before submission

Re-run this verification at camera-ready time. Publisher metadata changes, DOIs
are occasionally reassigned, and arXiv preprints acquire peer-reviewed venues —
`tufano2020athenatest` and `yang2023rephrased` are currently cited as
preprints and should be re-checked for a published version, since citing the
published form is preferable where one exists.

The verification script used for the Crossref pass is not committed; the checks
are reproducible from the table above by resolving each identifier.
