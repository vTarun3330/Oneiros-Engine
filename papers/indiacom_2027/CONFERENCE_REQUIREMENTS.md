# INDIACom-2027 — verified submission requirements

**Date checked:** 2026-10-13.
**Sources consulted:** only the official INDIACom/BVICAM site and the IEEE Author Center.

Every requirement below is quoted or paraphrased from the official page named next
to it. **Not stated** means the official pages checked do not say. Those points must
be confirmed with the conference before submission. They have not been inferred.

## Official URLs used

| Page | URL |
|---|---|
| Conference home | http://www.bvicam.ac.in/indiacom/ |
| Call for Papers | http://www.bvicam.ac.in/indiacom/callForPapers.asp |
| Guidelines to submit paper | http://www.bvicam.ac.in/indiacom/Procedural%20Guidelines.asp |
| Registration (fees, page limit, extra pages) | http://www.bvicam.ac.in/indiacom/Registration.asp |
| Review process, publication and indexing | http://www.bvicam.ac.in/indiacom/ReviewProcess.asp |
| Downloads (template, forms) | http://www.bvicam.ac.in/indiacom/Downloads.asp |
| FAQs | http://www.bvicam.ac.in/indiacom/faqs.asp |
| Full-paper template (PDF) | http://www.bvicam.ac.in/indiacom/downloads/Template_For_Full_Paper.pdf |
| IEEE policy on AI-generated content | https://conferences.ieeeauthorcenter.ieee.org/author-ethics/guidelines-and-policies/submission-policies/ |

## Conference identity

| Item | Verified value | Source |
|---|---|---|
| Conference | 21st INDIACom; INDIACom-2027, 14th International Conference on Computing for Sustainable Global Development | Home |
| Organiser | Bharati Vidyapeeth's Institute of Computer Applications and Management (BVICAM), New Delhi | Home |
| Dates | 07–09 April 2027 | Home, Guidelines |
| IEEE Conference ID | 74162 | Home, Review Process |
| Relevant track | **Track 4: Software Engineering and Emerging Technologies**. Chair: Dr. Ritika Wason; Co-Chair: Dr. Saumya Bansal; contact `track4.indiacom@gmail.com` | Home |
| Track 4 topics listed | Contemporary Software Design and Development; Software System; Blockchain and Distributed Ledger Technologies; Artificial Intelligence and Machine Learning; IoT and Embedded Systems | Call for Papers |

## Dates

| Milestone | Date | Source |
|---|---|---|
| **Full-length paper submission** | **07 December 2026** | Call for Papers, Guidelines |
| Acceptance notification | 08 February 2027 | Guidelines |
| Final version | 22 February 2027 | Guidelines |
| Early-bird registration deadline | 22 February 2027 | Guidelines, Registration |
| PPT submission | 08 March 2027 | Guidelines |

## Length and format

| Requirement | Verified value | Source |
|---|---|---|
| Maximum length | "Maximum paper length is of 6 A4 Size pages." | Registration |
| Extra pages | Up to 2 pages beyond 6 are allowed, charged at Rs. 2,000 per extra page (Indian) or US$100 per extra page (international) | Registration |
| References count toward the limit? | **Not stated.** This draft treats the 6-page limit as including references, the conservative reading. | — |
| Paper size | A4 (Registration page). **Conflict:** the downloadable `Template_For_Full_Paper.pdf` is US-Letter (612×792 pt) and says A4 users should use a separate "MSW_A4_format" file, which is not on the Downloads page. | Registration; template |
| Layout | The template is the IEEE conference template (two columns, Times New Roman, IEEE heading styles). Its running header still reads "Proceedings of the 20th INDIACom; INDIACom-2026", so the 2027 template has apparently not been refreshed. | Template |
| Template rules quoted | "Do Not Use Symbols, Special Characters, Footnotes, or Math in Paper Title or Abstract"; "Do not add any kind of pagination anywhere in the paper"; author names listed left to right; use IEEE numbered references ("Refer simply to the reference number, as in [3]") | Template |
| File format | Electronic submission only, through the site's own Submit Paper system; "Papers submitted by e-mail will not be considered." File type (PDF/DOC) is **not stated**. A LaTeX source is not mentioned; the official template is a Word file. | Call for Papers |
| IEEE PDF eXpress | A "How to Use IEEE PDF eXpress" guide is provided, so expect an IEEE Xplore-compatible PDF check at the camera-ready stage. | Downloads |

## Review, authorship and anonymisation

| Requirement | Verified value | Source |
|---|---|---|
| Review | Two rounds. Round 1 checks technical suitability, scope and plagiarism. Qualifying papers are then "double blind reviewed by three experts". | Call for Papers, Review Process |
| Anonymisation instructions | **Not stated.** The template has a full author block, and submission requires every co-author's site Member ID. Whether the uploaded PDF must be anonymised is **unresolved**: confirm with Track 4 chairs. `main.tex` has an `\anonymoustrue` switch so an anonymous review copy can be built. | — |
| Membership | Every author and co-author must register free on the site and give their Member ID at submission. | Guidelines |
| Plagiarism | "plagiarised papers will not be accepted"; papers are checked with Turnitin; an originality report may be uploaded. No similarity threshold is stated. | Guidelines, Review Process |
| Presentation requirement | Accepted papers must be presented. Papers that are not presented are not forwarded to IEEE Xplore. | FAQs |

## Publication status — not guaranteed

- Home page: technically approved by the IEEE Delhi Section. After presentation, papers are submitted to IEEE for *possible* inclusion in IEEE Xplore, depending on peer-review quality and IEEE's Technical Program Integrity Committee.
- Call for Papers: the conference "has applied for IEEE's Conference Publication Program".
- Review Process page: inclusion in IEEE Xplore is "subject to meeting IEEE Xplore's scope and quality requirements".

**Acceptance and IEEE Xplore inclusion are therefore not guaranteed.**

## Registration fees (Indian authors, INR)

| Category | Early bird (≤ 22 Feb 2027) | Regular | Spot |
|---|---|---|---|
| Full-time students | 6,400 (Member) / 8,000 (General) | 7,200 / 9,000 | 8,000 / 10,000 |
| Teachers / research scholars | 8,000 / 10,000 | 8,800 / 11,000 | 9,600 / 12,000 |
| Industry | 10,400 / 13,000 | 11,200 / 14,000 | 12,000 / 15,000 |

- Every accepted paper needs at least one full registration.
- Additional papers from the same author group cost Rs. 2,000 each.
- Fees are non-refundable (FAQs).
- "Member" and "General" are the site's categories. Which applies to the authors must be confirmed.

## AI-assistance policy

- **INDIACom:** no AI-content policy is stated on the pages checked.
- **IEEE (Author Center):** content generated by AI, "including but not limited to text, figures, images, and code", must be disclosed in the acknowledgments section. The disclosure must name the AI system, identify the sections affected, and briefly describe the level of use. AI used only for editing and grammar is "not required, but recommended" to disclose.

See `AI_ASSISTANCE_DISCLOSURE.md`.

## Open questions to send to the Track 4 chairs before submission

1. Must the submitted PDF be anonymised for the double-blind round?
2. Do references count toward the 6 pages?
3. Is a LaTeX-produced PDF in IEEE conference format acceptable in place of the Word template?
4. Where is the A4 version of the template, and the 2027 running header?
