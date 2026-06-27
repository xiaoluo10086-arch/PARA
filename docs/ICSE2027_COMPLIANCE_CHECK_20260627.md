# ICSE 2027 Research Track Compliance Check

Date: 2026-06-27

Official page checked:
<https://conf.researchr.org/track/icse-2027/icse-2027-research-track>

## Submission Requirements Extracted From The Official Page

- Research-track submission deadline: June 30, 2026.
- Format: IEEE conference proceedings template.
- LaTeX class requirement: `\documentclass[10pt,conference]{IEEEtran}`;
  do not use `compsoc` or `compsocconf`.
- Page limit: at most 10 pages for main text, inclusive of figures, tables,
  appendices, etc.; up to 2 additional pages containing only references.
- Submission must be PDF.
- Review model: double-anonymous; author names must be omitted, and prior work
  should be referenced in the third person.
- Reference integrity: hallucinated, fabricated, or unverifiable references can
  cause desk rejection.
- Open Science: artifact/data sharing is not mandatory for acceptance, but is
  expected by default; non-sharing must be justified. Authors should provide
  access instructions in the paper or explain why this is not possible.
- Generative AI disclosure: content generated with AI tools must be disclosed
  according to ACM/IEEE policies; ordinary spelling/grammar editing does not
  require disclosure.
- Human-subjects policy: if applicable, authors must declare compliance with
  ACM human-participants policy.
- Concurrent submission and plagiarism policies apply.

## Current Paper Status

Checked file:
`paper_final_icse2027/IEEE-conference-template-062824/para_icse2027_final_draft.tex`

Current compiled PDF:
`paper_final_icse2027/IEEE-conference-template-062824/para_icse2027_final_draft.pdf`

Status:

- Uses `\documentclass[10pt,conference]{IEEEtran}`.
- Compiles successfully with `pdflatex` + `bibtex` + `pdflatex` + `pdflatex`.
- Current PDF length: 7 pages total.
- Current PDF page size: letter.
- No undefined references.
- No overfull boxes in the latest log; only underfull warnings.
- Author block is anonymous.
- Added a `Data and Artifact Availability` section with double-anonymous
  artifact packaging language.
- Figure/table products were regenerated from the unified rerun matrix.

## Updated Figures And Tables

- Fig. 2 now uses `fig_unified_matrix.pdf`, generated from
  `results/unified_rerun_20260627/unified_core_summary_by_target_method.csv`.
- Table II now reports the 12-row unified matrix:
  4 targets x 3 methods, aggregated over 3 seeds.
- Fig. 4 high-complexity panel now reads the unified rerun matrix rather than
  hard-coded old table rows.
- The resulting paper has better data density than the prior 6-page draft:
  the main effectiveness evidence is visible as both heatmap and compact table.

## Remaining Submission Risks / To-Do

P0 before submission:

- Verify every bibliography entry against a traceable source. The ICSE page
  explicitly states that hallucinated or fabricated references can be
  desk-rejected.
- Create or prepare an anonymized artifact package/repository and ensure the
  paper's artifact statement matches what will actually be submitted.
- Decide whether an AI-use disclosure is required for this manuscript. If text,
  tables, graphs, code, or citations were generated with AI tools rather than
  only grammar/punctuation polishing, ACM/IEEE policies require disclosure.

P1 before submission:

- Add more prose if desired: the paper is currently 7 pages, below the 10-page
  main-text limit. There is room to strengthen related work, method details,
  and threats without risking the page limit.
- Re-check double-anonymous safety if an artifact link is inserted. Use an
  anonymous repository or supplemental upload.
- Consider shortening long target names in tables if final camera-ready layout
  becomes crowded after adding more content.

