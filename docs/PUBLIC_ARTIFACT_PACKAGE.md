# Public Artifact Package

This repository is the public PARA code and evidence package for the paper
"PARA: Path-Accountable Proof Trees for Software Architecture Relation
Reasoning."

## Uploaded in Git

- Core implementation under `src/para/`.
- CLI and experiment scripts under `scripts/`.
- ProofStrategy and agent decision-contract schemas under `src/para/schemas/`.
- A runnable toy task under `data/example/`.
- Curated paper evidence under `results/`, including the latest
  `results/final_icse2027/` package.
- Representative proof contracts and audit summaries.
- Figure source data and generated paper figures under `figures/`.
- Reproducibility, data-boundary, anonymity, and architecture documentation
  under `docs/`.

## Not Uploaded in Git

- API keys, `.env` files, private endpoints, account balances, or provider
  account metadata.
- Raw model/provider responses and API preflight files.
- Full Spring/TEAMMATES/OpenCV source checkouts, build directories, and caches.
- Full frozen task splits, raw graph materializations, and complete run logs.
- Manuscript draft history, internal review notes, and obsolete experiment
  branches.
- Files with local absolute paths or unclear redistribution licenses.

Large valid artifacts should be distributed as anonymous review supplements,
release assets, or archival records, then unpacked under `artifacts/` or
`external-data/`.

## ICSE-Style Review Use

For double-anonymous review, create a separate anonymous snapshot from this
commit and remove repository/account identifiers. The anonymous snapshot should
preserve:

- the same source tree;
- the same curated result files;
- the same relative artifact layout;
- enough precomputed outputs to inspect table values without API access.

The official named repository can be used after the anonymous review period or
for public release.
