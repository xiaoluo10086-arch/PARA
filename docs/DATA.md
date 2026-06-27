# Data and Artifact Boundary

## Included in Git

- A small runnable task in `data/example/`.
- Paper-facing CSV/JSON summaries in `results/summaries/`.
- ICSE 2027 paper-facing summaries and representative proof contracts in
  `results/final_icse2027/`.
- Sampled proof traces used by the proof audit.
- Counterfactual and near-miss accountability summaries.
- Figure source data and final paper PDFs in `figures/`.

## Distributed Separately

Full Spring Framework and TEAMMATES task splits, high-complexity frozen task
directories, complete run directories, raw provider responses, and intermediate
graph materializations are large derived artifacts. They should be published as
versioned release assets or in an archival artifact service, then unpacked
under:

```text
artifacts/
external-data/
```

Both directories are intentionally ignored by Git. Experiment scripts accept
explicit paths so the repository does not depend on one workstation layout.

`results/final_icse2027/` is generated from the frozen result bundle by:

```bash
python scripts/final_icse2027/export_public_artifacts.py \
  --source-results artifacts/paper_final_icse2027/results \
  --output-dir results/final_icse2027
```

The export step removes local filesystem paths and excludes API budget checks,
debug logs, and invalid transport runs.

## Not Redistributed

- LLM weights or checkpoints.
- API credentials or account metadata.
- Full third-party source checkouts and build outputs.
- Invalid transport/debug runs excluded by the experiment governance rules.
- Historical manuscript drafts and superseded result snapshots.

Third-party source code and derived datasets remain subject to the upstream
project licenses.
