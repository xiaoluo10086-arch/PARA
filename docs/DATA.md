# Data and Artifact Boundary

## Included in Git

- A small runnable task in `data/example/`.
- Paper-facing CSV/JSON summaries in `results/summaries/`.
- Sampled proof traces used by the proof audit.
- Counterfactual and near-miss accountability summaries.
- Figure source data.

## Distributed Separately

Full Spring Framework and TEAMMATES task splits, complete run directories, raw
provider responses, and intermediate graph materializations are large derived
artifacts. They should be published as versioned release assets or in an
archival artifact service, then unpacked under:

```text
artifacts/
external-data/
```

Both directories are intentionally ignored by Git. Experiment scripts accept
explicit paths so the repository does not depend on one workstation layout.

## Not Redistributed

- LLM weights or checkpoints.
- API credentials or account metadata.
- Full third-party source checkouts and build outputs.
- Invalid transport/debug runs excluded by the experiment governance rules.
- Historical manuscript drafts and superseded result snapshots.

Third-party source code and derived datasets remain subject to the upstream
project licenses.
