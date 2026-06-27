# PARA Anonymous Artifact README

This artifact supports the double-anonymous review of the PARA paper. It is
organized around the evidence layers reported in the paper rather than around
implementation history.

## Artifact Contents

- `src/para/`: implementation of the verifier-governed planner, symbolic
  verifier, proof engine, and ProofStrategy agent interface.
- `scripts/`: runnable experiment, audit, proof-contract, and table-generation
  entry points.
- `results/`: paper-facing result summaries, proof-contract samples,
  high-complexity stress results, and agent-control summaries.
- `figures/`: figure source data, plotting scripts, and final generated figures.
- `docs/`: data boundary, reproduction, anonymity, architecture, and evidence
  notes.

## Offline Reproduction

The following parts should be reproducible without external model APIs once the
artifact package is unpacked:

- regenerate paper figures from stored CSV summaries;
- compile the paper PDF;
- inspect learned rule libraries and ProofStrategy traces;
- replay stored proof-contract samples;
- verify table values against the included summary CSV files.
- inspect the matched Popper/ILP calibration logs and learned clauses.

Recommended commands from the repository root:

```bash
python -m pip install -e ".[dev,figures]"
pytest
python scripts/final_icse2027/export_public_artifacts.py \
  --source-results artifacts/paper_final_icse2027/results \
  --output-dir results/final_icse2027
```

Use the Python environment supplied by the artifact or any environment with the
declared optional dependencies.

## Result Sources and Baseline Scope

The paper reports several evidence layers with different reproduction costs.
Reviewers should treat them as follows:

- Unified table/figure summaries are paper-facing aggregates generated from
  archived CSV files under `results/`.
- Proof-contract samples and audit counts are offline replay artifacts: they
  can be checked without repeating model calls.
- The Popper/ILP row is a matched typed-bias calibration, not a weak baseline.
  It uses the same frozen split, background facts, held-out split, and typed
  predicate bias as the corresponding PARA runs. It does not receive
  path-program hints or a PARA-generated candidate path family.
- A path-hinted ILP rerun would evaluate a different interface: PARA-like path
  guidance plus an ILP backend. The included Popper calibration instead tests
  pure symbolic search under the documented fixed budget.
- High-complexity proof contracts are representative query-level
  materializations. Full proof materialization for every held-out query is a
  deployment-cost option rather than a required step for checking the reported
  aggregate metrics.

## API-Dependent Runs

Some strategy-generation experiments use commercial or externally hosted LLMs
such as GPT-4o, DeepSeek Chat, and Gemini 2.5 Pro. These runs are not required
for offline proof replay. The anonymized artifact should include:

- prompts or strategy-request templates;
- parser/verifier outputs;
- precomputed model responses or normalized ProofStrategy traces;
- metadata indicating which rows require an API key to rerun.

API-dependent reruns may differ because hosted models can change over time. The
paper therefore relies on the archived traces and verifier outputs for
review-time reproducibility.

## Precomputed Results

The artifact includes precomputed outputs for long-running or externally
dependent steps:

- unified Spring rerun matrix;
- high-complexity multi-seed strategy matrix;
- proof-accountability audit counts;
- agent memory/control traces;
- API robustness summaries;
- OpenCV package-slicing validation summaries.

These files are intended to let reviewers check the reported numbers without
rerunning every upstream extraction or model call.

## What Each Evidence Layer Checks

- Proof-status effectiveness: held-out query decisions under accepted
  run-local rules.
- Proof accountability: proof-contract consistency, fact-removal sensitivity,
  and near-miss false-support stress tests.
- Strategy governance: whether candidate strategies are admitted or rejected by
  the same symbolic verifier.
- Agent-assisted strategy proposal: memory-backed and reflection-backed path
  proposal under verifier control.
- Transfer and engineering validation: reuse of the proof core when typed facts
  and external validators are available.

## Anonymization Checklist

Before submission, remove or replace:

- absolute local paths;
- personal usernames;
- API keys, endpoint tokens, and shell history;
- non-anonymous repository URLs;
- generated caches such as `__pycache__`, `.pytest_cache`, and temporary logs.

The artifact should keep enough path compatibility for the table and figure
scripts to run, but it should not expose author identity.

## Expected Reviewer Workflow

1. Read the paper and inspect Fig. 1 to understand the proof-producing pipeline.
2. Regenerate figures and compile the PDF.
3. Check table values against the included CSV summaries.
4. Inspect representative ProofStrategy traces and proof-contract samples.
5. Optionally rerun API-dependent strategy generation if the reviewer supplies
   compatible API credentials.
