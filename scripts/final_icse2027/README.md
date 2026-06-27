# Final Paper Experiment Scripts

This directory contains the public, paper-facing scripts for regenerating
compact PARA evidence tables from a frozen artifact bundle.

## Local, No-API Scripts

- `export_public_artifacts.py`: copies compact final results into
  `results/final_icse2027/` and removes local filesystem paths.
- `build_high_complexity_multiseed_summary.py`: aggregates multi-seed
  high-complexity target runs.
- `build_unified_rerun_matrix.py`: builds the unified target-by-method matrix
  from frozen run manifests.
- `build_final_table_products.py`: converts sanitized summaries into
  paper-ready table products.
- `materialize_proof_contracts.py`: materializes sampled query-level proof
  contracts from frozen manifests and rule libraries.
- `summarize_agent_trace.py` and `summarize_final_suite.py`: summarize
  verifier-governed strategy traces and full-suite status files.

## Full Reproduction Boundary

The repository intentionally excludes raw logs, third-party source checkouts,
full task splits, API preflight files, and model/provider responses. To rerun
the complete experiment suite, unpack the frozen artifact bundle under
`artifacts/` and pass explicit input/output paths to these scripts.

LLM-backed planning requires user-provided model endpoints and keys through
environment variables. The deterministic proof replay, table aggregation, and
public artifact export steps do not require any provider account.
