# ICSE 2027 Paper-Facing Results

This directory contains compact, sanitized evidence products used by the PARA
paper. It intentionally excludes raw provider responses, API preflight files,
third-party source checkouts, frozen full task splits, local logs, and build
outputs.

## Contents

- `unified_rerun/`: canonical relation matrix over Spring targets and methods.
- `high_complexity_multiseed/`: multi-seed high-complexity target summaries.
- `contract_summary.{csv,json}`: sampled query-level proof-contract inventory.
- `proof_contracts/`: representative positive and negative proof contracts.

All local filesystem references are rewritten to placeholders such as
`<RESULTS_ROOT>` or `<WORKSPACE_ROOT>`. To reproduce from raw artifacts, unpack
the frozen artifact bundle and run:

```bash
python scripts/final_icse2027/export_public_artifacts.py \
  --source-results artifacts/paper_final_icse2027/results \
  --output-dir results/final_icse2027
```
