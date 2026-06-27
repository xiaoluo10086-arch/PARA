# Unified Rerun Matrix (2026-06-27)

This directory contains paper-facing tables built from the 2026-06-27 rerun.

## Inputs

- Fresh learn/admission rerun: `Agent‑Symbolic_Hybrid_Rule_Learning/paper_reasoning/paper_final_icse2027/results/final_suite_20260627_unified_p0a_learn_only/tables/unified_p0a_learn_only_summary.csv`
- Verified held-out Spring suite: `Agent‑Symbolic_Hybrid_Rule_Learning/paper_reasoning/proof_strategy_agent_v2/results/final_paper_verified_suite_20260627/tables/table1_effectiveness_rows.csv`
- Verified high-complexity suite: `Agent‑Symbolic_Hybrid_Rule_Learning/paper_reasoning/proof_strategy_agent_v2/results/final_paper_verified_suite_20260627/tables/table2_high_complexity_rows.csv`

## Outputs

- `unified_core_matrix.csv`: row-level project/target/seed/method matrix.
- `unified_core_summary_by_target_method.csv`: compact table for paper figures/tables.

## Interpretation

The fresh rerun completed the low-cost learn/admission layer for 27 Spring rows.
The combined matrix contains 36 rows, with 27 admitted rows across fresh and verified layers.
Full query-level proof search was not rerun for every Spring cell because proof materialization is the expensive layer;
held-out fields are therefore explicitly sourced from the existing verified suite.  This keeps the paper table
honest: admission is freshly rerun, while proof/search evidence is treated as a separate validated layer.

Summary rows: 12.
