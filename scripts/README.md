# Experiment Scripts

The scripts are grouped by purpose:

- Learning and reasoning: `run_learn_then_reason.py`
- Matched baselines: `run_matched_popper_baseline.py`,
  `run_direct_query_answering.py`, `run_direct_llm_reasoning_ablation.py`
- Accountability: `audit_proof_traces.py`,
  `analyze_structural_near_miss.py`, `materialize_false_supported_case.py`
- Capability checks: `run_recursive_idb_evaluation.py`
- Aggregation: the `summarize_*.py` scripts

Every external task or artifact location is passed as a command-line argument
or resolved under the repository-local ignored `artifacts/` directory. Run any
script with `--help` for its exact interface.
