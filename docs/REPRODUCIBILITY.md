# Reproducibility Guide

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,figures]"
pytest
```

## 2. Verify the Public Example

```bash
para inspect --task-dir data/example/can_call_class
para reason-eval \
  --task-dir data/example/can_call_class \
  --rule-library data/example/can_call_class/rule_library.json \
  --json
```

## 3. Full Experiments

Obtain the frozen artifact bundle and unpack it under `artifacts/`. The evidence
catalog maps every paper claim to its summary and generating script.

Principal entry points:

- `scripts/final_icse2027/export_public_artifacts.py`
- `scripts/final_icse2027/build_unified_rerun_matrix.py`
- `scripts/final_icse2027/build_high_complexity_multiseed_summary.py`
- `scripts/final_icse2027/materialize_proof_contracts.py`
- `scripts/run_learn_then_reason.py`
- `scripts/run_matched_popper_baseline.py`
- `scripts/run_direct_query_answering.py`
- `scripts/run_recursive_idb_evaluation.py`
- `scripts/audit_proof_traces.py`
- `scripts/analyze_structural_near_miss.py`

All task, split, model endpoint, and output locations are command-line
arguments. The scripts do not require the original authors' directory layout.

The compact paper-facing evidence package can be regenerated from the frozen
result bundle without API keys:

```bash
python scripts/final_icse2027/export_public_artifacts.py \
  --source-results artifacts/paper_final_icse2027/results \
  --output-dir results/final_icse2027
```

## 4. LLM Configuration

Use a local OpenAI-compatible server or a supported remote endpoint. Keep keys
in the environment, never in commands committed to Git.

```bash
export PARA_LLM_BASE_URL=http://127.0.0.1:8000
export PARA_LLM_MODEL=YOUR_MODEL
```

For remote services, set the provider's standard key variable. Exact model
versions, prompts, bounds, and run dates belong in the frozen artifact
manifest.

## 5. Deterministic Components

Given the same task, accepted rule library, and proof bounds, indexed execution,
candidate compilation, verification, and proof construction are deterministic.
Remote LLM planning can vary and must be reported separately from transport or
authentication failures.
