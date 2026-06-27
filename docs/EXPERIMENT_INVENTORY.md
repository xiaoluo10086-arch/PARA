# Experiment Inventory for Final Paper

Date: 2026-06-26

This document summarizes the current experiment evidence and assigns each item
to the final paper, appendix, or future rerun.

## A. Main Effectiveness Results

### Spring xlarge3, strict-agentic PARA

Source:

```text
paper_reasoning/results/icse_revision_20260620/e3_qwen_multiseed/
paper_reasoning/results/icse_revision_20260620/e6_e7_runtime_trace/
```

Current evidence:

- `canCallClass`: held-out accuracy 1.000, supported recall 1.000.
- `isAllowedToUse`: held-out accuracy about 0.941, supported recall about 0.824,
  no false supported negatives.
- `overridesMethod`: held-out recall 1.000, two seed-level false supported
  negatives in seeds 1 and 2.

Use in final paper:

- Main Table 1: full PARA held-out results across targets and seeds.
- Keep emphasis on supported precision, supported recall, false supported
  negatives, and positive abstention.
- Do not overemphasize raw accuracy because no-rule already gets 0.667 under
  the 1:2 label ratio.

### TEAMMATES clean9

Source:

```text
paper_reasoning/results/learn_then_reason_teammates_clean9/
paper_reasoning/results/icse_revision_20260620/e1_matched_graphrag/
```

Current evidence:

- Small/sparse tasks expose conservative rejection.
- Accepted cases avoid false support.

Use in final paper:

- Use as a secondary generality/conservativeness check.
- Compress to one small table or appendix.
- Do not let it compete with the main Spring narrative.

## B. Mechanism and Baseline Results

### Deterministic GraphRAG / typed retrieval

Source:

```text
paper_reasoning/results/icse_revision_20260620/e1_matched_graphrag/
paper_reasoning/applications/typed_symbolic_baseline/results/
```

Current evidence:

- Matched GraphRAG reaches the same held-out envelope as strict-agentic PARA on
  Spring xlarge3, but with higher retrieval/candidate-generation cost.
- Typed heuristic baseline accepts 2/3 Spring targets:
  - `canCallClass`: train F1 1.000, held-out recall 1.000.
  - `isAllowedToUse`: train F1 0.9362, held-out recall 0.820.
  - `overridesMethod`: candidate rejected, train F1 0.7952.

Use in final paper:

- Essential main-paper baseline.
- Reframe contribution: PARA includes a graph-indexed symbolic retrieval
  substrate; LLM planning is not the only source of accepted rules.
- Use this to prevent the reviewer from claiming the paper hides a strong
  deterministic baseline.

### Popper

Source:

```text
paper_reasoning/results/icse_revision_20260621/e14_matched_popper_spring_seed0/
```

Current evidence:

- Popper accepts `canCallClass` in 26.481s.
- Popper times out for `isAllowedToUse` and `overridesMethod` under 300s.

Use in final paper:

- Keep as an external ILP calibration, not a decisive "win" claim.
- State that Popper solves the simplest target and times out on the two others
  under this fixed budget.
- Avoid explaining the timeout as internal search explosion unless telemetry is
  added.

### Direct LLM query answering

Source:

```text
paper_reasoning/results/icse_revision_20260621/direct_query_answering_matched/
```

Current evidence:

- Direct QA can answer some `canCallClass` and `isAllowedToUse` queries.
- It is very weak on `overridesMethod`:
  - Qwen27B supported recall 0.007.
  - DeepSeek supported recall 0.007.

Use in final paper:

- Keep as baseline showing why direct query answering is not enough for
  compositional proof obligations.
- Use one compact figure or table.

## C. Agent-Specific Evidence

### Agentic loop audit

Source:

```text
paper_reasoning/applications/agentic_loop_probe/results/
```

Current evidence:

- 31/31 agent-like runs have persisted traces.
- 31/31 have verifier feedback.
- 14/31 have refiner actions.
- 1/31 trace-replay runs show candidate F1 improvement.

Use in final paper:

- Main paper: use as historical evidence that the agent records persisted
  state/action/feedback traces. The final wording should use the upgraded
  Verifier-Governed ProofStrategy Agent terminology.
- Do not claim the refiner generally improves accuracy.
- Use appendix for detailed replay statistics.

### Planner evidence ablation

Source:

```text
paper_reasoning/results/icse_revision_20260621/e15_planner_evidence_ablation/
```

Current evidence:

- `canCallClass`: full PARA improves held-out recall over witness-only and
  schema-only (0.687 -> 1.000).
- `isAllowedToUse`: all arms reach the same held-out recall 0.820.
- `overridesMethod`: schema-only is rejected; full PARA is accepted; witness-only
  is also accepted.

Use in final paper:

- This is the best current evidence for the agent's planner role.
- Phrase carefully: bounded train-witness evidence and agent planning can change
  the selected path program and is critical in selected cases, not universally.

## D. Accountability Evidence

Sources:

```text
paper_reasoning/results/icse_revision_20260620/e4_agent_proof_audit/
paper_reasoning/results/icse_revision_20260620/e5_accountability/
paper_reasoning/results/icse_revision_20260620/e8_integrity_audit/
paper_reasoning/results/icse_revision_20260621/structural_near_miss/
```

Current evidence:

- 50 sampled proof trees and 190 EDB mentions audited with 0 errors.
- Exact-signature near-miss negatives: 554 negatives, 0 false supported.
- Artifact isolation audit: 13 checks, 0 violations.
- Counterfactual evidence identifies critical versus alternative proof facts.

Use in final paper:

- Main contribution evidence.
- Keep one table and one figure.
- This is stronger than pure accuracy results because it supports the
  accountability claim.

## E. Application and Engineering Probes

### OpenCV static package/dependency slice

Source:

```text
paper_reasoning/applications/opencv_packaging_validation/
paper_reasoning/applications/opencv_full_validation/
paper_reasoning/applications/opencv_miniproj_slice/
paper_reasoning/applications/opencv_dep_slice/
```

Current evidence:

- OpenCV-like mini package demonstrations exist for fast sanity checks.
- Full OpenCV source validation uses a real 312M OpenCV checkout, extracts 484
  package/build facts over 26 modules, accepts 3 rules, rejects over-general
  candidates, and reports 80.8% source-level reduction.
- Four-module OpenCV slice: `core`, `features2d`, `imgcodecs`, `imgproc`.
- Build/link/runtime validation passes.
- 12/12 valid JPEG workloads pass; 2/2 negative workloads fail as expected.
- Leave-one-out removals fail.
- Baseline matrix shows PARA and post-build `ldd` match the observed runtime
  closure, while include-only and symbol-only heuristics miss transitive
  `core`.

Use in final paper:

- Use as an application scalability ladder: mini static package, full real
  source tree, build-facing validated package, and boundary diagnostics.
- Keep the compact row in the main RQ4 table and put full matrices in appendix.
- Do not present as a main benchmark or as global minimality.
- Use to show that the proof contract can transfer to static-file/package
  evidence when validators exist.
- If time allows, add one extra package probe only when it has local/frozen
  source, typed facts, proof contracts, and an automated validator.

### Static extractor boundary

Source:

```text
paper_reasoning/applications/static_extractor_boundary/
```

Use in final paper:

- Threats to validity / boundary contract.
- Shows `INCONCLUSIVE` is a review obligation, not negative proof.

## F. Current Problem With v6 Experiment Organization

The v6 paper has too many experimental roles inside the main Results:

- held-out reasoning;
- GraphRAG runtime;
- Popper;
- project-specific policy;
- TEAMMATES;
- proof audit;
- near miss;
- recursive IDB;
- planner ablation;
- multi-model API;
- direct QA;
- direct rule generation;
- OpenCV application.

Final paper should compress this into four main RQs and move the rest to
appendix/artifact evidence.
