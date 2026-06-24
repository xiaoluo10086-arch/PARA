# Project-Specific Spring Policy Supplemental Experiment

## Purpose

This supplemental experiment addresses the reviewer-risk that the main targets
(`canCallClass`, `isAllowedToUse`, `overridesMethod`) look like generic Java
relations.  It constructs a Spring-specific target, `springSameModuleUse/2`,
whose label depends on Spring's project module naming rather than Java language
semantics.

## Target

`springSameModuleUse(PkgA, PkgB)` means:

1. a class in `PkgA` imports a class in `PkgB`; and
2. `PkgA` and `PkgB` share the same Spring top-level module label, represented
   by the project-specific EDB fact `springModuleName(Pkg, Module)`.

This is not a universal Java relation.  A hand-written checker would need a
Spring-specific module taxonomy; PARA receives the taxonomy as graph facts and
must learn the evidence path and equality-style module constraint from examples.

## Configuration

| Item | Value |
|---|---|
| Source graph | `external-data/spring/spring_nshrl_isAllowedToUse` |
| Derived task | `artifacts/icse_revision_20260623/project_specific_spring_policy_v2/task_spring_same_module_use` |
| Package import edges | 3401 |
| Packages with module labels | 487 |
| Positive examples | 1226 |
| Negative examples | 23145 |
| Negative import edges across modules | 2175 |
| Negative same-module non-import pairs | 20970 |
| Train split | 50 positive / 100 negative |
| Held-out split | 1176 positive / 23045 negative |
| Main guide provider | Strict-agentic DeepSeek path-program planning |
| Deterministic guide provider | GraphRAG candidate generator with attribute constraints |
| Acceptance threshold | F1 >= 0.8 |

## Strict-Agentic Learned Rule

The strict-agentic DeepSeek run proposes a portfolio of executable path
programs, including the import path:

```json
[
  ["containsClass", "fwd"],
  ["importsClass", "fwd"],
  ["containsClass", "rev"]
]
```

The symbolic verifier then selects the import-path rule conjoined with the
project-specific module equality constraint:

```prolog
springSameModuleUse(A,B) :- containsClass(A,V1),importsClass(V1,V2),containsClass(B,V2),springModuleName(A,C1),springModuleName(B,C1) [0.980].
```

The deterministic GraphRAG-guided run independently reaches the same rule shape
with score 0.944.

## Results

| Stage | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Strict-agentic train verification | 1.000 | 1.000 | 1.000 | 1.000 | 50 | 0 | 100 | 0 |
| Strict-agentic held-out reasoning | 1.000 | 1.000 | 1.000 | 1.000 | 1176 | 0 | 23045 | 0 |
| Deterministic GraphRAG-guided train verification | 1.000 | 1.000 | 1.000 | 1.000 | 50 | 0 | 100 | 0 |
| Deterministic GraphRAG-guided held-out reasoning | 1.000 | 1.000 | 1.000 | 1.000 | 1176 | 0 | 23045 | 0 |
| No-rule policy | 0.951 | 0.000 | 0.000 | 0.000 | 0 | 0 | 23045 | 1176 |

## Matched Popper Baseline

We also ran the pure Popper baseline on the same v2 train split with the same
target predicate and background facts.  This run uses the corrected negative
set above, which includes same-module pairs without an import edge and
therefore prevents the shortcut rule `same module => positive`.

| Baseline artifact | Train precision | Train recall | Train F1 | Held-out precision | Held-out recall | Held-out accuracy | False supported | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Popper selected single rule | 1.000 | 0.280 | 0.438 | n/a | n/a | n/a | n/a | 308.70s |
| Popper stdout full program | 1.000 | 0.680 | 0.810 | 0.264 | 0.643 | 0.895 | 2113 | 308.70s + 62.56s eval |

The Popper process exceeded the configured 300s search budget after reporting a
multi-clause program with train F1 0.810.  The PARA baseline wrapper selected
the best single rule as `weak` (F1 0.438), but for fairness we also evaluated
the complete Popper stdout program as a separate rule library on the held-out
split.  That full program supports 756/1176 positives but also supports 2113
negative held-out queries, yielding held-out precision 0.264.  This is useful
as an ILP boundary result: the relation is learnable from the graph, but the
unconstrained multi-clause ILP program does not preserve the project-specific
constraint cleanly on held-out queries.

## Superseded v1 Note

The earlier `project_specific_spring_policy/` run is superseded by this v2
design.  In v1, negatives were limited to cross-module import edges.  Popper
could therefore learn a module-only shortcut, which did not test whether a
method must combine import evidence with the project-specific module
constraint.  The v2 negative set adds same-module package pairs without import
edges and should be used for paper claims.

## Interpretation

Use this result as a supplemental project-specific relation target, not as a
replacement for the main Spring xlarge3 evaluation.  It supports the revised
paper claim that strict-agentic PARA can operate on project-specific typed
facts and discover an auditable path-program rule for a relation whose meaning
is not built into the Java language.  The deterministic GraphRAG-guided run
reaches the same rule, so this target should not be used to claim raw-quality
superiority over deterministic retrieval.  Its main value is motivation
alignment: project-specific module facts matter, and the accepted proof rule
must combine import evidence with that project-specific equality constraint.

## Files

- Task summary: `artifacts/icse_revision_20260623/project_specific_spring_policy_v2/task_summary.json`
- Manifest: `artifacts/icse_revision_20260623/project_specific_spring_policy_v2/learn_then_reason/manifests/spring_same_module_use_train50_100_seed0_graphrag_manifest.json`
- Reasoning output: `artifacts/icse_revision_20260623/project_specific_spring_policy_v2/learn_then_reason/reason_eval/spring_same_module_use_train50_100_seed0_graphrag_test_reason_eval.json`
- No-rule output: `artifacts/icse_revision_20260623/project_specific_spring_policy_v2/learn_then_reason/reason_eval/spring_same_module_use_train50_100_seed0_graphrag_test_no_rule_reason_eval.json`
- Strict-agentic manifest: `artifacts/icse_revision_20260623/project_specific_spring_policy_v2/strict_agentic_deepseek/manifests/spring_same_module_use_train50_100_seed0_agent_strict_deepseek_manifest.json`
- Strict-agentic reasoning output: `artifacts/icse_revision_20260623/project_specific_spring_policy_v2/strict_agentic_deepseek/reason_eval/spring_same_module_use_train50_100_seed0_agent_strict_deepseek_test_reason_eval.json`
- Popper summary: `artifacts/icse_revision_20260623/project_specific_spring_policy_v2/pure_popper_clingo/learn/summary.json`
- Popper full-program held-out output: `artifacts/icse_revision_20260623/project_specific_spring_policy_v2/pure_popper_clingo/popper_full_program_reason_eval.json`
