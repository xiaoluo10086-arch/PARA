# PARA Experiment Evidence Catalog

Date: 2026-06-21

This document consolidates all experiment results that are either used in the
current PARA manuscript or valid for appendix / reviewer-response use. It is
organized by the paper question each experiment supports. Superseded,
diagnostic, invalid-infrastructure, transport-failure, and old full-matrix
results are not promoted as current main evidence unless explicitly marked as
context only.

## Use Rules

- **Main-text evidence**: results used directly in
  `the accompanying PARA manuscript`.
- **Appendix / response evidence**: valid experiments that support boundaries,
  robustness, or reviewer responses but are not necessary in the main text.
- **Context only**: prior deterministic/full-matrix evidence that may motivate
  the work but must not be pooled with held-out reasoning metrics.
- **Do not use as model-quality evidence**: network failures, sandbox failures,
  authentication failures, invalid Gemini output-budget settings, debug-repair
  probes, assisted-prior probes, or deterministic fallback runs.

## Main-Text Evidence

### E1. Spring xlarge3 strict-agentic learn-then-reason

**What it proves.**

PARA can learn accepted run-local rules from the current project graph and use
them to answer held-out architecture-relation queries with three-valued
judgements.

**Experiment configuration.**

- Project: Spring Framework xlarge3.
- Targets: `canCallClass`, `isAllowedToUse`, `overridesMethod`.
- Seeds: 0, 1, 2.
- Train split: 50 positives / 100 negatives per target.
- Held-out split: 150 positives / 300 negatives per target.
- Protocol: strict-agentic PARA, indexed path-program execution, no
  deterministic fallback, no direct-LLM candidate channel, no assisted prior.
- Portfolio: up to five path programs in one Planner/Refiner action.

**Experiment data.**

Source:

- `artifacts/icse_revision_20260620/e2_portfolio_k/k5/`
- `artifacts/icse_revision_20260620/e3_qwen_multiseed/`

**Experiment results.**

| Target | Accuracy | Supported precision | Supported recall | Negative non-support | Inconclusive |
|---|---:|---:|---:|---:|---:|
| `canCallClass` | 1.000 ± 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| `isAllowedToUse` | 0.941 ± 0.001 | 1.000 | 0.824 | 1.000 | 0.059 |
| `overridesMethod` | 0.999 ± 0.001 | 0.996 | 1.000 | 0.998 | 0.000 |

**Paper use.**

Main RQ1 result. Use it to support large-project held-out reasoning. Report
`isAllowedToUse` as a conservative INCONCLUSIVE boundary and
`overridesMethod` false-supported negatives in seeds 1/2 as an explicit
limitation.

### E2. Matched deterministic GraphRAG baseline

**What it proves.**

The path-program contract is compared against a strong non-agentic graph
retrieval baseline on the same frozen held-out reasoning splits. The main
supported contrast is operational: deterministic GraphRAG relies on online
BFS-style candidate generation, whereas PARA executes LLM-proposed path
programs through an index.

**Experiment configuration.**

- Same Spring xlarge3 targets and splits as E1.
- Baseline generator: canonical deterministic GraphRAG `main_attr_support2`.
- Reasoner: same bounded proof engine.
- Validation gate: exact split/input hash audit.

**Experiment data.**

Source:

- `artifacts/icse_revision_20260620/e1_matched_graphrag/`

**Experiment results.**

| Target | Accuracy / recall envelope | Method | Retrieval / exec time | End-to-end time |
|---|---:|---|---:|---:|
| `canCallClass` | 1.000 / 1.000 | GraphRAG | 5.0--9.5 s | 195.63 s |
| `canCallClass` | 1.000 / 1.000 | PARA | 0.002--0.467 s | 146.18 s |
| `isAllowedToUse` | 0.940--0.942 / 0.820--0.827 | GraphRAG | 40.6--111.8 s | 254.70 s |
| `isAllowedToUse` | 0.940--0.942 / 0.820--0.827 | PARA | 0.002--0.007 s | 173.30 s |
| `overridesMethod` | 0.998--1.000 / 1.000 | GraphRAG | 12.8--26.9 s | 223.03 s |
| `overridesMethod` | 0.998--1.000 / 1.000 | PARA | 0.002--0.006 s | 149.91 s |
| **Mean** | -- | GraphRAG | -- | **224.46 s** |
| **Mean** | -- | PARA | -- | **156.46 s** |

**Paper use.**

Main baseline / operational contrast table. Do not claim universal superiority
over deterministic GraphRAG. The valid claim is that PARA removes online BFS
enumeration from the strict-agentic execution step while maintaining the same
quality envelope on these matched Spring splits. Report end-to-end time
alongside retrieval / index execution time so the efficiency claim is not
component cherry-picking.

### E3. TEAMMATES clean9 learn-then-reason

**What it proves.**

The same learn-then-reason protocol remains conservative on smaller, sparser
tasks: rejected cases do not fabricate support.

**Experiment configuration.**

- Project: TEAMMATES clean9.
- Cases: 9 total, 3 targets × 3 complexity levels.
- Protocol: strict learn-then-reason with run-local rule export.
- Rejected cases export no rules and use the three-valued no-rule behavior.

**Experiment data.**

Source:

- `tables/teammates_clean9_learn_then_reason_seed0_summary.md`
- `results/learn_then_reason_teammates_clean9/`

**Experiment results.**

| Target | Accepted | Rejected | Mean accuracy |
|---|---:|---:|---:|
| `isAllowedToUse` | 3/3 | 0 | 1.000 |
| `canCallClass` | 2/3 | 1 small | 0.889 |
| `overridesMethod` | 1/3 | 2 small/mid | 0.778 |
| **All** | **6/9** | **3 conservative** | **0.889** |

**Paper use.**

Main small-project boundary table. Report that rejected cases yield no false
SUPPORTED decisions rather than using a separate all-zero false-support column.

### E4. Planner evidence and portfolio ablations

**What it proves.**

PARA is not merely a deterministic witness-path ranker, and schema text alone
is insufficient for the hardest relation. The full protocol needs both typed
schema and bounded train-witness sketches, then uses a bounded portfolio so
several executable evidence programs can compete under the same symbolic
verifier without fallback.

**Experiment configuration.**

- Project: Spring xlarge3.
- Model: Qwen3.5-27B local for the controlled ablations.
- Protocol: strict-agentic PARA, no fallback, no direct-rule channel.
- Evidence-source ablation arms:
  - `witness_only`: deterministic top train witness path, no LLM planning.
  - `schema_only`: LLM sees schema/examples but no bounded witness sketches.
  - `full_para`: final protocol with schema plus bounded witness sketches.
- Portfolio-size ablation only changes the maximum number of path programs
  (`k`).

**Experiment data.**

Source:

- `artifacts/icse_revision_20260621/e15_planner_evidence_ablation/`
- `artifacts/icse_revision_20260621/e15_planner_evidence_ablation/planner_evidence_ablation_summary.md`
- `artifacts/icse_revision_20260620/e2_portfolio_k/`
- `tables/top1_path_program_ablation_20260618_summary.md`
- `tables/targeted_portfolio_rerun_20260618_summary.md`

**Experiment results.**

Evidence-source ablation (seed 0; held-out supported recall):

| Target | Witness-only | Schema-only | Full PARA |
|---|---:|---:|---:|
| `canCallClass` | 0.687 | 0.687 | 1.000 |
| `isAllowedToUse` | 0.820 | 0.820 | 0.820 |
| `overridesMethod` | 1.000 | rejected (train F1 0.795) | 1.000 |

Portfolio-size ablation:

| Evidence | Target | Model(s) | Early/top-1 recall | Final recall |
|---|---|---|---:|---:|
| Controlled | `canCallClass` | Qwen | 0.687 (`k<=3`) | 1.000 (`k>=4`) |
| Controlled | `isAllowedToUse` | Qwen | 0.820 | 0.820 |
| Controlled | `overridesMethod` | Qwen | 1.000 | 1.000 |
| Targeted | `canCallClass` | 4 LLMs | 0.687 | 1.000 |
| Targeted | `overridesMethod` | DeepSeek | 0.913 | 1.000 |

**Paper use.**

Main RQ3 evidence. Do not claim every target improves. The precise claim is
that bounded witness sketches plus LLM portfolio planning recover evidence
paths missed by either witness-only or schema-only planning, while the small
bounded portfolio fixes a real conservative planning failure for `canCallClass`
and preserves the strict verifier gate.

### E5. Proof-accountability audit

**What it proves.**

Returned proof traces are grounded in real held-out graph facts and accepted
IDB rule applications. INCONCLUSIVE cases are audited as bounded absence within
the configured search limits, not as global proof of non-existence.

**Experiment configuration.**

- Source: strict-agentic Qwen3.5-27B, Spring xlarge3, `k=5`, seed 0.
- Sample: 10 audited decisions per target.
- Audit checks: EDB membership, IDB applications, substitution consistency,
  normal bounded termination for INCONCLUSIVE cases.

**Experiment data.**

Source:

- `artifacts/icse_revision_20260620/e4_agent_proof_audit/`

**Experiment results.**

| Target | Decisions | Proof trees | EDB fact mentions | IDB applications | INCONCLUSIVE audits | Errors |
|---|---:|---:|---:|---:|---:|---:|
| `canCallClass` | 10 | 19 | 57 | 19 | 0 | 0 |
| `isAllowedToUse` | 10 | 21 | 63 | 21 | 5 | 0 |
| `overridesMethod` | 10 | 10 | 70 | 10 | 0 | 0 |
| **Total** | **30** | **50** | **190** | **50** | **5** | **0** |

**Paper use.**

Main accountability evidence. The 190 EDB count is a mention count, not a
unique-fact count. Avoid presenting these mentions as independent statistical
trials; facts are nested within proof trees and queries.

### E6. Counterfactual fact ablation

**What it proves.**

Proof traces are operationally connected to reasoning outputs: removing
selected proof facts can remove support, while some relations have redundant
alternative evidence.

**Experiment configuration.**

- Spring split-learned sample-20.
- For initially SUPPORTED cases, selected EDB facts are removed and bounded
  proof search is rerun.

**Experiment data.**

Source:

- `artifacts/icse_revision_20260620/e5_accountability/`
- `tables/spring_xlarge3_split_counterfactual_sample20_summary.md`
- `tables/teammates_clean9_split_counterfactual_summary.md`

**Experiment results.**

| Target | Queries | Initially SUPPORTED | Critical fact rate | Alternative proof rate | All-selected alternative rate |
|---|---:|---:|---:|---:|---:|
| `canCallClass` | 20 | 20 | 0.667 | 0.333 | 0.200 |
| `isAllowedToUse` | 20 | 16 | 0.167 | 0.833 | 0.625 |
| `overridesMethod` | 20 | 20 | 1.000 | 0.000 | 0.000 |

**Paper use.**

Main accountability stress evidence. Explain that it supports operational
evidence sensitivity, not global causal minimality.

Terminology:

- `Alternative proof rate`: fraction of evaluated queries for which at least
  one removed selected fact still has an alternative proof.
- `All-selected alternative rate`: fraction of evaluated queries for which all
  removed selected facts still have alternative proofs.

### E7. Argument-sharing near-miss negative stress

**What it proves.**

PARA does not confuse close explicit negatives with positives under
argument-sharing stress. This is valid robustness evidence, but it has been
superseded in the main text by E11 exact-signature structural near-miss stress.

**Experiment configuration.**

- Near-miss definition: a held-out explicit negative shares its first or second
  argument with a held-out positive.
- Main current run: Spring/Qwen strict-agentic reasoning.
- Earlier valid TEAMMATES multi-model run: 150 near misses per model.

**Experiment data.**

Source:

- `artifacts/icse_revision_20260620/e5_accountability/`
- `tables/negative_near_miss_stress_20260618_summary.md`

**Experiment results.**

| Dataset / model group | Near-miss negatives | False SUPPORTED |
|---|---:|---:|
| Spring/Qwen current run | 105 | 0 |
| TEAMMATES/Qwen | 150 | 0 |
| TEAMMATES/GPT-4o | 150 | 0 |
| TEAMMATES/DeepSeek | 150 | 0 |
| TEAMMATES/Gemini 2.5 Pro | 150 | 0 |

**Paper use.**

Appendix / reviewer-response robustness evidence. Do not use as the strongest
main-text near-miss result; use E11 instead.

### E8. Direct LLM rule-generation ablation

**What it proves.**

Free-form direct LLM rule generation plus the same verifier does not replace
the path-program contract. A rule can pass train-split verification yet provide
poor held-out proof coverage.

**Experiment configuration.**

- Models: Qwen3.5-27B, GPT-4o, DeepSeek Chat, Gemini 2.5 Pro fixed I/O.
- Input: train facts and examples.
- Output: free-form candidate rule text.
- Acceptance: same verifier threshold (`min_f1 >= 0.8`).
- Reasoning: exported direct-rule library only; no fallback or assisted prior.

**Experiment data.**

Source:

- `tables/direct_llm_reasoning_ablation_20260617_summary.md`
- `results/direct_llm_reasoning_ablation_20260617/`

**Experiment results.**

Key paper-facing observations:

- Direct GPT-4o, DeepSeek, and Gemini solve `canCallClass` well.
- Direct GPT-4o and DeepSeek reach `overridesMethod` train F1 = 0.980 but only
  0.220 held-out supported recall.
- DeepSeek direct `isAllowedToUse` exports no accepted rule.
- Qwen3.5-27B direct `overridesMethod` exports no accepted rule.
- Gemini fixed-I/O direct `overridesMethod` reaches 0.913 recall, below the
  final path-program recall of 1.000.

**Paper use.**

Secondary RQ5 evidence. It supports the claim that train-split rule
verification is not enough, but the primary query-level direct baseline is now
E10. Do not claim direct LLM is uniformly weak or that PARA dominates direct
LLM on every target; `canCallClass` is an honest boundary where direct
generation is strong.

### E9. Strict-agentic multi-model portability

**What it proves.**

The same strict path-program protocol can run across a local 27B model and
three API models under the same no-fallback execution contract.

**Experiment configuration.**

- Models: Qwen3.5-27B, GPT-4o, DeepSeek Chat, Gemini 2.5 Pro.
- Datasets: Spring xlarge3 and TEAMMATES clean9.
- Protocol: strict-agentic PARA, indexed path-program execution, no fallback,
  no direct-rule candidate channel, no assisted prior.
- Gemini valid protocol: thinking mode enabled, `thinkingBudget=1024`,
  sufficiently large output budget.

**Experiment data.**

Source:

- `tables/spring_xlarge3_final_portfolio_20260618_summary.md`
- `tables/teammates_clean9_api_agentic_portfolio_20260618_summary.md`
- `results/portfolio_spring_xlarge3_final_20260618/`
- `results/teammates_clean9_agentic_portfolio_api_20260618/`

**Experiment results.**

Spring xlarge3:

| Model group | OK | Mean learn F1 | Mean reason accuracy | Mean supported recall |
|---|---:|---:|---:|---:|
| each of 4 LLMs | 3/3 | 0.979 | 0.980 | 0.940 |

TEAMMATES clean9:

| Model group | Valid cases | Accepted cases | Mean test accuracy | Supported negatives |
|---|---:|---:|---:|---:|
| each of 4 LLMs | 9/9 | 6/9 | 0.889 | 0/486 |

**Paper use.**

Main text reports this in prose because aggregate rows are identical and would
make a low-density table. Full model rows are valid appendix material.

### E10. Matched direct LLM query answering

**What it proves.**

Direct query answering from a deterministic local graph view does not replace
the executable path-program contract. The query-level baseline can recognize
salient call and package-dependency relations, but it has very low proof
coverage on the inheritance-plus-signature composition in `overridesMethod`.

**Experiment configuration.**

- Dataset: Spring xlarge3 seed 0.
- Queries: 450 per target, 1,350 per model.
- Models with complete runs: Qwen3.5-27B and DeepSeek V4 Pro.
- Input: schema, one fixed positive and one fixed negative train
  demonstration, and deterministic query-centered evidence.
- No learned PARA rule, Planner action, held-out label, or fallback is shown to
  the direct model.
- Output: one JSON three-valued decision and evidence-ID list per query.

**Experiment data.**

Source:

- `docs/PARA_DIRECT_QA_AND_STRUCTURAL_NEAR_MISS_20260621.md`
- `artifacts/icse_revision_20260621/direct_query_answering_matched/`

**Experiment results.**

| Model | Target | Supported precision | Supported recall |
|---|---|---:|---:|
| Qwen3.5-27B | `canCallClass` | 1.000 | 0.720 |
| Qwen3.5-27B | `isAllowedToUse` | 0.924 | 0.807 |
| Qwen3.5-27B | `overridesMethod` | 1.000 | 0.007 |
| DeepSeek V4 Pro | `canCallClass` | 1.000 | 0.820 |
| DeepSeek V4 Pro | `isAllowedToUse` | 0.816 | 0.740 |
| DeepSeek V4 Pro | `overridesMethod` | 0.500 | 0.007 |

All 2,700 final full-run decisions are parseable and cite only valid evidence
IDs. Append-only raw logs retain recovered Qwen truncation and DeepSeek
provider/API failures from earlier attempts, but those invalid rows were
excluded and rerun by query ID. GPT-4o has a valid 90-query pilot, but its
incomplete full run is excluded.

**Paper use.**

Main RQ5 evidence and Figure 3. Strong query-level baseline. Compare methods
primarily with supported precision and recall. Three-valued accuracy is not
directly matched because Direct QA does not receive held-out labels, while the
evaluation harness uses explicit negative labels to separate UNSUPPORTED from
INCONCLUSIVE after proof search. Do not include the incomplete GPT-4o full run
in the main table; GPT-4o remains valid in the strict-agentic multi-model
suite.

### E11. Exact-signature structural near-miss stress

**What it proves.**

PARA does not falsely support explicit negatives that closely match positive
queries in endpoint types, shortest path length, and complete directed path
signature.

**Experiment configuration.**

- Dataset: Spring xlarge3 seed 0.
- Model/run: final Qwen3.5-27B strict-agentic `k=5`.
- Selection uses graph structure and explicit-negative membership, never the
  model decision.
- A selected negative shares an exact predicate-and-direction path signature
  with a positive query for the same target.

**Experiment data.**

Source:

- `artifacts/icse_revision_20260621/structural_near_miss/spring_qwen27b_k5_exact_signature.*`
- `scripts/analyze_structural_near_miss.py`

**Experiment results.**

| Target | Structural near misses | False SUPPORTED |
|---|---:|---:|
| `canCallClass` | 166 | 0 |
| `isAllowedToUse` | 133 | 0 |
| `overridesMethod` | 255 | 0 |
| Total | 554 | 0 |

**Paper use.**

Main RQ2 accountability stress evidence and Table/Figure 2 annotation. It
supersedes argument-sharing near-miss as the strongest negative-discrimination
result, while E7 remains a broader multi-model robustness check.

### E12. Bounded recursive IDB execution

**What it proves.**

The proof engine executes multiple IDB rules recursively, respects proof-depth
and cycle bounds, and emits nested proof traces. The experiment validates
supplied-rule execution, not autonomous recursive-rule induction.

**Experiment configuration.**

- Rules: direct inheritance base case, recursive transitive-inheritance step,
  and a derived wrapper predicate.
- Controlled graph: chains through six hops, branching, disconnected
  components, and a directed cycle.
- Spring graph: 4,769 direct `inheritsClass` edges.
- Independent ground truth: breadth-first transitive closure, separate from
  the PARA reasoner.
- Spring sample: 50 positives at each shortest-path distance 1--5 and 250
  fully disconnected negatives.

**Experiment data.**

Source:

- `artifacts/icse_revision_20260620/e9_recursive_idb/results.json`
- `artifacts/icse_revision_20260620/e9_recursive_idb/summary.md`
- `scripts/run_recursive_idb_evaluation.py`

**Experiment results.**

| Setting | Positive queries | Recall | Negative queries | False support | Max IDB applications |
|---|---:|---:|---:|---:|---:|
| Controlled, depth 2 | 34 | 0.412 | 80 | 0 | 2 |
| Controlled, depth 8 | 34 | 1.000 | 80 | 0 | 7 |
| Spring, depth 8 | 250 | 1.000 | 250 | 0 | 8 |

The Spring run records 505 recursive-rule applications. The controlled cyclic
graph records 16 cycle prunes and reaches full recall once the configured bound
covers the longest sampled chain. An automated audit checks all 250 Spring
recursive proof trees, 756 EDB fact mentions, and 1,006 IDB applications with
zero errors.

**Paper use.**

Main RQ2 support for bounded recursive proof execution. Phrase the boundary
positively: the current Planner induces non-recursive target rules, while
recursive program induction is the next extension.

### E13. False-supported proof diagnosis

**What it proves.**

Proof accountability makes a real semantic error inspectable and suggests a
specific rule-level repair without rewriting the reported result.

**Experiment configuration.**

- Spring `overridesMethod`, seed 1, matched GraphRAG rule library.
- Gold label: negative; reported decision: SUPPORTED.
- Materialize the full proof trace and replay a diagnostic rule that adds
  shared method arity.

**Experiment data.**

Source:

- `artifacts/icse_revision_20260620/e10_false_supported_case/false_supported_case.json`
- `artifacts/icse_revision_20260620/e10_false_supported_case/case_study.md`
- `scripts/materialize_false_supported_case.py`

**Experiment results.**

The accepted rule matches class containment, class inheritance, and method
name `OPTIONS`, but omits method arity. The two methods have incompatible
signatures. A post-hoc diagnostic replay with shared `methodArity` changes the
decision from SUPPORTED to INCONCLUSIVE. The original false support remains in
all main metrics.

**Paper use.**

Discussion case study. Use it to distinguish an invalid proof from a valid
proof of an over-general learned rule.

## Appendix / Reviewer-Response Evidence

### A1. Refiner and trace statistics

**What it proves.**

Refiner is a conservative repair hook and traceable boundary, not the primary
source of success in the final protocol.

**Experiment configuration.**

- Source: Spring strict-agentic Qwen3.5-27B `k=5` traces.
- Rows: 9 Spring target/seed runs.

**Experiment data.**

Source:

- `artifacts/icse_revision_20260620/e6_e7_runtime_trace/e6_e7_with_e3_summary.md`
- `tables/refiner_conservative_planning_examples_20260618_summary.md`

**Experiment results.**

| Trace property | Count |
|---|---:|
| Final accepted | 9/9 |
| Refiner triggered | 3/9 |
| Accepted-but-conservative trigger | 3/9 |
| Candidate improved after Refiner | 0/3 |

**Paper use.**

Discussion / appendix only. It should not be used to claim Refiner is the main
success mechanism.

### A2. Runtime decomposition and trace boundary

**What it proves.**

Indexed path-program execution is cheap, but local 27B planning can dominate
end-to-end time. This supports a scoped efficiency claim.

**Experiment configuration.**

- Spring xlarge3 strict-agentic Qwen3.5-27B and matched GraphRAG baseline.
- Runtime components recorded by E6/E7 instrumentation.

**Experiment data.**

Source:

- `artifacts/icse_revision_20260620/e6_e7_runtime_trace/e6_e7_with_e3_summary.*`

**Experiment results.**

| Method | Retrieval behavior | Retrieval time |
|---|---|---:|
| Deterministic GraphRAG | BFS-style candidate generation | 5.0--111.8 s |
| Strict-agentic PARA | indexed path-program execution | 0.002--0.467 s |

**Paper use.**

Main text as part of matched baseline table; detailed decomposition can go to
appendix. Do not claim local-27B PARA is always faster end-to-end.

### E14. Matched pure Popper held-out baseline

**Status.**

Completed for Spring xlarge3 seed 0. This is valid current evidence, but it is
reported as a matched seed-0 baseline rather than pooled with the three-seed
PARA aggregate.

**What it proves.**

Pure ILP is a meaningful comparator under the same train/test split and proof
engine. It can recover simple structural rules, but it faces search-time
limits on more complex large-graph targets under the full static bias.

**Experiment configuration.**

- Project: Spring Framework xlarge3.
- Seed: 0.
- Targets: `canCallClass`, `isAllowedToUse`, `overridesMethod`.
- Train split: same 50 positives / 100 negatives as PARA.
- Held-out split: same 150 positives / 300 negatives as PARA.
- Baseline: pure Popper, full static predicate bias, no LLM guidance, no path
  programs, no candidate-first rules, no fallback.
- Timeout: 300s per Popper run.
- Evaluation: accepted Popper rules are exported to the same bounded proof
  engine used by PARA.

**Experiment data.**

Source:

- `artifacts/icse_revision_20260621/e14_matched_popper_spring_seed0/`
- `scripts/run_matched_popper_baseline.py`

**Experiment results.**

| Target | Train status | Popper time | Held-out accuracy | Supported recall | Negative non-support |
|---|---|---:|---:|---:|---:|
| `canCallClass` | ok | 26.5s | 1.000 | 1.000 | 1.000 |
| `isAllowedToUse` | timeout | 300s | 0.667 | 0.000 | 1.000 |
| `overridesMethod` | timeout | 300s | 0.667 | 0.000 | 1.000 |

**Relationship to historical evidence.**

The historical 60-case TEAMMATES and 48-case Spring pure-Popper matrices remain
valid evidence about full-task ILP scalability. They do not replace this
matched experiment because they do not use the current frozen held-out splits.

**Paper use.**

Main-text supplementary baseline table. Do not claim PARA uniformly dominates
Popper: Popper exactly solves `canCallClass` on the matched seed-0 split. The
valid claim is narrower and stronger: pure ILP can solve simple structural
targets, while the strict path-program protocol provides a bounded,
proof-accountable discovery route that remains effective on the complex targets
where Popper times out under the same full-bias setting.

## E15. Project-Specific Spring Module-Policy Target

**What this experiment proves.**

This experiment addresses the motivation/evaluation alignment risk that the
main targets can look like generic Java structure relations.  It adds a
Spring-specific target, `springSameModuleUse/2`, whose labels depend on a
project-specific module taxonomy encoded as typed EDB facts
`springModuleName(Pkg, Module)`.

**Experimental configuration.**

- Project: Spring Framework graph from `ParsingProject/spring_nshrl_isAllowedToUse`.
- Target: `springSameModuleUse(PkgA, PkgB)`.
- Positive label: a class in `PkgA` imports a class in `PkgB`, and both
  packages share the same Spring module label.
- Negative labels:
  - 2175 cross-module import edges;
  - 20970 same-module package pairs without an import edge.
- Train split: 50 positive / 100 negative.
- Held-out split: 1176 positive / 23045 negative.
- Main protocol: strict-agentic DeepSeek, indexed-plan-only, no fallback,
  path-program portfolio size 5.
- Deterministic variant: GraphRAG-guided candidate generation with attribute
  constraints.
- Matched ILP boundary: pure Popper with the same target, examples, and BK.

**Experiment data.**

Source:

- `artifacts/icse_revision_20260623/project_specific_spring_policy_v2/`
- `scripts/run_project_specific_spring_policy.py`
- Main report:
  `artifacts/icse_revision_20260623/project_specific_spring_policy_v2/PROJECT_SPECIFIC_SPRING_POLICY_REPORT.md`

**Experiment results.**

PARA learns the same interpretable rule shape under the strict-agentic and
deterministic GraphRAG-guided protocols:

```prolog
springSameModuleUse(A,B) :-
    containsClass(A,V1),
    importsClass(V1,V2),
    containsClass(B,V2),
    springModuleName(A,C1),
    springModuleName(B,C1).
```

| Method | Train F1 | Held-out precision | Held-out recall | Held-out accuracy | False supported |
|---|---:|---:|---:|---:|---:|
| Strict-agentic PARA (DeepSeek) | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| Deterministic GraphRAG-guided PARA | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| No-rule policy | n/a | 0.000 | 0.000 | 0.951 | 0 |
| Popper selected single rule | 0.438 | n/a | n/a | n/a | n/a |
| Popper stdout full program | 0.810 | 0.264 | 0.643 | 0.895 | 2113 |

**Paper use.**

Use as supplemental evidence that PARA can handle a project-specific policy
relation whose meaning is not built into Java.  It should not replace the main
Spring xlarge3 result and should not be framed as raw-quality superiority over
deterministic GraphRAG, because GraphRAG-guided PARA reaches the same rule.
The matched Popper result is useful as an ILP boundary: the best stdout program
that Popper reported before timeout overgeneralizes on held-out queries.

## Evidence-to-Claim Summary

| Paper claim | Best evidence | Current manuscript location |
|---|---|---|
| Strict-agentic PARA performs held-out reasoning. | E1 Spring multi-seed + E3 TEAMMATES | Table III, Table V |
| PARA is competitive with a strong graph baseline while avoiding online BFS enumeration. | E2 matched GraphRAG | Table IV |
| Pure ILP is a meaningful but search-limited matched baseline. | E14 matched Popper | Table IV(c) and RQ1 prose |
| Portfolio is useful and is not fallback. | E4 top-k ablation + targeted rerun | Table I, Table VII |
| Proofs are accountable. | E5 proof audit + E6 counterfactual + E11 exact-signature near-miss | Table VI |
| Free-form LLM rules do not replace path programs. | E8 direct LLM rule-generation ablation | Discussion / appendix |
| Protocol is portable across LLMs. | E9 multi-model Spring / TEAMMATES | RQ4 prose; appendix candidate |
| Direct query answering does not replace executable path programs. | E10 matched Direct QA | Figure 3 / RQ5 |
| Proof gating rejects structurally matched negatives. | E11 exact-signature near miss | Table VI / RQ2 |
| The proof engine supports bounded recursive IDB execution. | E12 controlled + Spring recursion | Table VI / RQ2 |
| Proof traces make false support actionable. | E13 false-supported diagnosis | Discussion |
| PARA handles a project-specific Spring policy target. | E15 Spring module-policy target | Supplemental RQ / appendix candidate |

## Current Manuscript Table Assessment

| Table | Keep in main text? | Rationale |
|---|---|---|
| Portfolio vs fallback | Yes | Method boundary; now compact yes/no table. |
| Fixed parameters | Yes | Reproducibility. |
| Spring xlarge3 | Yes | Main large-project result. |
| Matched GraphRAG | Yes | Baseline and efficiency contrast. |
| TEAMMATES clean9 | Yes | Small-project conservative behavior. |
| Proof accountability | Yes | Main accountability evidence; should remain numeric. |
| Portfolio evidence | Yes | Main top-k mechanism evidence. |
| Full multi-model rows | Appendix/prose | Rows are identical, low density. |
| Historical deterministic full matrix | Appendix/context | Different protocol. |

## Non-Use / Exclusion List

Do not use the following as current main evidence:

- invalid infrastructure attempts;
- sandbox/network/auth failures;
- Gemini empty-output probes before the fixed thinking-mode protocol;
- diagnostic debug-repair outputs;
- assisted-prior outputs;
- deterministic fallback outputs;
- direct LLM rules as PARA candidate channels;
- zero-shot Direct QA runs without fixed train demonstrations;
- incomplete GPT-4o Direct QA full runs;
- early Direct QA transport, truncated-JSON, and incompatible DeepSeek-mode runs;
- historical full-matrix deterministic F1 as if it were held-out proof recall.
