"""Command line interface for PARA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .baselines import run_pure_llm_baseline, run_pure_popper_baseline
from .agentic_graph_rag import AgenticGraphRAGGuideProvider, RetrievalPlan
from .display import compact_result_text, task_inspection_text
from .experiments import (
    aggregate_results,
    build_blind_review_pack,
    build_experiment_plan,
    run_statistical_tests,
)
from .guidance import HeuristicGuideProvider, JsonGuideProvider, LlamaCppGuideProvider
from .graph_rag import GraphRAGCandidateGenerator, GraphRAGGuideProvider
from .ilp_backends import (
    DEFAULT_ALEPH_SOURCE,
    DEFAULT_ILASP_BIN,
    DEFAULT_METAGOL_SOURCE,
    run_pure_aleph_baseline,
    run_pure_ilasp_baseline,
    run_pure_metagol_baseline,
)
from .incremental import analyze_incremental_change, build_rule_registry, changed_files_from_git
from .modular_guidance import ModularGuidanceConfig, ModularLLMGuideProvider
from .noise import inject_noise
from .pipeline import DEFAULT_CASE_TIMEOUT_SECONDS, DEFAULT_POPPER_PATH, NSHRLPipeline
from .prolog import load_task
from .reasoner import (
    export_rule_library,
    reason_counterfactual,
    reason_counterfactual_examples,
    reason_examples,
    reason_query,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PARA: Path Accountable Reasoning for Agentic Rule-Learning")
    sub = parser.add_subparsers(dest="command", required=True)

    # inspect 用于快速确认一个 Popper 任务目录是否完整，
    # 也可用于实验前统计事实数、正负例数量和候选谓词数量。
    inspect = sub.add_parser("inspect", help="Inspect a Popper task directory")
    inspect.add_argument("--task-dir", required=True)
    inspect.add_argument("--json", action="store_true", help="Print raw JSON instead of a compact text view")

    # learn 是主流程：LLM/启发式引导 -> bias 剪枝 -> Popper -> 评估 -> 输出解释。
    learn = sub.add_parser("learn", help="Run PARA rule learning")
    learn.add_argument("--task-dir", required=True)
    learn.add_argument("--output-dir", required=True)
    learn.add_argument("--popper-path", default=DEFAULT_POPPER_PATH)
    learn.add_argument("--timeout", type=int, default=DEFAULT_CASE_TIMEOUT_SECONDS)
    learn.add_argument("--predicate-budget", type=int, default=5)
    learn.add_argument("--rounds", type=int, default=2)
    learn.add_argument("--objective", default="")
    learn.add_argument("--guide-json", default="", help="Optional JSON guidance file from an external LLM")
    learn.add_argument(
        "--guide-provider",
        choices=["heuristic", "json", "llama", "modular-llama", "graphrag", "agent"],
        default="heuristic",
        help="Guidance source: local heuristic, JSON file, llama.cpp server, modular llama.cpp pipeline, GraphRAG retrieval, or agentic GraphRAG",
    )
    learn.add_argument("--llm-base-url", default="", help="llama.cpp base URL, e.g. http://HOST:8000")
    learn.add_argument("--llm-model", default="", help="llama.cpp model alias, e.g. Qwen3.5-27B-MaxCtx")
    learn.add_argument("--llm-timeout", type=int, default=120, help="LLM request timeout in seconds")
    learn.add_argument("--min-f1", type=float, default=0.5, help="Minimum F1 required for status=ok")
    learn.add_argument(
        "--allow-heuristic-fallback",
        action="store_true",
        help="Debug only: allow llama/agent guidance to fall back to local symbolic planning when the LLM fails",
    )
    learn.add_argument(
        "--merge-heuristic-guardrail",
        action="store_true",
        help="Debug/ablation only: merge llama guidance with local heuristic predicates and candidates",
    )
    learn.add_argument(
        "--reuse-previous-guidance",
        action="store_true",
        help="Debug only: reuse previous valid llama guidance if a feedback round fails",
    )
    learn.add_argument(
        "--candidate-first",
        action="store_true",
        help="Accept a verified LLM/guide candidate before running Popper search",
    )
    learn.add_argument(
        "--strict-candidate-first",
        action="store_true",
        help="Reject the PARA case if candidate-first does not produce a rule above --min-f1; do not fall through to Popper",
    )
    learn.add_argument(
        "--continue-on-candidate-rejected",
        action="store_true",
        help=(
            "Ablation switch: in strict candidate-first mode, use rejected candidate feedback "
            "to run another guidance round instead of immediately returning candidate_rejected"
        ),
    )
    learn.add_argument(
        "--bk-slice-examples",
        action="store_true",
        help="Prune bk.pl to facts reachable from train example constants after predicate pruning",
    )
    learn.add_argument("--bk-slice-depth", type=int, default=1, help="Expansion depth for --bk-slice-examples")
    learn.add_argument(
        "--max-facts-per-predicate",
        type=int,
        default=None,
        help="Optional deterministic cap per selected predicate after BK slicing; 0 or omitted means no cap",
    )
    learn.add_argument(
        "--modular-predicate-selector",
        choices=["llama", "heuristic", "all"],
        default="llama",
        help="Ablation switch for modular-llama Step 1",
    )
    learn.add_argument(
        "--modular-type-inferencer",
        choices=["llama", "task", "none"],
        default="task",
        help="Ablation switch for modular-llama Step 2",
    )
    learn.add_argument(
        "--modular-candidate-generator",
        choices=["llama", "heuristic", "none"],
        default="llama",
        help="Ablation switch for modular-llama Step 3",
    )
    learn.add_argument(
        "--modular-feedback-interpreter",
        choices=["llama", "heuristic", "none"],
        default="llama",
        help="Ablation switch for modular-llama Step 4",
    )
    learn.add_argument(
        "--modular-cache-dir",
        default="",
        help="Optional cache directory for modular guidance step outputs",
    )
    learn.add_argument("--graphrag-max-depth", type=int, default=5)
    learn.add_argument("--graphrag-max-positive-examples", type=int, default=8)
    learn.add_argument("--graphrag-max-paths-per-example", type=int, default=12)
    learn.add_argument("--graphrag-max-edges-per-node", type=int, default=80)
    learn.add_argument("--graphrag-max-candidates", type=int, default=30)
    learn.add_argument("--graphrag-disable-pair-constraints", action="store_true")
    learn.add_argument("--graphrag-traversal-strategy", choices=["bfs", "dfs"], default="bfs")
    learn.add_argument("--graphrag-constraint-mode", choices=["direct", "attribute", "both", "none"], default="direct")
    learn.add_argument("--graphrag-min-path-support", type=int, default=1)
    learn.add_argument("--agent-iterations", type=int, default=3, help="Planner/Retriever/Verifier/Refiner cycles for --guide-provider agent")
    learn.add_argument("--agent-acceptance-f1", type=float, default=0.8, help="Stop agentic refinement when the verifier reaches this F1")
    learn.add_argument("--agent-portfolio-size", type=int, default=4, help="Number of complementary retrieval plans per agent iteration")
    learn.add_argument(
        "--agent-max-path-queries",
        type=int,
        default=5,
        help="Maximum executable path programs kept from one Planner/Refiner action; use 1 for top-1 path-program ablation",
    )
    learn.add_argument(
        "--agent-disable-focused-retrieval",
        action="store_true",
        help="Ablation: ignore agent required_predicates during graph path retrieval",
    )
    learn.add_argument(
        "--agent-disable-indexed-path-execution",
        action="store_true",
        help="Ablation: ignore Planner path_queries and use graph traversal retrieval only",
    )
    learn.add_argument(
        "--agent-deterministic-fallback",
        action="store_true",
        help="Hybrid mode: merge a strong deterministic GraphRAG fallback plan into agent retrieval",
    )
    learn.add_argument(
        "--agent-indexed-plan-only",
        action="store_true",
        help="Research mode: execute only Planner/Refiner ordered path_queries over the 1-hop relation index; disable BFS retrieval",
    )
    learn.add_argument(
        "--agent-compact-actions",
        action="store_true",
        help="Student-model mode: request compact typed path programs and omit retrieval-policy JSON",
    )
    learn.add_argument(
        "--agent-typed-path-student",
        action="store_true",
        help="Use the leakage-audited typed-path SFT contract for Planner/Refiner actions",
    )
    learn.add_argument(
        "--agent-force-initial-retrieval-miss",
        action="store_true",
        help="Dataset construction only: clear the first executable action so Refiner receives a symbolic no_candidate signal",
    )
    learn.add_argument(
        "--agent-disable-symbolic-prior",
        action="store_true",
        help="Strict evaluation: do not merge target-name heuristic actions into Planner or Refiner output",
    )
    learn.add_argument(
        "--agent-schema-profile-mode",
        choices=["assisted", "raw"],
        default="assisted",
        help="Use raw to expose only schema types and bounded graph evidence without target-specific path hints",
    )
    learn.add_argument(
        "--agent-witness-evidence-mode",
        choices=["full", "schema_only", "deterministic_top1"],
        default="full",
        help=(
            "Planner-evidence ablation: full exposes bounded train witness signatures; "
            "schema_only hides them from the LLM; deterministic_top1 executes the highest-ranked witness without an LLM"
        ),
    )
    learn.add_argument(
        "--agent-paper-strict",
        action="store_true",
        help=(
            "PARA paper experiment guard: require strict agent settings "
            "(indexed plan only, compact actions, raw schema profile, no symbolic prior, no fallbacks, no text salvage)."
        ),
    )
    learn.add_argument(
        "--agent-candidate-evaluation-cap",
        type=int,
        default=500,
        help="Maximum merged agent candidates scored by the symbolic verifier per iteration",
    )
    learn.add_argument(
        "--agent-focused-retrieval-fact-limit",
        type=int,
        default=100000,
        help="Disable heap-based focused traversal above this fact count while keeping agent action reranking",
    )
    learn.add_argument(
        "--agent-seed-max-depth",
        type=int,
        default=3,
        help="Run the first controlled retrieval at this depth before expanding on a miss; 0 disables seed-first retrieval",
    )
    learn.add_argument(
        "--agent-seed-max-paths-per-example",
        type=int,
        default=8,
        help="Maximum short seed paths retained per positive example before controlled expansion",
    )
    learn.add_argument(
        "--agent-disable-seed-expansion-on-miss",
        action="store_true",
        help="Ablation: keep retrieval seed-only even when no short evidence path exists",
    )
    learn.add_argument("--json", action="store_true", help="Print full JSON result instead of a compact summary")

    # baseline-popper 是纯 ILP 对照组：完整谓词空间 + Popper 搜索，不读取 LLM 输出。
    baseline_popper = sub.add_parser("baseline-popper", help="Run the pure Popper baseline")
    baseline_popper.add_argument("--task-dir", required=True)
    baseline_popper.add_argument("--output-dir", required=True)
    baseline_popper.add_argument("--popper-path", default=DEFAULT_POPPER_PATH)
    baseline_popper.add_argument("--timeout", type=int, default=DEFAULT_CASE_TIMEOUT_SECONDS)
    baseline_popper.add_argument("--min-f1", type=float, default=0.5)
    baseline_popper.add_argument("--max-vars", type=int, default=None)
    baseline_popper.add_argument("--max-body", type=int, default=None)
    baseline_popper.add_argument("--max-clauses", type=int, default=None)
    baseline_popper.add_argument("--json", action="store_true", help="Print full JSON result")

    baseline_aleph = sub.add_parser("baseline-aleph", help="Run the pure Aleph baseline")
    baseline_aleph.add_argument("--task-dir", required=True)
    baseline_aleph.add_argument("--output-dir", required=True)
    baseline_aleph.add_argument("--aleph-source", default=str(DEFAULT_ALEPH_SOURCE))
    baseline_aleph.add_argument("--timeout", type=int, default=DEFAULT_CASE_TIMEOUT_SECONDS)
    baseline_aleph.add_argument("--min-f1", type=float, default=0.5)
    baseline_aleph.add_argument("--json", action="store_true", help="Print full JSON result")

    baseline_ilasp = sub.add_parser("baseline-ilasp", help="Run the pure ILASP baseline")
    baseline_ilasp.add_argument("--task-dir", required=True)
    baseline_ilasp.add_argument("--output-dir", required=True)
    baseline_ilasp.add_argument("--ilasp-bin", default=str(DEFAULT_ILASP_BIN))
    baseline_ilasp.add_argument("--ilasp-version", default="4", choices=["1", "2", "2i", "3", "4"])
    baseline_ilasp.add_argument("--timeout", type=int, default=DEFAULT_CASE_TIMEOUT_SECONDS)
    baseline_ilasp.add_argument("--min-f1", type=float, default=0.5)
    baseline_ilasp.add_argument("--max-body-literals", type=int, default=None)
    baseline_ilasp.add_argument("--json", action="store_true", help="Print full JSON result")

    baseline_metagol = sub.add_parser("baseline-metagol", help="Run the pure Metagol baseline")
    baseline_metagol.add_argument("--task-dir", required=True)
    baseline_metagol.add_argument("--output-dir", required=True)
    baseline_metagol.add_argument("--metagol-source", default=str(DEFAULT_METAGOL_SOURCE))
    baseline_metagol.add_argument("--timeout", type=int, default=DEFAULT_CASE_TIMEOUT_SECONDS)
    baseline_metagol.add_argument("--min-f1", type=float, default=0.5)
    baseline_metagol.add_argument("--max-clauses", type=int, default=None)
    baseline_metagol.add_argument("--json", action="store_true", help="Print full JSON result")

    # baseline-llm 是纯 LLM 对照组：模型直接输出规则，不运行 Popper。
    # base-url/model 是显式参数，方便后续批量实验切换多个模型。
    baseline_llm = sub.add_parser("baseline-llm", help="Run the pure LLM baseline")
    baseline_llm.add_argument("--task-dir", required=True)
    baseline_llm.add_argument("--output-dir", required=True)
    baseline_llm.add_argument("--llm-base-url", default="http://127.0.0.1:8000")
    baseline_llm.add_argument("--llm-model", default="Qwen3.5-27B-MaxCtx")
    baseline_llm.add_argument("--llm-timeout", type=int, default=120)
    baseline_llm.add_argument("--llm-max-tokens", type=int, default=2048)
    baseline_llm.add_argument("--fact-sample", type=int, default=40)
    baseline_llm.add_argument("--example-sample", type=int, default=6)
    baseline_llm.add_argument("--objective", default="")
    baseline_llm.add_argument("--max-rules", type=int, default=3)
    baseline_llm.add_argument("--min-f1", type=float, default=0.5)
    baseline_llm.add_argument("--json", action="store_true", help="Print full JSON result")

    # register-rules 把已有 summary.json 转成“规则依赖图”的 JSON 原型。
    register = sub.add_parser("register-rules", help="Build a rule dependency registry")
    register.add_argument("--summary", action="append", required=True, help="summary.json path or an output directory")
    register.add_argument("--output", required=True, help="Output registry JSON path")
    register.add_argument("--json", action="store_true", help="Print full registry JSON")

    export_reasoning = sub.add_parser(
        "export-rule-library",
        help="Export accepted PARA summaries as a query-time reasoning rule library",
    )
    export_reasoning.add_argument("--summary", action="append", required=True, help="summary.json path or output directory; repeatable")
    export_reasoning.add_argument("--output", required=True, help="Output rule library JSON path")
    export_reasoning.add_argument("--min-f1", type=float, default=0.8, help="Minimum accepted rule F1 to export")
    export_reasoning.add_argument(
        "--factor-path-helpers",
        action="store_true",
        help="Reasoning experiment only: factor each accepted rule body into a synthetic IDB helper predicate",
    )
    export_reasoning.add_argument("--json", action="store_true", help="Print full library JSON")

    reason = sub.add_parser("reason", help="Apply learned executable rules to one architecture-relation query")
    reason.add_argument("--task-dir", required=True)
    reason.add_argument("--rule-library", required=True)
    reason.add_argument(
        "--query",
        required=True,
        help="Query as `predicate(A,B)`, `predicate A B`, or `A B` for the task target predicate",
    )
    reason.add_argument("--threshold", type=float, default=0.8)
    reason.add_argument("--max-paths", type=int, default=5)
    reason.add_argument("--max-edges-per-node", type=int, default=120)
    reason.add_argument("--max-depth", type=int, default=4, help="Maximum backward-chaining proof depth")
    reason.add_argument("--max-proofs", type=int, default=5, help="Maximum proof trees returned")
    reason.add_argument("--max-states", type=int, default=2000, help="Maximum intermediate proof states per body literal")
    reason.add_argument("--output", default="", help="Optional JSON output path")
    reason.add_argument("--json", action="store_true", help="Print full reasoning JSON")

    reason_eval = sub.add_parser(
        "reason-eval",
        help="Evaluate learned executable rules as a three-valued reasoning interface on task examples",
    )
    reason_eval.add_argument("--task-dir", required=True)
    reason_eval.add_argument("--rule-library", required=True)
    reason_eval.add_argument("--threshold", type=float, default=0.8)
    reason_eval.add_argument("--max-paths", type=int, default=3)
    reason_eval.add_argument("--max-edges-per-node", type=int, default=120)
    reason_eval.add_argument("--max-depth", type=int, default=4, help="Maximum backward-chaining proof depth")
    reason_eval.add_argument("--max-proofs", type=int, default=3, help="Maximum proof trees returned per example")
    reason_eval.add_argument("--max-states", type=int, default=2000, help="Maximum intermediate proof states per body literal")
    reason_eval.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Optional cap per class for quick checks; 0 evaluates all examples",
    )
    reason_eval.add_argument("--output", default="", help="Optional JSON output path")
    reason_eval.add_argument("--json", action="store_true", help="Print full reasoning evaluation JSON")

    reason_cf = sub.add_parser(
        "reason-counterfactual",
        help="Ablate EDB facts from a proof trace and re-run bounded reasoning",
    )
    reason_cf.add_argument("--task-dir", required=True)
    reason_cf.add_argument("--rule-library", required=True)
    reason_cf.add_argument(
        "--query",
        required=True,
        help="Query as `predicate(A,B)`, `predicate A B`, or `A B` for the task target predicate",
    )
    reason_cf.add_argument("--threshold", type=float, default=0.8)
    reason_cf.add_argument("--max-paths", type=int, default=5)
    reason_cf.add_argument("--max-edges-per-node", type=int, default=120)
    reason_cf.add_argument("--max-depth", type=int, default=4)
    reason_cf.add_argument("--max-proofs", type=int, default=5)
    reason_cf.add_argument("--max-states", type=int, default=2000)
    reason_cf.add_argument("--max-ablation-facts", type=int, default=8)
    reason_cf.add_argument("--output", default="", help="Optional JSON output path")
    reason_cf.add_argument("--json", action="store_true", help="Print full counterfactual reasoning JSON")

    reason_cf_eval = sub.add_parser(
        "reason-counterfactual-eval",
        help="Run counterfactual evidence reasoning over positive task examples",
    )
    reason_cf_eval.add_argument("--task-dir", required=True)
    reason_cf_eval.add_argument("--rule-library", required=True)
    reason_cf_eval.add_argument("--threshold", type=float, default=0.8)
    reason_cf_eval.add_argument("--max-paths", type=int, default=5)
    reason_cf_eval.add_argument("--max-edges-per-node", type=int, default=120)
    reason_cf_eval.add_argument("--max-depth", type=int, default=4)
    reason_cf_eval.add_argument("--max-proofs", type=int, default=5)
    reason_cf_eval.add_argument("--max-states", type=int, default=2000)
    reason_cf_eval.add_argument("--max-ablation-facts", type=int, default=8)
    reason_cf_eval.add_argument("--max-queries", type=int, default=20)
    reason_cf_eval.add_argument("--output", default="", help="Optional JSON output path")
    reason_cf_eval.add_argument("--json", action="store_true", help="Print full counterfactual evaluation JSON")

    # incremental-analyze 根据 Git/文件变更判断哪些规则可复用、哪些需要局部重学习。
    incremental = sub.add_parser("incremental-analyze", help="Analyze changed files against registered rule dependencies")
    incremental.add_argument("--registry", required=True)
    incremental.add_argument("--changed-file", action="append", default=[], help="Changed file path; repeatable")
    incremental.add_argument("--base", default="HEAD~1", help="Git base revision if --changed-file is omitted")
    incremental.add_argument("--head", default="HEAD", help="Git head revision if --changed-file is omitted")
    incremental.add_argument("--git-cwd", default=".", help="Git working directory")
    incremental.add_argument("--output-dir", default="", help="Optional directory for JSON/Markdown reports")
    incremental.add_argument("--json", action="store_true", help="Print full analysis JSON")

    # experiment-plan 生成论文实验矩阵和可直接执行的 shell 脚本。
    exp_plan = sub.add_parser("experiment-plan", help="Generate a batch experiment matrix")
    exp_plan.add_argument("--task", action="append", required=True, help="Task spec: name=path or plain path; repeatable")
    exp_plan.add_argument("--output-dir", required=True)
    exp_plan.add_argument("--predicate-counts", default="5,10,15,20")
    exp_plan.add_argument("--noise-kinds", default="clean,label_flip,missing_bk,irrelevant_bk")
    exp_plan.add_argument("--noise-rates", default="0,0.1,0.2,0.3")
    exp_plan.add_argument("--seeds", default="1,2,3")
    exp_plan.add_argument("--methods", default="pure_popper,pure_llm,nshrl")
    exp_plan.add_argument("--timeout", type=int, default=DEFAULT_CASE_TIMEOUT_SECONDS)
    exp_plan.add_argument("--predicate-budget", type=int, default=5)
    exp_plan.add_argument("--llm-base-url", default="http://127.0.0.1:8000")
    exp_plan.add_argument("--llm-model", default="Qwen3.5-27B-MaxCtx")
    exp_plan.add_argument("--llm-timeout", type=int, default=120)
    exp_plan.add_argument(
        "--no-materialize-tasks",
        action="store_true",
        help="Only write the plan, do not create predicate/noise task variants",
    )
    exp_plan.add_argument("--json", action="store_true", help="Print full plan JSON")

    aggregate = sub.add_parser("aggregate-results", help="Aggregate summary.json files according to an experiment plan")
    aggregate.add_argument("--plan", required=True)
    aggregate.add_argument("--output-dir", required=True)
    aggregate.add_argument("--json", action="store_true", help="Print aggregate JSON")

    stats = sub.add_parser("stats", help="Run paired statistical tests from aggregated results")
    stats.add_argument("--results", required=True, help="results.json produced by aggregate-results")
    stats.add_argument("--output-dir", required=True)
    stats.add_argument("--json", action="store_true", help="Print statistical test JSON")

    blind = sub.add_parser("blind-review", help="Create anonymized interpretability blind-review materials")
    blind.add_argument("--summary", action="append", required=True, help="summary.json path or output directory; repeatable")
    blind.add_argument("--output-dir", required=True)
    blind.add_argument("--seed", type=int, default=42)
    blind.add_argument("--json", action="store_true", help="Print pack JSON")

    # noise 用于构造系统性实验数据集，不会覆盖原始干净任务目录。
    noise = sub.add_parser("noise", help="Create a noisy Popper task copy")
    noise.add_argument("--task-dir", required=True)
    noise.add_argument("--output-dir", required=True)
    noise.add_argument("--kind", choices=["label_flip", "missing_bk", "irrelevant_bk"], required=True)
    noise.add_argument("--rate", type=float, required=True)
    noise.add_argument("--seed", type=int, default=42)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inspect":
        # 只读取任务，不运行 Popper，适合调试导出器和检查实验输入。
        task = load_task(args.task_dir)
        payload = {
            "task_dir": task.task_dir,
            "target": task.target.signature,
            "predicates": [spec.signature for spec in task.predicates.values()],
            "fact_count": len(task.facts),
            "positive_examples": sum(1 for ex in task.examples if ex.positive),
            "negative_examples": sum(1 for ex in task.examples if not ex.positive),
            "limits": {
                "max_vars": task.max_vars,
                "max_body": task.max_body,
                "max_clauses": task.max_clauses,
            },
        }
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(task_inspection_text(payload))
        return

    if args.command == "learn":
        validate_paper_strict_agent(args)
        # 三种引导来源：
        # - heuristic: 完全本地、可复现；
        # - json: 使用外部 LLM 已生成的 JSON；
        # - llama: 直接调用 llama.cpp 的 OpenAI 兼容接口。
        provider = build_provider(args)
        result = NSHRLPipeline(
            task_dir=args.task_dir,
            output_dir=args.output_dir,
            popper_path=args.popper_path,
            provider=provider,
            predicate_budget=args.predicate_budget,
            rounds=args.rounds,
            timeout=args.timeout,
            objective=args.objective,
            min_f1=args.min_f1,
            candidate_first=args.candidate_first,
            strict_candidate_first=args.strict_candidate_first,
            continue_on_candidate_rejected=args.continue_on_candidate_rejected,
            bk_slice_examples=args.bk_slice_examples,
            bk_slice_depth=args.bk_slice_depth,
            max_facts_per_predicate=args.max_facts_per_predicate,
        ).run()
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(compact_result_text(result))
        return

    if args.command == "baseline-popper":
        result = run_pure_popper_baseline(
            task_dir=args.task_dir,
            output_dir=args.output_dir,
            popper_path=args.popper_path,
            timeout=args.timeout,
            min_f1=args.min_f1,
            max_vars=args.max_vars,
            max_body=args.max_body,
            max_clauses=args.max_clauses,
        )
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(compact_result_text(result))
        return

    if args.command == "baseline-aleph":
        result = run_pure_aleph_baseline(
            task_dir=args.task_dir,
            output_dir=args.output_dir,
            aleph_source=args.aleph_source,
            timeout=args.timeout,
            min_f1=args.min_f1,
        )
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(compact_result_text(result))
        return

    if args.command == "baseline-ilasp":
        result = run_pure_ilasp_baseline(
            task_dir=args.task_dir,
            output_dir=args.output_dir,
            ilasp_bin=args.ilasp_bin,
            ilasp_version=args.ilasp_version,
            timeout=args.timeout,
            min_f1=args.min_f1,
            max_body_literals=args.max_body_literals,
        )
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(compact_result_text(result))
        return

    if args.command == "baseline-metagol":
        result = run_pure_metagol_baseline(
            task_dir=args.task_dir,
            output_dir=args.output_dir,
            metagol_source=args.metagol_source,
            timeout=args.timeout,
            min_f1=args.min_f1,
            max_clauses=args.max_clauses,
        )
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(compact_result_text(result))
        return

    if args.command == "baseline-llm":
        result = run_pure_llm_baseline(
            task_dir=args.task_dir,
            output_dir=args.output_dir,
            base_url=args.llm_base_url,
            model=args.llm_model,
            request_timeout=args.llm_timeout,
            min_f1=args.min_f1,
            objective=args.objective,
            max_rules=args.max_rules,
            max_tokens=args.llm_max_tokens,
            fact_sample_size=args.fact_sample,
            example_sample_size=args.example_sample,
        )
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(compact_result_text(result))
        return

    if args.command == "register-rules":
        registry = build_rule_registry(args.summary, args.output)
        if args.json:
            print(json.dumps(registry, indent=2, ensure_ascii=False))
        else:
            print(f"Registered rules: {len(registry.get('rules', []))}")
            print(f"Registry: {args.output}")
        return

    if args.command == "export-rule-library":
        library = export_rule_library(
            args.summary,
            args.output,
            min_f1=args.min_f1,
            factor_path_helpers=args.factor_path_helpers,
        )
        if args.json:
            print(json.dumps(library, indent=2, ensure_ascii=False))
        else:
            print(f"Exported reasoning rules: {library.get('rule_count')}")
            print(f"Rule library: {args.output}")
        return

    if args.command == "reason":
        result = reason_query(
            task_dir=args.task_dir,
            rule_library=args.rule_library,
            query=args.query,
            threshold=args.threshold,
            max_paths=args.max_paths,
            max_edges_per_node=args.max_edges_per_node,
            max_depth=args.max_depth,
            max_proofs=args.max_proofs,
            max_states=args.max_states,
        )
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"Query: {result.get('query')}")
            print(f"Decision: {result.get('decision')}")
            print(f"Reason: {result.get('reason')}")
            print(f"Evidence chains: {result.get('evidence_count')}")
        return

    if args.command == "reason-eval":
        result = reason_examples(
            task_dir=args.task_dir,
            rule_library=args.rule_library,
            threshold=args.threshold,
            max_paths=args.max_paths,
            max_edges_per_node=args.max_edges_per_node,
            max_examples=args.max_examples,
            max_depth=args.max_depth,
            max_proofs=args.max_proofs,
            max_states=args.max_states,
        )
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"Task: {result.get('target')}")
            print(f"Examples: {result.get('examples')}")
            print(f"Supported precision: {result.get('supported_precision'):.3f}")
            print(f"Supported recall: {result.get('supported_recall'):.3f}")
            print(f"Negative non-support rate: {result.get('negative_non_support_rate'):.3f}")
            print(f"Abstention rate: {result.get('inconclusive_rate'):.3f}")
            print(f"Held-out accuracy: {result.get('held_out_accuracy'):.3f}")
        return

    if args.command == "reason-counterfactual":
        result = reason_counterfactual(
            task_dir=args.task_dir,
            rule_library=args.rule_library,
            query=args.query,
            threshold=args.threshold,
            max_paths=args.max_paths,
            max_edges_per_node=args.max_edges_per_node,
            max_depth=args.max_depth,
            max_proofs=args.max_proofs,
            max_states=args.max_states,
            max_ablation_facts=args.max_ablation_facts,
        )
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            summary = result.get("summary") or {}
            print(f"Query: {result.get('query')}")
            print(f"Initial decision: {result.get('initial_decision')}")
            print(f"Ablated facts: {summary.get('ablated_facts')}")
            print(f"Critical facts: {summary.get('critical_facts')}")
            print(f"Alternative-supported facts: {summary.get('alternative_supported')}")
            all_removed = result.get("all_selected_facts_removed") or {}
            print(f"All selected removed: {all_removed.get('effect')}")
        return

    if args.command == "reason-counterfactual-eval":
        result = reason_counterfactual_examples(
            task_dir=args.task_dir,
            rule_library=args.rule_library,
            threshold=args.threshold,
            max_paths=args.max_paths,
            max_edges_per_node=args.max_edges_per_node,
            max_depth=args.max_depth,
            max_proofs=args.max_proofs,
            max_states=args.max_states,
            max_ablation_facts=args.max_ablation_facts,
            max_queries=args.max_queries,
        )
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"Task: {result.get('target')}")
            print(f"Queries: {result.get('queries')}")
            print(f"Initial supported: {result.get('initial_supported')}")
            print(f"Critical fact rate: {result.get('critical_fact_rate'):.3f}")
            print(f"Alternative fact rate: {result.get('alternative_fact_rate'):.3f}")
            print(f"All-selected alternative rate: {result.get('all_selected_alternative_rate'):.3f}")
        return

    if args.command == "incremental-analyze":
        changed_files = args.changed_file or changed_files_from_git(args.base, args.head, cwd=args.git_cwd)
        result = analyze_incremental_change(
            registry_path=args.registry,
            changed_files=changed_files,
            output_dir=args.output_dir or None,
        )
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            recommendation = result.get("recommendation", {})
            print(f"Incremental action: {recommendation.get('action')}")
            print(f"Impacted predicates: {', '.join(result.get('impacted_predicates', [])) or 'none'}")
            print(f"Reuse rules: {len(result.get('reuse_rules', []))}")
            print(f"Relearn rules: {len(result.get('relearn_rules', []))}")
            if args.output_dir:
                print(f"Report: {Path(args.output_dir) / 'incremental_report.md'}")
        return

    if args.command == "experiment-plan":
        result = build_experiment_plan(
            task_specs=args.task,
            output_dir=args.output_dir,
            predicate_counts=[int(x) for x in _csv_values(args.predicate_counts)],
            noise_kinds=_csv_values(args.noise_kinds),
            noise_rates=[float(x) for x in _csv_values(args.noise_rates)],
            seeds=[int(x) for x in _csv_values(args.seeds)],
            methods=_csv_values(args.methods),
            timeout=args.timeout,
            predicate_budget=args.predicate_budget,
            llm_base_url=args.llm_base_url,
            llm_model=args.llm_model,
            llm_timeout=args.llm_timeout,
            materialize_tasks=not args.no_materialize_tasks,
        )
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"Experiment cases: {result.get('case_count')}")
            print(f"Plan: {Path(args.output_dir) / 'experiment_plan.json'}")
            print(f"Run script: {Path(args.output_dir) / 'run_experiments.sh'}")
        return

    if args.command == "aggregate-results":
        result = aggregate_results(args.plan, args.output_dir)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"Rows: {len(result.get('rows', []))}")
            print(f"Results: {Path(args.output_dir) / 'results.csv'}")
        return

    if args.command == "stats":
        result = run_statistical_tests(args.results, args.output_dir)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"Comparisons: {len(result.get('comparisons', []))}")
            print(f"Report: {Path(args.output_dir) / 'statistical_tests.md'}")
        return

    if args.command == "blind-review":
        result = build_blind_review_pack(args.summary, args.output_dir, seed=args.seed)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"Blind review items: {result.get('item_count')}")
            outputs = result.get("outputs", {})
            print(f"Items: {outputs.get('items')}")
            print(f"Form: {outputs.get('form')}")
            print(f"Key: {outputs.get('key')}")
        return

    if args.command == "noise":
        # 噪声生成后仍是标准 Popper 任务目录，可直接作为 learn 的 --task-dir。
        out = inject_noise(args.task_dir, args.output_dir, args.kind, args.rate, seed=args.seed)
        print(json.dumps({"output_dir": str(Path(out))}, indent=2))


def build_provider(args: argparse.Namespace):
    """Create the selected guidance provider.

    这里单独拆出函数，是为了让 CLI 参数和 provider 构造逻辑保持清晰。
    后续如果增加 OpenAI、Ollama 或 vLLM provider，只需要扩展这里。
    """

    if args.guide_provider == "json" or args.guide_json:
        if not args.guide_json:
            raise SystemExit("--guide-provider json requires --guide-json")
        return JsonGuideProvider(args.guide_json)
    if args.guide_provider == "llama":
        return LlamaCppGuideProvider(
            base_url=args.llm_base_url or None,
            model=args.llm_model or None,
            request_timeout=args.llm_timeout,
            strict=not args.allow_heuristic_fallback,
            merge_heuristic_guardrail=args.merge_heuristic_guardrail,
            reuse_previous_on_feedback_failure=args.reuse_previous_guidance,
        )
    if args.guide_provider == "modular-llama":
        return ModularLLMGuideProvider(
            ModularGuidanceConfig(
                predicate_selector=args.modular_predicate_selector,
                type_inferencer=args.modular_type_inferencer,
                candidate_generator=args.modular_candidate_generator,
                feedback_interpreter=args.modular_feedback_interpreter,
                base_url=args.llm_base_url or "http://127.0.0.1:8000",
                model=args.llm_model or "Qwen3.5-27B-MaxCtx",
                request_timeout=args.llm_timeout,
                cache_dir=args.modular_cache_dir,
            )
        )
    if args.guide_provider == "graphrag":
        return GraphRAGGuideProvider(
            GraphRAGCandidateGenerator(
                max_depth=args.graphrag_max_depth,
                max_positive_examples=args.graphrag_max_positive_examples,
                max_paths_per_example=args.graphrag_max_paths_per_example,
                max_edges_per_node=args.graphrag_max_edges_per_node,
                max_candidates=args.graphrag_max_candidates,
                enable_pair_constraints=not args.graphrag_disable_pair_constraints,
                traversal_strategy=args.graphrag_traversal_strategy,
                constraint_mode=args.graphrag_constraint_mode,
                min_path_support=args.graphrag_min_path_support,
            )
        )
    if args.guide_provider == "agent":
        return AgenticGraphRAGGuideProvider(
            base_plan=RetrievalPlan(
                max_depth=args.graphrag_max_depth,
                max_positive_examples=args.graphrag_max_positive_examples,
                max_paths_per_example=args.graphrag_max_paths_per_example,
                max_edges_per_node=args.graphrag_max_edges_per_node,
                max_candidates=args.graphrag_max_candidates,
                enable_pair_constraints=not args.graphrag_disable_pair_constraints,
                traversal_strategy=args.graphrag_traversal_strategy,
                constraint_mode=args.graphrag_constraint_mode,
                min_path_support=args.graphrag_min_path_support,
                seed_max_depth=args.agent_seed_max_depth,
                seed_max_paths_per_example=args.agent_seed_max_paths_per_example,
                expand_seed_on_miss=not args.agent_disable_seed_expansion_on_miss,
            ),
            max_iterations=args.agent_iterations,
            acceptance_f1=args.agent_acceptance_f1,
            portfolio_size=args.agent_portfolio_size,
            base_url=args.llm_base_url or None,
            model=args.llm_model or None,
            request_timeout=args.llm_timeout,
            strict_llm=not args.allow_heuristic_fallback,
            use_focused_retrieval=not args.agent_disable_focused_retrieval,
            use_indexed_path_execution=not args.agent_disable_indexed_path_execution,
            deterministic_fallback=args.agent_deterministic_fallback,
            indexed_plan_only=args.agent_indexed_plan_only,
            compact_actions=args.agent_compact_actions,
            typed_path_student=args.agent_typed_path_student,
            force_initial_retrieval_miss=args.agent_force_initial_retrieval_miss,
            use_symbolic_prior=not args.agent_disable_symbolic_prior,
            schema_profile_mode=args.agent_schema_profile_mode,
            candidate_evaluation_cap=args.agent_candidate_evaluation_cap,
            focused_retrieval_fact_limit=args.agent_focused_retrieval_fact_limit,
            max_path_queries=args.agent_max_path_queries,
            witness_evidence_mode=args.agent_witness_evidence_mode,
        )
    return HeuristicGuideProvider()


def validate_paper_strict_agent(args: argparse.Namespace) -> None:
    """Fail fast when a paper-strict run accidentally enables assisted/debug behavior."""

    if not getattr(args, "agent_paper_strict", False):
        return
    errors = []
    if args.guide_provider != "agent":
        errors.append("--agent-paper-strict requires --guide-provider agent")
    if args.allow_heuristic_fallback:
        errors.append("--allow-heuristic-fallback is debug-only")
    if args.agent_deterministic_fallback:
        errors.append("--agent-deterministic-fallback is assisted deployment only")
    if not args.agent_indexed_plan_only:
        errors.append("--agent-indexed-plan-only is required")
    if not args.agent_compact_actions:
        errors.append("--agent-compact-actions is required")
    if not args.agent_disable_symbolic_prior:
        errors.append("--agent-disable-symbolic-prior is required")
    if args.agent_schema_profile_mode != "raw":
        errors.append("--agent-schema-profile-mode raw is required")
    if args.agent_portfolio_size != 1:
        errors.append("--agent-portfolio-size 1 is required")
    if os.getenv("NSHRL_AGENT_ENABLE_TEXT_SALVAGE", "0") == "1":
        errors.append("NSHRL_AGENT_ENABLE_TEXT_SALVAGE=1 is debug-only")
    if errors:
        raise SystemExit("Invalid --agent-paper-strict configuration:\n- " + "\n- ".join(errors))


def _csv_values(text: str) -> list[str]:
    """Parse comma-separated CLI values while ignoring empty cells."""

    return [item.strip() for item in text.split(",") if item.strip()]


if __name__ == "__main__":
    main()
