#!/usr/bin/env python3
"""Run TEAMMATES clean9 learn-then-reason splits."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = build_parser().parse_args()
    task_root = Path(args.task_root)
    output_root = Path(args.output_root)
    runner = ROOT / "scripts/run_learn_then_reason.py"
    rows = []
    for target in ("isAllowedToUse", "canCallClass", "overridesMethod"):
        for complexity in ("small", "mid", "xlarge"):
            task_dir = task_root / target / complexity / "clean"
            if not task_dir.exists():
                continue
            pos, neg = count_examples(task_dir / "exs.pl")
            if pos < 2 or neg < 2:
                rows.append((target, complexity, "skipped_too_few_examples"))
                continue
            train_pos = max(1, min(args.max_train_pos, pos // 2))
            train_neg = max(1, min(args.max_train_neg, neg // 2))
            split_name = f"teammates_{target}_{complexity}_clean_seed{args.seed}_{args.run_label}"
            cmd = [
                sys.executable,
                str(runner),
                "--task-dir",
                str(task_dir),
                "--output-root",
                str(output_root),
                "--split-name",
                split_name,
                "--train-pos",
                str(train_pos),
                "--train-neg",
                str(train_neg),
                "--seed",
                str(args.seed),
                "--guide-provider",
                args.guide_provider,
                "--timeout",
                str(args.timeout),
                "--runner-timeout",
                str(args.runner_timeout),
                "--min-f1",
                str(args.min_f1),
                "--predicate-budget",
                str(args.predicate_budget),
                "--graphrag-max-depth",
                str(args.graphrag_max_depth),
                "--graphrag-constraint-mode",
                args.graphrag_constraint_mode,
                "--graphrag-min-path-support",
                str(args.graphrag_min_path_support),
                "--graphrag-max-positive-examples",
                str(args.graphrag_max_positive_examples),
                "--graphrag-max-paths-per-example",
                str(args.graphrag_max_paths_per_example),
                "--graphrag-max-edges-per-node",
                str(args.graphrag_max_edges_per_node),
                "--graphrag-max-candidates",
                str(args.graphrag_max_candidates),
                "--reason-max-proofs",
                str(args.reason_max_proofs),
                "--run-no-rule",
            ]
            if args.guide_provider == "agent":
                cmd.extend(
                    [
                        "--agent-iterations",
                        str(args.agent_iterations),
                        "--agent-acceptance-f1",
                        str(args.agent_acceptance_f1),
                        "--agent-portfolio-size",
                        str(args.agent_portfolio_size),
                        "--agent-max-path-queries",
                        str(args.agent_max_path_queries),
                        "--agent-candidate-evaluation-cap",
                        str(args.agent_candidate_evaluation_cap),
                        "--agent-schema-profile-mode",
                        args.agent_schema_profile_mode,
                        "--llm-timeout",
                        str(args.llm_timeout),
                    ]
                )
                if args.llm_base_url:
                    cmd.extend(["--llm-base-url", args.llm_base_url])
                if args.llm_model:
                    cmd.extend(["--llm-model", args.llm_model])
                if args.agent_indexed_plan_only:
                    cmd.append("--agent-indexed-plan-only")
                if args.agent_compact_actions:
                    cmd.append("--agent-compact-actions")
                if args.agent_disable_symbolic_prior:
                    cmd.append("--agent-disable-symbolic-prior")
            if args.agent_paper_strict:
                cmd.append("--agent-paper-strict")
            if args.skip_llm_preflight:
                cmd.append("--skip-llm-preflight")
            if args.continue_after_guidance_failed:
                cmd.append("--continue-after-guidance-failed")
            if args.exit_nonzero_on_invalid:
                cmd.append("--exit-nonzero-on-invalid")
            print(f"RUN {split_name} train_pos={train_pos} train_neg={train_neg}", flush=True)
            proc = subprocess.run(cmd, text=True)
            manifest_path = output_root / "manifests" / f"{split_name}_manifest.json"
            validity = read_validity(manifest_path)
            status = f"returncode={proc.returncode}"
            if validity:
                status += f" valid_for_paper={not validity.get('invalid_for_paper', False)} category={validity.get('category')}"
            rows.append((target, complexity, status))
            invalid = bool(validity and validity.get("invalid_for_paper"))
            if invalid and args.stop_on_invalid:
                raise SystemExit(f"Stopping after invalid paper run: {split_name} category={validity.get('category')}")
            if proc.returncode != 0 and args.stop_on_failure:
                raise SystemExit(proc.returncode)
    print("SUMMARY")
    for row in rows:
        print("\t".join(row))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TEAMMATES clean9 learn-then-reason")
    parser.add_argument(
        "--task-root",
        required=True,
        help="TEAMMATES task root organized as TARGET/COMPLEXITY/clean",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label", default="graphrag")
    parser.add_argument("--guide-provider", choices=["graphrag", "agent", "heuristic"], default="graphrag")
    parser.add_argument("--max-train-pos", type=int, default=50)
    parser.add_argument("--max-train-neg", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--runner-timeout", type=int, default=600)
    parser.add_argument("--min-f1", type=float, default=0.8)
    parser.add_argument("--predicate-budget", type=int, default=8)
    parser.add_argument("--graphrag-max-depth", type=int, default=5)
    parser.add_argument("--graphrag-constraint-mode", choices=["direct", "attribute", "both", "none"], default="attribute")
    parser.add_argument("--graphrag-min-path-support", type=int, default=1)
    parser.add_argument("--graphrag-max-positive-examples", type=int, default=8)
    parser.add_argument("--graphrag-max-paths-per-example", type=int, default=12)
    parser.add_argument("--graphrag-max-edges-per-node", type=int, default=80)
    parser.add_argument("--graphrag-max-candidates", type=int, default=30)
    parser.add_argument("--agent-iterations", type=int, default=2)
    parser.add_argument("--agent-acceptance-f1", type=float, default=0.8)
    parser.add_argument("--agent-portfolio-size", type=int, default=1)
    parser.add_argument("--agent-max-path-queries", type=int, default=5)
    parser.add_argument("--agent-candidate-evaluation-cap", type=int, default=40)
    parser.add_argument("--agent-indexed-plan-only", action="store_true")
    parser.add_argument("--agent-compact-actions", action="store_true")
    parser.add_argument("--agent-disable-symbolic-prior", action="store_true")
    parser.add_argument("--agent-schema-profile-mode", choices=["assisted", "raw"], default="raw")
    parser.add_argument("--agent-paper-strict", action="store_true")
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--llm-timeout", type=int, default=120)
    parser.add_argument("--reason-max-proofs", type=int, default=5)
    parser.add_argument("--skip-llm-preflight", action="store_true")
    parser.add_argument("--continue-after-guidance-failed", action="store_true")
    parser.add_argument("--exit-nonzero-on-invalid", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument(
        "--stop-on-invalid",
        action="store_true",
        default=True,
        help="Stop the suite when a split is marked invalid for paper reporting",
    )
    parser.add_argument(
        "--continue-on-invalid",
        dest="stop_on_invalid",
        action="store_false",
        help="Compatibility mode: keep running even after invalid infrastructure/model-output failures",
    )
    return parser


def count_examples(path: Path) -> tuple[int, int]:
    pos = 0
    neg = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("pos("):
            pos += 1
        elif stripped.startswith("neg("):
            neg += 1
    return pos, neg


def read_validity(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    validity = data.get("validity")
    return validity if isinstance(validity, dict) else None


if __name__ == "__main__":
    main()
