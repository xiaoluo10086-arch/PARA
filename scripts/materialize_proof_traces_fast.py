#!/usr/bin/env python3
"""Materialize auditable PARA proof traces with one shared reasoning engine.

This script is a faster companion to repeated `reason` CLI calls.  It preserves
the same proof construction path, but loads the task graph and rule library only
once before sampling labeled queries.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "src"


def main() -> None:
    args = build_parser().parse_args()
    code_root = Path(args.code_root).resolve()
    sys.path.insert(0, str(code_root))

    from para.prolog import load_task
    from para.reasoner import (
        BackwardChainingReasoner,
        example_label,
        load_rule_library,
        reasoning_result_from_proofs,
    )

    task = load_task(args.task_dir)
    rules = load_rule_library(args.rule_library)
    examples = [example for example in task.examples if example.positive]
    if args.include_negatives:
        examples = list(task.examples)
    if args.max_queries > 0:
        examples = examples[: args.max_queries]

    engine_build_started = time.perf_counter()
    engine = BackwardChainingReasoner(
        task=task,
        rules=rules,
        threshold=args.threshold,
        max_depth=args.max_depth,
        max_proofs=args.max_proofs,
        max_states=args.max_states,
        max_paths=args.max_paths,
        max_edges_per_node=args.max_edges_per_node,
        excluded_facts=set(),
        include_path_evidence=True,
    )
    engine_ready = time.perf_counter()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    for idx, example in enumerate(examples):
        started = time.perf_counter()
        proofs = engine.prove(example.literal)
        proof_done = time.perf_counter()
        result = reasoning_result_from_proofs(
            query_literal=example.literal,
            explicit_label=example_label(task.examples, example.literal),
            engine=engine,
            proofs=proofs,
            threshold=args.threshold,
            max_depth=args.max_depth,
            max_proofs=args.max_proofs,
            started=started,
            engine_ready=started,
            proof_done=proof_done,
            include_evidence=True,
        )
        proof_path = out_dir / f"proof_{idx:03d}.json"
        proof_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        rows.append(
            {
                "file": str(proof_path),
                "query": result.get("query"),
                "gold": "positive" if example.positive else "negative",
                "decision": result.get("decision"),
                "evidence_count": result.get("evidence_count"),
                "runtime_seconds": result.get("runtime_seconds"),
            }
        )

    summary = {
        "task_dir": str(Path(args.task_dir).resolve()),
        "rule_library": str(Path(args.rule_library).resolve()),
        "output_dir": str(out_dir.resolve()),
        "queries": len(rows),
        "shared_engine_build_seconds": engine_ready - engine_build_started,
        "total_runtime_seconds": (time.perf_counter() - engine_build_started),
        "rows": rows,
    }
    summary_path = out_dir / "materialized_proof_trace_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False) if args.json else compact(summary))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize proof traces with a shared reasoning engine")
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--rule-library", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--code-root", default=str(CODE_ROOT))
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--max-queries", type=int, default=10)
    parser.add_argument("--max-proofs", type=int, default=3)
    parser.add_argument("--max-paths", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-states", type=int, default=2000)
    parser.add_argument("--max-edges-per-node", type=int, default=120)
    parser.add_argument("--include-negatives", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def compact(summary: dict) -> str:
    lines = [
        f"Task: {summary['task_dir']}",
        f"Rule library: {summary['rule_library']}",
        f"Queries: {summary['queries']}",
        f"Shared engine build sec: {summary['shared_engine_build_seconds']:.3f}",
        f"Total sec: {summary['total_runtime_seconds']:.3f}",
    ]
    for row in summary.get("rows", []):
        lines.append(
            f"- {Path(row['file']).name}: {row['decision']} "
            f"evidence={row['evidence_count']} sec={float(row['runtime_seconds'] or 0.0):.4f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
