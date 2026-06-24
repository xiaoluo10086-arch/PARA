#!/usr/bin/env python3
"""Materialize a predeclared stratified audit sample from an agent rule library."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CODE_ROOT = ROOT / "src"


def main() -> None:
    args = build_parser().parse_args()
    code_root = Path(args.code_root).resolve()
    sys.path.insert(0, str(code_root))

    from para.prolog import load_task
    from para.reasoner import (
        BackwardChainingReasoner,
        INCONCLUSIVE,
        SUPPORTED,
        example_label,
        load_rule_library,
        reasoning_result_from_proofs,
    )

    task = load_task(args.task_dir)
    rules = load_rule_library(args.rule_library)
    examples = list(task.examples)
    random.Random(args.seed).shuffle(examples)
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
        include_path_evidence=False,
    )

    selected = {SUPPORTED: [], INCONCLUSIVE: []}
    scanned = 0
    for example in examples:
        scanned += 1
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
        decision = result["decision"]
        quota = target_quota(decision, args.supported, args.inconclusive)
        if decision in selected and len(selected[decision]) < quota:
            selected[decision].append(result)
        if len(selected[SUPPORTED]) >= args.supported and len(selected[INCONCLUSIVE]) >= args.inconclusive:
            break

    target_total = args.supported + args.inconclusive
    chosen = selected[SUPPORTED][: args.supported] + selected[INCONCLUSIVE][: args.inconclusive]
    if len(chosen) < target_total:
        # A fully covered target may expose no INCONCLUSIVE decisions. Fill
        # the predeclared target size with additional SUPPORTED cases while
        # retaining the achieved judgement distribution in the summary.
        chosen.extend(selected[SUPPORTED][args.supported : args.supported + target_total - len(chosen)])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_files = []
    for idx, result in enumerate(chosen):
        path = output_dir / f"proof_{idx:03d}_{str(result['decision']).lower()}.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        result_files.append(path)

    audit_path = output_dir / "proof_trace_audit.json"
    audit_cmd = [
        sys.executable,
        str(ROOT / "scripts/audit_proof_traces.py"),
        "--task-dir",
        str(Path(args.task_dir).resolve()),
        "--code-root",
        str(code_root),
        "--output",
        str(audit_path),
    ]
    for path in result_files:
        audit_cmd.extend(["--result", str(path)])
    audit = subprocess.run(audit_cmd, text=True, capture_output=True)

    summary = {
        "task_dir": str(Path(args.task_dir).resolve()),
        "rule_library": str(Path(args.rule_library).resolve()),
        "seed": args.seed,
        "predeclared_quota": {
            "SUPPORTED": args.supported,
            "INCONCLUSIVE": args.inconclusive,
        },
        "achieved_distribution": {
            "SUPPORTED": sum(1 for item in chosen if item["decision"] == SUPPORTED),
            "INCONCLUSIVE": sum(1 for item in chosen if item["decision"] == INCONCLUSIVE),
        },
        "examples_scanned": scanned,
        "sample_size": len(chosen),
        "audit_returncode": audit.returncode,
        "audit_output": str(audit_path),
        "files": [str(path) for path in result_files],
    }
    summary_path = output_dir / "selection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if audit.returncode != 0:
        print(audit.stdout, file=sys.stderr)
        print(audit.stderr, file=sys.stderr)
        raise SystemExit(audit.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a stratified agent-origin proof audit")
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--rule-library", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--code-root", default=str(DEFAULT_CODE_ROOT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--supported", type=int, default=5)
    parser.add_argument("--inconclusive", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--max-proofs", type=int, default=5)
    parser.add_argument("--max-paths", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-states", type=int, default=2000)
    parser.add_argument("--max-edges-per-node", type=int, default=120)
    return parser


def target_quota(decision: str, supported: int, inconclusive: int) -> int:
    if decision == "SUPPORTED":
        return supported + inconclusive
    if decision == "INCONCLUSIVE":
        return inconclusive
    return 0


if __name__ == "__main__":
    main()
