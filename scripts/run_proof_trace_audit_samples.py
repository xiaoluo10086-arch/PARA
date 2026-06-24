#!/usr/bin/env python3
"""Materialize and audit sampled full proof traces."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "src"


def main() -> None:
    args = build_parser().parse_args()
    task_dir = Path(args.task_dir)
    rule_library = Path(args.rule_library)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    queries = positive_queries(task_dir / "exs.pl")[: args.max_queries]
    result_files = []
    for idx, query in enumerate(queries):
        output = output_dir / f"proof_{idx:03d}.json"
        cmd = [
            sys.executable,
            "-m",
            "para.cli",
            "reason",
            "--task-dir",
            str(task_dir),
            "--rule-library",
            str(rule_library),
            "--query",
            query,
            "--max-proofs",
            str(args.max_proofs),
            "--output",
            str(output),
            "--json",
        ]
        subprocess.run(cmd, cwd=Path(args.code_root), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        result_files.append(output)
    audit_output = output_dir / "proof_trace_audit.json"
    audit_cmd = [
        sys.executable,
        str(ROOT / "scripts/audit_proof_traces.py"),
        "--task-dir",
        str(task_dir),
        "--output",
        str(audit_output),
    ]
    for result in result_files:
        audit_cmd.extend(["--result", str(result)])
    subprocess.run(audit_cmd, check=True, text=True)
    print(f"Materialized proofs: {len(result_files)}")
    print(f"Audit: {audit_output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sampled proof trace audit")
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--rule-library", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-queries", type=int, default=10)
    parser.add_argument("--max-proofs", type=int, default=3)
    parser.add_argument("--code-root", default=str(CODE_ROOT))
    return parser


def positive_queries(path: Path) -> list[str]:
    queries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("pos("):
            continue
        inner = stripped[len("pos(") :].rstrip(".")
        if inner.endswith(")"):
            inner = inner[:-1]
        queries.append(inner)
    return queries


if __name__ == "__main__":
    main()
