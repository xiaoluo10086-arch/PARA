#!/usr/bin/env python3
"""Run direct LLM baselines and summarize the results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "src"


def case_id(task_dir: Path) -> str:
    return "__".join(task_dir.parts[-3:])


def run_case(task_dir: Path, output_dir: Path, args: argparse.Namespace) -> Dict[str, object]:
    cmd = [
        sys.executable,
        "-m",
        "para.cli",
        "baseline-llm",
        "--task-dir",
        str(task_dir),
        "--output-dir",
        str(output_dir),
        "--llm-base-url",
        args.base_url,
        "--llm-model",
        args.model,
        "--llm-timeout",
        str(args.llm_timeout),
        "--llm-max-tokens",
        str(args.llm_max_tokens),
        "--fact-sample",
        str(args.fact_sample),
        "--example-sample",
        str(args.example_sample),
        "--max-rules",
        str(args.max_rules),
        "--min-f1",
        str(args.min_f1),
        "--json",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=CODE_ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=args.case_timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
    elapsed = time.perf_counter() - started
    if stdout:
        (output_dir / "runner.stdout.log").write_text(stdout, encoding="utf-8")
    if stderr:
        (output_dir / "runner.stderr.log").write_text(stderr, encoding="utf-8")
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    elif stdout.strip().startswith("{"):
        summary = json.loads(stdout)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        summary = {
            "status": "runner_timeout" if timed_out else "runner_failed",
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        }
    summary["task_dir"] = str(task_dir)
    summary["case_output_dir"] = str(output_dir)
    summary["case_elapsed_seconds"] = elapsed
    summary["runner_returncode"] = proc.returncode
    summary["runner_timed_out"] = timed_out
    summary["command"] = cmd
    return summary


def metrics_of(summary: Dict[str, object]) -> Dict[str, float]:
    metrics = summary.get("metrics") or summary.get("best_candidate_metrics") or {}
    return {
        "f1": float(metrics.get("f1", 0.0) or 0.0),
        "precision": float(metrics.get("precision", 0.0) or 0.0),
        "recall": float(metrics.get("recall", 0.0) or 0.0),
    }


def write_summary(rows: List[Dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "combined_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    flat = []
    for row in rows:
        task_dir = Path(str(row.get("task_dir", "")))
        metrics = metrics_of(row)
        flat.append(
            {
                "task": str(row.get("target", "")).split("/", 1)[0] or task_dir.parents[1].name,
                "complexity": task_dir.parent.name,
                "status": row.get("status"),
                **metrics,
                "elapsed_seconds": float(row.get("case_elapsed_seconds", 0.0) or 0.0),
                "rule": row.get("final_rule") or row.get("best_rule") or "",
                "output_dir": row.get("case_output_dir", ""),
            }
        )
    fields = ["task", "complexity", "status", "f1", "precision", "recall", "elapsed_seconds", "rule", "output_dir"]
    with (output_dir / "direct_llm_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    cases = len(flat)
    ok = sum(1 for row in flat if row["status"] == "ok")
    rejected = sum(1 for row in flat if row["status"] == "candidate_rejected")
    failed = sum(1 for row in flat if row["status"] not in {"ok", "candidate_rejected"})
    mean_f1 = sum(float(row["f1"]) for row in flat) / cases if cases else 0.0
    mean_time = sum(float(row["elapsed_seconds"]) for row in flat) / cases if cases else 0.0
    lines = [
        "# Direct LLM Suite Report",
        "",
        "| Cases | OK | Rejected | Failed | Mean F1 all | Mean wall time(s) |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {cases} | {ok} | {rejected} | {failed} | {mean_f1:.3f} | {mean_time:.2f} |",
        "",
        "## Per Case",
        "",
        "| Task | Complexity | Status | F1 | Seconds | Rule |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in flat:
        rule = str(row["rule"]).replace("|", "\\|")
        lines.append(f"| {row['task']} | {row['complexity']} | {row['status']} | {float(row['f1']):.3f} | {float(row['elapsed_seconds']):.2f} | `{rule}` |")
    (output_dir / "direct_llm_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--llm-timeout", type=int, default=180)
    parser.add_argument("--case-timeout", type=int, default=600)
    parser.add_argument("--llm-max-tokens", type=int, default=2048)
    parser.add_argument("--fact-sample", type=int, default=40)
    parser.add_argument("--example-sample", type=int, default=6)
    parser.add_argument("--max-rules", type=int, default=3)
    parser.add_argument("--min-f1", type=float, default=0.8)
    args = parser.parse_args()

    if not args.task_dir:
        parser.error("at least one --task-dir is required")
    task_dirs = [path.resolve() for path in args.task_dir]
    if args.max_cases:
        task_dirs = task_dirs[: args.max_cases]
    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task_dir in task_dirs:
        case_out = args.output_dir / "runs" / case_id(task_dir)
        row = run_case(task_dir, case_out, args)
        rows.append(row)
        metrics = metrics_of(row)
        print(f"{case_id(task_dir)}: {row.get('status')} f1={metrics['f1']:.3f}")
        write_summary(rows, args.output_dir)
    write_summary(rows, args.output_dir)
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
