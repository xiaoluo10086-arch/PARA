#!/usr/bin/env python3
"""Run direct-LLM rule generation followed by held-out reasoning.

For each Spring xlarge3 seed-0 split, this script asks an LLM to generate rules
from the train task, exports accepted direct rules into a run-local reasoning
library, and evaluates the library on the held-out test task.  It is intended as
the direct-rule counterpart to learn-then-reason and strict-agentic reasoning.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "src"
DIRECT_SUITE = ROOT / "scripts" / "run_direct_llm_suite.py"
DEFAULT_SPLIT_ROOT = ROOT / "artifacts" / "learn_then_reason_spring" / "splits"


TARGETS = ("canCallClass", "isAllowedToUse", "overridesMethod")


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).resolve()
    direct_root = output_root / "direct_generation"
    library_root = output_root / "rule_libraries"
    reason_root = output_root / "reason_eval"
    manifest_root = output_root / "manifests"
    for path in (direct_root, library_root, reason_root, manifest_root):
        path.mkdir(parents=True, exist_ok=True)

    train_tasks = [split_task(args.split_root, target, args.seed, "train_task") for target in TARGETS]
    test_tasks = {target: split_task(args.split_root, target, args.seed, "test_task") for target in TARGETS}

    direct_cmd = [
        sys.executable,
        str(DIRECT_SUITE),
        "--output-dir",
        str(direct_root),
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--llm-timeout",
        str(args.llm_timeout),
        "--case-timeout",
        str(args.case_timeout),
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
    ]
    for task in train_tasks:
        direct_cmd.extend(["--task-dir", str(task)])
    direct_summary = run_command(direct_cmd, cwd=Path.cwd(), timeout=args.suite_timeout)

    rows: List[Dict[str, object]] = []
    for target in TARGETS:
        split_name = split_name_for(target, args.seed)
        train_task = split_task(args.split_root, target, args.seed, "train_task")
        test_task = test_tasks[target]
        summary_dir = direct_case_dir(direct_root, split_name)
        rule_library = library_root / f"{split_name}_{args.model_tag}_direct_rule_library.json"
        reason_output = reason_root / f"{split_name}_{args.model_tag}_direct_reason_eval.json"
        no_rule_output = reason_root / f"{split_name}_{args.model_tag}_empty_reason_eval.json"

        export_summary = export_rule_library(args.python, args.code_root, summary_dir, rule_library, args.min_f1)
        reason_summary = run_reason_eval(args.python, args.code_root, test_task, rule_library, reason_output)
        no_rule_summary = None
        if args.run_no_rule:
            empty_library = library_root / f"{split_name}_{args.model_tag}_empty_rule_library.json"
            write_empty_rule_library(empty_library, args.min_f1)
            no_rule_summary = run_reason_eval(args.python, args.code_root, test_task, empty_library, no_rule_output)
        direct_case_summary = load_json(summary_dir / "summary.json")
        row = {
            "target": target,
            "split_name": split_name,
            "train_task": str(train_task),
            "test_task": str(test_task),
            "direct_summary": direct_case_summary,
            "rule_library": str(rule_library),
            "reason_output": str(reason_output),
            "direct_status": direct_case_summary.get("status"),
            "direct_f1": metric_value(direct_case_summary, "f1"),
            "export_rule_count": (export_summary.get("json") or {}).get("rule_count"),
            "reason_accuracy": (reason_summary.get("json") or {}).get("three_value_accuracy"),
            "reason_precision": (reason_summary.get("json") or {}).get("supported_precision"),
            "reason_recall": (reason_summary.get("json") or {}).get("supported_recall"),
            "reason_inconclusive": (reason_summary.get("json") or {}).get("inconclusive_rate"),
            "no_rule_accuracy": ((no_rule_summary or {}).get("json") or {}).get("three_value_accuracy") if no_rule_summary else None,
        }
        rows.append(row)

    manifest = {
        "protocol": "direct_llm_train_split_then_reason",
        "model_tag": args.model_tag,
        "model": args.model,
        "base_url": args.base_url,
        "seed": args.seed,
        "direct_generation_root": str(direct_root),
        "direct_suite": direct_summary,
        "rows": rows,
        "boundary": (
            "Direct LLM sees only train_task examples/facts. Held-out test_task "
            "examples are used only by reason-eval after rule export."
        ),
    }
    manifest_path = manifest_root / f"{args.model_tag}_direct_reasoning_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(output_root / f"{args.model_tag}_direct_reasoning_report.md", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False) if args.json else compact(manifest))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direct LLM train-split reasoning ablation")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-root", default=str(DEFAULT_SPLIT_ROOT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--code-root", default=str(CODE_ROOT))
    parser.add_argument("--llm-timeout", type=int, default=180)
    parser.add_argument("--case-timeout", type=int, default=600)
    parser.add_argument("--suite-timeout", type=int, default=2400)
    parser.add_argument("--llm-max-tokens", type=int, default=2048)
    parser.add_argument("--fact-sample", type=int, default=40)
    parser.add_argument("--example-sample", type=int, default=6)
    parser.add_argument("--max-rules", type=int, default=3)
    parser.add_argument("--min-f1", type=float, default=0.8)
    parser.add_argument("--run-no-rule", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def split_name_for(target: str, seed: int) -> str:
    return f"spring_{target}_train50_100_seed{seed}_graphrag"


def split_task(split_root: str | Path, target: str, seed: int, name: str) -> Path:
    path = Path(split_root) / split_name_for(target, seed) / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path.resolve()


def direct_case_dir(direct_root: Path, split_name: str) -> Path:
    path = direct_root / "runs" / f"splits__{split_name}__train_task"
    if not (path / "summary.json").exists():
        # Older direct-suite case IDs may omit the leading `splits` component.
        matches = sorted((direct_root / "runs").glob(f"*{split_name}*train_task*"))
        if matches:
            return matches[0]
    return path


def run_command(cmd: Sequence[str], cwd: Path, timeout: int) -> Dict[str, object]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=os.environ.copy(),
        )
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        return {
            "command": list(cmd),
            "returncode": -9,
            "timed_out": True,
            "elapsed_seconds": time.perf_counter() - started,
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
            "json": None,
        }
    return {
        "command": list(cmd),
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "json": parse_last_json(proc.stdout),
    }


def export_rule_library(python: str, code_root: str, summary_dir: Path, output: Path, min_f1: float) -> Dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    return run_command(
        [
            python,
            "-m",
            "para.cli",
            "export-rule-library",
            "--summary",
            str(summary_dir),
            "--output",
            str(output),
            "--min-f1",
            str(min_f1),
            "--json",
        ],
        cwd=Path(code_root),
        timeout=300,
    )


def run_reason_eval(python: str, code_root: str, task_dir: Path, rule_library: Path, output: Path) -> Dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    return run_command(
        [
            python,
            "-m",
            "para.cli",
            "reason-eval",
            "--task-dir",
            str(task_dir),
            "--rule-library",
            str(rule_library),
            "--output",
            str(output),
            "--json",
        ],
        cwd=Path(code_root),
        timeout=300,
    )


def write_empty_rule_library(path: Path, min_f1: float) -> None:
    payload = {
        "schema_version": 2,
        "library_kind": "ashrl_reasoning_rule_library",
        "min_f1": min_f1,
        "rule_count": 0,
        "dependency_graph": {
            "idb_predicates": [],
            "edb_predicates": [],
            "dependencies": {},
            "recursive_predicates": [],
        },
        "rules": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"status": "missing_summary", "summary_path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_last_json(text: str) -> object | None:
    stripped = text.strip()
    if not stripped:
        return None
    idx = stripped.find("{")
    if idx < 0:
        return None
    try:
        return json.loads(stripped[idx:])
    except json.JSONDecodeError:
        return None


def metric_value(summary: Dict[str, object], key: str) -> float:
    metrics = summary.get("metrics") or summary.get("best_candidate_metrics") or {}
    try:
        return float(metrics.get(key, 0.0) or 0.0)  # type: ignore[union-attr]
    except AttributeError:
        return 0.0


def write_report(path: Path, manifest: Dict[str, object]) -> None:
    rows = manifest.get("rows") or []
    lines = [
        f"# Direct LLM Reasoning Ablation: {manifest.get('model_tag')}",
        "",
        f"Model: `{manifest.get('model')}`",
        "",
        "| Target | Direct status | Direct F1 | Exported rules | Reason accuracy | Reason precision | Reason recall | Inconclusive | No-rule accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {target} | {direct_status} | {direct_f1:.3f} | {export_rule_count} | "
            "{reason_accuracy:.3f} | {reason_precision:.3f} | {reason_recall:.3f} | "
            "{reason_inconclusive:.3f} | {no_rule_accuracy:.3f} |".format(
                target=row.get("target"),
                direct_status=row.get("direct_status"),
                direct_f1=float(row.get("direct_f1") or 0.0),
                export_rule_count=row.get("export_rule_count"),
                reason_accuracy=float(row.get("reason_accuracy") or 0.0),
                reason_precision=float(row.get("reason_precision") or 0.0),
                reason_recall=float(row.get("reason_recall") or 0.0),
                reason_inconclusive=float(row.get("reason_inconclusive") or 0.0),
                no_rule_accuracy=float(row.get("no_rule_accuracy") or 0.0),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact(manifest: Dict[str, object]) -> str:
    lines = [
        f"Protocol: {manifest.get('protocol')}",
        f"Model: {manifest.get('model_tag')} ({manifest.get('model')})",
    ]
    for row in manifest.get("rows") or []:
        lines.append(
            f"- {row.get('target')}: direct={row.get('direct_status')} "
            f"f1={float(row.get('direct_f1') or 0.0):.3f} "
            f"rules={row.get('export_rule_count')} "
            f"reason_acc={float(row.get('reason_accuracy') or 0.0):.3f} "
            f"recall={float(row.get('reason_recall') or 0.0):.3f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
