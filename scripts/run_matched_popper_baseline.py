#!/usr/bin/env python3
"""Run a matched pure-Popper baseline on frozen PARA reasoning splits.

This script is intentionally narrow: it reuses the exact train/test task
directories already materialized for the PARA ICSE revision experiments, runs
the `baseline-popper` CLI on the train split, exports any accepted rule through
the same rule-library exporter, and evaluates held-out reasoning on the same
test split.

The baseline is "matched" in the sense that it uses the same BK, positive and
negative train examples, held-out queries, proof engine, and thresholds as the
PARA learn-then-reason protocol. It does not use LLM guidance, path programs,
candidate-first rules, direct-LLM rules, or deterministic fallback.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "src"
REVISION_ROOT = ROOT / "artifacts" / "icse_revision_20260620"


@dataclass(frozen=True)
class SplitCase:
    project: str
    target: str
    seed: int
    split_name: str
    split_dir: Path

    @property
    def train_task(self) -> Path:
        return self.split_dir / "train_task"

    @property
    def test_task(self) -> Path:
        return self.split_dir / "test_task"


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).resolve()
    code_root = Path(args.code_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cases = list(discover_cases(args))
    if args.target:
        wanted = {item.strip() for item in args.target.split(",") if item.strip()}
        cases = [case for case in cases if case.target in wanted]
    if args.seed:
        wanted_seeds = {int(item.strip()) for item in args.seed.split(",") if item.strip()}
        cases = [case for case in cases if case.seed in wanted_seeds]
    if args.max_cases:
        cases = cases[: args.max_cases]

    rows: List[Dict[str, Any]] = []
    for case in cases:
        row = run_case(case, output_root=output_root, code_root=code_root, python=args.python, args=args)
        rows.append(row)
        write_json(output_root / "results.json", {"cases": rows, "summary": summarize(rows)})

    payload = {
        "experiment": "matched_pure_popper_baseline",
        "case_count": len(rows),
        "cases": rows,
        "summary": summarize(rows),
    }
    write_json(output_root / "results.json", payload)
    write_summary_md(output_root / "summary.md", payload)
    print(json.dumps(payload if args.json else payload["summary"], indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Matched pure Popper baseline over frozen PARA splits")
    parser.add_argument("--output-root", default=str(ROOT / "results" / "matched_popper"))
    parser.add_argument("--code-root", default=str(CODE_ROOT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=900, help="Inner Popper timeout per case")
    parser.add_argument("--runner-timeout", type=int, default=1200, help="Outer timeout per CLI stage")
    parser.add_argument("--min-f1", type=float, default=0.8)
    parser.add_argument("--reason-max-proofs", type=int, default=3)
    parser.add_argument("--reason-max-depth", type=int, default=4)
    parser.add_argument("--reason-max-states", type=int, default=2000)
    parser.add_argument("--project", choices=["spring", "teammates", "both"], default="spring")
    parser.add_argument("--target", default="", help="Comma-separated target filter")
    parser.add_argument("--seed", default="", help="Comma-separated seed filter")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    return parser


def discover_cases(args: argparse.Namespace) -> Iterable[SplitCase]:
    if args.project in {"spring", "both"}:
        yield from spring_cases()
    if args.project in {"teammates", "both"}:
        yield from teammates_cases()


def spring_cases() -> Iterable[SplitCase]:
    # Seed 0 comes from the final k=5 portfolio run. Seeds 1/2 come from the
    # Qwen multiseed run. Together these are the current paper's Spring splits.
    seed0_root = REVISION_ROOT / "e2_portfolio_k/k5/splits"
    multi_root = REVISION_ROOT / "e3_qwen_multiseed/splits"
    targets = ["canCallClass", "isAllowedToUse", "overridesMethod"]
    for target in targets:
        split_name = f"spring_{target}_train50_100_seed0_agent_strict_qwen27b_k5"
        yield SplitCase("spring", target, 0, split_name, seed0_root / split_name)
    for seed in (1, 2):
        for target in targets:
            split_name = f"spring_{target}_train50_100_seed{seed}_agent_strict_qwen27b_k5"
            yield SplitCase("spring", target, seed, split_name, multi_root / split_name)


def teammates_cases() -> Iterable[SplitCase]:
    # These splits align with the current matched GraphRAG TEAMMATES clean9
    # context evidence; they are optional because the main ILP baseline risk is
    # the large Spring setting.
    split_root = REVISION_ROOT / "e1_matched_graphrag/teammates_clean9/splits"
    for split_dir in sorted(split_root.glob("teammates_*_clean_seed0_graphrag_support2")):
        name = split_dir.name
        parts = name.split("_")
        if len(parts) < 4:
            continue
        yield SplitCase("teammates", parts[1], 0, name, split_dir)


def run_case(
    case: SplitCase,
    *,
    output_root: Path,
    code_root: Path,
    python: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if not case.train_task.exists() or not case.test_task.exists():
        return {
            "project": case.project,
            "target": case.target,
            "seed": case.seed,
            "split_name": case.split_name,
            "status": "missing_split",
            "split_dir": str(case.split_dir),
        }

    learn_dir = output_root / "learn" / case.split_name
    library_path = output_root / "rule_libraries" / f"{case.split_name}_pure_popper_rule_library.json"
    reason_path = output_root / "reason_eval" / f"{case.split_name}_pure_popper_test_reason_eval.json"
    manifest_path = output_root / "manifests" / f"{case.split_name}_manifest.json"
    learn_dir.mkdir(parents=True, exist_ok=True)
    library_path.parent.mkdir(parents=True, exist_ok=True)
    reason_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    learn = run_cli(
        [
            python,
            "-m",
            "para.cli",
            "baseline-popper",
            "--task-dir",
            str(case.train_task),
            "--output-dir",
            str(learn_dir),
            "--timeout",
            str(args.timeout),
            "--min-f1",
            str(args.min_f1),
            "--json",
        ],
        cwd=code_root,
        timeout=args.runner_timeout,
    )
    summary = read_json_if_exists(learn_dir / "summary.json")

    export = None
    reason = None
    if summary.get("status") == "ok":
        export = run_cli(
            [
                python,
                "-m",
                "para.cli",
                "export-rule-library",
                "--summary",
                str(learn_dir),
                "--output",
                str(library_path),
                "--min-f1",
                str(args.min_f1),
                "--json",
            ],
            cwd=code_root,
            timeout=args.runner_timeout,
        )
        reason = run_cli(
            [
                python,
                "-m",
                "para.cli",
                "reason-eval",
                "--task-dir",
                str(case.test_task),
                "--rule-library",
                str(library_path),
                "--threshold",
                str(args.min_f1),
                "--max-depth",
                str(args.reason_max_depth),
                "--max-proofs",
                str(args.reason_max_proofs),
                "--max-states",
                str(args.reason_max_states),
                "--output",
                str(reason_path),
                "--json",
            ],
            cwd=code_root,
            timeout=args.runner_timeout,
        )
    else:
        write_empty_rule_library(library_path, args.min_f1)
        reason = run_cli(
            [
                python,
                "-m",
                "para.cli",
                "reason-eval",
                "--task-dir",
                str(case.test_task),
                "--rule-library",
                str(library_path),
                "--threshold",
                str(args.min_f1),
                "--max-depth",
                str(args.reason_max_depth),
                "--max-proofs",
                str(args.reason_max_proofs),
                "--max-states",
                str(args.reason_max_states),
                "--output",
                str(reason_path),
                "--json",
            ],
            cwd=code_root,
            timeout=args.runner_timeout,
        )

    reason_summary = read_json_if_exists(reason_path)
    elapsed = time.perf_counter() - started
    row: Dict[str, Any] = {
        "project": case.project,
        "target": case.target,
        "seed": case.seed,
        "split_name": case.split_name,
        "split_dir": str(case.split_dir),
        "status": summary.get("status", "missing_summary"),
        "learn_output": str(learn_dir),
        "rule_library": str(library_path),
        "reason_output": str(reason_path),
        "elapsed_seconds": elapsed,
        "learn": learn,
        "export": export,
        "reason": reason,
        "train_metrics": summary.get("metrics") or {},
        "final_rule": summary.get("final_rule"),
        "popper": summary.get("popper") or {},
        "held_out": summarize_reason(reason_summary),
    }
    write_json(manifest_path, row)
    return row


def run_cli(command: List[str], *, cwd: Path, timeout: int) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, timeout=timeout)
        elapsed = time.perf_counter() - started
        return {
            "returncode": proc.returncode,
            "elapsed_seconds": elapsed,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "timeout": False,
            "command": command,
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        return {
            "returncode": 124,
            "elapsed_seconds": elapsed,
            "stdout_tail": ensure_text(exc.stdout)[-4000:],
            "stderr_tail": ensure_text(exc.stderr)[-4000:],
            "timeout": True,
            "command": command,
        }


def summarize_reason(data: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "examples",
        "positive_examples",
        "negative_examples",
        "supported_precision",
        "supported_recall",
        "negative_non_support_rate",
        "inconclusive_rate",
        "held_out_accuracy",
        "false_supported",
        "true_supported",
        "positive_inconclusive",
    ]
    return {key: data.get(key) for key in keys if key in data}


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    completed = [row for row in rows if row.get("held_out")]
    accepted = [row for row in rows if row.get("status") == "ok"]
    return {
        "cases": len(rows),
        "completed_reason_eval": len(completed),
        "accepted_train_rules": len(accepted),
        "timed_out": sum(1 for row in rows if (row.get("popper") or {}).get("error") or row.get("learn", {}).get("timeout")),
        "by_target": {target: summarize_group([row for row in rows if row.get("target") == target]) for target in sorted({row.get("target") for row in rows})},
    }


def summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    held = [row.get("held_out") or {} for row in rows if row.get("held_out")]
    return {
        "cases": len(rows),
        "accepted_train_rules": sum(1 for row in rows if row.get("status") == "ok"),
        "held_out_accuracy_mean": mean([item.get("held_out_accuracy") for item in held]),
        "supported_recall_mean": mean([item.get("supported_recall") for item in held]),
        "negative_non_support_rate_mean": mean([item.get("negative_non_support_rate") for item in held]),
    }


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [float(value) for value in values if value is not None]
    return sum(nums) / len(nums) if nums else None


def write_empty_rule_library(path: Path, threshold: float) -> None:
    write_json(
        path,
        {
            "schema_version": 1,
            "threshold": threshold,
            "rule_count": 0,
            "rules": [],
        },
    )


def read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_summary_md(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# Matched Pure Popper Baseline",
        "",
        "This experiment runs pure Popper on frozen PARA train splits and evaluates",
        "any accepted rule with the same held-out bounded reasoner used by PARA.",
        "",
        "## Aggregate Summary",
        "",
        f"- Cases: {payload['summary'].get('cases')}",
        f"- Accepted train rules: {payload['summary'].get('accepted_train_rules')}",
        f"- Completed reason-eval: {payload['summary'].get('completed_reason_eval')}",
        "",
        "| Target | Cases | Accepted train rules | Held-out acc. | Supported recall | Negative non-support |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target, item in sorted((payload["summary"].get("by_target") or {}).items()):
        lines.append(
            "| {target} | {cases} | {accepted} | {acc} | {rec} | {neg} |".format(
                target=target,
                cases=item.get("cases"),
                accepted=item.get("accepted_train_rules"),
                acc=format_float(item.get("held_out_accuracy_mean")),
                rec=format_float(item.get("supported_recall_mean")),
                neg=format_float(item.get("negative_non_support_rate_mean")),
            )
        )
    lines.extend(["", "## Case Details", ""])
    lines.append("| Project | Target | Seed | Train status | Train F1 | Held-out acc. | Held-out recall | Popper time/error |")
    lines.append("|---|---|---:|---|---:|---:|---:|---|")
    for row in payload.get("cases") or []:
        popper = row.get("popper") or {}
        metrics = row.get("train_metrics") or {}
        held = row.get("held_out") or {}
        err = popper.get("error") or f"{format_float(popper.get('elapsed_seconds'))}s"
        lines.append(
            "| {project} | {target} | {seed} | {status} | {f1} | {acc} | {rec} | {err} |".format(
                project=row.get("project"),
                target=row.get("target"),
                seed=row.get("seed"),
                status=row.get("status"),
                f1=format_float(metrics.get("f1")),
                acc=format_float(held.get("held_out_accuracy")),
                rec=format_float(held.get("supported_recall")),
                err=str(err).replace("|", "/"),
            )
        )
    write_text(path, "\n".join(lines) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def format_float(value: Any) -> str:
    if value is None:
        return "--"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


if __name__ == "__main__":
    main()
