#!/usr/bin/env python3
"""Build E6 runtime and E7 repair/trace summaries from frozen manifests."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    args = build_parser().parse_args()
    manifest_paths = []
    for root in args.root:
        manifest_paths.extend(
            path
            for path in sorted(Path(root).rglob("*_manifest.json"))
            if path.parent.name == "manifests"
        )
    rows = []
    trace_rows = []
    for path in manifest_paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        summary = read_json(Path(manifest.get("learn_output") or "") / "summary.json")
        rows.append(runtime_row(path, manifest, summary))
        trace = extract_trace(summary)
        if trace:
            trace_rows.append(trace_row(path, manifest, summary, trace))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_count": len(manifest_paths),
        "runtime_rows": rows,
        "trace_rows": trace_rows,
        "trace_aggregate": aggregate_traces(trace_rows),
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(output.with_suffix(".md"), payload)
    print(f"Runtime rows: {len(rows)}")
    print(f"Trace rows: {len(trace_rows)}")
    print(f"Wrote {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize PARA runtime and trace statistics")
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser


def runtime_row(path: Path, manifest: dict, summary: dict) -> dict:
    command = ((manifest.get("learn_summary") or {}).get("command") or [])
    trace = extract_trace(summary)
    iteration = ((trace.get("iterations") or [{}])[0]) if trace else {}
    diagnostics = iteration.get("diagnostics") or {}
    plan_diagnostics = {}
    portfolio = diagnostics.get("portfolio") or []
    if portfolio:
        plan_diagnostics = (portfolio[0] or {}).get("diagnostics") or {}
    if not plan_diagnostics:
        plan_diagnostics = extract_deterministic_diagnostics(summary)
    timings = plan_diagnostics.get("timings_seconds") or {}
    reason = ((manifest.get("reason_summary") or {}).get("json") or {})
    guide = command_arg(command, "--guide-provider") or "-"
    return {
        "split_name": manifest.get("split_name"),
        "target": target_from_split(str(manifest.get("split_name") or "")),
        "seed": (manifest.get("split") or {}).get("seed"),
        "method": "strict-agentic" if guide == "agent" else guide,
        "max_path_queries": number(command_arg(command, "--agent-max-path-queries")),
        "min_path_support": number(command_arg(command, "--graphrag-min-path-support")),
        "learn_elapsed_seconds": (manifest.get("learn_summary") or {}).get("elapsed_seconds"),
        "retriever_total_seconds": timings.get("total"),
        "adjacency_seconds": timings.get("adjacency"),
        "retrieval_seconds": timings.get("retrieval"),
        "candidate_compile_seconds": timings.get("candidate_compile"),
        "reason_stage_elapsed_seconds": (manifest.get("reason_summary") or {}).get("elapsed_seconds"),
        "proof_search_total_seconds": reason.get("total_runtime_seconds"),
        "reason_engine_build_seconds": reason.get("shared_engine_build_seconds"),
        "end_to_end_seconds": manifest.get("elapsed_seconds"),
        "heldout_accuracy": reason.get("three_value_accuracy"),
        "heldout_recall": reason.get("supported_recall"),
        "false_supported_negatives": (reason.get("counts") or {}).get("supported_negative"),
        "manifest": str(path),
    }


def trace_row(path: Path, manifest: dict, summary: dict, trace: dict) -> dict:
    iterations = trace.get("iterations") or []
    first = iterations[0] if iterations else {}
    last = iterations[-1] if iterations else {}
    first_f1 = metric_f1(first)
    best_f1 = float(trace.get("best_f1") or 0.0)
    refiner_triggered = any(
        str(item.get("refiner_source") or "").startswith("llm_")
        for item in iterations
    )
    accepted_but_conservative = any(
        bool((item.get("diagnostics") or {}).get("accepted_but_conservative"))
        for item in iterations
    )
    return {
        "split_name": manifest.get("split_name"),
        "target": target_from_split(str(manifest.get("split_name") or "")),
        "seed": (manifest.get("split") or {}).get("seed"),
        "iterations": len(iterations),
        "initial_feedback": first.get("verifier_feedback"),
        "initial_f1": first_f1,
        "best_f1": best_f1,
        "final_status": summary.get("status"),
        "initial_rejected": first_f1 < float(trace.get("acceptance_f1") or 0.8),
        "refiner_triggered": refiner_triggered,
        "accepted_but_conservative": accepted_but_conservative,
        "candidate_improved": best_f1 > first_f1,
        "final_accepted": summary.get("status") == "ok",
        "no_improvement_after_refiner": bool(refiner_triggered and best_f1 <= first_f1),
        "first_path_queries": len((first.get("action") or {}).get("path_queries") or []),
        "final_refiner_source": last.get("refiner_source"),
        "manifest": str(path),
    }


def aggregate_traces(rows: list[dict]) -> dict:
    counter = Counter()
    by_target: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        for key in (
            "initial_rejected",
            "refiner_triggered",
            "accepted_but_conservative",
            "candidate_improved",
            "final_accepted",
            "no_improvement_after_refiner",
        ):
            if row.get(key):
                counter[key] += 1
                by_target[str(row.get("target"))][key] += 1
        counter["rows"] += 1
        by_target[str(row.get("target"))]["rows"] += 1
    return {
        "overall": dict(counter),
        "by_target": {target: dict(values) for target, values in sorted(by_target.items())},
    }


def extract_trace(summary: dict) -> dict:
    rounds = summary.get("rounds") or []
    if not rounds:
        return {}
    rationale = str(rounds[0].get("guidance_rationale") or "")
    marker = "Trace: "
    if marker not in rationale:
        return {}
    try:
        return json.loads(rationale.split(marker, 1)[1])
    except json.JSONDecodeError:
        match = re.search(r"Trace:\s*(\{.*\})\s*$", rationale)
        if not match:
            return {}
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return {}


def extract_deterministic_diagnostics(summary: dict) -> dict:
    import ast

    rounds = summary.get("rounds") or []
    if not rounds:
        return {}
    rationale = str(rounds[0].get("guidance_rationale") or "")
    marker = "Diagnostics: "
    if marker not in rationale:
        return {}
    try:
        value = ast.literal_eval(rationale.split(marker, 1)[1])
    except (SyntaxError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def metric_f1(iteration: dict) -> float:
    metrics = iteration.get("best_metrics") or {}
    try:
        return float(metrics.get("f1") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def command_arg(command: list, flag: str) -> str | None:
    try:
        return str(command[command.index(flag) + 1])
    except (ValueError, IndexError):
        return None


def number(value: str | None) -> int | float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return int(parsed) if parsed.is_integer() else parsed


def target_from_split(split_name: str) -> str:
    for target in ("canCallClass", "isAllowedToUse", "overridesMethod"):
        if target in split_name:
            return target
    return "-"


def write_markdown(path: Path, payload: dict) -> None:
    runtime_columns = [
        "target",
        "seed",
        "method",
        "max_path_queries",
        "heldout_accuracy",
        "heldout_recall",
        "retrieval_seconds",
        "retriever_total_seconds",
        "proof_search_total_seconds",
        "end_to_end_seconds",
    ]
    lines = ["# PARA Runtime and Trace Statistics", "", "## Runtime Rows", ""]
    lines.extend(markdown_table(payload["runtime_rows"], runtime_columns))
    lines.extend(["", "## Trace Aggregate", "", "```json", json.dumps(payload["trace_aggregate"], indent=2), "```", ""])
    trace_columns = [
        "target",
        "seed",
        "iterations",
        "initial_f1",
        "best_f1",
        "initial_rejected",
        "refiner_triggered",
        "accepted_but_conservative",
        "candidate_improved",
        "final_accepted",
    ]
    lines.extend(["## Trace Rows", ""])
    lines.extend(markdown_table(payload["trace_rows"], trace_columns))
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(rows: list[dict], columns: list[str]) -> list[str]:
    output = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        output.append("| " + " | ".join(format_value(row.get(column)) for column in columns) + " |")
    return output


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "-"
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()
