#!/usr/bin/env python3
"""Summarize versioned PARA revision manifests without merging old results."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from pathlib import Path


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    manifests = [
        path
        for path in sorted(root.rglob("*_manifest.json"))
        if path.parent.name == "manifests"
    ]
    rows = [summarize_manifest(path) for path in manifests]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "result_root": str(root),
        "manifest_count": len(rows),
        "source_snapshot": source_snapshot(Path(args.project_root).resolve()),
        "rows": rows,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output.with_suffix(".csv"), rows)
    write_markdown(output.with_suffix(".md"), rows, payload["source_snapshot"])
    print(f"Summarized {len(rows)} manifests into {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize PARA ICSE revision results")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    return parser


def summarize_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    command = ((manifest.get("learn_summary") or {}).get("command") or [])
    learn = ((manifest.get("learn_summary") or {}).get("json") or {})
    metrics = learn.get("metrics") or {}
    reason = ((manifest.get("reason_summary") or {}).get("json") or {})
    counts = reason.get("counts") or {}
    summary = read_json(Path(manifest.get("learn_output") or "") / "summary.json")
    trace = extract_trace(summary)
    iteration = ((trace.get("iterations") or [{}])[0]) if trace else {}
    action = iteration.get("action") or {}
    diagnostics = iteration.get("diagnostics") or {}
    plan_diagnostics = {}
    portfolio = diagnostics.get("portfolio") or []
    if portfolio:
        plan_diagnostics = (portfolio[0] or {}).get("diagnostics") or {}
    if not plan_diagnostics:
        plan_diagnostics = extract_deterministic_diagnostics(summary)
    timings = plan_diagnostics.get("timings_seconds") or {}
    validity = manifest.get("validity") or {}
    return {
        "manifest": str(path),
        "split_name": manifest.get("split_name"),
        "task_dir": manifest.get("task_dir"),
        "seed": (manifest.get("split") or {}).get("seed"),
        "guide_provider": command_arg(command, "--guide-provider"),
        "min_path_support": number(command_arg(command, "--graphrag-min-path-support")),
        "max_path_queries": number(command_arg(command, "--agent-max-path-queries")),
        "paper_valid": not bool(validity.get("invalid_for_paper")),
        "validity_category": validity.get("category", "ok"),
        "learn_status": learn.get("status"),
        "train_f1": metrics.get("f1"),
        "train_precision": metrics.get("precision"),
        "train_recall": metrics.get("recall"),
        "final_rule": summary.get("final_rule"),
        "path_programs_proposed": len(action.get("path_queries") or []),
        "candidate_count": iteration.get("candidate_count"),
        "heldout_accuracy": reason.get("three_value_accuracy"),
        "heldout_precision": reason.get("supported_precision"),
        "heldout_recall": reason.get("supported_recall"),
        "unsupported_recall": reason.get("unsupported_recall"),
        "inconclusive_rate": reason.get("inconclusive_rate"),
        "false_supported_negatives": counts.get("supported_negative"),
        "learn_elapsed_seconds": (manifest.get("learn_summary") or {}).get("elapsed_seconds"),
        "reason_stage_elapsed_seconds": (manifest.get("reason_summary") or {}).get("elapsed_seconds"),
        "proof_search_total_seconds": reason.get("total_runtime_seconds"),
        "end_to_end_seconds": manifest.get("elapsed_seconds"),
        "adjacency_seconds": timings.get("adjacency"),
        "constraint_index_seconds": timings.get("constraint_index"),
        "retrieval_seconds": timings.get("retrieval"),
        "candidate_compile_seconds": timings.get("candidate_compile"),
        "retriever_total_seconds": timings.get("total"),
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


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def source_snapshot(project_root: Path) -> dict:
    relative_paths = [
        "src/para/agentic_graph_rag.py",
        "src/para/graph_rag.py",
        "src/para/reasoner.py",
        "src/para/cli.py",
        "scripts/run_learn_then_reason.py",
        "scripts/run_teammates_clean9_learn_then_reason.py",
    ]
    files = {}
    for relative in relative_paths:
        path = project_root / relative
        if path.exists():
            files[relative] = sha256(path)
    return {"identity_rule": "sha256", "files": files}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict], snapshot: dict) -> None:
    columns = [
        "split_name",
        "guide_provider",
        "min_path_support",
        "max_path_queries",
        "learn_status",
        "train_f1",
        "heldout_accuracy",
        "heldout_precision",
        "heldout_recall",
        "inconclusive_rate",
        "false_supported_negatives",
        "end_to_end_seconds",
    ]
    lines = [
        "# PARA ICSE Revision Experiment Summary",
        "",
        f"Manifests: {len(rows)}",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(column)) for column in columns) + " |")
    lines.extend(["", "## Source Snapshot", "", "```json", json.dumps(snapshot, indent=2), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "-"
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()
