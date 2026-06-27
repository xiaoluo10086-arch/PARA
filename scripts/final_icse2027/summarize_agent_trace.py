#!/usr/bin/env python3
"""Summarize PARA agent traces from learn-then-reason JSON outputs.

The learner stores the agent trace inside ``guidance_rationale`` as:

    Agentic GraphRAG ... Trace: {...}

This script extracts the embedded JSON and writes a compact table for paper
figures: source counts, iteration counts, candidate counts, threshold status,
and final stop reasons.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="JSON files or directories to scan")
    parser.add_argument("--csv", dest="csv_path", help="Optional CSV output path")
    parser.add_argument("--json", dest="json_path", help="Optional JSON output path")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for path_text in args.paths:
        path = Path(path_text)
        for json_path in iter_json_files(path):
            rows.extend(extract_rows(json_path))

    rows.sort(key=lambda row: (row["source_file"], row["round_index"]))
    if args.csv_path:
        write_csv(Path(args.csv_path), rows)
    if args.json_path:
        Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_path).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.csv_path and not args.json_path:
        print(json.dumps(rows, ensure_ascii=False, indent=2))


def iter_json_files(path: Path) -> Iterable[Path]:
    if path.is_file() and path.suffix == ".json":
        yield path
    elif path.is_dir():
        yield from path.rglob("*.json")


def extract_rows(json_path: Path) -> List[Dict[str, Any]]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows: List[Dict[str, Any]] = []
    target = (
        data.get("target")
        or ((data.get("learn_summary") or {}).get("json") or {}).get("target")
        or ""
    )
    for round_index, record in enumerate(iter_round_records(data)):
        trace = parse_trace(record.get("guidance_rationale"))
        if not trace:
            continue
        accounting = trace_accounting(trace)
        rows.append(
            {
                "source_file": str(json_path),
                "round_index": round_index,
                "target": target or record.get("target") or "",
                "best_f1": trace.get("best_f1", ""),
                "acceptance_f1": trace.get("acceptance_f1", ""),
                "iteration_count": accounting.get("iteration_count", ""),
                "candidate_count_total": accounting.get("candidate_count_total", ""),
                "candidate_count_max": accounting.get("candidate_count_max", ""),
                "accepted_candidate_count": accounting.get("accepted_candidate_count", ""),
                "llm_planner_iteration_count": accounting.get("llm_planner_iteration_count", ""),
                "llm_refiner_iteration_count": accounting.get("llm_refiner_iteration_count", ""),
                "symbolic_feedback_iteration_count": accounting.get("symbolic_feedback_iteration_count", ""),
                "improved_best_iteration_count": accounting.get("improved_best_iteration_count", ""),
                "threshold_reached_iteration_count": accounting.get("threshold_reached_iteration_count", ""),
                "planned_path_query_count_total": accounting.get("planned_path_query_count_total", ""),
                "refiner_path_query_count_total": accounting.get("refiner_path_query_count_total", ""),
                "rejected_path_query_count_total": accounting.get("rejected_path_query_count_total", ""),
                "final_stop_reason": accounting.get("final_stop_reason", ""),
                "planner_source_counts": json.dumps(accounting.get("planner_source_counts", {}), sort_keys=True),
                "refiner_source_counts": json.dumps(accounting.get("refiner_source_counts", {}), sort_keys=True),
                "witness_evidence_mode": trace.get("witness_evidence_mode", ""),
                "indexed_plan_only": trace.get("indexed_plan_only", ""),
                "schema_profile_mode": trace.get("schema_profile_mode", ""),
            }
        )
    return rows


def trace_accounting(trace: Dict[str, Any]) -> Dict[str, Any]:
    accounting = trace.get("agent_accounting")
    if isinstance(accounting, dict) and accounting:
        return accounting
    iterations = trace.get("iterations") or []
    if not isinstance(iterations, list) or not iterations:
        return {}
    planner_sources: Dict[str, int] = {}
    refiner_sources: Dict[str, int] = {}
    for item in iterations:
        if not isinstance(item, dict):
            continue
        planner = str(item.get("planner_source") or "")
        refiner = str(item.get("refiner_source") or "")
        planner_sources[planner] = planner_sources.get(planner, 0) + 1
        refiner_sources[refiner] = refiner_sources.get(refiner, 0) + 1
    final = iterations[-1] if isinstance(iterations[-1], dict) else {}
    stop_reason = final.get("stop_reason")
    if not stop_reason:
        stop_reason = "acceptance_threshold" if float(trace.get("best_f1") or 0.0) >= float(trace.get("acceptance_f1") or 0.0) else "unknown"
    return {
        "iteration_count": len(iterations),
        "candidate_count_total": sum(int((item or {}).get("candidate_count") or 0) for item in iterations if isinstance(item, dict)),
        "candidate_count_max": max((int((item or {}).get("candidate_count") or 0) for item in iterations if isinstance(item, dict)), default=0),
        "accepted_candidate_count": sum(1 for item in iterations if isinstance(item, dict) and item.get("accepted_by_symbolic_guard")),
        "best_f1": trace.get("best_f1", ""),
        "final_stop_reason": stop_reason,
        "planner_source_counts": planner_sources,
        "refiner_source_counts": refiner_sources,
        "llm_planner_iteration_count": sum(count for name, count in planner_sources.items() if name.startswith("llm_")),
        "llm_refiner_iteration_count": sum(count for name, count in refiner_sources.items() if name.startswith("llm_")),
        "symbolic_feedback_iteration_count": sum(
            1
            for item in iterations
            if isinstance(item, dict)
            and item.get("verifier_feedback") in {"too_general", "too_specific", "mixed_errors", "no_candidate", "consistent"}
        ),
        "improved_best_iteration_count": sum(
            1
            for item in iterations
            if isinstance(item, dict) and (item.get("improved_best_candidate") or item.get("accepted_by_symbolic_guard"))
        ),
        "threshold_reached_iteration_count": sum(
            1
            for item in iterations
            if isinstance(item, dict)
            and (
                item.get("passes_acceptance_threshold")
                or float(((item.get("best_metrics") or {}).get("f1") or 0.0)) >= float(trace.get("acceptance_f1") or 0.0)
            )
        ),
        "planned_path_query_count_total": sum(
            len(((item.get("action") or {}).get("path_queries") or []))
            for item in iterations
            if isinstance(item, dict)
        ),
        "refiner_path_query_count_total": sum(
            len(((item.get("refiner_action") or {}).get("path_queries") or []))
            for item in iterations
            if isinstance(item, dict)
        ),
        "rejected_path_query_count_total": sum(
            len(((item.get("action") or {}).get("rejected_path_queries") or []))
            for item in iterations
            if isinstance(item, dict)
        ),
    }


def iter_round_records(data: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(data, dict):
        learn_json = (data.get("learn_summary") or {}).get("json")
        if isinstance(learn_json, dict):
            yield from iter_round_records(learn_json)
        rounds = data.get("rounds")
        if isinstance(rounds, list):
            for item in rounds:
                if isinstance(item, dict):
                    yield item
        elif "guidance_rationale" in data:
            yield data


def parse_trace(rationale: Any) -> Dict[str, Any]:
    if not isinstance(rationale, str):
        return {}
    marker = "Trace:"
    index = rationale.find(marker)
    if index < 0:
        return {}
    payload = rationale[index + len(marker) :].strip()
    try:
        parsed = json.loads(payload)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_file",
        "round_index",
        "target",
        "best_f1",
        "acceptance_f1",
        "iteration_count",
        "candidate_count_total",
        "candidate_count_max",
        "accepted_candidate_count",
        "llm_planner_iteration_count",
        "llm_refiner_iteration_count",
        "symbolic_feedback_iteration_count",
        "improved_best_iteration_count",
        "threshold_reached_iteration_count",
        "planned_path_query_count_total",
        "refiner_path_query_count_total",
        "rejected_path_query_count_total",
        "final_stop_reason",
        "planner_source_counts",
        "refiner_source_counts",
        "witness_evidence_mode",
        "indexed_plan_only",
        "schema_profile_mode",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
