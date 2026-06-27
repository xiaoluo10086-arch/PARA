#!/usr/bin/env python3
"""Build a paper-facing matrix from the 2026-06-27 unified rerun.

The rerun intentionally separates low-cost learn/admission evidence from
query-level held-out proof evidence.  This script preserves that separation:
fresh rerun rows provide admission/train/runtime fields, while the verified
suite supplies held-out fields where the project/target/seed/method match.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple


CORE_METHODS = {
    "agent_deterministic_top1": "proof_strategy",
    "proof_strategy": "proof_strategy",
    "typed_heuristic": "typed_heuristic",
    "heuristic": "typed_heuristic",
    "graphrag": "graphrag",
}


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    learn_rows = read_csv(Path(args.learn_summary))
    verified_rows = read_csv(Path(args.verified_effectiveness))
    high_complexity_rows = read_csv(Path(args.high_complexity))

    matrix = build_core_matrix(learn_rows, verified_rows)
    matrix.extend(build_high_complexity_matrix(high_complexity_rows))
    matrix.sort(key=lambda row: (row["project"], row["target"], int(row["seed"]), row["method"]))

    summary = summarize(matrix)
    write_csv(output_dir / "unified_core_matrix.csv", matrix)
    write_json(output_dir / "unified_core_matrix.json", matrix)
    write_csv(output_dir / "unified_core_summary_by_target_method.csv", summary)
    write_json(output_dir / "unified_core_summary_by_target_method.json", summary)
    write_readme(output_dir / "README.md", args, matrix, summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learn-summary", required=True)
    parser.add_argument("--verified-effectiveness", required=True)
    parser.add_argument("--high-complexity", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_core_matrix(
    learn_rows: List[Dict[str, str]],
    verified_rows: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    verified = {
        (row.get("target", ""), row.get("seed", ""), normalize_method(row.get("method", ""))): row
        for row in verified_rows
        if normalize_method(row.get("method", "")) in {"proof_strategy", "typed_heuristic", "graphrag"}
    }
    rows: List[Dict[str, Any]] = []
    for row in learn_rows:
        method = normalize_method(row.get("arm", ""))
        if method not in {"proof_strategy", "typed_heuristic", "graphrag"}:
            continue
        target = row.get("target", "")
        seed = row.get("seed", "")
        held = verified.get((target, seed, method), {})
        learn_status = row.get("learn_status", "")
        admitted = learn_status == "ok" and as_float(row.get("exported_rule_count")) > 0
        heldout_source = "verified_suite_20260627" if held else ""
        rows.append(
            {
                "project": "Spring",
                "target": target,
                "target_group": "canonical_architecture_relation",
                "seed": seed,
                "method": method,
                "model": "deterministic" if method != "proof_strategy" else "proof_strategy_agent",
                "admitted": admitted,
                "learn_status": learn_status,
                "train_f1": row.get("train_f1", ""),
                "train_precision": row.get("train_precision", ""),
                "train_recall": row.get("train_recall", ""),
                "best_candidate_f1": row.get("train_f1", ""),
                "best_candidate_precision": row.get("train_precision", ""),
                "best_candidate_recall": row.get("train_recall", ""),
                "heldout_precision": held.get("query_supported_precision", ""),
                "heldout_recall": held.get("query_supported_recall", ""),
                "negative_non_support": held.get("query_negative_non_support", ""),
                "inconclusive_rate": held.get("query_inconclusive_rate", ""),
                "accuracy": held.get("query_accuracy", ""),
                "runtime_seconds": row.get("total_elapsed_seconds", ""),
                "proof_contract_scope": "learn/admission rerun; held-out metrics reused" if held else "learn/admission rerun only",
                "fresh_rerun_manifest": row.get("manifest", ""),
                "heldout_manifest": held.get("manifest", ""),
                "source_run": "final_suite_20260627_unified_p0a_learn_only",
                "heldout_source": heldout_source,
                "notes": "query-level proof search skipped in rerun because full Spring proof materialization is high-cost",
            }
        )
    return rows


def build_high_complexity_matrix(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in rows:
        method = normalize_method(row.get("arm", ""))
        if method not in {"proof_strategy", "typed_heuristic", "graphrag"}:
            continue
        admitted = row.get("admitted", "").lower() == "true"
        output.append(
            {
                "project": "Spring",
                "target": row.get("target", ""),
                "target_group": "high_complexity_compositional_relation",
                "seed": row.get("seed", ""),
                "method": method,
                "model": "proof_strategy_agent" if method == "proof_strategy" else "deterministic",
                "admitted": admitted,
                "learn_status": row.get("status", ""),
                "train_f1": row.get("train_f1", ""),
                "train_precision": row.get("train_precision", ""),
                "train_recall": row.get("train_recall", ""),
                "best_candidate_f1": row.get("best_candidate_f1", row.get("train_f1", "")),
                "best_candidate_precision": row.get("best_candidate_precision", row.get("train_precision", "")),
                "best_candidate_recall": row.get("best_candidate_recall", row.get("train_recall", "")),
                "heldout_precision": row.get("heldout_precision", ""),
                "heldout_recall": row.get("heldout_recall", ""),
                "negative_non_support": "1.0" if row.get("heldout_fp", "") == "0" else "",
                "inconclusive_rate": "",
                "accuracy": row.get("heldout_accuracy", ""),
                "runtime_seconds": row.get("elapsed_seconds", ""),
                "proof_contract_scope": "rule-level held-out plus representative proof contracts",
                "fresh_rerun_manifest": "",
                "heldout_manifest": row.get("manifest", ""),
                "source_run": "final_paper_verified_suite_20260627",
                "heldout_source": "high_complexity_verified_suite_20260627",
                "notes": "high-complexity target retained as verified evidence layer",
            }
        )
    return output


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["project"], row["target"], row["method"])].append(row)
    summary: List[Dict[str, Any]] = []
    for (project, target, method), group in sorted(groups.items()):
        summary.append(
            {
                "project": project,
                "target": target,
                "method": method,
                "seeds": len(group),
                "admission_rate": round(mean(1.0 if row["admitted"] else 0.0 for row in group), 4),
                "mean_train_f1": fmt_mean(row.get("train_f1", "") for row in group),
                "mean_best_candidate_f1": fmt_mean(row.get("best_candidate_f1", "") for row in group),
                "mean_heldout_precision": fmt_mean(row.get("heldout_precision", "") for row in group),
                "mean_heldout_recall": fmt_mean(row.get("heldout_recall", "") for row in group),
                "mean_accuracy": fmt_mean(row.get("accuracy", "") for row in group),
                "mean_runtime_seconds": fmt_mean(row.get("runtime_seconds", "") for row in group),
                "evidence_layer": "; ".join(sorted({row.get("proof_contract_scope", "") for row in group if row.get("proof_contract_scope")})),
            }
        )
    return summary


def normalize_method(value: str) -> str:
    return CORE_METHODS.get(value, value)


def as_float(value: Any) -> float:
    try:
        if value == "" or value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def fmt_mean(values: Iterable[Any]) -> str:
    numeric = [as_float(value) for value in values if value not in ("", None)]
    if not numeric:
        return ""
    return f"{mean(numeric):.4f}"


def write_readme(path: Path, args: argparse.Namespace, matrix: List[Dict[str, Any]], summary: List[Dict[str, Any]]) -> None:
    passed = sum(1 for row in matrix if row.get("admitted"))
    fresh = [row for row in matrix if row.get("source_run") == "final_suite_20260627_unified_p0a_learn_only"]
    text = f"""# Unified Rerun Matrix (2026-06-27)

This directory contains paper-facing tables built from the 2026-06-27 rerun.

## Inputs

- Fresh learn/admission rerun: `{args.learn_summary}`
- Verified held-out Spring suite: `{args.verified_effectiveness}`
- Verified high-complexity suite: `{args.high_complexity}`

## Outputs

- `unified_core_matrix.csv`: row-level project/target/seed/method matrix.
- `unified_core_summary_by_target_method.csv`: compact table for paper figures/tables.

## Interpretation

The fresh rerun completed the low-cost learn/admission layer for {len(fresh)} Spring rows.
The combined matrix contains {len(matrix)} rows, with {passed} admitted rows across fresh and verified layers.
Full query-level proof search was not rerun for every Spring cell because proof materialization is the expensive layer;
held-out fields are therefore explicitly sourced from the existing verified suite.  This keeps the paper table
honest: admission is freshly rerun, while proof/search evidence is treated as a separate validated layer.

Summary rows: {len(summary)}.
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
