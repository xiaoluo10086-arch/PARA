#!/usr/bin/env python3
"""Materialize paper-facing table products for the final PARA suite."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.output_root)
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    final_rows = read_csv(tables / "final_suite_summary.csv")
    proof_rows = read_csv(tables / "proof_contract_summary.csv")
    agent_rows = read_csv(tables / "agent_trace_summary.csv")
    phase_rows = read_phase_status(out / "phase_status.tsv")

    write_csv(tables / "table1_effectiveness.csv", table1_effectiveness(final_rows))
    write_csv(tables / "table2_accountability.csv", table2_accountability(proof_rows, phase_rows))
    write_csv(tables / "table3_mechanism.csv", table3_mechanism(final_rows, phase_rows))
    write_csv(tables / "table4_agent_loop.csv", table4_agent_loop(final_rows, agent_rows))
    write_csv(tables / "table5_transfer_boundary.csv", table5_transfer_boundary(phase_rows))
    write_csv(tables / "appendix_package_scalability.csv", appendix_package_scalability(phase_rows))
    write_csv(tables / "final_suite_index.csv", final_suite_index(out, final_rows, phase_rows))

    payload = {
        "output_root": str(out),
        "table_files": [
            "tables/table1_effectiveness.csv",
            "tables/table2_accountability.csv",
            "tables/table3_mechanism.csv",
            "tables/table4_agent_loop.csv",
            "tables/table5_transfer_boundary.csv",
            "tables/appendix_package_scalability.csv",
            "tables/final_suite_index.csv",
        ],
        "counts": {
            "final_suite_rows": len(final_rows),
            "proof_contract_rows": len(proof_rows),
            "agent_trace_rows": len(agent_rows),
            "phase_rows": len(phase_rows),
        },
    }
    (tables / "table_product_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root")
    return parser


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_phase_status(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("phase\t"):
            continue
        parts = line.split("\t")
        while len(parts) < 6:
            parts.append("")
        rows.append(
            {
                "phase": parts[0],
                "item": parts[1],
                "status": parts[2],
                "exit_code": parts[3],
                "log": parts[4],
                "note": parts[5],
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"])
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def target_rows(rows: Iterable[Dict[str, str]], targets: set[str] | None = None) -> List[Dict[str, str]]:
    selected = []
    for row in rows:
        if targets and row.get("target") not in targets:
            continue
        selected.append(row)
    return selected


def table1_effectiveness(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    preferred_markers = ("agent_loop2", "agent_oneshot", "agent_deterministic_top1")
    selected = [
        row
        for row in rows
        if row.get("target") in {"canCallClass", "isAllowedToUse", "overridesMethod"}
        and any(marker in row.get("arm", "") for marker in preferred_markers)
    ]
    if not selected:
        selected = target_rows(rows, {"canCallClass", "isAllowedToUse", "overridesMethod"})
    return [
        {
            "target": row.get("target", ""),
            "seed": row.get("seed", ""),
            "arm": row.get("arm", ""),
            "supported_precision": row.get("held_out_supported_precision", ""),
            "supported_recall": row.get("held_out_supported_recall", ""),
            "false_supported_negatives": "",
            "inconclusive_rate": row.get("held_out_inconclusive_rate", ""),
            "accuracy": row.get("held_out_accuracy", ""),
            "manifest": row.get("manifest", ""),
        }
        for row in selected
    ]


def table2_accountability(
    proof_rows: List[Dict[str, str]],
    phase_rows: List[Dict[str, str]],
) -> List[Dict[str, object]]:
    proof_statuses = Counter(row.get("status", "") for row in proof_rows)
    rows: List[Dict[str, object]] = [
        {
            "audit_type": "proof_contract_materialization",
            "scope": "final-suite manifests",
            "unit_count": len(proof_rows),
            "error_or_false_support": proof_statuses.get("error", 0),
            "artifact": "tables/proof_contract_summary.csv",
            "interpretation": "SUPPORTED/INCONCLUSIVE samples have materialized contracts when available",
        }
    ]
    for phase in ("near_miss_exact", "proof_tree_audit", "isolation_audit", "counterfactual_audit", "recursive_idb_validation"):
        matches = [row for row in phase_rows if row["phase"] == phase]
        if matches:
            rows.append(
                {
                    "audit_type": phase,
                    "scope": "recorded final-suite phase",
                    "unit_count": len(matches),
                    "error_or_false_support": ",".join(sorted({row["status"] for row in matches})),
                    "artifact": ";".join(row["log"] for row in matches if row["log"]),
                    "interpretation": matches[-1].get("note", ""),
                }
            )
    return rows


def table3_mechanism(
    rows: List[Dict[str, str]],
    phase_rows: List[Dict[str, str]],
) -> List[Dict[str, object]]:
    mechanism = [
        row
        for row in rows
        if row.get("seed") in {"", "0"}
        and row.get("target") in {"canCallClass", "isAllowedToUse", "overridesMethod"}
    ]
    output = [
        {
            "method": row.get("arm", ""),
            "target": row.get("target", ""),
            "seed": row.get("seed", ""),
            "train_f1": row.get("train_f1", ""),
            "held_out_recall": row.get("held_out_supported_recall", ""),
            "supported_precision": row.get("held_out_supported_precision", ""),
            "candidate_or_rule_count": row.get("exported_rule_count", ""),
            "time_seconds": row.get("total_elapsed_seconds", ""),
            "artifact": row.get("manifest", ""),
        }
        for row in mechanism
    ]
    for phase in ("popper_typed_budgeted", "direct_qa", "direct_rules", "typed_heuristic"):
        for item in [row for row in phase_rows if row["phase"] == phase]:
            output.append(
                {
                    "method": item["item"],
                    "target": "",
                    "seed": "",
                    "train_f1": "",
                    "held_out_recall": "",
                    "supported_precision": "",
                    "candidate_or_rule_count": "",
                    "time_seconds": "",
                    "artifact": item["log"],
                    "status": item["status"],
                }
            )
    return output


def table4_agent_loop(
    final_rows: List[Dict[str, str]],
    agent_rows: List[Dict[str, str]],
) -> List[Dict[str, object]]:
    loop_rows = [
        row
        for row in final_rows
        if any(marker in row.get("arm", "") for marker in ("agent_oneshot", "agent_loop2", "agent_loop3"))
    ]
    by_split = {row.get("split_name", ""): row for row in agent_rows}
    return [
        {
            "target": row.get("target", ""),
            "seed": row.get("seed", ""),
            "arm": row.get("arm", ""),
            "train_f1": row.get("train_f1", ""),
            "held_out_recall": row.get("held_out_supported_recall", ""),
            "supported_precision": row.get("held_out_supported_precision", ""),
            "trace_status": by_split.get(row.get("split_name", ""), {}).get("status", ""),
            "refiner_actions": by_split.get(row.get("split_name", ""), {}).get("refiner_actions", ""),
            "stop_reason": by_split.get(row.get("split_name", ""), {}).get("stop_reason", ""),
            "manifest": row.get("manifest", ""),
        }
        for row in loop_rows
    ]


def table5_transfer_boundary(phase_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    wanted = {
        "teammates_clean9",
        "spring_policy",
        "opencv_miniproj",
        "opencv_full_source",
        "opencv_packaging",
        "opencv_workload_matrix",
        "opencv_manifest_consistency",
        "static_extractor_boundary",
    }
    return [
        {
            "scenario": row["phase"],
            "item": row["item"],
            "status": row["status"],
            "artifact": row["log"],
            "interpretation": row["note"],
        }
        for row in phase_rows
        if row["phase"] in wanted
    ]


def appendix_package_scalability(phase_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    return [
        {
            "package_probe": row["item"],
            "status": row["status"],
            "artifact": row["log"],
            "note": row["note"],
        }
        for row in phase_rows
        if row["phase"] in {"extra_package_probe", "opencv_full_source", "opencv_packaging"}
    ]


def final_suite_index(
    out: Path,
    final_rows: List[Dict[str, str]],
    phase_rows: List[Dict[str, str]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in final_rows:
        rows.append(
            {
                "kind": "learn_then_reason",
                "project": "spring",
                "target": row.get("target", ""),
                "seed": row.get("seed", ""),
                "arm": row.get("arm", ""),
                "status": row.get("learn_status", ""),
                "artifact": row.get("manifest", ""),
            }
        )
    for row in phase_rows:
        rows.append(
            {
                "kind": "phase",
                "project": "",
                "target": "",
                "seed": "",
                "arm": row["phase"],
                "status": row["status"],
                "artifact": row["log"],
                "note": row["note"],
            }
        )
    if not rows:
        rows.append({"kind": "empty", "status": "no rows", "artifact": str(out)})
    return rows


if __name__ == "__main__":
    main()
