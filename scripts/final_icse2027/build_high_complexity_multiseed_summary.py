#!/usr/bin/env python3
"""Aggregate high-complexity target/strategy runs for the final paper."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


TARGETS = ["signatureSafeOverride", "interfaceOverride", "packageCallDependency"]
ARMS = ["proof_strategy", "heuristic", "graphrag"]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        matrix = Path(args.input_template.format(seed=seed)) / "high_complexity_seed0_matrix.csv"
        if not matrix.exists():
            raise FileNotFoundError(matrix)
        for row in read_csv(matrix):
            manifest = read_json(Path(row.get("manifest", "")))
            clean = {
                "target": row["target"],
                "arm": row["arm"],
                "seed": seed,
                "returncode": row.get("returncode", ""),
                "valid_for_paper": row.get("valid_for_paper", ""),
                "validity_category": row.get("validity_category", ""),
                "learn_status": row.get("learn_status", ""),
                "train_f1": row.get("train_f1", ""),
                "supported_precision": row.get("supported_precision", ""),
                "supported_recall": row.get("supported_recall", ""),
                "mean_runtime_seconds": row.get("mean_runtime_seconds", ""),
                "elapsed_seconds": elapsed_seconds(manifest),
                "manifest": row.get("manifest", ""),
            }
            clean["admitted"] = "1" if clean["learn_status"] == "ok" else "0"
            clean["valid_cell"] = "1" if clean["valid_for_paper"] == "True" else "0"
            rows.append(clean)

    summary = []
    for target in TARGETS:
        for arm in ARMS:
            cells = [r for r in rows if r["target"] == target and r["arm"] == arm]
            valid_cells = [r for r in cells if r["valid_cell"] == "1"]
            admitted = [r for r in cells if r["admitted"] == "1"]
            train_f1 = [to_float(r["train_f1"]) for r in admitted if r["train_f1"] != ""]
            elapsed = [to_float(r["elapsed_seconds"]) for r in cells if r["elapsed_seconds"] != ""]
            status_counts = count_by(cells, "validity_category")
            summary.append(
                {
                    "target": target,
                    "arm": arm,
                    "seeds": len(cells),
                    "valid_cells": len(valid_cells),
                    "admitted": len(admitted),
                    "admission_rate": safe_div(len(admitted), len(cells)),
                    "valid_admission_rate": safe_div(len(admitted), len(valid_cells)),
                    "train_f1_mean": mean(train_f1),
                    "train_f1_sd": stdev(train_f1),
                    "elapsed_seconds_mean": mean(elapsed),
                    "elapsed_seconds_sd": stdev(elapsed),
                    "status_counts": status_counts,
                    "status_brief": brief_status(status_counts),
                }
            )

    write_csv(output_dir / "high_complexity_multiseed_rows.csv", rows)
    write_csv(output_dir / "high_complexity_multiseed_summary.csv", summary)
    (output_dir / "high_complexity_multiseed_summary.json").write_text(
        json.dumps({"rows": rows, "summary": summary}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "rows": len(rows), "summary": len(summary), "output_dir": str(output_dir)}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-template",
        default=(
            "artifacts/proof_strategy_agent/results/high_complexity_hard2_three_targets_seed{seed}"
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--output-dir",
        default="results/final_icse2027/high_complexity_multiseed",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    if not str(path) or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: str) -> float:
    return float(value)


def elapsed_seconds(manifest: dict) -> str:
    value = manifest.get("elapsed_seconds")
    if value is None:
        return ""
    return f"{float(value):.3f}"


def safe_div(num: int, den: int) -> str:
    if den == 0:
        return ""
    return f"{num / den:.3f}"


def mean(values: list[float]) -> str:
    if not values:
        return ""
    return f"{statistics.mean(values):.3f}"


def stdev(values: list[float]) -> str:
    if len(values) < 2:
        return "0.000" if values else ""
    return f"{statistics.stdev(values):.3f}"


def count_by(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key) or "missing"
        counts[value] = counts.get(value, 0) + 1
    return counts


def brief_status(counts: dict[str, int]) -> str:
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


if __name__ == "__main__":
    main()
