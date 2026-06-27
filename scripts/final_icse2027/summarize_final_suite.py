#!/usr/bin/env python3
"""Build final paper tables from PARA final-suite manifests."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


def main() -> None:
    args = build_parser().parse_args()
    rows: List[Dict[str, Any]] = []
    for root_text in args.paths:
        root = Path(root_text)
        for manifest in iter_manifests(root):
            row = summarize_manifest(manifest)
            if row:
                rows.append(row)
    rows.sort(key=lambda row: (row["target"], int(row["seed"]), row["arm"]))
    if args.csv:
        write_csv(Path(args.csv), rows)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.csv and not args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize PARA final-suite manifests")
    parser.add_argument("paths", nargs="+", help="Final-suite output directories or manifest files")
    parser.add_argument("--csv", help="Optional CSV output")
    parser.add_argument("--json", help="Optional JSON output")
    return parser


def iter_manifests(root: Path) -> Iterable[Path]:
    if root.is_file() and is_run_manifest(root):
        yield root
    elif root.is_dir():
        yield from (path for path in sorted(root.rglob("*_manifest.json")) if is_run_manifest(path))


def is_run_manifest(path: Path) -> bool:
    return path.name.endswith("_manifest.json") and path.name != "split_manifest.json"


def summarize_manifest(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    split = str(data.get("split_name") or path.stem.replace("_manifest", ""))
    learn_json = ((data.get("learn_summary") or {}).get("json") or {})
    reason_json = ((data.get("reason_summary") or {}).get("json") or {})
    no_rule_json = ((data.get("no_rule_summary") or {}).get("json") or {})
    export_json = ((data.get("export_summary") or {}).get("json") or {})
    metrics = learn_json.get("metrics") or learn_json.get("best_candidate_metrics") or {}
    target, seed, arm = parse_split_name(split)
    return {
        "manifest": str(path),
        "split_name": split,
        "target": target or learn_json.get("target") or reason_json.get("target") or "",
        "seed": seed,
        "arm": arm,
        "valid_for_paper": data.get("valid_for_paper", True),
        "validity_category": (data.get("validity") or {}).get("category", ""),
        "learn_status": learn_json.get("status", ""),
        "learn_rule": learn_json.get("final_rule", ""),
        "train_precision": metrics.get("precision", ""),
        "train_recall": metrics.get("recall", ""),
        "train_f1": metrics.get("f1", ""),
        "exported_rule_count": export_json.get("rule_count", ""),
        "held_out_examples": reason_json.get("examples", ""),
        "held_out_supported_precision": reason_json.get("supported_precision", ""),
        "held_out_supported_recall": reason_json.get("supported_recall", ""),
        "held_out_negative_non_support": reason_json.get("negative_non_support_rate", ""),
        "held_out_inconclusive_rate": reason_json.get("inconclusive_rate", ""),
        "held_out_accuracy": reason_json.get("held_out_accuracy", reason_json.get("three_value_accuracy", "")),
        "no_rule_inconclusive_rate": no_rule_json.get("inconclusive_rate", ""),
        "learn_elapsed_seconds": (data.get("learn_summary") or {}).get("elapsed_seconds", ""),
        "reason_elapsed_seconds": (data.get("reason_summary") or {}).get("elapsed_seconds", ""),
        "total_elapsed_seconds": data.get("elapsed_seconds", ""),
        "rule_library": data.get("rule_library", ""),
        "reason_output": data.get("reason_output", ""),
    }


def parse_split_name(split: str) -> tuple[str, str, str]:
    pattern = re.compile(r"^spring_(?P<target>.+?)_train\d+_\d+_seed(?P<seed>\d+)_(?P<arm>.+)$")
    match = pattern.match(split)
    if not match:
        return "", "0", split
    return match.group("target"), match.group("seed"), match.group("arm")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "manifest",
        "split_name",
        "target",
        "seed",
        "arm",
        "valid_for_paper",
        "validity_category",
        "learn_status",
        "learn_rule",
        "train_precision",
        "train_recall",
        "train_f1",
        "exported_rule_count",
        "held_out_examples",
        "held_out_supported_precision",
        "held_out_supported_recall",
        "held_out_negative_non_support",
        "held_out_inconclusive_rate",
        "held_out_accuracy",
        "no_rule_inconclusive_rate",
        "learn_elapsed_seconds",
        "reason_elapsed_seconds",
        "total_elapsed_seconds",
        "rule_library",
        "reason_output",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
