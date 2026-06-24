#!/usr/bin/env python3
"""Summarize the matched Planner-evidence ablation manifests."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("witness_only", "schema_only", "full_para")
TARGETS = ("canCallClass", "isAllowedToUse", "overridesMethod")


def manifest_for(results: Path, arm: str, target: str) -> Path:
    return results / arm / "manifests" / f"spring_{target}_train50_100_seed0_{arm}_manifest.json"


def row_for(results: Path, arm: str, target: str) -> dict[str, object]:
    path = manifest_for(results, arm, target)
    data = json.loads(path.read_text())
    learn = data["learn_summary"]["json"]
    reason = data["reason_summary"]["json"]
    learn_f1 = learn.get("metrics", {}).get("f1") if learn.get("metrics") else None
    if learn_f1 is None:
        report = Path(data["learn_output"]) / "report.md"
        if report.exists():
            match = re.search(r'"best_f1":\s*([0-9.]+)', report.read_text())
            if match:
                learn_f1 = float(match.group(1))
    if learn_f1 is None:
        learn_f1 = 0.0
    return {
        "arm": arm,
        "target": target,
        "valid_for_paper": bool(data["valid_for_paper"]),
        "learn_status": learn["status"],
        "learn_f1": learn_f1,
        "final_rule": learn.get("final_rule"),
        "held_out_accuracy": reason["held_out_accuracy"],
        "supported_precision": reason["supported_precision"],
        "supported_recall": reason["supported_recall"],
        "negative_non_support_rate": reason["negative_non_support_rate"],
        "positive_abstention_rate": 1.0 - reason["supported_recall"],
        "elapsed_seconds": data["elapsed_seconds"],
        "manifest": str(path.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "artifacts" / "e15_planner_evidence_ablation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "summaries" / "planner_evidence_ablation_summary.json",
    )
    args = parser.parse_args()
    rows = [row_for(args.results, arm, target) for arm in ARMS for target in TARGETS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"rows": rows}, indent=2) + "\n")

    lines = [
        "# Planner Evidence Ablation (Spring xlarge3, seed 0)",
        "",
        "All arms share the frozen split, indexed plan-only executor, candidate compiler, "
        "symbolic verifier (train F1 >= 0.8), and held-out proof engine.",
        "",
        "| Arm | Target | Train F1 | Held-out acc. | Supported precision | Supported recall | Positive abstention |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {arm} | {target} | {learn_f1:.3f} | {held_out_accuracy:.3f} | "
            "{supported_precision:.3f} | {supported_recall:.3f} | "
            "{positive_abstention_rate:.3f} |".format(**row)
        )
    lines.extend(["", "## Accepted rules", ""])
    for row in rows:
        lines.append(f"- `{row['arm']}/{row['target']}`: `{row['final_rule']}`")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
