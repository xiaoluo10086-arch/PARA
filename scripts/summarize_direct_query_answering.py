#!/usr/bin/env python3
"""Combine completed Direct QA runs into one paper-facing report."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


TARGETS = ("canCallClass", "isAllowedToUse", "overridesMethod")


def main() -> None:
    args = build_parser().parse_args()
    grouped: Dict[str, List[dict]] = defaultdict(list)
    sources: Dict[str, List[str]] = defaultdict(list)
    for run_name in args.run:
        run = Path(run_name).resolve()
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        model = str(summary["model_tag"])
        sources[model].append(str(run))
        for line in (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                grouped[model].append(json.loads(line))

    output = {
        "protocol": "direct_llm_query_answering",
        "comparison_boundary": (
            "Direct QA does not see held-out labels. PARA's evaluation harness uses explicit "
            "negative labels only to distinguish UNSUPPORTED from INCONCLUSIVE after proof search. "
            "Therefore supported precision/recall are the primary matched metrics; three-valued "
            "accuracy remains diagnostic."
        ),
        "models": {},
    }
    for model, rows in grouped.items():
        output["models"][model] = {
            "sources": sources[model],
            "queries": len(rows),
            "by_target": {
                target: metrics([row for row in rows if str(row["query"]).startswith(target + "(")])
                for target in TARGETS
            },
            "transport_or_parse_failures": sum(row["decision"] == "INVALID" for row in rows),
            "invalid_evidence_citations": sum(bool(row["invalid_evidence_ids"]) for row in rows),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(output_path.with_suffix(".md"), output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser


def metrics(rows: List[dict]) -> dict:
    counts = Counter(str(row["decision"]) for row in rows)
    positives = [row for row in rows if row["gold"] == "positive"]
    negatives = [row for row in rows if row["gold"] == "negative"]
    tp = sum(row["decision"] == "SUPPORTED" for row in positives)
    fp = sum(row["decision"] == "SUPPORTED" for row in negatives)
    tn = sum(row["decision"] == "UNSUPPORTED" for row in negatives)
    return {
        "queries": len(rows),
        "supported_precision": tp / (tp + fp) if tp + fp else 0.0,
        "supported_recall": tp / len(positives) if positives else 0.0,
        "unsupported_recall": tn / len(negatives) if negatives else 0.0,
        "inconclusive_rate": counts["INCONCLUSIVE"] / len(rows) if rows else 0.0,
        "diagnostic_three_value_accuracy": (tp + tn) / len(rows) if rows else 0.0,
        "invalid": counts["INVALID"],
        "positive_queries_with_no_retrieved_path": sum(
            row["gold"] == "positive" and not row["view_metadata"].get("path_count")
            for row in rows
        ),
    }


def write_markdown(path: Path, output: dict) -> None:
    lines = [
        "# Direct LLM Query Answering",
        "",
        "The LLM receives each held-out query and a deterministic query-centered graph view. "
        "It does not generate or export a rule library.",
        "",
        "**Comparison boundary.** " + output["comparison_boundary"],
        "",
        "| Model | Target | Queries | Supported precision | Supported recall | Unsupported recall | Inconclusive | Invalid |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, model_data in output["models"].items():
        for target in TARGETS:
            row = model_data["by_target"][target]
            if not row["queries"]:
                continue
            lines.append(
                f"| {model} | `{target}` | {row['queries']} | {row['supported_precision']:.3f} | "
                f"{row['supported_recall']:.3f} | {row['unsupported_recall']:.3f} | "
                f"{row['inconclusive_rate']:.3f} | {row['invalid']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
