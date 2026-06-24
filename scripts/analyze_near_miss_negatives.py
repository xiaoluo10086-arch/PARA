#!/usr/bin/env python3
"""Analyze held-out explicit negatives that are close to positive queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CODE_ROOT = ROOT / "src"


def main() -> None:
    args = build_parser().parse_args()
    sys.path.insert(0, str(Path(args.code_root).resolve()))
    from para.prolog import parse_example_line
    from para.reasoner import literal_to_text

    case_rows = []
    all_queries = set()
    aggregate = Counter()
    for manifest_name in args.manifest:
        manifest_path = Path(manifest_name).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        examples = [
            example
            for line in (Path(manifest["test_task"]) / "exs.pl").read_text(encoding="utf-8").splitlines()
            if (example := parse_example_line(line)) is not None
        ]
        positives = [example.literal for example in examples if example.positive]
        positive_first = {literal.args[0] for literal in positives if literal.args}
        positive_second = {literal.args[1] for literal in positives if len(literal.args) > 1}
        near_literals = [
            example.literal
            for example in examples
            if not example.positive
            and (
                (example.literal.args and example.literal.args[0] in positive_first)
                or (len(example.literal.args) > 1 and example.literal.args[1] in positive_second)
            )
        ]
        near_queries = {
            normalize(literal_to_text(literal, quote_constants=True))
            for literal in near_literals
        }
        all_queries.update(near_queries)
        reason_path = Path(manifest["reason_output"])
        reason = json.loads(reason_path.read_text(encoding="utf-8"))
        decisions = Counter()
        missing = []
        row_by_query = {
            normalize(str(row.get("query") or "")): row
            for row in reason.get("rows") or []
        }
        for query in sorted(near_queries):
            row = row_by_query.get(query)
            if row is None:
                missing.append(query)
                continue
            decisions[str(row.get("decision") or "MISSING")] += 1
        aggregate.update(decisions)
        case_rows.append(
            {
                "manifest": str(manifest_path),
                "split_name": manifest.get("split_name"),
                "test_task": manifest.get("test_task"),
                "near_miss_definition": "negative shares its first or second argument position with a held-out positive",
                "near_miss_negatives": len(near_queries),
                "decisions": dict(decisions),
                "missing_reason_rows": len(missing),
                "selection_sha256": hash_lines(near_queries),
            }
        )

    output = {
        "definition": "A held-out explicit negative is a near miss when its first argument occurs as the first argument of a held-out positive, or its second argument occurs as the second argument of a held-out positive.",
        "unique_negative_queries": len(all_queries),
        "model_decisions": sum(aggregate.values()),
        "decision_counts": dict(aggregate),
        "false_supported": int(aggregate.get("SUPPORTED", 0)),
        "selection_sha256": hash_lines(all_queries),
        "cases": case_rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if any(row["missing_reason_rows"] for row in case_rows):
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze explicit held-out near-miss negatives")
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-root", default=str(DEFAULT_CODE_ROOT))
    return parser


def normalize(text: str) -> str:
    return "".join(text.strip().rstrip(".").split())


def hash_lines(lines: set[str]) -> str:
    payload = "\n".join(sorted(lines)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    main()
