#!/usr/bin/env python3
"""Evaluate explicit negatives structurally matched to positive held-out queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

from query_evidence import GraphIndex, body_predicates_from_bias, jaccard


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CODE_ROOT = ROOT / "src"


def main() -> None:
    args = build_parser().parse_args()
    sys.path.insert(0, str(Path(args.code_root).resolve()))
    from para.prolog import parse_example_line

    manifests = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.manifest]
    if not manifests:
        raise SystemExit("No manifests")
    first_task = Path(manifests[0]["test_task"])
    allowed = body_predicates_from_bias(first_task / "bias.pl")
    graph = GraphIndex.from_bk(first_task / "bk.pl", allowed_predicates=allowed)
    case_rows = []
    all_selected: Set[str] = set()
    aggregate = Counter()
    for manifest in manifests:
        task = Path(manifest["test_task"])
        examples = [
            example
            for line in (task / "exs.pl").read_text(encoding="utf-8").splitlines()
            if (example := parse_example_line(line)) is not None
        ]
        descriptors = [
            descriptor(graph, example.literal.args[0], example.literal.args[1], args)
            for example in examples
        ]
        positives = [value for value, example in zip(descriptors, examples) if example.positive]
        selected = []
        for value, example in zip(descriptors, examples):
            if example.positive:
                continue
            match = best_match(value, positives)
            if match is None:
                continue
            level, similarity = match
            if level < args.min_level:
                continue
            selected.append((example, value, level, similarity))
        reason = json.loads(Path(manifest["reason_output"]).read_text(encoding="utf-8"))
        by_query = {normalize(str(row["query"])): row for row in reason.get("rows", [])}
        decisions = Counter()
        detail_rows = []
        for example, value, level, similarity in selected:
            query = literal_text(example.literal)
            result = by_query.get(normalize(query))
            decision = str((result or {}).get("decision") or "MISSING")
            decisions[decision] += 1
            all_selected.add(query)
            detail_rows.append(
                {
                    "query": query,
                    "level": level_name(level),
                    "signature_similarity": similarity,
                    "shortest_path_length": value["length"],
                    "signatures": sorted(value["signatures"]),
                    "decision": decision,
                }
            )
        aggregate.update(decisions)
        case_rows.append(
            {
                "split_name": manifest.get("split_name"),
                "test_task": str(task),
                "selected_negatives": len(selected),
                "decision_counts": dict(decisions),
                "false_supported": decisions["SUPPORTED"],
                "rows": detail_rows,
            }
        )
    output = {
        "protocol": "structure_matched_near_miss",
        "levels": {
            "1": "same endpoint type pair and same shortest path length",
            "2": "level 1 plus non-zero path-signature overlap",
            "3": "level 1 plus an exact path signature shared with a positive",
        },
        "min_level": args.min_level,
        "graph_binary_fact_count": graph.fact_count,
        "unique_negative_queries": len(all_selected),
        "decision_counts": dict(aggregate),
        "false_supported": aggregate["SUPPORTED"],
        "false_support_rate": aggregate["SUPPORTED"] / sum(aggregate.values()) if aggregate else 0.0,
        "selection_sha256": hashlib.sha256("\n".join(sorted(all_selected)).encode("utf-8")).hexdigest(),
        "cases": case_rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(output_path.with_suffix(".md"), output)
    print(json.dumps({key: output[key] for key in ("unique_negative_queries", "decision_counts", "false_supported", "false_support_rate")}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-root", default=str(DEFAULT_CODE_ROOT))
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-paths", type=int, default=12)
    parser.add_argument("--edge-cap", type=int, default=80)
    parser.add_argument("--min-level", type=int, choices=(1, 2, 3), default=2)
    return parser


def descriptor(graph: GraphIndex, source: str, target: str, args: argparse.Namespace) -> Dict[str, object]:
    paths = graph.bounded_paths(
        source,
        target,
        max_depth=args.max_depth,
        max_paths=args.max_paths,
        per_node_cap=args.edge_cap,
    )
    signatures = {
        "/".join(step.signature_token() for step in path)
        for path in paths
        if path
    }
    return {
        "types": (tuple(sorted(graph.entity_types.get(source, ()))), tuple(sorted(graph.entity_types.get(target, ())))),
        "length": min((len(path) for path in paths), default=None),
        "signatures": signatures,
    }


def best_match(negative: Dict[str, object], positives: Sequence[Dict[str, object]]) -> Tuple[int, float] | None:
    best: Tuple[int, float] | None = None
    for positive in positives:
        if negative["types"] != positive["types"] or negative["length"] != positive["length"]:
            continue
        similarity = jaccard(negative["signatures"], positive["signatures"])
        exact = bool(set(negative["signatures"]) & set(positive["signatures"]))
        level = 3 if exact else (2 if similarity > 0 else 1)
        candidate = (level, similarity)
        if best is None or candidate > best:
            best = candidate
    return best


def level_name(level: int) -> str:
    return {1: "type_length", 2: "signature_overlap", 3: "exact_signature"}[level]


def literal_text(literal: object) -> str:
    return f"{literal.predicate}({','.join(literal.args)})"


def normalize(text: str) -> str:
    return "".join(text.strip().rstrip(".").split())


def write_report(path: Path, output: Dict[str, object]) -> None:
    lines = [
        "# Structure-Matched Near-Miss Stress",
        "",
        f"- Unique explicit negatives: {output['unique_negative_queries']}",
        f"- False SUPPORTED: {output['false_supported']}",
        f"- False-support rate: {output['false_support_rate']:.3f}",
        f"- Selection SHA256: `{output['selection_sha256']}`",
        "",
        "| Split | Selected | SUPPORTED | UNSUPPORTED | INCONCLUSIVE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in output["cases"]:
        counts = row["decision_counts"]
        lines.append(
            f"| `{row['split_name']}` | {row['selected_negatives']} | {counts.get('SUPPORTED', 0)} | "
            f"{counts.get('UNSUPPORTED', 0)} | {counts.get('INCONCLUSIVE', 0)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
