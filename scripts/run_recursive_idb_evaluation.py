#!/usr/bin/env python3
"""Evaluate bounded recursive IDB reasoning without claiming recursive learning.

The experiment has two parts:

1. A controlled graph with chains, branches, disconnected components, and a
   cycle. This isolates depth bounds, recursive proof construction, and cycle
   termination.
2. A Spring Framework inheritance graph. Ground truth is computed by an
   independent breadth-first transitive-closure implementation, while PARA
   answers the same queries by bounded backward chaining over two IDB
   predicates.

The recursive rules are supplied to the reasoner. This validates recursive
execution and proof accountability, not autonomous recursive-rule induction.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import asdict
import json
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Dict, Iterable, List, Sequence, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "src"
sys.path.insert(0, str(CODE_ROOT))

from para.models import Literal, PredicateSpec, TaskData  # noqa: E402
from para.prolog import load_task, literal_to_text, parse_literal  # noqa: E402
from para.reasoner import (  # noqa: E402
    BackwardChainingReasoner,
    ReasoningRule,
    SUPPORTED,
)


Edge = Tuple[str, str]
Query = Tuple[str, str, int]


def recursive_rules() -> List[ReasoningRule]:
    specs = [
        (
            "inheritsTransitively/2",
            "inheritsTransitively(X,Y) :- inheritsClass(X,Y) [1.000].",
            "recursive_base",
        ),
        (
            "inheritsTransitively/2",
            "inheritsTransitively(X,Y) :- inheritsClass(X,Z),inheritsTransitively(Z,Y) [1.000].",
            "recursive_step",
        ),
        (
            "architectureAncestor/2",
            "architectureAncestor(X,Y) :- inheritsTransitively(X,Y) [1.000].",
            "idb_composition",
        ),
    ]
    return [
        ReasoningRule(
            target=target,
            target_types=("class", "class"),
            rule=rule,
            path_programs=(),
            f1=1.0,
            precision=1.0,
            recall=1.0,
            threshold=0.8,
            source_summary=f"recursive_idb_evaluation#{source}",
            source_method="supplied_recursive_rule",
            status="ok",
        )
        for target, rule, source in specs
    ]


def make_task(edges: Iterable[Edge], name: str) -> TaskData:
    facts = [Literal("inheritsClass", edge) for edge in sorted(set(edges))]
    predicates = {
        "inheritsClass/2": PredicateSpec("inheritsClass", 2, ("class", "class")),
        "inheritsTransitively/2": PredicateSpec("inheritsTransitively", 2, ("class", "class")),
    }
    return TaskData(
        task_dir=name,
        target=PredicateSpec("architectureAncestor", 2, ("class", "class")),
        predicates=predicates,
        facts=facts,
        examples=[],
        bias_lines=[],
        max_vars=4,
        max_body=3,
        max_clauses=3,
    )


def build_adjacency(edges: Iterable[Edge]) -> Dict[str, Set[str]]:
    adjacency: Dict[str, Set[str]] = defaultdict(set)
    for source, target in edges:
        adjacency[source].add(target)
        adjacency.setdefault(target, set())
    return adjacency


def shortest_distances(adjacency: Dict[str, Set[str]], source: str, max_distance: int | None = None) -> Dict[str, int]:
    distances = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        if max_distance is not None and distances[node] >= max_distance:
            continue
        for neighbor in adjacency.get(node, set()):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[node] + 1
            queue.append(neighbor)
    distances.pop(source, None)
    return distances


def ground_truth_pairs(edges: Iterable[Edge], max_distance: int = 5) -> Dict[int, List[Edge]]:
    adjacency = build_adjacency(edges)
    by_distance: Dict[int, List[Edge]] = defaultdict(list)
    for source in sorted(adjacency):
        for target, distance in shortest_distances(adjacency, source, max_distance).items():
            if 1 <= distance <= max_distance:
                by_distance[distance].append((source, target))
    return by_distance


def sample_queries(
    edges: Iterable[Edge],
    max_distance: int,
    positives_per_distance: int,
    negative_count: int,
    seed: int,
) -> Tuple[List[Query], List[Tuple[str, str]]]:
    rng = random.Random(seed)
    edge_set = set(edges)
    adjacency = build_adjacency(edge_set)
    by_distance = ground_truth_pairs(edge_set, max_distance=max_distance)
    positives: List[Query] = []
    for distance, pairs in sorted(by_distance.items()):
        chosen = pairs if len(pairs) <= positives_per_distance else rng.sample(pairs, positives_per_distance)
        positives.extend((source, target, distance) for source, target in chosen)

    # Negative truth uses the complete transitive closure, not only the positive
    # sampling distance. This prevents long-but-reachable pairs from being
    # mislabeled as disconnected negatives.
    reachable = {
        (source, target)
        for source in adjacency
        for target in shortest_distances(adjacency, source)
    }
    nodes = sorted(adjacency)
    candidates = [
        (source, target)
        for source in nodes
        for target in nodes
        if source != target and (source, target) not in reachable and (source, target) not in edge_set
    ]
    negatives = candidates if len(candidates) <= negative_count else rng.sample(candidates, negative_count)
    return positives, negatives


def trace_stats(trace: Dict[str, object]) -> Dict[str, int]:
    if not trace:
        return {"edb": 0, "idb": 0, "recursive": 0, "max_depth": 0}
    if trace.get("kind") == "edb_fact":
        return {"edb": 1, "idb": 0, "recursive": 0, "max_depth": 0}
    rule = str(trace.get("rule") or "")
    result = {
        "edb": 0,
        "idb": 1,
        "recursive": int("inheritsTransitively(Z,Y)" in rule),
        "max_depth": int(trace.get("depth") or 0) + 1,
    }
    for child in trace.get("subproofs") or []:
        child_stats = trace_stats(child)
        result["edb"] += child_stats["edb"]
        result["idb"] += child_stats["idb"]
        result["recursive"] += child_stats["recursive"]
        result["max_depth"] = max(result["max_depth"], child_stats["max_depth"])
    return result


def audit_trace(trace: Dict[str, object], edge_set: Set[Edge], errors: List[str]) -> Dict[str, int]:
    kind = trace.get("kind")
    if kind == "edb_fact":
        try:
            fact = parse_literal(str(trace.get("fact") or ""))
            if fact.predicate != "inheritsClass" or fact.arity != 2 or tuple(fact.args) not in edge_set:
                errors.append(f"EDB leaf is not a tested inheritance edge: {trace.get('fact')}")
        except ValueError as exc:
            errors.append(f"invalid EDB leaf: {exc}")
        return {"trees": 0, "edb": 1, "idb": 0}
    if kind != "derived_rule":
        errors.append(f"unknown proof node kind: {kind!r}")
        return {"trees": 0, "edb": 0, "idb": 0}
    totals = {"trees": 0, "edb": 0, "idb": 1}
    if not trace.get("rule"):
        errors.append("derived proof node has no rule")
    for child in trace.get("subproofs") or []:
        child_totals = audit_trace(child, edge_set, errors)
        for key, value in child_totals.items():
            totals[key] += value
    return totals


def evaluate_queries(
    task: TaskData,
    positives: Sequence[Query],
    negatives: Sequence[Edge],
    max_depth: int,
) -> Dict[str, object]:
    engine = BackwardChainingReasoner(
        task=task,
        rules=recursive_rules(),
        threshold=0.8,
        max_depth=max_depth,
        max_proofs=1,
        max_states=5000,
        max_paths=1,
        max_edges_per_node=1,
        excluded_facts=set(),
        include_path_evidence=False,
    )
    positive_rows = []
    negative_rows = []
    runtimes: List[float] = []
    cycle_prunes = 0
    depth_limit_hits = 0
    for source, target, distance in positives:
        started = time.perf_counter()
        proofs = engine.prove(Literal("architectureAncestor", (source, target)))
        runtimes.append(time.perf_counter() - started)
        stats = dict(engine.last_search_stats)
        cycle_prunes += int(stats.get("cycle_prunes") or 0)
        depth_limit_hits += int(stats.get("depth_limit_hits") or 0)
        proof_stats = trace_stats(proofs[0].trace) if proofs else trace_stats({})
        positive_rows.append(
            {
                "source": source,
                "target": target,
                "distance": distance,
                "decision": SUPPORTED if proofs else "INCONCLUSIVE",
                "search_stats": stats,
                "proof_stats": proof_stats,
                "proof_trace": proofs[0].trace if proofs else None,
            }
        )
    for source, target in negatives:
        started = time.perf_counter()
        proofs = engine.prove(Literal("architectureAncestor", (source, target)))
        runtimes.append(time.perf_counter() - started)
        stats = dict(engine.last_search_stats)
        cycle_prunes += int(stats.get("cycle_prunes") or 0)
        depth_limit_hits += int(stats.get("depth_limit_hits") or 0)
        negative_rows.append(
            {
                "source": source,
                "target": target,
                "decision": SUPPORTED if proofs else "INCONCLUSIVE",
                "search_stats": stats,
                "proof_trace": proofs[0].trace if proofs else None,
            }
        )

    by_distance: Dict[str, Dict[str, float | int]] = {}
    for distance in sorted({row["distance"] for row in positive_rows}):
        rows = [row for row in positive_rows if row["distance"] == distance]
        supported = sum(row["decision"] == SUPPORTED for row in rows)
        by_distance[str(distance)] = {
            "queries": len(rows),
            "supported": supported,
            "recall": supported / len(rows) if rows else 0.0,
        }
    supported_positive = sum(row["decision"] == SUPPORTED for row in positive_rows)
    false_supported = sum(row["decision"] == SUPPORTED for row in negative_rows)
    proof_rows = [row for row in positive_rows if row["decision"] == SUPPORTED]
    audit_errors: List[str] = []
    audit_totals = {"trees": 0, "edb": 0, "idb": 0}
    audited_edges = {
        (fact.args[0], fact.args[1])
        for fact in task.facts
        if fact.predicate == "inheritsClass" and fact.arity == 2
    }
    for row in proof_rows:
        audit_totals["trees"] += 1
        totals = audit_trace(row["proof_trace"], audited_edges, audit_errors)
        audit_totals["edb"] += totals["edb"]
        audit_totals["idb"] += totals["idb"]
    proof_samples = []
    for distance in sorted({row["distance"] for row in proof_rows}):
        row = next(item for item in proof_rows if item["distance"] == distance)
        proof_samples.append(
            {
                "distance": distance,
                "source": row["source"],
                "target": row["target"],
                "proof_stats": row["proof_stats"],
                "proof_trace": row["proof_trace"],
            }
        )
    return {
        "max_depth": max_depth,
        "positive_queries": len(positive_rows),
        "supported_positive": supported_positive,
        "supported_recall": supported_positive / len(positive_rows) if positive_rows else 0.0,
        "negative_queries": len(negative_rows),
        "false_supported": false_supported,
        "negative_non_support": 1.0 - false_supported / len(negative_rows) if negative_rows else 1.0,
        "by_distance": by_distance,
        "cycle_prunes": cycle_prunes,
        "depth_limit_hits": depth_limit_hits,
        "mean_query_seconds": statistics.fmean(runtimes) if runtimes else 0.0,
        "maximum_proof_depth": max((row["proof_stats"]["max_depth"] for row in proof_rows), default=0),
        "maximum_idb_applications": max((row["proof_stats"]["idb"] for row in proof_rows), default=0),
        "recursive_rule_applications": sum(row["proof_stats"]["recursive"] for row in proof_rows),
        "proof_audit": {
            "audited_trees": audit_totals["trees"],
            "edb_fact_mentions": audit_totals["edb"],
            "idb_rule_applications": audit_totals["idb"],
            "error_count": len(audit_errors),
            "errors": audit_errors,
        },
        "proof_samples": proof_samples,
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "dependency_graph": engine.dependency_graph.to_json(),
    }


def controlled_edges() -> Set[Edge]:
    edges: Set[Edge] = set()
    chain = [f"chain_{index}" for index in range(7)]
    edges.update(zip(chain, chain[1:]))
    edges.update(
        {
            ("branch_root", "branch_left"),
            ("branch_root", "branch_right"),
            ("branch_left", "branch_leaf_a"),
            ("branch_right", "branch_leaf_b"),
            ("cycle_a", "cycle_b"),
            ("cycle_b", "cycle_c"),
            ("cycle_c", "cycle_a"),
            ("isolated_a", "isolated_b"),
        }
    )
    return edges


def load_spring_inheritance(task_dir: Path) -> Set[Edge]:
    task = load_task(task_dir)
    return {
        (fact.args[0], fact.args[1])
        for fact in task.facts
        if fact.predicate == "inheritsClass" and fact.arity == 2
    }


def compact_result(result: Dict[str, object]) -> Dict[str, object]:
    return {key: value for key, value in result.items() if key not in {"positive_rows", "negative_rows"}}


def write_summary(output_dir: Path, payload: Dict[str, object]) -> None:
    controlled = payload["controlled"]
    spring = payload["spring"]
    lines = [
        "# PARA Recursive IDB Execution Evaluation",
        "",
        "The rules are supplied to the reasoner. The experiment validates bounded",
        "recursive execution and proof traces, not autonomous recursive-rule learning.",
        "",
        "## Controlled graph",
        "",
        "| Max depth | Positive recall | False support | Cycle prunes | Max proof depth | Max IDB apps |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for result in controlled:
        lines.append(
            f"| {result['max_depth']} | {result['supported_recall']:.3f} | "
            f"{result['false_supported']} / {result['negative_queries']} | "
            f"{result['cycle_prunes']} | {result['maximum_proof_depth']} | "
            f"{result['maximum_idb_applications']} |"
        )
    lines.extend(
        [
            "",
            "## Spring Framework inheritance closure",
            "",
            f"- Direct inheritance edges: {payload['spring_edge_count']}",
            f"- Sampled positive queries: {spring['positive_queries']}",
            f"- Sampled disconnected negatives: {spring['negative_queries']}",
            f"- Supported recall: {spring['supported_recall']:.3f}",
            f"- False-supported negatives: {spring['false_supported']}",
            f"- Maximum proof depth: {spring['maximum_proof_depth']}",
            f"- Maximum IDB applications in one proof: {spring['maximum_idb_applications']}",
            f"- Recursive rule applications: {spring['recursive_rule_applications']}",
            f"- Audited recursive proof trees: {spring['proof_audit']['audited_trees']}",
            f"- Recursive proof audit errors: {spring['proof_audit']['error_count']}",
            f"- Mean query time: {spring['mean_query_seconds']:.6f}s",
            "",
            "### Recall by independent shortest-path distance",
            "",
            "| Distance | Queries | Supported | Recall |",
            "|---:|---:|---:|---:|",
        ]
    )
    for distance, values in spring["by_distance"].items():
        lines.append(
            f"| {distance} | {values['queries']} | {values['supported']} | {values['recall']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The experiment demonstrates multi-rule IDB composition, recursive backward",
            "chaining, bounded termination, and nested proof traces. It does not establish",
            "that the Planner autonomously learns recursive rules.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spring-task-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "recursive_idb",
    )
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--spring-positive-per-distance", type=int, default=50)
    parser.add_argument("--spring-negatives", type=int, default=250)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    controlled_edge_set = controlled_edges()
    controlled_positives, controlled_negatives = sample_queries(
        controlled_edge_set,
        max_distance=6,
        positives_per_distance=30,
        negative_count=80,
        seed=args.seed,
    )
    controlled_task = make_task(controlled_edge_set, "controlled_recursive_graph")
    controlled_results = [
        evaluate_queries(controlled_task, controlled_positives, controlled_negatives, max_depth=depth)
        for depth in (1, 2, 3, 4, 6, 8)
    ]

    spring_edges = load_spring_inheritance(args.spring_task_dir)
    spring_positives, spring_negatives = sample_queries(
        spring_edges,
        max_distance=5,
        positives_per_distance=args.spring_positive_per_distance,
        negative_count=args.spring_negatives,
        seed=args.seed,
    )
    spring_task = make_task(spring_edges, str(args.spring_task_dir))
    spring_result = evaluate_queries(
        spring_task,
        spring_positives,
        spring_negatives,
        max_depth=8,
    )
    payload = {
        "experiment": "bounded_recursive_idb_execution",
        "claim_boundary": {
            "validated": [
                "multi-rule IDB composition",
                "recursive backward chaining",
                "depth-bounded termination",
                "cycle pruning",
                "nested proof traces",
            ],
            "not_validated": "autonomous recursive-rule learning by the LLM Planner",
        },
        "rules": [asdict(rule) for rule in recursive_rules()],
        "controlled_edge_count": len(controlled_edge_set),
        "controlled": [compact_result(result) for result in controlled_results],
        "spring_task_dir": str(args.spring_task_dir),
        "spring_edge_count": len(spring_edges),
        "spring": compact_result(spring_result),
        "generated_at_unix": time.time(),
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_summary(args.output_dir, payload)
    print(json.dumps(compact_result(spring_result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
