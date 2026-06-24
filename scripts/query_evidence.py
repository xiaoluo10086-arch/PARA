#!/usr/bin/env python3
"""Deterministic query-centered graph views for PARA baselines and audits."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Set, Tuple


@dataclass(frozen=True, order=True)
class Edge:
    predicate: str
    source: str
    target: str

    def fact_text(self) -> str:
        return f"{self.predicate}({self.source},{self.target})"


@dataclass(frozen=True)
class Step:
    edge: Edge
    direction: str
    next_node: str

    def signature_token(self) -> str:
        return f"{self.edge.predicate}:{self.direction}"


class GraphIndex:
    """A compact bidirectional index over binary ground facts."""

    def __init__(self) -> None:
        self._adjacency: DefaultDict[str, List[Step]] = defaultdict(list)
        self.entity_types: DefaultDict[str, Set[str]] = defaultdict(set)
        self.fact_count = 0

    @classmethod
    def from_bk(cls, path: str | Path, allowed_predicates: Set[str] | None = None) -> "GraphIndex":
        index = cls()
        with Path(path).open("r", encoding="utf-8") as handle:
            for raw in handle:
                parsed = parse_ground_fact(raw)
                if parsed is None:
                    continue
                predicate, args = parsed
                if len(args) == 1:
                    index.entity_types[args[0]].add(predicate)
                    continue
                if len(args) != 2 or (allowed_predicates is not None and predicate not in allowed_predicates):
                    continue
                edge = Edge(predicate, args[0], args[1])
                index._adjacency[edge.source].append(Step(edge, "fwd", edge.target))
                index._adjacency[edge.target].append(Step(edge, "rev", edge.source))
                index.fact_count += 1
        for node, steps in index._adjacency.items():
            index._adjacency[node] = sorted(set(steps), key=lambda item: (item.edge.predicate, item.direction, item.next_node))
        return index

    def neighbors(self, node: str, per_node_cap: int = 80) -> Sequence[Step]:
        return self._adjacency.get(node, ())[:per_node_cap]

    def bounded_paths(
        self,
        source: str,
        target: str,
        *,
        max_depth: int = 4,
        max_paths: int = 8,
        per_node_cap: int = 80,
    ) -> List[Tuple[Step, ...]]:
        if source == target:
            return [tuple()]
        left_depth = max_depth // 2
        right_depth = max_depth - left_depth
        left = self._partial_paths(source, left_depth, per_node_cap)
        right = self._partial_paths(target, right_depth, per_node_cap)
        candidates: Dict[Tuple[str, ...], Tuple[Step, ...]] = {}
        for meeting in sorted(set(left) & set(right)):
            for left_path in left[meeting]:
                for right_path in right[meeting]:
                    combined = left_path + reverse_path(right_path)
                    if not combined or len(combined) > max_depth:
                        continue
                    signature = tuple(
                        f"{step.edge.fact_text()}@{step.direction}"
                        for step in combined
                    )
                    candidates.setdefault(signature, combined)
        ordered = sorted(
            candidates.values(),
            key=lambda path: (
                len(path),
                tuple(step.signature_token() for step in path),
                tuple(step.edge.fact_text() for step in path),
            ),
        )
        return ordered[:max_paths]

    def _partial_paths(
        self,
        root: str,
        max_depth: int,
        per_node_cap: int,
    ) -> Dict[str, List[Tuple[Step, ...]]]:
        current = [(root, tuple(), frozenset({root}))]
        by_node: DefaultDict[str, List[Tuple[Step, ...]]] = defaultdict(list)
        by_node[root].append(tuple())
        for _ in range(max_depth):
            following = []
            for node, path, visited in current:
                for step in self.neighbors(node, per_node_cap=per_node_cap):
                    if step.next_node in visited:
                        continue
                    next_path = path + (step,)
                    if len(by_node[step.next_node]) < 4:
                        by_node[step.next_node].append(next_path)
                    following.append((step.next_node, next_path, visited | {step.next_node}))
            current = following
        return by_node

    def evidence_view(
        self,
        source: str,
        target: str,
        *,
        max_depth: int = 4,
        max_paths: int = 8,
        max_facts: int = 32,
        per_node_cap: int = 80,
    ) -> Dict[str, object]:
        paths = self.bounded_paths(
            source,
            target,
            max_depth=max_depth,
            max_paths=max_paths,
            per_node_cap=per_node_cap,
        )
        selected: Dict[str, Edge] = {}
        for path in paths:
            for step in path:
                selected[step.edge.fact_text()] = step.edge
        if len(selected) < max_facts:
            for endpoint in (source, target):
                for step in self.neighbors(endpoint, per_node_cap=per_node_cap):
                    selected.setdefault(step.edge.fact_text(), step.edge)
                    if len(selected) >= max_facts:
                        break
                if len(selected) >= max_facts:
                    break
        fact_texts = sorted(selected)[:max_facts]
        signatures = sorted(
            {
                tuple(step.signature_token() for step in path)
                for path in paths
                if path
            }
        )
        return {
            "source": source,
            "target": target,
            "source_types": sorted(self.entity_types.get(source, ())),
            "target_types": sorted(self.entity_types.get(target, ())),
            "path_count": len(paths),
            "shortest_path_length": min((len(path) for path in paths), default=None),
            "path_signatures": [list(signature) for signature in signatures],
            "facts": fact_texts,
        }


def parse_ground_fact(line: str) -> Tuple[str, Tuple[str, ...]] | None:
    text = line.strip()
    if not text or text.startswith("%") or ":-" in text or not text.endswith("."):
        return None
    text = text[:-1]
    if "(" not in text or not text.endswith(")"):
        return None
    predicate, raw_args = text.split("(", 1)
    if predicate in {"pos", "neg", "head_pred", "body_pred", "type"}:
        return None
    args = tuple(part.strip().strip("'") for part in raw_args[:-1].split(","))
    if not all(args):
        return None
    return predicate.strip(), args


def reverse_path(path: Sequence[Step]) -> Tuple[Step, ...]:
    reversed_steps = []
    for step in reversed(path):
        previous = step.edge.source if step.direction == "fwd" else step.edge.target
        reversed_steps.append(
            Step(
                edge=step.edge,
                direction="rev" if step.direction == "fwd" else "fwd",
                next_node=previous,
            )
        )
    return tuple(reversed_steps)


def body_predicates_from_bias(path: str | Path) -> Set[str]:
    predicates: Set[str] = set()
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text.startswith("body_pred("):
            continue
        inside = text[len("body_pred(") :].rstrip(").")
        predicates.add(inside.split(",", 1)[0].strip())
    return predicates


def render_schema(path: str | Path) -> List[str]:
    rows = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if text.startswith("type("):
            rows.append(text)
    return rows


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
