"""GraphRAG-style candidate generation over Popper background facts.

This module implements a lightweight, deterministic GraphRAG prototype for
PARA.  It treats `bk.pl` as a typed knowledge graph, retrieves evidence paths
between constants that appear in positive examples, and compiles frequent paths
into Horn-clause candidates.

The goal is not to replace Neo4j permanently.  Instead, this in-memory
implementation provides a fast experimental baseline that can later be mapped
to Cypher queries over the same predicate graph.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import heapq
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .models import Example, Literal, PredicateSpec, Rule, TaskData
from .models import Guidance
from .prolog import parse_rule


@dataclass(frozen=True)
class GraphEdge:
    """A directed edge in the retrieved fact graph.

    `reversed_edge=False` means the edge follows the original fact direction:
    p(arg0,arg1).  `reversed_edge=True` means the path traverses it backwards,
    which compiles to p(next,current) in the Horn body.
    """

    predicate: str
    reversed_edge: bool
    src: str
    dst: str


PathSignature = Tuple[Tuple[str, bool], ...]
RelationIndex = Dict[Tuple[str, str, bool], List[GraphEdge]]


class GraphRAGCandidateGenerator:
    """Generate candidate Horn clauses from graph-retrieved evidence paths."""

    def __init__(
        self,
        max_depth: int = 5,
        max_positive_examples: int = 30,
        max_paths_per_example: int = 40,
        max_edges_per_node: int = 300,
        max_candidates: int = 60,
        enable_pair_constraints: bool = True,
        traversal_strategy: str = "bfs",
        constraint_mode: str = "direct",
        min_path_support: int = 1,
        candidate_max_body: Optional[int] = None,
        focus_predicates: Optional[Sequence[str]] = None,
        seed_max_depth: int = 0,
        seed_max_paths_per_example: int = 8,
        expand_seed_on_miss: bool = True,
        planned_path_queries: Optional[Sequence[PathSignature]] = None,
        planned_queries_only: bool = False,
    ) -> None:
        # 最大检索跳数。overridesMethod 的主规则需要 5 个 body literal，
        # 因此默认允许到 5 跳。
        self.max_depth = max_depth
        self.max_positive_examples = max_positive_examples
        self.max_paths_per_example = max_paths_per_example
        self.max_edges_per_node = max_edges_per_node
        self.max_candidates = max_candidates
        self.enable_pair_constraints = enable_pair_constraints
        if constraint_mode not in {"direct", "attribute", "both", "none"}:
            raise ValueError(f"unsupported constraint_mode: {constraint_mode}")
        self.constraint_mode = "none" if not enable_pair_constraints else constraint_mode
        self.min_path_support = max(1, min_path_support)
        self.candidate_max_body = candidate_max_body
        self.focus_predicates = tuple(focus_predicates or ())
        self.seed_max_depth = max(0, seed_max_depth)
        self.seed_max_paths_per_example = max(1, seed_max_paths_per_example)
        self.expand_seed_on_miss = expand_seed_on_miss
        self.planned_path_queries = tuple(planned_path_queries or ())
        self.planned_queries_only = planned_queries_only
        if traversal_strategy not in {"bfs", "dfs"}:
            raise ValueError(f"unsupported traversal_strategy: {traversal_strategy}")
        self.traversal_strategy = traversal_strategy

    def generate(self, task: TaskData) -> Tuple[List[Rule], Dict[str, object]]:
        """Return candidate rules and retrieval diagnostics for a task."""

        started = time.perf_counter()
        binary_specs = {
            spec.name: spec
            for spec in task.predicates.values()
            if spec.arity == 2 and len(spec.types) == 2
        }
        if task.target.arity != 2 or len(task.target.types) != 2 or not binary_specs:
            return [], {"reason": "unsupported_target_or_no_binary_predicates"}
        body_budget = max(1, self.candidate_max_body or task.max_body)

        traversal_specs = {
            name: spec for name, spec in binary_specs.items() if not is_attribute_or_pair_constraint(name)
        }
        if not traversal_specs:
            traversal_specs = binary_specs
        adjacency = build_fact_graph(task.facts, traversal_specs)
        relation_index = build_relation_index(adjacency)
        adjacency_seconds = time.perf_counter() - started
        positives = [example for example in task.examples if example.positive][: self.max_positive_examples]
        target_start_type, target_end_type = task.target.types

        path_counter: Counter[PathSignature] = Counter()
        example_support: Dict[PathSignature, Set[int]] = defaultdict(set)
        pair_constraint_support: Counter[str] = Counter()
        attribute_constraint_support: Counter[str] = Counter()
        pair_constraint_predicates = {
            name
            for name, spec in binary_specs.items()
            if spec.types == (target_start_type, target_end_type)
        }
        constraint_index_started = time.perf_counter()
        fact_pairs = (
            fact_pair_index(task.facts, pair_constraint_predicates)
            if self.constraint_mode in {"direct", "both"}
            else set()
        )
        attribute_index = (
            attribute_fact_index(task.facts, binary_specs, task.target.types)
            if self.constraint_mode in {"attribute", "both"}
            else {}
        )
        constraint_index_seconds = time.perf_counter() - constraint_index_started

        retrieved_path_count = 0
        planned_query_path_count = 0
        seed_retrieved_path_count = 0
        seed_expanded_examples = 0
        retrieval_started = time.perf_counter()
        for ex_index, example in enumerate(positives):
            if len(example.literal.args) != 2:
                continue
            start, end = example.literal.args
            effective_max_depth = min(self.max_depth, body_budget)
            seed_depth = min(self.seed_max_depth, effective_max_depth)
            planned_paths = execute_path_queries(
                relation_index=relation_index,
                start=start,
                end=end,
                path_queries=self.planned_path_queries,
                max_paths=self.max_paths_per_example,
                max_edges_per_node=self.max_edges_per_node,
            )
            planned_query_path_count += len(planned_paths)
            if self.planned_queries_only:
                paths = planned_paths
            elif 0 < seed_depth < effective_max_depth:
                paths = retrieve_paths(
                    adjacency=adjacency,
                    start=start,
                    end=end,
                    max_depth=seed_depth,
                    max_paths=min(self.seed_max_paths_per_example, self.max_paths_per_example),
                    max_edges_per_node=self.max_edges_per_node,
                    traversal_strategy=self.traversal_strategy,
                    focus_predicates=self.focus_predicates,
                )
                seed_retrieved_path_count += len(paths)
                if not paths and self.expand_seed_on_miss:
                    seed_expanded_examples += 1
                    paths = retrieve_paths(
                        adjacency=adjacency,
                        start=start,
                        end=end,
                        max_depth=effective_max_depth,
                        max_paths=self.max_paths_per_example,
                        max_edges_per_node=self.max_edges_per_node,
                        traversal_strategy=self.traversal_strategy,
                        focus_predicates=self.focus_predicates,
                    )
                paths = dedupe_paths(planned_paths + paths)[: self.max_paths_per_example]
            else:
                paths = retrieve_paths(
                    adjacency=adjacency,
                    start=start,
                    end=end,
                    max_depth=effective_max_depth,
                    max_paths=self.max_paths_per_example,
                    max_edges_per_node=self.max_edges_per_node,
                    traversal_strategy=self.traversal_strategy,
                    focus_predicates=self.focus_predicates,
                )
                paths = dedupe_paths(planned_paths + paths)[: self.max_paths_per_example]
            retrieved_path_count += len(paths)
            seen_in_example: Set[PathSignature] = set()
            for path in paths:
                signature = tuple((edge.predicate, edge.reversed_edge) for edge in path)
                if not signature:
                    continue
                path_counter[signature] += 1
                seen_in_example.add(signature)
            for signature in seen_in_example:
                example_support[signature].add(ex_index)

            # 直接连接目标常量对的谓词可作为附加约束。例如 overridesMethod
            # 常需要 sameMethodName(A,B) 和 sameMethodArity(A,B)。
            if self.enable_pair_constraints:
                if self.constraint_mode in {"direct", "both"}:
                    for pred_name, spec in binary_specs.items():
                        if pred_name not in pair_constraint_predicates:
                            continue
                        if (pred_name, start, end) in fact_pairs:
                            pair_constraint_support[pred_name] += 1
                if self.constraint_mode in {"attribute", "both"}:
                    for pred_name, by_subject in attribute_index.items():
                        if by_subject.get(start, set()).intersection(by_subject.get(end, set())):
                            attribute_constraint_support[pred_name] += 1
        retrieval_seconds = time.perf_counter() - retrieval_started

        common_pair_constraints = [
            name
            for name, count in pair_constraint_support.most_common()
            if positives and count / len(positives) >= 0.5
        ]
        common_attribute_constraints = [
            name
            for name, count in attribute_constraint_support.most_common()
            if positives and count / len(positives) >= 0.5
        ]

        candidate_compile_started = time.perf_counter()
        candidates: List[str] = []
        seen_rules: Set[str] = set()
        ranked_signatures = sorted(
            (sig for sig in path_counter if len(example_support[sig]) >= self.min_path_support),
            key=lambda sig: (
                -len(example_support[sig]),
                -signature_semantic_score(task.target.name, sig),
                -path_counter[sig],
                len(sig),
                ",".join(f"{name}:{int(rev)}" for name, rev in sig),
            ),
        )

        for signature in ranked_signatures:
            if len(signature) > body_budget:
                continue
            body_literals = signature_to_body(signature)
            variants = [body_literals]

            # 附加直接 pair constraints，但不让规则体超过任务 bias 限制。
            remaining = max(0, body_budget - len(body_literals))
            if remaining and common_pair_constraints:
                constraints = [f"{name}(A,B)" for name in common_pair_constraints[:remaining]]
                if constraints:
                    variants.append(body_literals + constraints)
            if common_attribute_constraints:
                attr_variants = attribute_constraint_variants(
                    body_literals,
                    common_attribute_constraints,
                    body_budget,
                )
                variants.extend(attr_variants)

            for body in variants:
                body = dedupe_body_literals(body)
                text = f"{task.target.name}(A,B) :- {','.join(body)}."
                if text not in seen_rules:
                    candidates.append(text)
                    seen_rules.add(text)
                if len(candidates) >= self.max_candidates:
                    break
            if len(candidates) >= self.max_candidates:
                break

        rules = [parse_rule(text, source="graph_rag") for text in candidates]
        candidate_compile_seconds = time.perf_counter() - candidate_compile_started
        diagnostics = {
            "positive_examples_used": len(positives),
            "retrieved_path_count": retrieved_path_count,
            "seed_retrieved_path_count": seed_retrieved_path_count,
            "seed_expanded_examples": seed_expanded_examples,
            "planned_query_path_count": planned_query_path_count,
            "unique_path_signatures": len(path_counter),
            "eligible_path_signatures": len(ranked_signatures),
            "min_path_support": self.min_path_support,
            "pair_constraint_support": dict(pair_constraint_support),
            "attribute_constraint_support": dict(attribute_constraint_support),
            "common_pair_constraints": common_pair_constraints,
            "common_attribute_constraints": common_attribute_constraints,
            "candidate_count": len(rules),
            "traversal_strategy": self.traversal_strategy,
            "enable_pair_constraints": self.enable_pair_constraints,
            "constraint_mode": self.constraint_mode,
            "candidate_max_body": body_budget,
            "focus_predicates": list(self.focus_predicates),
            "traversal_predicates": sorted(traversal_specs),
            "excluded_constraint_predicates": sorted(set(binary_specs).difference(traversal_specs)),
            "pair_constraint_predicates": sorted(pair_constraint_predicates),
            "seed_max_depth": self.seed_max_depth,
            "seed_max_paths_per_example": self.seed_max_paths_per_example,
            "expand_seed_on_miss": self.expand_seed_on_miss,
            "planned_path_queries": [
                [{"predicate": predicate, "direction": "rev" if reversed_edge else "fwd"} for predicate, reversed_edge in query]
                for query in self.planned_path_queries
            ],
            "planned_queries_only": self.planned_queries_only,
            "timings_seconds": {
                "adjacency": round(adjacency_seconds, 6),
                "constraint_index": round(constraint_index_seconds, 6),
                "retrieval": round(retrieval_seconds, 6),
                "candidate_compile": round(candidate_compile_seconds, 6),
                "total": round(time.perf_counter() - started, 6),
            },
            "top_path_signatures": [
                {
                    "signature": [
                        {"predicate": pred, "direction": "rev" if rev else "fwd"}
                        for pred, rev in signature
                    ],
                    "path_count": path_counter[signature],
                    "example_support": len(example_support[signature]),
                }
                for signature in ranked_signatures[:10]
            ],
        }
        return rules, diagnostics


class GraphRAGGuideProvider:
    """PARA guide provider backed by graph evidence retrieval.

    This provider is the drop-in replacement for the DTE/heuristic candidate
    generator in end-to-end PARA experiments.  It does not enumerate all typed
    paths.  Instead, it retrieves evidence paths from the task's BK graph and
    returns the resulting Horn candidates through the same Guidance interface.
    """

    def __init__(self, generator: Optional[GraphRAGCandidateGenerator] = None) -> None:
        self.generator = generator or GraphRAGCandidateGenerator()

    def guide(
        self,
        task: TaskData,
        objective: str,
        predicate_budget: int,
        feedback: Optional[str] = None,
    ) -> Guidance:
        candidates, diagnostics = self.generator.generate(task)
        candidate_predicates = []
        seen = set()
        for rule in candidates:
            for literal in rule.body:
                if literal.predicate not in seen:
                    seen.add(literal.predicate)
                    candidate_predicates.append(literal.predicate)

        ranked = rank_predicates_from_graph(task, candidate_predicates)
        budget = max(predicate_budget, len(candidate_predicates))
        selected = ranked[: min(len(ranked), budget)]
        for name in candidate_predicates:
            if name not in selected:
                selected.append(name)

        return Guidance(
            ranked_predicates=ranked,
            selected_predicates=selected,
            candidate_rules=candidates,
            max_vars=task.max_vars,
            max_body=task.max_body,
            max_clauses=task.max_clauses,
            confidence=0.72,
            rationale=(
                "GraphRAG retrieved evidence paths from positive example pairs "
                f"and compiled {len(candidates)} graph-derived candidates. "
                f"Diagnostics: {diagnostics}"
            ),
        )


def build_fact_graph(
    facts: Iterable[Literal],
    binary_specs: Mapping[str, PredicateSpec],
) -> Dict[str, List[GraphEdge]]:
    """Build a bidirectional graph from binary BK facts."""

    adjacency: Dict[str, List[GraphEdge]] = defaultdict(list)
    for fact in facts:
        if fact.predicate not in binary_specs or fact.arity != 2:
            continue
        left, right = fact.args
        adjacency[left].append(GraphEdge(fact.predicate, False, left, right))
        adjacency[right].append(GraphEdge(fact.predicate, True, right, left))

    # 确定性排序，保证实验可复现。
    return {node: sorted(edges, key=lambda e: (e.predicate, e.reversed_edge, e.dst)) for node, edges in adjacency.items()}


def build_relation_index(adjacency: Mapping[str, Sequence[GraphEdge]]) -> RelationIndex:
    """Index 1-hop graph edges by entity, predicate, and direction."""

    output: RelationIndex = defaultdict(list)
    for node, edges in adjacency.items():
        for edge in edges:
            output[(node, edge.predicate, edge.reversed_edge)].append(edge)
    return {
        key: sorted(edges, key=lambda edge: edge.dst)
        for key, edges in output.items()
    }


def execute_path_queries(
    relation_index: Mapping[Tuple[str, str, bool], Sequence[GraphEdge]],
    start: str,
    end: str,
    path_queries: Sequence[PathSignature],
    max_paths: int,
    max_edges_per_node: int,
) -> List[List[GraphEdge]]:
    """Execute agent-planned relation programs over the 1-hop index."""

    output: List[List[GraphEdge]] = []
    for query in path_queries:
        if not query:
            continue
        frontier: List[Tuple[str, List[GraphEdge], Set[str]]] = [(start, [], {start})]
        for predicate, reversed_edge in query:
            next_frontier: List[Tuple[str, List[GraphEdge], Set[str]]] = []
            for current, path, visited in frontier:
                edges = relation_index.get((current, predicate, reversed_edge), ())[:max_edges_per_node]
                for edge in edges:
                    if edge.dst in visited and edge.dst != end:
                        continue
                    next_frontier.append((edge.dst, path + [edge], set(visited) | {edge.dst}))
            frontier = next_frontier
            if not frontier:
                break
        for current, path, _visited in frontier:
            if current != end:
                continue
            output.append(path)
            if len(output) >= max_paths:
                return dedupe_paths(output)
    return dedupe_paths(output)


def dedupe_paths(paths: Sequence[Sequence[GraphEdge]]) -> List[List[GraphEdge]]:
    """Deduplicate concrete evidence paths while preserving execution order."""

    output: List[List[GraphEdge]] = []
    seen = set()
    for path in paths:
        key = tuple((edge.predicate, edge.reversed_edge, edge.src, edge.dst) for edge in path)
        if key in seen:
            continue
        seen.add(key)
        output.append(list(path))
    return output


def rank_predicates_from_graph(task: TaskData, preferred: Sequence[str]) -> List[str]:
    """Rank body predicates with GraphRAG-used predicates first."""

    preferred_set = set(preferred)
    scored = []
    for key, spec in task.predicates.items():
        score = 0
        if spec.name in preferred_set:
            score += 100
        if spec.arity == 2:
            score += 5
        if len(spec.types) == spec.arity and set(spec.types).intersection(task.target.types):
            score += 3
        scored.append((score, spec.name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [name for _score, name in scored]


def fact_pair_index(
    facts: Iterable[Literal],
    predicate_names: Optional[Set[str]] = None,
) -> Set[Tuple[str, str, str]]:
    """Index binary fact pairs as (predicate,arg0,arg1)."""

    if predicate_names is not None and not predicate_names:
        return set()
    output: Set[Tuple[str, str, str]] = set()
    for fact in facts:
        if fact.arity == 2 and (predicate_names is None or fact.predicate in predicate_names):
            output.add((fact.predicate, fact.args[0], fact.args[1]))
    return output


def attribute_fact_index(
    facts: Iterable[Literal],
    binary_specs: Mapping[str, PredicateSpec],
    target_types: Tuple[str, ...],
) -> Dict[str, Dict[str, Set[str]]]:
    """Index attribute facts usable as lazy equality constraints.

    For method-level targets, facts such as `methodName(Method,Name)` and
    `methodArity(Method,Arity)` can replace large materialized pair facts like
    `sameMethodName(Method1,Method2)`.  A rule body
    `methodName(A,V), methodName(B,V)` is logically equivalent to checking that
    A and B share the same method name, while requiring only O(N) attribute
    facts instead of O(N^2) pair facts.
    """

    if len(target_types) != 2 or target_types[0] != target_types[1]:
        return {}
    target_type = target_types[0]
    attribute_predicates = {
        name
        for name, spec in binary_specs.items()
        if (
            spec.arity == 2
            and len(spec.types) == 2
            and spec.types[0] == target_type
            and spec.types[1] != target_type
            and is_attribute_or_pair_constraint(name)
        )
    }
    output: Dict[str, Dict[str, Set[str]]] = {name: defaultdict(set) for name in attribute_predicates}
    for fact in facts:
        if fact.predicate in output and fact.arity == 2:
            output[fact.predicate][fact.args[0]].add(fact.args[1])
    return {name: dict(by_subject) for name, by_subject in output.items()}


def attribute_constraint_variants(
    base_body: List[str],
    attribute_constraints: Sequence[str],
    max_body: int,
) -> List[List[str]]:
    """Build equality-style attribute constraint variants for A/B."""

    variants: List[List[str]] = []
    available_slots = max_body - len(base_body)
    if available_slots < 2:
        return variants
    selected = list(attribute_constraints)[: available_slots // 2]
    if not selected:
        return variants

    # Add each attribute independently, then one combined variant when it fits.
    combined = list(base_body)
    for idx, name in enumerate(selected, start=1):
        var = f"C{idx}"
        lits = [f"{name}(A,{var})", f"{name}(B,{var})"]
        variants.append(base_body + lits)
        combined.extend(lits)
    if len(combined) <= max_body and len(selected) > 1:
        variants.append(combined)
    return variants


def dedupe_body_literals(body: Sequence[str]) -> List[str]:
    """Remove exact duplicate body literals while preserving order."""

    output: List[str] = []
    seen = set()
    for literal in body:
        if literal in seen:
            continue
        seen.add(literal)
        output.append(literal)
    return output


def is_attribute_or_pair_constraint(predicate_name: str) -> bool:
    """Return True for predicates better used as constraints than path edges."""

    lower = predicate_name.lower()
    return any(word in lower for word in ("same", "name", "arity"))


def signature_semantic_score(target_name: str, signature: PathSignature) -> int:
    """Score a path signature with broad architecture-rule priors.

    This is not an oracle template: it never returns a complete rule for a
    target predicate.  It only nudges the ranking toward evidence predicates
    whose names match the target's coarse semantics, so the fixed candidate
    budget is spent on more plausible graph paths.
    """

    target = target_name.lower()
    score = 0
    names = [name.lower() for name, _rev in signature]
    for name in names:
        if ("call" in target or "invoke" in target) and "call" in name:
            score += 5
        if ("call" in target or "invoke" in target) and "method" in name:
            score += 2
        if ("override" in target or "inherit" in target) and ("inherit" in name or "extend" in name):
            score += 6
        if ("override" in target or "inherit" in target) and "method" in name:
            score += 2
        if ("allowed" in target or "use" in target or "depend" in target) and ("import" in name or "depend" in name):
            score += 5
        if ("allowed" in target or "use" in target or "depend" in target) and "class" in name:
            score += 1
    return score


def retrieve_paths(
    adjacency: Mapping[str, Sequence[GraphEdge]],
    start: str,
    end: str,
    max_depth: int,
    max_paths: int,
    max_edges_per_node: int,
    traversal_strategy: str = "bfs",
    focus_predicates: Sequence[str] = (),
) -> List[List[GraphEdge]]:
    """Retrieve simple paths between two constants with bounded BFS.

    BFS is important here: candidate rules should be based on the shortest
    explanatory evidence chains first.  A depth-first traversal can get trapped
    in package/class detours and miss the direct method-call path before the
    per-example path cap is reached.
    """

    focus = tuple(dict.fromkeys(focus_predicates))
    if focus:
        return retrieve_paths_focused(
            adjacency=adjacency,
            start=start,
            end=end,
            max_depth=max_depth,
            max_paths=max_paths,
            max_edges_per_node=max_edges_per_node,
            focus_predicates=focus,
        )

    paths: List[List[GraphEdge]] = []
    frontier = deque([(start, [], {start})])
    while frontier and len(paths) < max_paths:
        if traversal_strategy == "dfs":
            current, current_path, visited = frontier.pop()
        else:
            current, current_path, visited = frontier.popleft()
        if current_path and current == end:
            paths.append(current_path)
            continue
        if len(current_path) >= max_depth:
            continue
        for edge in adjacency.get(current, ())[:max_edges_per_node]:
            if edge.dst in visited and edge.dst != end:
                continue
            frontier.append((edge.dst, current_path + [edge], set(visited) | {edge.dst}))
    return paths


def retrieve_paths_focused(
    adjacency: Mapping[str, Sequence[GraphEdge]],
    start: str,
    end: str,
    max_depth: int,
    max_paths: int,
    max_edges_per_node: int,
    focus_predicates: Sequence[str],
) -> List[List[GraphEdge]]:
    """Retrieve paths with soft priority for focus predicates.

    This is not a hard constraint.  Paths that do not contain focus predicates
    may still be returned, but the queue explores paths with higher predicate
    coverage first.  This lets an agent guide exploration without deleting the
    deterministic GraphRAG fallback behavior.
    """

    paths: List[List[GraphEdge]] = []
    focus_set = set(focus_predicates)
    queue: List[Tuple[int, int, int, int, str, List[GraphEdge], Set[str]]] = []
    serial = 0
    heapq.heappush(queue, (0, 0, 0, serial, start, [], {start}))

    while queue and len(paths) < max_paths:
        _priority, _introduced, _depth, _serial, current, current_path, visited = heapq.heappop(queue)
        if current_path and current == end:
            paths.append(current_path)
            continue
        if len(current_path) >= max_depth:
            continue

        missing = focus_set.difference(edge.predicate for edge in current_path)
        edges = sorted(
            adjacency.get(current, ())[:max_edges_per_node],
            key=lambda edge: (
                0 if edge.predicate in missing else 1,
                0 if edge.predicate in focus_set else 1,
                edge.predicate,
                edge.reversed_edge,
                edge.dst,
            ),
        )
        for edge in edges:
            if edge.dst in visited and edge.dst != end:
                continue
            new_path = current_path + [edge]
            coverage = len({item.predicate for item in new_path}.intersection(focus_set))
            introduces_focus = 1 if edge.predicate in missing else 0
            serial += 1
            # Lower heap priority is popped first.  Coverage and newly
            # introduced focus predicates are negative so higher is better.
            heapq.heappush(
                queue,
                (
                    -coverage,
                    -introduces_focus,
                    len(new_path),
                    serial,
                    edge.dst,
                    new_path,
                    set(visited) | {edge.dst},
                ),
            )
    return paths


def signature_to_body(signature: PathSignature) -> List[str]:
    """Compile a path signature into Prolog body literals."""

    vars_by_position = ["A"] + [f"V{i}" for i in range(1, len(signature))] + ["B"]
    body: List[str] = []
    for idx, (predicate, reversed_edge) in enumerate(signature):
        left = vars_by_position[idx]
        right = vars_by_position[idx + 1]
        if reversed_edge:
            body.append(f"{predicate}({right},{left})")
        else:
            body.append(f"{predicate}({left},{right})")
    return body
