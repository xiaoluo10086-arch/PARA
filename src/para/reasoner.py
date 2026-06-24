"""Query-time architecture-relation reasoning over learned PARA rules.

The reasoner is intentionally a thin layer over the existing symbolic
components.  It does not learn new rules and it does not ask an LLM for a final
answer.  Instead, it applies accepted rules to a concrete relation query and
returns an auditable three-valued judgement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .evaluate import FactIndex, build_fact_index, indexed_candidates, matching_candidates, unify_literal
from .graph_rag import (
    GraphEdge,
    PathSignature,
    build_fact_graph,
    build_relation_index,
    execute_path_queries,
    is_attribute_or_pair_constraint,
)
from .models import Example, Literal, Rule, TaskData, is_variable
from .prolog import literal_to_text, load_task, parse_literal, parse_rule, rule_to_text


SUPPORTED = "SUPPORTED"
UNSUPPORTED = "UNSUPPORTED"
INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class ReasoningRule:
    """A learned rule plus the metadata needed for query-time reasoning."""

    target: str
    target_types: Tuple[str, ...]
    rule: str
    path_programs: Tuple[PathSignature, ...]
    f1: float
    precision: float
    recall: float
    threshold: float
    source_summary: str
    source_method: str
    status: str


@dataclass(frozen=True)
class ReasoningEvidence:
    """Concrete evidence for one rule application."""

    rule: str
    rule_f1: float
    rule_precision: float
    rule_recall: float
    body_facts: Tuple[str, ...]
    path_programs: Tuple[Tuple[Tuple[str, str], ...], ...]
    path_evidence: Tuple[Tuple[str, ...], ...]
    source_summary: str
    proof_score: float = 0.0
    proof_trace: Optional[Dict[str, object]] = None


@dataclass(frozen=True)
class ProofResult:
    """One bounded backward-chaining proof with its propagated binding."""

    binding: Dict[str, str]
    score: float
    trace: Dict[str, object]


@dataclass
class RuleDependencyGraph:
    """Static dependency graph for exported reasoning rules."""

    idb_predicates: Set[str]
    edb_predicates: Set[str]
    dependencies: Dict[str, List[Dict[str, str]]]
    recursive_predicates: Set[str]

    def to_json(self) -> Dict[str, object]:
        return {
            "idb_predicates": sorted(self.idb_predicates),
            "edb_predicates": sorted(self.edb_predicates),
            "dependencies": self.dependencies,
            "recursive_predicates": sorted(self.recursive_predicates),
        }


def export_rule_library(
    summaries: Sequence[str | Path],
    output: str | Path,
    min_f1: float = 0.8,
    factor_path_helpers: bool = False,
) -> Dict[str, object]:
    """Export accepted PARA summaries as a reusable reasoning library."""

    rules: List[Dict[str, object]] = []
    for summary in expand_summary_paths(summaries):
        data = json.loads(summary.read_text(encoding="utf-8"))
        item = reasoning_rule_from_summary(data, summary, min_f1=min_f1)
        if item is None:
            continue
        rules.append(rule_to_json(item))
    rules = dedupe_rule_jsons(rules)
    if factor_path_helpers:
        rules = factor_rule_jsons_into_helpers(rules)
    dependency_graph = build_dependency_graph([rule_from_json(item) for item in rules])
    payload = {
        "schema_version": 2,
        "library_kind": "para_reasoning_rule_library",
        "min_f1": min_f1,
        "factor_path_helpers": factor_path_helpers,
        "rule_count": len(rules),
        "dependency_graph": dependency_graph.to_json(),
        "agent_role": {
            "planner": "proposes executable path programs and rule-level subgoals",
            "refiner": "uses proof or verifier failures to revise path programs",
            "symbolic_engine": "executes bounded backward chaining and validates proof steps",
        },
        "rules": rules,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def dedupe_rule_jsons(rules: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Merge duplicate exported rules while preserving alternative path programs."""

    merged: Dict[Tuple[str, str], Dict[str, object]] = {}
    for item in rules:
        key = (str(item.get("target") or ""), str(item.get("rule") or ""))
        if key not in merged:
            merged[key] = dict(item)
            continue
        current = merged[key]
        current_paths = {json.dumps(path, sort_keys=True) for path in current.get("path_programs", [])}
        for path in item.get("path_programs", []) or []:
            encoded = json.dumps(path, sort_keys=True)
            if encoded not in current_paths:
                current.setdefault("path_programs", []).append(path)
                current_paths.add(encoded)
        for metric in ("f1", "precision", "recall"):
            current[metric] = max(float(current.get(metric) or 0.0), float(item.get(metric) or 0.0))
        sources = set(str(source) for source in current.get("source_summaries", []) or [])
        if current.get("source_summary"):
            sources.add(str(current.get("source_summary")))
        if item.get("source_summary"):
            sources.add(str(item.get("source_summary")))
        current["source_summaries"] = sorted(sources)
        if sources:
            current["source_summary"] = sorted(sources)[0]
    return sorted(merged.values(), key=lambda item: (str(item.get("target") or ""), str(item.get("rule") or "")))


def factor_rule_jsons_into_helpers(rules: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Factor accepted rule bodies into synthetic IDB helper predicates.

    This is an optional reasoning-layer normalization. It preserves the original
    rule semantics while forcing proof construction to compose at least two
    rules: target predicate -> helper predicate -> EDB evidence.
    """

    output: List[Dict[str, object]] = []
    for item in rules:
        rule_text = str(item.get("rule") or "")
        try:
            rule = parse_rule(rule_text, source=str(item.get("source_method") or "reasoning_factor"))
        except ValueError:
            output.append(dict(item))
            continue
        if not rule.body or rule.head.arity < 1:
            output.append(dict(item))
            continue
        helper_name = helper_predicate_name(rule)
        helper_head = Literal(helper_name, rule.head.args)
        confidence = f" [{float(item.get('f1') or 0.0):.3f}]"
        helper_rule_text = (
            f"{literal_to_text(helper_head)} :- "
            + ",".join(literal_to_text(literal) for literal in rule.body)
            + confidence
            + "."
        )
        target_rule_text = (
            f"{literal_to_text(rule.head)} :- {literal_to_text(helper_head)}"
            + confidence
            + "."
        )

        target_item = dict(item)
        target_item["rule"] = target_rule_text
        target_item["path_programs"] = []
        target_item["source_method"] = f"{item.get('source_method') or 'ashrl'}+factor_target"

        helper_item = dict(item)
        helper_item["target"] = f"{helper_name}/{rule.head.arity}"
        helper_item["rule"] = helper_rule_text
        helper_item["source_method"] = f"{item.get('source_method') or 'ashrl'}+factor_helper"
        helper_item["source_summary"] = str(item.get("source_summary") or "") + "#helper"
        if "source_summaries" in helper_item:
            helper_item["source_summaries"] = [str(source) + "#helper" for source in helper_item.get("source_summaries") or []]

        output.extend([target_item, helper_item])
    return dedupe_rule_jsons(output)


def helper_predicate_name(rule: Rule) -> str:
    digest = hashlib.sha1(rule_to_text(rule, with_confidence=True).encode("utf-8")).hexdigest()[:10]
    return f"ashrl_helper_{rule.head.predicate}_{digest}"


def reason_query(
    task_dir: str | Path,
    rule_library: str | Path,
    query: str,
    threshold: float = 0.8,
    max_paths: int = 5,
    max_edges_per_node: int = 120,
    max_depth: int = 4,
    max_proofs: int = 5,
    max_states: int = 2000,
) -> Dict[str, object]:
    """Apply learned executable rules to a concrete architecture relation."""

    task = load_task(task_dir)
    query_literal = parse_query_literal(query, task)
    return reason_literal(
        task=task,
        rules=load_rule_library(rule_library),
        query_literal=query_literal,
        threshold=threshold,
        max_paths=max_paths,
        max_edges_per_node=max_edges_per_node,
        max_depth=max_depth,
        max_proofs=max_proofs,
        max_states=max_states,
    )


def reason_counterfactual(
    task_dir: str | Path,
    rule_library: str | Path,
    query: str,
    threshold: float = 0.8,
    max_paths: int = 5,
    max_edges_per_node: int = 120,
    max_depth: int = 4,
    max_proofs: int = 5,
    max_states: int = 2000,
    max_ablation_facts: int = 8,
) -> Dict[str, object]:
    """Ablate proof facts and re-run bounded reasoning for one query."""

    task = load_task(task_dir)
    rules = load_rule_library(rule_library)
    query_literal = parse_query_literal(query, task)
    return reason_counterfactual_literal(
        task=task,
        rules=rules,
        query_literal=query_literal,
        threshold=threshold,
        max_paths=max_paths,
        max_edges_per_node=max_edges_per_node,
        max_depth=max_depth,
        max_proofs=max_proofs,
        max_states=max_states,
        max_ablation_facts=max_ablation_facts,
    )


def reason_counterfactual_examples(
    task_dir: str | Path,
    rule_library: str | Path,
    threshold: float = 0.8,
    max_paths: int = 5,
    max_edges_per_node: int = 120,
    max_depth: int = 4,
    max_proofs: int = 5,
    max_states: int = 2000,
    max_ablation_facts: int = 8,
    max_queries: int = 20,
) -> Dict[str, object]:
    """Run counterfactual evidence reasoning over positive task examples."""

    task = load_task(task_dir)
    rules = load_rule_library(rule_library)
    positives = [example for example in task.examples if example.positive]
    if max_queries > 0:
        positives = positives[:max_queries]
    rows: List[Dict[str, object]] = []
    for example in positives:
        result = reason_counterfactual_literal(
            task=task,
            rules=rules,
            query_literal=example.literal,
            threshold=threshold,
            max_paths=max_paths,
            max_edges_per_node=max_edges_per_node,
            max_depth=max_depth,
            max_proofs=max_proofs,
            max_states=max_states,
            max_ablation_facts=max_ablation_facts,
        )
        summary = result.get("summary") or {}
        all_removed = result.get("all_selected_facts_removed") or {}
        rows.append(
            {
                "query": result.get("query"),
                "initial_decision": result.get("initial_decision"),
                "initial_evidence_count": result.get("initial_evidence_count"),
                "initial_proof_score": result.get("initial_proof_score"),
                "ablated_facts": summary.get("ablated_facts", 0),
                "critical_facts": summary.get("critical_facts", 0),
                "alternative_supported": summary.get("alternative_supported", 0),
                "inconclusive_after_ablation": summary.get("inconclusive_after_ablation", 0),
                "all_selected_removed_effect": all_removed.get("effect"),
                "runtime_seconds": result.get("runtime_seconds", 0.0),
            }
        )
    supported_rows = [row for row in rows if row.get("initial_decision") == SUPPORTED]
    ablated = sum(int(row.get("ablated_facts") or 0) for row in supported_rows)
    critical = sum(int(row.get("critical_facts") or 0) for row in supported_rows)
    alternative = sum(int(row.get("alternative_supported") or 0) for row in supported_rows)
    all_alternative = sum(1 for row in supported_rows if row.get("all_selected_removed_effect") == "STILL_SUPPORTED_BY_ALTERNATIVE_PROOF")
    total_runtime = sum(float(row.get("runtime_seconds") or 0.0) for row in rows)
    return {
        "task_dir": str(task_dir),
        "target": task.target.signature,
        "queries": len(rows),
        "initial_supported": len(supported_rows),
        "initial_supported_rate": len(supported_rows) / len(rows) if rows else 0.0,
        "ablated_facts": ablated,
        "critical_facts": critical,
        "alternative_supported": alternative,
        "critical_fact_rate": critical / ablated if ablated else 0.0,
        "alternative_fact_rate": alternative / ablated if ablated else 0.0,
        "all_selected_alternative_rate": all_alternative / len(supported_rows) if supported_rows else 0.0,
        "total_runtime_seconds": total_runtime,
        "mean_runtime_seconds": total_runtime / len(rows) if rows else 0.0,
        "rows": rows,
    }


def reason_counterfactual_literal(
    task: TaskData,
    rules: Sequence[ReasoningRule],
    query_literal: Literal,
    threshold: float = 0.8,
    max_paths: int = 5,
    max_edges_per_node: int = 120,
    max_depth: int = 4,
    max_proofs: int = 5,
    max_states: int = 2000,
    max_ablation_facts: int = 8,
) -> Dict[str, object]:
    """Ablate proof facts for an already parsed query literal."""

    started = time.perf_counter()
    explicit_label = example_label(task.examples, query_literal)
    engine = BackwardChainingReasoner(
        task=task,
        rules=rules,
        threshold=threshold,
        max_depth=max_depth,
        max_proofs=max_proofs,
        max_states=max_states,
        max_paths=max_paths,
        max_edges_per_node=max_edges_per_node,
        excluded_facts=set(),
        include_path_evidence=True,
    )
    engine_ready = time.perf_counter()
    base_proofs = engine.prove(query_literal)
    proof_done = time.perf_counter()
    base = reasoning_result_from_proofs(
        query_literal=query_literal,
        explicit_label=explicit_label,
        engine=engine,
        proofs=base_proofs,
        threshold=threshold,
        max_depth=max_depth,
        max_proofs=max_proofs,
        started=started,
        engine_ready=engine_ready,
        proof_done=proof_done,
        include_evidence=True,
    )
    ablations: List[Dict[str, object]] = []
    if base.get("decision") != SUPPORTED or not base.get("evidence"):
        return {
            "query": base.get("query"),
            "initial_decision": base.get("decision"),
            "reason": "counterfactual ablation requires an initially supported query",
            "runtime_seconds": time.perf_counter() - started,
            "initial_runtime_seconds": base.get("runtime_seconds", 0.0),
            "ablations": ablations,
            "summary": {
                "ablated_facts": 0,
                "critical_facts": 0,
                "alternative_supported": 0,
                "inconclusive_after_ablation": 0,
            },
            "initial": base,
        }

    proof_trace = (base.get("evidence") or [{}])[0].get("proof_trace") or {}
    candidate_facts = unique_preserving_order(collect_edb_facts(proof_trace))[:max_ablation_facts]
    for fact_text in candidate_facts:
        try:
            fact = parse_literal(fact_text)
        except ValueError:
            continue
        rerun_started = time.perf_counter()
        after_proofs = engine.prove_with_exclusions(
            literal=query_literal,
            max_proofs=1,
            excluded_facts={fact},
        )
        rerun_done = time.perf_counter()
        after = reasoning_result_from_proofs(
            query_literal=query_literal,
            explicit_label=explicit_label,
            engine=engine,
            proofs=after_proofs,
            threshold=threshold,
            max_depth=max_depth,
            max_proofs=1,
            started=rerun_started,
            engine_ready=rerun_started,
            proof_done=rerun_done,
            include_evidence=False,
        )
        if after.get("decision") == SUPPORTED:
            effect = "STILL_SUPPORTED_BY_ALTERNATIVE_PROOF"
        elif after.get("decision") == INCONCLUSIVE:
            effect = "NO_LONGER_SUPPORTED_INCONCLUSIVE"
        else:
            effect = "NO_LONGER_SUPPORTED"
        ablations.append(
            {
                "removed_fact": fact_text,
                "after_decision": after.get("decision"),
                "effect": effect,
                "remaining_evidence_count": after.get("evidence_count"),
                "after_reason": after.get("reason"),
                "rerun_seconds": after.get("runtime_seconds", 0.0),
            }
        )

    all_removed_facts = {parse_literal(item) for item in candidate_facts}
    after_all = None
    if candidate_facts:
        rerun_started = time.perf_counter()
        after_all_proofs = engine.prove_with_exclusions(
            literal=query_literal,
            excluded_facts=all_removed_facts,
            max_proofs=1,
        )
        rerun_done = time.perf_counter()
        after_all = reasoning_result_from_proofs(
            query_literal=query_literal,
            explicit_label=explicit_label,
            engine=engine,
            proofs=after_all_proofs,
            threshold=threshold,
            max_depth=max_depth,
            max_proofs=1,
            started=rerun_started,
            engine_ready=rerun_started,
            proof_done=rerun_done,
            include_evidence=False,
        )
    critical = sum(1 for item in ablations if str(item.get("effect", "")).startswith("NO_LONGER_SUPPORTED"))
    alternative = sum(1 for item in ablations if item.get("effect") == "STILL_SUPPORTED_BY_ALTERNATIVE_PROOF")
    inconclusive = sum(1 for item in ablations if item.get("effect") == "NO_LONGER_SUPPORTED_INCONCLUSIVE")
    runtime_seconds = time.perf_counter() - started
    return {
        "query": base.get("query"),
        "initial_decision": base.get("decision"),
        "initial_evidence_count": base.get("evidence_count"),
        "initial_proof_score": ((base.get("evidence") or [{}])[0] or {}).get("proof_score"),
        "runtime_seconds": runtime_seconds,
        "initial_runtime_seconds": base.get("runtime_seconds", 0.0),
        "ablation_policy": "remove one EDB proof fact at a time, then remove all selected proof facts",
        "ablations": ablations,
        "all_selected_facts_removed": {
            "removed_facts": candidate_facts,
            "after_decision": after_all.get("decision") if after_all else None,
            "remaining_evidence_count": after_all.get("evidence_count") if after_all else None,
            "rerun_seconds": after_all.get("runtime_seconds") if after_all else None,
            "effect": (
                "STILL_SUPPORTED_BY_ALTERNATIVE_PROOF"
                if after_all and after_all.get("decision") == SUPPORTED
                else "NO_LONGER_SUPPORTED_INCONCLUSIVE"
                if after_all and after_all.get("decision") == INCONCLUSIVE
                else "NO_LONGER_SUPPORTED"
                if after_all
                else "NOT_RUN"
            ),
        },
        "summary": {
            "ablated_facts": len(ablations),
            "critical_facts": critical,
            "alternative_supported": alternative,
            "inconclusive_after_ablation": inconclusive,
        },
        "initial": base,
    }


def reason_examples(
    task_dir: str | Path,
    rule_library: str | Path,
    threshold: float = 0.8,
    max_paths: int = 3,
    max_edges_per_node: int = 120,
    max_examples: int = 0,
    max_depth: int = 4,
    max_proofs: int = 3,
    max_states: int = 2000,
) -> Dict[str, object]:
    """Evaluate the reasoning interface on the task's labeled examples."""

    task = load_task(task_dir)
    rules = load_rule_library(rule_library)
    examples = list(task.examples)
    if max_examples > 0:
        positives = [example for example in examples if example.positive][:max_examples]
        negatives = [example for example in examples if not example.positive][:max_examples]
        examples = positives + negatives

    engine_build_started = time.perf_counter()
    engine = BackwardChainingReasoner(
        task=task,
        rules=rules,
        threshold=threshold,
        max_depth=max_depth,
        max_proofs=max_proofs,
        max_states=max_states,
        max_paths=max_paths,
        max_edges_per_node=max_edges_per_node,
        excluded_facts=set(),
        include_path_evidence=False,
    )
    engine_build_seconds = time.perf_counter() - engine_build_started

    rows: List[Dict[str, object]] = []
    for example in examples:
        started = time.perf_counter()
        proofs = engine.prove(example.literal)
        proof_done = time.perf_counter()
        result = reasoning_result_from_proofs(
            query_literal=example.literal,
            explicit_label=example.positive,
            engine=engine,
            proofs=proofs,
            threshold=threshold,
            max_depth=max_depth,
            max_proofs=max_proofs,
            started=started,
            engine_ready=started,
            proof_done=proof_done,
            include_evidence=False,
        )
        rows.append(
            {
                "query": result["query"],
                "gold": "positive" if example.positive else "negative",
                "decision": result["decision"],
                "evidence_count": result["evidence_count"],
                "runtime_seconds": result.get("runtime_seconds", 0.0),
            }
        )

    positives = [row for row in rows if row["gold"] == "positive"]
    negatives = [row for row in rows if row["gold"] == "negative"]
    supported_pos = sum(1 for row in positives if row["decision"] == SUPPORTED)
    supported_neg = sum(1 for row in negatives if row["decision"] == SUPPORTED)
    negative_non_support = sum(1 for row in negatives if row["decision"] != SUPPORTED)
    inconclusive = sum(1 for row in rows if row["decision"] == INCONCLUSIVE)
    supported_total = supported_pos + supported_neg
    supported_precision = supported_pos / supported_total if supported_total else 0.0
    supported_recall = supported_pos / len(positives) if positives else 0.0
    negative_non_support_rate = negative_non_support / len(negatives) if negatives else 0.0
    held_out_accuracy = (
        sum(
            1
            for row in rows
            if (row["gold"] == "positive" and row["decision"] == SUPPORTED)
            or (row["gold"] == "negative" and row["decision"] != SUPPORTED)
        )
        / len(rows)
        if rows
        else 0.0
    )
    total_runtime = sum(float(row.get("runtime_seconds") or 0.0) for row in rows)
    return {
        "task_dir": str(task_dir),
        "target": task.target.signature,
        "examples": len(rows),
        "positive_examples": len(positives),
        "negative_examples": len(negatives),
        "supported_precision": supported_precision,
        "supported_recall": supported_recall,
        "negative_non_support_rate": negative_non_support_rate,
        "inconclusive_rate": inconclusive / len(rows) if rows else 0.0,
        "held_out_accuracy": held_out_accuracy,
        # Compatibility aliases for existing result consumers. New reports
        # should use the label-independent names above.
        "unsupported_recall": negative_non_support_rate,
        "three_value_accuracy": held_out_accuracy,
        "decision_semantics": {
            "SUPPORTED": "a bounded positive proof exists",
            "INCONCLUSIVE": "no bounded positive proof was found",
            "UNSUPPORTED": "reserved for an explicit negative proof",
            "negative_labels_used_for_decision": False,
        },
        "total_runtime_seconds": total_runtime,
        "mean_runtime_seconds": total_runtime / len(rows) if rows else 0.0,
        "shared_engine_build_seconds": engine_build_seconds,
        "counts": {
            "supported_positive": supported_pos,
            "supported_negative": supported_neg,
            "negative_non_support": negative_non_support,
            "inconclusive": inconclusive,
        },
        "rows": rows,
    }


def reason_literal(
    task: TaskData,
    rules: Sequence[ReasoningRule],
    query_literal: Literal,
    threshold: float = 0.8,
    max_paths: int = 5,
    max_edges_per_node: int = 120,
    max_depth: int = 4,
    max_proofs: int = 5,
    max_states: int = 2000,
    excluded_facts: Optional[Set[Literal]] = None,
    include_path_evidence: bool = True,
) -> Dict[str, object]:
    """Reason over one already parsed query literal."""

    started = time.perf_counter()
    explicit_label = example_label(task.examples, query_literal)
    engine = BackwardChainingReasoner(
        task=task,
        rules=rules,
        threshold=threshold,
        max_depth=max_depth,
        max_proofs=max_proofs,
        max_states=max_states,
        max_paths=max_paths,
        max_edges_per_node=max_edges_per_node,
        excluded_facts=excluded_facts or set(),
        include_path_evidence=include_path_evidence,
    )
    engine_ready = time.perf_counter()
    proofs = engine.prove(query_literal)
    proof_done = time.perf_counter()

    return reasoning_result_from_proofs(
        query_literal=query_literal,
        explicit_label=explicit_label,
        engine=engine,
        proofs=proofs,
        threshold=threshold,
        max_depth=max_depth,
        max_proofs=max_proofs,
        started=started,
        engine_ready=engine_ready,
        proof_done=proof_done,
        include_evidence=True,
    )


def reasoning_result_from_proofs(
    query_literal: Literal,
    explicit_label: Optional[bool],
    engine: "BackwardChainingReasoner",
    proofs: Sequence[ProofResult],
    threshold: float,
    max_depth: int,
    max_proofs: int,
    started: float,
    engine_ready: float,
    proof_done: float,
    include_evidence: bool,
) -> Dict[str, object]:
    """Materialize one reasoning judgement from already computed proofs."""

    evidence = [engine.proof_to_evidence(proof) for proof in proofs] if include_evidence else []
    evidence_done = time.perf_counter()
    if proofs:
        decision = SUPPORTED
        reason = "at least one bounded proof tree supports the query"
    else:
        decision = INCONCLUSIVE
        reason = "no bounded positive proof produced sufficient evidence"

    return {
        "query": literal_to_text(query_literal, quote_constants=True),
        "decision": decision,
        "reason": reason,
        "decision_semantics": {
            "SUPPORTED": "a bounded positive proof exists",
            "INCONCLUSIVE": "no bounded positive proof was found",
            "UNSUPPORTED": "reserved for an explicit negative proof",
            "negative_labels_used_for_decision": False,
        },
        "explicit_example_label": explicit_label,
        "threshold": threshold,
        "applicable_rule_count": engine.applicable_rule_count(query_literal),
        "reasoning_mode": "bounded_backward_chaining",
        "max_depth": max_depth,
        "max_proofs": max_proofs,
        "runtime_seconds": evidence_done - started,
        "runtime_breakdown": {
            "engine_build_seconds": engine_ready - started,
            "proof_search_seconds": proof_done - engine_ready,
            "evidence_materialization_seconds": evidence_done - proof_done,
        },
        "search_stats": dict(engine.last_search_stats),
        "dependency_graph": engine.dependency_graph.to_json(),
        "evidence_count": len(proofs),
        "evidence": [asdict(item) for item in evidence],
    }


class BackwardChainingReasoner:
    """Bounded backward chaining over EDB facts and exported PARA rules."""

    def __init__(
        self,
        task: TaskData,
        rules: Sequence[ReasoningRule],
        threshold: float,
        max_depth: int,
        max_proofs: int,
        max_states: int,
        max_paths: int,
        max_edges_per_node: int,
        excluded_facts: Set[Literal],
        include_path_evidence: bool,
    ) -> None:
        self.task = task
        self.rules = [rule for rule in rules if rule.f1 >= max(threshold, rule.threshold)]
        self.threshold = threshold
        self.max_depth = max_depth
        self.max_proofs = max_proofs
        self.max_states = max_states
        self.max_paths = max_paths
        self.max_edges_per_node = max_edges_per_node
        self.excluded_facts = excluded_facts
        self.include_path_evidence = include_path_evidence
        self.facts = list(task.facts)
        self.fact_index = build_fact_index(self.facts)
        self.parsed_rules: List[Tuple[ReasoningRule, Rule]] = []
        self.rules_by_head: Dict[str, List[Tuple[ReasoningRule, Rule]]] = {}
        for item in self.rules:
            try:
                parsed = parse_rule(item.rule, source=item.source_method or "reasoning_library")
            except ValueError:
                continue
            signature = literal_signature(parsed.head)
            self.parsed_rules.append((item, parsed))
            self.rules_by_head.setdefault(signature, []).append((item, parsed))
        self.dependency_graph = build_dependency_graph(self.rules)
        binary_specs = {
            spec.name: spec
            for spec in task.predicates.values()
            if spec.arity == 2 and len(spec.types) == 2 and not is_attribute_or_pair_constraint(spec.name)
        }
        self.relation_index = build_relation_index(build_fact_graph(self.facts, binary_specs)) if binary_specs else {}
        self.rule_instance_counter = 0
        self.last_search_stats: Dict[str, object] = {}

    def applicable_rule_count(self, literal: Literal) -> int:
        return len(self.rules_by_head.get(literal_signature(literal), []))

    def prove(self, literal: Literal) -> List[ProofResult]:
        self.last_search_stats = {
            "goal_calls": 0,
            "edb_candidate_count": 0,
            "depth_limit_hits": 0,
            "cycle_prunes": 0,
            "state_limit_hits": 0,
            "proof_limit_hits": 0,
            "max_depth_observed": 0,
            "normal_completion": False,
        }
        proofs = self._prove_literal(literal, {}, depth=0, visited=set())[: self.max_proofs]
        self.last_search_stats["proofs_found"] = len(proofs)
        self.last_search_stats["normal_completion"] = True
        if proofs:
            termination = "proof_found"
        elif any(
            int(self.last_search_stats.get(key, 0)) > 0
            for key in ("depth_limit_hits", "state_limit_hits")
        ):
            termination = "bounded_no_proof"
        else:
            termination = "exhausted_no_proof"
        self.last_search_stats["termination_reason"] = termination
        return proofs

    def prove_with_exclusions(self, literal: Literal, excluded_facts: Set[Literal], max_proofs: Optional[int] = None) -> List[ProofResult]:
        previous_exclusions = self.excluded_facts
        previous_max_proofs = self.max_proofs
        self.excluded_facts = excluded_facts
        if max_proofs is not None:
            self.max_proofs = max_proofs
        try:
            return self.prove(literal)
        finally:
            self.excluded_facts = previous_exclusions
            self.max_proofs = previous_max_proofs

    def _prove_literal(
        self,
        goal: Literal,
        binding: Dict[str, str],
        depth: int,
        visited: Set[str],
    ) -> List[ProofResult]:
        self.last_search_stats["goal_calls"] = int(self.last_search_stats.get("goal_calls", 0)) + 1
        self.last_search_stats["max_depth_observed"] = max(
            int(self.last_search_stats.get("max_depth_observed", 0)),
            depth,
        )
        if depth > self.max_depth:
            self.last_search_stats["depth_limit_hits"] = int(self.last_search_stats.get("depth_limit_hits", 0)) + 1
            return []
        grounded_goal = apply_binding(goal, binding)
        visit_key = literal_to_text(grounded_goal, quote_constants=True)
        if visit_key in visited:
            self.last_search_stats["cycle_prunes"] = int(self.last_search_stats.get("cycle_prunes", 0)) + 1
            return []
        if literal_signature(goal) not in self.rules_by_head:
            return self._prove_edb(goal, binding)
        output: List[ProofResult] = []
        next_visited = set(visited)
        next_visited.add(visit_key)
        for item, raw_rule in self.rules_by_head.get(literal_signature(goal), []):
            instantiated = instantiate_rule(raw_rule, self._next_rule_suffix())
            head_binding = dict(binding)
            if not unify_general(instantiated.head, goal, head_binding):
                continue
            body_proofs = self._prove_body(instantiated.body, head_binding, depth + 1, next_visited)
            for proof in body_proofs:
                score = min([item.f1] + [float(child.get("score", 1.0)) for child in proof.trace.get("subproofs", [])])
                if score < max(self.threshold, item.threshold):
                    continue
                trace = {
                    "kind": "derived_rule",
                    "goal": literal_to_text(apply_binding(goal, proof.binding), quote_constants=True),
                    "rule": item.rule,
                    "rule_f1": item.f1,
                    "score": score,
                    "confidence_aggregation": "min",
                    "depth": depth,
                    "source_summary": item.source_summary,
                    "subproofs": proof.trace.get("subproofs", []),
                }
                output.append(ProofResult(binding=proof.binding, score=score, trace=trace))
                if len(output) >= self.max_proofs:
                    self.last_search_stats["proof_limit_hits"] = int(self.last_search_stats.get("proof_limit_hits", 0)) + 1
                    return output
        return output

    def _prove_body(
        self,
        body: Sequence[Literal],
        binding: Dict[str, str],
        depth: int,
        visited: Set[str],
    ) -> List[ProofResult]:
        active: List[Tuple[ProofResult, Tuple[Literal, ...]]] = [
            (ProofResult(binding=dict(binding), score=1.0, trace={"subproofs": []}), tuple(body))
        ]
        completed: List[ProofResult] = []
        while active:
            next_active: List[Tuple[ProofResult, Tuple[Literal, ...]]] = []
            for state, remaining in active:
                if not remaining:
                    completed.append(state)
                    if len(completed) >= self.max_proofs:
                        self.last_search_stats["proof_limit_hits"] = int(self.last_search_stats.get("proof_limit_hits", 0)) + 1
                        return completed[: self.max_proofs]
                    continue
                literal_index = self._select_next_body_literal(remaining, state.binding)
                literal = remaining[literal_index]
                rest = remaining[:literal_index] + remaining[literal_index + 1 :]
                literal_proofs = self._prove_literal(literal, state.binding, depth, visited)
                for proof in literal_proofs:
                    next_active.append(
                        (
                            ProofResult(
                                binding=proof.binding,
                                score=min(state.score, proof.score),
                                trace={"subproofs": list(state.trace.get("subproofs", [])) + [proof.trace]},
                            ),
                            rest,
                        )
                    )
                    if len(next_active) >= self.max_states:
                        self.last_search_stats["state_limit_hits"] = int(self.last_search_stats.get("state_limit_hits", 0)) + 1
                        break
                if len(next_active) >= self.max_states:
                    break
            active = next_active[: self.max_states]
        return completed[: self.max_proofs]

    def _select_next_body_literal(self, body: Sequence[Literal], binding: Dict[str, str]) -> int:
        """Choose the next body literal without changing conjunction semantics.

        Popper and LLMs may emit logically equivalent clauses with different
        body orders.  A proof engine that expands a high-cardinality,
        fully-unbound predicate first can hit state bounds even when a simple
        bound-first ordering proves the same clause.  We therefore schedule
        body literals by the amount of already-bound information and by an
        index-based candidate estimate.
        """

        best_index = 0
        best_key: Optional[Tuple[int, int, int, int]] = None
        for index, literal in enumerate(body):
            resolved = [resolve_term(arg, binding) for arg in literal.args]
            bound_count = sum(1 for arg in resolved if not is_variable(arg))
            unbound_count = len(resolved) - bound_count
            candidate_estimate = self._candidate_estimate(literal, binding)
            key = (-bound_count, unbound_count, candidate_estimate, index)
            if best_key is None or key < best_key:
                best_key = key
                best_index = index
        return best_index

    def _candidate_estimate(self, literal: Literal, binding: Dict[str, str]) -> int:
        if literal_signature(literal) in self.rules_by_head:
            return 1
        pattern = apply_binding(literal, binding)
        try:
            return len(indexed_candidates(pattern, self.fact_index, {}))
        except Exception:
            return len(self.facts)

    def _prove_edb(self, literal: Literal, binding: Dict[str, str]) -> List[ProofResult]:
        pattern = apply_binding(literal, binding)
        candidates = indexed_candidates(pattern, self.fact_index, {})
        self.last_search_stats["edb_candidate_count"] = (
            int(self.last_search_stats.get("edb_candidate_count", 0)) + len(candidates)
        )
        output: List[ProofResult] = []
        for fact in candidates:
            if fact in self.excluded_facts:
                continue
            next_binding = dict(binding)
            if not unify_general(literal, fact, next_binding):
                continue
            trace = {
                "kind": "edb_fact",
                "goal": literal_to_text(apply_binding(literal, next_binding), quote_constants=True),
                "fact": literal_to_text(fact, quote_constants=True),
                "score": 1.0,
            }
            output.append(ProofResult(binding=next_binding, score=1.0, trace=trace))
            if len(output) >= self.max_states:
                self.last_search_stats["state_limit_hits"] = int(self.last_search_stats.get("state_limit_hits", 0)) + 1
                break
        return output

    def proof_to_evidence(self, proof: ProofResult) -> ReasoningEvidence:
        root = proof.trace
        rule_text = str(root.get("rule") or "")
        item = self._rule_metadata(rule_text)
        query_literal = parse_literal(str(root.get("goal") or "")) if root.get("goal") else None
        path_matches: List[List[GraphEdge]] = []
        if self.include_path_evidence and query_literal is not None and item is not None:
            path_matches = execute_rule_path_programs(
                relation_index=self.relation_index,
                start=query_literal.args[0] if query_literal.args else "",
                end=query_literal.args[1] if len(query_literal.args) > 1 else "",
                path_programs=item.path_programs,
                max_paths=self.max_paths,
                max_edges_per_node=self.max_edges_per_node,
            )
        return ReasoningEvidence(
            rule=rule_text,
            rule_f1=float(root.get("rule_f1") or proof.score),
            rule_precision=item.precision if item is not None else proof.score,
            rule_recall=item.recall if item is not None else proof.score,
            body_facts=tuple(collect_edb_facts(root)[: self.max_paths]),
            path_programs=tuple(path_signature_to_json(path) for path in (item.path_programs if item is not None else ())),
            path_evidence=tuple(tuple(edge_to_text(edge) for edge in path) for path in path_matches[: self.max_paths]),
            source_summary=str(root.get("source_summary") or (item.source_summary if item is not None else "")),
            proof_score=proof.score,
            proof_trace=root,
        )

    def _rule_metadata(self, rule_text: str) -> Optional[ReasoningRule]:
        for item in self.rules:
            if item.rule == rule_text:
                return item
        return None

    def _next_rule_suffix(self) -> str:
        self.rule_instance_counter += 1
        return f"R{self.rule_instance_counter}"


def reasoning_rule_from_summary(
    data: Dict[str, object],
    summary_path: Path,
    min_f1: float,
) -> Optional[ReasoningRule]:
    status = str(data.get("status") or "")
    metrics = data.get("metrics") or data.get("best_candidate_metrics") or {}
    f1 = float(metric_value(metrics, "f1"))
    if status != "ok" or f1 < min_f1:
        return None
    rule_text = str(data.get("final_rule") or "").strip()
    if not rule_text:
        return None
    try:
        rule = parse_rule(rule_text, source=str(data.get("method") or "ashrl"))
    except ValueError:
        return None
    trace = extract_agent_trace(data)
    path_programs = tuple(extract_accepted_path_programs(trace, rule))
    target_types = tuple(str(item) for item in (data.get("target_types") or []))
    if not target_types:
        target_types = tuple(str(item) for item in infer_target_types_from_trace(trace))
    return ReasoningRule(
        target=f"{rule.head.predicate}/{rule.head.arity}",
        target_types=target_types,
        rule=rule_to_text(rule, with_confidence=True),
        path_programs=path_programs,
        f1=f1,
        precision=float(metric_value(metrics, "precision")),
        recall=float(metric_value(metrics, "recall")),
        threshold=min_f1,
        source_summary=str(summary_path),
        source_method=str(data.get("method") or "ashrl"),
        status=status,
    )


def evidence_matches(
    rule: Rule,
    facts: Iterable[Literal],
    query_literal: Literal,
    fact_index: Optional[FactIndex] = None,
    max_matches: int = 3,
    max_intermediate_states: int = 2000,
) -> List[Tuple[Literal, ...]]:
    """Return concrete body fact matches for a grounded query."""

    if rule.head.predicate != query_literal.predicate or rule.head.arity != query_literal.arity:
        return []
    binding: Dict[str, str] = {}
    if not unify_literal(rule.head, query_literal, binding):
        return []
    facts_list = list(facts)
    if fact_index is None:
        fact_index = build_fact_index(facts_list)
    states: List[Tuple[Dict[str, str], List[Literal]]] = [(binding, [])]
    for lit in rule.body:
        next_states: List[Tuple[Dict[str, str], List[Literal]]] = []
        for current, used in states:
            candidates = indexed_candidates(lit, fact_index, current)
            for fact in matching_candidates(lit, candidates, current):
                new_binding = dict(current)
                if unify_literal(lit, fact, new_binding):
                    next_states.append((new_binding, used + [fact]))
                    if len(next_states) >= max_intermediate_states:
                        break
            if len(next_states) >= max_intermediate_states:
                break
        states = next_states
        if not states:
            return []
    return [tuple(used) for _binding, used in states[:max_matches]]


def build_dependency_graph(rules: Sequence[ReasoningRule]) -> RuleDependencyGraph:
    parsed: List[Rule] = []
    for item in rules:
        try:
            parsed.append(parse_rule(item.rule, source=item.source_method or "reasoning_library"))
        except ValueError:
            continue
    idb = {literal_signature(rule.head) for rule in parsed}
    edb: Set[str] = set()
    dependencies: Dict[str, List[Dict[str, str]]] = {}
    raw_edges: Dict[str, Set[str]] = {}
    for rule in parsed:
        head = literal_signature(rule.head)
        entries = dependencies.setdefault(head, [])
        known_entries = {(entry["predicate"], entry["kind"]) for entry in entries}
        for literal in rule.body:
            signature = literal_signature(literal)
            kind = "IDB" if signature in idb else "EDB"
            if kind == "EDB":
                edb.add(signature)
            else:
                raw_edges.setdefault(head, set()).add(signature)
            key = (signature, kind)
            if key not in known_entries:
                entries.append({"predicate": signature, "kind": kind})
                known_entries.add(key)
    recursive = {
        predicate
        for predicate in idb
        if dependency_reaches(predicate, predicate, raw_edges, visited=set())
    }
    return RuleDependencyGraph(
        idb_predicates=idb,
        edb_predicates=edb,
        dependencies={key: value for key, value in sorted(dependencies.items())},
        recursive_predicates=recursive,
    )


def dependency_reaches(start: str, target: str, edges: Dict[str, Set[str]], visited: Set[str]) -> bool:
    for child in edges.get(start, set()):
        if child == target:
            return True
        if child in visited:
            continue
        visited.add(child)
        if dependency_reaches(child, target, edges, visited):
            return True
    return False


def literal_signature(literal: Literal) -> str:
    return f"{literal.predicate}/{literal.arity}"


def resolve_term(term: str, binding: Dict[str, str]) -> str:
    seen: Set[str] = set()
    current = term
    while is_variable(current) and current in binding and binding[current] != current:
        if current in seen:
            break
        seen.add(current)
        current = binding[current]
    return current


def bind_term(variable: str, value: str, binding: Dict[str, str]) -> bool:
    resolved_variable = resolve_term(variable, binding)
    resolved_value = resolve_term(value, binding)
    if resolved_variable == resolved_value:
        return True
    if is_variable(resolved_variable):
        binding[resolved_variable] = resolved_value
        return True
    if is_variable(resolved_value):
        binding[resolved_value] = resolved_variable
        return True
    return resolved_variable == resolved_value


def unify_general(left: Literal, right: Literal, binding: Dict[str, str]) -> bool:
    """Unify two literals that may both contain variables."""

    if left.predicate != right.predicate or left.arity != right.arity:
        return False
    for left_arg, right_arg in zip(left.args, right.args):
        left_resolved = resolve_term(left_arg, binding)
        right_resolved = resolve_term(right_arg, binding)
        if is_variable(left_resolved):
            if not bind_term(left_resolved, right_resolved, binding):
                return False
        elif is_variable(right_resolved):
            if not bind_term(right_resolved, left_resolved, binding):
                return False
        elif left_resolved != right_resolved:
            return False
    return True


def apply_binding(literal: Literal, binding: Dict[str, str]) -> Literal:
    return Literal(literal.predicate, tuple(resolve_term(arg, binding) for arg in literal.args))


def instantiate_rule(rule: Rule, suffix: str) -> Rule:
    variable_map: Dict[str, str] = {}

    def rename(arg: str) -> str:
        if not is_variable(arg):
            return arg
        if arg not in variable_map:
            variable_map[arg] = f"{arg}_{suffix}"
        return variable_map[arg]

    return Rule(
        head=Literal(rule.head.predicate, tuple(rename(arg) for arg in rule.head.args)),
        body=tuple(Literal(literal.predicate, tuple(rename(arg) for arg in literal.args)) for literal in rule.body),
        confidence=rule.confidence,
        source=rule.source,
    )


def collect_edb_facts(trace: Dict[str, object]) -> List[str]:
    facts: List[str] = []
    if trace.get("kind") == "edb_fact" and trace.get("fact"):
        facts.append(str(trace["fact"]))
    for child in trace.get("subproofs", []) or []:
        if isinstance(child, dict):
            facts.extend(collect_edb_facts(child))
    return facts


def unique_preserving_order(items: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    output: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def execute_rule_path_programs(
    relation_index: Dict[Tuple[str, str, bool], List[GraphEdge]],
    start: str,
    end: str,
    path_programs: Sequence[PathSignature],
    max_paths: int,
    max_edges_per_node: int,
) -> List[List[GraphEdge]]:
    if not relation_index or not start or not end or not path_programs:
        return []
    return execute_path_queries(
        relation_index=relation_index,
        start=start,
        end=end,
        path_queries=path_programs,
        max_paths=max_paths,
        max_edges_per_node=max_edges_per_node,
    )


def expand_summary_paths(items: Sequence[str | Path]) -> List[Path]:
    output: List[Path] = []
    for raw in items:
        path = Path(raw)
        if path.is_dir():
            direct = path / "summary.json"
            if direct.exists():
                output.append(direct)
            else:
                output.extend(sorted(path.rglob("summary.json")))
        elif path.exists():
            output.append(path)
    return output


def load_rule_library(path: str | Path) -> List[ReasoningRule]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [rule_from_json(item) for item in data.get("rules", [])]


def parse_query_literal(query: str, task: TaskData) -> Literal:
    text = query.strip()
    if "(" in text:
        return parse_literal(text)
    parts = text.split()
    if len(parts) == 3:
        return Literal(parts[0], (parts[1], parts[2]))
    if len(parts) == 2 and task.target.arity == 2:
        return Literal(task.target.name, (parts[0], parts[1]))
    raise ValueError("Query must be `predicate(A,B)`, `predicate A B`, or `A B` for binary target tasks")


def example_label(examples: Sequence[Example], literal: Literal) -> Optional[bool]:
    for example in examples:
        if example.literal == literal:
            return example.positive
    return None


def extract_agent_trace(summary: Dict[str, object]) -> Dict[str, object]:
    rounds = summary.get("rounds") or []
    if not rounds:
        return {}
    rationale = str((rounds[0] or {}).get("guidance_rationale") or "")
    marker = "Trace: "
    if marker not in rationale:
        return {}
    try:
        return json.loads(rationale.split(marker, 1)[1])
    except json.JSONDecodeError:
        return {}


def extract_accepted_path_programs(trace: Dict[str, object], rule: Rule) -> List[PathSignature]:
    iterations = list(trace.get("iterations") or [])
    iterations.reverse()
    for item in iterations:
        metrics = item.get("best_metrics") or {}
        if str(item.get("best_rule") or "").strip().rstrip(".") != rule_to_text(rule).strip().rstrip("."):
            continue
        if float(metric_value(metrics, "f1")) <= 0.0:
            continue
        action = item.get("action") or {}
        parsed = parse_path_programs(action.get("path_queries"))
        if parsed:
            return parsed
        diagnostics = item.get("diagnostics") or {}
        parsed = parse_diagnostic_path_programs(diagnostics)
        if parsed:
            return parsed
    for item in iterations:
        action = item.get("action") or {}
        parsed = parse_path_programs(action.get("path_queries"))
        if parsed:
            return parsed
    return path_program_from_rule_body(rule)


def parse_diagnostic_path_programs(diagnostics: Dict[str, object]) -> List[PathSignature]:
    for portfolio in diagnostics.get("portfolio") or []:
        inner = (portfolio.get("diagnostics") or {}).get("planned_path_queries")
        parsed = parse_path_programs(inner)
        if parsed:
            return parsed
    return []


def parse_path_programs(value: object) -> List[PathSignature]:
    if not isinstance(value, list):
        return []
    output: List[PathSignature] = []
    for raw_path in value:
        steps = raw_path.get("steps") if isinstance(raw_path, dict) else raw_path
        if not isinstance(steps, list):
            continue
        parsed_steps: List[Tuple[str, bool]] = []
        for raw_step in steps:
            if isinstance(raw_step, dict):
                predicate = str(raw_step.get("predicate") or "")
                direction = str(raw_step.get("direction") or "fwd")
            elif isinstance(raw_step, list) and len(raw_step) == 2:
                predicate = str(raw_step[0])
                raw_direction = raw_step[1]
                if isinstance(raw_direction, bool):
                    direction = "rev" if raw_direction else "fwd"
                else:
                    direction = str(raw_direction)
            else:
                parsed_steps = []
                break
            if not predicate:
                parsed_steps = []
                break
            parsed_steps.append((predicate, direction == "rev"))
        if parsed_steps:
            signature = tuple(parsed_steps)
            if signature not in output:
                output.append(signature)
    return output


def path_program_from_rule_body(rule: Rule) -> List[PathSignature]:
    """Best-effort reconstruction: treat binary body literals as a path signature."""

    if not rule.body:
        return []
    steps: List[Tuple[str, bool]] = []
    current = rule.head.args[0] if rule.head.args else ""
    for lit in rule.body:
        if lit.arity != 2:
            continue
        left, right = lit.args
        if left == current:
            steps.append((lit.predicate, False))
            current = right
        elif right == current:
            steps.append((lit.predicate, True))
            current = left
    if current == (rule.head.args[1] if len(rule.head.args) > 1 else "") and steps:
        return [tuple(steps)]
    return []


def infer_target_types_from_trace(trace: Dict[str, object]) -> Tuple[str, ...]:
    return tuple()


def rule_to_json(rule: ReasoningRule) -> Dict[str, object]:
    return {
        "target": rule.target,
        "target_types": list(rule.target_types),
        "rule": rule.rule,
        "path_programs": [
            [[predicate, "rev" if reversed_edge else "fwd"] for predicate, reversed_edge in path]
            for path in rule.path_programs
        ],
        "f1": rule.f1,
        "precision": rule.precision,
        "recall": rule.recall,
        "threshold": rule.threshold,
        "source_summary": rule.source_summary,
        "source_method": rule.source_method,
        "status": rule.status,
    }


def rule_from_json(data: Dict[str, object]) -> ReasoningRule:
    return ReasoningRule(
        target=str(data.get("target") or ""),
        target_types=tuple(str(item) for item in (data.get("target_types") or [])),
        rule=str(data.get("rule") or ""),
        path_programs=tuple(parse_path_programs(data.get("path_programs"))),
        f1=float(data.get("f1") or 0.0),
        precision=float(data.get("precision") or 0.0),
        recall=float(data.get("recall") or 0.0),
        threshold=float(data.get("threshold") or 0.8),
        source_summary=str(data.get("source_summary") or ""),
        source_method=str(data.get("source_method") or ""),
        status=str(data.get("status") or ""),
    )


def path_signature_to_json(path: PathSignature) -> Tuple[Tuple[str, str], ...]:
    return tuple((predicate, "rev" if reversed_edge else "fwd") for predicate, reversed_edge in path)


def edge_to_text(edge: GraphEdge) -> str:
    if edge.reversed_edge:
        return f"{edge.predicate}({edge.dst},{edge.src})"
    return f"{edge.predicate}({edge.src},{edge.dst})"


def metric_value(metrics: object, key: str) -> float:
    if not isinstance(metrics, dict):
        return 0.0
    try:
        return float(metrics.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
