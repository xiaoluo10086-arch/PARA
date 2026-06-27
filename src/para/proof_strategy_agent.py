#!/usr/bin/env python3
"""Verifier-governed proof-strategy agent.

This module reads existing PARA artifacts as memory and emits auditable
ProofStrategy and decision-trace objects for verifier-governed planning.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ThresholdConfig:
    accept: float = 0.8
    refine: float = 0.9
    prefer_recall_over_precision: bool = False


@dataclass(frozen=True)
class SearchBounds:
    max_depth: int = 4
    max_proof_trees: int = 5
    max_states_per_node: int = 2000
    max_edges_per_node: int = 120


@dataclass(frozen=True)
class AuditConfig:
    counterfactual_sample_size: int = 20
    near_miss_stress_enabled: bool = True
    proof_tree_audit_depth: int = 3


@dataclass(frozen=True)
class IntentProfile:
    target: str
    strategy_style: str
    task_complexity: str
    sample_density: str
    rationale: str


@dataclass(frozen=True)
class ConfidenceProfile:
    score: float
    mode: str
    confidence_drivers: dict[str, float]
    rationale: str


@dataclass(frozen=True)
class ProofStrategy:
    strategy_id: str
    target: str
    path_programs: list[list[str]]
    threshold_config: ThresholdConfig = field(default_factory=ThresholdConfig)
    search_bounds: SearchBounds = field(default_factory=SearchBounds)
    audit_config: AuditConfig = field(default_factory=AuditConfig)
    fallback_strategy: str = "refine_with_symbolic_feedback"
    memory_enabled: bool = True
    intent_profile: IntentProfile | None = None
    confidence_profile: ConfidenceProfile | None = None
    selected_predicates: list[str] = field(default_factory=list)
    candidate_rules: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass(frozen=True)
class VerifierDiagnostic:
    status: str
    reason: str
    positive_coverage: float | None = None
    negative_contamination: float | None = None
    suggested_repair: str | None = None
    missing_relation_types: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Decision:
    step: int
    action: str
    rationale: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionTrace:
    trace_id: str
    target: str
    strategy_id: str
    decisions: list[Decision]
    verifier_diagnostic: VerifierDiagnostic


@dataclass(frozen=True)
class ReflectionRecord:
    trigger: str
    observation: str
    repair_action: str
    updated_assumption: str


@dataclass(frozen=True)
class StrategyLoop:
    loop_id: str
    target: str
    iterations: list[dict[str, Any]]
    final_strategy: ProofStrategy
    final_trace: DecisionTrace
    stop_reason: str


@dataclass
class MemoryRecord:
    target: str
    split_name: str
    source_file: str
    best_f1: float | None
    accepted_candidate_count: int
    candidate_count_total: int
    final_stop_reason: str
    witness_evidence_mode: str
    indexed_plan_only: bool
    train_positive: int = 0
    train_negative: int = 0
    test_positive: int = 0
    test_negative: int = 0
    path_programs: list[list[str]] = field(default_factory=list)
    selected_predicates: list[str] = field(default_factory=list)
    rule_texts: list[str] = field(default_factory=list)


@dataclass
class AgentMemory:
    records: list[MemoryRecord] = field(default_factory=list)

    def by_target(self, target: str) -> list[MemoryRecord]:
        normalized = normalize_target(target)
        return [record for record in self.records if normalize_target(record.target) == normalized]

    def best_for_target(self, target: str) -> MemoryRecord | None:
        records = self.by_target(target)
        if not records:
            return None
        return max(records, key=lambda rec: (rec.best_f1 or 0.0, rec.accepted_candidate_count))

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "records": [asdict(record) for record in self.records],
            "tiered_summary": self.tiered_summary(),
        }

    def tiered_summary(self) -> dict[str, Any]:
        by_target: dict[str, int] = {}
        for record in self.records:
            key = normalize_target(record.target)
            by_target[key] = by_target.get(key, 0) + 1
        domain_patterns = {
            "call_relation": "method/class call relations are usually structural path patterns",
            "policy_relation": "allowed-use relations usually need attribute or package constraints",
            "override_relation": "override relations usually need inheritance plus signature/name guards",
        }
        return {
            "session_memory_records": len(self.records),
            "task_memory_records_by_target": by_target,
            "domain_memory_patterns": domain_patterns,
        }


class ProofStrategyAgent:
    """Deterministic proof-strategy planner.

    The class uses existing artifacts as memory. It is deterministic by design
    so early experiments are reproducible and cheap.
    """

    def __init__(self, memory: AgentMemory, *, accept_f1: float = 0.8) -> None:
        self.memory = memory
        self.accept_f1 = accept_f1

    def plan(self, target: str, *, memory_enabled: bool = True) -> tuple[ProofStrategy, DecisionTrace]:
        normalized = normalize_target(target)
        intent = recognize_intent(normalized, self.memory.by_target(normalized))
        memory_record = self.memory.best_for_target(normalized) if memory_enabled else None
        path_programs = choose_path_programs(normalized, memory_record)
        threshold = choose_threshold(normalized, memory_record, self.accept_f1)
        bounds = choose_bounds(normalized, memory_record)
        confidence = estimate_confidence(intent, memory_record, path_programs)
        threshold, bounds = apply_confidence_mode(threshold, bounds, confidence)
        audit = AuditConfig(
            counterfactual_sample_size=20 if normalized != "canCallClass" else 10,
            near_miss_stress_enabled=True,
            proof_tree_audit_depth=3,
        )
        strategy_id = f"{normalized}_{'memory' if memory_record else 'cold'}_proof_strategy"
        rationale = build_rationale(normalized, memory_record, threshold, bounds)
        strategy = ProofStrategy(
            strategy_id=strategy_id,
            target=normalized,
            path_programs=path_programs,
            threshold_config=threshold,
            search_bounds=bounds,
            audit_config=audit,
            memory_enabled=memory_record is not None,
            intent_profile=intent,
            confidence_profile=confidence,
            selected_predicates=choose_selected_predicates(path_programs, memory_record),
            candidate_rules=memory_record.rule_texts[:5] if memory_record else [],
            rationale=rationale,
        )
        diagnostic = diagnose_strategy(strategy, memory_record)
        decisions = [
            Decision(
                step=1,
                action="RecognizeIntent",
                rationale="Classify the relation and choose a strategy style before planning.",
                payload={"intent_profile": asdict(intent)},
            ),
            Decision(
                step=2,
                action="LoadMemory",
                rationale="Query session, task, and domain memory for verifier-governed prior outcomes.",
                payload={
                    "memory_record": asdict(memory_record) if memory_record else None,
                    "tiered_memory": self.memory.tiered_summary(),
                },
            ),
            Decision(
                step=3,
                action="ProposeProofStrategy",
                rationale=rationale,
                payload={"strategy": asdict(strategy)},
            ),
            Decision(
                step=4,
                action="EstimateConfidence",
                rationale="Quantify uncertainty and switch to conservative/balanced/aggressive execution mode.",
                payload={"confidence_profile": asdict(confidence)},
            ),
            Decision(
                step=5,
                action="VerifierDiagnostic",
                rationale="Symbolic verifier remains the authority for admitting the strategy.",
                payload={"diagnostic": asdict(diagnostic)},
            ),
        ]
        trace = DecisionTrace(
            trace_id=f"trace_{strategy_id}",
            target=normalized,
            strategy_id=strategy_id,
            decisions=decisions,
            verifier_diagnostic=diagnostic,
        )
        return strategy, trace

    def plan_loop(
        self,
        target: str,
        *,
        memory_enabled: bool = True,
        max_refinements: int = 1,
    ) -> StrategyLoop:
        """Run a bounded strategy-planning loop.

        This is the minimal closed loop: propose a strategy, inspect verifier
        diagnostics and confidence, reflect, refine once, and emit a final trace
        that records the repair. It deliberately avoids hidden state or
        unbounded retries.
        """

        strategy, trace = self.plan(target, memory_enabled=memory_enabled)
        iterations: list[dict[str, Any]] = [
            {
                "iteration": 0,
                "strategy": asdict(strategy),
                "trace": asdict(trace),
                "reflection": None,
            }
        ]
        final_strategy = strategy
        final_trace = trace
        stop_reason = "verifier_admitted_initial_strategy"
        for index in range(1, max_refinements + 1):
            if not should_refine(final_strategy, final_trace.verifier_diagnostic):
                break
            reflection = reflect_on_diagnostic(final_strategy, final_trace.verifier_diagnostic)
            refined_strategy = refine_strategy(final_strategy, final_trace.verifier_diagnostic, reflection, index)
            refined_diagnostic = diagnose_refined_strategy(refined_strategy, final_trace.verifier_diagnostic)
            refined_trace = append_refinement_trace(final_trace, refined_strategy, refined_diagnostic, reflection)
            iterations.append(
                {
                    "iteration": index,
                    "strategy": asdict(refined_strategy),
                    "trace": asdict(refined_trace),
                    "reflection": asdict(reflection),
                }
            )
            final_strategy = refined_strategy
            final_trace = refined_trace
            if refined_diagnostic.status.startswith("admissible") or refined_diagnostic.status == "ready_for_symbolic_admission":
                stop_reason = f"refined_strategy_{refined_diagnostic.status}"
                break
            stop_reason = "refinement_budget_exhausted"
        return StrategyLoop(
            loop_id=f"loop_{target}_{'memory' if memory_enabled else 'cold'}",
            target=normalize_target(target),
            iterations=iterations,
            final_strategy=final_strategy,
            final_trace=final_trace,
            stop_reason=stop_reason,
        )


def build_memory(manifest_dir: Path, trace_summary_csv: Path | None = None) -> AgentMemory:
    records: list[MemoryRecord] = []
    trace_rows = load_trace_rows(trace_summary_csv) if trace_summary_csv and trace_summary_csv.exists() else {}
    for manifest_path in sorted(manifest_dir.glob("*_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_name = str(manifest.get("split_name", manifest_path.stem))
        target = infer_target(split_name, manifest)
        learn_json = (manifest.get("learn_summary") or {}).get("json") or {}
        trace_row = trace_rows.get(str(manifest_path.resolve())) or trace_rows.get(str(manifest_path))
        best_f1 = coerce_float(
            (trace_row or {}).get("best_f1")
            or learn_json.get("best_f1")
            or learn_json.get("best_train_f1")
        )
        accepted = coerce_int((trace_row or {}).get("accepted_candidate_count"), default=0)
        total = coerce_int((trace_row or {}).get("candidate_count_total"), default=0)
        stop_reason = str((trace_row or {}).get("final_stop_reason") or learn_json.get("status") or "unknown")
        witness_mode = str((trace_row or {}).get("witness_evidence_mode") or "unknown")
        indexed = str((trace_row or {}).get("indexed_plan_only", "")).lower() == "true"
        rule_payload = extract_rule_library_payload(Path(str(manifest.get("rule_library", ""))))
        split = manifest.get("split") or {}
        records.append(
            MemoryRecord(
                target=target,
                split_name=split_name,
                source_file=str(manifest_path),
                best_f1=best_f1,
                accepted_candidate_count=accepted,
                candidate_count_total=total,
                final_stop_reason=stop_reason,
                witness_evidence_mode=witness_mode,
                indexed_plan_only=indexed,
                train_positive=coerce_int(split.get("train_positive"), default=0),
                train_negative=coerce_int(split.get("train_negative"), default=0),
                test_positive=coerce_int(split.get("test_positive"), default=0),
                test_negative=coerce_int(split.get("test_negative"), default=0),
                path_programs=rule_payload["path_programs"] or extract_candidate_rules(learn_json),
                selected_predicates=rule_payload["selected_predicates"],
                rule_texts=rule_payload["rule_texts"],
            )
        )
    return AgentMemory(records=records)


def load_trace_rows(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = row.get("source_file")
            if source:
                rows[source] = row
                rows[str(Path(source).resolve())] = row
    return rows


def choose_path_programs(target: str, memory_record: MemoryRecord | None) -> list[list[str]]:
    if memory_record and memory_record.path_programs:
        return memory_record.path_programs[:5]
    defaults = {
        "canCallClass": [["containsMethod", "callsMethod", "containsMethod^-1"]],
        "isAllowedToUse": [["containsClass", "packagePolicy", "containsClass^-1"]],
        "overridesMethod": [["containsMethod", "inheritsClass", "containsMethod^-1", "methodSignatureEq"]],
    }
    return defaults.get(target, [["typedEdge", "typedEdge"]])


def recognize_intent(target: str, records: list[MemoryRecord]) -> IntentProfile:
    target_lower = target.lower()
    train_pos = max((record.train_positive for record in records), default=0)
    train_neg = max((record.train_negative for record in records), default=0)
    total_train = train_pos + train_neg
    if total_train == 0:
        sample_density = "unknown"
    elif train_pos < 10:
        sample_density = "sparse"
    elif train_pos < 50:
        sample_density = "moderate"
    else:
        sample_density = "dense"
    if "override" in target_lower:
        return IntentProfile(
            target=target,
            strategy_style="compositional_signature",
            task_complexity="complex",
            sample_density=sample_density,
            rationale="Override relations require inheritance structure plus method signature or name guards.",
        )
    if "allowed" in target_lower or "use" in target_lower:
        return IntentProfile(
            target=target,
            strategy_style="policy_constraint",
            task_complexity="medium",
            sample_density=sample_density,
            rationale="Allowed-use relations usually require structural reachability constrained by package or policy facts.",
        )
    if "call" in target_lower:
        return IntentProfile(
            target=target,
            strategy_style="structural_path",
            task_complexity="simple",
            sample_density=sample_density,
            rationale="Call relations are typically captured by direct structural path programs.",
        )
    return IntentProfile(
        target=target,
        strategy_style="typed_exploration",
        task_complexity="unknown",
        sample_density=sample_density,
        rationale="No known relation family matched; use bounded typed exploration.",
    )


def choose_threshold(target: str, memory_record: MemoryRecord | None, accept_f1: float) -> ThresholdConfig:
    best = memory_record.best_f1 if memory_record else None
    prefer_recall = target in {"isAllowedToUse", "overridesMethod"}
    if best is not None and best >= 0.95:
        return ThresholdConfig(accept=accept_f1, refine=0.95, prefer_recall_over_precision=prefer_recall)
    if best is not None and best < accept_f1:
        return ThresholdConfig(accept=accept_f1, refine=0.85, prefer_recall_over_precision=True)
    return ThresholdConfig(accept=accept_f1, refine=0.9, prefer_recall_over_precision=prefer_recall)


def choose_bounds(target: str, memory_record: MemoryRecord | None) -> SearchBounds:
    if target == "overridesMethod":
        return SearchBounds(max_depth=5, max_proof_trees=5, max_states_per_node=3000, max_edges_per_node=120)
    if memory_record and (memory_record.best_f1 or 0.0) < 0.8:
        return SearchBounds(max_depth=5, max_proof_trees=5, max_states_per_node=2500, max_edges_per_node=120)
    return SearchBounds()


def estimate_confidence(
    intent: IntentProfile,
    memory_record: MemoryRecord | None,
    path_programs: list[list[str]],
) -> ConfidenceProfile:
    historical_similarity = 0.0
    witness_coverage = 0.0
    schema_complexity = 0.5
    if memory_record:
        historical_similarity = min(1.0, 0.5 + 0.1 * memory_record.accepted_candidate_count)
        witness_coverage = min(1.0, memory_record.best_f1 or 0.0)
    if path_programs:
        avg_len = sum(len(path) for path in path_programs) / len(path_programs)
        schema_complexity = max(0.1, min(1.0, 1.0 - (avg_len - 1.0) * 0.12))
    complexity_penalty = {"simple": 0.05, "medium": 0.12, "complex": 0.2}.get(intent.task_complexity, 0.18)
    score = max(
        0.0,
        min(
            1.0,
            0.45 * witness_coverage
            + 0.35 * historical_similarity
            + 0.20 * schema_complexity
            - complexity_penalty,
        ),
    )
    if score < 0.6:
        mode = "conservative"
    elif score < 0.85:
        mode = "balanced"
    else:
        mode = "aggressive"
    return ConfidenceProfile(
        score=round(score, 3),
        mode=mode,
        confidence_drivers={
            "positive_witness_coverage": round(witness_coverage, 3),
            "historical_similarity": round(historical_similarity, 3),
            "schema_simplicity": round(schema_complexity, 3),
            "complexity_penalty": round(complexity_penalty, 3),
        },
        rationale=(
            "Confidence combines prior verifier F1, memory similarity, path/schema simplicity, "
            "and a task-complexity penalty."
        ),
    )


def apply_confidence_mode(
    threshold: ThresholdConfig,
    bounds: SearchBounds,
    confidence: ConfidenceProfile,
) -> tuple[ThresholdConfig, SearchBounds]:
    if confidence.mode == "conservative":
        return (
            ThresholdConfig(
                accept=max(threshold.accept, 0.85),
                refine=max(threshold.refine, 0.9),
                prefer_recall_over_precision=False,
            ),
            SearchBounds(
                max_depth=min(bounds.max_depth, 4),
                max_proof_trees=bounds.max_proof_trees,
                max_states_per_node=min(bounds.max_states_per_node, 2000),
                max_edges_per_node=min(bounds.max_edges_per_node, 100),
            ),
        )
    return threshold, bounds


def choose_selected_predicates(path_programs: list[list[str]], memory_record: MemoryRecord | None) -> list[str]:
    if memory_record and memory_record.selected_predicates:
        return memory_record.selected_predicates[:8]
    selected: list[str] = []
    seen: set[str] = set()
    for path in path_programs:
        for step in path:
            pred = step.split("^", 1)[0]
            if pred not in seen:
                selected.append(pred)
                seen.add(pred)
    return selected[:8]


def diagnose_strategy(strategy: ProofStrategy, memory_record: MemoryRecord | None) -> VerifierDiagnostic:
    if memory_record is None:
        return VerifierDiagnostic(
            status="needs_verifier_run",
            reason="cold_start_strategy_requires_symbolic_admission",
            suggested_repair="run train-split verifier before exporting a rule library",
        )
    best = memory_record.best_f1 or 0.0
    if best >= strategy.threshold_config.accept:
        return VerifierDiagnostic(
            status="admissible_from_memory",
            reason=memory_record.final_stop_reason,
            positive_coverage=best,
            negative_contamination=max(0.0, 1.0 - best),
        )
    return VerifierDiagnostic(
        status="repair_required",
        reason=f"best_f1_below_acceptance:{best:.3f}",
        positive_coverage=best,
        suggested_repair="expand bounds or add relation-specific equality guards",
    )


def should_refine(strategy: ProofStrategy, diagnostic: VerifierDiagnostic) -> bool:
    confidence = strategy.confidence_profile
    if diagnostic.status in {"repair_required", "needs_verifier_run"}:
        return True
    return bool(confidence and confidence.mode == "conservative")


def reflect_on_diagnostic(strategy: ProofStrategy, diagnostic: VerifierDiagnostic) -> ReflectionRecord:
    confidence = strategy.confidence_profile
    if diagnostic.status == "repair_required" and "best_f1_below_acceptance" in diagnostic.reason:
        return ReflectionRecord(
            trigger=diagnostic.reason,
            observation=(
                "The remembered verifier score is useful but below the confidence-adjusted "
                "acceptance threshold."
            ),
            repair_action="restore_base_acceptance_threshold_and_require_audit",
            updated_assumption=(
                "A memory strategy that clears the base threshold can be admitted only with "
                "explicit near-miss audit and held-out proof validation."
            ),
        )
    if diagnostic.status == "needs_verifier_run":
        return ReflectionRecord(
            trigger=diagnostic.reason,
            observation="Cold-start strategy has no verifier-admitted memory record.",
            repair_action="mark_ready_for_symbolic_admission",
            updated_assumption="The next step must run train-split symbolic admission before export.",
        )
    if confidence and confidence.mode == "conservative":
        return ReflectionRecord(
            trigger=f"low_confidence:{confidence.score}",
            observation="The strategy is admitted but uncertainty is high.",
            repair_action="tighten_audit_policy",
            updated_assumption="Keep the proof strategy but require stronger audit evidence.",
        )
    return ReflectionRecord(
        trigger=diagnostic.reason,
        observation="No repair required.",
        repair_action="none",
        updated_assumption="Verifier-admitted strategy can proceed.",
    )


def refine_strategy(
    strategy: ProofStrategy,
    diagnostic: VerifierDiagnostic,
    reflection: ReflectionRecord,
    iteration: int,
) -> ProofStrategy:
    threshold = strategy.threshold_config
    bounds = strategy.search_bounds
    audit = strategy.audit_config
    fallback = strategy.fallback_strategy
    rationale = strategy.rationale
    candidate_rules = list(strategy.candidate_rules)
    selected_predicates = list(strategy.selected_predicates)
    if reflection.repair_action == "restore_base_acceptance_threshold_and_require_audit":
        threshold = ThresholdConfig(
            accept=0.8,
            refine=max(threshold.refine, 0.9),
            prefer_recall_over_precision=threshold.prefer_recall_over_precision,
        )
        audit = AuditConfig(
            counterfactual_sample_size=max(audit.counterfactual_sample_size, 30),
            near_miss_stress_enabled=True,
            proof_tree_audit_depth=max(audit.proof_tree_audit_depth, 4),
        )
        fallback = "audit_guarded_memory_admission"
        rationale += " Refined by restoring the base verifier threshold and strengthening audit requirements."
    elif reflection.repair_action == "mark_ready_for_symbolic_admission":
        fallback = "run_symbolic_admission_before_export"
        rationale += " Refined by marking the strategy as ready for symbolic admission."
    elif reflection.repair_action == "tighten_audit_policy":
        audit = AuditConfig(
            counterfactual_sample_size=max(audit.counterfactual_sample_size, 30),
            near_miss_stress_enabled=True,
            proof_tree_audit_depth=max(audit.proof_tree_audit_depth, 4),
        )
        fallback = "admit_with_strengthened_audit"
        rationale += " Refined by tightening the audit policy under conservative confidence."
    if strategy.target == "overridesMethod":
        for predicate in ("methodArity", "methodName"):
            if predicate not in selected_predicates:
                selected_predicates.append(predicate)
    confidence = strategy.confidence_profile
    if confidence:
        confidence = ConfidenceProfile(
            score=confidence.score,
            mode="balanced" if reflection.repair_action != "mark_ready_for_symbolic_admission" else confidence.mode,
            confidence_drivers=confidence.confidence_drivers,
            rationale=confidence.rationale + " Reflection attached an explicit verifier obligation.",
        )
    return ProofStrategy(
        strategy_id=f"{strategy.strategy_id}_refined{iteration}",
        target=strategy.target,
        path_programs=strategy.path_programs,
        threshold_config=threshold,
        search_bounds=bounds,
        audit_config=audit,
        fallback_strategy=fallback,
        memory_enabled=strategy.memory_enabled,
        intent_profile=strategy.intent_profile,
        confidence_profile=confidence,
        selected_predicates=selected_predicates,
        candidate_rules=candidate_rules,
        rationale=rationale,
    )


def diagnose_refined_strategy(
    strategy: ProofStrategy,
    previous_diagnostic: VerifierDiagnostic,
) -> VerifierDiagnostic:
    coverage = previous_diagnostic.positive_coverage
    if coverage is not None and coverage >= strategy.threshold_config.accept:
        return VerifierDiagnostic(
            status="admissible_after_reflection",
            reason="memory_score_meets_refined_threshold_with_audit_obligation",
            positive_coverage=coverage,
            negative_contamination=max(0.0, 1.0 - coverage),
            suggested_repair=None,
        )
    if previous_diagnostic.status == "needs_verifier_run":
        return VerifierDiagnostic(
            status="ready_for_symbolic_admission",
            reason="cold_strategy_refined_to_explicit_verifier_obligation",
            positive_coverage=coverage,
            suggested_repair="execute train-split admission before rule export",
        )
    return VerifierDiagnostic(
        status="repair_required",
        reason=previous_diagnostic.reason,
        positive_coverage=coverage,
        suggested_repair=previous_diagnostic.suggested_repair,
        missing_relation_types=previous_diagnostic.missing_relation_types,
    )


def append_refinement_trace(
    trace: DecisionTrace,
    strategy: ProofStrategy,
    diagnostic: VerifierDiagnostic,
    reflection: ReflectionRecord,
) -> DecisionTrace:
    decisions = list(trace.decisions)
    next_step = max((decision.step for decision in decisions), default=0) + 1
    decisions.extend(
        [
            Decision(
                step=next_step,
                action="Reflect",
                rationale="Use verifier diagnostics and confidence to select a bounded repair.",
                payload={"reflection": asdict(reflection)},
            ),
            Decision(
                step=next_step + 1,
                action="RefineStrategy",
                rationale=reflection.updated_assumption,
                payload={"strategy": asdict(strategy)},
            ),
            Decision(
                step=next_step + 2,
                action="VerifierDiagnosticAfterRefinement",
                rationale="Re-evaluate the refined strategy before exposing it to execution.",
                payload={"diagnostic": asdict(diagnostic)},
            ),
        ]
    )
    return DecisionTrace(
        trace_id=f"trace_{strategy.strategy_id}",
        target=strategy.target,
        strategy_id=strategy.strategy_id,
        decisions=decisions,
        verifier_diagnostic=diagnostic,
    )


def build_rationale(
    target: str,
    memory_record: MemoryRecord | None,
    threshold: ThresholdConfig,
    bounds: SearchBounds,
) -> str:
    if memory_record:
        return (
            f"Use prior {target} verifier outcome from {memory_record.split_name}; "
            f"best_f1={memory_record.best_f1}, accepted={memory_record.accepted_candidate_count}, "
            f"threshold={threshold.accept}, depth={bounds.max_depth}."
        )
    return (
        f"Cold-start {target} with typed default path programs; "
        f"threshold={threshold.accept}, depth={bounds.max_depth}."
    )


def extract_candidate_rules(learn_json: dict[str, Any]) -> list[list[str]]:
    text = json.dumps(learn_json, ensure_ascii=False)
    rules = []
    for match in re.findall(r"([A-Za-z_][A-Za-z0-9_]*\\([^\\n]+?\\) :- [^\\n\"]+)", text):
        body = match.split(":-", 1)[1]
        predicates = []
        for pred in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\\(", body):
            predicates.append(pred)
        if predicates:
            rules.append(predicates)
    unique: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for rule in rules:
        key = tuple(rule)
        if key not in seen:
            unique.append(rule)
            seen.add(key)
    return unique


def extract_rule_library_payload(path: Path) -> dict[str, list[str] | list[list[str]]]:
    if not path.exists():
        return {"path_programs": [], "selected_predicates": [], "rule_texts": []}
    try:
        library = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"path_programs": [], "selected_predicates": [], "rule_texts": []}
    paths: list[list[str]] = []
    selected_predicates: list[str] = []
    rule_texts: list[str] = []
    for rule in library.get("rules", []):
        if isinstance(rule.get("rule"), str):
            rule_texts.append(rule["rule"])
        for program in rule.get("path_programs", []) or []:
            flattened: list[str] = []
            for step in program:
                if isinstance(step, (list, tuple)) and step:
                    predicate = str(step[0])
                    direction = str(step[1]) if len(step) > 1 else "fwd"
                    flattened.append(f"{predicate}^{direction}")
                elif isinstance(step, str):
                    flattened.append(step)
            if flattened:
                paths.append(flattened)
        if not paths and isinstance(rule.get("rule"), str):
            paths.extend(extract_candidate_rules({"rule": rule["rule"]}))
        for dep in (((library.get("dependency_graph") or {}).get("edb_predicates")) or []):
            selected_predicates.append(str(dep).split("/", 1)[0])
    unique: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for path_program in paths:
        key = tuple(path_program)
        if key not in seen:
            unique.append(path_program)
            seen.add(key)
    return {
        "path_programs": unique,
        "selected_predicates": sorted(set(selected_predicates)),
        "rule_texts": list(dict.fromkeys(rule_texts)),
    }


def infer_target(split_name: str, manifest: dict[str, Any]) -> str:
    for target in ("canCallClass", "isAllowedToUse", "overridesMethod"):
        if target in split_name:
            return target
    task_dir = str(manifest.get("task_dir", ""))
    for target in ("canCallClass", "isAllowedToUse", "overridesMethod"):
        if target in task_dir:
            return target
    return "unknown"


def normalize_target(target: str) -> str:
    return target.split("/", 1)[0]


def coerce_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_int(value: Any, *, default: int) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
