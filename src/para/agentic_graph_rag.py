"""Agentic GraphRAG guidance for PARA.

The provider in this module turns the existing deterministic GraphRAG
candidate generator into a small closed-loop agent workflow:

Planner -> Retriever -> Verifier -> Refiner.

It is deliberately conservative.  Agents may change retrieval parameters and
constraint policies, but candidate rules are accepted only when the symbolic
evaluator reports an improvement.  This makes the workflow suitable for paper
experiments where agentic exploration must remain reproducible.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .evaluate import build_fact_index, choose_best_rule, covers, evaluate_rule
from .graph_rag import (
    GraphRAGCandidateGenerator,
    PathSignature,
    build_fact_graph,
    is_attribute_or_pair_constraint,
    rank_predicates_from_graph,
    retrieve_paths,
)
from .guidance import GuideProvider, extract_json_object
from .llm_clients import chat_text, is_local_endpoint
from .models import Guidance, Rule, TaskData
from .prolog import parse_rule, rule_to_text


JSON_GBNF = r'''
root   ::= object
value  ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws (string ws ":" ws value ("," ws string ws ":" ws value)*)? "}" ws
array  ::= "[" ws (value ("," ws value)*)? "]" ws
string ::= "\"" ([^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
ws     ::= [ \t\n\r]*
'''

ACTION_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["path_queries"],
    "properties": {
        "path_queries": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "array",
                "items": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "string"},
                },
            },
        },
        "required_predicates": {"type": "array", "items": {"type": "string"}},
        "avoid_predicates": {"type": "array", "items": {"type": "string"}},
        "preferred_constraint_predicates": {"type": "array", "items": {"type": "string"}},
        "strategy_presets": {"type": "array", "items": {"type": "string"}},
        "feedback_label": {"type": "string"},
        "max_depth": {"type": "integer"},
        "max_positive_examples": {"type": "integer"},
        "max_paths_per_example": {"type": "integer"},
        "max_edges_per_node": {"type": "integer"},
        "max_candidates": {"type": "integer"},
        "enable_pair_constraints": {"type": "boolean"},
        "traversal_strategy": {"type": "string"},
        "constraint_mode": {"type": "string"},
        "min_path_support": {"type": "integer"},
        "candidate_body_extra": {"type": "integer"},
        "seed_max_depth": {"type": "integer"},
        "seed_max_paths_per_example": {"type": "integer"},
        "expand_seed_on_miss": {"type": "boolean"},
        "rationale": {"type": "string"},
        "reason": {"type": "string"},
        "repair_intent": {"type": "string"},
    },
}

ACTION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "ashrl_agent_action",
        "strict": True,
        "schema": ACTION_JSON_SCHEMA,
    },
}

COMPACT_ACTION_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "path_queries",
        "required_predicates",
        "avoid_predicates",
        "preferred_constraint_predicates",
        "strategy_presets",
    ],
    "properties": {
        "path_queries": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "array",
                "items": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "string"},
                },
            },
        },
        "required_predicates": {"type": "array", "items": {"type": "string"}},
        "avoid_predicates": {"type": "array", "items": {"type": "string"}},
        "preferred_constraint_predicates": {"type": "array", "items": {"type": "string"}},
        "strategy_presets": {"type": "array", "items": {"type": "string"}},
    },
}

COMPACT_ACTION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "ashrl_compact_agent_action",
        "strict": True,
        "schema": COMPACT_ACTION_JSON_SCHEMA,
    },
}

TYPED_PATH_STUDENT_SYSTEM_PROMPT = (
    "/no_think\n"
    "You are an agent-symbolic software architecture retrieval planner. "
    "Generate short typed predicate paths for later indexed symbolic execution. "
    "Return the JSON object immediately. Do not explain, reason aloud, use Markdown, "
    "or emit <think> text. The JSON object must have keys typed_paths, required_predicates, "
    "preferred_constraint_predicates, confidence, and rationale. "
    "Each step must have keys from, predicate, direction, and to. "
    "direction must be fwd or rev. Emit one to three short paths and use only the provided predicates."
)

STUDENT_ARCHITECTURE_PREDICATE_PRIORITY = (
    "containsPackage",
    "containsClass",
    "containsMethod",
    "importsClass",
    "callsMethod",
    "inheritsClass",
    "extendsClass",
    "implementsInterface",
    "methodName",
    "methodArity",
    "sameMethodName",
    "sameMethodArity",
)

STRICT_NEUTRAL_FORMAT_EXAMPLES = (
    {
        "note": "syntax only: one abstract forward edge",
        "output": {"path_queries": [[["relation_1", "fwd"]]]},
    },
    {
        "note": "syntax only: two abstract edges with a reverse traversal",
        "output": {
            "path_queries": [[["relation_1", "fwd"], ["relation_2", "rev"]]],
            "required_predicates": ["relation_1"],
        },
    },
)


@dataclass(frozen=True)
class RetrievalPlan:
    """Bounded retrieval policy proposed by the planner agent."""

    max_depth: int = 5
    max_positive_examples: int = 8
    max_paths_per_example: int = 12
    max_edges_per_node: int = 80
    max_candidates: int = 30
    enable_pair_constraints: bool = True
    traversal_strategy: str = "bfs"
    constraint_mode: str = "direct"
    min_path_support: int = 1
    candidate_body_extra: int = 0
    seed_max_depth: int = 3
    seed_max_paths_per_example: int = 8
    expand_seed_on_miss: bool = True


@dataclass(frozen=True)
class AgentAction:
    """Executable action proposed by Planner or Refiner."""

    required_predicates: Tuple[str, ...] = ()
    avoid_predicates: Tuple[str, ...] = ()
    preferred_constraint_predicates: Tuple[str, ...] = ()
    strategy_presets: Tuple[str, ...] = ()
    path_queries: Tuple[PathSignature, ...] = ()
    rejected_path_queries: Tuple[str, ...] = ()


def limit_action_path_queries(action: AgentAction, max_path_queries: int) -> AgentAction:
    """Return the same action with a bounded executable path-program portfolio."""

    cap = max(1, min(5, max_path_queries))
    if len(action.path_queries) <= cap:
        return action
    return AgentAction(
        required_predicates=action.required_predicates,
        avoid_predicates=action.avoid_predicates,
        preferred_constraint_predicates=action.preferred_constraint_predicates,
        strategy_presets=action.strategy_presets,
        path_queries=action.path_queries[:cap],
        rejected_path_queries=action.rejected_path_queries,
    )


@dataclass
class AgenticIteration:
    """Trace record for one planner-retriever-verifier-refiner cycle."""

    iteration: int
    incoming_feedback: Optional[str]
    plan: RetrievalPlan
    action: AgentAction
    planner_source: str
    planner_message: str
    candidate_count: int
    best_rule: Optional[str]
    best_metrics: Optional[Dict[str, object]]
    verifier_feedback: str
    refiner_feedback: Optional[str]
    refiner_action: AgentAction
    refiner_source: str
    refiner_message: str
    accepted_by_symbolic_guard: bool
    diagnostics: Dict[str, object]


@dataclass(frozen=True)
class PlanDecision:
    """Planner output plus provenance."""

    plan: RetrievalPlan
    action: AgentAction
    source: str
    message: str


@dataclass(frozen=True)
class RefinementDecision:
    """Refiner output plus provenance."""

    feedback: Optional[str]
    action: AgentAction
    source: str
    message: str


class AgentLLMClient:
    """Small OpenAI-compatible JSON client for the agent roles."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        request_timeout: int = 120,
        use_grammar: bool = True,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv("PARA_LLM_BASE_URL")
            or os.getenv("NSHRL_LLM_BASE_URL")
            or "http://127.0.0.1:8000"
        ).rstrip("/")
        self.model = model or os.getenv("PARA_LLM_MODEL") or os.getenv("NSHRL_LLM_MODEL") or "Qwen3.5-27B-MaxCtx"
        self.request_timeout = request_timeout
        self.use_grammar = use_grammar and is_local_endpoint(self.base_url)
        self.enable_text_salvage = os.getenv("NSHRL_AGENT_ENABLE_TEXT_SALVAGE", "0") == "1"
        trace_path = os.getenv("NSHRL_AGENT_LLM_TRACE_JSONL")
        self.trace_path = Path(trace_path) if trace_path else None
        self._call_counter = 0

    def chat_json(
        self,
        role: str,
        prompt: str,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        grammar: Optional[str] = None,
        response_format: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        self._call_counter += 1
        call_id = self._call_counter
        effective_system_prompt = system_prompt or default_agent_system_prompt(role)
        content = self._chat_text(
            role,
            prompt,
            max_tokens=max_tokens,
            system_prompt=effective_system_prompt,
            grammar=grammar,
            response_format=response_format,
        )
        try:
            parsed = extract_json_object(str(content))
            self._append_trace(
                {
                    "call_id": call_id,
                    "phase": "initial",
                    "role": role,
                    "model": self.model,
                    "system_prompt": effective_system_prompt,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "response_format": response_format or {"type": "json_object"},
                    "raw_output": str(content),
                    "parsed_json": parsed,
                    "parse_ok": True,
                }
            )
            return parsed
        except Exception as first_exc:
            self._append_trace(
                {
                    "call_id": call_id,
                    "phase": "initial",
                    "role": role,
                    "model": self.model,
                    "system_prompt": effective_system_prompt,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "response_format": response_format or {"type": "json_object"},
                    "raw_output": str(content),
                    "parse_ok": False,
                    "parse_error": str(first_exc),
                }
            )
            repair_prompt = action_repair_prompt(prompt, str(content), response_format=response_format)
            repair_max_tokens = max_tokens
            repaired = self._chat_text(
                role,
                repair_prompt,
                max_tokens=repair_max_tokens,
                system_prompt=(
                    "/no_think\n"
                    "You are a strict JSON action serializer. Return exactly one JSON object. "
                    "No prose. No markdown. The first character must be `{`."
                ),
                grammar=grammar,
                response_format=response_format,
            )
            try:
                parsed = extract_json_object(repaired)
                self._append_trace(
                    {
                        "call_id": call_id,
                        "phase": "repair",
                        "role": role,
                        "model": self.model,
                        "system_prompt": (
                            "/no_think\n"
                            "You are a strict JSON action serializer. Return exactly one JSON object. "
                            "No prose. No markdown. The first character must be `{`."
                        ),
                        "prompt": repair_prompt,
                        "max_tokens": repair_max_tokens,
                        "response_format": response_format or {"type": "json_object"},
                        "raw_output": str(repaired),
                        "parsed_json": parsed,
                        "parse_ok": True,
                    }
                )
                return parsed
            except Exception as second_exc:
                self._append_trace(
                    {
                        "call_id": call_id,
                        "phase": "repair",
                        "role": role,
                        "model": self.model,
                        "system_prompt": (
                            "/no_think\n"
                            "You are a strict JSON action serializer. Return exactly one JSON object. "
                            "No prose. No markdown. The first character must be `{`."
                        ),
                        "prompt": repair_prompt,
                        "max_tokens": repair_max_tokens,
                        "response_format": response_format or {"type": "json_object"},
                        "raw_output": str(repaired),
                        "parse_ok": False,
                        "parse_error": str(second_exc),
                    }
                )
                if response_format == ACTION_RESPONSE_FORMAT and self.enable_text_salvage:
                    try:
                        parsed = action_json_from_model_text("\n".join((str(content), str(repaired))))
                        self._append_trace(
                            {
                                "call_id": call_id,
                                "phase": "salvage",
                                "role": role,
                                "model": self.model,
                                "system_prompt": "local action serializer over model text",
                                "prompt": "Extract path_queries and constraint predicate names already emitted by the model.",
                                "max_tokens": 0,
                                "response_format": response_format or {"type": "json_object"},
                                "raw_output": "\n".join((str(content), str(repaired)))[:12000],
                                "parsed_json": parsed,
                                "parse_ok": True,
                            }
                        )
                        return parsed
                    except Exception as salvage_exc:
                        self._append_trace(
                            {
                                "call_id": call_id,
                                "phase": "salvage",
                                "role": role,
                                "model": self.model,
                                "system_prompt": "local action serializer over model text",
                                "prompt": "Extract path_queries and constraint predicate names already emitted by the model.",
                                "max_tokens": 0,
                                "response_format": response_format or {"type": "json_object"},
                                "raw_output": "\n".join((str(content), str(repaired)))[:12000],
                                "parse_ok": False,
                                "parse_error": str(salvage_exc),
                            }
                        )
                raise ValueError(
                    f"{first_exc}; repair also failed: {second_exc}; "
                    f"original={str(content)[:1000]!r}; repaired={str(repaired)[:1000]!r}"
                ) from second_exc

    def _chat_text(
        self,
        role: str,
        prompt: str,
        max_tokens: int,
        system_prompt: Optional[str] = None,
        grammar: Optional[str] = None,
        response_format: Optional[Dict[str, object]] = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or default_agent_system_prompt(role),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
            "response_format": response_format or {"type": "json_object"},
        }
        if self.use_grammar:
            # llama.cpp accepts grammar as an OpenAI-compatible extension.
            # Sending both spellings keeps this compatible across server builds.
            payload["grammar"] = grammar or JSON_GBNF
            payload["grammar_string"] = grammar or JSON_GBNF
        request_payload = dict(payload)
        if not self.use_grammar:
            request_payload.pop("grammar", None)
            request_payload.pop("grammar_string", None)
        return chat_text(
            base_url=self.base_url,
            model=self.model,
            messages=request_payload["messages"],
            request_timeout=self.request_timeout,
            max_tokens=max_tokens,
            response_format=request_payload.get("response_format"),
            temperature=0.0,
            extra_payload={
                key: value
                for key, value in request_payload.items()
                if key not in {"model", "messages", "temperature", "max_tokens", "stream", "response_format"}
            },
        )

    def _post_chat(self, payload: Dict[str, object]) -> str:
        """Compatibility shim for older callers that still expect raw JSON text."""

        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": chat_text(
                                base_url=self.base_url,
                                model=self.model,
                                messages=payload["messages"],
                                request_timeout=self.request_timeout,
                                max_tokens=int(payload.get("max_tokens", 1024)),
                                response_format=payload.get("response_format"),
                                temperature=float(payload.get("temperature", 0.0)),
                                extra_payload={
                                    key: value
                                    for key, value in payload.items()
                                    if key
                                    not in {
                                        "model",
                                        "messages",
                                        "temperature",
                                        "max_tokens",
                                        "stream",
                                        "response_format",
                                    }
                                },
                            )
                        }
                    }
                ]
            }
        )

    def _append_trace(self, record: Dict[str, object]) -> None:
        if self.trace_path is None:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def default_agent_system_prompt(role: str) -> str:
    return (
        "/no_think\n"
        f"You are the {role} in a multi-agent system for software-engineering rule learning. "
        "Return exactly one valid JSON object. No markdown. No prose outside JSON. "
        "Do not include <think> tags. The first character of the assistant message must be `{`."
    )


def action_repair_prompt(
    original_prompt: str,
    invalid_response: str,
    response_format: Optional[Dict[str, object]] = None,
) -> str:
    schema_hint = {
        "path_queries": [[["predicateName", "fwd"]]],
        "required_predicates": ["predicateName"],
        "preferred_constraint_predicates": ["predicateName"],
        "feedback_label": "too_general",
    }
    if response_format:
        schema_hint = {
            "path_queries": [[["predicateName", "fwd"], ["predicateName", "rev"]]],
            "required_predicates": [],
            "avoid_predicates": [],
            "preferred_constraint_predicates": [],
            "strategy_presets": [],
            "feedback_label": "too_general",
        }
    return "\n".join(
        [
            "/no_think",
            "Convert the previous answer into one compact JSON action. Do not solve the task again.",
            "If the previous answer contains a witness signature like [[\"p\",\"fwd\"]], copy it into path_queries.",
            "If it mentions direct pair constraints such as sameMethodName/sameMethodArity, put them in preferred_constraint_predicates.",
            "If only pair constraints are needed, use an empty path_queries array.",
            "Output shape:",
            json.dumps(schema_hint, ensure_ascii=False),
            "Rules:",
            "- first character must be `{`",
            "- do not explain",
            "- do not use markdown",
            "- use only predicate names present in the original task",
            "- path_queries is an array of paths; each path is an array of [predicate,direction] pairs",
            "- direction must be fwd or rev",
            "Relevant original task excerpt:",
            original_prompt[:2500],
            "Previous invalid answer:",
            invalid_response[:6000],
        ]
    )


def action_json_from_model_text(text: str) -> Dict[str, object]:
    """Recover an action JSON object from the model's own non-JSON analysis.

    This is intentionally narrow: it only serializes explicit [predicate,
    direction] arrays and predicate names already present in the model output.
    Semantic validation still happens later in ``coerce_agent_action``.
    """

    decoder = json.JSONDecoder()
    path_queries: List[List[List[str]]] = []
    for index, char in enumerate(text):
        if char != "[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except Exception:
            continue
        for query in _extract_path_query_lists(value):
            if query not in path_queries:
                path_queries.append(query)

    constraint_names = []
    for name in (
        "sameMethodName",
        "sameMethodArity",
        "methodName",
        "methodArity",
        "sameClassName",
        "samePackageName",
    ):
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text) and name not in constraint_names:
            constraint_names.append(name)

    if not path_queries and not constraint_names:
        raise ValueError("No explicit path_queries or pair-constraint predicates found in model text")

    required = []
    for query in path_queries:
        for predicate, _direction in query:
            if predicate not in required:
                required.append(predicate)
    return {
        "path_queries": path_queries[:5],
        "required_predicates": required,
        "avoid_predicates": [],
        "preferred_constraint_predicates": constraint_names,
        "strategy_presets": [],
    }


def _extract_path_query_lists(value: object) -> List[List[List[str]]]:
    if _is_path_query(value):
        return [[list(step) for step in value]]  # type: ignore[arg-type]
    if not isinstance(value, list):
        return []
    output: List[List[List[str]]] = []
    if value and all(_is_path_query(item) for item in value):
        for item in value:
            query = [list(step) for step in item]  # type: ignore[arg-type]
            if query not in output:
                output.append(query)
        return output
    for item in value:
        for query in _extract_path_query_lists(item):
            if query not in output:
                output.append(query)
    return output


def _is_path_query(value: object) -> bool:
    return isinstance(value, list) and bool(value) and all(_is_path_step(item) for item in value)


def _is_path_step(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], str)
        and value[1].lower() in {"fwd", "rev"}
    )


class PlannerAgent:
    """Choose a retrieval plan from target metadata and verifier feedback."""

    def __init__(
        self,
        base_plan: RetrievalPlan,
        llm: Optional[AgentLLMClient] = None,
        strict_llm: bool = True,
        compact_actions: bool = False,
        typed_path_student: bool = False,
        use_symbolic_prior: bool = True,
        schema_profile_mode: str = "assisted",
        max_path_queries: int = 5,
        witness_evidence_mode: str = "full",
    ) -> None:
        self.base_plan = base_plan
        self.llm = llm
        self.strict_llm = strict_llm
        self.compact_actions = compact_actions
        self.typed_path_student = typed_path_student
        self.use_symbolic_prior = use_symbolic_prior
        self.schema_profile_mode = schema_profile_mode
        self.max_path_queries = max(1, min(5, max_path_queries))
        self.witness_evidence_mode = witness_evidence_mode

    def plan(self, task: TaskData, objective: str, feedback: Optional[str], iteration: int) -> PlanDecision:
        symbolic_plan = self._symbolic_plan(task, feedback, iteration)
        symbolic_action = symbolic_action_for(task, feedback) if self.use_symbolic_prior else AgentAction()
        if self.witness_evidence_mode == "deterministic_top1":
            action = deterministic_witness_action(task, max_path_queries=1)
            return PlanDecision(
                symbolic_plan,
                action,
                "deterministic_witness_top1",
                "Selected the highest-ranked train-example witness signature without an LLM.",
            )
        if self.llm is None:
            if self.strict_llm:
                raise RuntimeError("agent Planner requires an LLM client; pass --llm-base-url/--llm-model or disable strict fallback")
            return PlanDecision(symbolic_plan, symbolic_action, "symbolic_fallback", "LLM planner disabled; used symbolic fallback.")

        prompt = planner_prompt(
            task,
            objective,
            feedback,
            iteration,
            self.base_plan,
            symbolic_plan,
            compact_actions=self.compact_actions,
            schema_profile_mode=self.schema_profile_mode,
            max_path_queries=self.max_path_queries,
            include_witness_evidence=self.witness_evidence_mode == "full",
        )
        try:
            if self.typed_path_student:
                data = self.llm.chat_json(
                    "Typed-Path Planner Agent",
                    typed_path_student_prompt(task),
                    max_tokens=512,
                    system_prompt=TYPED_PATH_STUDENT_SYSTEM_PROMPT,
                )
                action = limit_action_path_queries(
                    merge_actions(symbolic_action, typed_path_action_from_json(data, task)),
                    self.max_path_queries,
                )
                message = str(data.get("rationale") or "Typed-path student proposed an indexed path program.")
                return PlanDecision(symbolic_plan, action, "llm_typed_path_planner", message)
            data = self.llm.chat_json(
                "Planner Agent",
                prompt,
                max_tokens=1536 if self.compact_actions else 2048,
                response_format=COMPACT_ACTION_RESPONSE_FORMAT if self.compact_actions else ACTION_RESPONSE_FORMAT,
            )
            plan = coerce_retrieval_plan(data, symbolic_plan, task)
            action = limit_action_path_queries(
                merge_actions(symbolic_action, coerce_agent_action(data, task)),
                self.max_path_queries,
            )
            message = str(data.get("rationale") or data.get("reason") or "LLM planner proposed retrieval plan.")
            return PlanDecision(plan, action, "llm_planner", message)
        except Exception as exc:
            if self.strict_llm:
                raise RuntimeError(f"LLM planner failed: {exc}") from exc
            return PlanDecision(symbolic_plan, symbolic_action, "symbolic_fallback_after_llm_error", f"LLM planner failed: {exc}")

    def _symbolic_plan(self, task: TaskData, feedback: Optional[str], iteration: int) -> RetrievalPlan:
        plan = self.base_plan
        target_name = task.target.name.lower()
        max_body = max(1, task.max_body)

        max_depth = min(max_body, plan.max_depth)
        max_paths = plan.max_paths_per_example
        max_candidates = plan.max_candidates
        constraint_mode = plan.constraint_mode
        enable_constraints = plan.enable_pair_constraints
        min_support = plan.min_path_support
        seed_max_depth = min(max_body, plan.seed_max_depth)
        seed_max_paths = plan.seed_max_paths_per_example

        if feedback in {"too_specific", "no_candidate"}:
            max_depth = min(max_body, max_depth + iteration)
            max_paths = max_paths + 8 * iteration
            max_candidates = max_candidates + 20 * iteration
            seed_max_depth = min(max_depth, seed_max_depth + iteration)
            seed_max_paths = seed_max_paths + 4 * iteration
            min_support = 1
            if iteration > 1:
                constraint_mode = "none"
                enable_constraints = False
        elif feedback == "too_general":
            constraint_mode = "both"
            enable_constraints = True
            max_candidates = max_candidates + 10 * iteration
            if sum(1 for example in task.examples if example.positive) >= 4:
                min_support = max(1, min_support)
        elif feedback == "mixed_errors":
            max_depth = min(max_body, max_depth + 1)
            max_paths = max_paths + 6 * iteration
            max_candidates = max_candidates + 15 * iteration
            constraint_mode = "both"
            enable_constraints = True

        if self.use_symbolic_prior and "override" in target_name:
            constraint_mode = "both"
            enable_constraints = True
        elif self.use_symbolic_prior and "call" in target_name and feedback == "too_general":
            max_edges = max(20, plan.max_edges_per_node // 2)
            return RetrievalPlan(
                max_depth=max_depth,
                max_positive_examples=plan.max_positive_examples,
                max_paths_per_example=max_paths,
                max_edges_per_node=max_edges,
                max_candidates=max_candidates,
                enable_pair_constraints=enable_constraints,
                traversal_strategy=plan.traversal_strategy,
                constraint_mode=constraint_mode,
                min_path_support=min_support,
                candidate_body_extra=plan.candidate_body_extra,
                seed_max_depth=seed_max_depth,
                seed_max_paths_per_example=seed_max_paths,
                expand_seed_on_miss=plan.expand_seed_on_miss,
            )

        return RetrievalPlan(
            max_depth=max_depth,
            max_positive_examples=plan.max_positive_examples,
            max_paths_per_example=max_paths,
            max_edges_per_node=plan.max_edges_per_node,
            max_candidates=max_candidates,
            enable_pair_constraints=enable_constraints,
            traversal_strategy=plan.traversal_strategy,
            constraint_mode=constraint_mode,
            min_path_support=min_support,
            candidate_body_extra=plan.candidate_body_extra,
            seed_max_depth=seed_max_depth,
            seed_max_paths_per_example=seed_max_paths,
            expand_seed_on_miss=plan.expand_seed_on_miss,
        )


class RetrieverAgent:
    """Execute a retrieval plan with the existing GraphRAG generator."""

    def retrieve(
        self,
        task: TaskData,
        plan: RetrievalPlan,
        focus_predicates: Sequence[str] = (),
        path_queries: Sequence[PathSignature] = (),
        planned_queries_only: bool = False,
    ) -> Tuple[List[Rule], Dict[str, object]]:
        generator = GraphRAGCandidateGenerator(
            max_depth=plan.max_depth,
            max_positive_examples=plan.max_positive_examples,
            max_paths_per_example=plan.max_paths_per_example,
            max_edges_per_node=plan.max_edges_per_node,
            max_candidates=plan.max_candidates,
            enable_pair_constraints=plan.enable_pair_constraints,
            traversal_strategy=plan.traversal_strategy,
            constraint_mode=plan.constraint_mode,
            min_path_support=plan.min_path_support,
            candidate_max_body=task.max_body + max(0, plan.candidate_body_extra),
            focus_predicates=focus_predicates,
            seed_max_depth=plan.seed_max_depth,
            seed_max_paths_per_example=plan.seed_max_paths_per_example,
            expand_seed_on_miss=plan.expand_seed_on_miss,
            planned_path_queries=path_queries,
            planned_queries_only=planned_queries_only,
        )
        return generator.generate(task)


class VerifierAgent:
    """Score candidates with PARA's symbolic evaluator."""

    def verify(self, task: TaskData, candidates: Sequence[Rule]) -> Tuple[Optional[Rule], Optional[Dict[str, object]], str]:
        best = choose_best_rule(candidates, task.facts, task.examples)
        if best is None:
            return None, None, "no_candidate"
        rule, metrics = best
        return rule, asdict(metrics), metrics.feedback_label


class RefinerAgent:
    """Turn verifier output into the next planner feedback label."""

    def __init__(
        self,
        llm: Optional[AgentLLMClient] = None,
        strict_llm: bool = True,
        compact_actions: bool = False,
        typed_path_student: bool = False,
        use_symbolic_prior: bool = True,
        schema_profile_mode: str = "assisted",
        max_path_queries: int = 5,
        witness_evidence_mode: str = "full",
    ) -> None:
        self.llm = llm
        self.strict_llm = strict_llm
        self.compact_actions = compact_actions
        self.typed_path_student = typed_path_student
        self.use_symbolic_prior = use_symbolic_prior
        self.schema_profile_mode = schema_profile_mode
        self.max_path_queries = max(1, min(5, max_path_queries))
        self.witness_evidence_mode = witness_evidence_mode

    def next_feedback(
        self,
        task: TaskData,
        verifier_feedback: str,
        best_rule_text: Optional[str],
        best_metrics: Optional[Dict[str, object]],
        diagnostics: Dict[str, object],
    ) -> RefinementDecision:
        if verifier_feedback == "consistent":
            return RefinementDecision(None, AgentAction(), "symbolic_verifier", "Verifier found a consistent candidate.")
        if self.witness_evidence_mode == "deterministic_top1":
            return RefinementDecision(
                verifier_feedback,
                AgentAction(),
                "deterministic_witness_top1",
                "No LLM Refiner is used in the deterministic witness-only ablation.",
            )
        if self.llm is None:
            if self.strict_llm:
                raise RuntimeError("agent Refiner requires an LLM client; pass --llm-base-url/--llm-model or disable strict fallback")
            return RefinementDecision(
                verifier_feedback,
                symbolic_action_for(task, verifier_feedback) if self.use_symbolic_prior else AgentAction(),
                "symbolic_fallback",
                "LLM refiner disabled; reused verifier feedback.",
            )

        prompt = refiner_prompt(
            task,
            verifier_feedback,
            best_rule_text,
            best_metrics,
            diagnostics,
            compact_actions=self.compact_actions,
            schema_profile_mode=self.schema_profile_mode,
            max_path_queries=self.max_path_queries,
            include_witness_evidence=self.witness_evidence_mode == "full",
        )
        try:
            if self.typed_path_student:
                data = self.llm.chat_json(
                    "Typed-Path Refiner Agent",
                    typed_path_student_prompt(
                        task,
                        verifier_feedback=verifier_feedback,
                        best_rule_text=best_rule_text,
                        best_metrics=best_metrics,
                        diagnostics=diagnostics,
                    ),
                    max_tokens=512,
                    system_prompt=TYPED_PATH_STUDENT_SYSTEM_PROMPT,
                )
                action = limit_action_path_queries(
                    merge_actions(
                        symbolic_action_for(task, verifier_feedback) if self.use_symbolic_prior else AgentAction(),
                        typed_path_action_from_json(data, task),
                    ),
                    self.max_path_queries,
                )
                message = str(data.get("rationale") or "Typed-path student proposed a repaired indexed path program.")
                return RefinementDecision(verifier_feedback, action, "llm_typed_path_refiner", message)
            data = self.llm.chat_json(
                "Refiner Agent",
                prompt,
                max_tokens=2048,
                response_format=COMPACT_ACTION_RESPONSE_FORMAT if self.compact_actions else ACTION_RESPONSE_FORMAT,
            )
            proposed = str(data.get("feedback_label") or verifier_feedback)
            if proposed not in {"too_general", "too_specific", "mixed_errors", "no_candidate", "consistent"}:
                proposed = verifier_feedback
            feedback = None if proposed == "consistent" else proposed
            prior = symbolic_action_for(task, feedback) if self.use_symbolic_prior else AgentAction()
            action = limit_action_path_queries(merge_actions(prior, coerce_agent_action(data, task)), self.max_path_queries)
            message = str(data.get("rationale") or data.get("repair_intent") or "LLM refiner proposed next feedback.")
            return RefinementDecision(feedback, action, "llm_refiner", message)
        except Exception as exc:
            if self.strict_llm:
                raise RuntimeError(f"LLM refiner failed: {exc}") from exc
            return RefinementDecision(
                verifier_feedback,
                symbolic_action_for(task, verifier_feedback) if self.use_symbolic_prior else AgentAction(),
                "symbolic_fallback_after_llm_error",
                f"LLM refiner failed: {exc}",
            )


class AgenticGraphRAGGuideProvider(GuideProvider):
    """Closed-loop GraphRAG provider with a symbolic acceptance guard."""

    def __init__(
        self,
        base_plan: Optional[RetrievalPlan] = None,
        max_iterations: int = 3,
        acceptance_f1: float = 0.8,
        portfolio_size: int = 4,
        base_url: str | None = None,
        model: str | None = None,
        request_timeout: int = 120,
        strict_llm: bool = True,
        use_focused_retrieval: bool = True,
        use_indexed_path_execution: bool = True,
        deterministic_fallback: bool = False,
        indexed_plan_only: bool = False,
        compact_actions: bool = False,
        typed_path_student: bool = False,
        force_initial_retrieval_miss: bool = False,
        use_symbolic_prior: bool = True,
        schema_profile_mode: str = "assisted",
        candidate_evaluation_cap: int = 500,
        focused_retrieval_fact_limit: int = 100_000,
        max_path_queries: int = 5,
        witness_evidence_mode: str = "full",
    ) -> None:
        self.base_plan = base_plan or RetrievalPlan()
        self.max_iterations = max(1, max_iterations)
        self.acceptance_f1 = acceptance_f1
        self.portfolio_size = max(1, portfolio_size)
        self.llm = AgentLLMClient(base_url=base_url, model=model, request_timeout=request_timeout)
        self.strict_llm = strict_llm
        self.use_focused_retrieval = use_focused_retrieval
        self.use_indexed_path_execution = use_indexed_path_execution
        self.deterministic_fallback = deterministic_fallback
        self.indexed_plan_only = indexed_plan_only
        self.compact_actions = compact_actions
        self.typed_path_student = typed_path_student
        self.force_initial_retrieval_miss = force_initial_retrieval_miss
        self.use_symbolic_prior = use_symbolic_prior
        self.schema_profile_mode = schema_profile_mode
        if self.schema_profile_mode not in {"assisted", "raw"}:
            raise ValueError(f"unsupported schema_profile_mode: {self.schema_profile_mode}")
        self.witness_evidence_mode = witness_evidence_mode
        if self.witness_evidence_mode not in {"full", "schema_only", "deterministic_top1"}:
            raise ValueError(f"unsupported witness_evidence_mode: {self.witness_evidence_mode}")
        if self.indexed_plan_only and not self.use_indexed_path_execution:
            raise ValueError("indexed_plan_only requires indexed path execution")
        self.candidate_evaluation_cap = max(1, candidate_evaluation_cap)
        self.focused_retrieval_fact_limit = max(0, focused_retrieval_fact_limit)
        self.max_path_queries = max(1, min(5, max_path_queries))
        self.planner = PlannerAgent(
            self.base_plan,
            llm=self.llm,
            strict_llm=strict_llm,
            compact_actions=compact_actions,
            typed_path_student=typed_path_student,
            use_symbolic_prior=use_symbolic_prior,
            schema_profile_mode=schema_profile_mode,
            max_path_queries=self.max_path_queries,
            witness_evidence_mode=self.witness_evidence_mode,
        )
        self.retriever = RetrieverAgent()
        self.verifier = VerifierAgent()
        self.refiner = RefinerAgent(
            llm=self.llm,
            strict_llm=strict_llm,
            compact_actions=compact_actions,
            typed_path_student=typed_path_student,
            use_symbolic_prior=use_symbolic_prior,
            schema_profile_mode=schema_profile_mode,
            max_path_queries=self.max_path_queries,
            witness_evidence_mode=self.witness_evidence_mode,
        )

    def guide(
        self,
        task: TaskData,
        objective: str,
        predicate_budget: int,
        feedback: Optional[str] = None,
    ) -> Guidance:
        accepted_candidates: List[Rule] = []
        accepted_texts = set()
        traces: List[AgenticIteration] = []
        best_f1 = -1.0
        next_feedback = feedback
        next_action = AgentAction()

        for iteration in range(1, self.max_iterations + 1):
            plan_decision = self.planner.plan(task, objective, next_feedback, iteration)
            plan = plan_decision.plan
            action = merge_actions(next_action, plan_decision.action)
            counterfactual_original_action: Optional[AgentAction] = None
            if self.force_initial_retrieval_miss and iteration == 1:
                # Dataset-construction mode only: execute an empty action once
                # so the Refiner observes a real symbolic no_candidate signal.
                counterfactual_original_action = action
                action = AgentAction(
                    required_predicates=action.required_predicates,
                    avoid_predicates=action.avoid_predicates,
                    strategy_presets=action.strategy_presets,
                    rejected_path_queries=action.rejected_path_queries,
                )
            candidates, diagnostics = self._retrieve_staged(task, plan, next_feedback, action)
            if counterfactual_original_action is not None:
                diagnostics["counterfactual_initial_retrieval_miss"] = True
                diagnostics["counterfactual_original_action"] = asdict(counterfactual_original_action)
            best_rule, best_metrics, verifier_feedback = self.verifier.verify(task, candidates)
            diagnostics["schema_profile"] = schema_profile(task, mode=self.schema_profile_mode)
            diagnostics["top_candidate_summaries"] = candidate_feedback_summary(task, candidates, limit=5)
            diagnostics["failure_profile"] = failure_profile(
                task,
                best_rule,
                best_metrics,
                candidates,
                diagnostics,
                schema_profile_mode=self.schema_profile_mode,
            )
            current_f1 = float(best_metrics.get("f1", 0.0)) if best_metrics else 0.0
            accepted = current_f1 > best_f1

            if accepted:
                best_f1 = current_f1
                for rule in candidates:
                    text = rule_to_text(rule)
                    if text not in accepted_texts:
                        accepted_texts.add(text)
                        accepted_candidates.append(rule)

            best_rule_text = rule_to_text(best_rule) if best_rule else None
            accepted_but_conservative = is_accepted_but_conservative(best_metrics, self.acceptance_f1)
            diagnostics["accepted_but_conservative"] = accepted_but_conservative
            if verifier_feedback == "consistent":
                refinement = RefinementDecision(None, AgentAction(), "symbolic_guard", "Acceptance threshold reached; no further refinement.")
            elif best_f1 >= self.acceptance_f1 and accepted_but_conservative and iteration < self.max_iterations:
                refinement = self.refiner.next_feedback(
                    task=task,
                    verifier_feedback="too_specific",
                    best_rule_text=best_rule_text,
                    best_metrics=best_metrics,
                    diagnostics=diagnostics,
                )
            elif best_f1 >= self.acceptance_f1:
                refinement = RefinementDecision(None, AgentAction(), "symbolic_guard", "Acceptance threshold reached; no further refinement.")
            elif iteration >= self.max_iterations:
                refinement = RefinementDecision(verifier_feedback, AgentAction(), "iteration_budget", "Stopped before Refiner because this is the final agent iteration.")
            else:
                refinement = self.refiner.next_feedback(
                    task=task,
                    verifier_feedback=verifier_feedback,
                    best_rule_text=best_rule_text,
                    best_metrics=best_metrics,
                    diagnostics=diagnostics,
                )

            traces.append(
                AgenticIteration(
                    iteration=iteration,
                    incoming_feedback=next_feedback,
                    plan=plan,
                    action=action,
                    planner_source=(
                        "counterfactual_injected_miss"
                        if counterfactual_original_action is not None
                        else plan_decision.source
                    ),
                    planner_message=(
                        "Dataset construction injected one controlled retrieval miss. Original action: "
                        + json.dumps(asdict(counterfactual_original_action), ensure_ascii=False)
                        if counterfactual_original_action is not None
                        else plan_decision.message
                    ),
                    candidate_count=len(candidates),
                    best_rule=best_rule_text,
                    best_metrics=best_metrics,
                    verifier_feedback=verifier_feedback,
                    refiner_feedback=refinement.feedback,
                    refiner_action=refinement.action,
                    refiner_source=refinement.source,
                    refiner_message=refinement.message,
                    accepted_by_symbolic_guard=accepted,
                    diagnostics=diagnostics,
                )
            )

            if verifier_feedback == "consistent" or (best_f1 >= self.acceptance_f1 and not accepted_but_conservative):
                break
            next_feedback = refinement.feedback
            next_action = refinement.action

        candidate_predicates: List[str] = []
        seen_predicates = set()
        for rule in accepted_candidates:
            for literal in rule.body:
                if literal.predicate not in seen_predicates:
                    seen_predicates.add(literal.predicate)
                    candidate_predicates.append(literal.predicate)

        ranked = rank_predicates_from_graph(task, candidate_predicates)
        selected = ranked[: min(len(ranked), max(predicate_budget, len(candidate_predicates)))]
        for name in candidate_predicates:
            if name not in selected:
                selected.append(name)

        trace_payload = {
            "objective": objective,
            "acceptance_f1": self.acceptance_f1,
            "best_f1": max(best_f1, 0.0),
            "portfolio_size": self.portfolio_size,
            "use_focused_retrieval": self.use_focused_retrieval,
            "use_indexed_path_execution": self.use_indexed_path_execution,
            "deterministic_fallback": self.deterministic_fallback,
            "indexed_plan_only": self.indexed_plan_only,
            "typed_path_student": self.typed_path_student,
            "force_initial_retrieval_miss": self.force_initial_retrieval_miss,
            "use_symbolic_prior": self.use_symbolic_prior,
            "schema_profile_mode": self.schema_profile_mode,
            "witness_evidence_mode": self.witness_evidence_mode,
            "candidate_evaluation_cap": self.candidate_evaluation_cap,
            "focused_retrieval_fact_limit": self.focused_retrieval_fact_limit,
            "iterations": [
                {
                    **asdict(record),
                    "plan": asdict(record.plan),
                }
                for record in traces
            ],
        }

        return Guidance(
            ranked_predicates=ranked,
            selected_predicates=selected,
            candidate_rules=accepted_candidates,
            max_vars=task.max_vars,
            max_body=task.max_body,
            max_clauses=task.max_clauses,
            confidence=min(0.95, 0.55 + max(best_f1, 0.0) * 0.35),
            rationale=(
                "Agentic GraphRAG used Planner/Retriever/Verifier/Refiner cycles "
                "with a symbolic F1 guard. Trace: "
                + json.dumps(trace_payload, ensure_ascii=False)
            ),
        )

    def _retrieve_staged(
        self,
        task: TaskData,
        primary_plan: RetrievalPlan,
        feedback: Optional[str],
        action: AgentAction,
    ) -> Tuple[List[Rule], Dict[str, object]]:
        """Run primary retrieval first, then portfolio only if needed."""

        requested_focus_predicates = focus_predicates_for_action(action)
        focus_predicates = requested_focus_predicates
        focus_disabled_for_scale = len(task.facts) > self.focused_retrieval_fact_limit
        if not self.use_focused_retrieval or focus_disabled_for_scale:
            focus_predicates = ()
        executable_path_queries = action.path_queries if self.use_indexed_path_execution else ()
        primary_rules, primary_diag = self.retriever.retrieve(
            task,
            primary_plan,
            focus_predicates=focus_predicates,
            path_queries=executable_path_queries,
            planned_queries_only=self.indexed_plan_only,
        )
        primary_rules = apply_agent_action_to_candidates(task, primary_rules, action)
        primary_rules, primary_truncated = cap_candidate_rules(primary_rules, self.candidate_evaluation_cap)
        primary_best = self.verifier.verify(task, primary_rules)
        primary_metrics = primary_best[1]
        primary_f1 = float(primary_metrics.get("f1", 0.0)) if primary_metrics else 0.0
        if self.indexed_plan_only or self.portfolio_size <= 1 or primary_f1 >= self.acceptance_f1:
            return primary_rules, {
                "portfolio_size": 1,
                "merged_candidate_count": len(primary_rules),
                "primary_f1": primary_f1,
                "focus_predicates": list(focus_predicates),
                "requested_focus_predicates": list(requested_focus_predicates),
                "focus_disabled_for_scale": focus_disabled_for_scale,
                "portfolio_triggered": False,
                "portfolio_skipped_for_indexed_plan": self.indexed_plan_only,
                "use_indexed_path_execution": self.use_indexed_path_execution,
                "rejected_path_queries": list(action.rejected_path_queries),
                "candidate_evaluation_cap": self.candidate_evaluation_cap,
                "candidate_rules_truncated": primary_truncated,
                "portfolio": [
                    {
                        "index": 0,
                        "plan": asdict(primary_plan),
                        "candidate_count": len(primary_rules),
                        "diagnostics": primary_diag,
                    }
                ],
            }

        plans = retrieval_portfolio(primary_plan, self.base_plan, task, feedback, action)[: self.portfolio_size]
        merged: List[Rule] = []
        seen_rules = set()
        portfolio_diagnostics = [
            {
                "index": 0,
                "plan": asdict(primary_plan),
                "candidate_count": len(primary_rules),
                "diagnostics": primary_diag,
            }
        ]
        for rule in primary_rules:
            text = rule_to_text(rule)
            seen_rules.add(text)
            merged.append(rule)
        for index, plan in enumerate(plans):
            if plan == primary_plan:
                continue
            rules, diagnostics = self.retriever.retrieve(
                task,
                plan,
                focus_predicates=focus_predicates,
                path_queries=executable_path_queries,
                planned_queries_only=self.indexed_plan_only,
            )
            rules = apply_agent_action_to_candidates(task, rules, action)
            rules, rules_truncated = cap_candidate_rules(rules, self.candidate_evaluation_cap)
            for rule in rules:
                text = rule_to_text(rule)
                if text in seen_rules:
                    continue
                seen_rules.add(text)
                merged.append(rule)
                if len(merged) >= self.candidate_evaluation_cap:
                    break
            portfolio_diagnostics.append(
                {
                    "index": index,
                    "plan": asdict(plan),
                    "candidate_count": len(rules),
                    "candidate_rules_truncated": rules_truncated,
                    "diagnostics": diagnostics,
                }
            )
            if len(merged) >= self.candidate_evaluation_cap:
                break
        if self.deterministic_fallback and len(merged) < self.candidate_evaluation_cap:
            fallback_plan = deterministic_fallback_plan(self.base_plan, task)
            fallback_rules, fallback_diag = self.retriever.retrieve(task, fallback_plan, focus_predicates=())
            fallback_rules = apply_agent_action_to_candidates(task, fallback_rules, AgentAction())
            for rule in fallback_rules:
                text = rule_to_text(rule)
                if text in seen_rules:
                    continue
                seen_rules.add(text)
                merged.append(rule)
                if len(merged) >= self.candidate_evaluation_cap:
                    break
            portfolio_diagnostics.append(
                {
                    "index": "deterministic_fallback",
                    "plan": asdict(fallback_plan),
                    "candidate_count": len(fallback_rules),
                    "diagnostics": fallback_diag,
                }
            )
        return merged, {
            "portfolio_size": len(plans),
            "merged_candidate_count": len(merged),
            "primary_f1": primary_f1,
            "focus_predicates": list(focus_predicates),
            "requested_focus_predicates": list(requested_focus_predicates),
            "focus_disabled_for_scale": focus_disabled_for_scale,
            "portfolio_triggered": True,
            "portfolio_skipped_for_indexed_plan": False,
            "use_indexed_path_execution": self.use_indexed_path_execution,
            "rejected_path_queries": list(action.rejected_path_queries),
            "candidate_evaluation_cap": self.candidate_evaluation_cap,
            "candidate_rules_truncated": len(merged) >= self.candidate_evaluation_cap,
            "portfolio": portfolio_diagnostics,
        }


def planner_prompt(
    task: TaskData,
    objective: str,
    feedback: Optional[str],
    iteration: int,
    base_plan: RetrievalPlan,
    suggested_plan: RetrievalPlan,
    compact_actions: bool = False,
    schema_profile_mode: str = "assisted",
    max_path_queries: int = 5,
    include_witness_evidence: bool = True,
) -> str:
    """Prompt the model to act as a retrieval planner, not a rule writer."""

    predicate_lines = [
        f"{spec.name}/{spec.arity}:{'/'.join(spec.types)}"
        for spec in task.predicates.values()
    ]
    schema = schema_profile(task, mode=schema_profile_mode)
    evidence = (
        compact_task_evidence(task)
        if schema_profile_mode == "raw" and include_witness_evidence
        else {}
    )
    payload = {
        "role": "planner",
        "iteration": iteration,
        "objective": objective[:180],
        "target": {"name": task.target.name, "arity": task.target.arity, "types": list(task.target.types)},
        "task_limits": {"max_vars": task.max_vars, "max_body": task.max_body, "max_clauses": task.max_clauses},
        "feedback": feedback or "none",
        "available_predicates": predicate_lines,
        "schema_profile": schema,
        "task_evidence": evidence,
        "suggested_plan": asdict(suggested_plan),
        "enums": {
            "constraint_mode": ["direct", "attribute", "both", "none"],
            "traversal_strategy": ["bfs", "dfs"],
            "strategy_presets": ["wide_bfs", "constraint_heavy", "dfs_probe", "support2_attr", "extended_body"],
        },
        "output_keys": [
            "max_depth",
            "max_positive_examples",
            "max_paths_per_example",
            "max_edges_per_node",
            "max_candidates",
            "enable_pair_constraints",
            "traversal_strategy",
            "constraint_mode",
            "min_path_support",
            "candidate_body_extra",
            "seed_max_depth",
            "seed_max_paths_per_example",
            "expand_seed_on_miss",
            "path_queries",
            "required_predicates",
            "avoid_predicates",
            "preferred_constraint_predicates",
            "strategy_presets",
        ],
    }
    if compact_actions:
        portfolio_instruction = (
            "Emit exactly one path_query when task_evidence.candidate_path_signatures is non-empty. "
            if max_path_queries <= 1
            else f"Emit 3 to {max_path_queries} structurally diverse path_queries when task_evidence.candidate_path_signatures is non-empty. "
        )
        repair_hints = repair_hints_from_evidence(evidence, {}, list(task.target.types))
        payload = {
            "target_types": list(task.target.types),
            "feedback": feedback or "none",
            "available_predicates": predicate_lines,
            "schema": {
                "semantic_intent": schema.get("semantic_intent"),
                "expected_present": schema.get("expected_present", []),
                "fallback_candidates": schema.get("fallback_candidates", []),
                "expressibility": schema.get("expressibility"),
                "path_hints": schema.get("path_hints", []),
            },
            "task_evidence": evidence,
            "repair_hints": repair_hints if feedback else {},
            "neutral_format_examples": STRICT_NEUTRAL_FORMAT_EXAMPLES if schema_profile_mode == "raw" else (),
            "output_keys": [
                "path_queries",
                "required_predicates",
                "avoid_predicates",
                "preferred_constraint_predicates",
                "strategy_presets",
            ],
        }
        return (
            "/no_think\n"
            "Return one compact JSON action object. Use exactly output_keys; unused keys must be empty arrays. "
            "Do not repeat input keys such as target_types, feedback, available_predicates, schema, or task_evidence. "
            "neutral_format_examples teach JSON syntax only: their relation_1/relation_2 placeholders are not available predicates and must never be copied. "
            + portfolio_instruction
            +
            "Emit 0 path_queries only when direct_pair_constraint_support "
            "shows that constraints alone separate positives from negatives. Each non-empty path query must be a short array of "
            '["availableName","fwd|rev"] steps from target A to target B. '
            "For predicate p(T0,T1), fwd means T0 -> T1 and rev means T1 -> T0. "
            "Every query must start at target_types[0], type-check between adjacent steps, "
            "and end at target_types[1]. When task_evidence contains witness signatures, "
            "prefer candidate_path_signatures with high support_margin, high positive_support, and low negative_support. "
            "Cover distinct candidate_path_signatures when available instead of repeating one predicate family. "
            "Do not output near-duplicate prefixes; each path_query should differ by at least one structural predicate or direction. "
            "If feedback is not none, do not repeat a weak failing path when repair_hints.preferred_replacement_signatures is available. "
            "Never output a prefix of a witness signature: copy the complete steps needed to end at target_types[1]. "
            "Repeat a relation only when observed evidence or schema.path_hints supports a chain. "
            "Use schema.expected_present predicates when available. No rules, rationale, or retrieval parameters.\n"
            + json.dumps(payload, ensure_ascii=False)
        )
    return (
        "/no_think\n"
        "Return one compact JSON object. Use only output_keys. "
        "Predicate arrays must use names from available_predicates. Do not output rules. "
        "path_queries must be an array of ordered path programs; each program is an array of "
        '{"predicate":"availableName","direction":"fwd|rev"} steps from target A to target B. '
        "Direction contract: for predicate p(T0,T1), fwd means T0 -> T1 and rev means T1 -> T0. "
        "Every path query must start at target.types[0], type-check between adjacent steps, and end at target.types[1]. "
        "If schema_profile.expected_present is non-empty, actively use those predicates. "
        "If expected predicates are missing, use schema_profile.fallback_candidates or explain graph_not_expressive in rationale.\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def typed_path_student_prompt(
    task: TaskData,
    verifier_feedback: Optional[str] = None,
    best_rule_text: Optional[str] = None,
    best_metrics: Optional[Dict[str, object]] = None,
    diagnostics: Optional[Dict[str, object]] = None,
    max_examples: int = 6,
    facts_per_predicate: int = 4,
    max_predicates: int = 32,
) -> str:
    """Render the inference contract used by the leakage-audited 4B student."""

    predicate_specs = student_schema_predicates(task, max_predicates=max_predicates)
    facts_by_predicate: Dict[str, List[str]] = {spec.name: [] for spec in predicate_specs}
    for fact in task.facts:
        bucket = facts_by_predicate.get(fact.predicate)
        if bucket is not None and len(bucket) < facts_per_predicate:
            bucket.append(f"{fact.predicate}({','.join(fact.args)}).")
    evidence = [
        fact_text
        for spec in predicate_specs
        for fact_text in facts_by_predicate.get(spec.name, [])
    ]
    positive_examples = [
        list(example.literal.args)
        for example in task.examples
        if example.positive
    ][:max_examples]
    negative_examples = [
        list(example.literal.args)
        for example in task.examples
        if not example.positive
    ][: max_examples * 2]
    payload = {
        "task": "repair_typed_paths" if verifier_feedback else "generate_typed_paths",
        "target_predicate": task.target.signature,
        "target_types": list(task.target.types),
        "available_predicates": [spec.signature for spec in predicate_specs],
        "type_constraints": {spec.name: list(spec.types) for spec in predicate_specs if spec.types},
        "evidence_first_local_bk": evidence,
        "positive_examples": positive_examples,
        "negative_examples": negative_examples,
        "output_schema": {
            "typed_paths": [
                [
                    {
                        "from": "<current_type>",
                        "predicate": "<available_binary_predicate>",
                        "direction": "fwd|rev",
                        "to": "<next_type>",
                    },
                ],
            ],
            "required_predicates": ["<available_binary_predicate>"],
            "preferred_constraint_predicates": [],
            "confidence": 0.0,
            "rationale": "short reason",
        },
    }
    if verifier_feedback:
        diagnostics = diagnostics or {}
        payload["repair_context"] = {
            "verifier_feedback": verifier_feedback,
            "best_rule": best_rule_text,
            "best_metrics": best_metrics,
            "failure_profile": diagnostics.get("failure_profile", {}),
            "previous_rejected_path_queries": diagnostics.get("rejected_path_queries", []),
            "instruction": "Propose a different shortest type-safe path when the previous plan failed symbolic verification.",
        }
    return (
        "/no_think\nReturn only valid JSON. "
        "Replace angle-bracket placeholders with actual values from the provided schema. "
        "Emit one to three typed_paths. Choose short semantically relevant type-safe paths from target A to target B. "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def student_schema_predicates(task: TaskData, max_predicates: int = 32) -> List[object]:
    """Keep stable architecture relations visible in noisy project-specific schemas."""

    specs_by_name = {spec.name: spec for spec in task.predicates.values()}
    output = [
        specs_by_name[name]
        for name in STUDENT_ARCHITECTURE_PREDICATE_PRIORITY
        if name in specs_by_name
    ]
    for spec in sorted(task.predicates.values(), key=lambda item: item.signature):
        if spec not in output:
            output.append(spec)
        if len(output) >= max_predicates:
            break
    return output[:max_predicates]


def refiner_prompt(
    task: TaskData,
    verifier_feedback: str,
    best_rule_text: Optional[str],
    best_metrics: Optional[Dict[str, object]],
    diagnostics: Dict[str, object],
    compact_actions: bool = False,
    schema_profile_mode: str = "assisted",
    max_path_queries: int = 5,
    include_witness_evidence: bool = True,
) -> str:
    """Prompt the model to choose the next repair direction."""

    graph_summary = compact_graph_diagnostics(diagnostics)
    schema = schema_profile(task, mode=schema_profile_mode)
    evidence = (
        compact_task_evidence(task)
        if schema_profile_mode == "raw" and include_witness_evidence
        else {}
    )
    payload = {
        "role": "refiner",
        "target": {"name": task.target.name, "arity": task.target.arity, "types": list(task.target.types)},
        "verifier_feedback": verifier_feedback,
        "best_rule": best_rule_text,
        "best_metrics": best_metrics,
        "schema_profile": schema,
        "task_evidence": evidence,
        "failure_profile": diagnostics.get("failure_profile", {}),
        "graph_summary": graph_summary,
        "valid_feedback_labels": ["too_general", "too_specific", "mixed_errors", "no_candidate", "consistent"],
        "valid_predicate_names": [spec.name for spec in task.predicates.values()],
        "allowed_strategy_presets": ["wide_bfs", "constraint_heavy", "dfs_probe", "support2_attr", "extended_body"],
        "output_keys": [
            "feedback_label",
            "required_predicates",
            "avoid_predicates",
            "preferred_constraint_predicates",
            "strategy_presets",
            "path_queries",
        ],
        "few_shot_repairs": [
            {
                "condition": "fp>0 and fn==0",
                "repair": {
                    "feedback_label": "too_general",
                    "preferred_constraint_predicates": ["sameMethodName", "sameMethodArity", "methodName", "methodArity"],
                    "strategy_presets": ["constraint_heavy", "support2_attr"],
                },
            },
            {
                "condition": "fn>0 and expected_present contains structural predicates not in best_rule",
                "repair": {
                    "feedback_label": "too_specific",
                    "required_predicates": ["callsMethod", "inheritsClass", "extendsClass"],
                    "strategy_presets": ["wide_bfs", "extended_body"],
                },
            },
            {
                "condition": "expected_missing contains core structural relation",
                "repair": {
                    "feedback_label": "mixed_errors",
                    "required_predicates": ["containsClass", "importsClass", "containsMethod"],
                    "strategy_presets": ["wide_bfs", "constraint_heavy"],
                },
            },
        ],
    }
    if compact_actions:
        failure = diagnostics.get("failure_profile", {})
        portfolio_instruction = (
            "Emit exactly one revised path_query when repair evidence supports any executable path. "
            if max_path_queries <= 1
            else f"Emit up to {max_path_queries} short path_queries as arrays of [predicate,direction] steps. "
        )
        repair_hints = repair_hints_from_evidence(evidence, failure, list(task.target.types))
        payload = {
            "target_types": list(task.target.types),
            "feedback": verifier_feedback,
            "schema": {
                "expected_present": schema.get("expected_present", []),
                "fallback_candidates": schema.get("fallback_candidates", []),
                "expressibility": schema.get("expressibility"),
                "path_hints": schema.get("path_hints", []),
            },
            "failure": {
                "error_type": failure.get("error_type"),
                "body_predicates": failure.get("body_predicates", []),
                "missing_expected_in_best_rule": failure.get("missing_expected_in_best_rule", []),
                "recommended_repair": failure.get("recommended_repair"),
                "best_metrics": best_metrics,
                "accepted_but_conservative": diagnostics.get("accepted_but_conservative", False),
            },
            "acceptance_threshold": 0.8,
            "top_candidate_summaries": diagnostics.get("top_candidate_summaries", []),
            "available_predicates": [
                f"{spec.name}/{spec.arity}:{'/'.join(spec.types)}"
                for spec in task.predicates.values()
            ],
            "task_evidence": evidence,
            "repair_hints": repair_hints,
            "neutral_format_examples": STRICT_NEUTRAL_FORMAT_EXAMPLES if schema_profile_mode == "raw" else (),
            "output_keys": [
                "feedback_label",
                "path_queries",
                "required_predicates",
                "avoid_predicates",
                "preferred_constraint_predicates",
                "strategy_presets",
            ],
        }
        return (
            "/no_think\n"
            "Return one compact JSON repair action. Use exactly output_keys; unused keys must be empty arrays. "
            "Do not repeat input keys such as target_types, schema, failure, available_predicates, or task_evidence. "
            "neutral_format_examples teach JSON syntax only: never copy their relation_1/relation_2 placeholders. "
            + portfolio_instruction
            +
            "Every query must type-check from target A to target B. Use positive versus negative witness support "
            "to replace weak structural paths. If repair_hints.preferred_replacement_signatures is non-empty, "
            "emit at least one complete signature from that list exactly as a path_query. "
            "Use top_candidate_summaries for threshold-aware repair: if a candidate has precision near 1.0 "
            "but recall below threshold, preserve its high-precision constraints and broaden or replace only the narrow structural edge. "
            "If a candidate has high recall but low precision, add attribute constraints rather than replacing the whole path. "
            "If the best F1 is just below acceptance_threshold, make a minimal edit to the closest candidate instead of starting over. "
            "Do not output only a prefix of a witness signature; the final step must end at target_types[1]. "
            "If failure.body_predicates caused mixed_errors, do not put only those predicates in required_predicates. "
            "No rules, rationale, or retrieval parameters.\n"
            + json.dumps(payload, ensure_ascii=False)
        )
    return (
        "/no_think\n"
        "Return one compact JSON object. Use only output_keys. "
        "Do not output rules. Do not claim consistent unless fp=0 and fn=0. "
        "When changing structural retrieval, emit ordered path_queries using predicate and fwd|rev direction steps. "
        "For predicate p(T0,T1), fwd means T0 -> T1 and rev means T1 -> T0; type-check the full chain from target A to B. "
        "Use failure_profile to decide whether the next action should add constraints, broaden paths, or report schema limits.\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def coerce_retrieval_plan(data: Dict[str, object], fallback: RetrievalPlan, task: TaskData) -> RetrievalPlan:
    """Validate and clamp an LLM-proposed retrieval plan."""

    max_body = max(1, task.max_body)
    max_depth = clamp_int(data.get("max_depth"), 1, max_body, fallback.max_depth)
    max_positive_examples = clamp_int(data.get("max_positive_examples"), 1, 50, fallback.max_positive_examples)
    max_paths_per_example = clamp_int(data.get("max_paths_per_example"), 1, 200, fallback.max_paths_per_example)
    max_edges_per_node = clamp_int(data.get("max_edges_per_node"), 1, 1000, fallback.max_edges_per_node)
    max_candidates = clamp_int(data.get("max_candidates"), 1, 200, fallback.max_candidates)
    min_path_support = clamp_int(data.get("min_path_support"), 1, 20, fallback.min_path_support)
    candidate_body_extra = clamp_int(data.get("candidate_body_extra"), 0, 3, fallback.candidate_body_extra)
    seed_max_depth = clamp_int(data.get("seed_max_depth"), 0, max_body, fallback.seed_max_depth)
    seed_max_paths_per_example = clamp_int(
        data.get("seed_max_paths_per_example"),
        1,
        100,
        fallback.seed_max_paths_per_example,
    )
    expand_seed_on_miss = bool_value(data.get("expand_seed_on_miss"), fallback.expand_seed_on_miss)
    traversal_strategy = enum_value(data.get("traversal_strategy"), {"bfs", "dfs"}, fallback.traversal_strategy)
    constraint_mode = enum_value(data.get("constraint_mode"), {"direct", "attribute", "both", "none"}, fallback.constraint_mode)
    enable_pair_constraints = bool_value(data.get("enable_pair_constraints"), fallback.enable_pair_constraints)
    if constraint_mode == "none":
        enable_pair_constraints = False
    return RetrievalPlan(
        max_depth=max_depth,
        max_positive_examples=max_positive_examples,
        max_paths_per_example=max_paths_per_example,
        max_edges_per_node=max_edges_per_node,
        max_candidates=max_candidates,
        enable_pair_constraints=enable_pair_constraints,
        traversal_strategy=traversal_strategy,
        constraint_mode=constraint_mode,
        min_path_support=min_path_support,
        candidate_body_extra=candidate_body_extra,
        seed_max_depth=seed_max_depth,
        seed_max_paths_per_example=seed_max_paths_per_example,
        expand_seed_on_miss=expand_seed_on_miss,
    )


def compact_graph_diagnostics(diagnostics: Dict[str, object]) -> Dict[str, object]:
    """Keep Refiner prompts short and schema-relevant."""

    if "portfolio" not in diagnostics:
        return {
            "candidate_count": diagnostics.get("candidate_count"),
            "retrieved_path_count": diagnostics.get("retrieved_path_count"),
            "common_pair_constraints": diagnostics.get("common_pair_constraints", []),
            "common_attribute_constraints": diagnostics.get("common_attribute_constraints", []),
            "focus_predicates": diagnostics.get("focus_predicates", []),
            "seed_max_depth": diagnostics.get("seed_max_depth"),
            "seed_expanded_examples": diagnostics.get("seed_expanded_examples"),
            "planned_query_path_count": diagnostics.get("planned_query_path_count"),
            "planned_queries_only": diagnostics.get("planned_queries_only"),
            "rejected_path_queries": diagnostics.get("rejected_path_queries", []),
            "timings_seconds": diagnostics.get("timings_seconds", {}),
            "schema_profile": diagnostics.get("schema_profile", {}),
            "failure_profile": diagnostics.get("failure_profile", {}),
            "top_candidate_summaries": diagnostics.get("top_candidate_summaries", []),
        }

    summary = {
        "portfolio_size": diagnostics.get("portfolio_size"),
        "merged_candidate_count": diagnostics.get("merged_candidate_count"),
        "primary_f1": diagnostics.get("primary_f1"),
        "focus_predicates": diagnostics.get("focus_predicates", []),
        "rejected_path_queries": diagnostics.get("rejected_path_queries", []),
        "schema_profile": diagnostics.get("schema_profile", {}),
        "failure_profile": diagnostics.get("failure_profile", {}),
        "top_candidate_summaries": diagnostics.get("top_candidate_summaries", []),
        "plans": [],
    }
    for item in list(diagnostics.get("portfolio") or [])[:5]:
        diag = item.get("diagnostics") or {}
        summary["plans"].append(
            {
                "index": item.get("index"),
                "candidate_count": item.get("candidate_count"),
                "retrieved_path_count": diag.get("retrieved_path_count"),
                "common_pair_constraints": diag.get("common_pair_constraints", []),
                "common_attribute_constraints": diag.get("common_attribute_constraints", []),
                "focus_predicates": diag.get("focus_predicates", []),
                "seed_max_depth": diag.get("seed_max_depth"),
                "seed_expanded_examples": diag.get("seed_expanded_examples"),
                "planned_query_path_count": diag.get("planned_query_path_count"),
                "planned_queries_only": diag.get("planned_queries_only"),
                "timings_seconds": diag.get("timings_seconds", {}),
            }
        )
    return summary


def candidate_feedback_summary(task: TaskData, candidates: Sequence[Rule], limit: int = 5) -> List[Dict[str, object]]:
    """Summarize the verifier's top candidates for threshold-aware repair."""

    if not candidates:
        return []
    facts_list = list(task.facts)
    examples_list = list(task.examples)
    fact_index = build_fact_index(facts_list)
    scored: List[Tuple[float, float, float, int, Dict[str, object]]] = []
    for index, rule in enumerate(dedupe_rules(candidates)):
        metrics = evaluate_rule(rule, facts_list, examples_list, fact_index=fact_index)
        body = sorted(body_predicates(rule))
        scored.append(
            (
                metrics.f1,
                metrics.accuracy,
                metrics.precision,
                -len(rule.body),
                {
                    "index": index,
                    "rule": rule_to_text(rule),
                    "body_predicates": body,
                    "metrics": asdict(metrics),
                    "feedback": metrics.feedback_label,
                },
            )
        )
    scored.sort(key=lambda item: item[:4], reverse=True)
    return [item[-1] for item in scored[:limit]]


def compact_task_evidence(
    task: TaskData,
    max_positive_examples: int = 6,
    max_negative_examples: int = 4,
    max_paths_per_example: int = 24,
    max_depth: int = 3,
    max_edges_per_node: int = 80,
    max_signatures: int = 16,
) -> Dict[str, object]:
    """Render bounded graph witness sketches without target-specific templates.

    The observer never compiles or scores candidate rules.  It only exposes a
    compact sample of short paths that actually connect labeled endpoint pairs,
    leaving the Planner responsible for selecting an executable path program.
    """

    positives = [
        example.literal.args
        for example in task.examples
        if example.positive
    ][:max_positive_examples]
    negatives = [
        example.literal.args
        for example in task.examples
        if not example.positive
    ][:max_negative_examples]

    binary_specs = {
        spec.name: spec
        for spec in task.predicates.values()
        if spec.arity == 2
        and len(spec.types) == 2
        and not is_attribute_or_pair_constraint(spec.name)
    }
    adjacency = build_fact_graph(task.facts, binary_specs)

    def collect(examples: Sequence[Tuple[str, ...]]) -> List[Dict[str, object]]:
        signature_support: Dict[PathSignature, set[int]] = defaultdict(set)
        signature_count: Counter[PathSignature] = Counter()
        for ex_index, args in enumerate(examples):
            if len(args) != 2:
                continue
            for path in retrieve_paths(
                adjacency=adjacency,
                start=args[0],
                end=args[1],
                max_depth=max_depth,
                max_paths=max_paths_per_example,
                max_edges_per_node=max_edges_per_node,
            ):
                signature = tuple((edge.predicate, edge.reversed_edge) for edge in path)
                if not signature:
                    continue
                signature_support[signature].add(ex_index)
                signature_count[signature] += 1
        ranked = sorted(
            signature_count,
            key=lambda signature: (
                -len(signature_support[signature]),
                len(signature),
                -signature_count[signature],
                signature,
            ),
        )
        return [
            {
                "steps": [
                    [predicate, "rev" if reversed_edge else "fwd"]
                    for predicate, reversed_edge in signature
                ],
                "example_support": len(signature_support[signature]),
                "path_count": signature_count[signature],
            }
            for signature in ranked[:max_signatures]
        ]

    positive_signatures = collect(positives)
    negative_signatures = collect(negatives)
    negative_support_by_steps = {
        signature_steps_key(item.get("steps")): int(item.get("example_support", 0) or 0)
        for item in negative_signatures
    }
    candidate_path_signatures = []
    for item in positive_signatures:
        steps = signature_steps_key(item.get("steps"))
        if not steps:
            continue
        pos_support = int(item.get("example_support", 0) or 0)
        neg_support = negative_support_by_steps.get(steps, 0)
        candidate_path_signatures.append(
            {
                "steps": item.get("steps"),
                "positive_support": pos_support,
                "negative_support": neg_support,
                "support_margin": pos_support - neg_support,
                "path_length": len(steps),
            }
        )
    candidate_path_signatures.sort(
        key=lambda item: (
            -int(item["support_margin"]),
            int(item["negative_support"]),
            -int(item["positive_support"]),
            int(item["path_length"]),
            json.dumps(item["steps"], sort_keys=True),
        )
    )
    candidate_path_signatures = diversify_signature_items(candidate_path_signatures, max_signatures)

    direct_pair_facts = {
        (fact.predicate, fact.args[0], fact.args[1])
        for fact in task.facts
        if fact.arity == 2
    }

    def collect_direct_pair_support(examples: Sequence[Tuple[str, ...]]) -> List[Dict[str, object]]:
        support: Dict[str, set[int]] = defaultdict(set)
        for ex_index, args in enumerate(examples):
            if len(args) != 2:
                continue
            left, right = args
            for spec in task.predicates.values():
                if spec.arity != 2:
                    continue
                if (spec.name, left, right) in direct_pair_facts:
                    support[spec.name].add(ex_index)
        return [
            {"predicate": name, "example_support": len(indices)}
            for name, indices in sorted(
                support.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )
        ][:max_signatures]

    return {
        "observer_kind": "label_conditioned_witness_sketch",
        "target_name_conditioned": False,
        "compiles_or_scores_rules": False,
        "deterministic_order": True,
        "positive_example_count": len(positives),
        "negative_example_count": len(negatives),
        "positive_witness_signatures": positive_signatures,
        "negative_witness_signatures": negative_signatures,
        "candidate_path_signatures": candidate_path_signatures[:max_signatures],
        "positive_direct_pair_constraint_support": collect_direct_pair_support(positives),
        "negative_direct_pair_constraint_support": collect_direct_pair_support(negatives),
        "observer_limits": {
            "max_depth": max_depth,
            "max_paths_per_example": max_paths_per_example,
            "max_edges_per_node": max_edges_per_node,
            "excluded_constraint_predicates": sorted(
                spec.name
                for spec in task.predicates.values()
                if spec.arity == 2
                and len(spec.types) == 2
                and is_attribute_or_pair_constraint(spec.name)
            ),
        },
    }


def deterministic_witness_action(task: TaskData, max_path_queries: int = 1) -> AgentAction:
    """Select ranked train-example witness signatures without an LLM.

    This is an ablation comparator, not a fallback. It uses the same bounded
    witness observer as PARA and leaves compilation and acceptance to the same
    symbolic pipeline.
    """

    evidence = compact_task_evidence(task)
    path_queries: List[PathSignature] = []
    for item in evidence.get("candidate_path_signatures", []):
        if not isinstance(item, dict):
            continue
        steps = item.get("steps")
        if not isinstance(steps, list):
            continue
        query: List[Tuple[str, bool]] = []
        valid = True
        for step in steps:
            if (
                not isinstance(step, list)
                or len(step) != 2
                or not isinstance(step[0], str)
                or str(step[1]).lower() not in {"fwd", "rev"}
            ):
                valid = False
                break
            query.append((step[0], str(step[1]).lower() == "rev"))
        signature = tuple(query)
        if valid and signature and signature not in path_queries:
            path_queries.append(signature)
        if len(path_queries) >= max(1, max_path_queries):
            break
    return AgentAction(path_queries=tuple(path_queries))


def diversify_signature_items(items: Sequence[Dict[str, object]], limit: int) -> List[Dict[str, object]]:
    """Interleave structurally different witness signatures without target templates."""

    selected: List[Dict[str, object]] = []
    seen_predicate_sets: set[Tuple[str, ...]] = set()
    for item in items:
        steps = signature_steps_key(item.get("steps"))
        predicates = tuple(sorted({predicate for predicate, _direction in steps}))
        if not predicates or predicates in seen_predicate_sets:
            continue
        selected.append(dict(item))
        seen_predicate_sets.add(predicates)
        if len(selected) >= limit:
            return selected
    for item in items:
        steps = signature_steps_key(item.get("steps"))
        if not steps:
            continue
        if any(signature_steps_key(existing.get("steps")) == steps for existing in selected):
            continue
        selected.append(dict(item))
        if len(selected) >= limit:
            break
    return selected


def repair_hints_from_evidence(
    evidence: Dict[str, object],
    failure: Dict[str, object],
    target_types: Sequence[str],
) -> Dict[str, object]:
    """Derive prompt-only repair hints from witness support.

    The helper does not compile, score, or execute candidate rules. It only
    summarizes the existing witness sketch so the LLM can pick complete
    type-closed path programs instead of repeating a weak body predicate or
    outputting a prefix of a valid witness.
    """

    positive = list(evidence.get("candidate_path_signatures") or evidence.get("positive_witness_signatures") or [])
    negative = list(evidence.get("negative_witness_signatures") or [])
    body_predicates = set(str(name) for name in (failure.get("body_predicates") or []))
    negative_support: Dict[Tuple[Tuple[str, str], ...], int] = {}
    for item in negative:
        steps = signature_steps_key(item.get("steps"))
        if steps:
            negative_support[steps] = int(item.get("example_support", 0) or 0)

    ranked = []
    for item in positive:
        steps = signature_steps_key(item.get("steps"))
        if not steps:
            continue
        predicates = {predicate for predicate, _direction in steps}
        pos = int(item.get("positive_support", item.get("example_support", 0)) or 0)
        neg = int(item.get("negative_support", negative_support.get(steps, 0)) or 0)
        body_only = bool(predicates) and predicates.issubset(body_predicates)
        ranked.append(
            {
                "steps": [[predicate, direction] for predicate, direction in steps],
                "positive_support": pos,
                "negative_support": neg,
                "uses_only_failed_body_predicates": body_only,
            }
        )
    ranked.sort(
        key=lambda item: (
            bool(item["uses_only_failed_body_predicates"]),
            int(item["negative_support"]),
            -int(item["positive_support"]),
            len(item["steps"]),
            json.dumps(item["steps"], sort_keys=True),
        )
    )
    avoid_required = sorted(body_predicates) if failure.get("error_type") in {"mixed_errors", "too_general"} else []
    return {
        "target_type_chain": {
            "start": target_types[0] if target_types else None,
            "end": target_types[1] if len(target_types) > 1 else None,
            "instruction": "each path_query must end at this type; do not emit only a prefix",
        },
        "avoid_required_predicates_when_repairing": avoid_required,
        "preferred_replacement_signatures": ranked[:8],
    }


def signature_steps_key(value: object) -> Tuple[Tuple[str, str], ...]:
    if not isinstance(value, list):
        return ()
    output: List[Tuple[str, str]] = []
    for raw_step in value:
        if not isinstance(raw_step, list) or len(raw_step) != 2:
            return ()
        predicate = str(raw_step[0])
        direction = str(raw_step[1])
        if direction not in {"fwd", "rev"}:
            return ()
        output.append((predicate, direction))
    return tuple(output)


def schema_profile(task: TaskData, mode: str = "assisted") -> Dict[str, object]:
    """Describe whether the current task schema can express the target intent."""

    available = {spec.name for spec in task.predicates.values()}
    if mode == "raw":
        return {
            "profile_mode": "raw",
            "available_count": len(available),
            "expected_present": [],
            "expected_missing": [],
            "fallback_candidates": [],
            "required_group_status": [],
            "expressibility": "unknown",
            "notes": [],
            "path_hints": [],
        }
    if mode != "assisted":
        raise ValueError(f"unsupported schema profile mode: {mode}")
    target = task.target.name.lower()
    expected: List[str] = []
    fallback: List[str] = []
    required_groups: List[Dict[str, object]] = []
    expressibility = "unknown"
    notes: List[str] = []
    path_hints: List[List[List[str]]] = []

    intent = semantic_intent(task)
    if intent in {"method_call", "method_call_chain"}:
        expected = ["callsMethod"]
        fallback = ["sameMethodName", "sameMethodArity"]
        required_groups = [
            {"role": "method_call_relation", "any_of": ["callsMethod"]},
        ]
        expressibility = "direct" if "callsMethod" in available else "not_expressive"
        if intent == "method_call_chain":
            path_hints = [[["callsMethod", "fwd"], ["callsMethod", "fwd"]]]
            notes.append("The target denotes a two-hop method call chain; repeat callsMethod exactly twice.")
    elif intent == "class_call":
        expected = ["callsMethod", "containsMethod"]
        fallback = ["importsClass", "containsClass"]
        required_groups = [
            {"role": "method_call_path", "any_of": ["callsMethod"]},
            {"role": "method_to_class_bridge", "any_of": ["containsMethod"]},
        ]
        expressibility = "direct" if {"callsMethod", "containsMethod"}.issubset(available) else "fallback_only"
        if expressibility == "fallback_only":
            notes.append("Direct method-call evidence is absent; import/class containment is the fallback signal.")
    elif intent == "package_call":
        expected = ["containsClass", "containsMethod", "callsMethod"]
        fallback = ["importsClass", "containsPackage"]
        required_groups = [
            {"role": "package_to_class_bridge", "any_of": ["containsClass"]},
            {"role": "class_to_method_bridge", "any_of": ["containsMethod"]},
            {"role": "method_call_path", "any_of": ["callsMethod"]},
        ]
        expressibility = "direct" if set(expected).issubset(available) else "fallback_only"
    elif intent in {"class_import", "package_import"}:
        expected = ["importsClass"]
        fallback = ["containsClass", "containsPackage"]
        if intent == "package_import":
            expected = ["containsClass", "importsClass"]
        required_groups = [
            {"role": "import_relation", "any_of": ["importsClass"]},
        ]
        expressibility = "direct" if set(expected).issubset(available) else "fallback_only"
    elif intent == "class_hierarchy":
        expected = [name for name in ("inheritsClass", "extendsClass", "implementsInterface") if name.lower().replace("class", "") in target]
        if not expected:
            expected = ["inheritsClass", "extendsClass", "implementsInterface"]
        fallback = ["importsClass"]
        required_groups = [
            {"role": "class_hierarchy_relation", "any_of": ["inheritsClass", "extendsClass", "implementsInterface"]},
        ]
        expressibility = "direct" if set(expected).intersection(available) else "not_expressive"
    elif intent == "method_signature":
        expected = ["methodName", "methodArity"]
        fallback = ["sameMethodName", "sameMethodArity"]
        required_groups = [
            {"role": "method_name_match", "any_of": ["methodName", "sameMethodName"]},
            {"role": "method_arity_match", "any_of": ["methodArity", "sameMethodArity"]},
        ]
        has_name = bool({"methodName", "sameMethodName"}.intersection(available))
        has_arity = bool({"methodArity", "sameMethodArity"}.intersection(available))
        expressibility = "direct" if has_name and has_arity else "not_expressive"
    elif intent == "override":
        expected = ["containsMethod", "inheritsClass", "extendsClass", "sameMethodName", "sameMethodArity", "methodName", "methodArity"]
        fallback = ["containsClass", "importsClass", "containsMethod", "methodName", "methodArity"]
        required_groups = [
            {"role": "inheritance_relation", "any_of": ["inheritsClass", "extendsClass"]},
            {"role": "method_membership", "any_of": ["containsMethod"]},
            {"role": "method_signature_match", "any_of": ["sameMethodName", "methodName"]},
        ]
        has_inheritance = bool({"inheritsClass", "extendsClass"}.intersection(available))
        has_signature = bool({"sameMethodName", "methodName"}.intersection(available))
        if has_inheritance and has_signature and "containsMethod" in available:
            expressibility = "direct"
        elif has_signature and "containsMethod" in available:
            expressibility = "schema_limited"
            notes.append("Inheritance relation is absent; override can only be approximated by containment/import paths.")
        else:
            expressibility = "not_expressive"
            notes.append("Core method signature or membership evidence is absent.")
    elif intent == "allowed_use":
        expected = ["containsClass", "importsClass"]
        fallback = ["containsPackage", "package", "class"]
        required_groups = [
            {"role": "class_membership", "any_of": ["containsClass"]},
            {"role": "use_relation", "any_of": ["importsClass"]},
        ]
        expressibility = "direct" if {"containsClass", "importsClass"}.issubset(available) else "fallback_only"

    expected_present = [name for name in expected if name in available]
    expected_missing = [name for name in expected if name not in available]
    fallback_candidates = [name for name in fallback if name in available and name not in expected_present]
    group_status = []
    for group in required_groups:
        options = list(group.get("any_of") or [])
        present = [name for name in options if name in available]
        group_status.append({"role": group.get("role"), "present": present, "missing": [name for name in options if name not in available]})

    return {
        "semantic_intent": intent,
        "available_count": len(available),
        "expected_present": expected_present,
        "expected_missing": expected_missing,
        "fallback_candidates": fallback_candidates,
        "required_group_status": group_status,
        "expressibility": expressibility,
        "notes": notes,
        "path_hints": path_hints,
    }


def semantic_intent(task: TaskData) -> str:
    """Classify target intent with both semantic words and endpoint types."""

    target = task.target.name.lower()
    endpoint_types = tuple(task.target.types)
    if "override" in target:
        return "override"
    if "allowed" in target or "use" in target:
        return "allowed_use"
    if "signature" in target or target.startswith("same") or "samemethod" in target:
        return "method_signature"
    if any(word in target for word in ("extend", "inherit", "implement")):
        return "class_hierarchy"
    if "import" in target:
        return "package_import" if endpoint_types == ("package", "package") else "class_import"
    if "call" in target:
        if endpoint_types == ("method", "method"):
            return "method_call_chain" if "chain" in target else "method_call"
        if endpoint_types == ("package", "package"):
            return "package_call"
        return "class_call"
    return "unknown"


def failure_profile(
    task: TaskData,
    best_rule: Optional[Rule],
    best_metrics: Optional[Dict[str, object]],
    candidates: Sequence[Rule],
    diagnostics: Dict[str, object],
    schema_profile_mode: str = "assisted",
) -> Dict[str, object]:
    """Build compact structured evidence for the Refiner."""

    metrics = best_metrics or {}
    if best_rule is None:
        return {
            "error_type": "no_candidate",
            "candidate_count": len(candidates),
            "body_predicates": [],
            "missing_expected_in_best_rule": schema_profile(task, mode=schema_profile_mode).get("expected_present", []),
            "recommended_repair": "broaden_search_or_use_schema_fallback",
        }

    facts_list = list(task.facts)
    fact_index = build_fact_index(facts_list)
    fp_samples: List[List[str]] = []
    fn_samples: List[List[str]] = []
    for example in task.examples:
        predicted = covers(best_rule, facts_list, example, fact_index=fact_index)
        sample = list(example.literal.args)
        if not example.positive and predicted and len(fp_samples) < 5:
            fp_samples.append(sample)
        elif example.positive and not predicted and len(fn_samples) < 5:
            fn_samples.append(sample)

    schema = schema_profile(task, mode=schema_profile_mode)
    body = sorted(body_predicates(best_rule))
    missing_expected = [name for name in schema.get("expected_present", []) if name not in body]
    repair = "none"
    fp = int(metrics.get("fp", 0) or 0)
    fn = int(metrics.get("fn", 0) or 0)
    if fp and not fn:
        repair = "add_constraints"
    elif fn and not fp:
        repair = "broaden_paths"
    elif fp and fn:
        repair = "change_structural_path_then_add_constraints"
    if schema.get("expressibility") in {"schema_limited", "not_expressive"}:
        repair = "schema_limited_use_fallback_or_report_limit"

    return {
        "error_type": metrics_to_error_type(metrics),
        "tp": metrics.get("tp", 0),
        "fp": fp,
        "tn": metrics.get("tn", 0),
        "fn": fn,
        "false_positive_samples": fp_samples,
        "false_negative_samples": fn_samples,
        "body_predicates": body,
        "missing_expected_in_best_rule": missing_expected,
        "candidate_count": len(candidates),
        "portfolio_triggered": diagnostics.get("portfolio_triggered"),
        "recommended_repair": repair,
    }


def metrics_to_error_type(metrics: Dict[str, object]) -> str:
    fp = int(metrics.get("fp", 0) or 0)
    fn = int(metrics.get("fn", 0) or 0)
    if fp > 0 and fn == 0:
        return "too_general"
    if fn > 0 and fp == 0:
        return "too_specific"
    if fp > 0 and fn > 0:
        return "mixed_errors"
    return "consistent"


def is_accepted_but_conservative(metrics: Optional[Dict[str, object]], acceptance_f1: float) -> bool:
    """Allow one more refinement for high-precision accepted rules with low recall."""

    if not metrics:
        return False
    f1 = float(metrics.get("f1", 0.0) or 0.0)
    precision = float(metrics.get("precision", 0.0) or 0.0)
    recall = float(metrics.get("recall", 0.0) or 0.0)
    fp = int(metrics.get("fp", 0) or 0)
    fn = int(metrics.get("fn", 0) or 0)
    return f1 >= acceptance_f1 and precision >= 0.999 and recall < 0.9 and fp == 0 and fn > 0


def coerce_agent_action(data: Dict[str, object], task: TaskData) -> AgentAction:
    """Validate model-proposed action lists against the task schema."""

    available = {spec.name for spec in task.predicates.values()}
    allowed_presets = {"wide_bfs", "constraint_heavy", "dfs_probe", "support2_attr", "extended_body"}
    return AgentAction(
        required_predicates=tuple(names_from_json(data.get("required_predicates"), available)),
        avoid_predicates=tuple(names_from_json(data.get("avoid_predicates"), available)),
        preferred_constraint_predicates=tuple(names_from_json(data.get("preferred_constraint_predicates"), available)),
        strategy_presets=tuple(names_from_json(data.get("strategy_presets"), allowed_presets)),
        path_queries=tuple(path_queries_from_json(data.get("path_queries"), task)),
        rejected_path_queries=tuple(rejected_path_queries_from_json(data.get("path_queries"), task)),
    )


def typed_path_action_from_json(data: Dict[str, object], task: TaskData) -> AgentAction:
    """Convert leakage-audited student typed paths into indexed query actions."""

    raw_paths: List[object] = []
    if isinstance(data.get("typed_paths"), list):
        typed_paths = data["typed_paths"]
        if typed_paths and all(isinstance(item, dict) for item in typed_paths):
            raw_paths.append(typed_paths)
        else:
            raw_paths.extend(item for item in typed_paths if isinstance(item, list))
    if isinstance(data.get("typed_path"), list):
        raw_paths.append(data["typed_path"])

    path_queries: List[PathSignature] = []
    rejected: List[str] = []
    for raw_path in raw_paths:
        if not isinstance(raw_path, list):
            continue
        compact_steps: List[List[str]] = []
        syntax_ok = True
        for raw_step in raw_path:
            if not isinstance(raw_step, dict):
                syntax_ok = False
                break
            predicate = str(raw_step.get("predicate") or "")
            direction = str(raw_step.get("direction") or "fwd").lower()
            compact_steps.append([predicate, direction])
        parsed = path_queries_from_json([compact_steps], task) if syntax_ok else []
        if parsed:
            for query in parsed:
                if query not in path_queries:
                    path_queries.append(query)
        else:
            rejected.append(json.dumps(raw_path, ensure_ascii=False, sort_keys=True))
    soft_action = coerce_agent_action(data, task)
    return AgentAction(
        required_predicates=soft_action.required_predicates,
        avoid_predicates=soft_action.avoid_predicates,
        preferred_constraint_predicates=soft_action.preferred_constraint_predicates,
        strategy_presets=soft_action.strategy_presets,
        path_queries=tuple(merge_path_queries((soft_action.path_queries, tuple(path_queries[:12])))),
        rejected_path_queries=tuple(merge_names((soft_action.rejected_path_queries, tuple(rejected[:12])))),
    )


def symbolic_action_for(task: TaskData, feedback: Optional[str]) -> AgentAction:
    """Small schema-grounded action prior used when the model is vague."""

    available = {spec.name for spec in task.predicates.values()}
    intent = semantic_intent(task)
    required: List[str] = []
    constraints: List[str] = []
    presets: List[str] = []

    if intent in {"method_call", "method_call_chain"}:
        required = [name for name in ("callsMethod",) if name in available]
    elif intent == "class_call":
        required = [name for name in ("callsMethod", "containsMethod") if name in available]
        if not required:
            required = [name for name in ("importsClass", "containsClass") if name in available]
        if feedback in {"too_general", "mixed_errors"}:
            constraints = [name for name in ("importsClass",) if name in available]
            presets = ["constraint_heavy"]
    elif intent == "package_call":
        required = [name for name in ("containsClass", "containsMethod", "callsMethod") if name in available]
    elif intent == "package_import":
        required = [name for name in ("containsClass", "importsClass") if name in available]
    elif intent == "class_import":
        required = [name for name in ("importsClass",) if name in available]
    elif intent == "class_hierarchy":
        required = [name for name in ("inheritsClass", "extendsClass", "implementsInterface") if name in available]
    elif intent == "method_signature":
        constraints = [name for name in ("methodName", "methodArity", "sameMethodName", "sameMethodArity") if name in available]
    elif intent == "override":
        required = [name for name in ("containsMethod", "inheritsClass", "extendsClass") if name in available]
        if not {"inheritsClass", "extendsClass"}.intersection(available):
            required = [name for name in ("containsMethod", "containsClass", "importsClass") if name in available]
        constraints = [name for name in ("methodName", "methodArity", "sameMethodName", "sameMethodArity") if name in available]
        presets = ["support2_attr", "constraint_heavy", "extended_body"]
    elif intent == "allowed_use":
        required = [name for name in ("containsClass", "importsClass") if name in available]

    if feedback in {"too_specific", "no_candidate"}:
        presets.append("wide_bfs")

    return AgentAction(
        required_predicates=tuple(required),
        preferred_constraint_predicates=tuple(constraints),
        strategy_presets=tuple(presets),
    )


def names_from_json(value: object, allowed: set[str]) -> List[str]:
    """Normalize a JSON array of names and keep only allowed entries."""

    if not isinstance(value, list):
        return []
    output: List[str] = []
    for item in value:
        name = str(item)
        if name in allowed and name not in output:
            output.append(name)
    return output


def path_queries_from_json(value: object, task: TaskData) -> List[PathSignature]:
    """Validate ordered relation programs proposed by Planner or Refiner."""

    if not isinstance(value, list):
        return []
    specs = {spec.name: spec for spec in task.predicates.values()}
    output: List[PathSignature] = []
    for raw_query in value:
        raw_steps = raw_query.get("steps") if isinstance(raw_query, dict) else raw_query
        if not isinstance(raw_steps, list):
            continue
        if (
            len(raw_steps) == 1
            and isinstance(raw_steps[0], list)
            and raw_steps[0]
            and all(isinstance(item, (dict, list)) for item in raw_steps[0])
        ):
            # Small distilled models occasionally wrap one complete program in
            # an extra list. Normalize the syntax, then keep the full type-chain
            # validation below as the semantic guard.
            raw_steps = raw_steps[0]
        steps = []
        for raw_step in raw_steps:
            if isinstance(raw_step, dict):
                predicate = str(raw_step.get("predicate") or "")
                direction = str(raw_step.get("direction") or "fwd").lower()
            elif isinstance(raw_step, list) and len(raw_step) == 2:
                predicate = str(raw_step[0])
                direction = str(raw_step[1]).lower()
            else:
                steps = []
                break
            if predicate not in specs or direction not in {"fwd", "rev"}:
                steps = []
                break
            steps.append((predicate, direction == "rev"))
        query = tuple(steps)
        if query and is_type_safe_path_query(query, task, specs) and query not in output:
            output.append(query)
    return output[:12]


def rejected_path_queries_from_json(value: object, task: TaskData) -> List[str]:
    """Keep invalid model programs in traces for failure attribution."""

    if not isinstance(value, list):
        return []
    output: List[str] = []
    for raw_query in value:
        if path_queries_from_json([raw_query], task):
            continue
        text = json.dumps(raw_query, ensure_ascii=False, sort_keys=True)
        if text not in output:
            output.append(text)
    return output[:12]


def is_type_safe_path_query(
    query: PathSignature,
    task: TaskData,
    specs: Dict[str, object],
) -> bool:
    """Require an ordered query to form a type chain from target A to B."""

    if len(task.target.types) != 2:
        return False
    current_type = task.target.types[0]
    for predicate, reversed_edge in query:
        spec = specs.get(predicate)
        if spec is None or spec.arity != 2 or len(spec.types) != 2:
            return False
        source_type, destination_type = spec.types
        if reversed_edge:
            source_type, destination_type = destination_type, source_type
        if source_type != current_type:
            return False
        current_type = destination_type
    return current_type == task.target.types[1]


def merge_actions(*actions: AgentAction) -> AgentAction:
    """Merge action patches while preserving first occurrence order."""

    return AgentAction(
        required_predicates=tuple(merge_names(action.required_predicates for action in actions)),
        avoid_predicates=tuple(merge_names(action.avoid_predicates for action in actions)),
        preferred_constraint_predicates=tuple(merge_names(action.preferred_constraint_predicates for action in actions)),
        strategy_presets=tuple(merge_names(action.strategy_presets for action in actions)),
        path_queries=tuple(merge_path_queries(action.path_queries for action in actions)),
        rejected_path_queries=tuple(merge_names(action.rejected_path_queries for action in actions)),
    )


def merge_names(groups: Sequence[Sequence[str]]) -> List[str]:
    output: List[str] = []
    for group in groups:
        for name in group:
            if name not in output:
                output.append(name)
    return output


def merge_path_queries(groups: Sequence[Sequence[PathSignature]]) -> List[PathSignature]:
    output: List[PathSignature] = []
    for group in groups:
        for query in group:
            if query not in output:
                output.append(query)
    return output


def apply_agent_action_to_candidates(task: TaskData, rules: Sequence[Rule], action: AgentAction) -> List[Rule]:
    """Apply action-driven augmentation without discarding fallback candidates."""

    expanded = list(rules)
    expanded.extend(build_constraint_augmented_rules(task, rules, action.preferred_constraint_predicates))
    expanded = dedupe_rules(expanded)

    # Required/avoid predicates are treated as soft action preferences.  The
    # symbolic verifier must still see the unfiltered fallback candidates,
    # otherwise an overconfident LLM action can erase a strong GraphRAG rule.
    preferred: List[Rule] = []
    if action.avoid_predicates:
        avoid = set(action.avoid_predicates)
        preferred.extend(rule for rule in expanded if not body_predicates(rule).intersection(avoid))

    active_required = [name for name in action.required_predicates if any(name == lit.predicate for rule in expanded for lit in rule.body)]
    if active_required:
        required = set(active_required)
        preferred.extend(rule for rule in expanded if required.issubset(body_predicates(rule)))
        preferred.extend(rule for rule in expanded if body_predicates(rule).intersection(required))

    return dedupe_rules(preferred + expanded)


def focus_predicates_for_action(action: AgentAction) -> Tuple[str, ...]:
    """Use required predicates as soft retrieval focus."""

    # Constraint predicates such as methodName are better added after path
    # retrieval.  Focusing traversal on them would turn attribute nodes into
    # detours and crowd out structural evidence paths.
    return tuple(dict.fromkeys(action.required_predicates))


def build_constraint_augmented_rules(
    task: TaskData,
    rules: Sequence[Rule],
    constraint_predicates: Sequence[str],
) -> List[Rule]:
    """Create rule variants with legal direct or attribute constraints.

    Some architecture relations, such as method-signature equivalence, are
    pure constraints and do not need a structural path skeleton.  Keep that
    capability action-driven: only synthesize an empty-base variant when the
    Planner or Refiner explicitly selected constraint predicates.
    """

    if not constraint_predicates:
        return []
    specs = {spec.name: spec for spec in task.predicates.values()}
    variants: List[Rule] = []
    base_rules: Sequence[Optional[Rule]] = rules or (None,)
    for rule in base_rules:
        base_body = [literal_text(literal) for literal in rule.body] if rule is not None else []
        for body in constraint_body_variants(task, base_body, constraint_predicates, specs):
            text = f"{task.target.name}(A,B) :- {','.join(body)}."
            try:
                variants.append(parse_rule(text, source="agent_action_constraint"))
            except ValueError:
                continue
    return variants


def constraint_body_variants(
    task: TaskData,
    base_body: List[str],
    constraint_predicates: Sequence[str],
    specs: Dict[str, object],
) -> List[List[str]]:
    variants: List[List[str]] = []
    max_body = task.max_body
    pair_literals: List[str] = []
    for name in constraint_predicates:
        spec = specs.get(name)
        if spec is None or len(base_body) >= max_body:
            continue
        if spec.arity != 2 or len(spec.types) != 2:
            continue
        if tuple(spec.types) == tuple(task.target.types):
            lit = f"{name}(A,B)"
            if lit not in base_body:
                body = base_body + [lit]
                variants.append(body)
                pair_literals.append(lit)
            continue
        if (
            len(task.target.types) == 2
            and task.target.types[0] == task.target.types[1]
            and spec.types[0] == task.target.types[0]
            and len(base_body) + 2 <= max_body
        ):
            var = f"C{len(base_body) + len(variants) + 1}"
            lit1 = f"{name}(A,{var})"
            lit2 = f"{name}(B,{var})"
            if lit1 not in base_body and lit2 not in base_body:
                variants.append(base_body + [lit1, lit2])

    combined = base_body + [lit for lit in pair_literals if lit not in base_body]
    if len(combined) <= max_body and len(combined) > len(base_body):
        variants.append(combined)
    return [dedupe_strings(body) for body in variants]


def literal_text(literal) -> str:
    return f"{literal.predicate}({','.join(literal.args)})"


def body_predicates(rule: Rule) -> set[str]:
    return {literal.predicate for literal in rule.body}


def dedupe_rules(rules: Sequence[Rule]) -> List[Rule]:
    output: List[Rule] = []
    seen = set()
    for rule in rules:
        text = rule_to_text(rule)
        if text in seen:
            continue
        seen.add(text)
        output.append(rule)
    return output


def cap_candidate_rules(rules: Sequence[Rule], limit: int) -> Tuple[List[Rule], bool]:
    """Bound symbolic verification cost while preserving action-priority order."""

    deduped = dedupe_rules(rules)
    return deduped[:limit], len(deduped) > limit


def dedupe_strings(items: Sequence[str]) -> List[str]:
    output: List[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def retrieval_portfolio(
    primary: RetrievalPlan,
    base: RetrievalPlan,
    task: TaskData,
    feedback: Optional[str],
    action: AgentAction,
) -> List[RetrievalPlan]:
    """Build a compact set of complementary retrieval plans."""

    max_body = max(1, task.max_body)
    target_name = task.target.name.lower()
    plans = [primary]
    presets = set(action.strategy_presets)

    # Strong deterministic baseline: this mirrors the best current GraphRAG
    # setting used in clean9 candidate experiments, and gives the agent a
    # reliable anchor instead of forcing every iteration through one LLM plan.
    plans.append(
        RetrievalPlan(
            max_depth=min(max_body, max(base.max_depth, 5)),
            max_positive_examples=max(base.max_positive_examples, 30),
            max_paths_per_example=max(base.max_paths_per_example, 40),
            max_edges_per_node=max(base.max_edges_per_node, 300),
            max_candidates=max(base.max_candidates, 60),
            enable_pair_constraints=True,
            traversal_strategy="bfs",
            constraint_mode="attribute" if "override" in target_name else "direct",
            min_path_support=2,
        )
    )

    if feedback in {"too_specific", "no_candidate"} or "wide_bfs" in presets:
        plans.append(
            RetrievalPlan(
                max_depth=max_body,
                max_positive_examples=max(base.max_positive_examples, 20),
                max_paths_per_example=max(base.max_paths_per_example, 60),
                max_edges_per_node=max(base.max_edges_per_node, 500),
                max_candidates=max(base.max_candidates, 100),
                enable_pair_constraints=False,
                traversal_strategy="bfs",
                constraint_mode="none",
                min_path_support=1,
            )
        )
    if feedback not in {"too_specific", "no_candidate"} or "constraint_heavy" in presets:
        plans.append(
            RetrievalPlan(
                max_depth=min(max_body, max(base.max_depth, 5)),
                max_positive_examples=max(base.max_positive_examples, 16),
                max_paths_per_example=max(base.max_paths_per_example, 30),
                max_edges_per_node=max(base.max_edges_per_node, 200),
                max_candidates=max(base.max_candidates, 80),
                enable_pair_constraints=True,
                traversal_strategy="bfs",
                constraint_mode="both",
                min_path_support=1,
            )
        )

    if "dfs_probe" in presets or len(plans) < 4:
        plans.append(
            RetrievalPlan(
                max_depth=min(max_body, max(base.max_depth, 5)),
                max_positive_examples=max(base.max_positive_examples, 12),
                max_paths_per_example=max(base.max_paths_per_example, 24),
                max_edges_per_node=max(base.max_edges_per_node, 160),
                max_candidates=max(base.max_candidates, 60),
                enable_pair_constraints=True,
                traversal_strategy="dfs",
                constraint_mode=primary.constraint_mode,
                min_path_support=1,
            )
        )

    if "support2_attr" in presets:
        plans.append(
            RetrievalPlan(
                max_depth=min(max_body, max(base.max_depth, 5)),
                max_positive_examples=max(base.max_positive_examples, 30),
                max_paths_per_example=max(base.max_paths_per_example, 40),
                max_edges_per_node=max(base.max_edges_per_node, 300),
                max_candidates=max(base.max_candidates, 80),
                enable_pair_constraints=True,
                traversal_strategy="bfs",
                constraint_mode="attribute",
                min_path_support=2,
            )
        )

    if "extended_body" in presets:
        plans.append(
            RetrievalPlan(
                max_depth=min(max_body + 2, max(base.max_depth + 2, 7)),
                max_positive_examples=max(base.max_positive_examples, 30),
                max_paths_per_example=max(base.max_paths_per_example, 40),
                max_edges_per_node=max(base.max_edges_per_node, 300),
                max_candidates=max(base.max_candidates, 120),
                enable_pair_constraints=True,
                traversal_strategy="bfs",
                constraint_mode="attribute" if "override" in target_name else "both",
                min_path_support=1,
                candidate_body_extra=2,
            )
        )

    deduped: List[RetrievalPlan] = []
    seen = set()
    for plan in plans:
        key = tuple(asdict(plan).items())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(plan)
    return deduped


def deterministic_fallback_plan(base: RetrievalPlan, task: TaskData) -> RetrievalPlan:
    """Mirror the strong deterministic GraphRAG setting used as a safety net."""

    target_name = task.target.name.lower()
    return RetrievalPlan(
        max_depth=min(max(1, task.max_body), max(base.max_depth, 5)),
        max_positive_examples=max(base.max_positive_examples, 30),
        max_paths_per_example=max(base.max_paths_per_example, 40),
        max_edges_per_node=max(base.max_edges_per_node, 300),
        max_candidates=max(base.max_candidates, 80),
        enable_pair_constraints=True,
        traversal_strategy="bfs",
        constraint_mode="attribute" if "override" in target_name else "direct",
        min_path_support=2,
        seed_max_depth=0,
    )


def clamp_int(value: object, minimum: int, maximum: int, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(minimum, min(maximum, parsed))


def enum_value(value: object, allowed: set[str], fallback: str) -> str:
    text = str(value)
    return text if text in allowed else fallback


def bool_value(value: object, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return fallback
