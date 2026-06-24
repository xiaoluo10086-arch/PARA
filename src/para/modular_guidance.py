"""Composable LLM guidance modules for PARA.

This module splits the old single "generate one guide JSON" prompt into four
independent steps:

1. predicate selection,
2. type inference,
3. candidate rule generation,
4. feedback interpretation.

Each step can be backed by llama.cpp, a deterministic local implementation, or
disabled where that makes sense. This makes ablation experiments explicit:
turning off one module changes only that module, not the whole pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .guidance import GuideProvider, HeuristicGuideProvider, _parse_candidate_if_available, extract_json_object
from .models import Guidance, PredicateSpec, Rule, TaskData
from .prolog import rule_to_text


@dataclass
class ModularGuidanceConfig:
    """Configuration for modular guidance and ablation switches."""

    predicate_selector: str = "llama"
    type_inferencer: str = "task"
    candidate_generator: str = "llama"
    feedback_interpreter: str = "llama"
    base_url: str = "http://127.0.0.1:8000"
    model: str = "Qwen3.5-27B-MaxCtx"
    request_timeout: int = 120
    cache_dir: str = ""


class ModularLLMGuideProvider(GuideProvider):
    """Guide provider composed of four independently replaceable modules."""

    def __init__(self, config: ModularGuidanceConfig):
        self.config = config
        self.client = LlamaJsonClient(
            base_url=config.base_url,
            model=config.model,
            request_timeout=config.request_timeout,
            cache_dir=config.cache_dir,
        )
        self.heuristic = HeuristicGuideProvider()
        self.previous_guidance: Optional[Guidance] = None

    def guide(
        self,
        task: TaskData,
        objective: str,
        predicate_budget: int,
        feedback: Optional[str] = None,
    ) -> Guidance:
        objective = objective or f"Learn {task.target.signature} for software architecture rules."

        ranked = self._select_predicates(task, objective, predicate_budget, feedback)
        selected = ranked[:predicate_budget]
        if not selected:
            raise RuntimeError("Predicate Selector returned no selected predicates")

        correction_note = ""
        type_constraints = self._infer_types(task, selected, objective)
        candidates = self._generate_candidates(task, selected, type_constraints, objective, feedback)
        if feedback and self.previous_guidance is not None and self.config.feedback_interpreter != "none":
            correction = self._interpret_feedback(task, selected, candidates, feedback, objective)
            correction_note = f" Feedback Interpreter: {correction.get('rationale', '')}"
            previous_selected = list(selected)
            selected = apply_predicate_corrections(task, selected, correction, predicate_budget)
            if selected != previous_selected:
                # 反馈解释器可能只返回“增加/删除谓词”，不直接给 revised rules。
                # 这种情况下必须基于修正后的谓词集合重新推断类型并生成候选，
                # 否则 feedback 只会改变 bias，不会真正改变 candidate-first 路径。
                type_constraints = self._infer_types(task, selected, objective)
                candidates = self._generate_candidates(task, selected, type_constraints, objective, feedback)
            extra_candidates = parse_candidate_texts(task, correction.get("revised_candidate_rules", []), source="modular_feedback")
            if extra_candidates:
                candidates = extra_candidates + candidates

        guidance = Guidance(
            ranked_predicates=ranked,
            selected_predicates=selected,
            candidate_rules=candidates,
            type_constraints=type_constraints,
            max_vars=task.max_vars,
            max_body=task.max_body,
            max_clauses=task.max_clauses,
            confidence=0.75 if self.config.predicate_selector == "llama" else 0.6,
            rationale=(
                "Modular guidance: "
                f"predicate_selector={self.config.predicate_selector}, "
                f"type_inferencer={self.config.type_inferencer}, "
                f"candidate_generator={self.config.candidate_generator}, "
                f"feedback_interpreter={self.config.feedback_interpreter}."
                f"{correction_note}"
            ),
        )
        self.previous_guidance = guidance
        return guidance

    def _select_predicates(
        self,
        task: TaskData,
        objective: str,
        predicate_budget: int,
        feedback: Optional[str],
    ) -> List[str]:
        mode = self.config.predicate_selector
        if mode == "heuristic":
            return self.heuristic.guide(task, objective, predicate_budget, feedback=feedback).ranked_predicates
        if mode == "all":
            return [spec.name for spec in task.predicates.values()]
        if mode != "llama":
            raise RuntimeError(f"Unsupported predicate selector mode: {mode}")

        data = self.client.chat_json(
            module="predicate_selector",
            cache_key=task_cache_key(task, objective, predicate_budget, feedback, "predicate_selector"),
            prompt=predicate_selector_prompt(task, objective, predicate_budget, feedback),
            max_tokens=768,
        )
        ranked = [normalize_predicate_name(name) for name in require_string_list(data, "ranked_predicates")]
        available = {spec.name for spec in task.predicates.values()}
        ranked = [name for name in ranked if name in available]
        if not ranked:
            raise RuntimeError("Predicate Selector produced no available predicates")
        return ranked

    def _infer_types(
        self,
        task: TaskData,
        selected: List[str],
        objective: str,
    ) -> Dict[str, tuple[str, ...]]:
        mode = self.config.type_inferencer
        if mode == "task":
            return {
                spec.name: spec.types
                for name in selected
                if (spec := find_predicate(task, name)) is not None and spec.types
            }
        if mode == "none":
            return {}
        if mode != "llama":
            raise RuntimeError(f"Unsupported type inferencer mode: {mode}")

        data = self.client.chat_json(
            module="type_inferencer",
            cache_key=task_cache_key(task, objective, len(selected), None, "type_inferencer:" + ",".join(selected)),
            prompt=type_inferencer_prompt(task, selected, objective),
            max_tokens=768,
        )
        raw = data.get("type_constraints")
        if not isinstance(raw, dict):
            raise RuntimeError("Type Inferencer must return object field `type_constraints`")
        output: Dict[str, tuple[str, ...]] = {}
        for name, values in raw.items():
            clean_name = normalize_predicate_name(str(name))
            if clean_name not in selected or not isinstance(values, list):
                continue
            output[clean_name] = tuple(str(value) for value in values)
        if not output:
            raise RuntimeError("Type Inferencer produced no usable type constraints")
        return output

    def _generate_candidates(
        self,
        task: TaskData,
        selected: List[str],
        type_constraints: Dict[str, tuple[str, ...]],
        objective: str,
        feedback: Optional[str],
    ) -> List[Rule]:
        mode = self.config.candidate_generator
        if mode == "none":
            return []
        if mode == "heuristic":
            guide = Guidance(
                ranked_predicates=selected,
                selected_predicates=selected,
                candidate_rules=[],
                type_constraints=type_constraints,
            )
            # Reuse the deterministic typed-path generator with the selected
            # budget, then keep only candidates inside selected.
            candidates = self.heuristic.guide(task, objective, len(selected), feedback=feedback).candidate_rules
            return [rule for rule in candidates if all(lit.predicate in selected for lit in rule.body)]
        if mode != "llama":
            raise RuntimeError(f"Unsupported candidate generator mode: {mode}")

        data = self.client.chat_json(
            module="candidate_generator",
            cache_key=task_cache_key(task, objective, len(selected), feedback, "candidate_generator:" + ",".join(selected)),
            prompt=candidate_generator_prompt(task, selected, type_constraints, objective, feedback),
            max_tokens=1024,
        )
        candidate_texts = require_string_list(data, "candidate_rules")
        candidates = parse_candidate_texts(task, candidate_texts, source="modular_llm")
        if not candidates:
            raise RuntimeError("Candidate Rule Generator produced no parseable candidate rules")
        return candidates

    def _interpret_feedback(
        self,
        task: TaskData,
        selected: List[str],
        candidates: List[Rule],
        feedback: str,
        objective: str,
    ) -> Dict[str, object]:
        if self.config.feedback_interpreter == "none":
            return {}
        if self.config.feedback_interpreter == "heuristic":
            if feedback == "too_specific":
                ranked = self.heuristic.guide(task, objective, len(task.predicates), feedback=feedback).ranked_predicates
                return {"add_predicates": [name for name in ranked if name not in selected][:2], "rationale": "heuristic expansion"}
            if feedback == "too_general":
                return {"remove_predicates": selected[-1:], "rationale": "heuristic contraction"}
            return {}
        if self.config.feedback_interpreter != "llama":
            raise RuntimeError(f"Unsupported feedback interpreter mode: {self.config.feedback_interpreter}")

        data = self.client.chat_json(
            module="feedback_interpreter",
            cache_key=task_cache_key(task, objective, len(selected), feedback, "feedback_interpreter:" + ",".join(selected)),
            prompt=feedback_interpreter_prompt(task, selected, candidates, feedback, objective),
            max_tokens=768,
        )
        return data


class LlamaJsonClient:
    """Small llama.cpp JSON client with optional per-module caching."""

    def __init__(self, base_url: str, model: str, request_timeout: int, cache_dir: str = ""):
        self.base_url = (
            base_url
            or os.getenv("PARA_LLM_BASE_URL")
            or os.getenv("NSHRL_LLM_BASE_URL")
            or "http://127.0.0.1:8000"
        ).rstrip("/")
        self.model = model or os.getenv("PARA_LLM_MODEL") or os.getenv("NSHRL_LLM_MODEL") or "Qwen3.5-27B-MaxCtx"
        self.request_timeout = request_timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def chat_json(self, module: str, cache_key: str, prompt: str, max_tokens: int) -> Dict[str, object]:
        """Call one modular LLM step and require a JSON object.

        这里优先使用 OpenAI-compatible `json_schema` 响应格式。较新的
        llama.cpp server 会把 JSON Schema 转成生成约束，从采样阶段减少
        “自然语言分析 + JSON” 或坏 JSON。若当前服务不支持 `json_schema`
        并返回 HTTP 400/422，则退回到较弱的 `json_object`，避免实验因为
        服务版本差异直接不可运行；退回行为仍会被严格解析器检查。
        """

        cache_path = self.cache_path(module, cache_key)
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        response_format = module_response_format(module)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "/no_think\n"
                        "You are a strict JSON generator for inductive logic programming. "
                        "Return one compact JSON object only. No markdown. No prose."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
            "response_format": response_format,
        }
        raw = self._post_chat(payload)
        response = json.loads(raw)
        message = response["choices"][0]["message"]
        content = message.get("content") or message.get("reasoning_content") or ""
        try:
            data = extract_json_object(content)
        except Exception as first_exc:
            repair_prompt = "\n".join(
                [
                    "/no_think",
                    "Convert the previous answer to one compact valid JSON object.",
                    f"Expected schema for module `{module}`:",
                    module_prompt_schema_text(module),
                    "No explanation. No markdown. First character must be `{`.",
                    "Previous answer:",
                    content[:3000],
                ]
            )
            repaired = self.chat_text(repair_prompt, max_tokens=min(max_tokens, 768))
            try:
                data = extract_json_object(repaired)
            except Exception as second_exc:
                raise RuntimeError(f"{module} failed: {first_exc}; repair also failed: {second_exc}") from second_exc
        if cache_path:
            cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return data

    def chat_text(self, prompt: str, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "/no_think\nReturn one compact JSON object only. No prose. No markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        raw = self._post_chat(payload)
        response = json.loads(raw)
        message = response["choices"][0]["message"]
        return message.get("content") or message.get("reasoning_content") or ""

    def _post_chat(self, payload: Dict[str, object]) -> str:
        """Post to llama.cpp, retrying with `json_object` if schema is unsupported."""

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            # llama.cpp 版本不同，`json_schema` 支持并不总是一致。只有在
            # 服务器明确拒绝 schema 时才退回 json_object；其它错误继续抛出。
            if payload.get("response_format", {}).get("type") != "json_schema" or exc.code not in {400, 422}:
                raise
            fallback_payload = dict(payload)
            fallback_payload["response_format"] = {"type": "json_object"}
            fallback_req = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=json.dumps(fallback_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(fallback_req, timeout=self.request_timeout) as resp:
                return resp.read().decode("utf-8")

    def cache_path(self, module: str, cache_key: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        safe_module = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in module)
        path = self.cache_dir / safe_module
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{cache_key}.json"


def module_response_format(module: str) -> Dict[str, object]:
    """Return a module-specific JSON Schema for constrained generation."""

    schemas: Dict[str, Dict[str, object]] = {
        "predicate_selector": {
            "type": "object",
            "additionalProperties": False,
            "required": ["ranked_predicates", "rationale"],
            "properties": {
                "ranked_predicates": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "rationale": {"type": "string"},
            },
        },
        "type_inferencer": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type_constraints", "rationale"],
            "properties": {
                "type_constraints": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "rationale": {"type": "string"},
            },
        },
        "candidate_generator": {
            "type": "object",
            "additionalProperties": False,
            "required": ["candidate_rules", "rationale"],
            "properties": {
                "candidate_rules": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
                "rationale": {"type": "string"},
            },
        },
        "feedback_interpreter": {
            "type": "object",
            "additionalProperties": False,
            "required": ["add_predicates", "remove_predicates", "revised_candidate_rules", "rationale"],
            "properties": {
                "add_predicates": {"type": "array", "items": {"type": "string"}},
                "remove_predicates": {"type": "array", "items": {"type": "string"}},
                "revised_candidate_rules": {"type": "array", "items": {"type": "string"}},
                "rationale": {"type": "string"},
            },
        },
    }
    schema = schemas.get(module)
    if schema is None:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"nshrl_{module}",
            "strict": True,
            "schema": schema,
        },
    }


def module_prompt_schema_text(module: str) -> str:
    """Compact schema reminder used in repair prompts."""

    examples = {
        "predicate_selector": '{"ranked_predicates":["containsMethod","callsMethod"],"rationale":"short"}',
        "type_inferencer": '{"type_constraints":{"containsMethod":["class","method"]},"rationale":"short"}',
        "candidate_generator": '{"candidate_rules":["target(A,B) :- p(A,V1),q(V1,B)."],"rationale":"short"}',
        "feedback_interpreter": '{"add_predicates":[],"remove_predicates":[],"revised_candidate_rules":[],"rationale":"short"}',
    }
    return examples.get(module, "{}")


def predicate_selector_prompt(task: TaskData, objective: str, predicate_budget: int, feedback: Optional[str]) -> str:
    return "\n".join(
        [
            "/no_think",
            "Module: Predicate Selector.",
            "Select and rank body predicates that are most useful for the target rule.",
            "Return JSON schema: {\"ranked_predicates\":[\"predicateName\"],\"rationale\":\"short\"}",
            f"Objective: {objective}",
            f"Target: {task.target.signature}, types={task.target.types}",
            f"Budget: top {predicate_budget}",
            f"Feedback: {feedback or 'none'}",
            "Available predicates:",
            "\n".join(predicate_catalog(task)),
            "Positive examples:",
            "\n".join(example_lines(task, positive=True, limit=5)),
            "Negative examples:",
            "\n".join(example_lines(task, positive=False, limit=5)),
            "Return only JSON.",
        ]
    )


def type_inferencer_prompt(task: TaskData, selected: List[str], objective: str) -> str:
    return "\n".join(
        [
            "/no_think",
            "Module: Type Inferencer.",
            "Infer argument type constraints for selected predicates.",
            "Return JSON schema: {\"type_constraints\":{\"predicateName\":[\"type1\",\"type2\"]},\"rationale\":\"short\"}",
            f"Objective: {objective}",
            f"Target: {task.target.signature}, types={task.target.types}",
            "Selected predicates:",
            "\n".join(selected),
            "Metamodel predicate declarations:",
            "\n".join(predicate_catalog(task)),
            "Return only JSON.",
        ]
    )


def candidate_generator_prompt(
    task: TaskData,
    selected: List[str],
    type_constraints: Dict[str, tuple[str, ...]],
    objective: str,
    feedback: Optional[str],
) -> str:
    return "\n".join(
        [
            "/no_think",
            "Module: Candidate Rule Generator.",
            "Generate 1 to 3 Horn clauses for the target predicate.",
            "Rules may be imperfect but must use only selected predicates.",
            "Return JSON schema: {\"candidate_rules\":[\"target(A,B) :- p(A,C),q(C,B).\"],\"rationale\":\"short\"}",
            f"Objective: {objective}",
            f"Target: {task.target.name}/{task.target.arity}",
            f"Feedback: {feedback or 'none'}",
            "Selected predicates:",
            "\n".join(selected),
            "Type constraints:",
            json.dumps({k: list(v) for k, v in type_constraints.items()}, ensure_ascii=False),
            "Positive examples:",
            "\n".join(example_lines(task, positive=True, limit=6)),
            "Negative examples:",
            "\n".join(example_lines(task, positive=False, limit=6)),
            "Return only JSON.",
        ]
    )


def feedback_interpreter_prompt(
    task: TaskData,
    selected: List[str],
    candidates: List[Rule],
    feedback: str,
    objective: str,
) -> str:
    return "\n".join(
        [
            "/no_think",
            "Module: Feedback Interpreter.",
            "Given symbolic feedback, propose local repairs to predicate selection or candidate rules.",
            "Return JSON schema: {\"add_predicates\":[\"p\"],\"remove_predicates\":[\"q\"],\"revised_candidate_rules\":[\"target(A,B) :- ... .\"],\"rationale\":\"short\"}",
            f"Objective: {objective}",
            f"Target: {task.target.signature}",
            f"Feedback: {feedback}",
            "Current selected predicates:",
            "\n".join(selected),
            "Current candidate rules:",
            "\n".join(rule_to_text(rule) for rule in candidates) or "none",
            "Available predicates:",
            "\n".join(predicate_catalog(task)),
            "Return only JSON.",
        ]
    )


def predicate_catalog(task: TaskData) -> List[str]:
    return [f"- {spec.name}/{spec.arity}: {spec.types}" for spec in task.predicates.values()]


def example_lines(task: TaskData, positive: bool, limit: int) -> List[str]:
    rows = [example for example in task.examples if example.positive == positive][:limit]
    return [f"- {example.literal.predicate}{example.literal.args}" for example in rows] or ["- none"]


def require_string_list(data: Dict[str, object], key: str) -> List[str]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"Module output must contain non-empty list `{key}`")
    return [str(item) for item in value]


def parse_candidate_texts(task: TaskData, texts: object, source: str) -> List[Rule]:
    if not isinstance(texts, list):
        return []
    available = {spec.name for spec in task.predicates.values()}
    rules: List[Rule] = []
    for text in texts:
        rules.extend(_parse_candidate_if_available(str(text), source=source, available=available))
    return rules


def apply_predicate_corrections(
    task: TaskData,
    selected: List[str],
    correction: Dict[str, object],
    predicate_budget: int,
) -> List[str]:
    available = {spec.name for spec in task.predicates.values()}
    remove = {normalize_predicate_name(str(name)) for name in correction.get("remove_predicates", []) if isinstance(name, str)}
    add = [
        normalize_predicate_name(str(name))
        for name in correction.get("add_predicates", [])
        if isinstance(name, str) and normalize_predicate_name(str(name)) in available
    ]
    output = [name for name in selected if name not in remove]
    for name in add:
        if name not in output:
            output.append(name)
    return output[:predicate_budget]


def normalize_predicate_name(value: str) -> str:
    """Normalize LLM predicate names.

    模型经常把谓词写成 `importsClass/2` 或 Markdown 列表项
    `- importsClass/2: ...`。内部匹配只使用裸谓词名，所以这里把这些
    表达统一清洗为 `importsClass`。
    """

    text = value.strip().strip("`").lstrip("-").strip()
    if ":" in text:
        text = text.split(":", 1)[0].strip()
    if "/" in text:
        text = text.split("/", 1)[0].strip()
    return text


def find_predicate(task: TaskData, name: str) -> Optional[PredicateSpec]:
    for spec in task.predicates.values():
        if spec.name == name:
            return spec
    return None


def task_cache_key(
    task: TaskData,
    objective: str,
    predicate_budget: int,
    feedback: Optional[str],
    module_key: str,
) -> str:
    payload = {
        "task_dir": task.task_dir,
        "target": task.target.signature,
        "objective": objective,
        "predicate_budget": predicate_budget,
        "feedback": feedback,
        "module": module_key,
        "predicates": [spec.signature for spec in task.predicates.values()],
        "examples": [
            [example.positive, example.literal.predicate, example.literal.args]
            for example in task.examples[:20]
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:24]
