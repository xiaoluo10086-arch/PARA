"""LLM-guidance layer.

The project can run without network access by using `HeuristicGuideProvider`.
That provider plays the role of a deterministic LLM surrogate for experiments:
it ranks predicates by semantic relevance and emits candidate rules.  A real
LLM adapter can implement the same `guide` method and return a `Guidance`.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .models import Guidance, Literal, PredicateSpec, Rule, TaskData
from .prolog import parse_rule


class GuideProvider:
    """Interface for LLM or LLM-like guidance providers."""

    def guide(
        self,
        task: TaskData,
        objective: str,
        predicate_budget: int,
        feedback: Optional[str] = None,
    ) -> Guidance:
        # 所有引导层都只需要实现这一层接口。
        # 真实 LLM 接入时可以在这里组织 prompt、调用 API、解析 JSON；
        # 本地启发式 provider 则直接返回可复现的 Guidance。
        raise NotImplementedError


class HeuristicGuideProvider(GuideProvider):
    """Deterministic local substitute for an LLM.

    The ranking is intentionally transparent so it can be reported in the
    method section and used in offline experiments.
    """

    def guide(
        self,
        task: TaskData,
        objective: str,
        predicate_budget: int,
        feedback: Optional[str] = None,
    ) -> Guidance:
        # 第一步：对 bias.pl 中允许出现的 body_pred 做重要性排序。
        # 排序越靠前，越优先保留到剪枝后的 Popper bias 中。
        ranked = self._rank_predicates(task, objective)

        # 第二步：根据上一轮的错误类型调节搜索空间。
        # 规则过特化时扩大谓词预算和规则体长度；规则过泛化时略微收紧谓词预算。
        if feedback == "too_specific":
            budget = min(len(ranked), predicate_budget + 2)
            max_body = min(task.max_body + 2, 10)
        elif feedback == "too_general":
            budget = max(1, predicate_budget - 1)
            max_body = max(2, task.max_body)
        else:
            budget = min(len(ranked), predicate_budget)
            max_body = task.max_body

        selected = ranked[:budget]

        # 第三步：生成少量结构化候选规则。
        # 这些候选规则既可以直接作为 LLM-only 基线评估，也可以辅助解释 Popper 的搜索方向。
        candidates = self._candidate_rules(task, selected)
        return Guidance(
            ranked_predicates=ranked,
            selected_predicates=selected,
            candidate_rules=candidates,
            max_vars=task.max_vars,
            max_body=max_body,
            max_clauses=task.max_clauses,
            confidence=0.65,
            rationale=(
                "Local heuristic guidance ranked predicates by target-type overlap, "
                "software-architecture keywords, and generic typed predicate paths."
            ),
        )

    def _rank_predicates(self, task: TaskData, objective: str) -> List[str]:
        objective_lower = objective.lower()
        target_types = set(task.target.types)
        scored = []
        for key, spec in task.predicates.items():
            score = 0
            # 类型重合是最强信号。例如目标是 package->package 时，
            # containsClass(package,class) 比 class(class) 更可能连接目标变量。
            if target_types and target_types.intersection(spec.types):
                score += 6
            lname = spec.name.lower()

            # 软件架构规则中，结构关系谓词通常比纯类型谓词更有解释力。
            if any(word in lname for word in ("contains", "import", "call", "inherit", "implement")):
                score += 4

            # 针对 isAllowedToUse/2 的论文先验：
            # 包级 allowed-use 通常由“包包含类 + 类导入类”解释。
            if "allowed" in task.target.name.lower() and any(word in lname for word in ("contains", "import")):
                score += 5

            # 针对方法/类调用规则的先验：
            # callsMethod 和 containsMethod 应优先进入搜索空间。
            if "call" in task.target.name.lower() and any(word in lname for word in ("call", "method", "contains")):
                score += 5
            if "override" in task.target.name.lower() and any(
                word in lname for word in ("inherit", "extends", "implement", "method", "same")
            ):
                score += 5
            if spec.name.lower() in objective_lower:
                score += 3

            # 二元关系更容易连接规则头中的两个实体；一元类型谓词通常只作为辅助约束。
            if spec.arity == 2:
                score += 2
            if spec.arity == 1:
                score += 1
            scored.append((score, spec.name))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [name for _, name in scored]

    def _candidate_rules(self, task: TaskData, selected: List[str]) -> List[Rule]:
        """Generate candidates from the typed predicate graph.

        这里刻意不按 `isAllowedToUse`、`canCallClass`、`overridesMethod`
        这些目标名写模板。候选规则来自一个通用过程：

        1. 把每个二元谓词看作类型图中的一条有向边，并允许反向使用；
        2. 从 head 的第一个参数类型出发，枚举能到达第二个参数类型的短路径；
        3. 对同类型目标变量，附加 `same/equal/match/name/arity` 等约束谓词。

        因此，`class -> method -> method -> class` 可以自然生成类调用规则，
        `package -> class -> class -> package` 可以自然生成包使用规则，
        `method -> class -> class -> method + same*` 可以自然生成覆写规则。
        """

        target = task.target
        if target.arity != 2 or len(target.types) != 2:
            return self._fallback_candidates(target, selected)

        specs_by_name = {spec.name: spec for spec in task.predicates.values() if spec.name in selected}
        binary_specs = [specs_by_name[name] for name in selected if (name in specs_by_name and specs_by_name[name].arity == 2 and len(specs_by_name[name].types) == 2)]
        if not binary_specs:
            return self._fallback_candidates(target, selected)

        start_type, end_type = target.types
        paths = enumerate_typed_paths(
            binary_specs,
            start_type,
            end_type,
            max_len=min(max(1, task.max_body), 4),
            target_name=target.name,
        )
        pair_constraints = target_pair_constraints(binary_specs, target.types)

        candidates: List[str] = []
        seen = set()
        for path in paths:
            body_literals = path_to_literals(path)
            used = {literal.split("(", 1)[0] for literal in body_literals}
            extra_literals = [literal for name, literal in pair_constraints if name not in used]

            # 先加入纯路径规则，再加入带同名/同参等约束的规则。
            variants = [body_literals]
            if extra_literals:
                variants.append(body_literals + extra_literals[: max(0, task.max_body - len(body_literals))])

            for body in variants:
                if not body or len(body) > task.max_body:
                    continue
                text = f"{target.name}(A,B) :- {','.join(body)}."
                if text not in seen:
                    candidates.append(text)
                    seen.add(text)
            if len(candidates) >= 40:
                break
            if len(candidates) >= 40:
                break

        if not candidates:
            return self._fallback_candidates(target, selected)
        return [parse_rule(text, source="typed_path_heuristic") for text in candidates]

    def _fallback_candidates(self, target: PredicateSpec, selected: List[str]) -> List[Rule]:
        """Return simple non-oracle fallback candidates for diagnostics."""

        candidates: List[str] = []
        for name in selected[:3]:
            variables = ",".join(chr(ord("A") + i) for i in range(max(1, target.arity)))
            candidates.append(f"{target.name}({variables}) :- {name}({variables}).")
        return [parse_rule(text, source="typed_path_fallback") for text in candidates]


TypedEdge = Tuple[PredicateSpec, bool, str, str]


def enumerate_typed_paths(
    specs: Sequence[PredicateSpec],
    start_type: str,
    end_type: str,
    max_len: int,
    target_name: str = "",
) -> List[List[TypedEdge]]:
    """Enumerate short typed paths between two target argument types."""

    adjacency: Dict[str, List[TypedEdge]] = {}
    for spec in specs:
        left, right = spec.types
        adjacency.setdefault(left, []).append((spec, False, left, right))
        adjacency.setdefault(right, []).append((spec, True, right, left))

    paths: List[List[TypedEdge]] = []

    def dfs(current_type: str, current_path: List[TypedEdge], used_counts: Dict[str, int]) -> None:
        if current_path and current_type == end_type:
            paths.append(list(current_path))
        if len(current_path) >= max_len:
            return
        for edge in adjacency.get(current_type, []):
            spec, _reversed, _src, dst = edge
            # 包含关系常常需要正反向各用一次，例如
            # package -> class -> class -> package。允许同一谓词最多出现两次，
            # 但继续限制更多重复，避免类型图中产生无意义环路。
            if used_counts.get(spec.name, 0) >= 2:
                continue
            current_path.append(edge)
            used_counts[spec.name] = used_counts.get(spec.name, 0) + 1
            dfs(dst, current_path, used_counts)
            used_counts[spec.name] -= 1
            if used_counts[spec.name] == 0:
                del used_counts[spec.name]
            current_path.pop()

    dfs(start_type, [], {})
    paths.sort(key=lambda path: typed_path_score(path, target_name))
    return paths


def typed_path_score(path: List[TypedEdge], target_name: str = "") -> Tuple[int, int, int, str]:
    """Prefer short, architecture-meaningful paths."""

    names = [edge[0].name.lower() for edge in path]
    keyword_score = 0
    target_score = 0
    target_lower = target_name.lower()
    for name in names:
        if any(word in name for word in ("contains", "import", "call", "inherit", "extend", "implement")):
            keyword_score += 2
        if any(word in name for word in ("same", "name", "arity")):
            keyword_score -= 1
    # 这是语义排序，不是规则模板：目标名中的 call/use/override 等词只影响
    # 路径优先级，候选体仍由谓词类型图枚举产生。
    if "call" in target_lower and any("call" in name for name in names):
        target_score += 16
    if any(word in target_lower for word in ("use", "allowed", "allow")) and any("import" in name for name in names):
        target_score += 16
    if "override" in target_lower:
        if any("inherit" in name for name in names):
            target_score += 16
        elif any(("extend" in name or "implement" in name) for name in names):
            target_score += 8
    return (-target_score, len(path), -keyword_score, ",".join(names))


def path_to_literals(path: List[TypedEdge]) -> List[str]:
    """Instantiate a typed path as Prolog body literals."""

    var_by_position = ["A"] + [f"V{i}" for i in range(1, len(path))] + ["B"]
    literals = []
    for idx, (spec, reversed_edge, _src_type, _dst_type) in enumerate(path):
        src_var = var_by_position[idx]
        dst_var = var_by_position[idx + 1]
        args = (dst_var, src_var) if reversed_edge else (src_var, dst_var)
        literals.append(f"{spec.name}({args[0]},{args[1]})")
    return literals


def target_pair_constraints(
    specs: Sequence[PredicateSpec],
    target_types: Tuple[str, ...],
) -> List[Tuple[str, str]]:
    """Return generic same/equality-style constraints over A and B."""

    constraints = []
    for spec in specs:
        lname = spec.name.lower()
        if not any(word in lname for word in ("same", "equal", "match", "name", "arity", "signature")):
            continue
        if spec.types == target_types:
            constraints.append((spec.name, f"{spec.name}(A,B)"))
        elif spec.types == tuple(reversed(target_types)):
            constraints.append((spec.name, f"{spec.name}(B,A)"))
    constraints.sort(key=lambda item: item[0])
    return constraints


class JsonGuideProvider(GuideProvider):
    """Load guidance from a JSON file produced by an external LLM call."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def guide(
        self,
        task: TaskData,
        objective: str,
        predicate_budget: int,
        feedback: Optional[str] = None,
    ) -> Guidance:
        # JSON guide 是真实 LLM 接入的最轻量方式：
        # 先离线/在线调用模型得到 JSON，再交给本系统完成 Popper 验证和指标计算。
        data = json.loads(self.path.read_text(encoding="utf-8"))
        ranked = data.get("ranked_predicates", [])
        selected = data.get("selected_predicates") or ranked[:predicate_budget]
        available = {spec.name for spec in task.predicates.values()}
        ranked = [name for name in ranked if name in available]
        selected = [name for name in selected if name in available][:predicate_budget]
        candidates = [
            rule
            for text in data.get("candidate_rules", [])
            for rule in _parse_candidate_if_available(str(text), source="json_llm", available=available)
        ]
        return Guidance(
            ranked_predicates=ranked,
            selected_predicates=selected,
            candidate_rules=candidates,
            type_constraints={k: tuple(v) for k, v in data.get("type_constraints", {}).items()},
            max_vars=int(data.get("max_vars", task.max_vars)),
            max_body=int(data.get("max_body", task.max_body)),
            max_clauses=int(data.get("max_clauses", task.max_clauses)),
            confidence=float(data.get("confidence", 0.5)),
            rationale=data.get("rationale", "Loaded from JSON guide."),
        )


class LlamaCppGuideProvider(GuideProvider):
    """Use a llama.cpp OpenAI-compatible server as the LLM guidance layer.

    The user starts llama.cpp with `llama-server --port 8000 --alias ...`.
    llama-server exposes `/v1/chat/completions`, so this provider sends a
    JSON-only prompt and parses the model response into `Guidance`.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        request_timeout: int = 120,
        fallback: Optional[GuideProvider] = None,
        strict: bool = True,
        merge_heuristic_guardrail: bool = False,
        reuse_previous_on_feedback_failure: bool = False,
    ):
        # base_url 和 model 都支持环境变量，便于在批量实验脚本中统一配置。
        # 例如：
        #   export NSHRL_LLM_BASE_URL=http://dgx-host:8000
        #   export NSHRL_LLM_MODEL=Qwen3.5-27B-MaxCtx
        self.base_url = (
            base_url
            or os.getenv("PARA_LLM_BASE_URL")
            or os.getenv("NSHRL_LLM_BASE_URL")
            or "http://127.0.0.1:8000"
        ).rstrip("/")
        self.model = model or os.getenv("PARA_LLM_MODEL") or os.getenv("NSHRL_LLM_MODEL") or "Qwen3.5-27B-MaxCtx"
        self.request_timeout = request_timeout

        # strict=True 是正式实验默认值：LLM 引导失败就让 PARA 明确失败，
        # 不能静默退回 heuristic 或近似 pure Popper。这样论文统计能真实反映
        # LLM 引导层的可靠性，而不是被兜底逻辑掩盖。
        self.strict = strict
        self.merge_heuristic_guardrail = merge_heuristic_guardrail
        self.reuse_previous_on_feedback_failure = reuse_previous_on_feedback_failure

        # fallback 仅用于显式打开非严格调试模式，或显式请求 heuristic guardrail。
        # 正式 PARA/llama 实验默认不会使用它。
        self.fallback = fallback or HeuristicGuideProvider()
        self._last_online_guidance: Optional[Guidance] = None

    def guide(
        self,
        task: TaskData,
        objective: str,
        predicate_budget: int,
        feedback: Optional[str] = None,
    ) -> Guidance:
        # 构造 prompt 时不会把完整 bk.pl 全量塞给模型，只放谓词目录、
        # 少量正负例和事实样本，避免上下文浪费和大型事实库下的 token 爆炸。
        prompt = build_llm_prompt(task, objective, predicate_budget, feedback)
        try:
            data = self._chat_json(prompt)
            validate_guidance_schema(data)

            # 将模型 JSON 标准化为内部 Guidance 对象，同时过滤不存在的谓词，
            # 防止 LLM 幻觉出 bias.pl 中没有声明的 body predicate。
            guidance = guidance_from_dict(data, task, predicate_budget, source="llama_cpp")
            if not guidance.selected_predicates:
                raise ValueError("LLM response did not select any predicate")

            # 可选 guardrail 只在用户显式打开时启用。它适合工程调试，不适合
            # 论文中宣称“LLM 引导”的严格实验，因为它会混入本地先验。
            if self.merge_heuristic_guardrail:
                guidance = merge_with_heuristic_guidance(
                    guidance,
                    self.fallback.guide(task, objective, predicate_budget, feedback=feedback),
                    predicate_budget,
                )
            self._last_online_guidance = guidance
            return guidance
        except Exception as exc:
            # 非严格调试模式下，可以复用上一轮在线 LLM guidance；正式实验默认关闭。
            if self.reuse_previous_on_feedback_failure and self._last_online_guidance is not None:
                reused = Guidance(
                    ranked_predicates=list(self._last_online_guidance.ranked_predicates),
                    selected_predicates=list(self._last_online_guidance.selected_predicates),
                    candidate_rules=list(self._last_online_guidance.candidate_rules),
                    type_constraints=dict(self._last_online_guidance.type_constraints),
                    max_vars=self._last_online_guidance.max_vars,
                    max_body=self._last_online_guidance.max_body,
                    max_clauses=self._last_online_guidance.max_clauses,
                    confidence=max(0.1, self._last_online_guidance.confidence - 0.05),
                    rationale=(
                        "Reused previous successful online llama_cpp guidance because "
                        f"the feedback-round LLM response was not valid JSON: {exc}."
                    ),
                )
                return reused

            if self.strict:
                raise RuntimeError(f"LLM guidance failed at {self.base_url}: {exc}") from exc

            # 只有非严格调试模式才退回 heuristic。正式实验不要打开这个分支。
            fallback_guidance = self.fallback.guide(task, objective, predicate_budget, feedback=feedback)
            fallback_guidance.rationale = (
                f"FIRST_ROUND_LLM_GUIDANCE_FAILED at {self.base_url}: {exc}. "
                f"Used non-strict heuristic fallback. {fallback_guidance.rationale}"
            )
            return fallback_guidance

    def _chat_json(self, prompt: str) -> Dict[str, object]:
        # llama.cpp 的 /v1/chat/completions 接口兼容 OpenAI 格式。
        # 这里不依赖 openai/requests 包，故 rule_learning 环境只需 Python 标准库。
        content = self._chat_text(prompt, max_tokens=1536)
        try:
            return extract_json_object(content)
        except Exception as first_exc:
            # 本地 Qwen 偶尔会先输出分析文字，导致第一轮没有 JSON。
            # 第二轮只给它“修复格式”的任务，不再提供长事实样本，通常能稳定得到 JSON。
            repair_prompt = "\n".join(
                [
                    "/no_think",
                    "Your previous answer was not valid JSON.",
                    "Convert it to one compact JSON object only.",
                    "No explanation. No markdown. First character must be `{`.",
                    "Previous answer:",
                    content[:3000],
                ]
            )
            repaired = self._chat_text(repair_prompt, max_tokens=1024)
            try:
                return extract_json_object(repaired)
            except Exception as second_exc:
                raise ValueError(f"{first_exc}; repair also failed: {second_exc}") from second_exc

    def _chat_text(self, prompt: str, max_tokens: int) -> str:
        """Call llama.cpp and return the assistant text."""

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert in inductive logic programming and software architecture. "
                        "Do not think step by step. Do not explain. Your entire answer must start "
                        "with { and end with }. Return only one valid JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
            # llama.cpp 支持 OpenAI 风格的 JSON mode。
            # 对 Qwen 这类会输出 reasoning_content 的模型，JSON mode 能显著降低空 content
            # 或 Markdown 包裹导致的解析失败。
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # urlopen 可能抛出连接错误、超时或 HTTP 错误；外层 guide() 会捕获并降级。
        with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
            raw = resp.read().decode("utf-8")
        response = json.loads(raw)

        # Qwen/llama.cpp 可能把内容放在 content，也可能额外给 reasoning_content。
        # 优先取 content；为空时再尝试 reasoning_content。
        message = response["choices"][0]["message"]
        content = message.get("content") or message.get("reasoning_content") or ""
        return content


def build_llm_prompt(
    task: TaskData,
    objective: str,
    predicate_budget: int,
    feedback: Optional[str],
) -> str:
    """Build a compact prompt for predicate pruning and candidate generation."""

    predicate_lines = []
    for spec in task.predicates.values():
        # 把谓词类型一起给模型，让它知道哪些谓词能连接 package/class/method。
        types = f"({', '.join(spec.types)})" if spec.types else "(unknown)"
        predicate_lines.append(f"- {spec.name}/{spec.arity}: {types}")

    # 少量样例足以提示目标关系的方向；数量太多反而会让本地模型跑慢。
    pos_examples = [ex.literal for ex in task.examples if ex.positive][:4]
    neg_examples = [ex.literal for ex in task.examples if not ex.positive][:4]

    # fact_sample 用于给模型一点真实结构背景，但真正的规则验证由 Popper/evaluator 完成。
    fact_sample = task.facts[:16]

    # 控制提示长度：真实项目事实库可能很大，LLM 只需要谓词目录、少量事实样本和样例。
    return "\n".join(
        [
            "/no_think",
            "Return only one compact JSON object. No analysis, no prose, no markdown.",
            "The first character of your answer must be `{` and the last must be `}`.",
            "",
            f"Objective: {objective}",
            f"Target predicate: {task.target.name}/{task.target.arity} with types {task.target.types}",
            f"Predicate budget: select at most {predicate_budget} body predicates.",
            f"Previous feedback: {feedback or 'none'}",
            "",
            "Available predicates:",
            "\n".join(predicate_lines),
            "",
            "Positive examples:",
            "\n".join(f"- {lit.predicate}{lit.args}" for lit in pos_examples) or "- none",
            "",
            "Negative examples:",
            "\n".join(f"- {lit.predicate}{lit.args}" for lit in neg_examples) or "- none",
            "",
            "Background fact sample:",
            "\n".join(f"- {lit.predicate}{lit.args}" for lit in fact_sample),
            "",
            "Return a JSON object with this exact schema:",
            json.dumps(
                {
                    "ranked_predicates": ["predicateName"],
                    "selected_predicates": ["predicateName"],
                    "candidate_rules": ["target(A,B) :- body(A,C),other(C,B)."],
                    "type_constraints": {"predicateName": ["type1", "type2"]},
                    "max_vars": 6,
                    "max_body": 6,
                    "max_clauses": 4,
                    "confidence": 0.8,
                    "rationale": "short reason",
                },
                ensure_ascii=False,
            ),
            "Rules must use only the available predicate names and Prolog variables such as A,B,C,D.",
            "Do not explain your reasoning outside JSON.",
        ]
    )


def extract_json_object(text: str) -> Dict[str, object]:
    """Extract the first JSON object from a model response.

    Qwen-style models may emit short reasoning or accidental Markdown fences.
    This helper keeps the provider tolerant while still requiring valid JSON.
    """

    # 部分模型会输出 <think>...</think> 或 Markdown fenced code，
    # 这里尽量剥离外壳，只保留第一个 JSON object。
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = cleaned.replace("```json", "```").strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.strip("`").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in LLM response: {text[:200]}")
    return json.loads(cleaned[start : end + 1])


def guidance_from_dict(
    data: Dict[str, object],
    task: TaskData,
    predicate_budget: int,
    source: str,
) -> Guidance:
    """Normalize model JSON into a Guidance object."""

    # LLM 返回的数据不可信：字段可能缺失、类型可能不对、谓词名可能不存在。
    # 所以这里做最小标准化，之后再交给 Popper 做符号验证。
    ranked = [str(x) for x in data.get("ranked_predicates", [])]
    selected = [str(x) for x in data.get("selected_predicates", [])]
    if not selected:
        selected = ranked[:predicate_budget]
    selected = selected[:predicate_budget]

    # 过滤不存在的谓词，避免 LLM 幻觉污染 Popper bias。
    available = {spec.name for spec in task.predicates.values()}
    ranked = [name for name in ranked if name in available]
    selected = [name for name in selected if name in available]

    candidate_rules = []
    for text in data.get("candidate_rules", []):
        # 候选规则即使语法错也不应中断流程；跳过坏规则和当前任务不可用谓词。
        candidate_rules.extend(_parse_candidate_if_available(str(text), source=source, available=available))

    return Guidance(
        ranked_predicates=ranked,
        selected_predicates=selected,
        candidate_rules=candidate_rules,
        type_constraints={k: tuple(v) for k, v in dict(data.get("type_constraints", {})).items()},
        max_vars=int(data.get("max_vars", task.max_vars)),
        max_body=int(data.get("max_body", task.max_body)),
        max_clauses=int(data.get("max_clauses", task.max_clauses)),
        confidence=float(data.get("confidence", 0.5)),
        rationale=str(data.get("rationale", f"Loaded from {source} provider.")),
    )


def validate_guidance_schema(data: Dict[str, object]) -> None:
    """Reject malformed LLM guidance before it reaches the learner.

    llama.cpp currently gives us JSON mode, not full JSON Schema constrained
    decoding for every model. This validator is therefore the hard boundary:
    malformed guidance is a method failure in strict PARA, not a reason to
    continue with heuristic fallback.
    """

    required_lists = ("ranked_predicates", "selected_predicates", "candidate_rules")
    for key in required_lists:
        if key not in data or not isinstance(data[key], list):
            raise ValueError(f"LLM guidance JSON must contain list field `{key}`")
    if not data["ranked_predicates"]:
        raise ValueError("LLM guidance JSON has empty ranked_predicates")
    if not data["selected_predicates"]:
        raise ValueError("LLM guidance JSON has empty selected_predicates")
    if "type_constraints" in data and not isinstance(data["type_constraints"], dict):
        raise ValueError("LLM guidance JSON field `type_constraints` must be an object")
    for key in ("max_vars", "max_body", "max_clauses"):
        if key in data and not isinstance(data[key], int):
            raise ValueError(f"LLM guidance JSON field `{key}` must be an integer")
    if "confidence" in data and not isinstance(data["confidence"], (int, float)):
        raise ValueError("LLM guidance JSON field `confidence` must be a number")


def _parse_candidate_if_available(text: str, source: str, available: set[str]) -> List[Rule]:
    """Parse one candidate rule and keep it only if all body predicates exist.

    JSON/LLM guide 可能来自完整元模型提示，但当前实验任务可能已经按 D1 复杂度剪枝。
    若继续评估使用缺失谓词的候选规则，small/mid 预算下会出现“规则文本引用了
    bias.pl 没声明谓词”的 weak 结果。这里提前过滤，使候选规则和 Popper 搜索空间一致。
    """

    try:
        rule = parse_rule(text, source=source)
    except ValueError:
        return []
    if any(literal.predicate not in available for literal in rule.body):
        return []
    return [rule]


def merge_with_heuristic_guidance(
    primary: Guidance,
    heuristic: Guidance,
    predicate_budget: int,
) -> Guidance:
    """Fuse LLM guidance with deterministic architecture priors.

    This is the practical PARA guardrail: the LLM can rank and propose, but
    the system keeps enough high-value structural predicates for Popper to
    verify alternatives instead of being trapped by one incorrect LLM guess.
    """

    # primary 是 LLM 输出；heuristic 是本地先验。
    # 排序上先尊重 LLM，再补入 heuristic 遗漏的谓词。
    ranked = _stable_unique(primary.ranked_predicates + heuristic.ranked_predicates)

    # selected 会真正写入 pruned bias.pl。这里把 LLM 选择、LLM 排名前 k、
    # heuristic 选择合并后截断，避免 LLM 只选一个错误谓词造成搜索空间过窄。
    selected = _stable_unique(
        primary.selected_predicates
        + primary.ranked_predicates[:predicate_budget]
        + heuristic.selected_predicates
    )[:predicate_budget]

    # 候选规则同样合并：LLM 负责提供任务特化猜测，heuristic 提供稳健保底模板。
    candidate_rules = primary.candidate_rules + [
        rule for rule in heuristic.candidate_rules if rule not in primary.candidate_rules
    ]
    return Guidance(
        ranked_predicates=ranked,
        selected_predicates=selected,
        candidate_rules=candidate_rules,
        type_constraints=primary.type_constraints or heuristic.type_constraints,
        max_vars=max(primary.max_vars, heuristic.max_vars),
        max_body=max(primary.max_body, heuristic.max_body),
        max_clauses=max(primary.max_clauses, heuristic.max_clauses),
        confidence=primary.confidence,
        rationale=(
            f"{primary.rationale} Hybrid guardrail added heuristic predicates "
            f"and candidate rules for Popper validation."
        ),
    )


def _stable_unique(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
