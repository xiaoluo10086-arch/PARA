"""Baselines for PARA experiments.

本模块实现两条对照路线：

1. pure Popper：不使用 LLM 谓词筛选、不使用候选规则，只运行完整 ILP 搜索；
2. pure LLM：不运行 Popper，只让语言模型直接生成规则，再用同一个符号 evaluator 评分。

这两条路线与 PARA 共用 `TaskData`、规则解析、规则评估和输出格式，避免因为
工具链差异造成不公平比较。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .display import write_report
from .evaluate import choose_best_rule, evaluate_rule_set, weighted_rule
from .explain import explain_rule
from .graph_patterns import rule_to_cypher
from .guidance import extract_json_object
from .llm_clients import chat_text
from .models import Guidance, Rule, TaskData
from .pipeline import DEFAULT_CASE_TIMEOUT_SECONDS, DEFAULT_POPPER_PATH, _popper_report
from .popper_runner import PopperResult, run_popper
from .prolog import load_task, parse_rule, rule_to_text


class OpenAICompatibleRuleGenerator:
    """Ask an OpenAI-compatible chat endpoint to generate rules directly.

    当前主要面向用户已经启动的 llama.cpp `llama-server`，但接口只依赖
    `/v1/chat/completions`，后续可以替换为 vLLM、Ollama OpenAI 网关或云端模型。
    纯 LLM 基线不会合并 heuristic，也不会调用 Popper；模型输出错了就按错的评估。
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        request_timeout: int = 120,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        fact_sample_size: int = 40,
        example_sample_size: int = 6,
    ):
        self.base_url = (
            base_url
            or os.getenv("PARA_LLM_BASE_URL")
            or os.getenv("NSHRL_LLM_BASE_URL")
            or "http://127.0.0.1:8000"
        ).rstrip("/")
        self.model = model or os.getenv("PARA_LLM_MODEL") or os.getenv("NSHRL_LLM_MODEL") or "Qwen3.5-27B-MaxCtx"
        self.request_timeout = request_timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fact_sample_size = fact_sample_size
        self.example_sample_size = example_sample_size

    def generate(self, task: TaskData, objective: str = "", max_rules: int = 3) -> Dict[str, Any]:
        """Return raw model JSON plus normalized candidate rules."""

        prompt = build_pure_llm_prompt(
            task,
            objective=objective,
            max_rules=max_rules,
            fact_sample_size=self.fact_sample_size,
            example_sample_size=self.example_sample_size,
        )
        content = self._chat_text(prompt)

        # 纯 LLM 基线保留模型原始行为：如果它不按 JSON 输出，不使用 heuristic 修正，
        # 只从模型自己的文本中抽取看起来像 Horn clause 的规则。
        try:
            raw = extract_json_object(content)
            raw_rules = raw.get("candidate_rules", [])
            confidence = _safe_float(raw.get("confidence"), default=0.5)
            rationale = str(raw.get("rationale", "Direct LLM rule generation."))
        except Exception as exc:
            raw = {
                "raw_text": content,
                "parse_warning": f"Response was not valid JSON: {exc}",
            }
            raw_rules = extract_rule_strings_from_text(content, task.target.name)
            confidence = 0.5
            rationale = "Direct LLM response was not JSON; extracted Horn clauses from the model text."

        rules = normalize_llm_rules(raw_rules, task)
        return {
            "provider": "openai_compatible",
            "base_url": self.base_url,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "fact_sample_size": self.fact_sample_size,
            "example_sample_size": self.example_sample_size,
            "raw_response": raw,
            "candidate_rules": rules,
            "confidence": confidence,
            "rationale": rationale,
        }

    def _chat_text(self, prompt: str) -> str:
        # 为了保持环境轻量，这里只使用 Python 标准库发 HTTP 请求。
        return chat_text(
            base_url=self.base_url,
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in software architecture rule learning. "
                        "Return only one valid JSON object. Do not use Markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            request_timeout=self.request_timeout,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            json_response=True,
        )


def run_pure_popper_baseline(
    task_dir: str,
    output_dir: str,
    popper_path: str = DEFAULT_POPPER_PATH,
    timeout: int = DEFAULT_CASE_TIMEOUT_SECONDS,
    min_f1: float = 0.5,
    max_vars: Optional[int] = None,
    max_body: Optional[int] = None,
    max_clauses: Optional[int] = None,
) -> Dict[str, Any]:
    """Run pure Popper with the full static metamodel bias.

    “优化”的边界在这里很清楚：可以压缩 bk.pl，只保留 bias.pl 声明过的谓词事实；
    可以设置全局 max_* 和 timeout；但不允许用 LLM 排序、剪枝或候选规则。
    """

    task = load_task(task_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    popper_task = write_pure_popper_task(task, out / "popper_task", max_vars, max_body, max_clauses)

    popper_result = run_popper(popper_path, popper_task, timeout=timeout)
    final = choose_best_rule(popper_result.rules, task.facts, task.examples)
    program_metrics = evaluate_rule_set(popper_result.rules, task.facts, task.examples) if popper_result.rules else None
    result = _finalize_result(
        method="pure_popper",
        task=task,
        output_dir=out,
        candidate_rules=popper_result.rules,
        final=final,
        confidence=0.5,
        min_f1=min_f1,
        rounds=[
            {
                "round": 1,
                "selected_predicates": [spec.name for spec in task.predicates.values()],
                "ranked_predicates": [spec.name for spec in task.predicates.values()],
                "guidance_rationale": "Pure Popper baseline: full static bias, no LLM guidance.",
                "candidate_rules": [],
                "popper": _popper_report(popper_result),
                "best_rule": rule_to_text(final[0]) if final else None,
                "metrics": asdict(final[1]) if final else None,
                "feedback": final[1].feedback_label if final else "no_rule",
            }
        ],
        extra={
            "popper_task": str(popper_task),
            "popper": _popper_report(popper_result),
            "program_metrics": asdict(program_metrics) if program_metrics else None,
        },
    )
    return result


def run_pure_llm_baseline(
    task_dir: str,
    output_dir: str,
    base_url: str | None = None,
    model: str | None = None,
    request_timeout: int = 120,
    min_f1: float = 0.5,
    objective: str = "",
    max_rules: int = 3,
    max_tokens: int = 2048,
    fact_sample_size: int = 40,
    example_sample_size: int = 6,
) -> Dict[str, Any]:
    """Run direct LLM rule generation without Popper verification/search."""

    task = load_task(task_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    generator = OpenAICompatibleRuleGenerator(
        base_url=base_url,
        model=model,
        request_timeout=request_timeout,
        max_tokens=max_tokens,
        fact_sample_size=fact_sample_size,
        example_sample_size=example_sample_size,
    )

    start = time.perf_counter()
    try:
        generated = generator.generate(task, objective=objective, max_rules=max_rules)
        elapsed = time.perf_counter() - start
        rules = generated["candidate_rules"]
        error = None
    except Exception as exc:
        # 纯 LLM 基线不做 heuristic fallback；失败本身也是实验结果。
        elapsed = time.perf_counter() - start
        generated = {
            "provider": "openai_compatible",
            "base_url": (
                base_url
                or os.getenv("PARA_LLM_BASE_URL")
                or os.getenv("NSHRL_LLM_BASE_URL")
                or "http://127.0.0.1:8000"
            ),
            "model": model or os.getenv("PARA_LLM_MODEL") or os.getenv("NSHRL_LLM_MODEL") or "Qwen3.5-27B-MaxCtx",
            "max_tokens": max_tokens,
            "fact_sample_size": fact_sample_size,
            "example_sample_size": example_sample_size,
            "raw_response": {},
            "candidate_rules": [],
            "confidence": 0.0,
            "rationale": f"LLM baseline failed: {exc}",
        }
        rules = []
        error = str(exc)

    final = choose_best_rule(rules, task.facts, task.examples)
    result = _finalize_result(
        method="pure_llm",
        task=task,
        output_dir=out,
        candidate_rules=rules,
        final=final,
        confidence=float(generated.get("confidence", 0.5)),
        min_f1=min_f1,
        rounds=[
            {
                "round": 1,
                "selected_predicates": [],
                "ranked_predicates": [],
                "guidance_rationale": generated.get("rationale", ""),
                "candidate_rules": [rule_to_text(rule) for rule in rules],
                "popper": {
                    "returncode": None,
                    "elapsed_seconds": 0.0,
                    "rules": [],
                    "error": "Pure LLM baseline does not run Popper.",
                    "stdout_tail": "",
                    "stderr_tail": "",
                },
                "best_rule": rule_to_text(final[0]) if final else None,
                "metrics": asdict(final[1]) if final else None,
                "feedback": final[1].feedback_label if final else "no_rule",
            }
        ],
        extra={
            "llm": {
                "provider": generated.get("provider"),
                "base_url": generated.get("base_url"),
                "model": generated.get("model"),
                "elapsed_seconds": elapsed,
                "error": error,
                "raw_response": generated.get("raw_response"),
                "max_tokens": generated.get("max_tokens", max_tokens),
                "fact_sample_size": generated.get("fact_sample_size", fact_sample_size),
                "example_sample_size": generated.get("example_sample_size", example_sample_size),
            }
        },
    )
    return result


def write_pure_popper_task(
    task: TaskData,
    output_dir: str | Path,
    max_vars: Optional[int] = None,
    max_body: Optional[int] = None,
    max_clauses: Optional[int] = None,
) -> Path:
    """Create a Popper task for the pure ILP baseline.

    该目录保留完整 body_pred 集合；bk.pl 只删除 bias 中没有声明的事实谓词。
    对 Popper 来说这仍然是完整搜索空间，只是避免无关事实拖慢 grounding。
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(task.task_dir)

    selected = [spec.name for spec in task.predicates.values()]
    guidance = Guidance(
        ranked_predicates=selected,
        selected_predicates=selected,
        candidate_rules=[],
        max_vars=max_vars if max_vars is not None else task.max_vars,
        max_body=max_body if max_body is not None else task.max_body,
        max_clauses=max_clauses if max_clauses is not None else task.max_clauses,
        confidence=0.5,
        rationale="Pure Popper full-bias task.",
    )

    # 复用 bias 生成器，确保类型声明和 max_* 格式与 PARA 一致。
    from .bias import build_bias_text, write_pruned_bk

    write_pruned_bk(task, guidance, out / "bk.pl")
    shutil.copy2(src / "exs.pl", out / "exs.pl")
    (out / "bias.pl").write_text(build_bias_text(task, guidance), encoding="utf-8")
    return out


def build_pure_llm_prompt(
    task: TaskData,
    objective: str = "",
    max_rules: int = 3,
    fact_sample_size: int = 40,
    example_sample_size: int = 6,
) -> str:
    """Build a direct-rule prompt for the pure LLM baseline."""

    predicate_lines = []
    for spec in task.predicates.values():
        types = f"({', '.join(spec.types)})" if spec.types else "(unknown)"
        predicate_lines.append(f"- {spec.name}/{spec.arity}: {types}")

    pos_examples = [ex.literal for ex in task.examples if ex.positive][:example_sample_size]
    neg_examples = [ex.literal for ex in task.examples if not ex.positive][:example_sample_size]
    fact_sample = task.facts[:fact_sample_size]

    return "\n".join(
        [
            "/no_think",
            "Return only JSON. Do not call external tools.",
            "You are the pure LLM baseline. You must directly infer rules from examples.",
            "Do not mention Popper, ILP search, or verification.",
            "",
            f"Objective: {objective or f'Learn {task.target.name}/{task.target.arity} from static software facts.'}",
            f"Target predicate: {task.target.name}/{task.target.arity} with types {task.target.types}",
            f"Return at most {max_rules} candidate Horn rules.",
            "",
            "Available body predicates:",
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
            "Return this JSON schema exactly:",
            json.dumps(
                {
                    "candidate_rules": [
                        f"{task.target.name}(A,B) :- somePredicate(A,C),otherPredicate(C,B)."
                    ],
                    "confidence": 0.7,
                    "rationale": "short reason in Chinese or English",
                },
                ensure_ascii=False,
            ),
            "Rules must use only the target predicate in the head and only available body predicates.",
            "Use Prolog variables such as A,B,C,D,M1,M2. End each rule with a period.",
        ]
    )


def normalize_llm_rules(raw_rules: Iterable[Any], task: TaskData) -> List[Rule]:
    """Parse and filter direct LLM rules without correcting their semantics."""

    available = {spec.name for spec in task.predicates.values()}
    rules: List[Rule] = []
    for raw in raw_rules:
        try:
            rule = parse_rule(str(raw), source="pure_llm")
        except ValueError:
            continue
        if rule.head.predicate != task.target.name or rule.head.arity != task.target.arity:
            continue
        if any(lit.predicate not in available for lit in rule.body):
            continue
        rules.append(rule)
    return rules


def extract_rule_strings_from_text(text: str, target_name: str) -> List[str]:
    """Extract Horn clauses from a non-JSON LLM response."""

    # 这里只抽取模型已经写出来的规则，不进行语义补全。
    # 例如自然语言段落中的 `isAllowedToUse(A,B) :- ... .` 会被保留。
    pattern = re.compile(
        rf"{re.escape(target_name)}\s*\([^)]*\)\s*:-\s*[^.\n]+\.",
        flags=re.MULTILINE,
    )
    return [match.group(0).strip() for match in pattern.finditer(text)]


def _finalize_result(
    method: str,
    task: TaskData,
    output_dir: Path,
    candidate_rules: List[Rule],
    final: Optional[tuple],
    confidence: float,
    min_f1: float,
    rounds: List[Dict[str, Any]],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    """Write common baseline artifacts and return a summary dictionary."""

    if final is None:
        result: Dict[str, Any] = {
            "method": method,
            "status": "failed",
            "target": task.target.signature,
            "candidate_rules": [rule_to_text(rule) for rule in candidate_rules],
            "rounds": rounds,
            **extra,
        }
        (output_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_report(output_dir, result)
        return result

    best_rule, metrics = final
    final_rule = weighted_rule(best_rule, metrics, confidence)
    status = "ok" if metrics.f1 >= min_f1 else "weak"

    (output_dir / "learned_rules.pl").write_text(rule_to_text(final_rule, with_confidence=True) + "\n", encoding="utf-8")
    (output_dir / "graph_pattern.cypher").write_text(rule_to_cypher(final_rule) + "\n", encoding="utf-8")
    (output_dir / "explanation.md").write_text(explain_rule(final_rule, metrics), encoding="utf-8")

    result = {
        "method": method,
        "status": status,
        "target": task.target.signature,
        "final_rule": rule_to_text(final_rule, with_confidence=True),
        "metrics": asdict(metrics),
        "rounds": rounds,
        "outputs": {
            "learned_rules": str(output_dir / "learned_rules.pl"),
            "graph_pattern": str(output_dir / "graph_pattern.cypher"),
            "explanation": str(output_dir / "explanation.md"),
        },
        **extra,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(output_dir, result)
    return result


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
