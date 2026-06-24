"""End-to-end PARA learning pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from .bias import write_pruned_task
from .display import write_report
from .evaluate import choose_best_rule, evaluate_rule, weighted_rule
from .explain import explain_rule
from .graph_patterns import rule_to_cypher
from .guidance import GuideProvider, HeuristicGuideProvider
from .models import Guidance, Rule
from .popper_runner import PopperResult, run_popper
from .prolog import load_task, rule_to_text


DEFAULT_POPPER_PATH = os.getenv("PARA_POPPER_PATH", "popper.py")

# 论文级实验统一采用每个 case 900 秒硬超时。
# LLM 单次 HTTP 请求超时单独配置，不属于符号学习器 case timeout。
DEFAULT_CASE_TIMEOUT_SECONDS = 900


class NSHRLPipeline:
    """Coordinate PARA planning, symbolic verification, feedback, and outputs."""

    def __init__(
        self,
        task_dir: str,
        output_dir: str,
        popper_path: str = DEFAULT_POPPER_PATH,
        provider: Optional[GuideProvider] = None,
        predicate_budget: int = 5,
        rounds: int = 2,
        timeout: int = DEFAULT_CASE_TIMEOUT_SECONDS,
        objective: str = "",
        min_f1: float = 0.5,
        candidate_first: bool = False,
        strict_candidate_first: bool = False,
        continue_on_candidate_rejected: bool = False,
        bk_slice_examples: bool = False,
        bk_slice_depth: int = 1,
        max_facts_per_predicate: Optional[int] = None,
    ):
        self.task_dir = task_dir
        self.output_dir = Path(output_dir)
        self.popper_path = popper_path
        self.provider = provider or HeuristicGuideProvider()
        self.predicate_budget = predicate_budget
        self.rounds = rounds
        self.timeout = timeout
        self.objective = objective
        self.min_f1 = min_f1
        self.candidate_first = candidate_first
        self.strict_candidate_first = strict_candidate_first
        self.continue_on_candidate_rejected = continue_on_candidate_rejected
        self.bk_slice_examples = bk_slice_examples
        self.bk_slice_depth = bk_slice_depth
        self.max_facts_per_predicate = max_facts_per_predicate

    def run(self) -> Dict[str, object]:
        # 读取一个标准 Popper 任务目录。这里的 task 同时包含：
        # 1. bk.pl 中的背景事实；
        # 2. exs.pl 中的正负例；
        # 3. bias.pl 中的目标谓词、候选体谓词和类型约束。
        task = load_task(self.task_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # feedback 是闭环学习的核心信号：
        # - None: 第一轮，没有历史反馈；
        # - too_general: 当前规则覆盖了负例，需要收紧搜索空间；
        # - too_specific: 当前规则漏掉正例，需要放宽搜索空间；
        # - consistent: 当前规则已同时覆盖正例并排除负例。
        feedback = None
        all_rules: List[Rule] = []
        round_reports = []
        last_guidance: Optional[Guidance] = None
        final_eval_task = task

        objective = self.objective or f"Learn {task.target.name}/{task.target.arity} for software architecture rules."

        for round_idx in range(1, self.rounds + 1):
            # 引导层可以是真实 LLM，也可以是可复现实验用的本地启发式 provider。
            # 它负责输出谓词排序、被选中的谓词子集、候选规则和搜索限制。
            try:
                guidance = self.provider.guide(task, objective, self.predicate_budget, feedback=feedback)
            except Exception as exc:
                # 严格 PARA 下，LLM 引导层失败就是方法失败。
                # 不再退回 heuristic，也不继续让 Popper 在近似原空间里硬搜。
                result = {
                    "method": "nshrl",
                    "status": "guidance_failed",
                    "target": task.target.signature,
                    "error": str(exc),
                    "rounds": round_reports,
                }
                self._write_failure_summary(result)
                return result
            last_guidance = guidance
            round_dir = self.output_dir / f"round_{round_idx}"

            # 不改动 bk.pl 和 exs.pl，只重写 bias.pl。
            # 这样同一个干净事实库可以同时用于纯 ILP、纯 LLM 和 PARA 对比。
            pruned_task = write_pruned_task(
                task,
                guidance,
                round_dir / "popper_task",
                bk_slice_examples=self.bk_slice_examples,
                bk_slice_depth=self.bk_slice_depth,
                max_facts_per_predicate=self.max_facts_per_predicate,
            )
            # 当启用 BK slicing 时，候选规则验证也应使用同一个被剪枝的事实库。
            # 否则 candidate-first 虽然跳过 Popper，但 evaluator 仍要扫描 Spring
            # 的百万级 full BK，无法体现 PARA 的快速路径。
            eval_task = load_task(pruned_task) if self.bk_slice_examples or self.max_facts_per_predicate else task
            final_eval_task = eval_task

            # 候选规则和 Popper 规则统一进入本地 evaluator。
            # 这样即使 Popper 因环境问题失败，也能保留 LLM-only/候选规则基线的结果。
            candidate_rules = guidance.candidate_rules

            # 对大型方法调用图，Popper grounding 可能比验证一条候选规则慢很多。
            # candidate_first=True 时，先用符号 evaluator 检查 LLM/guide 候选规则；
            # 如果候选规则已达到质量门槛，就跳过 Popper 搜索，避免组合爆炸。
            candidate_best = choose_best_rule(candidate_rules, eval_task.facts, eval_task.examples)
            if self.candidate_first and candidate_best is not None and candidate_best[1].f1 >= self.min_f1:
                popper_result = PopperResult(
                    returncode=0,
                    elapsed_seconds=0.0,
                    stdout="SKIPPED: candidate_first accepted a verified candidate rule.",
                    stderr="",
                    rules=[],
                )
                round_rules = candidate_rules
            elif self.candidate_first and self.strict_candidate_first:
                # candidate-first 严格模式用于论文正式实验：LLM 不能给出可用候选时，
                # 不自动进入 Popper 搜索。否则 PARA 失败会被 Popper 兜底掩盖，
                # 难以判断 LLM 引导层到底是否有效。
                metrics_dict = asdict(candidate_best[1]) if candidate_best is not None else None
                best_rule_text = rule_to_text(candidate_best[0]) if candidate_best is not None else None
                feedback = candidate_best[1].feedback_label if candidate_best is not None else "no_candidate"
                popper_result = PopperResult(
                    returncode=None,
                    elapsed_seconds=0.0,
                    stdout="SKIPPED: strict_candidate_first rejected LLM candidates before Popper.",
                    stderr="",
                    rules=[],
                    error="strict_candidate_first rejected candidates; Popper was not run.",
                )
                round_reports.append(
                    {
                        "round": round_idx,
                        "selected_predicates": guidance.selected_predicates,
                        "ranked_predicates": guidance.ranked_predicates,
                        "guidance_rationale": guidance.rationale,
                        "candidate_rules": [rule_to_text(rule) for rule in candidate_rules],
                        "popper": _popper_report(popper_result),
                        "best_rule": best_rule_text,
                        "metrics": metrics_dict,
                        "feedback": feedback,
                    }
                )
                # 反馈闭环消融需要保留 strict candidate-first 的“不得回退 Popper”
                # 约束，同时允许下一轮 guide 根据 too_general/too_specific/mixed_errors
                # 修正候选谓词和规则。该开关只在还有剩余轮次时生效；最后一轮仍
                # 以 candidate_rejected 结束，避免把失败伪装成成功。
                if self.continue_on_candidate_rejected and round_idx < self.rounds:
                    continue
                result = {
                    "method": "nshrl",
                    "status": "candidate_rejected",
                    "target": task.target.signature,
                    "rounds": round_reports,
                }
                if best_rule_text is not None:
                    result["best_candidate_rule"] = best_rule_text
                    result["best_candidate_metrics"] = metrics_dict
                self._write_failure_summary(result)
                return result
            else:
                # Popper 是符号推理核心：它在剪枝后的假设空间中搜索逻辑正确的规则。
                popper_result = run_popper(self.popper_path, pruned_task, timeout=self.timeout)
                round_rules = popper_result.rules + candidate_rules

            all_rules.extend(round_rules)

            # 用当前任务的正负例计算 precision/recall/F1，并产生下一轮反馈标签。
            best = choose_best_rule(round_rules, eval_task.facts, eval_task.examples)
            if best is None:
                # 没有任何可评估规则时，默认认为搜索空间太窄。
                feedback = "too_specific"
                metrics_dict = None
                best_rule_text = None
            else:
                best_rule, metrics = best
                feedback = metrics.feedback_label
                metrics_dict = asdict(metrics)
                best_rule_text = rule_to_text(best_rule)

            round_reports.append(
                {
                    "round": round_idx,
                    "selected_predicates": guidance.selected_predicates,
                    "ranked_predicates": guidance.ranked_predicates,
                    "guidance_rationale": guidance.rationale,
                    "candidate_rules": [rule_to_text(rule) for rule in candidate_rules],
                    "popper": _popper_report(popper_result),
                    "best_rule": best_rule_text,
                    "metrics": metrics_dict,
                    "feedback": feedback,
                }
            )

            if feedback == "consistent":
                # 已经找到当前训练集上完全一致的规则，提前停止闭环迭代。
                break

        # 从所有轮次中重新选择全局最优规则，避免某一轮局部最优覆盖后续候选。
        final = choose_best_rule(all_rules, final_eval_task.facts, final_eval_task.examples)
        if final is None:
            result = {"method": "nshrl", "status": "failed", "rounds": round_reports}
            self._write_failure_summary(result)
            return result

        best_rule, metrics = final
        confidence = last_guidance.confidence if last_guidance else 0.5

        # 概率规则权重不是 Popper 原生输出，而是 PARA 的扩展：
        # 当前实现用符号验证分数为主、LLM 置信度为辅得到一个可解释权重。
        final_rule = weighted_rule(best_rule, metrics, confidence)
        status = "ok" if metrics.f1 >= self.min_f1 else "weak"

        # 三类最终产物分别服务于不同读者：
        # - learned_rules.pl 给 ILP/Prolog 工具继续使用；
        # - graph_pattern.cypher 给 Neo4j 可视化和证据检索；
        # - explanation.md 给架构师阅读和评审。
        (self.output_dir / "learned_rules.pl").write_text(rule_to_text(final_rule, with_confidence=True) + "\n", encoding="utf-8")
        (self.output_dir / "graph_pattern.cypher").write_text(rule_to_cypher(final_rule) + "\n", encoding="utf-8")
        (self.output_dir / "explanation.md").write_text(explain_rule(final_rule, metrics), encoding="utf-8")

        result = {
            "method": "nshrl",
            # weak 表示流程执行成功，但规则质量未达到实验门槛。
            # 这样展示层不会把低质量规则误报成成功结论。
            "status": status,
            "target": task.target.signature,
            "final_rule": rule_to_text(final_rule, with_confidence=True),
            "metrics": asdict(metrics),
            "rounds": round_reports,
            "outputs": {
                "learned_rules": str(self.output_dir / "learned_rules.pl"),
                "graph_pattern": str(self.output_dir / "graph_pattern.cypher"),
                "explanation": str(self.output_dir / "explanation.md"),
            },
        }
        (self.output_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_report(self.output_dir, result)
        return result

    def _write_failure_summary(self, result: Dict[str, object]) -> None:
        """Persist a failed strict PARA run in the same shape as successful runs."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_report(self.output_dir, result)


# Public name for new integrations. The historical name remains available so
# archived experiment commands continue to replay.
PARAPipeline = NSHRLPipeline


def _popper_report(result: PopperResult) -> Dict[str, object]:
    # summary.json 不保存完整 stdout/stderr，避免一次批量实验生成过大的日志。
    # 如果需要完整日志，可以在 popper_runner.py 中扩展文件落盘。
    return {
        "returncode": result.returncode,
        "elapsed_seconds": result.elapsed_seconds,
        "rules": [rule_to_text(rule) for rule in result.rules],
        "error": result.error,
        "stdout_tail": "\n".join(result.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(result.stderr.splitlines()[-20:]),
    }
