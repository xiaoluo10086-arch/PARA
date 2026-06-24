"""Batch experiment design, aggregation, statistics, and blind-review packs.

该模块把论文第 5-6 章需要的实验执行框架落成可运行代码：

1. 批量实验矩阵：方法 × 任务 × 谓词复杂度 × 噪声类型 × 噪声比例 × seed；
2. 结果聚合：从各实验输出目录读取 `summary.json`，生成 CSV/JSON 数据表；
3. 统计检验：对 PARA 与基线做配对符号检验、近似 Wilcoxon、Cliff's delta；
4. 可解释性盲测：将规则解释匿名化、随机化，生成架构师评分表和解盲表。

实现只依赖 Python 标准库，便于在 `conda run -n rule_learning` 下直接运行。
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import random
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .bias import build_bias_text, write_pruned_bk
from .models import Guidance
from .noise import inject_noise
from .prolog import load_task


METHODS = ("pure_popper", "pure_llm", "nshrl")
NOISE_KINDS = ("clean", "label_flip", "missing_bk", "irrelevant_bk")

# 复杂度预算不能简单取 bias.pl 中的前 N 个 body_pred。
# 不同解析任务的谓词声明顺序可能不同，若按顺序截断，mid 档可能漏掉
# canCallClass 必需的 callsMethod/2，导致实验结论失真。
# 这里把论文实验设计中的 D1 维度显式编码为命名谓词层级。
COMPLEXITY_PREDICATE_TIERS: Dict[int, List[str]] = {
    5: [
        "package",
        "class",
        "containsClass",
        "importsClass",
        "containsPackage",
    ],
    10: [
        "package",
        "class",
        "containsClass",
        "importsClass",
        "containsPackage",
        "method",
        "containsMethod",
        "callsMethod",
        "methodName",
        "methodArity",
    ],
    15: [
        "package",
        "class",
        "containsClass",
        "importsClass",
        "containsPackage",
        "method",
        "containsMethod",
        "callsMethod",
        "methodName",
        "methodArity",
        "inheritsClass",
        "implementsInterface",
        "extendsClass",
        "sameMethodName",
        "sameMethodArity",
    ],
    # 当前 ParsingProject 的静态事实集最多导出 15 个谓词。xlarge 档仍保留
    # 20 个预算入口，缺失谓词会写入 variant_report，作为后续扩展解析器的证据。
    20: [
        "package",
        "class",
        "containsClass",
        "importsClass",
        "containsPackage",
        "method",
        "containsMethod",
        "callsMethod",
        "methodName",
        "methodArity",
        "inheritsClass",
        "implementsInterface",
        "extendsClass",
        "sameMethodName",
        "sameMethodArity",
        "field",
        "containsField",
        "accessesField",
        "annotation",
        "hasAnnotation",
    ],
}


@dataclass
class ExperimentCase:
    """One executable row in the experiment matrix."""

    case_id: str
    task_name: str
    target: str
    method: str
    predicate_count: int
    noise_kind: str
    noise_rate: float
    seed: int
    task_dir: str
    output_dir: str
    command: str


def build_experiment_plan(
    task_specs: Sequence[str],
    output_dir: str | Path,
    predicate_counts: Sequence[int],
    noise_kinds: Sequence[str],
    noise_rates: Sequence[float],
    seeds: Sequence[int],
    methods: Sequence[str],
    timeout: int = 120,
    predicate_budget: int = 5,
    llm_base_url: str = "http://127.0.0.1:8000",
    llm_model: str = "Qwen3.5-27B-MaxCtx",
    llm_timeout: int = 120,
    materialize_tasks: bool = True,
) -> Dict[str, Any]:
    """Create a full factorial experiment plan and optional task variants."""

    out = Path(output_dir)
    tasks_root = out / "tasks"
    runs_root = out / "runs"
    out.mkdir(parents=True, exist_ok=True)
    tasks_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    parsed_tasks = [parse_task_spec(spec) for spec in task_specs]
    cases: List[ExperimentCase] = []

    for task_name, base_task_dir in parsed_tasks:
        base_task = load_task(base_task_dir)
        for predicate_count in predicate_counts:
            complexity_dir = tasks_root / task_name / f"pred_{predicate_count}"
            if materialize_tasks:
                write_complexity_task(base_task_dir, complexity_dir, predicate_count)

            for noise_kind, noise_rate, seed in itertools.product(noise_kinds, noise_rates, seeds):
                # clean 只允许 rate=0，避免同一个干净任务被重复表达成 clean_10/20。
                if noise_kind == "clean" and float(noise_rate) != 0.0:
                    continue
                if noise_kind != "clean" and float(noise_rate) == 0.0:
                    continue

                task_variant_dir = complexity_dir
                if noise_kind != "clean":
                    task_variant_dir = tasks_root / task_name / f"pred_{predicate_count}" / f"{noise_kind}_{noise_rate:g}_seed_{seed}"
                    if materialize_tasks:
                        inject_noise(complexity_dir, task_variant_dir, noise_kind, noise_rate, seed=seed)

                for method in methods:
                    if method not in METHODS:
                        raise ValueError(f"Unsupported method: {method}")
                    case_id = make_case_id(task_name, method, predicate_count, noise_kind, noise_rate, seed)
                    case_output = runs_root / case_id
                    command = build_case_command(
                        method=method,
                        task_dir=task_variant_dir,
                        output_dir=case_output,
                        timeout=timeout,
                        predicate_budget=predicate_budget,
                        llm_base_url=llm_base_url,
                        llm_model=llm_model,
                        llm_timeout=llm_timeout,
                    )
                    cases.append(
                        ExperimentCase(
                            case_id=case_id,
                            task_name=task_name,
                            target=base_task.target.signature,
                            method=method,
                            predicate_count=predicate_count,
                            noise_kind=noise_kind,
                            noise_rate=float(noise_rate),
                            seed=int(seed),
                            task_dir=str(task_variant_dir),
                            output_dir=str(case_output),
                            command=command,
                        )
                    )

    payload = {
        "schema": "nshrl-experiment-plan-v1",
        "output_dir": str(out),
        "case_count": len(cases),
        "factors": {
            "tasks": [name for name, _ in parsed_tasks],
            "methods": list(methods),
            "predicate_counts": list(predicate_counts),
            "noise_kinds": list(noise_kinds),
            "noise_rates": list(noise_rates),
            "seeds": list(seeds),
        },
        "cases": [asdict(case) for case in cases],
    }
    write_json(out / "experiment_plan.json", payload)
    write_cases_csv(out / "experiment_plan.csv", cases)
    write_shell_script(out / "run_experiments.sh", cases)
    return payload


def write_complexity_task(base_task_dir: str | Path, output_dir: str | Path, predicate_count: int) -> Path:
    """Create a task variant with an explicit D1 predicate-complexity budget.

    对 5/10/15/20 四个论文预算档，按 `COMPLEXITY_PREDICATE_TIERS`
    选择语义上对应的谓词；其它数值才退回到稳定顺序截断。
    """

    task = load_task(base_task_dir)
    out = Path(output_dir)
    if out.exists():
        # 保持可重复运行：只覆盖本函数生成的任务目录，不动用户手写代码。
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    selected, requested, missing = select_predicates_for_budget(task, predicate_count)
    guidance = Guidance(
        ranked_predicates=[spec.name for spec in task.predicates.values()],
        selected_predicates=selected,
        candidate_rules=[],
        max_vars=task.max_vars,
        max_body=task.max_body,
        max_clauses=task.max_clauses,
        confidence=0.5,
        rationale=(
            f"Predicate-complexity task with {len(selected)} available predicates "
            f"from a requested budget of {predicate_count}."
        ),
    )

    write_pruned_bk(task, guidance, out / "bk.pl")
    shutil.copy2(Path(task.task_dir) / "exs.pl", out / "exs.pl")
    (out / "bias.pl").write_text(build_bias_text(task, guidance), encoding="utf-8")
    write_json(
        out / "variant_report.json",
        {
            "base_task_dir": str(base_task_dir),
            "requested_predicate_budget": int(predicate_count),
            "predicate_count": len(selected),
            "requested_predicates": requested,
            "selected_predicates": selected,
            "missing_requested_predicates": missing,
            "target": task.target.signature,
        },
    )
    return out


def select_predicates_for_budget(task: Any, predicate_count: int) -> Tuple[List[str], List[str], List[str]]:
    """Select body predicates according to the full_experiment_1 D1 design.

    返回值分别是：
    - selected: 当前任务中真实可用的谓词；
    - requested: 该复杂度档理论上希望纳入的谓词；
    - missing: 理论档位中当前解析器尚未导出的谓词。
    """

    available = {spec.name for spec in task.predicates.values()}
    if predicate_count in COMPLEXITY_PREDICATE_TIERS:
        requested = COMPLEXITY_PREDICATE_TIERS[predicate_count]
        selected = [name for name in requested if name in available]
        missing = [name for name in requested if name not in available]
        # 如果目标项目没有任何命名层级谓词，退回到至少一个可用谓词，避免空 bias。
        if not selected:
            fallback = list(task.predicates.values())[:1]
            selected = [spec.name for spec in fallback]
        return selected, requested, missing

    selected_specs = list(task.predicates.values())[: max(1, min(predicate_count, len(task.predicates)))]
    selected = [spec.name for spec in selected_specs]
    return selected, selected, []


def aggregate_results(plan_path: str | Path, output_dir: str | Path) -> Dict[str, Any]:
    """Read experiment outputs and write a flat results table."""

    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    rows = []
    for case in plan.get("cases", []):
        summary_path = Path(case["output_dir"]) / "summary.json"
        summary = {}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(flatten_result_row(case, summary, summary_path.exists()))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "results.json", {"schema": "nshrl-results-v1", "rows": rows})
    write_rows_csv(out / "results.csv", rows)
    write_json(out / "summary_by_method.json", summarize_by_method(rows))
    return {"rows": rows, "output_dir": str(out)}


def run_statistical_tests(results_path: str | Path, output_dir: str | Path) -> Dict[str, Any]:
    """Run paired statistical comparisons between PARA and baselines."""

    rows = load_result_rows(results_path)
    comparisons = []
    metrics = ("success", "f1", "elapsed_seconds", "program_f1")
    baselines = ("pure_popper", "pure_llm")
    present_methods = {str(row.get("method")) for row in rows}
    challenger = "nshrl_full" if "nshrl_full" in present_methods else "nshrl"

    for baseline in baselines:
        for metric in metrics:
            pairs = paired_metric_values(rows, baseline=baseline, challenger=challenger, metric=metric)
            if not pairs:
                continue
            diffs = [challenger - base for base, challenger in pairs]
            comparisons.append(
                {
                    "baseline": baseline,
                    "challenger": challenger,
                    "metric": metric,
                    "n_pairs": len(pairs),
                    "baseline_mean": mean([base for base, _ in pairs]),
                    "challenger_mean": mean([challenger for _, challenger in pairs]),
                    "mean_difference": mean(diffs),
                    "median_difference": median(diffs),
                    "sign_test_p": sign_test_p_value(diffs),
                    "wilcoxon_z": wilcoxon_signed_rank_z(diffs),
                    "cliffs_delta": cliffs_delta([challenger for _, challenger in pairs], [base for base, _ in pairs]),
                    "interpretation": interpret_effect(metric, diffs),
                }
            )

    payload = {"schema": "nshrl-statistical-tests-v1", "comparisons": comparisons}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "statistical_tests.json", payload)
    write_stats_report(out / "statistical_tests.md", comparisons)
    return payload


def build_blind_review_pack(
    summary_paths: Sequence[str | Path],
    output_dir: str | Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """Create anonymous interpretability-rating materials."""

    rnd = random.Random(seed)
    items = []
    for idx, summary_path in enumerate(summary_paths, start=1):
        path = _resolve_summary_path(Path(summary_path))
        if path is None:
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        explanation_path = Path(summary.get("outputs", {}).get("explanation", path.parent / "explanation.md"))
        explanation = explanation_path.read_text(encoding="utf-8") if explanation_path.exists() else ""
        graph_path = Path(summary.get("outputs", {}).get("graph_pattern", path.parent / "graph_pattern.cypher"))
        graph_pattern = graph_path.read_text(encoding="utf-8") if graph_path.exists() else ""

        items.append(
            {
                "blind_id": f"R{idx:03d}",
                "method": summary.get("method", "nshrl"),
                "target": summary.get("target", ""),
                "summary_path": str(path),
                "rule": summary.get("final_rule", ""),
                "metrics": summary.get("metrics", {}),
                "review_text": render_review_item(summary, explanation, graph_pattern),
            }
        )

    rnd.shuffle(items)
    # 重新编号，避免 blind_id 暗示输入顺序。
    for idx, item in enumerate(items, start=1):
        item["blind_id"] = f"R{idx:03d}"

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    review_rows = [
        {
            "blind_id": item["blind_id"],
            "target": item["target"],
            "clarity_1_5": "",
            "correctness_1_5": "",
            "actionability_1_5": "",
            "trust_1_5": "",
            "comment": "",
        }
        for item in items
    ]
    key_rows = [
        {
            "blind_id": item["blind_id"],
            "method": item["method"],
            "target": item["target"],
            "rule": item["rule"],
            "summary_path": item["summary_path"],
            "f1": item["metrics"].get("f1", ""),
        }
        for item in items
    ]
    write_rows_csv(out / "blind_review_form.csv", review_rows)
    write_rows_csv(out / "blind_review_key.csv", key_rows)
    (out / "blind_review_items.md").write_text(render_review_markdown(items), encoding="utf-8")
    (out / "blind_review_protocol.md").write_text(render_blind_protocol(), encoding="utf-8")
    return {
        "schema": "nshrl-blind-review-pack-v1",
        "item_count": len(items),
        "outputs": {
            "items": str(out / "blind_review_items.md"),
            "form": str(out / "blind_review_form.csv"),
            "key": str(out / "blind_review_key.csv"),
            "protocol": str(out / "blind_review_protocol.md"),
        },
    }


def flatten_result_row(case: Dict[str, Any], summary: Dict[str, Any], exists: bool) -> Dict[str, Any]:
    metrics = summary.get("metrics") or {}
    program_metrics = summary.get("program_metrics") or {}
    elapsed = extract_elapsed_seconds(summary)
    return {
        "case_id": case.get("case_id"),
        "task_name": case.get("task_name"),
        "target": case.get("target"),
        "method": case.get("method"),
        "complexity": case.get("complexity"),
        "candidate_predicates": case.get("candidate_predicates"),
        "predicate_count": case.get("predicate_count"),
        "noise_profile": case.get("noise_profile"),
        "noise_kind": case.get("noise_kind"),
        "noise_rate": case.get("noise_rate"),
        "seed": case.get("seed"),
        "status": summary.get("status", "missing") if exists else "missing",
        "success": 1 if summary.get("status") == "ok" else 0,
        "precision": metrics.get("precision", 0.0),
        "recall": metrics.get("recall", 0.0),
        "accuracy": metrics.get("accuracy", 0.0),
        "f1": metrics.get("f1", 0.0),
        "program_f1": program_metrics.get("f1", metrics.get("f1", 0.0)),
        "elapsed_seconds": elapsed,
        "final_rule": summary.get("final_rule", ""),
        "summary_path": str(Path(case.get("output_dir", "")) / "summary.json"),
    }


def extract_elapsed_seconds(summary: Dict[str, Any]) -> float:
    if "llm" in summary:
        return float(summary.get("llm", {}).get("elapsed_seconds") or 0.0)
    if "popper" in summary:
        return float(summary.get("popper", {}).get("elapsed_seconds") or 0.0)
    rounds = summary.get("rounds") or []
    total = 0.0
    for round_info in rounds:
        total += float((round_info.get("popper") or {}).get("elapsed_seconds") or 0.0)
    return total


def paired_metric_values(
    rows: Sequence[Dict[str, Any]],
    baseline: str,
    challenger: str,
    metric: str,
) -> List[Tuple[float, float]]:
    grouped: Dict[Tuple[Any, ...], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("status") == "missing":
            continue
        key = (
            row.get("task_name"),
            row.get("target"),
            row.get("complexity") if row.get("complexity") is not None else row.get("predicate_count"),
            row.get("noise_kind"),
            row.get("noise_rate"),
            row.get("seed"),
        )
        grouped[key][row.get("method")] = row

    pairs = []
    for group in grouped.values():
        if baseline not in group or challenger not in group:
            continue
        pairs.append((float(group[baseline].get(metric) or 0.0), float(group[challenger].get(metric) or 0.0)))
    return pairs


def sign_test_p_value(diffs: Sequence[float]) -> float:
    non_zero = [diff for diff in diffs if diff != 0]
    n = len(non_zero)
    if n == 0:
        return 1.0
    wins = sum(1 for diff in non_zero if diff > 0)
    k = min(wins, n - wins)
    # Two-sided exact binomial sign test under p=0.5.
    prob = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2 * prob)


def wilcoxon_signed_rank_z(diffs: Sequence[float]) -> Optional[float]:
    non_zero = [(abs(diff), 1 if diff > 0 else -1) for diff in diffs if diff != 0]
    n = len(non_zero)
    if n < 2:
        return None
    non_zero.sort(key=lambda item: item[0])
    ranks = average_ranks([value for value, _ in non_zero])
    w_plus = sum(rank for rank, (_, sign) in zip(ranks, non_zero) if sign > 0)
    expected = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24
    if variance == 0:
        return None
    return (w_plus - expected) / math.sqrt(variance)


def cliffs_delta(xs: Sequence[float], ys: Sequence[float]) -> float:
    if not xs or not ys:
        return 0.0
    greater = less = 0
    for x in xs:
        for y in ys:
            if x > y:
                greater += 1
            elif x < y:
                less += 1
    return (greater - less) / (len(xs) * len(ys))


def average_ranks(values: Sequence[float]) -> List[float]:
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(values):
        end = idx + 1
        while end < len(values) and values[end] == values[idx]:
            end += 1
        avg = (idx + 1 + end) / 2
        for pos in range(idx, end):
            ranks[pos] = avg
        idx = end
    return ranks


def summarize_by_method(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("method"))].append(row)
    summary = {}
    for method, items in groups.items():
        summary[method] = {
            "n": len(items),
            "success_rate": mean([float(item.get("success") or 0.0) for item in items]),
            "mean_f1": mean([float(item.get("f1") or 0.0) for item in items]),
            "mean_program_f1": mean([float(item.get("program_f1") or 0.0) for item in items]),
            "mean_elapsed_seconds": mean([float(item.get("elapsed_seconds") or 0.0) for item in items]),
        }
    return summary


def render_review_item(summary: Dict[str, Any], explanation: str, graph_pattern: str) -> str:
    return "\n".join(
        [
            f"Target: {summary.get('target', '-')}",
            "",
            "Rule:",
            summary.get("final_rule", "-"),
            "",
            "Graph pattern:",
            graph_pattern.strip() or "-",
            "",
            "Natural-language explanation:",
            explanation.strip() or "-",
        ]
    )


def render_review_markdown(items: Sequence[Dict[str, Any]]) -> str:
    lines = ["# PARA Blind Review Items", ""]
    for item in items:
        lines.append(f"## {item['blind_id']}")
        lines.append("")
        lines.append(item["review_text"])
        lines.append("")
    return "\n".join(lines)


def render_blind_protocol() -> str:
    return """# Blind Interpretability Review Protocol

评分对象是匿名规则说明，不展示方法名称和指标。

请每位评审者独立评分：

- clarity_1_5：规则和解释是否清楚。
- correctness_1_5：规则是否符合你对架构关系的理解。
- actionability_1_5：该规则是否能帮助架构审查或违规定位。
- trust_1_5：你是否愿意把该规则用于后续架构分析。

评分尺度：

1 = 很差，2 = 较差，3 = 一般，4 = 较好，5 = 很好。

实验者在收回 `blind_review_form.csv` 后，再用 `blind_review_key.csv` 解盲，
比较 PARA、纯 Popper 和纯 LLM 的平均可解释性评分。
"""


def write_stats_report(path: Path, comparisons: Sequence[Dict[str, Any]]) -> None:
    lines = ["# PARA Statistical Tests", ""]
    lines.append("| Baseline | Metric | N | Baseline Mean | PARA Mean | Mean Diff | Sign p | Wilcoxon z | Cliff's delta |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in comparisons:
        z = item.get("wilcoxon_z")
        lines.append(
            "| {baseline} | {metric} | {n_pairs} | {baseline_mean:.3f} | {challenger_mean:.3f} | "
            "{mean_difference:.3f} | {sign_test_p:.4f} | {z} | {cliffs_delta:.3f} |".format(
                **item,
                z="-" if z is None else f"{z:.3f}",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_case_command(
    method: str,
    task_dir: str | Path,
    output_dir: str | Path,
    timeout: int,
    predicate_budget: int,
    llm_base_url: str,
    llm_model: str,
    llm_timeout: int,
) -> str:
    base = "python -m para.cli"
    if method == "pure_popper":
        return f"{base} baseline-popper --task-dir {task_dir} --output-dir {output_dir} --timeout {timeout}"
    if method == "pure_llm":
        return (
            f"{base} baseline-llm --task-dir {task_dir} --output-dir {output_dir} "
            f"--llm-base-url {llm_base_url} --llm-model {llm_model} --llm-timeout {llm_timeout} "
            "--llm-max-tokens 512 --fact-sample 20 --example-sample 4"
        )
    return (
        f"{base} learn --task-dir {task_dir} --output-dir {output_dir} "
        f"--guide-provider modular-llama --llm-base-url {llm_base_url} --llm-model {llm_model} "
        f"--llm-timeout {llm_timeout} --predicate-budget {predicate_budget} --rounds 1 "
        f"--timeout {timeout} --candidate-first --strict-candidate-first --bk-slice-examples --bk-slice-depth 1"
    )


def parse_task_spec(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        path = Path(spec)
        return path.name, spec
    name, path = spec.split("=", 1)
    return name.strip(), path.strip()


def make_case_id(task_name: str, method: str, predicate_count: int, noise_kind: str, noise_rate: float, seed: int) -> str:
    rate = str(noise_rate).replace(".", "p")
    return f"{task_name}__{method}__pred{predicate_count}__{noise_kind}_{rate}__seed{seed}"


def write_cases_csv(path: Path, cases: Sequence[ExperimentCase]) -> None:
    write_rows_csv(path, [asdict(case) for case in cases])


def write_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_shell_script(path: Path, cases: Sequence[ExperimentCase]) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for case in cases:
        lines.append(f"echo '[PARA] running {case.case_id}'")
        lines.append(case.command)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_result_rows(path: str | Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "rows" in data:
        return list(data["rows"])
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported results file: {path}")


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def interpret_effect(metric: str, diffs: Sequence[float]) -> str:
    avg = mean(diffs)
    if metric == "elapsed_seconds":
        return "PARA is faster on average." if avg < 0 else "PARA is slower or tied on average."
    return "PARA is better on average." if avg > 0 else "PARA is worse or tied on average."


def _resolve_summary_path(path: Path) -> Optional[Path]:
    if path.is_dir():
        candidate = path / "summary.json"
        return candidate if candidate.exists() else None
    return path if path.exists() else None
