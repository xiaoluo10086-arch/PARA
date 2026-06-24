"""Human-readable presentation helpers for PARA results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def compact_result_text(result: Dict[str, Any]) -> str:
    """Return a concise terminal summary for a learning run.

    The full `summary.json` is still written to disk by the pipeline.  This
    function only controls what the user sees in the terminal, so long Popper
    stdout/stderr blocks do not bury the actual result.
    """

    status = result.get("status", "unknown")
    method = result.get("method", "nshrl")
    lines = [f"Method: {method}", f"Status: {status}"]

    # failed 表示没有任何可用规则；weak 表示有规则但质量低于 --min-f1。
    # 两者都需要在终端上明显提示，避免用户把失败实验误读为成功结论。
    if status == "failed":
        lines.append("No final rule was produced. See summary.json for round details.")
        return "\n".join(lines)
    if status == "weak":
        lines.append("Warning: a rule was produced, but it did not meet the configured quality threshold.")

    metrics = result.get("metrics", {})
    outputs = result.get("outputs", {})

    # 终端只展示最重要的信息；完整 Popper 日志、候选规则和轮次细节在 summary.json/report.md。
    lines.extend(
        [
            f"Target: {result.get('target', '-')}",
            f"Final rule: {result.get('final_rule', '-')}",
            (
                "Metrics: "
                f"precision={metrics.get('precision', 0):.3f}, "
                f"recall={metrics.get('recall', 0):.3f}, "
                f"accuracy={metrics.get('accuracy', 0):.3f}, "
                f"f1={metrics.get('f1', 0):.3f}"
            ),
            f"Rounds: {len(result.get('rounds', []))}",
        ]
    )
    program_metrics = result.get("program_metrics")
    if program_metrics:
        # 纯 Popper 可能输出多条 clause 作为一个程序；这里额外展示规则集整体指标。
        lines.append(
            "Program metrics: "
            f"precision={program_metrics.get('precision', 0):.3f}, "
            f"recall={program_metrics.get('recall', 0):.3f}, "
            f"f1={program_metrics.get('f1', 0):.3f}"
        )
    for label, path in outputs.items():
        lines.append(f"{label}: {path}")
    return "\n".join(lines)


def task_inspection_text(payload: Dict[str, Any]) -> str:
    """Return a compact inspection view for a Popper task directory."""

    # inspect 是实验前的体检：目标谓词、可用谓词、事实规模、正负例数量都要一眼可见。
    predicates = ", ".join(payload.get("predicates", []))
    limits = payload.get("limits", {})
    return "\n".join(
        [
            f"Task: {payload.get('task_dir')}",
            f"Target: {payload.get('target')}",
            f"Predicates: {predicates}",
            f"Facts: {payload.get('fact_count')}",
            (
                "Examples: "
                f"pos={payload.get('positive_examples')}, "
                f"neg={payload.get('negative_examples')}"
            ),
            (
                "Limits: "
                f"max_vars={limits.get('max_vars')}, "
                f"max_body={limits.get('max_body')}, "
                f"max_clauses={limits.get('max_clauses')}"
            ),
        ]
    )


def write_report(output_dir: str | Path, result: Dict[str, Any]) -> Path:
    """Write a Markdown report that is easier to read than raw JSON."""

    out = Path(output_dir)
    report_path = out / "report.md"
    method = result.get("method", "nshrl")
    lines: List[str] = [f"# {method} Learning Report", ""]

    # Summary 部分对应论文实验表中的一行：最终规则和核心指标。
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Method: `{method}`")
    lines.append(f"- Status: `{result.get('status', 'unknown')}`")
    lines.append(f"- Target: `{result.get('target', '-')}`")
    lines.append(f"- Final rule: `{result.get('final_rule', '-')}`")

    metrics = result.get("metrics") or {}
    if metrics:
        # TP/FP/TN/FN 用于判断规则具体错在哪里，后续也可用于统计检验脚本。
        lines.extend(
            [
                f"- Precision: `{metrics.get('precision', 0):.3f}`",
                f"- Recall: `{metrics.get('recall', 0):.3f}`",
                f"- Accuracy: `{metrics.get('accuracy', 0):.3f}`",
                f"- F1: `{metrics.get('f1', 0):.3f}`",
                f"- TP/FP/TN/FN: `{metrics.get('tp', 0)}/{metrics.get('fp', 0)}/{metrics.get('tn', 0)}/{metrics.get('fn', 0)}`",
            ]
        )

    program_metrics = result.get("program_metrics") or {}
    if program_metrics:
        lines.extend(
            [
                f"- Program precision: `{program_metrics.get('precision', 0):.3f}`",
                f"- Program recall: `{program_metrics.get('recall', 0):.3f}`",
                f"- Program F1: `{program_metrics.get('f1', 0):.3f}`",
                (
                    "- Program TP/FP/TN/FN: "
                    f"`{program_metrics.get('tp', 0)}/{program_metrics.get('fp', 0)}/"
                    f"{program_metrics.get('tn', 0)}/{program_metrics.get('fn', 0)}`"
                ),
            ]
        )

    lines.extend(["", "## Rounds", ""])
    for round_info in result.get("rounds", []):
        # Rounds 部分记录每一轮 LLM/heuristic 选择、Popper 状态和反馈标签，
        # 是闭环学习可解释性的主要证据。
        lines.append(f"### Round {round_info.get('round')}")
        lines.append("")
        lines.append(f"- Selected predicates: `{', '.join(round_info.get('selected_predicates', []))}`")
        if round_info.get("guidance_rationale"):
            # rationale 可能来自真实 LLM，也可能来自 fallback；保留它方便追踪模型行为。
            lines.append(f"- Guidance rationale: {round_info.get('guidance_rationale')}")
        lines.append(f"- Feedback: `{round_info.get('feedback')}`")
        lines.append(f"- Best rule: `{round_info.get('best_rule')}`")
        popper = round_info.get("popper") or {}
        lines.append(f"- Popper return code: `{popper.get('returncode')}`")
        lines.append(f"- Popper elapsed seconds: `{popper.get('elapsed_seconds', 0):.3f}`")
        if popper.get("error"):
            lines.append(f"- Popper error: `{popper.get('error')}`")
        lines.append("")

    lines.extend(["", "## Output Files", ""])
    # 输出文件路径集中列出，用户可以直接打开规则、图模式和自然语言解释。
    for label, path in (result.get("outputs") or {}).items():
        lines.append(f"- {label}: `{path}`")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
