"""Natural-language explanations for architecture rules."""

from __future__ import annotations

from .models import Rule, RuleMetrics
from .prolog import rule_to_text


PREDICATE_DESCRIPTIONS = {
    "containsPackage": "一个包包含另一个子包",
    "containsClass": "一个包包含某个类",
    "containsMethod": "一个类声明或包含某个方法",
    "importsClass": "一个类导入另一个类",
    "inheritsClass": "一个类继承另一个类",
    "implementsInterface": "一个类实现某个接口",
    "callsMethod": "一个方法调用另一个方法",
}


def explain_rule(rule: Rule, metrics: RuleMetrics | None = None) -> str:
    """Generate multi-level Chinese explanation text."""

    lines = []
    lines.append("## 规则解释")
    lines.append("")
    lines.append("### 逻辑规则")
    lines.append("")
    lines.append("```prolog")
    lines.append(rule_to_text(rule, with_confidence=True))
    lines.append("```")
    lines.append("")
    lines.append("### 概要级解释")
    lines.append("")
    lines.append(summary(rule))
    lines.append("")
    lines.append("### 详细级解释")
    lines.append("")
    for lit in rule.body:
        desc = PREDICATE_DESCRIPTIONS.get(lit.predicate, f"谓词 {lit.predicate}/{lit.arity}")
        lines.append(f"- `{lit.predicate}({', '.join(lit.args)})`: {desc}。")
    lines.append("")
    lines.append("### 示例级解释模板")
    lines.append("")
    lines.append(
        "若规则头中的变量被一个具体正例替换，则系统会在背景事实中寻找规则体里所有关系是否同时成立；"
        "如果全部成立，该正例被规则覆盖。"
    )
    if metrics is not None:
        lines.append("")
        lines.append("### 统计解释")
        lines.append("")
        lines.append(
            f"该规则在当前样例上的 precision={metrics.precision:.3f}, "
            f"recall={metrics.recall:.3f}, accuracy={metrics.accuracy:.3f}, "
            f"F1={metrics.f1:.3f}。反馈标签为 `{metrics.feedback_label}`。"
        )
    return "\n".join(lines) + "\n"


def summary(rule: Rule) -> str:
    if rule.head.predicate == "isAllowedToUse":
        if {"containsClass", "importsClass"}.issubset({lit.predicate for lit in rule.body}):
            return (
                "如果包 A 中的某个类依赖或导入了包 B 中的某个类，"
                "则可以归纳出包 A 被允许使用包 B。"
            )
    if "callsMethod" in {lit.predicate for lit in rule.body}:
        return "该规则根据方法调用关系推断类或包之间的允许调用关系。"
    if "inheritsClass" in {lit.predicate for lit in rule.body}:
        return "该规则根据继承结构推断架构约束。"
    return f"该规则使用 {len(rule.body)} 个背景谓词推断 `{rule.head.predicate}/{rule.head.arity}`。"
