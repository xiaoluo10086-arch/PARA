"""Incremental learning prototype for static architecture facts.

核心思想：

1. 每条已学习规则都记录它依赖的 body predicate 集合；
2. Git 变更文件通过静态 Java 解析近似成“影响谓词集”；
3. 只有当影响谓词集与规则依赖有交集时，才建议对相关模块子图重新学习。

当前实现是原型级别，刻意不处理运行时动态行为；所有判断都来自源码文本和
Popper/PARA 已生成的规则产物。
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .models import Rule
from .prolog import parse_rule, rule_to_text


# 当 Java 文件被删除、重命名或无法读取时，静态影响分析必须保守。
# 这些谓词覆盖当前解析器支持的主要结构关系。
CONSERVATIVE_JAVA_IMPACT = {
    "package",
    "class",
    "method",
    "containsPackage",
    "containsClass",
    "containsMethod",
    "importsClass",
    "extendsClass",
    "implementsInterface",
    "inheritsClass",
    "callsMethod",
    "methodName",
    "methodArity",
    "sameMethodName",
    "sameMethodArity",
}


@dataclass
class RegisteredRule:
    """A learned rule plus the predicates it depends on."""

    rule_id: str
    method: str
    target: str
    rule_text: str
    dependencies: List[str]
    source_summary: str
    metrics: Dict[str, Any]


@dataclass
class FileImpact:
    """Static impact summary for one changed source file."""

    path: str
    exists: bool
    language: str
    packages: List[str]
    predicates: List[str]
    reason: str


def build_rule_registry(summary_paths: Sequence[str | Path], output_path: str | Path) -> Dict[str, Any]:
    """Build a JSON rule-dependency registry from summary files or directories."""

    rules: List[RegisteredRule] = []
    for path in summary_paths:
        summary_path = _resolve_summary_path(Path(path))
        if summary_path is None:
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        final_rule = data.get("final_rule")
        if not final_rule:
            continue
        try:
            rule = parse_rule(final_rule, source=str(data.get("method", "unknown")))
        except ValueError:
            continue
        method = str(data.get("method") or _infer_method_from_summary(summary_path))
        dependencies = sorted(rule_dependencies(rule))
        rule_id = f"{method}::{data.get('target', rule.head.predicate)}::{len(rules) + 1}"
        rules.append(
            RegisteredRule(
                rule_id=rule_id,
                method=method,
                target=str(data.get("target", rule.head.predicate)),
                rule_text=rule_to_text(rule, with_confidence=rule.confidence is not None),
                dependencies=dependencies,
                source_summary=str(summary_path),
                metrics=dict(data.get("metrics") or {}),
            )
        )

    payload = {
        "schema": "nshrl-rule-registry-v1",
        "description": "Rules and body-predicate dependencies for incremental PARA.",
        "rules": [asdict(rule) for rule in rules],
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def analyze_incremental_change(
    registry_path: str | Path,
    changed_files: Sequence[str | Path],
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Compare changed-file impacts with rule dependencies."""

    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    impacts = [analyze_file_impact(Path(path)) for path in changed_files]
    impacted_predicates = sorted({pred for impact in impacts for pred in impact.predicates})
    impacted_packages = sorted({pkg for impact in impacts for pkg in impact.packages})

    reuse_rules = []
    relearn_rules = []
    impacted_set = set(impacted_predicates)
    for rule in registry.get("rules", []):
        deps = set(rule.get("dependencies", []))
        overlap = sorted(deps.intersection(impacted_set))
        item = {
            **rule,
            "overlap": overlap,
            "decision": "relearn" if overlap else "reuse",
        }
        if overlap:
            relearn_rules.append(item)
        else:
            reuse_rules.append(item)

    result = {
        "schema": "nshrl-incremental-analysis-v1",
        "static_only": True,
        "changed_files": [str(path) for path in changed_files],
        "file_impacts": [asdict(impact) for impact in impacts],
        "impacted_predicates": impacted_predicates,
        "impacted_packages": impacted_packages,
        "reuse_rules": reuse_rules,
        "relearn_rules": relearn_rules,
        "recommendation": build_incremental_recommendation(relearn_rules, impacted_packages),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "incremental_analysis.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (out / "incremental_report.md").write_text(render_incremental_report(result), encoding="utf-8")
    return result


def changed_files_from_git(base: str = "HEAD~1", head: str = "HEAD", cwd: str | Path = ".") -> List[str]:
    """Return files changed between two Git revisions."""

    proc = subprocess.run(
        ["git", "diff", "--name-only", base, head],
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git diff failed with code {proc.returncode}")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def rule_dependencies(rule: Rule) -> Set[str]:
    """Return body predicate names used by a learned rule."""

    return {literal.predicate for literal in rule.body}


def analyze_file_impact(path: Path) -> FileImpact:
    """Map one changed file to likely affected architecture predicates."""

    language = _language_from_suffix(path)
    if language != "java":
        # 当前解析器以 Java 静态结构为主；非 Java 文件只影响实验配置或文档。
        return FileImpact(
            path=str(path),
            exists=path.exists(),
            language=language,
            packages=[],
            predicates=[],
            reason="Non-Java file; no architecture fact predicate is affected by the current static parser.",
        )

    if not path.exists():
        return FileImpact(
            path=str(path),
            exists=False,
            language="java",
            packages=[],
            predicates=sorted(CONSERVATIVE_JAVA_IMPACT),
            reason="Java file is deleted or not readable; conservatively mark all static Java predicates as impacted.",
        )

    text = path.read_text(encoding="utf-8", errors="ignore")
    packages = extract_java_packages(text)
    predicates = infer_java_predicate_impacts(text)
    reason = "Static Java text analysis over package/import/type/method/inheritance/call syntax."
    return FileImpact(
        path=str(path),
        exists=True,
        language="java",
        packages=packages,
        predicates=sorted(predicates),
        reason=reason,
    )


def extract_java_packages(text: str) -> List[str]:
    """Extract declared package and all parent package names."""

    match = re.search(r"^\s*package\s+([A-Za-z_][\w.]*)\s*;", text, flags=re.MULTILINE)
    if not match:
        return []
    package = match.group(1)
    parts = package.split(".")
    packages = [".".join(parts[:idx]) for idx in range(1, len(parts) + 1)]
    return packages


def infer_java_predicate_impacts(text: str) -> Set[str]:
    """Infer impacted fact predicates from Java source text.

    这里是轻量静态解析，不执行代码，也不依赖运行时调用信息。
    后续如果引入 tree-sitter-java，可以把本函数替换为 AST 版本，输出接口不变。
    """

    impacts: Set[str] = set()

    if re.search(r"^\s*package\s+[A-Za-z_][\w.]*\s*;", text, flags=re.MULTILINE):
        impacts.update({"package", "containsPackage"})

    if re.search(r"^\s*import\s+(?:static\s+)?[A-Za-z_][\w.*]*\s*;", text, flags=re.MULTILINE):
        impacts.add("importsClass")

    if re.search(r"\b(class|interface|enum|record)\s+[A-Za-z_]\w*", text):
        impacts.update({"class", "containsClass"})

    if re.search(r"\bextends\s+[A-Za-z_][\w.<>]*", text):
        impacts.update({"extendsClass", "inheritsClass", "overridesMethod"})

    if re.search(r"\bimplements\s+[A-Za-z_][\w.<>,\s]*", text):
        impacts.update({"implementsInterface", "inheritsClass", "overridesMethod"})

    if re.search(_JAVA_METHOD_DECL_RE, text, flags=re.MULTILINE):
        impacts.update(
            {
                "method",
                "containsMethod",
                "methodName",
                "methodArity",
                "sameMethodName",
                "sameMethodArity",
                "overridesMethod",
            }
        )

    if re.search(_JAVA_CALL_RE, text):
        # 调用表达式会影响方法调用边，也会影响以类间调用为目标的派生规则。
        impacts.update({"callsMethod", "canCallClass"})

    return impacts


def build_incremental_recommendation(relearn_rules: List[Dict[str, Any]], packages: List[str]) -> Dict[str, Any]:
    """Summarize the next action for an incremental run."""

    if not relearn_rules:
        return {
            "action": "reuse_all",
            "message": "No learned rule dependency intersects the impacted predicates; reuse all registered rules.",
            "suggested_scope": [],
        }

    return {
        "action": "partial_relearn",
        "message": "Relearn only rules whose body predicates intersect the static impact set.",
        "affected_rule_ids": [rule["rule_id"] for rule in relearn_rules],
        "suggested_scope": packages,
        "note": (
            "Use the affected packages to export a smaller Neo4j/Popper subtask, then run PARA "
            "only for the affected target rules."
        ),
    }


def render_incremental_report(result: Dict[str, Any]) -> str:
    """Render a compact Markdown report for architecture review."""

    lines = ["# PARA Incremental Analysis", ""]
    lines.append(f"- Static only: `{result.get('static_only')}`")
    lines.append(f"- Changed files: `{len(result.get('changed_files', []))}`")
    lines.append(f"- Impacted predicates: `{', '.join(result.get('impacted_predicates', [])) or 'none'}`")
    lines.append(f"- Impacted packages: `{', '.join(result.get('impacted_packages', [])) or 'none'}`")
    lines.append("")

    recommendation = result.get("recommendation", {})
    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"- Action: `{recommendation.get('action')}`")
    lines.append(f"- Message: {recommendation.get('message')}")
    if recommendation.get("suggested_scope"):
        lines.append(f"- Suggested package scope: `{', '.join(recommendation.get('suggested_scope', []))}`")
    lines.append("")

    lines.append("## Rules")
    lines.append("")
    for rule in result.get("relearn_rules", []):
        lines.append(f"- Relearn `{rule.get('rule_id')}` because of `{', '.join(rule.get('overlap', []))}`")
    for rule in result.get("reuse_rules", []):
        lines.append(f"- Reuse `{rule.get('rule_id')}`")
    lines.append("")
    return "\n".join(lines)


def _resolve_summary_path(path: Path) -> Optional[Path]:
    if path.is_dir():
        candidate = path / "summary.json"
        return candidate if candidate.exists() else None
    return path if path.exists() else None


def _infer_method_from_summary(path: Path) -> str:
    """Infer a readable method label for older summaries without `method`."""

    parent = path.parent.name.lower()
    if "baseline_popper" in parent or "pure_popper" in parent:
        return "pure_popper"
    if "baseline_llm" in parent or "pure_llm" in parent:
        return "pure_llm"
    return "nshrl"


def _language_from_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".java":
        return "java"
    if suffix == ".py":
        return "python"
    return suffix.lstrip(".") or "unknown"


_JAVA_METHOD_DECL_RE = (
    r"^\s*(?:public|protected|private|static|final|synchronized|abstract|native|\s)+"
    r"(?:<[^>]+>\s*)?(?:[A-Za-z_][\w.<>\[\],?]+\s+)+[A-Za-z_]\w*\s*\([^;{}]*\)\s*(?:throws\s+[^{;]+)?[;{]"
)

_JAVA_CALL_RE = r"(?<!\bif)(?<!\bfor)(?<!\bwhile)(?<!\bswitch)\b[A-Za-z_]\w*\s*\("
