"""Translate learned rules to Neo4j Cypher graph patterns."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .models import Literal, Rule


RELATION_MAP: Dict[str, Tuple[str, str, str]] = {
    # Prolog 谓词名 -> Neo4j 起点标签、关系类型、终点标签。
    # 这张表是逻辑规则和图数据库可视化之间的桥。
    "containsPackage": ("Package", "CONTAINS_PACKAGE", "Package"),
    "containsClass": ("Package", "CONTAINS_CLASS", "Class"),
    "containsMethod": ("Class", "CONTAINS_METHOD", "Method"),
    "methodInClass": ("Method", "METHOD_IN_CLASS", "Class"),
    "importsClass": ("Class", "IMPORTS", "Class"),
    "inheritsClass": ("Class", "INHERITS", "Class"),
    "implementsInterface": ("Class", "INHERITS", "Class"),
    "callsMethod": ("Method", "CALLS", "Method"),
}

METHOD_PROPERTY_MAP: Dict[str, str] = {
    # Popper 属性谓词 -> Neo4j Method 节点属性。
    # methodName(A,C), methodName(B,C) 会被转成 A.name = B.name。
    # methodArity(A,C), methodArity(B,C) 会被转成 A.arity = B.arity。
    "methodName": "name",
    "methodArity": "arity",
}


def rule_to_cypher(rule: Rule) -> str:
    """Convert a rule body to a Cypher MATCH query where possible."""

    patterns: List[str] = []
    comments: List[str] = []
    standalone_nodes: List[str] = []
    property_bindings: Dict[Tuple[str, str], List[str]] = {}
    for lit in rule.body:
        # 目前只转换二元关系谓词；一元类型谓词如 package(A) 通常不需要变成边。
        pattern = literal_to_pattern(lit)
        if pattern:
            patterns.append(pattern)
        elif lit.arity == 2 and lit.predicate in METHOD_PROPERTY_MAP:
            method_var, value_var = lit.args
            prop = METHOD_PROPERTY_MAP[lit.predicate]
            standalone_nodes.append(f"({method_var}:Method)")
            property_bindings.setdefault((prop, value_var), []).append(method_var)
        else:
            comments.append(f"// Predicate not mapped to a graph relation: {lit.predicate}/{lit.arity}")

    where_clauses: List[str] = []
    for (prop, _value_var), method_vars in sorted(property_bindings.items()):
        unique_vars = sorted(set(method_vars))
        if len(unique_vars) < 2:
            continue
        first = unique_vars[0]
        for other in unique_vars[1:]:
            where_clauses.append(f"{first}.{prop} = {other}.{prop}")

    for node in sorted(set(standalone_nodes)):
        if not any(node in pattern for pattern in patterns):
            patterns.append(node)

    if not patterns:
        # 如果规则体全是暂未映射的谓词，保留注释，提醒后续扩展 RELATION_MAP。
        return "\n".join(comments + ["// No Cypher pattern could be generated."])

    head_args = ", ".join(rule.head.args)
    lines = comments
    lines.append("MATCH")
    lines.append("  " + ",\n  ".join(patterns))
    if where_clauses:
        lines.append("WHERE " + " AND ".join(where_clauses))

    # 返回规则头变量，便于在 Neo4j 中直接看到该规则推断出的目标关系。
    lines.append(f"RETURN DISTINCT {head_args};")
    return "\n".join(lines)


def literal_to_pattern(lit: Literal) -> str | None:
    if lit.arity != 2 or lit.predicate not in RELATION_MAP:
        return None
    left_label, rel, right_label = RELATION_MAP[lit.predicate]
    left, right = lit.args
    return f"({left}:{left_label})-[:{rel}]->({right}:{right_label})"
