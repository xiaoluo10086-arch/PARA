"""Small Prolog parser/writer for Popper task files.

This module deliberately supports the subset used by this project: facts,
examples, `head_pred`, `body_pred`, `type`, max settings, and simple Horn
clauses.  It is not intended to be a complete Prolog parser.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .models import Example, Literal, PredicateSpec, Rule, TaskData, predicate_key


_PRED_DECL_RE = re.compile(r"^(head_pred|body_pred)\(([^,]+),\s*(\d+)\)\.$")
_TYPE_RE = re.compile(r"^type\(([^,]+),\s*\((.*)\)\)\.$")
_MAX_RE = re.compile(r"^max_(vars|body|clauses)\((\d+)\)\.$")


def strip_comment(line: str) -> str:
    """Remove comments while respecting simple quoted atoms."""

    # Prolog 用 `%` 开始行内注释，但事实中可能有被单引号包裹的字符串。
    # 因此不能简单 split("%")，需要跟踪是否位于 quote 内。
    in_quote = False
    escaped = False
    out = []
    for ch in line:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\" and in_quote:
            out.append(ch)
            escaped = True
            continue
        if ch == "'":
            in_quote = not in_quote
            out.append(ch)
            continue
        if ch == "%" and not in_quote:
            break
        out.append(ch)
    return "".join(out).strip()


def split_top_level(text: str, sep: str = ",") -> List[str]:
    """Split at top-level separators, ignoring parentheses and quotes."""

    # 规则体常见形式：p(A,B),q(B,C),r(f(C),D)。
    # 只能按顶层逗号切分，不能切开括号内部或引号内部的内容。
    parts: List[str] = []
    start = 0
    depth = 0
    in_quote = False
    escaped = False
    for idx, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_quote:
            escaped = True
            continue
        if ch == "'":
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def parse_literal(text: str) -> Literal:
    """Parse `pred(a,b)` into a Literal."""

    # Literal 是系统内部最小逻辑单元：
    # - 背景事实：containsClass(pkg,cls)
    # - 规则头：isAllowedToUse(A,B)
    # - 规则体：importsClass(C,D)
    text = text.strip().rstrip(".")
    if "(" not in text or not text.endswith(")"):
        raise ValueError(f"Not a literal: {text}")
    name, raw_args = text.split("(", 1)
    raw_args = raw_args[:-1]
    args = tuple(unquote_atom(arg.strip()) for arg in split_top_level(raw_args))
    return Literal(name.strip(), args)


def parse_fact_line(line: str) -> Optional[Literal]:
    """Parse a background fact line; skip directives and rules."""

    # bk.pl 中可能含有占位规则 `predicate(_,_) :- fail.`
    # 这类规则用于声明谓词存在，但不是可用于覆盖计算的地面事实。
    line = strip_comment(line)
    if not line or ":-" in line:
        return None
    if line.startswith(("pos(", "neg(", "head_pred(", "body_pred(", "type(", "max_")):
        return None
    if not line.endswith("."):
        return None
    return parse_literal(line)


def parse_example_line(line: str) -> Optional[Example]:
    """Parse Popper examples: pos(target(...)). or neg(target(...))."""

    # Popper 的训练样例用 pos/neg 包裹目标谓词。
    # 这里保留正负标签，后续 evaluator 会用它计算 TP/FP/TN/FN。
    line = strip_comment(line)
    if not line:
        return None
    positive = line.startswith("pos(")
    negative = line.startswith("neg(")
    if not positive and not negative:
        return None
    if not line.endswith(")."):
        raise ValueError(f"Invalid example line: {line}")
    inner = line[4:-2] if positive else line[4:-2]
    return Example(parse_literal(inner), positive)


def parse_rule(text: str, source: str = "popper") -> Rule:
    """Parse a simple Horn clause."""

    text = strip_comment(text).rstrip(".").strip()
    confidence = None

    # PARA 扩展允许规则末尾带权重，例如：
    # isAllowedToUse(A,B) :- ... [0.930].
    # Popper 原生规则没有该权重。
    weight_match = re.search(r"\[(0(?:\.\d+)?|1(?:\.0+)?)\]\s*$", text)
    if weight_match:
        confidence = float(weight_match.group(1))
        text = text[: weight_match.start()].strip()

    if ":-" in text:
        head_text, body_text = text.split(":-", 1)
        body = tuple(parse_literal(part) for part in split_top_level(body_text))
    else:
        head_text = text
        body = tuple()
    return Rule(parse_literal(head_text), body, confidence=confidence, source=source)


def parse_bias(lines: Iterable[str]) -> Tuple[PredicateSpec, Dict[str, PredicateSpec], Dict[str, int]]:
    """Read target/body predicate declarations and limits from bias lines."""

    # bias.pl 决定 Popper 能搜索什么：
    # - head_pred: 学习目标；
    # - body_pred: 允许出现在规则体中的谓词；
    # - type: 类型约束；
    # - max_*: 搜索空间大小限制。
    target: Optional[PredicateSpec] = None
    predicates: Dict[str, PredicateSpec] = {}
    type_decls: Dict[str, Tuple[str, ...]] = {}
    limits = {"max_vars": 6, "max_body": 6, "max_clauses": 4}

    for raw in lines:
        line = strip_comment(raw)
        if not line:
            continue
        pred_match = _PRED_DECL_RE.match(line)
        if pred_match:
            kind, name, arity_text = pred_match.groups()
            spec = PredicateSpec(name.strip(), int(arity_text))
            if kind == "head_pred":
                # 当前任务只支持一个 head predicate，这是 Popper 常见任务形态。
                target = spec
            else:
                predicates[predicate_key(spec.name, spec.arity)] = spec
            continue

        type_match = _TYPE_RE.match(line)
        if type_match:
            name, raw_types = type_match.groups()
            # 一元类型在 Popper 中写作 `(class,)`，split_top_level 会自然去掉空尾项。
            type_decls[name.strip()] = tuple(t.strip() for t in split_top_level(raw_types) if t.strip())
            continue

        max_match = _MAX_RE.match(line)
        if max_match:
            key, value = max_match.groups()
            limits[f"max_{key}"] = int(value)

    if target is None:
        raise ValueError("bias.pl does not contain head_pred/2")

    if target.name in type_decls:
        # 将 head_pred 与对应 type 合并，方便后续剪枝和解释使用。
        target = PredicateSpec(target.name, target.arity, type_decls[target.name])

    with_types: Dict[str, PredicateSpec] = {}
    for key, spec in predicates.items():
        with_types[key] = PredicateSpec(spec.name, spec.arity, type_decls.get(spec.name, ()))

    return target, with_types, limits


def load_task(task_dir: str | Path) -> TaskData:
    """Load bk.pl, exs.pl and bias.pl from a Popper task directory."""

    # 一个 PARA 输入目录必须首先是合法 Popper 任务目录。
    task_path = Path(task_dir)
    bk_path = task_path / "bk.pl"
    exs_path = task_path / "exs.pl"
    bias_path = task_path / "bias.pl"
    missing = [str(path) for path in (bk_path, exs_path, bias_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Popper task files: {missing}")

    bias_lines = bias_path.read_text(encoding="utf-8").splitlines()
    target, predicates, limits = parse_bias(bias_lines)

    # 只把地面事实读入 evaluator；规则、指令和注释都跳过。
    facts = [
        fact
        for line in bk_path.read_text(encoding="utf-8").splitlines()
        for fact in [parse_fact_line(line)]
        if fact is not None
    ]
    examples = [
        ex
        for line in exs_path.read_text(encoding="utf-8").splitlines()
        for ex in [parse_example_line(line)]
        if ex is not None
    ]
    return TaskData(
        task_dir=str(task_path),
        target=target,
        predicates=predicates,
        facts=facts,
        examples=examples,
        bias_lines=bias_lines,
        max_vars=limits["max_vars"],
        max_body=limits["max_body"],
        max_clauses=limits["max_clauses"],
    )


def quote_atom(value: str) -> str:
    """Write constants in a Popper-safe form."""

    # 简单小写 atom 可以裸写；包含点号、连字符、大写等字符时必须单引号包裹。
    if re.match(r"^[a-z][a-zA-Z0-9_]*$", value):
        return value
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def unquote_atom(value: str) -> str:
    if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        return value[1:-1].replace("\\'", "'").replace("\\\\", "\\")
    return value


def literal_to_text(literal: Literal, quote_constants: bool = False) -> str:
    args = []
    for arg in literal.args:
        # 输出训练样例时需要 quote 常量；输出规则时变量应保持 A/B/C 这种形式。
        if quote_constants and not (arg[:1].isupper() or arg.startswith("_")):
            args.append(quote_atom(arg))
        else:
            args.append(arg)
    return f"{literal.predicate}({','.join(args)})"


def rule_to_text(rule: Rule, with_confidence: bool = False) -> str:
    head = literal_to_text(rule.head)
    if rule.body:
        body = ",".join(literal_to_text(lit) for lit in rule.body)
        text = f"{head} :- {body}"
    else:
        text = head
    if with_confidence and rule.confidence is not None:
        text += f" [{rule.confidence:.3f}]"
    return text + "."


def write_examples(path: Path, examples: Iterable[Example]) -> None:
    # 噪声注入或数据切分后，可用该函数重新写出 Popper exs.pl。
    lines = []
    for ex in examples:
        wrapper = "pos" if ex.positive else "neg"
        lines.append(f"{wrapper}({literal_to_text(ex.literal, quote_constants=True)}).")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
