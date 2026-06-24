"""Shared data structures for PARA.

The classes in this file are intentionally small and serializable.  The
learning pipeline passes these objects between the parser, the LLM-guidance
layer, Popper execution, evaluation, and output generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PredicateSpec:
    """A predicate declaration from a Popper bias file."""

    # 谓词名，例如 containsClass。
    name: str

    # 谓词元数，例如 containsClass/2 的 arity 为 2。
    arity: int

    # Popper type 声明中的参数类型，例如 ("package", "class")。
    types: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def signature(self) -> str:
        return f"{self.name}/{self.arity}"


@dataclass(frozen=True)
class Literal:
    """A Prolog literal such as containsClass(A,C)."""

    # 谓词名。
    predicate: str

    # 参数列表。可以是变量 A/B/C，也可以是具体常量。
    args: Tuple[str, ...]

    @property
    def arity(self) -> int:
        return len(self.args)

    def variables(self) -> List[str]:
        # 返回 literal 中出现的 Prolog 变量，供规则分析和解释使用。
        return [arg for arg in self.args if is_variable(arg)]


@dataclass(frozen=True)
class Rule:
    """A Horn clause with an optional confidence weight."""

    # 规则头，例如 isAllowedToUse(A,B)。
    head: Literal

    # 规则体，例如 containsClass(A,C), importsClass(C,D)。
    body: Tuple[Literal, ...]

    # PARA 扩展权重，不是 Popper 原生字段。
    confidence: Optional[float] = None

    # 标记规则来源：popper、heuristic_llm、llama_cpp、json_llm 等。
    source: str = "unknown"


@dataclass(frozen=True)
class Example:
    """A positive or negative example for the target predicate."""

    # 目标谓词的具体实例。
    literal: Literal

    # True 表示 pos(...)，False 表示 neg(...)。
    positive: bool


@dataclass
class TaskData:
    """All information loaded from a Popper task directory."""

    # 原始 Popper 任务目录。
    task_dir: str

    # head_pred 对应的学习目标。
    target: PredicateSpec

    # body_pred 候选谓词，key 形如 containsClass/2。
    predicates: Dict[str, PredicateSpec]

    # bk.pl 中的地面事实。
    facts: List[Literal]

    # exs.pl 中的正负例。
    examples: List[Example]

    # 原始 bias.pl 行，必要时可用于调试或保持兼容。
    bias_lines: List[str]

    # Popper 搜索限制，后续可由 LLM/反馈机制调节。
    max_vars: int = 6
    max_body: int = 6
    max_clauses: int = 4


@dataclass
class Guidance:
    """LLM or heuristic guidance used to prune ILP search."""

    # LLM/启发式给出的完整谓词重要性排序。
    ranked_predicates: List[str]

    # 实际写入 pruned bias.pl 的谓词子集。
    selected_predicates: List[str]

    # LLM/启发式生成的候选规则，会和 Popper 学到的规则一起评估。
    candidate_rules: List[Rule]

    # 可选类型约束，主要用于记录 LLM 输出；当前 bias 写入以 TaskData 类型为准。
    type_constraints: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    # 搜索空间控制参数。
    max_vars: int = 6
    max_body: int = 6
    max_clauses: int = 4
    confidence: float = 0.5
    rationale: str = ""


@dataclass
class RuleMetrics:
    """Coverage metrics computed on a task's examples."""

    # 四格表：正例覆盖、负例误覆盖、负例正确排除、正例漏覆盖。
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    accuracy: float
    f1: float

    @property
    def feedback_label(self) -> str:
        """Classify a rule for closed-loop refinement."""

        # 误覆盖负例但没有漏掉正例：规则太宽，需要收紧。
        if self.fp > 0 and self.fn == 0:
            return "too_general"

        # 漏掉正例但没有误覆盖负例：规则太窄，需要放宽。
        if self.fn > 0 and self.fp == 0:
            return "too_specific"

        # 两类错误都有，说明当前搜索空间或样例本身可能存在冲突。
        if self.fp > 0 and self.fn > 0:
            return "mixed_errors"
        return "consistent"


def is_variable(token: str) -> bool:
    """Return True for Prolog-style variables.

    Popper outputs variables as `A`, `B`, `V1`, etc. Quoted atoms and lower-case
    identifiers are constants.
    """

    if not token:
        return False
    return token[0].isupper() or token[0] == "_"


def predicate_key(name: str, arity: int) -> str:
    # 统一字典 key，避免同名不同元数谓词冲突。
    return f"{name}/{arity}"


def ensure_sequence(value: Sequence[str] | str) -> Tuple[str, ...]:
    # 小工具：把单个字符串或字符串序列规范化为 tuple。
    if isinstance(value, str):
        return (value,)
    return tuple(value)
