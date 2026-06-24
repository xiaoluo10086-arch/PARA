"""Evaluate learned rules against Popper examples."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from .models import Example, Literal, Rule, RuleMetrics, is_variable

@dataclass
class FactIndex:
    """Indexes ground facts by predicate and by bound argument positions."""

    by_predicate: Dict[Tuple[str, int], List[Literal]]
    by_argument: Dict[Tuple[str, int, int, str], List[Literal]]


Binding = Dict[str, str]


def build_fact_index(facts: Iterable[Literal]) -> FactIndex:
    # 按 predicate/arity 和参数位置建索引，避免每个规则体 literal 都扫描全部背景事实。
    by_predicate: Dict[Tuple[str, int], List[Literal]] = defaultdict(list)
    by_argument: Dict[Tuple[str, int, int, str], List[Literal]] = defaultdict(list)
    for fact in facts:
        key = (fact.predicate, fact.arity)
        by_predicate[key].append(fact)
        for idx, arg in enumerate(fact.args):
            by_argument[(fact.predicate, fact.arity, idx, arg)].append(fact)
    return FactIndex(by_predicate=dict(by_predicate), by_argument=dict(by_argument))


def covers(rule: Rule, facts: Iterable[Literal], example: Example, fact_index: Optional[FactIndex] = None) -> bool:
    """Return True if a rule covers an example.

    The evaluator performs conjunctive query matching over ground background
    facts. This is sufficient for the architecture rules used in this project.
    """

    if rule.head.predicate != example.literal.predicate:
        return False
    if rule.head.arity != example.literal.arity:
        return False

    binding: Binding = {}
    if not unify_literal(rule.head, example.literal, binding):
        return False

    # bindings 保存“当前已经满足的规则体前缀”对应的变量绑定。
    # 每处理一个 body literal，就把绑定集合向前扩展一层。
    bindings = [binding]
    if fact_index is None:
        fact_index = build_fact_index(facts)
    for lit in rule.body:
        next_bindings: List[Binding] = []
        for current in bindings:
            candidates = indexed_candidates(lit, fact_index, current)
            for fact in matching_candidates(lit, candidates, current):
                new_binding = dict(current)
                # 如果当前 literal 可以和某个背景事实统一，就得到一个新的可行绑定。
                if unify_literal(lit, fact, new_binding):
                    next_bindings.append(new_binding)
        bindings = next_bindings
        if not bindings:
            # 任意一个 body literal 无法满足，整条合取规则就不覆盖该样例。
            return False
    return bool(bindings)


def indexed_candidates(lit: Literal, fact_index: FactIndex, binding: Binding) -> List[Literal]:
    """Return a small candidate fact list using bound variables/constants.

    对 Spring 这种百万级 BK，最关键的优化是避免扫描整个 containsClass、
    containsMethod 或 importsClass 谓词事实表。这里从当前绑定和常量参数中
    找到最小的参数位置索引，作为本轮 literal 的候选集合。
    """

    all_candidates = fact_index.by_predicate.get((lit.predicate, lit.arity), [])
    candidate_lists: List[List[Literal]] = []
    for idx, arg in enumerate(lit.args):
        value = binding.get(arg) if is_variable(arg) else arg
        if value is None:
            continue
        candidate_lists.append(fact_index.by_argument.get((lit.predicate, lit.arity, idx, value), []))
    if not candidate_lists:
        return all_candidates
    return min(candidate_lists, key=len)


def matching_candidates(lit: Literal, candidates: List[Literal], binding: Binding) -> Iterable[Literal]:
    """Yield facts that do not conflict with already bound variables.

    这是一个轻量剪枝：例如 head 已经绑定 A=Class1，而 body literal 是
    containsMethod(A,M)，那么只需要检查第一列等于 Class1 的 containsMethod
    facts。没有这一步，方法级规则会在 callsMethod/containsMethod 上产生很大
    的中间笛卡尔积。
    """

    for fact in candidates:
        ok = True
        for pattern_arg, fact_arg in zip(lit.args, fact.args):
            if is_variable(pattern_arg):
                bound = binding.get(pattern_arg)
                if bound is not None and bound != fact_arg:
                    ok = False
                    break
            elif pattern_arg != fact_arg:
                ok = False
                break
        if ok:
            yield fact


def unify_literal(pattern: Literal, ground: Literal, binding: Binding) -> bool:
    # pattern 可以包含变量，ground 应该是背景事实或具体样例。
    # 例如 containsClass(A,C) 与 containsClass(pkg,cls) 统一后：
    # A -> pkg, C -> cls。
    if pattern.predicate != ground.predicate or pattern.arity != ground.arity:
        return False
    for left, right in zip(pattern.args, ground.args):
        if is_variable(left):
            existing = binding.get(left)
            if existing is None:
                binding[left] = right
            elif existing != right:
                # 同一个变量在规则中多次出现时，必须绑定到同一个常量。
                return False
        elif left != right:
            return False
    return True


def evaluate_rule(
    rule: Rule,
    facts: Iterable[Literal],
    examples: Iterable[Example],
    fact_index: Optional[FactIndex] = None,
) -> RuleMetrics:
    # 这里的四格表直接对应实验指标：
    # TP: 正例被覆盖；FP: 负例被错误覆盖；
    # TN: 负例未覆盖；FN: 正例漏覆盖。
    tp = fp = tn = fn = 0
    facts_list = list(facts)
    if fact_index is None:
        fact_index = build_fact_index(facts_list)
    for ex in examples:
        predicted = covers(rule, facts_list, ex, fact_index=fact_index)
        if ex.positive and predicted:
            tp += 1
        elif ex.positive and not predicted:
            fn += 1
        elif not ex.positive and predicted:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if tp + fp + tn + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return RuleMetrics(tp=tp, fp=fp, tn=tn, fn=fn, precision=precision, recall=recall, accuracy=accuracy, f1=f1)


def evaluate_rule_set(rules: Iterable[Rule], facts: Iterable[Literal], examples: Iterable[Example]) -> RuleMetrics:
    """Evaluate a multi-clause program with disjunctive coverage.

    Popper 有时输出多条同一目标谓词的 clause，它们共同构成一个程序。
    对一个样例，只要任意一条规则覆盖，就认为整个规则集覆盖该样例。
    """

    rules_list = list(rules)
    facts_list = list(facts)
    fact_index = build_fact_index(facts_list)
    tp = fp = tn = fn = 0
    for ex in examples:
        predicted = any(covers(rule, facts_list, ex, fact_index=fact_index) for rule in rules_list)
        if ex.positive and predicted:
            tp += 1
        elif ex.positive and not predicted:
            fn += 1
        elif not ex.positive and predicted:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if tp + fp + tn + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return RuleMetrics(tp=tp, fp=fp, tn=tn, fn=fn, precision=precision, recall=recall, accuracy=accuracy, f1=f1)


def choose_best_rule(
    rules: Iterable[Rule],
    facts: Iterable[Literal],
    examples: Iterable[Example],
) -> Optional[Tuple[Rule, RuleMetrics]]:
    scored = []
    facts_list = list(facts)
    examples_list = list(examples)
    fact_index = build_fact_index(facts_list)
    for rule in rules:
        metrics = evaluate_rule(rule, facts_list, examples_list, fact_index=fact_index)
        # 排序优先级：
        # 1. F1 兼顾 precision 和 recall；
        # 2. accuracy 反映整体分类正确率；
        # 3. precision 降低过泛化风险；
        # 4. -len(body) 偏好更短、更可解释的规则。
        scored.append((metrics.f1, metrics.accuracy, metrics.precision, -len(rule.body), rule, metrics))
    if not scored:
        return None
    scored.sort(key=lambda item: item[:4], reverse=True)
    return scored[0][4], scored[0][5]

def weighted_rule(rule: Rule, metrics: RuleMetrics, llm_confidence: float = 0.5) -> Rule:
    """Attach a probability-like confidence weight to a rule."""

    # The symbolic score dominates. LLM confidence only breaks ties and reflects
    # how strongly the guidance layer supported the candidate.
    weight = 0.8 * metrics.f1 + 0.2 * llm_confidence
    return Rule(rule.head, rule.body, confidence=max(0.0, min(1.0, weight)), source=rule.source)
