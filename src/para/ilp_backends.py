"""Pure ILP backend baselines beyond Popper.

These adapters are intentionally conservative: they translate an existing
Popper-style task directory into the input format required by an external ILP
tool, run that tool, parse any learned Horn clauses, and score the clauses with
the same PARA evaluator used for Popper and LLM baselines.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .baselines import _finalize_result
from .evaluate import choose_best_rule, evaluate_rule_set
from .models import Literal, PredicateSpec, Rule, TaskData
from .pipeline import DEFAULT_CASE_TIMEOUT_SECONDS
from .prolog import load_task, parse_rule, quote_atom, rule_to_text


DEFAULT_ALEPH_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "ML-Based-Software-Architecture-Rule-Learning"
    / "experiments"
    / "experiment 1"
    / "Aleph"
    / "2both_all_facts"
    / "aleph.pl"
)
DEFAULT_ILASP_BIN = Path(__file__).resolve().parents[1] / "tools" / "ILASP"
DEFAULT_METAGOL_SOURCE = Path(__file__).resolve().parents[1] / "tools" / "metagol.pl"


def run_pure_aleph_baseline(
    task_dir: str,
    output_dir: str,
    aleph_source: str | Path = DEFAULT_ALEPH_SOURCE,
    timeout: int = DEFAULT_CASE_TIMEOUT_SECONDS,
    min_f1: float = 0.5,
) -> Dict[str, Any]:
    """Run Aleph as a pure ILP baseline."""

    task = load_task(task_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "aleph_task"
    write_aleph_task(task, work, aleph_source)
    result = _run_command(
        ["swipl", "-q", "-f", "run_aleph.pl"],
        cwd=work,
        timeout=timeout,
    )
    rules = extract_rules(result.stdout, task.target.name, source="aleph")
    return _finish_ilp_result("pure_aleph", task, out, rules, result, min_f1, {"task_dir": str(work)})


def run_pure_ilasp_baseline(
    task_dir: str,
    output_dir: str,
    ilasp_bin: str | Path = DEFAULT_ILASP_BIN,
    ilasp_version: str = "4",
    timeout: int = DEFAULT_CASE_TIMEOUT_SECONDS,
    min_f1: float = 0.5,
    max_body_literals: Optional[int] = None,
) -> Dict[str, Any]:
    """Run ILASP as a pure ILP baseline."""

    task = load_task(task_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    las_path = out / "ilasp_task.las"
    write_ilasp_task(task, las_path, max_body_literals=max_body_literals)
    result = _run_command(
        [str(Path(ilasp_bin).resolve()), f"--version={ilasp_version}", las_path.name],
        cwd=out,
        timeout=timeout,
    )
    rules = extract_rules(result.stdout, task.target.name, source="ilasp")
    return _finish_ilp_result(
        "pure_ilasp",
        task,
        out,
        rules,
        result,
        min_f1,
        {"task_file": str(las_path), "ilasp_version": ilasp_version},
    )


def run_pure_metagol_baseline(
    task_dir: str,
    output_dir: str,
    metagol_source: str | Path = DEFAULT_METAGOL_SOURCE,
    timeout: int = DEFAULT_CASE_TIMEOUT_SECONDS,
    min_f1: float = 0.5,
    max_clauses: Optional[int] = None,
) -> Dict[str, Any]:
    """Run Metagol as a pure MIL baseline."""

    task = load_task(task_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "metagol_task"
    write_metagol_task(task, work, metagol_source, timeout=timeout, max_clauses=max_clauses)
    result = _run_command(
        ["swipl", "-q", "-f", "run_metagol.pl"],
        cwd=work,
        timeout=timeout,
    )
    rules = extract_rules(result.stdout, task.target.name, source="metagol")
    return _finish_ilp_result("pure_metagol", task, out, rules, result, min_f1, {"task_dir": str(work)})


def write_aleph_task(task: TaskData, output_dir: str | Path, aleph_source: str | Path) -> Path:
    """Translate a Popper task into Aleph `.b`, `.f`, `.n` files."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(aleph_source, out / "aleph.pl")
    stem = "task"
    positives = [ex.literal for ex in task.examples if ex.positive]
    negatives = [ex.literal for ex in task.examples if not ex.positive]
    (out / f"{stem}.f").write_text("\n".join(format_fact(ex) for ex in positives) + "\n", encoding="utf-8")
    (out / f"{stem}.n").write_text("\n".join(format_fact(ex) for ex in negatives) + "\n", encoding="utf-8")
    (out / f"{stem}.b").write_text(build_aleph_background(task), encoding="utf-8")
    (out / "run_aleph.pl").write_text(
        "\n".join(
            [
                ":- initialization(main).",
                "main :-",
                "    consult(aleph),",
                "    set(i, 4),",
                "    set(nodes, 5000),",
                "    read_all(task),",
                "    induce,",
                "    halt.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return out


def build_aleph_background(task: TaskData) -> str:
    lines = [
        ":- style_check(-discontiguous).",
        f":- modeh(*,{mode_literal(task.target, head=True)}).",
    ]
    for spec in task.predicates.values():
        lines.append(f":- modeb(*,{mode_literal(spec)}).")
    for spec in task.predicates.values():
        lines.append(f":- determination({task.target.signature},{spec.signature}).")
    lines.append("")
    lines.extend(format_fact(fact) for fact in task.facts)
    return "\n".join(lines) + "\n"


def write_ilasp_task(
    task: TaskData,
    output_path: str | Path,
    max_body_literals: Optional[int] = None,
) -> Path:
    """Translate a Popper task into a compact ILASP learning task."""

    max_body = max_body_literals if max_body_literals is not None else task.max_body
    lines = [f"#maxv({task.max_vars}).", f"#max_penalty({max_body})."]
    lines.append(f"#modeh({ilasp_mode_literal(task.target)}).")
    for spec in task.predicates.values():
        lines.append(f"#modeb({ilasp_mode_literal(spec)}).")
    lines.append("")
    for ex in task.examples:
        literal = format_fact(ex.literal).rstrip(".")
        kind = "#pos" if ex.positive else "#neg"
        lines.append(f"{kind}({{{literal}}},{{}}).")
    lines.append("")
    lines.extend(format_fact(fact) for fact in task.facts)
    path = Path(output_path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_metagol_task(
    task: TaskData,
    output_dir: str | Path,
    metagol_source: str | Path,
    timeout: int,
    max_clauses: Optional[int] = None,
) -> Path:
    """Translate a Popper task into a Metagol program."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metagol_source, out / "metagol.pl")
    positives = ",\n        ".join(format_fact(ex.literal).rstrip(".") for ex in task.examples if ex.positive)
    negatives = ",\n        ".join(format_fact(ex.literal).rstrip(".") for ex in task.examples if not ex.positive)
    max_clause_text = max_clauses if max_clauses is not None else max(1, task.max_clauses)
    lines = [
        ":- use_module('./metagol').",
        ":- style_check(-discontiguous).",
        "",
        f":- metagol:set_option(max_clauses({max_clause_text})).",
        "% Metagol itself has no portable timeout directive; the Python wrapper",
        f"% kills this process after {timeout} seconds.",
        "",
    ]
    for spec in task.predicates.values():
        lines.append(f"body_pred({spec.signature}).")
    lines.extend(
        [
            "",
            "metarule([P,Q],[P,A,B],[[Q,A,B]]).",
            "metarule([P,Q],[P,A,B],[[Q,B,A]]).",
            "metarule([P,Q,R],[P,A,B],[[Q,A],[R,A,B]]).",
            "metarule([P,Q,R],[P,A,B],[[Q,A,B],[R,B]]).",
            "metarule([P,Q,R],[P,A,B],[[Q,A,C],[R,C,B]]).",
            "metarule([P,Q,R,S],[P,A,B],[[Q,A,C],[R,C,D],[S,B,D]]).",
            "",
        ]
    )
    lines.extend(format_fact(fact) for fact in task.facts)
    lines.extend(
        [
            "",
            ":- initialization(main).",
            "main :-",
            f"    Pos = [{positives}],",
            f"    Neg = [{negatives}],",
            "    learn(Pos,Neg),",
            "    halt.",
        ]
    )
    (out / "run_metagol.pl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def mode_literal(spec: PredicateSpec, head: bool = False) -> str:
    args = []
    for index in range(spec.arity):
        type_name = spec.types[index] if index < len(spec.types) else "term"
        direction = "+" if head and index == 0 else "-"
        args.append(f"{direction}{type_name}")
    return f"{spec.name}({','.join(args)})"


def ilasp_mode_literal(spec: PredicateSpec) -> str:
    args = []
    for index in range(spec.arity):
        type_name = spec.types[index] if index < len(spec.types) else "term"
        args.append(f"var({type_name})")
    return f"{spec.name}({','.join(args)})"


def format_fact(lit: Literal) -> str:
    return f"{lit.predicate}({','.join(quote_atom(arg) for arg in lit.args)})."


def extract_rules(text: str, target_name: str, source: str) -> List[Rule]:
    """Extract simple Horn clauses or ground target atoms from tool stdout."""

    if source == "aleph" and "[theory]" in text:
        text = text.split("[theory]", 1)[1].split("[Training set performance]", 1)[0]

    rules: List[Rule] = []
    pattern = re.compile(
        rf"{re.escape(target_name)}\s*\([^)]*\)\s*(?::-\s*[^.\n]+)?\.",
        flags=re.MULTILINE,
    )
    seen = set()
    for match in pattern.finditer(text):
        raw = match.group(0).strip()
        if source == "ilasp":
            # ILASP prints ASP-style bodies with semicolons in some versions.
            # Internally we evaluate Horn clauses with comma-separated body
            # literals, so normalize only after extraction from ILASP stdout.
            raw = raw.replace(";", ",")
        if raw in seen:
            continue
        seen.add(raw)
        try:
            rules.append(parse_rule(raw, source=source))
        except ValueError:
            continue
    return rules


def _finish_ilp_result(
    method: str,
    task: TaskData,
    output_dir: Path,
    rules: List[Rule],
    command: "CommandResult",
    min_f1: float,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    final = choose_best_rule(rules, task.facts, task.examples)
    program_metrics = evaluate_rule_set(rules, task.facts, task.examples) if rules else None
    result = _finalize_result(
        method=method,
        task=task,
        output_dir=output_dir,
        candidate_rules=rules,
        final=final,
        confidence=0.5,
        min_f1=min_f1,
        rounds=[
            {
                "round": 1,
                "selected_predicates": [spec.name for spec in task.predicates.values()],
                "ranked_predicates": [spec.name for spec in task.predicates.values()],
                "guidance_rationale": f"Pure {method.removeprefix('pure_')} baseline: no LLM guidance.",
                "candidate_rules": [rule_to_text(rule) for rule in rules],
                "best_rule": rule_to_text(final[0]) if final else None,
                "metrics": asdict(final[1]) if final else None,
                "feedback": final[1].feedback_label if final else "no_rule",
            }
        ],
        extra={
            method.replace("pure_", ""): {
                "returncode": command.returncode,
                "elapsed_seconds": command.elapsed_seconds,
                "error": command.error,
                "stdout_tail": command.stdout[-4000:],
                "stderr_tail": command.stderr[-4000:],
                "program_metrics": asdict(program_metrics) if program_metrics else None,
                **extra,
            }
        },
    )
    return result


class CommandResult:
    def __init__(self, returncode: int, elapsed_seconds: float, stdout: str, stderr: str, error: str | None = None):
        self.returncode = returncode
        self.elapsed_seconds = elapsed_seconds
        self.stdout = stdout
        self.stderr = stderr
        self.error = error


def _run_command(cmd: List[str], cwd: Path, timeout: int) -> CommandResult:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return CommandResult(proc.returncode, time.perf_counter() - start, proc.stdout, proc.stderr)
    except FileNotFoundError as exc:
        return CommandResult(127, time.perf_counter() - start, "", "", str(exc))
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return CommandResult(124, time.perf_counter() - start, stdout, stderr, f"Timed out after {timeout}s")
