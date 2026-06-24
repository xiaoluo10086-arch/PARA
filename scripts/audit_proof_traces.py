#!/usr/bin/env python3
"""Audit PARA reasoning proof traces.

Checks:

- every EDB fact in proof traces exists in the audited task's BK;
- every counterfactual removed fact comes from the initial proof trace;
- no-rule reasoning files do not contain SUPPORTED decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Set


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CODE_ROOT = ROOT / "src"


def main() -> None:
    args = build_parser().parse_args()
    sys.path.insert(0, str(Path(args.code_root).resolve()))
    task_dir = Path(args.task_dir)
    fact_texts = load_bk_fact_texts(task_dir / "bk.pl")
    result_files = [Path(item) for item in args.result]
    for item in args.result_dir:
        result_files.extend(sorted(Path(item).glob(args.result_glob)))
    output = {
        "task_dir": str(task_dir),
        "bk_fact_count": len(fact_texts),
        "files": [],
        "total_errors": 0,
        "total_warnings": 0,
    }
    for path in result_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        audit = audit_file(path, payload, fact_texts)
        output["files"].append(audit)
        output["total_errors"] += int(audit["error_count"])
        output["total_warnings"] += int(audit["warning_count"])
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False) if args.json else compact(output))
    raise SystemExit(1 if output["total_errors"] else 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit reasoning proof traces")
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--result", action="append", default=[])
    parser.add_argument("--result-dir", action="append", default=[])
    parser.add_argument("--result-glob", default="proof_*.json")
    parser.add_argument("--output", default="")
    parser.add_argument("--code-root", default=str(DEFAULT_CODE_ROOT))
    parser.add_argument("--json", action="store_true")
    return parser


def audit_file(path: Path, payload: dict, facts: Set[str]) -> dict:
    errors: List[str] = []
    warnings: List[str] = []
    proof_count = 0
    edb_count = 0
    missing_count = 0
    idb_count = 0
    invalid_idb_count = 0
    inconclusive_audited = 0

    if "evidence" in payload and "decision" in payload:
        for evidence_idx, evidence in enumerate(payload.get("evidence") or []):
            proof_count += 1
            trace_stats = audit_trace(
                evidence.get("proof_trace") or {},
                facts,
                errors,
                prefix=f"evidence {evidence_idx}",
            )
            edb_count += trace_stats["edb_fact_mentions"]
            missing_count += trace_stats["missing_fact_mentions"]
            idb_count += trace_stats["idb_applications"]
            invalid_idb_count += trace_stats["invalid_idb_applications"]
        if payload.get("decision") == "INCONCLUSIVE":
            inconclusive_audited += 1
            audit_inconclusive(payload, errors, warnings, prefix="query")

    if "rows" in payload and "initial_supported" not in payload:
        for idx, row in enumerate(payload.get("rows") or []):
            if "no_rule" in path.name and row.get("decision") == "SUPPORTED":
                errors.append(f"row {idx}: no-rule result has SUPPORTED decision")
            for evidence in row.get("evidence") or []:
                proof_count += 1
                trace_stats = audit_trace(
                    evidence.get("proof_trace") or {},
                    facts,
                    errors,
                    prefix=f"row {idx}",
                )
                edb_count += trace_stats["edb_fact_mentions"]
                missing_count += trace_stats["missing_fact_mentions"]
                idb_count += trace_stats["idb_applications"]
                invalid_idb_count += trace_stats["invalid_idb_applications"]
            if row.get("decision") == "INCONCLUSIVE":
                inconclusive_audited += 1
                audit_inconclusive(row, errors, warnings, prefix=f"row {idx}")

    if "initial" in payload or "ablations" in payload:
        proof_facts = set()
        for evidence in (payload.get("initial") or {}).get("evidence") or []:
            proof_count += 1
            for fact in collect_edb_facts(evidence.get("proof_trace") or {}):
                proof_facts.add(normalize_fact(fact))
                edb_count += 1
                if normalize_fact(fact) not in facts:
                    missing_count += 1
                    errors.append(f"initial proof fact not in BK: {fact}")
        for item in payload.get("ablations") or []:
            removed = normalize_fact(str(item.get("removed_fact") or ""))
            if removed and removed not in proof_facts:
                errors.append(f"removed fact was not in initial proof trace: {item.get('removed_fact')}")

    if "rows" in payload and "initial_supported" in payload:
        for row_idx, row in enumerate(payload.get("rows") or []):
            # Batch counterfactual files store summaries only. Full proof-trace
            # validation is performed on single-query files; batch files are
            # still useful for no-supported sanity and metric consistency.
            if row.get("initial_decision") == "SUPPORTED" and int(row.get("ablated_facts") or 0) == 0:
                warnings.append(f"row {row_idx}: supported query has zero ablated facts")

    return {
        "file": str(path),
        "proof_count": proof_count,
        "edb_fact_mentions": edb_count,
        "missing_fact_mentions": missing_count,
        "idb_applications": idb_count,
        "invalid_idb_applications": invalid_idb_count,
        "inconclusive_decisions_audited": inconclusive_audited,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors[:50],
        "warnings": warnings[:50],
    }


def load_bk_fact_texts(path: Path) -> Set[str]:
    facts: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%") or ":-" in stripped:
            continue
        if stripped.endswith("."):
            facts.add(normalize_fact(stripped[:-1]))
    return facts


def collect_edb_facts(trace: dict) -> List[str]:
    facts: List[str] = []
    if trace.get("kind") == "edb_fact" and trace.get("fact"):
        facts.append(str(trace["fact"]))
    for child in trace.get("subproofs") or []:
        if isinstance(child, dict):
            facts.extend(collect_edb_facts(child))
    return facts


def audit_trace(trace: dict, facts: Set[str], errors: List[str], prefix: str) -> dict:
    from para.prolog import parse_literal, parse_rule
    from para.reasoner import apply_binding, unify_general

    stats = {
        "edb_fact_mentions": 0,
        "missing_fact_mentions": 0,
        "idb_applications": 0,
        "invalid_idb_applications": 0,
    }
    kind = trace.get("kind")
    if kind == "edb_fact":
        fact = str(trace.get("fact") or "")
        goal = str(trace.get("goal") or "")
        stats["edb_fact_mentions"] += 1
        if normalize_fact(fact) not in facts:
            stats["missing_fact_mentions"] += 1
            errors.append(f"{prefix}: proof fact not in BK: {fact}")
        if fact and goal and normalize_fact(fact) != normalize_fact(goal):
            errors.append(f"{prefix}: EDB goal/fact mismatch: goal={goal} fact={fact}")
        return stats

    if kind != "derived_rule":
        errors.append(f"{prefix}: unknown or missing proof node kind: {kind!r}")
        return stats

    stats["idb_applications"] += 1
    children = [child for child in trace.get("subproofs") or [] if isinstance(child, dict)]
    try:
        rule = parse_rule(str(trace.get("rule") or ""), source="proof_audit")
        goal = parse_literal(str(trace.get("goal") or ""))
        binding = {}
        valid = unify_general(rule.head, goal, binding)
        if len(rule.body) != len(children):
            valid = False
            errors.append(
                f"{prefix}: rule body/subproof count mismatch: body={len(rule.body)} children={len(children)}"
            )
        for idx, (literal, child) in enumerate(zip(rule.body, children)):
            child_goal_text = str(child.get("goal") or child.get("fact") or "")
            child_goal = parse_literal(child_goal_text)
            if not unify_general(literal, child_goal, binding):
                valid = False
                errors.append(
                    f"{prefix}: body literal {idx} does not unify with child goal: "
                    f"literal={literal} child={child_goal_text}"
                )
            grounded = apply_binding(literal, binding)
            if any(arg and arg[0].isupper() for arg in grounded.args):
                valid = False
                errors.append(f"{prefix}: body literal {idx} remains ungrounded after substitution: {grounded}")
        if not valid:
            stats["invalid_idb_applications"] += 1
    except Exception as exc:
        stats["invalid_idb_applications"] += 1
        errors.append(f"{prefix}: invalid IDB application: {exc}")

    for idx, child in enumerate(children):
        child_stats = audit_trace(child, facts, errors, prefix=f"{prefix}.child[{idx}]")
        for key in stats:
            stats[key] += child_stats[key]
    return stats


def audit_inconclusive(payload: dict, errors: List[str], warnings: List[str], prefix: str) -> None:
    if int(payload.get("evidence_count") or 0) != 0:
        errors.append(f"{prefix}: INCONCLUSIVE decision contains evidence")
    stats = payload.get("search_stats")
    if not isinstance(stats, dict):
        errors.append(f"{prefix}: INCONCLUSIVE decision is missing search_stats")
        return
    if not stats.get("normal_completion"):
        errors.append(f"{prefix}: INCONCLUSIVE search did not record normal completion")
    if int(stats.get("proofs_found") or 0) != 0:
        errors.append(f"{prefix}: INCONCLUSIVE search recorded proofs")
    termination = stats.get("termination_reason")
    if termination not in {"exhausted_no_proof", "bounded_no_proof"}:
        errors.append(f"{prefix}: invalid INCONCLUSIVE termination reason: {termination!r}")
    if termination == "bounded_no_proof":
        warnings.append(
            f"{prefix}: no proof was found within configured bounds; this is not proof of global non-existence"
        )


def normalize_fact(text: str) -> str:
    return "".join(text.strip().rstrip(".").split())


def compact(output: dict) -> str:
    lines = [
        f"Task: {output['task_dir']}",
        f"Files: {len(output['files'])}",
        f"Errors: {output['total_errors']}",
        f"Warnings: {output['total_warnings']}",
    ]
    for item in output["files"]:
        lines.append(
            f"- {Path(item['file']).name}: errors={item['error_count']} "
            f"warnings={item['warning_count']} proof_count={item['proof_count']} "
            f"edb_mentions={item['edb_fact_mentions']} "
            f"idb_apps={item['idb_applications']} "
            f"inconclusive={item['inconclusive_decisions_audited']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
