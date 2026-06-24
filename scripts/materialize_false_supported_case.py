#!/usr/bin/env python3
"""Materialize one false-supported PARA decision with its full proof trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from para.reasoner import reason_query  # noqa: E402


DEFAULT_CASE = (
    "overridesMethod("
    "org_springframework_web_servlet_function_RouterFunctionBuilder_OPTIONS_HandlerFunction,"
    "org_springframework_web_servlet_function_Builder_OPTIONS_String_RequestPredicate_HandlerFunction"
    ")"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--rule-library",
        type=Path,
        required=True,
    )
    parser.add_argument("--query", default=DEFAULT_CASE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "false_supported_case",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = reason_query(
        task_dir=args.task_dir,
        rule_library=args.rule_library,
        query=args.query,
        threshold=0.8,
        max_depth=8,
        max_proofs=5,
        max_states=5000,
    )
    original_library = json.loads(args.rule_library.read_text(encoding="utf-8"))
    diagnostic_library = json.loads(json.dumps(original_library))
    diagnostic_rule = (
        "overridesMethod(A,B) :- containsMethod(V1,A),inheritsClass(V1,V2),"
        "containsMethod(V2,B),methodName(A,C2),methodName(B,C2),"
        "methodArity(A,C3),methodArity(B,C3) [0.944]."
    )
    diagnostic_library["rules"][0]["rule"] = diagnostic_rule
    diagnostic_library["rules"][0]["source_method"] = "posthoc_diagnostic_signature_constraint"
    diagnostic_path = args.output_dir / "diagnostic_rule_library.json"
    diagnostic_path.write_text(
        json.dumps(diagnostic_library, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    diagnostic_result = reason_query(
        task_dir=args.task_dir,
        rule_library=diagnostic_path,
        query=args.query,
        threshold=0.8,
        max_depth=8,
        max_proofs=5,
        max_states=5000,
    )
    payload = {
        "gold": "negative",
        "observed_decision": result["decision"],
        "query": args.query,
        "task_dir": str(args.task_dir),
        "rule_library": str(args.rule_library),
        "result": result,
        "diagnostic_replay": {
            "rule": diagnostic_rule,
            "decision": diagnostic_result["decision"],
            "evidence_count": diagnostic_result["evidence_count"],
            "status": "post-hoc diagnosis only; excluded from reported main metrics",
        },
        "diagnostic_interpretation": (
            "The accepted rule uses class containment, direct inheritance, and method-name equality, "
            "but omits method arity/signature equality. The proof trace exposes the exact class and "
            "method bindings responsible for the false support."
        ),
        "repair_hypothesis": (
            "Add methodArity equality or a stronger same-signature constraint, then re-run the "
            "unchanged held-out query. This is a diagnosis, not a post-hoc replacement of the "
            "reported main result."
        ),
    }
    (args.output_dir / "false_supported_case.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    evidence = result.get("evidence") or []
    facts = evidence[0].get("body_facts") if evidence else []
    lines = [
        "# False-Supported Case Audit",
        "",
        f"- Gold label: negative",
        f"- PARA decision: {result['decision']}",
        f"- Query: `{args.query}`",
        "",
        "## Accepted rule",
        "",
        f"`{evidence[0].get('rule') if evidence else 'No proof'}`",
        "",
        "## Grounded proof facts",
        "",
    ]
    lines.extend(f"- `{fact}`" for fact in facts)
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            "The rule checks containment, direct class inheritance, and method-name equality,",
            "but it does not require equal method arity or a complete signature match. The proof",
            "trace therefore reveals why a structurally plausible negative was supported.",
            "",
            "## Diagnostic replay",
            "",
            f"- Added constraint: shared `methodArity`",
            f"- Replayed decision: {diagnostic_result['decision']}",
            f"- Evidence count: {diagnostic_result['evidence_count']}",
            "",
            "This case is retained in the reported result. Adding a signature constraint is a",
            "repair hypothesis and is not used to rewrite the completed experiment.",
            "",
        ]
    )
    (args.output_dir / "case_study.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "evidence_count": result["evidence_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
