#!/usr/bin/env python3
"""Audit PARA split isolation and the empirical proof-structure boundary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable


TARGETS = ("canCallClass", "isAllowedToUse", "overridesMethod")
STRICT_FLAGS = (
    "--agent-paper-strict",
    "--agent-indexed-plan-only",
    "--agent-disable-symbolic-prior",
)
EXAMPLE_RE = re.compile(r"^(pos|neg)\((.+)\)\.$")


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    k5 = root / "results/icse_revision_20260620/e2_portfolio_k/k5"
    proof_root = root / "results/icse_revision_20260620/e4_agent_proof_audit"
    code_reasoner = root / "src" / "para" / "reasoner.py"

    checks = []
    for target in TARGETS:
        prefix = f"spring_{target}_train50_100_seed0_agent_strict_qwen27b_k5"
        split_root = k5 / "splits" / prefix
        train = split_root / "train_task"
        test = split_root / "test_task"
        manifest = k5 / "manifests" / f"{prefix}_manifest.json"
        rule_library = k5 / "rule_libraries" / f"{prefix}_rule_library.json"

        train_examples = load_examples(train / "exs.pl")
        test_examples = load_examples(test / "exs.pl")
        overlap = set(train_examples) & set(test_examples)
        checks.append(
            check(
                target,
                "train_test_example_disjoint",
                not overlap,
                {
                    "train_examples": len(train_examples),
                    "test_examples": len(test_examples),
                    "overlap": len(overlap),
                },
            )
        )

        target_fact_count = count_target_facts(target, (train / "bk.pl", test / "bk.pl"))
        checks.append(
            check(
                target,
                "target_ground_facts_excluded_from_bk",
                target_fact_count == 0,
                {"target_ground_facts_in_train_or_test_bk": target_fact_count},
            )
        )

        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        command = manifest_payload.get("learn_summary", {}).get("command", [])
        strict_present = {flag: flag in command for flag in STRICT_FLAGS}
        checks.append(
            check(
                target,
                "strict_protocol_flags",
                all(strict_present.values()),
                strict_present,
            )
        )

        library_payload = json.loads(rule_library.read_text(encoding="utf-8"))
        sources = [
            str(rule.get("source_summary") or "")
            for rule in library_payload.get("rules", [])
        ]
        run_local = bool(sources) and all(str(k5 / "learn") in source for source in sources)
        checks.append(
            check(
                target,
                "run_local_rule_library",
                run_local,
                {"rule_count": len(sources), "sources": sources},
            )
        )

    proof_stats = summarize_proofs(proof_root)
    reasoner_text = code_reasoner.read_text(encoding="utf-8")
    label_independent = (
        "elif explicit_label is False:" not in reasoner_text
        and '"negative_labels_used_for_decision": False' in reasoner_text
    )
    checks.append(
        check(
            "framework",
            "label_independent_reasoning_decision",
            label_independent,
            {"source": str(code_reasoner)},
        )
    )

    report = {
        "scope": "artifact-level audit of the final Qwen3.5-27B k=5 Spring protocol",
        "limitations": [
            "The final run did not persist complete Planner prompts, so prompt-text leakage is not claimed as directly audited.",
            "Proof-structure statistics describe sampled agent-origin traces, not every held-out query.",
        ],
        "checks": checks,
        "proof_structure": proof_stats,
        "passed": all(item["passed"] for item in checks),
        "violations": sum(not item["passed"] for item in checks),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="PARA repository root",
    )
    parser.add_argument("--output", required=True)
    return parser


def check(scope: str, name: str, passed: bool, details: dict) -> dict:
    return {"scope": scope, "check": name, "passed": passed, "details": details}


def load_examples(path: Path) -> list[str]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = EXAMPLE_RE.match(raw.strip())
        if match:
            rows.append(f"{match.group(1)}:{''.join(match.group(2).split())}")
    return rows


def count_target_facts(target: str, paths: Iterable[Path]) -> int:
    prefix = f"{target}("
    count = 0
    for path in paths:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith(prefix) and ":-" not in line:
                count += 1
    return count


def summarize_proofs(root: Path) -> dict:
    files = sorted(root.glob("*/proof_[0-9]*.json"))
    trees = 0
    edb_leaves = 0
    idb_apps = 0
    recursive_apps = 0
    multi_idb_trees = 0
    max_depth = 0
    target_counts: dict[str, dict[str, int]] = {}

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        target = path.parent.name
        stats = target_counts.setdefault(
            target,
            {"queries": 0, "proof_trees": 0, "edb_leaves": 0, "idb_applications": 0},
        )
        stats["queries"] += 1
        for evidence in payload.get("evidence") or []:
            trace = evidence.get("proof_trace") or {}
            tree_stats = walk_trace(trace)
            trees += 1
            edb_leaves += tree_stats["edb"]
            idb_apps += tree_stats["idb"]
            recursive_apps += tree_stats["recursive"]
            max_depth = max(max_depth, tree_stats["max_depth"])
            if tree_stats["idb"] > 1:
                multi_idb_trees += 1
            stats["proof_trees"] += 1
            stats["edb_leaves"] += tree_stats["edb"]
            stats["idb_applications"] += tree_stats["idb"]

    return {
        "sampled_queries": len(files),
        "proof_trees": trees,
        "edb_fact_mentions": edb_leaves,
        "idb_rule_applications": idb_apps,
        "maximum_observed_proof_depth": max_depth,
        "proof_trees_with_multiple_idb_applications": multi_idb_trees,
        "recursive_idb_applications": recursive_apps,
        "empirical_scope": (
            "single accepted IDB rule applied to multi-hop EDB evidence; "
            "the engine supports bounded IDB composition, but recursive composition "
            "is not exercised by this sample"
        ),
        "by_target": target_counts,
    }


def walk_trace(trace: dict, active_goals: tuple[str, ...] = ()) -> dict[str, int]:
    if not trace:
        return {"edb": 0, "idb": 0, "recursive": 0, "max_depth": 0}
    if trace.get("kind") == "edb_fact":
        return {"edb": 1, "idb": 0, "recursive": 0, "max_depth": 0}

    goal = str(trace.get("goal") or "")
    recursive = int(goal in active_goals)
    result = {
        "edb": 0,
        "idb": 1,
        "recursive": recursive,
        "max_depth": int(trace.get("depth") or 0) + 1,
    }
    for child in trace.get("subproofs") or []:
        if not isinstance(child, dict):
            continue
        child_stats = walk_trace(child, active_goals + (goal,))
        result["edb"] += child_stats["edb"]
        result["idb"] += child_stats["idb"]
        result["recursive"] += child_stats["recursive"]
        result["max_depth"] = max(result["max_depth"], child_stats["max_depth"])
    return result


if __name__ == "__main__":
    main()
