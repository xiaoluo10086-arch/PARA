#!/usr/bin/env python3
"""Materialize paper-facing PARA proof contracts from final-suite manifests."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
CODE_ROOT = PROJECT_ROOT / "code/latest"


def main() -> None:
    args = build_parser().parse_args()
    code_root = Path(args.code_root).resolve()
    sys.path.insert(0, str(code_root))

    rows: List[Dict[str, Any]] = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for manifest_path in iter_manifest_paths(args.manifest, args.manifest_dir):
        rows.extend(materialize_manifest_contracts(manifest_path, output_dir, args))

    rows.sort(key=lambda row: (row["split_name"], row["contract_index"]))
    if args.csv:
        write_csv(Path(args.csv), rows)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.csv and not args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize PARA proof contracts from run manifests")
    parser.add_argument("--manifest", action="append", default=[], help="One manifest JSON file; repeatable")
    parser.add_argument("--manifest-dir", action="append", default=[], help="Directory containing *_manifest.json files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--code-root", default=str(CODE_ROOT))
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--max-supported", type=int, default=3)
    parser.add_argument("--max-inconclusive", type=int, default=2)
    parser.add_argument("--max-proofs", type=int, default=3)
    parser.add_argument("--max-paths", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-states", type=int, default=2000)
    parser.add_argument("--max-edges-per-node", type=int, default=120)
    parser.add_argument("--include-negatives", action="store_true")
    parser.add_argument("--csv", help="Optional summary CSV path")
    parser.add_argument("--json", help="Optional summary JSON path")
    return parser


def iter_manifest_paths(paths: Sequence[str], dirs: Sequence[str]) -> Iterable[Path]:
    seen = set()
    for item in paths:
        path = Path(item)
        if path.is_file() and path not in seen:
            seen.add(path)
            yield path
    for item in dirs:
        root = Path(item)
        if root.is_file() and root.name.endswith("_manifest.json") and root not in seen:
            seen.add(root)
            yield root
        elif root.is_dir():
            for path in sorted(root.rglob("*_manifest.json")):
                if path not in seen:
                    seen.add(path)
                    yield path


def materialize_manifest_contracts(manifest_path: Path, output_root: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    from Rulelearning.nshrl.prolog import load_task
    from Rulelearning.nshrl.reasoner import (
        BackwardChainingReasoner,
        INCONCLUSIVE,
        SUPPORTED,
        example_label,
        load_rule_library,
        reasoning_result_from_proofs,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_name = str(manifest.get("split_name") or manifest_path.stem.replace("_manifest", ""))
    split_dir = output_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_valid = bool(manifest.get("valid_for_paper", True))
    test_task = Path(str(manifest.get("test_task") or ""))
    rule_library = Path(str(manifest.get("rule_library") or ""))
    if not manifest_valid or not test_task.exists() or not rule_library.exists():
        row = write_contract(
            split_dir=split_dir,
            contract_index=0,
            payload=invalid_contract(manifest, manifest_path, test_task, rule_library),
        )
        return [row]

    task = load_task(test_task)
    rules = load_rule_library(rule_library)
    engine_build_started = time.perf_counter()
    engine = BackwardChainingReasoner(
        task=task,
        rules=rules,
        threshold=args.threshold,
        max_depth=args.max_depth,
        max_proofs=args.max_proofs,
        max_states=args.max_states,
        max_paths=args.max_paths,
        max_edges_per_node=args.max_edges_per_node,
        excluded_facts=set(),
        include_path_evidence=True,
    )
    engine_ready = time.perf_counter()

    rows: List[Dict[str, Any]] = []
    supported = 0
    inconclusive = 0
    examples = list(task.examples) if args.include_negatives else [example for example in task.examples if example.positive]
    for example in examples:
        if supported >= args.max_supported and inconclusive >= args.max_inconclusive:
            break
        started = time.perf_counter()
        proofs = engine.prove(example.literal)
        proof_done = time.perf_counter()
        result = reasoning_result_from_proofs(
            query_literal=example.literal,
            explicit_label=example_label(task.examples, example.literal),
            engine=engine,
            proofs=proofs,
            threshold=args.threshold,
            max_depth=args.max_depth,
            max_proofs=args.max_proofs,
            started=started,
            engine_ready=started,
            proof_done=proof_done,
            include_evidence=True,
        )
        decision = str(result.get("decision") or "")
        if decision == SUPPORTED:
            if supported >= args.max_supported:
                continue
            supported += 1
        elif decision == INCONCLUSIVE:
            if inconclusive >= args.max_inconclusive:
                continue
            inconclusive += 1
        else:
            continue
        contract_index = len(rows)
        contract = build_contract(
            manifest=manifest,
            manifest_path=manifest_path,
            result=result,
            explicit_gold="positive" if example.positive else "negative",
            engine_build_seconds=engine_ready - engine_build_started,
        )
        rows.append(write_contract(split_dir=split_dir, contract_index=contract_index, payload=contract))

    summary_path = split_dir / "proof_contract_summary.json"
    summary_path.write_text(json.dumps({"split_name": split_name, "rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return rows


def invalid_contract(manifest: Dict[str, Any], manifest_path: Path, test_task: Path, rule_library: Path) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "contract_kind": "para_proof_contract",
        "split_name": manifest.get("split_name"),
        "manifest": str(manifest_path),
        "status": "INCONCLUSIVE",
        "status_category": "invalid_or_missing_artifact",
        "query": None,
        "reason": "manifest is invalid for paper reporting or required test/rule artifacts are missing",
        "artifact_checks": {
            "valid_for_paper": manifest.get("valid_for_paper"),
            "test_task_exists": test_task.exists(),
            "rule_library_exists": rule_library.exists(),
        },
        "review_obligation": "Inspect the run validity and artifact paths before using this split in the paper.",
    }


def build_contract(
    manifest: Dict[str, Any],
    manifest_path: Path,
    result: Dict[str, Any],
    explicit_gold: str,
    engine_build_seconds: float,
) -> Dict[str, Any]:
    evidence = list(result.get("evidence") or [])
    first_evidence = evidence[0] if evidence else {}
    decision = str(result.get("decision") or "INCONCLUSIVE")
    return {
        "schema_version": 1,
        "contract_kind": "para_proof_contract",
        "split_name": manifest.get("split_name"),
        "manifest": str(manifest_path),
        "task_dir": manifest.get("test_task"),
        "rule_library": manifest.get("rule_library"),
        "query": result.get("query"),
        "explicit_gold_label": explicit_gold,
        "status": decision,
        "status_category": "accepted_bounded_positive_proof" if decision == "SUPPORTED" else "review_obligation",
        "reason": result.get("reason"),
        "selected_rule": first_evidence.get("rule"),
        "selected_rule_metrics": {
            "f1": first_evidence.get("rule_f1"),
            "precision": first_evidence.get("rule_precision"),
            "recall": first_evidence.get("rule_recall"),
        },
        "proof_contract": {
            "idb_derivation": first_evidence.get("proof_trace"),
            "supporting_edb_facts": first_evidence.get("body_facts", []),
            "path_programs": first_evidence.get("path_programs", []),
            "path_evidence": first_evidence.get("path_evidence", []),
            "proof_score": first_evidence.get("proof_score"),
            "evidence_count": result.get("evidence_count", 0),
        },
        "validation_context": {
            "threshold": result.get("threshold"),
            "reasoning_mode": result.get("reasoning_mode"),
            "max_depth": result.get("max_depth"),
            "max_proofs": result.get("max_proofs"),
            "applicable_rule_count": result.get("applicable_rule_count"),
            "dependency_graph": result.get("dependency_graph"),
            "search_stats": result.get("search_stats"),
            "runtime_seconds": result.get("runtime_seconds"),
            "shared_engine_build_seconds": engine_build_seconds,
        },
        "decision_semantics": result.get("decision_semantics"),
        "review_obligation": (
            "No bounded positive proof was produced; inspect extractor coverage, rule expressibility, and bounds."
            if decision != "SUPPORTED"
            else None
        ),
        "boundary_statement": (
            "The contract certifies a bounded symbolic support trace over the extracted EDB facts. "
            "It is not a completeness proof over unextracted or runtime-only semantics."
        ),
    }


def write_contract(split_dir: Path, contract_index: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    path = split_dir / f"proof_contract_{contract_index:03d}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    contract = payload
    return {
        "split_name": str(contract.get("split_name") or split_dir.name),
        "contract_index": contract_index,
        "contract_file": str(path),
        "query": contract.get("query"),
        "gold_label": contract.get("explicit_gold_label"),
        "status": contract.get("status"),
        "status_category": contract.get("status_category"),
        "selected_rule": contract.get("selected_rule"),
        "evidence_count": (contract.get("proof_contract") or {}).get("evidence_count"),
        "supporting_edb_fact_count": len((contract.get("proof_contract") or {}).get("supporting_edb_facts") or []),
        "path_evidence_count": len((contract.get("proof_contract") or {}).get("path_evidence") or []),
        "review_obligation": contract.get("review_obligation"),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split_name",
        "contract_index",
        "contract_file",
        "query",
        "gold_label",
        "status",
        "status_category",
        "selected_rule",
        "evidence_count",
        "supporting_edb_fact_count",
        "path_evidence_count",
        "review_obligation",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
