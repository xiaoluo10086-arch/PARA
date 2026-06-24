#!/usr/bin/env python3
"""Run query-level direct LLM answering on frozen held-out PARA splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from query_evidence import GraphIndex, body_predicates_from_bias, render_schema


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "src"
DEFAULT_SPLIT_ROOT = ROOT / "artifacts" / "learn_then_reason_spring" / "splits"
TARGETS = ("canCallClass", "isAllowedToUse", "overridesMethod")
DECISIONS = {"SUPPORTED", "UNSUPPORTED", "INCONCLUSIVE"}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "direct_query_decisions",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "query_id": {"type": "string"},
                            "decision": {"type": "string", "enum": sorted(DECISIONS)},
                            "evidence_ids": {"type": "array", "items": {"type": "string"}},
                            "rationale": {"type": "string"},
                        },
                        "required": ["query_id", "decision", "evidence_ids", "rationale"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["answers"],
            "additionalProperties": False,
        },
    },
}


def main() -> None:
    args = build_parser().parse_args()
    sys.path.insert(0, str(Path(args.code_root).resolve()))
    from para.llm_clients import chat_text
    from para.prolog import parse_example_line

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    split_root = Path(args.split_root).resolve()
    first_task = task_dir(split_root, TARGETS[0], args.seed)
    allowed = body_predicates_from_bias(first_task / "bias.pl")
    graph = GraphIndex.from_bk(first_task / "bk.pl", allowed_predicates=allowed)
    decision_path = output_root / "decisions.jsonl"
    rows: List[Dict[str, object]] = load_existing_rows(decision_path) if args.resume else []
    completed_query_ids = {str(row.get("query_id")) for row in rows}
    demonstration_records: Dict[str, List[Dict[str, object]]] = {}
    raw_path = output_root / "raw_calls.jsonl"
    raw_handle = raw_path.open("a", encoding="utf-8")
    try:
        selected_targets = tuple(args.target or TARGETS)
        for target in selected_targets:
            task = task_dir(split_root, target, args.seed)
            examples = [
                example
                for line in (task / "exs.pl").read_text(encoding="utf-8").splitlines()
                if (example := parse_example_line(line)) is not None
            ]
            train_task = task.parent / "train_task"
            train_examples = [
                example
                for line in (train_task / "exs.pl").read_text(encoding="utf-8").splitlines()
                if (example := parse_example_line(line)) is not None
            ]
            demonstrations = build_demonstrations(graph, train_examples, args)
            demonstration_records[target] = demonstrations
            selected = stratified_examples(examples, args.per_target, args.positive_ratio)
            query_rows = []
            for index, example in enumerate(selected):
                source, destination = example.literal.args[:2]
                view = graph.evidence_view(
                    source,
                    destination,
                    max_depth=args.evidence_max_depth,
                    max_paths=args.evidence_max_paths,
                    max_facts=args.evidence_max_facts,
                    per_node_cap=args.evidence_edge_cap,
                )
                facts = [
                    {"id": f"F{fact_index + 1}", "fact": fact}
                    for fact_index, fact in enumerate(view["facts"])
                ]
                query_rows.append(
                    {
                        "query_id": f"{target}-{index:04d}",
                        "query": literal_text(example.literal),
                        "gold": "positive" if example.positive else "negative",
                        "evidence": facts,
                        "view_metadata": {key: value for key, value in view.items() if key != "facts"},
                    }
                )
            query_rows = [row for row in query_rows if str(row["query_id"]) not in completed_query_ids]
            schema = render_schema(task / "bias.pl")
            for batch in chunks(query_rows, args.batch_size):
                messages = build_messages(target, schema, demonstrations, batch)
                started = time.perf_counter()
                error = ""
                raw = ""
                parsed: Dict[str, object] = {}
                for attempt in range(args.retries + 1):
                    try:
                        raw = chat_text(
                            base_url=args.base_url,
                            model=args.model,
                            messages=messages,
                            request_timeout=args.timeout,
                            temperature=0.0,
                            max_tokens=args.max_tokens,
                            json_response=True,
                            response_format=(
                                RESPONSE_FORMAT
                                if args.structured_mode == "json_schema"
                                else {"type": "json_object"}
                            ),
                            extra_payload=(
                                {"thinking": {"type": "disabled"}}
                                if args.disable_thinking
                                else None
                            ),
                        )
                        parsed = parse_json(raw)
                        error = ""
                        break
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        if attempt >= args.retries or not retryable(error):
                            break
                        time.sleep(retry_delay(error, attempt, args.retry_delay))
                elapsed = time.perf_counter() - started
                answer_by_id = {
                    str(item.get("query_id")): item
                    for item in parsed.get("answers", [])
                    if isinstance(item, Mapping)
                }
                raw_handle.write(
                    json.dumps(
                        {
                            "target": target,
                            "query_ids": [item["query_id"] for item in batch],
                            "elapsed_seconds": elapsed,
                            "error": error,
                            "raw": raw,
                            "parsed": parsed,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                raw_handle.flush()
                for item in batch:
                    answer = answer_by_id.get(str(item["query_id"]), {})
                    decision = str(answer.get("decision") or "INVALID").upper()
                    evidence_ids = answer.get("evidence_ids") if isinstance(answer.get("evidence_ids"), list) else []
                    valid_ids = {fact["id"] for fact in item["evidence"]}
                    cited_ids = [str(value) for value in evidence_ids]
                    rows.append(
                        {
                            **item,
                            "model": args.model,
                            "decision": decision if decision in DECISIONS else "INVALID",
                            "evidence_ids": cited_ids,
                            "invalid_evidence_ids": sorted(set(cited_ids) - valid_ids),
                            "rationale": str(answer.get("rationale") or ""),
                            "call_error": error,
                            "batch_elapsed_seconds": elapsed,
                        }
                    )
                write_outputs(output_root, args, rows, graph.fact_count, demonstration_records)
    finally:
        raw_handle.close()
    write_outputs(output_root, args, rows, graph.fact_count, demonstration_records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split-root", default=str(DEFAULT_SPLIT_ROOT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", action="append", choices=TARGETS, default=[])
    parser.add_argument("--code-root", default=str(CODE_ROOT))
    parser.add_argument("--per-target", type=int, default=30, help="0 means all held-out queries")
    parser.add_argument("--positive-ratio", type=float, default=1 / 3)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--evidence-max-depth", type=int, default=4)
    parser.add_argument("--evidence-max-paths", type=int, default=8)
    parser.add_argument("--evidence-max-facts", type=int, default=32)
    parser.add_argument("--evidence-edge-cap", type=int, default=80)
    parser.add_argument("--demo-positive", type=int, default=1)
    parser.add_argument("--demo-negative", type=int, default=1)
    parser.add_argument("--demo-max-facts", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--structured-mode", choices=("json_schema", "json_object"), default="json_schema")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-delay", type=float, default=8.0)
    parser.add_argument("--resume", action="store_true", help="Skip query_ids already present in decisions.jsonl")
    return parser


def task_dir(split_root: Path, target: str, seed: int) -> Path:
    return split_root / f"spring_{target}_train50_100_seed{seed}_graphrag" / "test_task"


def stratified_examples(examples: Sequence[object], count: int, positive_ratio: float) -> List[object]:
    if count <= 0 or count >= len(examples):
        return list(examples)
    positives = [example for example in examples if example.positive]
    negatives = [example for example in examples if not example.positive]
    positive_count = min(len(positives), max(1, round(count * positive_ratio)))
    negative_count = min(len(negatives), count - positive_count)
    return positives[:positive_count] + negatives[:negative_count]


def literal_text(literal: object) -> str:
    return f"{literal.predicate}({','.join(literal.args)})"


def chunks(rows: Sequence[Dict[str, object]], size: int) -> Iterable[List[Dict[str, object]]]:
    for index in range(0, len(rows), size):
        yield list(rows[index : index + size])


def build_demonstrations(
    graph: GraphIndex,
    examples: Sequence[object],
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    positives = [example for example in examples if example.positive][: args.demo_positive]
    negatives = [example for example in examples if not example.positive][: args.demo_negative]
    rows = []
    for index, example in enumerate(positives + negatives):
        source, destination = example.literal.args[:2]
        view = graph.evidence_view(
            source,
            destination,
            max_depth=args.evidence_max_depth,
            max_paths=args.evidence_max_paths,
            max_facts=args.demo_max_facts,
            per_node_cap=args.evidence_edge_cap,
        )
        rows.append(
            {
                "example_id": f"D{index + 1}",
                "query": literal_text(example.literal),
                "label": "SUPPORTED" if example.positive else "UNSUPPORTED",
                "evidence": [
                    {"id": f"D{index + 1}F{fact_index + 1}", "fact": fact}
                    for fact_index, fact in enumerate(view["facts"])
                ],
            }
        )
    return rows


def build_messages(
    target: str,
    schema: Sequence[str],
    demonstrations: Sequence[Dict[str, object]],
    batch: Sequence[Dict[str, object]],
) -> List[Dict[str, str]]:
    payload = []
    for row in batch:
        payload.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "evidence": row["evidence"],
            }
        )
    system = (
        "You answer held-out software architecture relation queries using only the supplied graph facts. "
        "Return SUPPORTED only when the supplied facts contain a complete relation-specific evidence chain. "
        "Return UNSUPPORTED only when the supplied facts explicitly establish the negative claim; absence of "
        "support is not contradiction. Otherwise return INCONCLUSIVE. Do not invent facts, rules, or identifiers. "
        "Evidence IDs are local to each query. Keep each rationale under 40 words. "
        "Return one JSON object with the shape "
        "{\"answers\":[{\"query_id\":\"...\",\"decision\":\"SUPPORTED|UNSUPPORTED|INCONCLUSIVE\","
        "\"evidence_ids\":[\"F1\"],\"rationale\":\"...\"}]}."
    )
    user = (
        f"Target predicate: {target}\n"
        "Graph schema:\n"
        + "\n".join(schema)
        + "\n\nFixed labeled examples from the training split:\n"
        + json.dumps(demonstrations, ensure_ascii=False)
        + "\n\nQueries and deterministic query-centered evidence:\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n\nReturn one answer for every query_id."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_json(text: str) -> Dict[str, object]:
    content = text.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    parsed = json.loads(content)
    return parsed if isinstance(parsed, dict) else {}


def retryable(error: str) -> bool:
    lowered = error.lower()
    return any(
        marker in lowered
        for marker in (
            "429",
            "rate limit",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "connection reset",
            "empty response",
            "jsondecodeerror",
            "unterminated string",
        )
    )


def retry_delay(error: str, attempt: int, default: float) -> float:
    match = re.search(r"try again in ([0-9.]+)s", error, flags=re.IGNORECASE)
    if match:
        return max(default, float(match.group(1)) + 1.0)
    return default * (attempt + 1)


def load_existing_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [
        row
        for row in rows
        if row.get("decision") != "INVALID" and not row.get("call_error")
    ]


def write_outputs(
    output_root: Path,
    args: argparse.Namespace,
    rows: Sequence[Dict[str, object]],
    fact_count: int,
    demonstrations: Mapping[str, Sequence[Dict[str, object]]],
) -> None:
    row_path = output_root / "decisions.jsonl"
    row_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = summarize(rows)
    summary.update(
        {
            "protocol": "direct_llm_query_answering_with_fixed_train_demonstrations",
            "model_tag": args.model_tag,
            "model": args.model,
            "base_url": args.base_url,
            "seed": args.seed,
            "graph_binary_fact_count": fact_count,
            "evidence_config": {
                "max_depth": args.evidence_max_depth,
                "max_paths": args.evidence_max_paths,
                "max_facts": args.evidence_max_facts,
                "edge_cap": args.evidence_edge_cap,
                "selection": "deterministic shortest paths plus endpoint-incident facts",
                "train_demonstrations": {
                    "positive": args.demo_positive,
                    "negative": args.demo_negative,
                    "max_facts_each": args.demo_max_facts,
                    "selection": "fixed prefix within each train label",
                    "records": dict(demonstrations),
                    "sha256": hashlib.sha256(
                        json.dumps(demonstrations, sort_keys=True, ensure_ascii=False).encode("utf-8")
                    ).hexdigest(),
                },
            },
            "selection_sha256": hashlib.sha256(
                "\n".join(str(row["query"]) for row in rows).encode("utf-8")
            ).hexdigest(),
        }
    )
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        f"# Direct Query Answering: {args.model_tag}",
        "",
        "| Target | Queries | Accuracy | Supported precision | Supported recall | Unsupported recall | Inconclusive | Invalid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target, metrics in summary["by_target"].items():
        lines.append(
            f"| `{target}` | {metrics['queries']} | {metrics['accuracy']:.3f} | "
            f"{metrics['supported_precision']:.3f} | {metrics['supported_recall']:.3f} | "
            f"{metrics['unsupported_recall']:.3f} | {metrics['inconclusive_rate']:.3f} | {metrics['invalid']} |"
        )
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    by_target: Dict[str, Dict[str, float | int]] = {}
    for target in TARGETS:
        selected = [row for row in rows if str(row["query"]).startswith(target + "(")]
        counts = Counter(str(row["decision"]) for row in selected)
        positives = [row for row in selected if row["gold"] == "positive"]
        negatives = [row for row in selected if row["gold"] == "negative"]
        tp = sum(row["decision"] == "SUPPORTED" for row in positives)
        fp = sum(row["decision"] == "SUPPORTED" for row in negatives)
        tn = sum(row["decision"] == "UNSUPPORTED" for row in negatives)
        correct = tp + tn
        by_target[target] = {
            "queries": len(selected),
            "accuracy": correct / len(selected) if selected else 0.0,
            "supported_precision": tp / (tp + fp) if tp + fp else 0.0,
            "supported_recall": tp / len(positives) if positives else 0.0,
            "unsupported_recall": tn / len(negatives) if negatives else 0.0,
            "inconclusive_rate": counts["INCONCLUSIVE"] / len(selected) if selected else 0.0,
            "invalid": counts["INVALID"],
            "invalid_evidence_citations": sum(bool(row["invalid_evidence_ids"]) for row in selected),
        }
    return {"queries": len(rows), "by_target": by_target}


if __name__ == "__main__":
    main()
