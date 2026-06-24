#!/usr/bin/env python3
"""Run split-based PARA learn-then-reason experiments.

The script enforces the paper-facing protocol:

1. Materialize a train/test split from one target task.
2. Run PARA learning only on the train task.
3. Export a run-local rule library from that learning output.
4. Evaluate query-time reasoning only on the held-out test task.

No pre-existing rule library is used as an input to the learning or reasoning
stage.  Historical full-task rule libraries may be used only by separate
upper-bound scripts, not by this protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "src"


def main() -> None:
    args = build_parser().parse_args()
    task_dir = Path(args.task_dir).resolve()
    output_root = Path(args.output_root).resolve()
    split_root = output_root / "splits" / args.split_name
    train_task = split_root / "train_task"
    test_task = split_root / "test_task"
    learn_output = output_root / "learn" / args.split_name
    rule_library = output_root / "rule_libraries" / f"{args.split_name}_rule_library.json"
    reason_output = output_root / "reason_eval" / f"{args.split_name}_test_reason_eval.json"
    no_rule_output = output_root / "reason_eval" / f"{args.split_name}_test_no_rule_reason_eval.json"
    manifest_path = output_root / "manifests" / f"{args.split_name}_manifest.json"

    output_root.mkdir(parents=True, exist_ok=True)
    split_manifest = materialize_split(
        task_dir=task_dir,
        train_task=train_task,
        test_task=test_task,
        train_pos=args.train_pos,
        train_neg=args.train_neg,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )

    started = time.perf_counter()
    preflight_summary = None
    if should_run_llm_preflight(args):
        preflight_summary = run_llm_preflight(
            code_root=Path(args.code_root).resolve(),
            python=Path(args.python).resolve() if args.python else Path(sys.executable),
            args=args,
            output_dir=output_root / "preflight",
        )
        preflight_failure = classify_command_failure(preflight_summary)
        if preflight_failure["invalid_for_paper"]:
            manifest = build_manifest(
                args=args,
                task_dir=task_dir,
                output_root=output_root,
                split_name=args.split_name,
                train_task=train_task,
                test_task=test_task,
                split_manifest=split_manifest,
                learn_output=learn_output,
                rule_library=rule_library,
                reason_output=reason_output,
                no_rule_output=no_rule_output,
                started=started,
                preflight_summary=preflight_summary,
                learn_summary=None,
                export_summary=None,
                reason_summary=None,
                no_rule_summary=None,
                validity=preflight_failure,
            )
            write_manifest(manifest_path, manifest)
            print(json.dumps(manifest, indent=2, ensure_ascii=False) if args.json else compact_report(manifest))
            if args.exit_nonzero_on_invalid:
                raise SystemExit(2)
            return

    learn_summary = run_learning(
        code_root=Path(args.code_root).resolve(),
        python=Path(args.python).resolve() if args.python else Path(sys.executable),
        train_task=train_task,
        output_dir=learn_output,
        args=args,
    )
    learn_validity = classify_learn_validity(learn_summary)
    if learn_validity["stop_pipeline"] and not args.continue_after_guidance_failed:
        manifest = build_manifest(
            args=args,
            task_dir=task_dir,
            output_root=output_root,
            split_name=args.split_name,
            train_task=train_task,
            test_task=test_task,
            split_manifest=split_manifest,
            learn_output=learn_output,
            rule_library=rule_library,
            reason_output=reason_output,
            no_rule_output=no_rule_output,
            started=started,
            preflight_summary=preflight_summary,
            learn_summary=learn_summary,
            export_summary=None,
            reason_summary=None,
            no_rule_summary=None,
            validity=learn_validity,
        )
        write_manifest(manifest_path, manifest)
        print(json.dumps(manifest, indent=2, ensure_ascii=False) if args.json else compact_report(manifest))
        if args.exit_nonzero_on_invalid:
            raise SystemExit(2)
        return

    export_summary = export_rule_library(
        code_root=Path(args.code_root).resolve(),
        python=Path(args.python).resolve() if args.python else Path(sys.executable),
        learn_output=learn_output,
        rule_library=rule_library,
        min_f1=args.min_f1,
    )
    reason_summary = run_reason_eval(
        code_root=Path(args.code_root).resolve(),
        python=Path(args.python).resolve() if args.python else Path(sys.executable),
        test_task=test_task,
        rule_library=rule_library,
        output=reason_output,
        max_proofs=args.reason_max_proofs,
    )
    no_rule_summary = None
    if args.run_no_rule:
        empty_library = output_root / "rule_libraries" / f"{args.split_name}_empty_rule_library.json"
        write_empty_rule_library(empty_library, min_f1=args.min_f1)
        no_rule_summary = run_reason_eval(
            code_root=Path(args.code_root).resolve(),
            python=Path(args.python).resolve() if args.python else Path(sys.executable),
            test_task=test_task,
            rule_library=empty_library,
            output=no_rule_output,
            max_proofs=args.reason_max_proofs,
        )

    manifest = build_manifest(
        args=args,
        task_dir=task_dir,
        output_root=output_root,
        split_name=args.split_name,
        train_task=train_task,
        test_task=test_task,
        split_manifest=split_manifest,
        learn_output=learn_output,
        rule_library=rule_library,
        reason_output=reason_output,
        no_rule_output=no_rule_output,
        started=started,
        preflight_summary=preflight_summary,
        learn_summary=learn_summary,
        export_summary=export_summary,
        reason_summary=reason_summary,
        no_rule_summary=no_rule_summary,
        validity=learn_validity,
    )
    write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False) if args.json else compact_report(manifest))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split-based PARA learn-then-reason runner")
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split-name", default="split_seed0")
    parser.add_argument("--train-pos", type=int, default=50)
    parser.add_argument("--train-neg", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true", help="Use prefix examples instead of seeded stratified shuffle")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--code-root", default=str(CODE_ROOT))
    parser.add_argument("--guide-provider", choices=["graphrag", "agent", "heuristic"], default="graphrag")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--runner-timeout", type=int, default=900, help="Outer subprocess timeout for each stage")
    parser.add_argument("--min-f1", type=float, default=0.8)
    parser.add_argument("--predicate-budget", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--strict-candidate-first", action="store_true", default=True)
    parser.add_argument("--candidate-first", action="store_true", default=True)
    parser.add_argument("--graphrag-max-depth", type=int, default=5)
    parser.add_argument("--graphrag-max-positive-examples", type=int, default=8)
    parser.add_argument("--graphrag-max-paths-per-example", type=int, default=12)
    parser.add_argument("--graphrag-max-edges-per-node", type=int, default=80)
    parser.add_argument("--graphrag-max-candidates", type=int, default=30)
    parser.add_argument("--graphrag-constraint-mode", choices=["direct", "attribute", "both", "none"], default="attribute")
    parser.add_argument("--graphrag-min-path-support", type=int, default=1)
    parser.add_argument("--agent-iterations", type=int, default=2)
    parser.add_argument("--agent-acceptance-f1", type=float, default=0.8)
    parser.add_argument("--agent-portfolio-size", type=int, default=1)
    parser.add_argument("--agent-max-path-queries", type=int, default=5)
    parser.add_argument("--agent-candidate-evaluation-cap", type=int, default=40)
    parser.add_argument("--agent-indexed-plan-only", action="store_true")
    parser.add_argument("--agent-compact-actions", action="store_true")
    parser.add_argument("--agent-disable-symbolic-prior", action="store_true")
    parser.add_argument("--agent-schema-profile-mode", choices=["assisted", "raw"], default="raw")
    parser.add_argument(
        "--agent-witness-evidence-mode",
        choices=["full", "schema_only", "deterministic_top1"],
        default="full",
    )
    parser.add_argument("--agent-paper-strict", action="store_true")
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--llm-timeout", type=int, default=120)
    parser.add_argument(
        "--skip-llm-preflight",
        action="store_true",
        help="Do not issue the one-call API connectivity check before agentic remote-LLM runs",
    )
    parser.add_argument(
        "--continue-after-guidance-failed",
        action="store_true",
        help="Compatibility mode: export an empty library and run reasoning even after guidance_failed",
    )
    parser.add_argument(
        "--exit-nonzero-on-invalid",
        action="store_true",
        help="Exit with code 2 when the run is invalid for paper reporting",
    )
    parser.add_argument("--reason-max-proofs", type=int, default=5)
    parser.add_argument("--run-no-rule", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def build_manifest(
    *,
    args: argparse.Namespace,
    task_dir: Path,
    output_root: Path,
    split_name: str,
    train_task: Path,
    test_task: Path,
    split_manifest: dict,
    learn_output: Path,
    rule_library: Path,
    reason_output: Path,
    no_rule_output: Path,
    started: float,
    preflight_summary: dict | None,
    learn_summary: dict | None,
    export_summary: dict | None,
    reason_summary: dict | None,
    no_rule_summary: dict | None,
    validity: dict,
) -> dict:
    return {
        "protocol": "learn_then_reason",
        "task_dir": str(task_dir),
        "output_root": str(output_root),
        "split_name": split_name,
        "train_task": str(train_task),
        "test_task": str(test_task),
        "split": split_manifest,
        "code_root": str(Path(args.code_root).resolve()),
        "python": str(Path(args.python).resolve() if args.python else Path(sys.executable)),
        "llm_base_url": args.llm_base_url,
        "llm_model": args.llm_model,
        "preflight_summary": preflight_summary,
        "learn_output": str(learn_output),
        "rule_library": str(rule_library),
        "reason_output": str(reason_output),
        "no_rule_output": str(no_rule_output) if args.run_no_rule else None,
        "learn_summary": learn_summary,
        "export_summary": export_summary,
        "reason_summary": reason_summary,
        "no_rule_summary": no_rule_summary,
        "validity": validity,
        "valid_for_paper": not validity.get("invalid_for_paper", False),
        "elapsed_seconds": time.perf_counter() - started,
        "leakage_boundary": (
            "The reasoner uses only the rule library exported from this run's "
            "train-task learning output; held-out test examples are not used "
            "during learning or rule export."
        ),
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def should_run_llm_preflight(args: argparse.Namespace) -> bool:
    if args.skip_llm_preflight or args.guide_provider != "agent" or not args.llm_base_url:
        return False
    lowered = args.llm_base_url.lower()
    return not any(host in lowered for host in ("127.0.0.1", "localhost", "0.0.0.0"))


def run_llm_preflight(code_root: Path, python: Path, args: argparse.Namespace, output_dir: Path) -> dict:
    prompt = (
        "Return exactly this JSON object and nothing else: "
        '{"path_queries":[],"required_predicates":[],"avoid_predicates":[],'
        '"preferred_constraint_predicates":[],"strategy_presets":[]}'
    )
    code = (
        "import json, sys\n"
        "from para.llm_clients import chat_text\n"
        "base, model, timeout = sys.argv[1], sys.argv[2], int(sys.argv[3])\n"
        "messages=[{'role':'system','content':'Return one JSON object. No prose.'},"
        "{'role':'user','content':sys.argv[4]}]\n"
        "text=chat_text(base_url=base, model=model, messages=messages, request_timeout=timeout, "
        "temperature=0.0, max_tokens=128, json_response=True, response_format={'type':'json_object'})\n"
        "print(json.dumps({'ok': True, 'text': text[:1000]}, ensure_ascii=False))\n"
    )
    cmd = [
        str(python),
        "-c",
        code,
        args.llm_base_url,
        args.llm_model,
        str(min(args.llm_timeout, 60)),
        prompt,
    ]
    return run_command(cmd, cwd=code_root, output_dir=output_dir, timeout=min(args.llm_timeout + 20, 90))


def classify_command_failure(summary: dict | None) -> dict:
    if not summary:
        return {"invalid_for_paper": False, "category": "not_run", "stop_pipeline": False}
    text = " ".join(
        str(summary.get(key, ""))
        for key in ("stderr_tail", "stdout_tail", "returncode", "timed_out")
    )
    if summary.get("timed_out"):
        return invalidity("transport_timeout", text, stage="llm_preflight")
    if summary.get("returncode") not in (0, None):
        return invalidity(classify_error_text(text), text, stage="llm_preflight")
    parsed = summary.get("json")
    if not isinstance(parsed, dict) or not parsed.get("ok"):
        return invalidity("preflight_invalid_response", text, stage="llm_preflight")
    return {"invalid_for_paper": False, "category": "ok", "stage": "llm_preflight", "stop_pipeline": False}


def classify_learn_validity(summary: dict | None) -> dict:
    if not summary:
        return invalidity("learn_not_run", "", stage="learn")
    text = " ".join(
        str(summary.get(key, ""))
        for key in ("stderr_tail", "stdout_tail", "returncode", "timed_out")
    )
    if summary.get("timed_out"):
        return invalidity("runner_timeout", text, stage="learn")
    if summary.get("returncode") not in (0, None):
        return invalidity(classify_error_text(text), text, stage="learn")
    parsed = summary.get("json")
    if not isinstance(parsed, dict):
        return invalidity("learn_no_json_summary", text, stage="learn")
    status = parsed.get("status")
    if status == "guidance_failed":
        error = str(parsed.get("error", ""))
        category = classify_error_text(error)
        return invalidity(category, error, stage="learn")
    return {
        "invalid_for_paper": False,
        "category": str(status or "unknown"),
        "stage": "learn",
        "stop_pipeline": False,
    }


def classify_error_text(text: str) -> str:
    lowered = text.lower()
    if "operation not permitted" in lowered:
        return "sandbox_transport_blocked"
    if "tunnel connection failed" in lowered or "proxy" in lowered or "connection reset" in lowered:
        return "proxy_or_transport_failed"
    if "http error 401" in lowered or "http error 403" in lowered or "unauthorized" in lowered:
        return "auth_failed"
    if "http error 404" in lowered or "not found" in lowered:
        return "endpoint_failed"
    if "http error 429" in lowered or "rate limit" in lowered:
        return "rate_limited"
    if "invalid_argument" in lowered or "http error 400" in lowered or "returned error: 400" in lowered:
        return "endpoint_or_request_config_failed"
    if "http error 5" in lowered or "503 service unavailable" in lowered:
        return "provider_unavailable"
    if "timed out" in lowered or "timeout" in lowered:
        return "transport_timeout"
    if "json" in lowered or "parse" in lowered or "non-parse" in lowered:
        return "model_output_parse_failed"
    return "model_guidance_failed"


def invalidity(category: str, detail: str, *, stage: str) -> dict:
    return {
        "invalid_for_paper": True,
        "category": category,
        "stage": stage,
        "stop_pipeline": True,
        "detail": detail[-1000:],
    }


def materialize_split(
    task_dir: Path,
    train_task: Path,
    test_task: Path,
    train_pos: int,
    train_neg: int,
    seed: int,
    shuffle: bool,
) -> dict:
    pos_lines, neg_lines, other_lines = read_examples(task_dir / "exs.pl")
    if train_pos >= len(pos_lines) or train_neg >= len(neg_lines):
        raise ValueError(
            f"train split must leave held-out examples; got train_pos={train_pos}/{len(pos_lines)}, "
            f"train_neg={train_neg}/{len(neg_lines)}"
        )
    pos_order = list(range(len(pos_lines)))
    neg_order = list(range(len(neg_lines)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(pos_order)
        rng.shuffle(neg_order)
    train_pos_ids = set(pos_order[:train_pos])
    train_neg_ids = set(neg_order[:train_neg])
    train_pos_lines = [line for idx, line in enumerate(pos_lines) if idx in train_pos_ids]
    test_pos_lines = [line for idx, line in enumerate(pos_lines) if idx not in train_pos_ids]
    train_neg_lines = [line for idx, line in enumerate(neg_lines) if idx in train_neg_ids]
    test_neg_lines = [line for idx, line in enumerate(neg_lines) if idx not in train_neg_ids]

    copy_task_shell(task_dir, train_task)
    copy_task_shell(task_dir, test_task)
    write_examples(train_task / "exs.pl", other_lines, train_pos_lines, train_neg_lines)
    write_examples(test_task / "exs.pl", other_lines, test_pos_lines, test_neg_lines)
    manifest = {
        "seed": seed,
        "shuffle": shuffle,
        "source_positive": len(pos_lines),
        "source_negative": len(neg_lines),
        "train_positive": len(train_pos_lines),
        "train_negative": len(train_neg_lines),
        "test_positive": len(test_pos_lines),
        "test_negative": len(test_neg_lines),
    }
    (train_task / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (test_task / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def read_examples(path: Path) -> Tuple[List[str], List[str], List[str]]:
    pos: List[str] = []
    neg: List[str] = []
    other: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("pos("):
            pos.append(line)
        elif stripped.startswith("neg("):
            neg.append(line)
        else:
            other.append(line)
    return pos, neg, other


def write_examples(path: Path, other: Sequence[str], pos: Sequence[str], neg: Sequence[str]) -> None:
    lines = list(other)
    if lines and lines[-1].strip():
        lines.append("")
    lines.extend(pos)
    lines.extend(neg)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def copy_task_shell(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("bk.pl", "bias.pl", "metadata.json"):
        source = src / name
        if source.exists():
            shutil.copy2(source, dst / name)


def run_learning(code_root: Path, python: Path, train_task: Path, output_dir: Path, args: argparse.Namespace) -> dict:
    cmd = [
        str(python),
        "-m",
        "para.cli",
        "learn",
        "--task-dir",
        str(train_task),
        "--output-dir",
        str(output_dir),
        "--guide-provider",
        args.guide_provider,
        "--timeout",
        str(args.timeout),
        "--min-f1",
        str(args.min_f1),
        "--predicate-budget",
        str(args.predicate_budget),
        "--rounds",
        str(args.rounds),
        "--graphrag-max-depth",
        str(args.graphrag_max_depth),
        "--graphrag-max-positive-examples",
        str(args.graphrag_max_positive_examples),
        "--graphrag-max-paths-per-example",
        str(args.graphrag_max_paths_per_example),
        "--graphrag-max-edges-per-node",
        str(args.graphrag_max_edges_per_node),
        "--graphrag-max-candidates",
        str(args.graphrag_max_candidates),
        "--graphrag-constraint-mode",
        args.graphrag_constraint_mode,
        "--graphrag-min-path-support",
        str(args.graphrag_min_path_support),
        "--json",
    ]
    if args.candidate_first:
        cmd.append("--candidate-first")
    if args.strict_candidate_first:
        cmd.append("--strict-candidate-first")
    if args.guide_provider == "agent":
        cmd.extend(
            [
                "--agent-iterations",
                str(args.agent_iterations),
                "--agent-acceptance-f1",
                str(args.agent_acceptance_f1),
                "--agent-portfolio-size",
                str(args.agent_portfolio_size),
                "--agent-max-path-queries",
                str(args.agent_max_path_queries),
                "--agent-candidate-evaluation-cap",
                str(args.agent_candidate_evaluation_cap),
                "--agent-schema-profile-mode",
                args.agent_schema_profile_mode,
                "--agent-witness-evidence-mode",
                args.agent_witness_evidence_mode,
                "--llm-timeout",
                str(args.llm_timeout),
            ]
        )
        if args.llm_base_url:
            cmd.extend(["--llm-base-url", args.llm_base_url])
        if args.llm_model:
            cmd.extend(["--llm-model", args.llm_model])
        if args.agent_indexed_plan_only:
            cmd.append("--agent-indexed-plan-only")
        if args.agent_compact_actions:
            cmd.append("--agent-compact-actions")
        if args.agent_disable_symbolic_prior:
            cmd.append("--agent-disable-symbolic-prior")
        if args.agent_paper_strict:
            cmd.append("--agent-paper-strict")
    return run_command(cmd, cwd=code_root, output_dir=output_dir, timeout=args.runner_timeout)


def export_rule_library(code_root: Path, python: Path, learn_output: Path, rule_library: Path, min_f1: float) -> dict:
    cmd = [
        str(python),
        "-m",
        "para.cli",
        "export-rule-library",
        "--summary",
        str(learn_output),
        "--output",
        str(rule_library),
        "--min-f1",
        str(min_f1),
        "--json",
    ]
    return run_command(cmd, cwd=code_root, output_dir=rule_library.parent, timeout=300)


def run_reason_eval(code_root: Path, python: Path, test_task: Path, rule_library: Path, output: Path, max_proofs: int) -> dict:
    cmd = [
        str(python),
        "-m",
        "para.cli",
        "reason-eval",
        "--task-dir",
        str(test_task),
        "--rule-library",
        str(rule_library),
        "--max-proofs",
        str(max_proofs),
        "--output",
        str(output),
        "--json",
    ]
    return run_command(cmd, cwd=code_root, output_dir=output.parent, timeout=300)


def run_command(cmd: Sequence[str], cwd: Path, output_dir: Path, timeout: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")
        stderr += f"\nRUNNER_TIMEOUT after {timeout} seconds\n"
        returncode = -9
    elapsed = time.perf_counter() - started
    command_name = safe_name(cmd[4] if len(cmd) > 4 else "command")
    stamp = str(int(started * 1000))
    (output_dir / f"{command_name}_{stamp}.stdout.log").write_text(stdout, encoding="utf-8")
    (output_dir / f"{command_name}_{stamp}.stderr.log").write_text(stderr, encoding="utf-8")
    parsed = parse_last_json(stdout)
    return {
        "command": list(cmd),
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": elapsed,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
        "json": parsed,
    }


def parse_last_json(text: str) -> object | None:
    text = text.strip()
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError:
        return None


def write_empty_rule_library(path: Path, min_f1: float) -> None:
    payload = {
        "schema_version": 2,
        "library_kind": "ashrl_reasoning_rule_library",
        "min_f1": min_f1,
        "rule_count": 0,
        "dependency_graph": {
            "idb_predicates": [],
            "edb_predicates": [],
            "dependencies": {},
            "recursive_predicates": [],
        },
        "rules": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def compact_report(manifest: dict) -> str:
    learn_json = (manifest.get("learn_summary") or {}).get("json") or {}
    export_json = (manifest.get("export_summary") or {}).get("json") or {}
    reason_json = (manifest.get("reason_summary") or {}).get("json") or {}
    validity = manifest.get("validity") or {}
    lines = [
        f"Protocol: {manifest.get('protocol')}",
        f"Split: {manifest.get('split_name')}",
        f"Train/Test: {manifest.get('split')}",
        f"Valid for paper: {manifest.get('valid_for_paper')} category={validity.get('category')}",
        f"Learn status: {learn_json.get('status')} f1={metric_value(learn_json, 'f1')}",
        f"Exported rules: {export_json.get('rule_count')}",
        (
            "Reason test: "
            f"examples={reason_json.get('examples')} "
            f"acc={reason_json.get('held_out_accuracy', reason_json.get('three_value_accuracy'))} "
            f"supported_precision={reason_json.get('supported_precision')} "
            f"negative_non_support={reason_json.get('negative_non_support_rate')} "
            f"abstention={reason_json.get('inconclusive_rate')}"
        ),
        f"Manifest: {manifest.get('output_root')}/manifests/{manifest.get('split_name')}_manifest.json",
    ]
    if validity.get("invalid_for_paper"):
        lines.append(f"Invalid detail: {str(validity.get('detail', ''))[:300]}")
    if manifest.get("no_rule_summary"):
        no_rule_json = (manifest.get("no_rule_summary") or {}).get("json") or {}
        lines.append(
            "No-rule: "
            f"examples={no_rule_json.get('examples')} "
            f"acc={no_rule_json.get('held_out_accuracy', no_rule_json.get('three_value_accuracy'))} "
            f"abstention={no_rule_json.get('inconclusive_rate')}"
        )
    return "\n".join(lines)


def metric_value(summary: dict, key: str) -> object:
    metrics = summary.get("metrics") or summary.get("best_candidate_metrics") or {}
    return metrics.get(key)


if __name__ == "__main__":
    main()
