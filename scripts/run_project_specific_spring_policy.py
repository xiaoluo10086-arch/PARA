#!/usr/bin/env python3
"""Build and run a project-specific Spring architecture-policy task.

The task is intentionally separate from the main paper pipeline.  It derives a
Spring-specific target relation, ``springSameModuleUse/2``, from the existing
Spring static graph:

    springSameModuleUse(PkgA, PkgB)

holds when a class in ``PkgA`` imports a class in ``PkgB`` and both packages
belong to the same Spring top-level module (for example, ``beans`` or
``context``).  The module labels are project-specific facts inferred from
Spring package names, not Java language semantics.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "project_specific_spring_policy"
DEFAULT_RUNNER = ROOT / "scripts" / "run_learn_then_reason.py"


FACT_RE = re.compile(r"^(\w+)\((.*)\)\.$")


def main() -> None:
    args = build_parser().parse_args()
    source_task = Path(args.source_task).resolve()
    output_root = Path(args.output_root).resolve()
    task_dir = output_root / "task_spring_same_module_use"
    output_root.mkdir(parents=True, exist_ok=True)

    task_summary = build_task(source_task=source_task, task_dir=task_dir)
    summary_path = output_root / "task_summary.json"
    summary_path.write_text(json.dumps(task_summary, indent=2, ensure_ascii=False) + "\n")

    command = [
        sys.executable,
        str(Path(args.runner).resolve()),
        "--task-dir",
        str(task_dir),
        "--output-root",
        str(output_root / "learn_then_reason"),
        "--split-name",
        "spring_same_module_use_train50_100_seed0_graphrag",
        "--train-pos",
        str(args.train_pos),
        "--train-neg",
        str(args.train_neg),
        "--seed",
        str(args.seed),
        "--guide-provider",
        "graphrag",
        "--min-f1",
        "0.8",
        "--graphrag-constraint-mode",
        "attribute",
        "--graphrag-max-depth",
        "5",
        "--graphrag-max-candidates",
        "40",
        "--run-no-rule",
        "--json",
    ]
    result = subprocess.run(command, text=True, capture_output=True, timeout=args.timeout)
    run_summary = {
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-6000:],
        "stderr_tail": result.stderr[-6000:],
    }
    (output_root / "run_summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False) + "\n")
    if result.returncode != 0:
        print(json.dumps(run_summary, indent=2, ensure_ascii=False))
        raise SystemExit(result.returncode)

    report = summarize_results(output_root)
    report_path = output_root / "PROJECT_SPECIFIC_SPRING_POLICY_REPORT.md"
    report_path.write_text(report)
    print(report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-task",
        required=True,
        help="Spring isAllowedToUse task containing bk.pl, exs.pl, and bias.pl",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--runner", default=str(DEFAULT_RUNNER))
    parser.add_argument("--train-pos", type=int, default=50)
    parser.add_argument("--train-neg", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1200)
    return parser


def build_task(source_task: Path, task_dir: Path) -> dict:
    task_dir.mkdir(parents=True, exist_ok=True)
    facts = parse_facts(source_task / "bk.pl")
    contains_class = {cls: pkg for pkg, cls in facts.get("containsClass", [])}
    package_imports = sorted(
        {
            (contains_class[src_cls], contains_class[dst_cls])
            for src_cls, dst_cls in facts.get("importsClass", [])
            if src_cls in contains_class
            and dst_cls in contains_class
            and contains_class[src_cls] != contains_class[dst_cls]
        }
    )
    package_names = sorted({pkg for pair in package_imports for pkg in pair})
    module_by_package = {pkg: spring_module(pkg) for pkg in package_names}

    positives = [(a, b) for a, b in package_imports if module_by_package[a] == module_by_package[b]]
    negative_import_cross_module = [
        (a, b) for a, b in package_imports if module_by_package[a] != module_by_package[b]
    ]
    by_module: dict[str, list[str]] = {}
    for package, module in module_by_package.items():
        by_module.setdefault(module, []).append(package)
    negative_same_module_no_import: list[tuple[str, str]] = []
    import_edge_set = set(package_imports)
    for packages in by_module.values():
        ordered = sorted(packages)
        for source in ordered:
            for target in ordered:
                if source != target and (source, target) not in import_edge_set:
                    negative_same_module_no_import.append((source, target))
    negatives = sorted(set(negative_import_cross_module + negative_same_module_no_import))
    if len(positives) < 60 or len(negatives) < 120:
        raise RuntimeError(f"insufficient examples: pos={len(positives)} neg={len(negatives)}")

    shutil.copy2(source_task / "bk.pl", task_dir / "bk.pl")
    append_project_specific_facts(task_dir / "bk.pl", module_by_package)
    write_examples(task_dir / "exs.pl", positives, negatives)
    write_bias(task_dir / "bias.pl")
    metadata = {
        "target": "springSameModuleUse",
        "source_task": str(source_task),
        "task_dir": str(task_dir),
        "definition": "PkgA imports PkgB through class-level imports and both packages share the same Spring top-level module.",
        "project_specific_facts": "springModuleName(Package, Module) derived from Spring package naming.",
        "positive_examples": len(positives),
        "negative_examples": len(negatives),
        "negative_import_cross_module": len(negative_import_cross_module),
        "negative_same_module_no_import": len(negative_same_module_no_import),
        "package_import_edges": len(package_imports),
        "packages_with_module_names": len(module_by_package),
        "modules": sorted(set(module_by_package.values())),
        "example_positives": positives[:10],
        "example_negatives": negatives[:10],
    }
    (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    return metadata


def parse_facts(path: Path) -> dict[str, list[tuple[str, ...]]]:
    facts: dict[str, list[tuple[str, ...]]] = {}
    for line in path.read_text().splitlines():
        match = FACT_RE.match(line.strip())
        if not match:
            continue
        predicate = match.group(1)
        args = tuple(part.strip() for part in match.group(2).split(","))
        if "_" in args:
            continue
        facts.setdefault(predicate, []).append(args)
    return facts


def spring_module(package: str) -> str:
    prefix = "org_springframework_"
    if not package.startswith(prefix):
        return "external"
    rest = package[len(prefix) :]
    return rest.split("_", 1)[0] if rest else "root"


def append_project_specific_facts(bk_path: Path, module_by_package: dict[str, str]) -> None:
    with bk_path.open("a", encoding="utf-8") as handle:
        handle.write("\n% Project-specific Spring module-role facts for supplemental ICSE experiment.\n")
        handle.write("springModuleName(_,_) :- fail.\n")
        for package, module in sorted(module_by_package.items()):
            handle.write(f"springModuleName({package},spring_module_{module}).\n")


def write_examples(path: Path, positives: list[tuple[str, str]], negatives: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("% Project-specific Spring module-use examples.\n")
        for a, b in positives:
            handle.write(f"pos(springSameModuleUse({a},{b})).\n")
        for a, b in negatives:
            handle.write(f"neg(springSameModuleUse({a},{b})).\n")


def write_bias(path: Path) -> None:
    path.write_text(
        """% Bias for project-specific Spring same-module use policy.
head_pred(springSameModuleUse,2).
body_pred(package,1).
body_pred(class,1).
body_pred(method,1).
body_pred(containsPackage,2).
body_pred(containsClass,2).
body_pred(containsMethod,2).
body_pred(importsClass,2).
body_pred(extendsClass,2).
body_pred(implementsInterface,2).
body_pred(inheritsClass,2).
body_pred(callsMethod,2).
body_pred(methodName,2).
body_pred(methodArity,2).
body_pred(sameMethodName,2).
body_pred(sameMethodArity,2).
body_pred(springModuleName,2).

type(springSameModuleUse,(package,package)).
type(package,(package,)).
type(class,(class,)).
type(method,(method,)).
type(containsPackage,(package,package)).
type(containsClass,(package,class)).
type(containsMethod,(class,method)).
type(importsClass,(class,class)).
type(extendsClass,(class,class)).
type(implementsInterface,(class,class)).
type(inheritsClass,(class,class)).
type(callsMethod,(method,method)).
type(methodName,(method,name)).
type(methodArity,(method,arity)).
type(sameMethodName,(method,method)).
type(sameMethodArity,(method,method)).
type(springModuleName,(package,name)).

max_vars(8).
max_body(8).
max_clauses(4).
""",
        encoding="utf-8",
    )


def summarize_results(output_root: Path) -> str:
    task_summary = json.loads((output_root / "task_summary.json").read_text())
    manifest_path = output_root / "learn_then_reason" / "manifests" / "spring_same_module_use_train50_100_seed0_graphrag_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    learn = manifest.get("learn_summary", {}).get("json", {})
    reason_path = Path(manifest["reason_output"])
    no_rule_path = Path(manifest["no_rule_output"])
    reason = json.loads(reason_path.read_text())
    no_rule = json.loads(no_rule_path.read_text())
    learned_rules = Path(manifest["learn_output"]) / "learned_rules.pl"
    rule_text = learned_rules.read_text().strip() if learned_rules.exists() else ""

    def metrics(obj: dict) -> dict:
        if "metrics" in obj:
            return obj["metrics"]
        counts = obj.get("counts", {})
        tp = counts.get("supported_positive", 0)
        fp = counts.get("supported_negative", 0)
        fn = obj.get("positive_examples", 0) - tp
        tn = obj.get("negative_examples", 0) - fp
        precision = obj.get("supported_precision", 0.0)
        recall = obj.get("supported_recall", 0.0)
        f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
        return {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "accuracy": obj.get("held_out_accuracy", obj.get("three_value_accuracy", 0.0)),
            "f1": f1,
        }

    train_metrics = learn.get("metrics", {})
    test_metrics = metrics(reason)
    no_rule_metrics = metrics(no_rule)
    return f"""# Project-Specific Spring Policy Supplemental Experiment

## Purpose

This supplemental experiment addresses the reviewer-risk that the main targets
(`canCallClass`, `isAllowedToUse`, `overridesMethod`) look like generic Java
relations.  It constructs a Spring-specific target, `springSameModuleUse/2`,
whose label depends on Spring's project module naming rather than Java language
semantics.

## Target

`springSameModuleUse(PkgA, PkgB)` means:

1. a class in `PkgA` imports a class in `PkgB`; and
2. `PkgA` and `PkgB` share the same Spring top-level module label, represented
   by the project-specific EDB fact `springModuleName(Pkg, Module)`.

This is not a universal Java relation.  A hand-written checker would need a
Spring-specific module taxonomy; PARA receives the taxonomy as graph facts and
must learn the evidence path and equality-style module constraint from examples.

## Configuration

| Item | Value |
|---|---|
| Source graph | `{task_summary['source_task']}` |
| Derived task | `{task_summary['task_dir']}` |
| Package import edges | {task_summary['package_import_edges']} |
| Packages with module labels | {task_summary['packages_with_module_names']} |
| Positive examples | {task_summary['positive_examples']} |
| Negative examples | {task_summary['negative_examples']} |
| Negative import edges across modules | {task_summary['negative_import_cross_module']} |
| Negative same-module non-import pairs | {task_summary['negative_same_module_no_import']} |
| Train split | 50 positive / 100 negative |
| Held-out split | {manifest['split']['test_positive']} positive / {manifest['split']['test_negative']} negative |
| Guide provider | GraphRAG candidate generator with attribute constraints |
| Acceptance threshold | F1 >= 0.8 |

## Learned Rule

```prolog
{rule_text}
```

## Results

| Stage | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train verification | {train_metrics.get('accuracy', 0):.3f} | {train_metrics.get('precision', 0):.3f} | {train_metrics.get('recall', 0):.3f} | {train_metrics.get('f1', 0):.3f} | {train_metrics.get('tp', 0)} | {train_metrics.get('fp', 0)} | {train_metrics.get('tn', 0)} | {train_metrics.get('fn', 0)} |
| Held-out reasoning | {test_metrics.get('accuracy', 0):.3f} | {test_metrics.get('precision', 0):.3f} | {test_metrics.get('recall', 0):.3f} | {test_metrics.get('f1', 0):.3f} | {test_metrics.get('tp', 0)} | {test_metrics.get('fp', 0)} | {test_metrics.get('tn', 0)} | {test_metrics.get('fn', 0)} |
| No-rule policy | {no_rule_metrics.get('accuracy', 0):.3f} | {no_rule_metrics.get('precision', 0):.3f} | {no_rule_metrics.get('recall', 0):.3f} | {no_rule_metrics.get('f1', 0):.3f} | {no_rule_metrics.get('tp', 0)} | {no_rule_metrics.get('fp', 0)} | {no_rule_metrics.get('tn', 0)} | {no_rule_metrics.get('fn', 0)} |

## Interpretation

Use this result as a supplemental project-specific relation target, not as a
replacement for the main Spring xlarge3 evaluation.  It supports the revised
paper claim that PARA can operate on project-specific typed facts and discover
an auditable path-program rule for a relation whose meaning is not built into
the Java language.

## Files

- Task summary: `{output_root / 'task_summary.json'}`
- Manifest: `{manifest_path}`
- Reasoning output: `{reason_path}`
- No-rule output: `{no_rule_path}`
"""


if __name__ == "__main__":
    main()
