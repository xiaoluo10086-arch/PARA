#!/usr/bin/env python3
"""Export paper-facing PARA artifacts without local paths.

The final experiment workspace contains large raw logs, manifests, and frozen
task directories. This script copies only the compact evidence products that
belong in the public repository and rewrites local filesystem references into
artifact-relative placeholders.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any


ABS_PATH_RE = re.compile(r"(/(?:home|Users)/[^,\s\"']+)")
WORKSPACE_RE = re.compile(r"(/(?:home|Users)/[^,\s\"']+/ILPandLLM)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-results",
        type=Path,
        required=True,
        help="Root directory containing final paper result subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/final_icse2027"),
        help="Destination directory inside the public artifact repository.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source_results.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    copy_table_group(
        source / "unified_rerun_20260627",
        output / "unified_rerun",
        [
            "README.md",
            "unified_core_summary_by_target_method.csv",
            "unified_core_summary_by_target_method.json",
            "unified_core_matrix.csv",
            "unified_core_matrix.json",
        ],
        source,
    )
    copy_table_group(
        source / "high_complexity_multiseed_20260627",
        output / "high_complexity_multiseed",
        [
            "high_complexity_multiseed_summary.csv",
            "high_complexity_multiseed_summary.json",
            "high_complexity_multiseed_rows.csv",
        ],
        source,
    )
    export_contracts(
        source / "high_complexity_three_target_contracts_20260627",
        output / "proof_contracts",
        source,
    )
    write_readme(output)


def copy_table_group(src: Path, dst: Path, names: list[str], source_root: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.mkdir(parents=True, exist_ok=True)
    for name in names:
        src_file = src / name
        if not src_file.exists():
            continue
        dst_file = dst / name
        if src_file.suffix == ".json":
            payload = json.loads(src_file.read_text(encoding="utf-8"))
            write_json(dst_file, sanitize_value(payload, source_root))
        elif src_file.suffix == ".csv":
            sanitize_csv(src_file, dst_file, source_root)
        else:
            text = src_file.read_text(encoding="utf-8")
            dst_file.write_text(sanitize_text(text, source_root), encoding="utf-8")


def export_contracts(src: Path, dst: Path, source_root: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.mkdir(parents=True, exist_ok=True)

    summary_csv = src / "contract_summary.csv"
    if summary_csv.exists():
        rows: list[dict[str, str]] = []
        with summary_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                split = row.get("split_name", "unknown_split")
                original = Path(row.get("contract_file", "proof_contract.json")).name
                row["contract_file"] = f"proof_contracts/{split}/{original}"
                rows.append({key: sanitize_text(value, source_root) for key, value in row.items()})
        write_csv(dst.parent / "contract_summary.csv", rows)

    summary_json = src / "contract_summary.json"
    if summary_json.exists():
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        payload = sanitize_value(payload, source_root)
        for item in payload if isinstance(payload, list) else []:
            if isinstance(item, dict) and "contract_file" in item:
                split = item.get("split_name", "unknown_split")
                item["contract_file"] = f"proof_contracts/{split}/{Path(str(item['contract_file'])).name}"
        write_json(dst.parent / "contract_summary.json", payload)

    for contract in sorted(src.glob("*/proof_contract_*.json")):
        target_dir = dst / contract.parent.name
        target_dir.mkdir(parents=True, exist_ok=True)
        payload = json.loads(contract.read_text(encoding="utf-8"))
        write_json(target_dir / contract.name, sanitize_value(payload, source_root))

    for contract_summary in sorted(src.glob("*/proof_contract_summary.json")):
        target_dir = dst / contract_summary.parent.name
        target_dir.mkdir(parents=True, exist_ok=True)
        payload = json.loads(contract_summary.read_text(encoding="utf-8"))
        write_json(target_dir / contract_summary.name, sanitize_value(payload, source_root))


def sanitize_csv(src: Path, dst: Path, source_root: Path) -> None:
    with src.open(newline="", encoding="utf-8") as handle:
        rows = [
            {key: sanitize_text(value, source_root) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    write_csv(dst, rows)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sanitize_value(value: Any, source_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_value(item, source_root) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item, source_root) for item in value]
    if isinstance(value, str):
        return sanitize_text(value, source_root)
    return value


def sanitize_text(text: str | None, source_root: Path) -> str:
    if text is None:
        return ""
    sanitized = str(text)
    source_string = str(source_root)
    if source_string in sanitized:
        sanitized = sanitized.replace(source_string, "<RESULTS_ROOT>")
    sanitized = sanitized.replace(str(Path.home()), "<HOME>")
    sanitized = sanitized.replace("\\", "/")
    sanitized = WORKSPACE_RE.sub("<WORKSPACE_ROOT>", sanitized)
    sanitized = sanitized.replace(
        "<HOME>/ILPandLLM/Agent‑Symbolic_Hybrid_Rule_Learning/paper_reasoning/proof_strategy_agent_v2",
        "<ARTIFACT_ROOT>/artifacts/proof_strategy_agent",
    )
    sanitized = sanitized.replace(
        "Agent‑Symbolic_Hybrid_Rule_Learning/paper_reasoning/proof_strategy_agent_v2",
        "artifacts/proof_strategy_agent",
    )
    sanitized = sanitized.replace(
        "<HOME>/ILPandLLM/Agent‑Symbolic_Hybrid_Rule_Learning/paper_reasoning/paper_final_icse2027",
        "<ARTIFACT_ROOT>/artifacts/paper_final_icse2027",
    )
    sanitized = sanitized.replace(
        "Agent‑Symbolic_Hybrid_Rule_Learning/paper_reasoning/paper_final_icse2027",
        "artifacts/paper_final_icse2027",
    )
    sanitized = sanitized.replace("<HOME>/ILPandLLM", "<ARTIFACT_ROOT>")
    return ABS_PATH_RE.sub("<ABSOLUTE_PATH>", sanitized)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_readme(output: Path) -> None:
    readme = """# ICSE 2027 Paper-Facing Results

This directory contains compact, sanitized evidence products used by the PARA
paper. It intentionally excludes raw provider responses, API preflight files,
third-party source checkouts, frozen full task splits, local logs, and build
outputs.

## Contents

- `unified_rerun/`: canonical relation matrix over Spring targets and methods.
- `high_complexity_multiseed/`: multi-seed high-complexity target summaries.
- `contract_summary.{csv,json}`: sampled query-level proof-contract inventory.
- `proof_contracts/`: representative positive and negative proof contracts.

All local filesystem references are rewritten to placeholders such as
`<RESULTS_ROOT>` or `<WORKSPACE_ROOT>`. To reproduce from raw artifacts, unpack
the frozen artifact bundle and run:

```bash
python scripts/final_icse2027/export_public_artifacts.py \\
  --source-results artifacts/paper_final_icse2027/results \\
  --output-dir results/final_icse2027
```
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
