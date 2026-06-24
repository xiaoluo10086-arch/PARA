#!/usr/bin/env python3
"""Audit exact train/test input identity for matched PARA experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


AUDITED_FILES = ("bk.pl", "exs.pl", "bias.pl")


def main() -> None:
    args = build_parser().parse_args()
    left = Path(args.left).resolve()
    right = Path(args.right).resolve()
    rows = []
    for partition in ("train_task", "test_task"):
        for name in AUDITED_FILES:
            left_path = left / partition / name
            right_path = right / partition / name
            left_hash = sha256(left_path)
            right_hash = sha256(right_path)
            rows.append(
                {
                    "partition": partition,
                    "file": name,
                    "left": str(left_path),
                    "right": str(right_path),
                    "left_sha256": left_hash,
                    "right_sha256": right_hash,
                    "match": left_hash == right_hash,
                }
            )
    report = {
        "left": str(left),
        "right": str(right),
        "identity_rule": "exact_sha256",
        "matched": all(row["match"] for row in rows),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["matched"]:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit matched PARA split inputs")
    parser.add_argument("--left", required=True, help="Split root containing train_task and test_task")
    parser.add_argument("--right", required=True, help="Split root containing train_task and test_task")
    parser.add_argument("--output", required=True)
    return parser


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
