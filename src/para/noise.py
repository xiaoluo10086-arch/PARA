"""Noise injection utilities for systematic experiments."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import List


def inject_noise(task_dir: str | Path, output_dir: str | Path, kind: str, rate: float, seed: int = 42) -> Path:
    """Create a noisy copy of a Popper task directory.

    Supported kinds:
    - `label_flip`: swap a fraction of pos/neg examples.
    - `missing_bk`: remove a fraction of background facts.
    - `irrelevant_bk`: append irrelevant facts.
    """

    if not 0 <= rate <= 1:
        raise ValueError("rate must be in [0, 1]")

    src = Path(task_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 先复制一份完整任务，再只在副本中注入噪声。
    # 原始 clean 数据集必须保持不变，便于做 0% 噪声对照组。
    for name in ("bk.pl", "exs.pl", "bias.pl"):
        shutil.copy2(src / name, out / name)

    # 固定 seed 是为了让论文实验可复现。
    rnd = random.Random(seed)
    report = {"kind": kind, "rate": rate, "seed": seed}
    if kind == "label_flip":
        changed = _flip_labels(out / "exs.pl", rate, rnd)
    elif kind == "missing_bk":
        changed = _remove_bk_facts(out / "bk.pl", rate, rnd)
    elif kind == "irrelevant_bk":
        changed = _add_irrelevant_facts(out / "bk.pl", rate, rnd)
    else:
        raise ValueError(f"Unsupported noise kind: {kind}")

    report["changed_items"] = changed
    (out / "noise_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return out


def _flip_labels(path: Path, rate: float, rnd: random.Random) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()

    # 标注错误噪声：只修改 pos/neg 标签，不改变目标谓词参数。
    indices = [idx for idx, line in enumerate(lines) if line.strip().startswith(("pos(", "neg("))]
    count = int(round(len(indices) * rate))
    for idx in rnd.sample(indices, count):
        stripped = lines[idx].lstrip()
        prefix_len = len(lines[idx]) - len(stripped)
        prefix = lines[idx][:prefix_len]
        if stripped.startswith("pos("):
            lines[idx] = prefix + "neg(" + stripped[4:]
        elif stripped.startswith("neg("):
            lines[idx] = prefix + "pos(" + stripped[4:]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return count


def _remove_bk_facts(path: Path, rate: float, rnd: random.Random) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()

    # 信息缺失噪声：从背景知识中删除一部分地面事实。
    # 规则、注释和 Prolog 指令不参与删除。
    fact_indices = [
        idx
        for idx, line in enumerate(lines)
        if line.strip().endswith(".") and not line.strip().startswith("%") and ":-" not in line
    ]
    count = int(round(len(fact_indices) * rate))
    remove = set(rnd.sample(fact_indices, count))
    kept = [line for idx, line in enumerate(lines) if idx not in remove]
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return count


def _add_irrelevant_facts(path: Path, rate: float, rnd: random.Random) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()

    # 无关信息噪声：添加不会出现在 bias.pl body_pred 中的事实。
    # Popper 通常不会直接使用它们，但它们可以模拟大型事实库中的干扰信息。
    fact_count = sum(
        1 for line in lines if line.strip().endswith(".") and not line.strip().startswith("%") and ":-" not in line
    )
    count = int(round(fact_count * rate))
    additions = [f"irrelevantFact(noise_{i}, noise_{rnd.randint(0, 999999)})." for i in range(count)]
    path.write_text("\n".join(lines + [""] + additions) + "\n", encoding="utf-8")
    return count
