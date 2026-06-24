"""Popper subprocess integration."""

from __future__ import annotations

import re
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .models import Rule
from .prolog import parse_rule


@dataclass
class PopperResult:
    returncode: int
    elapsed_seconds: float
    stdout: str
    stderr: str
    rules: List[Rule] = field(default_factory=list)
    error: Optional[str] = None


def run_popper(popper_path: str | Path, task_dir: str | Path, timeout: int = 300) -> PopperResult:
    """Run Popper and parse any learned rules from stdout."""

    start = time.perf_counter()
    requested = str(popper_path)
    popper = Path(requested).expanduser()
    if popper.exists():
        executable = str(popper.resolve())
    else:
        executable = shutil.which(requested) or ""
    if not executable:
        # 不直接抛异常，是为了让 pipeline 可以继续评估 LLM 候选规则。
        # 这在没有安装 Popper 的机器上可以作为 LLM-only 降级基线。
        elapsed = time.perf_counter() - start
        return PopperResult(
            returncode=127,
            elapsed_seconds=elapsed,
            stdout="",
            stderr="",
            error=(
                f"Popper was not found: {requested}. Set PARA_POPPER_PATH, "
                "pass --popper-path, or install popper.py on PATH."
            ),
        )

    cmd = [sys.executable, executable, str(task_dir), "--timeout", str(timeout)]
    proc: subprocess.Popen | None = None
    try:
        # Popper 自己也有 --timeout；subprocess timeout 多给 10 秒，
        # 用于让 Python 进程有时间清理和输出错误信息。
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        stdout, stderr = proc.communicate(timeout=timeout + 10)
        elapsed = time.perf_counter() - start
        return PopperResult(
            returncode=proc.returncode or 0,
            elapsed_seconds=elapsed,
            stdout=stdout,
            stderr=stderr,
            rules=extract_rules(stdout),
        )
    except subprocess.TimeoutExpired as exc:
        if proc is not None:
            _terminate_process_group(proc)
            stdout, stderr = proc.communicate()
        else:
            stdout, stderr = _ensure_text(exc.stdout), _ensure_text(exc.stderr)
        # 超时是实验指标的一部分，因此返回结构化结果而不是中断整个批量实验。
        elapsed = time.perf_counter() - start
        return PopperResult(
            returncode=124,
            elapsed_seconds=elapsed,
            stdout=_ensure_text(stdout),
            stderr=_ensure_text(stderr),
            error=f"Popper timed out after {timeout}s",
        )


def extract_rules(output: str) -> List[Rule]:
    """Extract Horn clauses from Popper output."""

    rules: List[Rule] = []
    for raw in output.splitlines():
        line = raw.strip()

        # Popper 输出中混有搜索日志、分隔线和统计信息；
        # 这里只抽取形如 `target(A,B):- body(...).` 的 Horn clause。
        if not line or line.startswith(("%", "[", "********", "Program")):
            continue
        if ":-" not in line or not line.endswith("."):
            continue
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*\(", line):
            continue
        try:
            rules.append(parse_rule(line, source="popper"))
        except ValueError:
            # 单行解析失败不代表 Popper 失败，跳过该行继续抽取后续规则。
            continue
    return rules


def _ensure_text(value) -> str:
    """Normalize subprocess timeout output to text.

    Python may return bytes from TimeoutExpired even when subprocess.run uses
    text=True. Batch experiments should record the timeout, not crash while
    formatting stderr/stdout tails.
    """

    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """Kill Popper and solver descendants after a hard timeout."""

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
