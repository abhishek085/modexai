"""Describe the machine a benchmark ran on, so a recipe says *where* it works — not just *that* it works.

Only coarse, non-identifying facts are collected: chip/GPU model, memory size, OS family, architecture.
No hostnames, serial numbers, usernames or IPs.
"""

from __future__ import annotations

import contextlib
import os
import platform
import re
import shutil
import subprocess
from functools import lru_cache
from typing import Any


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


@lru_cache(maxsize=1)
def detect() -> dict[str, Any]:
    system = platform.system()
    info: dict[str, Any] = {
        "os": {"Darwin": "macOS", "Linux": "Linux", "Windows": "Windows"}.get(system, system),
        "arch": platform.machine().lower().replace("amd64", "x86_64"),
        "chip": platform.processor() or "unknown",
        "ram_gb": None,
        "gpu": None,
        "vram_gb": None,
        "accelerator": "cpu",
    }
    if system == "Darwin":
        info["chip"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or info["chip"]
        mem = _run(["sysctl", "-n", "hw.memsize"])
        if mem.isdigit():
            info["ram_gb"] = round(int(mem) / 2**30)
        if info["arch"] == "arm64":
            info["accelerator"] = "apple-silicon"
            info["gpu"] = info["chip"]
            info["vram_gb"] = info["ram_gb"]  # unified memory
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                m = re.search(r"model name\s*:\s*(.+)", f.read())
                if m:
                    info["chip"] = m.group(1).strip()
            with open("/proc/meminfo", encoding="utf-8") as f:
                m = re.search(r"MemTotal:\s*(\d+)", f.read())
                if m:
                    info["ram_gb"] = round(int(m.group(1)) / 2**20)
        except OSError:
            pass
    elif system == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            info["ram_gb"] = round(stat.ullTotalPhys / 2**30)
        except Exception:
            pass

    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        if out:
            name, _, mem = out.splitlines()[0].partition(",")
            info["gpu"] = name.strip()
            with contextlib.suppress(ValueError):
                info["vram_gb"] = round(float(mem) / 1024)
            info["accelerator"] = "cuda"
    return info


def label(hw: dict[str, Any] | None = None) -> str:
    """Short human label, e.g. 'Apple M4 Pro · 24 GB' or 'RTX 4090 24 GB · 64 GB RAM'."""
    hw = hw or detect()
    if hw.get("accelerator") == "apple-silicon":
        return f"{hw.get('chip', 'Apple Silicon')} · {hw.get('ram_gb')} GB"
    if hw.get("accelerator") == "cuda":
        gpu = (hw.get("gpu") or "GPU").replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
        return f"{gpu} {hw.get('vram_gb')} GB · {hw.get('ram_gb')} GB RAM"
    return f"{hw.get('chip', 'CPU')} · {hw.get('ram_gb')} GB RAM"


def memory_budget_gb(hw: dict[str, Any] | None = None) -> float:
    """Rough upper bound on model size that runs comfortably here (weights + KV cache + headroom)."""
    hw = hw or detect()
    override = os.environ.get("MODEXAI_MAX_MODEL_GB")
    if override:
        return float(override)
    if hw.get("accelerator") == "cuda" and hw.get("vram_gb"):
        return round(hw["vram_gb"] * 0.85, 1)
    if hw.get("ram_gb"):
        share = 0.6 if hw.get("accelerator") == "apple-silicon" else 0.5
        return round(hw["ram_gb"] * share, 1)
    return 4.0


def compatible(recipe_hw: dict[str, Any], mine: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Would a recipe measured on ``recipe_hw`` plausibly run here? Returns (ok, reason)."""
    mine = mine or detect()
    need = recipe_hw.get("model_size_gb")
    if need and need > memory_budget_gb(mine):
        return False, f"needs ~{need} GB, this machine's budget is {memory_budget_gb(mine)} GB"
    same_family = recipe_hw.get("accelerator") == mine.get("accelerator")
    return True, "same hardware family" if same_family else "different hardware — expect different speed"
