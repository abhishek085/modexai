"""Shadow → hire → delegate → train: let a local model apprentice under your agent.

1. **Shadow** — while a strong model (Claude, GPT, …) does a narrow, repetitive job, record each input
   and the answer it gave. Stored only on this machine, in ``~/.modexai/tasks/<task>.jsonl``.
2. **Hire** — once there are enough examples, test small local models against those answers and hire
   the smallest one that agrees often enough.
3. **Delegate** — route that task to the hire from now on (``ask_local_model(task=...)``).
4. **Train** — if no listed model is good enough, export the examples as a fine-tuning dataset.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import date
from pathlib import Path
from typing import Any

from . import config
from .backends import Backend
from .bench import ModelResult, Verdict, run_benchmark, verdict
from .registry import Registry
from .suites import Suite, suite_from_samples

MAX_EXAMPLES = 5000


def _tasks_dir() -> Path:
    p = config.home() / "tasks"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _task_path(task: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in task.lower()).strip("-")
    if not safe:
        raise ValueError("task name must contain letters or digits")
    return _tasks_dir() / f"{safe}.jsonl"


def _meta_path(task: str) -> Path:
    return _task_path(task).with_suffix(".meta.json")


def record(task: str, prompt: str, output: str, system: str | None = None, reference: str | None = None) -> dict:
    """Store one (input, strong-model answer) pair. Duplicates are ignored."""
    path = _task_path(task)
    h = hashlib.sha256(f"{system or ''}\x00{prompt}".encode()).hexdigest()[:16]
    existing = examples(task)
    if any(e.get("hash") == h for e in existing):
        return {"task": task, "recorded": False, "reason": "duplicate", "examples": len(existing)}
    if len(existing) >= MAX_EXAMPLES:
        return {"task": task, "recorded": False, "reason": f"limit of {MAX_EXAMPLES} reached", "examples": len(existing)}
    row = {"hash": h, "prompt": prompt, "expected": output, "ts": int(time.time())}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    meta = task_meta(task)
    if system and not meta.get("system"):
        meta["system"] = system
    if reference:
        meta["reference"] = reference
    _meta_path(task).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"task": task, "recorded": True, "examples": len(existing) + 1}


def examples(task: str) -> list[dict[str, Any]]:
    path = _task_path(task)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def task_meta(task: str) -> dict[str, Any]:
    p = _meta_path(task)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def list_tasks() -> list[dict[str, Any]]:
    out = []
    hires = load_hires()
    for p in sorted(_tasks_dir().glob("*.jsonl")):
        task = p.stem
        n = sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
        out.append({"task": task, "examples": n, "hired": hires.get(task, {}).get("model")})
    return out


def forget(task: str) -> bool:
    removed = False
    for p in (_task_path(task), _meta_path(task)):
        if p.exists():
            p.unlink()
            removed = True
    hires = load_hires()
    if hires.pop(task, None):
        _save_hires(hires)
    return removed


def task_suite(task: str, holdout: int | None = None) -> Suite:
    ex = examples(task)
    if not ex:
        raise ValueError(f"no examples recorded for task {task!r} yet — shadow it first (record_example)")
    if holdout:
        ex = ex[-holdout:]
    meta = task_meta(task)
    return suite_from_samples(ex, name=task, system=meta.get("system"))


# ---------------------------------------------------------------------------
# Hire
# ---------------------------------------------------------------------------


def _hires_path() -> Path:
    return config.home() / "hires.json"


def load_hires() -> dict[str, Any]:
    p = _hires_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save_hires(h: dict[str, Any]) -> None:
    _hires_path().write_text(json.dumps(h, indent=2), encoding="utf-8")


def hire(
    task: str,
    candidates: list[str],
    backend: Backend,
    registry: Registry | None = None,
    min_score: float = 0.9,
    limit: int | None = 50,
    on_event=None,
    min_examples: int = 10,
) -> tuple[Verdict, list[ModelResult], Suite]:
    """Test candidates against the strong model's recorded answers and hire the smallest that agrees."""
    n = len(examples(task))
    if n < min_examples:
        raise ValueError(f"only {n} examples for {task!r}; record at least {min_examples} before hiring")
    suite = task_suite(task, holdout=limit)
    results = run_benchmark(candidates, suite, backend, registry=registry, on_event=on_event)
    v = verdict(results, min_score=min_score)
    if v.winner:
        hires = load_hires()
        hires[task] = {
            "model": v.winner.model,
            "agreement": v.winner.score,
            "cases": v.winner.total,
            "p50_ms": v.winner.p50_ms,
            "system": task_meta(task).get("system"),
            "json_mode": suite.json_mode,
            "hired": date.today().isoformat(),
            "reference": task_meta(task).get("reference"),
        }
        _save_hires(hires)
    return v, results, suite


def hired_model(task: str) -> dict[str, Any] | None:
    return load_hires().get(task)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def export(task: str, out: Path | None = None, fmt: str = "messages") -> Path:
    """Write the shadow examples as a fine-tuning dataset.

    ``messages`` → ``{"messages": [{role, content}, ...]}`` per line, which MLX-LM, Unsloth, Axolotl and
    TRL's SFTTrainer all accept. ``prompt-completion`` → ``{"prompt", "completion"}`` per line.
    """
    ex = examples(task)
    if not ex:
        raise ValueError(f"no examples recorded for task {task!r}")
    system = task_meta(task).get("system")
    out = out or Path.cwd() / f"{task}.train.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for e in ex:
            expected = e["expected"] if isinstance(e["expected"], str) else json.dumps(e["expected"])
            if fmt == "prompt-completion":
                row = {"prompt": e["prompt"], "completion": expected}
            else:
                msgs = ([{"role": "system", "content": system}] if system else []) + [
                    {"role": "user", "content": e["prompt"]},
                    {"role": "assistant", "content": expected},
                ]
                row = {"messages": msgs}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out
