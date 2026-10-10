"""A local tally of work kept on your machine — counts and token totals only, never prompt text.

Lives in ``~/.modexai/ledger.jsonl``. Disable with ``MODEXAI_NO_LEDGER=1``.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any

from . import config


def _path():
    return config.home() / "ledger.jsonl"


def log(model: str, prompt_tokens: int, completion_tokens: int, latency_ms: float, task: str | None = None) -> None:
    if not config.ledger_enabled():
        return
    row = {"ts": int(time.time()), "model": model, "task": task, "in": prompt_tokens, "out": completion_tokens,
           "ms": round(latency_ms)}
    with _path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def summary(days: int | None = None) -> dict[str, Any]:
    pin, pout = config.baseline_price()
    cutoff = time.time() - days * 86400 if days else 0
    calls = tin = tout = 0
    by_model: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0})
    by_task: dict[str, int] = defaultdict(int)
    first = None
    p = _path()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["ts"] < cutoff:
                continue
            first = first or r["ts"]
            calls += 1
            tin += r.get("in", 0)
            tout += r.get("out", 0)
            m = by_model[r["model"]]
            m["calls"] += 1
            m["in"] += r.get("in", 0)
            m["out"] += r.get("out", 0)
            if r.get("task"):
                by_task[r["task"]] += 1
    saved = (tin * pin + tout * pout) / 1e6
    return {
        "calls_kept_local": calls,
        "tokens_in": tin,
        "tokens_out": tout,
        "est_cloud_cost_avoided_usd": round(saved, 2),
        "baseline_usd_per_million": {"input": pin, "output": pout},
        "since": time.strftime("%Y-%m-%d", time.localtime(first)) if first else None,
        "by_model": {k: {**v, "est_usd": round((v["in"] * pin + v["out"] * pout) / 1e6, 2)} for k, v in by_model.items()},
        "by_task": dict(by_task),
    }


def _fmt(n: float) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return f"{int(n)}"


def badge_svg(s: dict[str, Any] | None = None) -> str:
    """A shareable badge: 'kept local | 12.4k calls · ~$38'. Shows totals only."""
    s = s or summary()
    left = "kept local"
    right = f"{_fmt(s['calls_kept_local'])} calls · ~${s['est_cloud_cost_avoided_usd']:,.0f}"
    lw, rw = 8 + len(left) * 7, 10 + len(right) * 7
    w = lw + rw
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="22" role="img" aria-label="{left}: {right}">
<linearGradient id="g" x2="0" y2="100%"><stop offset="0" stop-color="#fff" stop-opacity=".12"/><stop offset="1" stop-opacity=".12"/></linearGradient>
<clipPath id="r"><rect width="{w}" height="22" rx="4"/></clipPath>
<g clip-path="url(#r)"><rect width="{lw}" height="22" fill="#111827"/><rect x="{lw}" width="{rw}" height="22" fill="#2A6DFF"/><rect width="{w}" height="22" fill="url(#g)"/></g>
<g fill="#fff" font-family="Verdana,DejaVu Sans,sans-serif" font-size="11">
<text x="{lw / 2}" y="15" text-anchor="middle">{left}</text><text x="{lw + rw / 2}" y="15" text-anchor="middle">{right}</text></g>
</svg>"""
