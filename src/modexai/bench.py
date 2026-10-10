"""Run a suite against one or more local models and summarise quality, speed and what it would have cost."""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from . import config
from .backends import Backend, BackendError
from .checks import grade
from .registry import Registry
from .suites import Suite

Event = Callable[[str, str, dict[str, Any]], None]


@dataclass
class CaseResult:
    id: str
    passed: bool
    detail: str
    latency_ms: float
    output: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class ModelResult:
    model: str
    suite: str
    suite_sha: str
    backend: str
    registry_id: str | None = None
    size_gb: float | None = None
    passed: int = 0
    total: int = 0
    p50_ms: float | None = None
    p95_ms: float | None = None
    tokens_per_s: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def score(self) -> float:
        return round(self.passed / self.total, 4) if self.total else 0.0

    def cloud_cost_usd(self, per_calls: int | None = None) -> float:
        """What the same tokens would cost at the configured cloud baseline (an estimate)."""
        pin, pout = config.baseline_price()
        cost = (self.prompt_tokens * pin + self.completion_tokens * pout) / 1e6
        if per_calls and self.total:
            cost = cost / self.total * per_calls
        return round(cost, 4)

    def to_dict(self, include_cases: bool = True) -> dict[str, Any]:
        d = asdict(self)
        d["score"] = self.score
        d["cloud_cost_per_1k_calls_usd"] = self.cloud_cost_usd(per_calls=1000)
        if not include_cases:
            d.pop("cases")
        return d


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(q * (len(s) - 1))))
    return round(s[k], 1)


def run_benchmark(
    models: list[str],
    suite: Suite,
    backend: Backend,
    registry: Registry | None = None,
    limit: int | None = None,
    warmup: bool = True,
    on_event: Event | None = None,
) -> list[ModelResult]:
    """Benchmark each model on the suite. Results come back in the order given."""
    emit = on_event or (lambda *_: None)
    items = suite.items[:limit] if limit else suite.items
    results: list[ModelResult] = []
    for ref in models:
        reg = registry.get(ref) if registry else None
        tag = reg.ollama_tag if reg else ref
        res = ModelResult(
            model=tag, suite=suite.ref, suite_sha=suite.sha, backend=backend.name,
            registry_id=reg.id if reg else None, size_gb=reg.size_gb if reg and reg.size_gb else None,
            total=len(items),
        )
        emit("start", tag, {"total": len(items)})
        try:
            if warmup and items and backend.name != "simulated":
                # First call loads weights into memory; keep it out of the latency numbers.
                backend.generate(tag, suite.messages(items[0]), max_tokens=1, item=items[0])
            latencies, speeds = [], []
            for it in items:
                gen = backend.generate(
                    tag, suite.messages(it), max_tokens=suite.max_tokens, json_mode=suite.json_mode, item=it,
                )
                ok, detail = grade(gen.text, it["check"])
                res.passed += int(ok)
                res.prompt_tokens += gen.prompt_tokens
                res.completion_tokens += gen.completion_tokens
                latencies.append(gen.latency_ms)
                if gen.tokens_per_s:
                    speeds.append(gen.tokens_per_s)
                case = CaseResult(it["id"], ok, detail, round(gen.latency_ms, 1), gen.text[:400],
                                  gen.prompt_tokens, gen.completion_tokens)
                res.cases.append(case)
                emit("case", tag, {"case": case, "done": len(res.cases), "passed": res.passed, "total": len(items)})
            res.p50_ms = _pct(latencies, 0.5)
            res.p95_ms = _pct(latencies, 0.95)
            res.tokens_per_s = round(statistics.median(speeds), 1) if speeds else None
        except BackendError as e:
            res.error = str(e)
            emit("error", tag, {"error": str(e)})
        emit("done", tag, {"result": res})
        results.append(res)
    return results


@dataclass
class Verdict:
    winner: ModelResult | None
    min_score: float
    reason: str
    ranked: list[ModelResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner.model if self.winner else None,
            "winner_score": self.winner.score if self.winner else None,
            "min_score": self.min_score,
            "reason": self.reason,
            "ranking": [
                {"model": r.model, "score": r.score, "p50_ms": r.p50_ms, "size_gb": r.size_gb, "error": r.error}
                for r in self.ranked
            ],
        }


def verdict(results: list[ModelResult], min_score: float = 0.8) -> Verdict:
    """Hire the *smallest* model that clears the bar — right-sizing, not leaderboard-chasing.

    If nothing clears it, say so plainly: that task is where cloud intelligence earns its place.
    """
    ok = [r for r in results if not r.error and r.total]
    ranked = sorted(ok, key=lambda r: (-r.score, r.size_gb or 99, r.p50_ms or 1e9))
    passing = [r for r in ok if r.score >= min_score]
    if passing:
        best = min(passing, key=lambda r: (r.size_gb or 99, -r.score, r.p50_ms or 1e9))
        size = f"{best.size_gb} GB" if best.size_gb else "local"
        return Verdict(best, min_score,
                       f"{best.model} clears the {min_score:.0%} bar at {best.score:.0%} — the smallest model that "
                       f"does ({size}, p50 {best.p50_ms} ms). Delegate this task to it.", ranked)
    if ranked:
        top = ranked[0]
        return Verdict(None, min_score,
                       f"No local model cleared {min_score:.0%} (best: {top.model} at {top.score:.0%}). Keep this "
                       f"task on a stronger model — this is where cloud earns its place — or fine-tune a specialist "
                       f"from your examples (`modexai export`).", ranked)
    return Verdict(None, min_score, "No model produced results (are they installed?).", ranked)
