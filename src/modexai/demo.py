"""`modexai demo` — the whole loop in under a minute.

Simulated by default (no Ollama needed, clearly labelled); ``--real`` races your installed models.
Uses a scratch MODEXAI_HOME so it never touches your real shadow examples or ledger.
"""

from __future__ import annotations

import json
import os
import tempfile
import time

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from . import hardware, recipes, service
from .backends import OllamaBackend, SimulatedBackend
from .suites import get_suite

BLUE, TEAL = "#2A6DFF", "#00C4B4"
DEMO_MODELS = ["qwen3:0.6b", "granite4:350m", "gemma3:1b", "qwen3:4b-instruct"]
FALLBACK = {"qwen3:0.6b": (0.2, 108), "granite4:350m": (0.5, 80), "gemma3:1b": (0.8, 156), "qwen3:4b-instruct": (0.85, 172)}


def _measured() -> tuple[dict[str, float], dict[str, float]]:
    """Simulate each model with the score and latency it actually measured on ticket-triage (from the index)."""
    skills, speed = {}, {}
    found = {r["model"]["tag"]: r["eval"] for r in recipes.load_index()
             if r.get("eval", {}).get("suite") == "ticket-triage@1" and r["kind"] == "benchmark"}
    for m in DEMO_MODELS:
        score, p50 = (found[m]["score"], found[m]["p50_ms"]) if m in found else FALLBACK[m]
        skills[m], speed[m] = score, p50
    return skills, speed


def run_demo(console: Console, real: bool = False, export_svg: str | None = None, fast: bool = False) -> int:
    rec_console = Console(record=True, width=console.width if not export_svg else 100) if export_svg else console
    c = rec_console
    pause = (lambda s: None) if fast or export_svg else time.sleep

    old_home = os.environ.get("MODEXAI_HOME")
    os.environ["MODEXAI_HOME"] = tempfile.mkdtemp(prefix="modexai-demo-")
    try:
        if real:
            backend = OllamaBackend()
            inst = backend.installed()
            models = [m for m in ("qwen3:0.6b", "gemma3:1b", "qwen3:1.7b", "phi4-mini") if m in inst] or sorted(inst)[:4]
        else:
            skills, speed = _measured()
            backend = SimulatedBackend(skills, speed)
            models = list(skills)
        ctx = service.Context(backend=backend)
        suite = get_suite("ticket-triage")

        c.print(f"[bold {BLUE}]modex[/][bold {TEAL}]ai[/] demo  [dim]{'real models' if real else 'simulated from measured scores — run with --real to use your models'}[/]")
        c.print(f"[dim]{hardware.label()} · model budget ~{hardware.memory_budget_gb()} GB[/]\n")
        pause(0.8)

        # 1. shadow
        c.print("[bold]1 · shadow[/]  Your agent (say, Claude) is triaging support tickets. Each answer it gives is saved [bold]locally[/].")
        with Live(console=c, refresh_per_second=20, transient=False) as live:
            for i, it in enumerate(suite.items, 1):
                service.record_example("ticket-triage", it["prompt"], it["check"]["expected"], system=suite.system,
                                       reference_model="claude")
                live.update(f"  [dim]~/.modexai/tasks/ticket-triage.jsonl[/]  [bold {TEAL}]{i}[/] examples  "
                            f"[dim]“{it['prompt'][:52]}…” → {it['check']['expected']}[/]")
                pause(0.06)
        c.print()
        pause(0.5)

        # 2. find recipes
        c.print("[bold]2 · ask the exchange[/]  What already works for this, on hardware like this?")
        found = recipes.find(recipes.load_index(), task="ticket-triage", limit=4)
        t = Table(box=None, padding=(0, 2), header_style="dim")
        for col in ("model", "score", "p50", "measured on"):
            t.add_column(col)
        for r in found:
            t.add_row(r["model"]["tag"], f"{r['eval']['score']:.0%}", f"{r['eval']['p50_ms']:.0f} ms", r["hardware"]["label"])
        c.print(t if found else "  [dim](no recipes in this index yet)[/]")
        pause(1.0)

        # 3. hire
        c.print("\n[bold]3 · hire[/]  Test small local models against the agent's own answers. Smallest one that agrees ≥ 80% gets the job.")
        from .cli import RaceBoard
        board = RaceBoard(models, "hiring for ticket-triage · vs Claude's answers", len(suite.items))

        def slow_event(kind, model, payload):
            board.on_event(kind, model, payload)
            if kind == "case":
                pause(0.04)

        with Live(board, console=c, refresh_per_second=12, transient=False):
            out = service.hire_model(ctx, "ticket-triage", models, min_score=0.8, on_event=slow_event)
        c.print()
        v = out["verdict"]
        c.print(Panel(v["reason"], title="verdict", border_style=TEAL if v["winner"] else "yellow"))
        pause(0.8)
        if not out["hired"]:
            return 0

        # 4. delegate
        c.print(f"\n[bold]4 · delegate[/]  300 new tickets go to [bold]{out['hired']}[/] on this machine.")
        total_in = total_out = 0
        with Live(console=c, refresh_per_second=20, transient=False) as live:
            for i in range(1, 301):
                it = suite.items[i % len(suite.items)]
                gen = backend.generate(out["hired"], suite.messages(it), max_tokens=8, item=it)
                total_in += gen.prompt_tokens
                total_out += gen.completion_tokens
                if i % 6 == 0 or i == 300:
                    live.update(f"  [bold {TEAL}]{i}[/]/300 tickets routed  [dim]0 tokens billed · 0 bytes sent[/]")
                    pause(0.02)
        c.print()
        from .config import baseline_price
        pin, pout = baseline_price()
        saved = (total_in * pin + total_out * pout) / 1e6 * 1000 / 300
        c.print(f"  [dim]at a ${pin:g}/${pout:g} per-M-token cloud baseline, that's ≈ ${saved:.2f} per 1,000 tickets kept local[/]")
        pause(0.8)

        # 5. share
        c.print("\n[bold]5 · share[/]  With your OK, the finding becomes a recipe other agents can find and reproduce:")
        draft = recipes.load_local()[0]
        shown = {k: draft[k] for k in ("kind", "title", "task", "model", "private_eval") if k in draft}
        shown["hardware"] = {"label": draft["hardware"]["label"], "accelerator": draft["hardware"]["accelerator"]}
        c.print(Panel(json.dumps(shown, indent=2, ensure_ascii=False), title="recipe preview — no prompts, no data, no weights",
                      border_style=BLUE))
        c.print("\n[bold]try it:[/] claude mcp add modexai -- uvx --from git+https://github.com/abhishek085/modexai modexai mcp")
    finally:
        if old_home is None:
            os.environ.pop("MODEXAI_HOME", None)
        else:
            os.environ["MODEXAI_HOME"] = old_home
    if export_svg:
        c.save_svg(export_svg, title="modexai demo")
        console.print(f"[green]✓[/] wrote {export_svg}")
    return 0
