"""`modexai` command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from . import __version__, config, hardware, ledger, recipes, service
from .backends import BackendError, get_backend
from .bench import ModelResult
from .registry import update_registry

BLUE, TEAL, INK, GRAY = "#2A6DFF", "#00C4B4", "#111827", "#6b7280"
console = Console()
err = Console(stderr=True)

LOGO = f"[bold {BLUE}]modex[/][bold {TEAL}]ai[/] [dim]· the local model exchange for agents · a nokast tool[/]"


# ---------------------------------------------------------------------------
# Live benchmark board
# ---------------------------------------------------------------------------


class RaceBoard:
    """A live leaderboard while models work through a suite."""

    def __init__(self, models: list[str], suite_name: str, total: int):
        self.suite_name = suite_name
        self.rows: dict[str, dict[str, Any]] = {
            m: {"done": 0, "passed": 0, "total": total, "lat": [], "status": "queued", "tps": None} for m in models
        }

    def on_event(self, kind: str, model: str, p: dict[str, Any]) -> None:
        row = self.rows.setdefault(model, {"done": 0, "passed": 0, "total": p.get("total", 0), "lat": [],
                                           "status": "queued", "tps": None})
        if kind == "start":
            row["status"] = "running"
            row["total"] = p["total"]
        elif kind == "case":
            row["done"], row["passed"] = p["done"], p["passed"]
            row["lat"].append(p["case"].latency_ms)
        elif kind == "error":
            row["status"] = "error"
            row["error"] = p["error"]
        elif kind == "done":
            res: ModelResult = p["result"]
            if row["status"] != "error":
                row["status"] = "done"
            row["tps"] = res.tokens_per_s

    def __rich__(self) -> Panel:
        t = Table(box=box.SIMPLE_HEAD, expand=True, header_style=f"bold {GRAY}")
        t.add_column("model", style="bold", no_wrap=True)
        t.add_column("progress", ratio=2)
        t.add_column("score", justify="right")
        t.add_column("p50", justify="right")
        t.add_column("tok/s", justify="right")
        t.add_column("", justify="left")
        best = max((r["passed"] / r["done"] if r["done"] else 0) for r in self.rows.values()) if self.rows else 0
        for m, r in self.rows.items():
            score = r["passed"] / r["done"] if r["done"] else 0
            lat = sorted(r["lat"])
            p50 = f"{lat[len(lat) // 2]:.0f} ms" if lat else "–"
            color = TEAL if score >= 0.8 else ("yellow" if score >= 0.6 else "red")
            mark = {"queued": "[dim]queued[/]", "running": f"[{BLUE}]running…[/]", "done": "[green]✓[/]",
                    "error": "[red]error[/]"}[r["status"]]
            if r["status"] == "done" and r["done"] and score == best:
                mark += " [bold]▲[/]"
            t.add_row(
                m,
                ProgressBar(total=max(r["total"], 1), completed=r["done"], width=None, complete_style=BLUE,
                            finished_style=TEAL),
                f"[{color}]{score:.0%}[/]" if r["done"] else "–",
                p50,
                f"{r['tps']:.0f}" if r["tps"] else "–",
                mark,
            )
        return Panel(t, title=f"[bold]{self.suite_name}[/]", subtitle="[dim]running on this machine · nothing leaves it[/]",
                     border_style=BLUE)


def _print_verdict(v: dict[str, Any]) -> None:
    ok = v["winner"] is not None
    console.print(Panel(Text(v["reason"]), title="[bold]verdict[/]", border_style=TEAL if ok else "yellow"))


def _ctx(args) -> service.Context:
    kw = {}
    if getattr(args, "base_url", None):
        kw["base_url"] = args.base_url
    return service.Context(backend=get_backend(args.backend, **kw))


def _dump(obj: Any) -> None:
    # Plain stdout, never Rich: soft-wrapping would corrupt JSON meant for pipes and agents.
    sys.stdout.write(json.dumps(obj, indent=2, default=str, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_doctor(args) -> int:
    ctx = _ctx(args)
    info = service.system_info(ctx)
    if args.json:
        _dump(info)
        return 0
    console.print(LOGO)
    hw = info["hardware"]
    t = Table(box=None, show_header=False, padding=(0, 2))
    t.add_row("machine", f"[bold]{hw['label']}[/] [dim]({hw['os']} {hw['arch']}, {hw['accelerator']})[/]")
    t.add_row("model budget", f"~{info['model_budget_gb']} GB [dim](override: MODEXAI_MAX_MODEL_GB)[/]")
    o = info.get("ollama")
    if o:
        state = f"[green]running[/] v{o['version']}" if o["running"] else "[red]not running[/] — install from ollama.com and run `ollama serve`"
        t.add_row("ollama", f"{state} [dim]{o['host']}[/]")
    t.add_row("installed", f"{len(info['installed_models'])} models · {len(info['installed_and_listed'])} listed in the registry")
    t.add_row("registry", f"{info['registry_models']} models · {len(recipes.load_index())} community recipes")
    t.add_row("local data", f"[dim]{info['data_dir']}[/] (shadow examples, receipts, ledger — never uploaded)")
    console.print(t)
    if o and o["running"] and not info["installed_and_listed"]:
        console.print("\n[bold]next:[/] `modexai recommend classification --auto-install` or `modexai install qwen/qwen3-1.7b`")
    return 0


def cmd_search(args) -> int:
    ctx = _ctx(args)
    res = service.search(ctx, " ".join(args.query), args.task, args.max_size, args.installed, args.limit)
    if args.json:
        _dump(res)
        return 0
    t = Table(box=box.SIMPLE_HEAD, header_style=f"bold {GRAY}")
    for col in ("model", "tasks", "size", "best measured", "license", ""):
        t.add_column(col)
    for m in res["models"]:
        bench = m.get("benchmarks") or {}
        best = max(bench.items(), key=lambda kv: kv[1]["score"]) if bench else None
        t.add_row(
            f"[bold]{m['id']}[/]\n[dim]{m['ollama_tag']}[/]",
            ", ".join(m["tasks"][:3]),
            f"{m['size_gb']} GB" if m["size_gb"] else "–",
            f"{best[1]['score']:.0%} {best[0]}" if best else "[dim]–[/]",
            m["license"],
            "[green]installed[/]" if m["installed"] else "",
        )
    console.print(t)
    console.print(f"[dim]showing models that fit ≤ {res['max_size_gb']} GB · tasks: {', '.join(res['known_tasks'][:12])}[/]")
    return 0


def cmd_show(args) -> int:
    ctx = _ctx(args)
    m = ctx.registry.get(args.model)
    if not m:
        err.print(f"[red]{args.model} is not in the registry[/] — try `modexai search {args.model}`")
        return 1
    if args.json:
        _dump(m.to_dict())
        return 0
    console.print(Panel(
        f"[bold]{m.name}[/]  [dim]{m.id}[/]\n{m.summary}\n\n"
        f"tasks: {', '.join(m.tasks)}\nsize: {m.size_gb} GB · params: {m.params_b}B · license: {m.license}\n"
        f"install: [bold]modexai install {m.id}[/]  (ollama tag: {m.ollama_tag})",
        border_style=BLUE))
    if m.benchmarks:
        t = Table(box=box.SIMPLE_HEAD, header_style=f"bold {GRAY}")
        for c in ("suite", "score", "p50", "tok/s", "hardware"):
            t.add_column(c)
        for s, b in m.benchmarks.items():
            t.add_row(s, f"{b['score']:.0%}", f"{b.get('p50_ms')} ms", str(b.get("tokens_per_s") or "–"), b.get("hardware", ""))
        console.print(t)
    found = recipes.find(recipes.load_index(), model=m.ollama_tag, fits_here=False, limit=5)
    if found:
        console.print(f"[bold]{len(found)} recipe(s)[/] — `modexai recipes find --model {m.ollama_tag}`")
    return 0


def cmd_suites(args) -> int:
    res = service.suites()
    if args.json:
        _dump(res)
        return 0
    t = Table(box=box.SIMPLE_HEAD, header_style=f"bold {GRAY}")
    for c in ("suite", "task", "cases", "what it tests"):
        t.add_column(c)
    for s in res["suites"]:
        t.add_row(f"[bold]{s['id']}[/]", s["task"], str(s["cases"]), s["description"])
    console.print(t)
    return 0


def _run_with_board(models: list[str], title: str, total: int, fn) -> Any:
    board = RaceBoard(models, title, total)
    with Live(board, console=console, refresh_per_second=8, transient=False):
        out = fn(board.on_event)
    console.print()
    return out


def cmd_bench(args) -> int:
    from .suites import get_suite, load_samples_file, suite_from_samples
    ctx = _ctx(args)
    samples = load_samples_file(args.samples) if args.samples else None
    if samples:
        s = suite_from_samples(samples, name=args.task or "custom", system=args.system)
    elif args.suite:
        s = get_suite(args.suite)
    elif args.task:
        s = service._pick_suite(args.task, None, None, None)
    else:
        err.print("[red]choose --suite, --task or --samples[/] (see `modexai suites`)")
        return 2
    total = min(len(s.items), args.limit) if args.limit else len(s.items)
    tags = [ctx.resolve(m)[0] for m in args.models]
    def call(ev):
        return service.benchmark(ctx, args.models, suite=s.id if s.builtin else None, task=args.task,
                                            samples=samples, system=args.system, limit=args.limit,
                                            min_score=args.min_score, on_event=ev)
    if args.json:
        _dump(call(None))
        return 0
    console.print(LOGO)
    out = _run_with_board(tags, s.name if s.builtin else f"{s.id} · your samples", total, call)
    if out["not_installed"]:
        console.print(f"[yellow]not installed:[/] {', '.join(out['not_installed'])} — `modexai install <model>`")
    for r in out["results"]:
        if r["error"]:
            console.print(f"[red]{r['model']}: {r['error']}[/]")
    _print_verdict(out["verdict"])
    _cost_line(out["results"])
    if out["recipe_drafts"]:
        console.print(f"[dim]recipe drafts saved: {', '.join(out['recipe_drafts'])}[/]")
        console.print(f"share a finding with other agents: [bold]modexai recipes share {out['recipe_drafts'][0]}[/]")
    return 0


def _cost_line(results: list[dict[str, Any]]) -> None:
    ok = [r for r in results if not r["error"] and r["total"]]
    if not ok:
        return
    pin, pout = config.baseline_price()
    r = ok[0]
    console.print(f"[dim]≈ ${r['cloud_cost_per_1k_calls_usd']:.2f} per 1,000 calls at a ${pin:g}/${pout:g} per-million-token "
                  f"cloud baseline (MODEXAI_BASELINE) · $0 marginal here[/]")


def cmd_install(args) -> int:
    ctx = _ctx(args)
    from rich.progress import BarColumn, Progress, TextColumn
    with Progress(TextColumn("{task.description}"), BarColumn(complete_style=BLUE), TextColumn("{task.percentage:>3.0f}%"),
                  console=console, transient=True) as prog:
        tid = prog.add_task(f"installing {args.model}", total=1.0)

        def cb(status: str, frac: float | None) -> None:
            prog.update(tid, description=f"{args.model} · {status}", completed=frac if frac is not None else 0)

        try:
            out = service.install(ctx, args.model, cb)
        except BackendError as e:
            err.print(f"[red]{e}[/]")
            return 1
    from .suites import suite_for_task
    m = ctx.registry.get(out["installed"])
    suite = next((s for s in (suite_for_task(t) for t in (m.tasks if m else [])) if s), None)
    console.print(f"[green]✓[/] installed [bold]{out['installed']}[/]")
    console.print(f"  try it:  modexai run {out['installed']} \"hello\"")
    console.print(f"  test it: modexai bench {out['installed']} --suite {suite.id if suite else 'ticket-triage'}")
    return 0


def cmd_run(args) -> int:
    ctx = _ctx(args)
    prompt = " ".join(args.prompt) if args.prompt else sys.stdin.read()
    try:
        out = service.ask(ctx, prompt, model=args.model, system=args.system, max_tokens=args.max_tokens,
                          json_mode=args.json_mode)
    except (BackendError, ValueError) as e:
        err.print(f"[red]{e}[/]")
        return 1
    if args.json:
        _dump(out)
    else:
        console.print(out["output"])
        err.print(f"[dim]{out['model']} · {out['latency_ms']} ms · {out['tokens']['out']} tokens · on this machine[/]")
    return 0


def cmd_recommend(args) -> int:
    from .suites import load_samples_file
    ctx = _ctx(args)
    samples = load_samples_file(args.samples) if args.samples else None
    console.print(LOGO)
    console.print(f"[dim]{hardware.label()} · budget {hardware.memory_budget_gb()} GB · finding candidates for '{args.task}'…[/]")
    board = RaceBoard([], args.task, 0)
    with Live(board, console=console, refresh_per_second=8):
        out = service.recommend(ctx, args.task, samples, args.max_size, args.min_score, args.candidates,
                                args.auto_install, args.limit, on_event=board.on_event)
    _print_verdict(out["verdict"])
    _cost_line(out["results"])
    if out["recipe_drafts"]:
        console.print(f"share it: [bold]modexai recipes share {out['recipe_drafts'][0]}[/]")
    return 0


def cmd_record(args) -> int:
    out = service.record_example(args.task, args.prompt, args.output, args.system, args.reference)
    console.print(out)
    return 0


def cmd_tasks(args) -> int:
    res = service.list_tasks()
    if args.json:
        _dump(res)
        return 0
    if not res["tasks"]:
        console.print("No shadowed tasks yet. Agents record them with the MCP tool `record_example`.")
        return 0
    t = Table(box=box.SIMPLE_HEAD, header_style=f"bold {GRAY}")
    for c in ("task", "examples", "hired"):
        t.add_column(c)
    for r in res["tasks"]:
        h = res["hires"].get(r["task"])
        t.add_row(r["task"], str(r["examples"]), f"[green]{h['model']}[/] ({h['agreement']:.0%})" if h else "–")
    console.print(t)
    return 0


def cmd_hire(args) -> int:
    ctx = _ctx(args)
    board = RaceBoard(args.candidates or [], f"hiring for '{args.task}' · vs your reference answers", 0)
    try:
        with Live(board, console=console, refresh_per_second=8):
            out = service.hire_model(ctx, args.task, args.candidates, args.min_score, args.max_size, on_event=board.on_event)
    except ValueError as e:
        err.print(f"[red]{e}[/]")
        return 1
    _print_verdict(out["verdict"])
    console.print(out["next"])
    return 0


def cmd_export(args) -> int:
    out = service.export_training_data(args.task, args.output, args.format)
    console.print(f"[green]✓[/] wrote {out['examples']} examples to [bold]{out['path']}[/]")
    for line in out["train_with"]:
        console.print(f"  [dim]{line}[/]")
    return 0


def _recipe_row(t: Table, r: dict[str, Any]) -> None:
    ev = r.get("eval") or r.get("private_eval") or {}
    t.add_row(
        f"[bold]{r['model']['tag']}[/]\n[dim]{r['id']}[/]",
        r["task"] + ("" if r.get("eval") else " [dim](private)[/]"),
        f"{ev.get('score', 0):.0%}",
        f"{ev.get('p50_ms')} ms",
        r.get("hardware", {}).get("label", "?") if isinstance(r.get("hardware"), dict) else str(r.get("hardware")),
        str(r.get("_repro", r.get("reproductions", 0))),
    )


def _recipe_table() -> Table:
    t = Table(box=box.SIMPLE_HEAD, header_style=f"bold {GRAY}")
    for c in ("model", "task", "score", "p50", "measured on", "repro"):
        t.add_column(c)
    return t


def cmd_recipes(args) -> int:
    sub = args.recipes_cmd
    if sub == "find":
        res = service.find_recipes(args.task, args.model, True, not args.all_hardware, args.limit)
        if args.json:
            _dump(res)
            return 0
        console.print(f"[dim]this machine: {res['this_machine']} · {res['note']}[/]")
        t = _recipe_table()
        for r in res["recipes"]:
            _recipe_row(t, {**r, "hardware": {"label": r["hardware"]}, "_repro": r["reproductions"]})
        console.print(t)
        return 0
    if sub == "mine":
        t = _recipe_table()
        for r in recipes.load_local():
            _recipe_row(t, r)
        console.print(t)
        return 0
    if sub == "show":
        _dump(service.preview_recipe(args.id))
        return 0
    if sub == "try":
        ctx = _ctx(args)
        r = service._get_recipe(args.id)
        board = RaceBoard([r["model"]["tag"]], f"reproducing {args.id}", 0)
        with Live(board, console=console, refresh_per_second=8):
            out = service.try_recipe(ctx, args.id, args.auto_install, on_event=board.on_event)
        _dump(out)
        return 0
    if sub == "share":
        prev = service.share_recipe(args.id, confirm=False, notes=args.notes, author=args.author)
        if prev.get("problems"):
            err.print(f"[red]not shareable:[/] {prev['problems']}")
            return 1
        console.print(Panel(json.dumps(prev["preview"], indent=2), title="this is exactly what will be shared",
                            subtitle="no prompts, outputs or data — setup, hardware class, scores", border_style=BLUE))
        if not args.yes and console.input("share it publicly? [y/N] ").strip().lower() not in ("y", "yes"):
            console.print("not shared.")
            return 0
        out = service.share_recipe(prev["preview"]["id"], confirm=True)
        console.print(f"[green]✓[/] {out['note']}\n{out['url']}")
        return 0
    if sub == "validate":
        raw = sys.stdin.read() if args.file == "-" else Path(args.file).read_text()
        d = json.loads(raw)
        problems = recipes.validate(d)
        if problems:
            for p in problems:
                err.print(f"[red]✗[/] {p}")
            return 1
        console.print(f"[green]✓[/] valid recipe {d.get('id')}")
        return 0
    if sub == "id":
        d = json.loads(Path(args.file).read_text())
        print(recipes.recipe_id(d))
        return 0
    return 2


def cmd_savings(args) -> int:
    s = ledger.summary(args.days)
    if args.badge:
        with open(args.badge, "w") as fh:
            fh.write(ledger.badge_svg(s))
        console.print(f"[green]✓[/] badge written to {args.badge}")
    if args.json:
        _dump(s)
        return 0
    console.print(Panel(
        f"[bold {TEAL}]{s['calls_kept_local']:,}[/] calls kept on this machine"
        + (f" since {s['since']}" if s["since"] else "")
        + f"\n[bold]{s['tokens_in'] + s['tokens_out']:,}[/] tokens that never left it"
        f"\n[bold]~${s['est_cloud_cost_avoided_usd']:,.2f}[/] cloud cost avoided "
        f"[dim](at ${s['baseline_usd_per_million']['input']:g}/${s['baseline_usd_per_million']['output']:g} per M tokens)[/]",
        title="kept local", border_style=TEAL))
    return 0


def cmd_update(args) -> int:
    path, n = update_registry()
    k = recipes.update_index()
    console.print(f"[green]✓[/] registry: {n} models · recipes: {k} · cached in {path.parent}")
    return 0


def cmd_mcp(args) -> int:
    from .mcp_server import run
    run()
    return 0


def cmd_ui(args) -> int:
    from .web_server import serve
    serve(args.port, open_browser=not args.no_open)
    return 0


def cmd_demo(args) -> int:
    from .demo import run_demo
    return run_demo(console, real=args.real, export_svg=args.export_svg, fast=args.fast)


def cmd_registry(args) -> int:
    from .build import build
    return build(check=args.check, refresh_sizes=args.refresh_sizes, console=console)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="modexai", description="The local model exchange for AI agents. A Nokast tool.")
    p.add_argument("--version", action="version", version=f"modexai {__version__}")
    p.add_argument("--backend", default="ollama", choices=["ollama", "openai", "simulated"])
    p.add_argument("--base-url", help="OpenAI-compatible server URL for --backend openai (llama.cpp, LM Studio, vLLM)")
    sp = p.add_subparsers(dest="cmd", metavar="command")

    def add(name, fn, help_):
        s = sp.add_parser(name, help=help_, description=help_)
        s.set_defaults(fn=fn)
        return s

    s = add("doctor", cmd_doctor, "check this machine, Ollama and what fits")
    s.add_argument("--json", action="store_true")

    s = add("search", cmd_search, "search the registry for models that fit this machine")
    s.add_argument("query", nargs="*")
    s.add_argument("--task")
    s.add_argument("--max-size", type=float)
    s.add_argument("--installed", action="store_true")
    s.add_argument("--limit", type=int, default=15)
    s.add_argument("--json", action="store_true")

    s = add("show", cmd_show, "show a model listing with measured results")
    s.add_argument("model")
    s.add_argument("--json", action="store_true")

    s = add("suites", cmd_suites, "list built-in benchmark suites")
    s.add_argument("--json", action="store_true")

    s = add("bench", cmd_bench, "benchmark local models on a suite or your own samples")
    s.add_argument("models", nargs="+")
    s.add_argument("--suite")
    s.add_argument("--task")
    s.add_argument("--samples", help=".jsonl/.json of {prompt, expected} — stays on this machine")
    s.add_argument("--system", help="system prompt for --samples")
    s.add_argument("--limit", type=int)
    s.add_argument("--min-score", type=float, default=0.8)
    s.add_argument("--json", action="store_true")

    s = add("install", cmd_install, "install a listed model into Ollama")
    s.add_argument("model")

    s = add("run", cmd_run, "run a prompt on a local model")
    s.add_argument("model")
    s.add_argument("prompt", nargs="*")
    s.add_argument("--system")
    s.add_argument("--max-tokens", type=int, default=512)
    s.add_argument("--json-mode", action="store_true")
    s.add_argument("--json", action="store_true")

    s = add("recommend", cmd_recommend, "find, test and pick the smallest model good enough for a task")
    s.add_argument("task")
    s.add_argument("--samples")
    s.add_argument("--max-size", type=float)
    s.add_argument("--min-score", type=float, default=0.8)
    s.add_argument("--candidates", type=int, default=4)
    s.add_argument("--limit", type=int)
    s.add_argument("--auto-install", action="store_true", help="pull missing candidates (downloads GBs)")

    s = add("record", cmd_record, "shadow mode: record one (input, reference answer) example")
    s.add_argument("task")
    s.add_argument("--prompt", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--system")
    s.add_argument("--reference")

    s = add("tasks", cmd_tasks, "list shadowed tasks and hires")
    s.add_argument("--json", action="store_true")

    s = add("hire", cmd_hire, "hire the smallest local model that agrees with your recorded answers")
    s.add_argument("task")
    s.add_argument("--candidates", nargs="+")
    s.add_argument("--min-score", type=float, default=0.9)
    s.add_argument("--max-size", type=float)

    s = add("export", cmd_export, "export a shadowed task as a fine-tuning dataset")
    s.add_argument("task")
    s.add_argument("-o", "--output")
    s.add_argument("--format", default="messages", choices=["messages", "prompt-completion"])

    s = add("recipes", cmd_recipes, "find, try and share community recipes")
    rs = s.add_subparsers(dest="recipes_cmd", required=True)
    r = rs.add_parser("find", help="find recipes that fit this machine")
    r.add_argument("task", nargs="?")
    r.add_argument("--model")
    r.add_argument("--all-hardware", action="store_true")
    r.add_argument("--limit", type=int, default=15)
    r.add_argument("--json", action="store_true")
    rs.add_parser("mine", help="your local recipe drafts")
    r = rs.add_parser("show", help="show a recipe")
    r.add_argument("id")
    r = rs.add_parser("try", help="reproduce a recipe on this machine")
    r.add_argument("id")
    r.add_argument("--auto-install", action="store_true")
    r = rs.add_parser("share", help="share a recipe (shows the exact JSON first)")
    r.add_argument("id")
    r.add_argument("--notes")
    r.add_argument("--author")
    r.add_argument("-y", "--yes", action="store_true")
    r = rs.add_parser("validate", help="validate a recipe JSON file ('-' for stdin)")
    r.add_argument("file")
    r = rs.add_parser("id", help="compute a recipe's content id")
    r.add_argument("file")

    s = add("savings", cmd_savings, "how much work stayed local")
    s.add_argument("--days", type=int)
    s.add_argument("--badge", help="write a shareable SVG badge")
    s.add_argument("--json", action="store_true")

    add("update", cmd_update, "refresh the registry and recipe index from GitHub")
    add("mcp", cmd_mcp, "run the MCP server (stdio) for Claude Code, Cursor, Codex, …")

    s = add("ui", cmd_ui, "open the exchange in your browser (served locally)")
    s.add_argument("--port", type=int, default=8484)
    s.add_argument("--no-open", action="store_true")

    s = add("demo", cmd_demo, "a 60-second tour (simulated unless --real)")
    s.add_argument("--real", action="store_true", help="use your installed Ollama models")
    s.add_argument("--fast", action="store_true")
    s.add_argument("--export-svg")

    s = add("registry", cmd_registry, "maintainers: validate listings + recipes and rebuild the index")
    s.add_argument("--check", action="store_true", help="fail if the built index is out of date")
    s.add_argument("--refresh-sizes", action="store_true", help="look up model sizes from the Ollama registry")
    return p


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if not getattr(args, "fn", None):
        console.print(LOGO)
        p.print_help()
        return 0
    try:
        return args.fn(args) or 0
    except BackendError as e:
        err.print(f"[red]{e}[/]")
        return 1
    except (KeyError, ValueError, FileNotFoundError) as e:
        err.print(f"[red]{e}[/]")
        return 1
    except KeyboardInterrupt:
        return 130
