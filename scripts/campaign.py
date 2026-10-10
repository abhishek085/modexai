"""Benchmark every installed, listed model on the suites relevant to it and write recipes.

Maintainers use this to seed `registry/recipes/` with real measurements; anyone can run it to contribute
results for their own hardware:

    python scripts/campaign.py                      # all installed listed models × relevant suites
    python scripts/campaign.py --suites ticket-triage --models qwen3:1.7b gemma3:1b
    python scripts/campaign.py --out ~/.modexai/recipes   # keep results local instead
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from modexai import recipes
from modexai.backends import OllamaBackend
from modexai.bench import run_benchmark
from modexai.registry import is_tag_installed, load_registry
from modexai.suites import list_suites

ROOT = Path(__file__).resolve().parents[1]


def relevant(model, suite) -> bool:
    return suite.task in model.tasks or "general" in model.tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="*")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--out", default=str(ROOT / "registry" / "recipes"))
    ap.add_argument("--author")
    args = ap.parse_args()

    backend = OllamaBackend()
    reg = load_registry(include_private=False)
    installed = backend.installed()
    version = backend.version()
    suites = [s for s in list_suites() if not args.suites or s.id in args.suites]
    models = [m for m in reg.models if m.kind == "chat" and is_tag_installed(m.ollama_tag, installed)]
    if args.models:
        models = [m for m in models if m.ollama_tag in args.models or m.id in args.models]
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    plan = [(m, s) for s in suites for m in models if relevant(m, s)]
    print(f"{len(plan)} runs: {len(models)} models × {len(suites)} suites", flush=True)
    t0 = time.time()
    for i, (m, s) in enumerate(plan, 1):
        res = run_benchmark([m.ollama_tag], s, backend, registry=reg)[0]
        if res.error:
            print(f"[{i}/{len(plan)}] {m.ollama_tag} × {s.id}: ERROR {res.error}", flush=True)
            continue
        r = recipes.from_result(res, task=s.task, model=m, json_mode=s.json_mode, runtime_version=version,
                                author=args.author)
        problems = recipes.validate(r)
        if problems:
            print(f"  invalid recipe: {problems}", file=sys.stderr)
            continue
        (out / f"{r['id']}.json").write_text(json.dumps(r, indent=2) + "\n")
        print(f"[{i}/{len(plan)}] {m.ollama_tag:<24} {s.id:<20} {res.score:>5.0%}  p50 {res.p50_ms} ms  "
              f"({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
