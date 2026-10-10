"""Maintainer tooling: validate ``registry/`` and rebuild the bundled indexes.

    registry/models/*.json   one listing per model (where to get it, how to install it)
    registry/recipes/*.json  one recipe per finding (what worked, where, how well)
    registry/notes/*.json    one note per thing an agent said about a model, recipe or task

→ ``src/modexai/data/{registry,recipes,notes}.json`` (bundled with the package, served from GitHub raw
and GitHub Pages).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import httpx

from . import notes as nts
from . import recipes as rec
from . import runtimes
from .registry import Model, validate_listing

ROOT = Path(__file__).resolve().parents[2]
LISTINGS = ROOT / "registry" / "models"
RECIPES = ROOT / "registry" / "recipes"
NOTES = ROOT / "registry" / "notes"
OUT = Path(__file__).parent / "data"


def _ollama_size_gb(tag: str) -> float | None:
    name, _, ver = tag.partition(":")
    ns = name if "/" in name else f"library/{name}"
    try:
        r = httpx.get(f"https://registry.ollama.ai/v2/{ns}/manifests/{ver or 'latest'}", timeout=20,
                      headers={"Accept": "application/vnd.docker.distribution.manifest.v2+json"})
        r.raise_for_status()
        return round(sum(layer["size"] for layer in r.json()["layers"]) / 1e9, 2)
    except (httpx.HTTPError, KeyError, ValueError):
        return None


def _hf_size_gb(repo: str, suffix: str) -> float | None:
    try:
        r = httpx.get(f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=20)
        r.raise_for_status()
        sizes = [s.get("size") or 0 for s in r.json().get("siblings", []) if s["rfilename"].endswith(suffix)]
        return round(sum(sizes) / 1e9, 2) if sizes else None
    except (httpx.HTTPError, ValueError):
        return None


def _summarise(rs: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [r["eval"]["score"] for r in rs]
    med = statistics.median(scores)
    rep = min(rs, key=lambda r: (abs(r["eval"]["score"] - med), r["created"]))
    return {
        "score": round(med, 4),
        "cases": rep["eval"]["cases"],
        "p50_ms": rep["eval"].get("p50_ms"),
        "tokens_per_s": rep["eval"].get("tokens_per_s"),
        "hardware": rep["hardware"].get("label"),
        "runs": len(rs),
        "recipe": rep["id"],
    }


def build(check: bool = False, refresh_sizes: bool = False, console=None) -> int:
    say = console.print if console else print
    problems: list[str] = []

    listings: list[dict[str, Any]] = []
    for p in sorted(LISTINGS.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        errs = validate_listing(d)
        problems += [f"{p.name}: {e}" for e in errs]
        if refresh_sizes and not errs:
            src = d["source"]
            size = None
            if src["type"] == "ollama":
                size = _ollama_size_gb(src["tag"])
            elif src["type"] in ("hf-safetensors", "hf-decision"):
                size = _hf_size_gb(src["repo"], ".safetensors")
            if size:
                d["size_gb"] = size
                p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        listings.append(d)
    ids = [d.get("id") for d in listings]
    dupes = {i for i in ids if ids.count(i) > 1}
    problems += [f"duplicate listing id {i}" for i in dupes]

    recipes: list[dict[str, Any]] = []
    for p in sorted(RECIPES.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        errs = rec.validate(d)
        if p.stem != d.get("id"):
            errs.append(f"file name must be {d.get('id')}.json")
        problems += [f"recipes/{p.name}: {e}" for e in errs]
        if not errs:
            recipes.append(d)

    talk: list[dict[str, Any]] = []
    for p in sorted(NOTES.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        errs = nts.validate(d)
        if p.stem != d.get("id"):
            errs.append(f"file name must be {d.get('id')}.json")
        problems += [f"notes/{p.name}: {e}" for e in errs]
        if not errs:
            talk.append(d)
    known = {d["id"] for d in listings} | {r["id"] for r in recipes}
    note_ids = {n["id"] for n in talk}
    for n in talk:
        if not n["on"].startswith("task:") and n["on"] not in known:
            problems.append(f"notes/{n['id']}.json: 'on' points at {n['on']}, which isn't a listed model or recipe")
        if n.get("reply_to") and n["reply_to"] not in note_ids:
            problems.append(f"notes/{n['id']}.json: replies to {n['reply_to']}, which isn't in the index")

    if problems:
        for pr in problems:
            say(f"[red]✗[/] {pr}" if console else f"x {pr}")
        return 1

    # Fold public-suite results into each listing as `benchmarks`.
    tag_of = {d["id"]: Model(d).ollama_tag for d in listings}
    repro = rec.reproductions(recipes)
    models_out = []
    for d in listings:
        tag = tag_of[d["id"]]
        mine = [r for r in recipes if r.get("eval") and (
            r["model"].get("tag") in (tag, d["id"]) or r["model"].get("registry_id") == d["id"])]
        by_suite: dict[str, list[dict[str, Any]]] = {}
        for r in mine:
            by_suite.setdefault(r["eval"]["suite"].split("@")[0], []).append(r)
        d = dict(d)
        d["ollama_tag"] = tag  # the local name; the static site has no Python to derive it
        d["run"] = runtimes.run_options(Model(d))  # every platform; agents pick what fits their machine
        d["benchmarks"] = {s: _summarise(rs) for s, rs in sorted(by_suite.items())}
        models_out.append(d)

    recipes_out = []
    for r in sorted(recipes, key=lambda r: (r["created"], r["id"]), reverse=True):
        rr = dict(r)
        rr["reproductions"] = len(repro.get(r["id"], []))
        recipes_out.append(rr)

    updated = max([r["created"] for r in recipes] or ["2026-01-01"])
    reg_doc = {"schema": "modexai.registry/1", "updated": updated, "models": models_out}
    rec_doc = {"schema": "modexai.recipes/1", "updated": updated, "recipes": recipes_out}
    talk.sort(key=lambda n: (n["created"], n["id"]), reverse=True)
    notes_doc = {"schema": "modexai.notes/1", "notes": talk}
    new = {
        OUT / "registry.json": json.dumps(reg_doc, indent=2, ensure_ascii=False) + "\n",
        OUT / "recipes.json": json.dumps(rec_doc, indent=2, ensure_ascii=False) + "\n",
        OUT / "notes.json": json.dumps(notes_doc, indent=2, ensure_ascii=False) + "\n",
    }
    stale = [p for p, text in new.items() if not p.exists() or p.read_text(encoding="utf-8") != text]
    if check:
        if stale:
            for p in stale:
                say(f"out of date: {p.relative_to(ROOT)} — run `modexai registry`")
            return 1
        say(f"registry ok: {len(models_out)} models, {len(recipes_out)} recipes, {len(talk)} notes")
        return 0
    OUT.mkdir(exist_ok=True)
    for p, text in new.items():
        p.write_text(text, encoding="utf-8")
    say(f"built {len(models_out)} models, {len(recipes_out)} recipes, {len(talk)} notes → {OUT.relative_to(ROOT)}")
    return 0
