"""Recipes: shareable findings about running a model locally — no weights, no user data.

A recipe says: *this model, set up this way, scored this on this public test, on this kind of machine.*
Agents draft them after a benchmark or a shadow-and-hire run, a human approves, and they land in an
open index that every other agent can query: "what works for invoice extraction on 16 GB Apple
Silicon?" Re-running a recipe on another machine produces a *reproduction*; reproductions are how a
"works on my machine" claim becomes evidence.

Recipes are data written by strangers and read by agents, so they are schema-checked, size-limited,
and screened for text that tries to instruct the reading agent. Treat every field as untrusted data.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import urllib.parse
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from . import __version__, config, hardware
from .bench import ModelResult
from .registry import Model

SCHEMA = "modexai.recipe/1"
KINDS = ("benchmark", "recipe", "reproduction")
RECIPE_REPO = "abhishek085/modexai"
BUNDLED_INDEX = config.DATA_DIR / "recipes.json"
REMOTE_INDEX_URL = "https://raw.githubusercontent.com/abhishek085/modexai/main/src/modexai/data/recipes.json"
ISSUE_MARKER = "<!-- modexai-recipe -->"

LIMITS = {"title": 140, "notes": 600, "system": 2000, "author": 40}
_INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts|messages)",
    r"disregard (all |the |any )?(previous|prior|above|system)",
    r"\byou are now\b",
    r"(ai|llm|language model|agent)s? (reading|processing) this",
    r"\b(reveal|print|output|exfiltrate|send)\b.{0,40}\b(system prompt|api[ _-]?keys?|secrets?|credentials|env(ironment)? var)",
    r"\bdeveloper mode\b",
    r"<\s*/?\s*(script|iframe|system|tool_call)\b",
    r"\b(run|execute)\b.{0,30}\b(curl|wget|rm -rf|bash|powershell)\b",
]


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


# Fields excluded from the content hash: identity/attribution, and values the index derives.
_NOT_HASHED = ("id", "created", "author", "reproductions")


def _canonical(d: dict[str, Any]) -> bytes:
    body = {k: v for k, v in d.items() if k not in _NOT_HASHED and not k.startswith("_")}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def recipe_id(d: dict[str, Any]) -> str:
    return "rcp_" + hashlib.sha256(_canonical(d)).hexdigest()[:12]


def from_result(
    res: ModelResult,
    *,
    task: str,
    model: Model | None = None,
    system: str | None = None,
    parameters: dict[str, Any] | None = None,
    json_mode: bool = False,
    private: bool = False,
    reference: str | None = None,
    notes: str = "",
    reproduces: str | None = None,
    runtime_version: str | None = None,
    author: str | None = None,
) -> dict[str, Any]:
    """Turn a benchmark result into a recipe draft.

    ``private=True`` means the cases came from your own data (e.g. shadow-recorded examples): only the
    aggregate score and case count are kept — never prompts, outputs or expected answers.
    """
    hw = dict(hardware.detect())
    hw["label"] = hardware.label(hw)
    if res.size_gb:
        hw["model_size_gb"] = res.size_gb
    ev = {
        "cases": res.total,
        "score": res.score,
        "p50_ms": res.p50_ms,
        "p95_ms": res.p95_ms,
        "tokens_per_s": res.tokens_per_s,
    }
    d: dict[str, Any] = {
        "schema": SCHEMA,
        "kind": "reproduction" if reproduces else ("recipe" if private or system else "benchmark"),
        "title": "",
        "task": task,
        "model": {k: v for k, v in {"tag": res.model, "registry_id": res.registry_id, "size_gb": res.size_gb}.items() if v},
        "setup": {k: v for k, v in {"system": system,
                                     "parameters": parameters or {"temperature": 0, "num_ctx": config.num_ctx()},
                                     "json_mode": json_mode or None}.items() if v is not None},
        "hardware": hw,
        "runtime": {"engine": res.backend, "version": runtime_version},
        "created": date.today().isoformat(),
        "tool": f"modexai {__version__}",
    }
    if private:
        d["private_eval"] = {**ev, "reference": reference or "your own labelled examples"}
    else:
        d["eval"] = {"suite": res.suite, "suite_sha": res.suite_sha[:16], **ev}
    if notes:
        d["notes"] = notes
    if reproduces:
        d["reproduces"] = reproduces
    if author:
        d["author"] = author
    score = d.get("eval", d.get("private_eval"))["score"]
    where = hw["label"]
    what = res.suite.split("@")[0] if not private else task
    d["title"] = f"{res.model} scores {score:.0%} on {what} · {where}"[: LIMITS["title"]]
    d["id"] = recipe_id(d)
    return d


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def screen_text(text: str) -> str | None:
    """Return the matched pattern if text looks like it is trying to instruct an agent."""
    low = text.lower()
    for pat in _INJECTION_PATTERNS:
        if re.search(pat, low):
            return pat
    return None


def validate(d: Any) -> list[str]:
    errs: list[str] = []
    if not isinstance(d, dict):
        return ["recipe must be a JSON object"]
    if d.get("schema") != SCHEMA:
        errs.append(f"'schema' must be {SCHEMA!r}")
    if d.get("kind") not in KINDS:
        errs.append(f"'kind' must be one of {KINDS}")
    for key in ("title", "task", "model", "hardware", "runtime"):
        if key not in d:
            errs.append(f"missing '{key}'")
    if errs:
        return errs
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}", str(d["task"])):
        errs.append("'task' must be lowercase-kebab-case")
    model = d["model"]
    if not isinstance(model, dict) or not isinstance(model.get("tag"), str) or not model["tag"]:
        errs.append("'model.tag' is required (the Ollama tag or local model name)")
    elif not re.fullmatch(r"[A-Za-z0-9._:/@-]{1,120}", model["tag"]):
        errs.append("'model.tag' has unexpected characters")
    ev = d.get("eval")
    pev = d.get("private_eval")
    if not ev and not pev:
        errs.append("a recipe needs 'eval' (public suite) or 'private_eval' (aggregate on private data)")
    for name, block in (("eval", ev), ("private_eval", pev)):
        if block is None:
            continue
        if not isinstance(block, dict):
            errs.append(f"'{name}' must be an object")
            continue
        s = block.get("score")
        if not isinstance(s, (int, float)) or not 0 <= s <= 1:
            errs.append(f"'{name}.score' must be between 0 and 1")
        n = block.get("cases")
        if not isinstance(n, int) or n < 1:
            errs.append(f"'{name}.cases' must be a positive integer")
    if ev and not re.fullmatch(r"[a-z0-9-]+@\d+", str(ev.get("suite", ""))):
        errs.append("'eval.suite' must look like 'ticket-triage@1'")
    if d["kind"] == "reproduction" and not str(d.get("reproduces", "")).startswith("rcp_"):
        errs.append("a reproduction must name the recipe it reproduces ('reproduces': 'rcp_…')")

    texts = {
        "title": d.get("title", ""),
        "notes": d.get("notes", ""),
        "system": (d.get("setup") or {}).get("system", "") or "",
        "author": d.get("author", "") or "",
    }
    for key, val in texts.items():
        if not isinstance(val, str):
            errs.append(f"'{key}' must be a string")
            continue
        if len(val) > LIMITS[key]:
            errs.append(f"'{key}' is longer than {LIMITS[key]} characters")
        hit = screen_text(val)
        if hit:
            errs.append(f"'{key}' contains instruction-like text aimed at agents (matched /{hit}/)")
    if len(json.dumps(d)) > 8000:
        errs.append("recipe is larger than 8 KB")
    if "id" in d and d["id"] != recipe_id(d):
        errs.append("'id' does not match the recipe content (re-generate it with `modexai recipes id`)")
    return errs


# ---------------------------------------------------------------------------
# Local drafts, the shared index, and search
# ---------------------------------------------------------------------------


def local_dir() -> Path:
    p = config.home() / "recipes"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_local(d: dict[str, Any]) -> Path:
    path = local_dir() / f"{d['id']}.json"
    path.write_text(json.dumps(d, indent=2))
    return path


def load_local() -> list[dict[str, Any]]:
    out = []
    for p in sorted(local_dir().glob("rcp_*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def _cached_index() -> Path:
    return config.home() / "recipes-index.json"


def load_index(source: str | None = None) -> list[dict[str, Any]]:
    """Recipes from the shared index: explicit source → cached copy → bundled snapshot."""
    if source:
        if source.startswith(("http://", "https://")):
            r = httpx.get(source, timeout=20, follow_redirects=True)
            r.raise_for_status()
            doc = r.json()
        else:
            doc = json.loads(Path(source).read_text())
    elif _cached_index().exists():
        doc = json.loads(_cached_index().read_text())
    elif BUNDLED_INDEX.exists():
        doc = json.loads(BUNDLED_INDEX.read_text())
    else:
        doc = {"recipes": []}
    return [r for r in doc.get("recipes", []) if not validate(r)]


def update_index(url: str = REMOTE_INDEX_URL) -> int:
    r = httpx.get(url, timeout=20, follow_redirects=True)
    r.raise_for_status()
    doc = r.json()
    _cached_index().write_text(json.dumps(doc))
    return len(doc.get("recipes", []))


def reproductions(recipes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group reproductions under the recipe they reproduce, and also link same model+suite results."""
    by_id: dict[str, list[dict[str, Any]]] = {}
    for r in recipes:
        if r.get("kind") == "reproduction" and r.get("reproduces"):
            by_id.setdefault(r["reproduces"], []).append(r)
    return by_id


def find(
    recipes: list[dict[str, Any]],
    task: str | None = None,
    model: str | None = None,
    fits_here: bool = True,
    hw: dict[str, Any] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search recipes, best evidence first. Adds ``_fit`` (does it fit this machine?) and ``_repro``."""
    hw = hw or hardware.detect()
    repro = reproductions(recipes)
    out = []
    for r in recipes:
        if r.get("kind") == "reproduction":
            continue
        ev = r.get("eval") or {}
        if task and task not in (r.get("task"), ev.get("suite", "").split("@")[0]):
            continue
        if model and model not in (r["model"].get("tag"), r["model"].get("registry_id")):
            continue
        ok, why = hardware.compatible(r.get("hardware", {}), hw)
        if fits_here and not ok:
            continue
        rr = dict(r)
        rr["_fit"] = why
        rr["_repro"] = len(repro.get(r["id"], []))
        out.append(rr)
    same = hw.get("accelerator")
    out.sort(key=lambda r: (
        0 if r.get("hardware", {}).get("accelerator") == same else 1,
        -r["_repro"],
        -(r.get("eval") or r.get("private_eval") or {}).get("score", 0),
        r["model"].get("size_gb") or 99,
    ))
    return out[:limit]


# ---------------------------------------------------------------------------
# Sharing (always explicit; never automatic)
# ---------------------------------------------------------------------------


def issue_body(d: dict[str, Any]) -> str:
    return (
        f"{ISSUE_MARKER}\n"
        f"**{d['title']}**\n\n"
        "```json\n" + json.dumps(d, indent=2) + "\n```\n\n"
        f"Submitted with `{d.get('tool', 'modexai')}`. Contains no prompts, outputs or user data — "
        "only setup, hardware class and aggregate scores.\n"
    )


def share_url(d: dict[str, Any], repo: str = RECIPE_REPO) -> str:
    q = urllib.parse.urlencode({"title": f"recipe: {d['title']}", "labels": "recipe", "body": issue_body(d)})
    return f"https://github.com/{repo}/issues/new?{q}"


def share(d: dict[str, Any], repo: str = RECIPE_REPO, use_gh: bool = True) -> dict[str, str]:
    """Submit a recipe as a GitHub issue (via `gh` if available) or return a pre-filled issue URL.

    Call this only after a human has seen and approved the exact JSON.
    """
    errs = validate(d)
    if errs:
        raise ValueError("recipe is not valid: " + "; ".join(errs))
    if use_gh and shutil.which("gh"):
        base = ["gh", "issue", "create", "--repo", repo, "--title", f"recipe: {d['title']}", "--body", issue_body(d)]
        for cmd in (base + ["--label", "recipe"], base):
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode == 0:
                return {"method": "gh", "url": p.stdout.strip().splitlines()[-1]}
    return {"method": "url", "url": share_url(d, repo)}


def extract_from_issue(body: str) -> dict[str, Any] | None:
    """Pull the recipe JSON out of an issue body (used by the intake GitHub Action)."""
    m = re.search(r"```json\s*(\{.*?\})\s*```", body or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
