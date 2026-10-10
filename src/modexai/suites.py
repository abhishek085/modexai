"""Benchmark suites: small, hand-written, deterministic test sets for jobs agents actually delegate.

Each suite is a JSON file. Items carry a ``check`` (see :mod:`modexai.checks`); ``check_defaults`` at
the suite level is merged into every item. Suites are versioned and content-hashed so a recipe can
say exactly which test it ran, and anyone can re-run the same one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config
from .checks import infer_check


@dataclass
class Suite:
    id: str
    version: int
    name: str
    task: str
    description: str
    items: list[dict[str, Any]]
    system: str | None = None
    prompt_prefix: str = ""
    max_tokens: int = 256
    json_mode: bool = False
    sha: str = ""
    builtin: bool = True

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"

    def messages(self, item: dict[str, Any]) -> list[dict[str, str]]:
        msgs = []
        system = item.get("system", self.system)
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": self.prompt_prefix + item["prompt"]})
        return msgs

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ref": self.ref,
            "name": self.name,
            "task": self.task,
            "description": self.description,
            "cases": len(self.items),
            "sha": self.sha[:12],
        }


def _from_doc(doc: dict[str, Any], builtin: bool) -> Suite:
    defaults = doc.get("check_defaults", {})
    items = []
    for i, it in enumerate(doc["items"]):
        it = dict(it)
        it.setdefault("id", f"{doc['id']}-{i + 1:03d}")
        it["check"] = {**defaults, **it.get("check", {})}
        items.append(it)
    canonical = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    return Suite(
        id=doc["id"],
        version=int(doc.get("version", 1)),
        name=doc.get("name", doc["id"]),
        task=doc.get("task", doc["id"]),
        description=doc.get("description", ""),
        items=items,
        system=doc.get("system"),
        prompt_prefix=doc.get("prompt_prefix", ""),
        max_tokens=int(doc.get("max_tokens", 256)),
        json_mode=bool(doc.get("json_mode", False)),
        sha=hashlib.sha256(canonical).hexdigest(),
        builtin=builtin,
    )


def list_suites() -> list[Suite]:
    return [_from_doc(json.loads(p.read_text()), True) for p in sorted(config.SUITES_DIR.glob("*.json"))]


def get_suite(ref: str) -> Suite:
    """Load a built-in suite by id (``ticket-triage`` or ``ticket-triage@1``) or a suite JSON file path."""
    sid = ref.split("@", 1)[0]
    path = config.SUITES_DIR / f"{sid}.json"
    if path.exists():
        return _from_doc(json.loads(path.read_text()), True)
    p = Path(ref).expanduser()
    if p.exists():
        return _from_doc(json.loads(p.read_text()), False)
    known = ", ".join(s.id for s in list_suites())
    raise KeyError(f"unknown suite {ref!r}. Built-in suites: {known}")


def suite_for_task(task: str) -> Suite | None:
    for s in list_suites():
        if task in (s.id, s.task):
            return s
    return None


def suite_from_samples(
    samples: list[dict[str, Any]],
    name: str = "custom",
    system: str | None = None,
    labels: list[str] | None = None,
) -> Suite:
    """Build a private, in-memory suite from ``[{"prompt": ..., "expected": ...}, ...]``.

    ``expected`` can be a label, a number, a JSON object or free text; the right check is inferred.
    An explicit ``check`` on a sample wins. Private suites are never written anywhere.
    """
    if labels is None:
        exp = [s.get("expected") for s in samples if isinstance(s.get("expected"), str)]
        distinct = {e.strip() for e in exp if e and len(e.split()) <= 4}
        if exp and len(distinct) <= max(2, len(exp) // 2) and len(distinct) <= 20:
            labels = sorted(distinct)
    items = []
    for i, s in enumerate(samples):
        if "prompt" not in s:
            raise ValueError(f"sample {i} has no 'prompt'")
        check = s.get("check") or (infer_check(s["expected"], labels) if "expected" in s else None)
        if check is None:
            raise ValueError(f"sample {i} needs 'expected' or 'check'")
        items.append({"id": s.get("id", f"{name}-{i + 1:03d}"), "prompt": s["prompt"], "check": check,
                      **({"system": s["system"]} if s.get("system") else {})})
    doc = {"id": name, "version": 1, "name": name, "task": name, "items": items}
    if system:
        doc["system"] = system
    if labels and system is None:
        doc["system"] = f"Answer with exactly one of: {', '.join(labels)}. No explanation."
    suite = _from_doc(doc, False)
    if any(it["check"]["type"] == "json_fields" for it in suite.items):
        suite.json_mode = True
        suite.max_tokens = 320
    return suite


def load_samples_file(path: str | Path) -> list[dict[str, Any]]:
    """Read samples from .jsonl (one object per line) or .json (a list)."""
    p = Path(path).expanduser()
    text = p.read_text()
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    if not isinstance(data, list):
        raise ValueError("samples file must be a JSON list or JSONL")
    return data
