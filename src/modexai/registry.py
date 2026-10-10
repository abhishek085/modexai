"""The registry: an open, static list of small models an agent can hire.

The registry is a single JSON document. ModexAI ships with a snapshot, can refresh it from GitHub on
request (``modexai update``), and merges any private listings you keep in ``~/.modexai/listings``.
No server, no account, no tracking: reading the registry is one GET of a public file — or zero.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from . import config

SOURCE_TYPES = ("ollama", "hf-gguf", "hf-safetensors", "hf-decision")
KINDS = ("chat", "embedding", "decision")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*$")
_TASK_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


@dataclass
class Model:
    raw: dict[str, Any]
    benchmarks: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.raw["id"]

    @property
    def name(self) -> str:
        return self.raw.get("name", self.id)

    @property
    def publisher(self) -> str:
        return self.raw.get("publisher", self.id.split("/")[0])

    @property
    def summary(self) -> str:
        return self.raw.get("summary", "")

    @property
    def tasks(self) -> list[str]:
        return list(self.raw.get("tasks", []))

    @property
    def kind(self) -> str:
        return self.raw.get("kind", "chat")

    @property
    def size_gb(self) -> float:
        return float(self.raw.get("size_gb") or 0)

    @property
    def params_b(self) -> float | None:
        v = self.raw.get("params_b")
        return float(v) if v is not None else None

    @property
    def license(self) -> str:
        return self.raw.get("license", "unknown")

    @property
    def source(self) -> dict[str, Any]:
        return self.raw.get("source", {})

    @property
    def ollama_tag(self) -> str:
        """The name this model has inside Ollama once installed."""
        src = self.source
        t = src.get("type")
        if t == "ollama":
            return src["tag"]
        if t == "hf-gguf":
            quant = src.get("quant")
            return f"hf.co/{src['repo']}" + (f":{quant}" if quant else "")
        if t == "hf-safetensors":
            return src["ollama_tag"]
        if t == "hf-decision":
            return src["repo"].rsplit("/", 1)[-1]  # the release name `osj serve --pull` uses
        return self.id

    def best_score(self, suite: str | None = None) -> float | None:
        """Best measured score on a suite (or across all suites when suite is None)."""
        if suite:
            b = self.benchmarks.get(suite)
            return b["score"] if b else None
        scores = [b["score"] for b in self.benchmarks.values()]
        return max(scores) if scores else None

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.raw.items() if not k.startswith("_")}
        d["ollama_tag"] = self.ollama_tag
        d["benchmarks"] = self.benchmarks
        return d


class Registry:
    def __init__(self, models: list[Model], meta: dict[str, Any] | None = None):
        self.models = models
        self.meta = meta or {}
        self._by_id = {m.id: m for m in models}
        self._by_tag = {m.ollama_tag: m for m in models}

    def __len__(self) -> int:
        return len(self.models)

    def get(self, ref: str) -> Model | None:
        """Look a model up by registry id (``qwen/qwen3-0.6b``) or Ollama tag (``qwen3:0.6b``)."""
        if ref in self._by_id:
            return self._by_id[ref]
        if ref in self._by_tag:
            return self._by_tag[ref]
        if ":" not in ref and f"{ref}:latest" in self._by_tag:
            return self._by_tag[f"{ref}:latest"]
        if ref.endswith(":latest") and ref[: -len(":latest")] in self._by_tag:
            return self._by_tag[ref[: -len(":latest")]]
        return None

    def tasks(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for m in self.models:
            for t in m.tasks:
                counts[t] = counts.get(t, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def search(
        self,
        query: str = "",
        task: str | None = None,
        max_size_gb: float | None = None,
        kind: str | tuple[str, ...] | None = "chat",
        include_general: bool = True,
        installed: set[str] | None = None,
        installed_only: bool = False,
        limit: int | None = None,
    ) -> list[Model]:
        words = [w for w in re.split(r"[^a-z0-9.]+", query.lower()) if w]
        ranked: list[tuple[tuple, Model]] = []
        for m in self.models:
            if kind and m.kind not in ((kind,) if isinstance(kind, str) else kind):
                continue
            if max_size_gb is not None and m.size_gb and m.size_gb > max_size_gb:
                continue
            is_installed = installed is not None and is_tag_installed(m.ollama_tag, installed)
            if installed_only and not is_installed:
                continue
            specialist = bool(task) and task in m.tasks
            if task and not specialist and not (include_general and "general" in m.tasks):
                continue
            text = " ".join([m.id, m.name, m.summary, " ".join(m.tasks), m.raw.get("base_model", "")]).lower()
            hits = sum(1 for w in words if w in text)
            if words and hits == 0:
                continue
            score = m.best_score(task) if task else m.best_score()
            key = (
                -hits,
                0 if specialist else 1,
                -(score if score is not None else -1),
                m.size_gb or 99,
            )
            ranked.append((key, m))
        ranked.sort(key=lambda kv: kv[0])
        out = [m for _, m in ranked]
        return out[:limit] if limit else out


def is_tag_installed(tag: str, installed: set[str]) -> bool:
    if tag in installed:
        return True
    if ":" not in tag:
        return f"{tag}:latest" in installed
    if tag.endswith(":latest"):
        return tag[: -len(":latest")] in installed
    return False


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _cached_registry_path() -> Path:
    return config.home() / "registry.json"


def _read_source(src: str) -> dict[str, Any]:
    if src.startswith(("http://", "https://")):
        r = httpx.get(src, timeout=20, follow_redirects=True)
        r.raise_for_status()
        return r.json()
    return json.loads(Path(src).expanduser().read_text(encoding="utf-8"))


def load_registry(source: str | None = None, include_private: bool = True) -> Registry:
    """Load the registry.

    Resolution order: explicit ``source`` → ``$MODEXAI_REGISTRY`` → cached copy from ``modexai update``
    → the snapshot bundled with this package. Private listings in ``~/.modexai/listings/*.json`` are
    merged on top (they never leave your machine).
    """
    src = source or os.environ.get("MODEXAI_REGISTRY")
    if src:
        doc = _read_source(src)
    elif _cached_registry_path().exists():
        doc = json.loads(_cached_registry_path().read_text(encoding="utf-8"))
    else:
        doc = json.loads(config.BUNDLED_REGISTRY.read_text(encoding="utf-8"))

    models = [Model(raw=m, benchmarks=m.get("benchmarks", {})) for m in doc.get("models", [])]

    if include_private:
        pdir = config.home() / "listings"
        if pdir.exists():
            known = {m.id for m in models}
            for p in sorted(pdir.glob("*.json")):
                try:
                    raw = json.loads(p.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                if validate_listing(raw):
                    continue
                raw["_private"] = True
                if raw["id"] in known:
                    models = [m for m in models if m.id != raw["id"]]
                models.append(Model(raw=raw, benchmarks=raw.get("benchmarks", {})))

    meta = {k: v for k, v in doc.items() if k != "models"}
    return Registry(models, meta)


def update_registry(url: str = config.REMOTE_REGISTRY_URL) -> tuple[Path, int]:
    """Fetch the latest public registry and cache it locally."""
    doc = _read_source(url)
    if "models" not in doc:
        raise ValueError("Not a ModexAI registry document")
    path = _cached_registry_path()
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path, len(doc["models"])


# ---------------------------------------------------------------------------
# Validation (no jsonschema dependency — the rules are small and explicit)
# ---------------------------------------------------------------------------


def validate_listing(d: Any) -> list[str]:
    """Return a list of problems with a listing; empty means valid."""
    errs: list[str] = []
    if not isinstance(d, dict):
        return ["listing must be a JSON object"]

    def need(key: str, typ: type | tuple[type, ...]) -> Any:
        if key not in d:
            errs.append(f"missing required field '{key}'")
            return None
        if not isinstance(d[key], typ):
            errs.append(f"'{key}' has the wrong type")
            return None
        return d[key]

    mid = need("id", str)
    if mid and not _ID_RE.match(mid):
        errs.append("'id' must look like 'publisher/model-name' (lowercase, digits, . _ -)")
    need("name", str)
    need("publisher", str)
    summary = need("summary", str)
    if summary and len(summary) > 220:
        errs.append("'summary' should be at most 220 characters")
    need("license", str)
    tasks = need("tasks", list)
    if tasks is not None:
        if not tasks:
            errs.append("'tasks' must list at least one task")
        for t in tasks:
            if not isinstance(t, str) or not _TASK_RE.match(t):
                errs.append(f"task {t!r} must be lowercase-kebab-case")
    kind = d.get("kind", "chat")
    if kind not in KINDS:
        errs.append(f"'kind' must be one of {KINDS}")
    size = d.get("size_gb")
    if size is not None and (not isinstance(size, (int, float)) or size < 0):
        errs.append("'size_gb' must be a non-negative number")
    params = d.get("params_b")
    if params is not None and (not isinstance(params, (int, float)) or params <= 0):
        errs.append("'params_b' must be a positive number (billions of parameters)")

    src = need("source", dict)
    if src is not None:
        t = src.get("type")
        if t not in SOURCE_TYPES:
            errs.append(f"source.type must be one of {SOURCE_TYPES}")
        elif t == "ollama" and not isinstance(src.get("tag"), str):
            errs.append("ollama sources need 'tag' (e.g. 'qwen3:0.6b')")
        elif t == "hf-gguf" and not isinstance(src.get("repo"), str):
            errs.append("hf-gguf sources need 'repo' (e.g. 'org/model-GGUF')")
        elif t == "hf-safetensors":
            if not isinstance(src.get("repo"), str):
                errs.append("hf-safetensors sources need 'repo'")
            if not isinstance(src.get("ollama_tag"), str):
                errs.append("hf-safetensors sources need 'ollama_tag' (the local name to create)")
        elif t == "hf-decision" and not isinstance(src.get("repo"), str):
            errs.append("hf-decision sources need 'repo' (e.g. 'org/spark-s1-4b-v6')")
        if (t == "hf-decision") != (kind == "decision"):
            errs.append("decision models use source.type 'hf-decision' and kind 'decision', together")

    mf = d.get("modelfile")
    if mf is not None:
        if not isinstance(mf, dict):
            errs.append("'modelfile' must be an object")
        else:
            unknown = set(mf) - {"system", "template", "parameters"}
            if unknown:
                errs.append(f"unknown modelfile keys: {sorted(unknown)}")
    return errs
