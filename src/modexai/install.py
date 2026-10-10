"""Get a listed model running in Ollama with one call.

- ``ollama``         → ``ollama pull <tag>``
- ``hf-gguf``        → ``ollama pull hf.co/<repo>:<quant>``
- ``hf-safetensors`` → download the repo, fix known config incompatibilities, write a Modelfile and
                       ``ollama create`` it. (Transformers v5 moved ``rope_theta`` under
                       ``rope_parameters``; Ollama's importer still reads the old key and silently falls
                       back to the wrong RoPE base, producing gibberish. We patch that.)
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from . import config
from .backends import BackendError, OllamaBackend
from .registry import Model

Progress = Callable[[str, float | None], None]

CHATML_TEMPLATE = (
    "{{- if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}"
    "{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}"
    "<|im_start|>assistant\n"
)
TEMPLATES = {"chatml": CHATML_TEMPLATE}
_HF_WANTED = (".json", ".safetensors", ".jinja", ".model", ".txt")


def install(model: Model, backend: OllamaBackend | None = None, progress: Progress | None = None) -> str:
    """Install a registry model; returns the Ollama tag to use."""
    backend = backend or OllamaBackend()
    say = progress or (lambda *_: None)
    src = model.source
    t = src.get("type")
    if t in ("ollama", "hf-gguf"):
        pull(model.ollama_tag, backend, say)
    elif t == "hf-safetensors":
        _install_safetensors(model, say)
    else:
        raise BackendError(f"don't know how to install source type {t!r}")
    return model.ollama_tag


def pull(tag: str, backend: OllamaBackend, say: Progress) -> None:
    last = ""
    for ev in backend.pull(tag):
        status = ev.get("status", "")
        total, done = ev.get("total"), ev.get("completed")
        frac = (done / total) if total and done is not None else None
        if status != last or frac is not None:
            say(status, frac)
            last = status


def _hf_files(repo: str) -> list[str]:
    r = httpx.get(f"https://huggingface.co/api/models/{repo}", timeout=30)
    r.raise_for_status()
    names = [s["rfilename"] for s in r.json().get("siblings", [])]
    return [n for n in names if "/" not in n and n.endswith(_HF_WANTED)]


def _download(url: str, dest: Path, say: Progress, label: str) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        got = 0
        with tmp.open("wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
                got += len(chunk)
                say(f"downloading {label}", got / total if total else None)
    tmp.replace(dest)


def patch_hf_config(cfg: dict[str, Any]) -> list[str]:
    """Back-fill legacy keys Ollama's importer expects. Returns the list of fixes applied."""
    fixes = []
    rp = cfg.get("rope_parameters")
    if isinstance(rp, dict) and "rope_theta" in rp and "rope_theta" not in cfg:
        cfg["rope_theta"] = rp["rope_theta"]
        fixes.append("rope_theta")
    if "dtype" in cfg and "torch_dtype" not in cfg:
        cfg["torch_dtype"] = cfg["dtype"]
        fixes.append("torch_dtype")
    return fixes


def modelfile_text(model: Model, from_ref: str) -> str:
    mf = model.raw.get("modelfile", {})
    lines = [f"FROM {from_ref}"]
    tmpl = mf.get("template")
    if tmpl:
        lines.append(f'TEMPLATE """{TEMPLATES.get(tmpl, tmpl)}"""')
    if mf.get("system"):
        lines.append(f'SYSTEM """{mf["system"]}"""')
    for k, v in (mf.get("parameters") or {}).items():
        for val in v if isinstance(v, list) else [v]:
            lines.append(f"PARAMETER {k} {val}")
    return "\n".join(lines) + "\n"


def _install_safetensors(model: Model, say: Progress) -> None:
    if not shutil.which("ollama"):
        raise BackendError("the `ollama` CLI is needed to import safetensors models — install it from ollama.com")
    repo = model.source["repo"]
    rev = model.source.get("revision", "main")
    cache = config.home() / "cache" / "hf" / repo.replace("/", "__")
    cache.mkdir(parents=True, exist_ok=True)
    for name in _hf_files(repo):
        dest = cache / name
        if not dest.exists():
            _download(f"https://huggingface.co/{repo}/resolve/{rev}/{name}", dest, say, name)
    cfg_path = cache / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        fixes = patch_hf_config(cfg)
        if fixes:
            cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            say(f"patched config.json ({', '.join(fixes)})", None)
    (cache / "Modelfile").write_text(modelfile_text(model, "."), encoding="utf-8")
    say("creating Ollama model", None)
    proc = subprocess.run(
        ["ollama", "create", model.ollama_tag, "-f", "Modelfile"],
        cwd=cache, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise BackendError(f"ollama create failed: {proc.stderr.strip() or proc.stdout.strip()}")
    say("done", 1.0)
