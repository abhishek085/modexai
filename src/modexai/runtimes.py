"""Local runtimes: which ones are running here, and how to run a listed model on each.

ModexAI doesn't own a runtime. The agent starts whichever fits the machine — MLX on Apple Silicon, vLLM on
NVIDIA, llama.cpp anywhere, Ollama as the easy fallback, open-spark-jev for decision models — and ModexAI
benchmarks whatever is serving. Detection only talks to localhost.
"""

from __future__ import annotations

from typing import Any

import httpx

from .registry import Model

# name → (port, how to recognise it, backend that speaks to it)
KNOWN = {
    "ollama": {"port": 11434, "probe": "/api/version", "backend": "ollama"},
    "llama.cpp": {"port": 8080, "probe": "/props", "backend": "openai"},
    "mlx-lm": {"port": 8081, "probe": "/v1/models", "backend": "openai"},
    "vllm": {"port": 8000, "probe": "/v1/models", "backend": "openai"},
    "lm-studio": {"port": 1234, "probe": "/v1/models", "backend": "openai"},
    "osj": {"port": 8400, "probe": "/healthz", "backend": "osj"},
}
PLATFORMS = {"apple-silicon": "Apple Silicon", "cuda": "NVIDIA GPU", "cpu": "any machine"}


def base_url(runtime: str) -> str:
    port = KNOWN[runtime]["port"]
    return f"http://127.0.0.1:{port}" + ("" if runtime == "ollama" else "/v1")


def detect(timeout: float = 0.4) -> list[dict[str, Any]]:
    """Runtimes answering on their usual localhost ports, with the models each is serving."""
    found = []
    with httpx.Client(timeout=timeout) as c:
        for name, k in KNOWN.items():
            root = f"http://127.0.0.1:{k['port']}"
            try:
                if c.get(root + k["probe"]).status_code != 200:
                    continue
                if name == "ollama":
                    models = [m["name"] for m in c.get(root + "/api/tags").json().get("models", [])]
                else:
                    models = [m["id"] for m in c.get(root + "/v1/models").json().get("data", [])]
            except (httpx.HTTPError, ValueError, KeyError):
                continue
            found.append({"runtime": name, "backend": k["backend"], "base_url": base_url(name), "models": models})
    return found


# Which weights each engine loads, in order of preference.
ENGINE_FORMATS = {"vllm": ("nvfp4", "safetensors"), "mlx-lm": ("mlx",), "llama.cpp": ("gguf",),
                  "lm-studio": ("gguf", "mlx")}


def served_name(m: Model | None, ref: str, engine: str) -> str:
    """The name to ask ``engine`` for: an Ollama tag for Ollama, the right Hugging Face repo for the rest."""
    if m is None:
        return ref
    if engine not in ENGINE_FORMATS:
        return m.ollama_tag
    if ref != m.id and ref != m.ollama_tag:
        return ref  # already a served name (a repo, maybe with a quant) — keep it as given
    w = m.weights
    return next((w[f] for f in ENGINE_FORMATS[engine] if w.get(f)), ref)


def engine_at(url: str) -> str:
    """Name the runtime behind an OpenAI-compatible URL from its port (generic 'openai' if unknown)."""
    port = httpx.URL(url).port
    return next((n for n, k in KNOWN.items() if k["port"] == port and k["backend"] == "openai"), "openai")


def run_options(m: Model, accelerator: str | None = None) -> list[dict[str, Any]]:
    """Ways to serve this model, best first for ``accelerator`` (or every platform when it is None).

    Each option is a command the agent runs itself, and how to point ModexAI at the result.
    """
    w = m.weights
    gated = " Accept the licence on Hugging Face and run `hf auth login` first." if w.get("gated") else ""
    opts: list[dict[str, Any]] = []

    def add(runtime: str, platforms: list[str], command: str, model: str, note: str = "") -> None:
        opts.append({
            "runtime": runtime, "platforms": platforms, "command": command,
            "then": {"backend": KNOWN[runtime]["backend"], "base_url": base_url(runtime), "model": model},
            **({"note": note} if note else {}),
        })

    if m.kind == "decision":
        repo = m.source["repo"]
        gpu = "Needs an NVIDIA Blackwell GPU (e.g. DGX Spark)." if "nvfp4" in w else "Needs an NVIDIA GPU."
        add("osj", ["cuda"], f"osj serve --pull {repo}", m.ollama_tag,
            f"open-spark-jev: pip install it from https://github.com/abhishek085/open-spark-jev. {gpu}")
    else:
        _chat_options(m, w, gated, add)
    if accelerator:
        opts = [o for o in opts if accelerator in o["platforms"] or "cpu" in o["platforms"]]
        native = {"apple-silicon": "mlx-lm", "cuda": "vllm"}.get(accelerator)
        opts.sort(key=lambda o: (o["runtime"] != native, o["runtime"] == "ollama"))
    return opts


def _chat_options(m: Model, w: dict[str, Any], gated: str, add) -> None:
    embed = m.kind == "embedding"
    if w.get("mlx") and not embed:
        add("mlx-lm", ["apple-silicon"], f"mlx_lm.server --model {w['mlx']} --port 8081", w["mlx"],
            "pip install mlx-lm. Usually the fastest option on a Mac.")
    if (w.get("safetensors") or w.get("nvfp4")) and not embed:
        repo = w.get("nvfp4") or w["safetensors"]
        add("vllm", ["cuda"], f"vllm serve {repo} --port 8000", repo,
            ("NVFP4 weights need an NVIDIA Blackwell GPU." if "nvfp4" in w else "pip install vllm.") + gated)
    if w.get("gguf"):
        add("llama.cpp", ["apple-silicon", "cuda", "cpu"],
            f"llama-server -hf {w['gguf']} --port 8080" + (" --embedding" if embed else " --jinja"), w["gguf"],
            "Runs anywhere; picks a sensible quantisation by default.")
    src = m.source.get("type")
    if src in ("ollama", "hf-gguf"):
        add("ollama", ["apple-silicon", "cuda", "cpu"], f"ollama pull {m.ollama_tag}", m.ollama_tag,
            "The easy fallback. Runs anywhere.")
    elif src == "hf-safetensors":
        add("ollama", ["apple-silicon", "cuda", "cpu"], f"modexai install {m.id}", m.ollama_tag,
            "Downloads the safetensors and imports them into Ollama.")
