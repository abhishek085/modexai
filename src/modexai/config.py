"""Paths, endpoints and defaults. Everything is overridable with environment variables."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).parent
DATA_DIR = PACKAGE_DIR / "data"
SUITES_DIR = PACKAGE_DIR / "suites"
WEB_DIR = PACKAGE_DIR / "web"
BUNDLED_REGISTRY = DATA_DIR / "registry.json"

REPO_URL = "https://github.com/abhishek085/modexai"
SITE_URL = "https://abhishek085.github.io/modexai/"
REMOTE_REGISTRY_URL = "https://raw.githubusercontent.com/abhishek085/modexai/main/src/modexai/data/registry.json"


def home() -> Path:
    """~/.modexai — local state (cached registry, private listings, receipts, ledger). Never uploaded."""
    p = Path(os.environ.get("MODEXAI_HOME", Path.home() / ".modexai")).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def ollama_host() -> str:
    """Normalise OLLAMA_HOST the same way the Ollama CLI does (scheme and port optional)."""
    raw = os.environ.get("OLLAMA_HOST", "").strip() or "http://127.0.0.1:11434"
    if "://" not in raw:
        raw = "http://" + raw
    scheme, rest = raw.split("://", 1)
    rest = rest.rstrip("/")
    host = rest.split("/", 1)[0]
    if host.startswith("0.0.0.0"):
        rest = rest.replace("0.0.0.0", "127.0.0.1", 1)
    if ":" not in host.split("]")[-1]:
        rest = rest.replace(host, host + ":11434", 1)
    return f"{scheme}://{rest}"


def backend_name() -> str:
    """Default inference backend (MODEXAI_BACKEND). 'auto' uses whichever local runtime is serving — MLX,
    llama.cpp, vLLM, LM Studio — and falls back to Ollama. The MCP server uses it too."""
    return os.environ.get("MODEXAI_BACKEND", "").strip() or "auto"


def backend_url() -> str | None:
    """Server URL for the openai and osj backends (MODEXAI_BASE_URL); each has its own default."""
    return os.environ.get("MODEXAI_BASE_URL", "").strip() or None


def baseline_price() -> tuple[float, float]:
    """Cloud baseline used for "what this would have cost" estimates, in USD per 1M tokens (input, output).

    It is an assumption, not a quote — set MODEXAI_BASELINE="in,out" to match the model you would
    otherwise call.
    """
    raw = os.environ.get("MODEXAI_BASELINE", "3,15")
    try:
        a, b = (float(x) for x in raw.split(","))
        return a, b
    except ValueError:
        return 3.0, 15.0


def num_ctx() -> int:
    """Context window requested from Ollama (MODEXAI_NUM_CTX, default 4096)."""
    try:
        return int(os.environ.get("MODEXAI_NUM_CTX", "4096"))
    except ValueError:
        return 4096


def ledger_enabled() -> bool:
    return os.environ.get("MODEXAI_NO_LEDGER", "") not in ("1", "true", "yes")
