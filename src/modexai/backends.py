"""Inference backends. Everything runs on your machine.

- ``ollama``: the Ollama HTTP API (default).
- ``openai``: any OpenAI-compatible local server — llama.cpp ``llama-server``, LM Studio, vLLM, MLX-LM.
- ``osj``: a local open-spark-jev gateway (``osj serve``) running a System One decision model. It answers
  closed questions — pick one of N labels — with probabilities, not generated text.
- ``simulated``: deterministic fake answers, used by ``modexai demo`` and the test-suite. Never used
  unless you ask for it, and every result it produces is labelled ``simulated``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from . import config


@dataclass
class Generation:
    text: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_s: float | None = None
    confidence: float | None = None  # decision models only: probability mass on the chosen label


class BackendError(RuntimeError):
    pass


class Backend(Protocol):
    name: str

    def generate(
        self, model: str, messages: list[dict[str, str]], *, max_tokens: int = 256, json_mode: bool = False,
        item: dict | None = None,
    ) -> Generation: ...

    def installed(self) -> set[str]: ...


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class OllamaBackend:
    name = "ollama"

    def __init__(self, host: str | None = None, timeout: float = 180.0):
        self.host = host or config.ollama_host()
        self.client = httpx.Client(base_url=self.host, timeout=timeout)
        self._no_think: set[str] = set()

    def available(self) -> bool:
        try:
            return self.client.get("/api/version", timeout=3).status_code == 200
        except httpx.HTTPError:
            return False

    def version(self) -> str | None:
        try:
            return self.client.get("/api/version", timeout=3).json().get("version")
        except (httpx.HTTPError, ValueError):
            return None

    def installed(self) -> set[str]:
        try:
            r = self.client.get("/api/tags", timeout=5)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise BackendError(f"Ollama is not reachable at {self.host} ({e}). Is `ollama serve` running?") from e
        return {m["name"] for m in r.json().get("models", [])}

    def show(self, model: str) -> dict[str, Any]:
        r = self.client.post("/api/show", json={"model": model})
        r.raise_for_status()
        return r.json()

    def generate(self, model, messages, *, max_tokens=256, json_mode=False, item=None) -> Generation:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            # Pin the context window: recent Ollama versions default to a model's full context (262k for
            # Qwen3), which can need 40+ GB and silently spill to CPU. Narrow tasks need a few thousand.
            "options": {"temperature": 0, "num_predict": max_tokens, "seed": 7, "num_ctx": config.num_ctx()},
            "keep_alive": "10m",
        }
        if json_mode:
            body["format"] = "json"
        # Reasoning models burn the token budget "thinking" on narrow tasks. Ask them not to; models
        # without a thinking mode reject the flag, so remember which ones and retry without it.
        if model not in self._no_think:
            body["think"] = False
        t0 = time.perf_counter()
        try:
            r = self.client.post("/api/chat", json=body)
            if r.status_code == 400 and "think" in r.text.lower() and "think" in body:
                self._no_think.add(model)
                body.pop("think")
                t0 = time.perf_counter()
                r = self.client.post("/api/chat", json=body)
            if r.status_code == 404:
                raise BackendError(f"model '{model}' is not installed — run `modexai install {model}`")
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise BackendError(f"Ollama request failed for {model}: {e}") from e
        latency = (time.perf_counter() - t0) * 1000
        d = r.json()
        text = d.get("message", {}).get("content", "")
        ec, ed = d.get("eval_count") or 0, d.get("eval_duration") or 0
        return Generation(
            text=text,
            latency_ms=latency,
            prompt_tokens=d.get("prompt_eval_count") or _approx_tokens(json.dumps(messages)),
            completion_tokens=ec or _approx_tokens(text),
            tokens_per_s=(ec / (ed / 1e9)) if ec and ed else None,
        )

    def pull(self, tag: str):
        """Stream pull progress events from Ollama."""
        with self.client.stream("POST", "/api/pull", json={"model": tag, "stream": True}, timeout=None) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    ev = json.loads(line)
                    if "error" in ev:
                        raise BackendError(ev["error"])
                    yield ev


class OpenAICompatBackend:
    name = "openai"

    def __init__(self, base_url: str = "http://127.0.0.1:8080/v1", api_key: str = "local", timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {api_key}"})

    def installed(self) -> set[str]:
        try:
            r = self.client.get(f"{self.base_url}/models", timeout=5)
            r.raise_for_status()
            return {m["id"] for m in r.json().get("data", [])}
        except httpx.HTTPError as e:
            raise BackendError(f"OpenAI-compatible server not reachable at {self.base_url}: {e}") from e

    def generate(self, model, messages, *, max_tokens=256, json_mode=False, item=None) -> Generation:
        body: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        try:
            r = self.client.post(f"{self.base_url}/chat/completions", json=body)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise BackendError(f"request failed for {model}: {e}") from e
        latency = (time.perf_counter() - t0) * 1000
        d = r.json()
        text = d["choices"][0]["message"].get("content") or ""
        usage = d.get("usage") or {}
        ct = usage.get("completion_tokens") or _approx_tokens(text)
        return Generation(
            text=text,
            latency_ms=latency,
            prompt_tokens=usage.get("prompt_tokens") or _approx_tokens(json.dumps(messages)),
            completion_tokens=ct,
            tokens_per_s=ct / (latency / 1000) if latency else None,
        )


class OsjBackend:
    """A System One decision model behind an open-spark-jev gateway (``osj serve``, POST /v1/decide).

    Decision models never write text: given a state and a closed question they return a probability per
    option. Each call becomes one ``choice`` question — the user message is the state, the system prompt
    is the question, and the options are the labels from the item's check (or ``labels`` passed to
    ``ask``). The chosen label comes back as the text, so the normal graders score it.
    """

    name = "osj"

    def __init__(self, base_url: str = "http://127.0.0.1:8400/v1", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def installed(self) -> set[str]:
        try:
            r = self.client.get(f"{self.base_url}/models", timeout=5)
            r.raise_for_status()
            return {m["id"] for m in r.json().get("data", [])}
        except httpx.HTTPError as e:
            raise BackendError(f"open-spark-jev gateway not reachable at {self.base_url} "
                               f"(start it with `osj serve --pull <model>`): {e}") from e

    def generate(self, model, messages, *, max_tokens=256, json_mode=False, item=None) -> Generation:
        labels = ((item or {}).get("check") or {}).get("labels")
        if not labels or len(labels) < 2:
            raise BackendError("decision models answer closed questions only: give two or more labels "
                               "(a label suite, or labels= when asking)")
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        state = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        body = {
            "state": {"content": state},
            "questions": [{"type": "choice", "prompt": system or "Which option fits?", "options": list(labels)}],
            "model": model,
        }
        t0 = time.perf_counter()
        try:
            r = self.client.post(f"{self.base_url}/decide", json=body)
            r.raise_for_status()
            ans = r.json()["answers"][0]
        except httpx.HTTPError as e:
            raise BackendError(f"decision request failed for {model}: {e}") from e
        except (KeyError, IndexError, ValueError) as e:
            raise BackendError(f"unexpected reply from the open-spark-jev gateway: {e}") from e
        latency = (time.perf_counter() - t0) * 1000
        return Generation(
            text=ans["selected"],
            latency_ms=latency,
            prompt_tokens=_approx_tokens(state + system),
            completion_tokens=1,
            confidence=ans.get("confidence"),
        )


class SimulatedBackend:
    """Deterministic stand-in for a real model. A model's 'skill' sets how often it answers correctly.

    The answer is derived from the case's check, so graders exercise the real code paths.
    """

    name = "simulated"

    def __init__(self, skills: dict[str, float] | None = None, speed: dict[str, float] | None = None):
        self.skills = skills or {}
        self.speed = speed or {}

    def installed(self) -> set[str]:
        return set(self.skills)

    def _roll(self, model: str, key: str) -> float:
        h = hashlib.sha256(f"{model}|{key}".encode()).digest()
        return int.from_bytes(h[:4], "big") / 2**32

    def generate(self, model, messages, *, max_tokens=256, json_mode=False, item=None) -> Generation:
        skill = self.skills.get(model, 0.6)
        key = (item or {}).get("id") or messages[-1]["content"]
        correct = self._roll(model, key) < skill
        text = _simulated_answer((item or {}).get("check"), correct)
        latency = self.speed.get(model, 150.0) * (0.8 + 0.4 * self._roll(model, key + "t"))
        ct = _approx_tokens(text)
        tps = 30000 / self.speed.get(model, 150.0)
        return Generation(text, latency, _approx_tokens(json.dumps(messages)), ct, round(tps, 1))


def _simulated_answer(check: dict | None, correct: bool) -> str:
    if not check:
        return "OK" if correct else "I am not sure."
    t, exp = check.get("type"), check.get("expected")
    if t == "label":
        if correct:
            return f"Label: {exp}"
        others = [lab for lab in check.get("labels", []) if lab not in (check.get("accept") or [exp])]
        return f"Label: {others[0]}" if others else "Label: unknown"
    if t == "json_fields":
        if correct:
            return json.dumps(exp)
        broken = dict(exp)
        if broken:
            broken[next(iter(broken))] = "???"
        return json.dumps(broken)
    if t == "number":
        return f"Answer: {exp if correct else float(exp) + 1}"
    if t == "tool_call":
        return json.dumps(exp if correct else {"name": "none", "arguments": {}})
    if t == "sql_exec":
        return check["gold"] if correct else "SELECT 1"
    if t == "contains":
        return " ".join(check.get("all", []) + check.get("any", [])[:1]) if correct else " ".join(check.get("none", []))
    if t in ("similar", "exact"):
        return exp if correct else "something else entirely"
    return "OK"


BACKENDS = ("ollama", "openai", "osj", "simulated")


def get_backend(name: str | None = None, **kw) -> Backend:
    name = name or config.backend_name()
    base_url = kw.get("base_url") or config.backend_url()
    if name == "ollama":
        return OllamaBackend(host=kw.get("host"))
    if name == "openai":
        return OpenAICompatBackend(base_url=base_url or "http://127.0.0.1:8080/v1")
    if name == "osj":
        return OsjBackend(base_url=base_url or "http://127.0.0.1:8400/v1")
    if name == "simulated":
        return SimulatedBackend(kw.get("skills"), kw.get("speed"))
    raise ValueError(f"unknown backend {name!r} ({' | '.join(BACKENDS)})")
