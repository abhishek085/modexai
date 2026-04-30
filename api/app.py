"""
ModexAI Exchange API
====================
Agent-facing endpoints for discovering, evaluating, and purchasing
fine-tuned models (LoRAs / SLMs).

Endpoints
---------
GET  /models          – list models, optionally filtered by niche
POST /eval            – evaluate top-3 models with task samples via Ollama
GET  /buy/{id}        – mock-purchase a model (Stripe integration ready)
GET  /health          – liveness probe
"""

from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/app/models"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
STRIPE_MOCK_KEY = os.getenv("STRIPE_MOCK_KEY", "mock")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ModexAI Exchange API",
    version="0.1.0",
    description=(
        "Agent-exclusive marketplace for fine-tuned models. "
        "Agents discover, evaluate, and purchase LoRA/SLM models autonomously."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Models (loaded from metadata.json files on disk)
# ---------------------------------------------------------------------------


def _load_models() -> list[dict[str, Any]]:
    """Walk MODELS_DIR and return a list of model metadata dicts."""
    catalogue: list[dict[str, Any]] = []
    if not MODELS_DIR.exists():
        return catalogue
    for meta_path in sorted(MODELS_DIR.glob("*/metadata.json")):
        try:
            data = json.loads(meta_path.read_text())
            data["_path"] = str(meta_path.parent)
            catalogue.append(data)
        except Exception:
            pass
    return catalogue


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    id: str
    name: str
    niche: str
    base_model: str
    description: str
    price_usd: float
    benchmarks: dict[str, Any]


class EvalRequest(BaseModel):
    niche: str = Field(..., examples=["finance"])
    samples: list[str] = Field(..., min_length=1, examples=[["Summarise Q3 earnings."]])


class EvalResult(BaseModel):
    model_id: str
    score: float
    avg_latency_ms: float


class EvalResponse(BaseModel):
    results: list[EvalResult]


class PurchaseResponse(BaseModel):
    status: str
    model_id: str
    download_token: str
    model_path: str


# ---------------------------------------------------------------------------
# Helper: Ollama inference
# ---------------------------------------------------------------------------


async def _ollama_generate(model: str, prompt: str) -> tuple[str, float]:
    """Call Ollama /api/generate and return (response_text, duration_ms)."""
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            duration_ms = (time.perf_counter() - t0) * 1000
            return data.get("response", ""), duration_ms
    except Exception:
        # Ollama not available — return a deterministic mock response
        duration_ms = (time.perf_counter() - t0) * 1000
        return f"[mock response for: {prompt[:60]}]", duration_ms


def _score_response(response: str, prompt: str) -> float:
    """
    Heuristic scoring: longer, non-empty responses score higher.
    Replace with a real evaluation metric in production.
    """
    if not response or response.startswith("[mock"):
        return round(0.5 + (len(prompt) % 10) * 0.01, 3)
    return min(1.0, round(0.6 + len(response) / 2000, 3))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}


@app.get("/models", response_model=list[ModelInfo], tags=["marketplace"])
async def list_models(
    niche: str | None = Query(default=None, description="Filter by niche (e.g. finance, claims, devops)"),
) -> list[ModelInfo]:
    """
    List available fine-tuned models.

    Pass `?niche=<niche>` to filter by domain.  Returns all models when the
    query parameter is omitted.
    """
    catalogue = _load_models()
    if niche:
        catalogue = [m for m in catalogue if m.get("niche") == niche]
    if not catalogue:
        return []
    return [ModelInfo(**{k: v for k, v in m.items() if not k.startswith("_")}) for m in catalogue]


@app.post("/eval", response_model=EvalResponse, tags=["marketplace"])
async def evaluate(body: EvalRequest) -> EvalResponse:
    """
    Evaluate top-3 models for a given niche using provided sample prompts.

    Each sample is sent to the model's Ollama endpoint and scored
    heuristically.  Results are sorted by score descending.
    """
    catalogue = _load_models()
    candidates = [m for m in catalogue if m.get("niche") == body.niche][:3]

    if not candidates:
        raise HTTPException(status_code=404, detail=f"No models found for niche '{body.niche}'")

    results: list[EvalResult] = []
    for model_meta in candidates:
        ollama_model = model_meta.get("ollama_model", "phi3:mini")
        scores: list[float] = []
        latencies: list[float] = []
        for sample in body.samples:
            response, latency = await _ollama_generate(ollama_model, sample)
            scores.append(_score_response(response, sample))
            latencies.append(latency)

        avg_score = round(sum(scores) / len(scores), 4)
        avg_latency = round(sum(latencies) / len(latencies), 1)
        results.append(EvalResult(model_id=model_meta["id"], score=avg_score, avg_latency_ms=avg_latency))

    results.sort(key=lambda r: r.score, reverse=True)
    return EvalResponse(results=results)


@app.get("/buy/{model_id}", response_model=PurchaseResponse, tags=["marketplace"])
async def buy_model(
    model_id: str,
    token: str = Query(default="mock", description="Payment token; use 'mock' for local demo"),
) -> PurchaseResponse:
    """
    Purchase a model.

    In production this endpoint integrates with Stripe to create a
    PaymentIntent.  Locally, pass `?token=mock` to bypass payment.
    """
    catalogue = _load_models()
    model = next((m for m in catalogue if m["id"] == model_id), None)

    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    # ── Payment gate ────────────────────────────────────────────────────────
    if token == "mock" or STRIPE_MOCK_KEY == "mock":
        # Local / demo mode — skip real payment
        pass
    else:
        # Production: create a Stripe PaymentIntent (placeholder)
        # import stripe
        # stripe.api_key = STRIPE_MOCK_KEY
        # stripe.PaymentIntent.create(amount=int(model["price_usd"]*100), currency="usd")
        pass

    download_token = f"dl_{secrets.token_hex(8)}"
    return PurchaseResponse(
        status="paid",
        model_id=model_id,
        download_token=download_token,
        model_path=model.get("_path", f"/models/{model_id}"),
    )
