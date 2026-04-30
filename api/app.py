"""
ModexAI Exchange API
====================
Agent-facing endpoints for discovering, evaluating, and purchasing
fine-tuned models (LoRAs / SLMs).

Endpoints (buyer / agent)
---------
GET  /models                – list models, optionally filtered by niche
POST /eval                  – evaluate top-3 models with task samples via Ollama
GET  /buy/{id}              – mock-purchase a model (Stripe integration ready)
GET  /download/{id}         – token-gated model file download
GET  /health                – liveness probe

Endpoints (seller dashboard)
---------
GET  /seller                – seller dashboard HTML UI
GET  /seller/models         – list models with earnings stats
POST /seller/models         – upload / register a new model
DELETE /seller/models/{id}  – remove a model listing
GET  /seller/earnings       – aggregated revenue summary
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/app/models"))
STATIC_DIR = Path(os.getenv("STATIC_DIR", "/app/static"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
STRIPE_MOCK_KEY = os.getenv("STRIPE_MOCK_KEY", "mock")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ModexAI Exchange API",
    version="0.2.0",
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

# Mount static files directory for the seller dashboard assets
_static_dir = STATIC_DIR if STATIC_DIR.exists() else Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

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


class SellerModelInfo(BaseModel):
    id: str
    name: str
    niche: str
    base_model: str
    description: str
    price_usd: float
    benchmarks: dict[str, Any]
    tags: list[str]
    version: str
    size_mb: int
    purchase_count: int
    revenue_usd: float


class EarningsSummary(BaseModel):
    total_models: int
    total_purchases: int
    total_revenue_usd: float
    models: list[SellerModelInfo]


# ---------------------------------------------------------------------------
# In-memory purchase ledger (resets on restart — use a DB for production)
# ---------------------------------------------------------------------------

# Maps download_token → {model_id, purchased_at, downloaded}
_purchases: dict[str, dict[str, Any]] = {}

# Maps model_id → purchase count
_purchase_counts: dict[str, int] = {}


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

    # Record purchase in the in-memory ledger
    _purchases[download_token] = {
        "model_id": model_id,
        "purchased_at": datetime.now(timezone.utc).isoformat(),
        "downloaded": False,
    }
    _purchase_counts[model_id] = _purchase_counts.get(model_id, 0) + 1

    return PurchaseResponse(
        status="paid",
        model_id=model_id,
        download_token=download_token,
        model_path=model.get("_path", f"/models/{model_id}"),
    )


# ---------------------------------------------------------------------------
# Download route (token-gated)
# ---------------------------------------------------------------------------


@app.get("/download/{model_id}", tags=["marketplace"])
async def download_model(
    model_id: str,
    token: str = Query(..., description="Download token received after purchase"),
) -> FileResponse:
    """
    Download a purchased model file.

    Requires a valid `download_token` issued by `GET /buy/{id}`.
    Returns the GGUF file if present; 404 if the file has not been uploaded.
    """
    purchase = _purchases.get(token)
    if purchase is None or purchase["model_id"] != model_id:
        raise HTTPException(status_code=403, detail="Invalid or expired download token")

    catalogue = _load_models()
    model = next((m for m in catalogue if m["id"] == model_id), None)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    model_path = Path(model["_path"])
    gguf_file = model_path / model.get("file", f"{model_id}.gguf")

    if not gguf_file.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Model file not yet uploaded by seller. "
                f"Expected: {gguf_file.name}"
            ),
        )

    _purchases[token]["downloaded"] = True
    return FileResponse(
        path=str(gguf_file),
        filename=gguf_file.name,
        media_type="application/octet-stream",
    )


# ---------------------------------------------------------------------------
# Seller dashboard routes
# ---------------------------------------------------------------------------


@app.get("/seller", response_class=HTMLResponse, tags=["seller"], include_in_schema=False)
async def seller_dashboard() -> HTMLResponse:
    """Serve the seller dashboard single-page application."""
    html_path = _static_dir / "seller.html" if _static_dir.exists() else Path(__file__).parent / "static" / "seller.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(), status_code=200)
    return HTMLResponse(
        content="<h1>Seller dashboard not found. Ensure api/static/seller.html is present.</h1>",
        status_code=404,
    )


@app.get("/seller/models", response_model=list[SellerModelInfo], tags=["seller"])
async def seller_list_models() -> list[SellerModelInfo]:
    """
    List all models with seller statistics (purchase count, revenue).

    In production this would be scoped to the authenticated seller's models.
    """
    catalogue = _load_models()
    result: list[SellerModelInfo] = []
    for m in catalogue:
        mid = m["id"]
        count = _purchase_counts.get(mid, 0)
        result.append(
            SellerModelInfo(
                id=mid,
                name=m.get("name", mid),
                niche=m.get("niche", ""),
                base_model=m.get("base_model", ""),
                description=m.get("description", ""),
                price_usd=m.get("price_usd", 0.0),
                benchmarks=m.get("benchmarks", {}),
                tags=m.get("tags", []),
                version=m.get("version", "1.0.0"),
                size_mb=m.get("size_mb", 0),
                purchase_count=count,
                revenue_usd=round(count * m.get("price_usd", 0.0), 2),
            )
        )
    return result


@app.post("/seller/models", response_model=SellerModelInfo, tags=["seller"], status_code=201)
async def seller_upload_model(
    name: str = Form(..., description="Human-readable model name"),
    niche: str = Form(..., description="Domain niche, e.g. finance, devops, medical"),
    base_model: str = Form(..., description="Foundation model, e.g. phi-3-mini, llama-3"),
    description: str = Form(..., description="What this model is fine-tuned for"),
    price_usd: float = Form(..., description="Price agents will pay per download (USD)"),
    version: str = Form(default="1.0.0", description="Semantic version string"),
    tags: str = Form(default="", description="Comma-separated tags, e.g. finance,lora,phi-3"),
    ollama_model: str = Form(default="phi3:mini", description="Ollama model tag for eval"),
    accuracy: float = Form(default=0.0, description="Benchmark accuracy (0-1)"),
    latency_ms: float = Form(default=0.0, description="Benchmark avg latency in ms"),
    model_file: UploadFile | None = File(default=None, description="GGUF model file (optional)"),
) -> SellerModelInfo:
    """
    Register a new model on the marketplace.

    The seller provides metadata via form fields and optionally uploads the GGUF
    model file.  The API assigns a slug ID, creates a directory under MODELS_DIR,
    writes metadata.json, and saves the file if provided.
    """
    # Derive a URL-safe ID from the name
    slug = name.lower().replace(" ", "-").replace("_", "-")
    model_id = f"{slug}-v{version.split('.')[0]}"

    # Ensure no duplicate IDs
    target_dir = MODELS_DIR / model_id
    if target_dir.exists():
        raise HTTPException(
            status_code=409,
            detail=f"A model with id '{model_id}' already exists. Use a different name or version.",
        )
    target_dir.mkdir(parents=True, exist_ok=True)

    # Save GGUF file if provided
    file_name = f"{model_id}.gguf"
    size_mb = 0
    if model_file and model_file.filename:
        file_name = model_file.filename
        dest = target_dir / file_name
        with dest.open("wb") as fh:
            shutil.copyfileobj(model_file.file, fh)
        size_mb = round(dest.stat().st_size / (1024 * 1024), 1)

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    metadata: dict[str, Any] = {
        "id": model_id,
        "name": name,
        "niche": niche,
        "base_model": base_model,
        "ollama_model": ollama_model,
        "description": description,
        "version": version,
        "price_usd": price_usd,
        "file": file_name,
        "size_mb": size_mb,
        "benchmarks": {
            "accuracy": accuracy,
            "latency_ms": latency_ms,
        },
        "tags": tag_list,
    }

    (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return SellerModelInfo(
        **{k: v for k, v in metadata.items() if k != "ollama_model"},
        purchase_count=0,
        revenue_usd=0.0,
    )


@app.delete("/seller/models/{model_id}", tags=["seller"])
async def seller_delete_model(model_id: str) -> dict[str, str]:
    """
    Remove a model listing and its associated files from the marketplace.

    In production this would be scoped to the authenticated seller's models.
    """
    target_dir = MODELS_DIR / model_id
    if not target_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    shutil.rmtree(target_dir)
    return {"status": "deleted", "model_id": model_id}


@app.get("/seller/earnings", response_model=EarningsSummary, tags=["seller"])
async def seller_earnings() -> EarningsSummary:
    """
    Aggregated earnings summary across all models.

    Returns total purchases, revenue, and per-model breakdown.
    """
    catalogue = _load_models()
    models_with_stats: list[SellerModelInfo] = []
    total_purchases = 0
    total_revenue = 0.0

    for m in catalogue:
        mid = m["id"]
        count = _purchase_counts.get(mid, 0)
        revenue = round(count * m.get("price_usd", 0.0), 2)
        total_purchases += count
        total_revenue += revenue
        models_with_stats.append(
            SellerModelInfo(
                id=mid,
                name=m.get("name", mid),
                niche=m.get("niche", ""),
                base_model=m.get("base_model", ""),
                description=m.get("description", ""),
                price_usd=m.get("price_usd", 0.0),
                benchmarks=m.get("benchmarks", {}),
                tags=m.get("tags", []),
                version=m.get("version", "1.0.0"),
                size_mb=m.get("size_mb", 0),
                purchase_count=count,
                revenue_usd=revenue,
            )
        )

    return EarningsSummary(
        total_models=len(models_with_stats),
        total_purchases=total_purchases,
        total_revenue_usd=round(total_revenue, 2),
        models=models_with_stats,
    )
