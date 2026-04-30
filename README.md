# ModexAI 🤖📈

**Agent-exclusive model marketplace** — AI agents autonomously discover, test, purchase, and deploy fine-tuned models (LoRAs/SLMs) like stocks on NYSE/NASDAQ, optimizing for cost & speed over big LLMs.Inentive for individuals to train more models and sell it on this marketplace. In future the training and selling can be done directly through Agents as well, who will name their price based on demand of the model and value it brings.

---

## Overview

ModexAI exposes a lightweight HTTP exchange that AI agents integrate with as a *tool*:

| Step | Action | Endpoint |
|------|--------|----------|
| 1 | **Discover** models by niche | `GET /models?niche=finance` |
| 2 | **Evaluate** top-3 with task samples | `POST /eval` |
| 3 | **Purchase** the winner | `GET /buy/{id}?token=mock` |

Agents run a full autonomy loop — no human in the loop required.

---
![animated representation](modexai.gif)

## Repo Structure

```
modexai/
├── README.md                 # This file
├── docker-compose.yml        # Local stack (API + Ollama)
├── .env.example              # Required environment variables
├── .gitignore
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py                # FastAPI: /models, /eval, /buy
│   └── models/               # Seed fine-tunes (GGUF metadata)
│       ├── finance-lora/
│       │   └── metadata.json
│       ├── claims-lora/
│       │   └── metadata.json
│       └── devops-lora/
│           └── metadata.json
├── agent-demo/
│   ├── demo_agent.py         # LangChain agent with modexai_tool
│   └── requirements.txt
└── docs/
    └── agent-tool.md         # OpenAI-compatible tool schema & agent protocol
```

---

## Quick Start (Local Demo)

### Prerequisites

- Docker Desktop ≥ 24
- Python ≥ 3.12 (for the agent demo)
- [Ollama](https://ollama.com) (pulled automatically by Compose)

### 1 — Clone & configure

```bash
git clone https://github.com/abhishek085/modexai
cd modexai
cp .env.example .env          # edit if needed
```

### 2 — Start the stack

```bash
docker compose up --build
```

The API is available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

### 3 — Run the agent demo

```bash
cd agent-demo
pip install -r requirements.txt
python demo_agent.py
```

The agent will autonomously search for a finance model, evaluate it, and purchase the winner.

---

## API Reference

### `GET /models`

List models filtered by niche.

| Query param | Type | Example |
|-------------|------|---------|
| `niche` | string | `finance`, `claims`, `devops` |

**Response**
```json
[
  {
    "id": "finance-lora-v1",
    "name": "Finance LoRA v1",
    "niche": "finance",
    "base_model": "phi-3-mini",
    "description": "Fine-tuned on financial analysis datasets",
    "price_usd": 4.99,
    "benchmarks": {"accuracy": 0.91, "latency_ms": 120}
  }
]
```

### `POST /eval`

Run evaluation samples against top-3 models for a given niche.

**Request body**
```json
{
  "niche": "finance",
  "samples": ["What is the P/E ratio of AAPL?", "Summarise Q3 earnings."]
}
```

**Response**
```json
{
  "results": [
    {"model_id": "finance-lora-v1", "score": 0.91, "avg_latency_ms": 130},
    {"model_id": "finance-lora-v2", "score": 0.87, "avg_latency_ms": 95}
  ]
}
```

### `GET /buy/{id}`

Purchase a model (mock Stripe flow).

| Query param | Type | Description |
|-------------|------|-------------|
| `token` | string | `mock` for local demo |

**Response**
```json
{
  "status": "paid",
  "model_id": "finance-lora-v1",
  "download_token": "dl_abc123",
  "model_path": "/models/finance-lora"
}
```

---

## Agent Tool Spec

See [`docs/agent-tool.md`](docs/agent-tool.md) for the full OpenAI-compatible tool schema used by agents to integrate with ModexAI.

---

## Deployment

| Target | Notes |
|--------|-------|
| **Local** | `docker compose up` — fully offline, M4 Pro optimized |
| **Railway** | Push to `main`; set env vars in Railway dashboard |
| **HF Hub** | Upload GGUF model files; update `metadata.json` with Hub URL |

---

## Environment Variables

See [`.env.example`](.env.example) for the full list.

---

## License

MIT

