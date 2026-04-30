# ModexAI Agent Tool Specification

This document describes the **OpenAI-compatible tool schema** for integrating with the ModexAI
Exchange API, along with the recommended agent protocol.

---

## Overview

ModexAI is an **agent-exclusive model marketplace**.  AI agents use three tools to complete a
full autonomy loop:

```
search_models → evaluate_models → buy_model
```

Each tool maps 1-to-1 to a ModexAI API endpoint and follows the
[OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling) schema.

---

## Tool Schemas (OpenAI JSON)

### `search_models`

Search for available fine-tuned models by niche.

```json
{
  "type": "function",
  "function": {
    "name": "search_models",
    "description": "Search the ModexAI marketplace for fine-tuned models by niche. Returns a list of available models with IDs, names, prices, and benchmark scores.",
    "parameters": {
      "type": "object",
      "properties": {
        "niche": {
          "type": "string",
          "description": "The model niche to search for, e.g. 'finance', 'claims', 'devops'"
        }
      },
      "required": ["niche"]
    }
  }
}
```

**Maps to:** `GET /models?niche={niche}`

---

### `evaluate_models`

Evaluate the top-3 models for a niche using task sample prompts.

```json
{
  "type": "function",
  "function": {
    "name": "evaluate_models",
    "description": "Evaluate the top-3 models for a given niche by running sample prompts through them. Returns ranked results with scores and latency. Use this before buying.",
    "parameters": {
      "type": "object",
      "properties": {
        "niche": {
          "type": "string",
          "description": "The model niche to evaluate"
        },
        "samples": {
          "type": "array",
          "items": { "type": "string" },
          "description": "A list of representative task prompts to score models against",
          "minItems": 1
        }
      },
      "required": ["niche", "samples"]
    }
  }
}
```

**Maps to:** `POST /eval`

---

### `buy_model`

Purchase a model from the ModexAI marketplace.

```json
{
  "type": "function",
  "function": {
    "name": "buy_model",
    "description": "Purchase a model from ModexAI. Pass the model_id of the model you want to buy. Use token='mock' for the local demo. Returns a download token and model path.",
    "parameters": {
      "type": "object",
      "properties": {
        "model_id": {
          "type": "string",
          "description": "The ID of the model to purchase (e.g. 'finance-lora-v1')"
        },
        "token": {
          "type": "string",
          "description": "Payment token. Use 'mock' for the local demo.",
          "default": "mock"
        }
      },
      "required": ["model_id"]
    }
  }
}
```

**Maps to:** `GET /buy/{model_id}?token={token}`

---

## Agent Protocol

The recommended agent protocol is a **sequential autonomy loop**:

```
┌─────────────────────────────────────────────┐
│              AGENT AUTONOMY LOOP             │
│                                              │
│  1. search_models(niche)                     │
│     └─ returns: list of ModelInfo objects    │
│                                              │
│  2. evaluate_models(niche, samples)          │
│     └─ returns: ranked EvalResult list       │
│                                              │
│  3. buy_model(model_id, token)               │
│     └─ returns: PurchaseResponse             │
│        (status, download_token, model_path)  │
└─────────────────────────────────────────────┘
```

### Rules

1. **Always complete the full loop.** Do not stop after search or eval.
2. **Select the highest-scoring model** from `evaluate_models` results.
3. **Use `token=mock`** in local / demo environments.
4. **One niche per loop.** Run multiple loops for multi-niche optimization.

---

## Data Types

### `ModelInfo`

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique model identifier |
| `name` | string | Human-readable name |
| `niche` | string | Domain niche |
| `base_model` | string | Foundation model (e.g. `phi-3-mini`) |
| `description` | string | Model description |
| `price_usd` | float | Price in USD |
| `benchmarks` | object | Key performance metrics |

### `EvalResult`

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | string | Model ID |
| `score` | float | Aggregate score (0–1) |
| `avg_latency_ms` | float | Average inference latency in ms |

### `PurchaseResponse`

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"paid"` on success |
| `model_id` | string | Purchased model ID |
| `download_token` | string | Token for secure model download |
| `model_path` | string | Local or remote path to model files |

---

## Integration Examples

### OpenAI Python SDK

```python
import openai, requests, json

client = openai.OpenAI()
MODEXAI_URL = "http://localhost:8000"

TOOLS = [
    # paste the three JSON schemas above here
]

messages = [{"role": "user", "content": "Optimize finance analysis using ModexAI."}]

while True:
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, tools=TOOLS, tool_choice="auto"
    )
    msg = response.choices[0].message
    if not msg.tool_calls:
        print(msg.content)
        break
    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments)
        if tc.function.name == "search_models":
            result = requests.get(f"{MODEXAI_URL}/models", params=args).text
        elif tc.function.name == "evaluate_models":
            result = requests.post(f"{MODEXAI_URL}/eval", json=args).text
        elif tc.function.name == "buy_model":
            model_id = args.pop("model_id")
            result = requests.get(f"{MODEXAI_URL}/buy/{model_id}", params=args).text
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
```

### LangChain (see `agent-demo/demo_agent.py`)

The `agent-demo/demo_agent.py` script provides a ready-to-run LangChain implementation using
`StructuredTool` and `create_tool_calling_agent`.

---

## MCP Compatibility

ModexAI tools follow the [Model Context Protocol](https://modelcontextprotocol.io) naming
conventions.  Each tool has:

- A unique `name` (snake_case)
- A `description` suitable for LLM tool selection
- A JSON Schema `parameters` block
- Deterministic, JSON-serialisable return values

This makes ModexAI tools drop-in compatible with any MCP-capable agent framework.
