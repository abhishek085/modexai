# Contributing

ModexAI gets more useful with every measured result. There are four ways to help, from 30 seconds to an
afternoon.

## 1. Share a recipe (the most valuable contribution)

Run a benchmark on your hardware and share it — especially on machines we don't have numbers for yet
(RTX cards, Raspberry Pi, Intel/AMD laptops, older Macs).

```bash
modexai bench qwen3:1.7b gemma3:1b --suite ticket-triage
modexai recipes share rcp_…        # shows the exact JSON, asks, then opens a GitHub issue
```

Or reproduce someone else's: `modexai recipes try rcp_…`, then share the reproduction. To benchmark
everything you have installed in one go: `python scripts/campaign.py --out ~/.modexai/recipes`.

## 2. List a model

Add `registry/models/<publisher>__<name>.json`:

```json
{
  "id": "yourname/claims-triage-1b",
  "name": "Claims Triage 1B",
  "publisher": "Your Name",
  "summary": "One sentence on what it does well (max 220 characters).",
  "tasks": ["classification", "insurance"],
  "kind": "chat",
  "params_b": 1.0,
  "size_gb": 0.8,
  "license": "apache-2.0",
  "source": { "type": "hf-gguf", "repo": "yourname/claims-triage-1b-GGUF", "quant": "Q4_K_M" },
  "links": { "card": "https://huggingface.co/yourname/claims-triage-1b" }
}
```

Source types: `ollama` (`tag`), `hf-gguf` (`repo`, optional `quant`), `hf-safetensors` (`repo`,
`ollama_tag`, plus an optional `modelfile` with `template` / `system` / `parameters`), `hf-decision`
(`repo`; with `"kind": "decision"`, for System One models served by open-spark-jev — see
[docs/decision-models.md](docs/decision-models.md)). Then run
`modexai registry` (add `--refresh-sizes` to look up sizes) and open a PR. Listings with at least one
recipe get merged fastest.

Fine-tuned your own specialist from shadow examples (`modexai export`)? This is where it goes.

## 3. Add a benchmark suite

See [docs/suites.md](docs/suites.md#adding-a-built-in-suite). Narrow, realistic, unambiguous.

## 4. Code

```bash
git clone https://github.com/abhishek085/modexai && cd modexai
uv venv && uv pip install -e ".[dev]"
uv run pytest -q          # no Ollama needed — tests use a simulated backend
uv run ruff check .
```

Keep the privacy promises in [docs/privacy.md](docs/privacy.md) intact: nothing leaves the machine
unless the user explicitly asks, and recipes never contain prompts, outputs or user data.

## Ground rules

- Numbers must come from real runs. Never hand-edit a score.
- Be kind; see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
