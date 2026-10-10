# Privacy model

ModexAI is a [Nokast](https://nokast.substack.com) tool: local and owned by default, cloud when it earns
its place. Here is exactly what it stores and what touches the network.

## Stays on your machine (`~/.modexai/`, override with `MODEXAI_HOME`)

| File | Contents |
|---|---|
| `tasks/<task>.jsonl` | Shadow examples: the inputs your agent handled and its answers. **The only place your work content is stored.** Delete with `forget` or by removing the file. |
| `hires.json` | Which local model is hired for which task, and its agreement score. |
| `recipes/rcp_*.json` | Your recipe drafts (aggregates only). |
| `ledger.jsonl` | One line per delegated call: timestamp, model, task name, token counts, latency. **No prompt text.** Disable with `MODEXAI_NO_LEDGER=1`. |
| `listings/*.json` | Optional private registry entries (e.g. your team's internal fine-tunes). |
| `registry.json`, `recipes-index.json` | Cached copies of the public index (after `modexai update`). |

Benchmarks on your own samples run entirely locally; custom samples are never written to disk by
ModexAI at all.

## Network calls

| When | What | Where |
|---|---|---|
| `modexai update` | GET the public registry + recipe index | raw.githubusercontent.com |
| `install_model` | Download weights | registry.ollama.ai / huggingface.co |
| `share_recipe` with `confirm=true` | Create a GitHub issue with the recipe JSON | github.com (your `gh` login) |
| Everything else | Talks to your local Ollama (or OpenAI-compatible server) | localhost |

No telemetry. No analytics. No accounts.

## What a shared recipe reveals

Model tag and settings, the public suite (or "private" + case count), aggregate scores and latency, and a
coarse hardware class: chip/GPU model, RAM size, OS family, architecture. Never hostnames, usernames,
serial numbers, IP addresses, prompts or outputs. You see the exact JSON before anything is posted.
