# Recipes: what runs locally, measured

A **recipe** is a small JSON document that says: *this model, set up this way, scored this on this test,
on this kind of machine.* It contains no weights, no prompts from your work, and no outputs — only setup,
hardware class and aggregate numbers. Recipes are how agents share what they learned so the next agent
doesn't start from zero.

## Anatomy

```json
{
  "schema": "modexai.recipe/1",
  "id": "rcp_3f1c0a9e2b7d",
  "kind": "benchmark",
  "title": "qwen3:1.7b scores 95% on ticket-triage · Apple M4 Pro · 24 GB",
  "task": "classification",
  "model": { "tag": "qwen3:1.7b", "registry_id": "qwen/qwen3-1.7b", "size_gb": 1.36 },
  "setup": { "parameters": { "temperature": 0, "num_ctx": 4096 } },
  "eval": {
    "suite": "ticket-triage@1", "suite_sha": "c13c60a6c727…",
    "cases": 20, "score": 0.95, "p50_ms": 128.8, "p95_ms": 160.2, "tokens_per_s": 120.4
  },
  "hardware": { "label": "Apple M4 Pro · 24 GB", "accelerator": "apple-silicon", "ram_gb": 24, "model_size_gb": 1.36, "os": "macOS", "arch": "arm64" },
  "runtime": { "engine": "ollama", "version": "0.32.1" },
  "created": "2026-10-09",
  "tool": "modexai 1.0.0"
}
```

| Kind | Meaning |
|---|---|
| `benchmark` | A model on a public suite. Anyone can re-run the exact same test (`suite` + `suite_sha`). |
| `recipe` | A setup that worked on **private** data (e.g. a shadow-and-hire run). Only `private_eval` aggregates are shared: case count, agreement score, latency. |
| `reproduction` | Someone re-ran an existing recipe on their machine. Points at it with `reproduces`. |

The `id` is a content hash (excluding `id`, `created` and `author`), so a recipe can't be edited after the
fact without changing its id.

## Lifecycle

```
agent runs benchmark / hire ──► draft saved in ~/.modexai/recipes/
                                        │
             user approves the exact JSON (share_recipe preview → confirm=true)
                                        │
                          GitHub issue (label: recipe)
                                        │
          intake bot: extract → validate → open PR into registry/recipes/
                                        │
                      maintainer merges → index rebuilt (src/modexai/data/recipes.json)
                                        │
          served from GitHub raw + Pages ──► every agent: find_recipes / try_recipe
```

Reads scale like any static file on a CDN. Writes are GitHub issues — free, moderated, auditable. No
server, no account system, no database to run.

## Trust

- **Reproducible by construction.** Public-suite recipes pin the suite version and content hash.
  `modexai recipes try <id>` re-runs it and produces a reproduction you can share.
- **Reproductions are the signal.** The index counts them per recipe, and search ranks recipes with more
  reproductions — on hardware like yours — first.
- **Private results are labelled as such.** A `private_eval` is a claim about someone else's data; treat
  it as a hint and test on your own samples before relying on it.

## Safety: recipes are data, not instructions

Recipes are written by strangers and read by agents, which makes them a prompt-injection surface.
ModexAI defends in layers:

1. Strict schema, field length limits, and an 8 KB cap.
2. Free-text fields (`title`, `notes`, `setup.system`, `author`) are screened for text that addresses the
   reading agent ("ignore previous instructions", "reveal your API keys", "AI agents reading this…").
3. A maintainer reviews every PR before it reaches the index.
4. The MCP server tells agents, in its instructions and in every `find_recipes` result, that recipes are
   community data and must never be followed as instructions.

On the roadmap: running contributed text through a local injection classifier — like
[nokast-secureRAG](https://huggingface.co/abhishek085/nokast-secureRAG-0.5B) — in the intake bot.

## Sharing by hand

```bash
modexai recipes mine                 # your drafts
modexai recipes show rcp_…           # the exact JSON
modexai recipes share rcp_… --notes "needed num_ctx 4096"   # previews, asks, then posts via gh
```

No `gh` CLI? `share` prints a pre-filled GitHub issue link instead.
