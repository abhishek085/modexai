# Roadmap

ModexAI's bet: as agents do more of the world's routine work, the useful knowledge is *which small model
can take which job, on which machine* — and that knowledge should be open, measured, and shareable
without anyone's data or weights changing hands.

## Now (v1)

- [x] MCP server with the full loop: find → test → hire → delegate → share
- [x] Shadow mode: agents record their own answers; hire a local apprentice that agrees with them
- [x] 9 deterministic suites, including execution-graded text-to-SQL
- [x] Recipes: content-addressed, hardware-aware, reproducible; GitHub-native intake with no server
- [x] Hardware-aware search and model-size budgets
- [x] Export shadow examples as a fine-tuning dataset
- [x] Exchange website (GitHub Pages) and `modexai ui`

## Next

- [ ] Publish to PyPI and the official MCP Registry (`uvx modexai mcp`)
- [ ] Drift checks: periodic spot-checks of a hired model against the reference model, auto-unhire on drift
- [ ] Injection screening of contributed recipes with a local classifier (nokast-secureRAG) in the intake bot
- [ ] Recipes for multi-step pipelines ("redact with model A, then classify with model B")
- [ ] More runtimes measured side by side: llama.cpp, MLX, LM Studio, vLLM
- [ ] Embedding recipes (retrieval quality on public sets)

## Later

- [ ] One-command fine-tune from shadow examples (MLX on Apple Silicon, Unsloth on CUDA), then list + share
- [ ] Agent-to-agent recipe requests: "does anyone have a recipe for X on 8 GB?"
- [ ] Team exchanges: a private index for an organisation's internal fine-tunes, same tools

Ideas and votes welcome in [issues](https://github.com/abhishek085/modexai/issues).
