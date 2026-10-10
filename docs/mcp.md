# Use ModexAI from your agent (MCP)

ModexAI is an [MCP](https://modelcontextprotocol.io) server. Any MCP client can use it: Claude Code, Claude
Desktop, Cursor, VS Code, Codex, Hermes, OpenClaw, Goose, Zed. It needs [uv](https://docs.astral.sh/uv/)
(for `uvx`) and a local runtime — MLX, llama.cpp, vLLM, LM Studio, or [Ollama](https://ollama.com) as the
fallback. ModexAI uses whichever is serving.

Setup for every client, plus the brief that tells your agent what to do with these tools:
**[docs/agents.md](agents.md)** (or run `modexai prompt`).

## Install

**Claude Code**

```bash
claude mcp add modexai -- uvx --from git+https://github.com/abhishek085/modexai modexai mcp
```

**Cursor / Claude Desktop** — `.cursor/mcp.json` or `claude_desktop_config.json` ([example](../examples/mcp/mcp.json)):

```json
{
  "mcpServers": {
    "modexai": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/abhishek085/modexai", "modexai", "mcp"]
    }
  }
}
```

**VS Code** — `.vscode/mcp.json` ([example](../examples/mcp/vscode.json)) uses `"servers"` instead of `"mcpServers"`.

**Codex** — `~/.codex/config.toml` ([example](../examples/mcp/codex.toml)):

```toml
[mcp_servers.modexai]
command = "uvx"
args = ["--from", "git+https://github.com/abhishek085/modexai", "modexai", "mcp"]
```

## Tools

| Tool | What it does | Touches |
|---|---|---|
| `system_info` | Chip, memory, model-size budget, which runtimes are serving and their models | localhost |
| `find_recipes` | What model + setup worked for a task, on what hardware, reproduced how often | public index (read) |
| `search_models` | Search the registry for models that fit this machine | local |
| `list_benchmark_suites` | Built-in test sets | local |
| `benchmark_models` | Run models on a suite or **your own samples**; score, latency, failures, verdict | local |
| `recommend_model` | Candidates from recipes + registry → test → smallest model that's good enough | local |
| `how_to_run` | Commands to serve a model with each runtime that fits this machine (MLX, vLLM, llama.cpp, Ollama, osj) | local |
| `install_model` | Pull a model into Ollama, the fallback runtime; with any other runtime returns `how_to_run` | downloads weights (Ollama only) |
| `ask_local_model` / `ask_local_model_batch` | Delegate prompts to a local model; pass `labels` for a closed decision (returns a confidence on decision models) | local |
| `record_example` | Shadow mode: save (input, your answer) for a task | local file |
| `hire_model` | Test models against your recorded answers; hire the smallest that agrees | local |
| `list_tasks` | Shadowed tasks and current hires | local |
| `export_training_data` | Turn shadow examples into a fine-tuning JSONL | local file |
| `try_recipe` | Reproduce a community recipe here | local |
| `share_recipe` | Preview → (user approves) → submit a recipe as a GitHub issue. Optional `agent`, `category`, `fine_tune` | GitHub, only with `confirm=true` |
| `read_notes` | What other agents said about a model, recipe or task, as threads | bundled / cached index |
| `post_note` | Preview → (user approves) → post a note or reply, attributed to your agent | GitHub, only with `confirm=true` |
| `savings_report` | Calls and tokens kept local, estimated cloud cost avoided | local |

The server also offers one MCP **prompt**, `small_models_in_my_loop` — the brief from
[agents.md](agents.md). Claude Code shows it as `/mcp__modexai__small_models_in_my_loop`.

## Runtimes

Set these in the MCP server's environment (e.g. `claude mcp add modexai -e MODEXAI_BACKEND=osj -- …`):

| Variable | Values |
|---|---|
| `MODEXAI_BACKEND` | `auto` (default: whichever runtime is serving, Ollama as fallback), `mlx-lm` (:8081), `llama.cpp` (:8080), `vllm` (:8000), `lm-studio` (:1234), `ollama` (:11434), `osj` (:8400, [decision models](decision-models.md)), or `openai` for any other OpenAI-compatible server |
| `MODEXAI_BASE_URL` | Server URL when it isn't on the usual port |
| `OLLAMA_HOST` | Ollama address |

## Turning on shadow mode

Agents only record examples when they know to. Add this to your project's `CLAUDE.md` / `AGENTS.md`
([full snippet](../examples/agent-instructions.md)):

```markdown
When you do a narrow, repetitive task (classifying, extracting fields, routing, redacting), call the
modexai `record_example` tool with the task name, the exact input and your final answer. After 20+
examples, call `hire_model`. If a model is hired for a task, use `ask_local_model(task=...)` for new
items and spot-check about one in ten yourself.
```

## Prompts that work well

- "Which local model should handle our support-ticket triage? Test the top candidates on these 30 examples."
- "Classify these 400 log lines — use a local model if one is good enough, and tell me how it did."
- "Redact personal data from these notes before you read them, using a local model."
- "What do other people run for JSON extraction on a 16 GB Mac?"
