# Use ModexAI from your agent (MCP)

ModexAI is an [MCP](https://modelcontextprotocol.io) server. Any MCP client can use it: Claude Code, Claude
Desktop, Cursor, VS Code, Codex, Goose, Zed. It needs [Ollama](https://ollama.com) running locally and
[uv](https://docs.astral.sh/uv/) (for `uvx`).

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
| `system_info` | Chip, memory, model-size budget, Ollama status, installed models | local |
| `find_recipes` | What model + setup worked for a task, on what hardware, reproduced how often | public index (read) |
| `search_models` | Search the registry for models that fit this machine | local |
| `list_benchmark_suites` | Built-in test sets | local |
| `benchmark_models` | Run models on a suite or **your own samples**; score, latency, failures, verdict | local |
| `recommend_model` | Candidates from recipes + registry → test → smallest model that's good enough | local |
| `install_model` | Pull / import a model into Ollama | downloads weights |
| `ask_local_model` / `ask_local_model_batch` | Delegate prompts to a local model | local |
| `record_example` | Shadow mode: save (input, your answer) for a task | local file |
| `hire_model` | Test models against your recorded answers; hire the smallest that agrees | local |
| `list_tasks` | Shadowed tasks and current hires | local |
| `export_training_data` | Turn shadow examples into a fine-tuning JSONL | local file |
| `try_recipe` | Reproduce a community recipe here | local |
| `share_recipe` | Preview → (user approves) → submit a recipe as a GitHub issue | GitHub, only with `confirm=true` |
| `savings_report` | Calls and tokens kept local, estimated cloud cost avoided | local |

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
