import asyncio
import json
from pathlib import Path

from modexai import mcp_server, service
from modexai.agents import BRIEF, CLIENTS, UVX
from modexai.backends import SimulatedBackend
from modexai.cli import main

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None

EXPECTED_TOOLS = {
    "system_info", "search_models", "list_benchmark_suites", "benchmark_models", "recommend_model", "install_model",
    "ask_local_model", "ask_local_model_batch", "record_example", "hire_model", "list_tasks", "export_training_data",
    "find_recipes", "try_recipe", "share_recipe", "savings_report", "read_notes", "post_note",
}


def test_mcp_exposes_the_full_loop():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    assert {t.name for t in tools} >= EXPECTED_TOOLS
    assert all(t.description for t in tools)


def test_mcp_benchmark_call_with_simulated_backend(monkeypatch):
    monkeypatch.setattr(mcp_server, "_ctx", service.Context(backend=SimulatedBackend({"qwen3:1.7b": 1.0})))
    res = asyncio.run(mcp_server.mcp.call_tool("benchmark_models", {"models": ["qwen3:1.7b"], "suite": "ticket-triage"}))
    data = json.loads(res.content[0].text)
    assert data["results"][0]["score"] == 1.0
    assert data["verdict"]["winner"] == "qwen3:1.7b"


def test_share_requires_confirmation(monkeypatch):
    monkeypatch.setattr(mcp_server, "_ctx", service.Context(backend=SimulatedBackend({"qwen3:1.7b": 1.0})))
    res = asyncio.run(mcp_server.mcp.call_tool("benchmark_models", {"models": ["qwen3:1.7b"], "suite": "ticket-triage"}))
    rid = json.loads(res.content[0].text)["recipe_drafts"][0]
    prev = service.share_recipe(rid, confirm=False)
    assert prev["shared"] is False and prev["preview"]["id"] == rid


def test_cli_commands(capsys):
    assert main(["suites", "--json"]) == 0
    assert "ticket-triage" in capsys.readouterr().out
    assert main(["--backend", "simulated", "bench", "qwen3:1.7b", "--suite", "math-word", "--json"]) == 0
    assert '"suite": "math-word@1"' in capsys.readouterr().out
    assert main(["search", "--task", "text-to-sql", "--json"]) == 0
    assert main(["savings", "--json"]) == 0


def test_cli_demo_runs_offline(capsys):
    assert main(["demo", "--fast"]) == 0
    assert "verdict" in capsys.readouterr().out


def test_agent_brief_is_a_prompt_a_command_and_a_doc(capsys):
    prompts = asyncio.run(mcp_server.mcp.list_prompts())
    assert [p.name for p in prompts] == ["small_models_in_my_loop"]
    got = asyncio.run(mcp_server.mcp.get_prompt("small_models_in_my_loop"))
    assert got.messages[0].content.text == BRIEF

    assert main(["prompt", "--client", "codex"]) == 0
    out = capsys.readouterr().out
    assert BRIEF in out and "[mcp_servers.modexai]" in out

    want = {"command": "uvx", "args": UVX}
    assert json.loads(CLIENTS["cursor"]["setup"])["mcpServers"]["modexai"] == want
    assert json.loads(CLIENTS["vscode"]["setup"])["servers"]["modexai"] == want
    if tomllib:
        assert tomllib.loads(CLIENTS["codex"]["setup"])["mcp_servers"]["modexai"] == want

    root = Path(__file__).resolve().parents[1]
    for doc in ("docs/agents.md", "examples/agent-instructions.md"):
        assert BRIEF in (root / doc).read_text(encoding="utf-8"), f"{doc} is out of date with agents.BRIEF"
    agents_doc = (root / "docs/agents.md").read_text(encoding="utf-8")
    assert all(c["setup"] in agents_doc for c in CLIENTS.values())
