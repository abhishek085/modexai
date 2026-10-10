import asyncio
import json

from modexai import mcp_server, service
from modexai.backends import SimulatedBackend
from modexai.cli import main

EXPECTED_TOOLS = {
    "system_info", "search_models", "list_benchmark_suites", "benchmark_models", "recommend_model", "install_model",
    "ask_local_model", "ask_local_model_batch", "record_example", "hire_model", "list_tasks", "export_training_data",
    "find_recipes", "try_recipe", "share_recipe", "savings_report",
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
