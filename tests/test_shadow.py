import json

import pytest

from modexai import service, shadow
from modexai.backends import SimulatedBackend
from modexai.suites import get_suite


def _shadow(n=20):
    suite = get_suite("ticket-triage")
    for it in suite.items[:n]:
        shadow.record("triage", it["prompt"], it["check"]["expected"], system=suite.system, reference="claude")


def test_record_dedupes_and_counts():
    assert shadow.record("t", "hello", "billing")["recorded"]
    assert not shadow.record("t", "hello", "billing")["recorded"]
    assert shadow.list_tasks()[0]["examples"] == 1


def test_hire_picks_smallest_agreeing_model_and_delegates():
    _shadow()
    ctx = service.Context(backend=SimulatedBackend({"qwen3:0.6b": 0.4, "qwen3:1.7b": 1.0, "phi4-mini": 1.0}))
    out = service.hire_model(ctx, "triage", ["qwen3:0.6b", "qwen3:1.7b", "phi4-mini"], min_score=0.9)
    assert out["hired"] == "qwen3:1.7b"
    assert out["recipe_drafts"]
    ans = service.ask(ctx, "I was charged twice", task="triage")
    assert ans["model"] == "qwen3:1.7b" and ans["ran_on"] == "this machine"
    assert service.savings()["calls_kept_local"] == 1


def test_hire_needs_enough_examples():
    _shadow(3)
    ctx = service.Context(backend=SimulatedBackend({"m": 1.0}))
    with pytest.raises(ValueError):
        service.hire_model(ctx, "triage", ["m"])


def test_export_is_chat_jsonl(tmp_path):
    _shadow(5)
    out = shadow.export("triage", tmp_path / "train.jsonl")
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 5
    assert [m["role"] for m in rows[0]["messages"]] == ["system", "user", "assistant"]


def test_forget_removes_everything():
    _shadow(2)
    assert shadow.forget("triage")
    assert shadow.examples("triage") == []
