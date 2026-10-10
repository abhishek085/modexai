import json

import httpx
import pytest

from modexai import service
from modexai.backends import BackendError, OsjBackend, get_backend
from modexai.bench import run_benchmark
from modexai.registry import load_registry, validate_listing
from modexai.suites import get_suite

GATE = get_suite("tool-call-gate")
ANSWERS = {it["prompt"]: it["check"]["expected"] for it in GATE.items}


def fake_gateway(answers=ANSWERS, seen=None):
    """A stand-in for `osj serve`: answers each choice question from a lookup table."""

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": "spark-s1-4b-v6"}]})
        body = json.loads(req.content)
        if seen is not None:
            seen.append(body)
        q = body["questions"][0]
        pick = answers.get(body["state"]["content"], q["options"][0])
        probs = [0.9 if o == pick else 0.1 / (len(q["options"]) - 1) for o in q["options"]]
        return httpx.Response(200, json={"answers": [{"type": "choice", "labels": q["options"], "probs": probs,
                                                      "selected": pick, "confidence": 0.9, "entropy": 0.4}],
                                         "model": body.get("model"), "latency_ms": 60.0})

    b = OsjBackend()
    b.client = httpx.Client(transport=httpx.MockTransport(handler))
    return b


def test_osj_backend_sends_a_typed_choice_question():
    seen = []
    b = fake_gateway(seen=seen)
    it = GATE.items[0]
    gen = b.generate("spark-s1-4b-v6", GATE.messages(it), item=it)
    assert gen.text == "allow" and gen.confidence == 0.9
    q = seen[0]["questions"][0]
    assert seen[0]["model"] == "spark-s1-4b-v6"
    assert seen[0]["state"]["content"] == it["prompt"]
    assert q == {"type": "choice", "prompt": GATE.system, "options": ["allow", "ask", "deny"]}
    assert b.installed() == {"spark-s1-4b-v6"}


def test_osj_backend_refuses_open_ended_prompts():
    with pytest.raises(BackendError, match="closed questions"):
        fake_gateway().generate("spark-s1-4b-v6", [{"role": "user", "content": "write a poem"}])


def test_decision_model_benchmarks_through_the_normal_graders():
    res = run_benchmark(["nokast/spark-s1-4b-v6"], GATE, fake_gateway(), registry=load_registry())[0]
    assert res.error is None and res.backend == "osj"
    assert res.model == "spark-s1-4b-v6" and res.registry_id == "nokast/spark-s1-4b-v6"
    assert res.score == 1.0


def test_decision_listings_are_valid_and_named_for_osj():
    reg = load_registry()
    m = reg.get("nokast/spark-s1-4b-v6")
    assert m.kind == "decision" and m.ollama_tag == "spark-s1-4b-v6"
    assert reg.get("spark-s1-4b-v6") is m
    bad = dict(m.raw, kind="chat")
    assert any("together" in e for e in validate_listing(bad))


def test_decision_models_show_in_search_but_not_in_chat_recommendations():
    ctx = service.Context(backend=get_backend("simulated"))
    found = service.search(ctx, task="tool-gate", max_size_gb=64)["models"]
    card = next(c for c in found if c["id"] == "nokast/spark-s1-4b-v6")
    assert card["kind"] == "decision" and "osj serve --pull" in card["run_with"]
    chat = ctx.registry.search(task="tool-gate", max_size_gb=64, kind="chat")
    assert all(m.kind == "chat" for m in chat)


def test_install_explains_how_to_serve_a_decision_model():
    out = service.install(service.Context(backend=get_backend("simulated")), "nokast/spark-s1-4b-v6")
    assert out["installed"] is False
    assert out["options"][0]["command"] == "osj serve --pull abhishek085/spark-s1-4b-v6"


def test_ask_with_labels_returns_the_decision_and_confidence():
    ctx = service.Context(backend=fake_gateway({"rm -rf ~/": "deny"}))
    out = service.ask(ctx, "rm -rf ~/", model="nokast/spark-s1-4b-v6", system="Approve this tool call?",
                      labels=["allow", "ask", "deny"])
    assert out["output"] == "deny" and out["confidence"] == 0.9


def test_backend_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("MODEXAI_BACKEND", "osj")
    monkeypatch.setenv("MODEXAI_BASE_URL", "http://spark.local:8400/v1")
    b = get_backend()
    assert isinstance(b, OsjBackend) and b.base_url == "http://spark.local:8400/v1"
