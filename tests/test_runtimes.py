import httpx

from modexai import recipes, runtimes, service
from modexai.backends import OllamaBackend, OpenAICompatBackend, get_backend
from modexai.bench import run_benchmark
from modexai.registry import load_registry, validate_listing
from modexai.suites import get_suite

REG = load_registry()


def test_every_listing_names_its_weights():
    for m in REG.models:
        assert m.weights, m.id
        assert validate_listing(m.raw) == [], m.id
    assert any("not a Hugging Face repo" in e or "repo id" in e
               for e in validate_listing({**REG.get("qwen/qwen3-1.7b").raw, "weights": {"gguf": "not a repo"}}))


def test_a_model_is_found_by_any_name_a_runtime_serves_it_under():
    m = REG.get("qwen/qwen3-1.7b")
    for name, fmt in (("Qwen/Qwen3-1.7B", "safetensors"), ("mlx-community/qwen3-1.7b-4bit", "mlx"),
                      ("ggml-org/Qwen3-1.7B-GGUF:Q4_K_M", "gguf"), ("qwen3:1.7b", "ollama")):
        assert REG.get(name) is m, name
        assert m.format_of(name) == fmt, name


def test_served_name_matches_the_engine():
    m = REG.get("qwen/qwen3-1.7b")
    assert runtimes.served_name(m, m.id, "ollama") == "qwen3:1.7b"
    assert runtimes.served_name(m, m.id, "vllm") == "Qwen/Qwen3-1.7B"
    assert runtimes.served_name(m, m.id, "mlx-lm") == "mlx-community/Qwen3-1.7B-4bit"
    assert runtimes.served_name(m, "unsloth/Qwen3-1.7B-GGUF", "llama.cpp") == "unsloth/Qwen3-1.7B-GGUF"
    assert runtimes.served_name(None, "my-own-model", "vllm") == "my-own-model"


def test_run_options_put_the_native_runtime_first():
    m = REG.get("qwen/qwen3-1.7b")
    mac = [o["runtime"] for o in runtimes.run_options(m, "apple-silicon")]
    nvidia = [o["runtime"] for o in runtimes.run_options(m, "cuda")]
    assert mac[0] == "mlx-lm" and mac[-1] == "ollama" and "vllm" not in mac
    assert nvidia[0] == "vllm" and "mlx-lm" not in nvidia
    assert [o["runtime"] for o in runtimes.run_options(m, "cpu")] == ["llama.cpp", "ollama"]
    gated = runtimes.run_options(REG.get("google/gemma3-1b"), "cuda")[0]
    assert "hf auth login" in gated["note"]
    spark = runtimes.run_options(REG.get("nokast/spark-s1-4b-v6"))
    assert [o["command"] for o in spark] == ["osj serve --pull abhishek085/spark-s1-4b-v6"]
    assert runtimes.run_options(REG.get("nokast/spark-s1-4b-v6"), "apple-silicon") == []


def test_how_to_run_says_when_nothing_fits_here(monkeypatch):
    monkeypatch.setattr(service.hardware, "detect", lambda: {"accelerator": "apple-silicon", "chip": "M4", "ram_gb": 24})
    ctx = service.Context(backend=get_backend("simulated"))
    assert service.how_to_run(ctx, "qwen/qwen3-1.7b")["fits_this_machine"] is True
    spark = service.how_to_run(ctx, "nokast/spark-s1-4b-v6")
    assert spark["fits_this_machine"] is False and spark["options"]


def test_engine_shortcuts_and_auto_fallback(monkeypatch):
    monkeypatch.delenv("MODEXAI_BACKEND", raising=False)
    monkeypatch.delenv("MODEXAI_BASE_URL", raising=False)
    b = get_backend("mlx-lm")
    assert isinstance(b, OpenAICompatBackend) and b.name == "mlx-lm" and b.base_url.endswith(":8081/v1")
    assert get_backend("openai", base_url="http://127.0.0.1:8000/v1").name == "vllm"
    monkeypatch.setattr(runtimes, "detect", lambda **_: [])
    assert isinstance(get_backend(), OllamaBackend)
    monkeypatch.setattr(runtimes, "detect", lambda **_: [
        {"runtime": "ollama", "backend": "ollama", "base_url": "http://127.0.0.1:11434", "models": []},
        {"runtime": "llama.cpp", "backend": "openai", "base_url": "http://127.0.0.1:8080/v1", "models": ["x"]}])
    picked = get_backend()
    assert isinstance(picked, OpenAICompatBackend) and picked.name == "llama.cpp"


def test_benchmark_on_mlx_records_the_weights_and_runtime():
    suite = get_suite("ticket-triage")
    answers = {it["prompt"]: it["check"]["expected"] for it in suite.items}
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        import json
        body = json.loads(req.content)
        seen.append(body["model"])
        text = answers.get(body["messages"][-1]["content"], "billing")
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}],
                                         "usage": {"prompt_tokens": 20, "completion_tokens": 2}})

    b = OpenAICompatBackend(base_url="http://127.0.0.1:8081/v1")
    b.client = httpx.Client(transport=httpx.MockTransport(handler))
    res = run_benchmark(["qwen/qwen3-1.7b"], suite, b, registry=REG, warmup=False)[0]
    assert set(seen) == {"mlx-community/Qwen3-1.7B-4bit"} and res.score == 1.0
    r = recipes.from_result(res, task="classification", model=REG.get(res.model))
    assert r["runtime"]["engine"] == "mlx-lm"
    assert r["model"]["format"] == "mlx" and r["model"]["registry_id"] == "qwen/qwen3-1.7b"
    assert recipes.validate(r) == []
