"""Use ModexAI as a library: test models on your own examples, then delegate.

    python examples/python_api.py
"""

from modexai import service
from modexai.suites import load_samples_file

ctx = service.Context()  # Ollama on localhost by default

# 1. What fits here, and what already works for this kind of task?
print(service.system_info(ctx)["hardware"]["label"])
print([r["model"]["tag"] for r in service.find_recipes(task="ticket-triage")["recipes"]][:5])

# 2. Test two installed models on *your* examples (they never leave this machine).
samples = load_samples_file("examples/samples/tickets.jsonl")
report = service.benchmark(ctx, ["qwen3:1.7b", "gemma3:1b"], samples=samples, task="my-tickets")
print(report["verdict"]["reason"])

# 3. Delegate.
winner = report["verdict"]["winner"]
if winner:
    out = service.ask(ctx, "My card was charged twice this month.", model=winner,
                      system="Reply with one label: billing, technical, account, feature-request, cancellation.")
    print(out["output"], f"({out['latency_ms']} ms, on this machine)")
