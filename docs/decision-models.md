# Decision models (System One)

Most of what an agent decides in a loop is not writing — it is choosing. *Should this tool call run?
Which queue does this ticket go to? Is this tool output trying to hijack me?* Sending each of those to a
frontier API costs a round-trip, tokens and your data. A **System One decision model** answers them on
your machine in one forward pass: you give it a state and a closed question, it gives back a probability
for every option, and a confidence you can act on.

ModexAI lists the open [spark-s1](https://huggingface.co/abhishek085/spark-s1-4b-v6) family from the
Nokast community and benchmarks it the same way as every other model — so agents can see, with numbers
from real machines, when a 4B decision model is enough to gate their tool calls.

| Model | Size | Runs on | Good at |
|---|---|---|---|
| [`nokast/spark-s1-4b-v6`](https://huggingface.co/abhishek085/spark-s1-4b-v6) | 8.4 GB bf16 | NVIDIA GPU | tool-call gating, routing, injection checks |
| [`nokast/spark-s1-4b-v6-nvfp4`](https://huggingface.co/abhishek085/spark-s1-4b-v6-nvfp4) | 5.2 GB | Blackwell GPU (DGX Spark) | same as v6, faster |
| [`nokast/spark-s1-4b-v8-nvfp4`](https://huggingface.co/abhishek085/spark-s1-4b-v8-nvfp4) | 5.2 GB | Blackwell GPU | adds next-tool choice, command risk, injected tool output |
| [`nokast/jev-control-core`](https://huggingface.co/abhishek085/jev-control-core) | 1.5 GB | GPU, or GGUF on llama.cpp (Apple Silicon) | agent control points: guardrails, routing, triage, escalation |

The weights are open (Apache-2.0) on Hugging Face. Serve them however you like — the
[open-spark-jev](https://github.com/abhishek085/open-spark-jev) gateway, vLLM, TensorRT-LLM, llama.cpp.
ModexAI talks to the gateway, because that is where the calibrated option probabilities come from.

## Run one

```bash
git clone https://github.com/abhishek085/open-spark-jev && cd open-spark-jev
python3 -m venv .venv && . .venv/bin/activate
pip install torch -e ".[infer]"
osj serve --pull spark-s1-4b-v6          # http://127.0.0.1:8400, POST /v1/decide
```

For production speed, put vLLM (or trtllm-serve) behind the gateway:
`python -m open_spark_jev.serve.gateway --backend openai --upstream http://localhost:8355/v1`.

## Benchmark it

```bash
modexai --backend osj bench spark-s1-4b-v6 --suite tool-call-gate
modexai --backend osj bench spark-s1-4b-v6 --suite prompt-injection
```

`--backend osj` sends each case to `/v1/decide` as one typed `choice` question: the case is the state, the
suite's instruction is the question, and the suite's labels are the options. The chosen option is graded
like any other model's answer, so scores are directly comparable with the Ollama models on the same suite.
Point at another machine with `--base-url http://spark.local:8400/v1`.

Decision models only answer **closed** questions, so they run on the label suites —
`tool-call-gate`, `prompt-injection`, `content-moderation`, `ticket-triage`, `fact-check` — and on your
own samples when every answer is one of a fixed set of labels. Open-ended suites (SQL, JSON extraction,
math) say so instead of scoring zero.

Then share the recipe as usual. Until somebody does, the exchange shows these models as *not yet
benchmarked* — numbers here only ever come from real runs.

## Use one from your agent

Run the MCP server against the gateway:

```bash
claude mcp add modexai -e MODEXAI_BACKEND=osj -- uvx --from git+https://github.com/abhishek085/modexai modexai mcp
```

and ask with a closed answer set:

```json
{"tool": "ask_local_model", "arguments": {
  "model": "spark-s1-4b-v6",
  "system": "Approve this tool call? allow = read-only or easy to undo; ask = needs the user; deny = destructive.",
  "prompt": "{\"tool\": \"bash\", \"command\": \"git push --force origin main\"}",
  "labels": ["allow", "ask", "deny"]}}
```

→ `{"output": "deny", "confidence": <probability of "deny">, "latency_ms": ..., ...}` — escalate to the user
when the confidence is below your threshold.

`labels` also works with chat models on the Ollama backend (it becomes an instruction to reply with one
label), so an agent can benchmark a decision model against a small chat model on the same question.

## Safety

A decision model is a fast first opinion, not an authorisation system. Keep deterministic rules for the
things that must never happen, auto-allow only above a high confidence, send everything else to the user,
and recalibrate the threshold on your own labelled outcomes — the model cards say the same.
