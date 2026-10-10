# Benchmark suites

Small, hand-written, deterministic test sets for jobs agents actually hand off. No LLM-as-judge: every
case has a checkable answer, and every check is reproducible.

| Suite | Task | Cases | Graded by |
|---|---|---|---|
| `ticket-triage` | classification | 20 | label match (5 queues) |
| `json-extraction` | extraction | 16 | field-by-field JSON comparison (numbers tolerant to formatting) |
| `prompt-injection` | prompt-injection | 20 | label match, includes judgment-flip pairs |
| `text-to-sql` | text-to-sql | 15 | **execution**: the model's query and the gold query run on the same in-memory SQLite DB |
| `tool-calling` | tool-calling | 15 | tool name + required arguments |
| `fact-check` | fact-checking | 16 | supported / unsupported |
| `math-word` | math | 15 | final number |
| `pii-redaction` | pii-redaction | 12 | nothing personal leaks, useful words survive |
| `content-moderation` | moderation | 14 | safe / unsafe (with edgy-but-safe distractors) |

Every suite is tested in CI: a perfect simulated model must score 100% and a broken one 0%, which proves
the gold answers and the graders agree.

## Your own test set

Any `.jsonl` with `prompt` and `expected`:

```jsonl
{"prompt": "Refund my duplicate charge", "expected": "billing"}
{"prompt": "Invoice INV-9 from Acme, total 1,200 EUR", "expected": {"vendor": "Acme", "total": 1200}}
```

```bash
modexai bench qwen3:1.7b gemma3:1b --samples my-cases.jsonl --system "Reply with one label: …"
```

ModexAI infers the check from `expected` (label, number, JSON fields, or fuzzy text match); add an
explicit `"check": {...}` to override. Custom samples never leave your machine.

## Adding a built-in suite

Add `src/modexai/suites/<id>.json` (copy an existing one), run `pytest`, and open a PR. Good suites are
narrow, realistic, 12–25 cases, and have unambiguous answers. Bump `version` whenever cases change, so
recipes stay comparable.
