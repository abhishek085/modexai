# Notes for coding agents working on this repo

- Python 3.10+, package in `src/modexai`. Install: `uv venv && uv pip install -e ".[dev]"`.
- Tests: `uv run pytest -q` — they use `SimulatedBackend`, never a real model. Keep it that way.
- Lint: `uv run ruff check .`
- `src/modexai/service.py` is the single source of behaviour; `mcp_server.py` and `cli.py` are thin layers.
- After changing `registry/` or `src/modexai/suites/`, run `uv run modexai registry` — CI checks the
  bundled index is current.
- Never hand-edit benchmark numbers or recipes; they come from `scripts/campaign.py` / `modexai bench`.
- Privacy is a feature: no new network calls, telemetry or on-disk storage of user content without an
  explicit user action. Update `docs/privacy.md` if you touch any of this.
