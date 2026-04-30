#!/usr/bin/env bash
# start-demo.sh — one-command local demo launcher for ModexAI
#
# Usage:
#   ./start-demo.sh            # Start API + Ollama, then run the agent demo
#   ./start-demo.sh --api-only # Start API + Ollama only (no agent)
#   ./start-demo.sh --seller   # Open the seller dashboard in the browser

set -euo pipefail

API_URL="http://localhost:8000"
API_ONLY=false
OPEN_SELLER=false

for arg in "$@"; do
  case "$arg" in
    --api-only) API_ONLY=true ;;
    --seller)   OPEN_SELLER=true ;;
  esac
done

# ── Helpers ────────────────────────────────────────────────────────────────

log()  { echo -e "\033[1;34m[modexai]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[modexai]\033[0m $*"; }
warn() { echo -e "\033[1;33m[modexai]\033[0m $*"; }
die()  { echo -e "\033[1;31m[modexai]\033[0m $*" >&2; exit 1; }

wait_for_api() {
  log "Waiting for API to be ready at $API_URL ..."
  local attempts=0
  until curl -sf "$API_URL/health" > /dev/null 2>&1; do
    attempts=$((attempts + 1))
    if [ $attempts -ge 40 ]; then
      die "API did not start within 120 s. Check 'docker compose logs api'."
    fi
    sleep 3
  done
  ok "API is up!"
}

open_browser() {
  local url="$1"
  if command -v open &>/dev/null; then
    open "$url"
  elif command -v xdg-open &>/dev/null; then
    xdg-open "$url"
  else
    warn "Could not open browser automatically. Visit: $url"
  fi
}

# ── Pre-flight checks ───────────────────────────────────────────────────────

command -v docker >/dev/null 2>&1 || die "Docker is not installed or not in PATH."

# ── Copy .env if missing ────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f .env ]; then
  log "No .env found — copying from .env.example"
  cp .env.example .env
fi

# ── Start Docker Compose stack ──────────────────────────────────────────────

log "Building and starting API + Ollama ..."
docker compose up --build --detach

wait_for_api

echo ""
ok "═══════════════════════════════════════════════"
ok "  ModexAI is running!"
ok ""
ok "  Marketplace API:      $API_URL"
ok "  Interactive API docs: $API_URL/docs"
ok "  Seller Dashboard:     $API_URL/seller"
ok "═══════════════════════════════════════════════"
echo ""

# ── Open seller dashboard ───────────────────────────────────────────────────

if $OPEN_SELLER; then
  log "Opening seller dashboard ..."
  open_browser "$API_URL/seller"
  exit 0
fi

if $API_ONLY; then
  log "API-only mode. To stop: docker compose down"
  exit 0
fi

# ── Run agent demo ──────────────────────────────────────────────────────────

log "Running agent demo ..."
echo ""

AGENT_DIR="$SCRIPT_DIR/agent-demo"

if ! command -v python3 &>/dev/null; then
  warn "Python3 not found — skipping agent demo. Run it manually:"
  warn "  cd agent-demo && pip install -r requirements.txt && python demo_agent.py"
  exit 0
fi

cd "$AGENT_DIR"

if [ ! -d .venv ]; then
  log "Creating agent-demo virtual environment ..."
  python3 -m venv .venv
fi

log "Installing agent-demo dependencies ..."
.venv/bin/pip install -q -r requirements.txt

MODEXAI_API_URL="$API_URL" .venv/bin/python demo_agent.py

echo ""
ok "Demo complete. Stack is still running."
ok "To stop: docker compose down"
