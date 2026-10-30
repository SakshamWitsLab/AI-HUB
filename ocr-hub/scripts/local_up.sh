#!/usr/bin/env bash
# Run OCR Hub natively (no Docker).
#   make local         → start (auto-initializes on first run)
#   make local-setup   → (re)install python deps into .venv only
#
# Env knobs:
#   OCR_HUB_PORT   (default 8000)
#   OCR_HUB_HOST   (default 0.0.0.0)
#   OCR_HUB_SKIP_TESSERACT=1 → don't auto-brew-install tesseract
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${OCR_HUB_HOST:-0.0.0.0}"
PORT="${OCR_HUB_PORT:-8000}"
VENV="$ROOT_DIR/.venv"
PY="$VENV/bin/python"

log() { printf '\033[36m[local]\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------- venv ------
if [ ! -x "$PY" ]; then
    log "creating .venv (python3.12 preferred)"
    if command -v python3.12 >/dev/null; then
        python3.12 -m venv "$VENV"
    else
        python3 -m venv "$VENV"
    fi
fi
"$PY" -m pip install --upgrade pip >/dev/null

# ------------------------------------------------------------- deps ---------
# NOTE: pymupdf is force-reinstalled once here — a corrupt local wheel
# (mixed .so versions) made `import fitz` dlopen-crash the server.
install_deps() {
    log "installing python deps into .venv (may take a few minutes)"
    "$PY" -m pip install --no-cache-dir -r backend/requirements.txt
    # runtime deps the Dockerfile installs but requirements.txt omits
    "$PY" -m pip install --no-cache-dir pymupdf easyocr docling-core
    "$PY" -m pip install --no-cache-dir --force-reinstall pymupdf
}

if [ "${1:-}" = "setup" ]; then
    install_deps
    log "setup complete — run 'make local' to start"
    exit 0
fi

# only auto-install when something is actually missing
if ! "$PY" -c "import fastapi, uvicorn, fitz, easyocr, docling_core" >/dev/null 2>&1; then
    install_deps
fi

# ------------------------------------------------------------ tesseract -----
if ! command -v tesseract >/dev/null && [ -z "${OCR_HUB_SKIP_TESSERACT:-}" ]; then
    if command -v brew >/dev/null; then
        log "installing tesseract via Homebrew (needed by the tesseract engine)"
        brew install tesseract
    else
        log "WARN: tesseract not found — the tesseract engine will be unavailable (brew install tesseract)"
    fi
fi

# ------------------------------------------------------------- frontend -----
if [ ! -f data/frontend_dist/index.html ]; then
    if command -v npm >/dev/null && [ -d frontend ]; then
        log "building frontend bundle"
        (cd frontend && npm install --no-progress && npm run build)
        rm -rf data/frontend_dist && mkdir -p data/frontend_dist
        cp -r frontend/dist/. data/frontend_dist/
    else
        log "WARN: no data/frontend_dist — UI will 404 (run 'make front' or check data/)"
    fi
fi

# ---------------------------------------------------------------- run -------
export OCR_HUB_DATA="$ROOT_DIR/data"
export PYTHONPATH="$ROOT_DIR"
mkdir -p "$OCR_HUB_DATA"

if [ -t 0 ] && lsof -ti tcp:"$PORT" >/dev/null 2>&1; then
    log "port $PORT busy — killing old server"
    lsof -ti tcp:"$PORT" | xargs kill 2>/dev/null || true
    sleep 1
fi

log "starting → http://localhost:$PORT  (data dir: $OCR_HUB_DATA)"
exec "$VENV/bin/uvicorn" backend.main:app --host "$HOST" --port "$PORT"
