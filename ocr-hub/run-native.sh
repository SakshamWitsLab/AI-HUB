#!/usr/bin/env bash
# Run OCR Hub natively — the only mode that supports Apple MPS (Metal) GPU.
# Usage: ./run-native.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "==> OCR Hub native run (MPS capable on Apple Silicon)"

# Venv
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

echo "==> Installing backend deps (skips torch reinstall if present)…"
pip install -q --upgrade pip
pip install -q fastapi "uvicorn[standard]" pydantic websockets huggingface_hub pytesseract pillow
pip install -q torch torchvision || true          # brings MPS support on macOS
pip install -q "transformers<5" sentencepiece protobuf tiktoken || true  # <5: trOCR tokenizer compat
python -c "import pytesseract" 2>/dev/null || true

# tesseract binary via Homebrew
if ! command -v tesseract >/dev/null 2>&1; then
  echo "==> Installing tesseract via Homebrew…"
  brew install tesseract || echo "   (brew failed — install tesseract manually for that engine)"
fi

echo "==> Starting server on http://localhost:8990"
OCR_HUB_DATA="$PWD/data" uvicorn backend.main:app --host 0.0.0.0 --port 8990
