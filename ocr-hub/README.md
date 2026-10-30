# 🔷 OCR Hub

A self-hosted hub for OCR models: browse a curated catalog, add models from
HuggingFace by ID, download them to your machine with progress tracking, and
test several models **in parallel** on the playground with bounding-box overlays.

## Run

```bash
make up        # build + start  →  http://localhost:8990
```

That's it — see [Makefile targets](#makefile) below (uses Docker Compose under the hood).

## Sections

| Tab | What it does |
|---|---|
| **Catalog** | Curated model cards with verified size/params/license/downloads. "Starter picks" = models under 1 GB. Nothing is pre-installed — download what you want. |
| **Add model** | Paste any HF id, e.g. `PaddlePaddle/PaddleOCR-VL-1.6` → live card preview → add + download. Auto-detects the engine from repo metadata. |
| **Playground** | Select 1..n installed models + device (auto-detects CUDA/MPS/CPU with ✅/❌ badges), drop an image, run all models in parallel, compare overlay / text / JSON per model. |
| **Settings** | HuggingFace token (validated, stored locally in `data/config.json`) for faster/unlimited downloads and gated models, preferred device, parallel run limit. |

## Default catalog (verified from HuggingFace)

| Model | Size | Params | Best for |
|---|---|---|---|
| PaddleOCR-VL 1.6 (`PaddlePaddle/PaddleOCR-VL-1.6`) | 1.93 GB | 0.96B | Doc parsing, 109 languages |
| TrOCR Small Printed (`microsoft/trocr-small-printed`) | **0.49 GB** | 61M | Printed lines, CPU-friendly |
| TrOCR Small Handwritten (`microsoft/trocr-small-handwritten`) | **0.25 GB** | ~60M | Handwriting, CPU-friendly |
| TrOCR Base Printed | 2.67 GB | 333M | Higher accuracy printed |
| TrOCR Base Handwritten | 2.67 GB | 333M | Higher accuracy handwriting |
| GOT-OCR 2.0 (`stepfun-ai/GOT-OCR2_0`) | 1.44 GB | 0.72B | Text, formulas, tables (gated) |
| Florence-2 Large | 3.12 GB | 0.78B | OCR + detection |
| Nanonets-OCR-s | 7.53 GB | 3.8B | PDF→Markdown |
| EasyOCR / Tesseract | built-in | — | Full-page OCR, 80+/100+ languages |

## Architecture

```
ocr-hub/
├── backend/           FastAPI + WS
│   ├── devices.py     CUDA / MPS / CPU auto-detection
│   ├── hf.py          HF metadata + threaded downloads w/ progress
│   ├── runner/        engine adapters (tesseract, easyocr, transformers, paddle)
│   └── store.py       catalog, settings, installed-model registry
├── frontend/          Vite + React + Tailwind (built dist served by backend)
├── data/              mounted volume: models/, config.json, results/
├── docker-compose.yml # --profile cpu | --profile gpu
### Makefile
| Target | What it does |
|---|---|
| `make up` | build image + start → http://localhost:8990 |
| `make stop` | stop the container (models in `data/` are kept) |
| `make logs` | follow backend logs |
| `make front` | rebuild the frontend bundle + restart container |
| `make clean` | remove container + downloaded models (fresh start) |
```

- Models are stored under `data/models/<org>__<name>/`.
- `data/config.json` holds your HF token locally — nothing leaves your machine except HuggingFace calls.
- Docker on macOS can't access Metal/MPS GPU — CPU is used there; run `--profile gpu` on Linux/NVIDIA hosts for CUDA.


```
cd "/Users/saksham/code/AI HUB/ocr-hub"
python3.12 -m venv .venv && source .venv/bin/activate
# mirror the Dockerfile's deps:
pip install fastapi "uvicorn[standard]" websockets huggingface_hub requests \
    pytesseract pillow pymupdf "transformers<5" sentencepiece protobuf tiktoken \
    simpleitk docling-core "torch torchvision" easyocr
brew install tesseract            # tesseract CLI, for the tesseract engine
export OCR_HUB_DATA="$PWD/data"
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```