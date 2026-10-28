"""Persistent store: settings (HF token etc.), catalog, installed models.

All state lives in DATA_DIR (../data by default) so it survives restarts and
is mounted as a Docker volume.
"""
import json
import os
import threading
from pathlib import Path

DATA_DIR = Path(os.environ.get("OCR_HUB_DATA", Path(__file__).resolve().parent.parent / "data"))
MODELS_DIR = DATA_DIR / "models"
SETTINGS_FILE = DATA_DIR / "config.json"
CATALOG_FILE = DATA_DIR / "catalog.json"
RESULTS_DIR = DATA_DIR / "results"

for _d in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
DEFAULT_SETTINGS = {
    "hf_token": "",
    "device_preference": "auto",  # auto | cuda | mps | cpu
    "parallel_limit": 6,
    "models_dir": str(MODELS_DIR),
    "oxr_cloud_enabled": False,   # SoMark OXR free API (uploads files to their servers)
}


def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text())
            return {**DEFAULT_SETTINGS, **data}
        except Exception:
            pass
    return dict(DEFAULT_SETTINGS)


def save_settings(patch: dict) -> dict:
    with _lock:
        cur = load_settings()
        cur.update({k: v for k, v in patch.items() if k in DEFAULT_SETTINGS})
        SETTINGS_FILE.write_text(json.dumps(cur, indent=2))
        return cur


def hf_token() -> str:
    tok = os.environ.get("HF_TOKEN", "")
    return tok or load_settings().get("hf_token", "")


# --------------------------------------------------------------------------- #
# Catalog / installed models
# --------------------------------------------------------------------------- #
DEFAULT_CATALOG = [
    {
        "id": "paddleocr-vl-1_6",
        "hf_id": "PaddlePaddle/PaddleOCR-VL-1.6",
        "name": "PaddleOCR-VL 1.6",
        "engine": "paddleocr",
        "size_gb": 1.93,
        "params": "0.96B",
        "languages": "109 languages",
        "task": "document-parse",
        "notes": "Latest PaddleOCR-VL. Strong document parsing, tables & formulas.",
        "tier": "curated",
        "downloads": 75397,
        "likes": 556,
    },
    {
        "id": "trocr-small-printed",
        "hf_id": "microsoft/trocr-small-printed",
        "name": "TrOCR Small (Printed)",
        "engine": "transformers",
        "size_gb": 0.49,
        "params": "61M",
        "languages": "en",
        "task": "printed-line-ocr",
        "notes": "Tiny printed-text line OCR. Runs comfortably on CPU.",
        "tier": "starter",
        "downloads": 16583,
        "likes": 58,
    },
    {
        "id": "trocr-small-handwritten",
        "hf_id": "microsoft/trocr-small-handwritten",
        "name": "TrOCR Small (Handwritten)",
        "engine": "transformers",
        "size_gb": 0.25,
        "params": "~60M",
        "languages": "en",
        "task": "handwritten-ocr",
        "notes": "Handwriting recognition, single text lines. CPU friendly.",
        "tier": "starter",
        "downloads": 100526,
        "likes": 74,
    },
    {
        "id": "nemotron-ocr-v2",
        "hf_id": "nvidia/nemotron-ocr-v2",
        "name": "NVIDIA Nemotron OCR v2",
        "engine": "nvidia-native",
        "size_gb": 0.55,
        "params": "det+rec .pth",
        "languages": "en / multilingual",
        "task": "ocr+layout",
        "notes": "NVIDIA document parser. Requires CUDA + Linux (builds a C++ extension from the official nemotron-ocr package) — not runnable on macOS/CPU; keep for GPU hosts.",
        "tier": "curated",
        "downloads": 1362,
        "likes": 268,
    },
    {
        "id": "nemotron-ocr-v1",
        "hf_id": "nvidia/nemotron-ocr-v1",
        "name": "NVIDIA Nemotron OCR v1",
        "engine": "nvidia-native",
        "size_gb": 0.21,
        "params": "det+rec .pth",
        "languages": "en",
        "task": "ocr+layout",
        "notes": "Compact NVIDIA document recognition. Same CUDA/Linux requirement as v2.",
        "tier": "curated",
        "downloads": 187,
        "likes": 121,
    },
    {
        "id": "lighton-ocr-1b",
        "hf_id": "lightonai/LightOnOCR-1B-1025",
        "name": "LightOnOCR 1B",
        "engine": "transformers",
        "size_gb": 2.34,
        "params": "1.4B",
        "languages": "multilingual",
        "task": "image-to-markdown",
        "notes": "Fast end-to-end OCR VLM producing markdown from document images.",
        "tier": "curated",
        "downloads": 19739,
        "likes": 256,
    },
    {
        "id": "granite-docling-258m",
        "hf_id": "ibm-granite/granite-docling-258M",
        "name": "Docling (Granite 258M)",
        "engine": "docling",
        "size_gb": 0.53,
        "params": "258M",
        "languages": "multilingual",
        "task": "document-parse",
        "notes": "IBM's tiny Docling VLM: tables, formulas, layout tags as structured output.",
        "tier": "starter",
        "downloads": 245871,
        "likes": 1287,
    },
    {
        "id": "oxr-1_0",
        "hf_id": "SoMarkAI/OXR-1.0",
        "name": "SoMark OXR-1.0 (cloud)",
        "engine": "somark-remote",
        "size_gb": 0,
        "params": "1.3B (run remotely)",
        "languages": "multilingual",
        "task": "doc-parse",
        "notes": "SOTA parses (#1 olmOCR-Bench): markdown/JSON + layout boxes via SoMark's free API. Nothing to download — but your file is uploaded to SoMark's free endpoint.",
        "tier": "starter",
        "cloud": True,
    },
    {
        "id": "trocr-base-printed",
        "hf_id": "microsoft/trocr-base-printed",
        "name": "TrOCR Base (Printed)",
        "engine": "transformers",
        "size_gb": 2.67,
        "params": "333M",
        "languages": "en",
        "task": "printed-line-ocr",
        "notes": "Higher-accuracy printed line recognition.",
        "tier": "curated",
    },
    {
        "id": "trocr-base-handwritten",
        "hf_id": "microsoft/trocr-base-handwritten",
        "name": "TrOCR Base (Handwritten)",
        "engine": "transformers",
        "size_gb": 2.67,
        "params": "333M",
        "languages": "en",
        "task": "handwritten-ocr",
        "notes": "Higher-accuracy handwriting recognition.",
        "tier": "curated",
    },
    {
        "id": "got-ocr2",
        "hf_id": "stepfun-ai/GOT-OCR2_0",
        "name": "GOT-OCR 2.0",
        "engine": "transformers",
        "size_gb": 1.44,
        "params": "0.72B",
        "languages": "multilingual",
        "task": "general-ocr",
        "notes": "General OCR 2.0: plain text, formulas, tables, charts.",
        "tier": "curated",
        "gated": True,
    },
    {
        "id": "florence-2-large",
        "hf_id": "microsoft/Florence-2-large",
        "name": "Florence-2 Large",
        "engine": "transformers",
        "size_gb": 3.12,
        "params": "0.78B",
        "languages": "multilingual",
        "task": "ocr+detection",
        "notes": "Vision multimodal: OCR + object detection + captioning.",
        "tier": "curated",
        "trust_remote_code": True,
    },
    {
        "id": "nanonets-ocr-s",
        "hf_id": "nanonets/Nanonets-OCR-s",
        "name": "Nanonets-OCR-s",
        "engine": "transformers",
        "size_gb": 7.53,
        "params": "3.8B",
        "languages": "en",
        "task": "pdf2markdown",
        "notes": "PDF -> Markdown with tables, LaTex formulas, signatures. Needs GPU.",
        "tier": "curated",
    },
    {
        "id": "easyocr-builtin",
        "hf_id": "builtin:easyocr",
        "name": "EasyOCR (built-in)",
        "engine": "easyocr",
        "size_gb": 0.1,
        "params": "detection+recognition",
        "languages": "80+ languages",
        "task": "full-page-ocr",
        "notes": "CRAFT detection + CRNN recognition. Lang models downloaded at first run.",
        "tier": "builtin",
    },
    {
        "id": "tesseract-builtin",
        "hf_id": "builtin:tesseract",
        "name": "Tesseract (built-in)",
        "engine": "tesseract",
        "size_gb": 0.02,
        "params": "classic engine",
        "languages": "100+ languages",
        "task": "full-page-ocr",
        "notes": "Battle-tested LSTM engine with language packs.",
        "tier": "builtin",
    },
]


def load_catalog() -> list:
    """Return the catalog, merging any default cards the local copy doesn't have."""
    base = list(DEFAULT_CATALOG)
    if CATALOG_FILE.exists():
        try:
            local = json.loads(CATALOG_FILE.read_text())
            ids = {c["id"] for c in local}
            for dflt in base:
                if dflt["id"] not in ids:
                    local.append(dflt)
            return local
        except Exception:
            pass
    return base


def save_catalog(catalog: list) -> list:
    with _lock:
        CATALOG_FILE.write_text(json.dumps(catalog, indent=2))
        return catalog


def register_model(card: dict) -> list:
    """Add/refresh a user-added model card (pre-registered, not installed)."""
    catalog = load_catalog()
    for i, c in enumerate(catalog):
        if c.get("hf_id") == card.get("hf_id"):
            catalog[i] = {**c, **card}
            break
    else:
        catalog.append(card)
    return save_catalog(catalog)


def all_hf_ids() -> dict:
    """Map hf_id -> catalog card."""
    return {c["hf_id"]: c for c in load_catalog()}


# --------------------------------------------------------------------------- #
# Downloaded models
# --------------------------------------------------------------------------- #
def model_path(hf_id: str) -> Path:
    safe = hf_id.replace("/", "__")
    return MODELS_DIR / safe


def is_installed(hf_id: str) -> bool:
    p = model_path(hf_id)
    if not p.exists():
        return False
    return (p / "INSTALL_OK").exists()


def mark_installed(hf_id: str) -> None:
    model_path(hf_id).mkdir(parents=True, exist_ok=True)
    (model_path(hf_id) / "INSTALL_OK").write_text("")


def installed_ids() -> list:
    out = []
    if MODELS_DIR.exists():
        for d in MODELS_DIR.iterdir():
            if (d / "INSTALL_OK").exists():
                out.append(d.name.replace("__", "/"))
    return out
