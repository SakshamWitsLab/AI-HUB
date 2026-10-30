"""Cloud & utility OCR providers (2026 APIs), subprocess-free.

These run directly in the hub process (like the OXR adapter) — no torch, no
subprocess. Every provider returns the same block format the runner engines
produce:  {text, confidence, box, engine}  where box = [l, t, r, b] pixels.

Providers
---------
- PyMuPDF      : local text-layer extraction from the *original* PDF page
- Mistral OCR  : POST https://api.mistral.ai/v1/ocr  (model mistral-ocr-latest)
- LlamaParse   : v2 tier API  api.cloud.llamaindex.ai  (async job + poll)
- Reducto      : POST https://api.reducto.ai/v1/parse (upload + OCR lines)
- Unstructured : POST $url/general/v0/general         (partition elements)
"""
import base64
import json
import os
import time

import requests
from pathlib import Path


def _key(settings: dict, name: str, env_var: str) -> str:
    """Provider key: env var wins over Settings-stored value."""
    return os.environ.get(env_var) or settings.get(name, "")


def _box_from_bbox(bbox: dict) -> list:
    """Mistral-style {top_left_x,...} or general bbox dict -> [l,t,r,b]."""
    return [
        bbox.get("top_left_x", bbox.get("left", 0)),
        bbox.get("top_left_y", bbox.get("top", 0)),
        bbox.get("bottom_right_x", bbox.get("left", 0) + bbox.get("width", 0)),
        bbox.get("bottom_right_y", bbox.get("top", 0) + bbox.get("height", 0)),
    ]


# --------------------------------------------------------------------------- #
# PyMuPDF — instant, local, reads the embedded text layer
# --------------------------------------------------------------------------- #
def run_pymupdf(pdf_path: str, page_no: int = 1) -> list:
    """Vector text layer of one PDF page via fitz `blocks` mode. No OCR —
    returns clean line-grouped blocks for born-digital/OCRed-underlying PDFs."""
    import fitz

    doc = fitz.open(pdf_path)
    if page_no < 1 or page_no > len(doc):
        raise RuntimeError(f"PyMuPDF: page {page_no} out of range (PDF has {len(doc)}p)")
    page = doc[page_no - 1]
    out = []
    for x0, y0, x1, y1, text, _, btype in page.get_text("blocks"):
        if btype != 0:  # 1 = image block, skip
            continue
        txt = text.strip()
        if not txt:
            continue
        out.append({
            "text": txt,
            "confidence": None,  # text layer is digital: no OCR confidence
            "box": [round(x0), round(y0), round(x1), round(y1)],
            "engine": "pymupdf",
        })
    doc.close()
    return out


def render_pymupdf_to_blocks(pdf_path: str, page_no: int = 1) -> list:
    """Fallback: if the text layer is empty the page is likely scanned — the
    caller surfaces this as an actionable message instead of empty success."""
    blocks = run_pymupdf(pdf_path, page_no)
    if not blocks:
        raise RuntimeError(
            "PyMuPDF found no text layer on this page — it's probably a scan. "
            "Use an OCR engine (Docling / EasyOCR / cloud) for scanned pages.")
    return blocks


# --------------------------------------------------------------------------- #
# Mistral OCR — POST /v1/ocr, sync, per-page image or whole PDF base64
# --------------------------------------------------------------------------- #
def run_mistral(page_image: str, api_key: str, cfg: dict | None = None) -> list:
    cfg = cfg or {}
    b64 = Path(page_image).read_bytes()
    mime = "image/png"
    doc = {
        "type": "image_url",
        "image_url": f"data:{mime};base64,{base64.b64encode(b64).decode()}",
    }
    r = requests.post(
        "https://api.mistral.ai/v1/ocr",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={
            "model": cfg.get("model", "mistral-ocr-latest"),
            "document": doc,
            "include_blocks": True,
            **({"pages": cfg["pages"]} if cfg.get("pages") else {}),
        },
        timeout=180,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Mistral OCR HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    page = (data.get("pages") or [{}])[0]
    blocks = []
    for b in page.get("blocks") or []:
        content = (b.get("content") or "").strip()
        if not content:
            continue
        blocks.append({
            "text": content,
            "confidence": b.get("confidence"),
            "box": _box_from_bbox(b),
            "engine": "mistral-ocr-latest",
            "type": b.get("type", "text"),
        })
    if not blocks and page.get("markdown"):
        # models without include_blocks: whole page as one block
        blocks.append({
            "text": page["markdown"].strip(), "confidence": None, "box": None,
            "engine": "mistral-ocr-latest", "type": "markdown"})
    return blocks


# --------------------------------------------------------------------------- #
# LlamaParse — v2 tier API, async job:  POST /parse/upload -> poll /parse/{id}
# --------------------------------------------------------------------------- #
_LLAMA_BASE = "https://api.cloud.llamaindex.ai"

_LLAMA_TO_BLOCKS_SKIP = {"page_number", "page_break"}


def run_llamaparse(file_path: str, api_key: str, cfg: dict | None = None,
                   poll_interval: float = 3.0, poll_timeout: float = 480.0) -> list:
    cfg = cfg or {}
    tier = cfg.get("tier", "cost_effective")
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"tier": tier, "version": cfg.get("version", "latest")}
    files = {"file": (Path(file_path).name, open(file_path, "rb"),
                      "application/octet-stream")}
    try:
        r = requests.post(f"{_LLAMA_BASE}/api/v2/parse/upload",
                          headers={**headers, "Accept": "application/json"},
                          data=data, files=files, timeout=300)
    finally:
        try:
            files["file"][1].close()
        except Exception:
            pass
    if r.status_code == 401:
        raise RuntimeError("LlamaParse: invalid LLAMA_CLOUD_API_KEY (401)")
    if r.status_code == 429:
        raise RuntimeError("LlamaParse: concurrent-job limit reached (429) — retry shortly")
    if r.status_code != 200:
        raise RuntimeError(f"LlamaParse upload HTTP {r.status_code}: {r.text[:300]}")
    job_id = r.json().get("id")
    if not job_id:
        raise RuntimeError(f"LlamaParse: no job id in response: {str(r.json())[:200]}")

    deadline = time.time() + poll_timeout
    want = "items" if tier != "fast" else "text"
    while True:
        g = requests.get(f"{_LLAMA_BASE}/api/v2/parse/{job_id}",
                         params={"expand": f"{want},markdown"},
                         headers=headers, timeout=60)
        if g.status_code == 200:
            job = g.json()
        elif g.status_code == 429:
            time.sleep(poll_interval)
            continue
        else:
            raise RuntimeError(f"LlamaParse poll HTTP {g.status_code}: {g.text[:300]}")
        status = (job.get("job") or {}).get("status", job.get("status", ""))
        if status in ("COMPLETED", "SUCCESS", "completed", "success"):
            break
        if status in ("FAILED", "failed"):
            raise RuntimeError(f"LlamaParse job failed: {str((job.get('job') or {}).get('error_message'))[:300]}")
        if time.time() > deadline:
            raise RuntimeError(f"LlamaParse job {job_id} still running after {int(poll_timeout)}s — check LlamaCloud dashboard")
        time.sleep(poll_interval)

    blocks = []
    items = job.get("items") or []
    for it in items:
        if not isinstance(it, dict):
            continue
        typ = (it.get("type") or it.get("kind") or "").lower()
        if typ in _LLAMA_TO_BLOCKS_SKIP:
            continue
        txt = str(it.get("text") or it.get("content") or it.get("markdown") or "").strip()
        if not txt:
            continue
        blk = {"text": txt, "confidence": None, "box": None, "engine": f"llamaparse ({tier})", "type": typ or "text"}
        coords = (it.get("metadata") or {}).get("coordinates") or it.get("coordinates")
        if coords and isinstance(coords, dict):
            pts = coords.get("points") or []
            if pts:
                xs = [p[0] if isinstance(p, (list, tuple)) else p.get("x", 0) for p in pts]
                ys = [p[1] if isinstance(p, (list, tuple)) else p.get("y", 0) for p in pts]
                blk["box"] = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
        blocks.append(blk)
    if not blocks and isinstance(job.get("markdown"), str) and job["markdown"].strip():
        blocks.append({"text": job["markdown"].strip(), "confidence": None,
                       "box": None, "engine": f"llamaparse ({tier})", "type": "markdown"})
    elif not blocks and isinstance(job.get("text"), str) and job["text"].strip():
        blocks.append({"text": job["text"].strip(), "confidence": None,
                       "box": None, "engine": f"llamaparse ({tier})", "type": "text"})
    return blocks


# --------------------------------------------------------------------------- #
# Reducto — POST /v1/parse (sync), upload page image multipart
# --------------------------------------------------------------------------- #
_REDUCTO = "https://api.reducto.ai/v1"


def run_reducto(page_image: str, api_key: str, cfg: dict | None = None) -> list:
    cfg = cfg or {}
    r = requests.post(
        f"{_REDUCTO}/parse",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": (Path(page_image).name, open(page_image, "rb"),
                        "application/octet-stream")},
        data={"config": json.dumps({
            "return_ocr_data": True,
            **(cfg.get("config") or {}),
        })},
        timeout=300)
    if r.status_code == 401:
        raise RuntimeError("Reducto: invalid REDUCTO_API_KEY (401)")
    if r.status_code == 429:
        raise RuntimeError("Reducto: rate limited (429) — retry shortly")
    if r.status_code != 200:
        raise RuntimeError(f"Reducto HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    result = data.get("result") or {}
    if result.get("type") == "url":  # oversized result: fetch full JSON
        result = requests.get(result["url"], timeout=60).json()
    result = result.get("result", result)
    blocks = []
    # word-level OCR lines (return_ocr_data) -> highest fidelity
    ocr = result.get("ocr") or {}
    for line in ocr.get("lines") or []:
        txt = (line.get("text") or "").strip()
        if not txt:
            continue
        b = line.get("bbox") or {}
        blk = {"text": txt, "confidence": line.get("confidence"),
               "box": None, "engine": "reducto", "type": "ocr-line"}
        if b:
            blk["box"] = [round(b.get("left", 0)), round(b.get("top", 0)),
                          round(b.get("left", 0) + b.get("width", 0)),
                          round(b.get("top", 0) + b.get("height", 0))]
        blocks.append(blk)
    if not blocks:  # fall back to chunk blocks with bbox
        for chunk in result.get("chunks") or []:
            for b in chunk.get("blocks") or []:
                txt = (b.get("content") or "").strip()
                if not txt:
                    continue
                blk = {"text": txt, "confidence": None, "box": None,
                       "engine": "reducto", "type": b.get("type", "text")}
                bb = b.get("bbox") or {}
                if bb:
                    blk["box"] = [round(bb.get("left", 0)), round(bb.get("top", 0)),
                                  round(bb.get("left", 0) + bb.get("width", 0)),
                                  round(bb.get("top", 0) + bb.get("height", 0))]
                blocks.append(blk)
    return blocks


# --------------------------------------------------------------------------- #
# OmniParse — self-hosted server (adithya-s-k/omniparse), Linux/GPU host
# --------------------------------------------------------------------------- #
def run_omniparse(pdf_path: str, url: str, cfg: dict | None = None) -> list:
    """Self-hosted OmniParse instance. You run it yourself (docker, on a Linux
    box):  docker run --gpus all -p 8000:8000 adithya11/omniparse:latest
    (image name per its docs). The hub POSTs the file and normalizes the
    markdown/elements it returns."""
    base = url.rstrip("/")
    endpoint = f"{base}/parse_document/pdf" if "/parse" not in base else base
    r = requests.post(
        endpoint,
        files={"file": (Path(pdf_path).name, open(pdf_path, "rb"), "application/pdf")},
        timeout=600)
    if r.status_code != 200:
        raise RuntimeError(f"OmniParse HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    # OmniParse returns a list of {text|content, coords|bbox?, metadata?} docs
    items = data if isinstance(data, list) else data.get("docs") or data.get("pages") or []
    blocks = []
    for it in items:
        if not isinstance(it, dict):
            continue
        txt = str(it.get("text") or it.get("content") or "").strip()
        if not txt:
            continue
        blk = {"text": txt, "confidence": None, "box": None,
               "engine": "omniparse", "type": "markdown"}
        c = it.get("coords") or it.get("bbox") or {}
        if isinstance(c, dict) and c.get("left") is not None:
            blk["box"] = _box_from_bbox(c)
        blocks.append(blk)
    return blocks


# --------------------------------------------------------------------------- #
# Unstructured — partition endpoint, multipart, vlm/ocr_only/hi_res strategy
# --------------------------------------------------------------------------- #
def run_unstructured(page_image: str, api_key: str, url: str = "",
                     cfg: dict | None = None) -> list:
    cfg = cfg or {}
    base = (url or "https://api.unstructured.io").rstrip("/")
    endpoint = f"{base}/general/v0/general" if "/general" not in base else base
    strategy = cfg.get("strategy", "ocr_only")
    data = {"strategy": strategy, "coordinates": "true",
            **(cfg.get("extra") or {})}
    r = requests.post(
        endpoint,
        headers={"unstructured-api-key": api_key},
        files={"files": (Path(page_image).name, open(page_image, "rb"),
                         "application/octet-stream")},
        data=data, timeout=300)
    if r.status_code == 401:
        raise RuntimeError("Unstructured: invalid UNSTRUCTURED_API_KEY (401)")
    if r.status_code == 422:
        d = r.json()
        raise RuntimeError(f"Unstructured validation error: {d.get('detail', r.text)[:300]}")
    if r.status_code == 429:
        raise RuntimeError("Unstructured: on-demand job limit (429) — retry shortly")
    if r.status_code != 200:
        raise RuntimeError(f"Unstructured HTTP {r.status_code}: {r.text[:300]}")
    blocks = []
    for el in r.json():
        if not isinstance(el, dict):
            continue
        txt = (el.get("text") or "").strip()
        if not txt:
            continue
        blk = {"text": txt, "confidence": None, "box": None,
               "engine": f"unstructured ({strategy})", "type": el.get("type", "text")}
        coords = ((el.get("metadata") or {}).get("coordinates") or {}).get("points") or []
        if coords:
            xs = [float(p.get("x", 0)) for p in coords if isinstance(p, dict)]
            ys = [float(p.get("y", 0)) for p in coords if isinstance(p, dict)]
            if xs and ys:
                blk["box"] = [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))]
        blocks.append(blk)
    return blocks
