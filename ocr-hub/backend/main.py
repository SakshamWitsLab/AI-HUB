"""OCR Hub — FastAPI backend."""
import asyncio
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import devices, hf, store

app = FastAPI(title="OCR Hub", version="0.2.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

ROOT_DIR = Path(__file__).resolve().parent.parent
WARMUP = None  # placeholder for device runner warmup


# --------------------------------------------------------------------------- #
# Models & catalog
# --------------------------------------------------------------------------- #
def _merge_catalog():
    s = store.load_settings()
    catalog = store.load_catalog()
    for c in catalog:
        if c["hf_id"].startswith("builtin:"):
            c["installed"] = True
        elif c.get("cloud"):
            c["installed"] = bool(s.get("oxr_cloud_enabled"))
        else:
            c["installed"] = store.is_installed(c["hf_id"])
        c["downloading"] = (
            c["hf_id"] in hf.downloads.jobs
            and hf.downloads.jobs[c["hf_id"]].get("status") in ("running", "queued")
        )
    return catalog


@app.get("/api/models")
def list_models():
    return {"models": _merge_catalog(), "downloads": hf.downloads.status()}


@app.get("/api/models/preview")
def preview_model(repo: str):
    try:
        meta = hf.fetch_metadata(repo)
    except ValueError as e:
        raise HTTPException(400, str(e))
    meta["installed"] = store.is_installed(repo)
    return meta


@app.post("/api/models/add")
def add_model(repo: str):
    try:
        meta = hf.fetch_metadata(repo)
    except ValueError as e:
        raise HTTPException(400, str(e))
    engine = "paddleocr" if "paddleocr" in repo.lower() else "transformers"
    card = {
        "id": repo.replace("/", "__").lower(),
        "hf_id": repo,
        "name": meta["name"],
        "engine": engine,
        "size_gb": meta["size_gb"],
        "params": meta["params"],
        "license": meta["license"],
        "task": meta.get("pipeline") or "general-ocr",
        "languages": "multilingual",
        "notes": f"Added from HuggingFace ({meta['downloads']} downloads, license: {meta['license']}).",
        "tier": "user",
        "installed": False,
    }
    store.register_model(card)
    return {"ok": True, "card": card}


@app.post("/api/models/{repo:path}/download")
def download_model(repo: str):
    if repo.startswith("builtin:"):
        raise HTTPException(400, "Built-in engines ship with the environment — nothing to download")
    try:
        hf.fetch_metadata(repo)  # validate existence / gating before spending time
    except ValueError as e:
        raise HTTPException(400, str(e))
    return hf.downloads.start(repo)


@app.delete("/api/models/{repo:path}")
def uninstall_model(repo: str):
    """Remove an installed model's files — leaves the card pre-registered."""
    if repo.startswith("builtin:"):
        raise HTTPException(400, "Built-in engines can't be removed")
    p = store.model_path(repo)
    if p.exists():
        import shutil

        shutil.rmtree(p)
    hf.downloads.jobs.pop(repo, None)
    return {"ok": True, "removed": repo}


@app.get("/api/storage")
def storage():
    def dir_size(p: Path) -> int:
        t = 0
        for root, _dirs, files in os.walk(p):
            for f in files:
                try:
                    t += (Path(root) / f).stat().st_size
                except OSError:
                    pass
        return t

    per_model = []
    if store.MODELS_DIR.exists():
        for d in sorted(store.MODELS_DIR.iterdir()):
            if d.is_dir():
                per_model.append({
                    "repo": d.name.replace("__", "/"),
                    "gb": round(dir_size(d) / 1e9, 2),
                })
    return {"total_gb": round(dir_size(store.MODELS_DIR) / 1e9, 2), "models": per_model}


# --------------------------------------------------------------------------- #
# Devices / settings
# --------------------------------------------------------------------------- #
@app.get("/api/devices")
def list_devices():
    return {
        "devices": devices.detect(),
        "preference": store.load_settings()["device_preference"],
    }


@app.get("/api/settings")
def get_settings():
    s = store.load_settings()
    s["hf_token"] = ("•" * 8 + s["hf_token"][-4:]) if s["hf_token"] else ""
    s["hf_token_set"] = bool(s["hf_token"])
    return s


class SettingsPatch(BaseModel):
    hf_token: str | None = None
    device_preference: str | None = None
    parallel_limit: int | None = None
    oxr_cloud_enabled: bool | None = None


@app.put("/api/settings")
def put_settings(patch: SettingsPatch):
    cur = store.save_settings({k: v for k, v in patch.model_dump().items() if v is not None})
    cur["hf_token"] = ("•" * 8 + cur["hf_token"][-4:]) if cur["hf_token"] else ""
    return cur


# --------------------------------------------------------------------------- #
# Run OCR — isolated subprocess per model × page
# --------------------------------------------------------------------------- #
class RunRequest(BaseModel):
    image: str  # base64 image OR pdf bytes (data URI allowed)
    model_ids: list[str]
    device: str | None = None


def _sanitize(blocks) -> list:
    """Ensure plain-JSON types (numpy doesn't serialize)."""
    out = []
    for b in blocks or []:
        box = b.get("box")
        try:
            box = [float(v) for v in box] if box is not None else None
        except Exception:
            box = None
        conf = b.get("confidence")
        out.append({
            "text": str(b.get("text", "")),
            "confidence": float(conf) if conf is not None else None,
            "box": box,
            "engine": str(b.get("engine", "")),
            **({"model_id": str(b["model_id"])} if b.get("model_id") else {}),
            **({"detail": str(b["detail"])} if b.get("detail") else {}),
        })
    return out


def _card_for(hf_id: str) -> dict:
    for c in store.load_catalog():
        if c["hf_id"] == hf_id:
            return c
    return {"engine": "transformers", "hf_id": hf_id, "trust_remote_code": False}


def pdf_to_pages(pdf_path: str) -> list[str]:
    """Render each PDF page to PNG at 2x scale; returns image paths."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    out = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        p = store.RESULTS_DIR / f"page_{uuid.uuid4().hex}.png"
        pix.save(str(p))
        out.append(str(p))
    doc.close()
    return out


GLOBAL_EXECUTOR = ThreadPoolExecutor(max_workers=6)


def _run_one(hf_id: str, img_path: str, device: str, cfg: dict | None = None,
             ctx: dict | None = None) -> dict:
    """Run one model on one page. Cloud models (OXR) go through their API in
    a subprocess-free fast path; local models run in an isolated subprocess,
    so a crash can't kill the hub."""
    t0 = time.time()
    card = _card_for(hf_id)
    engine = "docling" if card.get("engine") == "docling" else card.get("engine", "transformers")
    ctx = ctx or {}
    notify = lambda status, progress=0, result=None: hf.downloads.run_event(
        run_id=ctx.get("run_id"), page=ctx.get("page"), model_id=hf_id,
        status=status, progress=progress, result=result)

    if engine == "somark-remote":
        if not store.load_settings().get("oxr_cloud_enabled"):
            out = {"model_id": hf_id, "name": card.get("name", hf_id), "engine": engine,
                    "elapsed_s": 0, "blocks": [],
                    "error": "Cloud parsing is off — enable 'OXR cloud' in Settings"}
            notify("done", result=out)
            return out
        try:
            blocks = hf.oxr_parse(img_path)
            out = {"model_id": hf_id, "name": card.get("name", hf_id), "engine": engine,
                    "device": "cloud", "elapsed_s": round(time.time() - t0, 2),
                    "blocks": _sanitize(blocks), "error": None}
            notify("done", result=out)
            return out
        except Exception as e:
            out = {"model_id": hf_id, "name": card.get("name", hf_id), "engine": engine,
                    "elapsed_s": round(time.time() - t0, 2), "blocks": [],
                    "error": f"OXR API error: {str(e)[:300]}"}
            notify("done", result=out)
            return out

    cfg = cfg or {}
    notify("starting", 0)

    tmp_cfg = None
    try:
        block_cfg = {
            **cfg,
            "hf_id": hf_id,
            "local_path": str(store.model_path(hf_id)) if store.is_installed(hf_id) else hf_id,
        }
        fjson = {"engine": engine, "device": device, "image_path": img_path, **block_cfg}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(fjson, tf)
            tmp_cfg = tf.name

        notify("running", 5)
        proc = subprocess.run(
            [sys.executable, "-m", "backend.runner_cli", tmp_cfg],
            capture_output=True, text=True,
            timeout=int(cfg.get("timeout_s", 900)),
            cwd=str(ROOT_DIR),
            env={**os.environ, "PYTHONPATH": str(ROOT_DIR)},
        )
        last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "{}"
        try:
            data = json.loads(last)
        except Exception:
            data = {"crash": (proc.stderr or proc.stdout or "no output")[-300:]}

        if data.get("crash") or (proc.returncode not in (0, 3) and not data.get("blocks")):
            tail = f"Runner crashed: {str(data.get('crash'))[:300]}"
            stderr_tail = (proc.stderr or "").strip().splitlines()
            if stderr_tail:
                tail += f" | engine log: {' / '.join(stderr_tail[-3:])}"
            out = {
                "model_id": hf_id, "name": card.get("name", hf_id), "engine": engine,
                "elapsed_s": round(time.time() - t0, 2), "blocks": [],
                "error": tail[:500],
            }
            notify("done", result=out)
            return out
        out = {
            "model_id": hf_id,
            "name": card.get("name", hf_id),
            "engine": engine,
            "device": device,
            "elapsed_s": round(time.time() - t0, 2),
            "blocks": _sanitize(data.get("blocks", [])),
            "error": None,
        }
        notify("done", result=out)
        return out
    except subprocess.TimeoutExpired:
        out = {"model_id": hf_id, "name": card.get("name", hf_id), "engine": engine,
                "elapsed_s": round(time.time() - t0, 2), "blocks": [], "error": "Timed out"}
        notify("done", result=out)
        return out
    except Exception as e:
        out = {"model_id": hf_id, "name": card.get("name", hf_id), "engine": engine,
                "elapsed_s": round(time.time() - t0, 2), "blocks": [], "error": str(e)[:500]}
        notify("done", result=out)
        return out
    finally:
        if tmp_cfg:
            os.unlink(tmp_cfg)


@app.post("/api/run")
async def run(req: RunRequest):
    if not req.model_ids:
        raise HTTPException(400, "Select at least one model")
    raw = req.image
    if "," in raw and raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        data = base64.b64decode(raw)
    except Exception:
        raise HTTPException(400, "image must be base64-encoded")

    settings = store.load_settings()
    device = devices.resolve(req.device or settings["device_preference"])
    found = {c["hf_id"]: c for c in store.load_catalog()}

    source = store.RESULTS_DIR / f"src_{uuid.uuid4().hex}"
    is_pdf = data[:5] == b"%PDF-"
    if is_pdf:
        src = source.with_suffix(".pdf")
        src.write_bytes(data)
        pages = pdf_to_pages(str(src))
    else:
        src = source.with_suffix(".img")
        src.write_bytes(data)
        pages = [str(src)]

    def cfg_for(hf_id: str) -> dict:
        c = found.get(hf_id, {})
        return {
            "trust_remote_code": c.get("trust_remote_code", False),
            "prompt": (
                "Convert this page to docling."
                if hf_id.startswith("ibm-granite/granite-docling")
                else c.get("prompt", "Perform OCR on this image. Return the extracted text.")
            ),
        }

    pool = ThreadPoolExecutor(max_workers=max(settings.get("parallel_limit", 4), 1))
    loop = asyncio.get_running_loop()
    run_id = f"r{uuid.uuid4().hex[:8]}"
    page_results = []
    for idx, page_path in enumerate(pages):
        ctx = {"run_id": run_id, "page": idx + 1, "image": str(page_path)}
        tasks = {
            hf_id: loop.run_in_executor(pool, _run_one, hf_id, str(page_path), device,
                                        cfg_for(hf_id), ctx)
            for hf_id in req.model_ids
        }
        results = []
        for hf_id, fut in tasks.items():
            try:
                results.append(await fut)
            except Exception as e:
                results.append({"model_id": hf_id, "name": hf_id, "error": str(e)[:400], "blocks": [], "elapsed_s": None})
        page_results.append({
            "page": idx + 1,
            "image": str(page_path),
            "results": results,
        })
    return {"run_id": run_id, "device": device, "is_pdf": is_pdf, "n_pages": len(pages), "pages": page_results}


class RetryRequest(BaseModel):
    """Re-run specific models on one already-uploaded page (no re-upload)."""
    image_file: str   # filename in data/results, e.g. page_abc123.png
    model_ids: list[str]
    device: str | None = None
    page: int | None = None
    run_id: str | None = None


@app.post("/api/run/single")
async def run_single(req: RetryRequest):
    page_path = store.RESULTS_DIR / req.image_file
    if not page_path.exists() or ".." in req.image_file or "/" in req.image_file:
        raise HTTPException(400, "Unknown page image — re-upload the document")
    cfg = store.load_settings()
    device = devices.resolve(req.device or cfg["device_preference"])
    found = {c["hf_id"]: c for c in store.load_catalog()}

    def cfg_for(hf_id: str) -> dict:
        c = found.get(hf_id, {})
        return {
            "trust_remote_code": c.get("trust_remote_code", False),
            "prompt": (
                "Convert this page to docling."
                if hf_id.startswith("ibm-granite/granite-docling")
                else c.get("prompt", "Perform OCR on this image. Return the extracted text.")
            ),
        }

    loop = asyncio.get_running_loop()
    run_id = req.run_id or f"r{uuid.uuid4().hex[:8]}"
    page_no = req.page or 1
    ctx = {"run_id": run_id, "page": page_no, "image": str(page_path)}
    tasks = {
        hf_id: loop.run_in_executor(GLOBAL_EXECUTOR, _run_one, hf_id, str(page_path), device,
                                    cfg_for(hf_id), ctx)
        for hf_id in req.model_ids
    }
    results = []
    for hf_id, fut in tasks.items():
        try:
            results.append(await fut)
        except Exception as e:
            results.append({"model_id": hf_id, "name": hf_id, "error": str(e)[:400], "blocks": [], "elapsed_s": None})
    return {"run_id": run_id, "device": device, "is_pdf": False, "n_pages": 1,
            "pages": [{"page": page_no, "image": req.image_file, "results": results}]}


# --------------------------------------------------------------------------- #
# WS: download + run progress
# --------------------------------------------------------------------------- #
@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    queue = asyncio.Queue()
    hf.downloads.subscribers.add(queue)
    try:
        while True:
            evt = await queue.get()
            await sock.send_json(evt)
    except WebSocketDisconnect:
        pass
    finally:
        hf.downloads.subscribers.discard(queue)


@app.get("/api/health")
def health():
    return {"ok": True}


# Serve page images + built frontend
app.mount("/api/results", StaticFiles(directory=str(store.RESULTS_DIR)), name="pages")
FRONTEND_DIST = store.DATA_DIR / "frontend_dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="static")
