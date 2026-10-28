"""HuggingFace glue: metadata preview + background download with progress events."""
import asyncio
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

from . import store as store

HF_API = "https://huggingface.co/api/models/{repo}?blobs=true"
WEIGHT_EXTS = (".safetensors", ".bin", ".paddle", ".pdiparams")


def _auth_headers() -> dict:
    tok = store.hf_token()
    h = {}
    if tok:
        h["authorization"] = f"Bearer {tok}"
    return h


def fetch_metadata(repo: str) -> dict:
    """Card preview straight from HF — size, params, downloads, likes, license..."""
    req = urllib.request.Request(HF_API.format(repo=repo), headers=_auth_headers())
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            j = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = f"HuggingFace API returned {e.code}"
        if e.code == 401:
            detail = "Gated/private repo — make sure the model ID is right, or add a HuggingFace token in Settings"
        elif e.code == 404:
            detail = f"Model '{repo}' not found on HuggingFace"
        raise ValueError(detail) from e

    siblings = j.get("siblings") or []
    total = sum((s.get("size") or 0) for s in siblings)
    params = (j.get("safetensors") or {}).get("total")
    license_tags = [t for t in j.get("tags", []) if t.startswith("license:")]
    return {
        "hf_id": j["id"],
        "name": j["id"].split("/")[-1],
        "pipeline": j.get("pipeline_tag"),
        "size_gb": round(total / 1e9, 2),
        "params": f"{params/1e9:.2f}B" if params else "unknown",
        "license": license_tags[0].split(":", 1)[1].split("-", 1)[0] if license_tags else "unknown",
        "downloads": j.get("downloads", 0),
        "likes": j.get("likes", 0),
        "gated": bool(j.get("gated")),
        "url": f"https://huggingface.co/{j['id']}",
        "weight_files": [s["rfilename"] for s in siblings if s["rfilename"].endswith(WEIGHT_EXTS)],
        "files": [{"rfilename": s.get("rfilename"), "size": s.get("size")} for s in siblings],
    }


# --------------------------------------------------------------------------- #
# Download manager — background thread per repo, directory-size based progress
# --------------------------------------------------------------------------- #
class DownloadManager:
    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.subscribers: set = set()

    async def subscribe(self):
        q: asyncio.Queue = asyncio.Queue()
        self.subscribers.add(q)
        try:
            while True:
                yield await q.get()
        finally:
            self.subscribers.discard(q)

    def _emit(self, repo: str, **kw):
        self.jobs.setdefault(repo, {}).update(kw)
        evt = {"type": "download", "repo": repo, **self.jobs[repo]}
        for q in list(self.subscribers):
            q.put_nowait(evt)

    def status(self) -> list:
        return [{"repo": k, **v} for k, v in self.jobs.items()]

    def notify(self, kind: str, **kw):
        evt = {"type": kind, **kw}
        for q in list(self.subscribers):
            q.put_nowait(evt)

    def run_event(self, **kw):
        return self.notify("run", **kw)

    def start(self, repo: str) -> dict:
        if repo in self.jobs and self.jobs[repo].get("status") == "running":
            return {"ok": True, "already_running": True}
        self.jobs[repo] = {"status": "queued", "progress": 0}
        threading.Thread(target=self._run, args=(repo,), daemon=True).start()
        return {"ok": True}

    def _dir_size(self, p: Path) -> int:
        t = 0
        for root, _dirs, files in os.walk(p):
            for f in files:
                try:
                    t += (Path(root) / f).stat().st_size
                except OSError:
                    pass
        return t

    def _run(self, repo: str):
        self._emit(repo, status="running", progress=0)
        dest = store.model_path(repo)
        try:
            dest.mkdir(parents=True, exist_ok=True)
            # expected total size from metadata (weights+config)
            try:
                meta = fetch_metadata(repo)
                expected = sum(f["size"] or 0 for f in meta["files"])
            except Exception:
                expected = 0

            def poll():
                while self.jobs.get(repo, {}).get("status") == "running":
                    if expected:
                        pct = min(int(self._dir_size(dest) / expected * 95), 95)
                        self._emit(repo, progress=pct)
                    time.sleep(1.5)

            poller = threading.Thread(target=poll, daemon=True)
            poller.start()

            from huggingface_hub import snapshot_download

            patterns = [
                "*" + e for e in (
                    ".safetensors", ".bin", ".paddle", ".pdiparams", ".pth", ".onnx",
                    ".json", ".txt", ".model", ".tiktoken", ".md", ".jinja", ".model",
                )
            ] + ["tokenizer*", "vocab*", "merges*", "special_tokens*", "added_tokens*", "preprocessor*", "chat_template*", "generation_config*", "inference*"]
            snapshot_download(
                repo_id=repo,
                local_dir=str(dest),
                allow_patterns=patterns,
                token=store.hf_token() or None,
                max_workers=4,
            )
            weight_ok = any(
                p.suffix in (".safetensors", ".bin", ".paddle", ".pdiparams", ".pth", ".onnx")
                for p in dest.rglob("*")
            )
            if weight_ok:
                store.mark_installed(repo)
                self._emit(repo, status="done", progress=100)
            else:
                self._emit(repo, status="error", progress=0, error="No weight files found in this repo — nothing runnable here")
        except Exception as e:
            self._emit(repo, status="error", progress=0, error=str(e)[:500])


downloads = DownloadManager()


# --------------------------------------------------------------------------- #
# SoMark OXR — free cloud parse API (optional; no key)
# --------------------------------------------------------------------------- #
OXR_API = "https://free-oxr.somark.ai/api/v1"


def oxr_parse(img_path: str) -> list:
    """Parse one page via OXR's free API (POST {OXR_API}/parse/sync).
    Returns blocks in our standard shape. The user's file is uploaded to
    SoMark's servers — gated by the 'Enable cloud parsing' setting."""
    import requests

    fname = Path(img_path).name
    is_pdf = fname.endswith(".pdf")
    mime = "application/pdf" if is_pdf else "image/png"
    fname = fname if fname.lower().endswith((".pdf", ".png", ".jpg", ".jpeg", ".webp")) else ("page.pdf" if is_pdf else "page.png")
    with open(img_path, "rb") as f:
        r = requests.post(
            f"{OXR_API}/parse/sync",
            files={"file": (fname, f, mime)},
            data={"api_key": "dummy", "output_formats": "markdown", "output_formats": "json"},
            timeout=900,
        )
    if r.status_code != 200:
        raise ValueError(f"HTTP {r.status_code}: {r.text[:200]}")
    j = r.json()

    pages = ((j.get("data") or {}).get("result") or {}).get("outputs") or {}
    blocks = []
    for page in pages.get("json", {}).get("pages", []):
        for b in page.get("blocks", []):
            txt = b.get("text") or b.get("content") or ""
            if isinstance(txt, dict):
                txt = txt.get("text", "")
            blocks.append({
                "text": str(txt),
                "confidence": b.get("conf"),
                "box": b.get("bbox") or b.get("box"),
                "engine": "somark-oxr",
            })
    md = (pages.get("markdown") or "").strip()
    if md and not any(b["text"] for b in blocks):
        blocks.append({"text": md, "confidence": None, "box": None, "engine": "somark-oxr"})
    if not blocks:
        raise ValueError(f"OXR returned no content: {str(j)[:200]}")
    return blocks
