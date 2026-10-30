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
        j = self.jobs.get(repo)
        if j and j.get("status") == "running":
            return {"ok": True, "already_running": True}
        if j and j.get("status") == "paused":
            # resume from existing .part files
            j["status"] = "queued"
            j["pause_requested"] = False
            self._emit(repo, status="running", progress=j.get("progress", 0))
            threading.Thread(target=self._run, args=(repo,), daemon=True).start()
            return {"ok": True, "resumed": True}
        self.jobs[repo] = {"status": "queued", "progress": 0}
        threading.Thread(target=self._run, args=(repo,), daemon=True).start()
        return {"ok": True}

    def pause(self, repo: str) -> dict:
        j = self.jobs.get(repo)
        if not j or j.get("status") not in ("running", "queued"):
            return {"ok": False, "error": f"not running (status: {j and j.get('status')})"}
        j["pause_requested"] = True
        self._emit(repo, status="pausing", progress=j.get("progress", 0))
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

    DOWNLOAD_EXTS = (".safetensors", ".bin", ".paddle", ".pdiparams", ".pth", ".onnx",
                     ".json", ".txt", ".model", ".tiktoken", ".md", ".jinja")
    DOWNLOAD_PREFIXES = (
        "tokenizer*", "vocab*", "merges*", "special_tokens*", "added_tokens*",
        "preprocessor*", "chat_template*", "generation_config*", "inference*",
    )

    def _stream_file(self, repo: str, url: str, dest: Path,
                     base: int = 0, file_size: int = 0, expected: int = 0) -> bool:
        """Stream one HF file to `dest`, honoring pause. Returns True when the
        file is fully present on disk — resuming from `dest.part` (HTTP Range).
        Emits a WS progress event whenever the whole-percent changes, so multi-GB
        weight files show a moving bar instead of holding at 0% until done."""
        import requests

        part = Path(str(dest) + ".part")
        headers = {"Authorization": f"Bearer {store.hf_token()}"} if store.hf_token() else {}
        offset = part.stat().st_size if part.exists() else 0
        try:
            head = requests.head(url, headers=headers, timeout=30, allow_redirects=True)
            total = int(head.headers.get("content-length", 0)) or (file_size or None)
        except Exception:
            total = file_size or None
        if offset and total and offset >= total:
            part.rename(dest)
            return True
        if offset:
            headers["Range"] = f"bytes={offset}-"

        r = requests.get(url, headers=headers, stream=True, timeout=(15, 60), allow_redirects=True)
        r.raise_for_status()
        if r.status_code == 200 and offset:  # server ignored the Range — restart file
            offset = 0
        with open(part, "ab" if offset else "wb") as f:
            written = offset
            for chunk in r.iter_content(chunk_size=1 << 20):
                if self.jobs.get(repo, {}).get("pause_requested"):
                    return False
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
                    if expected:
                        pct = min(int((base + written) / expected * 95), 95)
                        cur = self.jobs[repo].get("progress", 0)
                        if pct != cur:
                            self.jobs[repo]["progress"] = pct
                            self._emit(repo, progress=pct)
        part.rename(dest)
        return True

    def _run(self, repo: str):
        self._emit(repo, status="running", progress=0)
        dest = store.model_path(repo)
        try:
            dest.mkdir(parents=True, exist_ok=True)
            expected = 0
            files = []
            try:
                import fnmatch
                meta = fetch_metadata(repo)
                wanted = set()
                for pat in self.DOWNLOAD_PREFIXES:
                    for f in meta["files"]:
                        if fnmatch.fnmatch(f["rfilename"], pat):
                            wanted.add(f["rfilename"])
                for f in meta["files"]:
                    name = f["rfilename"]
                    if name.endswith(self.DOWNLOAD_EXTS) or name in wanted:
                        files.append((name, int(f["size"] or 0)))
                expected = sum(sz for _, sz in files)
            except Exception:
                files = []
            if not files:  # fallback: snapshot as before via hub library
                from huggingface_hub import snapshot_download
                patterns = [
                    "*" + e for e in (".safetensors", ".bin", ".paddle", ".pdiparams", ".pth", ".onnx",
                                      ".json", ".txt", ".model", ".tiktoken", ".md", ".jinja")
                ] + list(self.DOWNLOAD_PREFIXES)
                snapshot_download(
                    repo_id=repo, local_dir=str(dest), allow_patterns=patterns,
                    token=store.hf_token() or None, max_workers=4,
                )
            else:
                downloaded = 0  # counted per file (complete + resumed offset)
                for name, sz in files:
                    if self.jobs.get(repo, {}).get("pause_requested"):
                        break
                    target = dest / name
                    # keep HF's relative path structure
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        url = f"https://huggingface.co/{repo}/resolve/main/{name}"
                        ok = self._stream_file(repo, url, target, base=downloaded,
                                               file_size=sz, expected=expected)
                        if not ok:
                            break  # paused mid-file
                        downloaded += (target.stat().st_size)
                        if expected:
                            pct = min(int(downloaded / expected * 95), 95)
                            if pct != self.jobs[repo].get("progress", 0):
                                self._emit(repo, progress=pct)
                    else:
                        downloaded += target.stat().st_size  # already on disk (resumed)

                if self.jobs.get(repo, {}).get("pause_requested"):
                    j = self.jobs[repo]
                    j["status"] = "paused"
                    j.pop("pause_requested", None)
                    self._emit(repo, status="paused", progress=j.get("progress", 0))
                    return

            weight_ok = any(
                p.suffix in (".safetensors", ".bin", ".paddle", ".pdiparams", ".pth", ".onnx")
                for p in dest.rglob("*") if p.is_file()
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


def oxr_parse(img_path: str, max_retries: int = 2) -> list:
    """Parse one page via OXR's free API (POST {OXR_API}/parse/sync).
    Returns blocks in our standard shape. The user's file is uploaded to
    SoMark's servers — gated by the 'Enable cloud parsing' setting.

    The free endpoint allows one task at a time; on HTTP 429 (code 1133,
    "task is already in progress") we retry with backoff instead of failing
    the page."""
    import time as _time
    import requests

    fname = Path(img_path).name
    is_pdf = fname.endswith(".pdf")
    mime = "application/pdf" if is_pdf else "image/png"
    fname = fname if fname.lower().endswith((".pdf", ".png", ".jpg", ".jpeg", ".webp")) else ("page.pdf" if is_pdf else "page.png")

    attempt, delay, last_err = 0, 2.0, None
    while attempt <= max_retries:
        attempt += 1
        try:
            with open(img_path, "rb") as f:
                r = requests.post(
                    f"{OXR_API}/parse/sync",
                    files={"file": (fname, f, mime)},
                    data={"api_key": "dummy", "output_formats": "markdown", "output_formats": "json"},
                    timeout=900,
                )
            if r.status_code == 200:
                j = r.json()
                break
            if r.status_code == 429:
                # honor Retry-After when the server provides one
                wait = r.headers.get("Retry-After")
                delay = max(delay, float(wait)) if wait and wait.isdigit() else delay
                last_err = f"HTTP 429: {r.text[:200]}"
            else:
                raise ValueError(f"HTTP {r.status_code}: {r.text[:200]}")
        except ValueError:
            raise
        except Exception as e:
            last_err = str(e)[:200]
        if attempt <= max_retries:
            _time.sleep(delay)
            delay = min(delay * 2, 15.0)
    else:
        raise ValueError(f"OXR rate limited after {max_retries + 1} attempts — {last_err}")
    if attempt > max_retries and r is not None and r.status_code != 200:
        raise ValueError(f"OXR rate limited: {last_err}")

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
