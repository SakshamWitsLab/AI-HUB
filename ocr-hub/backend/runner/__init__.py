"""Engine adapters. Each runner takes a local image and returns blocks:
{text, confidence?, box:[x0,y0,x1,y1], model_id, engine}.

Engines are imported lazily so the hub can serve catalog/downloads without
ML deps installed.
"""
from __future__ import annotations

import io
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None


def _load_image(path_or_bytes):
    if Image is None:
        raise RuntimeError("Pillow not installed — cannot load images")
    img = Image.open(io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes)
    return img


# --------------------------------------------------------------------------- #
# Tesseract
# --------------------------------------------------------------------------- #
import shutil
import subprocess


def run_tesseract(img_path: str, device: str = "cpu", lang: str = "eng") -> list:
    import pytesseract

    img = _load_image(img_path)
    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
    out = []
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt or int(data["conf"][i]) < 0:
            continue
        out.append(
            {
                "text": txt,
                "confidence": float(data["conf"][i]) / 100.0,
                "box": [
                    data["left"][i],
                    data["top"][i],
                    data["left"][i] + data["width"][i],
                    data["top"][i] + data["height"][i],
                ],
                "engine": "tesseract",
            }
        )
    return out


# --------------------------------------------------------------------------- #
# EasyOCR
# --------------------------------------------------------------------------- #
_easyocr_reader_cache: dict = {}


def run_easyocr(img_path: str, device: str = "cpu", lang: str = "en") -> list:
    import easyocr

    from .. import store

    cache = str(store.MODELS_DIR / "easyocr")
    import os

    os.makedirs(cache, exist_ok=True)
    key = (lang, device)
    if key not in _easyocr_reader_cache:
        # quantize=False: torch 2.x's dynamic quantized ops crash with
        # torch/issue#184982 on "template"-built images; fp32 inference is fine
        _easyocr_reader_cache[key] = easyocr.Reader(
            lang.split("+"),
            gpu=(device == "cuda"),
            model_storage_directory=cache,
            quantize=False,
        )
    reader = _easyocr_reader_cache[key]
    out = []
    for box, txt, conf in reader.readtext(img_path):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        out.append(
            {
                "text": txt,
                "confidence": round(float(conf), 3),
                "box": [min(xs), min(ys), max(xs), max(ys)],
                "engine": "easyocr",
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Transformers-family generic runner
# --------------------------------------------------------------------------- #
_tl_cache: dict = {}
_MPS_LOCK = __import__("threading").Lock()  # concurrent torch.generate on MPS crashes Metal encoders


class _NoLock:
    def __enter__(self): pass
    def __exit__(self, *a): pass
    async def __aenter__(self): pass
    async def __aexit__(self, *a): pass


def _load_tl(hf_id: str, path: str, device: str, trust_remote_code: bool = False):
    """Load tokenizer + image processor + model.

    Returns (tok, img_proc, model, proc).
    - Text-only repos (e.g. TrOCR): AutoProcessor returns the tokenizer; the
      image processor is loaded separately.
    - VL repos (e.g. GOT/Florence/Qwen/docling): AutoProcessor handles images+text.
    Callers should feed images through `proc`, never through a bare tokenizer
    (PreTrainedTokenizerFast rejects `images=...`).
    """
    key = hf_id
    if key in _tl_cache:
        return _tl_cache[key]
    import torch
    from transformers import AutoImageProcessor, AutoProcessor

    try:
        from transformers import AutoModelForVision2Seq as _VisionCls
    except ImportError:  # transformers >= 5: class renamed
        from transformers import AutoModelForImageTextToText as _VisionCls

    kwargs = {"trust_remote_code": trust_remote_code}
    proc = AutoProcessor.from_pretrained(path, **kwargs)
    if hasattr(proc, "images"):
        # combined processor (tokenizer + image)
        tok = getattr(proc, "tokenizer", None) or proc
        imgp = getattr(proc, "image_processor", None)
    else:
        # this proc is (probably) a tokenizer-only artifact
        from transformers import AutoTokenizer

        tok = proc if _is_tokenizer(proc) else AutoTokenizer.from_pretrained(path, **kwargs)
        try:
            imgp = AutoImageProcessor.from_pretrained(path, **kwargs)
        except Exception:
            imgp = None
    model = _VisionCls.from_pretrained(
        path,
        # "auto" yields bf16 which breaks on MPS; force fp32 on CPU/MPS
        torch_dtype=torch.float32 if device in ("cpu", "mps") else "auto",
        **kwargs,
    ).to(device)
    model.eval()
    _tl_cache[key] = (tok, imgp, model, proc)
    return _tl_cache[key]


def _is_tokenizer(obj) -> bool:
    return "Tokenizer" in type(obj).__name__


def _boxes_from(img, model_out, image_size) -> list:  # optional decode-level boxes
    return []


def run_transformers(img_path: str, device: str = "cpu", config: dict | None = None) -> list:
    import torch

    config = config or {}
    hf_id = config["hf_id"]
    path = config["local_path"]
    trust = bool(config.get("trust_remote_code"))

    tok, imgp, model, proc = _load_tl(hf_id, path, device, trust)
    img = _load_image(img_path).convert("RGB")

    init = model.config.architectures[0] if getattr(model.config, "architectures", None) else ""
    lock = _MPS_LOCK if device == "mps" else _NoLock()

    if "VisionEncoderDecoder" in init or model.__class__.__name__ == "VisionEncoderDecoderModel":
        # TrOCR-style: image-only input, decoder generates text
        px = imgp(images=img, return_tensors="pt").to(device)
        with torch.no_grad(), lock:
            gen = model.generate(**px, max_new_tokens=int(config.get("max_new_tokens", 256)))
        text = tok.batch_decode(gen, skip_special_tokens=True)[0]
        src = "vision-encoder-decoder"
    else:
        prompt = config.get("prompt", "Perform OCR on this image. Return the extracted text.")
        if _is_tokenizer(proc):
            # proc is the tokenizer itself (text-only repo) — build inputs
            # from the separate pieces (or fail loudly instead of sending
            # `images=` to PreTrainedTokenizerFast, which crashes the runner)
            if imgp is None:
                raise RuntimeError(
                    f"{hf_id}: no image processor available — cannot feed images to this model")
            ids = tok(prompt, return_tensors="pt")
            px = imgp(images=img, return_tensors="pt")
            inputs = {**ids, **px}
        elif hasattr(proc, "apply_chat_template"):
            # VLM with a chat template (image placeholders are inserted by
            # the template, keeping text/image token counts in balance)
            messages = [{"role": "user",
                         "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            chatted = proc.apply_chat_template(messages, add_generation_prompt=True)
            inputs = proc(text=chatted, images=[img], return_tensors="pt")
        else:
            # combined processor (tokenizer + image processor)
            inputs = proc(images=img, text=prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad(), lock:
            gen = model.generate(
                **inputs,
                max_new_tokens=int(config.get("max_new_tokens", 1024)),
                do_sample=False,
            )
        trimmed = gen[:, inputs["input_ids"].shape[1]:] if gen.shape[1] > inputs["input_ids"].shape[1] else gen
        text = tok.batch_decode(trimmed, skip_special_tokens=True)[0]
        src = "vision2seq"

    full_img_w, full_img_h = img.size
    return [
        {
            "text": text.strip() or "(no text detected)",
            "confidence": None,
            "box": [0, 0, full_img_w, full_img_h],
            "engine": f"transformers:{hf_id}",
            "model_id": hf_id,
            "detail": src,
        }
    ]


# --------------------------------------------------------------------------- #
# Docling runner (granite-docling) — generates DocTags, exports clean text
# --------------------------------------------------------------------------- #
def _doctags_to_text(doctags: str) -> str | None:
    """Convert a DocTags string into readable text using the model authors'
    reference library (docling-core). Returns None on any failure."""
    try:
        from docling_core.types.doc import DocTagsDocument, DoclingDocument
        dt = DocTagsDocument.from_doctags_and_image_pairs([doctags], [None])
        doc = DoclingDocument.load_from_doctags(dt, document_name="page")
        try:
            out = doc.export_to_text()
        except Exception:
            out = doc.export_to_markdown()
        return (out or "").strip() or None
    except Exception:
        return None


def _strip_doctags_fallback(doctags: str) -> str | None:
    """Last-resort: drop structural tags, keep inner text nodes."""
    import re
    txt = re.sub(r"<[^>]+>", "\n", doctags)
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    return "\n".join(lines) or None


def run_docling(img_path: str, device: str = "cpu", config: dict | None = None) -> list:
    """granite-docling-258M: generates DocTags, converts to readable text.
    Image long side is capped (config resize_longside_px, default 1024) so the
    vision encoder sees a sane token count instead of the 2x PDF raster."""
    tok, imgp, model, proc = _load_tl(
        config.get("hf_id") or "ibm-granite/granite-docling-258M",
        config.get("local_path"), device,
        bool(config.get("trust_remote_code", False)),
    )
    img = _load_image(img_path)
    longside = int(config.get("resize_longside_px", 1024))
    if max(img.size) > longside:
        scale = longside / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)))

    prompt = config.get("prompt", "Convert this page to docling.")
    # granite-docling is a VLM whose processor expands images into
    # <omni_img_token> placeholders inside the text — passing raw
    # text/images unbalanced raises "The number of images in the text [0]
    # and images [1] should be the same". Build via the chat template so the
    # image token is inserted for us (per the model card's quickstart).
    src = "processor-direct"
    if hasattr(proc, "apply_chat_template") and not _is_tokenizer(proc):
        src = "chat-template"
        try:
            messages = [{"role": "user",
                         "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = proc.apply_chat_template(messages, add_generation_prompt=True)
            inputs = proc(text=text, images=[img], return_tensors="pt")
        except (ValueError, TypeError) as e:
            # some processor versions strip image tokens from styled text —
            # fall back to the direct processor call and fail loudly if that
            # still yields the count mismatch
            try:
                inputs = proc(images=img, text=prompt, return_tensors="pt")
                src = "processor-direct"
            except Exception:
                raise RuntimeError(
                    f"granite-docling processor cannot balance image/text tokens "
                    f"(this container's transformers '{__import__('transformers').__version__}' "
                    f"with processor {type(proc).__name__}); last error: {e}") from e
    else:
        if _is_tokenizer(proc):
            if imgp is None:
                raise RuntimeError("granite-docling: no image processor available")
            inputs = {**tok(prompt, return_tensors="pt"),
                      **imgp(images=img, return_tensors="pt")}
        else:
            inputs = proc(images=img, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    import torch

    lock = _MPS_LOCK if device == "mps" else _NoLock()
    with torch.no_grad(), lock:
        gen = model.generate(
            **inputs,
            max_new_tokens=int(config.get("max_new_tokens", 4096)),
            do_sample=False,
            no_repeat_ngram_size=4,
        )
    trimmed = gen[:, inputs["input_ids"].shape[1]:]
    doctags = tok.batch_decode(trimmed, skip_special_tokens=False)[0]
    text = (_doctags_to_text(doctags) or _strip_doctags_fallback(doctags)
            or tok.batch_decode(trimmed, skip_special_tokens=True)[0])
    w, h = img.size
    return [{"text": text, "confidence": None, "box": [0, 0, w, h],
             "engine": "docling", "model_id": config.get("hf_id"),
             "detail": f"doctags ({src})"}]


# --------------------------------------------------------------------------- #
# PaddleOCR(-VL)
# --------------------------------------------------------------------------- #
def run_paddle(img_path: str, device: str = "cpu") -> list:
    try:
        from paddleocr import PaddleOCR  # PaddleX / PaddleOCR-VL pipeline
    except ImportError:
        raise RuntimeError("paddleocr not installed in this runner environment")

    ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
    result = ocr.predict(str(img_path))
    out = []
    for res in result:
        for txt, score, box in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"] or res["rec_boxes"]):
            xs = [p[0] for p in box] if isinstance(box[0], (list, tuple)) else [box[0], box[2]]
            ys = [p[1] for p in box] if isinstance(box[0], (list, tuple)) else [box[1], box[3]]
            out.append(
                {
                    "text": txt,
                    "confidence": round(float(score), 3),
                    "box": [min(xs), min(ys), max(xs), max(ys)],
                    "engine": "paddleocr",
                }
            )
    return out


# --------------------------------------------------------------------------- #
# NVIDIA Nemotron OCR (nvidia-native) — official nemotron-ocr package
# --------------------------------------------------------------------------- #
_nv_pipeline_cache: dict = {}


def run_nvidia(img_path: str, device: str = "cpu", config: dict | None = None) -> list:
    """NVIDIA Nemotron OCR (v1/v2) via nemotron_ocr.inference.pipeline_v2.

    NOT a transformers model: the official package bundles a detector +
    recognizer + relational model with a C++ CUDA extension. Requires
    Linux + CUDA (a GPU runner builds it via `pip install --no-build-isolation .`).
    On a machine without the package or without CUDA we fail with a clear
    message instead of 'Unknown engine'.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Nemotron OCR requires a CUDA GPU + Linux (builds a C++ extension "
            "from the official nemotron-ocr package) — no GPU visible here")

    try:
        from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2
    except ImportError as e:
        raise RuntimeError(
            "nemotron-ocr package not installed in this runner environment. "
            "On the GPU host: clone nvidia/nemotron-ocr-v2 with git-lfs, install "
            "PyTorch (CUDA build) first, then `pip install --no-build-isolation -v .` "
            "inside the repo (Python 3.12, CUDA toolkit + nvcc on PATH).") from e

    config = config or {}
    hf_id = config.get("hf_id") or "nvidia/nemotron-ocr-v2"
    # v1 repo runs the legacy pipeline; otherwise default to v2 multilingual
    # (lang="en" selects the English v2 word-level build).
    lang = config.get("lang") or ("v1" if hf_id.endswith("-v1") else "multi")
    merge_level = config.get("merge_level", "paragraph")

    # If the hub already downloaded the HF repo, run from disk — the weights
    # live in <repo>/v2_multilingual|v2_english (v2) or the repo root (v1).
    # A complete checkpoint dir must contain detector.pth (+charset.txt);
    # anything else falls back to lang-based hub download.
    if lang in ("en", "english"):
        candidates = ("v2_english", "v2_multilingual")
    elif lang in ("v1", "legacy"):
        candidates = ()
    else:
        candidates = ("v2_multilingual", "v2_english")

    def _complete(d: Path) -> bool:
        return (d / "detector.pth").is_file() and (d / "charset.txt").is_file()

    local_root = config.get("local_path") or ""
    model_dir = None
    root = Path(local_root) if local_root else None
    if root is not None and root.is_dir():
        for cand in (root,) + tuple(root / s for s in candidates):
            if _complete(cand):
                model_dir = cand
                break

    if model_dir is not None:
        cache_key = (str(model_dir), merge_level)
        if cache_key not in _nv_pipeline_cache:
            _nv_pipeline_cache[cache_key] = NemotronOCRV2(model_dir=str(model_dir))
        pipeline = _nv_pipeline_cache[cache_key]
    else:
        cache_key = (lang, merge_level)
        if cache_key not in _nv_pipeline_cache:
            _nv_pipeline_cache[cache_key] = NemotronOCRV2(lang=lang)
        pipeline = _nv_pipeline_cache[cache_key]

    preds = pipeline(str(img_path), merge_level=merge_level)
    out = []
    for p in preds:
        if not (p.get("text") or "").strip():
            continue
        out.append({
            "text": p["text"],
            "confidence": round(float(p.get("confidence", 0.0)), 3),
            "box": [p["left"], p["upper"], p["right"], p["lower"]],
            "engine": "nvidia-native",
            "model_id": hf_id,
        })
    if not out:
        out.append({
            "text": "(no text detected)", "confidence": None,
            "box": list(_load_image(img_path).size),
            "engine": "nvidia-native", "model_id": hf_id,
        })
    return out


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def run_rapidocr(img_path: str, device: str = "cpu") -> list:
    """RapidOCR: PP-OCR det+rec as ONNX — pure CPU, no torch, fast."""
    from rapidocr_onnxruntime import RapidOCR

    key = "rapidocr"
    if key not in _rapidocr_cache:
        _rapidocr_cache[key] = RapidOCR()
    engine = _rapidocr_cache[key]
    result, _elapsed = engine(img_path)
    out = []
    for item in result or []:
        if isinstance(item, dict):  # newer API
            txt = item.get("text") or item.get("txt") or ""
            box = item.get("box")
            conf = item.get("score")
        else:  # classic [box, text, score]
            box, txt, conf = item[0], item[1], item[2]
        txt = (txt or "").strip()
        if not txt:
            continue
        try:
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            b = [min(xs), min(ys), max(xs), max(ys)]
        except Exception:
            b = None
        out.append({
            "text": txt,
            "confidence": round(float(conf), 3) if conf is not None else None,
            "box": b,
            "engine": "rapidocr",
        })
    return out


_rapidocr_cache: dict = {}


def dispatch(engine: str, img_path: str, device: str = "cpu", config: dict | None = None) -> list:
    if engine == "tesseract":
        return run_tesseract(img_path, device, lang=(config or {}).get("lang", "eng"))
    if engine == "easyocr":
        return run_easyocr(img_path, device, lang=(config or {}).get("lang", "en"))
    if engine == "rapidocr":
        return run_rapidocr(img_path, device)
    if engine == "paddleocr":
        return run_paddle(img_path, device)
    if engine == "docling":
        cfg = (config or {}).copy()
        cfg["hf_id"] = cfg.get("hf_id") or "ibm-granite/granite-docling-258M"
        return run_docling(img_path, device, cfg)
    if engine == "nvidia-native":
        return run_nvidia(img_path, device, config)
    if engine in ("transformers", "auto"):
        return run_transformers(img_path, device, config)
    raise ValueError(f"Unknown engine '{engine}'")
