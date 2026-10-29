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

    Returns (tok, img_proc, model).
    - Text-only repos (e.g. TrOCR): AutoProcessor returns the tokenizer; the
      image processor is loaded separately.
    - VL repos (e.g. GOT/Florence/Qwen): AutoProcessor handles images+text.
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
    _tl_cache[key] = (tok, imgp, model)
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

    tok, imgp, model = _load_tl(hf_id, path, device, trust)
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
        if imgp is not None and not hasattr(imgp, "text"):  # we have separate pieces
            ids = tok(prompt, return_tensors="pt")
            px = imgp(images=img, return_tensors="pt")
            inputs = {**ids, **px}
            inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(device) for k, v in tok(images=img, text=prompt, return_tensors="pt").items()}
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
    tok, imgp, model = _load_tl(
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
    inputs = {k: v.to(device) for k, v in tok(images=img, text=prompt, return_tensors="pt").items()}
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
             "engine": "docling", "model_id": config.get("hf_id"), "detail": "doctags"}]


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
# Dispatch
# --------------------------------------------------------------------------- #
def dispatch(engine: str, img_path: str, device: str = "cpu", config: dict | None = None) -> list:
    if engine == "tesseract":
        return run_tesseract(img_path, device, lang=(config or {}).get("lang", "eng"))
    if engine == "easyocr":
        return run_easyocr(img_path, device, lang=(config or {}).get("lang", "en"))
    if engine == "paddleocr":
        return run_paddle(img_path, device)
    if engine == "docling":
        cfg = (config or {}).copy()
        cfg["hf_id"] = cfg.get("hf_id") or "ibm-granite/granite-docling-258M"
        return run_docling(img_path, device, cfg)
    if engine in ("transformers", "docling", "auto"):
        return run_transformers(img_path, device, config)
    raise ValueError(f"Unknown engine '{engine}'")
