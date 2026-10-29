# OCR Hub — one image for CPU & GPU.
#   docker compose --profile cpu up          (any machine, CPU)
#   docker compose --profile gpu up          (NVIDIA GPU host, Linux)
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        fastapi "uvicorn[standard]" websockets huggingface_hub requests pytesseract pillow \
        pymupdf "transformers<5" sentencepiece protobuf tiktoken simpleitk docling-core \
    && pip install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX} \
    && pip install --no-cache-dir easyocr

COPY backend/ /app/backend/

ENV OCR_HUB_DATA=/data PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
