# AI service (FastAPI + YOLO) untuk deployment di Render / Railway / VPS.
FROM python:3.11-slim

# Cegah pyc & buffering log.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Library sistem yang dibutuhkan opencv-headless & ultralytics.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependency dulu (biar layer cache-nya efektif).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Salin semua modul Python yang dipakai app + model.
# app.py meng-import vidToFrame.py dan ocr.py, jadi ketiganya wajib ikut.
COPY app.py vidToFrame.py ocr.py ./
COPY model ./model

# PORT disuntikkan oleh platform lewat env:
#   - Hugging Face Spaces: 7860 (default di sini)
#   - Render: menyuntikkan PORT-nya sendiri (otomatis meng-override)
ENV PORT=7860
EXPOSE 7860

# Jalankan uvicorn. SHELL FORM (tanpa []) supaya ${PORT} ter-expand oleh shell.
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
