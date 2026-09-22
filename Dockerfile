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

# Salin kode + model. .dockerignore memastikan file besar/tak perlu tidak ikut.
COPY app.py .
COPY model ./model

# PORT disuntikkan oleh platform lewat env:
#   - Hugging Face Spaces: 7860 (default di sini)
#   - Render: menyuntikkan PORT-nya sendiri (otomatis meng-override)

EXPOSE 7860

# Jalankan uvicorn. shell form supaya $PORT ter-expand.
CMD ["uvicorn", "main:app". "--host", "0.0.0.0", "--port", "7680"]
