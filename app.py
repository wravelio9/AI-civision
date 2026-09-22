"""
AI service untuk deteksi objek YOLO.

Alur:
    BE Express upload foto -> POST /predict (service ini) -> proses YOLO
    -> balas JSON deteksi + URL gambar hasil yang sudah digambari kotak.

Jalankan:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

from vidToFrame import annotate_video

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Model dasar (fallback) kalau belum ada hasil training.
FALLBACK_MODEL = BASE_DIR / "yolo26n.pt"


def _find_trained_model() -> Path:
    """
    Cari best.pt hasil training terbaru di runs/detect/*/weights/best.pt.

    Setiap kali `yolo train` dijalankan, hasilnya masuk ke folder baru
    (train, train2, train3, ...). Fungsi ini memilih best.pt yang paling
    baru dibuat, jadi tidak perlu mengubah path manual tiap training ulang.
    Bisa dioverride lewat env MODEL_PATH.
    """
    # 1. Override eksplisit lewat env (dipakai di produksi/Docker kalau perlu).
    override = os.environ.get("MODEL_PATH")
    if override and Path(override).exists():
        return Path(override)

    # 2. Model yang di-commit ke repo (dipakai saat deploy, mis. di Render).
    #    runs/ di-gitignore, jadi best.pt disalin ke model/best.pt agar ikut deploy.
    committed = BASE_DIR / "model" / "best.pt"
    if committed.exists():
        return committed

    # 3. Hasil training lokal terbaru di runs/detect/*/weights/best.pt.
    candidates = list((BASE_DIR / "runs" / "detect").glob("*/weights/best.pt"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    # 4. Fallback ke model dasar.
    return FALLBACK_MODEL


MODEL_PATH = _find_trained_model()

# Folder untuk menyimpan gambar hasil anotasi, lalu disajikan sebagai file statis.
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Ambang confidence default. Bisa dioverride lewat query ?conf=
DEFAULT_CONF = 0.5

# Ekstensi gambar yang diterima.
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}

# Tipe video yang diterima untuk /predict-video.
ALLOWED_VIDEO_TYPES = {
    "video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska",
    "video/webm", "video/avi", "application/octet-stream",
}

# Daftar origin yang boleh mengakses AI service (untuk request dari browser).
# Set lewat env ALLOWED_ORIGINS, dipisah koma, mis:
#   ALLOWED_ORIGINS=http://localhost:3000,https://app.domainmu.com
# Default "*" mengizinkan semua origin (praktis untuk development).
_origins_env = os.environ.get("ALLOWED_ORIGINS", "*").strip()
ALLOWED_ORIGINS = ["*"] if _origins_env == "*" else [
    o.strip() for o in _origins_env.split(",") if o.strip()
]

# ---------------------------------------------------------------------------
# Inisialisasi aplikasi + load model SEKALI saja (bukan per-request)
# ---------------------------------------------------------------------------
app = FastAPI(title="Civision AI Detection Service", version="1.0.0")

# CORS: izinkan BE (origin/port berbeda) memanggil service ini.
# Catatan: server-to-server (mis. Express -> AI via axios) tidak butuh CORS,
# tapi ini diperlukan kalau ada request langsung dari browser/frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sajikan folder outputs supaya BE bisa mengambil gambar hasil lewat URL.
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

print(f"[startup] Loading model: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))
print("[startup] Model loaded. Ready.")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _base_url(request_host: str) -> str:
    """Bangun base URL untuk link gambar hasil."""
    return request_host.rstrip("/")


def _run_detection(image_bytes: bytes, filename: str, conf: float, base_url: str) -> dict:
    """Jalankan YOLO pada satu gambar, simpan hasil anotasi, kembalikan dict."""
    import numpy as np
    import cv2

    # Decode bytes -> array gambar (BGR).
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("File bukan gambar yang valid atau rusak.")

    # Inference. verbose=False biar log tidak berisik.
    results = model.predict(source=img, conf=conf, verbose=False)
    result = results[0]

    # Kumpulkan deteksi jadi list JSON-friendly.
    detections = []
    names = result.names  # {id: label}
    for box in result.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            {
                "label": names.get(cls_id, str(cls_id)),
                "class_id": cls_id,
                "confidence": round(float(box.conf[0]), 4),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                },
            }
        )

    # Simpan gambar hasil yang sudah digambari kotak.
    annotated = result.plot()  # numpy array (BGR) dengan box tergambar
    out_name = f"{uuid.uuid4().hex}.jpg"
    out_path = OUTPUT_DIR / out_name
    cv2.imwrite(str(out_path), annotated)

    return {
        "filename": filename,
        "detections": detections,
        "count": len(detections),
        "annotated_image_url": f"{base_url}/outputs/{out_name}",
    }


def _predict_frame(frame, conf: float):
    """
    Jalankan YOLO pada satu frame video (numpy BGR).

    Dipakai oleh annotate_video di vidToFrame.py. Mengembalikan
    (frame_beranotasi, list_box) supaya video writer bisa menulisnya.
    """
    result = model.predict(source=frame, conf=conf, verbose=False)[0]
    return result.plot(), list(result.boxes)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Cek service hidup + model apa yang dipakai."""
    return {
        "status": "ok",
        "model": MODEL_PATH.name,
        "model_path": str(MODEL_PATH),
        "using_fallback": MODEL_PATH == FALLBACK_MODEL,
    }


@app.post("/predict-image")
async def predict(
    request: Request,
    files: list[UploadFile] = File(..., description="Satu atau banyak file gambar"),
    conf: float = DEFAULT_CONF,
):
    """
    Terima satu atau banyak gambar, jalankan deteksi, balas hasil per gambar.

    Form field: `files` (bisa lebih dari satu).
    Query opsional: `conf` (ambang confidence, default 0.5).
    """
    if not files:
        raise HTTPException(status_code=400, detail="Tidak ada file yang dikirim.")

    # base_url dipakai untuk menyusun link gambar hasil.
    # Prioritas: env PUBLIC_BASE_URL -> kalau tidak, ambil otomatis dari request
    # (jadi di Render/host mana pun URL gambar langsung benar tanpa konfigurasi).
    base_url = os.environ.get("PUBLIC_BASE_URL", "").strip()
    if not base_url:
        base_url = str(request.base_url).rstrip("/")

    results_out = []
    errors = []

    for f in files:
        if f.content_type not in ALLOWED_CONTENT_TYPES:
            errors.append(
                {"filename": f.filename, "error": f"Tipe file tidak didukung: {f.content_type}"}
            )
            continue
        try:
            content = await f.read()
            results_out.append(_run_detection(content, f.filename, conf, base_url))
        except Exception as exc:  # noqa: BLE001 - kembalikan error per file, jangan gagalkan semua
            errors.append({"filename": f.filename, "error": str(exc)})

    return JSONResponse(
        {
            "success": True,
            "conf": conf,
            "results": results_out,
            "errors": errors,
        }
    )


@app.post("/predict-video")
async def predict_video(
    request: Request,
    file: UploadFile = File(..., description="Satu file video"),
    conf: float = DEFAULT_CONF,
):
    """
    Terima 1 video, proses frame-per-frame dengan YOLO, kembalikan video
    beranotasi (kotak deteksi tergambar) + ringkasan deteksi.

    Alur:
        video masuk -> baca tiap frame -> prediksi -> gambar kotak
        -> tulis ke video output -> balas URL video hasil.

    Form field: `file` (satu video).
    Query opsional: `conf` (ambang confidence, default 0.5).
    """
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Tipe file tidak didukung: {file.content_type}. Kirim file video.",
        )

    base_url = os.environ.get("PUBLIC_BASE_URL", "").strip()
    if not base_url:
        base_url = str(request.base_url).rstrip("/")

    # Simpan upload ke file sementara (OpenCV butuh path file, bukan bytes).
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp_in = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_in = tmp.name

        # Video hasil disimpan di outputs/ supaya bisa diakses lewat URL.
        out_name = f"{uuid.uuid4().hex}.mp4"
        out_path = OUTPUT_DIR / out_name

        summary = annotate_video(
            input_path=tmp_in,
            output_path=str(out_path),
            predict_frame=_predict_frame,
            conf=conf,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Gagal memproses video: {exc}")
    finally:
        # Bersihkan file input sementara.
        if tmp_in and os.path.exists(tmp_in):
            try:
                os.remove(tmp_in)
            except OSError:
                pass

    return JSONResponse(
        {
            "success": True,
            "conf": conf,
            "filename": file.filename,
            "summary": summary,
            "annotated_video_url": f"{base_url}/outputs/{out_name}",
        }
    )
