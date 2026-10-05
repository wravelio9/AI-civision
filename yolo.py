import os
from ultralytics import YOLO
from pathlib import Path

# how to activate yolo.py: 
#   .\.venv\Scripts\Activate.ps1 
#   python yolo.py

BASE_DIR = Path(__file__).resolve().parent
FALLBACK_MODEL = BASE_DIR / "model/yolo26n.pt"

def _find_trained_model() -> Path:
    """
    Cari best.pt hasil training terbaru di runs/detect/*/weights/best.pt.

    Setiap kali `yolo train` dijalankan, hasilnya masuk ke folder baru
    (train, train2, train3, ...). Fungsi ini memilih best.pt yang paling
    baru dibuat, jadi tidak perlu mengubah path manual tiap training ulang.
    Bisa dioverride lewat env MODEL_PATH.
    """
    override = os.environ.get("MODEL_PATH")
    if override and Path(override).exists():
        return Path(override)

    candidates = list((BASE_DIR / "runs" / "detect").glob("*/weights/best.pt"))
    if candidates:
        # Ambil yang paling baru dimodifikasi.
        return max(candidates, key=lambda p: p.stat().st_mtime)

    return FALLBACK_MODEL


MODEL_PATH = _find_trained_model()

# mdl = "runs/detect/train2/weights/best.pt"
# yolo = "yolo26n.pt"
model = YOLO(FALLBACK_MODEL)

model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
)

metrics = model.val(
    data="dataset/data.yaml",
    split="test"
)

# print(f"Val: ${metrics}")

model.predict(
    source="dataset/test/images",
    save=True,
    conf=0.5
)

# model.tune(
#     data="dataset/data.yaml",
#     epochs=30,
#     iterations=50,
# )