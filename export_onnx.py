"""
Export model YOLO (best.pt) -> ONNX untuk inference di browser (ONNX Runtime Web).

PENTING:
    - TIDAK melakukan retrain. Hanya mengekspor model yang sudah ada.
    - best.pt tetap menjadi master/sumber untuk retraining & regenerasi ONNX.
    - Class custom PKL ('gerobak') otomatis ikut terbawa di metadata ONNX.

Cara pakai:
    .\.venv\Scripts\Activate.ps1
    python export_onnx.py

Hasil:
    model/best.onnx

Catatan setting ONNX untuk browser:
    - opset=12  : didukung baik oleh ONNX Runtime Web (WASM & WebGPU).
    - dynamic=False, dengan imgsz tetap 640x640 : shape input statis,
      lebih cepat & lebih kompatibel untuk browser.
    - nms=False : NMS TIDAK dibungkus ke dalam ONNX. Post-processing (termasuk
      NMS) dilakukan di frontend. Ini membuat model lebih portabel & output-nya
      standar YOLO (lihat MODEL_CONTRACT.md). Kalau mau NMS di dalam graph,
      set NMS_IN_MODEL=True di bawah (tapi ort-web perlu dukungan op NMS).
    - simplify=True : sederhanakan graph (butuh paket 'onnxslim'/'onnxsim' kalau
      tersedia; kalau gagal, ultralytics tetap menghasilkan ONNX tanpa simplify).
"""

import os
import shutil
from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PT = BASE_DIR / "model" / "best.pt"
MODEL_ONNX = BASE_DIR / "model" / "best.onnx"

IMG_SIZE = 640
OPSET = 12
NMS_IN_MODEL = False  # NMS dilakukan di frontend (lihat MODEL_CONTRACT.md)


def main() -> None:
    if not MODEL_PT.exists():
        raise FileNotFoundError(
            f"Tidak menemukan {MODEL_PT}. Pastikan best.pt ada di folder model/."
        )

    print(f"[export] Memuat model: {MODEL_PT}")
    model = YOLO(str(MODEL_PT))
    print(f"[export] task={model.task}  classes={model.names}")

    print(f"[export] Mengekspor ke ONNX (imgsz={IMG_SIZE}, opset={OPSET}, nms={NMS_IN_MODEL})...")
    exported = model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        opset=OPSET,
        dynamic=False,
        simplify=True,
        nms=NMS_IN_MODEL,
    )

    # ultralytics menaruh best.onnx di sebelah best.pt (yaitu model/best.onnx).
    exported_path = Path(exported)
    if exported_path.resolve() != MODEL_ONNX.resolve():
        shutil.move(str(exported_path), str(MODEL_ONNX))

    size_mb = MODEL_ONNX.stat().st_size / (1024 * 1024)
    print(f"[export] Selesai -> {MODEL_ONNX} ({size_mb:.2f} MB)")
    print("[export] Lanjutkan dengan verifikasi:  python verify_onnx.py")


if __name__ == "__main__":
    main()
