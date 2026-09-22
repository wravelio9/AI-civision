"""
Utilitas pemrosesan video untuk pipeline deteksi YOLO.

Dipakai oleh app.py untuk endpoint /predict-video dengan alur:
    video masuk -> baca frame -> prediksi tiap frame -> gambar kotak
    -> tulis frame ke video output -> kembalikan video hasil.

Ada dua fungsi utama:
    - extract_frames(): pisahkan video jadi file-file frame (kalau memang butuh
      frame terpisah di disk).
    - annotate_video(): proses video frame-per-frame (di memori) langsung jadi
      video beranotasi. Ini jalur efisien yang dipakai service.

Bisa juga dijalankan langsung dari terminal:
    python vidToFrame.py input.mp4 output.mp4
"""

from pathlib import Path
from typing import Callable, Optional

import cv2


def extract_frames(video_path: str, output_dir: str, every_n: int = 1) -> int:
    """
    Pisahkan video menjadi file-file gambar frame.

    Args:
        video_path: path video sumber.
        output_dir: folder tujuan menyimpan frame (dibuat kalau belum ada).
        every_n: simpan 1 frame setiap `every_n` frame (1 = semua frame).

    Returns:
        Jumlah frame yang tersimpan.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Tidak bisa membuka video: {video_path}")

    frame_number = 0
    saved = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_number % every_n == 0:
                cv2.imwrite(str(out / f"frame_{frame_number:06d}.jpg"), frame)
                saved += 1
            frame_number += 1
    finally:
        cap.release()

    return saved


def annotate_video(
    input_path: str,
    output_path: str,
    predict_frame: Callable,
    conf: float = 0.5,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """
    Proses video frame-per-frame menjadi video beranotasi.

    Untuk setiap frame: panggil `predict_frame(frame, conf)` yang harus
    mengembalikan (annotated_frame, detections_list). Frame beranotasi
    ditulis ke video output dengan fps & ukuran sama seperti input.

    Args:
        input_path: path video sumber.
        output_path: path video hasil (disarankan .mp4).
        predict_frame: fungsi (frame_bgr, conf) -> (annotated_bgr, detections).
        conf: ambang confidence yang diteruskan ke predict_frame.
        on_progress: callback opsional (frame_ke, total_frame) untuk log/progress.

    Returns:
        dict ringkasan: total_frames, frames_with_detection, total_detections,
        fps, width, height.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Tidak bisa membuka video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if width == 0 or height == 0:
        cap.release()
        raise ValueError("Dimensi video tidak valid (0). File mungkin rusak.")

    # mp4v: codec MP4 yang tersedia di OpenCV tanpa perlu ffmpeg terpisah.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Gagal membuat VideoWriter untuk output.")

    frames_done = 0
    frames_with_detection = 0
    total_detections = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, detections = predict_frame(frame, conf)
            writer.write(annotated)

            n = len(detections)
            if n > 0:
                frames_with_detection += 1
                total_detections += n

            frames_done += 1
            if on_progress and frames_done % 30 == 0:
                on_progress(frames_done, total)
    finally:
        cap.release()
        writer.release()

    return {
        "total_frames": frames_done,
        "frames_with_detection": frames_with_detection,
        "total_detections": total_detections,
        "fps": round(fps, 2),
        "width": width,
        "height": height,
    }


# ---------------------------------------------------------------------------
# CLI sederhana untuk tes cepat: python vidToFrame.py input.mp4 output.mp4
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from ultralytics import YOLO

    if len(sys.argv) < 3:
        print("Usage: python vidToFrame.py <input_video> <output_video> [model.pt] [conf]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    model_path = sys.argv[3] if len(sys.argv) > 3 else "yolo26n.pt"
    conf_arg = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

    yolo = YOLO(model_path)

    def _predict(frame, conf):
        res = yolo.predict(source=frame, conf=conf, verbose=False)[0]
        return res.plot(), list(res.boxes)

    print(f"Memproses {in_path} -> {out_path} (model={model_path}, conf={conf_arg})")
    summary = annotate_video(
        in_path, out_path, _predict, conf=conf_arg,
        on_progress=lambda d, t: print(f"  {d}/{t} frame"),
    )
    print("Selesai:", summary)
