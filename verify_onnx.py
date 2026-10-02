"""
Verifikasi best.onnx vs best.pt pada gambar yang SAMA.

Tujuan: memastikan ONNX menghasilkan deteksi yang masuk akal dibanding model
asli (.pt). Export sukses TIDAK menjamin inference benar, jadi kita bandingkan.

Yang dilakukan:
    1. Jalankan best.pt lewat ultralytics (ground truth).      [butuh torch]
    2. Jalankan best.onnx lewat onnxruntime + numpy (pure).    [TANPA torch]
       - preprocessing & postprocessing (NMS) persis seperti yang akan
         diimplementasikan di frontend (lihat MODEL_CONTRACT.md).
    3. Cetak class + confidence + bbox dari kedua jalur, lalu bandingkan.

Cara pakai:
    python verify_onnx.py [path_gambar]

Kalau torch diblokir di environment ini, jalur .pt otomatis dilewati dan
hanya jalur ONNX yang diuji (tetap berguna: membuktikan ONNX bisa di-infer).
"""

import sys
import glob
from pathlib import Path

import numpy as np
import cv2
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parent
MODEL_PT = BASE_DIR / "model" / "best.pt"
MODEL_ONNX = BASE_DIR / "model" / "best.onnx"

IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45
CLASS_NAMES = {0: "gerobak"}  # dari dataset/data.yaml (PKL)


# ---------------------------------------------------------------------------
# Preprocessing: letterbox 640x640, BGR->RGB, /255, HWC->CHW, batch dim.
# (Identik dengan yang harus dilakukan frontend.)
# ---------------------------------------------------------------------------
def letterbox(img, new_shape=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw /= 2
    dh /= 2
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
    return padded, r, (dw, dh)


def preprocess(img_bgr):
    padded, r, (dw, dh) = letterbox(img_bgr, IMG_SIZE)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None]  # (1,3,640,640)
    return np.ascontiguousarray(tensor), r, (dw, dh)


# ---------------------------------------------------------------------------
# Postprocessing: parse output YOLO + NMS. Mendukung dua layout output:
#   (1, 4+nc, N)  -> transpose dulu (format YOLOv8/YOLO11/YOLO26 tanpa NMS)
#   (1, N, 4+nc)  -> langsung
# ---------------------------------------------------------------------------
def nms(boxes, scores, iou_thres):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return keep


def postprocess(output, r, dwdh, conf_thres=CONF_THRES, iou_thres=IOU_THRES):
    out = output[0]  # buang batch -> bisa (4+nc, N) atau (N, 4+nc)
    if out.shape[0] < out.shape[1]:
        out = out.transpose(1, 0)  # jadikan (N, 4+nc)

    boxes_xywh = out[:, :4]
    cls_scores = out[:, 4:]
    class_ids = cls_scores.argmax(axis=1)
    confidences = cls_scores.max(axis=1)

    mask = confidences >= conf_thres
    boxes_xywh = boxes_xywh[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]
    if len(boxes_xywh) == 0:
        return []

    # xywh (center) -> xyxy, di ruang 640 (letterboxed)
    cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms(boxes, confidences, iou_thres)

    dw, dh = dwdh
    dets = []
    for i in keep:
        bx1 = (boxes[i, 0] - dw) / r
        by1 = (boxes[i, 1] - dh) / r
        bx2 = (boxes[i, 2] - dw) / r
        by2 = (boxes[i, 3] - dh) / r
        dets.append({
            "classId": int(class_ids[i]),
            "className": CLASS_NAMES.get(int(class_ids[i]), str(class_ids[i])),
            "confidence": round(float(confidences[i]), 4),
            "bbox": {"x1": round(float(bx1), 1), "y1": round(float(by1), 1),
                     "x2": round(float(bx2), 1), "y2": round(float(by2), 1)},
        })
    return dets


def run_onnx(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Gagal baca gambar: {img_path}")
    tensor, r, dwdh = preprocess(img)
    sess = ort.InferenceSession(str(MODEL_ONNX), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    out = sess.run(None, {inp: tensor})[0]
    print(f"[onnx] input name='{inp}' shape={tensor.shape}")
    print(f"[onnx] raw output shape={out.shape}")
    return postprocess(out, r, dwdh)


def run_pt(img_path):
    """Jalur .pt (ground truth). Dilewati kalau torch diblokir."""
    try:
        from ultralytics import YOLO
    except Exception as exc:  # noqa: BLE001
        print(f"[pt] DILEWATI (torch/ultralytics tidak bisa dimuat): {exc}")
        return None
    model = YOLO(str(MODEL_PT))
    res = model.predict(source=img_path, conf=CONF_THRES, iou=IOU_THRES, verbose=False)[0]
    dets = []
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        dets.append({
            "classId": int(b.cls[0]),
            "className": res.names.get(int(b.cls[0]), str(int(b.cls[0]))),
            "confidence": round(float(b.conf[0]), 4),
            "bbox": {"x1": round(x1, 1), "y1": round(y1, 1),
                     "x2": round(x2, 1), "y2": round(y2, 1)},
        })
    return dets


def _print(title, dets):
    print(f"\n=== {title} ===")
    if dets is None:
        print("  (tidak dijalankan)")
        return
    if not dets:
        print("  (tidak ada deteksi)")
        return
    for d in dets:
        print(f"  {d['className']} conf={d['confidence']} bbox={d['bbox']}")


def main():
    if not MODEL_ONNX.exists():
        raise FileNotFoundError(
            f"{MODEL_ONNX} belum ada. Jalankan dulu: python export_onnx.py"
        )

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        cand = sorted(glob.glob(str(BASE_DIR / "dataset" / "test" / "images" / "*")))
        if not cand:
            cand = sorted(glob.glob(str(BASE_DIR / "dataset" / "train" / "images" / "*")))
        if not cand:
            raise FileNotFoundError("Tidak ada gambar sample. Beri path: python verify_onnx.py <gambar>")
        img_path = cand[0]

    print(f"[verify] Gambar uji: {img_path}")
    onnx_dets = run_onnx(img_path)
    pt_dets = run_pt(img_path)

    _print("PT (ultralytics, ground truth)", pt_dets)
    _print("ONNX (onnxruntime + numpy, seperti frontend)", onnx_dets)

    print("\n=== RINGKASAN ===")
    print(f"  ONNX mendeteksi {len(onnx_dets)} objek.")
    if pt_dets is not None:
        print(f"  PT   mendeteksi {len(pt_dets)} objek.")
        if len(onnx_dets) == len(pt_dets):
            print("  OK: jumlah deteksi sama.")
        else:
            print("  CATATAN: jumlah beda (bisa karena perbedaan NMS/threshold kecil).")
    else:
        print("  Jalur PT dilewati; verifikasi ONNX saja (inference ONNX berhasil = OK dasar).")


if __name__ == "__main__":
    main()
