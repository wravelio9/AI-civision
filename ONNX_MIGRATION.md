# CiVision — Migrasi AI ke ONNX Runtime Web

Dokumen ini merangkum perubahan arsitektur AI CiVision dari **Python + YOLO + API
server** menjadi **YOLO → ONNX → ONNX Runtime Web → Browser**, sehingga inference
gambar normal **tidak lagi butuh server Python yang terus berjalan**.

---

## 1. Arsitektur baru

```
Frontend
   │  (unduh sekali, file statis)
   ▼
model/best.onnx
   │
   ▼
ONNX Runtime Web  (WASM / WebGPU)
   │
   ▼
Inference YOLO di browser  →  deteksi PKL (gerobak)
   │
   ▼
violation-logic.js  →  keputusan "melanggar / tidak"
```

Model `.pt` tetap jadi **master** untuk training & regenerasi ONNX. Tidak dihapus.

---

## 2. Files CHANGED / ADDED

### Ditambah (baru)
| File | Fungsi |
|------|--------|
| `export_onnx.py` | Export `best.pt` → `best.onnx` (tanpa retrain) |
| `verify_onnx.py` | Verifikasi: bandingkan output `.pt` vs `.onnx` pada gambar sama |
| `requirements-onnx.txt` | Dependency tooling export/verify (`onnx`, `onnxruntime`) |
| `MODEL_CONTRACT.md` | Kontrak input/output model untuk frontend |
| `web/yolo-onnx.js` | Inference YOLO di browser (preprocess + ORT + postprocess/NMS) |
| `web/violation-logic.js` | Logika pelanggaran, TERPISAH dari deteksi |
| `web/index.html` | Demo minimal: load model + deteksi gambar di browser |
| `ONNX_MIGRATION.md` | Dokumen ini |

### Tidak diubah / tetap dipertahankan
| File | Alasan |
|------|--------|
| `model/best.pt` | **Master model**. Dibutuhkan untuk retrain & regenerasi ONNX. |
| `yolo.py` | Script training. Tetap dipakai untuk melatih model. |
| `app.py`, `ocr.py`, `vidToFrame.py` | **Tidak dihapus** (lihat bagian Limitasi). Masih berguna untuk OCR & video yang tidak bisa jalan murni di browser. |
| `dataset/`, `Dockerfile`, `render.yaml` | Aset training & opsi deploy server (opsional) tetap tersedia. |

> **Tidak ada fitur yang dihapus.** Deteksi gambar dipindah ke browser; fungsi yang
> belum bisa dipindah (OCR, video) tetap ada sebagai server opsional.

---

## 3. Langkah menghasilkan `best.onnx`

> Butuh `torch` + `ultralytics` (dari `requirements.txt`) untuk membuka `best.pt`,
> plus `onnx`/`onnxruntime` (dari `requirements-onnx.txt`).

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-onnx.txt    # sekali saja
python export_onnx.py
```
Hasil: `model/best.onnx`.

### Verifikasi (WAJIB — jangan asumsikan export = benar)
```powershell
python verify_onnx.py
# atau dengan gambar tertentu:
python verify_onnx.py dataset/test/images/namafile.jpg
```
Script akan menjalankan `.pt` (ground truth) dan `.onnx` (jalur numpy seperti
frontend) pada gambar yang sama, lalu membandingkan jumlah & isi deteksi.

> **Catatan environment:** di mesin dev ini `torch` sempat diblokir oleh
> *Application Control policy* Windows (WinError 4551 pada `shm.dll`). Selama itu,
> export & jalur `.pt` tidak bisa dijalankan. `verify_onnx.py` otomatis melewati
> jalur `.pt` dan tetap menguji ONNX bila torch tak tersedia. Jalankan di mesin
> tanpa blok tersebut untuk menghasilkan & memverifikasi `best.onnx` sepenuhnya.
> **Logika preprocessing/postprocessing sudah diverifikasi terpisah** (NMS,
> konversi xywh→xyxy, un-letterbox) memakai output sintetis — hasilnya tepat.

---

## 4. Dependency frontend

Hanya satu:
```bash
npm install onnxruntime-web
```
Tidak ada dependency tambahan yang diperlukan untuk inference.

---

## 5. Cara frontend memuat & memakai model

```javascript
import { loadModel, detect } from "./yolo-onnx.js";
import { evaluateViolations } from "./violation-logic.js";

// 1. Muat model sekali (file statis best.onnx)
const session = await loadModel("/models/best.onnx");

// 2. Deteksi dari <img>, <canvas>, atau ImageBitmap
const imgEl = document.querySelector("#foto");
const { detections } = await detect(session, imgEl, {
  confThreshold: 0.25,
  iouThreshold: 0.45,
});
// detections: [{ classId, className, confidence, bbox:{x1,y1,x2,y2} }, ...]

// 3. (Terpisah) terapkan aturan pelanggaran
const rules = [
  { id: "trotoar", label: "Berjualan di trotoar", type: "zone-rect",
    area: { x1: 0, y1: 300, x2: 1280, y2: 720 }, minOverlap: 0.3 },
];
const { violations, summary } = evaluateViolations(detections, rules);
```

Model `best.onnx` cukup ditaruh sebagai **file statis** (mis. folder `public/models/`),
tidak perlu server inference.

---

## 6. Spesifikasi model (ringkas)

Lihat `MODEL_CONTRACT.md` untuk detail lengkap.

- **Input**: `float32` `[1,3,640,640]` NCHW, **RGB**, dinormalisasi `/255`, hasil **letterbox** 640×640.
- **Output**: `float32` `[1,5,8400]` = `[1, 4+nc, N]` (nc=1). Box `xywh (center)` di ruang 640, tanpa objectness, **tanpa NMS**.
- **Postprocessing (di frontend)**: filter conf → xywh→xyxy → NMS (IoU 0.45) → un-letterbox ke piksel gambar asli.
- **Kelas**: `0 = gerobak` (PKL).

---

## 7. Pemisahan deteksi vs pelanggaran

| Pertanyaan | Dijawab oleh | File |
|------------|--------------|------|
| "Di mana PKL?" | Model YOLO (deteksi) | `web/yolo-onnx.js` |
| "Apakah PKL melanggar?" | Logika aplikasi (aturan) | `web/violation-logic.js` |

Keduanya terpisah: model bisa diganti/retrain tanpa menyentuh aturan, dan aturan
(zona terlarang, trotoar, jarak) bisa diubah tanpa menyentuh model.

---

## 8. Limitasi & fungsi yang BELUM bisa dimigrasi ke browser

Jujur, tidak semua fitur server bisa pindah ke browser murni:

| Fitur | Status | Alasan |
|-------|--------|--------|
| **Deteksi gambar** | ✅ Pindah ke browser | Inti migrasi; jalan via ONNX Runtime Web. |
| **OCR koordinat** (`ocr.py`, EasyOCR) | ⚠️ Belum dipindah | EasyOCR = Python + torch, tak jalan di browser. Opsi: (a) tetap pakai endpoint OCR server hanya saat EXIF kosong, atau (b) ganti ke OCR berbasis JS (mis. Tesseract.js) di masa depan. `ocr.py` **tetap dipertahankan**. |
| **Proses video** (`vidToFrame.py`) | ⚠️ Belum dipindah | Decode video + tulis MP4 berat untuk browser. Bisa dibuat per-frame via `detect()` di `<video>`/`requestVideoFrameCallback`, tapi penggabungan jadi file MP4 lebih cocok di server. `vidToFrame.py` **tetap dipertahankan**. |
| **Server Python** (`app.py`) | ✅ Tidak wajib lagi | Untuk inference gambar normal tak perlu server. Tetap ada sebagai opsi untuk OCR/video. |

### Limitasi teknis lain
- **Ukuran model statis 640×640**: input harus di-letterbox ke 640. Sudah ditangani `yolo-onnx.js`.
- **Performa**: WASM berjalan di CPU browser (lebih lambat); WebGPU jauh lebih cepat bila didukung. `loadModel` otomatis memilih WebGPU bila ada.
- **Model belum di-verify end-to-end di mesin ini** karena blok torch — wajib dijalankan `export_onnx.py` + `verify_onnx.py` di mesin tanpa blok sebelum rilis.

---

## 9. Checklist sebelum rilis ke frontend
1. [ ] `python export_onnx.py` → `model/best.onnx` terbentuk.
2. [ ] `python verify_onnx.py` → deteksi `.onnx` masuk akal vs `.pt`.
3. [ ] Salin `best.onnx` ke folder statis frontend (mis. `public/models/`).
4. [ ] `npm install onnxruntime-web`.
5. [ ] Pakai `web/yolo-onnx.js` + `web/violation-logic.js` (atau port ke framework kamu).
6. [ ] Tes di browser dengan gambar nyata, cek bbox pas di gambar asli.
