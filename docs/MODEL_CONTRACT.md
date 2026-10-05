# CiVision — Model Contract (`best.onnx`)

Dokumen ini adalah kontrak antara model ONNX dan frontend. Dengan dokumen ini,
developer frontend bisa mengimplementasikan inference TANPA menebak.

Model: YOLO (YOLO26n) deteksi objek, 1 kelas custom (PKL).
Dihasilkan dari `model/best.pt` lewat `export_onnx.py` (lihat ONNX_MIGRATION.md).

---

## 1. Ringkasan

| Hal | Nilai |
|-----|-------|
| Format | ONNX (opset 12) |
| Task | Object detection |
| Jumlah kelas (nc) | 1 |
| Nama kelas | `0 = gerobak` (PKL) |
| Image size | 640 × 640 (tetap/statis) |
| NMS di dalam model? | **Tidak** — dilakukan di frontend |
| Eksekusi | ONNX Runtime Web (WASM, atau WebGPU bila didukung) |

---

## 2. INPUT

| Properti | Nilai |
|----------|-------|
| Nama input | `images` (ambil runtime: `session.inputNames[0]`) |
| Tipe data | `float32` |
| Shape | `[1, 3, 640, 640]` → `[batch, channel, height, width]` (NCHW) |
| Urutan channel | **RGB** (bukan BGR) |
| Normalisasi | setiap piksel dibagi 255 → rentang `[0.0, 1.0]` |
| Preprocessing wajib | **letterbox** ke 640×640 (jaga aspect ratio, pad warna abu 114,114,114) |
| Layout memori | contiguous, channel-first (CHW) |

### Langkah preprocessing (wajib sama seperti training)
1. Ambil gambar sumber (ukuran bebas).
2. **Letterbox**: skala gambar agar sisi terpanjang = 640, sisi lain dipadding
   dengan warna `rgb(114,114,114)` sampai jadi 640×640. Simpan faktor skala `r`
   dan padding `(dw, dh)` — dibutuhkan untuk mengembalikan koordinat nanti.
3. Konversi **BGR→RGB** bila sumbernya BGR (canvas browser biasanya sudah RGBA).
4. Bagi tiap nilai piksel dengan `255.0`.
5. Susun ulang dari HWC (tinggi, lebar, channel) menjadi CHW, lalu tambah batch
   dim di depan → `[1, 3, 640, 640]`.

> Catatan browser: dari `<canvas>` kamu dapat `Uint8ClampedArray` RGBA (HWC).
> Buang channel alpha, susun ke CHW float32 /255. Lihat `web/yolo-onnx.js`.

---

## 3. OUTPUT

Model mengeluarkan **satu** tensor (ambil nama: `session.outputNames[0]`).

| Properti | Nilai |
|----------|-------|
| Tipe data | `float32` |
| Shape | `[1, 4 + nc, 8400]` = `[1, 5, 8400]` (karena nc = 1) |
| Jumlah prediksi | 8400 kandidat box (anchor-free) |

### Interpretasi layout `[1, 5, 8400]`
Ini format YOLOv8/YOLO11/YOLO26 **tanpa NMS**. Tensornya "transposed":
baris = atribut, kolom = kandidat. Jadi untuk kandidat ke-`i`:

```
output[0][0][i] = cx   (pusat x, dalam piksel ruang 640)
output[0][1][i] = cy   (pusat y)
output[0][2][i] = w    (lebar box)
output[0][3][i] = h    (tinggi box)
output[0][4][i] = score kelas 0 (gerobak)   // kalau nc>1: index 4..4+nc-1
```

> Praktik aman: setelah dapat tensor, **transpose** jadi `[8400, 5]` supaya tiap
> baris = 1 kandidat `[cx, cy, w, h, score_cls0, ...]`. Kode frontend sudah
> menangani kedua layout `(1,4+nc,N)` dan `(1,N,4+nc)` secara otomatis.

### Penting
- Box dalam format **xywh (center)**, satuan **piksel pada ruang 640 (letterboxed)** —
  BUKAN koordinat gambar asli, BUKAN normalisasi 0–1.
- **Tidak ada objectness terpisah** (beda dari YOLOv5). Confidence = skor kelas.
- Model **tidak** melakukan NMS → banyak box tumpang tindih. Frontend WAJIB NMS.

---

## 4. POSTPROCESSING (wajib di frontend)

Urutan:
1. (Jika perlu) transpose output ke `[8400, 5]`.
2. Untuk tiap kandidat: `classId = argmax(scores)`, `confidence = max(scores)`.
3. **Filter confidence**: buang kandidat `< conf_threshold` (default `0.25`).
4. Konversi `xywh (center)` → `xyxy`:
   ```
   x1 = cx - w/2 ;  y1 = cy - h/2 ;  x2 = cx + w/2 ;  y2 = cy + h/2
   ```
5. **NMS** (Non-Max Suppression) per kelas, IoU threshold default `0.45`.
6. **Un-letterbox** koordinat kembali ke ukuran gambar asli:
   ```
   x_asli = (x_640 - dw) / r
   y_asli = (y_640 - dh) / r
   ```
   (`r`, `dw`, `dh` dari langkah letterbox saat preprocessing)

### Hasil akhir (bentuk konseptual)
```json
{
  "detections": [
    {
      "classId": 0,
      "className": "gerobak",
      "confidence": 0.94,
      "bbox": { "x1": 320, "y1": 180, "x2": 510, "y2": 430 }
    }
  ]
}
```
> `bbox` dalam piksel pada **gambar asli**. `className` dipetakan dari `classId`
> via `{0: "gerobak"}` (PKL).

---

## 5. Parameter yang bisa disetel

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `confThreshold` | 0.25 | ambang minimum confidence |
| `iouThreshold` | 0.45 | ambang IoU untuk NMS |
| `inputSize` | 640 | harus 640 (model statis) |

---

## 6. Catatan kompatibilitas ONNX Runtime Web

- **opset 12** dipilih karena stabil & didukung penuh oleh `onnxruntime-web`
  pada backend WASM maupun WebGPU.
- **Shape statis `[1,3,640,640]`** (bukan dynamic) → lebih cepat & lebih aman di
  browser; tidak ada operasi shape dinamis yang kadang tak didukung.
- **NMS tidak dibungkus ke graph** → menghindari ketergantungan op `NonMaxSuppression`
  yang implementasinya bisa berbeda/terbatas di ort-web. NMS dilakukan di JS.
