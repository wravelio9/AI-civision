# Civision — AI Detection Service

Deteksi objek YOLO yang diekspos sebagai HTTP service, dipanggil oleh Backend Express.js.

## Arsitektur

```
Client --upload foto--> BE (Express.js) --POST /predict--> AI Service (FastAPI + YOLO)
                                        <--JSON + URL gambar hasil--
```

- **AI Service** (`app.py`): FastAPI, load model YOLO sekali saat startup, endpoint `POST /predict` yang menerima banyak gambar sekaligus (batch).
- **BE** (`examples/express-client.js`): contoh Express yang menerima upload dari client dan meneruskannya ke AI service.

## 1. Setup AI Service

Sudah ada virtual environment di `.venv` (Python 3.11). Install dependency:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Jalankan service:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Cek hidup: buka http://localhost:8000/health

> Kalau `best.pt` belum ada, service otomatis pakai `yolo26n.pt` (model dasar COCO, 80 kelas umum) supaya tetap bisa dites. Field `using_fallback: true` pada `/health` menandakan ini.

## 2. Menghasilkan `best.pt` (training)

Model deteksi `gerobak` kamu dihasilkan lewat training. Jalankan `yolo.py`:

```powershell
.\.venv\Scripts\python.exe yolo.py
```

Setelah selesai, bobot terbaik ada di:

```
runs/detect/train/weights/best.pt
```

`app.py` otomatis memilih `best.pt` ini kalau file-nya ada (lihat `PREFERRED_MODEL` di `app.py`). Restart service setelah training agar model baru ter-load.

## 3. Endpoint AI Service

### `GET /health`
Cek status + model yang sedang dipakai.

### `POST /predict`
- **Form field**: `files` (bisa lebih dari satu file gambar).
- **Query opsional**: `conf` (ambang confidence, default `0.5`).

Contoh:

```powershell
curl.exe -X POST "http://localhost:8000/predict?conf=0.5" -F "files=@foto1.jpg" -F "files=@foto2.jpg"
```

Contoh response:

```json
{
  "success": true,
  "conf": 0.5,
  "results": [
    {
      "filename": "foto1.jpg",
      "detections": [
        {
          "label": "gerobak",
          "class_id": 0,
          "confidence": 0.91,
          "bbox": { "x1": 171.95, "y1": 53.62, "x2": 272.53, "y2": 315.4 }
        }
      ],
      "count": 1,
      "annotated_image_url": "http://localhost:8000/outputs/<hash>.jpg"
    }
  ],
  "errors": []
}
```

- `detections`: daftar objek terdeteksi (label, confidence, koordinat kotak `x1,y1,x2,y2`).
- `annotated_image_url`: URL gambar hasil yang sudah digambari kotak. BE bisa langsung meneruskan/menyimpan URL ini.
- `errors`: berisi file yang gagal diproses (tipe tidak didukung / rusak), sisanya tetap diproses.

> **Base URL gambar hasil**: default `http://localhost:8000`. Di produksi, set environment variable `PUBLIC_BASE_URL` ke alamat publik service, mis. `PUBLIC_BASE_URL=https://ai.domainmu.com`.

## 4. Menjalankan contoh BE Express

```powershell
cd examples
npm install
node express-client.js
```

Lalu kirim foto ke Express (yang akan meneruskan ke AI service):

```powershell
curl.exe -X POST http://localhost:3000/upload -F "photos=@foto1.jpg" -F "photos=@foto2.jpg"
```

Konfigurasi lewat env: `AI_SERVICE_URL` (default `http://localhost:8000`), `PORT` (default `3000`).

## Catatan produksi

- `outputs/` (gambar hasil) dan `.venv/`, `runs/`, `node_modules/` sudah di-ignore git.
- Gambar hasil menumpuk di `outputs/`; jadwalkan pembersihan berkala kalau volume tinggi.
- Untuk beban tinggi, jalankan uvicorn dengan beberapa worker atau di belakang reverse proxy.
