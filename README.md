---
title: Civision AI Detection
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Civision — AI Detection Service

Deteksi objek YOLO yang diekspos sebagai HTTP service, dipanggil oleh Backend Express.js.

> **Deploy gratis di Hugging Face Spaces** — lihat bagian "Deploy gratis (Hugging Face Spaces)" di bawah.

## Arsitektur

```
Client --upload foto--> BE (Express.js) --POST /predict--> AI Service (FastAPI + YOLO)
                                        <--JSON + URL gambar hasil--
```

- **AI Service** (`app.py`): FastAPI, load model YOLO sekali saat startup, endpoint `POST /predict` yang menerima banyak gambar sekaligus (batch).
- **BE** (`examples/express-client.js`): contoh Express yang menerima upload dari client dan meneruskannya ke AI service.

## 0. Setup pertama kali (clone baru / dari teman)

Folder `.venv` **tidak dibagikan lewat git** (isinya binary khusus mesin & terlalu besar).
Setelah clone repo, buat ulang virtual environment sendiri dari `requirements.lock.txt`
(berisi versi PERSIS yang sudah teruji).

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.lock.txt
```

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.lock.txt
```

Catatan:
- Pakai **Python 3.11** (versi lama seperti 3.7 tidak didukung model YOLO baru).
- `requirements.lock.txt` = versi terkunci (paling aman, sama persis).
  `requirements.txt` = daftar longgar (ambil versi terbaru) kalau mau update.
- Setelah venv aktif (`(.venv)` muncul di prompt), lanjut ke bagian 1.

## 1. Setup AI Service

Setelah venv ada (lihat bagian 0), install dependency kalau belum:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
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

---

## Deploy gratis (Hugging Face Spaces)

Hugging Face Spaces gratis, tanpa kartu kredit, dan punya RAM besar (cukup untuk torch + YOLO).

1. Buat akun di https://huggingface.co (gratis).
2. Klik **New** → **Space**.
3. Isi:
   - **Owner**: akun kamu
   - **Space name**: mis. `civision-ai`
   - **SDK**: pilih **Docker**
   - **Visibility**: Public (gratis)
4. Setelah Space dibuat, kamu punya git repo baru dari HF. Push kode ke situ:
   ```powershell
   git remote add hf https://huggingface.co/spaces/<username>/civision-ai
   git push hf main
   ```
   (login pakai token dari https://huggingface.co/settings/tokens)
5. HF otomatis build Dockerfile dan menjalankannya. Tunggu status jadi **Running**.
6. URL service kamu: `https://<username>-civision-ai.hf.space`
   - Cek: buka `https://<username>-civision-ai.hf.space/health`

### Hubungkan BE
Set di Express: `AI_SERVICE_URL=https://<username>-civision-ai.hf.space`

### Catatan free tier
- Space "tidur" setelah lama tidak ada trafik; request pertama agak lambat (bangun dulu).
- Tidak perlu set `PUBLIC_BASE_URL` — `app.py` otomatis pakai URL request.
