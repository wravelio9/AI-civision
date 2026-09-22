# Civision BE (Express)

BE Express yang menerima upload video dari client dan meneruskannya ke AI service
(FastAPI + YOLO) untuk deteksi objek per frame.

## Struktur (route -> controller -> service)

```
examples/
  server.js                       # entry point: setup express, daftar route, error handler
  .env.example                    # contoh konfigurasi
  src/
    config.js                     # baca env (port, AI_SERVICE_URL, dll)
    routes/predict.routes.js      # ROUTE  : URL + middleware -> controller
    controllers/predict.controller.js  # CONTROLLER: validasi request, bentuk response
    services/ai.service.js        # SERVICE : satu-satunya yang bicara ke AI service
    middleware/upload.js          # multer (memory, filter video, batas ukuran)
    middleware/error.js           # error handler terpusat
```

Alur request:
```
client --(multipart: video)--> BE /api/predict-video
   route -> controller -> service --(POST /predict-video)--> AI service
                                   <--(JSON: summary + annotated_video_url)--
```

Tanggung jawab tiap lapisan:
- **Route**: mendefinisikan endpoint & memasang middleware upload.
- **Controller**: validasi input, ambil `conf`, panggil service, susun response. Tidak tahu detail HTTP ke AI.
- **Service**: membangun multipart form, memanggil AI service via axios. Satu-satunya tempat detail AI berada.

## Setup

```bash
npm install
cp .env.example .env   # lalu sesuaikan AI_SERVICE_URL
npm start              # atau: npm run dev  (auto-reload)
```

## Endpoint

### `POST /api/predict-video`
- **form-data field**: `video` (satu file video)
- **query opsional**: `conf` (default dari `.env`)

Contoh:
```bash
curl -X POST "http://localhost:3000/api/predict-video?conf=0.5" \
  -F "video=@myvideo.mp4"
```

Contoh response:
```json
{
  "message": "Video berhasil diproses",
  "data": {
    "success": true,
    "conf": 0.5,
    "filename": "myvideo.mp4",
    "summary": {
      "total_frames": 20,
      "frames_with_detection": 20,
      "total_detections": 184,
      "fps": 5,
      "width": 692,
      "height": 442
    },
    "annotated_video_url": "http://localhost:8000/outputs/<hash>.mp4"
  }
}
```

`annotated_video_url` menunjuk ke video hasil di AI service. Client bisa langsung
mengunduh/memutar dari URL itu.

### `GET /health`
Cek BE hidup + AI service URL yang dipakai.

## Catatan
- Ukuran maksimum upload diatur `MAX_VIDEO_MB` di `.env` (default 200MB).
- Timeout request ke AI 15 menit (video panjang butuh waktu di CPU).
- Error ditangani terpusat: file bukan video -> 400, AI tak terjangkau -> 502, error AI -> status asli.
