/**
 * Contoh integrasi Express.js -> AI service (FastAPI).
 *
 * Alur:
 *   Client upload foto ke Express (POST /upload)
 *     -> Express teruskan ke AI service (POST /predict)
 *     -> AI balas JSON deteksi + URL gambar hasil
 *     -> Express kembalikan hasil ke client
 *
 * Dependency:
 *   npm install express multer axios form-data
 *
 * Jalankan:
 *   node express-client.js
 *
 * Pastikan AI service sudah hidup:
 *   uvicorn app:app --host 0.0.0.0 --port 8000
 */

const express = require("express");
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");

const app = express();

// Simpan file upload di memory (tidak ditulis ke disk). Cocok karena kita
// langsung meneruskannya ke AI service.
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }, // maks 10MB per file
});

// URL AI service. Di produksi, taruh di environment variable.
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || "http://localhost:8000";

/**
 * Endpoint upload. Field form: `photos` (bisa banyak file sekaligus).
 * Contoh (curl):
 *   curl -X POST http://localhost:3000/upload \
 *     -F "photos=@foto1.jpg" -F "photos=@foto2.jpg"
 */
app.post("/upload", upload.array("photos", 20), async (req, res) => {
  try {
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ error: "Tidak ada file yang diupload." });
    }

    // Bangun multipart form-data untuk dikirim ke AI service.
    // Field name harus `files` supaya cocok dengan parameter di FastAPI.
    const form = new FormData();
    for (const file of req.files) {
      form.append("files", file.buffer, {
        filename: file.originalname,
        contentType: file.mimetype,
      });
    }

    // Teruskan ke AI service. conf opsional (default 0.5 di sisi AI).
    const aiResponse = await axios.post(`${AI_SERVICE_URL}/predict`, form, {
      params: { conf: 0.5 },
      headers: form.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
      timeout: 120000, // beri waktu lebih untuk batch besar
    });

    // aiResponse.data berisi { success, conf, results: [...], errors: [...] }
    // Setiap item results: { filename, detections, count, annotated_image_url }
    return res.json({
      message: "Berhasil diproses",
      data: aiResponse.data,
    });
  } catch (err) {
    // Bedakan error dari AI service vs error jaringan.
    if (err.response) {
      return res.status(err.response.status).json({
        error: "AI service mengembalikan error",
        detail: err.response.data,
      });
    }
    return res.status(502).json({
      error: "Gagal menghubungi AI service",
      detail: err.message,
    });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Express server jalan di http://localhost:${PORT}`);
  console.log(`Meneruskan ke AI service di ${AI_SERVICE_URL}`);
});
