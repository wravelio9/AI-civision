// Entry point BE Express.
// Struktur: server -> routes -> controller -> service -> AI (FastAPI).
const express = require("express");
const config = require("./src/config");
const predictRoutes = require("./src/routes/predict.routes");
const errorHandler = require("./src/middleware/error");

const app = express();

app.use(express.json());

// Health check sederhana.
app.get("/health", (req, res) => {
  res.json({ status: "ok", aiServiceUrl: config.aiServiceUrl });
});

// Semua route prediksi di-prefix /api.
app.use("/api", predictRoutes);

// Error handler harus paling akhir.
app.use(errorHandler);

app.listen(config.port, () => {
  console.log(`BE Express jalan di http://localhost:${config.port}`);
  console.log(`Meneruskan ke AI service: ${config.aiServiceUrl}`);
});
