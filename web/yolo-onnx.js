/**
 * CiVision — YOLO inference di browser pakai ONNX Runtime Web.
 *
 * Modul ini HANYA menjawab: "Di mana PKL (gerobak)?" -> deteksi.
 * Logika pelanggaran TERPISAH (lihat violation-logic.js).
 *
 * Dependensi: onnxruntime-web
 *   import * as ort from "onnxruntime-web";
 *
 * Lihat MODEL_CONTRACT.md untuk spesifikasi lengkap input/output.
 */

import * as ort from "onnxruntime-web";

const INPUT_SIZE = 640;
const CLASS_NAMES = { 0: "gerobak" }; // PKL; sinkron dengan dataset/data.yaml

/**
 * Muat model ONNX. Panggil sekali, simpan session-nya.
 * @param {string} modelUrl  URL/path ke best.onnx (file statis).
 * @param {object} opts      { executionProviders?: string[] }
 * @returns {Promise<ort.InferenceSession>}
 */
export async function loadModel(modelUrl, opts = {}) {
  // Prefer WebGPU bila didukung, fallback ke WASM.
  const providers =
    opts.executionProviders ||
    ("gpu" in navigator ? ["webgpu", "wasm"] : ["wasm"]);
  return ort.InferenceSession.create(modelUrl, {
    executionProviders: providers,
    graphOptimizationLevel: "all",
  });
}

/**
 * Letterbox gambar ke 640x640 (jaga aspect ratio), hasilkan tensor NCHW float32.
 * @param {HTMLImageElement|HTMLCanvasElement|ImageBitmap} source
 * @returns {{tensor: ort.Tensor, r: number, dw: number, dh: number,
 *            srcW: number, srcH: number}}
 */
export function preprocess(source) {
  const srcW = source.width || source.videoWidth;
  const srcH = source.height || source.videoHeight;

  const r = Math.min(INPUT_SIZE / srcW, INPUT_SIZE / srcH);
  const newW = Math.round(srcW * r);
  const newH = Math.round(srcH * r);
  const dw = (INPUT_SIZE - newW) / 2;
  const dh = (INPUT_SIZE - newH) / 2;

  // Canvas 640x640 dengan background abu (114,114,114) = padding letterbox.
  const canvas = document.createElement("canvas");
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  ctx.drawImage(source, 0, 0, srcW, srcH, dw, dh, newW, newH);

  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE); // RGBA, HWC
  const area = INPUT_SIZE * INPUT_SIZE;
  const chw = new Float32Array(3 * area); // CHW, /255, RGB

  for (let i = 0; i < area; i++) {
    chw[i] = data[i * 4] / 255; // R plane
    chw[area + i] = data[i * 4 + 1] / 255; // G plane
    chw[2 * area + i] = data[i * 4 + 2] / 255; // B plane (alpha dibuang)
  }

  const tensor = new ort.Tensor("float32", chw, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  return { tensor, r, dw, dh, srcW, srcH };
}

/** IoU dua box [x1,y1,x2,y2]. */
function iou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter + 1e-9);
}

/** Non-Max Suppression. Mengembalikan index yang dipertahankan. */
function nms(boxes, scores, iouThreshold) {
  const order = scores
    .map((s, i) => [s, i])
    .sort((a, b) => b[0] - a[0])
    .map((p) => p[1]);
  const keep = [];
  const removed = new Set();
  for (let _i = 0; _i < order.length; _i++) {
    const i = order[_i];
    if (removed.has(i)) continue;
    keep.push(i);
    for (let _j = _i + 1; _j < order.length; _j++) {
      const j = order[_j];
      if (removed.has(j)) continue;
      if (iou(boxes[i], boxes[j]) > iouThreshold) removed.add(j);
    }
  }
  return keep;
}

/**
 * Postprocess output mentah ONNX menjadi daftar deteksi (koordinat gambar asli).
 * Mendukung layout output (1, 4+nc, N) maupun (1, N, 4+nc).
 */
export function postprocess(output, meta, opts = {}) {
  const confThreshold = opts.confThreshold ?? 0.25;
  const iouThreshold = opts.iouThreshold ?? 0.45;
  const { r, dw, dh } = meta;

  const dims = output.dims; // mis. [1, 5, 8400]
  const data = output.data; // Float32Array
  const d1 = dims[1];
  const d2 = dims[2];

  // Tentukan jumlah atribut (4+nc) dan jumlah kandidat (N).
  // Atribut selalu lebih kecil dari jumlah kandidat.
  const attrs = Math.min(d1, d2);
  const num = Math.max(d1, d2);
  const transposed = d1 < d2; // true kalau layout (1, attrs, N)
  const nc = attrs - 4;

  const get = (c, a) =>
    transposed ? data[a * num + c] : data[c * attrs + a];

  const boxes = [];
  const scores = [];
  const classIds = [];

  for (let i = 0; i < num; i++) {
    // skor kelas terbaik
    let bestCls = 0;
    let bestScore = -Infinity;
    for (let k = 0; k < nc; k++) {
      const s = get(i, 4 + k);
      if (s > bestScore) {
        bestScore = s;
        bestCls = k;
      }
    }
    if (bestScore < confThreshold) continue;

    const cx = get(i, 0);
    const cy = get(i, 1);
    const w = get(i, 2);
    const h = get(i, 3);

    // xywh(center, ruang 640) -> xyxy -> un-letterbox ke gambar asli
    const x1 = (cx - w / 2 - dw) / r;
    const y1 = (cy - h / 2 - dh) / r;
    const x2 = (cx + w / 2 - dw) / r;
    const y2 = (cy + h / 2 - dh) / r;

    boxes.push([x1, y1, x2, y2]);
    scores.push(bestScore);
    classIds.push(bestCls);
  }

  const keep = nms(boxes, scores, iouThreshold);
  return keep.map((i) => ({
    classId: classIds[i],
    className: CLASS_NAMES[classIds[i]] ?? String(classIds[i]),
    confidence: Math.round(scores[i] * 10000) / 10000,
    bbox: {
      x1: Math.round(boxes[i][0]),
      y1: Math.round(boxes[i][1]),
      x2: Math.round(boxes[i][2]),
      y2: Math.round(boxes[i][3]),
    },
  }));
}

/**
 * Pipeline lengkap: preprocess -> infer -> postprocess.
 * @returns {Promise<{detections: Array}>}
 */
export async function detect(session, source, opts = {}) {
  const meta = preprocess(source);
  const feeds = { [session.inputNames[0]]: meta.tensor };
  const results = await session.run(feeds);
  const output = results[session.outputNames[0]];
  const detections = postprocess(output, meta, opts);
  return { detections };
}
