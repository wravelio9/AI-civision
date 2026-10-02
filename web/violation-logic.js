/**
 * CiVision — Logika Pelanggaran (TERPISAH dari deteksi).
 *
 * PEMISAHAN TANGGUNG JAWAB:
 *   - yolo-onnx.js  menjawab: "Di mana PKL (gerobak)?"  -> deteksi (class, conf, bbox)
 *   - file ini      menjawab: "Apakah PKL ini melanggar aturan?" -> keputusan
 *
 * Modul ini TIDAK tahu apa-apa soal YOLO/ONNX. Ia hanya menerima daftar
 * `detections` (format dari MODEL_CONTRACT.md) + definisi aturan, lalu
 * mengembalikan keputusan pelanggaran. Jadi model deteksi bisa diganti/retrain
 * tanpa menyentuh logika aturan, dan sebaliknya.
 *
 * Bentuk detection yang diharapkan (dari yolo-onnx.js):
 *   { classId, className, confidence, bbox: { x1, y1, x2, y2 } }
 */

// --------------------------------------------------------------------------
// Util geometri dasar
// --------------------------------------------------------------------------

/** Titik tengah-bawah bbox (biasanya titik "kaki" objek menyentuh tanah). */
export function bboxGroundPoint(bbox) {
  return { x: (bbox.x1 + bbox.x2) / 2, y: bbox.y2 };
}

/** Pusat bbox. */
export function bboxCenter(bbox) {
  return { x: (bbox.x1 + bbox.x2) / 2, y: (bbox.y1 + bbox.y2) / 2 };
}

/** Luas irisan dua bbox. */
function intersectionArea(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  return Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
}

/** Luas bbox. */
function bboxArea(b) {
  return Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
}

/**
 * Rasio bbox PKL yang berada di dalam sebuah zona persegi (0..1).
 * Zona: { x1, y1, x2, y2 }.
 */
export function overlapRatioWithZone(bbox, zone) {
  const area = bboxArea(bbox);
  if (area === 0) return 0;
  return intersectionArea(bbox, zone) / area;
}

/** Apakah sebuah titik berada di dalam polygon (ray casting). */
export function pointInPolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x,
      yi = polygon[i].y;
    const xj = polygon[j].x,
      yj = polygon[j].y;
    const intersect =
      yi > point.y !== yj > point.y &&
      point.x < ((xj - xi) * (point.y - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

// --------------------------------------------------------------------------
// Aturan pelanggaran
// --------------------------------------------------------------------------
//
// Definisi aturan dipisah sebagai DATA, bukan hardcode. Satu aturan punya:
//   - id, label  : identitas aturan
//   - type       : "zone-rect" | "zone-polygon"
//   - area       : rect {x1,y1,x2,y2} atau polygon [{x,y}, ...]
//   - minOverlap : (zone-rect) rasio minimum bbox di dalam zona untuk dianggap melanggar
//   - test(detection) : (opsional) fungsi custom, menang atas type bawaan
//
// Koordinat aturan HARUS dalam ruang gambar asli (sama seperti bbox hasil
// un-letterbox dari yolo-onnx.js).

/**
 * Evaluasi satu deteksi terhadap satu aturan.
 * @returns {boolean} true kalau melanggar aturan ini.
 */
function evaluateRule(detection, rule) {
  if (typeof rule.test === "function") {
    return !!rule.test(detection);
  }

  if (rule.type === "zone-rect") {
    const ratio = overlapRatioWithZone(detection.bbox, rule.area);
    return ratio >= (rule.minOverlap ?? 0.3);
  }

  if (rule.type === "zone-polygon") {
    // Pakai titik "kaki" (tengah-bawah) sebagai acuan posisi PKL.
    return pointInPolygon(bboxGroundPoint(detection.bbox), rule.area);
  }

  return false;
}

/**
 * Evaluasi SEMUA deteksi terhadap SEMUA aturan.
 *
 * @param {Array} detections  hasil dari yolo-onnx.js
 * @param {Array} rules       daftar definisi aturan
 * @param {object} opts       { minConfidence?: number, targetClassIds?: number[] }
 * @returns {{ violations: Array, summary: object }}
 */
export function evaluateViolations(detections, rules, opts = {}) {
  const minConfidence = opts.minConfidence ?? 0.25;
  const targetClassIds = opts.targetClassIds ?? null; // null = semua kelas

  const violations = [];

  for (const det of detections) {
    if (det.confidence < minConfidence) continue;
    if (targetClassIds && !targetClassIds.includes(det.classId)) continue;

    const brokenRules = [];
    for (const rule of rules) {
      if (evaluateRule(det, rule)) {
        brokenRules.push({ ruleId: rule.id, label: rule.label });
      }
    }

    if (brokenRules.length > 0) {
      violations.push({
        detection: det,
        brokenRules,
        isViolation: true,
      });
    }
  }

  return {
    violations,
    summary: {
      totalDetections: detections.length,
      totalViolations: violations.length,
      hasViolation: violations.length > 0,
    },
  };
}
