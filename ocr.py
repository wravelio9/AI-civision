"""
OCR koordinat lokasi dari overlay teks di foto.

Dipakai untuk foto yang metadata GPS (EXIF)-nya kosong: banyak aplikasi kamera
(mis. "GPS Map Camera") mencetak koordinat langsung di atas foto sebagai teks.
Modul ini membaca teks itu dengan EasyOCR lalu mengekstrak lat/lon.

Fungsi utama:
    extract_coordinates(image_bgr) -> {"lat": float, "lon": float} | None

Model EasyOCR di-load sekali (lazy) saat pertama dipakai.
"""

import re
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Reader EasyOCR di-load lazy: hanya diinisialisasi saat OCR pertama dipanggil,
# supaya startup service tidak melambat kalau OCR tidak selalu dipakai.
# ---------------------------------------------------------------------------
_reader = None


# Status ketersediaan OCR: None = belum dicek, True/False = hasil cek.
_ocr_available = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr

        # Bahasa Inggris cukup untuk angka & label Lat/Long. gpu=False (CPU).
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def is_ocr_available() -> bool:
    """
    Cek apakah mesin OCR (EasyOCR) benar-benar bisa dipakai.

    Ini "memeriksa OCR sebelum membaca": mencoba meng-import dan menginisialisasi
    reader. Hasilnya di-cache supaya tidak menginisialisasi berulang kali.

    Returns:
        True kalau OCR siap dipakai, False kalau tidak (dependency hilang,
        model gagal di-download, dsb).
    """
    global _ocr_available
    if _ocr_available is None:
        try:
            _get_reader()  # coba load sungguhan
            _ocr_available = True
        except Exception as exc:  # noqa: BLE001
            print(f"[ocr] OCR tidak tersedia: {exc}")
            _ocr_available = False
    return _ocr_available


# ---------------------------------------------------------------------------
# Parsing koordinat dari teks
# ---------------------------------------------------------------------------

def _dms_to_decimal(deg: float, minute: float, sec: float, hemi: str) -> float:
    """Konversi derajat-menit-detik ke desimal. Hemisphere S/W jadi negatif."""
    val = deg + minute / 60.0 + sec / 3600.0
    if hemi.upper() in ("S", "W"):
        val = -val
    return val


def _valid_lat(v: float) -> bool:
    return -90.0 <= v <= 90.0


def _valid_lon(v: float) -> bool:
    return -180.0 <= v <= 180.0


def _normalize(text: str) -> str:
    """
    Bersihkan noise umum hasil OCR sebelum parsing:
      - '$' sering salah baca untuk 'S' (South).
      - 'lng'/'Ing'/'1ng' -> 'long'.
      - samakan pemisah & rapikan spasi.
    """
    t = " " + text.replace("\n", " ") + " "
    # '$' -> 'S' (hemisphere selatan yang salah terbaca).
    t = t.replace("$", "S")
    # variasi salah-baca 'lng' -> 'long'.
    t = re.sub(r"\b[l1I]ng\b", "long", t, flags=re.IGNORECASE)
    t = re.sub(r"\blng\b", "long", t, flags=re.IGNORECASE)
    return t


# Label + angka. Angka boleh: -7.257472  ATAU  -7257472 (titik hilang saat OCR).
# [^0-9\-]{0,6} = boleh ada sedikit karakter sampah antara label dan angka.
_LAT_LABELED = re.compile(r"lat(?:itude)?[^0-9\-]{0,6}(-?\d[\d.]*)", re.IGNORECASE)
_LON_LABELED = re.compile(r"lo?ng(?:itude)?[^0-9\-]{0,6}(-?\d[\d.]*)", re.IGNORECASE)

# DMS: 6°55'2.6"S  atau  6 55 2.6 S  (dengan/ tanpa simbol derajat).
_DMS = re.compile(
    r"(\d{1,3})\s*[°:\s]\s*(\d{1,2})\s*['′:\s]\s*(\d{1,2}(?:\.\d+)?)\s*[\"″]?\s*([NSEW])",
    re.IGNORECASE,
)

# Semua angka desimal (minimal 3 angka di belakang koma) di teks.
_ANY_DECIMAL = re.compile(r"-?\d{1,3}\.\d{3,}")


def _fix_decimal(raw: str, is_lat: bool) -> Optional[float]:
    """
    Ubah string angka jadi float koordinat, memperbaiki titik desimal yang
    hilang saat OCR. Contoh: '-7257472' (lat) -> -7.257472.
    """
    neg = raw.startswith("-")
    digits = raw.lstrip("-")

    if "." in digits:
        val = float(raw)
    else:
        # Titik desimal hilang. Sisipkan setelah bagian derajat.
        # Untuk lat, coba 1 digit derajat dulu (lebih umum, mis. -7.xxxxx),
        # baru 2. Untuk lon coba 2-3 digit dulu.
        d = digits
        placed = None
        order = (1, 2) if is_lat else (3, 2, 1)
        for deg_len in order:
            if deg_len >= len(d):
                continue
            head = d[:deg_len]
            candidate = float(f"{head}.{d[deg_len:]}")
            if is_lat and _valid_lat(candidate):
                placed = candidate
                break
            if not is_lat and _valid_lon(candidate):
                placed = candidate
                break
        if placed is None:
            return None
        val = placed

    if neg:
        val = -abs(val)
    return val


def parse_coordinates(text: str) -> Optional[dict]:
    """
    Ekstrak lat/lon dari teks hasil OCR (tahan noise).

    Strategi (urut prioritas):
      1. Label eksplisit "Lat ... Long ..." (paling andal, perbaiki titik hilang).
      2. Format DMS (derajat menit detik + N/S/E/W, dengan/tanpa simbol).
      3. Pasangan angka desimal: pilih kombinasi yang valid sebagai (lat, lon).
    """
    if not text:
        return None

    flat = _normalize(text)

    # --- 1. Label eksplisit ---
    lat_m = _LAT_LABELED.search(flat)
    lon_m = _LON_LABELED.search(flat)
    if lat_m and lon_m:
        lat = _fix_decimal(lat_m.group(1), is_lat=True)
        lon = _fix_decimal(lon_m.group(1), is_lat=False)
        if lat is not None and lon is not None and _valid_lat(lat) and _valid_lon(lon):
            return {"lat": round(lat, 6), "lon": round(lon, 6)}

    # --- 2. DMS ---
    dms_matches = _DMS.findall(flat)
    if len(dms_matches) >= 2:
        vals = []
        for deg, minute, sec, hemi in dms_matches[:2]:
            vals.append(
                (_dms_to_decimal(float(deg), float(minute), float(sec), hemi), hemi.upper())
            )
        lat = next((v for v, h in vals if h in ("N", "S")), None)
        lon = next((v for v, h in vals if h in ("E", "W")), None)
        if lat is not None and lon is not None and _valid_lat(lat) and _valid_lon(lon):
            return {"lat": round(lat, 6), "lon": round(lon, 6)}

    # --- 3. Desimal + huruf hemisphere (mis. "6.9174 S , 107.6191 E") ---
    #     Angka desimal yang diikuti N/S/E/W; tanda dari hemisphere.
    hemi_pairs = re.findall(r"(\d{1,3}\.\d{3,})\s*([NSEW])", flat, flags=re.IGNORECASE)
    if len(hemi_pairs) >= 2:
        lat = lon = None
        for num, h in hemi_pairs:
            h = h.upper()
            v = float(num)
            if h in ("N", "S") and lat is None:
                lat = -v if h == "S" else v
            elif h in ("E", "W") and lon is None:
                lon = -v if h == "W" else v
        if lat is not None and lon is not None and _valid_lat(lat) and _valid_lon(lon):
            return {"lat": round(lat, 6), "lon": round(lon, 6)}

    # --- 4. Pasangan desimal polos (ambil semua angka, cari pasangan valid) ---
    nums = [float(x) for x in _ANY_DECIMAL.findall(flat)]
    for i in range(len(nums)):
        for j in range(len(nums)):
            if i == j:
                continue
            a, b = nums[i], nums[j]
            if _valid_lat(a) and _valid_lon(b) and abs(a) <= 90:
                if abs(b) > 90 or i < j:
                    return {"lat": round(a, 6), "lon": round(b, 6)}

    return None


# ---------------------------------------------------------------------------
# OCR gambar -> koordinat
# ---------------------------------------------------------------------------
def ocr_text(image_bgr: np.ndarray) -> str:
    """Jalankan EasyOCR pada gambar (numpy BGR), kembalikan semua teks tergabung."""
    reader = _get_reader()
    # detail=0 -> hanya list string teks.
    lines = reader.readtext(image_bgr, detail=0)
    return " ".join(lines)


def extract_coordinates(image_bgr: np.ndarray) -> Optional[dict]:
    """
    Baca teks overlay di gambar dan ekstrak koordinat.

    CEK OCR DULU sebelum membaca: kalau mesin OCR tidak tersedia, langsung
    kembalikan None tanpa mencoba (menghindari error tiap gambar).

    Returns:
        {"lat": float, "lon": float} kalau ketemu, atau None kalau tidak terbaca.
    """
    result, _status = extract_coordinates_ex(image_bgr)
    return result


def extract_coordinates_ex(image_bgr: np.ndarray):
    """
    Versi detail dari extract_coordinates yang juga mengembalikan STATUS,
    supaya pemanggil bisa membedakan:
      - "unavailable" : mesin OCR tidak bisa dipakai (belum dibaca sama sekali)
      - "not_found"   : OCR jalan tapi tidak menemukan koordinat
      - "found"       : koordinat ketemu

    Returns:
        (coords_or_None, status_string)
    """
    # 1. Cek ketersediaan OCR SEBELUM mencoba membaca.
    if not is_ocr_available():
        return None, "unavailable"

    # 2. Jalankan OCR.
    try:
        text = ocr_text(image_bgr)
    except Exception:
        return None, "unavailable"

    # 3. Parse koordinat dari teks.
    coords = parse_coordinates(text)
    return (coords, "found") if coords else (None, "not_found")
