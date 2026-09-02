# Prediksi & Deteksi Karhutla Kalimantan

Project data science/ML untuk memahami dan mendeteksi kebakaran hutan dan
lahan (karhutla) di Kalimantan tahun 2026, menggabungkan data hotspot
satelit (NASA FIRMS) dengan deteksi perubahan lahan dari citra Sentinel-2.

**Status:** Tahap 1 & 2 selesai ✅ | Tahap 3 & 4 (prediksi ML) direncanakan

---

## Ringkasan hasil utama

- **~253.000 titik hotspot** terdeteksi di Kalimantan sepanjang Jan-Agu 2026
  (NASA FIRMS, gabungan sensor VIIRS & MODIS).
- **Eskalasi eksponensial**: dari baseline ~2.000-5.000 MW FRP (Fire
  Radiative Power) harian di Jan-Jul, melonjak ke **>250.000 MW** di akhir
  Agustus 2026 - lonjakan ~50x dalam beberapa minggu.
- **Kalimantan Tengah** adalah provinsi paling terdampak (~96.000 hotspot),
  disusul **Kalimantan Barat** (~85.000) - konsisten dengan liputan berita.
- **Studi kasus deteksi citra satelit** (area ~55km x 55km di perbatasan
  Kalteng, bbox `113.6,-2.4,114.4,-1.6`): **19,5% area (153.822 ha)**
  terindikasi mengalami perubahan akibat kebakaran (dNBR) antara
  Juli-September 2026.
- **Validasi silang**: dari 12.355 hotspot FIRMS di area studi kasus,
  **49,9% berada di piksel yang juga terdeteksi berubah** oleh Sentinel-2 -
  dua metode deteksi independen (termal vs spektral) saling mengkonfirmasi.

---

## Struktur project

```
karhutla-prediction/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py                  # bounding box, konstanta API
│   ├── fetch_firms.py             # Tahap 1: narik data hotspot FIRMS
│   ├── diagnose_unknown.py        # Tahap 1: diagnostik kategorisasi provinsi
│   ├── eda.py                     # Tahap 1: analisis & visualisasi hotspot
│   ├── gee_utils.py                # Tahap 2: fungsi Earth Engine bersama
│   ├── select_case_study.py       # Tahap 2: pilih AOI & tanggal studi kasus
│   ├── burn_detection.py          # Tahap 2: deteksi burn scar (dNBR)
│   └── validate_with_hotspots.py  # Tahap 2: validasi silang FIRMS x Sentinel-2
└── data/                          # hasil fetch & output (tidak di-commit)
```

---

## Tahap 1: Data Pipeline Hotspot (NASA FIRMS)

### Setup

1. **Dapatkan FIRMS MAP_KEY gratis**: https://firms.modaps.eosdis.nasa.gov/api/map_key/
2. **Set sebagai environment variable:**
   ```bash
   export FIRMS_MAP_KEY="your_map_key_here"      # macOS/Linux/Git Bash
   set FIRMS_MAP_KEY=your_map_key_here            # Windows CMD
   $env:FIRMS_MAP_KEY="your_map_key_here"         # Windows PowerShell
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt --break-system-packages
   ```

### Cara pakai

```bash
# Data historis (pakai sensor archive/SP - untuk tanggal > ~2 bulan lalu)
python src/fetch_firms.py --start 2026-01-01 --end 2026-04-30 --archive --out data/firms_early.csv

# Data terbaru (sensor NRT - untuk ~2 bulan terakhir)
python src/fetch_firms.py --start 2026-05-01 --end 2026-08-30 --out data/firms_recent.csv

# Gabungkan kalau datanya kepisah archive vs NRT
python -c "import pandas as pd; a=pd.read_csv('data/firms_early.csv'); b=pd.read_csv('data/firms_recent.csv'); c=pd.concat([a,b]).drop_duplicates(subset=['latitude','longitude','acq_date','acq_time','sensor']); c.to_csv('data/firms_combined.csv', index=False)"

# Analisis & visualisasi
python src/eda.py --input data/firms_combined.csv --outdir data/plots
```

### Catatan penting

- **Rate limit FIRMS**: max **5 hari per request** (bukan 10 seperti dokumentasi
  lama) - `fetch_firms.py` sudah menangani ini otomatis lewat chunking.
- **Sensor NRT vs SP**: data "near real-time" (NRT) cuma tersedia ~60 hari
  terakhir; untuk data lebih lama wajib pakai sensor "_SP" (archive), yang
  punya lag publikasi beberapa bulan.
- **User-Agent header wajib** - beberapa environment (khususnya PowerShell
  dengan proxy tertentu) memblokir request tanpa header ini; script sudah
  menyertakannya secara default.
- **Kategorisasi provinsi** pakai bounding box perkiraan (bukan shapefile
  presisi). Sudah divalidasi akurat untuk 5 ibu kota provinsi, dan secara
  eksplisit memisahkan titik yang berada di **Malaysia/Brunei** (bagian utara
  Borneo) dari kategori "Unknown" yang genuinely ambigu (area
  laut/perbatasan, kemungkinan false-positive dari flare kapal).
- 1 kejadian kebakaran bisa menghasilkan banyak "hotspot" (deteksi per
  piksel satelit) - jangan disamakan dengan jumlah kebakaran atau luas area.

---

## Tahap 2: Deteksi Burn Scar dari Citra Satelit (Sentinel-2)

Daripada memproses seluruh Kalimantan (berat dan tidak perlu untuk skala
portofolio), pendekatannya adalah **studi kasus**: ambil satu klaster
hotspot terpadat dari Tahap 1, lalu bandingkan citra Sentinel-2 sebelum vs
sesudah periode kebakaran menggunakan **dNBR (delta Normalized Burn Ratio)**
- metode standar remote sensing (dipakai USGS, UN-SPIDER) untuk memetakan
area terbakar dan tingkat keparahannya.

### Setup tambahan

1. **Daftar Google Earth Engine** (gratis, tier "Community" - tidak perlu
   billing account): https://code.earthengine.google.com/register
   Catat **Project ID** Google Cloud yang dipakai saat registrasi.
2. **Install dependency:**
   ```bash
   pip install earthengine-api geemap --break-system-packages
   ```
3. **Autentikasi** (sekali saja, buka browser untuk login):
   ```bash
   earthengine authenticate --auth_mode=notebook
   ```
   (Gunakan `--auth_mode=notebook` kalau mode default gagal redirect ke
   `localhost` - umum terjadi di balik firewall/antivirus tertentu.)

### Cara pakai

**Langkah 1 — pilih area & tanggal studi kasus otomatis dari data hotspot:**
```bash
python src/select_case_study.py --input data/firms_combined.csv --month 2026-08
```

**Langkah 2 — deteksi burn scar** (pakai output langkah 1):
```bash
python src/burn_detection.py \
    --bbox 113.600,-2.400,114.400,-1.600 \
    --before-start 2026-07-12 --before-end 2026-07-27 \
    --after-start 2026-08-25 --after-end 2026-09-15 \
    --project YOUR_GEE_PROJECT_ID \
    --max-cloud-pct 80
```

**Langkah 3 — validasi silang dengan hotspot FIRMS:**
```bash
python src/validate_with_hotspots.py \
    --input data/firms_combined.csv \
    --bbox 113.600,-2.400,114.400,-1.600 \
    --before-start 2026-07-12 --before-end 2026-07-27 \
    --after-start 2026-08-25 --after-end 2026-09-15 \
    --project YOUR_GEE_PROJECT_ID \
    --max-cloud-pct 80
```

### Hasil studi kasus (area contoh, ~55km x 55km, perbatasan Kalteng)

| Tingkat keparahan | Luas (ha) |
|---|---|
| Tidak terbakar | 633.500,6 |
| Rendah | 99.520,5 |
| Sedang-rendah | 36.761,9 |
| Sedang-tinggi | 14.533,3 |
| Tinggi | 3.006,6 |
| **Total terindikasi terbakar** | **153.822,2 (19,5%)** |

**Validasi silang** (12.355 hotspot FIRMS di area & periode yang sama):

| Kategori (Sentinel-2) | Jumlah hotspot | Persentase |
|---|---|---|
| Tidak terbakar | 6.195 | 50,1% |
| Rendah | 2.494 | 20,2% |
| Sedang-rendah | 1.725 | 14,0% |
| Sedang-tinggi | 1.415 | 11,5% |
| Tinggi | 526 | 4,3% |
| **Terkonfirmasi (≥1)** | **6.160** | **49,9%** |

### Catatan & keterbatasan penting

- **Asap = musuh ganda**: di puncak musim kebakaran, citra satelit sendiri
  sering tertutup asap tebal (bukan cuma awan biasa), sehingga `--max-cloud-pct`
  perlu dinaikkan cukup tinggi (kami pakai 80%) untuk tetap mendapat citra.
  Ini ironi yang menarik dicatat: makin parah kebakarannya, makin sulit
  memvalidasinya lewat citra optik.
- **~50% tingkat konfirmasi itu wajar**, bukan tanda kegagalan - FIRMS
  mendeteksi panas *real-time* per piksel ~375m-1km, sementara dNBR
  mendeteksi perubahan *permanen* pasca-kebakaran pada resolusi 10-20m. Ada
  lag alami dan perbedaan resolusi antara keduanya.
- **Artefak tile & false-positive**: peta severity mentah menunjukkan garis
  sambungan antar-tile Sentinel-2 dan beberapa bentuk geometris tajam
  (kemungkinan tambak/area pertanian yang berubah, bukan kebakaran). Untuk
  analisis produksi, ini perlu di-mask lebih lanjut.
- **Blob melingkar rapi berwarna abu-abu ("tidak terbakar")** di scatter
  plot validasi kemungkinan adalah badan air atau area industri - sumber
  false-positive umum di data FIRMS.
- dNBR **bukan bukti mutlak kebakaran** - perubahan lahan lain (panen,
  deforestasi) juga mengubah NBR. Validasi silang dengan FIRMS membantu,
  tapi verifikasi lapangan tetap ideal untuk klaim yang lebih kuat.

---

## Roadmap selanjutnya

3. **Feature engineering**: gabungkan hotspot historis + data cuaca BMKG
   (curah hujan, kelembapan) + tinggi muka air gambut
4. **Model prediksi risiko kebakaran** harian per kabupaten (XGBoost/LightGBM,
   atau LSTM untuk pendekatan time-series penuh) - divalidasi dengan burn
   scar Tahap 2 sebagai ground truth tambahan
5. (Opsional) Dashboard interaktif (Streamlit) untuk visualisasi peta risiko

## Tantangan teknis yang diselesaikan (log debugging)

Beberapa isu nyata yang muncul & diperbaiki selama pengembangan - dicatat
di sini karena mencerminkan proses debugging end-to-end yang sesungguhnya:

- Batas `day_range` FIRMS API ternyata 5 hari, bukan 10 seperti asumsi awal
  → fix di `config.py`.
- Request tanpa `User-Agent` header ditolak (400) oleh proxy/WAF di
  environment tertentu meski URL identik berhasil di browser → fix di
  `fetch_firms.py`.
- Kategorisasi provinsi berbasis bounding box awalnya salah mengelompokkan
  titik di Malaysia/Brunei sebagai "Unknown" → diperbaiki jadi kategori
  eksplisit di `config.py`.
- `PermissionError` saat menulis CSV di folder OneDrive akibat sinkronisasi
  file yang sedang berjalan → solusi: pindah folder kerja keluar dari
  OneDrive.
- `sampleRegions().getInfo()` Earth Engine dibatasi 5.000 elemen per
  panggilan → `validate_with_hotspots.py` memproses per-batch.

## Sumber data

- [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) — hotspot kebakaran aktif
- [Copernicus Sentinel-2](https://sentinels.copernicus.eu/) via
  [Google Earth Engine](https://earthengine.google.com/) — citra optik untuk
  deteksi burn scar
- [BMKG](https://www.bmkg.go.id/) — data cuaca/iklim (untuk tahap selanjutnya)
- [SIPONGI Kementerian Kehutanan](https://sipongi.menlhk.go.id/) — monitoring
  karhutla resmi pemerintah Indonesia
