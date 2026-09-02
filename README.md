# Prediksi & Deteksi Karhutla Kalimantan (Tahap 1: Data Pipeline)

Project portofolio ML/data science untuk memahami dan (nantinya) memprediksi
kebakaran hutan dan lahan (karhutla) di Kalimantan, menggunakan data satelit
NASA FIRMS.

Ini adalah **tahap paling awal**: mengambil data hotspot mentah dan melakukan
eksplorasi awal. Tahap berikutnya (deteksi citra satelit dengan Sentinel-2,
lalu model prediksi risiko) akan dibangun di atas fondasi ini.

## Struktur project

```
karhutla-prediction/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py        # bounding box Kalimantan & per-provinsi, konstanta API
│   ├── fetch_firms.py   # narik data hotspot dari NASA FIRMS API
│   └── eda.py           # analisis eksploratif + visualisasi
├── data/                # (kosong di repo ini - hasil fetch disimpan di sini)
└── notebooks/           # tempat untuk eksplorasi ad-hoc di Jupyter, kalau perlu
```

## Setup

1. **Dapatkan FIRMS MAP_KEY (gratis, ~1 menit)**
   Daftar di https://firms.modaps.eosdis.nasa.gov/api/map_key/ — key akan
   dikirim ke email kamu.

2. **Set sebagai environment variable** (jangan hardcode di kode!):
   ```bash
   export FIRMS_MAP_KEY="your_map_key_here"
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt --break-system-packages
   ```

## Cara pakai

### 1. Tarik data hotspot
```bash
# Data historis Jan-Agu 2026 (pakai sensor archive/SP karena > 60 hari lalu)
python src/fetch_firms.py --start 2026-01-01 --end 2026-08-30 --archive --out data/firms_kalimantan_2026.csv

# Data terbaru (60 hari terakhir, sensor NRT/near-real-time)
python src/fetch_firms.py --start 2026-07-01 --end 2026-08-30 --out data/firms_kalimantan_recent.csv
```

Script akan otomatis membagi rentang tanggal jadi potongan 10 hari (batasan
API FIRMS), menarik tiap sensor, menggabungkan, membuang duplikat, dan
memberi label provinsi (perkiraan, berbasis bounding box) ke tiap titik.

### 2. Eksplorasi data
```bash
python src/eda.py --input data/firms_kalimantan_2026.csv --outdir data/plots
```

Ini menghasilkan:
- `daily_hotspot_trend.png` — tren jumlah hotspot harian
- `hotspot_by_province.png` — perbandingan antar provinsi
- `confidence_distribution.png` — distribusi tingkat keyakinan deteksi
- `daily_frp_trend.png` — tren Fire Radiative Power (proxy intensitas api)

## Catatan penting soal data

- **Bounding box provinsi itu perkiraan**, bukan batas administratif asli
  (provinsi bentuknya poligon tidak beraturan, bukan kotak). Sudah cukup
  untuk eksplorasi, tapi kalau butuh akurasi tinggi (misal analisis per
  kabupaten), ganti dengan shapefile/GeoJSON asli dari Badan Informasi
  Geospasial dan lakukan spatial join pakai `geopandas`.
- **Sensor NRT vs SP**: data "near real-time" (NRT) cuma tersedia untuk
  ~60 hari terakhir. Untuk data yang lebih lama, wajib pakai sensor
  "_SP" (standard/archive product, kualitas lebih terjamin tapi ada lag
  publikasi).
- **1 kejadian kebakaran bisa menghasilkan banyak "hotspot"** — satelit
  mendeteksi per piksel, bukan per kejadian. Jangan menyamakan jumlah
  hotspot dengan jumlah kebakaran atau luas area terbakar secara langsung.

## Roadmap selanjutnya

1. ~~Data pipeline hotspot (FIRMS)~~ ✅ tahap ini
2. Deteksi area terbakar dari citra Sentinel-2 (NBR/dNBR) via Google Earth
   Engine — jadi label tambahan / ground truth
3. Feature engineering: gabungkan hotspot historis + data cuaca BMKG
   (curah hujan, kelembapan) + tinggi muka air gambut
4. Model prediksi risiko kebakaran harian per kabupaten (XGBoost/LightGBM
   atau LSTM untuk pendekatan time-series penuh)
5. (Opsional) Dashboard interaktif (Streamlit) untuk visualisasi peta risiko

## Sumber data

- [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) — hotspot kebakaran aktif
- [BMKG](https://www.bmkg.go.id/) — data cuaca/iklim (untuk tahap selanjutnya)
- [SIPONGI Kementerian Kehutanan](https://sipongi.menlhk.go.id/) — monitoring
  karhutla resmi pemerintah Indonesia
