"""
fetch_weather.py
=================
Tahap 3, langkah 1: ambil data cuaca historis (curah hujan, kelembapan, suhu,
dll) untuk tiap provinsi di Kalimantan, sebagai bahan feature engineering
untuk model prediksi risiko kebakaran.

Kenapa Open-Meteo, bukan BMKG?
BMKG memang API resminya Indonesia, tapi API publik mereka cuma menyediakan
PRAKIRAAN 3 hari ke depan per kode wilayah kelurahan - tidak ada endpoint
sederhana untuk data cuaca HISTORIS per koordinat bebas. Open-Meteo
menyediakan data reanalysis (ERA5/ECMWF - dipakai juga oleh BMKG dan
lembaga cuaca lain di dunia) secara gratis, tanpa API key, untuk koordinat
manapun di dunia termasuk Kalimantan.

Pendekatan: daripada narik cuaca per titik hotspot (jutaan titik, boros),
kita ambil 1 titik representatif (titik tengah bounding box) per provinsi -
cukup untuk analisis skala provinsi seperti yang kita pakai di Tahap 1 & 2.

Run:
    python src/fetch_weather.py --start 2026-01-01 --end 2026-08-30 --out data/weather_by_province.csv
"""

import argparse
import os
import sys
import time

import pandas as pd
import requests

sys.path.append(os.path.dirname(__file__))
from config import PROVINCE_BBOXES

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# Variabel harian yang kita minta. precipitation_sum & rain_sum penting untuk
# proxy kekeringan lahan gambut; et0 (evapotranspiration) & shortwave radiation
# jadi indikator tambahan potensi kekeringan.
DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "wind_speed_10m_max",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]
# Kelembapan relatif harian: nama parameter ini kurang terdokumentasi jelas
# di API Open-Meteo (masuk kategori "additional variables"), jadi kita coba
# sertakan, tapi fetch_province() akan otomatis retry TANPA parameter ini
# kalau ternyata ditolak API.
OPTIONAL_DAILY_VARS = ["relative_humidity_2m_mean"]


def province_centroid(bbox):
    west, south, east, north = bbox
    return (south + north) / 2, (west + east) / 2  # (lat, lon)


def fetch_province(province: str, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    daily_vars = DAILY_VARS + OPTIONAL_DAILY_VARS
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(daily_vars),
        "timezone": "Asia/Jakarta",
    }

    resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
    data = resp.json()

    if data.get("error"):
        # Kemungkinan besar gara-gara OPTIONAL_DAILY_VARS ditolak - retry tanpa itu.
        print(f"  Peringatan: {data.get('reason')}. Mencoba ulang tanpa variabel opsional...")
        params["daily"] = ",".join(DAILY_VARS)
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
        data = resp.json()
        if data.get("error"):
            raise RuntimeError(f"Gagal ambil data cuaca untuk {province}: {data.get('reason')}")

    daily = data["daily"]
    df = pd.DataFrame(daily)
    df["province"] = province
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="Format YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Format YYYY-MM-DD")
    parser.add_argument("--out", default="data/weather_by_province.csv")
    args = parser.parse_args()

    frames = []
    for province, bbox in PROVINCE_BBOXES.items():
        lat, lon = province_centroid(bbox)
        print(f"Mengambil data cuaca untuk {province} (titik representatif: {lat:.2f}, {lon:.2f})...")
        try:
            df = fetch_province(province, lat, lon, args.start, args.end)
            frames.append(df)
            print(f"  Berhasil: {len(df)} hari data.")
        except Exception as exc:
            print(f"  GAGAL: {exc}")
        time.sleep(1)  # jaga-jaga, sopan ke API gratis

    if not frames:
        sys.exit("Tidak ada data cuaca yang berhasil diambil.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"time": "date"})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    combined.to_csv(args.out, index=False)
    print(f"\nDisimpan {len(combined)} baris data cuaca ke {args.out}")
    print(f"Provinsi yang berhasil: {combined['province'].unique().tolist()}")


if __name__ == "__main__":
    main()
