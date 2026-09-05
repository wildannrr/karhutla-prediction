"""
build_features.py
==================
Tahap 3, langkah 2: gabungkan data hotspot (Tahap 1) + data cuaca (langkah
sebelumnya) jadi SATU TABEL FITUR per provinsi per hari, siap dipakai untuk
melatih model prediksi risiko kebakaran di Tahap 4.

Fitur yang dibuat:
- Jumlah hotspot & total FRP hari itu (dari data FIRMS)
- Cuaca hari itu (curah hujan, suhu, kelembapan, dst - dari Open-Meteo)
- ROLLING FEATURES (rata-rata/jumlah beberapa hari terakhir) - ini penting
  karena risiko kebakaran itu kumulatif, bukan cuma soal cuaca hari ini saja:
    - Jumlah hotspot 7 & 14 hari terakhir
    - Curah hujan kumulatif 7, 14, 30 hari terakhir (proxy kekeringan lahan)
    - Jumlah "hari kering berturut-turut" (curah hujan < 1mm) - indikator
      kuat risiko kebakaran gambut, karena gambut butuh waktu lama untuk
      benar-benar kering dan mudah terbakar
- TARGET untuk prediksi: jumlah hotspot BESOK (next-day), supaya model
  Tahap 4 belajar memprediksi 1 hari ke depan berdasarkan kondisi hari ini
  dan riwayat sebelumnya.

PENTING soal kebocoran data (data leakage): semua fitur "rolling" di sini
sengaja HANYA memakai data hingga hari H (bukan mengintip masa depan).
Target (`target_hotspot_next_day`) sengaja dipisah jelas dari fitur, supaya
saat training model nanti, tidak ada elemen kondisi "masa depan" yang
bocor ke input fitur.

Run:
    python src/build_features.py \
        --hotspot-input data/firms_kalimantan_2026_combined.csv \
        --weather-input data/weather_by_province.csv \
        --out data/feature_table.csv
"""

import argparse
import sys

import pandas as pd


def build_hotspot_daily(hotspot_df: pd.DataFrame) -> pd.DataFrame:
    """Agregasi hotspot dari level titik individual -> level provinsi per hari."""
    hotspot_df = hotspot_df.copy()
    hotspot_df["date"] = pd.to_datetime(hotspot_df["acq_date"]).dt.date

    daily = (
        hotspot_df.groupby(["province", "date"])
        .agg(hotspot_count=("latitude", "size"), total_frp=("frp", "sum"))
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def fill_date_gaps(df: pd.DataFrame, date_col: str, group_col: str) -> pd.DataFrame:
    """Isi tanggal yang tidak ada hotspot dengan 0 (bukan NaN) - penting supaya
    rolling window tidak salah hitung karena melompati hari tanpa kebakaran.
    """
    filled = []
    for group, sub in df.groupby(group_col):
        full_range = pd.date_range(sub[date_col].min(), sub[date_col].max(), freq="D")
        sub = sub.set_index(date_col).reindex(full_range)
        sub[group_col] = group
        sub = sub.fillna(0)
        sub.index.name = date_col
        filled.append(sub.reset_index())
    return pd.concat(filled, ignore_index=True)


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan fitur rolling (hotspot & cuaca) per provinsi, terurut tanggal."""
    df = df.sort_values(["province", "date"]).reset_index(drop=True)
    out_frames = []

    for province, sub in df.groupby("province"):
        sub = sub.sort_values("date").reset_index(drop=True)

        # Rolling hotspot: rata-rata & jumlah 7/14 hari terakhir (termasuk hari ini)
        sub["hotspot_count_roll7"] = sub["hotspot_count"].rolling(7, min_periods=1).sum()
        sub["hotspot_count_roll14"] = sub["hotspot_count"].rolling(14, min_periods=1).sum()
        sub["frp_roll7"] = sub["total_frp"].rolling(7, min_periods=1).sum()

        # Curah hujan kumulatif - proxy kekeringan lahan gambut
        if "precipitation_sum" in sub.columns:
            sub["rain_cumsum_7d"] = sub["precipitation_sum"].rolling(7, min_periods=1).sum()
            sub["rain_cumsum_14d"] = sub["precipitation_sum"].rolling(14, min_periods=1).sum()
            sub["rain_cumsum_30d"] = sub["precipitation_sum"].rolling(30, min_periods=1).sum()

            # Hari kering berturut-turut (curah hujan < 1mm dianggap "kering")
            is_dry = (sub["precipitation_sum"] < 1.0).astype(int)
            # Hitung consecutive run of 1s yang berakhir di tiap baris
            groups = (is_dry != is_dry.shift()).cumsum()
            sub["consecutive_dry_days"] = is_dry.groupby(groups).cumsum() * is_dry

        # TARGET: jumlah hotspot besok (next-day) - dipisah jelas dari fitur input
        sub["target_hotspot_next_day"] = sub["hotspot_count"].shift(-1)

        out_frames.append(sub)

    return pd.concat(out_frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotspot-input", required=True)
    parser.add_argument("--weather-input", required=True)
    parser.add_argument("--out", default="data/feature_table.csv")
    args = parser.parse_args()

    print("Memuat data hotspot...")
    hotspot_raw = pd.read_csv(args.hotspot_input)
    # Fokus ke 5 provinsi Indonesia saja - buang Malaysia/Brunei & Unknown
    # karena tidak ada data cuaca pembanding untuk kategori itu.
    valid_provinces = [
        "Kalimantan Barat", "Kalimantan Tengah", "Kalimantan Selatan",
        "Kalimantan Timur", "Kalimantan Utara",
    ]
    hotspot_raw = hotspot_raw[hotspot_raw["province"].isin(valid_provinces)]
    hotspot_daily = build_hotspot_daily(hotspot_raw)
    hotspot_daily = fill_date_gaps(hotspot_daily, "date", "province")
    print(f"  {len(hotspot_daily)} baris (provinsi x hari) hotspot setelah agregasi.")

    print("Memuat data cuaca...")
    weather = pd.read_csv(args.weather_input, parse_dates=["date"])
    print(f"  {len(weather)} baris data cuaca.")

    print("Menggabungkan hotspot + cuaca...")
    merged = pd.merge(hotspot_daily, weather, on=["province", "date"], how="inner")
    n_dropped = len(hotspot_daily) - len(merged)
    if n_dropped:
        print(f"  Peringatan: {n_dropped} baris hotspot tidak punya pasangan data cuaca "
              f"(kemungkinan di luar rentang tanggal weather) - baris ini dibuang.")

    print("Menghitung fitur rolling & target...")
    features = add_rolling_features(merged)

    # Buang baris terakhir per provinsi (target_hotspot_next_day = NaN, karena
    # tidak ada "besok" untuk hari terakhir dalam data)
    features = features.dropna(subset=["target_hotspot_next_day"])

    features.to_csv(args.out, index=False)
    print(f"\nTabel fitur final: {len(features)} baris, {len(features.columns)} kolom")
    print(f"Disimpan ke {args.out}")
    print(f"\nKolom yang tersedia:\n{list(features.columns)}")


if __name__ == "__main__":
    main()
