"""
analyze_features.py
====================
Tahap 3, langkah 3: cek cepat apakah fitur yang kita bikin di build_features.py
benar-benar informatif, SEBELUM buang waktu training model di Tahap 4.

Yang dicek:
1. Korelasi tiap fitur numerik terhadap target (jumlah hotspot besok) -
   fitur dengan korelasi mendekati 0 kemungkinan tidak berguna untuk model.
2. Plot time-series curah hujan kumulatif vs jumlah hotspot untuk satu
   provinsi - untuk konfirmasi visual bahwa "makin kering, makin banyak
   hotspot" benar-benar terlihat di data, bukan cuma asumsi.
3. Scatter plot hari kering berturut-turut vs jumlah hotspot - kalau pola
   naik terlihat jelas, ini pertanda fitur ini bakal berguna untuk model.

Run:
    python src/analyze_features.py --input data/feature_table.csv --outdir data/plots
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def print_correlations(df: pd.DataFrame, target_col: str):
    numeric_cols = df.select_dtypes(include="number").columns
    numeric_cols = [c for c in numeric_cols if c != target_col]

    corr = df[numeric_cols + [target_col]].corr()[target_col].drop(target_col)
    corr = corr.dropna()  # kolom dengan nilai konstan (std=0) hasilnya NaN - buang dari tampilan
    corr = corr.sort_values(key=abs, ascending=False)

    print(f"=== Korelasi tiap fitur terhadap '{target_col}' ===\n")
    for feature, value in corr.items():
        bar = "#" * int(abs(value) * 40)
        sign = "+" if value >= 0 else "-"
        print(f"  {feature:30s} {sign}{abs(value):.3f} {bar}")

    print("\nCatatan: korelasi mendekati 0 BUKAN berarti fitur itu pasti tidak")
    print("berguna (model non-linear seperti XGBoost bisa menangkap hubungan")
    print("yang tidak linear), tapi korelasi kuat adalah sinyal awal yang bagus.")


def plot_dryness_vs_fire(df: pd.DataFrame, province: str, outdir: str):
    sub = df[df["province"] == province].sort_values("date")
    if sub.empty:
        print(f"Tidak ada data untuk provinsi '{province}', lewati plot ini.")
        return

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(sub["date"], sub["hotspot_count_roll7"], color="firebrick", label="Hotspot (rolling 7 hari)")
    ax1.set_ylabel("Jumlah hotspot (rolling 7 hari)", color="firebrick")
    ax1.tick_params(axis="y", labelcolor="firebrick")
    ax1.set_xlabel("Tanggal")

    ax2 = ax1.twinx()
    ax2.plot(sub["date"], sub["rain_cumsum_30d"], color="steelblue", label="Curah hujan kumulatif 30 hari")
    ax2.set_ylabel("Curah hujan kumulatif 30 hari (mm)", color="steelblue")
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax2.invert_yaxis()  # dibalik: makin sedikit hujan (turun) ditampilkan naik, biar pola "berlawanan" gampang dilihat sejalan dengan hotspot

    plt.title(f"Hotspot vs Curah Hujan Kumulatif - {province}\n(sumbu curah hujan dibalik: turun = makin kering)")
    fig.tight_layout()

    path = os.path.join(outdir, "dryness_vs_fire.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot disimpan ke {path}")


def plot_dry_days_scatter(df: pd.DataFrame, outdir: str):
    plt.figure(figsize=(7, 5))
    plt.scatter(df["consecutive_dry_days"], df["hotspot_count"], alpha=0.3, s=15)
    plt.xlabel("Hari kering berturut-turut")
    plt.ylabel("Jumlah hotspot hari itu")
    plt.title("Hari Kering Berturut-turut vs Jumlah Hotspot")
    plt.tight_layout()

    path = os.path.join(outdir, "dry_days_scatter.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot disimpan ke {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/feature_table.csv")
    parser.add_argument("--outdir", default="data/plots")
    parser.add_argument("--province", default="Kalimantan Tengah",
                         help="Provinsi yang dipakai untuk plot time-series")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input, parse_dates=["date"])

    print_correlations(df, "target_hotspot_next_day")
    print()
    plot_dryness_vs_fire(df, args.province, args.outdir)
    plot_dry_days_scatter(df, args.outdir)


if __name__ == "__main__":
    main()
