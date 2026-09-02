"""
eda.py
======
Quick exploratory analysis on the FIRMS hotspot CSV produced by fetch_firms.py.

Run:  python src/eda.py --input data/firms_kalimantan_raw.csv --outdir data/plots
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["acq_date"])
    return df


def plot_daily_trend(df: pd.DataFrame, outdir: str):
    daily = df.groupby("acq_date").size()
    plt.figure(figsize=(11, 4))
    daily.plot()
    plt.title("Jumlah hotspot harian - Kalimantan")
    plt.xlabel("Tanggal")
    plt.ylabel("Jumlah hotspot")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "daily_hotspot_trend.png"), dpi=150)
    plt.close()


def plot_by_province(df: pd.DataFrame, outdir: str):
    plt.figure(figsize=(8, 5))
    order = df["province"].value_counts().index
    sns.countplot(data=df, y="province", order=order)
    plt.title("Total hotspot per provinsi")
    plt.xlabel("Jumlah hotspot")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "hotspot_by_province.png"), dpi=150)
    plt.close()


def plot_confidence_distribution(df: pd.DataFrame, outdir: str):
    if "confidence" not in df.columns:
        return
    plt.figure(figsize=(7, 4))
    # VIIRS confidence is categorical (l/n/h); MODIS is numeric 0-100.
    if df["confidence"].dtype == object:
        sns.countplot(data=df, x="confidence", order=["l", "n", "h"])
    else:
        sns.histplot(df["confidence"], bins=20)
    plt.title("Distribusi confidence deteksi")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confidence_distribution.png"), dpi=150)
    plt.close()


def plot_frp_over_time(df: pd.DataFrame, outdir: str):
    """FRP = Fire Radiative Power, a proxy for fire intensity."""
    if "frp" not in df.columns:
        return
    daily_frp = df.groupby("acq_date")["frp"].sum()
    plt.figure(figsize=(11, 4))
    daily_frp.plot(color="firebrick")
    plt.title("Total Fire Radiative Power (FRP) harian - proxy intensitas kebakaran")
    plt.xlabel("Tanggal")
    plt.ylabel("FRP (MW, dijumlahkan)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "daily_frp_trend.png"), dpi=150)
    plt.close()


def summarize(df: pd.DataFrame):
    print("=== Ringkasan dataset ===")
    print(f"Total deteksi hotspot : {len(df):,}")
    print(f"Rentang tanggal       : {df['acq_date'].min().date()} s.d. {df['acq_date'].max().date()}")
    print(f"Sensor yang dipakai   : {df['sensor'].unique().tolist()}")
    print("\nHotspot per provinsi:")
    print(df["province"].value_counts())
    if "frp" in df.columns:
        print(f"\nTotal FRP (proxy intensitas): {df['frp'].sum():,.1f} MW")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/firms_kalimantan_raw.csv")
    parser.add_argument("--outdir", default="data/plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load(args.input)

    summarize(df)
    plot_daily_trend(df, args.outdir)
    plot_by_province(df, args.outdir)
    plot_confidence_distribution(df, args.outdir)
    plot_frp_over_time(df, args.outdir)

    print(f"\nPlots saved to {args.outdir}/")


if __name__ == "__main__":
    main()
