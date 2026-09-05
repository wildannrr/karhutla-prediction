"""
train_model.py
===============
Tahap 4: latih model prediksi jumlah hotspot BESOK, berdasarkan kondisi
hari ini + riwayat cuaca & hotspot (fitur dari build_features.py).

Keputusan desain penting:

1. SPLIT BERDASARKAN WAKTU, BUKAN ACAK. Data ini time-series - kalau kita
   split acak (random train_test_split), model bisa "curi lihat" pola dari
   masa depan saat training (data leakage temporal), dan hasil evaluasinya
   akan terlihat bagus padahal di dunia nyata tidak akan seakurat itu. Jadi
   kita latih pakai bulan-bulan awal, uji pakai bulan-bulan terakhir -
   mensimulasikan kondisi nyata: memprediksi hari yang belum terjadi.

2. DIBANDINGKAN DENGAN BASELINE NAIF ("besok = hari ini"). Ini penting
   karena hotspot_count sendiri adalah fitur dengan korelasi tertinggi
   (0.91) - jadi model yang "malas" (cuma menyalin angka hari ini) bisa
   saja sudah kelihatan lumayan akurat. Kita perlu buktikan model ML kita
   BENAR-BENAR lebih baik dari sekadar menyalin, bukan cuma "keliatan
   akurat" karena datanya memang punya inersia tinggi.

Run:
    python src/train_model.py --input data/feature_table.csv --test-frac 0.2
"""

import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except ImportError:
    raise SystemExit(
        "Library 'xgboost' belum terinstall.\n"
        "Windows (CMD):        pip install xgboost scikit-learn joblib\n"
        "Linux/Mac:            pip install xgboost scikit-learn joblib --break-system-packages"
    )


FEATURE_COLS = [
    "hotspot_count", "total_frp",
    "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "rain_sum",
    "wind_speed_10m_max", "shortwave_radiation_sum", "et0_fao_evapotranspiration",
    "relative_humidity_2m_mean",
    "hotspot_count_roll7", "hotspot_count_roll14", "frp_roll7",
    "rain_cumsum_7d", "rain_cumsum_14d", "rain_cumsum_30d",
    "consecutive_dry_days",
]
TARGET_COL = "target_hotspot_next_day"


def time_based_split(df: pd.DataFrame, test_frac: float):
    """Split berdasarkan tanggal (bukan acak) - semua provinsi dipotong di
    tanggal yang sama, supaya simulasinya realistis: "kita di titik waktu X,
    prediksi apa yang terjadi setelahnya."
    """
    unique_dates = sorted(df["date"].unique())
    cutoff_idx = int(len(unique_dates) * (1 - test_frac))
    cutoff_date = unique_dates[cutoff_idx]

    train = df[df["date"] < cutoff_date]
    test = df[df["date"] >= cutoff_date]
    return train, test, cutoff_date


def evaluate(y_true, y_pred, label: str):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  [{label}] MAE={mae:.1f}  RMSE={rmse:.1f}  R2={r2:.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/feature_table.csv")
    parser.add_argument("--test-frac", type=float, default=0.2,
                         help="Proporsi data PALING BARU yang dipakai untuk testing (default 20%%)")
    parser.add_argument("--outdir", default="data")
    args = parser.parse_args()

    print("Memuat data...")
    df = pd.read_csv(args.input, parse_dates=["date"])

    # One-hot encode provinsi - biarkan model belajar "baseline" tiap provinsi
    # berbeda (Kalteng secara historis emang lebih rawan daripada Kalut, dst)
    province_dummies = pd.get_dummies(df["province"], prefix="prov")
    df = pd.concat([df, province_dummies], axis=1)
    feature_cols = FEATURE_COLS + list(province_dummies.columns)

    train_df, test_df, cutoff_date = time_based_split(df, args.test_frac)
    print(f"Split berdasarkan tanggal: train sebelum {cutoff_date.date()}, test mulai {cutoff_date.date()}")
    print(f"  Train: {len(train_df)} baris | Test: {len(test_df)} baris")

    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]

    # PENTING: kita latih model pada skala LOG (log1p), bukan skala asli.
    # Alasannya: jumlah hotspot itu sangat skewed (banyak hari dengan angka
    # kecil, sedikit hari dengan lonjakan ekstrem seperti Agustus kita).
    # Model pohon (XGBoost) memprediksi dengan "meniru" rentang nilai yang
    # pernah dilihat saat training - dia TIDAK BISA mengekstrapolasi ke
    # angka yang jauh lebih besar dari yang pernah ada di data training.
    # Bekerja di skala log mengecilkan rentang itu, sehingga model lebih
    # mampu menangkap LAJU PERUBAHAN (relatif) meski angka absolutnya
    # belum pernah terlihat sebelumnya.
    y_train_log = np.log1p(y_train)

    print("\nMelatih model XGBoost (skala log)...")
    model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        random_state=42,
    )
    model.fit(X_train, y_train_log)

    print("\n=== Evaluasi (setelah dikembalikan ke skala asli) ===")
    pred_train = np.expm1(model.predict(X_train))
    pred_test = np.expm1(model.predict(X_test))
    evaluate(y_train, pred_train, "Train")
    evaluate(y_test, pred_test, "Test (data belum pernah dilihat model)")

    # Baseline naif: "prediksi besok = hotspot_count hari ini"
    print("\n=== Baseline naif (besok = hari ini) - pembanding ===")
    naive_pred_test = test_df["hotspot_count"]
    evaluate(y_test, naive_pred_test, "Baseline naif (Test)")

    improvement = (
        (mean_absolute_error(y_test, naive_pred_test) - mean_absolute_error(y_test, pred_test))
        / mean_absolute_error(y_test, naive_pred_test) * 100
    )
    print(f"\n  -> Model XGBoost {'LEBIH BAIK' if improvement > 0 else 'LEBIH BURUK'} "
          f"{abs(improvement):.1f}% dibanding baseline naif (diukur dari MAE).")
    if improvement <= 0:
        print("     Catatan: ini BUKAN berarti proyeknya gagal - model berbasis pohon (XGBoost)")
        print("     memang punya keterbatasan mendasar dalam mengekstrapolasi ke kondisi yang")
        print("     belum pernah terlihat di data training (misal lonjakan tak terduga). Kalau")
        print("     baseline naif menang, itu insight teknis yang valid untuk didokumentasikan,")
        print("     bukan sesuatu yang perlu disembunyikan.")

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n=== 10 fitur paling penting menurut model ===")
    print(importance.head(10))

    os.makedirs(os.path.join(args.outdir, "plots"), exist_ok=True)

    plt.figure(figsize=(8, 6))
    importance.head(15).sort_values().plot(kind="barh")
    plt.xlabel("Feature importance (XGBoost)")
    plt.title("Fitur paling berpengaruh terhadap prediksi hotspot besok")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "plots", "feature_importance.png"), dpi=150)
    plt.close()

    # Plot prediksi vs aktual di periode test - paling penting untuk lihat
    # apakah model benar-benar menangkap LONJAKAN Agustus, bukan cuma rata-rata
    plt.figure(figsize=(12, 5))
    test_plot = test_df.copy()
    test_plot["prediction"] = pred_test
    for province, sub in test_plot.groupby("province"):
        sub = sub.sort_values("date")
        plt.plot(sub["date"], sub[TARGET_COL], label=f"{province} - Aktual", alpha=0.4)
        plt.plot(sub["date"], sub["prediction"], label=f"{province} - Prediksi", linestyle="--")
    plt.xlabel("Tanggal")
    plt.ylabel("Jumlah hotspot")
    plt.title("Prediksi vs Aktual - Periode Test")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "plots", "prediction_vs_actual.png"), dpi=150)
    plt.close()

    model_path = os.path.join(args.outdir, "fire_risk_model.joblib")
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
    print(f"\nModel disimpan ke {model_path}")
    print(f"Plot disimpan ke {args.outdir}/plots/feature_importance.png dan prediction_vs_actual.png")


if __name__ == "__main__":
    main()
