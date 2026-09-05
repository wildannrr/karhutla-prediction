"""
train_risk_classifier.py
=========================
Tahap 4b: reframe masalah dari "prediksi jumlah hotspot besok secara pasti"
(regresi - lihat train_model.py) menjadi "prediksi KATEGORI RISIKO besok"
(klasifikasi: Rendah/Sedang/Tinggi/Ekstrem).

Kenapa reframe ini masuk akal:
1. Untuk kebutuhan nyata (pemda, BNPB), tahu "besok risiko EKSTREM, siaga!"
   jauh lebih actionable daripada angka pasti "diprediksi 3.847 hotspot".
   Ini mirip sistem peringatan kebakaran resmi seperti Canadian Fire
   Weather Index yang juga berbasis kategori, bukan angka presisi.
2. Model regresi (train_model.py) terbukti kalah dari baseline naif saat
   diuji pada periode lonjakan ekstrem Agustus - karena model pohon tidak
   bisa mengekstrapolasi ke ANGKA yang belum pernah dilihat saat training.
   Klasifikasi risiko lebih toleran terhadap ini: model cukup mengenali
   "pola tanda bahaya" (curah hujan kumulatif sangat rendah, tren naik
   tajam), tidak perlu menebak besaran pastinya.

PENTING - cara menentukan ambang batas kategori:
Ambang batas (threshold) dihitung HANYA dari data TRAIN (bukan keseluruhan
dataset), lalu diterapkan apa adanya ke data test. Ini mensimulasikan
kondisi nyata: di dunia nyata, kita menentukan "berapa hotspot dianggap
tinggi" berdasarkan riwayat yang SUDAH terjadi, bukan riwayat yang belum
terjadi (data test).

Run:
    python src/train_risk_classifier.py --input data/feature_table.csv --test-frac 0.2
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

try:
    from xgboost import XGBClassifier
except ImportError:
    raise SystemExit(
        "Library 'xgboost' belum terinstall.\n"
        "Windows (CMD):        pip install xgboost scikit-learn joblib\n"
        "Linux/Mac:            pip install xgboost scikit-learn joblib --break-system-packages"
    )

sys.path.append(os.path.dirname(__file__))
from train_model import FEATURE_COLS, TARGET_COL, time_based_split

RISK_CATEGORIES = ["Rendah", "Sedang", "Tinggi", "Ekstrem"]


def compute_risk_thresholds(train_target: pd.Series) -> dict:
    """Ambang batas berdasarkan persentil distribusi TARGET DI DATA TRAIN saja.

    Dipilih 50/80/95 (bukan kuartil rata 25/50/75) karena data hotspot sangat
    skewed - sebagian besar hari itu "biasa saja", cuma sedikit hari yang
    benar-benar ekstrem. Skema ini meniru pola sistem peringatan dini nyata:
    kategori "Ekstrem" harus genuinely langka, bukan 25% dari semua hari.
    """
    return {
        "q50": train_target.quantile(0.50),
        "q80": train_target.quantile(0.80),
        "q95": train_target.quantile(0.95),
    }


def assign_risk_category(values: pd.Series, thresholds: dict) -> pd.Series:
    conditions = [
        values <= thresholds["q50"],
        (values > thresholds["q50"]) & (values <= thresholds["q80"]),
        (values > thresholds["q80"]) & (values <= thresholds["q95"]),
        values > thresholds["q95"],
    ]
    return pd.Series(np.select(conditions, RISK_CATEGORIES, default=RISK_CATEGORIES[0]), index=values.index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/feature_table.csv")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--outdir", default="data")
    args = parser.parse_args()

    print("Memuat data...")
    df = pd.read_csv(args.input, parse_dates=["date"])

    province_dummies = pd.get_dummies(df["province"], prefix="prov")
    df = pd.concat([df, province_dummies], axis=1)
    feature_cols = FEATURE_COLS + list(province_dummies.columns)

    train_df, test_df, cutoff_date = time_based_split(df, args.test_frac)
    print(f"Split berdasarkan tanggal: train sebelum {cutoff_date.date()}, test mulai {cutoff_date.date()}")
    print(f"  Train: {len(train_df)} baris | Test: {len(test_df)} baris")

    X_train, y_train_raw = train_df[feature_cols], train_df[TARGET_COL]
    X_test, y_test_raw = test_df[feature_cols], test_df[TARGET_COL]

    thresholds = compute_risk_thresholds(y_train_raw)
    print(f"\nAmbang batas kategori risiko (dihitung dari data TRAIN saja):")
    print(f"  Rendah  : hotspot besok <= {thresholds['q50']:.0f}")
    print(f"  Sedang  : {thresholds['q50']:.0f} < hotspot besok <= {thresholds['q80']:.0f}")
    print(f"  Tinggi  : {thresholds['q80']:.0f} < hotspot besok <= {thresholds['q95']:.0f}")
    print(f"  Ekstrem : hotspot besok > {thresholds['q95']:.0f}")

    y_train_cat = assign_risk_category(y_train_raw, thresholds)
    y_test_cat = assign_risk_category(y_test_raw, thresholds)

    # FITUR TAMBAHAN PENTING: kategori risiko HARI INI (bukan besok), sebagai
    # fitur eksplisit. Alasan: hotspot itu sangat "lengket" dari hari ke hari
    # (kalau hari ini ekstrem, besok besar kemungkinan masih ekstrem) - baseline
    # naif menang besar justru karena hanya mengandalkan pola ini. Dengan
    # memberi model kategori hari ini secara eksplisit (bukan cuma angka
    # mentah), model punya "titik awal" sekuat baseline, lalu tinggal belajar
    # KAPAN kategori itu berubah berdasarkan tren cuaca - bukan belajar dari
    # nol pola "biasanya sama seperti kemarin".
    today_cat_train = assign_risk_category(train_df["hotspot_count"], thresholds)
    today_cat_test = assign_risk_category(test_df["hotspot_count"], thresholds)
    today_cat_dummies_train = pd.get_dummies(today_cat_train, prefix="today_risk")
    today_cat_dummies_test = pd.get_dummies(today_cat_test, prefix="today_risk")
    # Pastikan kolom dummy konsisten antara train & test (kalau ada kategori
    # yang tidak muncul di salah satu sisi, isi 0)
    for col in today_cat_dummies_train.columns:
        if col not in today_cat_dummies_test.columns:
            today_cat_dummies_test[col] = 0
    today_cat_dummies_test = today_cat_dummies_test[today_cat_dummies_train.columns]

    X_train = pd.concat([X_train.reset_index(drop=True), today_cat_dummies_train.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True), today_cat_dummies_test.reset_index(drop=True)], axis=1)

    print(f"\nDistribusi kategori di TRAIN:\n{y_train_cat.value_counts()}")
    print(f"\nDistribusi kategori di TEST:\n{y_test_cat.value_counts()}")

    cat_to_idx = {cat: i for i, cat in enumerate(RISK_CATEGORIES)}
    y_train_idx = y_train_cat.map(cat_to_idx)
    y_test_idx = y_test_cat.map(cat_to_idx)

    print("\nMelatih model klasifikasi XGBoost...")
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        objective="multi:softprob",
        num_class=len(RISK_CATEGORIES),
        random_state=42,
        eval_metric="mlogloss",
    )
    clf.fit(X_train, y_train_idx)

    pred_test_idx = clf.predict(X_test)
    pred_test_cat = pd.Series(pred_test_idx).map({v: k for k, v in cat_to_idx.items()})

    print("\n=== Laporan klasifikasi (data TEST) ===")
    print(classification_report(
        y_test_idx, pred_test_idx,
        labels=list(range(len(RISK_CATEGORIES))),
        target_names=RISK_CATEGORIES, zero_division=0,
    ))

    acc = accuracy_score(y_test_idx, pred_test_idx)
    f1_macro = f1_score(y_test_idx, pred_test_idx, average="macro",
                          labels=list(range(len(RISK_CATEGORIES))), zero_division=0)
    print(f"Akurasi keseluruhan: {acc:.3f}")
    print(f"Macro F1-score     : {f1_macro:.3f}")

    # Baseline: kategori HARI INI dipakai sebagai prediksi kategori BESOK
    print("\n=== Baseline: kategori hari ini = prediksi kategori besok ===")
    today_idx = today_cat_test.map(cat_to_idx)
    baseline_acc = accuracy_score(y_test_idx, today_idx)
    baseline_f1 = f1_score(y_test_idx, today_idx, average="macro",
                             labels=list(range(len(RISK_CATEGORIES))), zero_division=0)
    print(f"Akurasi baseline : {baseline_acc:.3f}")
    print(f"Macro F1 baseline: {baseline_f1:.3f}")

    print(f"\n  -> Model klasifikasi {'LEBIH BAIK' if f1_macro > baseline_f1 else 'LEBIH BURUK'} "
          f"dibanding baseline (diukur dari macro F1-score).")

    # Confusion matrix
    cm = confusion_matrix(y_test_idx, pred_test_idx, labels=list(range(len(RISK_CATEGORIES))))
    os.makedirs(os.path.join(args.outdir, "plots"), exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Reds")
    plt.colorbar(label="Jumlah prediksi")
    plt.xticks(range(len(RISK_CATEGORIES)), RISK_CATEGORIES, rotation=45)
    plt.yticks(range(len(RISK_CATEGORIES)), RISK_CATEGORIES)
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title("Confusion Matrix - Klasifikasi Risiko Kebakaran")
    for i in range(len(RISK_CATEGORIES)):
        for j in range(len(RISK_CATEGORIES)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "plots", "risk_confusion_matrix.png"), dpi=150)
    plt.close()

    # Feature importance - feature_cols perlu disesuaikan karena kita
    # menambahkan kolom today_risk_* setelah X_train/X_test dibentuk
    all_feature_cols = list(X_train.columns)
    importance = pd.Series(clf.feature_importances_, index=all_feature_cols).sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    importance.head(15).sort_values().plot(kind="barh")
    plt.xlabel("Feature importance")
    plt.title("Fitur paling berpengaruh - Klasifikasi Risiko")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "plots", "risk_feature_importance.png"), dpi=150)
    plt.close()

    print(f"\nPlot disimpan ke {args.outdir}/plots/risk_confusion_matrix.png "
          f"dan risk_feature_importance.png")


if __name__ == "__main__":
    main()
