import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

try:
    import ee
except ImportError:
    sys.exit(
        "Library 'earthengine-api' belum terinstall.\n"
        "Jalankan: pip install earthengine-api geemap --break-system-packages"
    )

sys.path.append(os.path.dirname(__file__))
from gee_utils import SEVERITY_CLASS_NAMES, compute_severity_image


def load_hotspots_in_aoi(csv_path, west, south, east, north, start_date, end_date):
    """Filter hotspot CSV secara spasial+temporal DI SISI PANDAS dulu (murah),
    sebelum dikirim ke Earth Engine - supaya kita tidak upload titik yang
    jelas tidak relevan.
    """
    df = pd.read_csv(csv_path, parse_dates=["acq_date"])
    mask = (
        (df["longitude"] >= west) & (df["longitude"] <= east)
        & (df["latitude"] >= south) & (df["latitude"] <= north)
        & (df["acq_date"] >= start_date) & (df["acq_date"] <= end_date)
    )
    subset = df.loc[mask, ["latitude", "longitude", "acq_date", "frp"]].copy()
    subset = subset.drop_duplicates(subset=["latitude", "longitude", "acq_date"])
    return subset.reset_index(drop=True)


def sample_severity_at_points(severity_image, points_df, scale=20, batch_size=4000):
    """Sample nilai severity Earth Engine tepat di lokasi tiap titik hotspot.

    Earth Engine membatasi getInfo() ke maksimum 5.000 elemen per panggilan,
    jadi untuk dataset hotspot yang besar kita proses per-batch dan gabungkan
    hasilnya di akhir.
    """
    points_df = points_df.copy()
    severity_by_id = {}
    n_batches = (len(points_df) + batch_size - 1) // batch_size

    for batch_num, start in enumerate(range(0, len(points_df), batch_size), start=1):
        chunk = points_df.iloc[start:start + batch_size]
        print(f"  Batch {batch_num}/{n_batches} ({len(chunk)} titik)...", end=" ")

        features = [
            ee.Feature(ee.Geometry.Point([row.longitude, row.latitude]), {"point_id": int(i)})
            for i, row in chunk.iterrows()
        ]
        fc = ee.FeatureCollection(features)
        sampled = severity_image.sampleRegions(collection=fc, scale=scale, geometries=False)
        results = sampled.getInfo()["features"]

        for f in results:
            severity_by_id[f["properties"]["point_id"]] = f["properties"].get("burn_severity")
        print(f"selesai ({len(results)} hasil)")

    points_df["severity"] = points_df.index.map(severity_by_id)
    return points_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV hotspot gabungan dari Tahap 1")
    parser.add_argument("--bbox", required=True)
    parser.add_argument("--before-start", required=True)
    parser.add_argument("--before-end", required=True)
    parser.add_argument("--after-start", required=True)
    parser.add_argument("--after-end", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--max-cloud-pct", type=float, default=80)
    parser.add_argument("--out", default="data/validation_scatter.png")
    args = parser.parse_args()

    print("Menginisialisasi Google Earth Engine...")
    ee.Initialize(project=args.project)

    west, south, east, north = (float(v) for v in args.bbox.split(","))

    print(f"\nMemuat hotspot FIRMS di dalam AOI, periode {args.after_start} - {args.after_end}...")
    points_df = load_hotspots_in_aoi(
        args.input, west, south, east, north, args.after_start, args.after_end
    )
    print(f"Ditemukan {len(points_df)} titik hotspot di dalam AOI & periode studi kasus.")

    if points_df.empty:
        sys.exit("Tidak ada hotspot di AOI/periode ini - tidak ada yang bisa divalidasi.")

    aoi = ee.Geometry.Rectangle([west, south, east, north])
    severity, _ = compute_severity_image(
        aoi, args.before_start, args.before_end, args.after_start, args.after_end, args.max_cloud_pct
    )

    print(f"\nSampling nilai severity di {len(points_df)} lokasi hotspot...")
    points_df = sample_severity_at_points(severity, points_df)

    valid = points_df.dropna(subset=["severity"])
    n_missing = len(points_df) - len(valid)
    if n_missing:
        print(f"  ({n_missing} titik di luar area tercakup citra - dilewati)")

    counts = valid["severity"].astype(int).value_counts().sort_index()
    total = len(valid)

    print("\n=== Distribusi severity Sentinel-2 di LOKASI HOTSPOT FIRMS ===")
    for cls, name in SEVERITY_CLASS_NAMES.items():
        n = counts.get(cls, 0)
        pct = n / total * 100 if total else 0
        print(f"  {name:15s}: {n:5d} titik ({pct:5.1f}%)")

    detected_pct = (total - counts.get(0, 0)) / total * 100 if total else 0
    print(f"\n  -> {detected_pct:.1f}% hotspot FIRMS berada di piksel yang JUGA terdeteksi "
          f"berubah (severity >= 1) oleh Sentinel-2.")
    print("     Angka tinggi = dua metode saling mengkonfirmasi (bagus).")
    print("     Angka rendah = kemungkinan lag waktu, resolusi berbeda, atau tutupan awan/asap.")

    # Scatter plot: posisi hotspot, warna = severity yang di-sample
    plt.figure(figsize=(8, 7))
    palette = {0: "#cccccc", 1: "#ffff00", 2: "#ffa500", 3: "#ff4500", 4: "#8b0000"}
    for cls, name in SEVERITY_CLASS_NAMES.items():
        sub = valid[valid["severity"].astype(int) == cls]
        if not sub.empty:
            plt.scatter(sub["longitude"], sub["latitude"], c=palette[cls], s=8,
                        label=f"{name} (n={len(sub)})", alpha=0.7)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Lokasi hotspot FIRMS diwarnai berdasarkan severity Sentinel-2")
    plt.legend(loc="upper right", fontsize=8, markerscale=2)
    plt.gca().set_aspect("equal")
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"\nScatter plot validasi disimpan ke {args.out}")


if __name__ == "__main__":
    main()
