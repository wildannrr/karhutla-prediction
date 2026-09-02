"""
burn_detection.py
==================
Tahap 2, langkah 2: deteksi area terbakar (burn scar) dengan membandingkan
citra Sentinel-2 SEBELUM dan SESUDAH periode kebakaran, menggunakan indeks
dNBR (delta Normalized Burn Ratio) - metode standar di remote sensing untuk
memetakan area terbakar.

Konsep dasar:
- NBR = (NIR - SWIR2) / (NIR + SWIR2)
  Vegetasi sehat: NIR tinggi -> NBR tinggi (positif)
  Area terbakar : NIR turun drastis, SWIR2 naik -> NBR rendah/negatif
- dNBR = NBR_sebelum - NBR_sesudah
  dNBR tinggi = perubahan besar = kemungkinan area terbakar
  Threshold umum (USGS/UN-SPIDER):
    dNBR < 0.1        : unburned / tidak terbakar
    0.1 <= dNBR < 0.27: low severity burn
    0.27 <= dNBR < 0.66: moderate severity burn
    dNBR >= 0.66      : high severity burn

SETUP (wajib dilakukan sekali sebelum menjalankan script ini):
1. Daftar Google Earth Engine (gratis, non-commercial use):
   https://code.earthengine.google.com/register
2. Saat registrasi, pilih/bikin Google Cloud Project - catat PROJECT ID-nya.
3. Install dependencies:
   pip install earthengine-api geemap --break-system-packages
4. Autentikasi (sekali saja, akan buka browser untuk login):
   earthengine authenticate

Run:
    python src/burn_detection.py \
        --bbox 113.600,-2.400,114.400,-1.600 \
        --before-start 2026-07-12 --before-end 2026-07-27 \
        --after-start 2026-08-04 --after-end 2026-08-16 \
        --project YOUR_GEE_PROJECT_ID \
        --out data/burn_map
"""

import argparse
import os
import sys

try:
    import ee
except ImportError:
    sys.exit(
        "Library 'earthengine-api' belum terinstall.\n"
        "Jalankan: pip install earthengine-api geemap --break-system-packages"
    )

sys.path.append(os.path.dirname(__file__))
from gee_utils import SEVERITY_CLASS_NAMES, compute_severity_image


def main():
    parser = argparse.ArgumentParser(description="Deteksi burn scar pakai Sentinel-2 dNBR.")
    parser.add_argument("--bbox", required=True,
                         help="west,south,east,north (contoh dari select_case_study.py)")
    parser.add_argument("--before-start", required=True)
    parser.add_argument("--before-end", required=True)
    parser.add_argument("--after-start", required=True)
    parser.add_argument("--after-end", required=True)
    parser.add_argument("--project", required=True, help="Google Cloud / Earth Engine project ID kamu")
    parser.add_argument("--max-cloud-pct", type=float, default=40,
                         help="Maksimum persentase awan per citra (default 40)")
    parser.add_argument("--out", default="data/burn_map", help="Prefix path output (tanpa ekstensi)")
    args = parser.parse_args()

    print("Menginisialisasi Google Earth Engine...")
    try:
        ee.Initialize(project=args.project)
    except Exception as exc:
        sys.exit(
            f"Gagal inisialisasi Earth Engine: {exc}\n"
            "Pastikan sudah menjalankan 'earthengine authenticate' dan project ID benar."
        )

    west, south, east, north = (float(v) for v in args.bbox.split(","))
    aoi = ee.Geometry.Rectangle([west, south, east, north])

    severity, dnbr = compute_severity_image(
        aoi, args.before_start, args.before_end, args.after_start, args.after_end, args.max_cloud_pct
    )

    # Hitung luas area per kelas severity (dalam hektare)
    pixel_area_ha = ee.Image.pixelArea().divide(10000)
    area_by_class = (
        pixel_area_ha.addBands(severity)
        .reduceRegion(
            reducer=ee.Reducer.sum().group(groupField=1, groupName="severity_class"),
            geometry=aoi,
            scale=20,
            maxPixels=1e9,
        )
    )
    result = area_by_class.getInfo()

    class_names = SEVERITY_CLASS_NAMES
    print("\n=== Estimasi luas area per tingkat keparahan (hektare) ===")
    total_burned_ha = 0
    for group in result.get("groups", []):
        cls = int(group["severity_class"])
        area_ha = group["sum"]
        print(f"  {class_names.get(cls, cls):15s}: {area_ha:,.1f} ha")
        if cls > 0:
            total_burned_ha += area_ha
    print(f"\n  TOTAL area terindikasi terbakar: {total_burned_ha:,.1f} ha")

    # Export thumbnail PNG untuk visualisasi cepat (tidak perlu Google Drive)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    vis_params = {
        "min": 0, "max": 4,
        "palette": ["ffffff", "ffff00", "ffa500", "ff4500", "8b0000"],
    }
    thumb_url = severity.getThumbURL({
        **vis_params,
        "region": aoi,
        "dimensions": 800,
        "format": "png",
    })
    print(f"\nURL thumbnail peta severity (buka di browser untuk lihat/download):\n{thumb_url}")

    # Simpan URL & ringkasan ke file teks untuk didokumentasikan
    with open(f"{args.out}_summary.txt", "w") as f:
        f.write(f"Case study bbox: {args.bbox}\n")
        f.write(f"Before: {args.before_start} - {args.before_end}\n")
        f.write(f"After:  {args.after_start} - {args.after_end}\n\n")
        f.write("Luas area per tingkat keparahan (hektare):\n")
        for group in result.get("groups", []):
            cls = int(group["severity_class"])
            f.write(f"  {class_names.get(cls, cls)}: {group['sum']:,.1f} ha\n")
        f.write(f"\nTotal area terbakar: {total_burned_ha:,.1f} ha\n")
        f.write(f"\nThumbnail URL:\n{thumb_url}\n")

    print(f"\nRingkasan disimpan ke {args.out}_summary.txt")


if __name__ == "__main__":
    main()
