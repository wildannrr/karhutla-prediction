"""
gee_utils.py
============
Fungsi-fungsi Earth Engine yang dipakai bersama oleh burn_detection.py dan
validate_with_hotspots.py, supaya logikanya konsisten di kedua script dan
tidak duplikat.
"""

import ee


def mask_clouds_s2(image):
    """Masking awan pakai band QA60 Sentinel-2 (bit 10 = awan tebal, bit 11 = cirrus)."""
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = (
        qa.bitwiseAnd(cloud_bit_mask).eq(0)
        .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )
    return image.updateMask(mask)


def get_median_composite(aoi, start_date: str, end_date: str, max_cloud_pct: float = 40):
    """Ambil median composite Sentinel-2 SR untuk suatu periode, dengan cloud masking."""
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .map(mask_clouds_s2)
    )

    count = collection.size().getInfo()
    if count == 0:
        raise RuntimeError(
            f"Tidak ada citra Sentinel-2 yang cukup bersih (<{max_cloud_pct}% awan) "
            f"untuk periode {start_date} - {end_date}. Coba perlebar rentang tanggal "
            f"atau naikkan --max-cloud-pct."
        )
    print(f"  Ditemukan {count} citra Sentinel-2 untuk periode {start_date} - {end_date}")
    return collection.median().clip(aoi)


def compute_nbr(image):
    """NBR = (NIR - SWIR2) / (NIR + SWIR2). Band Sentinel-2: B8 = NIR, B12 = SWIR2."""
    return image.normalizedDifference(["B8", "B12"]).rename("NBR")


def classify_severity(dnbr):
    """Klasifikasi tingkat keparahan burn berdasarkan threshold dNBR standar USGS.

    0 = tidak terbakar, 1 = rendah, 2 = sedang-rendah, 3 = sedang-tinggi, 4 = tinggi
    """
    return (
        ee.Image(0)
        .where(dnbr.gte(0.1).And(dnbr.lt(0.27)), 1)
        .where(dnbr.gte(0.27).And(dnbr.lt(0.44)), 2)
        .where(dnbr.gte(0.44).And(dnbr.lt(0.66)), 3)
        .where(dnbr.gte(0.66), 4)
        .rename("burn_severity")
    )


def compute_severity_image(aoi, before_start, before_end, after_start, after_end, max_cloud_pct=40):
    """Pipeline lengkap: composite before/after -> NBR -> dNBR -> severity image."""
    print("\nMengambil citra SEBELUM periode kebakaran...")
    img_before = get_median_composite(aoi, before_start, before_end, max_cloud_pct)

    print("Mengambil citra SESUDAH periode kebakaran...")
    img_after = get_median_composite(aoi, after_start, after_end, max_cloud_pct)

    print("\nMenghitung NBR dan dNBR...")
    nbr_before = compute_nbr(img_before)
    nbr_after = compute_nbr(img_after)
    dnbr = nbr_before.subtract(nbr_after).rename("dNBR")

    severity = classify_severity(dnbr)
    return severity, dnbr


SEVERITY_CLASS_NAMES = {
    0: "Tidak terbakar",
    1: "Rendah",
    2: "Sedang-rendah",
    3: "Sedang-tinggi",
    4: "Tinggi",
}
