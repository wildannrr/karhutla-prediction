"""
diagnose_unknown.py
====================
Quick diagnostic: shows the lat/lon range of hotspots that didn't match any
province bounding box, so we can tighten up config.PROVINCE_BBOXES with real
numbers instead of guessing.

Run:  python src/diagnose_unknown.py --input data/firms_kalimantan_2026_combined.csv
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    unknown = df[df["province"] == "Unknown"]

    print(f"Total 'Unknown' points: {len(unknown)} / {len(df)} ({len(unknown)/len(df)*100:.1f}%)\n")

    if unknown.empty:
        print("No unknown points - nothing to fix!")
        return

    print("=== Lat/Lon range of Unknown points ===")
    print(f"Latitude:  {unknown['latitude'].min():.3f} to {unknown['latitude'].max():.3f}")
    print(f"Longitude: {unknown['longitude'].min():.3f} to {unknown['longitude'].max():.3f}")

    # Bucket into a coarse 1-degree grid so we can see WHERE the gaps cluster
    unknown = unknown.copy()
    unknown["lat_bucket"] = unknown["latitude"].round(0)
    unknown["lon_bucket"] = unknown["longitude"].round(0)

    print("\n=== Top 15 grid cells (1-degree) with most Unknown points ===")
    grid_counts = (
        unknown.groupby(["lat_bucket", "lon_bucket"])
        .size()
        .sort_values(ascending=False)
        .head(15)
    )
    print(grid_counts)

    print("\nUse these lat/lon clusters to widen or add province bounding boxes in config.py.")


if __name__ == "__main__":
    main()
