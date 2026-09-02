import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--grid-size", type=float, default=0.5,
                         help="Ukuran grid cell dalam derajat (default 0.5 ~= 55km)")
    parser.add_argument("--month", default="2026-08", help="Bulan yang dianalisis, format YYYY-MM")
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=["acq_date"])
    df = df[df["acq_date"].dt.strftime("%Y-%m") == args.month].copy()

    if df.empty:
        print(f"Tidak ada data untuk bulan {args.month}. Cek rentang tanggal di file input.")
        return

    # Grid ke sel 0.5 derajat, pakai TITIK TENGAH sel (bukan sudut) untuk AOI
    df["grid_lat"] = (df["latitude"] / args.grid_size).round() * args.grid_size
    df["grid_lon"] = (df["longitude"] / args.grid_size).round() * args.grid_size

    grid_counts = (
        df.groupby(["grid_lat", "grid_lon"])
        .agg(hotspot_count=("latitude", "size"), total_frp=("frp", "sum"))
        .sort_values("hotspot_count", ascending=False)
    )

    print(f"=== Top 5 grid cell terpadat untuk bulan {args.month} ===\n")
    print(grid_counts.head(5))

    top = grid_counts.iloc[0]
    lat_c, lon_c = grid_counts.index[0]
    half = args.grid_size / 2

    # AOI sedikit diperlebar (buffer) supaya area sekitar burn scar ikut kelihatan
    buffer = args.grid_size * 0.3
    west, east = lon_c - half - buffer, lon_c + half + buffer
    south, north = lat_c - half - buffer, lat_c + half + buffer

    print(f"\n=== AOI TERPILIH (grid terpadat) ===")
    print(f"Pusat        : lat={lat_c}, lon={lon_c}")
    print(f"Jumlah hotspot: {int(top['hotspot_count'])}")
    print(f"Total FRP    : {top['total_frp']:.1f} MW")
    print(f"Bounding box : west={west:.3f}, south={south:.3f}, east={east:.3f}, north={north:.3f}")

    # Tentukan tanggal hotspot pertama & puncak di cell ini untuk before/after
    cell_df = df[(df["grid_lat"] == lat_c) & (df["grid_lon"] == lon_c)]
    daily = cell_df.groupby(cell_df["acq_date"].dt.date).size()
    first_date = daily.index.min()
    peak_date = daily.idxmax()

    print(f"\nTanggal hotspot pertama di area ini : {first_date}")
    print(f"Tanggal puncak hotspot di area ini   : {peak_date} ({daily.max()} hotspot)")

    print("\n=== Rekomendasi untuk burn_detection.py ===")
    print(f"--bbox {west:.3f},{south:.3f},{east:.3f},{north:.3f}")
    print(f"--before-start {pd.Timestamp(first_date) - pd.Timedelta(days=20)} "
          f"--before-end {pd.Timestamp(first_date) - pd.Timedelta(days=5)}")
    print(f"--after-start {pd.Timestamp(peak_date) + pd.Timedelta(days=3)} "
          f"--after-end {pd.Timestamp(peak_date) + pd.Timedelta(days=15)}")


if __name__ == "__main__":
    main()
