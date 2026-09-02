import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests

sys.path.append(os.path.dirname(__file__))
from config import (
    FIRMS_AREA_URL_TEMPLATE,
    KALIMANTAN_BBOX,
    MALAYSIA_BRUNEI_LAT_THRESHOLD,
    MAX_DAY_RANGE,
    PROVINCE_BBOXES,
    SENSORS_ARCHIVE,
    SENSORS_NRT,
)


def daterange_chunks(start: date, end: date, step_days: int = MAX_DAY_RANGE):
    """Yield (chunk_end_date, num_days) pairs covering [start, end] inclusive.

    FIRMS' area API takes a single `date` plus a `day_range` and returns
    that many days *ending on* `date`. So we walk forward in `step_days`
    windows and, for each window, request it using its *last* day as the
    anchor date.
    """
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=step_days - 1), end)
        num_days = (chunk_end - current).days + 1
        yield chunk_end, num_days
        current = chunk_end + timedelta(days=1)


def fetch_chunk(map_key: str, sensor: str, bbox, chunk_end: date, num_days: int) -> pd.DataFrame:
    bbox_str = ",".join(str(v) for v in bbox)
    url = FIRMS_AREA_URL_TEMPLATE.format(
        map_key=map_key,
        sensor=sensor,
        bbox=bbox_str,
        day_range=num_days,
        date=chunk_end.isoformat(),
    )
    # Some servers/WAFs reject requests with the default python-requests
    # User-Agent (treated as bot traffic) even though the exact same URL
    # works fine in a browser. Sending a browser-like User-Agent fixes it.
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()

    if resp.text.strip().lower().startswith("invalid") or "error" in resp.text[:200].lower():
        raise RuntimeError(f"FIRMS API returned an error for {sensor} @ {chunk_end}: {resp.text[:200]}")

    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))
    if not df.empty:
        df["sensor"] = sensor
    return df


def tag_province(row) -> str:
    lat, lon = row["latitude"], row["longitude"]
    for province, (w, s, e, n) in PROVINCE_BBOXES.items():
        if w <= lon <= e and s <= lat <= n:
            return province
    # Not in any Indonesian province box. Likely Malaysia (Sarawak/Sabah) or
    # Brunei rather than a real gap in Indonesian territory - see the note
    # above MALAYSIA_BRUNEI_LAT_THRESHOLD in config.py.
    if lat >= MALAYSIA_BRUNEI_LAT_THRESHOLD:
        return "Malaysia/Brunei (Borneo)"
    return "Unknown (perbatasan/laut - cek manual)"


def fetch_range(map_key: str, start: date, end: date, sensors, bbox, pause_sec: float = 1.0) -> pd.DataFrame:
    frames = []
    chunks = list(daterange_chunks(start, end))
    total_calls = len(chunks) * len(sensors)
    call_num = 0

    for sensor in sensors:
        for chunk_end, num_days in chunks:
            call_num += 1
            print(f"[{call_num}/{total_calls}] {sensor}: last {num_days}d ending {chunk_end} ...", end=" ")
            try:
                df = fetch_chunk(map_key, sensor, bbox, chunk_end, num_days)
                print(f"{len(df)} rows")
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                print(f"FAILED ({exc})")
            time.sleep(pause_sec)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # De-duplicate: the same detection can appear in overlapping chunks/sensors.
    dedup_cols = [c for c in ["latitude", "longitude", "acq_date", "acq_time", "sensor"] if c in combined.columns]
    combined = combined.drop_duplicates(subset=dedup_cols)

    if {"latitude", "longitude"}.issubset(combined.columns):
        combined["province"] = combined.apply(tag_province, axis=1)

    return combined.sort_values("acq_date").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Fetch NASA FIRMS hotspot data for Kalimantan.")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--archive", action="store_true",
                         help="Use archive/SP sensors (needed for dates older than ~60 days)")
    parser.add_argument("--out", default="data/firms_kalimantan_raw.csv", help="Output CSV path")
    args = parser.parse_args()

    map_key = os.environ.get("FIRMS_MAP_KEY")
    if not map_key:
        sys.exit(
            "ERROR: Set your FIRMS MAP_KEY first:\n"
            "  export FIRMS_MAP_KEY='your_map_key_here'\n"
            "Get a free one at https://firms.modaps.eosdis.nasa.gov/api/map_key/"
        )

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    sensors = SENSORS_ARCHIVE if args.archive else SENSORS_NRT

    print(f"Fetching {sensors} for Kalimantan bbox {KALIMANTAN_BBOX}, {start} to {end}...\n")
    df = fetch_range(map_key, start, end, sensors, KALIMANTAN_BBOX)

    if df.empty:
        print("No data returned. Check your MAP_KEY, date range, and sensor choice.")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} hotspot records to {args.out}")
    print(df["province"].value_counts())


if __name__ == "__main__":
    main()
