"""Fetch real ground elevation for each study ward.

Source: api.opentopodata.org, SRTM 30 m dataset. Free, no API key.

SRTM measures bare-earth height, so a single 30 m sample in a built-up ward
reads low compared to street level. These values are therefore used as a
RELATIVE terrain index between wards (low ground vs high ground) rather than
as absolute flood-depth contours, which is the correct use for them here.
"""

import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config

API = "https://api.opentopodata.org/v1/srtm30m"
TIMEOUT = 60


def main() -> None:
    wards = config.WARDS
    locations = "|".join(f"{lat},{lon}" for _, lat, lon, _, _ in wards)
    print(f"Fetching SRTM30m elevation for {len(wards)} wards")

    r = requests.get(API, params={"locations": locations}, timeout=TIMEOUT)
    r.raise_for_status()
    results = r.json()["results"]

    rows = []
    for (name, lat, lon, cached, expect), res in zip(wards, results):
        elev = float(res["elevation"])
        rows.append({
            "ward": name,
            "lat": lat,
            "lon": lon,
            "elevation_m": elev,
            # relative index: 0 = lowest ward, 1 = highest ward
            "elevation_index": 0.0,  # filled below
            "flood_2015_expectation": expect,
            "source": "real:opentopodata/srtm30m",
        })

    df = pd.DataFrame(rows)
    lo, hi = df["elevation_m"].min(), df["elevation_m"].max()
    span = (hi - lo) or 1.0
    df["elevation_index"] = (df["elevation_m"] - lo) / span

    # The elevations in config were pre-verified on 2026-09-28. Flag any drift
    # so a silent API change can never quietly alter the study area.
    df["matches_config"] = [
        abs(got - cached) < 0.51
        for got, cached in zip(df["elevation_m"], [w[3] for w in wards])
    ]

    out = config.RAW / "elevations.csv"
    df.to_csv(out, index=False)

    print(f"  wrote {out}")
    for row in df.itertuples():
        flag = "" if row.matches_config else "  <-- DIFFERS FROM CONFIG"
        print(f"    {row.ward:16s} {row.elevation_m:6.1f} m  "
              f"idx={row.elevation_index:.2f}  {row.flood_2015_expectation}{flag}")
    drift = int((~df["matches_config"]).sum())
    if drift:
        print(f"  WARNING: {drift} ward(s) differ from the elevations in config.py")
    else:
        print("  all 15 elevations match the values frozen in config.py")


if __name__ == "__main__":
    main()
