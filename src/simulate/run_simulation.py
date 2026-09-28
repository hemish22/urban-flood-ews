"""Run the hydrology simulator and report calibration against the 2015 event.

Checks three things:
  1. Physical sanity over the full 6-year series (mass balance, no drift).
  2. The event window peaks at the documented intensity after bias correction.
  3. The flood / no-flood pattern across the 15 real wards matches what was
     actually documented in Dec 2015.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.simulate import hydrology

OUT = config.PROCESSED / "hourly_panel.parquet"


def main() -> None:
    rain = pd.read_csv(config.RAW / "rainfall.csv", parse_dates=["timestamp"])
    elev = pd.read_csv(config.RAW / "elevations.csv")

    print("=" * 68)
    print("1. FULL-SERIES SIMULATION (real ERA5 rainfall, 2010-2015)")
    print("=" * 68)

    # ERA5 under-resolves convective extremes, so apply the same correction
    # used for the replay to the training series. Models are then trained and
    # evaluated on a consistent rainfall distribution.
    series_dates = rain["timestamp"].dt.strftime("%Y-%m-%d").to_numpy()
    series_corrected = hydrology.bias_series_rainfall(
        rain["rainfall_mm"].to_numpy(), series_dates, config.EVENT_DAILY_TARGETS_MM
    )
    rain_c = rain.copy()
    rain_c["rainfall_mm"] = series_corrected

    d_raw = rain["rainfall_mm"].groupby(rain["timestamp"].dt.date).sum()
    d_cor = rain_c["rainfall_mm"].groupby(rain_c["timestamp"].dt.date).sum()
    print(f"  ERA5 max daily        {d_raw.max():7.1f} mm  ({d_raw.idxmax()})")
    print(f"  corrected max daily   {d_cor.max():7.1f} mm  ({d_cor.idxmax()})")
    print(f"  days > 100 mm         raw {int((d_raw > 100).sum()):3d}   "
          f"corrected {int((d_cor > 100).sum()):3d}")
    print(f"  days > 200 mm         raw {int((d_raw > 200).sum()):3d}   "
          f"corrected {int((d_cor > 200).sum()):3d}")

    # persist the exact rainfall the panel is simulated from, so feature
    # building can never drift out of sync with it
    rain_c[["timestamp", "rainfall_mm", "soil_moisture", "source"]].to_csv(
        config.PROCESSED / "rainfall_corrected.csv", index=False)
    print(f"  wrote {config.PROCESSED / 'rainfall_corrected.csv'}")

    panel = hydrology.run_series(rain_c, elev)

    for col in ("depth_mm", "surface_mm", "soil_mm"):
        assert (panel[col] >= -1e-9).all(), f"{col} went negative"

    # WATER BALANCE, per ward. Verified identity:
    #   rain = surface_out + et + d(surface) + d(soil) - nudge
    # where surface_out is everything leaving the surface store (runoff to
    # drain, gravity outfall, percolation into soil). Checking this per ward in
    # depth units is the only correct form: the wards have different areas, so
    # summing losses across all 15 and comparing to one hour of rain compares
    # one ward's rainfall against fifteen wards' water and reports a fake gap.
    balance = []
    for name, w in panel.groupby("ward", sort=False):
        w = w.sort_values("timestamp")
        ds = w["surface_mm"].diff().fillna(0.0)
        dz = w["soil_mm"].diff().fillna(0.0)
        # runoff_mm is the rain-not-infiltrated flux INTO the surface store,
        # and ds already contains it. Counting it as a sink double-counts it.
        out = w["drain_out_mm"]
        lhs = w["rainfall_mm"] + w["nudging_mm"]
        rhs = out + w["et_loss_mm"] + ds + dz
        balance.append(pd.DataFrame({
            "ward": name, "timestamp": w["timestamp"],
            "resid": lhs - rhs,
        }))
    bal = pd.concat(balance, ignore_index=True).iloc[1:]
    per_ward = bal.groupby("ward")["resid"].apply(lambda s: s.abs().mean())
    print(f"  water balance, per ward, mean|resid|    "
          f"{per_ward.mean():.2e} mm/hr   worst {per_ward.max():.2e}")
    # 6-year closure. FLUX terms sum; STOCK terms (surface, soil) must enter as
    # a final-minus-initial difference. Summing a standing stock over 52,000
    # hours inflates it by five orders of magnitude and looks like a huge leak.
    tot_rain = float(panel.groupby("ward")["rainfall_mm"].sum().mean())
    wsum = panel.groupby("ward")[["rainfall_mm", "drain_out_mm",
                                  "et_loss_mm", "nudging_mm"]].sum()
    wend = panel.groupby("ward")[["surface_mm", "soil_mm"]].last()
    wfirst = panel.groupby("ward")[["surface_mm", "soil_mm"]].first()
    lhs_t = wsum["rainfall_mm"] + wsum["nudging_mm"]
    rhs_t = (wsum["drain_out_mm"] + wsum["et_loss_mm"]
             + (wend["surface_mm"] - wfirst["surface_mm"])
             + (wend["soil_mm"] - wfirst["soil_mm"]))
    tot_gap = float((lhs_t - rhs_t).abs().mean())
    rel_gap = tot_gap / tot_rain * 100
    print(f"  water balance, 6-year total per ward    "
          f"rain {tot_rain:,.0f} mm   closure gap {tot_gap:,.2f} mm "
          f"({rel_gap:.4f}% of rainfall)")
    print(f"  depth range                    {panel['depth_mm'].min():.1f} .. "
          f"{panel['depth_mm'].max():.1f} mm")
    print(f"  hours above flood threshold    "
          f"{int((panel['depth_mm'] > config.FLOOD_DEPTH_MM).sum()):,} of {len(panel):,} "
          f"({(panel['depth_mm'] > config.FLOOD_DEPTH_MM).mean() * 100:.2f}%)")
    print(f"  mean depth by ward (mm), sorted:")
    mw = panel.groupby("ward")["depth_mm"].mean().sort_values(ascending=False)
    for w, v in mw.items():
        bar = "#" * int(v * 1.5)
        print(f"    {w:16s} {v:6.1f}  {bar}")

    print()
    print("=" * 68)
    print("2. 2015 EVENT REPLAY  (ERA5 shape, gauge magnitude)")
    print("=" * 68)
    ev = config.EVENT_2015
    mask = (rain["timestamp"] >= ev["start"]) & (rain["timestamp"] <= ev["end"])
    ev_rain = rain.loc[mask]
    corrected = hydrology.bias_series_rainfall(
        ev_rain["rainfall_mm"].to_numpy(),
        ev_rain["timestamp"].dt.strftime("%Y-%m-%d").to_numpy(),
        config.EVENT_DAILY_TARGETS_MM,
    )

    era5_daily = ev_rain.groupby(ev_rain["timestamp"].dt.date)["rainfall_mm"].sum()
    corr_daily = (
        pd.Series(corrected, index=ev_rain["timestamp"].dt.date)
        .groupby(level=0).sum()
    )
    print(f"  window {ev['start']} .. {ev['end']}  ({len(ev_rain)} h)")
    print(f"  ERA5 raw total          {era5_daily.sum():7.1f} mm")
    print(f"  bias-corrected total    {corr_daily.sum():7.1f} mm")
    print(f"  documented peak day     {ev['documented_peak_daily_mm']:7.1f} mm")
    print(f"  corrected peak day      {corr_daily.max():7.1f} mm")
    peak_ok = 0.6 <= corr_daily.max() / ev["documented_peak_daily_mm"] <= 1.5
    print(f"  peak day within 0.6-1.5x of documented: {peak_ok}")

    # replay the event with antecedent soil state from the real series
    ev_panel = hydrology.run_series(ev_rain, elev, rain_override=corrected)
    ev_panel.to_parquet(config.EVENTS / "event_2015_replay.parquet", index=False)
    print(f"  wrote {config.EVENTS / 'event_2015_replay.parquet'}")

    print()
    print("=" * 68)
    print("3. FLOOD / NO-FLOOD PATTERN vs DOCUMENTED OUTCOME")
    print("=" * 68)
    flood_depth = config.FLOOD_DEPTH_MM
    rows = []
    for ward, grp in ev_panel.groupby("ward"):
        hours_above = int((grp["depth_mm"] > flood_depth).sum())
        peak = float(grp["depth_mm"].max())
        cls = hydrology.physics_class(hours_above, peak)
        expect = grp["flood_2015_expectation"].iloc[0] if "flood_2015_expectation" in grp else None
        rows.append({
            "ward": ward,
            "elev_m": elev.set_index("ward").loc[ward, "elevation_m"],
            "peak_depth_mm": round(peak, 1),
            "hours_above_flood": hours_above,
            "physics_class": cls,
            "documented_2015": expect,
        })

    expect_map = dict(zip(elev["ward"], elev["flood_2015_expectation"]))
    for r in rows:
        r["documented_2015"] = expect_map[r["ward"]]

    res = pd.DataFrame(rows).sort_values("peak_depth_mm", ascending=False)
    print(f"  {'ward':16s} {'elev':>6s} {'peak mm':>9s} {'hrs>100':>8s} "
          f"{'class':10s} {'documented':12s} match")
    agree = 0
    for r in res.itertuples():
        flooded = r.peak_depth_mm > flood_depth
        should = r.documented_2015 in ("flooded", "waterlogged")
        ok = flooded == should
        agree += ok
        print(f"  {r.ward:16s} {r.elev_m:6.1f} {r.peak_depth_mm:9.1f} "
              f"{r.hours_above_flood:8d} {r.physics_class:10s} "
              f"{r.documented_2015:12s} {'OK' if ok else 'MISMATCH'}")

    print()
    print(f"  flood/no-flood agreement: {agree}/{len(res)} wards")

    panel.to_parquet(OUT, index=False)
    print(f"  wrote {OUT}  ({len(panel):,} rows)")
    res.to_csv(config.PROCESSED / "event_ward_summary.csv", index=False)
    print(f"  wrote {config.PROCESSED / 'event_ward_summary.csv'}")


if __name__ == "__main__":
    main()
