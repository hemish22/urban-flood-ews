"""Build model-ready features from the simulated panel.

Two products:

  1. SEQUENCES  (data/processed/sequences.npz) for the LSTM. Per ward, a
     sliding 24 h window of [rainfall, depth, soil, runoff] with targets at
     t+1..t+6. Windows are contiguous, never shuffled, and split by time.

  2. EPISODES   (data/processed/episodes.csv) for XGBoost. The natural unit of
     a susceptibility model is a ward x storm episode, not a ward-hour:
     flooding is rare (0.01% of hours), so an hourly classifier would learn
     nothing but "no flood". Each episode is a storm separated from the next by
     a dry spell, and carries static ward properties, antecedent conditions,
     storm forcing, and the simulator's own outcome as the label.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.simulate import hydrology

# a storm ends after this many consecutive dry hours
DRY_SEPARATION_H = 8
MIN_EPISODE_RAIN_MM = 8.0


def identify_episodes(rain: pd.DataFrame) -> list[tuple[int, int]]:
    """Split the rainfall series into storm episodes on dry separations."""
    r = rain["rainfall_mm"].to_numpy()
    wet = r > 0.2
    episodes: list[tuple[int, int]] = []
    start: int | None = None
    dry = 0
    for i, w in enumerate(wet):
        if w:
            if start is None:
                start = i
            dry = 0
        else:
            if start is None:
                continue
            dry += 1
            if dry >= DRY_SEPARATION_H:
                end = i - dry + 1
                if r[start:end].sum() >= MIN_EPISODE_RAIN_MM:
                    episodes.append((start, end))
                start = None
                dry = 0
    if start is not None and r[start:].sum() >= MIN_EPISODE_RAIN_MM:
        episodes.append((start, len(r)))
    return episodes


def build_episodes(panel: pd.DataFrame, rain: pd.DataFrame,
                   elev: pd.DataFrame) -> pd.DataFrame:
    """One row per ward x storm episode, labelled by the simulator."""
    ts = rain["timestamp"]
    episodes = identify_episodes(rain)
    rain_vals = rain["rainfall_mm"].to_numpy()
    sm_vals = rain["soil_moisture"].to_numpy()

    elev_m = dict(zip(elev["ward"], elev["elevation_m"]))
    elev_idx = dict(zip(elev["ward"], elev["elevation_index"]))

    rows = []
    for ward in config.WARD_NAMES:
        w = panel[panel["ward"] == ward]
        depth = w["depth_mm"].to_numpy()
        for (a, b) in episodes:
            seg_rain = rain_vals[a:b]
            total = float(seg_rain.sum())
            if total < MIN_EPISODE_RAIN_MM:
                continue
            seg_depth = depth[a:b]
            peak = float(seg_depth.max())
            hours_above = int((seg_depth > config.FLOOD_DEPTH_MM).sum())

            # antecedent: the 72 h before the storm, which is what decides
            # whether the soil can still absorb the rain
            a0 = max(a - 72, 0)
            ante_rain = float(rain_vals[a0:a].sum())
            ante_soil = float(np.nanmean(sm_vals[a0:a])) if a > a0 else float(sm_vals[0])

            rows.append({
                "ward": ward,
                "episode_start": ts.iloc[a],
                "episode_end": ts.iloc[b - 1],
                "duration_h": b - a,
                "rain_total_mm": round(total, 2),
                "rain_max_hour_mm": round(float(seg_rain.max()), 2),
                "rain_max_6h_mm": round(float(pd.Series(seg_rain).rolling(6).sum().max()), 2),
                "antecedent_rain_72h_mm": round(ante_rain, 2),
                "antecedent_soil_moisture": round(ante_soil, 4),
                "elevation_m": elev_m[ward],
                "elevation_index": round(elev_idx[ward], 4),
                "imperviousness": config.WARD_IMPERVIOUSNESS[ward],
                "drain_quality": config.WARD_DRAIN_QUALITY[ward],
                "drain_capacity_ratio": round(
                    config.PHYSICS["drain_base_mm_hr"] * (0.40 + config.WARD_DRAIN_QUALITY[ward])
                    / max(float(seg_rain.max()), 0.1), 3),
                "peak_depth_mm": round(peak, 2),
                "hours_above_flood": hours_above,
                "physics_class": hydrology.physics_class(hours_above, peak),
                "label_source": "simulated:bucket-model",
            })

    df = pd.DataFrame(rows)

    # --- two labels, used for two different questions ---------------------
    # 1. physics_class: absolute impact, from the level thresholds. This is
    #    what the alert engine shows a citizen. It is 98% "Low" because
    #    flooding genuinely is rare over six years, so it is useless as a
    #    classification target and is NOT used to train the classifier.
    #
    # 2. susceptibility_class: how badly this ward fared RELATIVE to the other
    #    14 wards hit by the same storm, quartiled. This is what a
    #    susceptibility map for drainage prioritisation actually means ("of
    #    the wards this storm hit, which are the worst?"). It is a statement
    #    about the ward, not about the storm: Perungalathur ponds more than
    #    Avadi in a 6 mm drizzle as surely as in a 320 mm cloudburst, and that
    #    ordering is what a drainage engineer needs to act on.
    #
    #    Deliberately NOT gated on absolute depth. A 98%-Low class distribution
    #    is not a modelling failure, it is the true base rate: over six years
    #    Chennai floods a few times, so almost every ward-storm pair is
    #    unremarkable in absolute terms. Gating on absolute depth would throw
    #    away the only learnable signal and leave 11 of 15 wards in one class.
    df["susceptibility_class"] = "Low"
    for _, grp in df.groupby("episode_start"):
        rank = grp["peak_depth_mm"].rank(method="first", pct=True)
        df.loc[grp.index, "susceptibility_class"] = np.select(
            [rank > 0.75, rank > 0.50, rank > 0.25],
            ["Severe", "High", "Moderate"],
            default="Low",
        )

    df["split"] = np.where(df["episode_start"] < config.SPLIT_DATE, "train", "test")
    return df.sort_values(["episode_start", "ward"]).reset_index(drop=True)


def build_sequences(panel: pd.DataFrame, out: Path) -> dict:
    """Sliding 24 h windows per ward with t+1..t+6 depth targets."""
    feats, targets, wards, stamps = [], [], [], []
    L = config.LSTM_LOOKBACK_H
    H = max(config.FORECAST_HORIZONS)

    for ward in config.WARD_NAMES:
        w = panel[panel["ward"] == ward].sort_values("timestamp")
        arr = w[["rainfall_mm", "depth_mm", "soil_mm", "runoff_mm"]].to_numpy(np.float32)
        ts = w["timestamp"].to_numpy()
        n = len(arr) - L - H + 1
        if n <= 0:
            continue
        for i in range(n):
            feats.append(arr[i:i + L])
            targets.append(arr[i + L: i + L + H, 1])  # depth channel
            wards.append(ward)
            stamps.append(ts[i + L - 1])

    data = {
        "X": np.stack(feats).astype(np.float32),
        "y": np.stack(targets).astype(np.float32),
        "ward": np.array(wards),
        "stamp": np.stack(stamps),
    }
    np.savez_compressed(out, **data)
    return data


def main() -> None:
    panel = pd.read_parquet(config.PROCESSED / "hourly_panel.parquet")
    rain = pd.read_csv(config.RAW / "rainfall.csv", parse_dates=["timestamp"])
    elev = pd.read_csv(config.RAW / "elevations.csv")

    # CRITICAL: read the rainfall the panel was ACTUALLY simulated from, not a
    # freshly recomputed correction. run_simulation.py already corrected the
    # series before simulating; re-applying the correction here double-corrects
    # it, which inflated test-period storms to 287 mm while training tops out at
    # 12 mm. Features and simulation must share one rainfall series.
    corr_path = config.PROCESSED / "rainfall_corrected.csv"
    if corr_path.exists():
        rain = pd.read_csv(corr_path, parse_dates=["timestamp"])
    else:
        rain = pd.read_csv(config.RAW / "rainfall.csv", parse_dates=["timestamp"])
        rain["rainfall_mm"] = hydrology.bias_series_rainfall(
            rain["rainfall_mm"].to_numpy(),
            rain["timestamp"].dt.strftime("%Y-%m-%d").to_numpy(),
            config.EVENT_DAILY_TARGETS_MM,
        )
        print("  WARNING: rainfall_corrected.csv missing, recomputed here. "
              "Re-run src.simulate.run_simulation to guarantee consistency.")

    print("=" * 68)
    print("EPISODES (XGBoost unit: ward x storm)")
    print("=" * 68)
    ep = build_episodes(panel, rain, elev)
    ep_path = config.PROCESSED / "episodes.csv"
    ep.to_csv(ep_path, index=False)
    print(f"  episodes found        {ep['episode_start'].nunique()}")
    print(f"  ward-episode rows     {len(ep):,}")
    print(f"  columns               {len(ep.columns)}")
    print()
    print("  physics_class (ABSOLUTE impact, drives alerts):")
    for k, v in ep["physics_class"].value_counts().reindex(
            ["Low", "Moderate", "High", "Severe"], fill_value=0).items():
        print(f"    {k:10s} {v:6,d}  ({v / len(ep) * 100:5.1f}%)")
    print()
    print("  susceptibility_class (RELATIVE rank within each storm, trains the model):")
    for k, v in ep["susceptibility_class"].value_counts().reindex(
            ["Low", "Moderate", "High", "Severe"], fill_value=0).items():
        print(f"    {k:10s} {v:6,d}  ({v / len(ep) * 100:5.1f}%)")
    print()
    print("  train/test split on", config.SPLIT_DATE)
    print(ep["split"].value_counts().to_string())
    print(f"  wrote {ep_path}")

    print()
    print("=" * 68)
    print("SEQUENCES (LSTM unit: 24 h window per ward-hour)")
    print("=" * 68)
    seq_path = config.PROCESSED / "sequences.npz"
    d = build_sequences(panel, seq_path)
    print(f"  windows               {len(d['X']):,}")
    print(f"  X shape               {d['X'].shape}")
    print(f"  y shape               {d['y'].shape}  (t+1..t+6 depth)")
    is_test = d["stamp"] >= np.datetime64(config.SPLIT_DATE)
    print(f"  train windows         {int((~is_test).sum()):,}")
    print(f"  test windows          {int(is_test.sum()):,}")
    print(f"  wrote {seq_path}")


if __name__ == "__main__":
    main()
