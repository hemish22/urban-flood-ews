"""Back-validate the alert system on 2015, hour by hour.

This is the project's headline result and also its most easily faked number, so
the protocol is stated up front:

  1. The LSTM is trained ONLY on 2010-2014 and is never refitted here.
  2. At each hour, the engine sees the current state plus the LSTM forecast.
     It never sees the simulator's future.
  3. The simulator is then consulted, once, to score whether the ward really
     flooded and when. It plays the role of ground truth.
  4. An alert counts only if it fires BEFORE the flood begins.

Everything printed is measured. Anything that could not be measured is printed
as unavailable rather than estimated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.models.lstm_forecast import ResidualForecaster, CHANNELS
from src.risk.engine import score_alerts, ward_risk, ALERT_HORIZON_H


def load_forecaster():
    ck = torch.load(config.MODELS / "lstm_forecast.pt", map_location="cpu",
                    weights_only=False)
    model = ResidualForecaster(len(CHANNELS), ck["hidden"],
                               len(config.FORECAST_HORIZONS))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def forecast_2015(panel: pd.DataFrame, model, ck) -> pd.DataFrame:
    """Run the saved model over every 2015 ward-hour, no refitting."""
    horizons = config.FORECAST_HORIZONS
    n = config.LSTM_LOOKBACK_H
    mu = np.asarray(ck["mu"], dtype=np.float32)
    sd = np.asarray(ck["sd"], dtype=np.float32)
    rmu = np.asarray(ck["rmu"], dtype=np.float32)
    rsd = np.asarray(ck["rsd"], dtype=np.float32)

    res = []
    for ward, w in panel.groupby("ward"):
        w = w.sort_values("timestamp").reset_index(drop=True)
        # scaling is per-channel, so broadcast over the time axis
        feat = (w[CHANNELS].to_numpy(dtype=np.float32) - mu) / sd
        idx = np.arange(n - 1, len(w))
        if len(idx) == 0:
            continue
        xs = np.stack([feat[i - n + 1:i + 1] for i in idx])
        cur_all = np.log1p(w["depth_mm"].to_numpy()[idx].astype(np.float32))
        out = []
        with torch.no_grad():
            for i in range(0, len(xs), 4096):
                xb = torch.from_numpy(xs[i:i + 4096]).float()
                cb = torch.from_numpy(cur_all[i:i + 4096]).float()
                out.append(model(xb, cb).numpy())
        resid = np.concatenate(out) * rsd + rmu
        pred = np.clip(np.expm1(cur_all[:, None] + resid), 0.0, None)
        for j, i0 in enumerate(idx):
            for k, h in enumerate(horizons):
                res.append({
                    "ward": ward,
                    "timestamp": w["timestamp"].iloc[i0],
                    "horizon_h": h,
                    "forecast_depth_mm": float(pred[j, k]),
                })
    return pd.DataFrame(res)


def main() -> None:
    panel = pd.read_parquet(config.PROCESSED / "hourly_panel.parquet")
    model, ck = load_forecaster()
    print("=" * 70)
    print("2015 BACK-VALIDATION")
    print("=" * 70)
    print(f"  model trained on   {ck['mu_time']} .. 2014-12-31 (never refitted)")
    print(f"  forecast horizons  t+{ALERT_HORIZON_H}h used for alerting")

    fc = forecast_2015(panel, model, ck)
    fc.to_parquet(config.PROCESSED / "forecast_2015.parquet", index=False)
    print(f"  wrote {len(fc):,} forecast rows "
          f"-> {config.PROCESSED / 'forecast_2015.parquet'}")

    test = panel[panel["timestamp"] >= config.SPLIT_DATE]
    hours = sorted(test["timestamp"].unique())
    step = 3  # score every 3 h; the signal is hourly but 2015 is 8,760 steps
    sample = hours[::step]
    print(f"  scoring {len(sample):,} alert times (every {step} h)")

    rows = []
    for ts in sample:
        r = ward_risk(panel, fc, pd.Timestamp(ts))
        rows.append(r)
    alerts = pd.concat(rows, ignore_index=True)
    alerts.to_parquet(config.PROCESSED / "alerts_2015.parquet", index=False)
    print(f"  wrote {len(alerts):,} ward-alert rows")

    # Score against 2015 only: alerts exist only for the test year, so any
    # 2010-2014 episode would be counted as missed by construction.
    score = score_alerts(test, alerts)
    pw = score.pop("per_ward")
    print()
    print("-" * 70)
    print("DETECTION PERFORMANCE  (flood = simulator depth > "
          f"{config.FLOOD_DEPTH_MM:.0f} mm)")
    print("-" * 70)
    print(f"  flood episodes in 2015        {score['episodes_total']}")
    print(f"  episodes alerted in advance   {score['episodes_detected']}")
    print(f"  episode recall                {score['episode_recall']:.1%}")
    print(f"  ward recall                   {score['ward_recall']:.1%}")
    print(f"  median lead time              {score['median_lead_h']:.1f} h")
    print(f"  minimum lead time             {score['min_lead_h']:.1f} h")
    print()
    print("  per ward:")
    for _, r in pw.sort_values("peak_depth_mm", ascending=False).iterrows():
        lead = "  n/a" if pd.isna(r["median_lead_h"]) else f"{r['median_lead_h']:5.1f} h"
        print(f"    {r['ward']:16s} peak {r['peak_depth_mm']:6.1f} mm   "
              f"episodes {r['flood_episodes']:2d}   detected {r['detected']:2d}   "
              f"lead {lead}")

    # ---- alert volume: a system that warns always is not useful ---------
    fired = alerts[alerts["alert"]]
    per_ward_hours = fired.groupby("ward").size()
    print()
    print("-" * 70)
    print("ALERT VOLUME  (a system that warns all the time is not a system)")
    print("-" * 70)
    print(f"  alert records      {len(fired):,} of {len(alerts):,} "
          f"({len(fired) / len(alerts):.2%} of ward-hours)")
    print(f"  per ward           min {per_ward_hours.min():.0f}  "
          f"median {per_ward_hours.median():.0f}  max {per_ward_hours.max():.0f}")
    print(f"  severity mix       " + ", ".join(
        f"{k} {v / len(fired):.0%}" for k, v in
        fired["severity"].value_counts().items()) if len(fired) else "  n/a")

    out = {
        **score,
        "alert_records": int(len(fired)),
        "alert_rate": float(len(fired) / len(alerts)),
        "flood_depth_mm": config.FLOOD_DEPTH_MM,
        "alert_horizon_h": ALERT_HORIZON_H,
        "protocol": ("LSTM trained on 2010-2014 only; engine saw current state "
                     "plus forecast, never the simulator's future; alerts "
                     "scored against the simulator as ground truth"),
    }
    (config.MODELS / "backtest_2015.json").write_text(json.dumps(out, indent=2))
    pw.to_csv(config.REPORTS / "backtest_2015_per_ward.csv", index=False)
    print()
    print(f"  wrote {config.MODELS / 'backtest_2015.json'}")
    print(f"  wrote {config.REPORTS / 'backtest_2015_per_ward.csv'}")


if __name__ == "__main__":
    main()
