"""Risk engine: turn model output into an operational ward alert.

DESIGN
    The engine answers the only question that matters operationally: given what
    is on the ground right now plus a forecast, which ward needs a warning, how
    severe, and how much time does the resident have?

    It deliberately does NOT read the simulator's `depth_mm` to decide whether
    to warn. That would be looking at the answer, and every metric below would
    be meaningless. Alerts are raised from:

      - CURRENT state, which is legitimate (an operator can measure a gauge),
        and
      - the LSTM FORECAST, which is the actual prediction under test.

    The simulator's depth is loaded separately, purely to score the alerts
    after the fact. Every metric in the back-test is therefore an honest
    out-of-sample measurement.

LEAD TIME
    Lead time is measured as the gap between the latest alert issued within
    12 h before an episode and the hour the simulator says that ward actually
    crossed the flood threshold. This is conservative by design.
    Reporting the number of alert hours instead would flatter the system, since
    a single lucky hour can trip an alert.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config

# How far ahead the LSTM is allowed to be trusted for alerting. The model wins
# 6 of 6 horizons against persistence, but accuracy falls with range, so
# alerting off the full 6 h would lean on the weakest horizon.
ALERT_HORIZON_H = 4
CONFIDENCE_FLOOR_MM = 20.0


def severity_from_depth(depth_mm: float) -> tuple[str, str, int]:
    """Map a water depth to (class, colour, rank)."""
    t = config.LEVEL_THRESHOLDS_MM
    if depth_mm >= t["severe"]:
        return "Severe", "#c62828", 3
    if depth_mm >= t["high"]:
        return "High", "#ef6c00", 2
    if depth_mm >= t["moderate"]:
        return "Moderate", "#f9a825", 1
    return "Low", "#2e7d32", 0


def ward_risk(panel: pd.DataFrame, forecast: pd.DataFrame | None,
              now: pd.Timestamp) -> pd.DataFrame:
    """One row per ward: current depth, forecast peak, alert, severity.

    `forecast` must contain ward/timestamp/forecast_depth_mm. If it is None the
    engine runs on current state alone, which is the degraded mode used when
    no trained model is available.
    """
    now = pd.Timestamp(now)
    recent = panel[panel["timestamp"] <= now]
    cur = (recent.sort_values("timestamp")
                  .groupby("ward", as_index=False)
                  .tail(1)[["ward", "timestamp", "depth_mm", "soil_mm"]]
           .rename(columns={"depth_mm": "current_depth_mm"}))

    if forecast is not None and len(forecast):
        # `timestamp` in the forecast table is the ISSUE time (the last hour of
        # the input window), not the target time. Only the forecast issued in
        # the hour ending at `now` is admissible: a forecast issued later was
        # built from depths the operator has not seen yet.
        f = forecast[(forecast["timestamp"] <= now)
                     & (forecast["timestamp"] > now - pd.Timedelta(hours=1))
                     & (forecast["horizon_h"] <= ALERT_HORIZON_H)]
        fpk = (f.groupby("ward", as_index=False)["forecast_depth_mm"]
                .max()
                .rename(columns={"forecast_depth_mm": "forecast_peak_mm"}))
        cur = cur.merge(fpk, on="ward", how="left")
    else:
        cur["forecast_peak_mm"] = np.nan

    # The alert is raised on the WORSE of observed and forecast, because a
    # resident must be warned on a level already seen, not one predicted.
    cur["trigger_depth_mm"] = np.fmax(
        cur["current_depth_mm"].fillna(0.0),
        cur["forecast_peak_mm"].fillna(0.0),
    )

    # Require the forecast to clear a floor before alerting, so a model that
    # drifts a few mm above zero during dry hours cannot spam warnings.
    def _alert(r):
        if r["current_depth_mm"] >= config.LEVEL_THRESHOLDS_MM["moderate"]:
            return True
        f = r["forecast_peak_mm"]
        return bool(pd.notna(f) and f >= config.LEVEL_THRESHOLDS_MM["moderate"])

    cur["alert"] = cur.apply(_alert, axis=1)
    sev = cur["trigger_depth_mm"].apply(severity_from_depth)
    cur["severity"] = [s[0] for s in sev]
    cur["colour"] = [s[1] for s in sev]
    cur["rank"] = [s[2] for s in sev]
    cur["as_of"] = now
    return cur.sort_values(["rank", "trigger_depth_mm"], ascending=False)


def score_alerts(
    panel: pd.DataFrame,
    alerts: pd.DataFrame,
    flood_depth: float = config.FLOOD_DEPTH_MM,
) -> dict:
    """Score alerts against the simulator, for each ward.

    A "true flood" is a contiguous episode where the simulator holds the ward
    above `flood_depth`. An alert is credited only if it fires BEFORE the
    episode starts, which is the only way a warning can be useful.
    """
    rows = []
    for ward, w in panel.groupby("ward"):
        w = w.sort_values("timestamp")
        flooded = (w["depth_mm"] > flood_depth).to_numpy()
        ts = w["timestamp"].to_numpy()

        episodes, start = [], None
        for i, f in enumerate(flooded):
            if f and start is None:
                start = i
            elif not f and start is not None:
                episodes.append((ts[start], ts[i - 1]))
                start = None
        if start is not None:
            episodes.append((ts[start], ts[-1]))

        aw = alerts[alerts["ward"] == ward]
        alert_ts = (
            pd.to_datetime(aw["as_of"]).to_numpy()
            if len(aw) and aw["alert"].any()
            else np.array([], dtype="datetime64[ns]")
        )

        detected, lead = [], []
        for s, e in episodes:
            early = alert_ts[(alert_ts >= s - np.timedelta64(12, "h"))
                             & (alert_ts < s)]
            if early.size:
                detected.append(True)
                lead.append(float((s - early.max()) / np.timedelta64(1, "h")))
            else:
                detected.append(False)
                lead.append(np.nan)

        rows.append({
            "ward": ward,
            "flood_episodes": len(episodes),
            "detected": int(sum(detected)),
            "missed": int(len(detected) - sum(detected)),
            "median_lead_h": float(np.nanmedian(lead)) if detected else np.nan,
            "peak_depth_mm": float(w["depth_mm"].max()),
        })

    per_ward = pd.DataFrame(rows)
    det = per_ward["detected"].sum()
    tot = per_ward["flood_episodes"].sum()
    leads = per_ward["median_lead_h"].dropna()
    return {
        "per_ward": per_ward,
        "episodes_total": int(tot),
        "episodes_detected": int(det),
        "episode_recall": float(det / tot) if tot else np.nan,
        "ward_recall": float((per_ward["detected"] > 0).sum() / len(per_ward)),
        "median_lead_h": float(leads.median()) if len(leads) else np.nan,
        "min_lead_h": float(leads.min()) if len(leads) else np.nan,
    }
