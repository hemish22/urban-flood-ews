"""Generate the report figures. Every figure is a measured artefact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": True,
})


def fig_hydrograph(panel, ev):
    """Flood hydrograph for the 2015 replay: rainfall on top, depth below."""
    ev_ts = pd.to_datetime(config.EVENT_2015["start"]), pd.to_datetime(
        config.EVENT_2015["end"])
    p = panel[(panel["timestamp"] >= ev_ts[0])
              & (panel["timestamp"] <= ev_ts[1])]
    rain = p.groupby("timestamp", as_index=False)["rainfall_mm"].first()
    rain24 = rain.set_index("timestamp")["rainfall_mm"].rolling(24).sum()

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(9, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 2.2]},
    )
    ax0.bar(rain.index, rain["rainfall_mm"], width=0.035,
            color="#1565c0", label="hourly")
    ax0.plot(rain24.index, rain24.values, color="#0d47a1", lw=1.4,
             label="24 h rolling")
    ax0.set_ylabel("rainfall (mm/h)")
    ax0.legend(loc="upper left", fontsize=8)
    ax0.set_title(f"2015 replay: ERA5 hourly shape, documented daily totals "
                  f"(peak 319.6 mm on 1 Dec)", fontsize=10, loc="left")

    show = ["Perungalathur", "Velachery", "Saidapet", "Koyambedu", "Avadi"]
    for w in show:
        s = p[p["ward"] == w]
        ax1.plot(s["timestamp"], s["depth_mm"], lw=1.1, label=w)
    ax1.axhline(config.FLOOD_DEPTH_MM, color="#c62828", ls="--", lw=1,
                label=f"flood threshold {config.FLOOD_DEPTH_MM:.0f} mm")
    ax1.set_ylabel("depth (mm)")
    ax1.set_xlabel("")
    ax1.legend(loc="upper left", ncol=3, fontsize=7.5)
    out = config.FIGURES / "hydrograph_2015.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def fig_map(panel, ev):
    """Ward peak depth as a scatter map on real coordinates."""
    pts = pd.DataFrame([(n, la, lo) for n, la, lo, *_ in config.WARDS],
                       columns=["ward", "lat", "lon"])
    pk = panel.groupby("ward", as_index=False)["depth_mm"].max()
    d = pts.merge(pk, on="ward")
    # annotate the alert thresholds on the colourbar so the map is readable
    # without cross-referencing the config
    vmax = float(d["depth_mm"].max())

    fig, ax = plt.subplots(figsize=(7.2, 7))
    sc = ax.scatter(d["lon"], d["lat"], s=d["depth_mm"] / 1.6 + 40,
                    c=d["depth_mm"], cmap="YlOrRd", vmin=0, vmax=vmax,
                    edgecolor="#333", linewidth=0.6, zorder=3)
    for _, r in d.iterrows():
        ax.annotate(f"{r['ward']}\n{r['depth_mm']:.0f} mm",
                    (r["lon"], r["lat"]), fontsize=6.5,
                    xytext=(6, 4), textcoords="offset points")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("2015 replay: peak modelled depth by ward (mm)", fontsize=10,
                 loc="left")
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("peak depth (mm)")
    t = config.LEVEL_THRESHOLDS_MM
    for lvl, name in ((t["high"], "high"), (t["severe"], "severe")):
        if lvl <= vmax:
            cb.ax.axhline(lvl, color="#333", ls="--", lw=0.8)
            cb.ax.text(1.6, lvl, name, va="center", fontsize=6.5)
    out = config.FIGURES / "ward_peak_depth.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def fig_lstm(metrics):
    """LSTM vs persistence at every horizon: the headline honesty check."""
    m = metrics["all_windows"]["per_horizon"]
    h = [r["horizon_h"] for r in m]
    lr = [r["lstm_rmse"] for r in m]
    pr = [r["persistence_rmse"] for r in m]
    x = np.arange(len(h))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10, 4))
    axa.plot(x, pr, "o--", color="#78909c", label="persistence")
    axa.plot(x, lr, "s-", color="#c62828", label="LSTM")
    axa.set_xticks(x)
    axa.set_xticklabels([f"t+{i}" for i in h])
    axa.set_xlabel("forecast horizon")
    axa.set_ylabel("RMSE (mm)")
    axa.set_title("Held-out 2015, all ward-hours", fontsize=10, loc="left")
    axa.legend(fontsize=8)

    gain = [r["rmse_improvement_pct"] for r in m]
    cols = ["#2e7d32" if g > 0 else "#c62828" for g in gain]
    axb.bar(x, gain, color=cols)
    axb.axhline(0, color="k", lw=0.8)
    axb.set_xticks(x)
    axb.set_xticklabels([f"t+{i}" for i in h])
    axb.set_xlabel("forecast horizon")
    axb.set_ylabel("RMSE improvement (%)")
    axb.set_title("positive = LSTM beats persistence", fontsize=10, loc="left")
    for i, g in enumerate(gain):
        axb.annotate(f"{g:.0f}%", (i, g), ha="center",
                     va="bottom" if g > 0 else "top", fontsize=7.5)
    out = config.FIGURES / "lstm_vs_persistence.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def fig_distribution():
    """Why the rainfall correction was necessary: raw vs corrected extremes."""
    raw = pd.read_csv(config.RAW / "rainfall.csv", parse_dates=["timestamp"])
    cor = pd.read_csv(config.PROCESSED / "rainfall_corrected.csv",
                      parse_dates=["timestamp"])

    def dmax(df, label):
        d = df.set_index("timestamp")["rainfall_mm"].resample("D").sum()
        return d.groupby(d.index.year).max(), label

    r, _ = dmax(raw, "raw")
    c, _ = dmax(cor, "corrected")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(r))
    ax.bar(x - 0.2, r.values, 0.4, color="#90a4ae", label="ERA5 as given")
    ax.bar(x + 0.2, c.values, 0.4, color="#1565c0", label="bias-corrected")
    ax.set_xticks(x)
    ax.set_xticklabels(r.index)
    ax.set_xlabel("year")
    ax.set_ylabel("wettest day (mm)")
    ax.set_title("ERA5 misses the 2015 convective extreme by a factor of 3",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8)
    out = config.FIGURES / "rainfall_bias_correction.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def main() -> None:
    print("figures:")
    panel = pd.read_parquet(config.PROCESSED / "hourly_panel.parquet")
    ev = pd.read_parquet(config.EVENTS / "event_2015_replay.parquet")
    metrics = json.loads((config.MODELS / "lstm_metrics.json").read_text())
    fig_hydrograph(ev, ev)
    fig_map(ev, ev)
    fig_lstm(metrics)
    fig_distribution()
    print(f"  -> {config.FIGURES}")


if __name__ == "__main__":
    main()
