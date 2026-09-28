"""Streamlit dashboard: ward flood risk, Chennai.

Run:  .venv/bin/streamlit run app/dashboard.py

The dashboard reads only saved artefacts, so it starts instantly and never
re-runs the simulation. It is organised as four tabs, in priority order:

  1. Situation   - what is happening right now, per ward, on a map.
  2. Ward detail - why, in numbers: rain, depth, and the LSTM forecast issued
                   at the replay hour.
  3. Validation  - how much to trust it, measured on held-out 2015.
  4. About       - method, data provenance and limitations.

A persistent banner states that depths are SIMULATED and that 2015 is a
reconstructed replay, so nobody can mistake this for a live gauge feed.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# `streamlit run app/dashboard.py` puts app/ on sys.path, not the project
# root, so `src` is unimportable until we add the root ourselves. Walk up from
# this file to the directory that actually contains the `src` package, rather
# than counting parent levels, which silently breaks if the file is moved.
_ROOT = Path(__file__).resolve()
for _p in _ROOT.parents:
    if (_p / "src" / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
else:  # pragma: no cover
    raise RuntimeError("project root with src/config.py not found")

from src import config
from src.risk.engine import ALERT_HORIZON_H, ward_risk

st.set_page_config(page_title="Chennai Flood Early Warning",
                   page_icon="🌧", layout="wide")

# Replay window. Loading only this slice keeps start-up fast: the full forecast
# table is 4.7M rows, of which the demo needs a few percent.
LOAD_FROM = pd.Timestamp("2015-11-20")
REPLAY_START = datetime(2015, 11, 25, 0)
REPLAY_END = datetime(2015, 12, 12, 0)

# Storm stages for one-click jumps during a live demo.
STAGES = {
    "Calm before": datetime(2015, 11, 30, 18),
    "Forecast warning": datetime(2015, 12, 1, 1),
    "Peak rain": datetime(2015, 12, 1, 12),
    "Worst flooding": datetime(2015, 12, 2, 12),
    "Draining": datetime(2015, 12, 4, 12),
    "Receded": datetime(2015, 12, 7, 0),
}

SEV = {label: (colour, action) for _, label, colour, action in config.ALERT_LEVELS}
WARD_META = {n: {"lat": la, "lon": lo, "elev": el, "exp": ex}
             for n, la, lo, el, ex in config.WARDS}

# ---------------------------------------------------------------- styling
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
.hero {
  background: linear-gradient(120deg, #0b2545 0%, #134074 55%, #13315c 100%);
  color: #fff; border-radius: 14px; padding: 22px 28px; margin-bottom: 14px;
}
.hero h1 {color:#fff; font-size: 1.9rem; margin: 0 0 4px 0; padding: 0;}
.hero p {color:#cfe0f5; margin: 0; font-size: 0.95rem;}
.hero .tag {
  display:inline-block; background:rgba(255,255,255,.14); color:#fff;
  border:1px solid rgba(255,255,255,.25); border-radius: 999px;
  padding: 2px 10px; font-size: 0.75rem; letter-spacing:.04em;
  margin-right:6px; margin-top:10px;
}
.status {
  border-radius: 12px; padding: 14px 20px; color: #fff; margin: 6px 0 14px 0;
  display:flex; align-items:center; gap:18px; flex-wrap:wrap;
}
.status .lvl {font-size:1.5rem; font-weight:800; letter-spacing:.03em;}
.status .msg {font-size:1rem; opacity:.95;}
.kpi {
  border:1px solid rgba(128,128,128,.25); border-radius: 12px;
  padding: 12px 16px; height: 100%;
}
.kpi .k {font-size:.78rem; opacity:.7; text-transform:uppercase; letter-spacing:.05em;}
.kpi .v {font-size:1.6rem; font-weight:700; line-height:1.2;}
.kpi .s {font-size:.8rem; opacity:.7;}
.chip {
  display:inline-block; color:#fff; border-radius:999px; padding:1px 10px;
  font-size:.75rem; font-weight:600; min-width:70px; text-align:center;
}
.wrow {
  display:grid; grid-template-columns: 1.5fr 1fr .9fr .9fr .4fr;
  align-items:center; gap:6px; padding:6px 4px;
  border-bottom:1px solid rgba(128,128,128,.15); font-size:.9rem;
}
.wrow.head {font-size:.72rem; opacity:.65; text-transform:uppercase;
            letter-spacing:.05em; border-bottom:1px solid rgba(128,128,128,.35);}
.wrow .num {text-align:right; font-variant-numeric: tabular-nums;}
.bell {text-align:center;}
.guide {display:flex; gap:10px; align-items:flex-start; margin:6px 0;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data
@st.cache_data
def load_panel():
    return pd.read_parquet(config.PROCESSED / "hourly_panel.parquet",
                           filters=[("timestamp", ">=", LOAD_FROM)])


@st.cache_data
def load_forecast():
    p = config.PROCESSED / "forecast_2015.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p, filters=[("timestamp", ">=", LOAD_FROM)])


@st.cache_data
def load_metrics():
    out = {}
    for name, f in (("lstm", "lstm_metrics.json"),
                    ("xgb", "susceptibility_metrics.json"),
                    ("backtest", "backtest_2015.json")):
        p = config.MODELS / f
        if p.exists():
            out[name] = json.loads(p.read_text())
    return out


@st.cache_data
def load_backtest_wards():
    p = config.REPORTS / "backtest_2015_per_ward.csv"
    return pd.read_csv(p) if p.exists() else None


def pooled_rmse(per_horizon, key):
    """RMSE pooled over all horizons (equal windows per horizon)."""
    return math.sqrt(np.mean([r[key] ** 2 for r in per_horizon]))


panel = load_panel()
fc = load_forecast()
M = load_metrics()
bt = M.get("backtest", {})
rain = (panel.groupby("timestamp", as_index=False)["rainfall_mm"].mean())

# ---------------------------------------------------------------- header
st.markdown("""
<div class="hero">
  <h1>🌧 Chennai Urban Flood Early-Warning System</h1>
  <p>Ward-level flood alerts from a calibrated hydrology model, an LSTM
     water-level nowcast and an XGBoost susceptibility ranking — replaying the
     December 2015 floods across 15 wards.</p>
  <span class="tag">SIMULATED REPLAY · DEC 2015</span>
  <span class="tag">LSTM NOWCAST t+1…t+6 h</span>
  <span class="tag">15 WARDS</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- controls
if "now" not in st.session_state:
    st.session_state.now = STAGES["Peak rain"]


def _jump():
    stage = st.session_state.get("stage")
    if stage:
        st.session_state.now = STAGES[stage]
    # behave like buttons: the slider is the single source of truth
    st.session_state.stage = None


c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
with c1:
    st.slider("Replay clock", min_value=REPLAY_START, max_value=REPLAY_END,
              step=timedelta(hours=1), format="DD MMM YYYY, HH:mm", key="now")
with c2:
    st.pills("Jump to storm stage", list(STAGES), key="stage",
             on_change=_jump, selection_mode="single")

now = pd.Timestamp(st.session_state.now)
risk = ward_risk(panel, fc, now)
risk = risk.merge(
    panel[panel["timestamp"] == now - pd.Timedelta(hours=3)][["ward", "depth_mm"]]
    .rename(columns={"depth_mm": "depth_3h_ago"}), on="ward", how="left")

alerts = risk[risk["alert"]]
# wards warned while still below the alert depth: the forecast's contribution
ahead = alerts[alerts["current_depth_mm"] < config.LEVEL_THRESHOLDS_MM["moderate"]]
top = risk.iloc[0]
top_colour, top_action = SEV[top["severity"]]
rain_24h = rain[(rain["timestamp"] > now - pd.Timedelta(hours=24))
                & (rain["timestamp"] <= now)]["rainfall_mm"].sum()

# ---------------------------------------------------------------- status
st.markdown(f"""
<div class="status" style="background:{top_colour}">
  <div class="lvl">{top['severity'].upper()}</div>
  <div class="msg"><b>{len(alerts)} of {len(risk)} wards on alert</b>
    at {now:%d %b %Y, %H:%M}. {top_action}
    {f"<br>⏱ <b>{len(ahead)} warned ahead of the water</b> by the LSTM "
     f"forecast alone — still dry now." if len(ahead) else ""}</div>
</div>
""", unsafe_allow_html=True)

st.caption("⚠️ **Depths are simulated, not gauged.** Rainfall is real ERA5 with "
           "a documented bias correction; elevations are real SRTM; water "
           "levels come from a bucket model calibrated against the December "
           "2015 event. A demonstration of the method, not an operational feed.")


def kpi(col, k, v, s=""):
    col.markdown(f'<div class="kpi"><div class="k">{k}</div>'
                 f'<div class="v">{v}</div><div class="s">{s}</div></div>',
                 unsafe_allow_html=True)


k = st.columns(5)
kpi(k[0], "Wards on alert", f"{len(alerts)} / {len(risk)}",
    f"{(risk['severity'] == 'Severe').sum()} severe")
kpi(k[1], "Rain, last 24 h", f"{rain_24h:.0f} mm", "city-wide, bias-corrected")
kpi(k[2], "Deepest ward", f"{risk['current_depth_mm'].max():.0f} mm",
    risk.loc[risk["current_depth_mm"].idxmax(), "ward"])
kpi(k[3], "Forecast horizon", f"t+{ALERT_HORIZON_H} h", "LSTM, used for alerting")
if bt:
    kpi(k[4], "2015 back-test", f"{bt.get('ward_recall', 0):.0%}",
        f"wards warned · median lead {bt.get('median_lead_h', 0):.0f} h")

st.write("")
tab_sit, tab_ward, tab_val, tab_about = st.tabs(
    ["🗺️ Situation", "📈 Ward detail", "✅ Validation", "ℹ️ About & limits"])

# ================================================================ situation
with tab_sit:
    # storm timeline, so the audience always sees where in the event we are
    ev = rain[(rain["timestamp"] >= pd.Timestamp(REPLAY_START))
              & (rain["timestamp"] <= pd.Timestamp(REPLAY_END))]
    tl = go.Figure(go.Bar(x=ev["timestamp"], y=ev["rainfall_mm"],
                          marker_color="#4f8fd6", name="rain",
                          hovertemplate="%{x|%d %b %H:%M}<br>%{y:.1f} mm/h"
                                        "<extra></extra>"))
    tl.add_vline(x=now, line_color="#c62828", line_width=2)
    tl.add_annotation(x=now, y=1, yref="paper", text="now", showarrow=False,
                      font=dict(color="#c62828"), yshift=8)
    tl.update_layout(height=150, margin=dict(t=18, b=10, l=10, r=10),
                     yaxis_title="rain mm/h", showlegend=False,
                     bargap=0)
    st.plotly_chart(tl, width="stretch", config={"displayModeBar": False})

    left, right = st.columns([3, 2])
    with left:
        fmap = folium.Map(location=[13.02, 80.21], zoom_start=11,
                          tiles="OpenStreetMap", control_scale=True)
        for _, r in risk.iterrows():
            m = WARD_META[r["ward"]]
            fp = r["forecast_peak_mm"]
            body = (f"<div style='font-family:sans-serif;font-size:13px'>"
                    f"<b style='font-size:14px'>{r['ward']}</b><br>"
                    f"<span style='background:{r['colour']};color:#fff;"
                    f"padding:1px 8px;border-radius:8px'>{r['severity']}</span>"
                    f"<br>modelled depth: <b>{r['current_depth_mm']:.0f} mm</b>")
            if pd.notna(fp):
                body += (f"<br>forecast peak (t+{ALERT_HORIZON_H} h): "
                         f"<b>{fp:.0f} mm</b>")
            body += f"<br>elevation: {m['elev']:.0f} m</div>"
            if r["alert"]:
                # soft halo marks wards under an active warning
                folium.CircleMarker(
                    location=[m["lat"], m["lon"]],
                    radius=16 + 6 * r["rank"], color=r["colour"], weight=0,
                    fill=True, fill_opacity=0.18,
                ).add_to(fmap)
            folium.CircleMarker(
                location=[m["lat"], m["lon"]], radius=7 + 4 * r["rank"],
                color="#ffffff", weight=2, fill=True,
                fill_color=r["colour"], fill_opacity=0.95,
                popup=folium.Popup(body, max_width=260),
                tooltip=f"{r['ward']}: {r['severity']} · "
                        f"{r['current_depth_mm']:.0f} mm",
            ).add_to(fmap)
            folium.Marker(
                location=[m["lat"], m["lon"]],
                icon=folium.DivIcon(
                    icon_size=(120, 16), icon_anchor=(-12, 22),
                    html=(f"<div style='font:600 11px sans-serif;color:#1d2b3a;"
                          f"text-shadow:0 0 3px #fff,0 0 3px #fff'>"
                          f"{r['ward']}</div>")),
            ).add_to(fmap)
        legend = "".join(
            f"<div><span style='display:inline-block;width:11px;height:11px;"
            f"border-radius:50%;background:{c};margin-right:6px'></span>{l}</div>"
            for _, l, c, _ in config.ALERT_LEVELS)
        fmap.get_root().html.add_child(folium.Element(
            "<div style='position:fixed;bottom:18px;left:18px;z-index:9999;"
            "background:rgba(255,255,255,.92);padding:8px 12px;border-radius:8px;"
            "font:12px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.25)'>"
            f"<b>Alert level</b>{legend}"
            "<div style='margin-top:4px;opacity:.7'>halo = active warning</div>"
            "</div>"))
        st.iframe(fmap._repr_html_(), height=520)

    with right:
        rows = ['<div class="wrow head"><div>Ward</div><div>Level</div>'
                '<div class="num">Now</div>'
                f'<div class="num">t+{ALERT_HORIZON_H} h</div>'
                '<div class="bell">Alert</div></div>']
        for _, r in risk.iterrows():
            fp = r["forecast_peak_mm"]
            d3 = r["depth_3h_ago"]
            arrow = ""
            if pd.notna(d3):
                delta = r["current_depth_mm"] - d3
                arrow = ("▲" if delta > 2 else "▼" if delta < -2 else "")
            bell = ""
            if r["alert"]:
                bell = ("⏱" if r["ward"] in set(ahead["ward"]) else "🔔")
            rows.append(
                f'<div class="wrow"><div><b>{r["ward"]}</b></div>'
                f'<div><span class="chip" style="background:{r["colour"]}">'
                f'{r["severity"]}</span></div>'
                f'<div class="num">{r["current_depth_mm"]:.0f} mm '
                f'<span style="opacity:.6">{arrow}</span></div>'
                f'<div class="num">'
                f'{"—" if pd.isna(fp) else f"{fp:.0f} mm"}</div>'
                f'<div class="bell">{bell}</div></div>')
        st.markdown("".join(rows), unsafe_allow_html=True)
        st.caption("▲▼ change over the last 3 h. Alert = observed or "
                   f"forecast depth ≥ {config.LEVEL_THRESHOLDS_MM['moderate']:.0f} mm. "
                   "⏱ = warned by the forecast before water arrived.")

    st.markdown("##### What residents are told")
    g = st.columns(4)
    t = config.LEVEL_THRESHOLDS_MM
    ranges = [f"< {t['moderate']:.0f} mm",
              f"{t['moderate']:.0f}–{t['high']:.0f} mm",
              f"{t['high']:.0f}–{t['severe']:.0f} mm",
              f"≥ {t['severe']:.0f} mm"]
    for col, (_, label, colour, action), rng in zip(g, config.ALERT_LEVELS,
                                                    ranges):
        n = int((risk["severity"] == label).sum())
        col.markdown(
            f'<div class="kpi" style="border-top:4px solid {colour}">'
            f'<div class="k" style="color:{colour};opacity:1">{label} · {rng}</div>'
            f'<div class="v">{n} <span style="font-size:.9rem;opacity:.6">'
            f'wards</span></div><div class="s">{action}</div></div>',
            unsafe_allow_html=True)

# ================================================================ ward detail
with tab_ward:
    wards = list(risk["ward"])
    ward = st.selectbox("Ward", wards, index=0,
                        help="Sorted by current risk; the top ward is selected "
                             "by default.")
    wr = risk[risk["ward"] == ward].iloc[0]
    meta = WARD_META[ward]

    d = st.columns(5)
    kpi(d[0], "Level now", wr["severity"],
        "🔔 alert active" if wr["alert"] else "no alert")
    kpi(d[1], "Modelled depth", f"{wr['current_depth_mm']:.0f} mm")
    kpi(d[2], "Elevation", f"{meta['elev']:.0f} m", "SRTM 30 m")
    kpi(d[3], "Imperviousness", f"{config.WARD_IMPERVIOUSNESS[ward]:.0%}",
        f"drain quality {config.WARD_DRAIN_QUALITY[ward]:.0%}")
    kpi(d[4], "Dec 2015 reality", meta["exp"].capitalize(),
        "documented impact")

    lo, hi = now - pd.Timedelta(hours=72), now + pd.Timedelta(hours=12)
    w = panel[(panel["ward"] == ward) & (panel["timestamp"] >= lo)
              & (panel["timestamp"] <= hi)]
    past, future = w[w["timestamp"] <= now], w[w["timestamp"] >= now]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.25, 0.75], vertical_spacing=0.04)
    fig.add_trace(go.Bar(x=w["timestamp"], y=w["rainfall_mm"], name="rainfall",
                         marker_color="#4f8fd6",
                         hovertemplate="%{y:.1f} mm/h<extra>rain</extra>"),
                  row=1, col=1)
    # threshold bands behind the depth curve
    bands = [(0, t["moderate"], "#2e7d32"), (t["moderate"], t["high"], "#f9a825"),
             (t["high"], t["severe"], "#ef6c00"), (t["severe"], 10_000, "#c62828")]
    for y0, y1, c in bands:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=c, opacity=0.07, line_width=0,
                      row=2, col=1)
    fig.add_trace(go.Scatter(x=past["timestamp"], y=past["depth_mm"],
                             name="modelled depth (seen)", mode="lines",
                             line=dict(color="#0b2545", width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=future["timestamp"], y=future["depth_mm"],
                             name="what actually happened (hidden from model)",
                             mode="lines",
                             line=dict(color="#0b2545", width=1.5, dash="dot"),
                             opacity=0.5), row=2, col=1)
    if fc is not None:
        issued = fc[(fc["ward"] == ward) & (fc["timestamp"] == now)]
        if len(issued):
            fx = [now] + [now + pd.Timedelta(hours=int(h))
                          for h in issued["horizon_h"]]
            fy = [wr["current_depth_mm"]] + list(issued["forecast_depth_mm"])
            fig.add_trace(go.Scatter(x=fx, y=fy, name="LSTM forecast issued now",
                                     mode="lines+markers",
                                     line=dict(color="#c62828", width=2.5),
                                     marker=dict(size=7)), row=2, col=1)
    fig.add_hline(y=config.FLOOD_DEPTH_MM, line_dash="dash",
                  line_color="#c62828", row=2, col=1,
                  annotation_text="flood threshold",
                  annotation_position="top left")
    fig.add_vline(x=now, line_color="#888", line_dash="dot")
    ymax = max(w["depth_mm"].max(), config.FLOOD_DEPTH_MM) * 1.15
    fig.update_yaxes(title_text="rain mm/h", row=1, col=1)
    fig.update_yaxes(title_text="depth mm", range=[0, ymax], row=2, col=1)
    fig.update_layout(height=520, margin=dict(t=20, b=10, l=10, r=10),
                      hovermode="x unified", bargap=0,
                      legend=dict(orientation="h", y=-0.08))
    st.plotly_chart(fig, width="stretch")
    st.caption("Solid line: what the operator has seen up to the replay clock. "
               "Red: the LSTM forecast issued at that hour, from past data only. "
               "Dotted: the simulator's future, shown for comparison but never "
               "given to the alerting engine.")

    bw = load_backtest_wards()
    if bw is not None and ward in set(bw["ward"]):
        b = bw[bw["ward"] == ward].iloc[0]
        lead = ("n/a" if pd.isna(b["median_lead_h"])
                else f"{b['median_lead_h']:.0f} h")
        st.info(f"**2015 back-test for {ward}:** {int(b['detected'])} of "
                f"{int(b['flood_episodes'])} flood episodes warned in advance, "
                f"median lead time {lead}, peak modelled depth "
                f"{b['peak_depth_mm']:.0f} mm.")

# ================================================================ validation
with tab_val:
    st.markdown("All numbers below are measured on **2015, a year no model "
                "trained on**. Alerts were raised from past data only and "
                "scored against the simulator afterwards.")
    lm, xm = M.get("lstm", {}), M.get("xgb", {})

    v = st.columns(4)
    if bt:
        kpi(v[0], "Ward recall", f"{bt['ward_recall']:.0%}",
            "wards warned before flooding")
        kpi(v[1], "Episode recall", f"{bt['episode_recall']:.0%}",
            f"{bt['episodes_detected']} of {bt['episodes_total']} flood episodes")
        kpi(v[2], "Median lead time", f"{bt['median_lead_h']:.0f} h",
            f"minimum {bt['min_lead_h']:.0f} h")
        kpi(v[3], "Alert rate", f"{bt['alert_rate']:.1%}",
            "of ward-hours — not crying wolf")
    else:
        st.info("Run `python -m src.validate.backtest` to populate this tab.")

    st.write("")
    a, b = st.columns(2)
    with a:
        st.markdown("**LSTM vs persistence, per forecast horizon**")
        if lm:
            ph = lm["all_windows"]["per_horizon"]
            hz = [f"t+{r['horizon_h']} h" for r in ph]
            f2 = go.Figure()
            f2.add_bar(x=hz, y=[r["persistence_rmse"] for r in ph],
                       name="persistence", marker_color="#9aa5b1")
            f2.add_bar(x=hz, y=[r["lstm_rmse"] for r in ph], name="LSTM",
                       marker_color="#134074",
                       text=[f"−{r['rmse_improvement_pct']:.0f}%" for r in ph],
                       textposition="outside")
            f2.update_layout(height=340, barmode="group", yaxis_title="RMSE (mm)",
                             margin=dict(t=10, b=10, l=10, r=10),
                             legend=dict(orientation="h", y=1.1))
            st.plotly_chart(f2, width="stretch")
            aw = lm["all_windows"]
            sw = lm["storm_active_windows"]
            st.caption(
                f"Beats persistence at **{aw['horizons_won']}/6** horizons. "
                f"Pooled RMSE {pooled_rmse(aw['per_horizon'], 'lstm_rmse'):.2f} mm "
                f"vs {pooled_rmse(aw['per_horizon'], 'persistence_rmse'):.2f} mm; "
                f"storm windows "
                f"{pooled_rmse(sw['per_horizon'], 'lstm_rmse'):.2f} mm vs "
                f"{pooled_rmse(sw['per_horizon'], 'persistence_rmse'):.2f} mm.")
        else:
            st.info("LSTM not trained.")

    with b:
        st.markdown("**Flood episodes warned in advance, per ward**")
        bw = load_backtest_wards()
        if bw is not None:
            bw = bw.sort_values("peak_depth_mm")
            f3 = go.Figure()
            f3.add_bar(y=bw["ward"], x=bw["detected"], orientation="h",
                       name="warned", marker_color="#2e7d32")
            f3.add_bar(y=bw["ward"], x=bw["flood_episodes"] - bw["detected"],
                       orientation="h", name="missed", marker_color="#e0a3a3")
            f3.update_layout(height=340, barmode="stack",
                             xaxis_title="flood episodes in 2015",
                             margin=dict(t=10, b=10, l=10, r=10),
                             legend=dict(orientation="h", y=1.1))
            st.plotly_chart(f3, width="stretch")
            st.caption("Alerts start at 30 mm and floods are scored at "
                       "100 mm, so most warnings come from water already "
                       "ponding; about one in five was raised by the "
                       "forecast alone.")

    st.markdown("**XGBoost ward susceptibility**")
    x1, x2 = st.columns([1, 2])
    if xm:
        with x1:
            kpi(st, "Macro F1", f"{xm['macro_f1']:.3f}",
                f"majority baseline {xm.get('baseline_f1', 0):.3f}")
            st.caption("Trained on relative ward ranking within each storm, so "
                       "it learns which ward floods first, not how bad the "
                       "storm is.")
        with x2:
            imp = pd.Series(xm["importance"]).sort_values()
            f4 = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                                  marker_color="#4f8fd6"))
            f4.update_layout(height=300, xaxis_title="feature importance",
                             margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(f4, width="stretch")
    else:
        st.info("XGBoost not trained.")

# ================================================================ about
with tab_about:
    st.markdown("""
#### How it works
```
ERA5 rainfall + soil moisture ─┐
                               ├─► bias correction ─► two-reservoir bucket model ─► depth
real SRTM elevations ──────────┘                                                   │
                                          ┌────────────────────────────────────────┤
                                          ▼                                        ▼
                                LSTM nowcast (t+1..t+6)               XGBoost susceptibility
                                          └──────────────┬─────────────────────────┘
                                                         ▼
                                     risk engine (worse of observed & forecast → alert)
                                                         ▼
                                           2015 back-test · this dashboard
```

#### Honest limitations
- **The target is the simulator.** The LSTM predicts the bucket model's
  output; it would need retraining on real gauges to be operational.
- **Warning lead time is short** — median 2 h. The ceiling is set by how fast
  the catchment responds; more lead needs rainfall nowcasting further ahead.
- **Episode recall is 98 %, but mostly from ponding.** Alerts start at 30 mm
  and floods are scored at 100 mm; only about one in five episodes was first
  flagged by the forecast alone.
- **False-alarm rate is 28 %,** above the 20 % target: that share of
  alert-hours is not followed by a flood within 12 h.
- **XGBoost's F1 is easier than it looks** — the label is a ward's rank within
  a storm, so it measures ranking skill, not storm-severity prediction.
- **December 2015 is reconstructed.** Hourly shape from ERA5, daily totals from
  documented magnitudes; it is a replay, not a verification.
- **T.Nagar is a known disagreement** between simulator and documented impact,
  left visible rather than tuned away.
""")
    fig_dir = config.FIGURES
    imgs = [("rainfall_bias_correction.png", "ERA5 rainfall bias correction"),
            ("hydrograph_2015.png", "December 2015 hydrograph"),
            ("ward_peak_depth.png", "Peak modelled depth per ward")]
    cols = st.columns(len(imgs))
    for col, (f, cap) in zip(cols, imgs):
        if (fig_dir / f).exists():
            col.image(str(fig_dir / f), caption=cap)

st.divider()
st.caption("Data: Open-Meteo ERA5 rainfall/soil moisture · OpenTopData SRTM30m "
           "elevations · Kaggle Chennai flood reference. Water levels simulated. "
           "See README for provenance and limitations.")
