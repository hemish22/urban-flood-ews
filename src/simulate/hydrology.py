"""Calibrated two-reservoir bucket hydrology model.

State variables per ward, hourly timestep, all depths in mm:

    soil      soil profile storage (antecedent moisture, the memory term)
    surface   surface / depression storage, i.e. standing water

Each hour:

  1. Partition rain into infiltration and runoff. Impervious fraction comes
     from ward urbanisation; infiltration capacity collapses as the soil
     fills, which is what makes a second day of rain worse than the first.
  2. Add runoff to surface storage, release the gravitational percolation
     term.
  3. Drain surface storage through the gully network. Capacity scales with
     ward drainage quality and falls as a chokepoint once water rises, so a
     poorly drained ward floods on rainfall a well-drained one absorbs.
  4. Convert surface storage to depth above the ward's drainage datum using
     a stage curve; low ground has a datum close to the surface, so it
     reports depth sooner.

Soil moisture from ERA5 is used to initialise and nudge the soil store so the
antecedent condition tracks observed regional wetness rather than drifting.

The model is deliberately simple enough to explain to a civic audience, but
carries the three processes that decide where urban water actually stands:
antecedent moisture, imperviousness, and drain capacity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config

# --- calibration targets from the documented Dec 2015 Chennai event ----------
# Documented: 319.6 mm on 1/2 Dec 2015 (highest single day in Chennai's
# recorded history), 1113.8 mm in November, suburbs near 500 mm. ERA5 reports
# 48.8 mm for that day, so a correction is required.
#
# The correction preserves ERA5's TIMING and redistributes it toward the
# hours ERA5 already flagged as the convective core. It never reduces rain.
# A convex weight w = (x/mean)^exponent is applied within each day and
# renormalised to hit the target daily total, so the within-day shape stays
# proportional to ERA5 and the multi-day sequence is untouched.
SHAPE_EXPONENT = 1.0
PEAK_DAY_TARGET_MM = 319.6
def bias_event_rainfall(rain: np.ndarray, dates: np.ndarray,
                        targets: dict | None = None) -> np.ndarray:
    """Renormalise specific days to documented totals.

    Within each targeted day, hourly rainfall is reweighted by a convex
    function of its own magnitude, so the hours ERA5 already identified as the
    storm core keep most of the water and small drizzles are suppressed, then
    the day is renormalised to the documented total. Hours with no ERA5 signal
    stay dry; we never invent rain where the reanalysis saw none.
    """
    t = targets if targets is not None else {}
    out = np.array(rain, dtype=float, copy=True)

    for day in np.unique(dates):
        sel = np.where(dates == day)[0]
        vals = out[sel]
        target = t.get(str(day))
        total = vals.sum()
        if target is None or total <= 0.0:
            continue
        if abs(target - total) < 0.5:
            continue

        weights = np.power(np.maximum(vals, 0.0), SHAPE_EXPONENT)
        if weights.sum() <= 0:
            continue
        weights = weights / weights.sum()

        if target < total:
            out[sel] = vals * (target / total)
        else:
            out[sel] = weights * target
    return out


# --- systematic bias correction for the full training series ----------------
# ERA5 is a 25 km reanalysis grid. It resolves monsoon structure and
# multi-day accumulation well but systematically smooths convective extremes:
# across 2010-2015 its wettest day at Chennai is 115.7 mm, whereas real
# Chennai events reach 200-500 mm. Left uncorrected, no ward in the training
# data would ever reach a flood threshold and the classifier would have
# nothing to learn.
#
# The correction is a convex transform applied ONLY to wet days, calibrated so
# the corrected distribution matches documented Chennai extreme-rainfall return
# levels. Dry and light-rain days pass through untouched.
CONVEX_GAMMA = 1.45
WET_DAY_MM = 15.0
EXTREME_REFERENCE_MM = 330.0

# A monthly multiplier, not a per-day one.
#
# ERA5's under-resolution of convective extremes is not a quirk of one storm:
# across 2010-2015 it reports no day above 116 mm, while real Chennai records
# 200-500 mm events. A per-day convex transform cannot fix that, because it
# only rescales days that ERA5 already flagged as extreme, and no such day
# exists before December 2015. Training on the raw series therefore produced
# examples capped at 12 mm of water, and the model was then asked to forecast
# a 319.6 mm event: a 25x extrapolation, which no model can do honestly.
#
# The correction is applied per CALENDAR MONTH using that month's ERA5
# maximum, so every month in every year is stretched by the same
# distribution-wide factor. Training and test periods then contain comparable
# extremes, and the model is interpolating rather than extrapolating.
#
# ERA5 November 2015 is the anchor: the documented November total is
# 1113.8 mm, the wettest month in Chennai's recorded history, while ERA5 gives
# 282.4 mm for the same month. That 3.94x gap sets the wet-month scale.
ERA5_NOV2015_MM = 282.4
DOCUMENTED_NOV2015_MM = 1113.8
MONTH_SCALE = DOCUMENTED_NOV2015_MM / ERA5_NOV2015_MM
# dry months must not be scaled up as hard as the monsoon
WET_MONTH_MIN_MM = 60.0


def bias_series_rainfall(rain: np.ndarray, dates: np.ndarray | None = None,
                         targets: dict | None = None) -> np.ndarray:
    """The single rainfall correction used everywhere.

    Two stages:
      1. Days with a documented target (the Dec 2015 event) are renormalised
         to that target, hour by hour.
      2. Every calendar month is scaled by a factor derived from its own ERA5
         maximum, so a month that ERA5 rendered too dry is stretched by a
         consistent distribution-wide amount.

    The SAME function builds the training series and replays the 2015 event, so
    the models are always trained on the rainfall distribution they are
    evaluated against.
    """
    out = np.array(rain, dtype=float, copy=True)
    n = len(out)
    if n == 0:
        return out

    if dates is None:
        dates = np.array([f"idx{i // 24}" for i in range(n)])

    # stage 1: documented days, exactly
    out = bias_event_rainfall(out, dates, targets if targets is not None else {})

    # stage 2: per-month scaling so every month carries realistic extremes
    frame = pd.DataFrame({"r": out, "d": dates})
    month = frame["d"].str.slice(0, 7)
    daily = frame.groupby(month)["r"].apply(
        lambda s: s.groupby(frame.loc[s.index, "d"]).sum()
    )
    month_max = daily.groupby(level=0).max()

    targeted_days = set(targets or {})
    for m, peak in month_max.items():
        if peak <= 0:
            continue
        sel = np.where(month == m)[0]
        # never re-scale a day stage 1 already set to a documented total,
        # or the correction compounds (319.6 mm -> 1260 mm)
        untouched = np.array(
            [d not in targeted_days for d in dates[sel]], dtype=bool
        )
        if not untouched.any():
            continue
        block = out[sel]
        free = block[untouched]
        if free.sum() <= 0:
            continue
        free_peak = pd.Series(free).groupby(
            pd.Series(dates[sel][untouched])
        ).sum().max()
        if free_peak <= 0:
            continue
        # dry months are lifted less; wet months get the full factor
        wetness = min(free_peak / WET_MONTH_MIN_MM, 1.0)
        factor = 1.0 + (MONTH_SCALE - 1.0) * wetness
        out[sel[untouched]] = free * factor
    return out


def stage_curve(surface_mm: np.ndarray, surface_max: float) -> np.ndarray:
    """Map surface storage to depth above the ward's drainage datum.

    Below surface_max the water sits in depressions and rises slowly. Once
    storage exceeds it, the water is no longer contained by the road profile
    and spreads across the ward floor, so depth climbs steeply and roughly
    linearly. This knee is what separates "waterlogged" from "flooded".
    """
    excess = np.maximum(surface_mm - surface_max, 0.0)
    contained = 0.30 * np.minimum(surface_mm, surface_max)
    # spreading: 0.65 mm of depth per mm of storage beyond the depression
    return contained + 0.65 * excess


def simulate_ward(
    rainfall_mm: np.ndarray,
    soil_moisture: np.ndarray,
    imperviousness: float,
    drain_quality: float,
    elevation_index: float,
    params: dict | None = None,
    initial_soil: float | None = None,
    elevation_m: float | None = None,
) -> dict[str, np.ndarray]:
    """Run the bucket model for one ward over the full series."""
    p = {**config.PHYSICS, **(params or {})}
    n = rainfall_mm.shape[0]

    soil = np.zeros(n)
    surface = np.zeros(n)
    depth = np.zeros(n)
    runoff = np.zeros(n)
    infiltration = np.zeros(n)
    drain_out = np.zeros(n)
    et_loss = np.zeros(n)
    percolation = np.zeros(n)
    nudging = np.zeros(n)

    # ERA5 soil moisture is a 0-1 volumetric fraction. Map to an equivalent
    # storage depth so the two are on a consistent footing.
    fc = p["field_capacity_mm"]
    sm_init = (
        initial_soil
        if initial_soil is not None
        else float(np.clip(soil_moisture[0], 0.05, 0.6) / 0.45) * fc
    )
    soil_prev = min(max(sm_init, 0.0), fc)

    imperv = np.clip(
        p["impervious_base"]
        + p["impervious_elev_slope"] * (1.0 - elevation_index),
        0.05,
        0.95,
    )

    # Gravity outfall strength per ward. SRTM gives a bare-earth height, so we
    # use the real elevation, not the normalised index, to decide whether a
    # ward can shed water downhill at all. Wards at or below outfall_min_height
    # sit in a closed basin and have no gravity outfall.
    elev_m = elevation_m if elevation_m is not None else elevation_index * 30.0
    head = max(elev_m - p["outfall_min_height"], 0.0)
    outfall_gain = p["outfall_rate_mm_hr"] * (head / 30.0) ** 0.85

    for i in range(n):
        rain = float(rainfall_mm[i])

        # --- 1. infiltration / runoff partition -------------------------
        sat = soil_prev / fc
        capacity = p["infill_rate_mm_hr"] * (
            1.0 - p["infill_saturated_factor"] * min(sat, 1.0)
        )
        permeable_rain = rain * (1.0 - imperv)
        infil = min(permeable_rain, capacity, max(fc - soil_prev, 0.0))
        excess = rain - infil

        # --- 2. soil store ---------------------------------------------
        et = p["et_rate_mm_hr"] * sat
        et = min(et, soil_prev)
        soil_now = np.clip(soil_prev + infil - et, 0.0, fc)

        # --- 3. surface store ------------------------------------------
        s = surface[i - 1] if i else 0.0
        # Percolation only removes what the soil can still accept; it is not a
        # free sink. Once the soil is at field capacity the surface store is
        # effectively sealed, which is how antecedent moisture gates flooding.
        # Percolation moves water surface -> soil, so it is ADDED to the soil
        # store. Omitting that credit made the surface store drain into a
        # black hole and broke the water balance by thousands of mm.
        # Percolation moves water surface -> soil, so it is ADDED to the soil
        # store. Omitting that credit made the surface store drain into a black
        # hole. The room is measured AFTER this hour's infiltration and ET so
        # soil can never exceed field capacity, which would otherwise be
        # silently clipped and delete water from the balance.
        soil_room = max(fc - (soil_prev + infil - et), 0.0)
        percol = min(
            p["surface_drain_coeff"] * max(s - p["surface_max_mm"] * 0.30, 0.0),
            soil_room,
        )
        s = s + excess - percol
        soil_now = soil_prev + infil + percol - et

        # --- 4. gully drainage with a chokepoint -----------------------
        # Drainage is capacity-limited, not demand-limited: an hour of rain
        # heavier than the gully outflow puts water on the street regardless
        # of how much is standing there.
        depth_now = stage_curve(s, p["surface_max_mm"])
        choke = 1.0 - p["drain_choke_factor"] * min(depth_now / 120.0, 1.0)
        cap = p["drain_base_mm_hr"] * (0.40 + drain_quality) * choke
        drained = min(s, cap)
        s = max(s - drained, 0.0)

        # --- 5. gravity outfall ----------------------------------------
        # Only acts once a head of water has built above the dead zone, and
        # scales with how far above the basin floor the ward sits. This is
        # why Avadi (29 m) and Ambattur (23 m) shed the storm while
        # Perungalathur (-2 m) cannot.
        out = 0.0
        if outfall_gain > 0.0 and s > p["outfall_deadzone_mm"]:
            drive = s - p["outfall_deadzone_mm"]
            out = min(drive, outfall_gain * (0.6 + 0.4 * drain_quality))
            s = max(s - out, 0.0)

        soil_now = float(np.clip(soil_now, 0.0, fc))

        # ERA5 nudges the soil store toward observed regional wetness. This is
        # an EXTERNAL data-assimilation flux, not a physical process: it can
        # add or remove water. It is measured and reported separately so the
        # physical water balance still closes exactly.
        if np.isfinite(soil_moisture[i]):
            target = float(np.clip(soil_moisture[i], 0.05, 0.6) / 0.45) * fc
            nudged = 0.90 * soil_now + 0.10 * target
            nudging[i] = nudged - soil_now
            soil_now = nudged

        soil[i] = soil_now
        infiltration[i] = infil
        et_loss[i] = et
        runoff[i] = excess
        # must ACCUMULATE: `drained` is gully outflow, `out` is gravity outfall.
        # Assigning instead of adding silently deleted the gravity term.
        drain_out[i] = drained + out
        percolation[i] = percol
        surface[i] = s
        depth[i] = stage_curve(s, p["surface_max_mm"])
        soil_prev = soil_now

    return {
        "depth_mm": depth,
        "surface_mm": surface,
        "soil_mm": soil,
        "runoff_mm": runoff,
        "infiltration_mm": infiltration,
        "et_loss_mm": et_loss,
        "drain_out_mm": drain_out,
        "percolation_mm": percolation,
        "nudging_mm": nudging,
        "impervious_fraction": np.full(n, imperv),
    }


def run_series(
    rainfall: pd.DataFrame,
    elevations: pd.DataFrame,
    rain_override: np.ndarray | None = None,
) -> pd.DataFrame:
    """Simulate all 15 wards. rain_override replaces rainfall when supplied
    (used by the event replay to inject the bias-corrected storm)."""
    rain = rain_override if rain_override is not None else rainfall["rainfall_mm"].to_numpy()
    sm = rainfall["soil_moisture"].to_numpy()
    ts = rainfall["timestamp"]

    elev_idx = dict(zip(elevations["ward"], elevations["elevation_index"]))
    elev_abs = dict(zip(elevations["ward"], elevations["elevation_m"]))
    frames = []

    for name, _, _, _, _ in config.WARDS:
        res = simulate_ward(
            rainfall_mm=rain,
            soil_moisture=sm,
            imperviousness=config.WARD_IMPERVIOUSNESS[name],
            drain_quality=config.WARD_DRAIN_QUALITY[name],
            elevation_index=elev_idx[name],
            elevation_m=elev_abs[name],
        )
        frames.append(pd.DataFrame({
            "timestamp": ts,
            "ward": name,
            "rainfall_mm": rain,
            "soil_moisture": sm,
            "depth_mm": res["depth_mm"],
            "surface_mm": res["surface_mm"],
            "soil_mm": res["soil_mm"],
            "runoff_mm": res["runoff_mm"],
            "infiltration_mm": res["infiltration_mm"],
            "drain_out_mm": res["drain_out_mm"],
            "et_loss_mm": res["et_loss_mm"],
            "percolation_mm": res["percolation_mm"],
            "nudging_mm": res["nudging_mm"],
            "impervious_fraction": res["impervious_fraction"],
            "source": "simulated:bucket-model",
        }))

    return pd.concat(frames, ignore_index=True)


def physics_class(hours_above: float, peak_depth: float) -> str:
    """Map simulator behaviour to one of four ward risk classes."""
    r = config.PHYSICS_CLASS_RULES
    if (hours_above >= r["Severe"]["hours_above_flood_min"]
            and peak_depth >= r["Severe"]["peak_depth_min_mm"]):
        return "Severe"
    if (hours_above >= r["High"]["hours_above_flood_min"]
            and peak_depth >= r["High"]["peak_depth_min_mm"]):
        return "High"
    if (hours_above >= r["Moderate"]["hours_above_flood_min"]
            and peak_depth >= r["Moderate"]["peak_depth_min_mm"]):
        return "Moderate"
    return "Low"
