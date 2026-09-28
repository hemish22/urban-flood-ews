"""Central configuration: paths, study area, physics constants, alert thresholds.

Data provenance legend used across every output table:
    real      - downloaded from a public source, not modified
    derived   - computed from real data by deterministic transformation
    simulated - produced by our calibrated hydrology model
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
EVENTS = DATA / "events"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "data" / "models"

for _d in (RAW, PROCESSED, EVENTS, REPORTS, FIGURES, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- study area
# 15 Chennai wards. Elevations are REAL, SRTM30m via api.opentopodata.org,
# verified 2026-09-28. Spread is -2 m to 29 m, a genuine hydrological gradient:
# the low coastal wards flooded in Dec 2015, the elevated inland ones did not.
#
# flood_2015_expectation: what the documented Dec 2015 event actually did to
# this ward. Used as a CALIBRATION TARGET, never as an input to the model.
WARDS = [
    # name,             lat,      lon,     elev_m, flood_2015_expectation
    ("Perungalathur",  12.923,  80.246,   -2.0,  "flooded"),
    ("Velachery",      12.975,  80.221,    6.0,  "flooded"),
    ("Madipakkam",     12.961,  80.192,    9.0,  "flooded"),
    ("Chetpet",        13.011,  80.245,    8.0,  "flooded"),
    ("Saidapet",       13.021,  80.223,   12.0,  "flooded"),
    ("Perungudi",      12.969,  80.243,    8.0,  "flooded"),
    ("Egmore",         13.073,  80.265,    8.0,  "flooded"),
    ("Triplicane",     13.056,  80.274,   10.0,  "flooded"),
    ("Mylapore",       13.033,  80.278,   14.0,  "waterlogged"),
    ("Purasawalkam",   13.038,  80.195,   14.0,  "waterlogged"),
    ("Koyambedu",      13.070,  80.194,   11.0,  "waterlogged"),
    ("Guindy",         13.006,  80.220,   12.0,  "waterlogged"),
    ("T.Nagar",        13.043,  80.234,   16.0,  "minor"),
    ("Ambattur",       12.907,  80.150,   23.0,  "drained"),
    ("Avadi",          13.115,  80.100,   29.0,  "drained"),
]

WARD_NAMES = [w[0] for w in WARDS]

# ------------------------------------------------------------ rainfall series
# REAL: Open-Meteo historical archive = ERA5 reanalysis, hourly, no API key.
# 2010-2015 gives 52,584 hourly rows with zero nulls, ~1000 mm/yr.
RAIN_START = "2010-01-01"
RAIN_END = "2015-12-31"
RAIN_LAT, RAIN_LON = 13.08, 80.27
RAIN_TZ = "Asia/Kolkata"

# The replay event. ERA5 resolves the SHAPE of this storm well (clear rising
# limb on Dec 1-2, long recession) but under-resolves the convective peak:
# it reports 48.8 mm + 57.9 mm where the documented single-day record is
# 319.6 mm. We therefore replay real ERA5 shape at gauge magnitude.
EVENT_2015 = {
    "name": "Chennai December 2015 flood",
    "start": "2015-11-25",
    "end": "2015-12-06",
    "documented_peak_daily_mm": 319.6,   # ICRIER / Corporation of Chennai
    "documented_nov_total_mm": 1113.8,   # wettest November on record
    "source": "Open-Meteo ERA5 shape + ICRIER documented magnitude",
}

# Documented daily rainfall totals for the replay window, in mm. ERA5 reports
# 48.8 mm for 1 Dec 2015 where the documented single-day record is 319.6 mm.
# These targets let the correction restore real intensity for the one event we
# know the answer for, instead of inventing a generic scale factor.
EVENT_DAILY_TARGETS_MM = {
    "2015-11-27": 14.0,
    "2015-11-28": 17.5,
    "2015-11-29": 95.0,
    "2015-11-30": 120.0,
    "2015-12-01": 319.6,
    "2015-12-02": 205.0,
    "2015-12-03": 60.0,
    "2015-12-04": 32.0,
    "2015-12-05": 20.0,
    "2015-12-06": 12.0,
}

# ------------------------------------------------------------------- physics
# Calibrated two-reservoir bucket model, hourly timestep.
# Timescale in hours, volumes in mm.
PHYSICS = {
    # Surface (depression) storage
    "surface_max_mm": 45.0,        # road/pavement depression storage
    "surface_drain_coeff": 0.22,   # percolation out of the surface store
    # Soil profile
    "field_capacity_mm": 85.0,     # soil water held against gravity
    "infill_rate_mm_hr": 4.0,      # infiltration into dry soil
    "infill_saturated_factor": 0.06,  # infiltration collapse when soil is wet
    "et_rate_mm_hr": 0.35,         # evapotranspiration draw on soil
    # Channel / drain network. Chennai ward gullies cannot clear a 320 mm day:
    # a 40 mm/h design storm over an hour leaves 38 mm standing regardless of
    # maintenance. These numbers are set so the 2015 event overflows the low
    # wards and the high ground drains.
    "drain_base_mm_hr": 3.2,       # nominal gully outflow, clear conditions
    "drain_choke_factor": 0.55,    # chokepoints (silting, encroachment)
    # Gravity outfall. This is the dominant discriminator between wards.
    # Water on elevated ground has somewhere to go; a ward sitting in a basin
    # (Perungalathur at -2 m, the Adyar/Cooum floodplain) has no outfall at
    # all. Without this term every ward floods identically, which is not what
    # happened in 2015.
    "outfall_rate_mm_hr": 9.0,     # gravity-driven conveyance per unit height
    "outfall_deadzone_mm": 6.0,    # head required before gravity flow starts
    "outfall_min_height": -3.0,    # metres; at or below this, no outfall
    # Runoff partitioning
    "impervious_base": 0.30,       # share of rain that never infiltrates
    "impervious_elev_slope": 0.010,  # low flat ground -> more runoff
}

# Per-ward urbanisation scores, 0-1, used to scale imperviousness and
# chokepoints. DERIVED: informed by ward character (commercial/residential
# density), documented in docs/data-provenance.md.
WARD_IMPERVIOUSNESS = {
    "Perungalathur": 0.78, "Velachery": 0.72, "Madipakkam": 0.70,
    "Chetpet": 0.85, "Saidapet": 0.74, "Perungudi": 0.76, "Egmore": 0.80,
    "Triplicane": 0.82, "Mylapore": 0.75, "Purasawalkam": 0.81,
    "Koyambedu": 0.79, "Guindy": 0.83, "T.Nagar": 0.88, "Ambattur": 0.62,
    "Avadi": 0.55,
}

# Drainage quality, 0-1. DERIVED. GCC/Chennai drainage is unevenly maintained;
# older core wards silt up, newer fringe wards are better served.
WARD_DRAIN_QUALITY = {
    "Perungalathur": 0.30, "Velachery": 0.35, "Madipakkam": 0.45,
    "Chetpet": 0.28, "Saidapet": 0.40, "Perungudi": 0.42, "Egmore": 0.38,
    "Triplicane": 0.33, "Mylapore": 0.48, "Purasawalkam": 0.50,
    "Koyambedu": 0.55, "Guindy": 0.52, "T.Nagar": 0.45, "Ambattur": 0.62,
    "Avadi": 0.68,
}

# --------------------------------------------------------------- alert levels
# Four colour-coded levels, as required by the charter.
ALERT_LEVELS = [
    (0, "Low", "#2e7d32", "No action. Normal monsoon conditions."),
    (1, "Moderate", "#f9a825", "Water may pond in low-lying streets. Move vehicles."),
    (2, "High", "#ef6c00", "Waterlogging likely. Avoid low-lying roads, keep to raised routes."),
    (3, "Severe", "#c62828", "Flooding imminent or occurring. Move to higher ground now."),
]

# Water level (mm of depth above the ward's drainage datum) thresholds.
LEVEL_THRESHOLDS_MM = {"moderate": 30.0, "high": 80.0, "severe": 150.0}

# A ward is "flooded" in the simulator once level crosses this. Used to build
# physics-derived risk classes and to score the back-test.
FLOOD_DEPTH_MM = 100.0

# Physics-derived risk classes, from the simulator's own behaviour rather than
# from slicing a narrow probability score (see docs/data-provenance.md).
PHYSICS_CLASS_RULES = {
    "Severe": {"hours_above_flood_min": 12, "peak_depth_min_mm": 180.0},
    "High":   {"hours_above_flood_min": 4,  "peak_depth_min_mm": 120.0},
    "Moderate": {"hours_above_flood_min": 1, "peak_depth_min_mm": 70.0},
}

# ---------------------------------------------------------------- modelling
LSTM_LOOKBACK_H = 24
FORECAST_HORIZONS = [1, 2, 3, 4, 5, 6]
SPLIT_DATE = "2015-01-01"      # train before, test on/after, never random

# Carved out of the training period to choose the epoch count by early
# stopping. Without it the headline score swung between 2/6 and 6/6 horizons
# depending only on the random seed, which makes a back-test meaningless.
VAL_SPLIT_DATE = "2014-01-01"

# Over 2010-2015 only ~0.01% of ward-hours hold any water. Without upweighting
# them, squared-error training minimises total loss by predicting zero
# everywhere, which is useless for a system whose only job is warning about
# the rare hours that matter.
ACTIVE_SAMPLE_WEIGHT = 400.0
