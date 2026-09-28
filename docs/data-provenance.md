# Data provenance

Every input is labelled **real**, **derived**, or **simulated**. Nothing here is
presented as an observation it is not.

---

## 1. Rainfall and soil moisture — real, with a documented correction

| | |
|---|---|
| Source | Open-Meteo Historical Weather API (ERA5 reanalysis) |
| Variables | `precipitation`, `soil_moisture_0_to_7cm` |
| Location | 13.0827 N, 80.2707 E (Chennai) |
| Period | 2010-01-01 to 2015-12-31 |
| Rows | 52,584 hourly, zero nulls |
| Mean annual | 998.4 mm |
| File | `data/raw/rainfall.csv` |

### Why a correction is unavoidable

ERA5 is a reanalysis product on a roughly 31 km grid. It smooths away
convective extremes, and the December 2015 Chennai event was exactly that kind
of storm.

| Quantity | ERA5 | Documented | Ratio |
|---|---|---|---|
| Wettest daily total, whole record | 115.7 mm (2015-11-16) | 319.6 mm (2015-12-01) | 2.8x |
| November 2015 monthly total | 282.4 mm | 1113.8 mm | 3.9x |
| Days above 100 mm | 2 | - | - |
| Days above 200 mm | 0 | - | - |

The November 2015 ratio (3.94x) is the anchor, because 1113.8 mm is documented
as the wettest month in Chennai's recorded history.

### What the correction does

Two stages, in `src/simulate/hydrology.py::bias_series_rainfall`:

1. **Documented days** are renormalised hour by hour to their documented daily
   total. December 1 2015 lands on exactly 319.6 mm.
2. **Every other calendar month** is scaled by a factor derived from that
   month's own ERA5 maximum, damped for dry months:
   `factor = 1 + (3.94 - 1) * min(month_max / 60, 1)`.

Days already fixed in stage 1 are excluded from stage 2, otherwise the
correction compounds. This actually happened during development: 319.6 mm
became 1260 mm before the exclusion was added.

### Why per-month and not per-day

The first attempt used a convex per-day transform. It failed. ERA5 contains no
day above 115.7 mm before December 2015, so there was no "extreme day" for the
transform to amplify. Training examples topped out at 12 mm of water while the
2015 replay reached 287 mm, and the LSTM was being asked to extrapolate 25x
beyond anything it had seen. Per-month scaling gives every year realistic
extremes (138 to 350 mm peak depth) and turns the model into an interpolator.

**The same function builds the training series and the 2015 replay.** A single
rainfall series is written to `data/processed/rainfall_corrected.csv` and read
back by feature building, so the two cannot drift apart. An earlier version
re-derived the correction independently inside `features/build.py` and silently
double-corrected the test period.

### Status

**Derived.** Hourly *shape* is ERA5; hourly *magnitude* on documented days is
imposed. The replay is described throughout as "ERA5 shape, gauge-magnitude
event" and never as a reanalysis of the real storm.

---

## 2. Elevations — real

| | |
|---|---|
| Source | OpenTopData SRTM30m |
| Sampling | One point per ward centroid |
| Wards | 15 |
| File | `data/raw/elevations.csv` |

Range is -2 m (Perungalathur) to 29 m (Avadi). SRTM gives a bare-earth height,
so it is used to decide whether a ward can shed water downhill at all: wards
at or below the outfall minimum sit in a closed basin with no gravity outfall.

| Ward | Elevation (m) | Ward | Elevation (m) |
|---|---|---|---|
| Perungalathur | -2 | Koyambedu | 11 |
| Velachery | 6 | Saidapet | 12 |
| Chetpet | 8 | Guindy | 12 |
| Perungudi | 8 | Mylapore | 14 |
| Egmore | 8 | Purasawalkam | 14 |
| Madipakkam | 9 | T.Nagar | 16 |
| Triplicane | 10 | Ambattur | 23 |

Sorted values: -2, 6, 8, 8, 8, 9, 10, 11, 12, 12, 14, 14, 16, 23, 29 m.

**Real.** No synthetic values.

---

## 3. Static ward attributes — assumed

Imperviousness and drain quality are plausible assigned values, not survey
data. They live in `src/config.py::WARD_IMPERVIOUSNESS` and
`WARD_DRAIN_QUALITY` and are clearly marked as assumptions.

**Assumed.** The simulator's response to them is calibrated so that the
resulting ward ordering is physically sensible, but the specific numbers are
not evidence. This is a real limitation: the XGBoost model's top features
include imperviousness and drain quality, so part of its 0.895 F1 rests on
assumed inputs.

---

## 4. Water levels — simulated

| | |
|---|---|
| Source | `src/simulate/hydrology.py` bucket model |
| Scale | 788,760 ward-hours (15 wards x 52,584 hours) |
| File | `data/processed/hourly_panel.parquet` |

Two stores per ward: a soil column and a surface store. Infiltration is
capacity-limited and falls as the soil fills. Percolation is gated by remaining
soil storage, so a saturated soil seals the surface and that is what gates
flooding. Gully drainage is capacity-limited, not demand-limited, with a
depth-dependent chokepoint. Gravity outfall scales with real elevation. A stage
curve converts storage to depth above the ward's drainage datum.

### Water balance

The verified identity, per ward, per hour:

```
rain + nudge = surface_out + et + d(surface) + d(soil)
```

`runoff_mm` is a flux *into* the surface store and is already contained in
`d(surface)`; counting it as a sink double-counts it. Stock terms enter as
final-minus-initial differences, never as sums.

Current residual: **3.5e-06 mm/hr** hourly, **0.20 mm** over six years
(**0.0012%** of the 16,560 mm that fell).

`nudging` is the ERA5 soil-moisture assimilation term. It is a data-assimilation
flux, not a physical process, and can add or remove water. It is reported
separately so the physical balance still closes exactly.

### Calibration

Physics parameters were tuned so the December 2015 replay reproduces the
documented event: peak daily rainfall 319.6 mm, flood/no-flood agreement in
**14 of 15** wards. T.Nagar disagrees and is left as a discrepancy rather than
tuned away.

**Simulated.** This is the single most important limitation in the project. The
LSTM learns to predict this model's output, so its measured skill is skill at
predicting the simulator. None of it transfers to real gauges without
retraining on observations.

---

## 5. Kaggle reference dataset — real, but not used operationally

| | |
|---|---|
| Rows | 1,117,957 |
| Features | 20 |
| File | `data/raw/kaggle_chennai_flood.csv` |

Used only to reproduce the plan's stated reference method, which reaches
**0.610 macro F1**.

That score is hollow and is not presented as a result. The target is
quartile-banded from a distribution with standard deviation 0.051, so the
model is largely learning quartile boundaries rather than flood risk. The
projected probability ranges per band confirm it: q1 0.285-0.465, q2
0.470-0.500, q3 0.505-0.535, q4 0.540-0.725.

**Real data, misleading task.** Demonstrates that a high F1 can be meaningless,
which is the point of including it.

---

## Summary

| Input | Status | Used for |
|---|---|---|
| Hourly rainfall shape | real (ERA5) | model input, feature |
| Hourly rainfall magnitude, documented days | derived | event replay |
| Monthly rainfall extremes | derived | training realism |
| Elevation | real (SRTM30m) | outfall physics, features |
| Imperviousness, drain quality | assumed | physics, features |
| Water depth | simulated | target, ground truth |
| December 2015 daily totals | documented | calibration |
| Kaggle labels | real, narrow target | reference only |
