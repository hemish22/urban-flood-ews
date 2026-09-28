# Method notes

Why the models are built the way they are, including the parts that did not
work. The failures are the more useful half.

---

## 1. Why a residual formulation for the LSTM

Water depth is strongly autocorrelated. A "assume it stays where it is"
forecast is a genuinely hard baseline to beat, and at t+1 the level has barely
moved in an hour, so persistence is close to optimal there.

Asking a network to output an absolute level makes it spend capacity
re-deriving the current level from its own inputs. The fix is to predict the
correction instead:

```
log1p(depth[t+h]) = log1p(depth[t]) + correction[t+h]
```

The head is **zero-initialised**, so at initialisation the model reproduces
persistence exactly and every subsequent weight is a genuine improvement over
the baseline. That framing also makes the metric interpretable: the model's
`log` correction at each horizon is directly the mm of rise or fall it
predicts over the next few hours.

### Per-horizon scaling

Corrections are scaled by each horizon's own training standard deviation:

```
horizon std   h1=0.0507  h2=0.0834  h3=0.1086  h4=0.1295  h5=0.1474  h6=0.1638
```

With one shared scalar the long horizons dominate the gradient and actively
damage the short ones. Adding per-horizon scaling moved t+1 from -183% to
+25.8% in a single change.

---

## 2. Why the loss is `log1p` with storm upweighting

Over 2010-2015 about 68% of ward-hours hold no water at all, and the depth
distribution is extremely heavy-tailed. Under squared error the loss-optimal
prediction is near zero almost everywhere.

The first trained model showed the signature clearly: **MAE 0.97 mm alongside
RMSE 11.4 mm**. Low average error with enormous errors on the rare windows. The
network was not failing at the task; it was correctly minimising a loss that
never asked it about the storms.

Three changes, applied together:

1. `log1p` target, so 200 mm of depth matters as much as 1 mm.
2. Sample weight of 400 on storm-active windows, defined as any window with
   depth > 1 mm, any 24 h rain total > 2 mm, or any forecast depth > 1 mm.
3. Metrics reported over **both** all windows and the storm-active subset.

Point 3 is the important one. An early-warning claim is only meaningful on the
storm subset, and publishing the all-window number alone would hide the
difference. Both are printed, and the JSON records both.

---

## 3. Why the epoch count is chosen on a validation year

This is the single most important reproducibility finding in the project.

With the epoch count set by hand, the headline result swung between **2/6 and
6/6 horizons won** on the random seed alone, with identical code and data. The
run that produced 6/6 was luck. Reporting it would have been reporting noise.

The fix has two parts:

- **Seeding.** `set_seed()` covers Python, NumPy, Torch and the MPS backend.
  Two consecutive runs now produce byte-identical metrics.
- **Early stopping.** 2014 is carved out of the training period as a
  validation year. The schedule runs up to 30 epochs with patience 6, and the
  best-validation weights are restored. It selects **epoch 4** of 30, where
  validation RMSE bottoms out at 1.4586 mm and then degrades.

The validation curve is why this works: loss keeps falling to epoch 10 while
validation RMSE has already turned upward from epoch 4. The model is
overfitting from the fifth epoch onward, and only a held-out set can see that.

Early stopping is not an optimisation convenience here. It is what makes the
back-test mean anything.

---

## 4. Why XGBoost predicts ward rank, not storm severity

The susceptibility target is a ward's **rank within a single storm** (Low,
Moderate, High, Severe, 1/4 each). This is a deliberate choice.

An absolute-severity target would be almost pure noise: storm-wide features
like total rainfall are shared by all 15 wards, so they cannot explain which
ward is worst. Within a storm, those features are constant and drop out, and
what remains is exactly the static ward character (elevation, imperviousness,
drain quality) that determines local flood response.

The result is a model that answers "which ward floods first in this storm",
which is the operationally useful question. Top features confirm the intent:
`rain_max_hour_mm`, `elevation_m`, `elevation_index`, `imperviousness`,
`drain_quality`.

**The honest caveat:** the 0.895 macro F1 is easier than it looks. Rank
targeting balances the classes and removes storm-scale variance. A high score
here means ward ranking is learnable from ward character. It does not mean the
model can predict how bad a storm will be. The physics-derived absolute
classes are computed separately and drive the actual alerts.

---

## 5. Why the back-test protocol is written the way it is

The headline number in a flood back-test is trivially fakeable. The
countermeasures, in order of importance:

1. **The engine never sees the simulator's future.** Alerts are raised from
   current state plus LSTM forecast. The simulator is consulted once,
   afterwards, purely to score.
2. **Training stops at 2014-12-31.** The model is never refitted on 2015.
3. **An alert only counts if it fires before the flood begins.** Crediting an
   alert that arrives after the water is already there would inflate recall
   without warning anyone of anything.
4. **Lead time is measured from the latest alert before the flood onset**,
   not by counting alert hours. This is deliberately conservative: an alert
   that has been on for ten hours still scores only the gap from its most
   recent issue.
5. **Episodes are scored on 2015 only.** Alerts exist only for the test year;
   an earlier version scored against 2010-2015 and counted every pre-2015
   episode as missed, reporting 41.7% episode recall instead of 97.7%.
6. **The alert rate is reported.** 3.1% of ward-hours. A system that warned
   constantly would score well on recall while being useless.

---

## 6. Physics parameter calibration

Parameters in `src/config.py::PHYSICS` were tuned so the December 2015 replay
reproduces the documented event: peak daily rainfall 319.6 mm and flood
agreement in 14 of 15 wards.

This is **calibration to a single event**, and that is its weakness. A single
storm cannot identify 12-plus parameters; many combinations fit December 2015
while implying different behaviour on other storms. The parameters are
physically motivated rather than fitted freely, which limits the damage, but
they are not independently identified.

T.Nagar is a known disagreement with the documented 2015 impact. It was left
unfixed deliberately. Tuning until every ward agrees would demonstrate that
the parameters can be fitted, not that the model works.

---

## 7. What would make this operational

In rough order of value:

1. **Real gauge data.** The entire evaluation is against a simulator. Nothing
   here establishes skill at predicting real water levels.
2. **Rainfall nowcasting.** The 1-3 h lead time is capped by how fast the
   catchment responds, not by LSTM quality. Forecasting rainfall further ahead
   buys more warning than any model improvement.
3. **Multi-event calibration.** Calibrating on December 2015 alone is one
   storm. Several documented events would constrain the parameters.
4. **Richer antecedents.** Soil moisture and 72 h antecedent rainfall are
   coarse. Real antecedent wetness is the dominant control on flood response.
5. **Street-level drainage.** Ward-mean imperviousness cannot represent a
   ward's low points, which is where water actually collects.
