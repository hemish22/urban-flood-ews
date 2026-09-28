"""XGBoost ward-susceptibility classifier.

TARGET
    susceptibility_class from episodes.csv: how badly each ward fared
    relative to the other 14 wards hit by the same storm. This is a static
    property of the ward (elevation, imperviousness, drainage), which is what
    a drainage-prioritisation map needs. See src/features/build.py for why the
    absolute physics class is not used as the training target.

ALSO TRAINED
    A model on the Kaggle Flood Prediction Dataset, as the project plan
    specifies. That dataset is a public teaching corpus with a narrow target
    (stdev 0.051) and near-identically distributed features; it is reported as
    a pipeline demonstration and a feature-importance cross-check, NOT as
    Chennai flood risk. The two results are never mixed.

SPLIT
    Strictly by time, never random. Train on storms before 2015-01-01, test
    after. Random splitting would leak storm structure across the boundary and
    inflate every metric in the report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.ingest.kaggle_flood import FEATURES as KAGGLE_FEATURES

CLASSES = ["Low", "Moderate", "High", "Severe"]
FEATURES = [
    "elevation_m", "elevation_index", "imperviousness", "drain_quality",
    "antecedent_rain_72h_mm", "antecedent_soil_moisture",
    "rain_total_mm", "rain_max_hour_mm", "rain_max_6h_mm",
    "duration_h", "drain_capacity_ratio",
]


def train_susceptibility(ep: pd.DataFrame) -> dict:
    train = ep[ep["split"] == "train"]
    test = ep[ep["split"] == "test"]

    # ordinal encoding, fixed order so class 3 is always Severe
    to_code = {c: i for i, c in enumerate(CLASSES)}
    ytr = train["susceptibility_class"].map(to_code).to_numpy()
    yte = test["susceptibility_class"].map(to_code).to_numpy()
    Xtr = train[FEATURES].to_numpy()
    Xte = test[FEATURES].to_numpy()
    assert not np.isnan(ytr).any() and not np.isnan(yte).any(), "unmapped class label"

    print(f"  train {len(train)} / test {len(test)} ward-storms")
    print(f"  train class counts: {dict(pd.Series(ytr).value_counts())}")
    print(f"  test  class counts: {dict(pd.Series(yte).value_counts())}")

    model = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.06,
        subsample=0.9, colsample_bytree=0.9, min_child_weight=3,
        objective="multi:softprob", num_class=4, eval_metric="mlogloss",
        random_state=42, n_jobs=-1,
    )
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    # y is ordinal-encoded, so "present" must be checked on the codes, not names
    present_codes = sorted(set(yte.tolist()))
    present = [CLASSES[c] for c in present_codes]
    present_idx = present_codes
    if len(present) < 4:
        print(f"  NOTE: test split contains only {present}; metrics cover those classes")
    macro = f1_score(yte, pred, average="macro", labels=present_idx, zero_division=0)
    weighted = f1_score(yte, pred, average="weighted", labels=present_idx, zero_division=0)

    maj = int(pd.Series(ytr).mode().iloc[0])
    base_pred = np.full(len(yte), maj)
    base_f1 = f1_score(yte, base_pred, average="macro", labels=present_idx, zero_division=0)

    print()
    print(f"  macro F1        {macro:.3f}")
    print(f"  weighted F1     {weighted:.3f}")
    print(f"  majority-class  {base_f1:.3f}  (baseline to beat)")
    print()
    print(classification_report(yte, pred, labels=present_idx,
                                target_names=present, zero_division=0))

    print("  confusion matrix (rows = actual, cols = predicted):")
    cm = confusion_matrix(yte, pred, labels=present_idx)
    print("            " + "".join(f"{c:>10s}" for c in present))
    for c, row in zip(present, cm):
        print(f"  {c:10s} " + "".join(f"{v:10d}" for v in row))

    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print()
    print("  feature importance:")
    for f, v in imp.items():
        bar = "#" * int(v * 200)
        print(f"    {f:26s} {v:.3f}  {bar}")

    model.save_model(config.MODELS / "xgb_susceptibility.json")
    scaler_note = "tree model, no scaling required"
    print(f"\n  saved {config.MODELS / 'xgb_susceptibility.json'} ({scaler_note})")

    return {
        "model": model,
        "macro_f1": float(macro),
        "weighted_f1": float(weighted),
        "baseline_f1": float(base_f1),
        "report": classification_report(yte, pred, labels=present_idx,
                                       target_names=present, output_dict=True,
                                       zero_division=0),
        "confusion": cm.tolist(),
        "confusion_labels": present,
        "classes_present_in_test": present,
        "importance": {k: float(v) for k, v in imp.items()},
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }


def train_kaggle_reference() -> dict:
    """The plan's stated method, run and reported honestly.

    Bands the Kaggle FloodProbability into quartiles and fits XGBoost. The
    resulting F1 is high, but it is high because the quartiles are arbitrary
    slices of a narrow continuous score, not because the model has learnt
    anything transferable. We report it with that caveat attached rather than
    presenting it as a flood-risk result.
    """
    path = config.RAW / "kaggle" / "train.csv"
    if not path.exists():
        print("  Kaggle dataset absent, skipping reference model")
        return {}

    df = pd.read_csv(path)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(df), size=300_000, replace=False)
    sub = df.iloc[idx]

    y_cont = sub["FloodProbability"]
    edges = np.percentile(y_cont, [25, 50, 75])
    y = np.digitize(y_cont, edges)  # 0..3

    split = int(len(sub) * 0.8)
    Xtr, ytr = sub[KAGGLE_FEATURES].iloc[:split], y[:split]
    Xte, yte = sub[KAGGLE_FEATURES].iloc[split:], y[split:]

    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        objective="multi:softprob", num_class=4,
        random_state=42, n_jobs=-1,
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    macro = f1_score(yte, pred, average="macro", zero_division=0)

    print(f"  sampled {len(sub):,} of {len(df):,} rows, quartile-banded target")
    print(f"  macro F1        {macro:.3f}   <- high, but see caveat")

    # what the bands actually mean
    for i, lab in enumerate(["q1", "q2", "q3", "q4"]):
        sel = y == i
        if sel.any():
            print(f"    {lab}: n={int(sel.sum()):7,d}  "
                  f"prob range {y_cont[sel].min():.3f}-{y_cont[sel].max():.3f}")

    imp = pd.Series(model.feature_importances_, index=KAGGLE_FEATURES).sort_values(ascending=False)
    print("  top features:", ", ".join(imp.index[:5]))

    model.save_model(config.MODELS / "xgb_kaggle_reference.json")
    return {"macro_f1": float(macro), "n_sampled": int(len(sub)),
            "top_features": list(imp.index[:10]),
            "band_edges": [float(e) for e in edges],
            "caveat": ("Quartile bands of a target with stdev 0.051. High F1 here "
                       "reflects a narrow continuous score, not transferable "
                       "flood-risk skill. Reported for completeness only.")}


def main() -> None:
    print("=" * 68)
    print("1. SUSCEPTIBILITY MODEL  (our ward-storm episodes)")
    print("=" * 68)
    ep = pd.read_csv(config.PROCESSED / "episodes.csv", parse_dates=["episode_start"])
    res = train_susceptibility(ep)

    print()
    print("=" * 68)
    print("2. KAGGLE REFERENCE MODEL  (project plan's stated method)")
    print("=" * 68)
    res["kaggle"] = train_kaggle_reference()

    out = config.MODELS / "susceptibility_metrics.json"
    payload = {k: v for k, v in res.items() if k != "model"}
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
