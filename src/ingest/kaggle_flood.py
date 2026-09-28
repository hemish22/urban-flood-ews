"""Download the Kaggle Flood Prediction Dataset.

Source: https://www.kaggle.com/datasets/aritra100/flood-prediction-dataset
Ships 1,117,957 rows x 20 integer features (0-19) plus a continuous
FloodProbability target in [0.285, 0.725].

WHAT THIS DATASET IS GOOD FOR
    A large, free, public corpus that exercises the XGBoost pipeline and
    demonstrates which physical factors carry relative risk signal. It is a
    public teaching dataset, generated rather than surveyed in any one place.

WHAT IT IS NOT
    It is not Chennai, and it is not observed flood data. Its target has a
    standard deviation of 0.051, so banding it into quartiles yields four
    arbitrary slices of a narrow score. The ward risk classes the dashboard
    colours come from the simulator's own physics; the Kaggle score is
    reported alongside as a cross-check, clearly labelled as such.

The CSV is 29 MB zipped. We cache the extraction and skip re-download.
"""

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config

URL = "https://www.kaggle.com/api/v1/datasets/download/aritra100/flood-prediction-dataset"
TRAIN_ENTRY = "train.csv/train.csv"
FEATURES = [
    "MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Deforestation",
    "Urbanization", "ClimateChange", "DamsQuality", "Siltation",
    "AgriculturalPractices", "Encroachments", "IneffectiveDisasterPreparedness",
    "DrainageSystems", "CoastalVulnerability", "Landslides", "Watersheds",
    "DeterioratingInfrastructure", "PopulationScore", "WetlandLoss",
    "InadequatePlanning", "PoliticalFactors",
]
TIMEOUT = 300


def main() -> None:
    out_dir = config.RAW / "kaggle"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "train.csv"

    if target.exists():
        print(f"cached, skipping download: {target}")
    else:
        print("Downloading Kaggle flood dataset (~29 MB)...")
        r = requests.get(URL, timeout=TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        with zf.open(TRAIN_ENTRY) as src, open(target, "wb") as dst:
            dst.write(src.read())
        print(f"  wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")

    df = pd.read_csv(target)
    missing = [c for c in FEATURES + ["FloodProbability"] if c not in df.columns]
    if missing:
        raise RuntimeError(f"expected columns missing from dataset: {missing}")

    tgt = df["FloodProbability"]
    print(f"  rows            {len(df)}")
    print(f"  features        {len(FEATURES)}")
    print(f"  target min/max  {tgt.min():.3f} / {tgt.max():.3f}")
    print(f"  target stdev    {tgt.std():.4f}  <- narrow; see module docstring")
    print(f"  mean feature    {df[FEATURES].to_numpy().mean():.2f} "
          f"(features are near-identically distributed)")

    summary = config.RAW / "kaggle_summary.csv"
    df[FEATURES + ["FloodProbability"]].describe().T.reset_index().rename(
        columns={"index": "feature"}
    ).to_csv(summary, index=False)
    print(f"  wrote {summary}")


if __name__ == "__main__":
    main()
