"""Data generation, loading, feature engineering and preprocessing."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "traffic_data.csv"

# junction -> (scale, morning weight, evening weight, midday weight, leisure weight)
JUNCTIONS = {
    "City Center": (1.00, 1.0, 1.0, 0.7, 0.9),
    "Airport Road": (0.85, 0.8, 0.9, 0.9, 0.6),
    "Industrial Zone": (0.90, 1.1, 1.0, 0.3, 0.3),
    "Residential Area": (0.70, 0.9, 1.0, 0.5, 1.0),
}
WEATHER = ["Clear", "Rain", "Fog"]
WEATHER_FACTOR = {"Clear": 1.0, "Rain": 0.88, "Fog": 0.82}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

CATEGORICAL = ["junction", "weather"]
NUMERIC = ["hour_sin", "hour_cos", "day_of_week", "is_weekend", "is_holiday", "temperature"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "vehicles"


def _g(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def congestion_level(vehicles: float) -> str:
    if vehicles < 100:
        return "Low"
    if vehicles < 200:
        return "Moderate"
    if vehicles < 300:
        return "High"
    return "Severe"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns from hour / day_of_week."""
    df = df.copy()
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def generate_synthetic_data(days: int = 180, seed: int = 42) -> pd.DataFrame:
    """Hourly vehicle counts for several junctions with realistic daily patterns."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=days * 24, freq="h")
    base = pd.DataFrame({"timestamp": ts})
    base["hour"] = ts.hour
    base["day_of_week"] = ts.dayofweek
    dates = ts.normalize().unique()
    holidays = set(rng.choice(dates, size=max(1, days // 20), replace=False))
    base["is_holiday"] = ts.normalize().isin(holidays).astype(int)
    blocks = rng.choice(WEATHER, size=len(ts) // 6 + 1, p=[0.72, 0.2, 0.08])
    base["weather"] = np.repeat(blocks, 6)[: len(ts)]
    doy = ts.dayofyear.to_numpy()
    base["temperature"] = (
        27 + 6 * np.sin(2 * np.pi * (doy - 110) / 365) + 4 * np.sin(2 * np.pi * (base["hour"] - 9) / 24)
        - np.where(base["weather"] == "Rain", 3, 0) + rng.normal(0, 1.2, len(ts))
    ).round(1)
    base = add_features(base)

    frames = []
    h = base["hour"].to_numpy()
    off_day = ((base["is_weekend"] == 1) | (base["is_holiday"] == 1)).to_numpy()
    wf = base["weather"].map(WEATHER_FACTOR).to_numpy()
    for name, (scale, mw, ew, mid, leis) in JUNCTIONS.items():
        work = 40 + 260 * (mw * _g(h, 8.0, 1.6) + ew * _g(h, 17.5, 1.9)) + 90 * mid * _g(h, 13, 3)
        off = 40 + 140 * mid * _g(h, 13, 3.5) + 170 * leis * _g(h, 19, 2.5)
        flow = scale * np.where(off_day, off, work) * wf
        flow = flow * rng.lognormal(0, 0.07, len(h))
        f = base.copy()
        f["junction"] = name
        f[TARGET] = np.clip(flow, 5, None).round().astype(int)
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df["congestion"] = df[TARGET].map(congestion_level)
    return df.sort_values(["timestamp", "junction"]).reset_index(drop=True)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the dataset, generating (and saving) it first if it doesn't exist."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        generate_synthetic_data().to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=["timestamp"])


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )


if __name__ == "__main__":
    df = generate_synthetic_data()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df):,} rows to {DATA_PATH}")
