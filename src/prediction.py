"""Load the trained model and make traffic predictions."""
import json
from functools import lru_cache

import joblib
import pandas as pd

from src.preprocessing import FEATURES, add_features, congestion_level
from src.train_model import METRICS_PATH, MODEL_PATH


def model_exists() -> bool:
    return MODEL_PATH.exists()


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}


def _frame(junction, hours, day_of_week, weather, temperature, is_holiday) -> pd.DataFrame:
    df = pd.DataFrame({"hour": list(hours)})
    df["junction"] = junction
    df["day_of_week"] = day_of_week
    df["weather"] = weather
    df["temperature"] = temperature
    df["is_holiday"] = int(is_holiday)
    return add_features(df)[FEATURES]


def predict_traffic(junction, hour, day_of_week, weather="Clear", temperature=28.0, is_holiday=False) -> dict:
    """Predict vehicles/hour for one situation."""
    X = _frame(junction, [hour], day_of_week, weather, temperature, is_holiday)
    vehicles = max(0, int(round(load_model().predict(X)[0])))
    return {"vehicles": vehicles, "congestion": congestion_level(vehicles)}


def forecast_day(junction, day_of_week, weather="Clear", temperature=28.0, is_holiday=False) -> pd.DataFrame:
    """Predict all 24 hours for a junction."""
    X = _frame(junction, range(24), day_of_week, weather, temperature, is_holiday)
    out = pd.DataFrame({"hour": range(24), "vehicles": load_model().predict(X).clip(0).round().astype(int)})
    out["congestion"] = out["vehicles"].map(congestion_level)
    return out
