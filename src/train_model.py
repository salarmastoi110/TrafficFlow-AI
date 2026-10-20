"""Train the traffic volume model and save it to models/traffic_model.pkl."""
import json
import sys
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing import FEATURES, ROOT, TARGET, build_preprocessor, load_data

MODEL_PATH = ROOT / "models" / "traffic_model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"


def train(save: bool = True) -> dict:
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline(
        [
            ("prep", build_preprocessor()),
            ("reg", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=42)),
        ]
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {
        "mae": round(float(mean_absolute_error(y_te, pred)), 2),
        "rmse": round(float(mean_squared_error(y_te, pred) ** 0.5), 2),
        "r2": round(float(r2_score(y_te, pred)), 4),
        "train_rows": int(len(X_tr)),
        "test_rows": int(len(X_te)),
    }
    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    m = train()
    print("Model saved to", MODEL_PATH)
    print(json.dumps(m, indent=2))
