# 🚦 TrafficFlow-AI

Traffic analytics and prediction dashboard. Explore hourly traffic patterns across city junctions, forecast
vehicle volume for any hour/day/weather combination, and see the predicted congestion level.

## Features
- **Dashboard**: KPIs, hourly traffic by junction, weekly heatmap
- **Prediction**: predicted vehicles/hour + congestion level (Low / Moderate / High / Severe) and a 24-hour forecast
- **Analytics**: model metrics (MAE, RMSE, R²), weather impact, congestion distribution, daily trends, CSV download

## Project structure
```
app.py                  Streamlit app
data/traffic_data.csv   Synthetic hourly dataset (4 junctions, 180 days)
models/traffic_model.pkl Trained model (HistGradientBoostingRegressor pipeline)
notebooks/              Training / EDA notebook
src/preprocessing.py    Data generation, feature engineering, preprocessing
src/train_model.py      Training + evaluation
src/prediction.py       Inference helpers
screenshots/, demo/     Add your screenshots and demo video here
```

## Quick start
```bash
pip install -r requirements.txt
python src/preprocessing.py     # (re)generate data/traffic_data.csv
python src/train_model.py       # train and save models/traffic_model.pkl
streamlit run app.py
```
The app trains the model automatically on first launch if `traffic_model.pkl` is missing.

## Model
Features: junction, weather, cyclical hour (sin/cos), day of week, weekend, holiday, temperature.
Target: vehicles per hour. Congestion level is derived from thresholds (<100 Low, <200 Moderate, <300 High, 300+ Severe).

## Using real data
Replace `data/traffic_data.csv` with your own file using the same columns
(`timestamp, hour, day_of_week, is_holiday, weather, temperature, is_weekend, hour_sin, hour_cos, junction, vehicles, congestion`)
and re-run `python src/train_model.py`.

## Screenshots
Run the app and save captures as `screenshots/dashboard.png`, `prediction.png`, `analytics.png`
(a generated `analytics.png` is included as a starting point). Put a screen recording at `demo/TrafficFlow-AI-Demo.mp4`.

## License
MIT
