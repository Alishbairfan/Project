import pandas as pd
import joblib
from datetime import timedelta
import hopsworks
import os
import numpy as np

def fetch_last_24_hours(api_key):
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    fg = fs.get_feature_group("karachi_realtime_features", version=1)
    df = fg.read()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp").resample("H").ffill().reset_index()

    latest_time = df["timestamp"].max()
    last_24h = latest_time - pd.Timedelta(hours=23)
    df_24h = df[df["timestamp"] >= last_24h].reset_index(drop=True)

    numeric_cols = [
        "pm25","pm10","no2","o3","temperature","humidity","wind_speed",
        "hour","day","month","weekday","is_weekend",
        "aqi_change_rate","pm25_to_pm10_ratio","temp_change","humidity_change","weather_code"
    ]
    for col in numeric_cols:
        if col in df_24h.columns:
            df_24h[col] = pd.to_numeric(df_24h[col], errors="coerce")
    df_24h = df_24h.dropna(subset=numeric_cols)

    return df_24h

def load_model_from_registry(api_key, model_name="aqi_predictor", version=None):
    project = hopsworks.login(api_key_value=api_key)
    mr = project.get_model_registry()
    model = mr.get_model(model_name, version=version)
    if model is None:
        raise ValueError(f"Model '{model_name}' not found!")
    model_dir = model.download()
    pkl_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl file in {model_dir}")
    return joblib.load(os.path.join(model_dir, pkl_files[0]))

def predict_next_hours(api_key, model_name="aqi_predictor", output_csv="predicted_aqi_predictions.csv"):
    model = load_model_from_registry(api_key, model_name)
    df_24h = fetch_last_24_hours(api_key)

    if df_24h.empty:
        print("No data to predict.")
        return

    numeric_cols = [
        "pm25","pm10","no2","o3","temperature","humidity","wind_speed",
        "hour","day","month","weekday","is_weekend",
        "aqi_change_rate","pm25_to_pm10_ratio","temp_change","humidity_change","weather_code"
    ]

    df_features = df_24h[numeric_cols].copy()
    last_timestamp = pd.Timestamp.now(tz="Asia/Karachi")  # <-- use current date/time
    last_aqi = df_24h["aqi"].iloc[-1]
    next_pred = last_aqi
    hourly_preds = []

    for hour in range(1, 73):
        X_input = df_features.iloc[[-1]].copy()

        X_input['temperature'] = X_input['temperature'] + np.random.uniform(-1, 1)
        X_input['humidity'] = np.clip(X_input['humidity'] + np.random.uniform(-2, 2), 0, 100)
        X_input['wind_speed'] = np.clip(X_input['wind_speed'] + np.random.uniform(-1, 1), 0, None)
        X_input['pm25'] = np.clip(X_input['pm25'] * np.random.uniform(0.95, 1.05), 0, None)
        X_input['pm10'] = np.clip(X_input['pm10'] * np.random.uniform(0.95, 1.05), 0, None)
        X_input['no2'] = np.clip(X_input['no2'] * np.random.uniform(0.95, 1.05), 0, None)
        X_input['o3'] = np.clip(X_input['o3'] * np.random.uniform(0.95, 1.05), 0, None)
        X_input['hour'] = (X_input['hour'] + 1) % 24
        day_increment = 1 if X_input['hour'].iloc[0] == 0 else 0
        X_input['day'] = (last_timestamp + timedelta(hours=hour)).day
        X_input['month'] = (last_timestamp + timedelta(hours=hour)).month
        X_input['weekday'] = (last_timestamp + timedelta(hours=hour)).weekday()
        X_input['is_weekend'] = 1 if X_input['weekday'].iloc[0] >= 5 else 0

        # Update AQI change rate
        X_input['aqi_change_rate'] = next_pred - last_aqi
        X_input['pm25_to_pm10_ratio'] = X_input['pm25'] / X_input['pm10'] if X_input['pm10'].iloc[0] != 0 else 0
        # Predict
        next_pred = model.predict(X_input.values)[0]
        pred_time = last_timestamp + timedelta(hours=hour)
        hourly_preds.append({"timestamp": pred_time, "predicted_aqi": round(next_pred, 2)})

        
        last_aqi = next_pred
       
        df_features = pd.concat([df_features, X_input], ignore_index=True)

        
        if len(df_features) > 24:
            df_features = df_features.iloc[-24:].reset_index(drop=True)

    hourly_df = pd.DataFrame(hourly_preds)
    hourly_df.to_csv(output_csv, index=False)
    print(f"Saved hourly predictions to {output_csv}")

    # Next 3 days averages
    next_3_days = []
    for day in range(1, 4):
        day_data = hourly_df.iloc[(day-1)*24:day*24]
        avg_aqi = round(day_data["predicted_aqi"].mean(), 2)
        next_3_days.append({
            "day": day,
            "date": (last_timestamp + timedelta(days=day)).strftime("%Y-%m-%d"),
            "avg_predicted_aqi": avg_aqi
        })

    print(f"\nNext Hour AQI: {hourly_df.iloc[0]['predicted_aqi']}")
    print("Next 3 Days Average AQI:")
    for d in next_3_days:
        print(f"Day {d['day']} ({d['date']}): {d['avg_predicted_aqi']}")

    return hourly_df, next_3_days

if __name__ == "__main__":
    api_key = "37GNR2LvcIsDqZd1.NHO665gQEMZGDRaZYCKszx7nenA0CJqLvLdOSMBvbpK2YzVJ7TD7hqw9HxCjRRk6"
    predict_next_hours(api_key)
