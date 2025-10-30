import pandas as pd
import joblib
from datetime import timedelta, datetime
import hopsworks

def fetch_last_24_hours(api_key):
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    fg = fs.get_feature_group("karachi_realtime_features", version=1)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_24h = df.sort_values("timestamp").tail(24).reset_index(drop=True)
    
    numeric_cols = [
        "pm25","pm10","no2","o3","temperature","humidity","wind_speed",
        "hour","day","month","weekday","is_weekend",
        "aqi_change_rate","pm25_to_pm10_ratio","temp_change","humidity_change","weather_code","aqi"
    ]
    for col in numeric_cols:
        if col in df_24h.columns:
            df_24h[col] = pd.to_numeric(df_24h[col], errors="coerce")
    df_24h = df_24h.dropna(subset=numeric_cols)
    return df_24h

def predict_next_3_days(model_path, api_key, output_csv="predicted_aqi_predictions.csv"):
    model = joblib.load(model_path)
    df = fetch_last_24_hours(api_key)
    if df.empty:
        print("No data to predict.")
        return

    hourly_preds = []
    last_row = df.iloc[-1:].copy()
    last_timestamp = df["timestamp"].iloc[-1]

    numeric_cols = df.drop(columns=["timestamp","aqi"]).select_dtypes(include=["number","bool"]).columns

    for hour in range(1, 73):
        X_input = last_row[numeric_cols]
        next_pred = model.predict(X_input)[0]

        pred_time = last_timestamp + timedelta(hours=hour)
        hourly_preds.append({"timestamp": pred_time, "predicted_aqi": round(next_pred,2)})

        new_row = last_row.copy()
        new_row["aqi"] = next_pred
        last_row = new_row

    hourly_df = pd.DataFrame(hourly_preds)

    combined_preds = []

    combined_preds.append({
        "timestamp": hourly_df["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
        "type": "next_hour",
        "predicted_aqi": hourly_df["predicted_aqi"].iloc[0]
    })

    for day in range(1, 4):
        day_data = hourly_df.iloc[(day-1)*24:day*24]
        combined_preds.append({
            "timestamp": (last_timestamp + timedelta(days=day)).strftime("%Y-%m-%d"),
            "type": "daily_aggregate",
            "predicted_aqi": round(day_data["predicted_aqi"].mean(),2)
        })

    result_df = pd.DataFrame(combined_preds)
    result_df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")
    return result_df

if __name__ == "__main__":

    API_KEY = "xo0aJonG6geNXT6N.Y1Xs6QMqrk1qExXtGDelouhghHJlKlDKTm4Erx1sWrTrmIjPyP8nZTqlS2xEgEOC"
    predict_next_3_days("best_aqi_model.pkl", API_KEY)
