import pandas as pd
import joblib

def predict_next_3_days(model_path="best_aqi_model.pkl", csv_path="computed_features.csv", output_csv="predicted_aqi_predictions.csv"):

    model = joblib.load(model_path)
    df = pd.read_csv(csv_path)
    df = df.replace([float('inf'), float('-inf')], None).dropna()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    recent_data = df.tail(24).copy()
    last_timestamp = df["timestamp"].iloc[-1] if "timestamp" in df.columns and pd.notnull(df["timestamp"].iloc[-1]) else pd.Timestamp.now()

    hourly_preds = []

    for hour in range(1, 73):
        X_input = recent_data.drop(columns=["aqi"]).iloc[-1:].select_dtypes(include=["number", "bool"])
        next_pred = model.predict(X_input)[0]

        hourly_preds.append({
            "timestamp": last_timestamp + pd.Timedelta(hours=hour),
            "predicted_aqi": next_pred
        })
        
        new_row = recent_data.iloc[-1:].copy()
        new_row["aqi"] = next_pred
        recent_data = pd.concat([recent_data, new_row]).tail(24)

    hourly_df = pd.DataFrame(hourly_preds)

    next_hour = hourly_df.iloc[0]
    print("Next hour prediction:", round(next_hour["predicted_aqi"], 2))

    combined_preds = []

    combined_preds.append({
        "timestamp": next_hour["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
        "type": "next_hour",
        "predicted_aqi": round(next_hour["predicted_aqi"], 2)
    })

    for day in range(1, 4):
        day_data = hourly_df.iloc[(day-1)*24:day*24]
        combined_preds.append({
            "timestamp": (last_timestamp + pd.Timedelta(days=day)).strftime("%Y-%m-%d"),
            "type": "daily_aggregate",
            "predicted_aqi": round(day_data["predicted_aqi"].mean(), 2)
        })

    result_df = pd.DataFrame(combined_preds)
    result_df.to_csv(output_csv, index=False)
    print(f"Saved next-hour and 3-day predictions to {output_csv}")

    return result_df

if __name__ == "__main__":
    result = predict_next_3_days()
    print(result)
