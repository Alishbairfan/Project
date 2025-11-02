import os
import pandas as pd
from datetime import datetime
from fetch_data import fetch_current_features
import hopsworks

def get_last_record_from_hopsworks():
    project = hopsworks.login(api_key_value="xo0aJonG6geNXT6N.Y1Xs6QMqrk1qExXtGDelouhghHJlKlDKTm4Erx1sWrTrmIjPyP8nZTqlS2xEgEOC")
    fs = project.get_feature_store()
    fg = fs.get_feature_group("karachi_realtime_features", version=1)
    df = fg.read()

    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    return df.iloc[-1]

def compute_features(df):
    prev_record = get_last_record_from_hopsworks()

    if prev_record is not None:
        df['aqi_change_rate'] = df['aqi'] - prev_record['aqi']
        df['temp_change'] = df['temperature'] - prev_record['temperature']
        df['humidity_change'] = df['humidity'] - prev_record['humidity']
    else:
        df['aqi_change_rate'] = 0
        df['temp_change'] = 0
        df['humidity_change'] = 0

    df['pm25_to_pm10_ratio'] = df['pm25'] / df['pm10']

    weather_map = {
        'Broken Clouds': 0,
        'Clear Sky': 1,
        'Drizzle': 2,
        'Few Clouds': 3,
        'Fog': 4,
        'Haze': 5,
        'Light Rain': 6,
        'Moderate Rain': 7,
        'Overcast Clouds': 8,
        'Scattered Clouds': 9
    }

    if 'weather_description' in df.columns:
        df['weather_description'] = df['weather_description'].str.title()
        df['weather_code'] = df['weather_description'].map(weather_map).fillna(-1).astype(int)
        df = df.drop(columns=['weather_description'])

    return df


def prepare_features(current_features):
    df = pd.DataFrame([current_features])
    df = compute_features(df)
    return df

def save_to_csv(df, path="computed_features.csv"):
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if os.path.exists(path):
        existing = pd.read_csv(path)
        existing["timestamp"] = pd.to_datetime(existing["timestamp"])

        new_time = df["timestamp"].iloc[0]
        same_hour = existing[existing["timestamp"].dt.floor("h") == new_time.floor("h")]

        if same_hour.empty:
            df.to_csv(path, mode="a", header=False, index=False)
            print(f"Added new hourly data for {new_time}.")
        else:
            print(f"Data for {new_time} already exists in CSV, skipping append.")
    else:
        df.to_csv(path, index=False)
        print("Created new CSV and added first record.")


def store_in_hopsworks(df):
    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value="xo0aJonG6geNXT6N.Y1Xs6QMqrk1qExXtGDelouhghHJlKlDKTm4Erx1sWrTrmIjPyP8nZTqlS2xEgEOC")
    fs = project.get_feature_store()

    desired_order = [
        "timestamp", "aqi", "pm25", "pm10", "no2", "o3",
        "temperature", "humidity", "wind_speed", "hour", "day",
        "month", "weekday", "is_weekend", "aqi_change_rate",
        "pm25_to_pm10_ratio", "temp_change", "humidity_change", "weather_code"
    ]
    df = df[desired_order]
    df = df.astype({
        "temperature": "float64",
        "wind_speed": "float64",
        "temp_change": "float64",
        "aqi_change_rate": "float64",
        "pm25_to_pm10_ratio": "float64",
        "humidity_change": "float64",
        "weather_code": "int32",
        "aqi": "float64",
        "pm25": "float64",
        "pm10": "float64",
        "no2": "float64",
        "o3": "float64",
        "humidity": "float64",
        "hour": "int32",
        "day": "int32",
        "month": "int32",
        "weekday": "int32",
        "is_weekend": "int32"
    })
    df["timestamp"] = df["timestamp"].astype(str)
    print("\n[INFO] DataFrame dtypes before insert:")
    print(df.dtypes)
    fg = fs.get_or_create_feature_group(
        name="karachi_realtime_features",
        version=1,
        primary_key=["timestamp"],
        description="Karachi aqi computed features",
        online_enabled=True
    )
    
    existing_df = fg.read()
    existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"])

    new_time = df["timestamp"].iloc[0]
    if new_time not in existing_df["timestamp"].values:
        fg.insert(df)
        print(f"Inserted new hourly record for {new_time} into Hopsworks.")
    else:
        print(f"Record for {new_time} already exists in Hopsworks, skipping insert.")

if __name__ == "__main__":
    current_features = fetch_current_features()
    df_features = prepare_features(current_features)

    if "timestamp" not in df_features.columns:
        df_features["timestamp"] = datetime.now()

    desired_order = [
        "timestamp", "aqi", "pm25", "pm10", "no2", "o3",
        "temperature", "humidity", "wind_speed", "hour", "day",
        "month", "weekday", "is_weekend", "aqi_change_rate",
        "pm25_to_pm10_ratio", "temp_change", "humidity_change", "weather_code"
    ]
    df_features = df_features[desired_order]

    print(df_features)

    save_to_csv(df_features)
    store_in_hopsworks(df_features)
