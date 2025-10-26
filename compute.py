import pandas as pd
import os
from datetime import datetime
import hopsworks

API_KEY = os.getenv("HOPSWORKS_API_KEY")  
project = hopsworks.login(api_key_value=API_KEY)
fs = project.get_feature_store()

def compute_features(df):
  
    try:
        prev_features = pd.read_csv("prev_features.csv").iloc[-1]
    except FileNotFoundError:
        prev_features = None

    if prev_features is not None:
        df['aqi_change_rate'] = df['aqi'] - prev_features['aqi']
        df['temp_change'] = df['temperature'] - prev_features['temperature']
        df['humidity_change'] = df['humidity'] - prev_features['humidity']
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

def prepare_features(current_features: dict) -> pd.DataFrame:
  
    df = pd.DataFrame([current_features])
    df = compute_features(df)
    return df

if __name__ == "__main__":
  
    df_features = prepare_features(current_features)

    desired_order = [
        "timestamp", "aqi","pm25", "pm10","no2","o3","temperature","humidity","wind_speed",
        "hour", "day", "month", "weekday", "is_weekend",
        "aqi_change_rate","pm25_to_pm10_ratio","temp_change","humidity_change",
        "weather_code"
    ]
    df_features = df_features[desired_order]

    
    try:
        fg = fs.get_feature_group(name="karachi_realtime_features", version=1)
        fg.insert(df_features, write_options={"wait_for_job": True})
        print("Inserted features into existing Hopsworks Feature Group.")
    except hopsworks.errors.RestAPIError:
      
        fg = fs.create_feature_group(
            name="karachi_realtime_features",
            version=1,
            description="Processed AQI and weather of realtime features",
            primary_key=["timestamp"],
            event_time="timestamp",
            df=df_features
        )
        fg.insert(df_features, write_options={"wait_for_job": True})
        print("Created and inserted features into new Hopsworks Feature Group.")

    df_features.to_csv("prev_features.csv", index=False)
    print("Features saved locally as prev_features.csv")
