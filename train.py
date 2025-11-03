import pandas as pd
from datetime import datetime, timedelta
import joblib
from xgboost import XGBRegressor
import hopsworks

def fetch_last_24_hours(use_hopsworks=True, csv_path="computed_features.csv",
                        fg_name="karachi_realtime_features", fg_version=1,
                        api_key="37GNR2LvcIsDqZd1.NHO665gQEMZGDRaZYCKszx7nenA0CJqLvLdOSMBvbpK2YzVJ7TD7hqw9HxCjRRk6"):
    if use_hopsworks:
        project = hopsworks.login(api_key_value=api_key)
        fs = project.get_feature_store()
        fg = fs.get_feature_group(fg_name, version=fg_version)
        df = fg.read()
    else:
        if not pd.io.common.file_exists(csv_path):
            return pd.DataFrame()
        df = pd.read_csv(csv_path)

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce" ,utc=True)
    df = df.dropna(subset=["timestamp"])  
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp").resample("H").ffill().reset_index()         
    latest_time = df["timestamp"].max()
    last_24h = latest_time - pd.Timedelta(hours=24)
    df_24h = df[df["timestamp"] > last_24h]

    return df_24h

def train_daily_model():
    df_train = fetch_last_24_hours()
    if df_train.empty:
        print("No data available for last 24 hours. Skipping training.")
        return None

    df_train["aqi_next_hour"] = df_train["aqi"].shift(-1)
    df_train = df_train.dropna(subset=["aqi_next_hour"])

    X_train = df_train.drop(columns=["timestamp", "aqi", "aqi_next_hour"])
    y_train = df_train["aqi_next_hour"]

    project = hopsworks.login(api_key_value="37GNR2LvcIsDqZd1.NHO665gQEMZGDRaZYCKszx7nenA0CJqLvLdOSMBvbpK2YzVJ7TD7hqw9HxCjRRk6")
    mr = project.get_model_registry()
    model_name = "aqi_predictor"

    try:
        model_meta = mr.get_model(model_name)
        model_dir = model_meta.download()
        model = joblib.load(model_dir + "/best_aqi_model.pkl")
        print(f"Loaded existing model version {model_meta.version}")
    except:
        model = XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        print("Created new model.")

    
    model.fit(X_train, y_train, xgb_model=None if 'model_meta' not in locals() else model.get_booster())

    joblib.dump(model, "best_aqi_model.pkl")
    model_meta = mr.python.create_model(
        name=model_name,
        description="AQI next-hour prediction model"
    )
    model_meta.save("best_aqi_model.pkl")
    print(f"Model trained and registered in Hopsworks Registry as version {model_meta.version}")

    return model
if __name__ == "__main__":
    train_daily_model()
