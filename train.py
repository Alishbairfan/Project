import os
import pandas as pd
import joblib
from xgboost import XGBRegressor

def train_or_update_xgb(model_path="best_aqi_model.pkl", csv_path="computed_features.csv", ts_path="last_timestamp.txt"):
   

    df = pd.read_csv(csv_path)

  
    if "timestamp" not in df.columns:
        raise ValueError("'timestamp' column not found in CSV file")

    
    if os.path.exists(ts_path):
        with open(ts_path, "r") as f:
            last_ts = f.read().strip()
        df_new = df[df["timestamp"] > last_ts]
    else:
        df_new = df  # First run: use all data

    if df_new.empty:
        print(" No new data found. Skipping training update.")
        return None

    
    df_new = df_new.copy()
    df_new["aqi_next_hour"] = df_new["aqi"].shift(-1)
    X_new = df_new.drop(columns=["timestamp", "aqi", "aqi_next_hour"]).iloc[:-1]
    y_new = df_new["aqi_next_hour"].iloc[:-1]

    
    if os.path.exists(model_path):
        print(" Loading existing XGBoost model...")
        model = joblib.load(model_path)
    else:
        print("Training new XGBoost model...")
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

   
    model.fit(X_new, y_new, xgb_model=None if not os.path.exists(model_path) else model.get_booster())
    print("Model updated with latest data.")

   
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    
    new_last_ts = str(df_new["timestamp"].max())
    with open(ts_path, "w") as f:
        f.write(new_last_ts)
    print(f" Updated last timestamp to: {new_last_ts}")

    return model

if __name__ == "__main__":
    train_or_update_xgb()
