import joblib
import hopsworks
from xgboost import XGBRegressor
model = joblib.load("best_aqi_model.pkl")


project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()
fg = fs.get_feature_group("karachi_realtime_features", version=1)

# Fetch latest rows (filter by timestamp if needed)
df_new = fg.read()  # or filter df_new[df_new['timestamp'] > last_timestamp]
df_new['aqi_next_hour'] = df_new['aqi'].shift(-1)
X_new = df_new.drop(columns=['timestamp', 'aqi', 'aqi_next_hour'])
y_new = df_new['aqi_next_hour']

X_new = X_new.iloc[:-1]
y_new = y_new.iloc[:-1]


# Add 50 new trees (adjust n_estimators if needed)
model.fit(X_new, y_new, xgb_model=model)  
joblib.dump(model, "best_aqi_model.pkl")
print("Trained model updated with real-time data, historical training not repeated")
