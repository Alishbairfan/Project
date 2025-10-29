import streamlit as st
import pandas as pd
from predict import predict_next_3_days  

st.set_page_config(page_title="AQI Prediction Dashboard", layout="centered")
st.title("Karachi AQI Prediction Dashboard")
def aqi_alert(aqi):
    if aqi <= 50:
        return "Good", "green"
    elif aqi <= 100:
        return "Moderate", "yellow"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "orange"
    elif aqi <= 200:
        return "Unhealthy", "red"
    elif aqi <= 300:
        return "Very Unhealthy", "purple"
    else:
        return "Hazardous", "maroon"

st.sidebar.header("Settings")
model_path = st.sidebar.text_input("Model Path", "best_aqi_model.pkl")
csv_path = st.sidebar.text_input("CSV Path", "computed_features.csv")
output_csv = st.sidebar.text_input("Output CSV Path", "predicted_aqi_predictions.csv")

if st.sidebar.button("Predict AQI"):
   
    result_df = predict_next_3_days(model_path, csv_path, output_csv)

    next_hour_df = result_df[result_df["type"] == "next_hour"]
    daily_df = result_df[result_df["type"] == "daily_aggregate"]

    st.subheader("Next Hour AQI")
    next_hour_value = next_hour_df["predicted_aqi"].values[0]
    next_hour_time = next_hour_df["timestamp"].values[0]
    status, color = aqi_alert(next_hour_value)
    st.metric(label=f"Predicted AQI at {next_hour_time}", value=next_hour_value)

   
    st.subheader("Next 3 Days AQI (Daily Average)")

    daily_display = daily_df[["timestamp", "predicted_aqi"]].rename(columns={"timestamp": "Date", "predicted_aqi": "AQI"})
    daily_display["Alert"] = daily_display["AQI"].apply(lambda x: aqi_alert(x)[0])
    st.dataframe(daily_display)
    st.line_chart(daily_df.set_index("timestamp")["predicted_aqi"])
