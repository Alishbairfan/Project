import streamlit as st
import pandas as pd

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
csv_path = st.sidebar.text_input("Predicted AQI CSV Path", "predicted_aqi_predictions.csv")

if st.sidebar.button("Load Predictions"):
    try:
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        st.error(f"Could not load CSV: {e}")
        st.stop()

    next_hour = df.iloc[0]
    next_hour_value = next_hour['predicted_aqi']
    next_hour_time = next_hour['timestamp']
    
    next_hour_time_str = pd.to_datetime(next_hour_time).strftime("%Y-%m-%d %H:%M")
 

    status, color = aqi_alert(next_hour_value)
    st.subheader("Next Hour AQI")
    st.markdown(
        f"<h2 style='color:{color}'>Predicted AQI at {next_hour_time_str}: {next_hour_value} ({status})</h2>",
        unsafe_allow_html=True
)

    st.subheader("Next 3 Days AQI (Daily Average)")
    df['date'] = df['timestamp'].dt.date
    daily_avg = df.groupby('date')['predicted_aqi'].mean().reset_index()
    daily_avg['Alert'] = daily_avg['predicted_aqi'].apply(lambda x: aqi_alert(x)[0])
    st.dataframe(daily_avg.rename(columns={'date': 'Date', 'predicted_aqi': 'AQI'}))

    st.subheader("Next 72 Hours AQI Prediction")
    st.line_chart(df.set_index('timestamp')['predicted_aqi'])
