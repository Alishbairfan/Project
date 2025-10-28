import requests
from datetime import datetime
from zoneinfo import ZoneInfo
import os
from pathlib import Path
OPENWEATHER_API_KEY ="8e01acd56128708936dd99cfdcb59500"
WEATHERBIT_API_KEY ="e3876c4116194da1a79e2ed6fed6055c"
LAT = 24.8607
LON = 67.0011
LOCAL_TZ = "Asia/Karachi"


def fetch_openweather_pollutants(lat=LAT, lon=LON, api_key=OPENWEATHER_API_KEY):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(url)
    response.raise_for_status()

    data = response.json()['list'][0]
    components = data['components']

    dt_utc = datetime.utcfromtimestamp(data['dt'])
    dt_local = dt_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo(LOCAL_TZ))

    timestamp = f"{dt_local.day}/{dt_local.month}/{dt_local.year} {dt_local.hour}:{dt_local.minute:02d}"

    pollutants = {
        'timestamp': timestamp,
        'pm25': components.get('pm2_5'),
        'pm10': components.get('pm10'),
        'no2': components.get('no2'),
        'o3': components.get('o3'),
    }
    return pollutants


def fetch_weatherbit_weather(lat=LAT, lon=LON, api_key=WEATHERBIT_API_KEY):
   
    url_weather = f"https://api.weatherbit.io/v2.0/current?lat={lat}&lon={lon}&key={api_key}"
    response_weather = requests.get(url_weather)
    response_weather.raise_for_status()
    data_weather = response_weather.json()['data'][0]

    
    url_aqi = f"https://api.weatherbit.io/v2.0/current/airquality?lat={lat}&lon={lon}&key={api_key}"
    response_aqi = requests.get(url_aqi)
    response_aqi.raise_for_status()
    data_aqi = response_aqi.json()['data'][0]

    dt_utc = datetime.utcfromtimestamp(data_weather['ts'])
    dt_local = dt_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo(LOCAL_TZ))
    timestamp = f"{dt_local.day}/{dt_local.month}/{dt_local.year} {dt_local.hour}:{dt_local.minute:02d}"

    weather = {
        'timestamp': timestamp,
        'temperature': data_weather.get('temp'),
        'humidity': data_weather.get('rh'),
        'wind_speed': data_weather.get('wind_spd'),
        'hour': dt_local.hour,
        'day': dt_local.day,
        'month': dt_local.month,
        'weekday': dt_local.weekday(),
        'is_weekend': 1 if dt_local.weekday() >= 5 else 0,
        'aqi': data_aqi.get('aqi'),
        'weather_description': data_weather['weather']['description']  
    }
    return weather


def fetch_current_features():
    pollutants = fetch_openweather_pollutants()
    weather = fetch_weatherbit_weather()

    if pollutants['timestamp'] != weather['timestamp']:
        print(f"Warning: timestamps differ! Aligning Weatherbit to OpenWeather timestamp.")
        weather['timestamp'] = pollutants['timestamp']

    features = {**pollutants, **weather}
    return features

if __name__ == "__main__":
    current_features = fetch_current_features()
    print(current_features)
    df =pd.DataFrame([current_features])
    df.to_csv("realtime_features.csv", index=False)
    print("data saved in realtime_features.csv")
        
