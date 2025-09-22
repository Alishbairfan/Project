import requests
import pandas as pd
from datetime import datetime
import os

API_KEY = os.getenv("OPENWEATHER_API_KEY")  
CITY = "Karachi"
URL = f"http://api.openweathermap.org/data/2.5/air_pollution?lat=24.8607&lon=67.0011&appid={API_KEY}"

def fetch_aqi():
    response = requests.get(URL)
    if response.status_code == 200:
        data = response.json()
        # Extract AQI and timestamp
        aqi = data['list'][0]['main']['aqi']
        timestamp = datetime.utcfromtimestamp(data['list'][0]['dt']).strftime('%Y-%m-%d %H:%M:%S')
        
      
        df = pd.DataFrame([{
            "timestamp": timestamp,
            "city": CITY,
            "aqi": aqi
        }])
        
        file_name = "aqi_data.csv"
        
       
        if os.path.exists(file_name):
            df.to_csv(file_name, mode='a', header=False, index=False)
        else:
            df.to_csv(file_name, index=False)
            
        print(f"[✅] Data saved for {timestamp}")
    else:
        print(f"[❌] Failed to fetch data: {response.status_code} - {response.text}")

if __name__ == "__main__":
    fetch_aqi()
