import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create sample data
def create_sample_data():
    # Create three stations
    stations = [
        {'id': 'Station1', 'lat': 34.05, 'lon': -118.25, 'city': 'Los Angeles'},
        {'id': 'Station2', 'lat': 34.10, 'lon': -118.30, 'city': 'Los Angeles'},
        {'id': 'Station3', 'lat': 35.20, 'lon': -120.70, 'city': 'San Luis Obispo'}
    ]
    
    # Create time series (7 days hourly)
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(24*7)]
    
    # Create empty dataframe
    data = []
    
    # Add data for each station
    for station in stations:
        base_value = 20 + np.random.normal(0, 5)  # Base PM2.5 value
        for date in dates:
            # Create some daily pattern
            hour_factor = 1 + 0.5 * np.sin(date.hour / 24 * 2 * np.pi)
            # Add some noise
            noise = np.random.normal(0, 3)
            pm25 = base_value * hour_factor + noise
            
            data.append({
                'station_loc': station['id'],
                'latitude': station['lat'],
                'longitude': station['lon'],
                'city': station['city'],
                'datetime': date,
                'PM2.5 (ug/m3)': max(0, pm25)  # Ensure non-negative values
            })
    
    # Create dataframe
    df = pd.DataFrame(data)
    df.to_csv('/home/naradalinux/dev/GCNTFT/outputs/tests/sample_air_quality.csv', index=False)
    print(f"Created sample data with {len(stations)} stations and {len(dates)} time points")
    
if __name__ == "__main__":
    create_sample_data()