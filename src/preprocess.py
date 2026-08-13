import pandas as pd
import numpy as np

def calculate_distance(df):
    """Calculates the Haversine distance in kilometers from GPS coordinates."""
    R = 6371.0 # Earth radius in kilometers
    
    lat1, lon1 = np.radians(df['pickup_latitude']), np.radians(df['pickup_longitude'])
    lat2, lon2 = np.radians(df['dropoff_latitude']), np.radians(df['dropoff_longitude'])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def engineer_features(file_path):
    print(f"Engineering features for {file_path}...")
    df = pd.read_csv(file_path, parse_dates=['pickup_datetime'])
    
    # 1. Location Feature: Distance
    df['distance_km'] = calculate_distance(df)
    
    # 2. Time Features
    df['hour_of_day'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 3. Drop columns we don't need for ML training
    # dropoff_datetime is a data leak (we won't know it when predicting ETA!)
    cols_to_drop = ['id', 'vendor_id', 'pickup_datetime', 'dropoff_datetime', 
                    'store_and_fwd_flag', 'pickup_longitude', 'pickup_latitude', 
                    'dropoff_longitude', 'dropoff_latitude']
    
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    out_path = "data/processed/engineered_taxi.csv"
    df.to_csv(out_path, index=False)
    print(f"Feature engineering complete. Saved to {out_path}")

if __name__ == "__main__":
    engineer_features("data/processed/validated_taxi.csv")