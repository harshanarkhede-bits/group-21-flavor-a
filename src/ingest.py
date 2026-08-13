import pandas as pd
import os

def ingest_and_validate(file_path):
    print(f"Ingesting data from {file_path}...")
    
    # We will sample 100,000 rows to keep training fast for the mini-project. 
    # You can remove `nrows=100000` later if your PC can handle all 1.45 million rows!
    df = pd.read_csv(file_path, nrows=100000)
    
    initial_shape = df.shape
    
    # Validation 1: Drop missing values
    df = df.dropna()
    
    # Validation 2: Ensure correct datetime format
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    
    # Validation 3: Filter out extreme outliers (e.g., trips > 10 hours or < 1 minute)
    # The target variable 'trip_duration' is in seconds
    df = df[(df['trip_duration'] > 60) & (df['trip_duration'] < 36000)]
    
    # Validation 4: Remove trips with 0 passengers
    df = df[df['passenger_count'] > 0]
    
    print(f"Dropped {initial_shape[0] - df.shape[0]} invalid rows.")
    
    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/validated_taxi.csv"
    df.to_csv(out_path, index=False)
    print(f"Validation complete. Saved to {out_path}")

if __name__ == "__main__":
    ingest_and_validate("data/raw/train.csv")