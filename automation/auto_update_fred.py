import requests
import pandas as pd
from datetime import datetime
import os

# YOUR API KEY HERE (get from https://fred.stlouisfed.org/docs/api/api_key.html)
FRED_API_KEY = '35611f0651c9fc40cca2771bede3e6b3'  # ← CHANGE THIS!

print("=" * 70)
print("AUTOMATED FRED DATA UPDATE")
print("=" * 70)

# FRED series codes
SERIES = {
    'GDPC1': 'GDP',
    'UNRATE': 'Unemployment',
    'CPIAUCSL': 'Inflation',
    'UMCSENT': 'Consumer Confidence'
}

def download_fred_series(series_id, api_key):
    """Download data from FRED API"""
    url = f'https://api.stlouisfed.org/fred/series/observations'
    
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'sort_order': 'asc'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        observations = data['observations']
        
        # Convert to DataFrame
        df = pd.DataFrame(observations)
        df = df[['date', 'value']]
        df.columns = ['DATE', 'VALUE']
        
        # Remove non-numeric values
        df = df[df['VALUE'] != '.']
        df['VALUE'] = pd.to_numeric(df['VALUE'])
        
        return df
    
    except Exception as e:
        print(f"   ❌ Error downloading {series_id}: {e}")
        return None

def update_all_data():
    """Download and save all FRED series"""
    
    print(f"\nStarting update at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if FRED_API_KEY == 'YOUR_API_KEY_HERE':
        print("❌ ERROR: You need to add your FRED API key!")
        print()
        print("1. Get free key: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("2. Open automation/auto_update_fred.py")
        print("3. Replace 'YOUR_API_KEY_HERE' with your actual key")
        print()
        return False
    
    # Create directory if it doesn't exist
    output_dir = 'real_data/economic'
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    
    for series_id, series_name in SERIES.items():
        print(f"Downloading {series_name} ({series_id})...")
        
        df = download_fred_series(series_id, FRED_API_KEY)
        
        if df is not None:
            # Save to CSV
            output_file = os.path.join(output_dir, f'{series_id}.csv')
            df.to_csv(output_file, index=False)
            
            # Show latest value
            latest_date = df.iloc[-1]['DATE']
            latest_value = df.iloc[-1]['VALUE']
            
            print(f"   ✅ Downloaded {len(df)} records")
            print(f"   Latest: {latest_date} = {latest_value:.2f}")
            print()
            
            success_count += 1
        else:
            print()
    
    print("=" * 70)
    print(f"Update complete: {success_count}/{len(SERIES)} series updated")
    print("=" * 70)
    
    return success_count == len(SERIES)

def retrain_models():
    """Automatically retrain models with new data"""
    print("\nRetraining models with updated data...")
    
    import subprocess
    
    try:
        result = subprocess.run(
            ['python', 'train_with_economics.py'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Models retrained successfully")
            return True
        else:
            print("⚠️  Model retraining had issues:")
            print(result.stderr)
            return False
    
    except Exception as e:
        print(f"❌ Error retraining models: {e}")
        return False

if __name__ == "__main__":
    # Update data
    if update_all_data():
        print("\n" + "=" * 70)
        print("DATA UPDATE SUCCESSFUL")
        print("=" * 70)
        
        # Ask if should retrain
        print("\nRetrain models with new data? (y/n): ", end='')
        response = input().lower()
        
        if response == 'y':
            retrain_models()
        
        print("\n✅ All done!")
    else:
        print("\n❌ Update failed - check errors above")