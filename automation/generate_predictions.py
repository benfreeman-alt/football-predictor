import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

print("=" * 70)
print("2026 PREDICTION GENERATOR")
print("=" * 70)

# State mapping
STATE_MAPPING = {
    'AK': 'Alaska', 'AL': 'Alabama', 'AR': 'Arkansas', 'AZ': 'Arizona',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'IA': 'Iowa',
    'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'MA': 'Massachusetts', 'MD': 'Maryland',
    'ME': 'Maine', 'MI': 'Michigan', 'MN': 'Minnesota', 'MO': 'Missouri',
    'MS': 'Mississippi', 'MT': 'Montana', 'NC': 'North Carolina', 'ND': 'North Dakota',
    'NE': 'Nebraska', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico',
    'NV': 'Nevada', 'NY': 'New York', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VA': 'Virginia', 'VT': 'Vermont', 'WA': 'Washington', 'WI': 'Wisconsin',
    'WV': 'West Virginia', 'WY': 'Wyoming', 'DC': 'District of Columbia'
}

def load_latest_economic_data():
    """Load the most recent economic indicators"""
    print("\n1. Loading latest economic data...")
    
    gdp_df = pd.read_csv('real_data/economic/GDPC1.csv')
    unemployment_df = pd.read_csv('real_data/economic/UNRATE.csv')
    inflation_df = pd.read_csv('real_data/economic/CPIAUCSL.csv')
    confidence_df = pd.read_csv('real_data/economic/UMCSENT.csv')
    
    # Rename columns
    gdp_df.columns = ['DATE', 'gdp']
    unemployment_df.columns = ['DATE', 'unemployment']
    inflation_df.columns = ['DATE', 'inflation']
    confidence_df.columns = ['DATE', 'consumer_confidence']
    
    # Convert dates
    for df in [gdp_df, unemployment_df, inflation_df, confidence_df]:
        df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Get latest values
    latest_gdp = gdp_df.iloc[-1]['gdp']
    latest_unemployment = unemployment_df.iloc[-1]['unemployment']
    latest_inflation = inflation_df.iloc[-1]['inflation']
    latest_confidence = confidence_df.iloc[-1]['consumer_confidence']
    
    latest_date = confidence_df.iloc[-1]['DATE']
    
    print(f"   Data as of: {latest_date.strftime('%Y-%m-%d')}")
    print(f"   GDP: {latest_gdp:.2f}")
    print(f"   Unemployment: {latest_unemployment:.1f}%")
    print(f"   CPI: {latest_inflation:.2f}")
    print(f"   Consumer Confidence: {latest_confidence:.1f}")
    
    # Calculate metrics for 2026
    # This would need proper Q3 2026 data when available
    econ_2026 = {
        'gdp_growth': 0,  # Placeholder - calculate from actual data
        'unemployment': latest_unemployment,
        'inflation': 0,  # Placeholder - calculate YoY
        'consumer_confidence': latest_confidence
    }
    
    print("\n   ⚠️  Using latest available data as placeholder")
    print("   Update with Q3 2026 data when available")
    
    return econ_2026

def load_latest_polling_data():
    """Load 2026 polling data if available"""
    print("\n2. Checking for 2026 polling data...")
    
    poll_files = []
    poll_dir = 'real_data/polling'
    
    if os.path.exists(poll_dir):
        for f in os.listdir(poll_dir):
            if '2026' in f and f.endswith('.csv'):
                poll_files.append(f)
    
    if poll_files:
        print(f"   ✅ Found {len(poll_files)} 2026 polling files")
        # Load and process polls (similar to existing code)
        return None  # Placeholder
    else:
        print("   ⚠️  No 2026 polling data found yet")
        print("   Will use economics-only predictions")
        return None

def load_historical_results():
    """Load historical election results for feature engineering"""
    print("\n3. Loading historical results...")
    
    elections = pd.read_csv('data/real_election_results.csv')
    latest_year = elections['election_year'].max()
    
    print(f"   Latest results: {latest_year}")
    print(f"   Total records: {len(elections)}")
    
    return elections

def generate_predictions_for_2026():
    """Generate predictions for 2026 Senate races"""
    print("\n4. Generating 2026 predictions...")
    
    # Load model
    model = joblib.load('data/models/best_model_real_data.pkl')
    scaler = joblib.load('data/models/scaler_real_data.pkl')
    
    print("   ✅ Model loaded")
    
    # 2026 Senate races (33-34 seats typically)
    senate_2026_states = [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM',
        'NC', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
        'VA', 'WA', 'WV', 'WY'
    ]
    
    print(f"   Predicting {len(senate_2026_states)} Senate races")
    
    # For now, create placeholder predictions
    print("\n   ⚠️  Full automation requires:")
    print("   1. Complete 2024 election results")
    print("   2. Q3 2026 economic data")
    print("   3. 2026 polling data (optional)")
    print("\n   This is a template - fill in when data available")
    
    return senate_2026_states

def calculate_confidence_scores():
    """Calculate confidence for each prediction"""
    print("\n5. Calculating confidence scores...")
    
    # Placeholder - would use actual data
    print("   Will compare economics vs polling (when available)")
    
    return None

def generate_betting_recommendations():
    """Create actionable betting recommendations"""
    print("\n6. Generating betting recommendations...")
    
    recommendations = """
======================================================================
BETTING RECOMMENDATIONS - 2026 SENATE
======================================================================

Generated: {date}

HIGH CONFIDENCE BETS (1.0x Kelly):
[ ] State: Prediction, Market Price, Edge, Bet Size
[ ] (None yet - need 2026 data)

MEDIUM CONFIDENCE BETS (0.5x Kelly):
[ ] State: Prediction, Market Price, Edge, Bet Size
[ ] (None yet - need 2026 data)

SKIP (LOW CONFIDENCE):
[ ] State: Reason for skipping
[ ] (None yet - need 2026 data)

======================================================================
NOTES:
- Update this after each data refresh
- Only bet when markets are open
- Track all predictions vs actual results

======================================================================
""".format(date=datetime.now().strftime('%Y-%m-%d'))
    
    output_file = 'automation/betting_recommendations_2026.txt'
    with open(output_file, 'w') as f:
        f.write(recommendations)
    
    print(f"   ✅ Recommendations saved to: {output_file}")
    print(recommendations)

# Run pipeline
if __name__ == "__main__":
    econ_data = load_latest_economic_data()
    poll_data = load_latest_polling_data()
    historical = load_historical_results()
    predictions = generate_predictions_for_2026()
    confidence = calculate_confidence_scores()
    generate_betting_recommendations()
    
    print("\n" + "=" * 70)
    print("AUTOMATION STATUS")
    print("=" * 70)
    print("\n✅ Phase 1 Complete: Data pipeline framework")
    print("⏳ Phase 2 Pending: Need 2026 data")
    print("⏳ Phase 3 Pending: Live market integration")
    
    print("\n" + "=" * 70)