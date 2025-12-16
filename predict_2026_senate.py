import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

print("=" * 70)
print("2026 SENATE RACE PREDICTIONS WITH CONFIDENCE SCORES")
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

# 2026 Senate races (actual seats up in 2026)
SENATE_2026 = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'DE', 'GA', 'HI', 'ID', 
    'IL', 'IA', 'KS', 'KY', 'LA', 'ME', 'MA', 'MI', 'MN', 'MS',
    'MT', 'NE', 'NH', 'NJ', 'NM', 'NC', 'OK', 'OR', 'RI', 'SC',
    'SD', 'TN', 'TX', 'VA', 'WV', 'WY'
]

def load_latest_economics():
    """Load Q3 2026 economic indicators"""
    print("\n1. Loading Q3 2026 Economic Data...")
    
    gdp_df = pd.read_csv('real_data/economic/GDPC1.csv')
    unemployment_df = pd.read_csv('real_data/economic/UNRATE.csv')
    inflation_df = pd.read_csv('real_data/economic/CPIAUCSL.csv')
    confidence_df = pd.read_csv('real_data/economic/UMCSENT.csv')
    
    gdp_df.columns = ['DATE', 'gdp']
    unemployment_df.columns = ['DATE', 'unemployment']
    inflation_df.columns = ['DATE', 'inflation']
    confidence_df.columns = ['DATE', 'consumer_confidence']
    
    for df in [gdp_df, unemployment_df, inflation_df, confidence_df]:
        df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Get Q3 2026 data (July-Sept 2026)
    gdp_2026_q3 = gdp_df[(gdp_df['DATE'].dt.year == 2026) & (gdp_df['DATE'].dt.quarter == 3)]
    gdp_2025_q3 = gdp_df[(gdp_df['DATE'].dt.year == 2025) & (gdp_df['DATE'].dt.quarter == 3)]
    
    if len(gdp_2026_q3) > 0 and len(gdp_2025_q3) > 0:
        gdp_growth = ((gdp_2026_q3.iloc[0]['gdp'] - gdp_2025_q3.iloc[0]['gdp']) / 
                      gdp_2025_q3.iloc[0]['gdp'] * 100)
    else:
        print("   ⚠️  Q3 2026 GDP not available yet, using latest")
        gdp_growth = 0
    
    # Unemployment (average of July, Aug, Sept 2026)
    unemp_2026 = unemployment_df[
        (unemployment_df['DATE'].dt.year == 2026) & 
        (unemployment_df['DATE'].dt.month.isin([7, 8, 9]))
    ]
    
    if len(unemp_2026) > 0:
        unemployment = unemp_2026['unemployment'].mean()
    else:
        unemployment = unemployment_df.iloc[-1]['unemployment']
        print(f"   ⚠️  Using latest unemployment: {unemployment:.1f}%")
    
    # Inflation (Sept 2026 vs Sept 2025)
    infl_2026_sept = inflation_df[
        (inflation_df['DATE'].dt.year == 2026) & 
        (inflation_df['DATE'].dt.month == 9)
    ]
    infl_2025_sept = inflation_df[
        (inflation_df['DATE'].dt.year == 2025) & 
        (inflation_df['DATE'].dt.month == 9)
    ]
    
    if len(infl_2026_sept) > 0 and len(infl_2025_sept) > 0:
        inflation = ((infl_2026_sept.iloc[0]['inflation'] - infl_2025_sept.iloc[0]['inflation']) /
                    infl_2025_sept.iloc[0]['inflation'] * 100)
    else:
        inflation = 0
        print("   ⚠️  Sept 2026 inflation not available yet")
    
    # Consumer confidence (Sept 2026)
    conf_2026_sept = confidence_df[
        (confidence_df['DATE'].dt.year == 2026) & 
        (confidence_df['DATE'].dt.month == 9)
    ]
    
    if len(conf_2026_sept) > 0:
        consumer_confidence = conf_2026_sept.iloc[0]['consumer_confidence']
    else:
        consumer_confidence = confidence_df.iloc[-1]['consumer_confidence']
        print(f"   ⚠️  Using latest consumer confidence: {consumer_confidence:.1f}")
    
    print(f"\n   Q3 2026 Economic Conditions:")
    print(f"   GDP Growth: {gdp_growth:.2f}%")
    print(f"   Unemployment: {unemployment:.1f}%")
    print(f"   Inflation: {inflation:.2f}%")
    print(f"   Consumer Confidence: {consumer_confidence:.1f}")
    
    return {
        'gdp_growth': gdp_growth,
        'unemployment': unemployment,
        'inflation': inflation,
        'consumer_confidence': consumer_confidence
    }

def load_2024_results_for_features():
    """Load 2024 election results to create lagged features"""
    print("\n2. Loading 2024 Results for Historical Features...")
    
    elections = pd.read_csv('data/real_election_results.csv')
    
    # Get 2024 results
    results_2024 = elections[elections['election_year'] == 2024]
    
    print(f"   ✅ Loaded 2024 results for {len(results_2024)} states")
    
    return results_2024

def load_2026_polling():
    """Load 2026 Senate polling if available"""
    print("\n3. Checking for 2026 Polling Data...")
    
    # Check if 2026 polling file exists
    import os
    poll_files = []
    poll_dir = 'real_data/polling'
    
    if os.path.exists(poll_dir):
        for f in os.listdir(poll_dir):
            if '2026' in f.lower() and f.endswith('.csv'):
                poll_files.append(f)
    
    if poll_files:
        print(f"   ✅ Found 2026 polling: {poll_files}")
        # You would process these here (similar to historical polls)
        # For now, return None - you'll add this logic when data exists
        return None
    else:
        print("   ⚠️  No 2026 polling data found")
        print("   Will use economics-only predictions")
        return None

def calculate_confidence(econ_pred_margin, poll_margin, consumer_confidence):
    """Calculate betting confidence"""
    
    # No polling data
    if poll_margin is None or pd.isna(poll_margin):
        if consumer_confidence < 75:
            return 'MEDIUM-HIGH', 0.75, f'Strong economic signal (conf: {consumer_confidence:.1f}), no polls'
        elif consumer_confidence < 85:
            return 'MEDIUM', 0.5, f'Moderate economic signal (conf: {consumer_confidence:.1f}), no polls'
        else:
            return 'MEDIUM-LOW', 0.25, f'Weak economic signal (conf: {consumer_confidence:.1f}), no polls'
    
    # Have both economics and polling
    econ_direction = 'R' if econ_pred_margin > 0 else 'D'
    poll_direction = 'R' if poll_margin > 0 else 'D'
    
    if econ_direction == poll_direction:
        margin_diff = abs(abs(econ_pred_margin) - abs(poll_margin))
        
        if margin_diff < 3:
            return 'HIGH', 1.0, f'Strong agreement (econ: {econ_pred_margin:+.1f}%, poll: {poll_margin:+.1f}%)'
        elif margin_diff < 7:
            return 'MEDIUM-HIGH', 0.75, f'Good agreement (econ: {econ_pred_margin:+.1f}%, poll: {poll_margin:+.1f}%)'
        else:
            return 'MEDIUM', 0.5, f'Agree on winner (econ: {econ_pred_margin:+.1f}%, poll: {poll_margin:+.1f}%)'
    else:
        return 'LOW', 0.0, f'DISAGREEMENT - SKIP! (econ: {econ_direction}, poll: {poll_direction})'

def generate_predictions():
    """Generate predictions for all 2026 Senate races"""
    print("\n4. Generating Predictions...")
    
    # Load model and scaler
    model = joblib.load('data/models/best_model_real_data.pkl')
    scaler = joblib.load('data/models/scaler_real_data.pkl')
    
    # Load data
    econ_2026 = load_latest_economics()
    results_2024 = load_2024_results_for_features()
    polling_2026 = load_2026_polling()
    
    # Create predictions for each Senate race
    predictions = []
    
    for state in SENATE_2026:
        # Get 2024 results for this state (for lagged features)
        state_2024 = results_2024[results_2024['state'] == state]
        
        if len(state_2024) == 0:
            print(f"   ⚠️  No 2024 data for {state}, skipping")
            continue
        
        state_2024 = state_2024.iloc[0]
        
        # Calculate historical average (you'd do this properly with full history)
        # For now, just use 2024 margin as proxy
        historical_avg_margin = state_2024['margin']
        
        # Create feature vector
        features = {
            'prev_margin': state_2024['margin'],
            'prev_dem_pct': state_2024['dem_pct'],
            'prev_rep_pct': state_2024['rep_pct'],
            'historical_avg_margin': historical_avg_margin,
            'gdp_growth': econ_2026['gdp_growth'],
            'unemployment': econ_2026['unemployment'],
            'inflation': econ_2026['inflation'],
            'consumer_confidence': econ_2026['consumer_confidence']
        }
        
        # Create feature array in correct order
        feature_cols = ['prev_margin', 'prev_dem_pct', 'prev_rep_pct', 'historical_avg_margin',
                       'gdp_growth', 'unemployment', 'inflation', 'consumer_confidence']
        
        X = np.array([[features[col] for col in feature_cols]])
        
        # Scale and predict
        X_scaled = scaler.transform(X)
        pred_class = model.predict(X_scaled)[0]
        pred_proba = model.predict_proba(X_scaled)[0]
        
        # Calculate predicted margin
        econ_pred_margin = (pred_proba[1] - 0.5) * 100 * 2  # Convert prob to margin
        
        # Get polling margin (None for now, you'll add when data exists)
        poll_margin = None  # Would load from polling_2026 if available
        
        # Calculate confidence
        confidence_level, bet_multiplier, reason = calculate_confidence(
            econ_pred_margin,
            poll_margin,
            econ_2026['consumer_confidence']
        )
        
        predictions.append({
            'state': state,
            'state_full': STATE_MAPPING.get(state, state),
            'prediction': 'Republican' if pred_class == 1 else 'Democrat',
            'pred_class': pred_class,
            'rep_probability': pred_proba[1],
            'dem_probability': pred_proba[0],
            'econ_pred_margin': econ_pred_margin,
            'poll_margin': poll_margin,
            'confidence_level': confidence_level,
            'bet_multiplier': bet_multiplier,
            'reason': reason,
            'should_bet': bet_multiplier >= 0.5
        })
    
    return pd.DataFrame(predictions)

def display_recommendations(predictions_df):
    """Display betting recommendations"""
    print("\n5. BETTING RECOMMENDATIONS")
    print("=" * 70)
    
    # Sort by confidence
    predictions_df = predictions_df.sort_values(['bet_multiplier', 'rep_probability'], ascending=[False, False])
    
    # HIGH Confidence
    high_conf = predictions_df[predictions_df['confidence_level'] == 'HIGH']
    if len(high_conf) > 0:
        print("\n🎯 HIGH CONFIDENCE BETS (1.0x Kelly):")
        for _, row in high_conf.iterrows():
            print(f"\n   {row['state']} - {row['state_full']}")
            print(f"   Prediction: {row['prediction']}")
            print(f"   Probability: {row['rep_probability']:.1%} R, {row['dem_probability']:.1%} D")
            print(f"   Reason: {row['reason']}")
    
    # MEDIUM-HIGH Confidence
    med_high = predictions_df[predictions_df['confidence_level'] == 'MEDIUM-HIGH']
    if len(med_high) > 0:
        print("\n✅ MEDIUM-HIGH CONFIDENCE BETS (0.75x Kelly):")
        for _, row in med_high.iterrows():
            print(f"\n   {row['state']} - {row['state_full']}")
            print(f"   Prediction: {row['prediction']}")
            print(f"   Probability: {row['rep_probability']:.1%} R, {row['dem_probability']:.1%} D")
            print(f"   Reason: {row['reason']}")
    
    # MEDIUM Confidence
    medium = predictions_df[predictions_df['confidence_level'] == 'MEDIUM']
    if len(medium) > 0:
        print("\n👍 MEDIUM CONFIDENCE BETS (0.5x Kelly):")
        for _, row in medium.iterrows():
            print(f"\n   {row['state']} - {row['state_full']}")
            print(f"   Prediction: {row['prediction']}")
            print(f"   Probability: {row['rep_probability']:.1%} R, {row['dem_probability']:.1%} D")
            print(f"   Reason: {row['reason']}")
    
    # SKIP
    skip = predictions_df[predictions_df['bet_multiplier'] < 0.5]
    if len(skip) > 0:
        print("\n⚠️  SKIP (LOW CONFIDENCE):")
        for _, row in skip.iterrows():
            print(f"\n   {row['state']} - {row['state_full']}")
            print(f"   Prediction: {row['prediction']}")
            print(f"   Reason: {row['reason']}")
    
    # Summary
    total_races = len(predictions_df)
    should_bet = predictions_df['should_bet'].sum()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal races: {total_races}")
    print(f"Recommended bets: {should_bet} ({should_bet/total_races:.1%})")
    print(f"Skip: {total_races - should_bet} ({(total_races - should_bet)/total_races:.1%})")
    
    return predictions_df

def save_predictions(predictions_df):
    """Save predictions to CSV for reference"""
    output_file = 'automation/2026_senate_predictions.csv'
    predictions_df.to_csv(output_file, index=False)
    print(f"\n✅ Predictions saved to: {output_file}")
    return output_file

# Run it
if __name__ == "__main__":
    predictions = generate_predictions()
    display_recommendations(predictions)
    save_predictions(predictions)
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\n1. Review predictions above")
    print("2. Open Smarkets or Betfair")
    print("3. Find each race in the predictions")
    print("4. Compare market odds to your predictions")
    print("5. Calculate bet sizes (see below)")
    print("\n" + "=" * 70)