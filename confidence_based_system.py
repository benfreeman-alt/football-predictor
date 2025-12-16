import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

print("=" * 70)
print("CONFIDENCE-BASED BETTING SYSTEM")
print("Economics for Direction, Polling for Confidence")
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

# Load all data
print("\n1. Loading data...")
elections = pd.read_csv('data/real_election_results.csv')
elections['state_full'] = elections['state'].map(STATE_MAPPING)

# Load economic data
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
    df['year'] = df['DATE'].dt.year
    df['quarter'] = df['DATE'].dt.quarter

def get_q3_economics(year):
    gdp_q3 = gdp_df[(gdp_df['year'] == year) & (gdp_df['quarter'] == 3)]
    gdp_val = gdp_q3['gdp'].values[0] if len(gdp_q3) > 0 else None
    
    gdp_prev = gdp_df[(gdp_df['year'] == year-1) & (gdp_df['quarter'] == 3)]
    gdp_prev_val = gdp_prev['gdp'].values[0] if len(gdp_prev) > 0 else None
    gdp_growth = ((gdp_val - gdp_prev_val) / gdp_prev_val * 100) if (gdp_val and gdp_prev_val) else 0
    
    unemp_q3 = unemployment_df[(unemployment_df['year'] == year) & 
                               (unemployment_df['DATE'].dt.month.isin([7,8,9]))]
    unemployment = unemp_q3['unemployment'].mean() if len(unemp_q3) > 0 else None
    
    infl_sept = inflation_df[(inflation_df['year'] == year) & 
                             (inflation_df['DATE'].dt.month == 9)]
    infl_prev = inflation_df[(inflation_df['year'] == year-1) & 
                             (inflation_df['DATE'].dt.month == 9)]
    
    if len(infl_sept) > 0 and len(infl_prev) > 0:
        inflation = ((infl_sept['inflation'].values[0] - infl_prev['inflation'].values[0]) / 
                    infl_prev['inflation'].values[0] * 100)
    else:
        inflation = None
    
    conf_sept = confidence_df[(confidence_df['year'] == year) & 
                              (confidence_df['DATE'].dt.month == 9)]
    consumer_conf = conf_sept['consumer_confidence'].values[0] if len(conf_sept) > 0 else None
    
    return {
        'gdp_growth': gdp_growth,
        'unemployment': unemployment,
        'inflation': inflation,
        'consumer_confidence': consumer_conf,
        'election_year': year
    }

economic_features = []
for year in elections['election_year'].unique():
    economic_features.append(get_q3_economics(int(year)))

econ_df = pd.DataFrame(economic_features)
elections = elections.merge(econ_df, on='election_year', how='left')

# Load polling
polls_historical = pd.read_csv('real_data/polling/pres_pollaverages_1968-2016.csv')
polls_2020_2024 = pd.read_csv('real_data/polling/presidential_general_averages_2024-09-12_uncorrected.csv')

polls_hist_clean = polls_historical[['cycle', 'state', 'modeldate', 'candidate_name', 'pct_trend_adjusted']].copy()
polls_hist_clean['date'] = pd.to_datetime(polls_hist_clean['modeldate'])
polls_hist_clean = polls_hist_clean.rename(columns={'cycle': 'election_year', 'pct_trend_adjusted': 'pct'})

polls_2020_clean = polls_2020_2024[['cycle', 'state', 'date', 'candidate', 'pct_trend_adjusted']].copy()
polls_2020_clean['date'] = pd.to_datetime(polls_2020_clean['date'])
polls_2020_clean = polls_2020_clean.rename(columns={'cycle': 'election_year', 'candidate': 'candidate_name', 'pct_trend_adjusted': 'pct'})

all_polls = pd.concat([polls_hist_clean, polls_2020_clean], ignore_index=True)

dem_candidates = ['Joseph R. Biden Jr.', 'Hillary Rodham Clinton', 'Barack Obama', 'John Kerry', 'Al Gore']
rep_candidates = ['Donald Trump', 'Mitt Romney', 'John McCain', 'George W. Bush']

all_polls['party'] = all_polls['candidate_name'].apply(
    lambda x: 'DEM' if x in dem_candidates else ('REP' if x in rep_candidates else 'OTHER')
)
all_polls = all_polls[all_polls['party'].isin(['DEM', 'REP'])]

def get_polling_features(state_full, year):
    state_polls = all_polls[(all_polls['state'] == state_full) & (all_polls['election_year'] == year)]
    
    if len(state_polls) == 0:
        return {'final_poll_dem': None, 'final_poll_rep': None, 'poll_margin': None, 'num_polls': 0}
    
    election_date = pd.Timestamp(year=int(year), month=11, day=3)
    final_cutoff = election_date - pd.Timedelta(days=14)
    final_polls = state_polls[state_polls['date'] >= final_cutoff]
    
    if len(final_polls) == 0:
        final_polls = state_polls[state_polls['date'] >= (election_date - pd.Timedelta(days=30))]
    
    if len(final_polls) == 0:
        return {'final_poll_dem': None, 'final_poll_rep': None, 'poll_margin': None, 'num_polls': len(state_polls)}
    
    dem_polls = final_polls[final_polls['party'] == 'DEM']['pct'].mean()
    rep_polls = final_polls[final_polls['party'] == 'REP']['pct'].mean()
    poll_margin = rep_polls - dem_polls if (pd.notna(rep_polls) and pd.notna(dem_polls)) else None
    
    return {'final_poll_dem': dem_polls, 'final_poll_rep': rep_polls, 'poll_margin': poll_margin, 'num_polls': len(state_polls)}

polling_features = []
for _, row in elections.iterrows():
    features = get_polling_features(row['state_full'], row['election_year'])
    features['state'] = row['state']
    features['election_year'] = row['election_year']
    polling_features.append(features)

polling_df = pd.DataFrame(polling_features)
elections = elections.merge(polling_df, on=['state', 'election_year'], how='left')

# Create election features
elections = elections.sort_values(['state', 'election_year'])
elections['prev_margin'] = elections.groupby('state')['margin'].shift(1)
elections['prev_dem_pct'] = elections.groupby('state')['dem_pct'].shift(1)
elections['prev_rep_pct'] = elections.groupby('state')['rep_pct'].shift(1)
elections['historical_avg_margin'] = elections.groupby('state')['margin'].transform(
    lambda x: x.expanding().mean().shift(1)
)

elections_with_features = elections[elections['election_year'] > 2000].copy()

print("✅ All data loaded and processed")

# Define confidence scoring function
def calculate_confidence(econ_pred_margin, poll_margin, consumer_confidence, num_polls):
    """
    Calculate betting confidence based on agreement and signal strength
    
    Returns: confidence_level (HIGH/MEDIUM/LOW), bet_multiplier (0.0 to 1.0)
    """
    
    # No polling data
    if pd.isna(poll_margin) or num_polls == 0:
        # Use economic confidence
        if pd.notna(consumer_confidence):
            if consumer_confidence < 75:  # Low confidence = strong signal
                return 'MEDIUM-HIGH', 0.75, 'Strong economic signal, no polls'
            else:
                return 'MEDIUM', 0.5, 'Weak economic signal, no polls'
        else:
            return 'MEDIUM', 0.5, 'No polls, unclear economics'
    
    # Have both economics and polling
    econ_direction = 'R' if econ_pred_margin > 0 else 'D'
    poll_direction = 'R' if poll_margin > 0 else 'D'
    
    # Check agreement
    if econ_direction == poll_direction:
        # Both agree on winner
        margin_diff = abs(abs(econ_pred_margin) - abs(poll_margin))
        
        if margin_diff < 3:  # Very close agreement
            return 'HIGH', 1.0, f'Strong agreement (econ: {econ_pred_margin:+.1f}%, poll: {poll_margin:+.1f}%)'
        elif margin_diff < 7:  # Moderate agreement
            return 'MEDIUM-HIGH', 0.75, f'Good agreement (econ: {econ_pred_margin:+.1f}%, poll: {poll_margin:+.1f}%)'
        else:  # Agree on winner but different confidence
            return 'MEDIUM', 0.5, f'Agree on winner, different margins (econ: {econ_pred_margin:+.1f}%, poll: {poll_margin:+.1f}%)'
    else:
        # Disagree on winner - RED FLAG
        return 'LOW', 0.25, f'DISAGREEMENT (econ: {econ_direction} {econ_pred_margin:+.1f}%, poll: {poll_direction} {poll_margin:+.1f}%)'

# Backtest with confidence-based betting
print("\n2. Backtesting Confidence-Based System")
print("=" * 70)

test_years = [2016, 2020]
all_results = []

econ_features = ['prev_margin', 'prev_dem_pct', 'prev_rep_pct', 'historical_avg_margin',
                 'gdp_growth', 'unemployment', 'inflation', 'consumer_confidence']

for test_year in test_years:
    print(f"\n{'=' * 70}")
    print(f"TESTING {test_year}")
    print('=' * 70)
    
    train_data = elections_with_features[elections_with_features['election_year'] < test_year]
    test_data = elections_with_features[elections_with_features['election_year'] == test_year]
    
    if len(train_data) == 0 or len(test_data) == 0:
        continue
    
    # Train economics-only model
    X_train = train_data[econ_features].fillna(0)
    y_train = (train_data['margin'] > 0).astype(int)
    
    X_test = test_data[econ_features].fillna(0)
    y_test = (test_data['margin'] > 0).astype(int)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Get predictions and probabilities
    econ_pred = model.predict(X_test_scaled)
    econ_proba = model.predict_proba(X_test_scaled)
    
    # Calculate predicted margins
    test_data_reset = test_data.reset_index(drop=True)
    test_data_reset['econ_pred'] = econ_pred
    test_data_reset['econ_rep_prob'] = econ_proba[:, 1]
    test_data_reset['econ_pred_margin'] = (test_data_reset['econ_rep_prob'] - 0.5) * 100 * 2  # Convert prob to margin
    
    # Calculate confidence for each race
    confidence_data = []
    
    for idx, row in test_data_reset.iterrows():
        conf_level, bet_mult, reason = calculate_confidence(
            row['econ_pred_margin'],
            row['poll_margin'],
            row['consumer_confidence'],
            row['num_polls']
        )
        
        confidence_data.append({
            'state': row['state'],
            'confidence_level': conf_level,
            'bet_multiplier': bet_mult,
            'reason': reason,
            'econ_pred': 'R' if row['econ_pred'] == 1 else 'D',
            'poll_pred': 'R' if row['poll_margin'] > 0 else 'D' if pd.notna(row['poll_margin']) else 'N/A',
            'actual': 'R' if row['margin'] > 0 else 'D',
            'econ_correct': row['econ_pred'] == (row['margin'] > 0),
            'would_bet': bet_mult >= 0.5  # Only bet if confidence >= MEDIUM
        })
    
    conf_df = pd.DataFrame(confidence_data)
    
    # Strategy 1: Bet everything (baseline)
    all_bets_correct = conf_df['econ_correct'].sum()
    all_bets_total = len(conf_df)
    all_bets_accuracy = all_bets_correct / all_bets_total
    
    # Strategy 2: Only bet with MEDIUM+ confidence
    conf_bets = conf_df[conf_df['would_bet']]
    conf_bets_correct = conf_bets['econ_correct'].sum()
    conf_bets_total = len(conf_bets)
    conf_bets_accuracy = conf_bets_correct / conf_bets_total if conf_bets_total > 0 else 0
    
    # Strategy 3: Weight by confidence (expected profit)
    conf_df['weighted_profit'] = conf_df['bet_multiplier'] * (conf_df['econ_correct'] * 2 - 1)  # Win = +multiplier, Lose = -multiplier
    expected_roi = conf_df['weighted_profit'].sum() / conf_df['bet_multiplier'].sum() if conf_df['bet_multiplier'].sum() > 0 else 0
    
    print(f"\nSTRATEGY COMPARISON:")
    print(f"\n1. Bet Everything (Baseline):")
    print(f"   Bets: {all_bets_total}")
    print(f"   Wins: {all_bets_correct}")
    print(f"   Accuracy: {all_bets_accuracy:.1%}")
    
    print(f"\n2. Confidence-Based Betting (MEDIUM+):")
    print(f"   Bets: {conf_bets_total} ({conf_bets_total/all_bets_total:.1%} of opportunities)")
    print(f"   Wins: {conf_bets_correct}")
    print(f"   Accuracy: {conf_bets_accuracy:.1%}")
    print(f"   Improvement: {(conf_bets_accuracy - all_bets_accuracy):+.1%}")
    
    print(f"\n3. Weighted Kelly Betting:")
    print(f"   Expected ROI: {expected_roi:.1%}")
    
    # Show swing states with confidence levels
    print(f"\nSWING STATE ANALYSIS:")
    swing_states = ['PA', 'WI', 'MI', 'GA', 'AZ', 'NV', 'NC', 'FL', 'OH']
    swing_df = conf_df[conf_df['state'].isin(swing_states)]
    
    for _, row in swing_df.iterrows():
        correct_symbol = '✓' if row['econ_correct'] else '✗'
        bet_decision = 'BET' if row['would_bet'] else 'SKIP'
        
        print(f"  {row['state']}: {row['confidence_level']:12s} | Econ: {row['econ_pred']} | Poll: {row['poll_pred']} | Actual: {row['actual']} {correct_symbol} | {bet_decision:4s} (mult: {row['bet_multiplier']:.2f})")
        print(f"       → {row['reason']}")
    
    all_results.append({
        'year': test_year,
        'all_bets_accuracy': all_bets_accuracy,
        'conf_bets_accuracy': conf_bets_accuracy,
        'conf_bets_taken': conf_bets_total,
        'conf_bets_pct': conf_bets_total / all_bets_total,
        'improvement': conf_bets_accuracy - all_bets_accuracy,
        'expected_roi': expected_roi
    })

# Summary
print("\n" + "=" * 70)
print("SUMMARY ACROSS ALL YEARS")
print("=" * 70)

results_df = pd.DataFrame(all_results)

print("\nYear-by-Year Results:")
for _, row in results_df.iterrows():
    print(f"\n{int(row['year'])}:")
    print(f"  Bet everything:        {row['all_bets_accuracy']:.1%}")
    print(f"  Confidence-based:      {row['conf_bets_accuracy']:.1%}")
    print(f"  Bets taken:            {row['conf_bets_taken']}/51 ({row['conf_bets_pct']:.1%})")
    print(f"  Improvement:           {row['improvement']:+.1%}")
    print(f"  Expected ROI:          {row['expected_roi']:+.1%}")

avg_all = results_df['all_bets_accuracy'].mean()
avg_conf = results_df['conf_bets_accuracy'].mean()
avg_improvement = results_df['improvement'].mean()
avg_roi = results_df['expected_roi'].mean()

print(f"\nAVERAGE PERFORMANCE:")
print(f"  Bet everything:        {avg_all:.1%}")
print(f"  Confidence-based:      {avg_conf:.1%}")
print(f"  Average improvement:   {avg_improvement:+.1%}")
print(f"  Average ROI:           {avg_roi:+.1%}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if avg_improvement > 0.03:
    print("\n🎯 EXCELLENT: Confidence-based betting significantly improves results!")
    print("   By selectively betting, you avoid mistakes and increase win rate.")
elif avg_improvement > 0:
    print("\n✅ GOOD: Confidence-based betting provides modest improvement")
elif avg_roi > avg_all:
    print("\n👍 POSITIVE: While accuracy similar, ROI is better with selective betting")
else:
    print("\n⚠️ MIXED: Confidence-based approach needs refinement")

print("\n" + "=" * 70)