import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

print("=" * 70)
print("VALIDATING POLLING ON HISTORICAL ELECTIONS")
print("=" * 70)

# Load election results
elections = pd.read_csv('data/real_election_results.csv')

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

# Load polling data
print("\n1. Loading polling data...")
polls_historical = pd.read_csv('real_data/polling/pres_pollaverages_1968-2016.csv')
polls_2020 = pd.read_csv('real_data/polling/presidential_general_averages_2024-09-12_uncorrected.csv')

# Process polls
polls_hist_clean = polls_historical[['cycle', 'state', 'modeldate', 'candidate_name', 'pct_trend_adjusted']].copy()
polls_hist_clean['date'] = pd.to_datetime(polls_hist_clean['modeldate'])
polls_hist_clean = polls_hist_clean.rename(columns={'cycle': 'election_year', 'pct_trend_adjusted': 'pct'})

polls_2020_clean = polls_2020[['cycle', 'state', 'date', 'candidate', 'pct_trend_adjusted']].copy()
polls_2020_clean['date'] = pd.to_datetime(polls_2020_clean['date'])
polls_2020_clean = polls_2020_clean.rename(columns={'cycle': 'election_year', 'candidate': 'candidate_name', 'pct_trend_adjusted': 'pct'})

all_polls = pd.concat([polls_hist_clean, polls_2020_clean], ignore_index=True)

# Identify parties
dem_candidates = ['Joseph R. Biden Jr.', 'Hillary Clinton', 'Barack Obama', 'John Kerry', 'Al Gore']
rep_candidates = ['Donald Trump', 'Mitt Romney', 'John McCain', 'George W. Bush']

all_polls['party'] = all_polls['candidate_name'].apply(
    lambda x: 'DEM' if x in dem_candidates else ('REP' if x in rep_candidates else 'OTHER')
)
all_polls = all_polls[all_polls['party'].isin(['DEM', 'REP'])]

print(f"✅ Loaded {len(all_polls)} poll records")
print(f"   Years with polling: {sorted(all_polls['election_year'].unique())}")

# Create polling features
def get_polling_features(state, year):
    state_polls = all_polls[(all_polls['state'] == state) & (all_polls['election_year'] == year)]
    
    if len(state_polls) == 0:
        return {
            'final_poll_dem': None,
            'final_poll_rep': None,
            'poll_margin': None,
            'poll_volatility': None,
            'num_polls': 0
        }
    
    election_date = pd.Timestamp(year=int(year), month=11, day=3)
    final_cutoff = election_date - pd.Timedelta(days=14)
    final_polls = state_polls[state_polls['date'] >= final_cutoff]
    
    if len(final_polls) == 0:
        final_polls = state_polls.tail(20)
    
    dem_polls = final_polls[final_polls['party'] == 'DEM']['pct'].mean()
    rep_polls = final_polls[final_polls['party'] == 'REP']['pct'].mean()
    poll_margin = rep_polls - dem_polls if (pd.notna(rep_polls) and pd.notna(dem_polls)) else None
    
    volatility_cutoff = election_date - pd.Timedelta(days=30)
    recent_polls = state_polls[state_polls['date'] >= volatility_cutoff]
    
    if len(recent_polls) > 5:
        poll_volatility = recent_polls.groupby('party')['pct'].std().mean()
    else:
        poll_volatility = None
    
    return {
        'final_poll_dem': dem_polls,
        'final_poll_rep': rep_polls,
        'poll_margin': poll_margin,
        'poll_volatility': poll_volatility if pd.notna(poll_volatility) else 0,
        'num_polls': len(state_polls)
    }

polling_features = []
for _, row in elections.iterrows():
    features = get_polling_features(row['state'], row['election_year'])
    features['state'] = row['state']
    features['election_year'] = row['election_year']
    polling_features.append(features)

polling_df = pd.DataFrame(polling_features)
elections = elections.merge(polling_df, on=['state', 'election_year'], how='left')

print(f"✅ Created polling features")

# Check which years have good polling data
polling_coverage = polling_df.groupby('election_year')['num_polls'].agg(['sum', 'mean'])
print("\nPolling coverage by year:")
print(polling_coverage)

# Create all features
elections = elections.sort_values(['state', 'election_year'])
elections['prev_margin'] = elections.groupby('state')['margin'].shift(1)
elections['prev_dem_pct'] = elections.groupby('state')['dem_pct'].shift(1)
elections['prev_rep_pct'] = elections.groupby('state')['rep_pct'].shift(1)
elections['historical_avg_margin'] = elections.groupby('state')['margin'].transform(
    lambda x: x.expanding().mean().shift(1)
)

elections_with_features = elections[elections['election_year'] > 2000].copy()
elections_with_features = elections_with_features.ffill()

# Feature sets
econ_features = ['prev_margin', 'prev_dem_pct', 'prev_rep_pct', 'historical_avg_margin',
                 'gdp_growth', 'unemployment', 'inflation', 'consumer_confidence']

polling_features_list = ['final_poll_dem', 'final_poll_rep', 'poll_margin', 'poll_volatility']

all_features = econ_features + polling_features_list

print("\n2. Testing: Economics-Only vs Economics+Polling")
print("=" * 70)

# Test on years where we have polling data
test_years = [2020, 2016]

results = []

for test_year in test_years:
    print(f"\n{'=' * 70}")
    print(f"TESTING {test_year}")
    print('=' * 70)
    
    # Train on all years before test year
    train_data = elections_with_features[elections_with_features['election_year'] < test_year]
    test_data = elections_with_features[elections_with_features['election_year'] == test_year]
    
    if len(train_data) == 0 or len(test_data) == 0:
        continue
    
    y_train = (train_data['margin'] > 0).astype(int)
    y_test = (test_data['margin'] > 0).astype(int)
    
    # Count how many states have polling in test year
    test_with_polls = test_data[test_data['num_polls'] > 0]
    print(f"\nStates with polling data in {test_year}: {len(test_with_polls)}/51")
    
    # Test 1: Economics Only
    X_train_econ = train_data[econ_features].fillna(0)
    X_test_econ = test_data[econ_features].fillna(0)
    
    scaler_econ = StandardScaler()
    X_train_econ_scaled = scaler_econ.fit_transform(X_train_econ)
    X_test_econ_scaled = scaler_econ.transform(X_test_econ)
    
    model_econ = LogisticRegression(random_state=42, max_iter=1000)
    model_econ.fit(X_train_econ_scaled, y_train)
    
    pred_econ = model_econ.predict(X_test_econ_scaled)
    acc_econ = accuracy_score(y_test, pred_econ)
    
    # Test 2: Economics + Polling
    X_train_all = train_data[all_features].fillna(0)
    X_test_all = test_data[all_features].fillna(0)
    
    scaler_all = StandardScaler()
    X_train_all_scaled = scaler_all.fit_transform(X_train_all)
    X_test_all_scaled = scaler_all.transform(X_test_all)
    
    model_all = LogisticRegression(random_state=42, max_iter=1000)
    model_all.fit(X_train_all_scaled, y_train)
    
    pred_all = model_all.predict(X_test_all_scaled)
    acc_all = accuracy_score(y_test, pred_all)
    
    # Calculate swing state accuracy
    swing_states = ['PA', 'WI', 'MI', 'GA', 'AZ', 'NV', 'NC', 'FL', 'OH']
    test_data_reset = test_data.reset_index(drop=True)
    
    swing_mask = test_data_reset['state'].isin(swing_states)
    
    if swing_mask.sum() > 0:
        swing_acc_econ = accuracy_score(y_test[swing_mask.values], pred_econ[swing_mask.values])
        swing_acc_all = accuracy_score(y_test[swing_mask.values], pred_all[swing_mask.values])
    else:
        swing_acc_econ = None
        swing_acc_all = None
    
    # Results
    print(f"\nECONOMICS ONLY:")
    print(f"  Overall accuracy: {acc_econ:.1%} ({int(acc_econ*51)}/51)")
    if swing_acc_econ is not None:
        print(f"  Swing states: {swing_acc_econ:.1%}")
    
    print(f"\nECONOMICS + POLLING:")
    print(f"  Overall accuracy: {acc_all:.1%} ({int(acc_all*51)}/51)")
    if swing_acc_all is not None:
        print(f"  Swing states: {swing_acc_all:.1%}")
    
    improvement = acc_all - acc_econ
    print(f"\n{'🎯' if improvement > 0 else '⚠️'} IMPROVEMENT: {improvement:+.1%}")
    
    # Show swing state details
    print(f"\nSwing state breakdown:")
    for state in swing_states:
        if state in test_data_reset['state'].values:
            idx = test_data_reset[test_data_reset['state'] == state].index[0]
            
            actual_winner = 'R' if y_test.iloc[idx] == 1 else 'D'
            pred_econ_winner = 'R' if pred_econ[idx] == 1 else 'D'
            pred_all_winner = 'R' if pred_all[idx] == 1 else 'D'
            
            econ_correct = '✓' if pred_econ[idx] == y_test.iloc[idx] else '✗'
            all_correct = '✓' if pred_all[idx] == y_test.iloc[idx] else '✗'
            
            row = test_data_reset.iloc[idx]
            num_polls = row['num_polls']
            poll_marg = row.get('poll_margin', 0)
            
            print(f"  {state}: Actual {actual_winner} | Econ: {pred_econ_winner} {econ_correct} | +Polling: {pred_all_winner} {all_correct} | Polls: {num_polls:3.0f} (margin: {poll_marg:+.1f})")
    
    results.append({
        'year': test_year,
        'econ_only': acc_econ,
        'econ_polling': acc_all,
        'improvement': improvement,
        'swing_econ': swing_acc_econ,
        'swing_polling': swing_acc_all
    })

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(results)
print("\nOverall Results:")
for _, row in results_df.iterrows():
    print(f"\n{int(row['year'])}:")
    print(f"  Economics only:     {row['econ_only']:.1%}")
    print(f"  + Polling:          {row['econ_polling']:.1%}")
    print(f"  Improvement:        {row['improvement']:+.1%}")
    if pd.notna(row['swing_econ']):
        print(f"  Swing (econ only):  {row['swing_econ']:.1%}")
        print(f"  Swing (+ polling):  {row['swing_polling']:.1%}")

avg_improvement = results_df['improvement'].mean()
print(f"\nAverage improvement: {avg_improvement:+.1%}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if avg_improvement >= 0.05:
    print("\n🎯 EXCELLENT: Polling provides significant improvement!")
    print("   Adding polling is worth the effort")
    print("   Expected to help in future elections")
elif avg_improvement >= 0.02:
    print("\n✅ GOOD: Polling provides moderate improvement")
    print("   Worth including for marginal edge")
elif avg_improvement >= 0:
    print("\n👍 SLIGHT: Polling helps a little")
    print("   Economics is doing most of the work")
else:
    print("\n⚠️ NO BENEFIT: Polling doesn't help in this dataset")
    print("   Economics alone is sufficient")

print("\n" + "=" * 70)