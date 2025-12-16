import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os

print("=" * 70)
print("TRAINING WITH REAL DATA + ECONOMICS + POLLING")
print("=" * 70)

# Load election results
print("\n1. Loading election data...")
elections = pd.read_csv('data/real_election_results.csv')
print(f"✅ Loaded {len(elections)} election records")

# Load economic data
print("\n2. Loading economic indicators...")
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

print("✅ Economic data merged")

# Load and process polling data
print("\n3. Loading and processing polling data...")

# Load historical polls (1968-2016)
polls_historical = pd.read_csv('real_data/polling/pres_pollaverages_1968-2016.csv')
print(f"   Loaded {len(polls_historical)} historical poll records")

# Load 2020 polls
polls_2020 = pd.read_csv('real_data/polling/presidential_general_averages_2024-09-12_uncorrected.csv')
print(f"   Loaded {len(polls_2020)} 2020 poll records")

# Process historical polls
polls_hist_clean = polls_historical[['cycle', 'state', 'modeldate', 'candidate_name', 'pct_trend_adjusted']].copy()
polls_hist_clean['date'] = pd.to_datetime(polls_hist_clean['modeldate'])
polls_hist_clean = polls_hist_clean.rename(columns={'cycle': 'election_year', 'pct_trend_adjusted': 'pct'})

# Process 2020 polls
polls_2020_clean = polls_2020[['cycle', 'state', 'date', 'candidate', 'pct_trend_adjusted']].copy()
polls_2020_clean['date'] = pd.to_datetime(polls_2020_clean['date'])
polls_2020_clean = polls_2020_clean.rename(columns={'cycle': 'election_year', 'candidate': 'candidate_name', 'pct_trend_adjusted': 'pct'})

# Combine polls
all_polls = pd.concat([polls_hist_clean, polls_2020_clean], ignore_index=True)

# Identify party for each candidate
dem_candidates = ['Joseph R. Biden Jr.', 'Hillary Clinton', 'Barack Obama', 'John Kerry', 'Al Gore']
rep_candidates = ['Donald Trump', 'Mitt Romney', 'John McCain', 'George W. Bush']

all_polls['party'] = all_polls['candidate_name'].apply(
    lambda x: 'DEM' if x in dem_candidates else ('REP' if x in rep_candidates else 'OTHER')
)

# Filter for major parties only
all_polls = all_polls[all_polls['party'].isin(['DEM', 'REP'])]

print(f"✅ Combined and cleaned {len(all_polls)} poll records")
print(f"   Years: {sorted(all_polls['election_year'].unique())}")

# Create polling features for each state-year
print("\n4. Creating polling features...")

def get_polling_features(state, year):
    """Extract polling features for a state-year"""
    
    state_polls = all_polls[(all_polls['state'] == state) & (all_polls['election_year'] == year)]
    
    if len(state_polls) == 0:
        return {
            'final_poll_dem': None,
            'final_poll_rep': None,
            'poll_margin': None,
            'poll_volatility': None,
            'num_polls': 0
        }
    
    # Get election date (assume November 3rd)
    election_date = pd.Timestamp(year=int(year), month=11, day=3)
    
    # Final polls (last 14 days before election)
    final_cutoff = election_date - pd.Timedelta(days=14)
    final_polls = state_polls[state_polls['date'] >= final_cutoff]
    
    if len(final_polls) == 0:
        final_polls = state_polls.tail(20)  # Use last 20 polls if no recent ones
    
    # Calculate averages by party
    dem_polls = final_polls[final_polls['party'] == 'DEM']['pct'].mean()
    rep_polls = final_polls[final_polls['party'] == 'REP']['pct'].mean()
    
    # Poll margin
    poll_margin = rep_polls - dem_polls if (pd.notna(rep_polls) and pd.notna(dem_polls)) else None
    
    # Poll volatility (std dev of last 30 days)
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

# Create polling features for all state-year combinations
polling_features = []

for _, row in elections.iterrows():
    features = get_polling_features(row['state'], row['election_year'])
    features['state'] = row['state']
    features['election_year'] = row['election_year']
    polling_features.append(features)

polling_df = pd.DataFrame(polling_features)

print(f"✅ Created polling features for {len(polling_df)} state-year combinations")

# Show sample
print("\nSample polling features:")
sample = polling_df[polling_df['election_year'] == 2020].head()
print(sample[['state', 'election_year', 'final_poll_dem', 'final_poll_rep', 'poll_margin', 'num_polls']])

# Merge with elections
elections = elections.merge(polling_df, on=['state', 'election_year'], how='left')

print("\n5. Creating all features...")

# Create lagged election features
elections = elections.sort_values(['state', 'election_year'])
elections['prev_margin'] = elections.groupby('state')['margin'].shift(1)
elections['prev_dem_pct'] = elections.groupby('state')['dem_pct'].shift(1)
elections['prev_rep_pct'] = elections.groupby('state')['rep_pct'].shift(1)
elections['historical_avg_margin'] = elections.groupby('state')['margin'].transform(
    lambda x: x.expanding().mean().shift(1)
)

# Remove 2000 (no previous data)
elections_with_features = elections[elections['election_year'] > 2000].copy()
elections_with_features = elections_with_features.ffill()

# Fill missing polling data with zero or mean
elections_with_features['final_poll_dem'] = elections_with_features['final_poll_dem'].fillna(
    elections_with_features.groupby('election_year')['final_poll_dem'].transform('mean')
)
elections_with_features['final_poll_rep'] = elections_with_features['final_poll_rep'].fillna(
    elections_with_features.groupby('election_year')['final_poll_rep'].transform('mean')
)
elections_with_features['poll_margin'] = elections_with_features['poll_margin'].fillna(0)
elections_with_features['poll_volatility'] = elections_with_features['poll_volatility'].fillna(0)

print(f"✅ Created all features for {len(elections_with_features)} records")

# Prepare training data
print("\n6. Preparing training/test split...")

train_data = elections_with_features[elections_with_features['election_year'] < 2024]
test_data = elections_with_features[elections_with_features['election_year'] == 2024]

print(f"   Training: {len(train_data)} records")
print(f"   Testing: {len(test_data)} records")

# Select features - NOW WITH POLLING!
feature_cols = [
    'prev_margin', 'prev_dem_pct', 'prev_rep_pct', 'historical_avg_margin',
    'gdp_growth', 'unemployment', 'inflation', 'consumer_confidence',
    'final_poll_dem', 'final_poll_rep', 'poll_margin', 'poll_volatility'
]

X_train = train_data[feature_cols].fillna(0)
y_train = (train_data['margin'] > 0).astype(int)

X_test = test_data[feature_cols].fillna(0)
y_test = (test_data['margin'] > 0).astype(int)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Features ({len(feature_cols)}): {feature_cols}")

# Train models
print("\n7. Training models with POLLING features...")
print("-" * 70)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
}

results = {}

for name, model in models.items():
    print(f"\n{name}:")
    model.fit(X_train_scaled, y_train)
    
    train_pred = model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_pred)
    
    test_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"  Training accuracy: {train_acc:.1%}")
    print(f"  Test accuracy: {test_acc:.1%}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importances = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n  Top 8 features:")
        for idx, row in importances.head(8).iterrows():
            print(f"    {row['feature']}: {row['importance']:.3f}")
    
    # Swing state predictions
    test_data_copy = test_data.copy()
    test_data_copy['predicted'] = test_pred
    test_data_copy['correct'] = test_pred == y_test
    
    print(f"\n  Swing state predictions (2024):")
    swing_states = ['PA', 'WI', 'MI', 'GA', 'AZ', 'NV', 'NC']
    for state in swing_states:
        if state in test_data_copy['state'].values:
            row = test_data_copy[test_data_copy['state'] == state].iloc[0]
            pred_winner = 'Republican' if row['predicted'] == 1 else 'Democrat'
            actual_winner = 'Republican' if row['margin'] > 0 else 'Democrat'
            correct = '✓' if row['correct'] else '✗'
            margin = abs(row['margin'])
            poll_margin = row.get('poll_margin', 0)
            print(f"    {state}: Predicted {pred_winner:10s}, Actual {actual_winner:10s} {correct} (margin: {margin:.1f}%, poll: {poll_margin:+.1f}%)")
    
    results[name] = {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc
    }

# Save best model
print("\n8. Saving models...")
os.makedirs('data/models', exist_ok=True)

best_model_name = max(results, key=lambda x: results[x]['test_acc'])
best_model = results[best_model_name]['model']

joblib.dump(best_model, 'data/models/best_model_with_polling.pkl')
joblib.dump(scaler, 'data/models/scaler_with_polling.pkl')

with open('data/models/features_with_polling.txt', 'w') as f:
    f.write('\n'.join(feature_cols))

print(f"✅ Saved best model: {best_model_name}")
print(f"   Test accuracy: {results[best_model_name]['test_acc']:.1%}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE WITH POLLING!")
print("=" * 70)
print(f"\nBest Model: {best_model_name}")
print(f"Test Accuracy: {results[best_model_name]['test_acc']:.1%}")
print(f"Improvement over economics-only: {results[best_model_name]['test_acc'] - 1.0:+.1%}")
print(f"\nFeatures used: {len(feature_cols)}")
print(f"  - 4 historical election features")
print(f"  - 4 economic features")
print(f"  - 4 polling features (NEW!)")
print("\nModels saved to: data/models/")