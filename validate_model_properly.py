import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

print("=" * 70)
print("PROPER MODEL VALIDATION - LEAVE-ONE-ELECTION-OUT")
print("=" * 70)

# Load data with all features
elections = pd.read_csv('data/real_election_results.csv')

# Load and process economic data
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

# Add economic features
economic_features = []
for year in elections['election_year'].unique():
    economic_features.append(get_q3_economics(int(year)))

econ_df = pd.DataFrame(economic_features)
elections = elections.merge(econ_df, on='election_year', how='left')

# Create lagged features
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

feature_cols = [
    'prev_margin', 'prev_dem_pct', 'prev_rep_pct', 'historical_avg_margin',
    'gdp_growth', 'unemployment', 'inflation', 'consumer_confidence'
]

print("\nLEAVE-ONE-ELECTION-OUT CROSS-VALIDATION")
print("(Train on all years except one, test on that year)")
print("-" * 70)

# Test each election year
test_years = [2024, 2020, 2016, 2012, 2008, 2004]
all_results = []

for test_year in test_years:
    # Train on all years BEFORE this one
    train_data = elections_with_features[elections_with_features['election_year'] < test_year]
    test_data = elections_with_features[elections_with_features['election_year'] == test_year]
    
    if len(train_data) == 0 or len(test_data) == 0:
        continue
    
    X_train = train_data[feature_cols].fillna(0)
    y_train = (train_data['margin'] > 0).astype(int)
    
    X_test = test_data[feature_cols].fillna(0)
    y_test = (test_data['margin'] > 0).astype(int)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, pred)
    
    # Count swing state accuracy
    swing_states = ['PA', 'WI', 'MI', 'GA', 'AZ', 'NV', 'NC', 'FL', 'OH']
    swing_mask = test_data['state'].isin(swing_states)
    
    if swing_mask.sum() > 0:
        swing_pred = pred[swing_mask.values]
        swing_actual = y_test[swing_mask.values]
        swing_acc = accuracy_score(swing_actual, swing_pred)
    else:
        swing_acc = None
    
    training_years = sorted(train_data['election_year'].unique())
    
    print(f"\nTesting {test_year}:")
    print(f"  Training years: {training_years}")
    print(f"  Overall accuracy: {acc:.1%} ({int(acc*len(test_data))}/{len(test_data)} states)")
    if swing_acc is not None:
        print(f"  Swing state accuracy: {swing_acc:.1%}")
    
    # Show sample predictions - FIXED
    test_data_reset = test_data.reset_index(drop=True)
    for i in range(min(5, len(test_data_reset))):
        row = test_data_reset.iloc[i]
        pred_winner = 'R' if pred[i] == 1 else 'D'
        actual_winner = 'R' if row['margin'] > 0 else 'D'
        correct = '✓' if pred[i] == (row['margin'] > 0) else '✗'
        print(f"    {row['state']}: {pred_winner} (actual: {actual_winner}) {correct}")
    
    all_results.append({
        'test_year': test_year,
        'accuracy': acc,
        'swing_accuracy': swing_acc
    })

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(all_results)
print(f"\nResults by year:")
for _, row in results_df.iterrows():
    swing_str = f", Swing: {row['swing_accuracy']:.1%}" if pd.notna(row['swing_accuracy']) else ""
    print(f"  {int(row['test_year'])}: {row['accuracy']:.1%}{swing_str}")

print(f"\nAverage accuracy across all elections: {results_df['accuracy'].mean():.1%}")
print(f"Best: {results_df['accuracy'].max():.1%} in {int(results_df.loc[results_df['accuracy'].idxmax(), 'test_year'])}")
print(f"Worst: {results_df['accuracy'].min():.1%} in {int(results_df.loc[results_df['accuracy'].idxmin(), 'test_year'])}")

swing_results = results_df[results_df['swing_accuracy'].notna()]
if len(swing_results) > 0:
    print(f"\nSwing states average: {swing_results['swing_accuracy'].mean():.1%}")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

avg_acc = results_df['accuracy'].mean()

if avg_acc >= 0.95:
    print("\n🎯 EXCELLENT: Model consistently performs at 95%+")
    print("   The 100% on 2024 appears to be REAL predictive power")
    print("   Economic features provide genuine edge")
    print("   This model is production-ready!")
elif avg_acc >= 0.90:
    print("\n✅ VERY GOOD: Model consistently performs at 90%+")
    print("   The 100% on 2024 was slightly lucky but mostly skill")
    print("   Model has real predictive power")
    print("   Ready for careful paper trading")
elif avg_acc >= 0.85:
    print("\n👍 GOOD: Model consistently performs at 85%+")
    print("   The 100% on 2024 was lucky")
    print("   But model still beats random chance significantly")
    print("   Needs more features before real money")
else:
    print("\n⚠️  CONCERNING: Average accuracy below 85%")
    print("   The 100% on 2024 was mostly luck")
    print("   Model needs improvement")

print("\n" + "=" * 70)