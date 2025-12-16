import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

print("=" * 70)
print("TRAINING MODELS WITH REAL DATA + ECONOMICS")
print("=" * 70)

# Load real election results
print("\n1. Loading real election data...")
elections = pd.read_csv('data/real_election_results.csv')
print(f"✅ Loaded {len(elections)} election records")

# Load economic data
print("\n2. Loading economic indicators...")

# Read economic CSVs
gdp_df = pd.read_csv('real_data/economic/GDPC1.csv')
unemployment_df = pd.read_csv('real_data/economic/UNRATE.csv')
inflation_df = pd.read_csv('real_data/economic/CPIAUCSL.csv')
confidence_df = pd.read_csv('real_data/economic/UMCSENT.csv')

# Rename columns to standard format
gdp_df.columns = ['DATE', 'gdp']
unemployment_df.columns = ['DATE', 'unemployment']
inflation_df.columns = ['DATE', 'inflation']
confidence_df.columns = ['DATE', 'consumer_confidence']

# Convert dates
for df in [gdp_df, unemployment_df, inflation_df, confidence_df]:
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['year'] = df['DATE'].dt.year
    df['quarter'] = df['DATE'].dt.quarter

print("✅ Loaded all economic indicators")

# Get Q3 (pre-election) economic data for each election year
def get_q3_economics(year):
    """Get Q3 economic indicators for election year"""
    
    # GDP (quarterly data) - Q3
    gdp_q3 = gdp_df[(gdp_df['year'] == year) & (gdp_df['quarter'] == 3)]
    gdp_val = gdp_q3['gdp'].values[0] if len(gdp_q3) > 0 else None
    
    # Get previous year Q3 for growth calculation
    gdp_prev = gdp_df[(gdp_df['year'] == year-1) & (gdp_df['quarter'] == 3)]
    gdp_prev_val = gdp_prev['gdp'].values[0] if len(gdp_prev) > 0 else None
    gdp_growth = ((gdp_val - gdp_prev_val) / gdp_prev_val * 100) if (gdp_val and gdp_prev_val) else 0
    
    # Unemployment (monthly) - Average of July, Aug, Sept
    unemp_q3 = unemployment_df[(unemployment_df['year'] == year) & 
                               (unemployment_df['DATE'].dt.month.isin([7,8,9]))]
    unemployment = unemp_q3['unemployment'].mean() if len(unemp_q3) > 0 else None
    
    # Inflation (monthly) - September value (most recent)
    infl_sept = inflation_df[(inflation_df['year'] == year) & 
                             (inflation_df['DATE'].dt.month == 9)]
    
    # Calculate year-over-year inflation
    infl_prev = inflation_df[(inflation_df['year'] == year-1) & 
                             (inflation_df['DATE'].dt.month == 9)]
    
    if len(infl_sept) > 0 and len(infl_prev) > 0:
        inflation = ((infl_sept['inflation'].values[0] - infl_prev['inflation'].values[0]) / 
                    infl_prev['inflation'].values[0] * 100)
    else:
        inflation = None
    
    # Consumer confidence (monthly) - September value
    conf_sept = confidence_df[(confidence_df['year'] == year) & 
                              (confidence_df['DATE'].dt.month == 9)]
    consumer_conf = conf_sept['consumer_confidence'].values[0] if len(conf_sept) > 0 else None
    
    return {
        'gdp_growth': gdp_growth,
        'unemployment': unemployment,
        'inflation': inflation,
        'consumer_confidence': consumer_conf
    }

# Add economic data for each election year
print("\n3. Merging economic data with elections...")

economic_features = []
for year in elections['election_year'].unique():
    econ = get_q3_economics(int(year))
    econ['election_year'] = year
    economic_features.append(econ)

econ_df = pd.DataFrame(economic_features)
print(f"✅ Created economic features for {len(econ_df)} election years")
print("\nEconomic conditions by year:")
print(econ_df)

# Merge with elections
elections = elections.merge(econ_df, on='election_year', how='left')

print("\n4. Creating election features...")

# Add previous election results (lagged features)
elections = elections.sort_values(['state', 'election_year'])
elections['prev_margin'] = elections.groupby('state')['margin'].shift(1)
elections['prev_dem_pct'] = elections.groupby('state')['dem_pct'].shift(1)
elections['prev_rep_pct'] = elections.groupby('state')['rep_pct'].shift(1)

# Add historical average
elections['historical_avg_margin'] = elections.groupby('state')['margin'].transform(
    lambda x: x.expanding().mean().shift(1)
)

# Remove first election (no previous data)
elections_with_features = elections[elections['election_year'] > 2000].copy()

# Fill any missing economic data
elections_with_features = elections_with_features.fillna(method='ffill')

print(f"✅ Created all features for {len(elections_with_features)} records")

# Prepare training data
print("\n5. Preparing training/test split...")

# Use 2004-2020 for training, 2024 for testing
train_data = elections_with_features[elections_with_features['election_year'] < 2024]
test_data = elections_with_features[elections_with_features['election_year'] == 2024]

print(f"   Training: {len(train_data)} records (2004-2020)")
print(f"   Testing: {len(test_data)} records (2024)")

# Select features - NOW WITH ECONOMICS!
feature_cols = [
    'prev_margin', 
    'prev_dem_pct', 
    'prev_rep_pct', 
    'historical_avg_margin',
    'gdp_growth',
    'unemployment',
    'inflation',
    'consumer_confidence'
]

X_train = train_data[feature_cols].fillna(0)
y_train = (train_data['margin'] > 0).astype(int)

X_test = test_data[feature_cols].fillna(0)
y_test = (test_data['margin'] > 0).astype(int)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Features: {feature_cols}")

# Train models
print("\n6. Training models with economic features...")
print("-" * 70)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
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
        
        print(f"\n  Top 5 features:")
        for idx, row in importances.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance']:.3f}")
    
    # Show predictions for swing states
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
            print(f"    {state}: Predicted {pred_winner:10s}, Actual {actual_winner:10s} {correct} (margin: {margin:.1f}%)")
    
    results[name] = {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc
    }

# Save best model
print("\n7. Saving models...")
os.makedirs('data/models', exist_ok=True)

best_model_name = max(results, key=lambda x: results[x]['test_acc'])
best_model = results[best_model_name]['model']

joblib.dump(best_model, 'data/models/best_model_with_economics.pkl')
joblib.dump(scaler, 'data/models/scaler_with_economics.pkl')

# Save feature names
with open('data/models/features_with_economics.txt', 'w') as f:
    f.write('\n'.join(feature_cols))

print(f"✅ Saved best model: {best_model_name}")
print(f"   Test accuracy: {results[best_model_name]['test_acc']:.1%}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE WITH ECONOMICS!")
print("=" * 70)
print(f"\nBest Model: {best_model_name}")
print(f"Test Accuracy: {results[best_model_name]['test_acc']:.1%}")
print(f"Improvement: {results[best_model_name]['test_acc'] - 0.902:.1%} vs baseline")
print(f"\nFeatures used: {len(feature_cols)} (4 election + 4 economic)")
print(f"Training records: {len(train_data)}")
print(f"\nModels saved to: data/models/")