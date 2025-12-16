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
print("TRAINING MODELS WITH REAL DATA")
print("=" * 70)

# Load real election results
print("\n1. Loading real election data...")
elections = pd.read_csv('data/real_election_results.csv')
print(f"✅ Loaded {len(elections)} election records")
print(f"   Years: {sorted(elections['election_year'].unique())}")
print(f"   States: {elections['state'].nunique()}")

# Create simple features from real data
print("\n2. Creating features...")

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

print(f"✅ Created features for {len(elections_with_features)} records")

# Prepare training data
print("\n3. Preparing training/test split...")

# Use 2000-2020 for training, 2024 for testing
train_data = elections_with_features[elections_with_features['election_year'] < 2024]
test_data = elections_with_features[elections_with_features['election_year'] == 2024]

print(f"   Training: {len(train_data)} records (2004-2020)")
print(f"   Testing: {len(test_data)} records (2024)")

# Select features
feature_cols = ['prev_margin', 'prev_dem_pct', 'prev_rep_pct', 'historical_avg_margin']

X_train = train_data[feature_cols].fillna(0)
y_train = (train_data['margin'] > 0).astype(int)  # 1 = Republican win, 0 = Democrat win

X_test = test_data[feature_cols].fillna(0)
y_test = (test_data['margin'] > 0).astype(int)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Features: {feature_cols}")

# Train models
print("\n4. Training models...")
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
    
    # Show predictions for a few states
    test_data_copy = test_data.copy()
    test_data_copy['predicted'] = test_pred
    test_data_copy['correct'] = test_pred == y_test
    
    print(f"\n  Sample predictions (2024):")
    sample_states = ['PA', 'WI', 'MI', 'GA', 'AZ', 'NV']
    for state in sample_states:
        if state in test_data_copy['state'].values:
            row = test_data_copy[test_data_copy['state'] == state].iloc[0]
            pred_winner = 'Republican' if row['predicted'] == 1 else 'Democrat'
            actual_winner = 'Republican' if row['margin'] > 0 else 'Democrat'
            correct = '✓' if row['correct'] else '✗'
            print(f"    {state}: Predicted {pred_winner}, Actual {actual_winner} {correct}")
    
    results[name] = {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc
    }

# Save best model
print("\n5. Saving models...")
os.makedirs('data/models', exist_ok=True)

best_model_name = max(results, key=lambda x: results[x]['test_acc'])
best_model = results[best_model_name]['model']

joblib.dump(best_model, 'data/models/best_model_real_data.pkl')
joblib.dump(scaler, 'data/models/scaler_real_data.pkl')

print(f"✅ Saved best model: {best_model_name}")
print(f"   Test accuracy: {results[best_model_name]['test_acc']:.1%}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"\nBest Model: {best_model_name}")
print(f"Test Accuracy: {results[best_model_name]['test_acc']:.1%}")
print(f"Training Records: {len(train_data)} (6 elections × 51 states)")
print(f"\nModels saved to: data/models/")
print("\nNext: Run backtesting with real data!")