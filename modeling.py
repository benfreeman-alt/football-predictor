"""
Predictive Modeling for Election Prediction Markets
====================================================

This module builds and evaluates multiple ML models to predict election outcomes
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import joblib
import os


class ElectionPredictor:
    """Build and train models to predict election outcomes"""
    
    def __init__(self, data_dir: str = "/home/claude/data"):
        self.data_dir = data_dir
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load preprocessed training and test data"""
        train_df = pd.read_csv(os.path.join(self.data_dir, "train_data.csv"))
        test_df = pd.read_csv(os.path.join(self.data_dir, "test_data.csv"))
        
        return train_df, test_df
    
    def prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target variable
        
        Target: Whether Republican wins (margin > 0)
        """
        # Select feature columns (exclude identifiers and target)
        feature_cols = [
            'prev_election_margin', 'historical_avg_margin',
            'final_poll_dem', 'final_poll_rep', 'final_poll_margin',
            'poll_momentum_dem', 'poll_momentum_rep', 'poll_volatility',
            'gdp_growth_q3', 'unemployment_q3', 'inflation_q3',
            'consumer_confidence_q3', 'gdp_trend', 'unemployment_trend'
        ]
        
        # Keep only columns that exist and have data
        available_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[available_cols].copy()
        
        # Handle missing values more robustly
        # First pass: fill with column means
        for col in X.columns:
            if X[col].isna().any():
                mean_val = X[col].mean()
                if pd.isna(mean_val):  # If all values are NaN, use 0
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(mean_val)
        
        # Second pass: any remaining NaNs to 0
        X = X.fillna(0)
        
        # Store feature names
        if is_training:
            self.feature_names = available_cols
        
        # Scale features
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Check for any remaining NaNs
        if np.isnan(X_scaled).any():
            print("Warning: NaN values detected after preprocessing. Filling with 0.")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        
        # Target: Republican win (1) or loss (0)
        if 'margin' in df.columns:
            y = (df['margin'] > 0).astype(int).values
        else:
            y = None
        
        return X_scaled, y
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Train multiple models and return performance metrics
        """
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=5,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                random_state=42
            )
        }
        
        results = {}
        
        print("\nTraining models...")
        print("-" * 70)
        
        for name, model in models.items():
            print(f"\n{name}:")
            
            # Train model
            model.fit(X_train, y_train)
            self.models[name] = model
            
            # Training accuracy
            train_pred = model.predict(X_train)
            train_acc = accuracy_score(y_train, train_pred)
            
            print(f"  Training accuracy: {train_acc:.3f}")
            
            # Get feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                print(f"\n  Top 5 features:")
                for idx, row in feature_importance.head().iterrows():
                    print(f"    {row['feature']}: {row['importance']:.4f}")
            
            results[name] = {
                'model': model,
                'train_accuracy': train_acc
            }
        
        return results
    
    def predict_with_probability(self, X: np.ndarray, model_name: str = 'Random Forest') -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with probability estimates
        
        Returns:
            predictions: Class predictions (0 or 1)
            probabilities: Probability of Republican win
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.models[model_name]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]  # Probability of class 1 (Republican win)
        
        return predictions, probabilities
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, model_name: str = 'Random Forest') -> Dict:
        """Evaluate model performance on test set"""
        predictions, probabilities = self.predict_with_probability(X_test, model_name)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'predicted_class': predictions[0],
            'predicted_probability': probabilities[0],
            'actual_outcome': y_test[0] if y_test is not None else None
        }
        
        return metrics
    
    def create_prediction_report(self, test_df: pd.DataFrame, X_test: np.ndarray, y_test: np.ndarray):
        """Create a comprehensive prediction report"""
        print("\n" + "=" * 70)
        print("PREDICTION REPORT")
        print("=" * 70)
        
        for model_name in self.models.keys():
            print(f"\n{model_name}")
            print("-" * 70)
            
            predictions, probabilities = self.predict_with_probability(X_test, model_name)
            
            for idx in range(len(test_df)):
                state = test_df.iloc[idx]['state']
                year = test_df.iloc[idx]['election_year']
                
                pred_winner = 'Republican' if predictions[idx] == 1 else 'Democrat'
                rep_prob = probabilities[idx] * 100
                dem_prob = (1 - probabilities[idx]) * 100
                
                print(f"\n{year} - {state}:")
                print(f"  Predicted Winner: {pred_winner}")
                print(f"  Republican Probability: {rep_prob:.1f}%")
                print(f"  Democrat Probability: {dem_prob:.1f}%")
                
                if y_test is not None:
                    actual_winner = 'Republican' if y_test[idx] == 1 else 'Democrat'
                    correct = '✓' if predictions[idx] == y_test[idx] else '✗'
                    print(f"  Actual Winner: {actual_winner} {correct}")
                
                # Show key features influencing prediction
                if 'final_poll_margin' in test_df.columns:
                    poll_margin = test_df.iloc[idx]['final_poll_margin']
                    print(f"  Final Poll Margin: {poll_margin:+.1f}% (R-D)")
    
    def save_models(self):
        """Save trained models"""
        model_dir = os.path.join(self.data_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        
        for name, model in self.models.items():
            filename = name.lower().replace(' ', '_') + '.pkl'
            filepath = os.path.join(model_dir, filename)
            joblib.dump(model, filepath)
            print(f"Saved {name} to {filepath}")
        
        # Save scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}")
        
        # Save feature names
        feature_path = os.path.join(model_dir, 'features.txt')
        with open(feature_path, 'w') as f:
            f.write('\n'.join(self.feature_names))
        print(f"Saved feature names to {feature_path}")


def main():
    """Main modeling workflow"""
    print("=" * 70)
    print("ELECTION PREDICTION MODEL TRAINING")
    print("=" * 70)
    
    # Initialize predictor
    predictor = ElectionPredictor()
    
    # Load data
    print("\n1. Loading training data...")
    print("-" * 70)
    train_df, test_df = predictor.load_training_data()
    print(f"Training set: {len(train_df)} records")
    print(f"Test set: {len(test_df)} records")
    
    # Prepare features
    print("\n2. Preparing features...")
    print("-" * 70)
    X_train, y_train = predictor.prepare_features(train_df, is_training=True)
    X_test, y_test = predictor.prepare_features(test_df, is_training=False)
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Feature names: {predictor.feature_names}")
    
    # Train models
    print("\n3. Training multiple models...")
    print("-" * 70)
    results = predictor.train_models(X_train, y_train)
    
    # Make predictions
    print("\n4. Making predictions on 2024 election...")
    print("-" * 70)
    predictor.create_prediction_report(test_df, X_test, y_test)
    
    # Save models
    print("\n5. Saving trained models...")
    print("-" * 70)
    predictor.save_models()
    
    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Backtest the strategy on historical data")
    print("2. Optimize betting parameters (Kelly criterion, stake sizes)")
    print("3. Build automation system for live trading")


if __name__ == "__main__":
    main()
