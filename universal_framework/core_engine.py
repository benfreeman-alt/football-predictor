import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os

class PredictionEngine:
    """Universal prediction engine for any market"""
    
    def __init__(self, market_name):
        self.market_name = market_name
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.predictions = None
        
    def train_model(self, X_train, y_train, feature_names):
        """Train the prediction model"""
        
        self.feature_names = feature_names
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X_train_scaled, y_train)
        
        # Training accuracy
        train_pred = self.model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)
        
        return train_acc
    
    def predict(self, X_test):
        """Make predictions on new data"""
        
        if self.model is None:
            raise Exception("Model not trained yet!")
        
        X_test_scaled = self.scaler.transform(X_test)
        
        predictions = self.model.predict(X_test_scaled)
        probabilities = self.model.predict_proba(X_test_scaled)
        
        return predictions, probabilities
    
    def calculate_confidence(self, economic_signal, polling_signal=None, 
                            signal_strength=None):
        """
        Universal confidence calculator
        
        Can be adapted for any market by passing different signals
        """
        
        # No secondary signal (like polling)
        if polling_signal is None:
            if signal_strength == 'STRONG':
                return 'MEDIUM-HIGH', 0.75, 'Strong primary signal, no secondary'
            elif signal_strength == 'MODERATE':
                return 'MEDIUM', 0.5, 'Moderate primary signal, no secondary'
            else:
                return 'MEDIUM-LOW', 0.25, 'Weak primary signal, no secondary'
        
        # Have both signals - check agreement
        if economic_signal == polling_signal:
            return 'HIGH', 1.0, 'Both signals agree'
        else:
            return 'LOW', 0.0, 'DISAGREEMENT - SKIP!'
    
    def save_model(self, filepath):
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(self.model, f'{filepath}_model.pkl')
        joblib.dump(self.scaler, f'{filepath}_scaler.pkl')
        
        with open(f'{filepath}_features.txt', 'w') as f:
            f.write('\n'.join(self.feature_names))
    
    def load_model(self, filepath):
        """Load trained model"""
        self.model = joblib.load(f'{filepath}_model.pkl')
        self.scaler = joblib.load(f'{filepath}_scaler.pkl')
        
        with open(f'{filepath}_features.txt', 'r') as f:
            self.feature_names = f.read().strip().split('\n')

class BettingCalculator:
    """Universal betting calculator"""
    
    @staticmethod
    def calculate_edge(your_prob, market_odds):
        """Calculate your edge vs market"""
        market_implied_prob = 1 / market_odds
        edge = your_prob - market_implied_prob
        return edge
    
    @staticmethod
    def calculate_kelly(your_prob, market_odds, confidence_multiplier=1.0, 
                       conservative_fraction=0.25):
        """
        Calculate Kelly Criterion bet size
        
        conservative_fraction: Use 0.25 for Quarter Kelly (recommended)
        """
        b = market_odds - 1
        p = your_prob
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # Apply conservative fraction (Quarter Kelly)
        kelly *= conservative_fraction
        
        # Apply confidence multiplier
        kelly *= confidence_multiplier
        
        return kelly
    
    @staticmethod
    def should_bet(edge, confidence_level, min_edge_thresholds=None):
        """Determine if bet meets minimum edge requirements"""
        
        if min_edge_thresholds is None:
            min_edge_thresholds = {
                'HIGH': 0.10,
                'MEDIUM-HIGH': 0.10,
                'MEDIUM': 0.15,
                'MEDIUM-LOW': 0.20,
                'LOW': 1.0  # Never bet
            }
        
        min_edge = min_edge_thresholds.get(confidence_level, 0.15)
        return edge >= min_edge

class PerformanceTracker:
    """Track betting performance"""
    
    def __init__(self):
        self.bets = []
    
    def add_bet(self, bet_info):
        """Add a bet to tracking"""
        self.bets.append(bet_info)
    
    def calculate_roi(self):
        """Calculate ROI from completed bets"""
        completed = [b for b in self.bets if 'result' in b]
        
        if not completed:
            return None
        
        total_stake = sum(b['stake'] for b in completed)
        total_profit = sum(b.get('profit', 0) for b in completed)
        
        roi = (total_profit / total_stake) if total_stake > 0 else 0
        
        return roi
    
    def get_stats(self):
        """Get performance statistics"""
        completed = [b for b in self.bets if 'result' in b]
        
        if not completed:
            return None
        
        wins = sum(1 for b in completed if b['result'] == 'WIN')
        total = len(completed)
        
        stats = {
            'total_bets': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': wins / total,
            'roi': self.calculate_roi()
        }
        
        return stats