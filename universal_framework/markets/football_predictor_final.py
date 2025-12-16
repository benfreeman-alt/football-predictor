"""
FINAL FOOTBALL PREDICTOR

V4 Historical Model (74.4% accuracy)
+ Post-prediction injury adjustment

This is the PRODUCTION model.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from football_optimized_v4_historical import FootballPredictorV4Historical
from injury_scraper import InjuryScraper

class FinalFootballPredictor:
    """Production Football Predictor with Injury Adjustment"""
    
    def __init__(self, league='Premier League'):
        self.predictor = FootballPredictorV4Historical(league)
        self.injury_scraper = InjuryScraper()
        
        print("🎯 Initializing FINAL Production System...")
        print("   V4 Model (74.4%) + Injury Adjustment")
    
    def load_data(self, df):
        """Load match data"""
        self.predictor.load_data(df)
    
    def load_historical_xg(self, seasons=['2022-2023', '2023-2024', '2024-2025']):
        """Load historical xG"""
        self.predictor.load_historical_xg(seasons)
    
    def load_injuries(self):
        """Load current injuries"""
        print("\n🏥 Loading injury data...")
        self.injury_scraper.scrape_premier_league_injuries()
    
    def train_model(self, test_split=0.2):
        """Train V4 model"""
        return self.predictor.train_model(test_split)
    
    def predict_match(self, home_team, away_team, match_date=None):
        """
        Predict match with injury adjustment
        
        Process:
        1. Get base prediction from V4 (74.4% accuracy)
        2. Adjust for current injuries
        3. Return adjusted prediction
        """
        
        # Get base prediction from V4
        base_prediction = self.predictor.predict_match(home_team, away_team, match_date)
        
        # Get injury impacts
        home_injury = self.injury_scraper.get_injury_impact(home_team)
        away_injury = self.injury_scraper.get_injury_impact(away_team)
        
        # Calculate injury adjustment
        # Each 0.1 injury impact = ~3% probability shift
        injury_adjustment = (away_injury - home_injury) * 0.3
        
        # Adjust probability
        base_prob = base_prediction['probabilities']['home_win']
        adjusted_prob = base_prob + injury_adjustment
        
        # Clamp between 0.1 and 0.9
        adjusted_prob = max(0.1, min(0.9, adjusted_prob))
        
        # Determine new prediction based on adjusted probability
        if adjusted_prob >= 0.70:
            prediction = 'Home Win'
            confidence = 'HIGH'
        elif adjusted_prob >= 0.60:
            prediction = 'Home Win'
            confidence = 'MEDIUM-HIGH'
        elif adjusted_prob >= 0.55:
            prediction = 'Home Win'
            confidence = 'MEDIUM'
        elif adjusted_prob <= 0.30:
            prediction = 'Away Win'
            confidence = 'HIGH'
        elif adjusted_prob <= 0.40:
            prediction = 'Away Win'
            confidence = 'MEDIUM-HIGH'
        elif adjusted_prob <= 0.45:
            prediction = 'Away Win'
            confidence = 'MEDIUM'
        else:
            prediction = 'SKIP'
            confidence = 'LOW'
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'home_win': adjusted_prob,
                'away_win': 1 - adjusted_prob,
                'base_home_win': base_prob,
            },
            'injury_adjustment': injury_adjustment,
            'injury_details': {
                'home_impact': home_injury,
                'away_impact': away_injury,
                'home_injuries': self.injury_scraper.get_injury_count(home_team),
                'away_injuries': self.injury_scraper.get_injury_count(away_team),
            }
        }

# Testing
if __name__ == "__main__":
    from football_data import FootballDataLoader
    
    print("=" * 70)
    print("🎯 FINAL PRODUCTION FOOTBALL PREDICTOR")
    print("=" * 70)
    
    # Load data
    loader = FootballDataLoader()
    data = loader.get_latest_results('E0', num_seasons=3)
    
    if data is not None:
        clean_data = loader.clean_match_data(data)
        
        # Create final predictor
        predictor = FinalFootballPredictor('Premier League')
        predictor.load_data(clean_data)
        predictor.load_historical_xg(['2022-2023', '2023-2024', '2024-2025'])
        predictor.load_injuries()
        
        # Train model (V4)
        results = predictor.train_model(test_split=0.2)
        
        # Make prediction
        prediction = predictor.predict_match('Arsenal', 'Man United')
        
        print("\n" + "=" * 70)
        print("EXAMPLE PREDICTION")
        print("=" * 70)
        print(f"\n{prediction['home_team']} vs {prediction['away_team']}")
        print(f"Prediction: {prediction['prediction']}")
        print(f"Confidence: {prediction['confidence']}")
        print(f"\nProbabilities:")
        print(f"  Base (V4): {prediction['probabilities']['base_home_win']:.1%}")
        print(f"  Adjusted: {prediction['probabilities']['home_win']:.1%}")
        print(f"  Injury adjustment: {prediction['injury_adjustment']:+.1%}")
        
        # Injury details
        inj = prediction['injury_details']
        print(f"\n🏥 Injury Analysis:")
        print(f"  {prediction['home_team']}: {inj['home_injuries']} injuries (impact: {inj['home_impact']:.2f})")
        print(f"  {prediction['away_team']}: {inj['away_injuries']} injuries (impact: {inj['away_impact']:.2f})")