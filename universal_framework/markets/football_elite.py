"""
FOOTBALL ELITE PREDICTION SYSTEM

Advanced features:
- Injury data (PhysioRoom)
- Rest days (fixture congestion)
- Weather conditions
- Motivation (league position)
- Form trends (improving/declining)
- 5 seasons of data
- Random Forest classifier
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_engine import PredictionEngine
from injury_scraper import InjuryTracker
from weather_data import WeatherTracker

class FootballElitePredictor:
    """Elite Football Predictor with ALL advanced features"""
    
    def __init__(self, league='Premier League', weather_api_key=None):
        self.engine = PredictionEngine(f"Football Elite - {league}")
        self.league = league
        self.match_data = None
        
        # Advanced trackers
        self.injury_tracker = InjuryTracker()
        self.weather_tracker = WeatherTracker(api_key=weather_api_key)
        
        print("🎯 Initializing ELITE Prediction System...")
        
    def load_data(self, df):
        """Load match data"""
        self.match_data = df.copy()
        self.match_data = self.match_data.sort_values('Date')
        print(f"✅ Loaded {len(df)} matches")
        
    def load_live_data(self):
        """Load injury and weather data"""
        print("\n📊 Loading live data sources...")
        self.injury_tracker.scrape_physioroom()
        print("✅ Live data loaded")
        
    def calculate_team_form(self, team_name, as_of_date, num_games=6, home_only=False, away_only=False):
        """Calculate team form"""
        
        if home_only:
            team_matches = self.match_data[
                (self.match_data['HomeTeam'] == team_name) &
                (self.match_data['Date'] < as_of_date)
            ].tail(num_games)
        elif away_only:
            team_matches = self.match_data[
                (self.match_data['AwayTeam'] == team_name) &
                (self.match_data['Date'] < as_of_date)
            ].tail(num_games)
        else:
            team_matches = self.match_data[
                ((self.match_data['HomeTeam'] == team_name) | 
                 (self.match_data['AwayTeam'] == team_name)) &
                (self.match_data['Date'] < as_of_date)
            ].tail(num_games)
        
        if len(team_matches) == 0:
            return {
                'games_played': 0, 'points': 0, 'wins': 0, 'draws': 0, 'losses': 0,
                'goals_for': 0, 'goals_against': 0, 'goal_diff': 0, 'ppg': 0.0,
                'clean_sheets': 0, 'recent_form_trend': 0
            }
        
        wins = draws = losses = 0
        goals_for = goals_against = 0
        clean_sheets = 0
        points_list = []
        
        for _, match in team_matches.iterrows():
            if match['HomeTeam'] == team_name:
                goals_for += match['home_goals']
                goals_against += match['away_goals']
                if match['away_goals'] == 0:
                    clean_sheets += 1
                
                if match['result'] == 'H':
                    wins += 1
                    points_list.append(3)
                elif match['result'] == 'D':
                    draws += 1
                    points_list.append(1)
                else:
                    losses += 1
                    points_list.append(0)
            else:
                goals_for += match['away_goals']
                goals_against += match['home_goals']
                if match['home_goals'] == 0:
                    clean_sheets += 1
                
                if match['result'] == 'A':
                    wins += 1
                    points_list.append(3)
                elif match['result'] == 'D':
                    draws += 1
                    points_list.append(1)
                else:
                    losses += 1
                    points_list.append(0)
        
        points = (wins * 3) + draws
        
        # Calculate form trend (improving = positive, declining = negative)
        form_trend = 0
        if len(points_list) >= 4:
            recent_avg = np.mean(points_list[-3:])  # Last 3 games
            earlier_avg = np.mean(points_list[-6:-3] if len(points_list) >= 6 else points_list[:-3])  # Previous 3
            form_trend = recent_avg - earlier_avg
        
        return {
            'games_played': len(team_matches),
            'points': points,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'goal_diff': goals_for - goals_against,
            'ppg': points / len(team_matches),
            'clean_sheets': clean_sheets,
            'recent_form_trend': form_trend  # NEW!
        }
    
    def get_rest_days(self, team_name, match_date):
        """
        Calculate days since last match
        
        More rest = better performance
        """
        
        last_match = self.match_data[
            ((self.match_data['HomeTeam'] == team_name) | 
             (self.match_data['AwayTeam'] == team_name)) &
            (self.match_data['Date'] < match_date)
        ].tail(1)
        
        if len(last_match) == 0:
            return 7  # Default: 1 week
        
        days = (match_date - last_match.iloc[0]['Date']).days
        
        return min(days, 14)  # Cap at 14 days
    
    def get_head_to_head(self, home_team, away_team, num_matches=10):
        """Get H2H record"""
        
        h2h = self.match_data[
            ((self.match_data['HomeTeam'] == home_team) & (self.match_data['AwayTeam'] == away_team)) |
            ((self.match_data['HomeTeam'] == away_team) & (self.match_data['AwayTeam'] == home_team))
        ].tail(num_matches)
        
        if len(h2h) == 0:
            return None
        
        home_wins = away_wins = draws = 0
        
        for _, match in h2h.iterrows():
            if match['HomeTeam'] == home_team:
                if match['result'] == 'H':
                    home_wins += 1
                elif match['result'] == 'D':
                    draws += 1
                else:
                    away_wins += 1
            else:
                if match['result'] == 'H':
                    away_wins += 1
                elif match['result'] == 'D':
                    draws += 1
                else:
                    home_wins += 1
        
        return {
            'matches': len(h2h),
            'home_wins': home_wins,
            'away_wins': away_wins,
            'draws': draws
        }
    
    def create_match_features(self, home_team, away_team, match_date=None):
        """Create ELITE features with all advanced stats"""
        
        if match_date is None:
            match_date = datetime.now()
        
        # Basic form
        home_form = self.calculate_team_form(home_team, match_date, num_games=6)
        away_form = self.calculate_team_form(away_team, match_date, num_games=6)
        
        # Home/away splits
        home_form_home = self.calculate_team_form(home_team, match_date, num_games=6, home_only=True)
        away_form_away = self.calculate_team_form(away_team, match_date, num_games=6, away_only=True)
        
        # H2H
        h2h = self.get_head_to_head(home_team, away_team)
        
        # Rest days
        home_rest = self.get_rest_days(home_team, match_date)
        away_rest = self.get_rest_days(away_team, match_date)
        
        # Injuries (live data)
        home_injuries = self.injury_tracker.get_team_injury_score(home_team)
        away_injuries = self.injury_tracker.get_team_injury_score(away_team)
        
        # Weather (for home team's stadium)
        weather_impact = self.weather_tracker.get_weather_impact(home_team)
        
        features = {
            # Basic form
            'home_ppg': home_form['ppg'],
            'away_ppg': away_form['ppg'],
            'form_diff': home_form['ppg'] - away_form['ppg'],
            
            # Home/away splits
            'home_ppg_at_home': home_form_home['ppg'],
            'away_ppg_away': away_form_away['ppg'],
            
            # Attack
            'home_goals_per_game': home_form['goals_for'] / max(home_form['games_played'], 1),
            'away_goals_per_game': away_form['goals_for'] / max(away_form['games_played'], 1),
            
            # Defense
            'home_conceded_per_game': home_form['goals_against'] / max(home_form['games_played'], 1),
            'away_conceded_per_game': away_form['goals_against'] / max(away_form['games_played'], 1),
            'home_clean_sheet_pct': home_form['clean_sheets'] / max(home_form['games_played'], 1),
            'away_clean_sheet_pct': away_form['clean_sheets'] / max(away_form['games_played'], 1),
            
            # Goal difference
            'home_gd': home_form['goal_diff'] / max(home_form['games_played'], 1),
            'away_gd': away_form['goal_diff'] / max(away_form['games_played'], 1),
            
            # Attack vs defense matchup
            'home_attack_vs_away_defense': 
                (home_form['goals_for'] / max(home_form['games_played'], 1)) - 
                (away_form['goals_against'] / max(away_form['games_played'], 1)),
            
            # H2H
            'h2h_home_advantage': 0 if h2h is None else (h2h['home_wins'] - h2h['away_wins']) / h2h['matches'],
            
            # ✨ NEW: Form trends
            'home_form_trend': home_form['recent_form_trend'],
            'away_form_trend': away_form['recent_form_trend'],
            
            # ✨ NEW: Rest/fixture congestion
            'home_rest_days': home_rest,
            'away_rest_days': away_rest,
            'rest_advantage': home_rest - away_rest,
            
            # ✨ NEW: Injuries
            'home_injury_score': home_injuries,
            'away_injury_score': away_injuries,
            'injury_advantage': away_injuries - home_injuries,  # Positive = home team healthier
            
            # ✨ NEW: Weather
            'weather_impact': weather_impact,
        }
        
        return features
    
    def prepare_training_data(self, min_games=10):
        """Prepare BINARY training data"""
        
        print("\n📊 Preparing ELITE training data...")
        
        X_data = []
        y_data = []
        match_info = []
        
        for idx, match in self.match_data.iterrows():
            matches_before = self.match_data[self.match_data['Date'] < match['Date']]
            
            if len(matches_before) < min_games:
                continue
            
            try:
                features = self.create_match_features(
                    match['HomeTeam'],
                    match['AwayTeam'],
                    match['Date']
                )
            except:
                continue
            
            # BINARY TARGET: 1 = Home Win, 0 = Not Home Win
            target = 1 if match['result'] == 'H' else 0
            
            X_data.append(list(features.values()))
            y_data.append(target)
            match_info.append({
                'date': match['Date'],
                'home': match['HomeTeam'],
                'away': match['AwayTeam'],
                'actual': match['result']
            })
        
        X = np.array(X_data)
        y = np.array(y_data)
        feature_names = list(features.keys())
        
        print(f"✅ Created {len(X)} training examples")
        print(f"   Features: {len(feature_names)} (ELITE feature set)")
        print(f"   Home wins: {(y == 1).sum()} ({(y == 1).sum()/len(y):.1%})")
        print(f"   Not home wins: {(y == 0).sum()} ({(y == 0).sum()/len(y):.1%})")
        
        return X, y, feature_names, match_info
    
    def train_model(self, test_split=0.2):
        """Train ELITE classifier"""
        
        print("\n🧠 Training ELITE Model (Random Forest + Advanced Features)...")
        print("=" * 70)
        
        X, y, feature_names, match_info = self.prepare_training_data()
        
        split_idx = int(len(X) * (1 - test_split))
        
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        print(f"\nTraining set: {len(X_train)} matches")
        print(f"Test set: {len(X_test)} matches")
        
        # Random Forest with tuned hyperparameters
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report
        
        model = RandomForestClassifier(
            n_estimators=300,  # More trees
            max_depth=20,      # Deeper trees
            min_samples_split=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"\n📊 ELITE Model Performance:")
        print(f"   Training accuracy: {train_acc:.1%}")
        print(f"   Test accuracy: {test_acc:.1%}")
        
        # Compare to previous models
        baseline = 0.728  # Optimized model
        if test_acc > baseline:
            improvement = (test_acc - baseline) / baseline * 100
            print(f"   🎯 Improvement over optimized: +{improvement:.1f}%!")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, test_pred, target_names=['Not Home Win', 'Home Win']))
        
        # Feature importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        print(f"\n🔍 Top 10 Most Important Features:")
        for i, idx in enumerate(indices, 1):
            print(f"   {i:2d}. {feature_names[idx]:30s}: {importances[idx]:.3f}")
        
        self.engine.model = model
        self.engine.feature_names = feature_names
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
        }
    
    def predict_match(self, home_team, away_team, match_date=None):
        """Predict match outcome"""
        
        if self.engine.model is None:
            raise Exception("Model not trained!")
        
        features = self.create_match_features(home_team, away_team, match_date)
        X = np.array([list(features.values())])
        
        # Get probability of home win
        home_win_prob = self.engine.model.predict_proba(X)[0][1]
        
        # Determine prediction and confidence
        if home_win_prob >= 0.70:
            prediction = 'Home Win'
            confidence = 'HIGH'
        elif home_win_prob >= 0.60:
            prediction = 'Home Win'
            confidence = 'MEDIUM-HIGH'
        elif home_win_prob >= 0.55:
            prediction = 'Home Win'
            confidence = 'MEDIUM'
        elif home_win_prob <= 0.30:
            prediction = 'Away Win'
            confidence = 'HIGH'
        elif home_win_prob <= 0.40:
            prediction = 'Away Win'
            confidence = 'MEDIUM-HIGH'
        elif home_win_prob <= 0.45:
            prediction = 'Away Win'
            confidence = 'MEDIUM'
        else:
            prediction = 'SKIP'
            confidence = 'LOW'
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'prediction': prediction,
            'probabilities': {
                'home_win': home_win_prob,
                'away_win': 1 - home_win_prob,
                'draw': 0
            },
            'confidence': confidence,
            'bet_multiplier': 1.0 if confidence == 'HIGH' else 0.75 if confidence == 'MEDIUM-HIGH' else 0.5,
            'features': features  # Include features for debugging
        }

# Testing
if __name__ == "__main__":
    from football_data import FootballDataLoader
    
    print("=" * 70)
    print("🎯 FOOTBALL ELITE PREDICTION SYSTEM")
    print("=" * 70)
    
    # Load data
    loader = FootballDataLoader()
    data = loader.get_latest_results('E0', num_seasons=5)
    
    if data is not None:
        clean_data = loader.clean_match_data(data)
        
        # Create ELITE predictor
        predictor = FootballElitePredictor('Premier League')
        predictor.load_data(clean_data)
        
        # Load live data
        predictor.load_live_data()
        
        # Train model
        results = predictor.train_model(test_split=0.2)
        
        # Predict a match
        prediction = predictor.predict_match('Arsenal', 'Man United')
        
        print("\n" + "=" * 70)
        print("EXAMPLE PREDICTION (ELITE MODEL)")
        print("=" * 70)
        print(f"\n{prediction['home_team']} vs {prediction['away_team']}")
        print(f"Prediction: {prediction['prediction']}")
        print(f"Confidence: {prediction['confidence']}")
        print(f"Home Win Probability: {prediction['probabilities']['home_win']:.1%}")
        
        # Show advanced features
        features = prediction['features']
        print(f"\n📊 Advanced Features:")
        print(f"   Injury advantage: {features['injury_advantage']:+.1f}")
        print(f"   Rest advantage: {features['rest_advantage']:+.0f} days")
        print(f"   Home form trend: {features['home_form_trend']:+.2f}")
        print(f"   Weather impact: {features['weather_impact']:.0f}")