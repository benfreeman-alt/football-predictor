"""
ENHANCED FOOTBALL PREDICTION MODULE

Predict match outcomes using:
- Team form (recent results)
- Head-to-head record
- Goals scored/conceded
- Home advantage
- Advanced stats (shots, corners, possession)
- 10+ seasons of historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_engine import PredictionEngine, BettingCalculator

class FootballPredictor:
    """Football match outcome predictor - ENHANCED"""
    
    def __init__(self, league='Premier League'):
        self.engine = PredictionEngine(f"Football - {league}")
        self.league = league
        self.match_data = None
        self.team_stats = {}
        
    def load_data(self, df):
        """Load match data"""
        self.match_data = df.copy()
        self.match_data = self.match_data.sort_values('Date')
        
        print(f"✅ Loaded {len(df)} matches")
        
        # Check what stats we have
        has_shots = 'home_shots' in df.columns
        has_corners = 'home_corners' in df.columns
        
        if has_shots and has_corners:
            print(f"   📊 Advanced stats: AVAILABLE ✅")
        else:
            print(f"   📊 Advanced stats: Limited")
        
    def calculate_team_form(self, team_name, as_of_date, num_games=5, home_only=False, away_only=False):
        """Calculate team's recent form"""
        
        # Get team's recent matches before this date
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
        
        # Default return if no matches found
        if len(team_matches) == 0:
            return {
                'games_played': 0,
                'points': 0,
                'wins': 0,
                'draws': 0,
                'losses': 0,
                'goals_for': 0,
                'goals_against': 0,
                'goal_diff': 0,
                'ppg': 0.0
            }
        
        # Calculate stats
        wins = 0
        draws = 0
        losses = 0
        goals_for = 0
        goals_against = 0
        
        for _, match in team_matches.iterrows():
            if match['HomeTeam'] == team_name:
                # Team was home
                goals_for += match['home_goals']
                goals_against += match['away_goals']
                
                if match['result'] == 'H':
                    wins += 1
                elif match['result'] == 'D':
                    draws += 1
                else:
                    losses += 1
            else:
                # Team was away
                goals_for += match['away_goals']
                goals_against += match['home_goals']
                
                if match['result'] == 'A':
                    wins += 1
                elif match['result'] == 'D':
                    draws += 1
                else:
                    losses += 1
        
        points = (wins * 3) + (draws * 1)
        ppg = points / len(team_matches) if len(team_matches) > 0 else 0.0
        
        return {
            'games_played': len(team_matches),
            'points': points,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'goal_diff': goals_for - goals_against,
            'ppg': ppg
        }
    
    def calculate_advanced_stats(self, team_name, as_of_date, num_games=5):
        """
        Calculate advanced statistics for a team
        
        Returns: Dict with shots, corners, etc.
        """
        
        # Get team's recent matches
        team_matches = self.match_data[
            ((self.match_data['HomeTeam'] == team_name) | 
             (self.match_data['AwayTeam'] == team_name)) &
            (self.match_data['Date'] < as_of_date)
        ].tail(num_games)
        
        if len(team_matches) == 0:
            return {
                'shots_per_game': 0,
                'shots_on_target_per_game': 0,
                'corners_per_game': 0,
            }
        
        total_shots = 0
        total_shots_on_target = 0
        total_corners = 0
        
        for _, match in team_matches.iterrows():
            if match['HomeTeam'] == team_name:
                # Team was home
                total_shots += match.get('home_shots', 0)
                total_shots_on_target += match.get('home_shots_target', 0)
                total_corners += match.get('home_corners', 0)
            else:
                # Team was away
                total_shots += match.get('away_shots', 0)
                total_shots_on_target += match.get('away_shots_target', 0)
                total_corners += match.get('away_corners', 0)
        
        games = len(team_matches)
        
        return {
            'shots_per_game': total_shots / games,
            'shots_on_target_per_game': total_shots_on_target / games,
            'corners_per_game': total_corners / games,
        }
    
    def get_head_to_head(self, home_team, away_team, num_matches=10):
        """Get head-to-head record"""
        
        h2h = self.match_data[
            ((self.match_data['HomeTeam'] == home_team) & (self.match_data['AwayTeam'] == away_team)) |
            ((self.match_data['HomeTeam'] == away_team) & (self.match_data['AwayTeam'] == home_team))
        ].tail(num_matches)
        
        if len(h2h) == 0:
            return None
        
        home_wins = 0
        away_wins = 0
        draws = 0
        
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
        """Create features for a match prediction - ENHANCED"""
        
        if match_date is None:
            match_date = datetime.now()
        
        # Overall form (last 5 games)
        home_form = self.calculate_team_form(home_team, match_date, num_games=5)
        away_form = self.calculate_team_form(away_team, match_date, num_games=5)
        
        # Home/away specific form
        home_form_home = self.calculate_team_form(home_team, match_date, num_games=5, home_only=True)
        away_form_away = self.calculate_team_form(away_team, match_date, num_games=5, away_only=True)
        
        # Advanced stats (last 5 games)
        home_advanced = self.calculate_advanced_stats(home_team, match_date, num_games=5)
        away_advanced = self.calculate_advanced_stats(away_team, match_date, num_games=5)
        
        # Head to head
        h2h = self.get_head_to_head(home_team, away_team)
        
        # Create feature vector
        features = {
            # Basic form
            'home_ppg': home_form['ppg'],
            'away_ppg': away_form['ppg'],
            'ppg_diff': home_form['ppg'] - away_form['ppg'],
            
            # Home/away specific
            'home_ppg_home': home_form_home['ppg'],
            'away_ppg_away': away_form_away['ppg'],
            
            # Goals
            'home_goals_for': home_form['goals_for'] / max(home_form['games_played'], 1),
            'home_goals_against': home_form['goals_against'] / max(home_form['games_played'], 1),
            'away_goals_for': away_form['goals_for'] / max(away_form['games_played'], 1),
            'away_goals_against': away_form['goals_against'] / max(away_form['games_played'], 1),
            
            # Goal difference
            'home_goal_diff': home_form['goal_diff'] / max(home_form['games_played'], 1),
            'away_goal_diff': away_form['goal_diff'] / max(away_form['games_played'], 1),
            
            # NEW: Advanced stats
            'home_shots_per_game': home_advanced.get('shots_per_game', 0),
            'away_shots_per_game': away_advanced.get('shots_per_game', 0),
            'home_shots_on_target_per_game': home_advanced.get('shots_on_target_per_game', 0),
            'away_shots_on_target_per_game': away_advanced.get('shots_on_target_per_game', 0),
            'home_corners_per_game': home_advanced.get('corners_per_game', 0),
            'away_corners_per_game': away_advanced.get('corners_per_game', 0),
            
            # Attack vs Defense matchup
            'home_attack_vs_away_defense': (home_form['goals_for'] / max(home_form['games_played'], 1)) - 
                                            (away_form['goals_against'] / max(away_form['games_played'], 1)),
            'away_attack_vs_home_defense': (away_form['goals_for'] / max(away_form['games_played'], 1)) - 
                                            (home_form['goals_against'] / max(home_form['games_played'], 1)),
            
            # Head to head
            'h2h_home_advantage': 0 if h2h is None else (h2h['home_wins'] - h2h['away_wins']) / h2h['matches']
        }
        
        return features
    
    def prepare_training_data(self, min_games=10):
        """Prepare training data from historical matches"""
        
        print("\n📊 Preparing training data...")
        
        X_data = []
        y_data = []
        match_info = []
        
        for idx, match in self.match_data.iterrows():
            # Skip early season matches
            matches_before = self.match_data[self.match_data['Date'] < match['Date']]
            
            if len(matches_before) < min_games:
                continue
            
            # Create features
            try:
                features = self.create_match_features(
                    match['HomeTeam'],
                    match['AwayTeam'],
                    match['Date']
                )
            except Exception as e:
                continue
            
            # Create target
            if match['result'] == 'H':
                target = 2  # Home win
            elif match['result'] == 'D':
                target = 1  # Draw
            else:
                target = 0  # Away win
            
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
        print(f"   Features: {len(feature_names)}")
        print(f"   Home wins: {(y == 2).sum()} ({(y == 2).sum()/len(y):.1%})")
        print(f"   Draws: {(y == 1).sum()} ({(y == 1).sum()/len(y):.1%})")
        print(f"   Away wins: {(y == 0).sum()} ({(y == 0).sum()/len(y):.1%})")
        
        return X, y, feature_names, match_info
    
    def train_model(self, test_split=0.2):
        """Train the prediction model"""
        
        print("\n🧠 Training ENHANCED Football Prediction Model...")
        print("=" * 70)
        
        # Prepare data
        X, y, feature_names, match_info = self.prepare_training_data()
        
        # Train/test split (temporal)
        split_idx = int(len(X) * (1 - test_split))
        
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        print(f"\nTraining set: {len(X_train)} matches")
        print(f"Test set: {len(X_test)} matches")
        
        # Train model
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, classification_report
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial')
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"\n📊 Model Performance:")
        print(f"   Training accuracy: {train_acc:.1%}")
        print(f"   Test accuracy: {test_acc:.1%}")
        
        # Improvement vs basic model
        baseline = 0.588  # Your previous model
        improvement = (test_acc - baseline) / baseline * 100
        
        if improvement > 0:
            print(f"   🎯 Improvement: +{improvement:.1f}% vs basic model!")
        
        # Detailed report
        print(f"\n📋 Test Set Classification Report:")
        target_names = ['Away Win', 'Draw', 'Home Win']
        print(classification_report(y_test, test_pred, target_names=target_names))
        
        # Save model
        self.engine.model = model
        self.engine.scaler = scaler
        self.engine.feature_names = feature_names
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'test_predictions': test_pred,
            'test_actual': y_test,
            'test_matches': match_info[split_idx:]
        }
    
    def predict_match(self, home_team, away_team, match_date=None):
        """Predict a specific match"""
        
        if self.engine.model is None:
            raise Exception("Model not trained! Run train_model() first.")
        
        # Create features
        features = self.create_match_features(home_team, away_team, match_date)
        
        # Make prediction
        X = np.array([list(features.values())])
        X_scaled = self.engine.scaler.transform(X)
        
        prediction = self.engine.model.predict(X_scaled)[0]
        probabilities = self.engine.model.predict_proba(X_scaled)[0]
        
        # Map prediction
        outcome_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
        predicted_outcome = outcome_map[prediction]
        
        # Calculate confidence
        max_prob = probabilities.max()
        
        if max_prob >= 0.6:
            confidence = 'HIGH'
            multiplier = 1.0
        elif max_prob >= 0.5:
            confidence = 'MEDIUM-HIGH'
            multiplier = 0.75
        elif max_prob >= 0.45:
            confidence = 'MEDIUM'
            multiplier = 0.5
        else:
            confidence = 'LOW'
            multiplier = 0.25
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'prediction': predicted_outcome,
            'probabilities': {
                'home_win': probabilities[2],
                'draw': probabilities[1],
                'away_win': probabilities[0]
            },
            'confidence': confidence,
            'bet_multiplier': multiplier,
            'features': features
        }

# Testing
if __name__ == "__main__":
    from football_data import FootballDataLoader
    
    # Load MORE data (10 seasons)
    loader = FootballDataLoader()
    data = loader.get_latest_results('E0', num_seasons=10)  # ← 10 seasons!
    
    if data is not None:
        clean_data = loader.clean_match_data(data)
        
        # Create predictor
        predictor = FootballPredictor('Premier League')
        predictor.load_data(clean_data)
        
        # Train model
        results = predictor.train_model(test_split=0.2)
        
        # Predict a match
        prediction = predictor.predict_match('Arsenal', 'Man United')
        
        print("\n" + "=" * 70)
        print("EXAMPLE PREDICTION (ENHANCED MODEL)")
        print("=" * 70)
        print(f"\n{prediction['home_team']} vs {prediction['away_team']}")
        print(f"\nPrediction: {prediction['prediction']}")
        print(f"Confidence: {prediction['confidence']}")
        print(f"\nProbabilities:")
        print(f"  Home Win: {prediction['probabilities']['home_win']:.1%}")
        print(f"  Draw:     {prediction['probabilities']['draw']:.1%}")
        print(f"  Away Win: {prediction['probabilities']['away_win']:.1%}")