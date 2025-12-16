"""
FOOTBALL OPTIMIZED V2 - Enhanced with Shot/Corner Analysis

New features:
- Shots per game
- Shots on target per game  
- Shot accuracy (%)
- Corners per game
- Corner efficiency

Using data you already have from football-data.co.uk!
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_engine import PredictionEngine

class FootballPredictorV2:
    """Enhanced Football Predictor with Shot & Corner Analysis"""
    
    def __init__(self, league='Premier League'):
        self.engine = PredictionEngine(f"Football V2 - {league}")
        self.league = league
        self.match_data = None
        
        print("🎯 Initializing ENHANCED Prediction System v2...")
        
    def load_data(self, df):
        """Load match data"""
        self.match_data = df.copy()
        self.match_data = self.match_data.sort_values('Date')
        
        # Check what advanced stats we have
        has_shots = 'home_shots' in df.columns
        has_corners = 'home_corners' in df.columns
        
        print(f"✅ Loaded {len(df)} matches")
        print(f"   📊 Shots data: {'✅ Available' if has_shots else '❌ Missing'}")
        print(f"   🎯 Corners data: {'✅ Available' if has_corners else '❌ Missing'}")
        
    def calculate_team_form_advanced(self, team_name, as_of_date, num_games=6, home_only=False, away_only=False):
        """
        Calculate team form WITH shot/corner analysis
        """
        
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
            return self.get_default_stats()
        
        # Basic stats
        wins = draws = losses = 0
        goals_for = goals_against = 0
        clean_sheets = 0
        points_list = []
        
        # NEW: Advanced stats
        total_shots = 0
        total_shots_on_target = 0
        total_corners = 0
        
        for _, match in team_matches.iterrows():
            if match['HomeTeam'] == team_name:
                # Team was home
                goals_for += match['home_goals']
                goals_against += match['away_goals']
                if match['away_goals'] == 0:
                    clean_sheets += 1
                
                # NEW: Collect shot/corner data
                total_shots += match.get('home_shots', 0)
                total_shots_on_target += match.get('home_shots_target', 0)
                total_corners += match.get('home_corners', 0)
                
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
                # Team was away
                goals_for += match['away_goals']
                goals_against += match['home_goals']
                if match['home_goals'] == 0:
                    clean_sheets += 1
                
                # NEW: Collect shot/corner data
                total_shots += match.get('away_shots', 0)
                total_shots_on_target += match.get('away_shots_target', 0)
                total_corners += match.get('away_corners', 0)
                
                if match['result'] == 'A':
                    wins += 1
                    points_list.append(3)
                elif match['result'] == 'D':
                    draws += 1
                    points_list.append(1)
                else:
                    losses += 1
                    points_list.append(0)
        
        games = len(team_matches)
        points = (wins * 3) + draws
        
        # Calculate form trend
        form_trend = 0
        if len(points_list) >= 4:
            recent_avg = np.mean(points_list[-3:])
            earlier_avg = np.mean(points_list[-6:-3] if len(points_list) >= 6 else points_list[:-3])
            form_trend = recent_avg - earlier_avg
        
        # Calculate shot efficiency
        shot_accuracy = (total_shots_on_target / total_shots * 100) if total_shots > 0 else 0
        goals_per_shot = (goals_for / total_shots) if total_shots > 0 else 0
        
        return {
            # Basic stats
            'games_played': games,
            'points': points,
            'ppg': points / games,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'goal_diff': goals_for - goals_against,
            'clean_sheets': clean_sheets,
            'recent_form_trend': form_trend,
            
            # NEW: Shot statistics
            'shots_per_game': total_shots / games,
            'shots_on_target_per_game': total_shots_on_target / games,
            'shot_accuracy': shot_accuracy,
            'goals_per_shot': goals_per_shot,
            
            # NEW: Corner statistics
            'corners_per_game': total_corners / games,
            'goals_per_corner': (goals_for / total_corners) if total_corners > 0 else 0,
        }
    
    def get_default_stats(self):
        """Default stats when no data available"""
        return {
            'games_played': 0, 'points': 0, 'ppg': 0.0, 'wins': 0, 'draws': 0, 'losses': 0,
            'goals_for': 0, 'goals_against': 0, 'goal_diff': 0, 'clean_sheets': 0,
            'recent_form_trend': 0, 'shots_per_game': 0, 'shots_on_target_per_game': 0,
            'shot_accuracy': 0, 'goals_per_shot': 0, 'corners_per_game': 0, 'goals_per_corner': 0
        }
    
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
        """Create ENHANCED features with shot/corner analysis"""
        
        if match_date is None:
            match_date = datetime.now()
        
        # Get advanced form stats
        home_form = self.calculate_team_form_advanced(home_team, match_date, num_games=6)
        away_form = self.calculate_team_form_advanced(away_team, match_date, num_games=6)
        
        # Home/away splits
        home_form_home = self.calculate_team_form_advanced(home_team, match_date, num_games=6, home_only=True)
        away_form_away = self.calculate_team_form_advanced(away_team, match_date, num_games=6, away_only=True)
        
        # H2H
        h2h = self.get_head_to_head(home_team, away_team)
        
        features = {
            # Basic form
            'home_ppg': home_form['ppg'],
            'away_ppg': away_form['ppg'],
            'form_diff': home_form['ppg'] - away_form['ppg'],
            
            # Home/away splits
            'home_ppg_at_home': home_form_home['ppg'],
            'away_ppg_away': away_form_away['ppg'],
            
            # Goals
            'home_goals_per_game': home_form['goals_for'] / max(home_form['games_played'], 1),
            'away_goals_per_game': away_form['goals_for'] / max(away_form['games_played'], 1),
            'home_conceded_per_game': home_form['goals_against'] / max(home_form['games_played'], 1),
            'away_conceded_per_game': away_form['goals_against'] / max(away_form['games_played'], 1),
            
            # Defense
            'home_clean_sheet_pct': home_form['clean_sheets'] / max(home_form['games_played'], 1),
            'away_clean_sheet_pct': away_form['clean_sheets'] / max(away_form['games_played'], 1),
            
            # Goal difference
            'home_gd': home_form['goal_diff'] / max(home_form['games_played'], 1),
            'away_gd': away_form['goal_diff'] / max(away_form['games_played'], 1),
            
            # Attack vs defense
            'home_attack_vs_away_defense': 
                (home_form['goals_for'] / max(home_form['games_played'], 1)) - 
                (away_form['goals_against'] / max(away_form['games_played'], 1)),
            
            # H2H
            'h2h_home_advantage': 0 if h2h is None else (h2h['home_wins'] - h2h['away_wins']) / h2h['matches'],
            
            # Form trends
            'home_form_trend': home_form['recent_form_trend'],
            'away_form_trend': away_form['recent_form_trend'],
            
            # ✨ NEW: Shot statistics
            'home_shots_per_game': home_form['shots_per_game'],
            'away_shots_per_game': away_form['shots_per_game'],
            'home_shots_on_target_per_game': home_form['shots_on_target_per_game'],
            'away_shots_on_target_per_game': away_form['shots_on_target_per_game'],
            'home_shot_accuracy': home_form['shot_accuracy'],
            'away_shot_accuracy': away_form['shot_accuracy'],
            'home_goals_per_shot': home_form['goals_per_shot'],
            'away_goals_per_shot': away_form['goals_per_shot'],
            
            # ✨ NEW: Corner statistics
            'home_corners_per_game': home_form['corners_per_game'],
            'away_corners_per_game': away_form['corners_per_game'],
            'home_goals_per_corner': home_form['goals_per_corner'],
            'away_goals_per_corner': away_form['goals_per_corner'],
        }
        
        return features
    
    def prepare_training_data(self, min_games=10):
        """Prepare training data with enhanced features"""
        
        print("\n📊 Preparing ENHANCED training data (v2)...")
        
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
            
            # Binary target
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
        print(f"   Features: {len(feature_names)} (Enhanced with shots/corners)")
        print(f"   Home wins: {(y == 1).sum()} ({(y == 1).sum()/len(y):.1%})")
        print(f"   Not home wins: {(y == 0).sum()} ({(y == 0).sum()/len(y):.1%})")
        
        return X, y, feature_names, match_info
    
    def train_model(self, test_split=0.2):
        """Train enhanced model"""
        
        print("\n🧠 Training ENHANCED Model v2...")
        print("=" * 70)
        
        X, y, feature_names, match_info = self.prepare_training_data()
        
        split_idx = int(len(X) * (1 - test_split))
        
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        print(f"\nTraining set: {len(X_train)} matches")
        print(f"Test set: {len(X_test)} matches")
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report
        
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"\n📊 Enhanced Model Performance:")
        print(f"   Training accuracy: {train_acc:.1%}")
        print(f"   Test accuracy: {test_acc:.1%}")
        
        # Compare to v1
        baseline = 0.728
        if test_acc > baseline:
            improvement = (test_acc - baseline) / baseline * 100
            print(f"   🎯 Improvement over v1: +{improvement:.1f}%!")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, test_pred, target_names=['Not Home Win', 'Home Win']))
        
        # Feature importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        print(f"\n🔍 Top 10 Most Important Features:")
        for i, idx in enumerate(indices, 1):
            print(f"   {i:2d}. {feature_names[idx]:35s}: {importances[idx]:.3f}")
        
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
        
        home_win_prob = self.engine.model.predict_proba(X)[0][1]
        
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
        }

# Testing
if __name__ == "__main__":
    from football_data import FootballDataLoader
    
    print("=" * 70)
    print("🎯 FOOTBALL PREDICTOR V2 - Enhanced with Shots/Corners")
    print("=" * 70)
    
    loader = FootballDataLoader()
    data = loader.get_latest_results('E0', num_seasons=5)
    
    if data is not None:
        clean_data = loader.clean_match_data(data)
        
        predictor = FootballPredictorV2('Premier League')
        predictor.load_data(clean_data)
        
        results = predictor.train_model(test_split=0.2)
        
        prediction = predictor.predict_match('Arsenal', 'Man United')
        
        print("\n" + "=" * 70)
        print("EXAMPLE PREDICTION (V2 MODEL)")
        print("=" * 70)
        print(f"\n{prediction['home_team']} vs {prediction['away_team']}")
        print(f"Prediction: {prediction['prediction']}")
        print(f"Confidence: {prediction['confidence']}")
        print(f"Home Win Probability: {prediction['probabilities']['home_win']:.1%}")