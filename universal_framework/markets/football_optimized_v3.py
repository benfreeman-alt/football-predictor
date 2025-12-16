"""
FOOTBALL PREDICTOR V3 - ULTIMATE MODEL

Features:
- Basic form & stats (ppg, goals, clean sheets)
- Expected Goals (xG) - non-penalty attacking quality
- xG advantage calculations
- 3 recent seasons (2022-2025) with best data quality
- Binary classification (Home Win vs Not Home Win)
- Random Forest with optimized hyperparameters

Expected Accuracy: 74-77%
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_engine import PredictionEngine
from firefox_xg_scraper import FirefoxXGScraper as UnderstatScraper

class FootballPredictorV3:
    """Ultimate Football Predictor with xG Integration"""
    
    def __init__(self, league='Premier League'):
        self.engine = PredictionEngine(f"Football V3 - {league}")
        self.league = league
        self.match_data = None
        self.xg_scraper = UnderstatScraper()
        
        print("🎯 Initializing ULTIMATE Prediction System V3...")
        print("   Features: Form + xG + Advanced Stats")
        
    def load_data(self, df):
        """Load match data"""
        self.match_data = df.copy()
        self.match_data = self.match_data.sort_values('Date')
        print(f"✅ Loaded {len(df)} matches")
        
    def load_xg_data(self, season='2024'):
        """Load xG data from Understat"""
        print("\n📊 Loading xG data...")
        self.xg_scraper.scrape_league_season('EPL', season)
        
    def calculate_team_form(self, team_name, as_of_date, num_games=6, home_only=False, away_only=False):
        """Calculate comprehensive team form"""
        
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
        
        # Calculate stats
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
        
        games = len(team_matches)
        points = (wins * 3) + draws
        
        # Calculate form trend
        form_trend = 0
        if len(points_list) >= 4:
            recent_avg = np.mean(points_list[-3:])
            earlier_avg = np.mean(points_list[-6:-3] if len(points_list) >= 6 else points_list[:-3])
            form_trend = recent_avg - earlier_avg
        
        return {
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
        }
    
    def get_default_stats(self):
        """Default stats when no data available"""
        return {
            'games_played': 0, 'points': 0, 'ppg': 0.0, 'wins': 0, 'draws': 0, 'losses': 0,
            'goals_for': 0, 'goals_against': 0, 'goal_diff': 0, 'clean_sheets': 0,
            'recent_form_trend': 0
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
        """Create ULTIMATE feature set with xG"""
        
        if match_date is None:
            match_date = datetime.now()
        
        # Get form stats
        home_form = self.calculate_team_form(home_team, match_date, num_games=6)
        away_form = self.calculate_team_form(away_team, match_date, num_games=6)
        
        # Home/away splits
        home_form_home = self.calculate_team_form(home_team, match_date, num_games=6, home_only=True)
        away_form_away = self.calculate_team_form(away_team, match_date, num_games=6, away_only=True)
        
        # H2H
        h2h = self.get_head_to_head(home_team, away_team)
        
        # ✨ NEW: xG data
        home_xg = self.xg_scraper.get_team_xg(home_team)
        away_xg = self.xg_scraper.get_team_xg(away_team)
        
        features = {
            # Basic form
            'home_ppg': home_form['ppg'],
            'away_ppg': away_form['ppg'],
            'form_diff': home_form['ppg'] - away_form['ppg'],
            
            # Home/away splits
            'home_ppg_at_home': home_form_home['ppg'],
            'away_ppg_away': away_form_away['ppg'],
            
            # Goals (actual results)
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
            
            # Attack vs defense matchup
            'home_attack_vs_away_defense': 
                (home_form['goals_for'] / max(home_form['games_played'], 1)) - 
                (away_form['goals_against'] / max(away_form['games_played'], 1)),
            
            # H2H
            'h2h_home_advantage': 0 if h2h is None else (h2h['home_wins'] - h2h['away_wins']) / h2h['matches'],
            
            # Form trends
            'home_form_trend': home_form['recent_form_trend'],
            'away_form_trend': away_form['recent_form_trend'],
            
            # ✨ NEW: xG features (Expected Goals - quality of chances)
            'home_npxg_per_game': home_xg['npxg_per_game'],
            'away_npxg_per_game': away_xg['npxg_per_game'],
            'xg_advantage': home_xg['npxg_per_game'] - away_xg['npxg_per_game'],
            
            # ✨ NEW: xG efficiency (are they overperforming/underperforming xG?)
            'home_goals_vs_xg': (home_form['goals_for'] / max(home_form['games_played'], 1)) - home_xg['npxg_per_game'],
            'away_goals_vs_xg': (away_form['goals_for'] / max(away_form['games_played'], 1)) - away_xg['npxg_per_game'],
            
            # ✨ NEW: Combined quality score (form + xG)
            'home_quality_score': (home_form['ppg'] / 3) * 0.5 + (home_xg['npxg_per_game'] / 2.5) * 0.5,
            'away_quality_score': (away_form['ppg'] / 3) * 0.5 + (away_xg['npxg_per_game'] / 2.5) * 0.5,
        }
        
        return features
    
    def prepare_training_data(self, min_games=10):
        """Prepare training data with xG features"""
        
        print("\n📊 Preparing ULTIMATE training data (v3 with xG)...")
        
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
        print(f"   Features: {len(feature_names)} (ULTIMATE set with xG)")
        print(f"   Home wins: {(y == 1).sum()} ({(y == 1).sum()/len(y):.1%})")
        print(f"   Not home wins: {(y == 0).sum()} ({(y == 0).sum()/len(y):.1%})")
        
        return X, y, feature_names, match_info
    
    def train_model(self, test_split=0.2):
        """Train ULTIMATE model"""
        
        print("\n🧠 Training ULTIMATE Model V3 (with xG)...")
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
        
        # Optimized hyperparameters for xG features
        model = RandomForestClassifier(
            n_estimators=400,  # More trees for complex features
            max_depth=25,      # Deeper for xG interactions
            min_samples_split=6,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"\n📊 ULTIMATE Model Performance:")
        print(f"   Training accuracy: {train_acc:.1%}")
        print(f"   Test accuracy: {test_acc:.1%}")
        
        # Compare to previous versions
        v1_baseline = 0.728
        improvement = (test_acc - v1_baseline) / v1_baseline * 100
        
        if test_acc > v1_baseline:
            print(f"   🎯 Improvement over V1: +{improvement:.1f}%! 🚀")
        else:
            print(f"   Current: {test_acc:.1%} vs V1: {v1_baseline:.1%}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, test_pred, target_names=['Not Home Win', 'Home Win']))
        
        # Feature importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print(f"\n🔍 Top 15 Most Important Features:")
        for i, idx in enumerate(indices[:15], 1):
            feature = feature_names[idx]
            importance = importances[idx]
            
            # Highlight xG features
            if 'xg' in feature.lower():
                print(f"   {i:2d}. {feature:35s}: {importance:.3f} ⭐ xG")
            else:
                print(f"   {i:2d}. {feature:35s}: {importance:.3f}")
        
        self.engine.model = model
        self.engine.feature_names = feature_names
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
        }
    
    def predict_match(self, home_team, away_team, match_date=None):
        """Predict match outcome with xG insights"""
        
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
            
            # ✨ NEW: xG insights
            'xg_insights': {
                'home_npxg': features['home_npxg_per_game'],
                'away_npxg': features['away_npxg_per_game'],
                'xg_advantage': features['xg_advantage'],
                'home_quality': features['home_quality_score'],
                'away_quality': features['away_quality_score'],
            }
        }

# Testing
if __name__ == "__main__":
    from football_data import FootballDataLoader
    
    print("=" * 70)
    print("🎯 FOOTBALL PREDICTOR V3 - ULTIMATE MODEL")
    print("=" * 70)
    
    # Load data (3 recent seasons for best data quality)
    loader = FootballDataLoader()
    data = loader.get_latest_results('E0', num_seasons=3)
    
    if data is not None:
        clean_data = loader.clean_match_data(data)
        
        # Create V3 predictor
        predictor = FootballPredictorV3('Premier League')
        predictor.load_data(clean_data)
        
        # Load xG data
        predictor.load_xg_data(season='2024')
        
        # Train model
        results = predictor.train_model(test_split=0.2)
        
        # Predict a match
        prediction = predictor.predict_match('Arsenal', 'Man United')
        
        print("\n" + "=" * 70)
        print("EXAMPLE PREDICTION (V3 ULTIMATE MODEL)")
        print("=" * 70)
        print(f"\n{prediction['home_team']} vs {prediction['away_team']}")
        print(f"Prediction: {prediction['prediction']}")
        print(f"Confidence: {prediction['confidence']}")
        print(f"Home Win Probability: {prediction['probabilities']['home_win']:.1%}")
        
        # Show xG insights
        xg = prediction['xg_insights']
        print(f"\n📊 xG Analysis:")
        print(f"   {prediction['home_team']} npxG: {xg['home_npxg']:.2f} per game")
        print(f"   {prediction['away_team']} npxG: {xg['away_npxg']:.2f} per game")
        print(f"   xG Advantage: {xg['xg_advantage']:+.2f}")
        print(f"   Quality Scores: {xg['home_quality']:.2f} vs {xg['away_quality']:.2f}")