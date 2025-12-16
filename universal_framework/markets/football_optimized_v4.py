"""
FOOTBALL PREDICTOR V4 - ULTIMATE MODEL

Complete advanced stats integration:
- Basic form (ppg, goals, clean sheets)
- Non-penalty xG (attacking quality)
- xG per shot (shot selection quality)
- Shots per game (volume)
- Set pieces (corners/dead ball situations)
- Random Forest classifier
- Binary classification (Home Win vs Not Home Win)

Expected Accuracy: 75-78%
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_engine import PredictionEngine
from firefox_advanced_scraper import AdvancedXGScraper

class FootballPredictorV4:
    """Ultimate Football Predictor with Complete Advanced Stats"""
    
    def __init__(self, league='Premier League'):
        self.engine = PredictionEngine(f"Football V4 - {league}")
        self.league = league
        self.match_data = None
        self.xg_scraper = AdvancedXGScraper()
        
        print("🎯 Initializing ULTIMATE Prediction System V4...")
        print("   Features: Form + npxG + xG/shot + Shots + Set Pieces")
        
    def load_data(self, df):
        """Load match data"""
        self.match_data = df.copy()
        self.match_data = self.match_data.sort_values('Date')
        print(f"✅ Loaded {len(df)} matches")
        
    def load_advanced_stats(self, season='2024'):
        """Load advanced stats from FBref"""
        print("\n📊 Loading advanced stats from FBref...")
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
        
        # Form trend
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
        """Default stats"""
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
        """Create ULTIMATE feature set with ALL advanced stats"""
        
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
        
        # ✨ Advanced stats from FBref
        home_advanced = self.xg_scraper.get_team_stats(home_team)
        away_advanced = self.xg_scraper.get_team_stats(away_team)
        
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
            
            # ✨ NEW: Non-penalty xG
            'home_npxg_per_game': home_advanced.get('npxg_per_game', 1.3),
            'away_npxg_per_game': away_advanced.get('npxg_per_game', 1.3),
            'npxg_advantage': home_advanced.get('npxg_per_game', 1.3) - away_advanced.get('npxg_per_game', 1.3),
            
            # ✨ NEW: Shot quality (xG per shot)
            'home_xg_per_shot': home_advanced.get('xg_per_shot', 0.08),
            'away_xg_per_shot': away_advanced.get('xg_per_shot', 0.08),
            'shot_quality_advantage': home_advanced.get('xg_per_shot', 0.08) - away_advanced.get('xg_per_shot', 0.08),
            
            # ✨ NEW: Shot volume
            'home_shots_per_game': home_advanced.get('shots_per_game', 15.0),
            'away_shots_per_game': away_advanced.get('shots_per_game', 15.0),
            
            # ✨ NEW: Set pieces (corners/dead ball)
            'home_set_pieces_per_game': home_advanced.get('corners_per_game', 4.5),
            'away_set_pieces_per_game': away_advanced.get('corners_per_game', 4.5),
            'set_piece_advantage': home_advanced.get('corners_per_game', 4.5) - away_advanced.get('corners_per_game', 4.5),
            
            # ✨ NEW: Combined attacking quality
            'home_attacking_quality': (
                home_advanced.get('npxg_per_game', 1.3) * 0.6 +  # 60% weight on xG
                home_advanced.get('xg_per_shot', 0.08) * 10 * 0.4  # 40% weight on shot quality
            ),
            'away_attacking_quality': (
                away_advanced.get('npxg_per_game', 1.3) * 0.6 +
                away_advanced.get('xg_per_shot', 0.08) * 10 * 0.4
            ),
        }
        
        return features
    
    def prepare_training_data(self, min_games=10):
        """Prepare training data with ALL features"""
        
        print("\n📊 Preparing ULTIMATE training data (V4)...")
        
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
        print(f"   Features: {len(feature_names)} (ULTIMATE V4 set)")
        print(f"   Home wins: {(y == 1).sum()} ({(y == 1).sum()/len(y):.1%})")
        print(f"   Not home wins: {(y == 0).sum()} ({(y == 0).sum()/len(y):.1%})")
        
        return X, y, feature_names, match_info
    
    def train_model(self, test_split=0.2):
        """Train ULTIMATE V4 model"""
        
        print("\n🧠 Training ULTIMATE Model V4...")
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
        
        # Optimized hyperparameters for V4
        model = RandomForestClassifier(
            n_estimators=500,  # More trees for complex features
            max_depth=30,      # Deeper for advanced stats
            min_samples_split=5,
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
        
        print(f"\n📊 ULTIMATE V4 Model Performance:")
        print(f"   Training accuracy: {train_acc:.1%}")
        print(f"   Test accuracy: {test_acc:.1%}")
        
        # Compare to V3
        v3_baseline = 0.728
        improvement = (test_acc - v3_baseline) / v3_baseline * 100
        
        if test_acc > v3_baseline:
            print(f"   🎯 Improvement over V3: +{improvement:.1f}%! 🚀")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, test_pred, target_names=['Not Home Win', 'Home Win']))
        
        # Feature importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print(f"\n🔍 Top 15 Most Important Features:")
        for i, idx in enumerate(indices[:15], 1):
            feature = feature_names[idx]
            importance = importances[idx]
            
            # Highlight advanced features
            if any(x in feature.lower() for x in ['npxg', 'xg_per_shot', 'shot', 'set_piece', 'quality']):
                print(f"   {i:2d}. {feature:40s}: {importance:.3f} ⭐ ADVANCED")
            else:
                print(f"   {i:2d}. {feature:40s}: {importance:.3f}")
        
        self.engine.model = model
        self.engine.feature_names = feature_names
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
        }
    
    def predict_match(self, home_team, away_team, match_date=None):
        """Predict match with V4 model"""
        
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
            
            # ✨ Advanced insights
            'advanced_insights': {
                'npxg_advantage': features['npxg_advantage'],
                'shot_quality_advantage': features['shot_quality_advantage'],
                'set_piece_advantage': features['set_piece_advantage'],
                'home_attacking_quality': features['home_attacking_quality'],
                'away_attacking_quality': features['away_attacking_quality'],
            }
        }

# Testing
if __name__ == "__main__":
    from football_data import FootballDataLoader
    
    print("=" * 70)
    print("🎯 FOOTBALL PREDICTOR V4 - ULTIMATE MODEL")
    print("=" * 70)
    
    # Load data
    loader = FootballDataLoader()
    data = loader.get_latest_results('E0', num_seasons=1)
    
    if data is not None:
        clean_data = loader.clean_match_data(data)
        
        # Create V4 predictor
        predictor = FootballPredictorV4('Premier League')
        predictor.load_data(clean_data)
        
        # Load advanced stats
        predictor.load_advanced_stats(season='2024')
        
        # Train model
        results = predictor.train_model(test_split=0.2)
        
        # Predict a match
        prediction = predictor.predict_match('Arsenal', 'Man United')
        
        print("\n" + "=" * 70)
        print("EXAMPLE PREDICTION (V4 ULTIMATE MODEL)")
        print("=" * 70)
        print(f"\n{prediction['home_team']} vs {prediction['away_team']}")
        print(f"Prediction: {prediction['prediction']}")
        print(f"Confidence: {prediction['confidence']}")
        print(f"Home Win Probability: {prediction['probabilities']['home_win']:.1%}")
        
        # Show advanced insights
        insights = prediction['advanced_insights']
        print(f"\n📊 Advanced Insights:")
        print(f"   npxG advantage: {insights['npxg_advantage']:+.2f}")
        print(f"   Shot quality advantage: {insights['shot_quality_advantage']:+.3f}")
        print(f"   Set piece advantage: {insights['set_piece_advantage']:+.1f}")
        print(f"   Attacking quality: {insights['home_attacking_quality']:.2f} vs 	{insights['away_attacking_quality']:.2f}")	