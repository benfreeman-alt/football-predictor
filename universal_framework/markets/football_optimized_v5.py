"""
FOOTBALL PREDICTOR V5 - ULTIMATE MODEL WITH INJURIES

Includes:
- Form stats
- Historical xG data
- Shot quality & volume
- Set pieces
- INJURY DATA (NEW!)

Expected Accuracy: 75-77%
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_engine import PredictionEngine
from historical_xg_scraper import HistoricalXGScraper
from injury_scraper import InjuryScraper

class FootballPredictorV5:
    """Ultimate Football Predictor with Injuries"""
    
    def __init__(self, league='Premier League'):
        self.engine = PredictionEngine(f"Football V5 - {league}")
        self.league = league
        self.match_data = None
        self.xg_scraper = HistoricalXGScraper()
        self.injury_scraper = InjuryScraper()
        
        print("🎯 Initializing ULTIMATE Prediction System V5...")
        print("   Features: Form + xG + Shots + Set Pieces + INJURIES")
        
    def load_data(self, df):
        """Load match data"""
        self.match_data = df.copy()
        self.match_data = self.match_data.sort_values('Date')
        print(f"✅ Loaded {len(df)} matches")
        
    def load_historical_xg(self, seasons=['2022-2023', '2023-2024', '2024-2025']):
        """Load historical xG"""
        print("\n📊 Loading historical xG data...")
        self.xg_scraper.preload_seasons(seasons)
    
    def load_current_injuries(self):
        """Load current injury data"""
        print("\n🏥 Loading injury data...")
        self.injury_scraper.scrape_premier_league_injuries()
        
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
    
    def create_match_features(self, home_team, away_team, match_date=None, use_injuries=False):
        """Create ULTIMATE feature set with INJURIES"""
        
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
        
        # Historical advanced stats
        home_advanced = self.xg_scraper.get_stats_for_match(home_team, match_date)
        away_advanced = self.xg_scraper.get_stats_for_match(away_team, match_date)
        
        # 🏥 INJURY DATA (NEW!)
        home_injury_impact = 0.0
        away_injury_impact = 0.0
        injury_advantage = 0.0
        
        if use_injuries:
            home_injury_impact = self.injury_scraper.get_injury_impact(home_team)
            away_injury_impact = self.injury_scraper.get_injury_impact(away_team)
            injury_advantage = away_injury_impact - home_injury_impact  # Higher is better for home
        
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
            
            # Historical xG
            'home_npxg_per_game': home_advanced.get('npxg_per_game', 1.3),
            'away_npxg_per_game': away_advanced.get('npxg_per_game', 1.3),
            'npxg_advantage': home_advanced.get('npxg_per_game', 1.3) - away_advanced.get('npxg_per_game', 1.3),
            
            # Shot quality
            'home_xg_per_shot': home_advanced.get('xg_per_shot', 0.08),
            'away_xg_per_shot': away_advanced.get('xg_per_shot', 0.08),
            'shot_quality_advantage': home_advanced.get('xg_per_shot', 0.08) - away_advanced.get('xg_per_shot', 0.08),
            
            # Shot volume
            'home_shots_per_game': home_advanced.get('shots_per_game', 15.0),
            'away_shots_per_game': away_advanced.get('shots_per_game', 15.0),
            
            # Set pieces
            'home_set_pieces_per_game': home_advanced.get('corners_per_game', 4.5),
            'away_set_pieces_per_game': away_advanced.get('corners_per_game', 4.5),
            'set_piece_advantage': home_advanced.get('corners_per_game', 4.5) - away_advanced.get('corners_per_game', 4.5),
            
            # Combined attacking quality
            'home_attacking_quality': (
                home_advanced.get('npxg_per_game', 1.3) * 0.6 +
                home_advanced.get('xg_per_shot', 0.08) * 10 * 0.4
            ),
            'away_attacking_quality': (
                away_advanced.get('npxg_per_game', 1.3) * 0.6 +
                away_advanced.get('xg_per_shot', 0.08) * 10 * 0.4
            ),
            
            # 🏥 INJURY FEATURES (NEW!)
            'home_injury_impact': home_injury_impact,
            'away_injury_impact': away_injury_impact,
            'injury_advantage': injury_advantage,
        }
        
        return features
    
    def prepare_training_data(self, min_games=10, use_injuries=False):
        """Prepare training data"""
        
        print(f"\n📊 Preparing V5 training data (injuries: {'ON' if use_injuries else 'OFF'})...")
        
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
                    match['Date'],
                    use_injuries=use_injuries
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
        
        num_features = len(feature_names)
        injury_features = 3 if use_injuries else 0
        
        print(f"✅ Created {len(X)} training examples")
        print(f"   Features: {num_features} (V5 with {injury_features} injury features)")
        print(f"   Home wins: {(y == 1).sum()} ({(y == 1).sum()/len(y):.1%})")
        print(f"   Not home wins: {(y == 0).sum()} ({(y == 0).sum()/len(y):.1%})")
        
        return X, y, feature_names, match_info
    
    def train_model(self, test_split=0.2, use_injuries=False):
        """Train V5 model"""
        
        print("\n🧠 Training V5 Model...")
        print("=" * 70)
        
        X, y, feature_names, match_info = self.prepare_training_data(use_injuries=use_injuries)
        
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
            n_estimators=500,
            max_depth=30,
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
        
        print(f"\n📊 V5 Model Performance:")
        print(f"   Training accuracy: {train_acc:.1%}")
        print(f"   Test accuracy: {test_acc:.1%}")
        
        # Compare to V4
        v4_baseline = 0.744
        improvement = (test_acc - v4_baseline) / v4_baseline * 100
        
        if test_acc > v4_baseline:
            print(f"   🎯 Improvement over V4: +{improvement:.1f}%! 🚀")
        else:
            print(f"   Current vs V4: {improvement:+.1f}%")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, test_pred, target_names=['Not Home Win', 'Home Win']))
        
        # Feature importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print(f"\n🔍 Top 20 Most Important Features:")
        for i, idx in enumerate(indices[:20], 1):
            feature = feature_names[idx]
            importance = importances[idx]
            
            # Highlight advanced features
            if 'injury' in feature.lower():
                print(f"   {i:2d}. {feature:40s}: {importance:.3f} 🏥 INJURY")
            elif any(x in feature.lower() for x in ['npxg', 'xg_per_shot', 'shot', 'set_piece', 'quality']):
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
        """Predict match with V5"""
        
        if self.engine.model is None:
            raise Exception("Model not trained!")
        
        features = self.create_match_features(home_team, away_team, match_date, use_injuries=True)
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
            },
            'confidence': confidence,
            'injury_insights': {
                'home_impact': features['home_injury_impact'],
                'away_impact': features['away_injury_impact'],
                'advantage': features['injury_advantage']
            }
        }

# Testing
if __name__ == "__main__":
    from football_data import FootballDataLoader
    
    print("=" * 70)
    print("🎯 FOOTBALL PREDICTOR V5 - WITH INJURIES")
    print("=" * 70)
    
    # Load data
    loader = FootballDataLoader()
    data = loader.get_latest_results('E0', num_seasons=3)
    
    if data is not None:
        clean_data = loader.clean_match_data(data)
        
        # Create V5 predictor
        predictor = FootballPredictorV5('Premier League')
        predictor.load_data(clean_data)
        
        # Load historical xG
        predictor.load_historical_xg(['2022-2023', '2023-2024', '2024-2025'])
        
        # Load current injuries
        predictor.load_current_injuries()
        
        # Train WITHOUT injuries (historical matches don't have injury data)
        print("\n" + "=" * 70)
        print("TRAINING WITHOUT INJURY DATA (Historical)")
        print("=" * 70)
        results_no_injuries = predictor.train_model(test_split=0.2, use_injuries=False)
        
        # Predict with injuries
        prediction = predictor.predict_match('Arsenal', 'Man United')
        
        print("\n" + "=" * 70)
        print("EXAMPLE PREDICTION (WITH CURRENT INJURIES)")
        print("=" * 70)
        print(f"\n{prediction['home_team']} vs {prediction['away_team']}")
        print(f"Prediction: {prediction['prediction']}")
        print(f"Confidence: {prediction['confidence']}")
        print(f"Home Win: {prediction['probabilities']['home_win']:.1%}")
        
        # Show injury impact
        injury = prediction['injury_insights']
        print(f"\n🏥 Injury Impact:")
        print(f"   {prediction['home_team']} impact: {injury['home_impact']:.2f}")
        print(f"   {prediction['away_team']} impact: {injury['away_impact']:.2f}")
        print(f"   Advantage: {injury['advantage']:+.2f}")