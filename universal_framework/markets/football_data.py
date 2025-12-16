"""
FOOTBALL DATA LOADER

Downloads historical Premier League data from football-data.co.uk
"""

import pandas as pd
import requests
from datetime import datetime
import os

class FootballDataLoader:
    """Download and manage football data"""
    
    BASE_URL = "https://www.football-data.co.uk/mmz4281"
    
    def __init__(self):
        self.data_dir = 'data/football'
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_season(self, league_code, season):
        """Download a specific season"""
        
        url = f"{self.BASE_URL}/{season}/{league_code}.csv"
        filepath = os.path.join(self.data_dir, f"{league_code}_{season}.csv")
        
        if os.path.exists(filepath):
            print(f"  ✅ Downloaded {league_code} {season} (cached)")
            return pd.read_csv(filepath)
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"  ✅ Downloaded {league_code} {season}")
            return pd.read_csv(filepath)
        
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            return None
    
    def get_latest_results(self, league_code='E0', num_seasons=3):
        """
        Get recent seasons with FULL statistics
        
        Only 2017+ has shots/corners/advanced stats
        """
        
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        if current_month < 8:
            current_year -= 1
        
        seasons = []
        for i in range(num_seasons):
            year = current_year - i
            season_code = f"{str(year)[2:]}{str(year + 1)[2:]}"
            seasons.append(season_code)
        
        print(f"\n📥 Downloading {league_code} data...")
        print(f"Seasons: {seasons}")
        
        all_data = []
        
        for season in seasons:
            data = self.download_season(league_code, season)
            if data is not None:
                all_data.append(data)
        
        if not all_data:
            return None
        
        combined = pd.concat(all_data, ignore_index=True)
        
        print(f"\n✅ Total matches: {len(combined)}")
        
        # Check what columns we have
        has_shots = 'HS' in combined.columns
        has_corners = 'HC' in combined.columns
        
        if has_shots and has_corners:
            print(f"   📊 Advanced stats: AVAILABLE ✅")
        else:
            print(f"   ⚠️  Advanced stats: LIMITED")
            print(f"      Shots: {'✅' if has_shots else '❌'}")
            print(f"      Corners: {'✅' if has_corners else '❌'}")
        
        return combined
    
    def clean_match_data(self, df):
        """Clean and standardize match data"""
        
        required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"⚠️  Missing columns: {missing}")
            return None
        
        cols_to_keep = required_cols.copy()
        
        # Add advanced stats if available
        advanced_cols = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC']
        for col in advanced_cols:
            if col in df.columns:
                cols_to_keep.append(col)
        
        df_clean = df[cols_to_keep].copy()
        
        # Rename columns
        rename_dict = {
            'FTHG': 'home_goals',
            'FTAG': 'away_goals',
            'FTR': 'result',
        }
        
        if 'HS' in df.columns:
            rename_dict.update({
                'HS': 'home_shots',
                'AS': 'away_shots',
                'HST': 'home_shots_target',
                'AST': 'away_shots_target',
                'HC': 'home_corners',
                'AC': 'away_corners'
            })
        
        df_clean = df_clean.rename(columns=rename_dict)
        
        # Convert date
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], format='%d/%m/%Y', errors='coerce')
        
        # Remove missing data
        df_clean = df_clean.dropna(subset=['home_goals', 'away_goals', 'result'])
        
        # Fill missing advanced stats
        for col in ['home_shots', 'away_shots', 'home_shots_target', 
                    'away_shots_target', 'home_corners', 'away_corners']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(0)
        
        df_clean['goal_diff'] = df_clean['home_goals'] - df_clean['away_goals']
        
        return df_clean