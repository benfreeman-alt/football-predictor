"""
AUTOMATED DATA COLLECTION

Scrapes new match results and updates training data weekly
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from bs4 import BeautifulSoup

class AutoDataCollector:
    """Automatically collect new Premier League match data"""
    
    def __init__(self, data_dir='data/football'):
        self.data_dir = data_dir
        self.current_season = self._get_current_season()
        os.makedirs(data_dir, exist_ok=True)
    
    def _get_current_season(self):
        """Determine current season based on date"""
        today = datetime.now()
        
        # Season runs Aug-May
        if today.month >= 8:
            return today.year
        else:
            return today.year - 1
    
    def collect_new_results(self, days_back=7):
        """
        Collect match results from last N days
        
        Args:
            days_back: How many days to look back
        
        Returns:
            DataFrame with new results
        """
        
        print(f"\n⚽ Collecting results from last {days_back} days...")
        
        # Try multiple sources
        results = None
        
        # Source 1: football-data.org
        results = self._fetch_from_football_data_org()
        
        # Source 2: API-Football (if available)
        if results is None or len(results) == 0:
            results = self._fetch_from_api_football()
        
        # Source 3: Web scraping (fallback)
        if results is None or len(results) == 0:
            results = self._scrape_from_web()
        
        if results is not None and len(results) > 0:
            print(f"   ✅ Collected {len(results)} new results")
            return results
        else:
            print(f"   ⚠️  No new results found")
            return pd.DataFrame()
    
    def _fetch_from_football_data_org(self):
        """Fetch finished matches from football-data.org"""
        
        try:
            api_token = os.getenv('FOOTBALL_DATA_TOKEN')
            if not api_token:
                return None
            
            print("   Trying football-data.org...")
            
            url = "https://api.football-data.org/v4/competitions/PL/matches"
            headers = {'X-Auth-Token': api_token}
            params = {'status': 'FINISHED'}
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                matches = data.get('matches', [])
                
                # Convert to our format
                results = []
                for match in matches:
                    try:
                        # Only include last 7 days
                        match_date = datetime.fromisoformat(match['utcDate'].replace('Z', '+00:00'))
                        if (datetime.now() - match_date).days > 7:
                            continue
                        
                        results.append({
                            'Date': match_date.strftime('%d/%m/%Y'),
                            'HomeTeam': self._standardize_team_name(match['homeTeam']['name']),
                            'AwayTeam': self._standardize_team_name(match['awayTeam']['name']),
                            'FTHG': match['score']['fullTime']['home'],
                            'FTAG': match['score']['fullTime']['away'],
                            'FTR': self._get_result(
                                match['score']['fullTime']['home'],
                                match['score']['fullTime']['away']
                            ),
                            'Season': self.current_season
                        })
                    except Exception as e:
                        continue
                
                if results:
                    return pd.DataFrame(results)
            
        except Exception as e:
            print(f"   ❌ football-data.org failed: {e}")
        
        return None
    
    def _fetch_from_api_football(self):
        """Fetch from API-Football"""
        # Similar implementation to football-data.org
        # Left as placeholder
        return None
    
    def _scrape_from_web(self):
        """Web scraping fallback"""
        # Scrape from premierleague.com or BBC Sport
        # Left as placeholder
        return None
    
    def _standardize_team_name(self, name):
        """Convert API team names to our format"""
        
        name_map = {
            'Manchester United FC': 'Man United',
            'Manchester City FC': 'Man City',
            'Tottenham Hotspur FC': 'Tottenham',
            'Newcastle United FC': 'Newcastle',
            'West Ham United FC': 'West Ham',
            'Leicester City FC': 'Leicester',
            'Brighton & Hove Albion FC': 'Brighton',
            'Wolverhampton Wanderers FC': 'Wolves',
            'Nottingham Forest FC': "Nott'm Forest",
            # Add more mappings as needed
        }
        
        return name_map.get(name, name.replace(' FC', '').strip())
    
    def _get_result(self, home_goals, away_goals):
        """Determine match result"""
        if home_goals > away_goals:
            return 'H'
        elif away_goals > home_goals:
            return 'A'
        else:
            return 'D'
    
    def append_to_dataset(self, new_results):
        """
        Append new results to existing dataset
        
        Args:
            new_results: DataFrame with new match results
        """
        
        if new_results.empty:
            print("   No new results to append")
            return
        
        # Load existing data
        season_file = os.path.join(self.data_dir, f'E0_{self.current_season}.csv')
        
        if os.path.exists(season_file):
            existing_data = pd.read_csv(season_file)
            
            # Merge with new results (avoid duplicates)
            combined = pd.concat([existing_data, new_results], ignore_index=True)
            combined = combined.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], keep='last')
            
            # Save
            combined.to_csv(season_file, index=False)
            print(f"   ✅ Updated {season_file}")
            print(f"   Total matches: {len(combined)}")
        else:
            # Create new file
            new_results.to_csv(season_file, index=False)
            print(f"   ✅ Created {season_file}")
            print(f"   Total matches: {len(new_results)}")
    
    def update_xg_data(self, new_results):
        """
        Update xG cache with new match data
        
        Args:
            new_results: DataFrame with new results
        """
        
        print("\n📊 Updating xG data...")
        
        try:
            from historical_xg_scraper import HistoricalXGScraper
            
            scraper = HistoricalXGScraper()
            
            # For each new match, scrape xG if available
            for _, match in new_results.iterrows():
                try:
                    xg_data = scraper.get_match_xg(
                        home_team=match['HomeTeam'],
                        away_team=match['AwayTeam'],
                        date=match['Date']
                    )
                    
                    if xg_data:
                        print(f"   ✅ xG: {match['HomeTeam']} vs {match['AwayTeam']}")
                
                except Exception as e:
                    print(f"   ⚠️  xG failed for {match['HomeTeam']} vs {match['AwayTeam']}")
            
            print("   ✅ xG data updated")
        
        except Exception as e:
            print(f"   ❌ xG update failed: {e}")

# Testing
if __name__ == "__main__":
    collector = AutoDataCollector()
    
    # Collect new results
    new_results = collector.collect_new_results(days_back=7)
    
    if not new_results.empty:
        print("\n" + "="*70)
        print("NEW RESULTS")
        print("="*70)
        print(new_results)
        
        # Append to dataset
        collector.append_to_dataset(new_results)
        
        # Update xG
        collector.update_xg_data(new_results)