"""
FOOTBALL ODDS FETCHER

Fetches live Premier League odds from The-Odds-API
Compares multiple bookmakers to find best odds
"""

import requests
import os
from datetime import datetime, timedelta

class FootballOddsFetcher:
    """Fetch live Premier League betting odds"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        self.base_url = "https://api.the-odds-api.com/v4"
        
        # Premier League sport key
        self.sport = "soccer_epl"
        
        # Markets we care about
        self.markets = "h2h,spreads,totals"  # Home/Away/Draw, Handicap, Over/Under
    
    def get_upcoming_matches_odds(self):
        """Get odds for upcoming Premier League matches"""
        
        print("\n💰 Fetching live Premier League odds...")
        
        if not self.api_key:
            print("   ⚠️  No API key found (set ODDS_API_KEY)")
            return {}
        
        try:
            url = f"{self.base_url}/sports/{self.sport}/odds"
            
            params = {
                'apiKey': self.api_key,
                'regions': 'uk',  # UK bookmakers
                'markets': self.markets,
                'oddsFormat': 'decimal',
                'dateFormat': 'iso'
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Parse odds data
                odds_by_match = {}
                
                for match in data:
                    try:
                        home_team = self._clean_team_name(match['home_team'])
                        away_team = self._clean_team_name(match['away_team'])
                        
                        match_key = f"{home_team} vs {away_team}"
                        
                        # Extract best odds from all bookmakers
                        best_odds = self._extract_best_odds(match)
                        
                        odds_by_match[match_key] = {
                            'home_team': home_team,
                            'away_team': away_team,
                            'commence_time': match['commence_time'],
                            'odds': best_odds
                        }
                    
                    except Exception as e:
                        continue
                
                print(f"   ✅ Retrieved odds for {len(odds_by_match)} matches")
                print(f"   📊 API requests remaining: {response.headers.get('x-requests-remaining', 'unknown')}")
                
                return odds_by_match
            
            else:
                print(f"   ❌ API error: {response.status_code}")
                return {}
        
        except Exception as e:
            print(f"   ❌ Error fetching odds: {e}")
            return {}
    
    def _extract_best_odds(self, match):
        """Extract best odds from all bookmakers for a match"""
        
        best_odds = {
            'home_win': 0,
            'away_win': 0,
            'draw': 0,
            'double_chance_home_draw': 0,
            'double_chance_away_draw': 0,
            'double_chance_home_away': 0,
        }
        
        # Go through all bookmakers
        for bookmaker in match.get('bookmakers', []):
            
            for market in bookmaker.get('markets', []):
                
                # H2H market (Home/Draw/Away)
                if market['key'] == 'h2h':
                    for outcome in market['outcomes']:
                        if outcome['name'] == match['home_team']:
                            best_odds['home_win'] = max(best_odds['home_win'], outcome['price'])
                        elif outcome['name'] == match['away_team']:
                            best_odds['away_win'] = max(best_odds['away_win'], outcome['price'])
                        elif outcome['name'] == 'Draw':
                            best_odds['draw'] = max(best_odds['draw'], outcome['price'])
        
        # Calculate double chance odds (approximate)
        if best_odds['home_win'] and best_odds['draw']:
            # Home or Draw = 1/((1/home) + (1/draw))
            best_odds['double_chance_home_draw'] = 1 / ((1/best_odds['home_win']) + (1/best_odds['draw']))
        
        if best_odds['away_win'] and best_odds['draw']:
            # Away or Draw
            best_odds['double_chance_away_draw'] = 1 / ((1/best_odds['away_win']) + (1/best_odds['draw']))
        
        if best_odds['home_win'] and best_odds['away_win']:
            # Home or Away
            best_odds['double_chance_home_away'] = 1 / ((1/best_odds['home_win']) + (1/best_odds['away_win']))
        
        return best_odds
    
    def _clean_team_name(self, name):
        """Standardize team names to match our model"""
        
        name_map = {
            'Manchester United': 'Man United',
            'Manchester City': 'Man City',
            'Tottenham Hotspur': 'Tottenham',
            'Newcastle United': 'Newcastle',
            'West Ham United': 'West Ham',
            'Leicester City': 'Leicester',
            'Brighton and Hove Albion': 'Brighton',
            'Wolverhampton Wanderers': 'Wolves',
            'Nottingham Forest': "Nott'm Forest",
            'AFC Bournemouth': 'Bournemouth',
            'Ipswich Town': 'Ipswich',
        }
        
        return name_map.get(name, name)
    
    def get_odds_for_match(self, home_team, away_team, all_odds):
        """Get odds for a specific match"""
        
        match_key = f"{home_team} vs {away_team}"
        
        if match_key in all_odds:
            return all_odds[match_key]['odds']
        
        # Try reverse
        match_key_reverse = f"{away_team} vs {home_team}"
        if match_key_reverse in all_odds:
            odds = all_odds[match_key_reverse]['odds']
            # Swap home/away
            return {
                'home_win': odds['away_win'],
                'away_win': odds['home_win'],
                'draw': odds['draw'],
                'double_chance_home_draw': odds['double_chance_away_draw'],
                'double_chance_away_draw': odds['double_chance_home_draw'],
                'double_chance_home_away': odds['double_chance_home_away'],
            }
        
        return None

# Testing
if __name__ == "__main__":
    fetcher = FootballOddsFetcher()
    
    odds = fetcher.get_upcoming_matches_odds()
    
    if odds:
        print("\n" + "=" * 70)
        print("💰 PREMIER LEAGUE ODDS")
        print("=" * 70)
        
        for match_key, match_data in list(odds.items())[:5]:
            print(f"\n{match_key}")
            print(f"  Home Win: {match_data['odds']['home_win']:.2f}")
            print(f"  Draw: {match_data['odds']['draw']:.2f}")
            print(f"  Away Win: {match_data['odds']['away_win']:.2f}")
            print(f"  Double Chance (Away/Draw): {match_data['odds']['double_chance_away_draw']:.2f}")