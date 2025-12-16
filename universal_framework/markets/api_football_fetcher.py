"""
API-FOOTBALL FIXTURE FETCHER

Automatically gets upcoming Premier League fixtures from API-Football
Free tier: 100 requests/day (more than enough!)

Sign up: https://www.api-football.com/
Get your API key from dashboard
"""

import requests
import json
import os
from datetime import datetime, timedelta

class APIFootballFetcher:
    """Fetch fixtures from API-Football"""
    
    def __init__(self, api_key=None, cache_dir='data/fixture_cache'):
        self.api_key = api_key or os.getenv('API_FOOTBALL_KEY')
        self.base_url = "https://v3.football.api-sports.io"
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Premier League ID
        self.premier_league_id = 39
        # Current ACTIVE Premier League season is 2025-2026
        # (Season runs Aug 2025 - May 2026)
        self.season = 2025
    
    def get_upcoming_fixtures(self, days_ahead=14):
        """
        Get upcoming Premier League fixtures
        
        Args:
            days_ahead: Number of days to look ahead (default 14)
        
        Returns:
            List of fixtures
        """
        
        # Check cache first (valid for 12 hours)
        cached = self._load_from_cache()
        if cached:
            print("   📂 Using cached fixtures")
            return cached
        
        if not self.api_key:
            print("   ⚠️  No API key found, using manual fixtures")
            return self._get_manual_fixtures()
        
        print(f"\n⚽ Fetching fixtures from API-Football...")
        
        try:
            # Get current date range
            today = datetime.now().date()
            end_date = today + timedelta(days=days_ahead)
            
            # API endpoint
            url = f"{self.base_url}/fixtures"
            
            headers = {
                'x-rapidapi-host': 'v3.football.api-sports.io',
                'x-rapidapi-key': self.api_key
            }
            
            params = {
                'league': self.premier_league_id,
                'season': self.season,
                'from': today.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if data['response']:
                    fixtures = self._parse_fixtures(data['response'])
                    
                    # Cache for next time
                    self._save_to_cache(fixtures)
                    
                    print(f"   ✅ Retrieved {len(fixtures)} upcoming fixtures")
                    return fixtures
                else:
                    print("   ⚠️  No fixtures found in API response")
                    return self._get_manual_fixtures()
            else:
                print(f"   ⚠️  API error: {response.status_code}")
                return self._get_manual_fixtures()
        
        except Exception as e:
            print(f"   ⚠️  Error fetching fixtures: {e}")
            return self._get_manual_fixtures()
    
    def _parse_fixtures(self, fixtures_data):
        """Parse API response into our format"""
        
        parsed = []
        
        for fixture in fixtures_data:
            try:
                # Extract match details
                home_team = fixture['teams']['home']['name']
                away_team = fixture['teams']['away']['name']
                
                # Parse date/time
                fixture_date = datetime.fromisoformat(fixture['fixture']['date'].replace('Z', '+00:00'))
                
                # Clean team names
                home_team = self._clean_team_name(home_team)
                away_team = self._clean_team_name(away_team)
                
                parsed.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'date': fixture_date.strftime('%Y-%m-%d'),
                    'time': fixture_date.strftime('%H:%M'),
                    'venue': fixture['fixture']['venue']['name'],
                    'status': fixture['fixture']['status']['short']
                })
            
            except Exception as e:
                continue
        
        return parsed
    
    def _clean_team_name(self, name):
        """Standardize team names to match our model"""
        
        name_map = {
            'Manchester United': 'Man United',
            'Manchester City': 'Man City',
            'Tottenham Hotspur': 'Tottenham',
            'Newcastle United': 'Newcastle',
            'West Ham United': 'West Ham',
            'Leicester City': 'Leicester',
            'Brighton & Hove Albion': 'Brighton',
            'Brighton and Hove Albion': 'Brighton',
            'Wolverhampton Wanderers': 'Wolves',
            'Nottingham Forest': "Nott'm Forest",
            'AFC Bournemouth': 'Bournemouth',
            'Ipswich Town': 'Ipswich',
        }
        
        return name_map.get(name, name)
    
    def _get_manual_fixtures(self):
        """Fallback manual fixtures (updated weekly)"""
        
        print("   📊 Using manual fixture list")
        
        # MANUAL FIXTURES - Update this weekly!
        # Current: December 15-29, 2024
        fixtures = [
            # Gameweek 17 (December 21-22, 2024)
            {'home_team': 'Man City', 'away_team': 'Man United', 'date': '2024-12-21', 'time': '12:30', 'venue': 'Etihad Stadium', 'status': 'NS'},
            {'home_team': 'Crystal Palace', 'away_team': 'Arsenal', 'date': '2024-12-21', 'time': '15:00', 'venue': 'Selhurst Park', 'status': 'NS'},
            {'home_team': 'Bournemouth', 'away_team': 'West Ham', 'date': '2024-12-21', 'time': '15:00', 'venue': 'Vitality Stadium', 'status': 'NS'},
            {'home_team': 'Fulham', 'away_team': 'Southampton', 'date': '2024-12-22', 'time': '14:00', 'venue': 'Craven Cottage', 'status': 'NS'},
            {'home_team': 'Chelsea', 'away_team': 'Everton', 'date': '2024-12-22', 'time': '14:00', 'venue': 'Stamford Bridge', 'status': 'NS'},
            
            # Gameweek 18 - Boxing Day! (December 26, 2024)
            {'home_team': 'Liverpool', 'away_team': 'Leicester', 'date': '2024-12-26', 'time': '20:00', 'venue': 'Anfield', 'status': 'NS'},
            {'home_team': 'Man City', 'away_team': 'Everton', 'date': '2024-12-26', 'time': '12:30', 'venue': 'Etihad Stadium', 'status': 'NS'},
            {'home_team': 'Newcastle', 'away_team': 'Aston Villa', 'date': '2024-12-26', 'time': '15:00', 'venue': 'St James Park', 'status': 'NS'},
            {'home_team': "Nott'm Forest", 'away_team': 'Tottenham', 'date': '2024-12-26', 'time': '15:00', 'venue': 'City Ground', 'status': 'NS'},
            {'home_team': 'Southampton', 'away_team': 'West Ham', 'date': '2024-12-26', 'time': '15:00', 'venue': 'St Marys Stadium', 'status': 'NS'},
            {'home_team': 'Wolves', 'away_team': 'Man United', 'date': '2024-12-26', 'time': '17:30', 'venue': 'Molineux Stadium', 'status': 'NS'},
            
            # Gameweek 18 continued (December 27-29, 2024)
            {'home_team': 'Arsenal', 'away_team': 'Ipswich', 'date': '2024-12-27', 'time': '20:00', 'venue': 'Emirates Stadium', 'status': 'NS'},
            {'home_team': 'Leicester', 'away_team': 'Man City', 'date': '2024-12-29', 'time': '15:00', 'venue': 'King Power Stadium', 'status': 'NS'},
            {'home_team': 'Tottenham', 'away_team': 'Wolves', 'date': '2024-12-29', 'time': '13:30', 'venue': 'Tottenham Hotspur Stadium', 'status': 'NS'},
            {'home_team': 'Brighton', 'away_team': 'Brentford', 'date': '2024-12-29', 'time': '15:00', 'venue': 'Amex Stadium', 'status': 'NS'},
        ]
        
        return fixtures
    
    def _load_from_cache(self):
        """Load cached fixtures (valid for 12 hours)"""
        
        cache_file = os.path.join(self.cache_dir, 'api_fixtures.json')
        
        if not os.path.exists(cache_file):
            return None
        
        # Check if cache is fresh
        import time
        file_time = os.path.getmtime(cache_file)
        age = time.time() - file_time
        
        if age > 43200:  # 12 hours
            return None
        
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    def _save_to_cache(self, fixtures):
        """Save fixtures to cache"""
        
        cache_file = os.path.join(self.cache_dir, 'api_fixtures.json')
        
        with open(cache_file, 'w') as f:
            json.dump(fixtures, f, indent=2)
        
        print(f"   💾 Cached {len(fixtures)} fixtures")

# Testing
if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("⚽ API-FOOTBALL FIXTURE FETCHER TEST")
    print("=" * 70)
    
    # Check for API key
    api_key = os.getenv('API_FOOTBALL_KEY')
    
    if not api_key:
        print("\n⚠️  No API key found!")
        print("\nTo use API-Football:")
        print("1. Sign up: https://www.api-football.com/")
        print("2. Get your API key from dashboard")
        print("3. Set environment variable:")
        print("   Windows: set API_FOOTBALL_KEY=your_key_here")
        print("   Or add to .env file")
        print("\n✅ For now, using manual fixtures as fallback\n")
    
    fetcher = APIFootballFetcher(api_key)
    fixtures = fetcher.get_upcoming_fixtures(days_ahead=14)
    
    if fixtures:
        print("\n📅 UPCOMING FIXTURES:")
        print("-" * 70)
        
        for i, f in enumerate(fixtures[:10], 1):
            print(f"\n{i}. {f['home_team']} vs {f['away_team']}")
            print(f"   Date: {f['date']} at {f['time']}")
            print(f"   Venue: {f.get('venue', 'TBD')}")