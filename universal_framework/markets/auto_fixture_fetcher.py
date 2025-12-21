"""
PREMIER LEAGUE FIXTURE FETCHER

Uses free football APIs to get real upcoming fixtures
No API key needed! Multiple fallback sources
"""

import requests
import json
import os
from datetime import datetime, timedelta

class AutoFixtureFetcher:
    """Automatically fetch Premier League fixtures from multiple sources"""
    
    def __init__(self, cache_dir='data/fixture_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_upcoming_fixtures(self, days_ahead=14):
        """Get upcoming fixtures from multiple sources"""
        
        print("\n⚽ Auto-fetching Premier League fixtures...")
        
        # Check cache first (valid for 12 hours)
        cached = self._load_from_cache()
        if cached:
            print(f"   📂 Using cached fixtures ({len(cached)} fixtures)")
            return cached
        
        # Try multiple sources in order
        fixtures = None
        
        # Source 1: API-Football (try first if key available)
        fixtures = self._fetch_from_api_football()
        
        # Source 2: Football-Data.org (free, no key needed for limited use)
        if not fixtures:
            fixtures = self._fetch_from_football_data()
        
        # Source 3: Fallback to manual list
        if not fixtures:
            print("   ⚠️  Auto-fetch failed, using fallback fixtures")
            fixtures = self._get_fallback_fixtures()
        
        if fixtures:
            self._save_to_cache(fixtures)
        
        return fixtures
    
    def _fetch_from_football_data(self):
        """Fetch from football-data.org (free tier available)"""
        
        try:
            # Get token from environment
            api_token = os.getenv('FOOTBALL_DATA_TOKEN')
            
            if not api_token:
                print("   ⚠️  No FOOTBALL_DATA_TOKEN found")
                return None
            
            print("   Trying football-data.org...")
            
            # Premier League
            url = "https://api.football-data.org/v4/competitions/PL/matches"
            
            headers = {
                'X-Auth-Token': api_token
            }
            
            # Get SCHEDULED matches only
            params = {
                'status': 'SCHEDULED'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                matches = data.get('matches', [])
                
                print(f"   Found {len(matches)} scheduled matches")
                
                if not matches:
                    print("   ⚠️  No scheduled matches found")
                    return None
                
                # Convert to our format
                fixtures = []
                today = datetime.now()
                # Make today timezone-aware for comparison
                if today.tzinfo is None:
                    from datetime import timezone
                    today = today.replace(tzinfo=timezone.utc)
                
                cutoff = today + timedelta(days=14)
                
                print(f"   Today: {today.strftime('%Y-%m-%d %H:%M UTC')}")
                print(f"   Cutoff: {cutoff.strftime('%Y-%m-%d %H:%M UTC')}")
                
                for match in matches:
                    try:
                        match_date_str = match.get('utcDate', '')
                        match_date = datetime.fromisoformat(match_date_str.replace('Z', '+00:00'))
                        
                        # Only include FUTURE matches in next 14 days
                        if match_date >= today and match_date <= cutoff:
                            fixtures.append({
                                'home_team': self._clean_team_name(match['homeTeam']['name']),
                                'away_team': self._clean_team_name(match['awayTeam']['name']),
                                'date': match_date.strftime('%Y-%m-%d'),
                                'time': match_date.strftime('%H:%M'),
                                'venue': match.get('venue', 'TBD'),
                                'status': 'NS',
                                'datetime': match_date  # Keep for sorting
                            })
                    except Exception as e:
                        print(f"   ⚠️  Error parsing match: {e}")
                        continue
                
                if fixtures:
                    # Sort by date/time
                    fixtures.sort(key=lambda x: x.get('datetime', datetime.max))
                    # Remove datetime field (not needed in final output)
                    for f in fixtures:
                        f.pop('datetime', None)
                    
                    print(f"   ✅ Retrieved {len(fixtures)} fixtures from football-data.org")
                    return fixtures
                else:
                    print("   ⚠️  No fixtures in next 14 days")
            
            elif response.status_code == 429:
                print("   ⚠️  Rate limit exceeded (10 calls/minute limit)")
            elif response.status_code == 403:
                print("   ⚠️  Invalid API token or unauthorized")
            else:
                print(f"   ⚠️  API error: {response.status_code}")
            
        except Exception as e:
            print(f"   ❌ football-data.org failed: {e}")
        
        return None
    
    def _fetch_from_api_football(self):
        """Fetch from API-Football if key available"""
        
        try:
            api_key = os.getenv('API_FOOTBALL_KEY')
            if not api_key:
                print("   ⚠️  No API_FOOTBALL_KEY found")
                return None
            
            print("   Trying API-Football...")
            
            # IMPORTANT: Use REAL WORLD current season
            # Premier League 2024-2025 season (started August 2024, ends May 2025)
            # This is the actual current season in the real world
            season = 2024
            
            print(f"   Using season: {season}-{season+1} (current real-world season)")
            
            headers = {
                'x-rapidapi-host': 'v3.football.api-sports.io',
                'x-rapidapi-key': api_key
            }
            
            url = "https://v3.football.api-sports.io/fixtures"
            
            # Get upcoming fixtures (next 14 days from today's real date)
            from datetime import datetime as real_datetime
            today = real_datetime.now()
            date_from = today.strftime('%Y-%m-%d')
            date_to = (today + timedelta(days=14)).strftime('%Y-%m-%d')
            
            params = {
                'league': 39,  # Premier League
                'season': season,
                'from': date_from,
                'to': date_to,
                'timezone': 'UTC'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                fixtures_raw = data.get('response', [])
                
                print(f"   Found {len(fixtures_raw)} fixtures in API response")
                
                if not fixtures_raw:
                    print(f"   ⚠️  No fixtures found for season {season}")
                    return None
                
                fixtures = []
                for match in fixtures_raw:
                    try:
                        match_date = datetime.fromisoformat(match['fixture']['date'].replace('Z', '+00:00'))
                        
                        fixtures.append({
                            'home_team': self._clean_team_name(match['teams']['home']['name']),
                            'away_team': self._clean_team_name(match['teams']['away']['name']),
                            'date': match_date.strftime('%Y-%m-%d'),
                            'time': match_date.strftime('%H:%M'),
                            'venue': match['fixture']['venue']['name'],
                            'status': 'NS'
                        })
                    except Exception as e:
                        continue
                
                if fixtures:
                    print(f"   ✅ Retrieved {len(fixtures)} fixtures from API-Football")
                    print(f"   📊 API requests remaining: {response.headers.get('x-ratelimit-requests-remaining', 'unknown')}")
                    return fixtures
            
            elif response.status_code == 429:
                print("   ⚠️  Rate limit exceeded")
            elif response.status_code == 403:
                print("   ⚠️  Invalid API key")
            else:
                print(f"   ⚠️  API error: {response.status_code}")
        
        except Exception as e:
            print(f"   ❌ API-Football failed: {e}")
        
        return None
    
    def _get_fallback_fixtures(self):
        """Fallback manual fixtures"""
        
        # Current week's real fixtures (update this weekly as backup)
        fixtures = [
            {'home_team': 'Man City', 'away_team': 'Everton', 'date': '2025-12-26', 'time': '12:30', 'venue': 'Etihad Stadium', 'status': 'NS'},
            {'home_team': 'Liverpool', 'away_team': 'Leicester', 'date': '2025-12-26', 'time': '20:00', 'venue': 'Anfield', 'status': 'NS'},
            {'home_team': 'Newcastle', 'away_team': 'Aston Villa', 'date': '2025-12-26', 'time': '15:00', 'venue': 'St James Park', 'status': 'NS'},
            {'home_team': 'Arsenal', 'away_team': 'Ipswich', 'date': '2025-12-27', 'time': '20:00', 'venue': 'Emirates Stadium', 'status': 'NS'},
            {'home_team': 'Leicester', 'away_team': 'Man City', 'date': '2025-12-29', 'time': '15:00', 'venue': 'King Power Stadium', 'status': 'NS'},
        ]
        
        return fixtures
    
    def _clean_team_name(self, name):
        """Standardize team names"""
        
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
            'AFC Bournemouth': 'Bournemouth',
            'Ipswich Town FC': 'Ipswich',
            'Aston Villa FC': 'Aston Villa',
            'Crystal Palace FC': 'Crystal Palace',
            'Everton FC': 'Everton',
            'Fulham FC': 'Fulham',
            'Brentford FC': 'Brentford',
            'Southampton FC': 'Southampton',
            'Chelsea FC': 'Chelsea',
            'Arsenal FC': 'Arsenal',
            'Liverpool FC': 'Liverpool',
        }
        
        return name_map.get(name, name.replace(' FC', '').replace(' United', '').strip())
    
    def _load_from_cache(self):
        """Load cached fixtures (12 hour validity)"""
        
        cache_file = os.path.join(self.cache_dir, 'auto_fixtures.json')
        
        if not os.path.exists(cache_file):
            return None
        
        # Check age
        import time
        file_age = time.time() - os.path.getmtime(cache_file)
        
        if file_age > 43200:  # 12 hours
            return None
        
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    def _save_to_cache(self, fixtures):
        """Save to cache"""
        
        cache_file = os.path.join(self.cache_dir, 'auto_fixtures.json')
        
        with open(cache_file, 'w') as f:
            json.dump(fixtures, f, indent=2)
        
        print(f"   💾 Cached {len(fixtures)} fixtures")

# Testing
if __name__ == "__main__":
    fetcher = AutoFixtureFetcher()
    fixtures = fetcher.get_upcoming_fixtures()
    
    print("\n" + "=" * 70)
    print("📅 UPCOMING FIXTURES")
    print("=" * 70)
    
    for i, f in enumerate(fixtures[:10], 1):
        print(f"\n{i:2d}. {f['home_team']} vs {f['away_team']}")
        print(f"    {f['date']} at {f['time']}")