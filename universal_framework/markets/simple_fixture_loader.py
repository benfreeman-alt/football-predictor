"""
SIMPLE FIXTURE LOADER

Loads fixtures from fixtures.json file
Update the JSON file weekly - takes 5 minutes!
"""

import json
import os
from datetime import datetime

class SimpleFixtureLoader:
    """Load fixtures from JSON file"""
    
    def __init__(self, fixtures_file='fixtures.json'):
        self.fixtures_file = fixtures_file
        
        # Try different locations
        possible_locations = [
            fixtures_file,
            os.path.join('data', fixtures_file),
            os.path.join(os.path.dirname(__file__), fixtures_file),
            os.path.join(os.path.dirname(__file__), 'data', fixtures_file),
        ]
        
        for location in possible_locations:
            if os.path.exists(location):
                self.fixtures_file = location
                break
    
    def get_upcoming_fixtures(self, days_ahead=14):
        """Load fixtures - tries auto-fetch first, then JSON file"""
        
        print(f"\n📅 Loading fixtures...")
        
        # Try auto-fetch first
        try:
            from auto_fixture_fetcher import AutoFixtureFetcher
            
            auto_fetcher = AutoFixtureFetcher()
            fixtures = auto_fetcher.get_upcoming_fixtures(days_ahead)
            
            if fixtures and len(fixtures) > 0:
                print(f"   ✅ Auto-fetched {len(fixtures)} fixtures")
                return fixtures
        
        except Exception as e:
            print(f"   ⚠️  Auto-fetch failed: {e}")
        
        # Fall back to manual JSON file
        print(f"   📂 Loading from {os.path.basename(self.fixtures_file)}...")
        
        try:
            if not os.path.exists(self.fixtures_file):
                print(f"   ⚠️  File not found: {self.fixtures_file}")
                return self._get_default_fixtures()
            
            with open(self.fixtures_file, 'r') as f:
                fixtures = json.load(f)
            
            # Filter to upcoming only
            today = datetime.now().date()
            upcoming = []
            
            for fixture in fixtures:
                try:
                    fixture_date = datetime.strptime(fixture['date'], '%Y-%m-%d').date()
                    
                    # Only include future fixtures
                    if fixture_date >= today:
                        # Add status if not present
                        if 'status' not in fixture:
                            fixture['status'] = 'NS'
                        upcoming.append(fixture)
                
                except:
                    continue
            
            print(f"   ✅ Loaded {len(upcoming)} upcoming fixtures from JSON")
            return upcoming
        
        except Exception as e:
            print(f"   ❌ Error loading fixtures: {e}")
            return self._get_default_fixtures()
    
    def _get_default_fixtures(self):
        """Default fixtures if file not found"""
        return [
            {'home_team': 'Man United', 'away_team': 'Bournemouth', 'date': '2025-12-22', 'time': '14:00', 'venue': 'Old Trafford', 'status': 'NS'},
            {'home_team': 'Liverpool', 'away_team': 'Leicester', 'date': '2025-12-26', 'time': '20:00', 'venue': 'Anfield', 'status': 'NS'},
            {'home_team': 'Arsenal', 'away_team': 'Ipswich', 'date': '2025-12-27', 'time': '20:00', 'venue': 'Emirates Stadium', 'status': 'NS'},
        ]

# Testing
if __name__ == "__main__":
    loader = SimpleFixtureLoader()
    fixtures = loader.get_upcoming_fixtures()
    
    print("\n" + "=" * 70)
    print("📅 UPCOMING FIXTURES")
    print("=" * 70)
    
    for i, f in enumerate(fixtures, 1):
        print(f"\n{i:2d}. {f['home_team']} vs {f['away_team']}")
        print(f"    Date: {f['date']} at {f['time']}")
        print(f"    Venue: {f['venue']}")