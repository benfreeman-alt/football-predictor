"""
DETAILED 2025 SEASON DEBUG

Check what fixtures actually exist in API-Football for 2025 season
"""

import requests
import os
from datetime import datetime, timedelta

api_key = os.getenv('API_FOOTBALL_KEY')

print("=" * 70)
print("🔍 2025 SEASON FIXTURE DEBUG")
print("=" * 70)

if not api_key:
    print("\n❌ No API key!")
    exit()

print(f"\n✅ API Key: {api_key[:10]}...")

headers = {
    'x-rapidapi-host': 'v3.football.api-sports.io',
    'x-rapidapi-key': api_key
}

# Test: Get ALL fixtures for 2025 season (no date filter)
print("\n" + "=" * 70)
print("TEST: Get ALL 2025 Season Fixtures")
print("=" * 70)

url = "https://v3.football.api-sports.io/fixtures"

params = {
    'league': 39,  # Premier League
    'season': 2025,
}

print(f"\nFetching fixtures for Premier League 2025 season...")
print(f"(This might take a moment...)")

try:
    response = requests.get(url, headers=headers, params=params)
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        fixtures = data.get('response', [])
        
        print(f"Total fixtures in 2025 season: {len(fixtures)}")
        
        if len(fixtures) > 0:
            print("\n✅ Fixtures exist in 2025 season!")
            
            # Group by date
            from collections import defaultdict
            by_date = defaultdict(list)
            
            for fixture in fixtures:
                try:
                    date_str = fixture['fixture']['date']
                    date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    date_key = date_obj.strftime('%Y-%m')
                    by_date[date_key].append({
                        'date': date_obj,
                        'home': fixture['teams']['home']['name'],
                        'away': fixture['teams']['away']['name'],
                        'status': fixture['fixture']['status']['short']
                    })
                except:
                    continue
            
            print("\nFixtures by month:")
            for month in sorted(by_date.keys()):
                count = len(by_date[month])
                print(f"  {month}: {count} fixtures")
            
            # Show December 2025 fixtures
            december_key = '2025-12'
            if december_key in by_date:
                print(f"\n📅 DECEMBER 2025 FIXTURES:")
                print("-" * 70)
                
                dec_fixtures = sorted(by_date[december_key], key=lambda x: x['date'])
                
                for i, f in enumerate(dec_fixtures[:20], 1):
                    date_str = f['date'].strftime('%Y-%m-%d %H:%M')
                    print(f"{i:2d}. {f['home']} vs {f['away']}")
                    print(f"    {date_str} - Status: {f['status']}")
            else:
                print(f"\n⚠️  No fixtures found for December 2025")
                print("\nAvailable months:")
                for month in sorted(by_date.keys()):
                    print(f"  - {month}")
            
            # Find fixtures from today onwards
            print("\n" + "=" * 70)
            print("UPCOMING FIXTURES (from today)")
            print("=" * 70)
            
            today = datetime.now()
            upcoming = []
            
            for fixture in fixtures:
                try:
                    date_str = fixture['fixture']['date']
                    date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    
                    if date_obj >= today:
                        upcoming.append({
                            'date': date_obj,
                            'home': fixture['teams']['home']['name'],
                            'away': fixture['teams']['away']['name'],
                            'status': fixture['fixture']['status']['short'],
                            'venue': fixture['fixture']['venue']['name']
                        })
                except:
                    continue
            
            if upcoming:
                upcoming_sorted = sorted(upcoming, key=lambda x: x['date'])[:15]
                
                print(f"\nFound {len(upcoming)} fixtures from {today.strftime('%Y-%m-%d')} onwards")
                print(f"\nNext 15 fixtures:")
                print("-" * 70)
                
                for i, f in enumerate(upcoming_sorted, 1):
                    date_str = f['date'].strftime('%Y-%m-%d %H:%M')
                    print(f"\n{i:2d}. {f['home']} vs {f['away']}")
                    print(f"    Date: {date_str}")
                    print(f"    Venue: {f['venue']}")
                    print(f"    Status: {f['status']}")
            else:
                print(f"\n⚠️  No upcoming fixtures found from {today.strftime('%Y-%m-%d')}")
        
        else:
            print("\n❌ No fixtures found for 2025 season!")
            print("\nPossible reasons:")
            print("- Season 2025 not yet in API database")
            print("- Fixtures not yet scheduled")
            print("- API using different season format")
    
    else:
        print(f"❌ API Error: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("END OF DEBUG")
print("=" * 70)