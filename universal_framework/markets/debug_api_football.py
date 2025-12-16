"""
API-FOOTBALL DEBUG SCRIPT

See exactly what the API returns for different season parameters
"""

import requests
import os
from datetime import datetime, timedelta
import json

api_key = os.getenv('API_FOOTBALL_KEY')

if not api_key:
    print("❌ No API_FOOTBALL_KEY found")
    exit()

print("="*70)
print("API-FOOTBALL DEBUG")
print("="*70)

headers = {
    'x-rapidapi-host': 'v3.football.api-sports.io',
    'x-rapidapi-key': api_key
}

# Test different season values
seasons_to_try = [2024, 2025, 2026]

for season in seasons_to_try:
    print(f"\n{'='*70}")
    print(f"TESTING SEASON: {season}")
    print(f"{'='*70}")
    
    url = "https://v3.football.api-sports.io/fixtures"
    
    today = datetime.now()
    date_from = today.strftime('%Y-%m-%d')
    date_to = (today + timedelta(days=30)).strftime('%Y-%m-%d')
    
    params = {
        'league': 39,  # Premier League
        'season': season,
        'from': date_from,
        'to': date_to
    }
    
    print(f"\nRequest parameters:")
    print(f"  League: 39 (Premier League)")
    print(f"  Season: {season}")
    print(f"  From: {date_from}")
    print(f"  To: {date_to}")
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Rate Limit Remaining: {response.headers.get('x-ratelimit-requests-remaining', 'unknown')}")
        
        if response.status_code == 200:
            data = response.json()
            
            fixtures = data.get('response', [])
            print(f"Fixtures found: {len(fixtures)}")
            
            if fixtures:
                print(f"\nFirst 3 fixtures:")
                for i, match in enumerate(fixtures[:3], 1):
                    home = match['teams']['home']['name']
                    away = match['teams']['away']['name']
                    date = match['fixture']['date']
                    print(f"  {i}. {home} vs {away} - {date}")
            else:
                print("\n⚠️  No fixtures in response")
                
                # Check if there's an error message
                if 'errors' in data and data['errors']:
                    print(f"API Errors: {data['errors']}")
        
        else:
            print(f"❌ Request failed")
            print(f"Response: {response.text[:200]}")
    
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "="*70)
print("CHECKING CURRENT SEASON INFO")
print("="*70)

# Get league info to see current season
url = "https://v3.football.api-sports.io/leagues"
params = {
    'id': 39,  # Premier League
    'current': 'true'
}

try:
    response = requests.get(url, headers=headers, params=params, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        leagues = data.get('response', [])
        
        if leagues:
            league = leagues[0]
            print(f"\nPremier League Info:")
            print(f"  Name: {league['league']['name']}")
            
            seasons = league.get('seasons', [])
            print(f"\nAvailable Seasons: {len(seasons)}")
            
            for season_info in seasons[-3:]:  # Last 3 seasons
                year = season_info['year']
                current = season_info.get('current', False)
                start = season_info.get('start', 'Unknown')
                end = season_info.get('end', 'Unknown')
                
                marker = " ← CURRENT" if current else ""
                print(f"  Season {year}: {start} to {end}{marker}")

except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*70)
print("END DEBUG")
print("="*70)