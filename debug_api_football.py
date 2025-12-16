"""
DEBUG VERSION - API Football Fetcher

This will show exactly what the API is returning
"""

import requests
import json
import os
from datetime import datetime, timedelta

api_key = os.getenv('API_FOOTBALL_KEY') or "YOUR_KEY_HERE"

print("=" * 70)
print("🔍 API-FOOTBALL DEBUG TEST")
print("=" * 70)

if not api_key or api_key == "YOUR_KEY_HERE":
    print("\n❌ No API key set!")
    exit()

print(f"\n✅ API Key found: {api_key[:10]}...")

# Test 1: Check API connection
print("\n" + "=" * 70)
print("TEST 1: API Connection")
print("=" * 70)

url = "https://v3.football.api-sports.io/status"
headers = {
    'x-rapidapi-host': 'v3.football.api-sports.io',
    'x-rapidapi-key': api_key
}

try:
    response = requests.get(url, headers=headers)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ API Connected!")
        print(f"Account: {data.get('response', {}).get('account', {})}")
        print(f"Requests Today: {data.get('response', {}).get('requests', {})}")
    else:
        print(f"❌ Error: {response.text}")
        exit()
except Exception as e:
    print(f"❌ Connection failed: {e}")
    exit()

# Test 2: Get current season info
print("\n" + "=" * 70)
print("TEST 2: Current Season Info")
print("=" * 70)

url = "https://v3.football.api-sports.io/leagues"
params = {
    'id': 39,  # Premier League
    'current': 'true'
}

try:
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['response']:
            league_info = data['response'][0]
            seasons = league_info.get('seasons', [])
            
            print("Premier League Seasons Available:")
            for season in seasons[-3:]:  # Last 3 seasons
                print(f"  - {season['year']}: {season['start']} to {season['end']} (Current: {season['current']})")
            
            # Find current season
            current_season = [s for s in seasons if s['current']][0]
            season_year = current_season['year']
            print(f"\n✅ Current Season: {season_year}")
        else:
            print("❌ No league data returned")
            exit()
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# Test 3: Get fixtures for current season
print("\n" + "=" * 70)
print("TEST 3: Fetching Fixtures")
print("=" * 70)

url = "https://v3.football.api-sports.io/fixtures"

# Try different date ranges
today = datetime.now().date()
tomorrow = today + timedelta(days=1)
next_week = today + timedelta(days=7)
next_month = today + timedelta(days=30)

test_ranges = [
    (today, next_week, "Next 7 days"),
    (today, next_month, "Next 30 days"),
    (tomorrow, next_month, "Tomorrow onwards"),
]

for start_date, end_date, label in test_ranges:
    print(f"\n🔍 Testing: {label}")
    print(f"   From: {start_date}")
    print(f"   To: {end_date}")
    
    params = {
        'league': 39,
        'season': season_year,
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d')
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            num_fixtures = len(data.get('response', []))
            
            print(f"   Status: {response.status_code}")
            print(f"   Fixtures found: {num_fixtures}")
            
            if num_fixtures > 0:
                print(f"\n   ✅ SUCCESS! Found {num_fixtures} fixtures")
                print("\n   First 5 fixtures:")
                
                for i, fixture in enumerate(data['response'][:5], 1):
                    home = fixture['teams']['home']['name']
                    away = fixture['teams']['away']['name']
                    date = fixture['fixture']['date']
                    status = fixture['fixture']['status']['short']
                    
                    print(f"   {i}. {home} vs {away}")
                    print(f"      Date: {date}")
                    print(f"      Status: {status}")
                
                print(f"\n✅ WORKING DATE RANGE: {label}")
                print(f"   From: {start_date}")
                print(f"   To: {end_date}")
                print(f"   Season: {season_year}")
                break
            else:
                print(f"   ⚠️  No fixtures in this range")
        else:
            print(f"   ❌ Error: {response.status_code}")
    
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
print("END OF DEBUG TEST")
print("=" * 70)