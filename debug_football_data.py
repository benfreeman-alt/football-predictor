"""
FOOTBALL-DATA.ORG DEBUG SCRIPT

Test if the API works and see what it returns
"""

import requests
import os
from datetime import datetime, timedelta

api_token = os.getenv('FOOTBALL_DATA_TOKEN')

print("="*70)
print("FOOTBALL-DATA.ORG DEBUG")
print("="*70)

if not api_token:
    print("\n❌ No FOOTBALL_DATA_TOKEN environment variable found!")
    print("\nTo set it:")
    print("  PowerShell: $env:FOOTBALL_DATA_TOKEN = \"your_token\"")
    exit()

print(f"\n✅ Token found: {api_token[:10]}...")

headers = {
    'X-Auth-Token': api_token
}

# Test 1: Get competition info
print("\n" + "="*70)
print("TEST 1: Get Premier League Competition Info")
print("="*70)

url = "https://api.football-data.org/v4/competitions/PL"

try:
    response = requests.get(url, headers=headers, timeout=10)
    
    print(f"Status Code: {response.status_code}")
    print(f"Rate Limit: {response.headers.get('X-Requests-Available-Minute', 'unknown')}/10 remaining")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Competition: {data.get('name')}")
        print(f"   Current Season: {data.get('currentSeason', {}).get('id', 'Unknown')}")
        print(f"   Start: {data.get('currentSeason', {}).get('startDate', 'Unknown')}")
        print(f"   End: {data.get('currentSeason', {}).get('endDate', 'Unknown')}")
    
    elif response.status_code == 403:
        print("❌ 403 Forbidden - Invalid token or unauthorized")
    elif response.status_code == 429:
        print("❌ 429 Rate limit exceeded")
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text[:200]}")

except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Get upcoming matches
print("\n" + "="*70)
print("TEST 2: Get Upcoming Matches (All Statuses)")
print("="*70)

url = "https://api.football-data.org/v4/competitions/PL/matches"

params = {}  # No status filter - get all

try:
    response = requests.get(url, headers=headers, params=params, timeout=10)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        matches = data.get('matches', [])
        
        print(f"\nTotal matches in response: {len(matches)}")
        
        if matches:
            # Group by status
            from collections import defaultdict
            by_status = defaultdict(list)
            
            for match in matches:
                status = match.get('status')
                by_status[status].append(match)
            
            print("\nMatches by status:")
            for status, match_list in by_status.items():
                print(f"  {status}: {len(match_list)} matches")
            
            # Show upcoming matches
            print("\n" + "="*70)
            print("UPCOMING SCHEDULED MATCHES:")
            print("="*70)
            
            today = datetime.now()
            upcoming = []
            
            for match in matches:
                if match.get('status') in ['SCHEDULED', 'TIMED']:
                    try:
                        match_date_str = match.get('utcDate', '')
                        match_date = datetime.fromisoformat(match_date_str.replace('Z', '+00:00'))
                        
                        if match_date >= today:
                            upcoming.append({
                                'date': match_date,
                                'home': match['homeTeam']['name'],
                                'away': match['awayTeam']['name'],
                                'status': match['status']
                            })
                    except:
                        continue
            
            upcoming.sort(key=lambda x: x['date'])
            
            if upcoming:
                print(f"\nFound {len(upcoming)} upcoming matches:")
                for i, m in enumerate(upcoming[:10], 1):
                    date_str = m['date'].strftime('%Y-%m-%d %H:%M')
                    print(f"{i:2d}. {m['home']} vs {m['away']}")
                    print(f"    {date_str} ({m['status']})")
            else:
                print("\n⚠️  No upcoming matches found!")
                print("\nThis could mean:")
                print("  - Season ended")
                print("  - No matches scheduled yet")
                print("  - Winter break")
        
        else:
            print("\n⚠️  No matches in response at all!")
    
    elif response.status_code == 403:
        print("❌ 403 Forbidden - Check your token")
    elif response.status_code == 404:
        print("❌ 404 Not Found - Competition might not exist")
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text[:300]}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Try with status filter
print("\n" + "="*70)
print("TEST 3: Get SCHEDULED Matches Only")
print("="*70)

params = {'status': 'SCHEDULED'}

try:
    response = requests.get(url, headers=headers, params=params, timeout=10)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        matches = data.get('matches', [])
        print(f"SCHEDULED matches found: {len(matches)}")
        
        if matches:
            for i, match in enumerate(matches[:5], 1):
                home = match['homeTeam']['name']
                away = match['awayTeam']['name']
                date = match.get('utcDate', 'Unknown')
                print(f"  {i}. {home} vs {away} - {date}")

except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*70)
print("END DEBUG")
print("="*70)