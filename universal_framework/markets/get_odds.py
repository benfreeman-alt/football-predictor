"""
THE-ODDS-API INTEGRATION

Simple, reliable odds from multiple bookmakers
Free tier: 500 requests/month (perfect for weekly use)

Sign up: https://the-odds-api.com/
"""

import requests
from datetime import datetime

def get_odds_from_api(api_key):
    """
    Get Premier League odds from The-Odds-API
    
    Free API key: https://the-odds-api.com/
    """
    
    print("\n📊 Fetching odds from The-Odds-API...")
    
    # API endpoint
    url = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds/"
    
    params = {
        'apiKey': api_key,
        'regions': 'uk',  # UK bookmakers
        'markets': 'h2h',  # Head-to-head (match winner)
        'oddsFormat': 'decimal'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Check remaining requests
        remaining = response.headers.get('x-requests-remaining')
        used = response.headers.get('x-requests-used')
        
        print(f"   API Usage: {used} used, {remaining} remaining this month")
        
        all_odds = {}
        
        for match in data:
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Get best odds from all bookmakers
            if match['bookmakers']:
                # Use first bookmaker (or average across all)
                bookmaker = match['bookmakers'][0]
                markets = bookmaker['markets'][0]  # h2h market
                
                outcomes = markets['outcomes']
                
                odds_dict = {}
                
                for outcome in outcomes:
                    name = outcome['name']
                    price = outcome['price']
                    
                    if name == home_team:
                        odds_dict['home win'] = price
                    elif name == away_team:
                        odds_dict['away win'] = price
                    elif name == 'Draw':
                        odds_dict['draw'] = price
                
                if len(odds_dict) == 3:
                    # Clean team names
                    from get_fixtures import clean_team_name
                    home_clean = clean_team_name(home_team)
                    away_clean = clean_team_name(away_team)
                    
                    all_odds[(home_clean, away_clean)] = odds_dict
                    
                    print(f"   {home_clean} vs {away_clean}")
                    print(f"      Home: {odds_dict['home win']:.2f}")
                    print(f"      Draw: {odds_dict['draw']:.2f}")
                    print(f"      Away: {odds_dict['away win']:.2f}")
                    print()
        
        print(f"   ✅ Retrieved odds for {len(all_odds)} matches")
        
        return all_odds
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("   ❌ Invalid API key")
        elif e.response.status_code == 429:
            print("   ❌ Rate limit exceeded")
        else:
            print(f"   ❌ HTTP Error: {e}")
        return {}
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {}

# Testing
if __name__ == "__main__":
    API_KEY = 'YOUR_API_KEY_HERE'
    
    if API_KEY == 'YOUR_API_KEY_HERE':
        print("⚠️  Get your free API key:")
        print("   1. Visit: https://the-odds-api.com/")
        print("   2. Click 'Get a Free API Key'")
        print("   3. Sign up (email only)")
        print("   4. Copy key and update this script")
    else:
        odds = get_odds_from_api(API_KEY)
        
        if odds:
            print(f"\n✅ Success! {len(odds)} matches with odds")