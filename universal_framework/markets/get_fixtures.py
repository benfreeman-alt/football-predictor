"""
FIXTURE FETCHER

Gets upcoming Premier League fixtures
"""

import requests
from datetime import datetime

def clean_team_name(name):
    """Standardize team names"""
    
    name_map = {
        'Man United': 'Man United',
        'Manchester United FC': 'Man United',
        'Manchester United': 'Man United',
        'Man City': 'Man City',
        'Manchester City FC': 'Man City',
        'Manchester City': 'Man City',
        'Spurs': 'Tottenham',
        'Tottenham': 'Tottenham',
        'Tottenham Hotspur FC': 'Tottenham',
        'Tottenham Hotspur': 'Tottenham',
        'West Ham': 'West Ham',
        'West Ham United FC': 'West Ham',
        'West Ham United': 'West Ham',
        'Newcastle': 'Newcastle',
        'Newcastle United FC': 'Newcastle',
        'Newcastle United': 'Newcastle',
        'Leicester': 'Leicester',
        'Leicester City FC': 'Leicester',
        'Leicester City': 'Leicester',
        'Brighton': 'Brighton',
        'Brighton & Hove Albion FC': 'Brighton',
        'Brighton & Hove Albion': 'Brighton',
        'Brighton and Hove Albion': 'Brighton',
        'Wolves': 'Wolves',
        'Wolverhampton Wanderers FC': 'Wolves',
        'Wolverhampton Wanderers': 'Wolves',
        "Nott'm Forest": "Nott'm Forest",
        'Nottingham Forest FC': "Nott'm Forest",
        'Nottingham Forest': "Nott'm Forest",
        'Arsenal': 'Arsenal',
        'Arsenal FC': 'Arsenal',
        'Liverpool': 'Liverpool',
        'Liverpool FC': 'Liverpool',
        'Chelsea': 'Chelsea',
        'Chelsea FC': 'Chelsea',
        'Everton': 'Everton',
        'Everton FC': 'Everton',
        'Aston Villa': 'Aston Villa',
        'Aston Villa FC': 'Aston Villa',
        'Crystal Palace': 'Crystal Palace',
        'Crystal Palace FC': 'Crystal Palace',
        'Fulham': 'Fulham',
        'Fulham FC': 'Fulham',
        'Bournemouth': 'Bournemouth',
        'AFC Bournemouth': 'Bournemouth',
        'Brentford': 'Brentford',
        'Brentford FC': 'Brentford',
        'Southampton': 'Southampton',
        'Southampton FC': 'Southampton',
        'Ipswich': 'Ipswich',
        'Ipswich Town FC': 'Ipswich',
        'Ipswich Town': 'Ipswich',
    }
    
    return name_map.get(name, name)

def get_fallback_fixtures():
    """Manual fixtures if API/scraping fails"""
    
    print("\n   Using manual fixtures...")
    print("   Check actual fixtures: https://www.premierleague.com/fixtures\n")
    
    # UPDATE THESE MANUALLY EACH WEEK
    # Current fixtures (December 14-15, 2025)
    fixtures = [
        ('Arsenal', 'Everton'),
        ('Man City', 'Man United'),
        ('Liverpool', 'Fulham'),
        ('Newcastle', 'Leicester'),
        ('Tottenham', 'Southampton'),
        ('Brighton', 'Crystal Palace'),
        ('Bournemouth', 'West Ham'),
        ('Ipswich', 'Brentford'),
    ]
    
    for i, (home, away) in enumerate(fixtures, 1):
        print(f"   {i}. {home} vs {away}")
    
    return fixtures

def get_this_weekends_fixtures():
    """Get this weekend's fixtures"""
    
    print("\n📅 Fetching this weekend's fixtures...")
    
    # For now, use fallback
    # You can add API integration later
    fixtures = get_fallback_fixtures()
    
    return fixtures

# Testing
if __name__ == "__main__":
    fixtures = get_this_weekends_fixtures()
    print(f"\n✅ Retrieved {len(fixtures)} fixtures")