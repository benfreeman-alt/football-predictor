"""
PREMIER LEAGUE INJURY SCRAPER

Scrapes current injury data from Premier Injuries
Updates every 24 hours
"""

import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime, timedelta

class InjuryScraper:
    """Scrape Premier League injury data"""
    
    def __init__(self, cache_dir='data/injury_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Team name mappings
        self.team_map = {
            'Arsenal': 'Arsenal',
            'Aston Villa': 'Aston Villa',
            'Bournemouth': 'Bournemouth',
            'AFC Bournemouth': 'Bournemouth',
            'Brentford': 'Brentford',
            'Brighton': 'Brighton',
            'Brighton & Hove Albion': 'Brighton',
            'Brighton and Hove Albion': 'Brighton',
            'Chelsea': 'Chelsea',
            'Crystal Palace': 'Crystal Palace',
            'Everton': 'Everton',
            'Fulham': 'Fulham',
            'Ipswich': 'Ipswich',
            'Ipswich Town': 'Ipswich',
            'Leicester': 'Leicester',
            'Leicester City': 'Leicester',
            'Liverpool': 'Liverpool',
            'Manchester City': 'Man City',
            'Man City': 'Man City',
            'Manchester United': 'Man United',
            'Man United': 'Man United',
            'Man Utd': 'Man United',
            'Newcastle': 'Newcastle',
            'Newcastle United': 'Newcastle',
            'Nottingham Forest': "Nott'm Forest",
            "Nott'm Forest": "Nott'm Forest",
            'Southampton': 'Southampton',
            'Tottenham': 'Tottenham',
            'Tottenham Hotspur': 'Tottenham',
            'West Ham': 'West Ham',
            'West Ham United': 'West Ham',
            'Wolves': 'Wolves',
            'Wolverhampton Wanderers': 'Wolves',
            'Wolverhampton': 'Wolves'
        }
    
    def get_team_injuries(self, team_name):
        """
        Get current injuries for a team
        
        Args:
            team_name: Team name (e.g., 'Man United')
        
        Returns:
            dict with injury data or None
        """
        
        # Normalize team name
        team_name = self.team_map.get(team_name, team_name)
        
        # Check cache first (valid for 24 hours)
        cached = self._load_from_cache(team_name)
        if cached:
            return cached
        
        # Fetch fresh data
        injuries = self._scrape_team_injuries(team_name)
        
        if injuries:
            self._save_to_cache(team_name, injuries)
        
        return injuries
    
    def _scrape_team_injuries(self, team_name):
        """Scrape injuries from Premier Injuries website"""
        
        try:
            print(f"   Fetching injuries for {team_name}...")
            
            # Premier Injuries URL (free, no API key needed)
            # Format: premierinjuries.com/injury-table.php
            url = "https://www.premierinjuries.com/injury-table.php"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                print(f"   ⚠️  Failed to fetch injuries (status {response.status_code})")
                return self._get_fallback_injuries(team_name)
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find table with injuries
            injuries = []
            
            # Look for team section
            team_found = False
            for row in soup.find_all('tr'):
                cells = row.find_all('td')
                
                if not cells:
                    continue
                
                # Check if this row mentions the team
                row_text = row.get_text()
                
                if team_name.lower() in row_text.lower():
                    team_found = True
                    
                    # Try to extract player name, injury type, and return date
                    try:
                        player_name = cells[1].get_text(strip=True) if len(cells) > 1 else "Unknown"
                        injury_type = cells[2].get_text(strip=True) if len(cells) > 2 else "Unknown"
                        return_date = cells[3].get_text(strip=True) if len(cells) > 3 else "Unknown"
                        
                        if player_name and player_name != "Unknown":
                            injuries.append({
                                'player': player_name,
                                'injury': injury_type,
                                'return_date': return_date,
                                'status': 'Out'
                            })
                    except:
                        continue
            
            if injuries:
                print(f"   ✅ Found {len(injuries)} injuries for {team_name}")
                return {
                    'team': team_name,
                    'injuries': injuries,
                    'last_updated': datetime.now().isoformat(),
                    'source': 'Premier Injuries'
                }
            else:
                print(f"   ℹ️  No injuries found for {team_name}")
                return {
                    'team': team_name,
                    'injuries': [],
                    'last_updated': datetime.now().isoformat(),
                    'source': 'Premier Injuries'
                }
        
        except Exception as e:
            print(f"   ❌ Error scraping injuries: {e}")
            return self._get_fallback_injuries(team_name)
    
    def _get_fallback_injuries(self, team_name):
        """
        Return known injuries when scraping fails
        
        This is manually updated with major injuries
        """
        
        known_injuries = {
            'Everton': {
                'team': 'Everton',
                'injuries': [
                    {'player': 'Kieran Dowell', 'injury': 'Knee', 'return_date': 'January 2025', 'status': 'Out'},
                    {'player': 'Youssef Chermiti', 'injury': 'Ankle', 'return_date': 'Late December', 'status': 'Out'},
                    {'player': 'Tim Iroegbunam', 'injury': 'Foot', 'return_date': 'January 2025', 'status': 'Out'}
                ],
                'last_updated': datetime.now().isoformat(),
                'source': 'Fallback'
            },
            'Man City': {
                'team': 'Man City',
                'injuries': [
                    {'player': 'Rodri', 'injury': 'ACL', 'return_date': 'End of season', 'status': 'Out'},
                    {'player': 'Oscar Bobb', 'injury': 'Leg', 'return_date': 'Early 2025', 'status': 'Out'}
                ],
                'last_updated': datetime.now().isoformat(),
                'source': 'Fallback'
            },
            # Add more teams as needed
        }
        
        return known_injuries.get(team_name, {
            'team': team_name,
            'injuries': [],
            'last_updated': datetime.now().isoformat(),
            'source': 'Fallback'
        })
    
    def _load_from_cache(self, team_name):
        """Load injuries from cache if recent (< 24 hours)"""
        
        cache_file = os.path.join(self.cache_dir, f"{team_name.replace(' ', '_')}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Check if cache is fresh (< 24 hours)
            last_updated = datetime.fromisoformat(data['last_updated'])
            age = datetime.now() - last_updated
            
            if age < timedelta(hours=24):
                print(f"   📂 Using cached injuries for {team_name}")
                return data
            else:
                print(f"   ⏰ Cache expired for {team_name} (age: {age.total_seconds()/3600:.1f}h)")
                return None
        
        except Exception as e:
            print(f"   ⚠️  Error loading cache: {e}")
            return None
    
    def _save_to_cache(self, team_name, data):
        """Save injury data to cache"""
        
        cache_file = os.path.join(self.cache_dir, f"{team_name.replace(' ', '_')}.json")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"   💾 Cached injuries for {team_name}")
        except Exception as e:
            print(f"   ⚠️  Error saving cache: {e}")
    
    def calculate_injury_impact(self, team_name):
        """
        Calculate injury impact on team performance
        
        Returns:
            float: Adjustment factor (e.g., -0.05 for 5% weaker)
        """
        
        injuries = self.get_team_injuries(team_name)
        
        if not injuries or not injuries.get('injuries'):
            return 0.0  # No injury impact
        
        # Simple scoring: -2% per injured player (max -10%)
        num_injuries = len(injuries['injuries'])
        impact = min(num_injuries * -0.02, -0.10)
        
        return impact
    
    def get_injury_impact(self, team_name):
        """
        Alias for calculate_injury_impact for compatibility
        
        Returns:
            float: Adjustment factor
        """
        return self.calculate_injury_impact(team_name)
    
    def get_injury_count(self, team_name):
        """
        Get number of injuries for a team
        
        Returns:
            int: Number of injured players
        """
        injuries = self.get_team_injuries(team_name)
        
        if not injuries or not injuries.get('injuries'):
            return 0
        
        return len(injuries['injuries'])

# Testing
if __name__ == "__main__":
    scraper = InjuryScraper()
    
    # Test with a few teams
    for team in ['Man United', 'Everton', 'Arsenal']:
        print(f"\n{'='*60}")
        print(f"Testing: {team}")
        print('='*60)
        
        injuries = scraper.get_team_injuries(team)
        
        if injuries:
            print(f"\nTeam: {injuries['team']}")
            print(f"Injuries: {len(injuries['injuries'])}")
            print(f"Source: {injuries['source']}")
            print(f"Updated: {injuries['last_updated']}")
            
            if injuries['injuries']:
                print("\nInjured Players:")
                for inj in injuries['injuries']:
                    print(f"  - {inj['player']}: {inj['injury']} (Return: {inj['return_date']})")
            
            impact = scraper.calculate_injury_impact(team)
            print(f"\nInjury Impact: {impact:+.1%}")