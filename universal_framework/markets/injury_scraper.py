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
        """Scrape injuries from PhysioRoom website"""
        
        try:
            print(f"   🔍 Fetching injuries for {team_name}...")
            
            # PhysioRoom URL - more reliable for scraping
            url = "https://www.physioroom.com/news/english_premier_league/epl_injury_table.php"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://www.physioroom.com/',
            }
            
            print(f"   Fetching from: {url}")
            response = requests.get(url, headers=headers, timeout=15)
            print(f"   Response status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"   ⚠️  Failed to fetch injuries (status {response.status_code})")
                print(f"   Using fallback (empty injuries)...")
                return self._get_fallback_injuries(team_name)
            
            print(f"   ✅ Page loaded successfully, parsing HTML...")
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Debug: Show what team names are actually on the page
            page_text = soup.get_text().lower()
            possible_variations = [
                team_name,
                team_name.replace("'", ""),  # Nott'm → Nottm
                team_name.replace(" ", ""),   # Man City → ManCity
                team_name.split()[0] if " " in team_name else team_name,  # Man City → Man
            ]
            
            # Add common variations
            name_map = {
                "Man City": ["manchester city", "man city", "man. city"],
                "Man United": ["manchester united", "man united", "man utd", "man. united"],
                "Tottenham": ["tottenham", "spurs", "tottenham hotspur"],
                "Newcastle": ["newcastle", "newcastle united"],
                "West Ham": ["west ham", "west ham united"],
                "Brighton": ["brighton", "brighton & hove albion", "brighton and hove albion"],
                "Nott'm Forest": ["nottingham forest", "nottm forest", "nott'm forest", "forest"],
                "Leicester": ["leicester", "leicester city"],
                "Wolves": ["wolves", "wolverhampton", "wolverhampton wanderers"],
                "Bournemouth": ["bournemouth", "afc bournemouth"],
                "Ipswich": ["ipswich", "ipswich town"],
            }
            
            # Get variations for this team
            search_terms = name_map.get(team_name, [team_name.lower()])
            
            print(f"   Searching for variations: {search_terms[:3]}")
            
            # Find which variation exists on page
            found_variation = None
            for variation in search_terms:
                if variation.lower() in page_text:
                    found_variation = variation
                    print(f"   ✅ Found '{variation}' in page content")
                    break
            
            if not found_variation:
                print(f"   ⚠️  None of the variations found on page")
                # Debug: show what teams ARE on the page
                prem_teams = ['arsenal', 'chelsea', 'liverpool', 'man city', 'man united', 
                             'tottenham', 'newcastle', 'manchester', 'brighton', 'villa']
                found_teams = [t for t in prem_teams if t in page_text]
                print(f"   Teams found on page: {found_teams[:5]}")
                return self._get_fallback_injuries(team_name)
            
            injuries = []
            
            # Method 1: Find team name in headers using the found variation
            team_headers = soup.find_all(['h3', 'h4', 'h2', 'div', 'p', 'span', 'strong', 'b'], 
                                        string=lambda text: text and found_variation.lower() in text.lower() if text else False)
            
            print(f"   Found {len(team_headers)} potential team headers for '{found_variation}'")
            
            injury_table = None
            
            if team_headers:
                # Get the first matching header
                team_header = team_headers[0]
                print(f"   ✅ Found team header: {team_header.name} - {team_header.get_text(strip=True)[:50]}")
                
                # Find the next table after this header
                current = team_header
                
                # Walk through siblings to find the table
                for sibling in team_header.find_next_siblings():
                    if sibling.name == 'table':
                        injury_table = sibling
                        break
                    # Stop if we hit another team header
                    if sibling.name in ['h3', 'h4'] and sibling != team_header:
                        break
                
                if not injury_table:
                    # Try finding parent's next table
                    parent = team_header.parent
                    if parent:
                        injury_table = parent.find_next('table')
            
            # Method 2 (Fallback): Look through all tables for team name in rows
            if not injury_table:
                print(f"   ⚠️  Header method failed, trying table scan...")
                all_tables = soup.find_all('table')
                print(f"   Scanning {len(all_tables)} tables...")
                
                for table in all_tables:
                    # Check if table or nearby text contains team name
                    table_context = str(table.parent)[:2000].lower()  # Check parent context
                    
                    # Use the found variation
                    if found_variation and found_variation.lower() in table_context:
                        # Make sure it's not just the team name appearing in injury data
                        # Check if there's actual player data
                        rows = table.find_all('tr')
                        if len(rows) > 1:  # Has more than just header
                            print(f"   ✅ Found potential table via scan (matched '{found_variation}')")
                            injury_table = table
                            break
            
            if not injury_table:
                print(f"   ⚠️  No injury table found after team header")
                return {
                    'team': team_name,
                    'injuries': [],
                    'last_updated': datetime.now().isoformat(),
                    'source': 'PhysioRoom (scraped - no injuries)'
                }
            
            print(f"   ✅ Found injury table for {team_name}")
            
            # Parse the table
            rows = injury_table.find_all('tr')
            print(f"   Processing {len(rows)} rows...")
            
            for row in rows[1:]:  # Skip header row
                cols = row.find_all('td')
                
                if len(cols) >= 2:
                    try:
                        player_name = cols[0].get_text(strip=True)
                        injury_type = cols[1].get_text(strip=True)
                        return_date = cols[2].get_text(strip=True) if len(cols) > 2 else "Unknown"
                        
                        # Skip header rows, empty cells, or team names
                        skip_values = ['Player', 'Name', '', 'Injury', 'Status', 'Return', team_name]
                        if player_name and player_name not in skip_values and len(player_name) > 2:
                            injuries.append({
                                'player': player_name,
                                'injury': injury_type,
                                'return_date': return_date,
                                'status': 'Out'
                            })
                            print(f"      • {player_name}: {injury_type}")
                    except Exception as e:
                        print(f"      ⚠️  Error parsing row: {e}")
                        continue
            
            if injuries:
                print(f"   ✅ Scraped {len(injuries)} injuries for {team_name}")
                return {
                    'team': team_name,
                    'injuries': injuries,
                    'last_updated': datetime.now().isoformat(),
                    'source': 'PhysioRoom (scraped)'
                }
            else:
                print(f"   ℹ️  No injuries found for {team_name}")
                return {
                    'team': team_name,
                    'injuries': [],
                    'last_updated': datetime.now().isoformat(),
                    'source': 'PhysioRoom (scraped - no injuries)'
                }
        
        except requests.exceptions.Timeout:
            print(f"   ❌ Request timeout after 15 seconds")
            return self._get_fallback_injuries(team_name)
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Network error: {e}")
            return self._get_fallback_injuries(team_name)
        except Exception as e:
            print(f"   ❌ Error scraping injuries: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_injuries(team_name)
    
    def _find_teams_on_page(self, soup):
        """Debug helper: Find what teams are mentioned on the page"""
        try:
            page_text = soup.get_text().lower()
            premier_league_teams = [
                'arsenal', 'aston villa', 'bournemouth', 'brentford', 'brighton',
                'chelsea', 'crystal palace', 'everton', 'fulham', 'ipswich',
                'leicester', 'liverpool', 'manchester city', 'manchester united',
                'newcastle', 'nottingham forest', 'southampton', 'tottenham',
                'west ham', 'wolves'
            ]
            found_teams = [team for team in premier_league_teams if team in page_text]
            return found_teams[:5]  # Return first 5 found
        except:
            return []
    
    def _get_fallback_injuries(self, team_name):
        """
        Return empty injuries when scraping fails
        
        The scraper will keep trying to fetch live data on next refresh
        """
        
        print(f"   📝 Returning empty injuries for {team_name} (will retry next time)")
        
        return {
            'team': team_name,
            'injuries': [],
            'last_updated': datetime.now().isoformat(),
            'source': 'Fallback (empty)'
        }
    
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