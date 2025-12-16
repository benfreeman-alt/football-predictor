"""
INJURY DATA SCRAPER

Scrapes injury data from Premier Injuries
Tracks key player absences
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
import os
from datetime import datetime

class InjuryScraper:
    """Scrape injury data from Premier Injuries"""
    
    def __init__(self, cache_dir='data/injury_cache'):
        self.injury_data = {}
        self.driver = None
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def init_driver(self, headless=True):
        """Initialize Firefox"""
        
        print("\n🦊 Initializing Firefox for injury scraping...")
        
        try:
            firefox_options = Options()
            if headless:
                firefox_options.add_argument('--headless')
            
            try:
                from webdriver_manager.firefox import GeckoDriverManager
                service = Service(GeckoDriverManager().install())
                self.driver = webdriver.Firefox(service=service, options=firefox_options)
                print("   ✅ Firefox initialized")
            except ImportError:
                self.driver = webdriver.Firefox(options=firefox_options)
                print("   ✅ Firefox initialized")
            
            return True
        
        except Exception as e:
            print(f"   ❌ Could not initialize: {e}")
            return False
    
    def scrape_premier_league_injuries(self):
        """Scrape current Premier League injuries"""
        
        print("\n🏥 Scraping Premier League injury data...")
        
        # Check cache first (valid for 24 hours)
        cached = self.load_from_cache()
        if cached:
            return cached
        
        if not self.driver:
            if not self.init_driver():
                return self._get_default_injuries()
        
        try:
            url = "https://www.premierinjuries.com/injury-table.php"
            print(f"   Loading {url}...")
            
            self.driver.get(url)
            time.sleep(3)
            
            # Get page source
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Find injury table
            injury_data = {}
            
            # Look for team sections
            teams = soup.find_all('div', class_='team-injuries')
            
            if not teams:
                # Try alternative parsing
                print("   Trying alternative parsing method...")
                injury_data = self._parse_alternative(soup)
            else:
                for team_section in teams:
                    try:
                        team_name = team_section.find('h3').text.strip()
                        team_name = self.clean_team_name(team_name)
                        
                        injuries = []
                        injury_rows = team_section.find_all('tr')
                        
                        for row in injury_rows[1:]:  # Skip header
                            cols = row.find_all('td')
                            if len(cols) >= 3:
                                player = cols[0].text.strip()
                                injury_type = cols[1].text.strip()
                                status = cols[2].text.strip()
                                
                                injuries.append({
                                    'player': player,
                                    'injury': injury_type,
                                    'status': status
                                })
                        
                        injury_data[team_name] = injuries
                    
                    except Exception as e:
                        continue
            
            if injury_data:
                self.injury_data = injury_data
                self.save_to_cache(injury_data)
                print(f"\n   ✅ Retrieved injury data for {len(injury_data)} teams")
                return injury_data
            else:
                print("   ⚠️  No injury data found, using defaults")
                return self._get_default_injuries()
        
        except Exception as e:
            print(f"   ❌ Error scraping injuries: {e}")
            return self._get_default_injuries()
        
        finally:
            if self.driver:
                try:
                    self.driver.quit()
                    print("   🔒 Browser closed")
                except:
                    pass
            self.driver = None
    
    def _parse_alternative(self, soup):
        """Alternative parsing method"""
        
        injury_data = {}
        
        try:
            # Try to find all tables
            tables = soup.find_all('table')
            
            for table in tables:
                rows = table.find_all('tr')
                
                for row in rows:
                    cols = row.find_all('td')
                    
                    if len(cols) >= 4:
                        try:
                            team_name = cols[0].text.strip()
                            player = cols[1].text.strip()
                            injury = cols[2].text.strip()
                            status = cols[3].text.strip()
                            
                            team_name = self.clean_team_name(team_name)
                            
                            if team_name not in injury_data:
                                injury_data[team_name] = []
                            
                            injury_data[team_name].append({
                                'player': player,
                                'injury': injury,
                                'status': status
                            })
                        
                        except:
                            continue
        
        except Exception as e:
            pass
        
        return injury_data
    
    def _get_default_injuries(self):
        """Return default (no injuries) for all teams"""
        
        print("\n   📊 Using default injury data (no injuries)")
        
        teams = [
            'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
            'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich',
            'Leicester', 'Liverpool', 'Man City', 'Man United', 'Newcastle',
            "Nott'm Forest", 'Southampton', 'Tottenham', 'West Ham', 'Wolves'
        ]
        
        return {team: [] for team in teams}
    
    def clean_team_name(self, name):
        """Standardize team names"""
        
        name_map = {
            'Manchester United': 'Man United',
            'Manchester City': 'Man City',
            'Tottenham Hotspur': 'Tottenham',
            'Newcastle United': 'Newcastle',
            'West Ham United': 'West Ham',
            'Leicester City': 'Leicester',
            'Brighton & Hove Albion': 'Brighton',
            'Brighton and Hove Albion': 'Brighton',
            'Wolverhampton Wanderers': 'Wolves',
            'Nottingham Forest': "Nott'm Forest",
            'AFC Bournemouth': 'Bournemouth',
            'Ipswich Town': 'Ipswich',
        }
        
        return name_map.get(name, name)
    
    def get_injury_impact(self, team_name):
        """
        Calculate injury impact score
        
        Returns:
            impact_score: 0.0 (no impact) to 1.0 (severe impact)
        """
        
        if team_name not in self.injury_data:
            return 0.0
        
        injuries = self.injury_data[team_name]
        
        if not injuries:
            return 0.0
        
        # Weight by severity keywords
        total_impact = 0
        
        for injury in injuries:
            status = injury.get('status', '').lower()
            injury_type = injury.get('injury', '').lower()
            
            # Severity scoring
            if 'doubt' in status or 'late test' in status:
                impact = 0.3
            elif 'out' in status or 'injured' in status:
                impact = 0.5
            elif 'long' in status or 'season' in status:
                impact = 0.7
            else:
                impact = 0.2
            
            # Boost for key positions (rough heuristic)
            if any(word in injury_type.lower() for word in ['hamstring', 'knee', 'acl']):
                impact *= 1.2
            
            total_impact += impact
        
        # Cap at 1.0
        return min(total_impact / 5, 1.0)  # Normalize by expected max of 5 injuries
    
    def get_injury_count(self, team_name):
        """Get number of injuries for a team"""
        
        if team_name not in self.injury_data:
            return 0
        
        return len(self.injury_data[team_name])
    
    def load_from_cache(self):
        """Load cached injury data (valid for 24 hours)"""
        
        cache_file = os.path.join(self.cache_dir, 'current_injuries.json')
        
        if not os.path.exists(cache_file):
            return None
        
        # Check if cache is fresh (< 24 hours old)
        file_time = os.path.getmtime(cache_file)
        age = time.time() - file_time
        
        if age > 86400:  # 24 hours in seconds
            return None
        
        print("   📂 Loading injury data from cache...")
        
        with open(cache_file, 'r') as f:
            data = json.load(f)
        
        self.injury_data = data
        return data
    
    def save_to_cache(self, data):
        """Save injury data to cache"""
        
        cache_file = os.path.join(self.cache_dir, 'current_injuries.json')
        
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"   💾 Cached injury data")

# Testing
if __name__ == "__main__":
    scraper = InjuryScraper()
    
    # Scrape current injuries
    injuries = scraper.scrape_premier_league_injuries()
    
    if injuries:
        print("\n" + "=" * 70)
        print("📊 INJURY REPORT")
        print("=" * 70)
        
        for team, team_injuries in sorted(injuries.items()):
            if team_injuries:
                impact = scraper.get_injury_impact(team)
                print(f"\n{team}: {len(team_injuries)} injuries (Impact: {impact:.2f})")
                
                for injury in team_injuries[:3]:  # Show top 3
                    print(f"  - {injury.get('player', 'Unknown')}: {injury.get('injury', 'Unknown')} ({injury.get('status', 'Unknown')})")
        
        # Test impact scores
        print("\n" + "=" * 70)
        print("TEAMS MOST AFFECTED BY INJURIES")
        print("=" * 70)
        
        impact_scores = [(team, scraper.get_injury_impact(team)) for team in injuries.keys()]
        impact_scores.sort(key=lambda x: x[1], reverse=True)
        
        for team, impact in impact_scores[:10]:
            if impact > 0:
                print(f"{team:20s}: Impact {impact:.2f} ({scraper.get_injury_count(team)} injuries)")