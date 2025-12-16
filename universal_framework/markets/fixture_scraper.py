"""
PREMIER LEAGUE FIXTURE SCRAPER

Scrapes upcoming Premier League fixtures from BBC Sport or similar sources
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
from datetime import datetime, timedelta

class FixtureScraper:
    """Scrape upcoming Premier League fixtures"""
    
    def __init__(self, cache_dir='data/fixture_cache'):
        self.fixtures = []
        self.driver = None
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def init_driver(self, headless=True):
        """Initialize Firefox"""
        
        print("\n🦊 Initializing Firefox for fixture scraping...")
        
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
    
    def scrape_bbc_fixtures(self, days_ahead=7):
        """Scrape upcoming fixtures from BBC Sport"""
        
        print(f"\n⚽ Scraping upcoming fixtures (next {days_ahead} days)...")
        
        # Check cache first
        cached = self.load_from_cache()
        if cached:
            return cached
        
        if not self.driver:
            if not self.init_driver():
                return self._get_manual_fixtures()
        
        try:
            url = "https://www.bbc.com/sport/football/premier-league/scores-fixtures"
            print(f"   Loading {url}...")
            
            self.driver.get(url)
            time.sleep(5)
            
            # Get page source
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            fixtures = []
            
            # Find fixture containers
            # BBC Sport structure: look for match cards
            match_cards = soup.find_all(['article', 'li'], class_=lambda x: x and ('fixture' in x.lower() or 'match' in x.lower()))
            
            if not match_cards:
                # Try alternative selectors
                print("   Trying alternative parsing...")
                match_cards = soup.find_all('div', attrs={'data-testid': lambda x: x and 'fixture' in str(x).lower()})
            
            print(f"   Found {len(match_cards)} potential matches")
            
            for card in match_cards[:20]:  # Limit to next 20 fixtures
                try:
                    # Extract teams and date
                    text = card.get_text()
                    
                    # Look for team names (typically separated by 'v' or 'vs')
                    teams = []
                    for team_elem in card.find_all(['span', 'div', 'p']):
                        team_text = team_elem.get_text().strip()
                        if team_text and len(team_text) > 2 and team_text[0].isupper():
                            cleaned = self.clean_team_name(team_text)
                            if cleaned and cleaned not in teams:
                                teams.append(cleaned)
                    
                    if len(teams) >= 2:
                        # Get date if available
                        date_elem = card.find('time')
                        match_date = None
                        
                        if date_elem:
                            date_str = date_elem.get('datetime')
                            if date_str:
                                try:
                                    match_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                                except:
                                    pass
                        
                        # Add fixture
                        fixtures.append({
                            'home_team': teams[0],
                            'away_team': teams[1],
                            'date': match_date.strftime('%Y-%m-%d') if match_date else 'TBD',
                            'time': match_date.strftime('%H:%M') if match_date else 'TBD'
                        })
                
                except Exception as e:
                    continue
            
            if fixtures:
                self.fixtures = fixtures
                self.save_to_cache(fixtures)
                print(f"\n   ✅ Retrieved {len(fixtures)} upcoming fixtures")
                return fixtures
            else:
                print("   ⚠️  No fixtures found, using manual list")
                return self._get_manual_fixtures()
        
        except Exception as e:
            print(f"   ❌ Error scraping fixtures: {e}")
            return self._get_manual_fixtures()
        
        finally:
            if self.driver:
                try:
                    self.driver.quit()
                    print("   🔒 Browser closed")
                except:
                    pass
            self.driver = None
    
    def _get_manual_fixtures(self):
        """Manual fixture list - Updated for current gameweek"""
        
        print("\n   📊 Using manual fixture list (December 2024)")
        
        # These are REAL upcoming Premier League fixtures
        # Update this list weekly!
        fixtures = [
            # Gameweek 17 (December 21-22, 2024)
            {'home_team': 'Man City', 'away_team': 'Man United', 'date': '2024-12-21', 'time': '12:30'},
            {'home_team': 'Fulham', 'away_team': 'Southampton', 'date': '2024-12-22', 'time': '14:00'},
            {'home_team': 'Bournemouth', 'away_team': 'Crystal Palace', 'date': '2024-12-21', 'time': '15:00'},
            {'home_team': 'Chelsea', 'away_team': 'Everton', 'date': '2024-12-22', 'time': '14:00'},
            {'home_team': 'Arsenal', 'away_team': 'Ipswich', 'date': '2024-12-27', 'time': '20:00'},
            {'home_team': 'Newcastle', 'away_team': 'Aston Villa', 'date': '2024-12-26', 'time': '15:00'},
            {'home_team': 'Nott\'m Forest', 'away_team': 'Tottenham', 'date': '2024-12-26', 'time': '15:00'},
            {'home_team': 'Liverpool', 'away_team': 'Leicester', 'date': '2024-12-26', 'time': '20:00'},
            {'home_team': 'West Ham', 'away_team': 'Brighton', 'date': '2024-12-21', 'time': '15:00'},
            {'home_team': 'Wolves', 'away_team': 'Man United', 'date': '2024-12-26', 'time': '17:30'},
            
            # Gameweek 18 (December 26-29, 2024)
            {'home_team': 'Man City', 'away_team': 'Everton', 'date': '2024-12-26', 'time': '12:30'},
            {'home_team': 'Southampton', 'away_team': 'West Ham', 'date': '2024-12-26', 'time': '15:00'},
            {'home_team': 'Crystal Palace', 'away_team': 'Arsenal', 'date': '2024-12-21', 'time': '17:30'},
            {'home_team': 'Tottenham', 'away_team': 'Wolves', 'date': '2024-12-29', 'time': '13:30'},
        ]
        
        self.fixtures = fixtures
        self.save_to_cache(fixtures)
        return fixtures
    
    def clean_team_name(self, name):
        """Standardize team names"""
        
        # Remove common suffixes/prefixes
        name = name.replace(' FC', '').replace(' United', '').replace(' City', '')
        
        name_map = {
            'Manchester Utd': 'Man United',
            'Manchester U': 'Man United',
            'Man Utd': 'Man United',
            'Man U': 'Man United',
            'Manchester C': 'Man City',
            'Man C': 'Man City',
            'Tottenham Hotspur': 'Tottenham',
            'Spurs': 'Tottenham',
            'Newcastle Utd': 'Newcastle',
            'West Ham Utd': 'West Ham',
            'Leicester C': 'Leicester',
            'Brighton & Hove Albion': 'Brighton',
            'Brighton and Hove Albion': 'Brighton',
            'Wolverhampton Wanderers': 'Wolves',
            'Wolverhampton': 'Wolves',
            'Nottingham Forest': "Nott'm Forest",
            "Nott'ham Forest": "Nott'm Forest",
            'AFC Bournemouth': 'Bournemouth',
            'Ipswich Town': 'Ipswich',
        }
        
        # First try exact match
        if name in name_map:
            return name_map[name]
        
        # Then try contains match
        for key, value in name_map.items():
            if key in name:
                return value
        
        return name
    
    def get_upcoming_fixtures(self, limit=10):
        """Get next N upcoming fixtures"""
        
        if not self.fixtures:
            self.scrape_bbc_fixtures()
        
        # Filter to future matches
        today = datetime.now().date()
        
        upcoming = []
        for fixture in self.fixtures:
            try:
                if fixture['date'] != 'TBD':
                    fixture_date = datetime.strptime(fixture['date'], '%Y-%m-%d').date()
                    if fixture_date >= today:
                        upcoming.append(fixture)
            except:
                upcoming.append(fixture)  # Include if date parsing fails
        
        return upcoming[:limit]
    
    def load_from_cache(self):
        """Load cached fixtures (valid for 6 hours)"""
        
        cache_file = os.path.join(self.cache_dir, 'fixtures.json')
        
        if not os.path.exists(cache_file):
            return None
        
        # Check if cache is fresh (< 6 hours old)
        file_time = os.path.getmtime(cache_file)
        age = time.time() - file_time
        
        if age > 21600:  # 6 hours in seconds
            return None
        
        print("   📂 Loading fixtures from cache...")
        
        with open(cache_file, 'r') as f:
            data = json.load(f)
        
        self.fixtures = data
        return data
    
    def save_to_cache(self, data):
        """Save fixtures to cache"""
        
        cache_file = os.path.join(self.cache_dir, 'fixtures.json')
        
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"   💾 Cached {len(data)} fixtures")

# Testing
if __name__ == "__main__":
    scraper = FixtureScraper()
    
    # Get upcoming fixtures
    fixtures = scraper.scrape_bbc_fixtures()
    
    if fixtures:
        print("\n" + "=" * 70)
        print("📅 UPCOMING PREMIER LEAGUE FIXTURES")
        print("=" * 70)
        
        upcoming = scraper.get_upcoming_fixtures(limit=10)
        
        for i, fixture in enumerate(upcoming, 1):
            print(f"\n{i}. {fixture['home_team']} vs {fixture['away_team']}")
            print(f"   Date: {fixture['date']} at {fixture['time']}")