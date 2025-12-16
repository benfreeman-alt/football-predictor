"""
BBC SPORT FIXTURE SCRAPER

Scrapes REAL upcoming Premier League fixtures from BBC Sport
No API key needed - always up-to-date!
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from bs4 import BeautifulSoup
import time
import json
import os
from datetime import datetime, timedelta

class BBCFixtureScraper:
    """Scrape upcoming fixtures from BBC Sport"""
    
    def __init__(self, cache_dir='data/fixture_cache'):
        self.fixtures = []
        self.driver = None
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def init_driver(self, headless=True):
        """Initialize Firefox"""
        
        print("\n🦊 Initializing Firefox...")
        
        try:
            firefox_options = Options()
            if headless:
                firefox_options.add_argument('--headless')
            
            try:
                from webdriver_manager.firefox import GeckoDriverManager
                service = Service(GeckoDriverManager().install())
                self.driver = webdriver.Firefox(service=service, options=firefox_options)
            except ImportError:
                self.driver = webdriver.Firefox(options=firefox_options)
            
            return True
        
        except Exception as e:
            print(f"   ❌ Could not initialize: {e}")
            return False
    
    def scrape_upcoming_fixtures(self, days_ahead=14):
        """Scrape upcoming Premier League fixtures from BBC Sport"""
        
        print(f"\n⚽ Scraping upcoming fixtures from BBC Sport...")
        
        # Check cache first (valid for 6 hours)
        cached = self._load_from_cache()
        if cached:
            print(f"   📂 Using cached fixtures ({len(cached)} fixtures)")
            return cached
        
        if not self.driver:
            if not self.init_driver():
                return []
        
        try:
            url = "https://www.bbc.com/sport/football/premier-league/scores-fixtures"
            print(f"   Loading {url}...")
            
            self.driver.get(url)
            time.sleep(5)  # Wait for dynamic content
            
            # Get page source
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            fixtures = []
            today = datetime.now()
            cutoff_date = today + timedelta(days=days_ahead)
            
            # Find all fixture containers
            # BBC uses different structures, try multiple selectors
            
            # Try method 1: data-testid
            fixture_containers = soup.find_all(attrs={'data-testid': lambda x: x and 'FIXTURE' in str(x).upper()})
            
            if not fixture_containers:
                # Try method 2: class-based
                fixture_containers = soup.find_all('li', class_=lambda x: x and ('fixture' in str(x).lower() or 'match' in str(x).lower()))
            
            if not fixture_containers:
                # Try method 3: article tags
                fixture_containers = soup.find_all('article')
            
            print(f"   Found {len(fixture_containers)} potential fixtures")
            
            for container in fixture_containers:
                try:
                    fixture_data = self._parse_fixture(container)
                    
                    if fixture_data:
                        # Filter to upcoming only
                        if fixture_data.get('date') and fixture_data['date'] != 'TBD':
                            try:
                                fixture_date = datetime.strptime(fixture_data['date'], '%Y-%m-%d')
                                if today <= fixture_date <= cutoff_date:
                                    fixtures.append(fixture_data)
                            except:
                                fixtures.append(fixture_data)  # Include if date parsing fails
                        else:
                            fixtures.append(fixture_data)  # Include TBD fixtures
                
                except Exception as e:
                    continue
            
            # Remove duplicates
            seen = set()
            unique_fixtures = []
            for f in fixtures:
                key = (f['home_team'], f['away_team'], f['date'])
                if key not in seen:
                    seen.add(key)
                    unique_fixtures.append(f)
            
            if unique_fixtures:
                self.fixtures = unique_fixtures
                self._save_to_cache(unique_fixtures)
                print(f"   ✅ Retrieved {len(unique_fixtures)} upcoming fixtures")
                return unique_fixtures
            else:
                print(f"   ⚠️  No fixtures found")
                return []
        
        except Exception as e:
            print(f"   ❌ Error scraping: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        finally:
            if self.driver:
                try:
                    self.driver.quit()
                    print("   🔒 Browser closed")
                except:
                    pass
            self.driver = None
    
    def _parse_fixture(self, container):
        """Extract fixture data from container"""
        
        try:
            # Get all text from container
            text = container.get_text(' ', strip=True)
            
            # Look for team names
            teams = []
            
            # Try to find team elements
            team_elements = container.find_all(['span', 'div', 'p', 'a'])
            
            for elem in team_elements:
                elem_text = elem.get_text(strip=True)
                
                # Check if this looks like a team name
                if elem_text and len(elem_text) > 2:
                    # Common team names
                    known_teams = [
                        'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
                        'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich',
                        'Leicester', 'Liverpool', 'Man City', 'Man Utd', 'Manchester City',
                        'Manchester United', 'Newcastle', 'Nottingham Forest', 'Southampton',
                        'Tottenham', 'West Ham', 'Wolves'
                    ]
                    
                    # Check if text contains a team name
                    for team in known_teams:
                        if team.lower() in elem_text.lower() and elem_text not in teams:
                            teams.append(self._clean_team_name(elem_text))
                            break
            
            # Remove duplicates while preserving order
            seen = set()
            unique_teams = []
            for team in teams:
                if team not in seen:
                    seen.add(team)
                    unique_teams.append(team)
            
            if len(unique_teams) >= 2:
                # Try to extract date/time
                date_str = 'TBD'
                time_str = 'TBD'
                
                # Look for time element
                time_elem = container.find('time')
                if time_elem:
                    datetime_attr = time_elem.get('datetime')
                    if datetime_attr:
                        try:
                            dt = datetime.fromisoformat(datetime_attr.replace('Z', '+00:00'))
                            date_str = dt.strftime('%Y-%m-%d')
                            time_str = dt.strftime('%H:%M')
                        except:
                            pass
                
                return {
                    'home_team': unique_teams[0],
                    'away_team': unique_teams[1],
                    'date': date_str,
                    'time': time_str,
                    'venue': 'TBD',
                    'status': 'NS'
                }
        
        except Exception as e:
            pass
        
        return None
    
    def _clean_team_name(self, name):
        """Standardize team names"""
        
        # Remove common suffixes
        name = name.replace(' FC', '').replace(' United', '').strip()
        
        name_map = {
            'Manchester Utd': 'Man United',
            'Man Utd': 'Man United',
            'Man U': 'Man United',
            'Manchester U': 'Man United',
            'Manchester City': 'Man City',
            'Man C': 'Man City',
            'Tottenham Hotspur': 'Tottenham',
            'Spurs': 'Tottenham',
            'Newcastle Utd': 'Newcastle',
            'West Ham Utd': 'West Ham',
            'Leicester C': 'Leicester',
            'Leicester City': 'Leicester',
            'Brighton & Hove Albion': 'Brighton',
            'Brighton and Hove Albion': 'Brighton',
            'Wolverhampton Wanderers': 'Wolves',
            'Wolverhampton': 'Wolves',
            'Nottingham Forest': "Nott'm Forest",
            "Nott'ham Forest": "Nott'm Forest",
            'AFC Bournemouth': 'Bournemouth',
            'Ipswich Town': 'Ipswich',
        }
        
        # Check map
        if name in name_map:
            return name_map[name]
        
        # Check contains
        for key, value in name_map.items():
            if key in name:
                return value
        
        return name
    
    def _load_from_cache(self):
        """Load cached fixtures (valid for 6 hours)"""
        
        cache_file = os.path.join(self.cache_dir, 'bbc_fixtures.json')
        
        if not os.path.exists(cache_file):
            return None
        
        # Check if cache is fresh
        import time
        file_time = os.path.getmtime(cache_file)
        age = time.time() - file_time
        
        if age > 21600:  # 6 hours
            return None
        
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    def _save_to_cache(self, fixtures):
        """Save fixtures to cache"""
        
        cache_file = os.path.join(self.cache_dir, 'bbc_fixtures.json')
        
        with open(cache_file, 'w') as f:
            json.dump(fixtures, f, indent=2)
        
        print(f"   💾 Cached {len(fixtures)} fixtures")

# Testing
if __name__ == "__main__":
    scraper = BBCFixtureScraper()
    fixtures = scraper.scrape_upcoming_fixtures(days_ahead=14)
    
    if fixtures:
        print("\n" + "=" * 70)
        print("📅 UPCOMING PREMIER LEAGUE FIXTURES")
        print("=" * 70)
        
        for i, f in enumerate(fixtures[:15], 1):
            print(f"\n{i:2d}. {f['home_team']} vs {f['away_team']}")
            print(f"    Date: {f['date']} at {f['time']}")
    else:
        print("\n❌ No fixtures found - scraping failed")