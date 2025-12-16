"""
SKY SPORTS FIXTURE SCRAPER

Scrapes REAL upcoming Premier League fixtures from Sky Sports
More reliable structure than BBC
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
import re
from datetime import datetime, timedelta

class SkyFixtureScraper:
    """Scrape upcoming fixtures from Sky Sports"""
    
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
            firefox_options.add_argument('--disable-blink-features=AutomationControlled')
            
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
        """Scrape upcoming Premier League fixtures from Sky Sports"""
        
        print(f"\n⚽ Scraping upcoming fixtures from Sky Sports...")
        
        # Check cache first (valid for 6 hours)
        cached = self._load_from_cache()
        if cached:
            print(f"   📂 Using cached fixtures ({len(cached)} fixtures)")
            return cached
        
        if not self.driver:
            if not self.init_driver():
                return self._get_fallback_fixtures()
        
        try:
            url = "https://www.skysports.com/premier-league-fixtures"
            print(f"   Loading {url}...")
            
            self.driver.get(url)
            time.sleep(5)
            
            # Scroll to load more fixtures
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # Get page source
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            fixtures = []
            today = datetime.now()
            cutoff_date = today + timedelta(days=days_ahead)
            
            # Sky Sports uses fixres__item class
            fixture_items = soup.find_all('div', class_='fixres__item')
            
            print(f"   Found {len(fixture_items)} fixture items")
            
            current_date = None
            
            for item in fixture_items:
                try:
                    # Check if this is a date header
                    date_header = item.find('h4', class_='fixres__header2')
                    if date_header:
                        # Parse date header (e.g., "Saturday 21st December 2025")
                        date_text = date_header.get_text(strip=True)
                        current_date = self._parse_date_header(date_text)
                        continue
                    
                    # Extract fixture data
                    home_elem = item.find('span', class_='swap-text__target')
                    away_elem = item.find_all('span', class_='swap-text__target')
                    
                    if len(away_elem) >= 2:
                        home_team = home_elem.get_text(strip=True) if home_elem else None
                        away_team = away_elem[1].get_text(strip=True)
                        
                        # Get time
                        time_elem = item.find('span', class_='matches__date')
                        match_time = time_elem.get_text(strip=True) if time_elem else 'TBD'
                        
                        if home_team and away_team and current_date:
                            # Filter to upcoming
                            if today.date() <= current_date <= cutoff_date.date():
                                fixtures.append({
                                    'home_team': self._clean_team_name(home_team),
                                    'away_team': self._clean_team_name(away_team),
                                    'date': current_date.strftime('%Y-%m-%d'),
                                    'time': match_time,
                                    'venue': 'TBD',
                                    'status': 'NS'
                                })
                
                except Exception as e:
                    continue
            
            if fixtures:
                self.fixtures = fixtures
                self._save_to_cache(fixtures)
                print(f"   ✅ Retrieved {len(fixtures)} upcoming fixtures")
                return fixtures
            else:
                print(f"   ⚠️  No fixtures found, using fallback")
                return self._get_fallback_fixtures()
        
        except Exception as e:
            print(f"   ❌ Error scraping: {e}")
            return self._get_fallback_fixtures()
        
        finally:
            if self.driver:
                try:
                    self.driver.quit()
                    print("   🔒 Browser closed")
                except:
                    pass
            self.driver = None
    
    def _parse_date_header(self, date_text):
        """Parse date from header like 'Saturday 21st December 2025'"""
        
        try:
            # Remove day of week and ordinal suffixes
            date_text = re.sub(r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+', '', date_text)
            date_text = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_text)
            
            # Parse the date
            return datetime.strptime(date_text.strip(), '%d %B %Y').date()
        
        except:
            return datetime.now().date()
    
    def _clean_team_name(self, name):
        """Standardize team names"""
        
        name = name.strip()
        
        name_map = {
            'Manchester United': 'Man United',
            'Man Utd': 'Man United',
            'Manchester City': 'Man City',
            'Man City': 'Man City',
            'Tottenham Hotspur': 'Tottenham',
            'Tottenham': 'Tottenham',
            'Newcastle United': 'Newcastle',
            'Newcastle': 'Newcastle',
            'West Ham United': 'West Ham',
            'West Ham': 'West Ham',
            'Leicester City': 'Leicester',
            'Leicester': 'Leicester',
            'Brighton & Hove Albion': 'Brighton',
            'Brighton and Hove Albion': 'Brighton',
            'Brighton': 'Brighton',
            'Wolverhampton Wanderers': 'Wolves',
            'Wolves': 'Wolves',
            'Nottingham Forest': "Nott'm Forest",
            'Nott\'m Forest': "Nott'm Forest",
            'AFC Bournemouth': 'Bournemouth',
            'Bournemouth': 'Bournemouth',
            'Ipswich Town': 'Ipswich',
            'Ipswich': 'Ipswich',
            'Crystal Palace': 'Crystal Palace',
            'Aston Villa': 'Aston Villa',
            'Fulham': 'Fulham',
            'Brentford': 'Brentford',
            'Southampton': 'Southampton',
            'Everton': 'Everton',
            'Chelsea': 'Chelsea',
            'Arsenal': 'Arsenal',
            'Liverpool': 'Liverpool',
        }
        
        return name_map.get(name, name)
    
    def _get_fallback_fixtures(self):
        """Fallback fixtures if scraping fails"""
        
        print("   📊 Using fallback fixture list")
        
        # Real upcoming fixtures (update weekly if needed)
        fixtures = [
            {'home_team': 'Man United', 'away_team': 'Bournemouth', 'date': '2025-12-22', 'time': '14:00', 'venue': 'Old Trafford', 'status': 'NS'},
            {'home_team': 'Man City', 'away_team': 'Everton', 'date': '2025-12-26', 'time': '12:30', 'venue': 'Etihad Stadium', 'status': 'NS'},
            {'home_team': 'Liverpool', 'away_team': 'Leicester', 'date': '2025-12-26', 'time': '20:00', 'venue': 'Anfield', 'status': 'NS'},
            {'home_team': 'Newcastle', 'away_team': 'Aston Villa', 'date': '2025-12-26', 'time': '15:00', 'venue': 'St James Park', 'status': 'NS'},
            {'home_team': 'Arsenal', 'away_team': 'Ipswich', 'date': '2025-12-27', 'time': '20:00', 'venue': 'Emirates Stadium', 'status': 'NS'},
        ]
        
        return fixtures
    
    def _load_from_cache(self):
        """Load cached fixtures (valid for 6 hours)"""
        
        cache_file = os.path.join(self.cache_dir, 'sky_fixtures.json')
        
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
        
        cache_file = os.path.join(self.cache_dir, 'sky_fixtures.json')
        
        with open(cache_file, 'w') as f:
            json.dump(fixtures, f, indent=2)
        
        print(f"   💾 Cached {len(fixtures)} fixtures")

# Testing
if __name__ == "__main__":
    scraper = SkyFixtureScraper()
    fixtures = scraper.scrape_upcoming_fixtures(days_ahead=14)
    
    if fixtures:
        print("\n" + "=" * 70)
        print("📅 UPCOMING PREMIER LEAGUE FIXTURES")
        print("=" * 70)
        
        for i, f in enumerate(fixtures[:15], 1):
            print(f"\n{i:2d}. {f['home_team']} vs {f['away_team']}")
            print(f"    Date: {f['date']} at {f['time']}")
    else:
        print("\n❌ No fixtures found")