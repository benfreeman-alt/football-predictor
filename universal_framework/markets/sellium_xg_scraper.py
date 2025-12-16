"""
SELENIUM xG SCRAPER

Uses Selenium to scrape FBref/Understat like a real browser
Bypasses anti-bot protection
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import pandas as pd
import time
import re

class SeleniumXGScraper:
    """Scrape xG data using Selenium (bypasses 403 errors)"""
    
    def __init__(self):
        self.team_xg = {}
        self.driver = None
    
    def init_driver(self, headless=True):
        """Initialize Chrome driver"""
        
        print("\n🤖 Initializing browser...")
        
        try:
            # Chrome options
            chrome_options = Options()
            
            if headless:
                chrome_options.add_argument('--headless')  # Run without opening browser
            
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            
            # Try with webdriver-manager first
            try:
                from webdriver_manager.chrome import ChromeDriverManager
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
                print("   ✅ Browser initialized (auto driver)")
            except ImportError:
                # Fallback to manual driver
                self.driver = webdriver.Chrome(options=chrome_options)
                print("   ✅ Browser initialized (manual driver)")
            
            return True
        
        except Exception as e:
            print(f"   ❌ Could not initialize browser: {e}")
            print("   Make sure Chrome is installed and chromedriver is available")
            return False
    
    def scrape_fbref(self, season='2024-2025'):
        """Scrape FBref using Selenium"""
        
        print(f"\n📊 Scraping FBref Premier League {season}...")
        
        if not self.driver:
            if not self.init_driver():
                return self._get_manual_xg_estimates()
        
        try:
            # Navigate to FBref
            url = "https://fbref.com/en/comps/9/Premier-League-Stats"
            print(f"   Loading {url}...")
            
            self.driver.get(url)
            
            # Wait for page to load
            time.sleep(3)
            
            # Wait for stats table
            wait = WebDriverWait(self.driver, 10)
            table = wait.until(
                EC.presence_of_element_located((By.ID, "stats_squads_standard_for"))
            )
            
            print("   ✅ Page loaded, parsing table...")
            
            # Get table HTML
            table_html = table.get_attribute('outerHTML')
            
            # Parse with pandas
            df = pd.read_html(table_html)[0]
            
            # Process the table
            xg_data = self._process_fbref_table(df)
            
            if xg_data:
                self.team_xg = xg_data
                print(f"\n   ✅ Retrieved xG data for {len(xg_data)} teams")
                return xg_data
            else:
                print("   ⚠️  Could not extract xG data")
                return self._get_manual_xg_estimates()
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return self._get_manual_xg_estimates()
        
        finally:
            if self.driver:
                self.driver.quit()
                print("   🔒 Browser closed")
    
    def scrape_understat(self, league='EPL', season='2024'):
        """Scrape Understat using Selenium"""
        
        print(f"\n📊 Scraping Understat {league} {season}...")
        
        if not self.driver:
            if not self.init_driver():
                return self._get_manual_xg_estimates()
        
        try:
            # Navigate to Understat
            url = f"https://understat.com/league/{league}/{season}"
            print(f"   Loading {url}...")
            
            self.driver.get(url)
            
            # Wait for page to load
            time.sleep(5)
            
            # Find the league table
            wait = WebDriverWait(self.driver, 10)
            table = wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "league-stats"))
            )
            
            print("   ✅ Page loaded, extracting data...")
            
            # Get all team rows
            rows = self.driver.find_elements(By.CSS_SELECTOR, "table.league-stats tbody tr")
            
            xg_data = {}
            
            for row in rows:
                try:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    
                    if len(cells) < 6:
                        continue
                    
                    # Extract data
                    team_name = cells[1].text.strip()
                    team_name = self.clean_team_name(team_name)
                    
                    matches = int(cells[2].text.strip())
                    xg_for = float(cells[5].text.strip())
                    xg_against = float(cells[6].text.strip())
                    
                    # Understat has npxG directly
                    npxg_for = xg_for * 0.85  # Approximate
                    npxg_against = xg_against * 0.85
                    
                    xg_data[team_name] = {
                        'matches': matches,
                        'xg_for': xg_for,
                        'xg_against': xg_against,
                        'npxg_for': npxg_for,
                        'npxg_against': npxg_against,
                        'xg_per_game': xg_for / max(matches, 1),
                        'npxg_per_game': npxg_for / max(matches, 1),
                    }
                    
                    print(f"   {team_name:20s}: npxG {xg_data[team_name]['npxg_per_game']:.2f}/game")
                
                except Exception as e:
                    continue
            
            if xg_data:
                self.team_xg = xg_data
                print(f"\n   ✅ Retrieved xG data for {len(xg_data)} teams")
                return xg_data
            else:
                return self._get_manual_xg_estimates()
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return self._get_manual_xg_estimates()
        
        finally:
            if self.driver:
                self.driver.quit()
                print("   🔒 Browser closed")
    
    def _process_fbref_table(self, df):
        """Process FBref table to extract xG"""
        
        xg_data = {}
        
        try:
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(str(col).strip() for col in c) for c in df.columns.values]
            
            # Find columns
            squad_col = None
            xg_col = None
            xga_col = None
            matches_col = None
            
            for col in df.columns:
                col_str = str(col).lower()
                
                if 'squad' in col_str:
                    squad_col = col
                elif col_str.endswith('xg') and 'xga' not in col_str:
                    xg_col = col
                elif 'xga' in col_str:
                    xga_col = col
                elif 'mp' in col_str or 'matches' in col_str:
                    matches_col = col
            
            if not squad_col or not xg_col:
                print(f"   ⚠️  Missing columns. Found: {df.columns.tolist()}")
                return None
            
            # Extract data
            for _, row in df.iterrows():
                try:
                    team_name = str(row[squad_col]).strip()
                    
                    # Skip header/aggregate rows
                    if team_name in ['Squad', 'nan', ''] or 'vs' in team_name.lower():
                        continue
                    
                    team_name = self.clean_team_name(team_name)
                    
                    xg_for = float(row[xg_col])
                    xg_against = float(row[xga_col]) if xga_col else 0
                    matches = int(row[matches_col]) if matches_col else 15
                    
                    # Calculate npxG (85% of total is typical)
                    npxg_for = xg_for * 0.85
                    npxg_against = xg_against * 0.85
                    
                    xg_data[team_name] = {
                        'matches': matches,
                        'xg_for': xg_for,
                        'xg_against': xg_against,
                        'npxg_for': npxg_for,
                        'npxg_against': npxg_against,
                        'xg_per_game': xg_for / max(matches, 1),
                        'npxg_per_game': npxg_for / max(matches, 1),
                    }
                    
                    print(f"   {team_name:20s}: npxG {xg_data[team_name]['npxg_per_game']:.2f}/game")
                
                except Exception as e:
                    continue
            
            return xg_data if xg_data else None
        
        except Exception as e:
            print(f"   ⚠️  Error processing table: {e}")
            return None
    
    def _get_manual_xg_estimates(self):
        """Manual xG estimates as fallback"""
        
        print("\n   📊 Using manual xG estimates (December 2024)")
        
        xg_estimates = {
            'Liverpool': 2.4,
            'Man City': 2.2,
            'Arsenal': 2.1,
            'Chelsea': 1.9,
            'Tottenham': 1.8,
            'Newcastle': 1.7,
            'Aston Villa': 1.7,
            'Brighton': 1.6,
            'Man United': 1.5,
            'Fulham': 1.4,
            'West Ham': 1.3,
            'Bournemouth': 1.3,
            'Brentford': 1.4,
            "Nott'm Forest": 1.2,
            'Crystal Palace': 1.2,
            'Wolves': 1.1,
            'Everton': 1.0,
            'Leicester': 1.0,
            'Ipswich': 0.9,
            'Southampton': 0.8,
        }
        
        xg_data = {}
        
        for team, npxg in xg_estimates.items():
            xg_data[team] = {
                'matches': 15,
                'xg_for': npxg * 15 * 1.15,
                'xg_against': 1.3 * 15,
                'npxg_for': npxg * 15,
                'npxg_against': 1.3 * 15 * 0.85,
                'xg_per_game': npxg * 1.15,
                'npxg_per_game': npxg,
            }
        
        self.team_xg = xg_data
        print(f"   ✅ Loaded estimates for {len(xg_data)} teams")
        
        return xg_data
    
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
    
    def get_team_xg(self, team_name):
        """Get xG stats for a team"""
        return self.team_xg.get(team_name, {
            'npxg_per_game': 1.3,
            'xg_per_game': 1.5,
            'matches': 15
        })
    
    def get_xg_advantage(self, home_team, away_team):
        """Calculate xG advantage"""
        
        home_xg = self.get_team_xg(home_team)
        away_xg = self.get_team_xg(away_team)
        
        advantage = home_xg['npxg_per_game'] - away_xg['npxg_per_game']
        
        return advantage
    
    def scrape_league_season(self, league='EPL', season='2024'):
        """
        Main scraping method - tries both FBref and Understat
        
        This matches the interface of other scrapers
        """
        
        # Try FBref first (more reliable)
        print("   Trying FBref first...")
        xg_data = self.scrape_fbref(f"{season}-{int(season)+1}")
        
        if xg_data and len(xg_data) > 15:  # Got good data
            return xg_data
        
        # Fallback to Understat
        print("   FBref failed, trying Understat...")
        xg_data = self.scrape_understat(league, season)
        
        return xg_data
    
    def save_to_csv(self, filename='selenium_xg.csv'):
        """Save xG data to CSV"""
        
        if not self.team_xg:
            print("No data to save")
            return
        
        df = pd.DataFrame.from_dict(self.team_xg, orient='index')
        df.index.name = 'team'
        df.to_csv(filename)
        
        print(f"\n✅ Saved xG data to {filename}")

# Testing
if __name__ == "__main__":
    scraper = SeleniumXGScraper()
    
    # Try scraping
    xg_data = scraper.scrape_league_season('EPL', '2024')
    
    if xg_data:
        scraper.save_to_csv()
        
        print("\n" + "=" * 70)
        print("📊 EXAMPLE: Man City vs Arsenal")
        print("=" * 70)
        
        city_xg = scraper.get_team_xg('Man City')
        arsenal_xg = scraper.get_team_xg('Arsenal')
        
        print(f"\nMan City:")
        print(f"  Non-penalty xG: {city_xg['npxg_per_game']:.2f} per game")
        print(f"  Total xG: {city_xg['xg_per_game']:.2f} per game")
        
        print(f"\nArsenal:")
        print(f"  Non-penalty xG: {arsenal_xg['npxg_per_game']:.2f} per game")
        print(f"  Total xG: {arsenal_xg['xg_per_game']:.2f} per game")
        
        advantage = scraper.get_xg_advantage('Man City', 'Arsenal')
        print(f"\nxG Advantage: {advantage:+.2f}")