"""
FIREFOX SELENIUM xG SCRAPER

Uses Firefox instead of Chrome (more reliable on Windows)
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
import pandas as pd
import time

class FirefoxXGScraper:
    """Scrape xG data using Firefox/Selenium"""
    
    def __init__(self):
        self.team_xg = {}
        self.driver = None
    
    def init_driver(self, headless=True):
        """Initialize Firefox driver"""
        
        print("\n🦊 Initializing Firefox browser...")
        
        try:
            firefox_options = Options()
            
            if headless:
                firefox_options.add_argument('--headless')
            
            # Try with webdriver-manager
            try:
                from webdriver_manager.firefox import GeckoDriverManager
                service = Service(GeckoDriverManager().install())
                self.driver = webdriver.Firefox(service=service, options=firefox_options)
                print("   ✅ Firefox initialized")
            except ImportError:
                self.driver = webdriver.Firefox(options=firefox_options)
                print("   ✅ Firefox initialized (manual driver)")
            
            return True
        
        except Exception as e:
            print(f"   ❌ Could not initialize Firefox: {e}")
            return False
    
    def scrape_fbref(self, season='2024-2025'):
        """Scrape FBref using Firefox"""
        
        print(f"\n📊 Scraping FBref Premier League {season}...")
        
        if not self.driver:
            if not self.init_driver():
                return self._get_manual_xg_estimates()
        
        try:
            url = "https://fbref.com/en/comps/9/Premier-League-Stats"
            print(f"   Loading {url}...")
            
            self.driver.get(url)
            time.sleep(5)  # Wait for page to load
            
            # Wait for stats table
            wait = WebDriverWait(self.driver, 15)
            table = wait.until(
                EC.presence_of_element_located((By.ID, "stats_squads_standard_for"))
            )
            
            print("   ✅ Page loaded, parsing table...")
            
            # Get table HTML and parse
            table_html = table.get_attribute('outerHTML')
            df = pd.read_html(table_html)[0]
            
            # Process
            xg_data = self._process_fbref_table(df)
            
            if xg_data:
                self.team_xg = xg_data
                print(f"\n   ✅ Retrieved REAL xG data for {len(xg_data)} teams!")
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
        """Process FBref table"""
        
        xg_data = {}
        
        try:
            # Flatten columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(str(c).strip() for c in col) for col in df.columns.values]
            
            # Find columns
            squad_col = xg_col = xga_col = matches_col = None
            
            for col in df.columns:
                col_str = str(col).lower()
                if 'squad' in col_str:
                    squad_col = col
                elif col_str.endswith('xg') and 'xga' not in col_str:
                    xg_col = col
                elif 'xga' in col_str:
                    xga_col = col
                elif 'mp' in col_str:
                    matches_col = col
            
            if not squad_col or not xg_col:
                return None
            
            # Extract data
            for _, row in df.iterrows():
                try:
                    team_name = str(row[squad_col]).strip()
                    
                    if team_name in ['Squad', 'nan', ''] or 'vs' in team_name.lower():
                        continue
                    
                    team_name = self.clean_team_name(team_name)
                    
                    xg_for = float(row[xg_col])
                    xg_against = float(row[xga_col]) if xga_col else 0
                    matches = int(row[matches_col]) if matches_col else 15
                    
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
                
                except:
                    continue
            
            return xg_data if xg_data else None
        
        except Exception as e:
            return None
    
    def _get_manual_xg_estimates(self):
        """Manual fallback"""
        
        print("\n   📊 Using manual xG estimates (December 2024)")
        
        xg_estimates = {
            'Liverpool': 2.4, 'Man City': 2.2, 'Arsenal': 2.1, 'Chelsea': 1.9,
            'Tottenham': 1.8, 'Newcastle': 1.7, 'Aston Villa': 1.7, 'Brighton': 1.6,
            'Man United': 1.5, 'Fulham': 1.4, 'West Ham': 1.3, 'Bournemouth': 1.3,
            'Brentford': 1.4, "Nott'm Forest": 1.2, 'Crystal Palace': 1.2,
            'Wolves': 1.1, 'Everton': 1.0, 'Leicester': 1.0, 'Ipswich': 0.9,
            'Southampton': 0.8,
        }
        
        xg_data = {}
        for team, npxg in xg_estimates.items():
            xg_data[team] = {
                'matches': 15, 'xg_for': npxg * 15 * 1.15, 'xg_against': 1.3 * 15,
                'npxg_for': npxg * 15, 'npxg_against': 1.3 * 15 * 0.85,
                'xg_per_game': npxg * 1.15, 'npxg_per_game': npxg,
            }
        
        self.team_xg = xg_data
        print(f"   ✅ Loaded estimates for {len(xg_data)} teams")
        return xg_data
    
    def clean_team_name(self, name):
        """Standardize names"""
        name_map = {
            'Manchester United': 'Man United', 'Manchester City': 'Man City',
            'Tottenham Hotspur': 'Tottenham', 'Newcastle United': 'Newcastle',
            'West Ham United': 'West Ham', 'Leicester City': 'Leicester',
            'Brighton & Hove Albion': 'Brighton', 'Brighton and Hove Albion': 'Brighton',
            'Wolverhampton Wanderers': 'Wolves', 'Nottingham Forest': "Nott'm Forest",
            'AFC Bournemouth': 'Bournemouth', 'Ipswich Town': 'Ipswich',
        }
        return name_map.get(name, name)
    
    def get_team_xg(self, team_name):
        """Get xG for team"""
        return self.team_xg.get(team_name, {
            'npxg_per_game': 1.3, 'xg_per_game': 1.5, 'matches': 15
        })
    
    def get_xg_advantage(self, home_team, away_team):
        """Calculate advantage"""
        home_xg = self.get_team_xg(home_team)
        away_xg = self.get_team_xg(away_team)
        return home_xg['npxg_per_game'] - away_xg['npxg_per_game']
    
    def scrape_league_season(self, league='EPL', season='2024'):
        """Main scraping method"""
        return self.scrape_fbref(f"{season}-{int(season)+1}")
    
    def save_to_csv(self, filename='firefox_xg.csv'):
        """Save to CSV"""
        if self.team_xg:
            df = pd.DataFrame.from_dict(self.team_xg, orient='index')
            df.index.name = 'team'
            df.to_csv(filename)
            print(f"\n✅ Saved xG data to {filename}")

# Testing
if __name__ == "__main__":
    scraper = FirefoxXGScraper()
    xg_data = scraper.scrape_league_season('EPL', '2024')
    
    if xg_data:
        scraper.save_to_csv()
        
        print("\n" + "=" * 70)
        print("📊 REAL xG DATA vs Manual Estimates")
        print("=" * 70)
        
        for team in ['Man City', 'Arsenal', 'Liverpool'][:3]:
            xg = scraper.get_team_xg(team)
            print(f"\n{team}: {xg['npxg_per_game']:.2f} npxG/game")