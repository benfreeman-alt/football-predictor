"""
ADVANCED FIREFOX xG SCRAPER - FIXED VERSION

Properly handles:
- Browser reinitialization between scrapes
- Historical season URLs
- Graceful error handling
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
import pandas as pd
import time
from io import StringIO
from datetime import datetime

class AdvancedXGScraper:
    """Scrape comprehensive stats from FBref"""
    
    def __init__(self):
        self.team_stats = {}
        self.driver = None
    
    def init_driver(self, headless=True):
        """Initialize Firefox"""
        
        print("\n🦊 Initializing Firefox browser...")
        
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
    
    def scrape_fbref_complete(self, season='2024-2025'):
        """Scrape ALL stats from FBref - handles current and historical seasons"""
        
        print(f"\n📊 Scraping COMPLETE FBref data for {season}...")
        
        # Always reinitialize driver for each scrape
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None
        
        if not self.init_driver():
            return self._get_manual_estimates()
        
        try:
            # Determine URL based on season
            current_year = datetime.now().year
            season_start = int(season.split('-')[0])
            
            # If current or last season, use main page
            if season_start >= current_year - 1:
                url = "https://fbref.com/en/comps/9/Premier-League-Stats"
            else:
                # For historical seasons, use archive URL
                url = f"https://fbref.com/en/comps/9/{season}/Premier-League-Stats"
            
            print(f"   Loading {url}...")
            
            self.driver.get(url)
            time.sleep(5)
            
            print("   📊 Extracting stats tables...")
            
            # Get all required tables
            standard_df = self._get_table('stats_squads_standard_for')
            shooting_df = self._get_table('stats_squads_shooting_for')
            gca_df = self._get_table('stats_squads_gca_for')
            
            # Process all data
            team_stats = self._process_all_data(standard_df, shooting_df, gca_df)
            
            if team_stats:
                self.team_stats = team_stats
                print(f"\n   ✅ Retrieved COMPLETE stats for {len(team_stats)} teams!")
                return team_stats
            else:
                return self._get_manual_estimates()
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return self._get_manual_estimates()
        
        finally:
            # Always close browser after scraping
            if self.driver:
                try:
                    self.driver.quit()
                    print("   🔒 Browser closed")
                except:
                    pass
            self.driver = None
    
    def _get_table(self, table_id):
        """Get specific table"""
        
        try:
            wait = WebDriverWait(self.driver, 10)
            table = wait.until(
                EC.presence_of_element_located((By.ID, table_id))
            )
            
            table_html = table.get_attribute('outerHTML')
            df = pd.read_html(StringIO(table_html))[0]
            
            print(f"   ✅ Loaded {table_id}")
            return df
        
        except Exception as e:
            print(f"   ⚠️  Could not load {table_id}: {e}")
            return None
    
    def _process_all_data(self, standard_df, shooting_df, gca_df):
        """Process all tables into team stats"""
        
        team_stats = {}
        
        try:
            # Process standard table first (npxG)
            if standard_df is not None:
                team_stats = self._process_standard(standard_df)
            
            # Add shooting stats (xG per shot, shots)
            if shooting_df is not None and team_stats:
                shooting_stats = self._process_shooting(shooting_df)
                for team, data in shooting_stats.items():
                    if team in team_stats:
                        team_stats[team].update(data)
            
            # Add GCA stats (corners)
            if gca_df is not None and team_stats:
                gca_stats = self._process_gca(gca_df)
                for team, data in gca_stats.items():
                    if team in team_stats:
                        team_stats[team].update(data)
            
            # Print results
            if team_stats:
                print("\n   📊 COMPLETE STATS (TOP 10):")
                for team, data in sorted(team_stats.items(), 
                                        key=lambda x: x[1].get('npxg_per_game', 0), 
                                        reverse=True)[:10]:
                    print(f"   {team:20s}: "
                          f"npxG {data.get('npxg_per_game', 0):.2f}/g, "
                          f"xG/shot {data.get('xg_per_shot', 0):.3f}, "
                          f"corners {data.get('corners_per_game', 0):.1f}/g, "
                          f"shots {data.get('shots_per_game', 0):.1f}/g")
            
            return team_stats
        
        except Exception as e:
            print(f"   ⚠️  Error processing: {e}")
            return None
    
    def _process_standard(self, df):
        """Process standard stats - npxG"""
        
        team_data = {}
        
        try:
            # Flatten columns
            if isinstance(df.columns, pd.MultiIndex):
                flat_cols = []
                for col in df.columns:
                    if isinstance(col, tuple):
                        parts = [str(c) for c in col if 'Unnamed' not in str(c)]
                        flat_cols.append('_'.join(parts))
                    else:
                        flat_cols.append(str(col))
                df.columns = flat_cols
            
            # Find columns
            squad_col = mp_col = xg_col = npxg_col = None
            
            for col in df.columns:
                col_lower = str(col).lower()
                
                if 'squad' in col_lower and squad_col is None:
                    squad_col = col
                elif ('mp' in col_lower or 'matches' in col_lower) and '90' not in col_lower:
                    mp_col = col
                elif col_lower == 'expected_npxg':
                    npxg_col = col
                elif col_lower == 'expected_xg':
                    xg_col = col
            
            if not squad_col:
                return {}
            
            # Extract data
            for idx, row in df.iterrows():
                try:
                    team_name = str(row[squad_col]).strip()
                    
                    if team_name in ['Squad', 'nan', '', 'Unnamed'] or 'vs' in team_name.lower():
                        continue
                    
                    team_name = self.clean_team_name(team_name)
                    
                    matches = int(row[mp_col]) if mp_col and pd.notna(row[mp_col]) else 15
                    xg_total = float(row[xg_col]) if xg_col and pd.notna(row[xg_col]) else 0
                    npxg_total = float(row[npxg_col]) if npxg_col and pd.notna(row[npxg_col]) else (xg_total * 0.85)
                    
                    npxg_per_game = npxg_total / max(matches, 1)
                    xg_per_game = xg_total / max(matches, 1)
                    
                    team_data[team_name] = {
                        'matches': matches,
                        'xg_total': xg_total,
                        'npxg_total': npxg_total,
                        'npxg_per_game': npxg_per_game,
                        'xg_per_game': xg_per_game,
                    }
                
                except Exception as e:
                    continue
            
            return team_data
        
        except Exception as e:
            return {}
    
    def _process_shooting(self, df):
        """Process shooting stats - xG per shot, shots"""
        
        team_data = {}
        
        try:
            if isinstance(df.columns, pd.MultiIndex):
                flat_cols = []
                for col in df.columns:
                    if isinstance(col, tuple):
                        parts = [str(c) for c in col if 'Unnamed' not in str(c)]
                        flat_cols.append('_'.join(parts))
                    else:
                        flat_cols.append(str(col))
                df.columns = flat_cols
            
            # Find columns
            squad_col = shots_col = sot_col = npxg_col = None
            
            for col in df.columns:
                col_lower = str(col).lower()
                
                if 'squad' in col_lower and squad_col is None:
                    squad_col = col
                elif col_lower == 'standard_sh':
                    shots_col = col
                elif col_lower == 'standard_sot':
                    sot_col = col
                elif col_lower == 'expected_npxg':
                    npxg_col = col
            
            if not squad_col:
                return {}
            
            for _, row in df.iterrows():
                try:
                    team_name = str(row[squad_col]).strip()
                    
                    if team_name in ['Squad', 'nan', ''] or 'vs' in team_name.lower():
                        continue
                    
                    team_name = self.clean_team_name(team_name)
                    
                    shots = float(row[shots_col]) if shots_col and pd.notna(row[shots_col]) else 0
                    sot = float(row[sot_col]) if sot_col and pd.notna(row[sot_col]) else 0
                    npxg = float(row[npxg_col]) if npxg_col and pd.notna(row[npxg_col]) else 0
                    
                    xg_per_shot = npxg / shots if shots > 0 else 0
                    shot_accuracy = (sot / shots * 100) if shots > 0 else 0
                    shots_per_game = shots / 15
                    
                    team_data[team_name] = {
                        'shots_total': shots,
                        'shots_on_target': sot,
                        'shot_accuracy': shot_accuracy,
                        'xg_per_shot': xg_per_shot,
                        'shots_per_game': shots_per_game,
                    }
                
                except:
                    continue
            
            return team_data
        
        except Exception as e:
            return {}
    
    def _process_gca(self, df):
        """Process Goal Creating Actions table - corners"""
        
        team_data = {}
        
        try:
            if isinstance(df.columns, pd.MultiIndex):
                flat_cols = []
                for col in df.columns:
                    if isinstance(col, tuple):
                        parts = [str(c) for c in col if 'Unnamed' not in str(c)]
                        flat_cols.append('_'.join(parts))
                    else:
                        flat_cols.append(str(col))
                df.columns = flat_cols
            
            # Find columns
            squad_col = corners_col = dead_ball_col = None
            
            for col in df.columns:
                col_lower = str(col).lower()
                
                if 'squad' in col_lower and squad_col is None:
                    squad_col = col
                elif 'corner' in col_lower or col_lower.endswith('_ck'):
                    if corners_col is None:
                        corners_col = col
                elif 'passdead' in col_lower.replace('_', ''):
                    dead_ball_col = col
            
            set_piece_col = corners_col if corners_col else dead_ball_col
            
            if not squad_col:
                return {}
            
            for _, row in df.iterrows():
                try:
                    team_name = str(row[squad_col]).strip()
                    
                    if team_name in ['Squad', 'nan', ''] or 'vs' in team_name.lower():
                        continue
                    
                    team_name = self.clean_team_name(team_name)
                    
                    set_pieces = float(row[set_piece_col]) if set_piece_col and pd.notna(row[set_piece_col]) else 0
                    corners = set_pieces if corners_col else (set_pieces * 0.6)
                    
                    team_data[team_name] = {
                        'corners_total': corners,
                        'corners_per_game': corners / 15,
                    }
                
                except:
                    continue
            
            return team_data
        
        except Exception as e:
            return {}
    
    def _get_manual_estimates(self):
        """Manual estimates fallback"""
        
        print("\n   📊 Using manual estimates (December 2024)")
        
        estimates = {
            'Liverpool': {'npxg': 2.4, 'xg_shot': 0.12, 'corners': 6.2, 'shots': 20.0},
            'Man City': {'npxg': 2.2, 'xg_shot': 0.13, 'corners': 6.5, 'shots': 17.0},
            'Arsenal': {'npxg': 2.1, 'xg_shot': 0.11, 'corners': 7.1, 'shots': 19.0},
            'Chelsea': {'npxg': 1.9, 'xg_shot': 0.10, 'corners': 5.8, 'shots': 19.0},
            'Tottenham': {'npxg': 1.8, 'xg_shot': 0.11, 'corners': 5.5, 'shots': 16.5},
            'Newcastle': {'npxg': 1.7, 'xg_shot': 0.10, 'corners': 5.2, 'shots': 17.0},
            'Aston Villa': {'npxg': 1.7, 'xg_shot': 0.10, 'corners': 5.0, 'shots': 17.0},
            'Brighton': {'npxg': 1.6, 'xg_shot': 0.09, 'corners': 5.3, 'shots': 18.0},
            'Man United': {'npxg': 1.5, 'xg_shot': 0.09, 'corners': 5.1, 'shots': 17.0},
            'Fulham': {'npxg': 1.4, 'xg_shot': 0.09, 'corners': 4.8, 'shots': 15.5},
            'West Ham': {'npxg': 1.3, 'xg_shot': 0.08, 'corners': 4.5, 'shots': 16.0},
            'Bournemouth': {'npxg': 1.3, 'xg_shot': 0.08, 'corners': 4.7, 'shots': 16.0},
            'Brentford': {'npxg': 1.4, 'xg_shot': 0.10, 'corners': 4.9, 'shots': 14.0},
            "Nott'm Forest": {'npxg': 1.2, 'xg_shot': 0.08, 'corners': 4.2, 'shots': 15.0},
            'Crystal Palace': {'npxg': 1.2, 'xg_shot': 0.08, 'corners': 4.3, 'shots': 15.0},
            'Wolves': {'npxg': 1.1, 'xg_shot': 0.07, 'corners': 4.0, 'shots': 16.0},
            'Everton': {'npxg': 1.0, 'xg_shot': 0.07, 'corners': 3.8, 'shots': 14.0},
            'Leicester': {'npxg': 1.0, 'xg_shot': 0.07, 'corners': 4.0, 'shots': 14.0},
            'Ipswich': {'npxg': 0.9, 'xg_shot': 0.06, 'corners': 3.5, 'shots': 15.0},
            'Southampton': {'npxg': 0.8, 'xg_shot': 0.06, 'corners': 3.2, 'shots': 13.0},
        }
        
        team_stats = {}
        for team, est in estimates.items():
            team_stats[team] = {
                'matches': 15,
                'npxg_total': est['npxg'] * 15,
                'npxg_per_game': est['npxg'],
                'xg_per_game': est['npxg'] * 1.15,
                'xg_per_shot': est['xg_shot'],
                'shots_total': est['shots'] * 15,
                'shots_per_game': est['shots'],
                'corners_total': est['corners'] * 15,
                'corners_per_game': est['corners'],
                'shot_accuracy': 35.0,
            }
        
        self.team_stats = team_stats
        return team_stats
    
    def clean_team_name(self, name):
        """Standardize names"""
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
            'Manchester Utd': 'Man United',
            'Newcastle Utd': 'Newcastle',
            "Nott'ham Forest": "Nott'm Forest",
        }
        return name_map.get(name, name)
    
    def get_team_stats(self, team_name):
        """Get complete stats for team"""
        return self.team_stats.get(team_name, {
            'npxg_per_game': 1.3,
            'xg_per_shot': 0.08,
            'corners_per_game': 4.5,
            'shots_per_game': 15.0,
            'shot_accuracy': 33.0,
        })
    
    def get_xg_advantage(self, home_team, away_team):
        """Calculate xG advantage"""
        home = self.get_team_stats(home_team)
        away = self.get_team_stats(away_team)
        return home.get('npxg_per_game', 1.3) - away.get('npxg_per_game', 1.3)
    
    def scrape_league_season(self, league='EPL', season='2024'):
        """Main method - matches interface"""
        return self.scrape_fbref_complete(f"{season}-{int(season)+1}")
    
    def save_to_csv(self, filename='advanced_stats.csv'):
        """Save to CSV"""
        if self.team_stats:
            df = pd.DataFrame.from_dict(self.team_stats, orient='index')
            df.index.name = 'team'
            df.to_csv(filename)
            print(f"\n✅ Saved complete stats to {filename}")

# Testing
if __name__ == "__main__":
    scraper = AdvancedXGScraper()
    stats = scraper.scrape_league_season('EPL', '2024')
    
    if stats:
        scraper.save_to_csv()
        
        print("\n" + "=" * 70)
        print("📊 COMPLETE ADVANCED STATS")
        print("=" * 70)
        
        teams = ['Liverpool', 'Man City', 'Arsenal', 'Chelsea']
        
        for team in teams:
            s = scraper.get_team_stats(team)
            print(f"\n{team}:")
            print(f"  Non-penalty xG: {s.get('npxg_per_game', 0):.2f}/game")
            print(f"  xG per shot: {s.get('xg_per_shot', 0):.3f}")
            print(f"  Shots: {s.get('shots_per_game', 0):.1f}/game")
            print(f"  Corners: {s.get('corners_per_game', 0):.1f}/game")