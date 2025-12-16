"""
FBREF xG DATA SCRAPER

Scrapes expected goals (xG) data from FBref.com (StatsBomb data)
Free, comprehensive, reliable
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

class FBrefScraper:
    """Scrape xG data from FBref"""
    
    BASE_URL = "https://fbref.com"
    
    def __init__(self):
        self.team_xg = {}
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
    
    def scrape_premier_league(self, season='2024-2025'):
        """
        Scrape Premier League xG data for current season
        
        Args:
            season: '2024-2025' for current season
        """
        
        print(f"\n📊 Scraping FBref Premier League {season}...")
        
        # FBref Premier League stats page
        url = f"{self.BASE_URL}/en/comps/9/Premier-League-Stats"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            # Parse with pandas (reads HTML tables)
            tables = pd.read_html(response.text)
            
            print(f"   Found {len(tables)} tables on page")
            
            # Find the table with xG data
            xg_table = None
            
            for i, table in enumerate(tables):
                # Check if this table has xG columns
                cols_str = str(table.columns).lower()
                
                if 'xg' in cols_str and 'squad' in cols_str:
                    xg_table = table
                    print(f"   ✅ Found xG table (table #{i})")
                    break
            
            if xg_table is None:
                print("   ⚠️  Could not find xG table, trying alternative method...")
                return self._scrape_alternative_method(response.text)
            
            # Process the table
            xg_data = self._process_fbref_table(xg_table)
            
            if xg_data:
                self.team_xg = xg_data
                print(f"\n   ✅ Retrieved xG data for {len(xg_data)} teams")
            
            return xg_data
        
        except Exception as e:
            print(f"   ❌ Error scraping FBref: {e}")
            print("   Using manual estimates...")
            return self._get_manual_xg_estimates()
    
    def _process_fbref_table(self, table):
        """Process FBref stats table to extract xG"""
        
        xg_data = {}
        
        try:
            # FBref tables often have multi-level columns
            # Flatten if needed
            if isinstance(table.columns, pd.MultiIndex):
                table.columns = ['_'.join(col).strip() for col in table.columns.values]
            
            # Find relevant columns
            squad_col = None
            xg_col = None
            xga_col = None
            matches_col = None
            
            for col in table.columns:
                col_lower = str(col).lower()
                
                if 'squad' in col_lower and squad_col is None:
                    squad_col = col
                elif 'xg' in col_lower and 'xga' not in col_lower and xg_col is None:
                    xg_col = col
                elif 'xga' in col_lower and xga_col is None:
                    xga_col = col
                elif 'mp' in col_lower or 'matches' in col_lower:
                    matches_col = col
            
            if not all([squad_col, xg_col]):
                print("   ⚠️  Missing required columns")
                return None
            
            # Extract data
            for _, row in table.iterrows():
                try:
                    team_name = str(row[squad_col]).strip()
                    
                    # Skip aggregate rows
                    if team_name in ['Squad', '', 'nan'] or 'vs' in team_name:
                        continue
                    
                    # Clean team name
                    team_name = self.clean_team_name(team_name)
                    
                    # Get xG values
                    xg_for = float(row[xg_col])
                    xg_against = float(row[xga_col]) if xga_col else 0
                    matches = int(row[matches_col]) if matches_col else 15
                    
                    # Calculate non-penalty xG (approximate: 85% of total)
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
    
    def _scrape_alternative_method(self, html):
        """Alternative scraping method using BeautifulSoup"""
        
        print("   Trying BeautifulSoup method...")
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find stats table
            stats_table = soup.find('table', {'id': re.compile(r'stats_squads_standard')})
            
            if not stats_table:
                return None
            
            rows = stats_table.find_all('tr')
            
            xg_data = {}
            
            for row in rows[1:]:  # Skip header
                cells = row.find_all(['td', 'th'])
                
                if len(cells) < 5:
                    continue
                
                try:
                    team_name = cells[0].get_text(strip=True)
                    team_name = self.clean_team_name(team_name)
                    
                    # Try to find xG columns
                    # This varies by page structure
                    for i, cell in enumerate(cells):
                        text = cell.get_text(strip=True)
                        
                        # Look for numeric values that could be xG
                        if text.replace('.', '').isdigit():
                            val = float(text)
                            
                            # xG is usually between 0.5 and 3.0 per game
                            if 10 < val < 50:  # Total xG for season
                                xg_data[team_name] = {
                                    'xg_for': val,
                                    'npxg_per_game': (val * 0.85) / 15,
                                }
                                break
                
                except:
                    continue
            
            return xg_data if xg_data else None
        
        except Exception as e:
            print(f"   ⚠️  Alternative method failed: {e}")
            return None
    
    def _get_manual_xg_estimates(self):
        """
        Manual xG estimates - Updated December 2024
        Based on current season performance
        """
        
        print("\n   📊 Using manual xG estimates (December 2024)")
        
        xg_estimates = {
            # Elite attacking teams
            'Liverpool': 2.4,
            'Man City': 2.2,
            'Arsenal': 2.1,
            'Chelsea': 1.9,
            
            # Strong attacking
            'Tottenham': 1.8,
            'Newcastle': 1.7,
            'Aston Villa': 1.7,
            'Brighton': 1.6,
            'Man United': 1.5,
            
            # Mid-table
            'Fulham': 1.4,
            'West Ham': 1.3,
            'Bournemouth': 1.3,
            'Brentford': 1.4,
            "Nott'm Forest": 1.2,
            'Crystal Palace': 1.2,
            
            # Lower teams
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
        
        print(f"   ✅ Loaded estimates for {len(xg_data)} teams")
        
        return xg_data
    
    def clean_team_name(self, name):
        """Standardize team names"""
        
        # Remove extra text
        name = name.replace(' FC', '').replace(' United', '')
        
        name_map = {
            'Manchester': 'Man',
            'Tottenham Hotspur': 'Tottenham',
            'Tottenham': 'Tottenham',
            'Newcastle': 'Newcastle',
            'West Ham': 'West Ham',
            'Leicester City': 'Leicester',
            'Leicester': 'Leicester',
            'Brighton & Hove Albion': 'Brighton',
            'Brighton and Hove Albion': 'Brighton',
            'Brighton': 'Brighton',
            'Wolverhampton Wanderers': 'Wolves',
            'Wolves': 'Wolves',
            'Nottingham Forest': "Nott'm Forest",
            "Nott'm Forest": "Nott'm Forest",
            'AFC Bournemouth': 'Bournemouth',
            'Bournemouth': 'Bournemouth',
            'Ipswich Town': 'Ipswich',
            'Ipswich': 'Ipswich',
            'Man City': 'Man City',
            'Manchester City': 'Man City',
            'Man United': 'Man United',
            'Manchester United': 'Man United',
        }
        
        for key, val in name_map.items():
            if key in name:
                return val
        
        return name
    
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
    
    def save_to_csv(self, filename='fbref_xg.csv'):
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
    scraper = FBrefScraper()
    
    # Scrape current season
    xg_data = scraper.scrape_premier_league('2024-2025')
    
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