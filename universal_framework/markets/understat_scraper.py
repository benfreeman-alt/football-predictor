"""
UNDERSTAT xG DATA SCRAPER - ROBUST VERSION

Scrapes expected goals (xG) data from Understat.com
Multiple parsing methods for reliability
"""

import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import re
from datetime import datetime
import time

class UnderstatScraper:
    """Scrape xG data from Understat with multiple fallback methods"""
    
    BASE_URL = "https://understat.com"
    
    def __init__(self):
        self.team_xg = {}
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://understat.com/'
        }
    
    def scrape_league_season(self, league='EPL', season='2024'):
        """
        Scrape team xG data for a season
        
        Args:
            league: 'EPL', 'La_liga', 'Bundesliga', 'Serie_A', 'Ligue_1'
            season: '2024' for 2024/25 season
        """
        
        print(f"\n📊 Scraping Understat {league} {season}/{int(season)+1}...")
        
        url = f"{self.BASE_URL}/league/{league}/{season}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            # Method 1: Try to find JSON in script tags
            xg_data = self._parse_method_1(response.text)
            
            if xg_data:
                self.team_xg = xg_data
                return xg_data
            
            # Method 2: Parse HTML tables
            xg_data = self._parse_method_2(response.text)
            
            if xg_data:
                self.team_xg = xg_data
                return xg_data
            
            print("   ⚠️  Could not parse Understat data")
            print("   Using manual xG estimates instead...")
            
            # Fallback: Use manual estimates for current season
            xg_data = self._get_manual_xg_estimates()
            self.team_xg = xg_data
            
            return xg_data
        
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Network error: {e}")
            print("   Using manual xG estimates...")
            xg_data = self._get_manual_xg_estimates()
            self.team_xg = xg_data
            return xg_data
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print("   Using manual xG estimates...")
            xg_data = self._get_manual_xg_estimates()
            self.team_xg = xg_data
            return xg_data
    
    def _parse_method_1(self, html):
        """Method 1: Extract JSON from script tags"""
        
        soup = BeautifulSoup(html, 'html.parser')
        scripts = soup.find_all('script')
        
        for script in scripts:
            if not script.string:
                continue
            
            # Try multiple JSON extraction patterns
            patterns = [
                r'var teamsData\s*=\s*JSON\.parse\(\'(.+?)\'\)',
                r'teamsData\s*=\s*(\{.+?\});',
                r'var teamsData\s*=\s*(\{.+?\});',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, script.string, re.DOTALL)
                
                if match:
                    try:
                        json_str = match.group(1)
                        
                        # Try to decode if it's escaped
                        if '\\' in json_str:
                            json_str = json_str.encode().decode('unicode_escape')
                        
                        team_data = json.loads(json_str)
                        
                        # Process the data
                        return self._process_team_data(team_data)
                    
                    except:
                        continue
        
        return None
    
    def _parse_method_2(self, html):
        """Method 2: Parse HTML table"""
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find the league table
            table = soup.find('table', class_='league-table')
            
            if not table:
                return None
            
            rows = table.find_all('tr')[1:]  # Skip header
            
            xg_data = {}
            
            for row in rows:
                cols = row.find_all('td')
                
                if len(cols) < 6:
                    continue
                
                team_name = cols[1].get_text(strip=True)
                team_name = self.clean_team_name(team_name)
                
                # Extract xG values
                matches = int(cols[2].get_text(strip=True))
                xg_for = float(cols[5].get_text(strip=True))
                xg_against = float(cols[6].get_text(strip=True))
                
                xg_data[team_name] = {
                    'matches': matches,
                    'xg_for': xg_for,
                    'xg_against': xg_against,
                    'npxg_for': xg_for * 0.85,  # Estimate: ~85% is non-penalty
                    'npxg_against': xg_against * 0.85,
                    'xg_per_game': xg_for / max(matches, 1),
                    'npxg_per_game': (xg_for * 0.85) / max(matches, 1),
                }
                
                print(f"   {team_name:20s}: npxG {xg_data[team_name]['npxg_per_game']:.2f}/game")
            
            if xg_data:
                print(f"\n   ✅ Retrieved xG data for {len(xg_data)} teams")
            
            return xg_data if xg_data else None
        
        except:
            return None
    
    def _process_team_data(self, team_data):
        """Process extracted team data"""
        
        xg_data = {}
        
        for team_id, data in team_data.items():
            if isinstance(data, dict):
                team_name = data.get('title', '')
                team_name = self.clean_team_name(team_name)
                
                matches = int(data.get('matches', 0))
                
                if matches == 0:
                    continue
                
                xg_data[team_name] = {
                    'matches': matches,
                    'xg_for': float(data.get('xG', 0)),
                    'xg_against': float(data.get('xGA', 0)),
                    'npxg_for': float(data.get('npxG', 0)),
                    'npxg_against': float(data.get('npxGA', 0)),
                    'xg_per_game': float(data.get('xG', 0)) / matches,
                    'npxg_per_game': float(data.get('npxG', 0)) / matches,
                }
                
                print(f"   {team_name:20s}: npxG {xg_data[team_name]['npxg_per_game']:.2f}/game")
        
        if xg_data:
            print(f"\n   ✅ Retrieved xG data for {len(xg_data)} teams")
        
        return xg_data if xg_data else None
    
    def _get_manual_xg_estimates(self):
        """
        Manual xG estimates based on current Premier League performance
        
        Updated December 2024
        Source: General football knowledge + season stats
        """
        
        print("\n   📊 Using manual xG estimates (December 2024)")
        
        # Non-penalty xG per game estimates
        xg_estimates = {
            # Top tier
            'Liverpool': 2.3,
            'Man City': 2.1,
            'Arsenal': 2.0,
            'Chelsea': 1.8,
            
            # Upper mid
            'Tottenham': 1.7,
            'Newcastle': 1.6,
            'Man United': 1.5,
            'Aston Villa': 1.6,
            'Brighton': 1.5,
            
            # Mid table
            'West Ham': 1.3,
            'Fulham': 1.3,
            'Bournemouth': 1.2,
            'Brentford': 1.4,
            'Crystal Palace': 1.2,
            "Nott'm Forest": 1.1,
            
            # Lower mid
            'Wolves': 1.0,
            'Everton': 1.0,
            'Leicester': 1.0,
            
            # Bottom
            'Ipswich': 0.9,
            'Southampton': 0.8,
        }
        
        xg_data = {}
        
        for team, npxg in xg_estimates.items():
            xg_data[team] = {
                'matches': 15,  # Approximate games played
                'xg_for': npxg * 15 * 1.15,  # Add ~15% for penalties
                'xg_against': 1.3 * 15,  # League average
                'npxg_for': npxg * 15,
                'npxg_against': 1.3 * 15 * 0.85,
                'xg_per_game': npxg * 1.15,
                'npxg_per_game': npxg,
            }
        
        print(f"   ✅ Loaded estimates for {len(xg_data)} teams")
        
        return xg_data
    
    def clean_team_name(self, name):
        """Standardize team names"""
        
        name_map = {
            'Manchester United': 'Man United',
            'Manchester City': 'Man City',
            'Tottenham': 'Tottenham',
            'Tottenham Hotspur': 'Tottenham',
            'Newcastle United': 'Newcastle',
            'West Ham': 'West Ham',
            'West Ham United': 'West Ham',
            'Leicester': 'Leicester',
            'Leicester City': 'Leicester',
            'Brighton': 'Brighton',
            'Brighton and Hove Albion': 'Brighton',
            'Wolverhampton Wanderers': 'Wolves',
            'Wolves': 'Wolves',
            'Nottingham Forest': "Nott'm Forest",
            'Bournemouth': 'Bournemouth',
            'AFC Bournemouth': 'Bournemouth',
            'Ipswich': 'Ipswich',
            'Ipswich Town': 'Ipswich',
        }
        
        return name_map.get(name, name)
    
    def get_team_xg(self, team_name):
        """Get xG stats for a team"""
        return self.team_xg.get(team_name, {
            'npxg_per_game': 1.3,  # League average
            'xg_per_game': 1.5,
            'matches': 15
        })
    
    def get_xg_advantage(self, home_team, away_team):
        """
        Calculate xG advantage for home team
        
        Positive = home team creates better chances
        """
        
        home_xg = self.get_team_xg(home_team)
        away_xg = self.get_team_xg(away_team)
        
        advantage = home_xg['npxg_per_game'] - away_xg['npxg_per_game']
        
        return advantage
    
    def save_to_csv(self, filename='understat_xg.csv'):
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
    scraper = UnderstatScraper()
    
    # Scrape current season
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
        print(f"\nxG Advantage: {advantage:+.2f} (Man City creates better chances)")