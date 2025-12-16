"""
HISTORICAL xG SCRAPER - FIXED VERSION

Scrapes and caches xG data for multiple seasons
Properly handles browser cleanup between scrapes
"""

from firefox_advanced_scraper import AdvancedXGScraper
import json
import os
from datetime import datetime

class HistoricalXGScraper:
    """Manage xG data across multiple seasons"""
    
    def __init__(self, cache_dir='data/xg_cache'):
        self.cache_dir = cache_dir
        self.season_data = {}
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_season_from_date(self, match_date):
        """Determine season from match date"""
        
        if isinstance(match_date, str):
            match_date = datetime.strptime(match_date, '%Y-%m-%d')
        
        year = match_date.year
        month = match_date.month
        
        # Season runs Aug-May
        # Aug-Dec = current year season
        # Jan-May = previous year season
        if month >= 8:
            return f"{year}-{year+1}"
        else:
            return f"{year-1}-{year}"
    
    def get_cache_filename(self, season):
        """Get cache file path for season"""
        return os.path.join(self.cache_dir, f'xg_{season}.json')
    
    def load_season_from_cache(self, season):
        """Load cached season data"""
        
        cache_file = self.get_cache_filename(season)
        
        if os.path.exists(cache_file):
            print(f"   📂 Loading {season} from cache...")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        return None
    
    def save_season_to_cache(self, season, data):
        """Save season data to cache"""
        
        cache_file = self.get_cache_filename(season)
        
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"   💾 Cached {season} data")
    
    def scrape_season(self, season):
        """Scrape xG data for a specific season"""
        
        print(f"\n📊 Getting xG data for {season}...")
        
        # Check cache first
        cached_data = self.load_season_from_cache(season)
        if cached_data:
            return cached_data
        
        # Extract start year (e.g., "2023-2024" -> "2023")
        start_year = season.split('-')[0]
        
        # Create new scraper instance for this season
        scraper = AdvancedXGScraper()
        
        try:
            # Scrape from FBref
            print(f"   Scraping {season} (starting year: {start_year})...")
            stats = scraper.scrape_fbref_complete(season)
            
            if stats and len(stats) > 0:
                # Save to cache
                self.save_season_to_cache(season, stats)
                return stats
            else:
                print(f"   ⚠️  No data scraped for {season}, using estimates")
                return scraper._get_manual_estimates()
        
        except Exception as e:
            print(f"   ⚠️  Error scraping {season}: {e}")
            return scraper._get_manual_estimates()
    
    def get_stats_for_match(self, team_name, match_date):
        """Get xG stats for a team at a specific date"""
        
        # Determine season
        season = self.get_season_from_date(match_date)
        
        # Load season data if not already loaded
        if season not in self.season_data:
            self.season_data[season] = self.scrape_season(season)
        
        # Get team stats
        season_stats = self.season_data[season]
        
        if team_name in season_stats:
            return season_stats[team_name]
        else:
            # Return default if team not found
            return {
                'npxg_per_game': 1.3,
                'xg_per_shot': 0.08,
                'corners_per_game': 4.5,
                'shots_per_game': 15.0,
            }
    
    def preload_seasons(self, seasons):
        """Preload multiple seasons"""
        
        print("\n📊 Preloading historical xG data...")
        print("=" * 70)
        
        for season in seasons:
            self.season_data[season] = self.scrape_season(season)
        
        print("\n✅ All seasons loaded!")
        print(f"   Cached seasons: {list(self.season_data.keys())}")

# Testing
if __name__ == "__main__":
    scraper = HistoricalXGScraper()
    
    # Test: Load last 3 seasons
    seasons = ['2022-2023', '2023-2024', '2024-2025']
    
    print("🧹 Clearing cache for fresh scrape...")
    import shutil
    if os.path.exists('data/xg_cache'):
        shutil.rmtree('data/xg_cache')
    os.makedirs('data/xg_cache', exist_ok=True)
    
    scraper.preload_seasons(seasons)
    
    # Test: Get stats for a specific match
    test_date = '2024-09-15'
    team = 'Arsenal'
    
    stats = scraper.get_stats_for_match(team, test_date)
    
    print("\n" + "=" * 70)
    print(f"TEST: {team} stats for match on {test_date}")
    print("=" * 70)
    print(f"Season detected: {scraper.get_season_from_date(test_date)}")
    print(f"npxG per game: {stats.get('npxg_per_game', 0):.2f}")
    print(f"xG per shot: {stats.get('xg_per_shot', 0):.3f}")
    print(f"Shots per game: {stats.get('shots_per_game', 0):.1f}")
    print(f"Set pieces per game: {stats.get('corners_per_game', 0):.1f}")