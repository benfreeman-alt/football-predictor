from firefox_advanced_scraper import AdvancedXGScraper

scraper = AdvancedXGScraper()
stats = scraper.scrape_league_season('EPL', '2024')

if stats:
    print("\n" + "=" * 70)
    print("TOP 5 TEAMS - COMPLETE STATS")
    print("=" * 70)
    
    for team in ['Man City', 'Arsenal', 'Liverpool', 'Chelsea', 'Tottenham']:
        s = scraper.get_team_stats(team)
        print(f"\n{team}:")
        print(f"  npxG: {s.get('npxg_per_game', 0):.2f}/game")
        print(f"  xG/shot: {s.get('xg_per_shot', 0):.3f}")
        print(f"  Shots: {s.get('shots_per_game', 0):.1f}/game")
        print(f"  Corners: {s.get('corners_per_game', 0):.1f}/game")