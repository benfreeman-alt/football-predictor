"""
FOOTBALL BETTING OPPORTUNITY FINDER - COMPLETE CLEAN VERSION

Finds value bets for Premier League matches with LIVE odds
"""

import pandas as pd
import requests
from datetime import datetime
from football_data import FootballDataLoader
from football_optimized_v4_historical import FootballPredictorV4Historical as FootballPredictor

# Load xG data
predictor.load_xg_data(season='2024')

# ============================================================================
# CONFIGURATION - UPDATE THESE!
# ============================================================================

# Your FREE API key from: https://the-odds-api.com/
THE_ODDS_API_KEY = '5c08725603fba442011b0a242bf5d2a8'

# Your betting bankroll (£)
BANKROLL = 1000

# ============================================================================
# TEAM NAME CLEANER
# ============================================================================

def clean_team_name(name):
    """Standardize team names to match our data"""
    
    name_map = {
        'Manchester United': 'Man United',
        'Manchester United FC': 'Man United',
        'Man United': 'Man United',
        'Manchester City': 'Man City',
        'Manchester City FC': 'Man City',
        'Man City': 'Man City',
        'Tottenham Hotspur': 'Tottenham',
        'Tottenham Hotspur FC': 'Tottenham',
        'Tottenham': 'Tottenham',
        'Spurs': 'Tottenham',
        'West Ham United': 'West Ham',
        'West Ham United FC': 'West Ham',
        'West Ham': 'West Ham',
        'Newcastle United': 'Newcastle',
        'Newcastle United FC': 'Newcastle',
        'Newcastle': 'Newcastle',
        'Leicester City': 'Leicester',
        'Leicester City FC': 'Leicester',
        'Leicester': 'Leicester',
        'Brighton & Hove Albion': 'Brighton',
        'Brighton and Hove Albion': 'Brighton',
        'Brighton & Hove Albion FC': 'Brighton',
        'Brighton': 'Brighton',
        'Wolverhampton Wanderers': 'Wolves',
        'Wolverhampton Wanderers FC': 'Wolves',
        'Wolves': 'Wolves',
        'Nottingham Forest': "Nott'm Forest",
        'Nottingham Forest FC': "Nott'm Forest",
        "Nott'm Forest": "Nott'm Forest",
        'Arsenal FC': 'Arsenal',
        'Arsenal': 'Arsenal',
        'Liverpool FC': 'Liverpool',
        'Liverpool': 'Liverpool',
        'Chelsea FC': 'Chelsea',
        'Chelsea': 'Chelsea',
        'Everton FC': 'Everton',
        'Everton': 'Everton',
        'Aston Villa FC': 'Aston Villa',
        'Aston Villa': 'Aston Villa',
        'Crystal Palace FC': 'Crystal Palace',
        'Crystal Palace': 'Crystal Palace',
        'Fulham FC': 'Fulham',
        'Fulham': 'Fulham',
        'AFC Bournemouth': 'Bournemouth',
        'Bournemouth': 'Bournemouth',
        'Brentford FC': 'Brentford',
        'Brentford': 'Brentford',
        'Southampton FC': 'Southampton',
        'Southampton': 'Southampton',
        'Ipswich Town': 'Ipswich',
        'Ipswich Town FC': 'Ipswich',
        'Ipswich': 'Ipswich',
    }
    
    return name_map.get(name, name)

# ============================================================================
# LIVE ODDS FETCHER (The-Odds-API)
# ============================================================================

def get_live_odds(api_key):
    """
    Get live Premier League odds from The-Odds-API
    
    Free tier: 500 requests/month
    """
    
    print("\n📊 Fetching live odds from The-Odds-API...")
    
    url = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds/"
    
    params = {
        'apiKey': api_key,
        'regions': 'uk',
        'markets': 'h2h',
        'oddsFormat': 'decimal'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Show API usage
        remaining = response.headers.get('x-requests-remaining', 'Unknown')
        used = response.headers.get('x-requests-used', 'Unknown')
        
        print(f"   API Usage: {used} used, {remaining} remaining this month\n")
        
        all_odds = {}
        fixtures = []
        
        for match in data:
            home_team = match['home_team']
            away_team = match['away_team']
            
            if match['bookmakers']:
                bookmaker = match['bookmakers'][0]
                bookmaker_name = bookmaker['title']
                
                markets = bookmaker['markets'][0]
                outcomes = markets['outcomes']
                
                odds_dict = {}
                
                for outcome in outcomes:
                    name = outcome['name']
                    price = outcome['price']
                    
                    if name == home_team:
                        odds_dict['home win'] = price
                    elif name == away_team:
                        odds_dict['away win'] = price
                    elif name == 'Draw':
                        odds_dict['draw'] = price
                
                if len(odds_dict) == 3:
                    home_clean = clean_team_name(home_team)
                    away_clean = clean_team_name(away_team)
                    
                    all_odds[(home_clean, away_clean)] = odds_dict
                    fixtures.append((home_clean, away_clean))
                    
                    print(f"   {home_clean} vs {away_clean} ({bookmaker_name})")
                    print(f"      Home: {odds_dict['home win']:.2f}, Draw: {odds_dict['draw']:.2f}, Away: {odds_dict['away win']:.2f}")
        
        print(f"\n   ✅ Retrieved odds for {len(all_odds)} matches")
        
        return fixtures, all_odds
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("   ❌ Invalid API key")
        elif e.response.status_code == 429:
            print("   ❌ Rate limit exceeded")
        else:
            print(f"   ❌ HTTP Error: {e}")
        return [], {}
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return [], {}

# ============================================================================
# PREDICTION & ANALYSIS
# ============================================================================

def display_predictions(predictor, fixtures):
    """Display predictions for all fixtures"""
    
    print("\n" + "=" * 70)
    print("🔮 PREMIER LEAGUE PREDICTIONS")
    print("=" * 70)
    
    for home, away in fixtures:
        try:
            pred = predictor.predict_match(home, away)
            
            print(f"\n{home} vs {away}")
            print(f"  Prediction: {pred['prediction']} ({pred['confidence']})")
            print(f"  Probabilities:")
            print(f"    Home: {pred['probabilities']['home_win']:.1%}")
            print(f"    Draw: {pred['probabilities']['draw']:.1%}")
            print(f"    Away: {pred['probabilities']['away_win']:.1%}")
        except Exception as e:
            print(f"\n{home} vs {away}")
            print(f"  ⚠️  Could not generate prediction (teams may not be in training data)")

def find_value_bets(predictor, fixtures, market_odds, bankroll):
    """Find value betting opportunities"""
    
    print("\n" + "=" * 70)
    print("💰 VALUE BETTING OPPORTUNITIES")
    print("=" * 70)
    print(f"\nBankroll: £{bankroll:,.0f}")
    print(f"Strategy: Skip draws, only bet high confidence (>10% edge)\n")
    
    if not market_odds:
        print("❌ No market odds available")
        return []
    
    recommendations = []
    
    for home, away in fixtures:
        try:
            pred = predictor.predict_match(home, away)
            
            if (home, away) not in market_odds:
                continue
            
            odds = market_odds[(home, away)]
            
            # Check home win and away win (SKIP DRAWS)
            for outcome in ['home_win', 'away_win']:
                
                your_prob = pred['probabilities'][outcome]
                outcome_key = outcome.replace('_', ' ')
                
                if outcome_key not in odds:
                    continue
                
                market_prob = 1 / odds[outcome_key]
                edge = your_prob - market_prob
                
                # Minimum edge requirement
                min_edge = 0.10 if pred['confidence'] in ['HIGH', 'MEDIUM-HIGH'] else 0.15
                
                # Check if value bet
                if edge >= min_edge and your_prob >= 0.50:
                    
                    bet_size = bankroll * 0.02  # 2% of bankroll
                    potential_profit = bet_size * (odds[outcome_key] - 1)
                    
                    recommendations.append({
                        'match': f"{home} vs {away}",
                        'bet': outcome.replace('_', ' ').title(),
                        'your_prob': your_prob,
                        'market_odds': odds[outcome_key],
                        'market_prob': market_prob,
                        'edge': edge,
                        'confidence': pred['confidence'],
                        'bet_size': bet_size,
                        'potential_profit': potential_profit
                    })
        
        except Exception as e:
            continue
    
    # Display recommendations
    if recommendations:
        print(f"✅ Found {len(recommendations)} value bets:\n")
        
        total_stake = 0
        total_potential = 0
        
        for i, bet in enumerate(recommendations, 1):
            print(f"{i}. {bet['match']}")
            print(f"   Bet: {bet['bet']}")
            print(f"   Your probability: {bet['your_prob']:.1%}")
            print(f"   Market odds: {bet['market_odds']:.2f} (implies {bet['market_prob']:.1%})")
            print(f"   Edge: {bet['edge']:.1%} ✅")
            print(f"   Confidence: {bet['confidence']}")
            print(f"   ")
            print(f"   💰 Bet: £{bet['bet_size']:.2f}")
            print(f"   💵 Potential profit: £{bet['potential_profit']:.2f}")
            print()
            
            total_stake += bet['bet_size']
            total_potential += bet['potential_profit']
        
        print("=" * 70)
        print(f"TOTAL STAKE: £{total_stake:.2f}")
        print(f"TOTAL POTENTIAL PROFIT: £{total_potential:.2f}")
        print(f"EXPECTED PROFIT (58.8% win rate): £{total_potential * 0.588:.2f}")
        print("=" * 70)
        
    else:
        print("❌ No value bets found this week")
        print("\n   Market is efficient - no edges above threshold")
        print("   👍 Skipping is profitable - wait for better opportunities")
    
    return recommendations

def save_recommendations(recommendations):
    """Save recommendations to CSV"""
    
    if recommendations:
        df = pd.DataFrame(recommendations)
        df['date_generated'] = datetime.now().strftime('%Y-%m-%d %H:%M')
        df.to_csv('football_bets.csv', index=False)
        print(f"\n✅ Recommendations saved to: football_bets.csv")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 70)
    print("⚽ FOOTBALL BETTING OPPORTUNITY FINDER")
    print("=" * 70)
    print(f"\nGenerated: {datetime.now().strftime('%A, %B %d, %Y at %H:%M')}")
    
    # Step 1: Load data and train model
    print("\n1. Loading Premier League data...")
    loader = FootballDataLoader()
    data = loader.get_latest_results('E0', num_seasons=5)
    
    if data is None:
        print("❌ Failed to load data")
        exit()
    
    clean_data = loader.clean_match_data(data)
    
    # Step 2: Train predictor
    print("\n2. Training prediction model...")
    predictor = FootballPredictor('Premier League')
    predictor.load_data(clean_data)
    results = predictor.train_model(test_split=0.2)
    
    # Step 3: Get fixtures and odds from API
    print("\n3. Loading fixtures and live odds...")
    
    if THE_ODDS_API_KEY == 'YOUR_API_KEY_HERE':
        print("\n❌ API key not set!")
        print("   Update THE_ODDS_API_KEY at top of script")
        exit()
    
    fixtures, market_odds = get_live_odds(THE_ODDS_API_KEY)
    
    if not fixtures:
        print("\n❌ No fixtures available")
        print("   This might mean:")
        print("   • No Premier League matches this weekend")
        print("   • API issue")
        print("   • Try again later")
        exit()
    
    print(f"\n   ✅ {len(fixtures)} matches this weekend")
    
    # Step 4: Show predictions
    display_predictions(predictor, fixtures)
    
    # Step 5: Find value bets
    recommendations = find_value_bets(predictor, fixtures, market_odds, BANKROLL)
    
    # Step 6: Save
    if recommendations:
        save_recommendations(recommendations)
        
        print("\n" + "=" * 70)
        print("NEXT STEPS:")
        print("=" * 70)
        print("\n1. Review recommendations above")
        print("2. Log into Smarkets/Betfair")
        print("3. Verify current odds (they change!)")
        print("4. Place recommended bets")
        print("5. Track results in football_bets.csv")
        print("\n💡 TIP: Start small (£5-10) to build confidence")
        print("\n" + "=" * 70)
    
    print("\n✅ Analysis complete!")