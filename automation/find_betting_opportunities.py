import pandas as pd
import requests
from datetime import datetime
import json

print("=" * 70)
print("AUTOMATED BET FINDER - 2026 SENATE")
print("=" * 70)

def load_predictions():
    """Load your model predictions"""
    print("\n1. Loading your predictions...")
    
    predictions_file = 'automation/2026_senate_predictions.csv'
    
    try:
        predictions = pd.read_csv(predictions_file)
        print(f"   ✅ Loaded {len(predictions)} predictions")
        return predictions
    except FileNotFoundError:
        print(f"   ❌ Predictions file not found!")
        print(f"   Run: python predict_2026_senate.py first")
        return None

def get_smarkets_odds():
    """
    Get current odds from Smarkets
    
    NOTE: This is a template - Smarkets requires API access
    For now, you'll input market odds manually
    """
    print("\n2. Getting market odds...")
    print("   ⚠️  Automatic odds fetching requires API access")
    print("   For now, we'll use manual input method")
    
    # In future, with Smarkets API:
    # odds = fetch_from_smarkets_api()
    
    return None

def manual_odds_entry(predictions):
    """
    Manual method: Create template for you to fill in market odds
    """
    print("\n2. Creating market odds template...")
    
    # Create template with just the races you should bet on
    bettable = predictions[predictions['should_bet'] == True].copy()
    
    if len(bettable) == 0:
        print("   ⚠️  No bets recommended (all LOW confidence)")
        return None
    
    # Create template
    template = bettable[['state', 'state_full', 'prediction', 'confidence_level', 'rep_probability']].copy()
    template['market_rep_odds'] = None
    template['market_dem_odds'] = None
    
    template_file = 'automation/market_odds_template.csv'
    template.to_csv(template_file, index=False)
    
    print(f"   ✅ Template created: {template_file}")
    print(f"\n   INSTRUCTIONS:")
    print(f"   1. Open {template_file} in Excel")
    print(f"   2. For each state, go to Smarkets/Betfair")
    print(f"   3. Fill in market_rep_odds and market_dem_odds columns")
    print(f"   4. Save the file")
    print(f"   5. Run this script again")
    
    return template_file

def load_market_odds():
    """Load market odds that you filled in"""
    odds_file = 'automation/market_odds_template.csv'
    
    try:
        odds = pd.read_csv(odds_file)
        
        # Check if filled in
        if odds['market_rep_odds'].isna().any():
            print("   ⚠️  Market odds not filled in yet")
            print(f"   Open {odds_file} and add market odds")
            return None
        
        print(f"   ✅ Loaded market odds for {len(odds)} races")
        return odds
    
    except FileNotFoundError:
        return None

def calculate_edge(your_prob, market_odds):
    """
    Calculate your edge vs market
    
    your_prob: Your probability (e.g., 0.72 for 72%)
    market_odds: Market decimal odds (e.g., 1.85)
    """
    market_implied_prob = 1 / market_odds
    edge = your_prob - market_implied_prob
    return edge

def calculate_kelly_bet(your_prob, market_odds, bankroll, confidence_multiplier, max_pct=0.05):
    """
    Calculate optimal bet size using Kelly Criterion
    
    Returns bet size capped at max_pct of bankroll
    """
    # Kelly formula: (bp - q) / b
    # b = odds - 1
    # p = your probability
    # q = 1 - p
    
    b = market_odds - 1
    p = your_prob
    q = 1 - p
    
    kelly = (b * p - q) / b
    
    # Quarter Kelly (conservative)
    quarter_kelly = kelly / 4
    
    # Apply confidence multiplier
    adjusted_kelly = quarter_kelly * confidence_multiplier
    
    # Cap at max percentage
    capped_kelly = min(adjusted_kelly, max_pct)
    
    # Calculate bet size
    bet_size = bankroll * capped_kelly
    
    return bet_size, kelly, quarter_kelly

def generate_bet_recommendations(predictions, market_odds, bankroll=1000):
    """
    Generate specific betting recommendations with exact amounts
    """
    print("\n3. Generating bet recommendations...")
    print(f"   Bankroll: £{bankroll:,.0f}")
    print()
    
    # Merge predictions with market odds
    merged = predictions.merge(
        market_odds[['state', 'market_rep_odds', 'market_dem_odds']], 
        on='state', 
        how='left'
    )
    
    # Confidence multipliers
    confidence_map = {
        'HIGH': 1.0,
        'MEDIUM-HIGH': 0.75,
        'MEDIUM': 0.5,
        'MEDIUM-LOW': 0.25,
        'LOW': 0.0
    }
    
    recommendations = []
    
    for _, row in merged.iterrows():
        if not row['should_bet']:
            continue
        
        if pd.isna(row['market_rep_odds']):
            continue
        
        # Determine which side to bet
        if row['prediction'] == 'Republican':
            your_prob = row['rep_probability']
            market_odds = row['market_rep_odds']
            bet_side = 'Republican'
        else:
            your_prob = row['dem_probability']
            market_odds = row['market_dem_odds']
            bet_side = 'Democrat'
        
        # Calculate edge
        edge = calculate_edge(your_prob, market_odds)
        edge_pct = edge * 100
        
        # Get confidence multiplier
        conf_mult = confidence_map.get(row['confidence_level'], 0.5)
        
        # Calculate bet size
        bet_size, full_kelly, quarter_kelly = calculate_kelly_bet(
            your_prob, 
            market_odds, 
            bankroll, 
            conf_mult
        )
        
        # Calculate potential profit
        potential_profit = bet_size * (market_odds - 1)
        
        # Determine if should bet (need >10% edge for HIGH/MED-HIGH, >15% for MEDIUM)
        min_edge_map = {
            'HIGH': 0.10,
            'MEDIUM-HIGH': 0.10,
            'MEDIUM': 0.15,
            'MEDIUM-LOW': 0.20,
            'LOW': 1.0  # Never bet
        }
        
        min_edge = min_edge_map.get(row['confidence_level'], 0.15)
        
        should_actually_bet = edge >= min_edge and bet_size >= 5  # Min £5 bet
        
        recommendations.append({
            'state': row['state'],
            'state_full': row['state_full'],
            'bet_side': bet_side,
            'confidence': row['confidence_level'],
            'your_prob': your_prob,
            'market_odds': market_odds,
            'edge': edge,
            'edge_pct': edge_pct,
            'bet_size': bet_size,
            'potential_profit': potential_profit,
            'min_edge_required': min_edge * 100,
            'should_bet': should_actually_bet,
            'reason': row['reason'] if 'reason' in row else ''
        })
    
    return pd.DataFrame(recommendations)

def display_recommendations(recommendations, bankroll):
    """Display betting recommendations in clear format"""
    
    if len(recommendations) == 0:
        print("   ❌ No betting opportunities found")
        return
    
    # Filter to actual bets
    actual_bets = recommendations[recommendations['should_bet'] == True].copy()
    skips = recommendations[recommendations['should_bet'] == False].copy()
    
    print("=" * 70)
    print("BET RECOMMENDATIONS")
    print("=" * 70)
    
    if len(actual_bets) > 0:
        print(f"\n🎯 RECOMMENDED BETS ({len(actual_bets)} races):")
        print()
        
        total_stake = 0
        total_potential = 0
        
        # Sort by confidence then edge
        actual_bets = actual_bets.sort_values(['confidence', 'edge_pct'], ascending=[True, False])
        
        for i, (_, row) in enumerate(actual_bets.iterrows(), 1):
            print(f"{i}. {row['state']} - {row['state_full']}")
            print(f"   Bet: {row['bet_side']}")
            print(f"   Confidence: {row['confidence']}")
            print(f"   Your probability: {row['your_prob']:.1%}")
            print(f"   Market odds: {row['market_odds']:.2f} (implied: {1/row['market_odds']:.1%})")
            print(f"   Edge: {row['edge_pct']:+.1f}% ✅" if row['edge_pct'] > 15 else f"   Edge: {row['edge_pct']:+.1f}%")
            print(f"   ")
            print(f"   💰 BET SIZE: £{row['bet_size']:.2f}")
            print(f"   💵 Potential profit: £{row['potential_profit']:.2f}")
            print()
            
            total_stake += row['bet_size']
            total_potential += row['potential_profit']
        
        print("-" * 70)
        print(f"TOTAL STAKES: £{total_stake:.2f} ({total_stake/bankroll:.1%} of bankroll)")
        print(f"TOTAL POTENTIAL PROFIT: £{total_potential:.2f}")
        print(f"EXPECTED PROFIT (at your accuracy): £{total_potential * 0.95:.2f}")
        print("-" * 70)
        
        # Position limits check
        if total_stake > bankroll * 0.25:
            print(f"\n⚠️  WARNING: Total stake exceeds 25% limit!")
            print(f"   Recommended max: £{bankroll * 0.25:.2f}")
            print(f"   Current total: £{total_stake:.2f}")
            print(f"   Reduce bet sizes proportionally")
    else:
        print("\n⚠️  No bets meet minimum edge requirements")
    
    if len(skips) > 0:
        print(f"\n\n⏭️  SKIPPED ({len(skips)} races):")
        print()
        
        for _, row in skips.iterrows():
            reason = "Edge too small" if row['edge_pct'] < row['min_edge_required'] else "Bet size too small"
            print(f"   {row['state']}: {row['bet_side']} - {reason}")
            print(f"       Edge: {row['edge_pct']:+.1f}% (need {row['min_edge_required']:.0f}%)")
            print()

def save_recommendations(recommendations):
    """Save recommendations to CSV for reference"""
    output_file = 'automation/betting_recommendations.csv'
    recommendations.to_csv(output_file, index=False)
    print(f"\n✅ Full recommendations saved to: {output_file}")
    return output_file

def create_bet_tracking_sheet(recommendations):
    """Create a spreadsheet to track your bets"""
    
    actual_bets = recommendations[recommendations['should_bet'] == True].copy()
    
    if len(actual_bets) == 0:
        return
    
    tracking = actual_bets[[
        'state', 'state_full', 'bet_side', 'confidence', 
        'market_odds', 'bet_size', 'potential_profit'
    ]].copy()
    
    tracking['date_placed'] = datetime.now().strftime('%Y-%m-%d')
    tracking['actual_result'] = ''
    tracking['won'] = ''
    tracking['actual_profit'] = ''
    tracking['notes'] = ''
    
    tracking_file = 'automation/bet_tracking_sheet.csv'
    tracking.to_csv(tracking_file, index=False)
    
    print(f"✅ Bet tracking sheet created: {tracking_file}")
    print(f"\n   Use this to track your bets after placing them")

# Main execution
if __name__ == "__main__":
    # Settings
    BANKROLL = 1000  # ← CHANGE THIS TO YOUR ACTUAL BANKROLL
    
    print(f"\nSettings:")
    print(f"  Bankroll: £{BANKROLL:,.0f}")
    print(f"  Max per bet: {5}% (£{BANKROLL * 0.05:.2f})")
    print(f"  Max total deployed: {25}% (£{BANKROLL * 0.25:.2f})")
    
    # Step 1: Load predictions
    predictions = load_predictions()
    
    if predictions is None:
        print("\n❌ Cannot proceed without predictions")
        print("   Run: python predict_2026_senate.py")
        exit()
    
    # Step 2: Get market odds
    market_odds = load_market_odds()
    
    if market_odds is None:
        # Create template for manual entry
        manual_odds_entry(predictions)
        print("\n" + "=" * 70)
        print("NEXT STEPS:")
        print("=" * 70)
        print("\n1. Open automation/market_odds_template.csv")
        print("2. Fill in market odds from Smarkets/Betfair")
        print("3. Save the file")
        print("4. Run this script again: python automation/find_betting_opportunities.py")
        exit()
    
    # Step 3: Generate recommendations
    recommendations = generate_bet_recommendations(predictions, market_odds, BANKROLL)
    
    # Step 4: Display
    display_recommendations(recommendations, BANKROLL)
    
    # Step 5: Save
    save_recommendations(recommendations)
    create_bet_tracking_sheet(recommendations)
    
    print("\n" + "=" * 70)
    print("READY TO BET!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review recommendations above")
    print("2. Log into Smarkets/Betfair")
    print("3. Place each recommended bet")
    print("4. Update bet_tracking_sheet.csv with results")
    print("\n" + "=" * 70)