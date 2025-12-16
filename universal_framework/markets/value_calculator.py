"""
VALUE BET CALCULATOR

Compares model predictions with bookmaker odds
Calculates expected value and identifies profitable bets
"""

class ValueCalculator:
    """Calculate betting value and make recommendations"""
    
    def __init__(self, min_edge=0.05):
        """
        Args:
            min_edge: Minimum edge required to recommend bet (default 5%)
        """
        self.min_edge = min_edge
    
    def calculate_value(self, prediction_data, odds_data):
        """
        Calculate betting value for a match
        
        Args:
            prediction_data: Dict with model prediction
            odds_data: Dict with bookmaker odds
        
        Returns:
            Dict with value analysis and recommendations
        """
        
        if not odds_data:
            return self._no_odds_recommendation(prediction_data)
        
        home_win_prob = prediction_data['probabilities']['home_win']
        not_home_win_prob = 1 - home_win_prob
        
        # Get odds
        home_win_odds = odds_data.get('home_win', 0)
        away_win_odds = odds_data.get('away_win', 0)
        draw_odds = odds_data.get('draw', 0)
        double_chance_away_draw = odds_data.get('double_chance_away_draw', 0)
        double_chance_home_draw = odds_data.get('double_chance_home_draw', 0)
        
        # Calculate expected values
        home_win_ev = (home_win_prob * home_win_odds) - 1 if home_win_odds else -1
        not_home_win_ev = (not_home_win_prob * double_chance_away_draw) - 1 if double_chance_away_draw else -1
        
        # Determine best bet
        recommendation = self._make_recommendation(
            home_win_prob,
            home_win_ev,
            not_home_win_ev,
            home_win_odds,
            double_chance_away_draw,
            prediction_data['confidence']
        )
        
        return {
            'match': prediction_data['event'],
            'model_prediction': {
                'home_win_prob': home_win_prob,
                'not_home_win_prob': not_home_win_prob,
                'confidence': prediction_data['confidence']
            },
            'odds': {
                'home_win': home_win_odds,
                'away_win': away_win_odds,
                'draw': draw_odds,
                'double_chance_away_draw': double_chance_away_draw,
                'double_chance_home_draw': double_chance_home_draw,
            },
            'value_analysis': {
                'home_win_ev': home_win_ev,
                'not_home_win_ev': not_home_win_ev,
                'home_win_implied_prob': 1/home_win_odds if home_win_odds else 0,
                'not_home_win_implied_prob': 1/double_chance_away_draw if double_chance_away_draw else 0,
            },
            'recommendation': recommendation
        }
    
    def _make_recommendation(self, home_prob, home_ev, not_home_ev, home_odds, dc_odds, confidence):
        """Make betting recommendation based on value and confidence"""
        
        # Determine what the model actually predicts
        model_predicts_home = home_prob > 0.55
        model_predicts_not_home = home_prob < 0.45
        
        # Only consider bets that match model prediction
        best_bet = None
        best_ev = -1
        bet_type = None
        odds = 0
        
        if model_predicts_home:
            # Model says home win - only consider home win bet
            if home_ev > self.min_edge:
                best_ev = home_ev
                best_bet = "HOME_WIN"
                bet_type = "Home Win"
                odds = home_odds
        
        elif model_predicts_not_home:
            # Model says not home win - only consider double chance
            if not_home_ev > self.min_edge:
                best_ev = not_home_ev
                best_bet = "NOT_HOME_WIN"
                bet_type = "Away Win or Draw (Double Chance)"
                odds = dc_odds
        
        # No value found OR model uncertain
        if not best_bet:
            return {
                'action': 'SKIP',
                'reason': 'No positive expected value' if (model_predicts_home or model_predicts_not_home) else 'Model uncertain (45-55%)',
                'bet_type': None,
                'odds': 0,
                'expected_value': 0,
                'stake_recommendation': 0,
                'rating': '❌'
            }
        
        # Calculate stake (Kelly Criterion simplified)
        if confidence == 'HIGH':
            stake_pct = min(2.0, best_ev * 0.25)  # Up to 2% for high confidence
        elif confidence == 'MEDIUM-HIGH':
            stake_pct = min(1.5, best_ev * 0.20)  # Up to 1.5%
        elif confidence == 'MEDIUM':
            stake_pct = min(1.0, best_ev * 0.15)  # Up to 1%
        else:
            stake_pct = 0.5  # Minimum
        
        # Rating
        if best_ev > 0.20 and confidence == 'HIGH':
            rating = '⭐⭐⭐'  # Excellent value
        elif best_ev > 0.15 and confidence in ['HIGH', 'MEDIUM-HIGH']:
            rating = '⭐⭐'  # Good value
        elif best_ev > 0.10:
            rating = '⭐'  # Decent value
        else:
            rating = '✓'  # Small edge
        
        return {
            'action': 'BET',
            'bet_type': bet_type,
            'odds': odds,
            'expected_value': best_ev,
            'stake_recommendation': stake_pct,
            'rating': rating,
            'reason': f'{best_ev*100:.1f}% edge with {confidence} confidence'
        }
    
    def _no_odds_recommendation(self, prediction_data):
        """Recommendation when no odds available"""
        
        confidence = prediction_data['confidence']
        home_prob = prediction_data['probabilities']['home_win']
        
        if confidence in ['HIGH', 'MEDIUM-HIGH']:
            if home_prob > 0.55:
                bet_type = "Home Win"
            elif home_prob < 0.45:
                bet_type = "Away Win or Draw (Double Chance)"
            else:
                bet_type = None
        else:
            bet_type = None
        
        return {
            'match': prediction_data['event'],
            'model_prediction': {
                'home_win_prob': home_prob,
                'confidence': confidence
            },
            'odds': None,
            'value_analysis': None,
            'recommendation': {
                'action': 'BET' if bet_type else 'SKIP',
                'bet_type': bet_type,
                'odds': 0,
                'expected_value': 0,
                'stake_recommendation': 1.0 if bet_type else 0,
                'rating': '?' if bet_type else '❌',
                'reason': 'No odds data - based on model confidence only'
            }
        }
    
    def get_top_value_bets(self, all_recommendations, max_bets=10):
        """Get top value bets sorted by expected value"""
        
        # Filter to only bets
        bets = [r for r in all_recommendations if r['recommendation']['action'] == 'BET']
        
        # Sort by expected value
        bets.sort(key=lambda x: x['recommendation']['expected_value'], reverse=True)
        
        return bets[:max_bets]

# Testing
if __name__ == "__main__":
    calculator = ValueCalculator(min_edge=0.05)
    
    # Test case
    prediction = {
        'event': 'Man United vs Bournemouth',
        'probabilities': {
            'home_win': 0.281,
        },
        'confidence': 'HIGH'
    }
    
    odds = {
        'home_win': 2.10,
        'away_win': 3.50,
        'draw': 3.40,
        'double_chance_away_draw': 1.65,
    }
    
    result = calculator.calculate_value(prediction, odds)
    
    print("=" * 70)
    print("VALUE ANALYSIS TEST")
    print("=" * 70)
    print(f"\nMatch: {result['match']}")
    print(f"Model: {result['model_prediction']['home_win_prob']:.1%} home win")
    print(f"Confidence: {result['model_prediction']['confidence']}")
    
    if result['odds']:
        print(f"\nOdds:")
        print(f"  Home Win: {result['odds']['home_win']:.2f}")
        print(f"  Double Chance: {result['odds']['double_chance_away_draw']:.2f}")
        
        print(f"\nExpected Value:")
        print(f"  Home Win EV: {result['value_analysis']['home_win_ev']*100:+.1f}%")
        print(f"  Not Home Win EV: {result['value_analysis']['not_home_win_ev']*100:+.1f}%")
    
    rec = result['recommendation']
    print(f"\n{'='*70}")
    print(f"RECOMMENDATION: {rec['action']} {rec['rating']}")
    print(f"{'='*70}")
    
    if rec['action'] == 'BET':
        print(f"Bet Type: {rec['bet_type']}")
        print(f"Odds: {rec['odds']:.2f}")
        print(f"Expected Value: {rec['expected_value']*100:+.1f}%")
        print(f"Stake: {rec['stake_recommendation']:.1f}% of bankroll")
        print(f"Reason: {rec['reason']}")