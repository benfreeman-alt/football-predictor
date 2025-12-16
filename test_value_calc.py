"""
TEST VALUE CALCULATOR LOGIC

This will show if the bug is fixed
"""

import sys
sys.path.insert(0, 'C:\\Users\\User\\Desktop\\prediction-markets\\universal_framework\\markets')

from value_calculator import ValueCalculator

calculator = ValueCalculator(min_edge=0.05)

# Test case: Model says AWAY WIN but home odds are attractive
prediction = {
    'event': 'Wolves vs Brentford',
    'probabilities': {
        'home_win': 0.359,  # 35.9% - Model says AWAY/DRAW will win!
    },
    'confidence': 'MEDIUM-HIGH'
}

odds = {
    'home_win': 3.90,  # High odds (attractive)
    'away_win': 2.15,
    'draw': 3.60,
    'double_chance_away_draw': 1.44,  # What model predicts
}

result = calculator.calculate_value(prediction, odds)

print("=" * 70)
print("VALUE CALCULATOR TEST")
print("=" * 70)
print(f"\nMatch: {result['match']}")
print(f"Model: {result['model_prediction']['home_win_prob']:.1%} home win")
print(f"Model prediction: NOT HOME WIN (Away/Draw)")
print(f"Confidence: {result['model_prediction']['confidence']}")

print(f"\nExpected Values:")
print(f"  Home Win EV: {result['value_analysis']['home_win_ev']*100:+.1f}%")
print(f"  Not Home Win EV: {result['value_analysis']['not_home_win_ev']*100:+.1f}%")

rec = result['recommendation']
print(f"\n{'='*70}")
print(f"RECOMMENDATION: {rec['action']}")
print(f"{'='*70}")

if rec['action'] == 'BET':
    print(f"Bet Type: {rec['bet_type']}")
    print(f"\n⚠️  CHECK THIS:")
    if rec['bet_type'] == 'Home Win':
        print("❌ BUG STILL EXISTS!")
        print("   System recommending HOME WIN when model says AWAY/DRAW!")
    else:
        print("✅ CORRECT!")
        print("   System recommending what model predicts!")
else:
    print("Reason:", rec['reason'])