"""
FOOTBALL MARKET DIAGNOSTIC

Run this to test if predictions are working
"""

import sys
import os

# Add path
sys.path.insert(0, os.path.join(os.getcwd(), 'universal_framework', 'markets'))

print("=" * 70)
print("🔍 FOOTBALL MARKET DIAGNOSTIC TEST")
print("=" * 70)

try:
    print("\n1️⃣ Importing football_market...")
    from football_market import FootballMarket
    print("   ✅ Import successful")
    
    print("\n2️⃣ Creating FootballMarket instance...")
    market = FootballMarket()
    print("   ✅ Instance created")
    
    print("\n3️⃣ Initializing market (this takes 30-60 seconds)...")
    print("   Loading data, training model, loading xG, loading injuries...")
    
    success = market.initialize()
    
    if success:
        print("   ✅ Market initialized successfully!")
        
        print("\n4️⃣ Getting predictions...")
        predictions = market.get_predictions()
        
        print(f"\n   📊 Result: {len(predictions)} predictions found")
        
        if predictions:
            print("\n✅ SUCCESS! Predictions are working!")
            print("\n📋 Sample Predictions:")
            print("-" * 70)
            
            for i, pred in enumerate(predictions[:5], 1):
                print(f"\n{i}. {pred['event']}")
                print(f"   Prediction: {pred['prediction']}")
                print(f"   Confidence: {pred['confidence']}")
                print(f"   Probability: {pred['probability']:.1%}")
                print(f"   Injury adjustment: {pred['injury_adjustment']:+.1%}")
        else:
            print("\n❌ PROBLEM: No predictions returned!")
            print("\nPossible causes:")
            print("- get_predictions() returned empty list")
            print("- Predictor not initialized properly")
            print("- Error in prediction loop")
    else:
        print("   ❌ Market initialization failed!")
        print("\nPossible causes:")
        print("- Data files missing")
        print("- Model training error")
        print("- xG cache missing")

except ImportError as e:
    print(f"\n❌ IMPORT ERROR: {e}")
    print("\nPossible causes:")
    print("- football_market.py not in universal_framework/markets/")
    print("- Path issue")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("END OF DIAGNOSTIC")
print("=" * 70)