"""
TEST WHAT DASHBOARD LOADS

See exactly what fixtures the dashboard is getting
"""

import sys
sys.path.insert(0, 'C:\\Users\\User\\Desktop\\prediction-markets\\universal_framework\\markets')

from simple_fixture_loader import SimpleFixtureLoader

print("="*70)
print("TESTING SIMPLE FIXTURE LOADER")
print("="*70)

loader = SimpleFixtureLoader()
fixtures = loader.get_upcoming_fixtures()

print(f"\n{'='*70}")
print(f"LOADED {len(fixtures)} FIXTURES")
print(f"{'='*70}")

for i, f in enumerate(fixtures[:10], 1):
    print(f"\n{i:2d}. {f['home_team']} vs {f['away_team']}")
    print(f"    Date: {f['date']} at {f['time']}")
    print(f"    Venue: {f.get('venue', 'Unknown')}")

print(f"\n{'='*70}")
print("END TEST")
print(f"{'='*70}")