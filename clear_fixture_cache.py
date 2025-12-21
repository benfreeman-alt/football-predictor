"""
Clear fixture cache to force fresh fetch
Run this to remove old/past fixtures
"""

import os

cache_file = 'data/fixture_cache/auto_fixtures.json'

if os.path.exists(cache_file):
    os.remove(cache_file)
    print(f"✅ Deleted {cache_file}")
    print("Next dashboard load will fetch fresh fixtures!")
else:
    print(f"⚠️  Cache file not found: {cache_file}")