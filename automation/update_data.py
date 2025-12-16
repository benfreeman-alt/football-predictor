import pandas as pd
import requests
from datetime import datetime, timedelta
import os

print("=" * 70)
print("AUTOMATED DATA UPDATE SYSTEM")
print("=" * 70)

def update_economic_data():
    """
    Update economic indicators from FRED
    """
    print("\n1. Updating Economic Data...")
    
    # FRED API (free, requires key from https://fred.stlouisfed.org/docs/api/api_key.html)
    # For now, we'll create a manual update script
    
    indicators = {
        'GDPC1': 'GDP',
        'UNRATE': 'Unemployment', 
        'CPIAUCSL': 'Inflation',
        'UMCSENT': 'Consumer Confidence'
    }
    
    print("   Economic data sources:")
    for code, name in indicators.items():
        print(f"   - {name} ({code}): https://fred.stlouisfed.org/series/{code}")
    
    print("\n   ⚠️  Manual step required:")
    print("   1. Visit FRED links above")
    print("   2. Click 'Download' → CSV")
    print("   3. Save to real_data/economic/{CODE}.csv")
    print("   4. Or sign up for FRED API key for full automation")
    
    # Check if files exist and are recent
    econ_dir = 'real_data/economic'
    if os.path.exists(econ_dir):
        files = os.listdir(econ_dir)
        print(f"\n   Current files ({len(files)}):")
        for f in files:
            if f.endswith('.csv'):
                filepath = os.path.join(econ_dir, f)
                mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                age_days = (datetime.now() - mod_time).days
                status = '✅' if age_days < 60 else '⚠️'
                print(f"   {status} {f}: Updated {age_days} days ago")

def update_polling_data():
    """
    Update polling data from FiveThirtyEight or other sources
    """
    print("\n2. Updating Polling Data...")
    
    sources = {
        'FiveThirtyEight': 'https://projects.fivethirtyeight.com/polls/',
        'RealClearPolitics': 'https://www.realclearpolitics.com/epolls/latest_polls/',
        'Silver Bulletin': 'https://www.natesilver.net/'
    }
    
    print("   Polling data sources:")
    for name, url in sources.items():
        print(f"   - {name}: {url}")
    
    print("\n   ⚠️  Manual step required:")
    print("   1. Check sources for 2026 polling data")
    print("   2. Download CSV files when available")
    print("   3. Save to real_data/polling/2026_polls.csv")
    print("   4. Script will auto-detect and process")
    
    # Check existing polling files
    poll_dir = 'real_data/polling'
    if os.path.exists(poll_dir):
        files = os.listdir(poll_dir)
        print(f"\n   Current files ({len(files)}):")
        for f in files:
            if f.endswith('.csv'):
                filepath = os.path.join(poll_dir, f)
                df = pd.read_csv(filepath, nrows=5)
                print(f"   ✅ {f}: {len(pd.read_csv(filepath))} records")

def check_election_results():
    """
    Check for new election results (after 2024)
    """
    print("\n3. Checking Election Results...")
    
    results_file = 'data/real_election_results.csv'
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        latest_year = df['election_year'].max()
        print(f"   Latest election data: {latest_year}")
        
        if latest_year < 2024:
            print("   ⚠️  Missing 2024 results!")
            print("   Update from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VOQCHQ")
        else:
            print("   ✅ Up to date")
    else:
        print("   ❌ No results file found")

def check_model_freshness():
    """
    Check when models were last trained
    """
    print("\n4. Checking Model Freshness...")
    
    model_dir = 'data/models'
    if os.path.exists(model_dir):
        models = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        print(f"   Found {len(models)} model files:")
        
        for model_file in models:
            filepath = os.path.join(model_dir, model_file)
            mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            age_days = (datetime.now() - mod_time).days
            
            if age_days < 30:
                status = '✅ Fresh'
            elif age_days < 90:
                status = '⚠️  Consider retraining'
            else:
                status = '❌ Needs retraining'
            
            print(f"   {status}: {model_file} (trained {age_days} days ago)")
    else:
        print("   ❌ No models directory found")

def generate_update_checklist():
    """
    Create a checklist for manual updates
    """
    print("\n5. Generating Update Checklist...")
    
    checklist = f"""
=" * 70)
UPDATE CHECKLIST - {datetime.now().strftime('%Y-%m-%d')}
======================================================================

MONTHLY UPDATES (1st of each month):
[ ] Download latest GDP data (GDPC1) from FRED
[ ] Download latest Unemployment data (UNRATE) from FRED  
[ ] Download latest CPI/Inflation data (CPIAUCSL) from FRED
[ ] Download latest Consumer Confidence (UMCSENT) from FRED
[ ] Run: python train_with_economics.py

WEEKLY UPDATES (During election season - Aug-Nov 2026):
[ ] Check FiveThirtyEight for new 2026 Senate polling
[ ] Download polling CSVs to real_data/polling/
[ ] Run: python train_with_polling.py
[ ] Run: python confidence_based_system.py

AS NEEDED:
[ ] When new markets open: Run prediction script
[ ] After election: Update results, retrain models
[ ] Quarterly: Review model performance

======================================================================
"""
    
    checklist_file = 'automation/update_checklist.txt'
    os.makedirs('automation', exist_ok=True)
    
    with open(checklist_file, 'w') as f:
        f.write(checklist)
    
    print(f"   ✅ Checklist saved to: {checklist_file}")
    print(checklist)

# Run all checks
if __name__ == "__main__":
    update_economic_data()
    update_polling_data()
    check_election_results()
    check_model_freshness()
    generate_update_checklist()
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\n1. Set calendar reminders:")
    print("   - Monthly: 1st of month → Update economic data")
    print("   - Weekly (Aug-Nov 2026): Check for new polls")
    print("   - Quarterly: Review and retrain models")
    
    print("\n2. Optional: Get FRED API key")
    print("   - Sign up: https://fred.stlouisfed.org/docs/api/api_key.html")
    print("   - Enables automatic economic data updates")
    
    print("\n3. Monitor for 2026 Senate markets")
    print("   - PredictIt: https://www.predictit.org/")
    print("   - Polymarket: https://polymarket.com/")
    print("   - Kalshi: https://kalshi.com/")
    
    print("\n" + "=" * 70)