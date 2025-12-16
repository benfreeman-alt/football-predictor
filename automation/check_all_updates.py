import os
from datetime import datetime
import requests

print("=" * 70)
print("CHECKING ALL DATA SOURCES FOR UPDATES")
print("=" * 70)

def check_github_for_2026_polls():
    """Check if 2026 polling data is available"""
    print("\n1. Checking for 2026 Senate Polling...")
    
    urls_to_check = [
        'https://github.com/fivethirtyeight/data/tree/master/polls',
        'https://projects.fivethirtyeight.com/polls/senate/2026/'
    ]
    
    found_2026 = False
    
    for url in urls_to_check:
        try:
            response = requests.get(url, timeout=5)
            if '2026' in response.text and 'senate' in response.text.lower():
                print(f"   ✅ FOUND: 2026 data may be available at {url}")
                found_2026 = True
            else:
                print(f"   ⚠️  No 2026 data yet at {url}")
        except:
            print(f"   ❌ Could not check {url}")
    
    if not found_2026:
        current_month = datetime.now().strftime('%B %Y')
        print(f"\n   Current: {current_month}")
        print(f"   Expected: June 2026 or later")
        print(f"   Action: Check again next month")
    
    return found_2026

def check_mit_election_data():
    """Check if new election results are available"""
    print("\n2. Checking for New Election Results...")
    
    # Check what we have
    elections_file = 'real_data/elections/countypres_2000-2024.csv'
    
    if os.path.exists(elections_file):
        import pandas as pd
        df = pd.read_csv(elections_file)
        
        if 'year' in df.columns:
            latest_year = df['year'].max()
        else:
            latest_year = 2024  # Assume based on filename
        
        print(f"   Current data: Through {latest_year}")
        
        current_year = datetime.now().year
        
        if current_year > latest_year and datetime.now().month >= 11:
            print(f"   ⚠️  {current_year} election has occurred!")
            print(f"   Action: Update from MIT Election Lab")
            print(f"   URL: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VOQCHQ")
            return False
        else:
            print(f"   ✅ Up to date (next election: {current_year if current_year % 2 == 0 else current_year + 1})")
            return True
    else:
        print(f"   ❌ Elections file not found!")
        return False

def check_economic_data():
    """Check if economic data is fresh"""
    print("\n3. Checking Economic Data Freshness...")
    
    econ_files = {
        'GDPC1.csv': 'GDP',
        'UNRATE.csv': 'Unemployment',
        'CPIAUCSL.csv': 'Inflation',
        'UMCSENT.csv': 'Consumer Confidence'
    }
    
    econ_dir = 'real_data/economic'
    all_fresh = True
    
    for filename, name in econ_files.items():
        filepath = os.path.join(econ_dir, filename)
        
        if os.path.exists(filepath):
            mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            age_days = (datetime.now() - mod_time).days
            
            if age_days < 45:  # Updated in last 45 days
                print(f"   ✅ {name}: Fresh ({age_days} days old)")
            else:
                print(f"   ⚠️  {name}: Needs update ({age_days} days old)")
                all_fresh = False
        else:
            print(f"   ❌ {name}: File missing!")
            all_fresh = False
    
    if not all_fresh:
        print(f"\n   Action: Run python automation/auto_update_fred.py")
    
    return all_fresh

def generate_action_plan():
    """Create action plan based on checks"""
    print("\n" + "=" * 70)
    print("ACTION PLAN")
    print("=" * 70)
    
    actions = []
    
    # Check if we're in election season
    current_date = datetime.now()
    current_month = current_date.month
    current_year = current_date.year
    
    if current_year == 2026:
        if 6 <= current_month <= 11:  # June-November 2026
            actions.append("🔔 ELECTION SEASON - Check for polling data weekly")
            actions.append("   Visit: https://github.com/fivethirtyeight/data/tree/master/polls")
        
        if current_month == 9:  # September
            actions.append("🔔 CRITICAL MONTH - September Q3 economic data released!")
            actions.append("   Run: python automation/auto_update_fred.py")
            actions.append("   Run: python train_with_economics.py")
        
        if current_month == 10:  # October
            actions.append("🔔 BETTING MONTH - Generate predictions and place bets")
            actions.append("   Run: python predict_2026_senate.py")
        
        if current_month == 11:  # November
            actions.append("🔔 ELECTION MONTH - Results coming!")
        
        if current_month == 12:  # December
            actions.append("🔔 POST-ELECTION - Update with 2026 results")
            actions.append("   Download: MIT Election Lab data")
            actions.append("   Run: python load_real_data.py")
            actions.append("   Run: python train_with_economics.py")
    
    if not actions:
        actions.append("✅ Nothing urgent - normal monthly monitoring")
        actions.append("   Economic data updates automatically on 1st of month")
    
    print()
    for action in actions:
        print(action)
    
    print("\n" + "=" * 70)

# Run all checks
if __name__ == "__main__":
    check_economic_data()
    found_polls = check_github_for_2026_polls()
    elections_current = check_mit_election_data()
    generate_action_plan()
    
    print("\n✅ Check complete!")
    print("\nRerun this monthly: python automation/check_all_updates.py")
    print("=" * 70)