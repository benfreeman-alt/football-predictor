import pandas as pd
import os
from datetime import datetime

print("=" * 70)
print("AUTOMATED DATA UPDATE SYSTEM")
print("=" * 70)

# Detect base directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"\nBase directory: {base_dir}")

def update_economic_data():
    """Check economic data freshness"""
    print("\n1. Updating Economic Data...")
    
    indicators = {
        'GDPC1': 'GDP',
        'UNRATE': 'Unemployment', 
        'CPIAUCSL': 'Inflation',
        'UMCSENT': 'Consumer Confidence'
    }
    
    print("   Economic data sources:")
    for code, name in indicators.items():
        print(f"   - {name} ({code}): https://fred.stlouisfed.org/series/{code}")
    
    # Check if files exist
    econ_dir = os.path.join(base_dir, 'real_data', 'economic')
    
    if os.path.exists(econ_dir):
        files = [f for f in os.listdir(econ_dir) if f.endswith('.csv')]
        print(f"\n   Current files ({len(files)}):")
        
        for f in files:
            filepath = os.path.join(econ_dir, f)
            mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            age_days = (datetime.now() - mod_time).days
            
            # Load and check latest date
            try:
                df = pd.read_csv(filepath)
                df.columns = ['DATE', 'VALUE']
                df['DATE'] = pd.to_datetime(df['DATE'])
                latest_date = df['DATE'].max()
                latest_value = df['VALUE'].iloc[-1]
                
                status = '✅' if age_days < 60 else '⚠️'
                print(f"   {status} {f}")
                print(f"      Last updated: {mod_time.strftime('%Y-%m-%d')} ({age_days} days ago)")
                print(f"      Latest data: {latest_date.strftime('%Y-%m-%d')} = {latest_value:.2f}")
            except Exception as e:
                print(f"   ❌ {f}: Error reading file")
    else:
        print(f"   ❌ Directory not found: {econ_dir}")

def update_polling_data():
    """Check polling data"""
    print("\n2. Updating Polling Data...")
    
    sources = {
        'FiveThirtyEight': 'https://projects.fivethirtyeight.com/polls/',
        'RealClearPolitics': 'https://www.realclearpolitics.com/epolls/latest_polls/',
        'Silver Bulletin': 'https://www.natesilver.net/'
    }
    
    print("   Polling data sources:")
    for name, url in sources.items():
        print(f"   - {name}: {url}")
    
    # Check existing polling files
    poll_dir = os.path.join(base_dir, 'real_data', 'polling')
    
    if os.path.exists(poll_dir):
        files = [f for f in os.listdir(poll_dir) if f.endswith('.csv')]
        print(f"\n   Current files ({len(files)}):")
        
        for f in files:
            filepath = os.path.join(poll_dir, f)
            try:
                df = pd.read_csv(filepath)
                num_records = len(df)
                
                # Try to find year column
                year_col = None
                for col in ['cycle', 'year', 'election_year']:
                    if col in df.columns:
                        year_col = col
                        break
                
                if year_col:
                    years = sorted(df[year_col].unique())
                    print(f"   ✅ {f}: {num_records:,} records, years: {years}")
                else:
                    print(f"   ✅ {f}: {num_records:,} records")
            except Exception as e:
                print(f"   ❌ {f}: Error reading file")
    else:
        print(f"   ❌ Directory not found: {poll_dir}")
    
    print("\n   📅 2026 Polling Timeline:")
    print("   - Jan-May 2026: Early polling (unreliable)")
    print("   - Jun-Aug 2026: Start tracking regularly")
    print("   - Sep-Nov 2026: Update weekly")

def check_election_results():
    """Check for election results"""
    print("\n3. Checking Election Results...")
    
    results_file = os.path.join(base_dir, 'data', 'real_election_results.csv')
    
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        latest_year = df['election_year'].max()
        num_states = df[df['election_year'] == latest_year]['state'].nunique()
        
        print(f"   ✅ Results file found")
        print(f"   Latest election: {latest_year}")
        print(f"   States covered: {num_states}")
        
        if latest_year < 2024:
            print(f"\n   ⚠️  Missing 2024 results!")
            print("   Update from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VOQCHQ")
        else:
            print(f"   ✅ Includes 2024 results")
    else:
        print(f"   ❌ No results file at: {results_file}")

def check_model_freshness():
    """Check model training dates"""
    print("\n4. Checking Model Freshness...")
    
    model_dir = os.path.join(base_dir, 'data', 'models')
    
    if os.path.exists(model_dir):
        models = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        
        if models:
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
                
                print(f"   {status}: {model_file}")
                print(f"      Trained: {mod_time.strftime('%Y-%m-%d')} ({age_days} days ago)")
        else:
            print("   ⚠️  No model files found")
    else:
        print(f"   ❌ Models directory not found: {model_dir}")

def analyze_system_readiness():
    """Overall system readiness for 2026"""
    print("\n5. System Readiness for 2026...")
    
    checks = {
        'economic_data': os.path.exists(os.path.join(base_dir, 'real_data', 'economic', 'GDPC1.csv')),
        'polling_data': os.path.exists(os.path.join(base_dir, 'real_data', 'polling')),
        'election_results': os.path.exists(os.path.join(base_dir, 'data', 'real_election_results.csv')),
        'trained_models': os.path.exists(os.path.join(base_dir, 'data', 'models', 'best_model_real_data.pkl')),
        'automation_scripts': os.path.exists(os.path.join(base_dir, 'automation'))
    }
    
    ready_count = sum(checks.values())
    total_count = len(checks)
    
    print(f"\n   System Components: {ready_count}/{total_count} ready")
    print(f"   ✅ Economic data: {'Yes' if checks['economic_data'] else 'No'}")
    print(f"   ✅ Polling infrastructure: {'Yes' if checks['polling_data'] else 'No'}")
    print(f"   ✅ Historical results: {'Yes' if checks['election_results'] else 'No'}")
    print(f"   ✅ Trained models: {'Yes' if checks['trained_models'] else 'No'}")
    print(f"   ✅ Automation: {'Yes' if checks['automation_scripts'] else 'No'}")
    
    if ready_count == total_count:
        print(f"\n   🎯 SYSTEM READY for 2026!")
        print(f"   Next: Monitor for Q3 2026 economic data and polling")
    elif ready_count >= total_count - 1:
        print(f"\n   ⚠️  Almost ready - missing 1 component")
    else:
        print(f"\n   ❌ Need to set up {total_count - ready_count} more components")

def generate_2026_timeline():
    """Create timeline for 2026 predictions"""
    print("\n6. 2026 Prediction Timeline...")
    
    timeline = """
======================================================================
2026 MIDTERM ELECTION TIMELINE
======================================================================

PREPARATION PHASE (Now - May 2026):
  ✅ Dec 2025: Build prediction system
  [ ] Jan 2026: Monthly economic data updates
  [ ] Feb 2026: Monthly economic data updates
  [ ] Mar 2026: Monthly economic data updates
  [ ] Apr 2026: Monthly economic data updates
  [ ] May 2026: Quarterly model retraining

EARLY TRACKING (Jun - Aug 2026):
  [ ] Jun 2026: Markets open, start tracking
  [ ] Jul 2026: Q2 economic data available
  [ ] Aug 2026: Begin weekly polling checks
  [ ] Aug 2026: Generate initial predictions

ACTIVE BETTING (Sep - Nov 2026):
  [ ] Sep 2026: Q3 economic data (CRITICAL!)
  [ ] Sep 2026: Retrain with Q3 2026 data
  [ ] Sep 2026: Weekly polling updates
  [ ] Oct 2026: Generate final predictions
  [ ] Oct 2026: Place confidence-based bets
  [ ] Nov 5, 2026: ELECTION DAY

POST-ELECTION (Nov 2026 - Jan 2027):
  [ ] Nov 2026: Collect final results
  [ ] Dec 2026: Analyze performance
  [ ] Dec 2026: Update models with 2026 data
  [ ] Jan 2027: Document lessons learned

KEY DATES:
  - September 2026: Q3 economic data critical for predictions
  - October 2026: Peak betting period
  - November 5, 2026: Election Day

======================================================================
"""
    
    timeline_file = os.path.join(base_dir, 'automation', '2026_timeline.txt')
    
    with open(timeline_file, 'w') as f:
        f.write(timeline)
    
    print(timeline)
    print(f"   ✅ Timeline saved to: {timeline_file}")

def generate_update_checklist():
    """Create monthly checklist"""
    print("\n7. Generating Monthly Checklist...")
    
    checklist = f"""
======================================================================
MONTHLY UPDATE CHECKLIST - {datetime.now().strftime('%B %Y')}
======================================================================

DATA UPDATES (Do on 1st of each month):
[ ] 1. Visit https://fred.stlouisfed.org/series/GDPC1
       Download → Save as real_data/economic/GDPC1.csv
       
[ ] 2. Visit https://fred.stlouisfed.org/series/UNRATE
       Download → Save as real_data/economic/UNRATE.csv
       
[ ] 3. Visit https://fred.stlouisfed.org/series/CPIAUCSL
       Download → Save as real_data/economic/CPIAUCSL.csv
       
[ ] 4. Visit https://fred.stlouisfed.org/series/UMCSENT
       Download → Save as real_data/economic/UMCSENT.csv

MODEL UPDATES:
[ ] 5. Run: python train_with_economics.py
       (Only needed after new economic data)

POLLING UPDATES (Aug-Nov 2026 only):
[ ] 6. Check https://projects.fivethirtyeight.com/polls/
       Download any new 2026 Senate polling
       
[ ] 7. Run: python train_with_polling.py
       (Only if new polling available)

PREDICTION GENERATION:
[ ] 8. Run: python automation/generate_predictions.py
       Review betting recommendations

TRACKING:
[ ] 9. Document any predictions made
[ ] 10. Track market prices vs predictions

======================================================================
NEXT UPDATE DUE: {(datetime.now().replace(day=1) + pd.DateOffset(months=1)).strftime('%B 1, %Y')}
======================================================================
"""
    
    checklist_file = os.path.join(base_dir, 'automation', 'monthly_checklist.txt')
    
    with open(checklist_file, 'w') as f:
        f.write(checklist)
    
    print(f"   ✅ Checklist saved to: {checklist_file}")
    print("\n   First 3 items from checklist:")
    print(checklist.split('\n')[6:12])
    print("   ... (see full file for complete list)")

# Run all checks
if __name__ == "__main__":
    update_economic_data()
    update_polling_data()
    check_election_results()
    check_model_freshness()
    analyze_system_readiness()
    generate_2026_timeline()
    generate_update_checklist()
    
    print("\n" + "=" * 70)
    print("AUTOMATION SETUP COMPLETE")
    print("=" * 70)
    print("\n✅ System diagnostics run")
    print("✅ Checklists generated")
    print("✅ Timeline created")
    
    print("\n📅 Next Actions:")
    print("   1. Set calendar reminder: 1st of each month")
    print("   2. Bookmark FRED data links")
    print("   3. Monitor for 2026 Senate markets (likely June 2026)")
    print("   4. Run this script monthly to check freshness")
    
    print("\n" + "=" * 70)