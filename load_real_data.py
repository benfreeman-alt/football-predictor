
import pandas as pd
import os

DATA_DIR = "real_data"

def load_election_results():
    print("\nLoading election results...")
    elections_dir = os.path.join(DATA_DIR, "elections")
    
    if not os.path.exists(elections_dir):
        print(f"❌ Folder not found: {elections_dir}")
        return None
    
    csv_files = [f for f in os.listdir(elections_dir) if f.endswith('.csv')]
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f}")
    
    if not csv_files:
        print("❌ No CSV files found")
        return None
    
    target_file = None
    for f in csv_files:
        if '2000-2024' in f or '2000-2020' in f:
            target_file = f
            break
    
    if not target_file:
        target_file = csv_files[0]
    
    mit_file = os.path.join(elections_dir, target_file)
    print(f"\nTrying to load: {target_file}")
    
    try:
        df = pd.read_csv(mit_file, encoding='latin-1', low_memory=False)
        print(f"✅ Loaded! Rows: {len(df)}")
        print(f"   Columns: {df.columns.tolist()}")
        
        df = df[df['year'] >= 2000]
        
        party_col = None
        for col in ['party_simplified', 'party_detailed', 'party']:
            if col in df.columns:
                party_col = col
                break
        
        if party_col is None:
            print("❌ Can't find party column")
            print(f"   Available columns: {df.columns.tolist()}")
            return None
        
        print(f"   Using party column: {party_col}")
        
        dem_names = ['DEMOCRAT', 'DEMOCRATIC', 'DEM']
        rep_names = ['REPUBLICAN', 'REP']
        
        df_dem = df[df[party_col].str.upper().isin(dem_names)].copy()
        df_rep = df[df[party_col].str.upper().isin(rep_names)].copy()
        
        df_dem['party_clean'] = 'DEMOCRAT'
        df_rep['party_clean'] = 'REPUBLICAN'
        
        df_filtered = pd.concat([df_dem, df_rep])
        
        votes_col = None
        for col in ['candidatevotes', 'votes', 'vote_total']:
            if col in df.columns:
                votes_col = col
                break
        
        if votes_col is None:
            print("❌ Can't find votes column")
            return None
        
        print(f"   Using votes column: {votes_col}")
        
        state_col = 'state_po' if 'state_po' in df.columns else 'state'
        
        state_results = df_filtered.groupby(['year', state_col, 'party_clean'])[votes_col].sum().reset_index()
        
        state_pivot = state_results.pivot_table(
            index=['year', state_col],
            columns='party_clean',
            values=votes_col,
            fill_value=0
        ).reset_index()
        
        state_pivot['total_votes'] = state_pivot['DEMOCRAT'] + state_pivot['REPUBLICAN']
        state_pivot['dem_pct'] = state_pivot['DEMOCRAT'] / state_pivot['total_votes'] * 100
        state_pivot['rep_pct'] = state_pivot['REPUBLICAN'] / state_pivot['total_votes'] * 100
        state_pivot['margin'] = state_pivot['rep_pct'] - state_pivot['dem_pct']
        
        state_pivot = state_pivot.rename(columns={'year': 'election_year', state_col: 'state'})
        
        print(f"✅ Processed {len(state_pivot)} state-level records")
        print(f"   Years: {sorted(state_pivot['election_year'].unique())}")
        return state_pivot
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_economic_data():
    print("\nLoading economic data...")
    economic_dir = os.path.join(DATA_DIR, "economic")
    
    if not os.path.exists(economic_dir):
        print(f"❌ Folder not found: {economic_dir}")
        return {}
    
    csv_files = [f for f in os.listdir(economic_dir) if f.endswith('.csv')]
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f}")
    
    economic_data = {}
    
    for csv_file in csv_files:
        filepath = os.path.join(economic_dir, csv_file)
        name = csv_file.replace('.csv', '')
        
        try:
            df = pd.read_csv(filepath)
            
            first_col = df.columns[0]
            second_col = df.columns[1]
            
            df = df.rename(columns={first_col: 'DATE', second_col: 'VALUE'})
            df['DATE'] = pd.to_datetime(df['DATE'])
            
            economic_data[name] = df
            print(f"✅ Loaded {name}: {len(df)} rows")
            
        except Exception as e:
            print(f"❌ Error loading {csv_file}: {e}")
    
    return economic_data

def main():
    print("=" * 70)
    print("LOADING REAL HISTORICAL DATA")
    print("=" * 70)
    
    elections = load_election_results()
    economic = load_economic_data()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if elections is not None:
        print(f"\n✅ ELECTIONS: {len(elections)} records")
        print(f"   Years: {sorted(elections['election_year'].unique())}")
        print(f"   States: {elections['state'].nunique()}")
        
        os.makedirs('data', exist_ok=True)
        elections.to_csv('data/real_election_results.csv', index=False)
        print("   Saved to: data/real_election_results.csv")
        
        print("\n   Sample data:")
        print(elections[['election_year', 'state', 'dem_pct', 'rep_pct', 'margin']].head(10))
    else:
        print("\n❌ No election data loaded")
    
    if economic:
        print(f"\n✅ ECONOMIC: {len(economic)} indicators")
        for name in economic.keys():
            print(f"   - {name}")
    else:
        print("\n❌ No economic data loaded")
    
    print("\n" + "=" * 70)
    if elections is not None or economic:
        print("✅ READY TO RETRAIN MODELS!")
    else:
        print("❌ NO DATA LOADED - CHECK FILE LOCATIONS")
    print("=" * 70)

if __name__ == "__main__":
    main()