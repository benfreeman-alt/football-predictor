import pandas as pd
import os

print("=" * 70)
print("EXPLORING POLLING DATA")
print("=" * 70)

polling_dir = "real_data/polling"

# Find all CSV files
csv_files = [f for f in os.listdir(polling_dir) if f.endswith('.csv')]

print(f"\nFound {len(csv_files)} polling files:")
for f in csv_files:
    print(f"  - {f}")

# Load and explore each file
for csv_file in csv_files:
    filepath = os.path.join(polling_dir, csv_file)
    
    print(f"\n{'=' * 70}")
    print(f"FILE: {csv_file}")
    print('=' * 70)
    
    try:
        df = pd.read_csv(filepath, nrows=1000)  # Load first 1000 rows
        
        print(f"\nRows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(f"\nColumn names:")
        for col in df.columns:
            print(f"  - {col}")
        
        # Check for key columns we need
        key_cols = {
            'year/cycle': ['year', 'cycle', 'election_year'],
            'state': ['state', 'state_po', 'location'],
            'date': ['date', 'end_date', 'poll_date'],
            'dem_pct': ['dem', 'democrat', 'pct_estimate', 'answer'],
            'rep_pct': ['rep', 'republican', 'pct_estimate', 'answer']
        }
        
        print(f"\nKey column detection:")
        for key, possible_names in key_cols.items():
            found = [col for col in df.columns if any(name.lower() in col.lower() for name in possible_names)]
            if found:
                print(f"  ✓ {key}: {found[0]}")
            else:
                print(f"  ✗ {key}: Not found")
        
        # Show sample data
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        # Check date range if date column exists
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            print(f"\nDate range:")
            print(f"  Earliest: {df[date_col].min()}")
            print(f"  Latest: {df[date_col].max()}")
        
        # Check for election years
        year_cols = [col for col in df.columns if col.lower() in ['year', 'cycle', 'election_year']]
        if year_cols:
            year_col = year_cols[0]
            print(f"\nElection years covered:")
            print(f"  {sorted(df[year_col].dropna().unique())}")
        
    except Exception as e:
        print(f"✗ Error loading file: {e}")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print("\nBased on the structure above, we'll:")
print("1. Parse the polling data")
print("2. Aggregate polls by state and election")
print("3. Create polling features (final poll average, momentum, etc.)")
print("4. Merge with election results")
print("5. Retrain model with polling features")