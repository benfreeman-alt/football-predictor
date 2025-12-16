import pandas as pd

print("=" * 70)
print("DEBUGGING POLLING DATA")
print("=" * 70)

# Load the historical polls file
polls = pd.read_csv('real_data/polling/pres_pollaverages_1968-2016.csv')

print("\n1. Historical polls (1968-2016):")
print(f"   Total records: {len(polls)}")
print(f"\n   Columns: {list(polls.columns)}")

# Check what's in there
print(f"\n   Unique cycles: {sorted(polls['cycle'].unique())}")
print(f"\n   Unique states (first 10): {sorted(polls['state'].unique())[:10]}")

# Look at 2016 data specifically
polls_2016 = polls[polls['cycle'] == 2016]
print(f"\n2. 2016 data:")
print(f"   Records: {len(polls_2016)}")
print(f"   States: {polls_2016['state'].nunique()}")

# Sample of 2016 data
print(f"\n   Sample records:")
print(polls_2016[['cycle', 'state', 'modeldate', 'candidate_name', 'pct_estimate']].head(20))

# Check candidate names
print(f"\n3. Candidate names in 2016:")
print(polls_2016['candidate_name'].value_counts())

# Check dates
polls_2016_copy = polls_2016.copy()
polls_2016_copy['modeldate'] = pd.to_datetime(polls_2016_copy['modeldate'])
print(f"\n4. Date range for 2016:")
print(f"   Earliest: {polls_2016_copy['modeldate'].min()}")
print(f"   Latest: {polls_2016_copy['modeldate'].max()}")

# Now check 2020 file
print("\n" + "=" * 70)
polls_2020 = pd.read_csv('real_data/polling/presidential_general_averages_2024-09-12_uncorrected.csv')

print("\n5. 2020 polls file:")
print(f"   Total records: {len(polls_2020)}")
print(f"   Columns: {list(polls_2020.columns)}")

# Check what years
print(f"\n   Unique cycles: {sorted(polls_2020['cycle'].unique())}")

# Look at 2020 specifically
polls_2020_data = polls_2020[polls_2020['cycle'] == 2020]
print(f"\n6. 2020 data:")
print(f"   Records: {len(polls_2020_data)}")
print(f"   States: {polls_2020_data['state'].nunique()}")

print(f"\n   Sample records:")
print(polls_2020_data[['cycle', 'state', 'date', 'candidate', 'pct_trend_adjusted']].head(20))

# Candidate names
print(f"\n7. Candidate names in 2020:")
print(polls_2020_data['candidate'].value_counts())

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)
print("\nLooks like the data IS there, but our extraction logic isn't working.")
print("Need to fix how we're matching states and filtering polls.")