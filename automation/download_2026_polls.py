import requests
import os
from datetime import datetime

print("=" * 70)
print("ATTEMPT TO DOWNLOAD 2026 SENATE POLLING")
print("=" * 70)

# Possible GitHub URLs where 2026 data might appear
POSSIBLE_URLS = [
    'https://raw.githubusercontent.com/fivethirtyeight/data/master/polls/senate_polls_2026.csv',
    'https://raw.githubusercontent.com/fivethirtyeight/data/master/polls/2026_senate_polls.csv',
    'https://raw.githubusercontent.com/fivethirtyeight/data/master/senate-polls/senate_polls_2026.csv',
    'https://projects.fivethirtyeight.com/polls/data/senate_polls_2026.csv',
]

def try_download_polls():
    """Try to download 2026 polling from known locations"""
    
    print("\nAttempting to download from possible locations...\n")
    
    for url in POSSIBLE_URLS:
        print(f"Trying: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                # Check if it's actually CSV data (not a 404 page)
                if 'cycle' in response.text or 'date' in response.text or 'poll' in response.text.lower():
                    # Looks like real data!
                    output_dir = 'real_data/polling'
                    os.makedirs(output_dir, exist_ok=True)
                    
                    output_file = os.path.join(output_dir, 'senate_2026.csv')
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    
                    print(f"   ✅ SUCCESS! Downloaded to: {output_file}")
                    print(f"   File size: {len(response.text)} bytes")
                    
                    # Show sample
                    lines = response.text.split('\n')[:5]
                    print(f"\n   Sample of data:")
                    for line in lines:
                        print(f"   {line}")
                    
                    return True
                else:
                    print(f"   ⚠️  URL exists but doesn't contain polling data")
            else:
                print(f"   ✗ Not found (status: {response.status_code})")
        
        except Exception as e:
            print(f"   ✗ Error: {e}")
        
        print()
    
    return False

def manual_instructions():
    """Show manual download instructions"""
    print("=" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print("\nAutomatic download failed. Here's how to get it manually:")
    print("\n1. Visit: https://github.com/fivethirtyeight/data/tree/master/polls")
    print("\n2. Look for a file containing '2026' and 'senate'")
    print("   Examples:")
    print("   - senate_polls_2026.csv")
    print("   - 2026_senate_polls.csv")
    print("   - senate-polls-2026.csv")
    print("\n3. Click the file, then click 'Raw' or 'Download'")
    print("\n4. Save to: C:\\Users\\User\\Desktop\\prediction-markets\\real_data\\polling\\")
    print("\n5. Rename to: senate_2026.csv")
    print("\n6. Run: python train_with_polling.py")
    print("\n" + "=" * 70)

# Run it
if __name__ == "__main__":
    print(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n")
    
    success = try_download_polls()
    
    if success:
        print("\n✅ Polling data downloaded successfully!")
        print("\nNext steps:")
        print("1. Verify the data looks correct")
        print("2. Run: python train_with_polling.py")
    else:
        print("\n⚠️  Automatic download failed")
        manual_instructions()
    
    print("\n" + "=" * 70)