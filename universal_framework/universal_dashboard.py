UNIVERSAL PREDICTION MARKET DASHBOARD

Select which market you want to trade, then use the same tools.
"""

import os
import sys

def print_market_menu():
    """Show available markets"""
    print("=" * 70)
    print("🌍 UNIVERSAL PREDICTION MARKET SYSTEM")
    print("=" * 70)
    print("\nAvailable Markets:\n")
    print("  1. 🇺🇸 US Elections")
    print("  2. 🇬🇧 UK Elections")
    print("  3. ⚽ Football/Soccer")
    print("  4. 📊 Economic Events")
    print("  5. 🏏 Cricket (Coming Soon)")
    print("  6. 🏈 American Football (Coming Soon)")
    print("\n  0. Exit")
    print("\n" + "=" * 70)

def launch_us_elections():
    """Launch US elections dashboard"""
    # Your existing dashboard
    import sys
    sys.path.append('..')
    import dashboard
    dashboard.main()

def launch_uk_elections():
    """Launch UK elections module"""
    print("\n🇬🇧 UK ELECTIONS MODULE")
    print("\nFeatures:")
    print("  • Predict constituency results")
    print("  • ONS economic data integration")
    print("  • YouGov polling analysis")
    print("\⚠️  Coming soon! Framework ready, needs data connection.")
    input("\nPress Enter to return...")

def launch_football():
    """Launch football module"""
    print("\n⚽ FOOTBALL PREDICTION MODULE")
    print("\nFeatures:")
    print("  • Match outcome predictions")
    print("  • Team form analysis")
    print("  • Historical head-to-head")
    print("\n⚠️  Coming soon! Framework ready, needs data source.")
    input("\nPress Enter to return...")

def main():
    """Main universal dashboard"""
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print_market_menu()
        
        choice = input("Select market: ").strip()
        
        if choice == '0':
            print("\n👋 Goodbye!\n")
            break
        elif choice == '1':
            launch_us_elections()
        elif choice == '2':
            launch_uk_elections()
        elif choice == '3':
            launch_football()
        elif choice == '4':
            print("\n📊 Economic Events - Coming Soon!")
            input("\nPress Enter to return...")
        else:
            print("\n❌ Invalid option")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()