"""
LAY BETTING CALCULATOR

Calculate equivalent lay bets and compare with traditional betting
"""

class LayCalculator:
    """Calculate lay betting stakes and compare with traditional bets"""
    
    def __init__(self, commission_rate=0.02):
        """
        Args:
            commission_rate: Betfair commission (default 2% = 0.02)
        """
        self.commission_rate = commission_rate
    
    def calculate_lay_stake_from_risk(self, target_liability, lay_odds):
        """
        Calculate backer's stake needed to match a target liability
        
        Args:
            target_liability: How much you want to risk (£)
            lay_odds: Lay odds on exchange
        
        Returns:
            Backer's stake needed
        """
        return target_liability / (lay_odds - 1)
    
    def calculate_liability(self, backers_stake, lay_odds):
        """
        Calculate your liability (what you risk)
        
        Args:
            backers_stake: The stake you're laying
            lay_odds: Lay odds on exchange
        
        Returns:
            Your liability
        """
        return backers_stake * (lay_odds - 1)
    
    def calculate_lay_profit(self, backers_stake, lay_odds):
        """
        Calculate profit if lay bet wins (after commission)
        
        Args:
            backers_stake: The stake you're laying
            lay_odds: Lay odds on exchange
        
        Returns:
            Profit after commission
        """
        gross_profit = backers_stake
        commission = gross_profit * self.commission_rate
        return gross_profit - commission
    
    def calculate_traditional_profit(self, stake, odds):
        """
        Calculate profit from traditional bet
        
        Args:
            stake: Your stake
            odds: Decimal odds
        
        Returns:
            Profit if win
        """
        return stake * (odds - 1)
    
    def compare_bets(self, traditional_stake, traditional_odds, lay_odds):
        """
        Compare traditional bet vs lay bet
        
        Args:
            traditional_stake: Stake for traditional bet
            traditional_odds: Traditional odds (e.g., double chance)
            lay_odds: Lay odds on exchange
        
        Returns:
            Dictionary with comparison data
        """
        # Traditional bet
        trad_risk = traditional_stake
        trad_profit = self.calculate_traditional_profit(traditional_stake, traditional_odds)
        trad_roi = (trad_profit / trad_risk) * 100
        
        # Lay bet - match the same risk
        lay_stake = self.calculate_lay_stake_from_risk(trad_risk, lay_odds)
        lay_liability = self.calculate_liability(lay_stake, lay_odds)
        lay_profit = self.calculate_lay_profit(lay_stake, lay_odds)
        lay_roi = (lay_profit / lay_liability) * 100
        
        # Comparison
        profit_difference = lay_profit - trad_profit
        better_option = "LAY" if lay_profit > trad_profit else "TRADITIONAL"
        
        return {
            'traditional': {
                'stake': traditional_stake,
                'odds': traditional_odds,
                'risk': trad_risk,
                'profit': trad_profit,
                'roi': trad_roi
            },
            'lay': {
                'backers_stake': lay_stake,
                'odds': lay_odds,
                'liability': lay_liability,
                'profit': lay_profit,
                'roi': lay_roi,
                'commission': lay_stake * self.commission_rate
            },
            'comparison': {
                'profit_difference': profit_difference,
                'better_option': better_option,
                'percentage_better': (abs(profit_difference) / trad_profit) * 100 if trad_profit > 0 else 0
            }
        }
    
    def get_lay_recommendation(self, traditional_stake, traditional_odds, lay_odds):
        """
        Get a simple recommendation
        
        Returns:
            String recommendation
        """
        comparison = self.compare_bets(traditional_stake, traditional_odds, lay_odds)
        
        if comparison['comparison']['better_option'] == 'LAY':
            diff = comparison['comparison']['profit_difference']
            pct = comparison['comparison']['percentage_better']
            return f"✅ LAY is better! Extra £{diff:.2f} profit ({pct:.1f}% more)"
        else:
            return "⚠️ Traditional bet is better at these odds"
    
    def calculate_equivalent_traditional_odds(self, lay_odds):
        """
        Calculate what traditional odds would be equivalent to lay odds
        (after commission)
        
        Args:
            lay_odds: Lay odds on exchange
        
        Returns:
            Equivalent traditional odds
        """
        # Probability of lay winning
        lay_win_prob = 1 - (1 / lay_odds)
        
        # After commission
        effective_prob = lay_win_prob * (1 - self.commission_rate)
        
        # Convert back to odds
        if effective_prob > 0:
            return 1 / effective_prob
        return 0

# Testing
if __name__ == "__main__":
    calc = LayCalculator(commission_rate=0.02)
    
    # Example: £100 traditional bet @ 1.78 vs laying @ 2.20
    result = calc.compare_bets(
        traditional_stake=100,
        traditional_odds=1.78,
        lay_odds=2.20
    )
    
    print("="*70)
    print("LAY CALCULATOR TEST")
    print("="*70)
    
    print("\nTRADITIONAL BET:")
    print(f"  Stake: £{result['traditional']['stake']:.2f}")
    print(f"  Odds: {result['traditional']['odds']:.2f}")
    print(f"  Risk: £{result['traditional']['risk']:.2f}")
    print(f"  Profit if win: £{result['traditional']['profit']:.2f}")
    print(f"  ROI: {result['traditional']['roi']:.1f}%")
    
    print("\nLAY BET:")
    print(f"  Backer's Stake: £{result['lay']['backers_stake']:.2f}")
    print(f"  Lay Odds: {result['lay']['odds']:.2f}")
    print(f"  Your Liability: £{result['lay']['liability']:.2f}")
    print(f"  Profit if win: £{result['lay']['profit']:.2f}")
    print(f"  Commission: £{result['lay']['commission']:.2f}")
    print(f"  ROI: {result['lay']['roi']:.1f}%")
    
    print("\nCOMPARISON:")
    print(f"  Better option: {result['comparison']['better_option']}")
    print(f"  Profit difference: £{result['comparison']['profit_difference']:.2f}")
    print(f"  Percentage better: {result['comparison']['percentage_better']:.1f}%")
    
    print("\n" + calc.get_lay_recommendation(100, 1.78, 2.20))