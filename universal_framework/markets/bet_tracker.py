"""
BET TRACKING MODULE

Track your bets, calculate ROI, and monitor performance
"""

import pandas as pd
import json
import os
from datetime import datetime

class BetTracker:
    """Track betting performance"""
    
    def __init__(self, data_file='data/bet_tracking.json'):
        self.data_file = data_file
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        self.bets = self._load_bets()
    
    def _load_bets(self):
        """Load existing bets from file"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_bets(self):
        """Save bets to file"""
        with open(self.data_file, 'w') as f:
            json.dump(self.bets, f, indent=2)
    
    def add_bet(self, match, bet_type, odds, stake, result=None, profit=None, date=None, bet_direction='BACK'):
        """
        Add a new bet
        
        Args:
            match: "Team A vs Team B"
            bet_type: "Home Win", "Away Win or Draw", etc.
            odds: Decimal odds (e.g., 1.85)
            stake: Amount bet (e.g., 10.00) - For LAY bets, this is backer's stake
            result: "Won", "Lost", "Push", or None if pending
            profit: Actual profit/loss, or None to calculate
            date: Date of match (defaults to today)
            bet_direction: "BACK" or "LAY"
        """
        
        if profit is None and result:
            if bet_direction == 'LAY':
                # LAY BET LOGIC
                # stake parameter = liability (what we're risking)
                if result == "Won":
                    # We won - selection lost
                    # We keep the backer's stake minus 2% commission
                    backers_stake = stake / (odds - 1)
                    profit = backers_stake * 0.98
                elif result == "Lost":
                    # We lost - selection won
                    # We pay the liability
                    profit = -stake
                else:  # Push
                    profit = 0
            else:
                # BACK BET LOGIC (original)
                if result == "Won":
                    profit = stake * (odds - 1)
                elif result == "Lost":
                    profit = -stake
                else:  # Push
                    profit = 0
        
        bet = {
            'id': len(self.bets) + 1,
            'date': date or datetime.now().strftime('%Y-%m-%d'),
            'match': match,
            'bet_type': bet_type,
            'bet_direction': bet_direction,
            'odds': odds,
            'stake': stake,
            'result': result,
            'profit': profit,
            'timestamp': datetime.now().isoformat()
        }
        
        self.bets.append(bet)
        self._save_bets()
        
        return bet
    
    def update_bet_result(self, bet_id, result):
        """
        Update bet result after match finishes
        
        Args:
            bet_id: ID of bet to update
            result: "Won", "Lost", or "Push"
        """
        
        for bet in self.bets:
            if bet['id'] == bet_id:
                bet['result'] = result
                
                # Get bet direction (default to BACK for old bets)
                bet_direction = bet.get('bet_direction', 'BACK')
                
                if bet_direction == 'LAY':
                    # LAY BET LOGIC
                    # stake = liability (what we're risking)
                    # If we WIN the lay bet (selection loses), we keep the backer's stake minus commission
                    # If we LOSE the lay bet (selection wins), we pay the liability
                    
                    if result == "Won":
                        # We won - selection lost
                        # We keep the backer's stake (liability / (odds - 1)) minus 2% commission
                        backers_stake = bet['stake'] / (bet['odds'] - 1)
                        bet['profit'] = backers_stake * 0.98  # Win backer's stake minus commission
                    elif result == "Lost":
                        # We lost - selection won
                        # We pay the liability (which is the stake we stored)
                        bet['profit'] = -bet['stake']  # Lose liability
                    else:  # Push
                        bet['profit'] = 0
                else:
                    # BACK BET LOGIC
                    if result == "Won":
                        bet['profit'] = bet['stake'] * (bet['odds'] - 1)
                    elif result == "Lost":
                        bet['profit'] = -bet['stake']
                    else:  # Push
                        bet['profit'] = 0
                
                self._save_bets()
                return bet
        
        return None
    
    def delete_bet(self, bet_id):
        """Delete a bet"""
        self.bets = [b for b in self.bets if b['id'] != bet_id]
        self._save_bets()
    
    def get_stats(self):
        """Calculate betting statistics"""
        
        if not self.bets:
            return {
                'total_bets': 0,
                'pending_bets': 0,
                'settled_bets': 0,
                'wins': 0,
                'losses': 0,
                'pushes': 0,
                'win_rate': 0,
                'total_staked': 0,
                'total_profit': 0,
                'roi': 0,
                'avg_odds': 0
            }
        
        settled = [b for b in self.bets if b['result'] not in ['Pending', '', None]]
        pending = [b for b in self.bets if b['result'] in ['Pending', '', None]]
        
        wins = [b for b in settled if b['result'] == 'Won']
        losses = [b for b in settled if b['result'] == 'Lost']
        pushes = [b for b in settled if b['result'] == 'Push']
        
        total_staked = sum(b['stake'] for b in self.bets)
        total_profit = sum(b.get('profit', 0) for b in settled)
        
        return {
            'total_bets': len(self.bets),
            'pending_bets': len(pending),
            'settled_bets': len(settled),
            'wins': len(wins),
            'losses': len(losses),
            'pushes': len(pushes),
            'win_rate': (len(wins) / len(settled) * 100) if settled else 0,
            'total_staked': total_staked,
            'total_profit': total_profit,
            'roi': (total_profit / total_staked * 100) if total_staked > 0 else 0,
            'avg_odds': sum(b['odds'] for b in self.bets) / len(self.bets) if self.bets else 0
        }
    
    def get_bets_dataframe(self):
        """Get bets as pandas DataFrame for display"""
        if not self.bets:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.bets)
        
        # Add status column
        df['status'] = df['result'].apply(lambda x: 'Pending' if pd.isna(x) or x is None else x)
        
        return df
    
    def get_performance_by_confidence(self):
        """Analyze performance by confidence level (if tracked)"""
        # Could be enhanced to track confidence levels
        pass

# Testing
if __name__ == "__main__":
    tracker = BetTracker(data_file='test_bets.json')
    
    # Add test bet
    tracker.add_bet(
        match="Man United vs Newcastle",
        bet_type="Away Win or Draw",
        odds=1.78,
        stake=20.00,
        result="Won"
    )
    
    # Get stats
    stats = tracker.get_stats()
    print("Stats:", stats)
    
    # Get DataFrame
    df = tracker.get_bets_dataframe()
    print("\nBets:")
    print(df)