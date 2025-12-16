"""
Backtesting and Strategy Optimization
======================================

This module handles:
1. Historical backtesting of prediction strategies
2. Kelly Criterion for optimal bet sizing
3. Performance metrics (ROI, Sharpe ratio, max drawdown)
4. Strategy optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os


class BettingStrategy:
    """Implement various betting strategies"""
    
    @staticmethod
    def kelly_criterion(probability: float, odds: float, kelly_fraction: float = 0.25) -> float:
        """
        Calculate optimal bet size using Kelly Criterion
        
        Args:
            probability: Our estimated probability of winning (0-1)
            odds: Market odds (e.g., 1.5 means you get $1.50 per $1 bet)
            kelly_fraction: Fraction of Kelly to use (0.25 = Quarter Kelly, conservative)
        
        Returns:
            Fraction of bankroll to bet
        """
        # Kelly formula: f = (bp - q) / b
        # where: b = odds - 1 (net odds), p = probability, q = 1 - p
        
        b = odds - 1  # Net odds
        p = probability
        q = 1 - p
        
        kelly_bet = (b * p - q) / b
        
        # Apply fractional Kelly (for risk management)
        kelly_bet = kelly_bet * kelly_fraction
        
        # Never bet negative amounts or more than 100%
        kelly_bet = max(0, min(kelly_bet, 1.0))
        
        return kelly_bet
    
    @staticmethod
    def fixed_stake(stake_percent: float = 0.02) -> float:
        """Fixed percentage stake strategy"""
        return stake_percent
    
    @staticmethod
    def calculate_edge(model_prob: float, market_prob: float) -> float:
        """
        Calculate betting edge
        
        Args:
            model_prob: Our model's probability estimate
            market_prob: Market's implied probability
        
        Returns:
            Edge (positive means we have advantage)
        """
        return model_prob - market_prob


class Backtester:
    """Backtest prediction market strategies"""
    
    def __init__(self, initial_bankroll: float = 10000):
        self.initial_bankroll = initial_bankroll
        self.trades = []
        
    def simulate_trade(self, 
                      prediction_prob: float,
                      market_price: float,
                      outcome: bool,
                      strategy: str = 'kelly',
                      kelly_fraction: float = 0.25,
                      min_edge: float = 0.05) -> Dict:
        """
        Simulate a single trade
        
        Args:
            prediction_prob: Model's predicted probability
            market_price: Market price (0-1, represents implied probability)
            outcome: True if our predicted side won
            strategy: 'kelly' or 'fixed'
            kelly_fraction: Fraction of Kelly to use
            min_edge: Minimum edge required to place bet
        
        Returns:
            Trade result dictionary
        """
        # Calculate edge
        edge = BettingStrategy.calculate_edge(prediction_prob, market_price)
        
        # Only bet if we have sufficient edge
        if edge < min_edge:
            return {
                'bet_placed': False,
                'edge': edge,
                'reason': 'Insufficient edge'
            }
        
        # Calculate odds from market price
        # If market price is 0.6, odds are 1/0.6 = 1.67
        odds = 1 / market_price
        
        # Determine bet size
        if strategy == 'kelly':
            bet_fraction = BettingStrategy.kelly_criterion(
                prediction_prob, 
                odds, 
                kelly_fraction
            )
        else:  # fixed
            bet_fraction = BettingStrategy.fixed_stake()
        
        # Calculate profit/loss
        if outcome:
            # Win: profit is (odds - 1) * stake
            pnl_fraction = bet_fraction * (odds - 1)
        else:
            # Loss: lose the stake
            pnl_fraction = -bet_fraction
        
        return {
            'bet_placed': True,
            'bet_fraction': bet_fraction,
            'edge': edge,
            'odds': odds,
            'prediction_prob': prediction_prob,
            'market_price': market_price,
            'outcome': outcome,
            'pnl_fraction': pnl_fraction
        }
    
    def run_backtest(self, 
                    predictions_df: pd.DataFrame,
                    strategy: str = 'kelly',
                    kelly_fraction: float = 0.25,
                    min_edge: float = 0.05) -> pd.DataFrame:
        """
        Run full backtest on historical predictions
        
        predictions_df should have columns:
        - date
        - model_prob (our prediction)
        - market_prob (market implied probability)
        - outcome (1 if our side won, 0 otherwise)
        """
        results = []
        current_bankroll = self.initial_bankroll
        
        for idx, row in predictions_df.iterrows():
            trade_result = self.simulate_trade(
                prediction_prob=row['model_prob'],
                market_price=row['market_prob'],
                outcome=row['outcome'],
                strategy=strategy,
                kelly_fraction=kelly_fraction,
                min_edge=min_edge
            )
            
            if trade_result['bet_placed']:
                # Calculate dollar amounts
                bet_amount = current_bankroll * trade_result['bet_fraction']
                pnl_amount = current_bankroll * trade_result['pnl_fraction']
                current_bankroll += pnl_amount
                
                results.append({
                    'date': row.get('date', idx),
                    'market': row.get('market', f'Market_{idx}'),
                    'bet_amount': bet_amount,
                    'pnl': pnl_amount,
                    'bankroll': current_bankroll,
                    'edge': trade_result['edge'],
                    'odds': trade_result['odds'],
                    'outcome': trade_result['outcome']
                })
        
        return pd.DataFrame(results)
    
    def calculate_performance_metrics(self, backtest_results: pd.DataFrame) -> Dict:
        """Calculate strategy performance metrics"""
        if len(backtest_results) == 0:
            return {
                'total_trades': 0,
                'roi': 0,
                'win_rate': 0,
                'avg_edge': 0
            }
        
        initial = self.initial_bankroll
        final = backtest_results['bankroll'].iloc[-1] if len(backtest_results) > 0 else initial
        
        roi = (final - initial) / initial
        win_rate = backtest_results['outcome'].mean()
        total_pnl = backtest_results['pnl'].sum()
        avg_edge = backtest_results['edge'].mean()
        
        # Calculate Sharpe Ratio (assuming daily returns)
        returns = backtest_results['pnl'] / backtest_results['bankroll'].shift(1).fillna(initial)
        sharpe = returns.mean() / returns.std() if len(returns) > 1 else 0
        
        # Max drawdown
        cumulative = backtest_results['bankroll']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        metrics = {
            'total_trades': len(backtest_results),
            'winning_trades': backtest_results['outcome'].sum(),
            'losing_trades': len(backtest_results) - backtest_results['outcome'].sum(),
            'win_rate': win_rate,
            'roi': roi,
            'roi_percent': roi * 100,
            'final_bankroll': final,
            'total_pnl': total_pnl,
            'avg_edge': avg_edge,
            'sharpe_ratio': sharpe * np.sqrt(252),  # Annualized
            'max_drawdown': max_drawdown,
            'max_drawdown_percent': max_drawdown * 100
        }
        
        return metrics


def create_sample_backtest_data() -> pd.DataFrame:
    """
    Create sample historical prediction data for backtesting
    In production, this would use actual historical model predictions and market prices
    """
    np.random.seed(42)
    
    # Simulate 50 historical prediction opportunities
    n_trades = 50
    
    data = {
        'date': pd.date_range(start='2020-01-01', periods=n_trades, freq='W'),
        'market': [f'Election_{i}' for i in range(n_trades)],
        # Model predictions (slightly better than market)
        'model_prob': np.random.beta(5, 5, n_trades),
        # Market implied probabilities
        'market_prob': np.random.beta(5, 5, n_trades),
    }
    
    df = pd.DataFrame(data)
    
    # Add some edge to our model (make it slightly better than random)
    df['model_prob'] = df['model_prob'] * 0.8 + np.random.random(n_trades) * 0.2
    
    # Generate outcomes based on true probabilities (with some noise)
    true_prob = (df['model_prob'] * 0.7 + df['market_prob'] * 0.3)
    df['outcome'] = (np.random.random(n_trades) < true_prob).astype(int)
    
    return df


def optimize_strategy(backtest_data: pd.DataFrame) -> Dict:
    """
    Optimize strategy parameters
    Test different Kelly fractions and minimum edge thresholds
    """
    print("\n" + "=" * 70)
    print("STRATEGY OPTIMIZATION")
    print("=" * 70)
    
    # Parameter grid
    kelly_fractions = [0.1, 0.25, 0.5, 1.0]  # 10%, 25%, 50%, 100% Kelly
    min_edges = [0.0, 0.02, 0.05, 0.10]  # 0%, 2%, 5%, 10% minimum edge
    
    results = []
    
    for kelly_frac in kelly_fractions:
        for min_edge in min_edges:
            backtester = Backtester(initial_bankroll=10000)
            backtest_results = backtester.run_backtest(
                backtest_data,
                strategy='kelly',
                kelly_fraction=kelly_frac,
                min_edge=min_edge
            )
            
            metrics = backtester.calculate_performance_metrics(backtest_results)
            
            results.append({
                'kelly_fraction': kelly_frac,
                'min_edge': min_edge,
                'total_trades': metrics['total_trades'],
                'roi': metrics['roi_percent'],
                'win_rate': metrics['win_rate'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown_percent']
            })
    
    results_df = pd.DataFrame(results)
    
    # Find best strategy by Sharpe ratio
    best_idx = results_df['sharpe_ratio'].idxmax()
    best_strategy = results_df.loc[best_idx]
    
    print("\nOptimization Results:")
    print("-" * 70)
    print(results_df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("BEST STRATEGY")
    print("=" * 70)
    print(f"Kelly Fraction: {best_strategy['kelly_fraction']}")
    print(f"Minimum Edge: {best_strategy['min_edge']:.1%}")
    print(f"Expected ROI: {best_strategy['roi']:.2f}%")
    print(f"Win Rate: {best_strategy['win_rate']:.1%}")
    print(f"Sharpe Ratio: {best_strategy['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {best_strategy['max_drawdown']:.2f}%")
    
    return best_strategy.to_dict()


def main():
    """Main backtesting workflow"""
    print("=" * 70)
    print("BACKTESTING AND STRATEGY OPTIMIZATION")
    print("=" * 70)
    
    # Create sample backtest data
    print("\n1. Loading historical prediction data...")
    print("-" * 70)
    backtest_data = create_sample_backtest_data()
    print(f"Loaded {len(backtest_data)} historical predictions")
    print("\nSample data:")
    print(backtest_data.head())
    
    # Run backtest with default parameters
    print("\n2. Running backtest with Quarter Kelly strategy...")
    print("-" * 70)
    
    backtester = Backtester(initial_bankroll=10000)
    results = backtester.run_backtest(
        backtest_data,
        strategy='kelly',
        kelly_fraction=0.25,
        min_edge=0.05
    )
    
    print(f"Executed {len(results)} trades")
    
    if len(results) > 0:
        print("\nFirst 5 trades:")
        print(results.head())
        
        # Calculate performance
        print("\n3. Performance Metrics...")
        print("-" * 70)
        
        metrics = backtester.calculate_performance_metrics(results)
        
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Winning Trades: {metrics['winning_trades']}")
        print(f"Losing Trades: {metrics['losing_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.1%}")
        print(f"\nInitial Bankroll: ${backtester.initial_bankroll:,.2f}")
        print(f"Final Bankroll: ${metrics['final_bankroll']:,.2f}")
        print(f"Total P&L: ${metrics['total_pnl']:,.2f}")
        print(f"ROI: {metrics['roi_percent']:.2f}%")
        print(f"\nAverage Edge: {metrics['avg_edge']:.1%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown_percent']:.2f}%")
        
        # Save results
        results.to_csv("/home/claude/data/backtest_results.csv", index=False)
        print(f"\nResults saved to: /home/claude/data/backtest_results.csv")
    
    # Optimize strategy
    print("\n4. Optimizing strategy parameters...")
    print("-" * 70)
    best_params = optimize_strategy(backtest_data)
    
    # Save optimization results
    with open("/home/claude/data/optimal_strategy.txt", 'w') as f:
        f.write("OPTIMAL BETTING STRATEGY\n")
        f.write("=" * 50 + "\n\n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
    
    print("\n" + "=" * 70)
    print("BACKTESTING COMPLETE")
    print("=" * 70)
    print("\nNext step: Build automation system for live trading")


if __name__ == "__main__":
    main()
