"""
Data Preprocessing and Feature Engineering
===========================================

This module handles:
1. Data cleaning and validation
2. Feature engineering for prediction models
3. Merging multiple data sources
4. Creating training/test datasets
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict
import os


class DataPreprocessor:
    """Clean and preprocess raw data"""
    
    def __init__(self, data_dir: str = "/home/claude/data"):
        self.data_dir = data_dir
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all collected datasets"""
        data = {}
        
        files = {
            'elections': 'election_results.csv',
            'polling': 'polling_data.csv',
            'economic': 'economic_indicators.csv'
        }
        
        for key, filename in files.items():
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                data[key] = pd.read_csv(filepath)
                print(f"Loaded {key}: {len(data[key])} records")
            else:
                print(f"Warning: {filename} not found")
                
        return data
    
    def clean_election_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean election results data"""
        df = df.copy()
        
        # Convert types
        df['election_year'] = df['election_year'].astype(int)
        df['votes'] = df['votes'].astype(int)
        df['vote_share'] = df['vote_share'].astype(float)
        
        # Calculate margin
        df_pivot = df.pivot_table(
            index=['election_year', 'state'],
            columns='party',
            values='vote_share'
        ).reset_index()
        
        if 'Republican' in df_pivot.columns and 'Democratic' in df_pivot.columns:
            df_pivot['margin'] = df_pivot['Republican'] - df_pivot['Democratic']
        
        return df_pivot
    
    def clean_polling_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean polling data"""
        df = df.copy()
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate polling margin and spread
        df['poll_margin'] = df['candidate_rep'] - df['candidate_dem']
        df['poll_spread'] = abs(df['poll_margin'])
        
        # Calculate days until election (assuming Nov 5)
        df['days_until_election'] = (
            pd.to_datetime(df['election_year'].astype(str) + '-11-05') - df['date']
        ).dt.days
        
        return df
    
    def clean_economic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean economic indicators"""
        df = df.copy()
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'])
        
        # Add year and quarter for merging
        df['year'] = df['date'].dt.year
        df['quarter'] = df['date'].dt.quarter
        
        # Calculate changes/trends
        for col in ['gdp_growth', 'unemployment_rate', 'inflation_rate']:
            df[f'{col}_change'] = df[col].diff()
            df[f'{col}_trend'] = df[col].rolling(window=4, min_periods=1).mean()
        
        return df


class FeatureEngineer:
    """Create features for machine learning models"""
    
    def create_polling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from polling data
        - Polling averages at different time windows
        - Polling momentum/trends
        - Volatility measures
        """
        df = df.copy()
        
        # Group by election and state
        features_list = []
        
        for (year, state), group in df.groupby(['election_year', 'state']):
            group = group.sort_values('date')
            
            features = {
                'election_year': year,
                'state': state,
                # Final polls (last 2 weeks)
                'final_poll_dem': group[group['days_until_election'] <= 14]['candidate_dem'].mean(),
                'final_poll_rep': group[group['days_until_election'] <= 14]['candidate_rep'].mean(),
                'final_poll_margin': group[group['days_until_election'] <= 14]['poll_margin'].mean(),
                
                # Early polls (30+ days out)
                'early_poll_dem': group[group['days_until_election'] >= 30]['candidate_dem'].mean(),
                'early_poll_rep': group[group['days_until_election'] >= 30]['candidate_rep'].mean(),
                
                # Polling momentum (change from early to final)
                'poll_momentum_dem': 0,  # Will calculate below
                'poll_momentum_rep': 0,
                
                # Polling volatility
                'poll_volatility': group['poll_margin'].std(),
                
                # Number of polls
                'num_polls': len(group)
            }
            
            # Calculate momentum
            if not pd.isna(features['final_poll_dem']) and not pd.isna(features['early_poll_dem']):
                features['poll_momentum_dem'] = features['final_poll_dem'] - features['early_poll_dem']
                features['poll_momentum_rep'] = features['final_poll_rep'] - features['early_poll_rep']
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def create_economic_features(self, df: pd.DataFrame, election_years: list) -> pd.DataFrame:
        """
        Create economic features for election years
        Focus on Q3 data (most recent before Nov election)
        """
        features_list = []
        
        for year in election_years:
            # Get Q3 data for the election year
            q3_data = df[(df['year'] == year) & (df['quarter'] == 3)]
            
            if len(q3_data) > 0:
                row = q3_data.iloc[0]
                features = {
                    'election_year': year,
                    'gdp_growth_q3': row['gdp_growth'],
                    'unemployment_q3': row['unemployment_rate'],
                    'inflation_q3': row['inflation_rate'],
                    'consumer_confidence_q3': row['consumer_confidence'],
                    'gdp_trend': row.get('gdp_growth_trend', row['gdp_growth']),
                    'unemployment_trend': row.get('unemployment_rate_trend', row['unemployment_rate']),
                    'inflation_trend': row.get('inflation_rate_trend', row['inflation_rate'])
                }
                features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def create_historical_features(self, election_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on historical patterns
        - Previous election results
        - Historical party performance
        """
        df = election_df.copy()
        
        # Sort by state and year
        df = df.sort_values(['state', 'election_year'])
        
        # Previous election margin (lagged feature)
        df['prev_election_margin'] = df.groupby('state')['margin'].shift(1)
        
        # Average historical margin
        df['historical_avg_margin'] = df.groupby('state')['margin'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        return df
    
    def merge_all_features(self, 
                          election_df: pd.DataFrame,
                          polling_features: pd.DataFrame,
                          economic_features: pd.DataFrame) -> pd.DataFrame:
        """
        Merge all features into a single dataframe
        """
        # Start with election results
        merged = election_df.copy()
        
        # Merge polling features
        merged = merged.merge(
            polling_features,
            on=['election_year', 'state'],
            how='left'
        )
        
        # Merge economic features
        merged = merged.merge(
            economic_features,
            on='election_year',
            how='left'
        )
        
        return merged


def create_modeling_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to create clean modeling dataset
    
    Returns:
        train_df: Historical data for training
        test_df: Most recent data for testing (2024 election)
    """
    print("=" * 70)
    print("DATA PREPROCESSING AND FEATURE ENGINEERING")
    print("=" * 70)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    print("\n1. Loading raw data...")
    print("-" * 70)
    data = preprocessor.load_all_data()
    
    # Clean data
    print("\n2. Cleaning data...")
    print("-" * 70)
    
    if 'elections' in data:
        elections_clean = preprocessor.clean_election_data(data['elections'])
        print(f"Cleaned elections: {len(elections_clean)} records")
    
    if 'polling' in data:
        polling_clean = preprocessor.clean_polling_data(data['polling'])
        print(f"Cleaned polling: {len(polling_clean)} records")
    
    if 'economic' in data:
        economic_clean = preprocessor.clean_economic_data(data['economic'])
        print(f"Cleaned economic: {len(economic_clean)} records")
    
    # Feature engineering
    print("\n3. Engineering features...")
    print("-" * 70)
    
    engineer = FeatureEngineer()
    
    # Create polling features
    polling_features = engineer.create_polling_features(polling_clean)
    print(f"Created polling features: {len(polling_features.columns)} columns")
    print("Polling features:", list(polling_features.columns))
    
    # Create economic features
    election_years = elections_clean['election_year'].unique()
    economic_features = engineer.create_economic_features(economic_clean, election_years)
    print(f"\nCreated economic features: {len(economic_features.columns)} columns")
    print("Economic features:", list(economic_features.columns))
    
    # Create historical features
    elections_with_history = engineer.create_historical_features(elections_clean)
    print(f"\nAdded historical features")
    
    # Merge all features
    print("\n4. Merging all features...")
    print("-" * 70)
    
    final_df = engineer.merge_all_features(
        elections_with_history,
        polling_features,
        economic_features
    )
    
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Features: {list(final_df.columns)}")
    
    # Split into train/test
    print("\n5. Splitting train/test...")
    print("-" * 70)
    
    # Use 2024 as test set, earlier years for training
    train_df = final_df[final_df['election_year'] < 2024].copy()
    test_df = final_df[final_df['election_year'] == 2024].copy()
    
    print(f"Training set: {len(train_df)} records (years: {sorted(train_df['election_year'].unique())})")
    print(f"Test set: {len(test_df)} records (year: 2024)")
    
    # Save processed data
    print("\n6. Saving processed datasets...")
    print("-" * 70)
    
    train_df.to_csv("/home/claude/data/train_data.csv", index=False)
    test_df.to_csv("/home/claude/data/test_data.csv", index=False)
    final_df.to_csv("/home/claude/data/final_features.csv", index=False)
    
    print("Saved:")
    print("  - train_data.csv")
    print("  - test_data.csv")
    print("  - final_features.csv")
    
    # Display sample
    print("\n7. Sample of final features:")
    print("-" * 70)
    print(final_df.head())
    
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    
    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = create_modeling_dataset()
    
    print("\nReady for modeling!")
    print("Next step: Build predictive models using train_data.csv")
