"""
AUTOMATED MODEL TRAINING

Retrains model with new data and validates performance
"""

import os
import sys
import pandas as pd
from datetime import datetime
import joblib
import json

class AutoTrainer:
    """Automatically retrain model with new data"""
    
    def __init__(self, model_dir='models', data_dir='data/football'):
        self.model_dir = model_dir
        self.data_dir = data_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def should_retrain(self):
        """
        Determine if model should be retrained
        
        Returns:
            bool: True if retraining needed
        """
        
        # Check when model was last trained
        model_file = os.path.join(self.model_dir, 'last_trained.json')
        
        if not os.path.exists(model_file):
            print("   ⚠️  No training history found - should retrain")
            return True
        
        with open(model_file, 'r') as f:
            training_info = json.load(f)
        
        last_trained = datetime.fromisoformat(training_info['timestamp'])
        days_since = (datetime.now() - last_trained).days
        
        print(f"   Model last trained {days_since} days ago")
        
        # Retrain if:
        # - More than 7 days since last training
        # - Accuracy dropped below threshold
        if days_since >= 7:
            print("   ✅ 7+ days since last training - should retrain")
            return True
        
        if 'accuracy' in training_info and training_info['accuracy'] < 0.72:
            print(f"   ⚠️  Accuracy ({training_info['accuracy']:.1%}) below 72% - should retrain")
            return True
        
        print("   Model is current - no retraining needed")
        return False
    
    def train_model(self):
        """
        Retrain model with all available data
        
        Returns:
            dict: Training results
        """
        
        print("\n🤖 Retraining model...")
        
        try:
            # Import model
            sys.path.insert(0, 'universal_framework/markets')
            from football_optimized_v4_historical import FootballPredictorV4Historical
            
            # Initialize predictor
            predictor = FootballPredictorV4Historical(data_dir=self.data_dir)
            
            # Train model
            print("   Loading data...")
            predictor.load_data()
            
            print("   Training model...")
            results = predictor.train()
            
            # Save model
            model_file = os.path.join(self.model_dir, 'football_model_v4.pkl')
            predictor.save_model(model_file)
            print(f"   ✅ Model saved to {model_file}")
            
            # Save training info
            training_info = {
                'timestamp': datetime.now().isoformat(),
                'accuracy': results.get('accuracy', 0),
                'samples': results.get('samples', 0),
                'version': 'v4'
            }
            
            info_file = os.path.join(self.model_dir, 'last_trained.json')
            with open(info_file, 'w') as f:
                json.dump(training_info, f, indent=2)
            
            print(f"   ✅ Training complete!")
            print(f"   Accuracy: {results.get('accuracy', 0):.2%}")
            print(f"   Samples: {results.get('samples', 0)}")
            
            return results
        
        except Exception as e:
            print(f"   ❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def validate_model(self):
        """
        Validate model on recent matches
        
        Returns:
            dict: Validation results
        """
        
        print("\n✅ Validating model...")
        
        try:
            # Load recent matches (last 20)
            current_season = datetime.now().year if datetime.now().month >= 8 else datetime.now().year - 1
            season_file = os.path.join(self.data_dir, f'E0_{current_season}.csv')
            
            if not os.path.exists(season_file):
                print("   ⚠️  No current season data for validation")
                return None
            
            data = pd.read_csv(season_file)
            recent = data.tail(20)
            
            # Load model
            sys.path.insert(0, 'universal_framework/markets')
            from football_predictor_final import FinalFootballPredictor
            
            predictor = FinalFootballPredictor()
            
            # Test predictions
            correct = 0
            total = 0
            
            for _, match in recent.iterrows():
                try:
                    pred = predictor.predict_match(
                        home_team=match['HomeTeam'],
                        away_team=match['AwayTeam'],
                        match_date=match['Date']
                    )
                    
                    # Check if prediction was correct
                    if pred['prediction'] == 'Home Win' and match['FTR'] == 'H':
                        correct += 1
                    elif pred['prediction'] in ['Away Win', 'Away Win or Draw'] and match['FTR'] != 'H':
                        correct += 1
                    
                    total += 1
                
                except Exception as e:
                    continue
            
            accuracy = correct / total if total > 0 else 0
            
            print(f"   ✅ Validation complete")
            print(f"   Accuracy on last 20 matches: {accuracy:.2%}")
            print(f"   Correct: {correct}/{total}")
            
            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }
        
        except Exception as e:
            print(f"   ❌ Validation failed: {e}")
            return None
    
    def run_auto_update(self):
        """
        Complete auto-update workflow
        
        Returns:
            bool: True if update successful
        """
        
        print("\n" + "="*70)
        print("AUTO-UPDATE WORKFLOW")
        print("="*70)
        
        # Step 1: Collect new data
        from auto_data_collector import AutoDataCollector
        
        collector = AutoDataCollector(data_dir=self.data_dir)
        new_results = collector.collect_new_results(days_back=7)
        
        if new_results.empty:
            print("\n⚠️  No new results - skipping update")
            return False
        
        # Step 2: Update dataset
        collector.append_to_dataset(new_results)
        collector.update_xg_data(new_results)
        
        # Step 3: Check if retraining needed
        if not self.should_retrain():
            print("\n✅ Model is current - no update needed")
            return False
        
        # Step 4: Retrain
        results = self.train_model()
        
        if results is None:
            print("\n❌ Retraining failed")
            return False
        
        # Step 5: Validate
        validation = self.validate_model()
        
        if validation and validation['accuracy'] < 0.70:
            print(f"\n⚠️  Validation accuracy ({validation['accuracy']:.2%}) below threshold")
            print("   Model NOT deployed")
            return False
        
        # Step 6: Success!
        print("\n" + "="*70)
        print("✅ AUTO-UPDATE COMPLETE")
        print("="*70)
        print(f"New data: {len(new_results)} matches")
        print(f"Training accuracy: {results.get('accuracy', 0):.2%}")
        print(f"Validation accuracy: {validation['accuracy']:.2%}")
        print("\n🚀 Model ready for deployment!")
        
        return True

# Testing
if __name__ == "__main__":
    trainer = AutoTrainer()
    success = trainer.run_auto_update()
    
    if success:
        print("\n✅ Auto-update successful!")
    else:
        print("\n❌ Auto-update failed or not needed")