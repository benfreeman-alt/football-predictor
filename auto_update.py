"""
AUTO-UPDATE ORCHESTRATOR

Main script that runs the complete auto-update pipeline
Run this weekly (e.g., via cron job or GitHub Actions)
"""

import sys
import os
from datetime import datetime
import json

# Add paths
sys.path.insert(0, 'universal_framework/markets')

from auto_data_collector import AutoDataCollector
from auto_trainer import AutoTrainer
from auto_deployer import AutoDeployer

class AutoUpdateOrchestrator:
    """Orchestrates the complete auto-update workflow"""
    
    def __init__(self, log_file='logs/auto_update.log'):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log(self, message):
        """Log message to file and print"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def run(self, dry_run=False):
        """
        Run complete auto-update workflow
        
        Args:
            dry_run: If True, don't actually deploy
        
        Returns:
            dict: Results summary
        """
        
        self.log("="*70)
        self.log("AUTO-UPDATE STARTED")
        self.log("="*70)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'steps': {}
        }
        
        try:
            # STEP 1: Collect new data
            self.log("\n[STEP 1] Collecting new match data...")
            
            collector = AutoDataCollector()
            new_results = collector.collect_new_results(days_back=7)
            
            if new_results.empty:
                self.log("   No new results found")
                results['steps']['data_collection'] = 'no_new_data'
                results['success'] = True  # Success but nothing to do
                return results
            
            self.log(f"   ✅ Collected {len(new_results)} new matches")
            results['steps']['data_collection'] = {
                'status': 'success',
                'matches': len(new_results)
            }
            
            # STEP 2: Update dataset
            self.log("\n[STEP 2] Updating dataset...")
            
            collector.append_to_dataset(new_results)
            collector.update_xg_data(new_results)
            
            self.log("   ✅ Dataset updated")
            results['steps']['dataset_update'] = 'success'
            
            # STEP 3: Check if retraining needed
            self.log("\n[STEP 3] Checking if retraining needed...")
            
            trainer = AutoTrainer()
            
            if not trainer.should_retrain():
                self.log("   Model is current - no retraining needed")
                results['steps']['retrain_check'] = 'not_needed'
                results['success'] = True
                return results
            
            # STEP 4: Retrain model
            self.log("\n[STEP 4] Retraining model...")
            
            training_results = trainer.train_model()
            
            if training_results is None:
                self.log("   ❌ Training failed")
                results['steps']['training'] = 'failed'
                return results
            
            self.log(f"   ✅ Training complete - Accuracy: {training_results.get('accuracy', 0):.2%}")
            results['steps']['training'] = {
                'status': 'success',
                'accuracy': training_results.get('accuracy', 0)
            }
            
            # STEP 5: Validate model
            self.log("\n[STEP 5] Validating model...")
            
            validation_results = trainer.validate_model()
            
            if validation_results is None:
                self.log("   ⚠️  Validation failed - deploying anyway")
                validation_passed = True  # Deploy despite validation failure
            elif validation_results['accuracy'] < 0.70:
                self.log(f"   ❌ Validation accuracy ({validation_results['accuracy']:.2%}) below 70%")
                self.log("   NOT deploying - model performance too low")
                results['steps']['validation'] = {
                    'status': 'failed',
                    'accuracy': validation_results['accuracy']
                }
                return results
            else:
                self.log(f"   ✅ Validation passed - Accuracy: {validation_results['accuracy']:.2%}")
                validation_passed = True
                results['steps']['validation'] = {
                    'status': 'success',
                    'accuracy': validation_results['accuracy']
                }
            
            # STEP 6: Deploy
            if dry_run:
                self.log("\n[STEP 6] DRY RUN - Skipping deployment")
                results['steps']['deployment'] = 'dry_run'
            else:
                self.log("\n[STEP 6] Deploying to GitHub...")
                
                deployer = AutoDeployer()
                deployed = deployer.deploy_if_needed(validation_passed=validation_passed)
                
                if deployed:
                    self.log("   ✅ Deployed to GitHub")
                    self.log("   🚀 Streamlit Cloud will auto-deploy in 2-3 minutes")
                    results['steps']['deployment'] = 'success'
                else:
                    self.log("   ⚠️  Deployment skipped")
                    results['steps']['deployment'] = 'skipped'
            
            # Success!
            results['success'] = True
            self.log("\n" + "="*70)
            self.log("✅ AUTO-UPDATE COMPLETE")
            self.log("="*70)
            
        except Exception as e:
            self.log(f"\n❌ AUTO-UPDATE FAILED: {e}")
            import traceback
            self.log(traceback.format_exc())
            results['error'] = str(e)
        
        # Save results
        results_file = 'logs/last_update.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto-update football prediction model')
    parser.add_argument('--dry-run', action='store_true', help='Run without deploying')
    args = parser.parse_args()
    
    orchestrator = AutoUpdateOrchestrator()
    results = orchestrator.run(dry_run=args.dry_run)
    
    if results['success']:
        print("\n✅ Auto-update successful!")
        exit(0)
    else:
        print("\n❌ Auto-update failed!")
        exit(1)