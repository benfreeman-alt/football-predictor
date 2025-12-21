"""
AUTO-DEPLOYMENT TO GITHUB

Automatically commits and pushes updated model to GitHub
which triggers Streamlit Cloud redeployment
"""

import os
import subprocess
from datetime import datetime

class AutoDeployer:
    """Automatically deploy updated model to GitHub"""
    
    def __init__(self, repo_path='.'):
        self.repo_path = repo_path
    
    def check_git_status(self):
        """Check if there are changes to commit"""
        
        try:
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.stdout.strip():
                print(f"   Changes detected:")
                for line in result.stdout.strip().split('\n')[:5]:
                    print(f"     {line}")
                return True
            else:
                print("   No changes to deploy")
                return False
        
        except Exception as e:
            print(f"   ❌ Git status check failed: {e}")
            return False
    
    def commit_and_push(self, message=None):
        """
        Commit changes and push to GitHub
        
        Args:
            message: Commit message (auto-generated if None)
        
        Returns:
            bool: True if successful
        """
        
        if message is None:
            message = f"Auto-update model - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        try:
            # Add all changes
            print("\n📤 Deploying to GitHub...")
            
            subprocess.run(
                ['git', 'add', '.'],
                cwd=self.repo_path,
                check=True
            )
            print("   ✅ Changes staged")
            
            # Commit
            subprocess.run(
                ['git', 'commit', '-m', message],
                cwd=self.repo_path,
                check=True
            )
            print(f"   ✅ Committed: {message}")
            
            # Push
            subprocess.run(
                ['git', 'push'],
                cwd=self.repo_path,
                check=True
            )
            print("   ✅ Pushed to GitHub")
            print("\n🚀 Streamlit Cloud will auto-deploy in 2-3 minutes")
            
            return True
        
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Deployment failed: {e}")
            return False
    
    def deploy_if_needed(self, validation_passed=True):
        """
        Deploy only if validation passed and changes exist
        
        Args:
            validation_passed: Whether model validation passed
        
        Returns:
            bool: True if deployed
        """
        
        if not validation_passed:
            print("\n⚠️  Validation failed - NOT deploying")
            return False
        
        if not self.check_git_status():
            print("\n✅ No changes to deploy")
            return False
        
        return self.commit_and_push()

# Testing
if __name__ == "__main__":
    deployer = AutoDeployer()
    
    # Check for changes
    has_changes = deployer.check_git_status()
    
    if has_changes:
        print("\nWould you like to deploy? (y/n): ", end='')
        response = input()
        
        if response.lower() == 'y':
            deployer.commit_and_push()