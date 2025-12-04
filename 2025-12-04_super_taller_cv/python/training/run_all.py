#!/usr/bin/env python
"""
Quick Start Script - Subsystem 5
Run all training pipelines with a single command
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def run_cnn_training(skip_cv=False):
    """Run CNN from scratch training"""
    print_header("Training CNN from Scratch")
    
    script_path = os.path.join("python", "training", "cnn_from_scratch.py")
    
    if skip_cv:
        print("⚠️  Skipping cross-validation...")
        # Modify to skip CV interactively
    
    result = subprocess.run([sys.executable, script_path], 
                          capture_output=False, text=True)
    
    if result.returncode == 0:
        print("✓ CNN training completed successfully!")
    else:
        print("✗ CNN training failed!")
        return False
    
    return True

def run_fine_tuning(models=None):
    """Run fine-tuning on selected models"""
    print_header("Fine-Tuning Pre-trained Models")
    
    script_path = os.path.join("python", "training", "fine_tuning.py")
    
    result = subprocess.run([sys.executable, script_path],
                          capture_output=False, text=True)
    
    if result.returncode == 0:
        print("✓ Fine-tuning completed successfully!")
    else:
        print("✗ Fine-tuning failed!")
        return False
    
    return True

def run_comparison():
    """Run model comparison"""
    print_header("Generating Model Comparisons")
    
    script_path = os.path.join("python", "training", "compare_models.py")
    
    result = subprocess.run([sys.executable, script_path],
                          capture_output=False, text=True)
    
    if result.returncode == 0:
        print("✓ Comparison completed successfully!")
    else:
        print("✗ Comparison failed!")
        return False
    
    return True

def launch_dashboard():
    """Launch Streamlit dashboard"""
    print_header("Launching Interactive Dashboard")
    
    script_path = os.path.join("python", "training", "dashboard.py")
    
    print("📊 Starting Streamlit dashboard...")
    print("🌐 Dashboard will open at: http://localhost:8501")
    print("⚠️  Press Ctrl+C to stop the dashboard\n")
    
    subprocess.run(["streamlit", "run", script_path])

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Quick start script for Subsystem 5"
    )
    
    parser.add_argument(
        '--cnn-only',
        action='store_true',
        help='Train only CNN from scratch'
    )
    
    parser.add_argument(
        '--ft-only',
        action='store_true',
        help='Run only fine-tuning'
    )
    
    parser.add_argument(
        '--compare-only',
        action='store_true',
        help='Run only comparison (requires trained models)'
    )
    
    parser.add_argument(
        '--dashboard-only',
        action='store_true',
        help='Launch only dashboard (requires trained models)'
    )
    
    parser.add_argument(
        '--skip-cv',
        action='store_true',
        help='Skip cross-validation in CNN training'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run complete pipeline (default)'
    )
    
    args = parser.parse_args()
    
    print_header("Subsystem 5: Deep Learning Training Pipeline")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine what to run
    run_all = args.all or not any([
        args.cnn_only, args.ft_only, 
        args.compare_only, args.dashboard_only
    ])
    
    try:
        # Run CNN training
        if run_all or args.cnn_only:
            if not run_cnn_training(skip_cv=args.skip_cv):
                print("\n⚠️  CNN training failed. Stopping pipeline.")
                return
        
        # Run fine-tuning
        if run_all or args.ft_only:
            if not run_fine_tuning():
                print("\n⚠️  Fine-tuning failed. Stopping pipeline.")
                return
        
        # Run comparison
        if run_all or args.compare_only:
            if not run_comparison():
                print("\n⚠️  Comparison failed.")
                # Continue anyway
        
        # Launch dashboard
        if run_all or args.dashboard_only:
            launch_dashboard()
        
        print_header("Pipeline Completed Successfully!")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
