#!/usr/bin/env python3
"""
SIMPLE RUNNER - Execute Full Pipeline
Just run: python run_all.py
"""

import sys
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

def run_command(cmd, name):
    """Run a command and handle errors"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {name}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        print(f"\n✅ {name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {name} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ {name} failed: {e}")
        return False

def main():
    print("\n" + "="*80)
    print("3D CARDIAC RECONSTRUCTION - COMPLETE PIPELINE")
    print("="*80)
    print("\nThis will run:")
    print("  1. Main Training (10 subjects) - 15 min")
    print("  2. Ablation Studies (3 subjects) - 30 min")
    print("  3. Sparse Reconstruction (3 subjects) - 20 min")
    print("  4. Train/Validation Split (10 subjects) - 20 min")
    print("\nTotal estimated time: ~1.5 hours on GPU")
    print("\n" + "="*80)
    
    response = input("\nContinue? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    results = {}
    
    # Stage 1: Main training
    if run_command(
        [sys.executable, str(SCRIPT_DIR / 'minimal_starter_5.py')],
        'STAGE 1: Main Training'
    ):
        results['training'] = '✅'
    else:
        results['training'] = '❌'
    
    # Stage 2: Ablations
    if run_command(
        [sys.executable, str(SCRIPT_DIR / 'ablation_studies_7.py')],
        'STAGE 2: Ablation Studies'
    ):
        results['ablations'] = '✅'
    else:
        results['ablations'] = '❌'
    
    # Stage 3: Sparse reconstruction
    if run_command(
        [sys.executable, str(SCRIPT_DIR / 'sparse_reconstruction_2.py')],
        'STAGE 3: Sparse Reconstruction'
    ):
        results['sparse'] = '✅'
    else:
        results['sparse'] = '❌'
    
    # Stage 4: Integrated pipeline (includes train/val)
    if run_command(
        [sys.executable, str(SCRIPT_DIR / 'integrated_pipeline.py')],
        'STAGE 4: Integrated Pipeline'
    ):
        results['integrated'] = '✅'
    else:
        results['integrated'] = '❌'
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    print("\nResults:")
    print(f"  Stage 1 - Training:             {results.get('training', '?')}")
    print(f"  Stage 2 - Ablations:            {results.get('ablations', '?')}")
    print(f"  Stage 3 - Sparse Recon:         {results.get('sparse', '?')}")
    print(f"  Stage 4 - Integrated Pipeline:  {results.get('integrated', '?')}")
    
    all_passed = all(v == '✅' for v in results.values())
    
    if all_passed:
        print("\n✅ ALL STAGES COMPLETED SUCCESSFULLY!")
        print("\nOutput files:")
        print("  ./training_results.json")
        print("  ./ablation_results.json")
        print("  ./sparse_reconstruction_results/")
        print("  ./pipeline_results/")
    else:
        print("\n⚠️  Some stages failed. Check output above.")
    
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
