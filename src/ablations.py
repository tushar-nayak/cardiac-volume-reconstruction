#!/usr/bin/env python3
"""
ablations_wandb.py

High-Intensity Ablation Runner for Echo3D Hybrid.
- Imports logic from 'echo3d_hybrid.py'
- Uses Random Search to explore huge hyperparameter spaces.
- Reconstruction of the 'Mixed' pipeline to capture specific metrics.
- Logs everything to Weights & Biases.
"""

import copy
import random
import time
import traceback
import numpy as np
import torch
import wandb
from collections import defaultdict
from typing import Dict, Any

# =========================================================================
# 1. IMPORT MASTER CODE
# =========================================================================
try:
    import FINAL_2_gpu_optmized as core
    print("✅ Successfully imported 'echo3d_hybrid.py'")
except ImportError:
    print("❌ ERROR: Could not import 'echo3d_hybrid.py'.")
    print("   Please ensure the master code is saved as 'echo3d_hybrid.py'")
    exit(1)

# =========================================================================
# 2. CONFIGURATION & SEARCH SPACE
# =========================================================================

PROJECT_NAME = "Echo3D-Hybrid-Ablation-Study"
NUM_RUNS = 50  # How many different experiments to run

# Parameters that stay constant for all ablation runs (Fast settings)
BASE_OVERRIDES = {
    "mode": "mixed",
    "save_nifti": False,       # Save disk space
    "print_every": 1000,       # Reduce console spam
    "num_epochs": 10,          # Faster global training
    "steps_per_epoch": 100,
    "mesh_threshold": 0.5,
    "subject_category_mode": "healthy_only" # Keep data consistent
}

# The Hyperparameter Search Space (Random Search)
SEARCH_SPACE = {
    # --- Optimization Dynamics ---
    "mixed_refine_steps": [50, 100, 200, 400],    # Test-time training steps
    "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3],    # INR plasticity
    
    # --- Regularization ---
    "vol_supervision_weight": [0.0, 0.1, 1.0, 5.0], 
    "pose_reg_weight": [0.0, 1e-4, 1e-2],
    
    # --- Architecture ---
    "hidden_dim": [32, 64, 128],  
    "num_inr_layers": [2, 4, 6],
    
    # --- Data Scarcity ---
    "num_views": [3, 5, 7],       # How many slices do we have?
    "learn_pose": [True, False]   # Do we trust the scanner metadata?
}

# =========================================================================
# 3. UTILITIES
# =========================================================================

def get_random_config(search_space: Dict[str, Any]) -> Dict[str, Any]:
    """Randomly samples one value for each key in the search space."""
    return {k: random.choice(v) for k, v in search_space.items()}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =========================================================================
# 4. CUSTOM PIPELINE RUNNER
# =========================================================================
# We re-implement the high-level logic of 'main_mixed' here so we can 
# capture metrics and log them to W&B directly.

def run_mixed_pipeline_with_logging(config: Dict, run_name: str):
    """
    Executes the Mixed Mode pipeline using functions from echo3d_hybrid.
    """
    # 1. Update the Global Config in the imported module
    # This is necessary because the core functions rely on echo3d_hybrid.CONFIG
    core.CONFIG.update(config)
    set_seed(core.CONFIG["random_seed"])
    device = torch.device(core.CONFIG["device"])
    
    print(f"\n🚀 Starting Run: {run_name}")
    
    # 2. Build Dataset
    # We use the core function to load data
    dataset = core.build_scan_dataset(core.CONFIG)
    train_scans = dataset["train"]
    val_scans = dataset["val"]
    test_scans = dataset["test"]

    if not train_scans:
        raise RuntimeError("No training data found.")

    # 3. Train Global Prior (Baseline B)
    print("   TRAINING GLOBAL PRIOR...")
    model_global = core.train_global_inr(train_scans, val_scans, core.CONFIG)
    
    # 4. Refine on Test Split (The "Mixed" Step)
    print("   REFINING ON TEST SET...")
    
    # Metrics accumulators
    metrics_acc = defaultdict(list)
    refine_steps = core.CONFIG["mixed_refine_steps"]
    
    # We iterate through test scans manually to log progress
    for i, scan in enumerate(test_scans):
        
        # Clone the global model
        model_local = core.ImplicitNeuralRepresentation(
            hidden_dim=core.CONFIG["hidden_dim"],
            num_layers=core.CONFIG["num_inr_layers"]
        ).to(device)
        model_local.load_state_dict(model_global.state_dict())
        
        # Initialize Pose
        pose_layer = core.PoseParameters(core.CONFIG["num_views"]).to(device)
        if not core.CONFIG["learn_pose"]:
            with torch.no_grad():
                pose_layer.pose.zero_()
                
        # Refine (Optimization)
        # We reuse the core optimization function
        model_local, pose_layer, _ = core.optimize_single_subject(
            model=model_local,
            slices_2d=scan.slices_2d,
            contours_2d=scan.contours_2d,
            pose_layer=pose_layer,
            chosen=scan.chosen_z,
            config=core.CONFIG,
            num_steps=refine_steps,
            D=scan.D,
            seg_vol=scan.seg_c
        )
        
        # Evaluate
        m3d = core.evaluate_subject_3d_and_mesh(
            model_local, scan.seg_c, f"{scan.scan_id}_ablation", core.CONFIG, scan.chosen_z
        )
        
        # Accumulate
        metrics_acc["dice_3d"].append(m3d["dice_3d"])
        metrics_acc["iou_3d"].append(m3d["iou_3d"])
        metrics_acc["dice_3d_central"].append(m3d["dice_3d_central"])

    # 5. Aggregate Results
    final_results = {
        "final/dice_3d_mean": np.mean(metrics_acc["dice_3d"]),
        "final/dice_3d_std": np.std(metrics_acc["dice_3d"]),
        "final/iou_3d_mean": np.mean(metrics_acc["iou_3d"]),
        "final/dice_central_mean": np.mean(metrics_acc["dice_3d_central"]),
    }
    
    return final_results

# =========================================================================
# 5. MAIN ABLATION LOOP
# =========================================================================

def main():
    print(f"Starting W&B Ablation Study: {PROJECT_NAME}")
    print(f"Total Runs: {NUM_RUNS}")
    
    for i in range(NUM_RUNS):
        # A. Sample Hyperparameters
        sampled_params = get_random_config(SEARCH_SPACE)
        
        # B. Merge with Base Overrides
        run_config = core.CONFIG.copy() # Start with defaults
        run_config.update(BASE_OVERRIDES) # Apply fixed settings
        run_config.update(sampled_params) # Apply random settings
        
        # C. Generate Run Name
        run_name = f"run_{i:02d}_views{sampled_params['num_views']}_refine{sampled_params['mixed_refine_steps']}"
        
        # D. Init W&B
        try:
            wandb.init(
                project=PROJECT_NAME,
                name=run_name,
                config=run_config,
                reinit=True
            )
            
            start_time = time.time()
            
            # E. Run Pipeline
            results = run_mixed_pipeline_with_logging(run_config, run_name)
            
            duration = time.time() - start_time
            
            # F. Log Metrics
            wandb.log(results)
            wandb.log({"perf/duration_seconds": duration})
            
            print(f"   ✅ Finished Run {i}. 3D Dice: {results['final/dice_3d_mean']:.4f}")
            
        except Exception as e:
            print(f"   ❌ FAILED Run {i}: {e}")
            traceback.print_exc()
            wandb.log({"error": str(e)})
            
        finally:
            wandb.finish()
            # Clean up GPU memory
            torch.cuda.empty_cache()

    print("\n========================================")
    print("Study Complete.")
    print(f"View results at https://wandb.ai/home -> Project: {PROJECT_NAME}")
    print("========================================")

if __name__ == "__main__":
    main()