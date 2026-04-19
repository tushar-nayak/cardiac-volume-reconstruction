#!/usr/bin/env python3
"""
ABLATION STUDIES: Complete & Working Version
Runs ablations on all 10 subjects with proper evaluation metrics.

KEY FIXES:
- Uses internal evaluation logic (doesn't depend on external evaluate_subject)
- Proper device handling for tensors
- Correct Dice/IoU computation
- Averages metrics across all contours (not just first)
- Adds occupancy threshold robustness
- Increased optimization steps to 150 for better convergence
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import traceback
from collections import defaultdict

# Import ONLY data utilities from minimal_starter
from minimal_starter_5 import (
    load_mitea_subject,
    extract_synthetic_2d_slices,
    find_mitea_image_files,
    optimize_single_subject,
    ImplicitNeuralRepresentation,
    CONFIG
)


# ============================================================================
# INTERNAL EVALUATION (Bypass External Dependencies)
# ============================================================================

def evaluate_reconstruction(model, contours_2d, device, resolution=64):
    """
    Evaluate reconstruction quality across ALL contours.
    Returns averaged Dice and IoU metrics.
    
    CRITICAL FIX:
    - Evaluates on ALL views, not just first view
    - Uses adaptive thresholding (otsu-like) instead of fixed 0.5
    - Returns mean metrics across views for robustness
    """
    model.eval()
    with torch.no_grad():
        # Sample occupancy grid
        occupancy_grid = model.sample_grid(resolution=resolution, device=device, requires_grad=False)
        
        # Project 3D -> 2D via max-pooling
        pred_projection = torch.max(occupancy_grid, dim=0)[0]  # (64, 64)
        
        # Move contours to device
        contours_device = contours_2d.to(device)
        
        # Compute metrics for EACH contour and average
        all_dice = []
        all_iou = []
        
        for view_idx in range(contours_device.shape[0]):
            # Resize projection to match contour size
            pred_resized = F.interpolate(
                pred_projection.unsqueeze(0).unsqueeze(0),
                size=contours_device[view_idx].shape,
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            # Adaptive threshold: use median of predicted values
            threshold = torch.median(pred_resized).item()
            threshold = max(threshold, 0.3)  # Ensure minimum threshold
            
            # Binarize both
            pred_binary = (pred_resized > threshold).float()
            target_binary = (contours_device[view_idx] > 0.5).float()
            
            # Compute metrics
            intersection = torch.sum(pred_binary * target_binary).item()
            union = torch.sum(pred_binary).item() + torch.sum(target_binary).item()
            
            dice = (2.0 * intersection) / (union + 1e-6)
            iou = intersection / (union - intersection + 1e-6)
            
            all_dice.append(dice)
            all_iou.append(iou)
        
        # Return mean across all views
        return {
            'dice': np.mean(all_dice),
            'iou': np.mean(all_iou)
        }


# ============================================================================
# ABLATION STUDIES CLASS
# ============================================================================

class AblationStudies:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path('./ablation_results')
        self.output_dir.mkdir(exist_ok=True)
        self.aggregated = defaultdict(lambda: defaultdict(list))
    
    def run_single_experiment(self, parameter_name, parameter_values, subject_pair):
        """Run one ablation experiment on one subject"""
        
        img_file, label_file = subject_pair
        vol, seg = load_mitea_subject(img_file, label_file)
        
        if vol is None:
            print(f"    Failed to load {img_file.name}")
            return None
        
        # Extract 6 views
        all_slices, all_contours = extract_synthetic_2d_slices(vol, seg, num_views=6)
        results = {parameter_name: [], 'dice': []}
        
        for val in parameter_values:
            cfg = self.config.copy()
            num_views = 6
            
            # Configure based on parameter
            if parameter_name == 'num_views':
                num_views = val
            elif parameter_name == 'hidden_dim':
                cfg['hidden_dim'] = val
            elif parameter_name == 'num_layers':
                cfg['num_inr_layers'] = val
            elif parameter_name == 'learning_rate':
                cfg['learning_rate'] = val
            
            # Use subset of views
            slices = all_slices[:num_views]
            contours = all_contours[:num_views]
            
            try:
                # Create model
                model = ImplicitNeuralRepresentation(
                    hidden_dim=cfg['hidden_dim'],
                    num_layers=cfg['num_inr_layers']
                )
                
                # Optimize with increased steps for better convergence
                model, _ = optimize_single_subject(
                    model, slices, contours, cfg, num_steps=150
                )
                
                # Evaluate with internal function
                device = torch.device(cfg['device'])
                metrics = evaluate_reconstruction(model, contours, device)
                
                results[parameter_name].append(val)
                results['dice'].append(metrics['dice'])
                self.aggregated[parameter_name][val].append(metrics['dice'])
                
            except Exception as e:
                print(f"    Error {parameter_name}={val}: {type(e).__name__}: {str(e)[:30]}")
                results[parameter_name].append(val)
                results['dice'].append(0.0)
        
        return results

    def plot_results(self):
        """Generate and save plots"""
        print("\n" + "="*60)
        print("GENERATING PLOTS")
        print("="*60)
        
        for param, data_dict in self.aggregated.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sort keys numerically if possible
            try:
                keys = sorted(data_dict.keys(), key=float)
            except:
                keys = sorted(data_dict.keys())
            
            means = [np.mean(data_dict[k]) for k in keys]
            stds = [np.std(data_dict[k]) for k in keys]
            
            if param == 'learning_rate':
                # Log scale for learning rate
                labels = [f'{k:.0e}' for k in keys]
                x_pos = np.arange(len(labels))
                ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels, rotation=45)
            else:
                # Linear scale
                ax.errorbar(keys, means, yerr=stds, fmt='o-', capsize=5,
                           linewidth=2, markersize=8, label='Dice')
            
            ax.set_ylabel('Dice Coefficient')
            ax.set_ylim([0, 1.0])
            ax.set_xlabel(param)
            ax.set_title(f'Ablation: {param}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            out = self.output_dir / f"ablation_{param}.png"
            plt.savefig(out, dpi=150)
            plt.close()
            print(f"✓ Saved: {out}")

    def print_summary(self):
        """Print aggregated statistics"""
        print("\n" + "="*60)
        print("ABLATION SUMMARY")
        print("="*60)
        
        for param, data in self.aggregated.items():
            print(f"\n{param.upper()}:")
            for val in sorted(data.keys()):
                scores = data[val]
                mean_s = np.mean(scores)
                std_s = np.std(scores)
                print(f"  {str(val):>8}: Dice = {mean_s:.4f} ± {std_s:.4f} (N={len(scores)})")

    def save_results(self):
        """Save JSON results"""
        out = self.output_dir / "aggregated_results.json"
        data = {k: {str(vk): vv for vk, vv in v.items()} for k, v in self.aggregated.items()}
        with open(out, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"\n✓ Results saved to: {out}")


def main():
    print("="*60)
    print("ABLATION STUDIES - ROBUST EVALUATION")
    print("="*60)
    
    # Get subjects
    data_path = CONFIG['data_path']
    subjects = find_mitea_image_files(data_path)[:10]
    
    if not subjects:
        print("ERROR: No subjects found!")
        return
    
    print(f"Found {len(subjects)} subjects\n")
    
    ablations = AblationStudies(CONFIG)
    
    # Run ablations on all subjects
    for i, sub in enumerate(subjects):
        sid = sub[0].stem
        print(f"[{i+1}/{len(subjects)}] {sid}")
        
        try:
            ablations.run_single_experiment("num_views", [2, 3, 4, 6], sub)
            ablations.run_single_experiment("hidden_dim", [32, 64, 128, 256], sub)
            ablations.run_single_experiment("num_layers", [2, 4, 6, 8], sub)
            ablations.run_single_experiment("learning_rate", [1e-4, 5e-4, 1e-3, 5e-3], sub)
            print("  ✓ Done")
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
    
    # Generate outputs
    ablations.plot_results()
    ablations.print_summary()
    ablations.save_results()
    
    print("\n" + "="*60)
    print("✅ ABLATION STUDIES COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
