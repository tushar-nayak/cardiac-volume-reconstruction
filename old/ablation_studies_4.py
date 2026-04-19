#!/usr/bin/env python3
"""
ABLATION STUDIES V2: Self-contained & Robust
Run this to perform ablation studies on MITEA subjects.

This script defines its own Model and Optimization logic to avoid 
dependency issues with external files.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import traceback
from collections import defaultdict

# Only import data loading utilities
try:
    from minimal_starter_5 import (
        load_mitea_subject,
        extract_synthetic_2d_slices,
        find_mitea_image_files,
        CONFIG
    )
except ImportError:
    print("ERROR: minimal_starter.py not found. Please ensure it is in the same directory.")
    exit(1)


# ============================================================================
# INTERNAL MODEL DEFINITION (Ensures Correct Gradient Flow)
# ============================================================================

class PositionalEncoding:
    """Positional encoding for coordinates."""
    def __init__(self, num_freqs=4):
        self.num_freqs = num_freqs
    
    def encode(self, coords):
        """coords: (..., 3) in [-1, 1]"""
        pe = []
        for i in range(self.num_freqs):
            freq = 2.0 ** i
            pe.append(torch.sin(freq * np.pi * coords))
            pe.append(torch.cos(freq * np.pi * coords))
        return torch.cat(pe, dim=-1)


class ImplicitNeuralRepresentation(nn.Module):
    """INR: f(x,y,z) -> occupancy ∈ [0,1]"""
    def __init__(self, hidden_dim=64, num_layers=4):
        super().__init__()
        self.pe = PositionalEncoding(num_freqs=4)
        input_dim = 8 * 3
        
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else 1
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, coords):
        pe_coords = self.pe.encode(coords)
        return torch.sigmoid(self.mlp(pe_coords))
    
    def sample_grid(self, resolution=64, device='cpu', requires_grad=False):
        """Sample occupancy on 3D grid with explicit gradient control"""
        linspace = torch.linspace(-1, 1, resolution, device=device)
        grid = torch.stack(torch.meshgrid(linspace, linspace, linspace, indexing='ij'), dim=-1)
        
        if requires_grad:
            return self.forward(grid).squeeze(-1)
        else:
            with torch.no_grad():
                return self.forward(grid).squeeze(-1)


# ============================================================================
# INTERNAL OPTIMIZATION & EVALUATION LOGIC
# ============================================================================

def contour_reprojection_loss(occupancy_grid, target_contour):
    # Max-pooling projection
    projection = torch.max(occupancy_grid, dim=0)[0]
    
    # Resize to match contour
    projection_resized = F.interpolate(
        projection.unsqueeze(0).unsqueeze(0),
        size=target_contour.shape,
        mode='bilinear',
        align_corners=False
    ).squeeze()
    
    return F.binary_cross_entropy(projection_resized, target_contour)

def optimize_subject_internal(model, slices_2d, contours_2d, config, num_steps=50):
    device = torch.device(config['device'])
    model = model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    contours_device = contours_2d.to(device)
    
    losses = []
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # CRITICAL: requires_grad=True for backprop
        occupancy_grid = model.sample_grid(resolution=64, device=device, requires_grad=True)
        
        loss = 0.0
        for view_idx in range(contours_2d.shape[0]):
            loss += contour_reprojection_loss(occupancy_grid, contours_device[view_idx])
        
        # Smoothness
        if occupancy_grid.requires_grad:
            lap = (
                torch.roll(occupancy_grid, 1, dims=0) + torch.roll(occupancy_grid, -1, dims=0) +
                torch.roll(occupancy_grid, 1, dims=1) + torch.roll(occupancy_grid, -1, dims=1) +
                torch.roll(occupancy_grid, 1, dims=2) + torch.roll(occupancy_grid, -1, dims=2) -
                6 * occupancy_grid
            )
            loss += 0.01 * torch.mean(lap ** 2)
        
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    return model, losses

def evaluate_subject_internal(model, occupancy_grid, contours_2d, device):
    contours_device = contours_2d.to(device)
    
    # Project 3D -> 2D
    pred_projection = torch.max(occupancy_grid, dim=0)[0]
    
    # Resize
    pred_projection_resized = F.interpolate(
        pred_projection.unsqueeze(0).unsqueeze(0),
        size=contours_device[0].shape,
        mode='bilinear',
        align_corners=False
    ).squeeze()
    
    # Binarize
    pred_binary = (pred_projection_resized > 0.5).float()
    target_binary = (contours_device[0] > 0.5).float()
    
    # Metrics
    intersection = torch.sum(pred_binary * target_binary)
    union = torch.sum(pred_binary) + torch.sum(target_binary)
    dice = (2 * intersection) / (union + 1e-6)
    iou = intersection / (union - intersection + 1e-6)
    
    return {'dice': dice.item(), 'iou': iou.item()}


# ============================================================================
# ABLATION CLASS
# ============================================================================

class AblationStudies:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path('./ablation_results')
        self.output_dir.mkdir(exist_ok=True)
        self.aggregated = defaultdict(lambda: defaultdict(list))
    
    def run_single_experiment(self, parameter_name, parameter_values, subject_pair):
        img_file, label_file = subject_pair
        vol, seg = load_mitea_subject(img_file, label_file)
        
        if vol is None: 
            return None

        # Extract max views (6)
        all_slices, all_contours = extract_synthetic_2d_slices(vol, seg, num_views=6)
        
        results = {parameter_name: [], 'dice': []}
        
        for val in parameter_values:
            # Setup config
            cfg = self.config.copy()
            num_views = 6
            
            if parameter_name == 'num_views':
                num_views = val
            elif parameter_name == 'hidden_dim':
                cfg['hidden_dim'] = val
            elif parameter_name == 'num_layers':
                cfg['num_inr_layers'] = val
            elif parameter_name == 'learning_rate':
                cfg['learning_rate'] = val
            
            # Select views
            slices_subset = all_slices[:num_views]
            contours_subset = all_contours[:num_views]
            
            try:
                # 1. Initialize Internal Model
                model = ImplicitNeuralRepresentation(
                    hidden_dim=cfg['hidden_dim'],
                    num_layers=cfg['num_inr_layers']
                )
                
                # 2. Optimize (Internal Function)
                model, _ = optimize_subject_internal(model, slices_subset, contours_subset, cfg)
                
                # 3. Evaluate (Internal Function)
                device = torch.device(cfg['device'])
                with torch.no_grad():
                    occ = model.sample_grid(resolution=64, device=device, requires_grad=False)
                    metrics = evaluate_subject_internal(model, occ, contours_subset, device)
                
                # Record
                results[parameter_name].append(val)
                results['dice'].append(metrics['dice'])
                self.aggregated[parameter_name][val].append(metrics['dice'])
                
            except Exception as e:
                print(f"    Error {parameter_name}={val}: {e}")
                traceback.print_exc()
                results[parameter_name].append(val)
                results['dice'].append(0.0)
        
        return results

    def plot_results(self):
        print("\n" + "="*40)
        print("SAVING PLOTS")
        print("="*40)
        for param, data in self.aggregated.items():
            plt.figure(figsize=(8, 5))
            
            keys = sorted(data.keys())
            # Sort based on numeric value if possible
            try:
                keys.sort(key=float)
            except:
                pass
                
            means = [np.mean(data[k]) for k in keys]
            stds = [np.std(data[k]) for k in keys]
            
            plt.errorbar(keys, means, yerr=stds, fmt='-o', capsize=5, label='Dice')
            plt.xlabel(param)
            plt.ylabel('Dice Coefficient')
            plt.title(f'Ablation: {param}')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.0)
            
            if param == 'learning_rate':
                plt.xscale('log')
                
            out_path = self.output_dir / f"ablation_{param}.png"
            plt.savefig(out_path)
            print(f"Saved {out_path}")
            plt.close()

    def save_summary(self):
        out_path = self.output_dir / "aggregated_results.json"
        serializable = {k: {str(vk): vv for vk, vv in v.items()} for k, v in self.aggregated.items()}
        with open(out_path, 'w') as f:
            json.dump(serializable, f, indent=4)
        print(f"Saved summary to {out_path}")


def main():
    print("="*60)
    print("ABLATION STUDIES V2 (Robust)")
    print("="*60)
    
    data_path = CONFIG['data_path']
    subjects = find_mitea_image_files(data_path)[:10]
    
    if not subjects:
        print("No subjects found.")
        return

    ablations = AblationStudies(CONFIG)
    
    for i, sub in enumerate(subjects):
        sid = sub[0].stem
        print(f"\n[{i+1}/{len(subjects)}] {sid}")
        
        ablations.run_single_experiment("num_views", [2, 3, 4, 6], sub)
        ablations.run_single_experiment("hidden_dim", [32, 64, 128, 256], sub)
        ablations.run_single_experiment("num_layers", [2, 4, 6, 8], sub)
        ablations.run_single_experiment("learning_rate", [1e-4, 5e-4, 1e-3, 5e-3], sub)
        print("  Done.")

    ablations.plot_results()
    ablations.save_summary()
    print("\nDONE.")

if __name__ == '__main__':
    main()
