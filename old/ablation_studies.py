#!/usr/bin/env python3
"""
ABLATION STUDIES: Test contribution of each component
Run after getting baseline to work
"""

import torch
import numpy as np
from minimal_starter import (
    ImplicitNeuralRepresentation,
    load_mitea_subject,
    extract_synthetic_2d_slices,
    optimize_single_subject,
    evaluate_subject,
    CONFIG
)
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# ABLATION EXPERIMENTS
# ============================================================================

class AblationStudies:
    def __init__(self, config):
        self.config = config
        self.results = {}
    
    def ablation_num_views(self, subject_path, max_views=6):
        """Test effect of varying number of views"""
        print("\n" + "="*60)
        print("ABLATION: Number of Views")
        print("="*60)
        
        vol, seg = load_mitea_subject(subject_path)
        slices_2d, contours_2d = extract_synthetic_2d_slices(vol, seg, num_views=max_views)
        
        results = {'num_views': [], 'dice': [], 'iou': []}
        
        for num_views in [2, 3, 4, 6]:
            print(f"Testing with {num_views} views...")
            
            model = ImplicitNeuralRepresentation(
                hidden_dim=self.config['hidden_dim'],
                num_layers=self.config['num_inr_layers']
            )
            
            # Use first num_views
            subset_contours = contours_2d[:num_views]
            
            model, _ = optimize_single_subject(
                model,
                slices_2d[:num_views],
                subset_contours,
                self.config,
                num_steps=self.config['num_optimization_steps']
            )
            
            # Evaluate
            device = torch.device(self.config['device'])
            with torch.no_grad():
                occupancy_grid = model.sample_grid(resolution=64, device=device)
                metrics = evaluate_subject(model, occupancy_grid, subset_contours)
            
            results['num_views'].append(num_views)
            results['dice'].append(metrics['dice'])
            results['iou'].append(metrics['iou'])
            
            print(f"  Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}")
        
        self.results['num_views'] = results
        return results
    
    def ablation_hidden_dimension(self, subject_path, num_views=6):
        """Test effect of model capacity"""
        print("\n" + "="*60)
        print("ABLATION: Hidden Dimension (Model Capacity)")
        print("="*60)
        
        vol, seg = load_mitea_subject(subject_path)
        slices_2d, contours_2d = extract_synthetic_2d_slices(vol, seg, num_views=num_views)
        
        results = {'hidden_dim': [], 'dice': [], 'iou': [], 'params': []}
        
        for hidden_dim in [16, 32, 64, 128]:
            print(f"Testing with hidden_dim={hidden_dim}...")
            
            model = ImplicitNeuralRepresentation(
                hidden_dim=hidden_dim,
                num_layers=self.config['num_inr_layers']
            )
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            
            model, _ = optimize_single_subject(
                model,
                slices_2d,
                contours_2d,
                self.config,
                num_steps=self.config['num_optimization_steps']
            )
            
            device = torch.device(self.config['device'])
            with torch.no_grad():
                occupancy_grid = model.sample_grid(resolution=64, device=device)
                metrics = evaluate_subject(model, occupancy_grid, contours_2d)
            
            results['hidden_dim'].append(hidden_dim)
            results['dice'].append(metrics['dice'])
            results['iou'].append(metrics['iou'])
            results['params'].append(num_params)
            
            print(f"  Params: {num_params}, Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}")
        
        self.results['hidden_dim'] = results
        return results
    
    def ablation_num_layers(self, subject_path, num_views=6):
        """Test effect of network depth"""
        print("\n" + "="*60)
        print("ABLATION: Network Depth (Number of Layers)")
        print("="*60)
        
        vol, seg = load_mitea_subject(subject_path)
        slices_2d, contours_2d = extract_synthetic_2d_slices(vol, seg, num_views=num_views)
        
        results = {'num_layers': [], 'dice': [], 'iou': [], 'params': []}
        
        for num_layers in [2, 3, 4, 6]:
            print(f"Testing with num_layers={num_layers}...")
            
            model = ImplicitNeuralRepresentation(
                hidden_dim=self.config['hidden_dim'],
                num_layers=num_layers
            )
            
            num_params = sum(p.numel() for p in model.parameters())
            
            model, _ = optimize_single_subject(
                model,
                slices_2d,
                contours_2d,
                self.config,
                num_steps=self.config['num_optimization_steps']
            )
            
            device = torch.device(self.config['device'])
            with torch.no_grad():
                occupancy_grid = model.sample_grid(resolution=64, device=device)
                metrics = evaluate_subject(model, occupancy_grid, contours_2d)
            
            results['num_layers'].append(num_layers)
            results['dice'].append(metrics['dice'])
            results['iou'].append(metrics['iou'])
            results['params'].append(num_params)
            
            print(f"  Params: {num_params}, Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}")
        
        self.results['num_layers'] = results
        return results
    
    def ablation_learning_rate(self, subject_path, num_views=6):
        """Test effect of learning rate"""
        print("\n" + "="*60)
        print("ABLATION: Learning Rate")
        print("="*60)
        
        vol, seg = load_mitea_subject(subject_path)
        slices_2d, contours_2d = extract_synthetic_2d_slices(vol, seg, num_views=num_views)
        
        results = {'learning_rate': [], 'dice': [], 'iou': []}
        
        for lr in [1e-4, 5e-4, 1e-3, 5e-3]:
            print(f"Testing with learning_rate={lr}...")
            
            model = ImplicitNeuralRepresentation(
                hidden_dim=self.config['hidden_dim'],
                num_layers=self.config['num_inr_layers']
            )
            
            # Temporarily override learning rate
            orig_lr = self.config['learning_rate']
            self.config['learning_rate'] = lr
            
            model, _ = optimize_single_subject(
                model,
                slices_2d,
                contours_2d,
                self.config,
                num_steps=self.config['num_optimization_steps']
            )
            
            self.config['learning_rate'] = orig_lr
            
            device = torch.device(self.config['device'])
            with torch.no_grad():
                occupancy_grid = model.sample_grid(resolution=64, device=device)
                metrics = evaluate_subject(model, occupancy_grid, contours_2d)
            
            results['learning_rate'].append(lr)
            results['dice'].append(metrics['dice'])
            results['iou'].append(metrics['iou'])
            
            print(f"  Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}")
        
        self.results['learning_rate'] = results
        return results
    
    def plot_results(self, output_dir='./results'):
        """Plot ablation study results"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Plot 1: Number of views
        if 'num_views' in self.results:
            r = self.results['num_views']
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.plot(r['num_views'], r['dice'], 'o-', label='Dice', linewidth=2)
            ax1.set_xlabel('Number of Views')
            ax1.set_ylabel('Dice Coefficient')
            ax1.set_title('Effect of Number of Views')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(r['num_views'], r['iou'], 's-', label='IoU', color='orange', linewidth=2)
            ax2.set_xlabel('Number of Views')
            ax2.set_ylabel('IoU')
            ax2.set_title('Effect of Number of Views (IoU)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/ablation_num_views.png', dpi=150)
            print(f"Saved: {output_dir}/ablation_num_views.png")
            plt.close()
        
        # Plot 2: Hidden dimension
        if 'hidden_dim' in self.results:
            r = self.results['hidden_dim']
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.plot(r['hidden_dim'], r['dice'], 'o-', linewidth=2)
            ax1.set_xlabel('Hidden Dimension')
            ax1.set_ylabel('Dice Coefficient')
            ax1.set_title('Effect of Model Capacity')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(r['params'], r['iou'], 's-', color='orange', linewidth=2)
            ax2.set_xlabel('Number of Parameters')
            ax2.set_ylabel('IoU')
            ax2.set_title('Performance vs Model Size')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/ablation_hidden_dim.png', dpi=150)
            print(f"Saved: {output_dir}/ablation_hidden_dim.png")
            plt.close()
        
        # Plot 3: Network depth
        if 'num_layers' in self.results:
            r = self.results['num_layers']
            fig, ax = plt.subplots(figsize=(10, 5))
            
            ax.plot(r['num_layers'], r['dice'], 'o-', label='Dice', linewidth=2)
            ax.plot(r['num_layers'], r['iou'], 's-', label='IoU', linewidth=2)
            ax.set_xlabel('Number of Layers')
            ax.set_ylabel('Metric Value')
            ax.set_title('Effect of Network Depth')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/ablation_num_layers.png', dpi=150)
            print(f"Saved: {output_dir}/ablation_num_layers.png")
            plt.close()
        
        # Plot 4: Learning rate
        if 'learning_rate' in self.results:
            r = self.results['learning_rate']
            fig, ax = plt.subplots(figsize=(10, 5))
            
            lrs = [f"{lr:.0e}" for lr in r['learning_rate']]
            x = np.arange(len(lrs))
            width = 0.35
            
            ax.bar(x - width/2, r['dice'], width, label='Dice')
            ax.bar(x + width/2, r['iou'], width, label='IoU')
            ax.set_xlabel('Learning Rate')
            ax.set_ylabel('Metric Value')
            ax.set_title('Effect of Learning Rate')
            ax.set_xticks(x)
            ax.set_xticklabels(lrs)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/ablation_learning_rate.png', dpi=150)
            print(f"Saved: {output_dir}/ablation_learning_rate.png")
            plt.close()


def main():
    """Run ablation studies"""
    print("="*60)
    print("ABLATION STUDIES")
    print("="*60)
    
    # Setup
    data_path = CONFIG['data_path']
    subject_dirs = sorted([d for d in (data_path / 'segmentations').glob('*')])[:3]  # First 3 subjects
    
    if not subject_dirs:
        print(f"ERROR: No subjects found at {data_path}")
        return
    
    subject_path = data_path / 'volumes' / subject_dirs[0].name
    
    # Run ablations
    ablations = AblationStudies(CONFIG.copy())
    
    ablations.ablation_num_views(subject_path)
    ablations.ablation_hidden_dimension(subject_path)
    ablations.ablation_num_layers(subject_path)
    ablations.ablation_learning_rate(subject_path)
    
    # Plot
    ablations.plot_results('./results')
    
    print("\n" + "="*60)
    print("ABLATION STUDIES COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
