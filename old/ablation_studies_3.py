#!/usr/bin/env python3
"""
ABLATION STUDIES: Test contribution of each component
Run after getting baseline to work

This version runs ablations on all 10 subjects from minimal_starter
to ensure statistical robustness and understand failure modes.

FIXES:
- JSON saving bug (PosixPath object has no attribute 'write')
- Dice scores all 0.0 (evaluate_subject device parameter)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import traceback
from collections import defaultdict

# Import components from your working minimal_starter.py
from minimal_starter_5 import (
    ImplicitNeuralRepresentation,
    load_mitea_subject,
    extract_synthetic_2d_slices,
    optimize_single_subject,
    evaluate_subject,
    find_mitea_image_files,
    CONFIG
)


# ============================================================================
# ABLATION EXPERIMENTS CLASS
# ============================================================================

class AblationStudies:
    def __init__(self, config):
        self.config = config
        self.results = {}
        self.output_dir = Path('./ablation_results')
        self.output_dir.mkdir(exist_ok=True)
        
        # For aggregating across subjects
        self.aggregated = defaultdict(lambda: defaultdict(list))
    
    def run_single_experiment(self, experiment_name, parameter_name, parameter_values, subject_pair):
        """Run one experiment (e.g., num_views) on one subject"""
        
        # Load subject data
        img_file, label_file = subject_pair
        vol, seg = load_mitea_subject(img_file, label_file)
        if vol is None:
            print(f"  Failed to load {img_file.name}")
            return None
        
        # Extract max views needed
        max_views = 6
        all_slices_2d, all_contours_2d = extract_synthetic_2d_slices(vol, seg, num_views=max_views)
        
        results = {parameter_name: [], 'dice': [], 'iou': [], 'params': []}
        
        for val in parameter_values:
            # Configure experiment
            current_config = self.config.copy()
            
            # Handle parameter overrides
            if parameter_name == 'num_views':
                num_views = val
                slices_2d = all_slices_2d[:num_views]
                contours_2d = all_contours_2d[:num_views]
            else:
                # For model/optim params, use full 6 views
                num_views = 6
                slices_2d = all_slices_2d
                contours_2d = all_contours_2d
                
                if parameter_name == 'hidden_dim':
                    current_config['hidden_dim'] = val
                elif parameter_name == 'num_layers':
                    current_config['num_inr_layers'] = val
                elif parameter_name == 'learning_rate':
                    current_config['learning_rate'] = val
            
            try:
                # Initialize Model
                model = ImplicitNeuralRepresentation(
                    hidden_dim=current_config['hidden_dim'],
                    num_layers=current_config['num_inr_layers']
                )
                
                # Count parameters
                num_params = sum(p.numel() for p in model.parameters())
                
                # Optimize
                model, losses = optimize_single_subject(
                    model,
                    slices_2d,
                    contours_2d,
                    current_config,
                    num_steps=50  # Reduced for ablation speed
                )
                
                # Evaluate
                device = torch.device(current_config['device'])
                with torch.no_grad():
                    occupancy_grid = model.sample_grid(resolution=64, device=device, requires_grad=False)
                    # FIX: Pass device parameter to evaluate_subject
                    metrics = evaluate_subject(model, occupancy_grid, contours_2d, device)
                
                results[parameter_name].append(val)
                results['dice'].append(metrics['dice'])
                results['iou'].append(metrics['iou'])
                results['params'].append(num_params)
                
                # Aggregate for cross-subject analysis
                self.aggregated[parameter_name][val].append(metrics['dice'])
                
            except Exception as e:
                print(f"    Error with {parameter_name}={val}: {str(e)[:50]}")
                results[parameter_name].append(val)
                results['dice'].append(0.0)
                results['iou'].append(0.0)
                results['params'].append(num_params)
        
        return results
    
    def run_all_experiments_on_subject(self, subject_pair, subject_id):
        """Run all ablation experiments on a single subject"""
        print(f"\n  Running ablations for {subject_id}...")
        
        subject_results = {}
        
        # 1. View Count
        print(f"    - View Count ablation...", end='', flush=True)
        result = self.run_single_experiment(
            "view_count", "num_views", [2, 3, 4, 6], subject_pair
        )
        if result:
            subject_results['num_views'] = result
        print(" ✓")
        
        # 2. Hidden Dimension
        print(f"    - Hidden Dim ablation...", end='', flush=True)
        result = self.run_single_experiment(
            "model_capacity", "hidden_dim", [32, 64, 128, 256], subject_pair
        )
        if result:
            subject_results['hidden_dim'] = result
        print(" ✓")
        
        # 3. Network Layers
        print(f"    - Network Depth ablation...", end='', flush=True)
        result = self.run_single_experiment(
            "network_depth", "num_layers", [2, 4, 6, 8], subject_pair
        )
        if result:
            subject_results['num_layers'] = result
        print(" ✓")
        
        # 4. Learning Rate
        print(f"    - Learning Rate ablation...", end='', flush=True)
        result = self.run_single_experiment(
            "learning_rate", "learning_rate", [1e-4, 5e-4, 1e-3, 5e-3], subject_pair
        )
        if result:
            subject_results['learning_rate'] = result
        print(" ✓")
        
        # Save per-subject results
        self._save_json(f"subject_{subject_id}", subject_results)
        
        return subject_results
    
    def _save_json(self, name, data):
        """Save results to JSON - FIX: Properly open file"""
        filepath = self.output_dir / f"{name}.json"
        try:
            # FIX: Open file first before passing to json.dump
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4, default=str)
        except Exception as e:
            print(f"\nWarning: Could not save {filepath}: {e}")
    
    def plot_aggregated_results(self):
        """Plot average results across all subjects"""
        print("\n" + "="*60)
        print("PLOTTING AGGREGATED RESULTS")
        print("="*60)
        
        # 1. View Count
        if 'num_views' in self.aggregated:
            self._plot_ablation(
                'num_views',
                'Number of Views',
                self.aggregated['num_views']
            )
        
        # 2. Hidden Dimension
        if 'hidden_dim' in self.aggregated:
            self._plot_ablation(
                'hidden_dim',
                'Hidden Dimension',
                self.aggregated['hidden_dim']
            )
        
        # 3. Network Layers
        if 'num_layers' in self.aggregated:
            self._plot_ablation(
                'num_layers',
                'Number of Layers',
                self.aggregated['num_layers']
            )
        
        # 4. Learning Rate
        if 'learning_rate' in self.aggregated:
            lr_dict = {}
            for lr, scores in self.aggregated['learning_rate'].items():
                lr_dict[f'{lr:.0e}'] = scores
            self._plot_ablation(
                'learning_rate',
                'Learning Rate',
                lr_dict,
                is_categorical=True
            )
    
    def _plot_ablation(self, param_name, param_label, data_dict, is_categorical=False):
        """Generic plotting for ablation results"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if is_categorical:
            # Categorical x-axis (learning rate)
            categories = list(data_dict.keys())
            means = [np.mean(scores) for scores in data_dict.values()]
            stds = [np.std(scores) for scores in data_dict.values()]
            
            x_pos = np.arange(len(categories))
            ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(categories, rotation=45)
            ax.set_xlabel(param_label)
        else:
            # Continuous x-axis
            sorted_items = sorted(data_dict.items(), key=lambda x: float(x[0]))
            x_vals = [float(k) for k, v in sorted_items]
            means = [np.mean(v) for k, v in sorted_items]
            stds = [np.std(v) for k, v in sorted_items]
            
            ax.errorbar(x_vals, means, yerr=stds, fmt='o-', capsize=5, 
                       linewidth=2, markersize=8, label='Mean ± Std', color='steelblue')
            ax.set_xlabel(param_label)
            ax.set_xscale('log' if param_name == 'learning_rate' else 'linear')
        
        ax.set_ylabel('Dice Coefficient')
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f'Ablation Study: Effect of {param_label}')
        
        plt.tight_layout()
        output_path = self.output_dir / f"ablation_{param_name}.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"✓ Saved: {output_path}")
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "="*60)
        print("ABLATION SUMMARY STATISTICS")
        print("="*60)
        
        for param_name, data in self.aggregated.items():
            print(f"\n{param_name.upper()}:")
            for val, scores in sorted(data.items()):
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                count = len(scores)
                print(f"  {val:>8}: Dice = {mean_score:.4f} ± {std_score:.4f} (N={count})")


# ============================================================================
# MAIN RUNNER
# ============================================================================

def main():
    """Run ablation studies on all 10 subjects from minimal_starter"""
    
    print("\n" + "="*60)
    print("MULTI-SUBJECT ABLATION STUDIES")
    print("="*60)
    
    data_path = CONFIG['data_path']
    print(f"Data Path: {data_path}")
    
    # 1. Get the 10 subjects (same as minimal_starter)
    print("\nFinding subjects...")
    subject_pairs = find_mitea_image_files(data_path)[:10]
    
    if not subject_pairs:
        print("ERROR: No subjects found!")
        return
    
    print(f"Found {len(subject_pairs)} subjects for ablation")
    
    # 2. Initialize Ablation Suite
    ablations = AblationStudies(CONFIG)
    
    # 3. Run experiments on ALL subjects
    print(f"\n{'='*60}")
    print("RUNNING EXPERIMENTS")
    print(f"{'='*60}")
    
    for i, subject_pair in enumerate(subject_pairs):
        subject_id = subject_pair[0].stem
        print(f"\n[{i+1}/{len(subject_pairs)}] {subject_id}")
        
        try:
            ablations.run_all_experiments_on_subject(subject_pair, subject_id)
        except Exception as e:
            print(f"ERROR on subject {subject_id}: {e}")
            traceback.print_exc()
            continue
    
    # 4. Plot aggregated results
    print(f"\n{'='*60}")
    print("VISUALIZATION")
    print(f"{'='*60}")
    ablations.plot_aggregated_results()
    
    # 5. Print summary
    ablations.print_summary()
    
    # 6. Save aggregated results
    agg_data = {
        k: {str(v_k): v_v for v_k, v_v in v.items()}
        for k, v in ablations.aggregated.items()
    }
    ablations._save_json("aggregated_results", agg_data)
    
    print(f"\n{'='*60}")
    print("✅ ABLATION STUDIES COMPLETE")
    print(f"Results saved in: {ablations.output_dir}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
