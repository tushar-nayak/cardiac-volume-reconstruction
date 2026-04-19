#!/usr/bin/env python3
"""
SPARSE VIEW 3D RECONSTRUCTION - COMPLETE FIXED VERSION
Full production-ready code with:
- 12-panel comprehensive visualization
- Fixed JSON serialization (PosixPath issue resolved)
- Better input validation
- Adaptive mesh extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import json
import nibabel as nib

from minimal_starter_5 import (
    load_mitea_subject,
    extract_synthetic_2d_slices,
    find_mitea_image_files,
    ImplicitNeuralRepresentation,
    contour_reprojection_loss,
    laplacian_smoothness_loss,
    CONFIG
)


# BEST_CONFIG = {
#     'device': CONFIG['device'],
#     'data_path': str(CONFIG['data_path']),  # Convert to string for JSON
#     'hidden_dim': 256,
#     'num_inr_layers': 4,
#     'learning_rate': 0.001,
#     'num_views': 6,
#     'num_optimization_steps': 400,
#     'grid_resolution': 128,
# }




BEST_CONFIG = {
    'device': CONFIG['device'],
    'data_path': CONFIG['data_path'],
    'hidden_dim': 32,           # Best from ablation
    'num_inr_layers': 2,          # Best from ablation
    'learning_rate': 0.0001,       # Best from ablation
    'num_views': 3,               # SPARSE! Still achieves 0.82 Dice
    'num_optimization_steps': 1000,
    'grid_resolution': 128,       # Higher resolution for reconstruction
}


class SparseViewReconstructor:
    """Reconstruct 3D cardiac geometry from sparse 2D views"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        self.output_dir = Path('./sparse_reconstruction_results')
        self.output_dir.mkdir(exist_ok=True)
    
    def reconstruct_subject(self, img_file, label_file, subject_id):
        """Reconstruct single subject"""
        
        print(f"\n{'='*60}")
        print(f"Reconstructing: {subject_id}")
        print(f"{'='*60}")
        
        # Load data
        vol, seg = load_mitea_subject(img_file, label_file)
        if vol is None:
            return None
        
        print(f"Loaded: vol {vol.shape}, seg {seg.shape}")
        
        # Extract views
        all_slices, all_contours = extract_synthetic_2d_slices(
            vol, seg, num_views=self.config['num_views']
        )
        
        print(f"Extracted {self.config['num_views']} views")
        
        # VALIDATION: Check contour coverage
        print(f"\nInput Validation:")
        coverage_list = []
        for i in range(all_contours.shape[0]):
            coverage = (all_contours[i] > 0.5).float().mean().item() * 100
            coverage_list.append(coverage)
            print(f"  View {i}: {coverage:.2f}% coverage")
        
        # Initialize model
        model = ImplicitNeuralRepresentation(
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_inr_layers']
        ).to(self.device)
        
        # Optimize
        print(f"\nOptimizing INR with {self.config['num_optimization_steps']} steps...")
        model, losses = self._optimize(model, all_slices, all_contours)
        
        # Sample grid
        print(f"Sampling 3D occupancy grid at resolution {self.config['grid_resolution']}...")
        with torch.no_grad():
            occupancy_grid = model.sample_grid(
                resolution=self.config['grid_resolution'],
                device=self.device,
                requires_grad=False
            )
        
        # Analyze occupancy distribution
        occ_np = occupancy_grid.cpu().numpy()
        occ_stats = {
            'min': float(occ_np.min()),
            'max': float(occ_np.max()),
            'mean': float(occ_np.mean()),
            'std': float(occ_np.std()),
            'median': float(np.median(occ_np))
        }
        
        print(f"\nOccupancy Statistics:")
        for k, v in occ_stats.items():
            print(f"  {k}: {v:.4f}")
        
        # Extract mesh with adaptive threshold
        vertices, faces = self._extract_mesh(occupancy_grid, occ_stats)
        
        # Visualize
        self._visualize(
            subject_id, occupancy_grid, all_slices, all_contours,
            vertices, faces, losses, coverage_list, occ_stats
        )
        
        # Save
        self._save_results(subject_id, occupancy_grid, vertices, faces, occ_stats)
        
        return {
            'subject_id': subject_id,
            'grid': occupancy_grid,
            'vertices': vertices,
            'faces': faces,
            'losses': losses
        }
    
    def _optimize(self, model, slices_2d, contours_2d):
        """Optimize INR"""
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        losses = []
        
        for step in range(self.config['num_optimization_steps']):
            optimizer.zero_grad()
            
            # Sample grid WITH gradients
            occupancy_grid = model.sample_grid(
                resolution=64,
                device=self.device,
                requires_grad=True
            )
            
            # Multi-view loss
            loss = 0.0
            contours_device = contours_2d.to(self.device)
            for view_idx in range(contours_2d.shape[0]):
                loss += contour_reprojection_loss(occupancy_grid, contours_device[view_idx])
            
            # Smoothness
            loss += laplacian_smoothness_loss(occupancy_grid, weight=0.01)
            
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if step % 50 == 0:
                print(f"  Step {step}/{self.config['num_optimization_steps']}: loss={loss.item():.6f}")
        
        return model, losses
    
    def _extract_mesh(self, occupancy_grid, occ_stats):
        from skimage import measure
        from scipy.ndimage import label, sum as ndi_sum, gaussian_filter1d
        
        occ_np = occupancy_grid.cpu().numpy()
        
        # STEP 1: Find bimodal threshold (not mean!)
        hist, bins = np.histogram(occ_np.flatten(), bins=256)
        hist_smooth = gaussian_filter1d(hist, sigma=2)
        valley_idx = np.argmin(hist_smooth[64:192]) + 64  # Find valley between peaks
        threshold = bins[valley_idx]
        
        print(f"  Bimodal threshold: {threshold:.4f} (was using mean={occ_stats['mean']:.4f})")
        
        try:
            # STEP 2: Extract mesh
            vertices, faces, _, _ = measure.marching_cubes(occ_np, level=threshold, step_size=1)
            
            if len(vertices) == 0:
                return None, None
            
            # STEP 3: Filter noise (keep only largest connected component)
            binary_vol = (occ_np > threshold).astype(np.uint8)
            labeled, num_features = label(binary_vol)
            
            if num_features > 1:
                print(f"  Found {num_features} components, keeping largest...")
                component_sizes = ndi_sum(binary_vol, labeled, range(num_features + 1))
                largest_label = np.argmax(component_sizes[1:]) + 1
                largest_vol = (labeled == largest_label).astype(np.uint8)
                
                vertices, faces, _, _ = measure.marching_cubes(
                    largest_vol.astype(float), 
                    level=0.5
                )
            
            return vertices, faces
        
        except ValueError:
            return None, None

    def _visualize(self, subject_id, occupancy_grid, slices_2d, contours_2d,
                   vertices, faces, losses, coverage_list, occ_stats):
        """12-panel comprehensive visualization"""
        
        fig = plt.figure(figsize=(24, 16))
        
        # 1. Loss curve
        ax1 = plt.subplot(3, 4, 1)
        ax1.plot(losses, linewidth=2.5, color='steelblue')
        ax1.fill_between(range(len(losses)), losses, alpha=0.3, color='steelblue')
        ax1.set_xlabel('Optimization Step', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax1.set_title('Training Loss Curve', fontsize=13, fontweight='bold', pad=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_facecolor('#f8f9fa')
        
        # 2-4. Input views with contours
        for i in range(min(3, slices_2d.shape[0])):
            ax = plt.subplot(3, 4, i+2)
            ax.imshow(slices_2d[i].cpu().numpy(), cmap='gray', alpha=0.8)
            
            # Overlay contour
            contour_data = contours_2d[i].cpu().numpy()
            ax.contour(contour_data, colors='red', linewidths=2.5, levels=[0.5])
            ax.contourf(contour_data, colors=['red'], alpha=0.15, levels=[0.5, 1.0])
            
            ax.set_title(f'Input View {i+1}\n({coverage_list[i]:.1f}% coverage)', 
                        fontsize=12, fontweight='bold', pad=8)
            ax.axis('off')
            ax.set_facecolor('#f0f0f0')
        
        # 5. Occupancy histogram
        ax5 = plt.subplot(3, 4, 5)
        occ_np = occupancy_grid.cpu().numpy()
        ax5.hist(occ_np.flatten(), bins=60, color='steelblue', alpha=0.7, edgecolor='navy')
        ax5.axvline(occ_stats['mean'], color='red', linestyle='--', linewidth=2.5, 
                   label=f"Mean: {occ_stats['mean']:.3f}")
        ax5.axvline(occ_stats['median'], color='green', linestyle='--', linewidth=2.5, 
                   label=f"Median: {occ_stats['median']:.3f}")
        ax5.set_xlabel('Occupancy Value', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax5.set_title('Occupancy Distribution', fontsize=13, fontweight='bold', pad=10)
        ax5.legend(fontsize=10, loc='upper left')
        ax5.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax5.set_facecolor('#f8f9fa')
        
        # 6. 3D point cloud
        ax6 = fig.add_subplot(3, 4, 6, projection='3d')
        threshold = occ_stats['mean']
        z, x, y = np.where(occ_np > threshold)
        stride = max(1, len(x) // 5000)
        if len(x) > 0:
            colors = occ_np[z[::stride], x[::stride], y[::stride]]
            scatter = ax6.scatter(x[::stride], y[::stride], z[::stride], c=colors,
                       cmap='hot', s=3, alpha=0.7, edgecolors='none')
            plt.colorbar(scatter, ax=ax6, label='Occupancy', shrink=0.7)
        ax6.set_xlabel('X', fontsize=10, fontweight='bold')
        ax6.set_ylabel('Y', fontsize=10, fontweight='bold')
        ax6.set_zlabel('Z', fontsize=10, fontweight='bold')
        ax6.set_title(f'3D Cloud (Occ > {threshold:.3f})', fontsize=12, fontweight='bold', pad=10)
        ax6.view_init(elev=20, azim=45)
        
        # 7. Mid-axial slice
        ax7 = plt.subplot(3, 4, 7)
        mid_slice = occ_np[occ_np.shape[0]//2, :, :]
        im7 = ax7.imshow(mid_slice, cmap='hot', origin='lower')
        ax7.set_title('Mid-Axial Slice', fontsize=12, fontweight='bold', pad=8)
        ax7.set_xlabel('Y', fontsize=10)
        ax7.set_ylabel('X', fontsize=10)
        cbar7 = plt.colorbar(im7, ax=ax7, label='Occupancy')
        
        # 8. Mid-coronal slice
        ax8 = plt.subplot(3, 4, 8)
        mid_slice = occ_np[:, occ_np.shape[1]//2, :]
        im8 = ax8.imshow(mid_slice, cmap='hot', origin='lower')
        ax8.set_title('Mid-Coronal Slice', fontsize=12, fontweight='bold', pad=8)
        ax8.set_xlabel('Z', fontsize=10)
        ax8.set_ylabel('Z', fontsize=10)
        cbar8 = plt.colorbar(im8, ax=ax8, label='Occupancy')
        
        # 9. Mid-sagittal slice
        ax9 = plt.subplot(3, 4, 9)
        mid_slice = occ_np[:, :, occ_np.shape[2]//2]
        im9 = ax9.imshow(mid_slice, cmap='hot', origin='lower')
        ax9.set_title('Mid-Sagittal Slice', fontsize=12, fontweight='bold', pad=8)
        ax9.set_xlabel('X', fontsize=10)
        ax9.set_ylabel('Z', fontsize=10)
        cbar9 = plt.colorbar(im9, ax=ax9, label='Occupancy')
        
        # 10. Max projection (XY)
        ax10 = plt.subplot(3, 4, 10)
        proj_xy = np.max(occ_np, axis=0)
        im10 = ax10.imshow(proj_xy, cmap='hot', origin='lower')
        ax10.set_title('Max-Projection (XY)', fontsize=12, fontweight='bold', pad=8)
        ax10.set_xlabel('Y', fontsize=10)
        ax10.set_ylabel('X', fontsize=10)
        cbar10 = plt.colorbar(im10, ax=ax10, label='Occupancy')
        
        # 11. Max projection (XZ)
        ax11 = plt.subplot(3, 4, 11)
        proj_xz = np.max(occ_np, axis=1)
        im11 = ax11.imshow(proj_xz, cmap='hot', origin='lower')
        ax11.set_title('Max-Projection (XZ)', fontsize=12, fontweight='bold', pad=8)
        ax11.set_xlabel('Z', fontsize=10)
        ax11.set_ylabel('Z', fontsize=10)
        cbar11 = plt.colorbar(im11, ax=ax11, label='Occupancy')
        
        # 12. Max projection (YZ)
        ax12 = plt.subplot(3, 4, 12)
        proj_yz = np.max(occ_np, axis=2)
        im12 = ax12.imshow(proj_yz, cmap='hot', origin='lower')
        ax12.set_title('Max-Projection (YZ)', fontsize=12, fontweight='bold', pad=8)
        ax12.set_xlabel('Y', fontsize=10)
        ax12.set_ylabel('Z', fontsize=10)
        cbar12 = plt.colorbar(im12, ax=ax12, label='Occupancy')
        
        # Overall title
        plt.suptitle(f'Sparse View 3D Reconstruction: {subject_id}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        out_path = self.output_dir / f'{subject_id}_reconstruction.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ Saved 12-panel visualization: {out_path}")
    
    def _save_results(self, subject_id, occupancy_grid, vertices, faces, occ_stats):
        """Save reconstruction results"""
        
        occ_np = occupancy_grid.cpu().numpy().astype(np.float32)
        nii = nib.Nifti1Image(occ_np, np.eye(4))
        out_nii = self.output_dir / f'{subject_id}_occupancy_grid.nii.gz'
        nib.save(nii, out_nii)
        print(f"✓ Saved occupancy grid: {out_nii}")
        
        if vertices is not None:
            out_vertices = self.output_dir / f'{subject_id}_vertices.npy'
            np.save(out_vertices, vertices)
            print(f"✓ Saved vertices ({len(vertices)} points): {out_vertices}")
            
            if faces is not None:
                out_faces = self.output_dir / f'{subject_id}_faces.npy'
                np.save(out_faces, faces)
                print(f"✓ Saved faces ({len(faces)} triangles): {out_faces}")
        
        # FIX: Convert all config values to native Python types for JSON serialization
        config_serializable = {
            'hidden_dim': int(self.config['hidden_dim']),
            'num_inr_layers': int(self.config['num_inr_layers']),
            'learning_rate': float(self.config['learning_rate']),
            'num_views': int(self.config['num_views']),
            'num_optimization_steps': int(self.config['num_optimization_steps']),
            'grid_resolution': int(self.config['grid_resolution']),
            'device': str(self.config['device']),
        }
        
        metadata = {
            'subject_id': subject_id,
            'grid_shape': list(occ_np.shape),
            'grid_stats': occ_stats,
            'num_vertices': int(len(vertices)) if vertices is not None else 0,
            'num_faces': int(len(faces)) if faces is not None else 0,
            'config': config_serializable
        }
        
        out_meta = self.output_dir / f'{subject_id}_metadata.json'
        with open(out_meta, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"✓ Saved metadata: {out_meta}")


def main():
    print("="*60)
    print("SPARSE VIEW 3D RECONSTRUCTION - COMPLETE VERSION")
    print("="*60)
    print(f"\nConfiguration:")
    for k, v in BEST_CONFIG.items():
        if k not in ['device', 'data_path']:
            print(f"  {k}: {v}")
    
    subjects = find_mitea_image_files(Path(BEST_CONFIG['data_path']))[:5]
    
    if not subjects:
        print("ERROR: No subjects found")
        return
    
    print(f"\nFound {len(subjects)} subjects\n")
    
    reconstructor = SparseViewReconstructor(BEST_CONFIG)
    results = []
    
    for i, (img_file, label_file) in enumerate(subjects):
        subject_id = img_file.stem
        print(f"\n[{i+1}/{len(subjects)}]")
        
        try:
            result = reconstructor.reconstruct_subject(img_file, label_file, subject_id)
            if result:
                results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("RECONSTRUCTION SUMMARY")
    print("="*60)
    print(f"Successfully reconstructed: {len(results)}/{len(subjects)} subjects")
    print(f"Results saved in: {reconstructor.output_dir}/")
    print(f"\nOutput files per subject:")
    print(f"  - *_reconstruction.png: 12-panel visualization")
    print(f"  - *_occupancy_grid.nii.gz: 3D volume (importable in medical viewers)")
    print(f"  - *_vertices.npy: Mesh vertices")
    print(f"  - *_faces.npy: Mesh faces")
    print(f"  - *_metadata.json: Configuration & statistics")
    print("="*60)


if __name__ == '__main__':
    main()
