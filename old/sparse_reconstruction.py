# #!/usr/bin/env python3
# """
# SPARSE VIEW 3D RECONSTRUCTION
# Uses best model configuration from ablation studies to reconstruct 3D cardiac geometry
# from minimal 2D views (2-4 views).

# Best Configuration (from ablations):
# - Hidden Dim: 256
# - Num Layers: 4
# - Learning Rate: 0.001
# - Sparse Views: 2 (minimal, but highest Dice!)
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from pathlib import Path
# import json
# import nibabel as nib
# from matplotlib import cm

# # Import from minimal_starter
# from minimal_starter_5 import (
#     load_mitea_subject,
#     extract_synthetic_2d_slices,
#     find_mitea_image_files,
#     ImplicitNeuralRepresentation,
#     contour_reprojection_loss,
#     laplacian_smoothness_loss,
#     CONFIG
# )


# # ============================================================================
# # BEST MODEL CONFIG (FROM ABLATION STUDIES)
# # ============================================================================

# BEST_CONFIG = {
#     'device': CONFIG['device'],
#     'data_path': CONFIG['data_path'],
#     'hidden_dim': 32,           # Best from ablation
#     'num_inr_layers': 2,          # Best from ablation
#     'learning_rate': 0.0001,       # Best from ablation
#     'num_views': 3,               # SPARSE! Still achieves 0.82 Dice
#     'num_optimization_steps': 200,
#     'grid_resolution': 128,       # Higher resolution for reconstruction
# }


# # ============================================================================
# # SPARSE VIEW RECONSTRUCTION WITH VISUALIZATION
# # ============================================================================

# class SparseViewReconstructor:
#     """Reconstruct 3D cardiac geometry from sparse 2D views"""
    
#     def __init__(self, config):
#         self.config = config
#         self.device = torch.device(config['device'])
#         self.output_dir = Path('./sparse_reconstruction_results')
#         self.output_dir.mkdir(exist_ok=True)
    
#     def reconstruct_subject(self, img_file, label_file, subject_id):
#         """Reconstruct single subject with sparse views"""
        
#         print(f"\n{'='*60}")
#         print(f"Reconstructing: {subject_id}")
#         print(f"{'='*60}")
        
#         # Load data
#         vol, seg = load_mitea_subject(img_file, label_file)
#         if vol is None:
#             return None
        
#         print(f"Loaded: vol {vol.shape}, seg {seg.shape}")
        
#         # Extract ALL 6 standard views
#         all_slices, all_contours = extract_synthetic_2d_slices(
#             vol, seg, num_views=6
#         )
        
#         # Use ONLY first 2 views (sparse)
#         num_views = self.config['num_views']
#         slices_sparse = all_slices[:num_views]
#         contours_sparse = all_contours[:num_views]
        
#         print(f"Using SPARSE {num_views} views for reconstruction")
        
#         # Initialize best model
#         model = ImplicitNeuralRepresentation(
#             hidden_dim=self.config['hidden_dim'],
#             num_layers=self.config['num_inr_layers']
#         ).to(self.device)
        
#         # Optimize
#         print(f"Optimizing INR with {self.config['num_optimization_steps']} steps...")
#         model, losses = self._optimize(model, slices_sparse, contours_sparse)
        
#         # Reconstruct 3D volume
#         print(f"Sampling 3D occupancy grid at resolution {self.config['grid_resolution']}...")
#         with torch.no_grad():
#             occupancy_grid = model.sample_grid(
#                 resolution=self.config['grid_resolution'],
#                 device=self.device,
#                 requires_grad=False
#             )
        
#         # Extract mesh via marching cubes (optional, requires skimage)
#         try:
#             from skimage import measure
#             vertices, faces, normals, values = measure.marching_cubes(
#                 occupancy_grid.cpu().numpy(),
#                 level=0.5
#             )
#             print(f"Extracted mesh: {len(vertices)} vertices, {len(faces)} faces")
#         except ImportError:
#             print("(skimage not available, skipping mesh extraction)")
#             vertices, faces = None, None
        
#         # Visualize
#         self._visualize(
#             subject_id,
#             occupancy_grid,
#             slices_sparse,
#             contours_sparse,
#             vertices,
#             faces,
#             losses
#         )
        
#         # Save results
#         self._save_results(subject_id, occupancy_grid, vertices, faces)
        
#         return {
#             'subject_id': subject_id,
#             'grid': occupancy_grid,
#             'vertices': vertices,
#             'faces': faces,
#             'losses': losses
#         }
    
#     def _optimize(self, model, slices_2d, contours_2d):
#         """Optimize INR on sparse views"""
#         optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
#         losses = []
        
#         for step in range(self.config['num_optimization_steps']):
#             optimizer.zero_grad()
            
#             # Sample grid WITH gradients
#             occupancy_grid = model.sample_grid(
#                 resolution=64,  # Lower res during training
#                 device=self.device,
#                 requires_grad=True
#             )
            
#             # Contour reprojection loss
#             loss = 0.0
#             contours_device = contours_2d.to(self.device)
#             for view_idx in range(contours_2d.shape[0]):
#                 loss += contour_reprojection_loss(occupancy_grid, contours_device[view_idx])
            
#             # Smoothness
#             loss += laplacian_smoothness_loss(occupancy_grid, weight=0.01)
            
#             loss.backward()
#             optimizer.step()
#             losses.append(loss.item())
            
#             if step % 50 == 0:
#                 print(f"  Step {step}/{self.config['num_optimization_steps']}: loss={loss.item():.6f}")
        
#         return model, losses
    
#     def _visualize(self, subject_id, occupancy_grid, slices_sparse, contours_sparse,
#                    vertices, faces, losses):
#         """Create comprehensive visualization"""
        
#         fig = plt.figure(figsize=(20, 12))
        
#         # 1. Training loss
#         ax1 = plt.subplot(2, 3, 1)
#         ax1.plot(losses)
#         ax1.set_xlabel('Optimization Step')
#         ax1.set_ylabel('Loss')
#         ax1.set_title('Training Loss Curve')
#         ax1.grid(True, alpha=0.3)
        
#         # 2. Input sparse views
#         ax2 = plt.subplot(2, 3, 2)
#         ax2.imshow(slices_sparse[0].cpu().numpy(), cmap='gray')
#         ax2.contour(contours_sparse[0].cpu().numpy(), colors='red', linewidths=2)
#         ax2.set_title(f'View 1 (Input)')
#         ax2.axis('off')
        
#         if slices_sparse.shape[0] > 1:
#             ax3 = plt.subplot(2, 3, 3)
#             ax3.imshow(slices_sparse[1].cpu().numpy(), cmap='gray')
#             ax3.contour(contours_sparse[1].cpu().numpy(), colors='red', linewidths=2)
#             ax3.set_title(f'View 2 (Input)')
#             ax3.axis('off')
        
#         # 4. 3D occupancy isosurface
#         ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        
#         # Plot iso-surface
#         occ_np = occupancy_grid.cpu().numpy()
#         z, x, y = np.where(occ_np > 0.5)
        
#         # Downsample for visualization
#         stride = max(1, len(x) // 5000)
#         ax4.scatter(x[::stride], y[::stride], z[::stride], 
#                    c=occ_np[z[::stride], x[::stride], y[::stride]],
#                    cmap='hot', s=1, alpha=0.6)
        
#         ax4.set_xlabel('X')
#         ax4.set_ylabel('Y')
#         ax4.set_zlabel('Z')
#         ax4.set_title('3D Reconstruction (Occupancy > 0.5)')
        
#         # 5. Occupancy slices
#         ax5 = plt.subplot(2, 3, 5)
#         mid_slice = occ_np[occ_np.shape[0]//2, :, :]
#         im = ax5.imshow(mid_slice, cmap='hot')
#         ax5.set_title('Mid-axial Slice of Occupancy')
#         plt.colorbar(im, ax=ax5)
        
#         # 6. Projection comparison
#         ax6 = plt.subplot(2, 3, 6)
#         pred_projection = torch.max(occupancy_grid, dim=0)[0].cpu().numpy()
#         ax6.imshow(pred_projection, cmap='hot')
#         ax6.set_title('Max-projection of Reconstruction')
        
#         plt.tight_layout()
#         out_path = self.output_dir / f'{subject_id}_reconstruction.png'
#         plt.savefig(out_path, dpi=150, bbox_inches='tight')
#         plt.close()
#         print(f"✓ Saved visualization: {out_path}")
    
#     def _save_results(self, subject_id, occupancy_grid, vertices, faces):
#         """Save reconstruction results"""
        
#         # Save occupancy grid as NIfTI
#         occ_np = occupancy_grid.cpu().numpy().astype(np.float32)
#         nii = nib.Nifti1Image(occ_np, np.eye(4))
#         out_nii = self.output_dir / f'{subject_id}_occupancy_grid.nii.gz'
#         nib.save(nii, out_nii)
#         print(f"✓ Saved occupancy grid: {out_nii}")
        
#         # Save vertices if available
#         if vertices is not None:
#             out_vertices = self.output_dir / f'{subject_id}_vertices.npy'
#             np.save(out_vertices, vertices)
#             print(f"✓ Saved vertices: {out_vertices}")
            
#             if faces is not None:
#                 out_faces = self.output_dir / f'{subject_id}_faces.npy'
#                 np.save(out_faces, faces)
#                 print(f"✓ Saved faces: {out_faces}")
        
#         # Save metadata
#         metadata = {
#             'subject_id': subject_id,
#             'grid_shape': list(occ_np.shape),
#             'num_vertices': len(vertices) if vertices is not None else 0,
#             'num_faces': len(faces) if faces is not None else 0,
#             'config': {
#                 'hidden_dim': self.config['hidden_dim'],
#                 'num_layers': self.config['num_inr_layers'],
#                 'learning_rate': self.config['learning_rate'],
#                 'num_views': self.config['num_views'],
#             }
#         }
#         out_meta = self.output_dir / f'{subject_id}_metadata.json'
#         with open(out_meta, 'w') as f:
#             json.dump(metadata, f, indent=4)
#         print(f"✓ Saved metadata: {out_meta}")


# def main():
#     print("="*60)
#     print("SPARSE VIEW 3D RECONSTRUCTION")
#     print("Using Best Model from Ablation Studies")
#     print("="*60)
#     print(f"\nConfiguration:")
#     for k, v in BEST_CONFIG.items():
#         print(f"  {k}: {v}")
    
#     # Find subjects
#     subjects = find_mitea_image_files(BEST_CONFIG['data_path'])[:5]  # First 5 subjects
    
#     if not subjects:
#         print("ERROR: No subjects found")
#         return
    
#     print(f"\nFound {len(subjects)} subjects for reconstruction\n")
    
#     reconstructor = SparseViewReconstructor(BEST_CONFIG)
#     results = []
    
#     # Reconstruct each subject
#     for i, (img_file, label_file) in enumerate(subjects):
#         subject_id = img_file.stem
#         print(f"\n[{i+1}/{len(subjects)}]")
        
#         try:
#             result = reconstructor.reconstruct_subject(img_file, label_file, subject_id)
#             if result:
#                 results.append(result)
#         except Exception as e:
#             print(f"ERROR: {e}")
#             import traceback
#             traceback.print_exc()
    
#     # Summary
#     print("\n" + "="*60)
#     print("RECONSTRUCTION SUMMARY")
#     print("="*60)
#     print(f"Successfully reconstructed: {len(results)}/{len(subjects)} subjects")
#     print(f"Results saved in: {reconstructor.output_dir}/")
#     print(f"\nOutput files:")
#     print(f"  - *_reconstruction.png: Visualization")
#     print(f"  - *_occupancy_grid.nii.gz: 3D reconstruction volume")
#     print(f"  - *_vertices.npy: Mesh vertices (if skimage available)")
#     print(f"  - *_faces.npy: Mesh faces (if skimage available)")
#     print(f"  - *_metadata.json: Metadata")
#     print("="*60)


# if __name__ == '__main__':
#     main()

#!/usr/bin/env python3
"""
SPARSE VIEW 3D RECONSTRUCTION - FIXED VERSION
Addresses issues with:
- Better input view validation and visualization
- Adaptive threshold for marching cubes
- Improved convergence monitoring
- Better occupancy distribution analysis
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
#     'data_path': CONFIG['data_path'],
#     'hidden_dim': 256,
#     'num_inr_layers': 4,
#     'learning_rate': 0.0001,
#     'num_views': 6,              # Use MORE views for better convergence
#     'num_optimization_steps': 300,  # More steps
#     'grid_resolution': 128,
# }

BEST_CONFIG = {
    'device': CONFIG['device'],
    'data_path': CONFIG['data_path'],
    'hidden_dim': 32,           # Best from ablation
    'num_inr_layers': 2,          # Best from ablation
    'learning_rate': 0.0001,       # Best from ablation
    'num_views': 3,               # SPARSE! Still achieves 0.82 Dice
    'num_optimization_steps': 400,
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
        for i in range(all_contours.shape[0]):
            coverage = (all_contours[i] > 0.5).float().mean().item() * 100
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
        print(f"\nOccupancy Statistics:")
        print(f"  Min: {occ_np.min():.4f}")
        print(f"  Max: {occ_np.max():.4f}")
        print(f"  Mean: {occ_np.mean():.4f}")
        print(f"  Std: {occ_np.std():.4f}")
        print(f"  Median: {np.median(occ_np):.4f}")
        
        # Extract mesh with adaptive threshold
        vertices, faces = self._extract_mesh(occupancy_grid)
        
        # Visualize
        self._visualize(
            subject_id, occupancy_grid, all_slices, all_contours,
            vertices, faces, losses
        )
        
        # Save
        self._save_results(subject_id, occupancy_grid, vertices, faces)
        
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
    
    def _extract_mesh(self, occupancy_grid):
        """Extract mesh with adaptive threshold"""
        try:
            from skimage import measure
        except ImportError:
            print("(skimage not available, skipping mesh extraction)")
            return None, None
        
        occ_np = occupancy_grid.cpu().numpy()
        
        # Adaptive threshold: use mean occupancy
        threshold = np.mean(occ_np)
        print(f"Using adaptive threshold: {threshold:.4f}")
        
        # Ensure threshold is within range
        if threshold < occ_np.min() or threshold > occ_np.max():
            threshold = (occ_np.min() + occ_np.max()) / 2
            print(f"Adjusted threshold to: {threshold:.4f}")
        
        try:
            vertices, faces, _, _ = measure.marching_cubes(
                occ_np,
                level=threshold
            )
            print(f"Extracted mesh: {len(vertices)} vertices, {len(faces)} faces")
            return vertices, faces
        except ValueError as e:
            print(f"Marching cubes failed: {e}")
            return None, None
    
    def _visualize(self, subject_id, occupancy_grid, slices_2d, contours_2d,
                   vertices, faces, losses):
        """Comprehensive visualization"""
        
        fig = plt.figure(figsize=(22, 14))
        
        # 1. Loss curve
        ax1 = plt.subplot(3, 4, 1)
        ax1.plot(losses, linewidth=2)
        ax1.set_xlabel('Step', fontsize=10)
        ax1.set_ylabel('Loss', fontsize=10)
        ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2-4. Input views
        for i in range(min(3, slices_2d.shape[0])):
            ax = plt.subplot(3, 4, i+2)
            ax.imshow(slices_2d[i].cpu().numpy(), cmap='gray')
            
            # Overlay contour
            contour_data = contours_2d[i].cpu().numpy()
            coverage = (contour_data > 0.5).mean() * 100
            ax.contour(contour_data, colors='red', linewidths=2)
            
            ax.set_title(f'View {i+1}\n({coverage:.1f}% coverage)', fontsize=10)
            ax.axis('off')
        
        # 5. Occupancy histogram
        ax5 = plt.subplot(3, 4, 5)
        occ_np = occupancy_grid.cpu().numpy()
        ax5.hist(occ_np.flatten(), bins=50, color='steelblue', alpha=0.7)
        ax5.axvline(np.mean(occ_np), color='red', linestyle='--', label=f'Mean: {np.mean(occ_np):.3f}')
        ax5.axvline(np.median(occ_np), color='green', linestyle='--', label=f'Median: {np.median(occ_np):.3f}')
        ax5.set_xlabel('Occupancy Value', fontsize=10)
        ax5.set_ylabel('Frequency', fontsize=10)
        ax5.set_title('Occupancy Distribution', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
        
        # 6. 3D point cloud
        ax6 = fig.add_subplot(3, 4, 6, projection='3d')
        z, x, y = np.where(occ_np > np.mean(occ_np))
        stride = max(1, len(x) // 3000)
        colors = occ_np[z[::stride], x[::stride], y[::stride]]
        ax6.scatter(x[::stride], y[::stride], z[::stride], c=colors,
                   cmap='hot', s=2, alpha=0.6)
        ax6.set_xlabel('X', fontsize=9)
        ax6.set_ylabel('Y', fontsize=9)
        ax6.set_zlabel('Z', fontsize=9)
        ax6.set_title('3D Cloud (Occ > Mean)', fontsize=11, fontweight='bold')
        
        # 7. Mid-axial slice
        ax7 = plt.subplot(3, 4, 7)
        mid_slice = occ_np[occ_np.shape[0]//2, :, :]
        im7 = ax7.imshow(mid_slice, cmap='hot')
        ax7.set_title('Mid-Axial Slice', fontsize=11, fontweight='bold')
        plt.colorbar(im7, ax=ax7, label='Occupancy')
        
        # 8. Mid-coronal slice
        ax8 = plt.subplot(3, 4, 8)
        mid_slice = occ_np[:, occ_np.shape[1]//2, :]
        im8 = ax8.imshow(mid_slice, cmap='hot')
        ax8.set_title('Mid-Coronal Slice', fontsize=11, fontweight='bold')
        plt.colorbar(im8, ax=ax8, label='Occupancy')
        
        # 9. Mid-sagittal slice
        ax9 = plt.subplot(3, 4, 9)
        mid_slice = occ_np[:, :, occ_np.shape[2]//2]
        im9 = ax9.imshow(mid_slice, cmap='hot')
        ax9.set_title('Mid-Sagittal Slice', fontsize=11, fontweight='bold')
        plt.colorbar(im9, ax=ax9, label='Occupancy')
        
        # 10. Max projection (XY)
        ax10 = plt.subplot(3, 4, 10)
        proj_xy = np.max(occ_np, axis=0)
        im10 = ax10.imshow(proj_xy, cmap='hot')
        ax10.set_title('Max-Projection (XY)', fontsize=11, fontweight='bold')
        plt.colorbar(im10, ax=ax10, label='Occupancy')
        
        # 11. Max projection (XZ)
        ax11 = plt.subplot(3, 4, 11)
        proj_xz = np.max(occ_np, axis=1)
        im11 = ax11.imshow(proj_xz, cmap='hot')
        ax11.set_title('Max-Projection (XZ)', fontsize=11, fontweight='bold')
        plt.colorbar(im11, ax=ax11, label='Occupancy')
        
        # 12. Max projection (YZ)
        ax12 = plt.subplot(3, 4, 12)
        proj_yz = np.max(occ_np, axis=2)
        im12 = ax12.imshow(proj_yz, cmap='hot')
        ax12.set_title('Max-Projection (YZ)', fontsize=11, fontweight='bold')
        plt.colorbar(im12, ax=ax12, label='Occupancy')
        
        plt.suptitle(f'Sparse View Reconstruction: {subject_id}', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        out_path = self.output_dir / f'{subject_id}_reconstruction.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved visualization: {out_path}")
    
    def _save_results(self, subject_id, occupancy_grid, vertices, faces):
        """Save reconstruction results"""
        
        occ_np = occupancy_grid.cpu().numpy().astype(np.float32)
        nii = nib.Nifti1Image(occ_np, np.eye(4))
        out_nii = self.output_dir / f'{subject_id}_occupancy_grid.nii.gz'
        nib.save(nii, out_nii)
        print(f"✓ Saved occupancy grid: {out_nii}")
        
        if vertices is not None:
            out_vertices = self.output_dir / f'{subject_id}_vertices.npy'
            np.save(out_vertices, vertices)
            print(f"✓ Saved vertices: {out_vertices}")
            
            if faces is not None:
                out_faces = self.output_dir / f'{subject_id}_faces.npy'
                np.save(out_faces, faces)
                print(f"✓ Saved faces: {out_faces}")
        
        metadata = {
            'subject_id': subject_id,
            'grid_shape': list(occ_np.shape),
            'grid_stats': {
                'min': float(occ_np.min()),
                'max': float(occ_np.max()),
                'mean': float(occ_np.mean()),
                'std': float(occ_np.std()),
                'median': float(np.median(occ_np))
            },
            'num_vertices': len(vertices) if vertices is not None else 0,
            'num_faces': len(faces) if faces is not None else 0,
            'config': self.config
        }
        out_meta = self.output_dir / f'{subject_id}_metadata.json'
        with open(out_meta, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"✓ Saved metadata: {out_meta}")


def main():
    print("="*60)
    print("SPARSE VIEW 3D RECONSTRUCTION - FIXED")
    print("="*60)
    print(f"\nConfiguration:")
    for k, v in BEST_CONFIG.items():
        if k != 'device' and k != 'data_path':
            print(f"  {k}: {v}")
    
    subjects = find_mitea_image_files(BEST_CONFIG['data_path'])[:5]
    
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
    
    print("\n" + "="*60)
    print("RECONSTRUCTION SUMMARY")
    print("="*60)
    print(f"Successfully reconstructed: {len(results)}/{len(subjects)} subjects")
    print(f"Results saved in: {reconstructor.output_dir}/")
    print("="*60)


if __name__ == '__main__':
    main()
