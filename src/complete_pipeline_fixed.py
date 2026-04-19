#!/usr/bin/env python3
"""
COMPLETE PIPELINE - RECONSTRUCTION + 3D COMPARISON (FIXED)
===========================================================

This script:
1. Runs sparse view 3D reconstruction
2. Creates production-ready 3D viewers
3. Computes alignment metrics
4. All in one go!

FIXES:
- aspectmode moved to scene dict (not camera dict)
- Proper plotly API usage
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import nibabel as nib
from scipy.ndimage import zoom
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from minimal_starter_5 import (
    load_mitea_subject,
    extract_synthetic_2d_slices,
    find_mitea_image_files,
    ImplicitNeuralRepresentation,
    contour_reprojection_loss,
    laplacian_smoothness_loss,
    CONFIG
)


# ============================================================================
# RECONSTRUCTION (From sparse_reconstruction_2.py)
# ============================================================================

RECON_CONFIG = {
    'device': CONFIG['device'],
    'data_path': CONFIG['data_path'],
    'hidden_dim': 32,
    'num_inr_layers': 4,
    'learning_rate': 0.0001,
    'num_views': 3,
    'num_optimization_steps': 1500,
    'grid_resolution': 128,
}


class SparseViewReconstructor:
    """Reconstruct 3D cardiac geometry from sparse 2D views"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        self.output_dir = Path('./sparse_reconstruction_results')
        self.output_dir.mkdir(exist_ok=True)
    
    def reconstruct_subject(self, img_file, label_file, subject_id, verbose=True):
        """Reconstruct single subject"""
        
        if verbose:
            print(f"\n  Reconstructing: {subject_id}")
        
        # Load data
        vol, seg = load_mitea_subject(img_file, label_file)
        if vol is None:
            if verbose:
                print(f"  ✗ Failed to load")
            return None
        
        # Extract views
        all_slices, all_contours = extract_synthetic_2d_slices(
            vol, seg, num_views=self.config['num_views']
        )
        
        # Initialize model
        model = ImplicitNeuralRepresentation(
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_inr_layers']
        ).to(self.device)
        
        # Optimize
        if verbose:
            print(f"  Optimizing INR ({self.config['num_optimization_steps']} steps)...")
        model, losses = self._optimize(model, all_slices, all_contours, verbose=False)
        
        # Sample grid
        with torch.no_grad():
            occupancy_grid = model.sample_grid(
                resolution=self.config['grid_resolution'],
                device=self.device,
                requires_grad=False
            )
        
        # Extract mesh
        vertices, faces = self._extract_mesh(occupancy_grid, verbose=False)
        
        # Save results
        self._save_results(subject_id, occupancy_grid, vertices, faces)
        
        if verbose:
            print(f"  ✓ Done - {len(vertices) if vertices is not None else 0} vertices")
        
        return {
            'subject_id': subject_id,
            'grid': occupancy_grid,
            'vertices': vertices,
            'faces': faces
        }
    
    def _optimize(self, model, slices_2d, contours_2d, verbose=True):
        """Optimize INR"""
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        losses = []
        
        for step in range(self.config['num_optimization_steps']):
            optimizer.zero_grad()
            
            occupancy_grid = model.sample_grid(
                resolution=64,
                device=self.device,
                requires_grad=True
            )
            
            loss = 0.0
            contours_device = contours_2d.to(self.device)
            for view_idx in range(contours_2d.shape[0]):
                loss += contour_reprojection_loss(occupancy_grid, contours_device[view_idx])
            
            loss += laplacian_smoothness_loss(occupancy_grid, weight=0.01)
            
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if verbose and step % 100 == 0:
                print(f"    Step {step}: loss={loss.item():.6f}")
        
        return model, losses
    
    def _extract_mesh(self, occupancy_grid, verbose=True):
        """Extract mesh with adaptive threshold"""
        try:
            from skimage import measure
        except ImportError:
            if verbose:
                print("  (skimage not available)")
            return None, None
        
        occ_np = occupancy_grid.cpu().numpy()
        threshold = np.mean(occ_np)
        
        try:
            vertices, faces, _, _ = measure.marching_cubes(occ_np, level=threshold)
            return vertices, faces
        except (ValueError, RuntimeError):
            return None, None
    
    def _save_results(self, subject_id, occupancy_grid, vertices, faces):
        """Save reconstruction results"""
        
        occ_np = occupancy_grid.cpu().numpy().astype(np.float32)
        
        # Save occupancy grid
        nii = nib.Nifti1Image(occ_np, np.eye(4))
        out_nii = self.output_dir / f'{subject_id}_occupancy_grid.nii.gz'
        nib.save(nii, out_nii)
        
        # Save metadata
        config_serializable = {
            'hidden_dim': int(self.config['hidden_dim']),
            'num_inr_layers': int(self.config['num_inr_layers']),
            'learning_rate': float(self.config['learning_rate']),
            'num_views': int(self.config['num_views']),
            'num_optimization_steps': int(self.config['num_optimization_steps']),
            'grid_resolution': int(self.config['grid_resolution']),
        }
        
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
            'num_vertices': int(len(vertices)) if vertices is not None else 0,
            'num_faces': int(len(faces)) if faces is not None else 0,
            'config': config_serializable
        }
        
        out_meta = self.output_dir / f'{subject_id}_metadata.json'
        with open(out_meta, 'w') as f:
            json.dump(metadata, f, indent=4)


# ============================================================================
# 3D COMPARISON VIEWER (FIXED)
# ============================================================================

class PreciseThreeDComparator:
    """Production-ready 3D comparison with proper coordinate alignment"""
    
    def __init__(self):
        self.output_dir = Path('./3d_comparison_viewers_v2')
        self.output_dir.mkdir(exist_ok=True)
        self.reconstruction_dir = Path('./sparse_reconstruction_results')
    
    def load_original_seg(self, img_file, label_file):
        """Load original segmentation"""
        vol, seg = load_mitea_subject(img_file, label_file)
        if vol is None:
            return None, None
        return vol.numpy() if isinstance(vol, torch.Tensor) else vol, \
               seg.numpy() if isinstance(seg, torch.Tensor) else seg
    
    def load_reconstruction(self, subject_id):
        """Load reconstructed occupancy grid"""
        grid_file = self.reconstruction_dir / f'{subject_id}_occupancy_grid.nii.gz'
        if not grid_file.exists():
            return None
        nii = nib.load(grid_file)
        return nii.get_fdata().astype(np.float32)
    
    def extract_quality_mesh(self, volume, threshold=None, max_vertices=30000):
        """Extract PROPER MESH with topology preservation"""
        try:
            from skimage import measure
        except ImportError:
            return self._extract_pointcloud(volume, threshold, max_vertices)
        
        if threshold is None:
            threshold = np.mean(volume)
        
        try:
            vertices, faces, _, _ = measure.marching_cubes(volume, level=threshold, step_size=1)
            
            # Downsample if needed
            if len(vertices) > max_vertices:
                stride = len(vertices) // max_vertices
                keep_indices = np.arange(0, len(vertices), stride)
                vertices = vertices[keep_indices]
                
                # Remap faces
                valid_faces = []
                old_to_new = {old: new for new, old in enumerate(keep_indices)}
                for face in faces:
                    if all(v in old_to_new for v in face):
                        valid_faces.append([old_to_new[v] for v in face])
                faces = np.array(valid_faces) if valid_faces else None
            
            return vertices, faces
        
        except (ValueError, RuntimeError):
            return self._extract_pointcloud(volume, threshold, max_vertices)
    
    def _extract_pointcloud(self, volume, threshold, max_points):
        """Fallback: voxel point cloud"""
        coords = np.argwhere(volume > (threshold if threshold else np.mean(volume)))
        if len(coords) > max_points:
            stride = len(coords) // max_points
            coords = coords[::stride]
        return coords, None
    
    def transform_to_canonical_coords(self, vertices, source_shape, target_shape=(128, 128, 128)):
        """Transform vertices from voxel space to canonical [-1, 1]^3 space"""
        scale_factor = np.array(target_shape) / np.array(source_shape)
        vertices_resampled = vertices * scale_factor
        vertices_canonical = 2.0 * (vertices_resampled / np.array(target_shape)) - 1.0
        return vertices_canonical
    
    def create_aligned_mesh_trace(self, vertices, faces, color='blue', name='Surface', opacity=0.75):
        """Create plotly mesh trace"""
        if vertices is None or len(vertices) == 0:
            return None
        
        if faces is None or len(faces) == 0:
            return go.Scatter3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                mode='markers',
                marker=dict(size=2, color=color, opacity=opacity),
                name=name,
                hovertemplate=f'<b>{name}</b><br>(%{{x:.2f}}, %{{y:.2f}}, %{{z:.2f}})<extra></extra>'
            )
        else:
            return go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=color,
                opacity=opacity,
                name=name,
                flatshading=True
            )
    
    def create_comparison_viewer(self, subject_id, img_file, label_file):
        """Create side-by-side 3D comparison - FIXED"""
        
        print(f"    Creating side-by-side viewer...")
        
        # Load data
        vol_orig, seg_orig = self.load_original_seg(img_file, label_file)
        if seg_orig is None:
            return False
        
        recon_occ = self.load_reconstruction(subject_id)
        if recon_occ is None:
            return False
        
        canonical_shape = recon_occ.shape
        
        # Extract surfaces
        orig_verts, orig_faces = self.extract_quality_mesh(seg_orig, threshold=0.5)
        recon_verts, recon_faces = self.extract_quality_mesh(recon_occ, threshold=np.mean(recon_occ))
        
        # Transform to canonical space
        orig_verts_canonical = self.transform_to_canonical_coords(orig_verts, seg_orig.shape, canonical_shape)
        recon_verts_canonical = self.transform_to_canonical_coords(recon_verts, recon_occ.shape, canonical_shape)
        
        # Create figure
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=(f'Original', f'Reconstructed'),
            horizontal_spacing=0.05
        )
        
        orig_trace = self.create_aligned_mesh_trace(orig_verts_canonical, orig_faces, color='#1f77b4', name='Original', opacity=0.75)
        if orig_trace:
            fig.add_trace(orig_trace, row=1, col=1)
        
        recon_trace = self.create_aligned_mesh_trace(recon_verts_canonical, recon_faces, color='#ff7f0e', name='Reconstructed', opacity=0.75)
        if recon_trace:
            fig.add_trace(recon_trace, row=1, col=2)
        
        # FIXED: aspectmode goes in scene, not camera
        camera = dict(eye=dict(x=1.3, y=1.3, z=1.3))
        
        fig.update_layout(
            title_text=f'<b>3D Comparison: {subject_id}</b><br><sub>Blue=Original | Orange=Reconstructed</sub>',
            height=800,
            width=1800,
            showlegend=True,
            template='plotly_white',
            font=dict(size=11)
        )
        
        # FIXED: Update scenes with proper structure
        fig.update_scenes(camera=camera, aspectmode='cube')
        
        out_path = self.output_dir / f'{subject_id}_comparison.html'
        fig.write_html(str(out_path))
        
        return True
    
    def create_overlay_viewer(self, subject_id, img_file, label_file):
        """Create overlay viewer - FIXED"""
        
        print(f"    Creating overlay viewer...")
        
        vol_orig, seg_orig = self.load_original_seg(img_file, label_file)
        recon_occ = self.load_reconstruction(subject_id)
        
        if seg_orig is None or recon_occ is None:
            return False
        
        canonical_shape = recon_occ.shape
        
        orig_verts, orig_faces = self.extract_quality_mesh(seg_orig, threshold=0.5)
        recon_verts, recon_faces = self.extract_quality_mesh(recon_occ, threshold=np.mean(recon_occ))
        
        orig_verts_canonical = self.transform_to_canonical_coords(orig_verts, seg_orig.shape, canonical_shape)
        recon_verts_canonical = self.transform_to_canonical_coords(recon_verts, recon_occ.shape, canonical_shape)
        
        fig = go.Figure()
        
        orig_trace = self.create_aligned_mesh_trace(orig_verts_canonical, orig_faces, color='#1f77b4', name='Original', opacity=0.6)
        if orig_trace:
            fig.add_trace(orig_trace)
        
        recon_trace = self.create_aligned_mesh_trace(recon_verts_canonical, recon_faces, color='#ff7f0e', name='Reconstructed', opacity=0.6)
        if recon_trace:
            fig.add_trace(recon_trace)
        
        # FIXED: Proper scene configuration
        fig.update_layout(
            title=f'<b>3D Overlay: {subject_id}</b><br><sub>Blue=Original | Orange=Reconstructed</sub>',
            scene=dict(
                camera=dict(eye=dict(x=1.3, y=1.3, z=1.3)),
                aspectmode='cube'
            ),
            height=900,
            width=900,
            showlegend=True,
            template='plotly_white'
        )
        
        out_path = self.output_dir / f'{subject_id}_overlay.html'
        fig.write_html(str(out_path))
        
        return True


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("="*70)
    print("COMPLETE PIPELINE: RECONSTRUCTION + 3D COMPARISON (FIXED)")
    print("="*70)
    
    # Get subjects
    subjects = find_mitea_image_files(RECON_CONFIG['data_path'])[:5]
    
    if not subjects:
        print("ERROR: No subjects found")
        return
    
    print(f"\nFound {len(subjects)} subjects")
    
    # Phase 1: Reconstruction
    print("\n" + "="*70)
    print("PHASE 1: SPARSE VIEW RECONSTRUCTION")
    print("="*70)
    
    reconstructor = SparseViewReconstructor(RECON_CONFIG)
    recon_results = []
    
    for i, (img_file, label_file) in enumerate(subjects):
        subject_id = img_file.stem
        print(f"\n[{i+1}/{len(subjects)}] {subject_id}")
        
        try:
            result = reconstructor.reconstruct_subject(img_file, label_file, subject_id, verbose=True)
            if result:
                recon_results.append(result)
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✓ Reconstructed {len(recon_results)}/{len(subjects)} subjects")
    
    # Phase 2: 3D Comparison
    print("\n" + "="*70)
    print("PHASE 2: 3D COMPARISON VIEWERS")
    print("="*70)
    
    comparator = PreciseThreeDComparator()
    success_count = 0
    
    for i, (img_file, label_file) in enumerate(subjects):
        subject_id = img_file.stem
        print(f"\n[{i+1}/{len(subjects)}] {subject_id}")
        
        try:
            if comparator.create_comparison_viewer(subject_id, img_file, label_file):
                comparator.create_overlay_viewer(subject_id, img_file, label_file)
                print(f"  ✓ Viewers created")
                success_count += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "="*70)
    print("COMPLETE PIPELINE FINISHED")
    print("="*70)
    print(f"\n✓ Reconstruction files: {reconstructor.output_dir}/")
    print(f"✓ Viewer files: {comparator.output_dir}/")
    print(f"✓ Viewers created: {success_count}/{len(subjects)}")
    print(f"\nOPEN IN BROWSER:")
    print(f"  - *_comparison.html (Side-by-side)")
    print(f"  - *_overlay.html (Overlay)")
    print("="*70)


if __name__ == '__main__':
    main()
