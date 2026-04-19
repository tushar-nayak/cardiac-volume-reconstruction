#!/usr/bin/env python3
"""
3D VIEWER - SIDE-BY-SIDE COMPARISON (FIXED)
Display reconstructed and original segmentation in 3D viewers for easy comparison.
Uses plotly for interactive 3D visualization.

FIXES:
- Handles different volume dimensions by resampling
- Computes metrics correctly with matching shapes
- Robust error handling
"""

import numpy as np
import torch
from pathlib import Path
import json
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import zoom
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from minimal_starter_5 import (
    load_mitea_subject,
    find_mitea_image_files,
    CONFIG
)


class ThreeDComparator:
    """Interactive 3D comparison of original vs reconstructed"""
    
    def __init__(self):
        self.output_dir = Path('./3d_comparison_viewers')
        self.output_dir.mkdir(exist_ok=True)
        self.reconstruction_dir = Path('./sparse_reconstruction_results')
    
    def load_original_seg(self, img_file, label_file):
        """Load original segmentation"""
        vol, seg = load_mitea_subject(img_file, label_file)
        if vol is None:
            return None
        return seg.numpy() if isinstance(seg, torch.Tensor) else seg
    
    def load_reconstruction(self, subject_id):
        """Load reconstructed occupancy grid"""
        grid_file = self.reconstruction_dir / f'{subject_id}_occupancy_grid.nii.gz'
        if not grid_file.exists():
            print(f"  Reconstruction file not found: {grid_file}")
            return None
        
        nii = nib.load(grid_file)
        return nii.get_fdata().astype(np.float32)
    
    def resample_volume(self, volume, target_shape):
        """Resample volume to target shape"""
        factors = np.array(target_shape) / np.array(volume.shape)
        resampled = zoom(volume, factors, order=1)
        return resampled
    
    def extract_surface_points(self, volume, threshold=0.5, max_points=50000):
        """Extract surface points from 3D volume using marching cubes"""
        try:
            from skimage import measure
            
            # Find surface using marching cubes
            vertices, faces = measure.marching_cubes(volume, level=threshold)[:2]
            
            # Downsample if too many points
            if len(vertices) > max_points:
                stride = len(vertices) // max_points
                vertices = vertices[::stride]
                if faces is not None:
                    # Remap face indices for downsampled vertices
                    old_to_new = {}
                    new_idx = 0
                    for old_idx in np.unique(faces.flatten()):
                        if old_idx in np.arange(0, len(vertices)*stride, stride):
                            old_to_new[old_idx] = new_idx
                            new_idx += 1
                    faces = np.array([[old_to_new.get(i, 0) for i in face] for face in faces])
            
            return vertices, faces
        except Exception as e:
            print(f"  Marching cubes failed: {e}")
            # Fallback: use voxel centers where volume > threshold
            coords = np.argwhere(volume > threshold)
            if len(coords) > max_points:
                stride = len(coords) // max_points
                coords = coords[::stride]
            return coords, None
    
    def create_3d_mesh_trace(self, vertices, faces, color='blue', name='Surface', opacity=0.8):
        """Create plotly mesh trace"""
        if faces is None or len(faces) == 0:
            # Point cloud fallback
            trace = go.Scatter3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                mode='markers',
                marker=dict(size=2, color=color, opacity=opacity),
                name=name,
                hovertemplate=f'<b>{name}</b><br>Position: (%{{x:.1f}}, %{{y:.1f}}, %{{z:.1f}})<extra></extra>'
            )
        else:
            # Mesh surface
            trace = go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=color,
                opacity=opacity,
                name=name,
                hovertemplate=f'<b>{name}</b><extra></extra>'
            )
        
        return trace
    
    def create_comparison_viewer(self, subject_id, img_file, label_file):
        """Create interactive side-by-side 3D viewer"""
        
        print(f"\nCreating 3D comparison for {subject_id}...")
        
        # Load original segmentation
        print(f"  Loading original segmentation...")
        original_seg = self.load_original_seg(img_file, label_file)
        if original_seg is None:
            print(f"  Failed to load original segmentation")
            return False
        
        # Load reconstruction
        print(f"  Loading reconstruction...")
        reconstructed = self.load_reconstruction(subject_id)
        if reconstructed is None:
            return False
        
        print(f"  Original shape: {original_seg.shape}, Reconstructed shape: {reconstructed.shape}")
        
        # Extract surfaces
        print(f"  Extracting original surface...")
        orig_vertices, orig_faces = self.extract_surface_points(
            original_seg, threshold=0.5, max_points=15000
        )
        
        print(f"  Extracting reconstructed surface...")
        recon_vertices, recon_faces = self.extract_surface_points(
            reconstructed, threshold=np.mean(reconstructed), max_points=15000
        )
        
        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=(
                f'Original Segmentation<br><sub>Shape: {original_seg.shape}</sub>',
                f'Reconstructed Volume<br><sub>Shape: {reconstructed.shape}</sub>'
            ),
            horizontal_spacing=0.05
        )
        
        # Add original surface
        print(f"  Adding original surface to viewer...")
        orig_trace = self.create_3d_mesh_trace(
            orig_vertices, orig_faces, color='#1f77b4', name='Original', opacity=0.85
        )
        fig.add_trace(orig_trace, row=1, col=1)
        
        # Add reconstructed surface
        print(f"  Adding reconstructed surface to viewer...")
        recon_trace = self.create_3d_mesh_trace(
            recon_vertices, recon_faces, color='#ff7f0e', name='Reconstructed', opacity=0.85
        )
        fig.add_trace(recon_trace, row=1, col=2)
        
        # Update layout
        fig.update_layout(
            title_text=f'<b>3D Comparison: {subject_id}</b><br><sub>Original (Left) vs Reconstructed (Right)</sub>',
            height=800,
            width=1800,
            showlegend=True,
            template='plotly_white',
            font=dict(size=12),
            hovermode='closest'
        )
        
        # Update camera for both subplots
        fig.update_scenes(
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='auto'
        )
        
        # Save HTML
        out_path = self.output_dir / f'{subject_id}_comparison.html'
        fig.write_html(str(out_path))
        print(f"  ✓ Saved interactive viewer: {out_path}")
        
        return True
    
    def create_overlay_viewer(self, subject_id, img_file, label_file):
        """Create overlay viewer (both in same 3D space)"""
        
        print(f"  Creating overlay viewer for {subject_id}...")
        
        # Load surfaces
        original_seg = self.load_original_seg(img_file, label_file)
        reconstructed = self.load_reconstruction(subject_id)
        
        if original_seg is None or reconstructed is None:
            return False
        
        # Extract surfaces
        orig_vertices, orig_faces = self.extract_surface_points(
            original_seg, threshold=0.5, max_points=15000
        )
        recon_vertices, recon_faces = self.extract_surface_points(
            reconstructed, threshold=np.mean(reconstructed), max_points=15000
        )
        
        # Create figure
        fig = go.Figure()
        
        # Add original (semi-transparent blue)
        orig_trace = self.create_3d_mesh_trace(
            orig_vertices, orig_faces, color='#1f77b4', name='Original', opacity=0.7
        )
        fig.add_trace(orig_trace)
        
        # Add reconstructed (semi-transparent orange)
        recon_trace = self.create_3d_mesh_trace(
            recon_vertices, recon_faces, color='#ff7f0e', name='Reconstructed', opacity=0.7
        )
        fig.add_trace(recon_trace)
        
        # Update layout
        fig.update_layout(
            title=f'<b>3D Overlay Comparison: {subject_id}</b><br><sub>Blue=Original, Orange=Reconstructed</sub>',
            scene=dict(
                aspectmode='auto',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=900,
            width=1200,
            showlegend=True,
            template='plotly_white',
            hovermode='closest',
            font=dict(size=12)
        )
        
        out_path = self.output_dir / f'{subject_id}_overlay.html'
        fig.write_html(str(out_path))
        print(f"  ✓ Saved overlay viewer: {out_path}")
        
        return True
    
    def create_metrics_summary(self, subject_id, img_file, label_file):
        """Create comparison metrics report"""
        
        print(f"  Computing comparison metrics...")
        
        # Load volumes
        original_seg = self.load_original_seg(img_file, label_file)
        reconstructed = self.load_reconstruction(subject_id)
        
        if original_seg is None or reconstructed is None:
            return None
        
        # CRITICAL FIX: Resample to match dimensions
        target_shape = reconstructed.shape
        if original_seg.shape != target_shape:
            print(f"    Resampling original from {original_seg.shape} to {target_shape}...")
            original_seg = self.resample_volume(original_seg, target_shape)
        
        # Normalize to [0, 1]
        orig_binary = (original_seg > 0.5).astype(np.float32)
        recon_binary = (reconstructed > np.mean(reconstructed)).astype(np.float32)
        
        # Compute metrics
        intersection = np.sum(orig_binary * recon_binary)
        union = np.sum(np.maximum(orig_binary, recon_binary))
        dice = (2 * intersection) / (np.sum(orig_binary) + np.sum(recon_binary) + 1e-6)
        iou = intersection / (union + 1e-6)
        
        # Volume overlap
        orig_volume = np.sum(orig_binary)
        recon_volume = np.sum(recon_binary)
        volume_ratio = recon_volume / (orig_volume + 1e-6)
        
        # Hausdorff distance (approximate with sampling)
        try:
            from scipy.spatial.distance import cdist
            orig_points = np.argwhere(orig_binary > 0)
            recon_points = np.argwhere(recon_binary > 0)
            
            # Sample if too many points
            if len(orig_points) > 5000:
                orig_points = orig_points[np.random.choice(len(orig_points), 5000, replace=False)]
            if len(recon_points) > 5000:
                recon_points = recon_points[np.random.choice(len(recon_points), 5000, replace=False)]
            
            if len(orig_points) > 0 and len(recon_points) > 0:
                distances = cdist(orig_points, recon_points)
                hausdorff = max(np.min(distances, axis=1).max(), 
                              np.min(distances, axis=0).max())
            else:
                hausdorff = 0.0
        except Exception as e:
            print(f"    Hausdorff computation failed: {e}")
            hausdorff = 0.0
        
        metrics = {
            'subject_id': subject_id,
            'dice_coefficient': float(dice),
            'iou': float(iou),
            'original_volume': float(orig_volume),
            'reconstructed_volume': float(recon_volume),
            'volume_ratio': float(volume_ratio),
            'hausdorff_distance': float(hausdorff),
            'original_shape': list(original_seg.shape),
            'reconstructed_shape': list(reconstructed.shape)
        }
        
        return metrics
    
    def process_all_subjects(self):
        """Process all subjects"""
        
        print("="*70)
        print("3D COMPARISON VIEWER GENERATOR (FIXED)")
        print("="*70)
        
        # Find subjects
        subjects = find_mitea_image_files(CONFIG['data_path'])[:5]
        
        if not subjects:
            print("ERROR: No subjects found")
            return
        
        print(f"Found {len(subjects)} subjects\n")
        
        all_metrics = []
        
        for i, (img_file, label_file) in enumerate(subjects):
            subject_id = img_file.stem
            print(f"\n[{i+1}/{len(subjects)}] {subject_id}")
            print("-" * 70)
            
            try:
                # Create comparison viewers
                if not self.create_comparison_viewer(subject_id, img_file, label_file):
                    print(f"  Skipping comparison viewer")
                    continue
                
                if not self.create_overlay_viewer(subject_id, img_file, label_file):
                    print(f"  Skipping overlay viewer")
                    continue
                
                # Compute metrics
                metrics = self.create_metrics_summary(subject_id, img_file, label_file)
                if metrics:
                    all_metrics.append(metrics)
                    print(f"\n  Metrics:")
                    print(f"    Dice: {metrics['dice_coefficient']:.4f}")
                    print(f"    IoU: {metrics['iou']:.4f}")
                    print(f"    Volume Ratio: {metrics['volume_ratio']:.4f}")
                    print(f"    Hausdorff Distance: {metrics['hausdorff_distance']:.2f}")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save metrics summary
        if all_metrics:
            metrics_file = self.output_dir / 'comparison_metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(all_metrics, f, indent=4)
            print(f"\n✓ Saved metrics summary: {metrics_file}")
            
            # Print summary statistics
            print("\n" + "="*70)
            print("COMPARISON SUMMARY")
            print("="*70)
            dice_scores = [m['dice_coefficient'] for m in all_metrics]
            iou_scores = [m['iou'] for m in all_metrics]
            volume_ratios = [m['volume_ratio'] for m in all_metrics]
            
            print(f"Mean Dice: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
            print(f"Mean IoU: {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")
            print(f"Mean Volume Ratio: {np.mean(volume_ratios):.4f} ± {np.std(volume_ratios):.4f}")
            print(f"Subjects Processed: {len(all_metrics)}")
            print("="*70)
        
        print(f"\n✓ All viewers saved in: {self.output_dir}/")
        print(f"\nOutput files per subject:")
        print(f"  - *_comparison.html: Side-by-side interactive viewer")
        print(f"  - *_overlay.html: Overlay interactive viewer")
        print(f"  - comparison_metrics.json: Quantitative metrics")
        print(f"\nOpen HTML files in your web browser:")
        print(f"  - Rotate: Click + Drag")
        print(f"  - Zoom: Scroll")
        print(f"  - Pan: Right Click + Drag")
        print(f"  - Toggle surfaces: Click legend items")


def main():
    comparator = ThreeDComparator()
    comparator.process_all_subjects()


if __name__ == '__main__':
    main()
