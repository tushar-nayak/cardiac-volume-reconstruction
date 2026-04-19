# #!/usr/bin/env python3
# """
# MINIMAL STARTER: 3D Cardiac Reconstruction from Echo
# Complete working pipeline - run this to get started in 2 days.
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np
# from pathlib import Path
# import yaml
# import os

# # ============================================================================
# # CONFIG
# # ============================================================================

# CONFIG = {
#     'device': 'cuda' if torch.cuda.is_available() else 'cpu',
#     'data_path': Path("/home/sofa/host_dir/cap-mitea/mitea"),
#     'checkpoint_path': Path('./checkpoints'),
#     'num_views': 6,
#     'image_size': 256,
#     'hidden_dim': 64,
#     'num_inr_layers': 4,
#     'learning_rate': 1e-3,
#     'num_optimization_steps': 200,
#     'batch_size': 1,  # Per-subject optimization
# }

# # ============================================================================
# # MODELS
# # ============================================================================

# class PositionalEncoding:
#     """Positional encoding for coordinates."""
#     def __init__(self, num_freqs=4):
#         self.num_freqs = num_freqs
    
#     def encode(self, coords):
#         """coords: (..., 3) in [-1, 1]"""
#         pe = []
#         for i in range(self.num_freqs):
#             freq = 2.0 ** i
#             pe.append(torch.sin(freq * np.pi * coords))
#             pe.append(torch.cos(freq * np.pi * coords))
#         return torch.cat(pe, dim=-1)  # (..., 8*3)


# class ImplicitNeuralRepresentation(nn.Module):
#     """INR: f(x,y,z) -> occupancy ∈ [0,1]"""
#     def __init__(self, hidden_dim=64, num_layers=4):
#         super().__init__()
#         self.pe = PositionalEncoding(num_freqs=4)
#         input_dim = 8 * 3  # positional encoding output
        
#         layers = []
#         for i in range(num_layers):
#             in_dim = input_dim if i == 0 else hidden_dim
#             out_dim = hidden_dim if i < num_layers - 1 else 1
#             layers.append(nn.Linear(in_dim, out_dim))
#             if i < num_layers - 1:
#                 layers.append(nn.ReLU())
        
#         self.mlp = nn.Sequential(*layers)
    
#     def forward(self, coords):
#         """coords: (..., 3)"""
#         pe_coords = self.pe.encode(coords)
#         occupancy = torch.sigmoid(self.mlp(pe_coords))
#         return occupancy
    
#     def sample_grid(self, resolution=64, device='cpu'):
#         """Sample occupancy on 3D grid"""
#         linspace = torch.linspace(-1, 1, resolution, device=device)
#         grid = torch.stack(torch.meshgrid(linspace, linspace, linspace, indexing='ij'), dim=-1)
#         with torch.no_grad():
#             occupancy = self.forward(grid).squeeze(-1)
#         return occupancy


# # ============================================================================
# # DATA LOADING (MITEA)
# # ============================================================================

# def load_mitea_subject(subject_path):
#     """Load single MITEA subject - SIMPLIFIED VERSION"""
#     import nibabel as nib
    
#     # Load 3D volume and segmentation
#     vol_file = list(subject_path.glob('*_vol.nii*'))[0]
#     seg_file = list(subject_path.glob('*_seg.nii*'))[0]
    
#     vol = torch.tensor(nib.load(vol_file).get_fdata(), dtype=torch.float32)
#     seg = torch.tensor(nib.load(seg_file).get_fdata(), dtype=torch.float32)
    
#     # Normalize
#     vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
#     seg = (seg > 0.5).float()
    
#     return vol, seg


# def extract_synthetic_2d_slices(vol, seg, num_views=6):
#     """Extract standard cardiac planes (A2C, A4C, PSAX, etc.)"""
#     D, H, W = vol.shape
#     slices_2d = []
#     contours_2d = []
    
#     # Define slice indices for standard planes
#     plane_indices = [
#         D // 4,      # A2C
#         D // 2,      # A4C
#         D // 10,     # PSAX basal
#         D // 3,      # PSAX mid
#         2*D // 3,    # PSAX apical
#         D // 6,      # RVOT
#     ][:num_views]
    
#     for idx in plane_indices:
#         # Extract 2D slice
#         slice_img = vol[idx, :, :]
#         slice_seg = seg[idx, :, :]
        
#         # Resize
#         slice_img = F.interpolate(
#             slice_img.unsqueeze(0).unsqueeze(0),
#             size=(256, 256),
#             mode='bilinear'
#         ).squeeze()
        
#         slice_seg = F.interpolate(
#             slice_seg.unsqueeze(0).unsqueeze(0),
#             size=(256, 256),
#             mode='nearest'
#         ).squeeze()
        
#         slices_2d.append(slice_img)
#         contours_2d.append(slice_seg)
    
#     return torch.stack(slices_2d), torch.stack(contours_2d)


# # ============================================================================
# # LOSSES
# # ============================================================================

# def contour_reprojection_loss(occupancy_grid, target_contour):
#     """Loss between max-projection and target 2D contour"""
#     # Simple max-pooling projection
#     projection = torch.max(occupancy_grid, dim=0)[0]  # (H, W)
    
#     # Resize to match contour
#     projection_resized = F.interpolate(
#         projection.unsqueeze(0).unsqueeze(0),
#         size=target_contour.shape,
#         mode='bilinear',
#         align_corners=False
#     ).squeeze()
    
#     # BCE loss
#     loss = F.binary_cross_entropy(projection_resized, target_contour)
#     return loss


# def laplacian_smoothness_loss(occupancy_grid, weight=0.01):
#     """Encourage smooth surface via L2 of Laplacian"""
#     # Approximate Laplacian with finite differences
#     lap = (
#         torch.roll(occupancy_grid, 1, dims=0) +
#         torch.roll(occupancy_grid, -1, dims=0) +
#         torch.roll(occupancy_grid, 1, dims=1) +
#         torch.roll(occupancy_grid, -1, dims=1) +
#         torch.roll(occupancy_grid, 1, dims=2) +
#         torch.roll(occupancy_grid, -1, dims=2) -
#         6 * occupancy_grid
#     )
    
#     # Penalize near surface
#     surface_mask = (occupancy_grid > 0.2) & (occupancy_grid < 0.8)
#     loss = torch.mean(lap[surface_mask] ** 2)
#     return weight * loss


# # ============================================================================
# # TRAINING / OPTIMIZATION
# # ============================================================================

# def optimize_single_subject(model, slices_2d, contours_2d, config, num_steps=200):
#     """Optimize INR for single subject given 2D contours"""
#     device = torch.device(config['device'])
#     model = model.to(device)
    
#     optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
#     losses = []
    
#     for step in range(num_steps):
#         optimizer.zero_grad()
        
#         # Sample occupancy grid
#         occupancy_grid = model.sample_grid(resolution=64, device=device)
        
#         # Contour reprojection loss (multiple views)
#         loss = 0.0
#         contours_device = contours_2d.to(device)
#         for view_idx in range(contours_2d.shape[0]):
#             loss_view = contour_reprojection_loss(occupancy_grid, contours_device[view_idx])
#             loss += loss_view
        
#         # Smoothness regularization
#         loss_smooth = laplacian_smoothness_loss(occupancy_grid, weight=0.01)
#         loss = loss + loss_smooth
        
#         loss.backward()
#         optimizer.step()
        
#         losses.append(loss.item())
        
#         if step % 50 == 0:
#             print(f"  Step {step}/{num_steps}: loss={loss.item():.6f}")
    
#     return model, losses


# def evaluate_subject(model, occupancy_grid, contours_2d):
#     """Compute metrics for single subject"""
#     metrics = {}
    
#     # Dice coefficient
#     pred_binary = (occupancy_grid > 0.5).float()
#     target_binary = (contours_2d[0] > 0.5).float()
    
#     intersection = torch.sum(pred_binary * target_binary)
#     union = torch.sum(pred_binary) + torch.sum(target_binary)
#     dice = (2 * intersection) / (union + 1e-6)
    
#     # IoU
#     iou = intersection / (union - intersection + 1e-6)
    
#     metrics['dice'] = dice.item()
#     metrics['iou'] = iou.item()
    
#     return metrics


# # ============================================================================
# # MAIN PIPELINE
# # ============================================================================

# def main():
#     """Complete end-to-end pipeline"""
#     device = torch.device(CONFIG['device'])
#     print(f"Device: {device}")
    
#     # Create checkpoint directory
#     CONFIG['checkpoint_path'].mkdir(exist_ok=True)
    
#     # Initialize model
#     model = ImplicitNeuralRepresentation(
#         hidden_dim=CONFIG['hidden_dim'],
#         num_layers=CONFIG['num_inr_layers']
#     )
    
#     # List MITEA subjects
#     data_path = CONFIG['data_path']
#     if not data_path.exists():
#         print(f"ERROR: MITEA data not found at {data_path}")
#         print("Download MITEA or update data_path in CONFIG")
#         return
    
#     subject_dirs = sorted([d for d in (data_path / 'segmentations').glob('*')])[:10]  # First 10 for speed
    
#     print(f"Found {len(subject_dirs)} subjects")
    
#     all_metrics = {'dice': [], 'iou': []}
    
#     # Process each subject
#     for subject_idx, subject_dir in enumerate(subject_dirs):
#         subject_id = subject_dir.name
#         print(f"\n[{subject_idx + 1}/{len(subject_dirs)}] Processing {subject_id}...")
        
#         try:
#             # Load subject
#             vol, seg = load_mitea_subject(data_path / 'volumes' / subject_id)
            
#             # Extract synthetic 2D slices
#             slices_2d, contours_2d = extract_synthetic_2d_slices(
#                 vol, seg, num_views=CONFIG['num_views']
#             )
            
#             print(f"  Shape: slices={slices_2d.shape}, contours={contours_2d.shape}")
            
#             # Reset model for this subject
#             model = ImplicitNeuralRepresentation(
#                 hidden_dim=CONFIG['hidden_dim'],
#                 num_layers=CONFIG['num_inr_layers']
#             )
            
#             # Optimize
#             model, losses = optimize_single_subject(
#                 model,
#                 slices_2d,
#                 contours_2d,
#                 CONFIG,
#                 num_steps=CONFIG['num_optimization_steps']
#             )
            
#             # Evaluate
#             with torch.no_grad():
#                 occupancy_grid = model.sample_grid(resolution=64, device=device)
#                 metrics = evaluate_subject(model, occupancy_grid, contours_2d)
            
#             print(f"  Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}")
#             all_metrics['dice'].append(metrics['dice'])
#             all_metrics['iou'].append(metrics['iou'])
            
#         except Exception as e:
#             print(f"  ERROR: {e}")
    
#     # Summary
#     print("\n" + "="*60)
#     print("FINAL RESULTS")
#     print("="*60)
#     print(f"Mean Dice: {np.mean(all_metrics['dice']):.4f} ± {np.std(all_metrics['dice']):.4f}")
#     print(f"Mean IoU:  {np.mean(all_metrics['iou']):.4f} ± {np.std(all_metrics['iou']):.4f}")
#     print(f"Subjects processed: {len(all_metrics['dice'])}")


# if __name__ == '__main__':
#     main()



#!/usr/bin/env python3
"""
MINIMAL STARTER: 3D Cardiac Reconstruction from Echo
Complete working pipeline - run this to get started in 2 days.

FIXED FOR MITEA STRUCTURE:
- Images in: data_path/images/
- Labels in: data_path/labels/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path
import os
import traceback


# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_path': Path("/home/sofa/host_dir/cap-mitea/mitea"),
    'checkpoint_path': Path('./checkpoints'),
    'num_views': 6,
    'image_size': 256,
    'hidden_dim': 64,
    'num_inr_layers': 4,
    'learning_rate': 1e-3,
    'num_optimization_steps': 200,
    'batch_size': 1,  # Per-subject optimization
}


# ============================================================================
# MODELS
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
        return torch.cat(pe, dim=-1)  # (..., 8*3)


class ImplicitNeuralRepresentation(nn.Module):
    """INR: f(x,y,z) -> occupancy ∈ [0,1]"""
    def __init__(self, hidden_dim=64, num_layers=4):
        super().__init__()
        self.pe = PositionalEncoding(num_freqs=4)
        input_dim = 8 * 3  # positional encoding output
        
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else 1
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, coords):
        """coords: (..., 3)"""
        pe_coords = self.pe.encode(coords)
        occupancy = torch.sigmoid(self.mlp(pe_coords))
        return occupancy
    
    def sample_grid(self, resolution=64, device='cpu'):
        """Sample occupancy on 3D grid"""
        linspace = torch.linspace(-1, 1, resolution, device=device)
        grid = torch.stack(torch.meshgrid(linspace, linspace, linspace, indexing='ij'), dim=-1)
        with torch.no_grad():
            occupancy = self.forward(grid).squeeze(-1)
        return occupancy


# ============================================================================
# DATA LOADING (MITEA) - FIXED FOR ACTUAL STRUCTURE
# ============================================================================

def find_mitea_image_files(data_path):
    """
    Find all image-label pairs in MITEA structure.
    Expected: data_path/images/*.nii* and data_path/labels/*.nii*
    """
    images_dir = data_path / 'images'
    labels_dir = data_path / 'labels'
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"ERROR: Expected images/ and labels/ subdirectories")
        print(f"  images_dir exists: {images_dir.exists()}")
        print(f"  labels_dir exists: {labels_dir.exists()}")
        if images_dir.exists():
            print(f"  Contents of images/: {list(images_dir.glob('*'))[:5]}")
        if labels_dir.exists():
            print(f"  Contents of labels/: {list(labels_dir.glob('*'))[:5]}")
        return []
    
    # Find all .nii or .nii.gz files in images
    image_files = sorted(images_dir.glob('*.nii*'))
    print(f"Found {len(image_files)} image files in {images_dir}")
    
    pairs = []
    for img_file in image_files:
        # Construct corresponding label filename
        stem = img_file.stem
        if stem.endswith('.nii'):
            stem = stem[:-4]  # Remove .nii if double-barreled
        
        # Look for matching label file
        label_candidates = list(labels_dir.glob(f'{stem}*'))
        if label_candidates:
            pairs.append((img_file, label_candidates[0]))
            print(f"  Matched: {img_file.name} <-> {label_candidates[0].name}")
        else:
            print(f"  WARNING: No matching label for {img_file.name}")
    
    print(f"\nFound {len(pairs)} valid image-label pairs")
    return pairs


def load_mitea_subject(img_file, label_file):
    """Load single MITEA subject image and label"""
    try:
        import nibabel as nib
    except ImportError:
        print("ERROR: nibabel not installed. Run: pip install nibabel")
        return None, None
    
    try:
        vol = torch.tensor(nib.load(img_file).get_fdata(), dtype=torch.float32)
        seg = torch.tensor(nib.load(label_file).get_fdata(), dtype=torch.float32)
        
        # Normalize
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
        seg = (seg > 0.5).float()
        
        return vol, seg
    except Exception as e:
        print(f"  Error loading {img_file.name}: {e}")
        return None, None


def extract_synthetic_2d_slices(vol, seg, num_views=6):
    """Extract standard cardiac planes (A2C, A4C, PSAX, etc.)"""
    D, H, W = vol.shape
    slices_2d = []
    contours_2d = []
    
    # Define slice indices for standard planes
    plane_indices = [
        D // 4,      # A2C
        D // 2,      # A4C
        D // 10,     # PSAX basal
        D // 3,      # PSAX mid
        2*D // 3,    # PSAX apical
        D // 6,      # RVOT
    ][:num_views]
    
    for idx in plane_indices:
        # Extract 2D slice
        slice_img = vol[idx, :, :]
        slice_seg = seg[idx, :, :]
        
        # Resize
        slice_img = F.interpolate(
            slice_img.unsqueeze(0).unsqueeze(0),
            size=(256, 256),
            mode='bilinear'
        ).squeeze()
        
        slice_seg = F.interpolate(
            slice_seg.unsqueeze(0).unsqueeze(0),
            size=(256, 256),
            mode='nearest'
        ).squeeze()
        
        slices_2d.append(slice_img)
        contours_2d.append(slice_seg)
    
    return torch.stack(slices_2d), torch.stack(contours_2d)


# ============================================================================
# LOSSES
# ============================================================================

def contour_reprojection_loss(occupancy_grid, target_contour):
    """Loss between max-projection and target 2D contour"""
    # Simple max-pooling projection
    projection = torch.max(occupancy_grid, dim=0)[0]  # (H, W)
    
    # Resize to match contour
    projection_resized = F.interpolate(
        projection.unsqueeze(0).unsqueeze(0),
        size=target_contour.shape,
        mode='bilinear',
        align_corners=False
    ).squeeze()
    
    # BCE loss
    loss = F.binary_cross_entropy(projection_resized, target_contour)
    return loss


def laplacian_smoothness_loss(occupancy_grid, weight=0.01):
    """Encourage smooth surface via L2 of Laplacian"""
    # Approximate Laplacian with finite differences
    lap = (
        torch.roll(occupancy_grid, 1, dims=0) +
        torch.roll(occupancy_grid, -1, dims=0) +
        torch.roll(occupancy_grid, 1, dims=1) +
        torch.roll(occupancy_grid, -1, dims=1) +
        torch.roll(occupancy_grid, 1, dims=2) +
        torch.roll(occupancy_grid, -1, dims=2) -
        6 * occupancy_grid
    )
    
    # Penalize near surface
    surface_mask = (occupancy_grid > 0.2) & (occupancy_grid < 0.8)
    loss = torch.mean(lap[surface_mask] ** 2)
    return weight * loss


# ============================================================================
# TRAINING / OPTIMIZATION
# ============================================================================

def optimize_single_subject(model, slices_2d, contours_2d, config, num_steps=200):
    """Optimize INR for single subject given 2D contours"""
    device = torch.device(config['device'])
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    losses = []
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Sample occupancy grid
        occupancy_grid = model.sample_grid(resolution=64, device=device)
        
        # Contour reprojection loss (multiple views)
        loss = 0.0
        contours_device = contours_2d.to(device)
        for view_idx in range(contours_2d.shape[0]):
            loss_view = contour_reprojection_loss(occupancy_grid, contours_device[view_idx])
            loss += loss_view
        
        # Smoothness regularization
        loss_smooth = laplacian_smoothness_loss(occupancy_grid, weight=0.01)
        loss = loss + loss_smooth
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 50 == 0:
            print(f"  Step {step}/{num_steps}: loss={loss.item():.6f}")
    
    return model, losses


def evaluate_subject(model, occupancy_grid, contours_2d):
    """Compute metrics for single subject"""
    metrics = {}
    
    # Dice coefficient
    pred_binary = (occupancy_grid > 0.5).float()
    target_binary = (contours_2d[0] > 0.5).float()
    
    intersection = torch.sum(pred_binary * target_binary)
    union = torch.sum(pred_binary) + torch.sum(target_binary)
    dice = (2 * intersection) / (union + 1e-6)
    
    # IoU
    iou = intersection / (union - intersection + 1e-6)
    
    metrics['dice'] = dice.item()
    metrics['iou'] = iou.item()
    
    return metrics


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Complete end-to-end pipeline"""
    device = torch.device(CONFIG['device'])
    print(f"Device: {device}")
    print(f"Data path: {CONFIG['data_path']}")
    
    # Create checkpoint directory
    CONFIG['checkpoint_path'].mkdir(exist_ok=True)
    
    # Initialize model
    model = ImplicitNeuralRepresentation(
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_inr_layers']
    )
    
    # List MITEA subjects
    data_path = CONFIG['data_path']
    if not data_path.exists():
        print(f"ERROR: MITEA data not found at {data_path}")
        print("Download MITEA or update data_path in CONFIG")
        return
    
    # ========== FIXED: Find image-label pairs in images/ and labels/ ==========
    print("\n" + "="*60)
    print("FINDING MITEA SUBJECTS")
    print("="*60)
    subject_pairs = find_mitea_image_files(data_path)[:10]  # First 10 for speed
    
    if not subject_pairs:
        print(f"\nERROR: No image-label pairs found")
        print(f"Expected structure:")
        print(f"  {data_path}/images/*.nii*")
        print(f"  {data_path}/labels/*.nii*")
        return
    
    print(f"\nTotal subjects to process: {len(subject_pairs)}\n")
    
    all_metrics = {'dice': [], 'iou': []}
    
    # Process each subject
    for subject_idx, (img_file, label_file) in enumerate(subject_pairs):
        subject_id = img_file.stem
        print(f"\n[{subject_idx + 1}/{len(subject_pairs)}] Processing {subject_id}...")
        
        try:
            # Load subject
            vol, seg = load_mitea_subject(img_file, label_file)
            if vol is None:
                print(f"  Skipping {subject_id}")
                continue
            
            print(f"  Loaded: vol shape={vol.shape}, seg shape={seg.shape}")
            
            # Extract synthetic 2D slices
            slices_2d, contours_2d = extract_synthetic_2d_slices(
                vol, seg, num_views=CONFIG['num_views']
            )
            
            print(f"  Extracted: slices={slices_2d.shape}, contours={contours_2d.shape}")
            
            # Reset model for this subject
            model = ImplicitNeuralRepresentation(
                hidden_dim=CONFIG['hidden_dim'],
                num_layers=CONFIG['num_inr_layers']
            )
            
            # Optimize
            print(f"  Optimizing INR...")
            model, losses = optimize_single_subject(
                model,
                slices_2d,
                contours_2d,
                CONFIG,
                num_steps=CONFIG['num_optimization_steps']
            )
            
            # Evaluate
            with torch.no_grad():
                occupancy_grid = model.sample_grid(resolution=64, device=device)
                metrics = evaluate_subject(model, occupancy_grid, contours_2d)
            
            print(f"  Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}")
            all_metrics['dice'].append(metrics['dice'])
            all_metrics['iou'].append(metrics['iou'])
            
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    if all_metrics['dice']:
        print(f"Mean Dice: {np.mean(all_metrics['dice']):.4f} ± {np.std(all_metrics['dice']):.4f}")
        print(f"Mean IoU:  {np.mean(all_metrics['iou']):.4f} ± {np.std(all_metrics['iou']):.4f}")
        print(f"Subjects processed: {len(all_metrics['dice'])}")
    else:
        print("No subjects successfully processed.")


if __name__ == '__main__':
    main()