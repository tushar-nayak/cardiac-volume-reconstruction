#!/usr/bin/env python3
"""
MINIMAL STARTER: 3D Cardiac Reconstruction from Echo
Complete working pipeline - run this to get started in 2 days.

FIXED FOR MITEA STRUCTURE:
- Images in: data_path/images/
- Labels in: data_path/labels/

GRADIENT FIX: occupancy_grid computed WITHOUT no_grad() for backprop
DIMENSION FIX: evaluate_subject uses 2D projection, not 3D grid
DEVICE FIX: Move contours_2d to GPU in evaluate_subject
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
    'data_path': Path(__file__).resolve().parents[1] / "cap-mitea" / "mitea",
    'checkpoint_path': Path('./checkpoints'),
    'num_views': 3,
    'image_size': 256,
    'hidden_dim': 64,
    'num_inr_layers': 4,
    'learning_rate': 0.0001,
    'num_optimization_steps': 1500,
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
        
    def sample_grid(self, resolution=64, device='cpu', requires_grad=False):
        """Sample occupancy on 3D grid"""
        # FIX 1: Ensure resolution is a plain Python int (not tensor/float)
        resolution = int(resolution)
        
        # FIX 2: Convert device string to torch.device
        if isinstance(device, str):
            device = torch.device(device)
        
        # Now torch.linspace will work properly
        linspace = torch.linspace(-1, 1, resolution, device=device, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(linspace, linspace, linspace, indexing='ij'), dim=-1)
        
        if requires_grad:
            occupancy = self.forward(grid).squeeze(-1)
        else:
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
        print ("SEG")
        print(seg)
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

# AFTER (✅ CORRECT - BETTER CONSTRAINED)
def contour_reprojection_loss(occupancy_grid, target_contour):
    projection = torch.mean(occupancy_grid, dim=0)  # ← AVERAGE: Preserves structure
    
    projection_resized = F.interpolate(
        projection.unsqueeze(0).unsqueeze(0),
        size=target_contour.shape,
        mode='bilinear',
        align_corners=False
    ).squeeze()
    
    loss = F.binary_cross_entropy(projection_resized, target_contour)
    
    # ADD BOUNDARY CONSISTENCY LOSS
    grad_x = torch.diff(projection_resized, dim=0, prepend=projection_resized[:1])
    grad_y = torch.diff(projection_resized, dim=1, prepend=projection_resized[:, :1])
    grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
    boundary_loss = torch.mean(grad_norm * (1 - target_contour))
    
    return loss + 0.5 * boundary_loss


def laplacian_smoothness_loss(occupancy_grid, weight=0.2):  # ← 10X STRONGER!
    lap = (
        torch.roll(occupancy_grid, 1, dims=0) +
        torch.roll(occupancy_grid, -1, dims=0) +
        torch.roll(occupancy_grid, 1, dims=1) +
        torch.roll(occupancy_grid, -1, dims=1) +
        torch.roll(occupancy_grid, 1, dims=2) +
        torch.roll(occupancy_grid, -1, dims=2) -
        6 * occupancy_grid
    )
    surface_mask = (occupancy_grid > 0.2) & (occupancy_grid < 0.8)
    loss = torch.mean(lap[surface_mask] ** 2)
    return weight * loss

def volume_entropy_loss(occupancy_grid, weight=0.1):  # ← NEW!
    """Enforce binary occupancy (0 or 1, not 0.5)"""
    eps = 1e-6
    entropy = -(occupancy_grid * torch.log(occupancy_grid + eps) + 
                (1 - occupancy_grid) * torch.log(1 - occupancy_grid + eps))
    return weight * torch.mean(entropy)

def surface_area_loss(occupancy_grid, weight=0.05):  # ← NEW!
    """Prefer minimal surface (reduces noise)"""
    grad_x = torch.diff(occupancy_grid, dim=0, prepend=occupancy_grid[:1])
    grad_y = torch.diff(occupancy_grid, dim=1, prepend=occupancy_grid[:, :1])
    grad_z = torch.diff(occupancy_grid, dim=2, prepend=occupancy_grid[:, :, :1])
    grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-6)
    return weight * torch.mean(grad_norm)

# ============================================================================
# TRAINING / OPTIMIZATION
# ============================================================================

def optimize_single_subject(model, slices_2d, contours_2d, config, num_steps=1500):
    """Optimize INR for single subject given 2D contours"""
    
    device = torch.device(config['device'])
    model = model.to(device)
    
    # Use HIGHER initial learning rate
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'] * 10)
    
    # Add learning rate scheduler (cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_steps,
        eta_min=config['learning_rate'] / 100
    )

    losses = []  # ← ADD THIS LINE

    # AFTER (✅ STRONG - MULTI-TERM)
    for step in range(num_steps):
        optimizer.zero_grad()
        
        occupancy_grid = model.sample_grid(
            resolution=64,
            device=device,
            requires_grad=True
)        
        # Multi-term loss - all significant
        loss_projection = 0.0
        contours_device = contours_2d.to(device)  # ← ADD THIS LINE

        for view_idx in range(contours_2d.shape[0]):
            loss_projection += contour_reprojection_loss(occupancy_grid, contours_device[view_idx])
        
        loss_smooth = laplacian_smoothness_loss(occupancy_grid, weight=0.2)
        loss_entropy = volume_entropy_loss(occupancy_grid, weight=0.1)
        loss_area = surface_area_loss(occupancy_grid, weight=0.05)
        
        loss = loss_projection + loss_smooth + loss_entropy + loss_area
        
        loss.backward()
        optimizer.step()
        scheduler.step()  # New: learning rate decay

        losses.append(loss.item())
        
        if step % 50 == 0:
            print(f"  Step {step}/{num_steps}: loss={loss.item():.6f}")
    
    return model, losses


def evaluate_subject(model, occupancy_grid, contours_2d, device):
    """Compute metrics for single subject"""
    metrics = {}
    
    # ====== FIX: Move contours to same device as occupancy_grid ======
    contours_2d_device = contours_2d.to(device)
    
    # Project 3D grid to 2D for comparison
    # Project occupancy_grid via max-pooling (same as loss function)
    pred_projection = torch.max(occupancy_grid, dim=0)[0]  # (64, 64)
    
    # Resize projection to match contour size (256, 256)
    pred_projection_resized = F.interpolate(
        pred_projection.unsqueeze(0).unsqueeze(0),
        size=contours_2d_device[0].shape,  # (256, 256)
        mode='bilinear',
        align_corners=False
    ).squeeze()
    
    # Binarize
    pred_binary = (pred_projection_resized > 0.5).float()
    target_binary = (contours_2d_device[0] > 0.5).float()
    
    # Compute metrics
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
                occupancy_grid = model.sample_grid(resolution=64, device=device, requires_grad=False)
                metrics = evaluate_subject(model, occupancy_grid, contours_2d, device)
            
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
