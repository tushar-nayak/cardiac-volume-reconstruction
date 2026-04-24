#!/usr/bin/env python3
"""
Echo3D-style minimal starter (Option B)
- Adds per-view pose parameters
- Adds differentiable slice projection from INR using planar sampling
- Alternating optimization: shape <-> pose
- Drop-in replacement for your previous script
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path
import os
import traceback

DATA_PATH = Path(
    os.getenv(
        "CARDIAC_DATA_PATH",
        str(Path(__file__).resolve().parents[1] / "cap-mitea" / "mitea"),
    )
).expanduser()

# ==========================
# CONFIG
# ==========================
CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_path': DATA_PATH,
    'checkpoint_path': Path('./checkpoints'),
    'num_views': 3,
    'image_size': 256,
    'hidden_dim': 64,
    'num_inr_layers': 4,
    'learning_rate': 1e-5,  # Lower learning rate for stability
    'pose_learning_rate': 1e-6,  # Lower learning rate for pose
    'num_optimization_steps': 1200,
    'batch_size': 1,
    'alternate_every': 20,  # Alternate optimization every 20 steps
    'proj_resolution': 256,  # Projection resolution
}

# ==========================
# MODELS
# ==========================
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
        """
        coords: (..., 3) in [-1,1]
        returns: (..., 1) occupancy
        """
        orig_shape = coords.shape
        coords_flat = coords.view(-1, 3)
        pe_coords = self.pe.encode(coords_flat)  # (N, 24)
        out = torch.sigmoid(self.mlp(pe_coords))  # (N,1)
        return out.view(*orig_shape[:-1], 1)
       
    def sample_grid(self, resolution=64, device='cpu', requires_grad=False):
        """Sample occupancy on 3D grid (resolution^3)"""
        resolution = int(resolution)
        if isinstance(device, str):
            device = torch.device(device)
        linspace = torch.linspace(-1, 1, resolution, device=device, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(linspace, linspace, linspace, indexing='ij'), dim=-1)  # (R,R,R,3)
        if requires_grad:
            occupancy = self.forward(grid).squeeze(-1)
        else:
            with torch.no_grad():
                occupancy = self.forward(grid).squeeze(-1)
        return occupancy


class PoseParameters(nn.Module):
    """
    Learnable per-view poses: Euler angles (rx, ry, rz) in radians and translations (tx, ty, tz)
    Initialization near identity (small noise).
    """
    def __init__(self, num_views, init_sigma=1e-4):  # Slightly larger noise
        super().__init__()
        # Correct initialization of pose parameters
        init = torch.zeros(num_views, 6)  # Initialize `init` as zeros first
        init += init_sigma * torch.randn_like(init)  # Add small random noise
        self.pose = nn.Parameter(init)  # learnable parameter

    def get_matrices(self, device=None):
        """Return (num_views, 4, 4) extrinsic matrices mapping from plane coordinates -> world coordinates."""
        if device is None:
            device = self.pose.device
        p = self.pose.to(device)
        rx, ry, rz = p[:, 0], p[:, 1], p[:, 2]
        tx, ty, tz = p[:, 3], p[:, 4], p[:, 5]

        # Compute rotation matrices from Euler angles (Z * Y * X)
        cx, sx = torch.cos(rx), torch.sin(rx)
        cy, sy = torch.cos(ry), torch.sin(ry)
        cz, sz = torch.cos(rz), torch.sin(rz)

        R = torch.zeros((p.shape[0], 3, 3), device=device)
        R[:,0,0] = cz * cy
        R[:,0,1] = cz * sy * sx - sz * cx
        R[:,0,2] = cz * sy * cx + sz * sx
        R[:,1,0] = sz * cy
        R[:,1,1] = sz * sy * sx + cz * cx
        R[:,1,2] = sz * sy * cx - cz * sx
        R[:,2,0] = -sy
        R[:,2,1] = cy * sx
        R[:,2,2] = cy * cx

        extrinsics = torch.eye(4, device=device).unsqueeze(0).repeat(p.shape[0], 1, 1)
        extrinsics[:, :3, :3] = R
        extrinsics[:, :3, 3] = torch.stack([tx, ty, tz], dim=-1)
        return extrinsics



# ==========================
# DATA LOADING (MITEA)
# ==========================
def find_mitea_image_files(data_path):
    images_dir = data_path / 'images'
    labels_dir = data_path / 'labels'
   
    if not images_dir.exists() or not labels_dir.exists():
        print(f"ERROR: Expected images/ and labels/ subdirectories")
        return []
   
    image_files = sorted(images_dir.glob('*.nii*'))
    print(f"Found {len(image_files)} image files in {images_dir}")
   
    pairs = []
    for img_file in image_files:
        stem = img_file.stem
        label_candidates = list(labels_dir.glob(f'{stem}*'))
        if label_candidates:
            pairs.append((img_file, label_candidates[0]))
    return pairs


# def load_mitea_subject(img_file, label_file):
#     try:
#         import nibabel as nib
#     except ImportError:
#         print("ERROR: nibabel not installed. Run: pip install nibabel")
#         return None, None
   
#     try:
#         vol = torch.tensor(nib.load(img_file).get_fdata(), dtype=torch.float32)
#         seg = torch.tensor(nib.load(label_file).get_fdata(), dtype=torch.float32)
#         vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
#         print("Raw Segmentation Mask (before thresholding):")
#         print(seg)

#         seg = (seg > 0.5).float()
#         print ("SEG")
#         print(seg)
#         return vol, seg
#     except Exception as e:
#         print(f"  Error loading {img_file.name}: {e}")
#         return None, None
def load_mitea_subject(img_file, label_file):
    try:
        import nibabel as nib
    except ImportError:
        print("ERROR: nibabel not installed. Run: pip install nibabel")
        return None, None
   
    try:
        # Load NIfTI files using nibabel
        vol = torch.tensor(nib.load(img_file).get_fdata(), dtype=torch.float32)
        seg = torch.tensor(nib.load(label_file).get_fdata(), dtype=torch.float32)
        
        # Normalize the volume (this is common for image data)
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)

        # Debugging: Check if the volume and segmentation are non-zero
        print("Volume (before normalization):")
        print(f"Min value: {vol.min()}, Max value: {vol.max()}, Mean value: {vol.mean()}")
        
        print("Segmentation (before thresholding):")
        print(f"Min value: {seg.min()}, Max value: {seg.max()}, Mean value: {seg.mean()}")
        
        # If the segmentation or volume is all zeros, output an error
        if torch.all(vol == 0):
            print(f"WARNING: Volume {img_file} contains all zeros.")
        
        if torch.all(seg == 0):
            print(f"WARNING: Segmentation {label_file} contains all zeros.")
        
        # Threshold the segmentation to make it binary (0 or 1)
        seg = (seg > 0.1).float()
        
        # Additional debug: Check the thresholded segmentation
        print("Segmentation after thresholding:")
        print(f"Min value: {seg.min()}, Max value: {seg.max()}, Mean value: {seg.mean()}")

        # Return the processed volume and segmentation
        return vol, seg
    except Exception as e:
        print(f"Error loading {img_file.name}: {e}")
        return None, None


def extract_synthetic_2d_slices(vol, seg, num_views=6, out_size=256):
    D, H, W = vol.shape
    slices_2d = []
    contours_2d = []
    plane_indices = [
        D // 4, D // 2, D // 10, D // 3, 2*D // 3, D // 6,
    ][:num_views]
   
    for idx in plane_indices:
        slice_img = vol[idx, :, :]
        slice_seg = seg[idx, :, :]
        slice_img = F.interpolate(slice_img.unsqueeze(0).unsqueeze(0), size=(out_size, out_size), mode='bilinear').squeeze()
        slice_seg = F.interpolate(slice_seg.unsqueeze(0).unsqueeze(0), size=(out_size, out_size), mode='nearest').squeeze()
        slices_2d.append(slice_img)
        contours_2d.append(slice_seg)
    return torch.stack(slices_2d), torch.stack(contours_2d)


# ==========================
# LOSSES
# ==========================
def contour_bce_loss(pred, target):
    """pred and target: (H,W) with values in [0,1]"""
    pred = pred.clamp(1e-6, 1-1e-6)
    return F.binary_cross_entropy(pred, target)


def laplacian_smoothness_loss(occupancy_grid, weight=0.02):
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
    if surface_mask.any():
        loss = torch.mean(lap[surface_mask] ** 2)
    else:
        loss = torch.mean(lap ** 2)
    return weight * loss

def volume_entropy_loss(occupancy_grid, weight=0.01):
    eps = 1e-6
    entropy = -(occupancy_grid * torch.log(occupancy_grid + eps) + (1 - occupancy_grid) * torch.log(1 - occupancy_grid + eps))
    return weight * torch.mean(entropy)

def surface_area_loss(occupancy_grid, weight=0.005):
    grad_x = torch.diff(occupancy_grid, dim=0, prepend=occupancy_grid[:1])
    grad_y = torch.diff(occupancy_grid, dim=1, prepend=occupancy_grid[:, :1])
    grad_z = torch.diff(occupancy_grid, dim=2, prepend=occupancy_grid[:, :, :1])
    grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-6)
    return weight * torch.mean(grad_norm)


# ==========================
# DIFFERENTIABLE PROJECTION
# ==========================
def project_slice_from_inr(model, pose_matrix, resolution=256, device='cpu'):
    lin = torch.linspace(-1, 1, resolution, device=device)
    xv, yv = torch.meshgrid(lin, lin, indexing='ij')  # (H,W)
    zv = torch.zeros_like(xv)
    ones = torch.ones_like(xv)
    pts_plane = torch.stack([xv, yv, zv, ones], dim=-1)  # (H,W,4)

    # Apply pose (plane -> world)
    pts_world = torch.matmul(pts_plane, pose_matrix.T)
    pts_world = pts_world[..., :3]  # (H,W,3)

    coords = pts_world.view(-1, 3)
    coords = coords.to(next(model.parameters()).device)

    with torch.set_grad_enabled(True):
        batch = 4096
        out_chunks = []
        for i in range(0, coords.shape[0], batch):
            c = coords[i:i+batch]
            out = model(c)  # (N,1)
            out_chunks.append(out)
        occ = torch.cat(out_chunks, dim=0).view(resolution, resolution)  # (H,W)
    return occ


# ==========================
# TRAINING / OPTIMIZATION
# ==========================

def optimize_single_subject(model, slices_2d, contours_2d, pose_layer, config, num_steps=1500):
    device = torch.device(config['device'])
    model = model.to(device)
    pose_layer = pose_layer.to(device)

    optimizer_shape = optim.Adam(model.parameters(), lr=config['learning_rate'])
    optimizer_pose  = optim.Adam(pose_layer.parameters(), lr=config['pose_learning_rate'])

    scheduler_shape = optim.lr_scheduler.CosineAnnealingLR(optimizer_shape, T_max=num_steps, eta_min=config['learning_rate']/10)
    scheduler_pose = optim.lr_scheduler.CosineAnnealingLR(optimizer_pose, T_max=num_steps, eta_min=config['pose_learning_rate']/10)

    losses = []
    contours_device = contours_2d.to(device)

    for step in range(num_steps):
        # Compute projections for each view
        extrinsics = pose_layer.get_matrices(device=device)  # (V,4,4)
        loss_projection = 0.0
        preds = []
        for v in range(contours_device.shape[0]):
            pred = project_slice_from_inr(model, extrinsics[v], resolution=config['proj_resolution'], device=device)
            preds.append(pred)
            target = contours_device[v]
            loss_projection += contour_bce_loss(pred, target)

        # Compute smoothness, entropy, and surface area losses
        occ_grid = model.sample_grid(resolution=32, device=device, requires_grad=True)  # (32,32,32)
        loss_smooth = laplacian_smoothness_loss(occ_grid, weight=0.02)
        loss_entropy = volume_entropy_loss(occ_grid, weight=0.01)
        loss_area = surface_area_loss(occ_grid, weight=0.005)

        # Total loss
        loss = loss_projection + loss_smooth + loss_entropy + loss_area

        # Alternating update
        if (step // config.get('alternate_every', 1)) % 2 == 0:
            optimizer_shape.zero_grad()
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients for the model
            torch.nn.utils.clip_grad_norm_(pose_layer.parameters(), max_norm=1.0)  # Clip gradients for the pose parameters

            optimizer_shape.step()
            scheduler_shape.step()
        else:
            optimizer_pose.zero_grad()
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients for the model
            torch.nn.utils.clip_grad_norm_(pose_layer.parameters(), max_norm=1.0)  # Clip gradients for the pose parameters

            optimizer_pose.step()
            scheduler_pose.step()

        # Track loss
        losses.append(loss.item())

        # Log projection loss and pose parameters at regular intervals
        if step % 50 == 0:
            print(f"Step {step}/{num_steps}: total loss={loss.item():.6f}, projection loss={loss_projection.item():.6f}")

        # Debugging: Log pose parameters at regular intervals
        if step % 100 == 0:
            print(f"Pose parameters at step {step}: {pose_layer.pose.detach().cpu().numpy()}")

        # Debugging: Log the range of predicted projections and contours every 100 steps
        if step % 100 == 0:
            print(f"Predicted projection at step {step}: {preds[0].detach().cpu().numpy()}")
            print(f"Ground truth contours at step {step}: {contours_device[0].detach().cpu().numpy()}")

        # Optionally: visualize predictions and ground truth at every N steps
        if step % 200 == 0:
            import matplotlib.pyplot as plt
            plt.subplot(1, 2, 1)
            plt.imshow(preds[0].detach().cpu().numpy(), cmap='gray')
            plt.title(f"Predicted (step {step})")
            plt.subplot(1, 2, 2)
            plt.imshow(contours_device[0].detach().cpu().numpy(), cmap='gray')
            plt.title(f"Ground Truth (step {step})")
            plt.show()

    return model, pose_layer, losses

# ==========================
# EVALUATION
# ==========================
def evaluate_subject(model, contours_2d, pose_layer, config):
    device = torch.device(config['device'])
    model = model.to(device)
    contours_device = contours_2d.to(device)
    extrinsics = pose_layer.get_matrices(device=device)

    dices = []
    ious = []
    for v in range(contours_device.shape[0]):
        pred = project_slice_from_inr(model, extrinsics[v], resolution=config['proj_resolution'], device=device)
        pred_binary = (pred > 0.5).float()
        target_binary = (contours_device[v] > 0.5).float()

        intersection = torch.sum(pred_binary * target_binary)
        union = torch.sum(pred_binary) + torch.sum(target_binary)
        dice = (2 * intersection) / (union + 1e-6)
        iou = intersection / (union - intersection + 1e-6)

        dices.append(dice.item())
        ious.append(iou.item())

    metrics = {'dice': float(np.mean(dices)), 'iou': float(np.mean(ious))}
    return metrics


# ==========================
# MAIN PIPELINE
# ==========================
def main():
    device = torch.device(CONFIG['device'])
    print(f"Device: {device}")
    print(f"Data path: {CONFIG['data_path']}")

    CONFIG['checkpoint_path'].mkdir(exist_ok=True)

    # Initialize model
    model = ImplicitNeuralRepresentation(hidden_dim=CONFIG['hidden_dim'], num_layers=CONFIG['num_inr_layers'])

    data_path = CONFIG['data_path']
    if not data_path.exists():
        print(f"ERROR: MITEA data not found at {data_path}")
        return

    print("\n" + "="*60)
    print("FINDING MITEA SUBJECTS")
    print("="*60)
    subject_pairs = find_mitea_image_files(data_path)[:10]

    if not subject_pairs:
        print(f"\nERROR: No image-label pairs found")
        return

    print(f"\nTotal subjects to process: {len(subject_pairs)}\n")
    all_metrics = {'dice': [], 'iou': []}

    for subject_idx, (img_file, label_file) in enumerate(subject_pairs):
        subject_id = img_file.stem
        print(f"\n[{subject_idx + 1}/{len(subject_pairs)}] Processing {subject_id}...")

        try:
            vol, seg = load_mitea_subject(img_file, label_file)
            if vol is None:
                print(f"  Skipping {subject_id}")
                continue
            print(f"  Loaded: vol shape={vol.shape}, seg shape={seg.shape}")

            slices_2d, contours_2d = extract_synthetic_2d_slices(vol, seg, num_views=CONFIG['num_views'], out_size=CONFIG['image_size'])
            print(f"  Extracted: slices={slices_2d.shape}, contours={contours_2d.shape}")

            # Reset model & pose for this subject
            model = ImplicitNeuralRepresentation(hidden_dim=CONFIG['hidden_dim'], num_layers=CONFIG['num_inr_layers'])
            pose_layer = PoseParameters(CONFIG['num_views'])

            # Optimize
            print(f"  Optimizing INR + poses (alternating)...")
            model, pose_layer, losses = optimize_single_subject(model, slices_2d, contours_2d, pose_layer, CONFIG, num_steps=CONFIG['num_optimization_steps'])

            # Evaluate
            metrics = evaluate_subject(model, contours_2d, pose_layer, CONFIG)
            print(f"  Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}")
            all_metrics['dice'].append(metrics['dice'])
            all_metrics['iou'].append(metrics['iou'])

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

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
