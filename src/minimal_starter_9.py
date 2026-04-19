#!/usr/bin/env python3
"""
Hybrid Echo3D minimal starter (Hybrid C, pro version)
-----------------------------------------------------

Key features:
- STRICT non-empty slice selection with *stratified* z-coverage.
- Alternating optimization of INR (shape) and PoseParameters (pose).
- Differentiable planar sampling projection from INR -> 2D slices.
- Strong regularizers (laplacian, entropy, surface area) for shape steps.
- Consistent z-depth mapping in training and evaluation.
- Canonical pose by default (learn_pose = False) for aligned datasets like MITEA.
- Optional pose learning + L2 pose regularization when learn_pose = True.
- Optional hybrid 2D + 3D volumetric supervision using the full segmentation.
- Optional cropping to the heart's bounding box for better 3D focus.
- 3D evaluation:
    * Full-volume Dice / IoU
    * "Central" Dice / IoU restricted to z-range of supervised slices.
- 2D HTML overlays:
    red   = prediction only
    green = ground truth only
    yellow = overlap
- 3D mesh HTML overlays:
    red = prediction
    green = ground truth
"""

import os
import traceback
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import nibabel as nib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------
# CONFIG
# ----------------------------
CONFIG: Dict = {
    # device
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # data paths
    "data_path": Path("/home/sofa/host_dir/cap-mitea/mitea"),
    "checkpoint_path": Path("./checkpoints"),

    # dataset control
    # None or <=0 = use all pairs; otherwise limit to first N subjects
    "max_subjects": 10,

    # views / slices
    "num_views": 6,
    "min_mask_pixels": 100,
    # "topk" = previous behavior (largest masks),
    # "stratified" = spread indices across the non-empty z-range (recommended)
    "slice_selection_strategy": "stratified",

    # network + training
    "image_size": 256,
    "hidden_dim": 64,
    "num_inr_layers": 4,
    "learning_rate": 1e-3,
    "pose_learning_rate": 1e-4,
    "num_optimization_steps": 1000,
    "alternate_every": 20,       # alternate shape/pose every N steps
    "proj_resolution": 256,      # projection resolution (H,W)

    # pose behavior
    # For MITEA (already aligned), default to False for canonical z alignment.
    "learn_pose": False,
    "pose_reg_weight": 1e-3,     # L2 regularization on pose (only if learn_pose=True)

    # ===== Hybrid 3D volumetric supervision =====
    # 0.0 → original behavior (2D slices only); >0 → add volumetric BCE term
    "vol_supervision_weight": 1.0,
    # number of random 3D points per shape step for volumetric BCE
    "vol_supervision_samples": 32768,

    # sampling / grid
    "grid_resolution": 64,       # for regularizers (Laplace / entropy / area)
    "eval_grid_resolution": 128, # for 3D volume metrics + mesh export

    # ===== Cropping to heart bounding box =====
    # If True, crop volume + segmentation to seg>0 bounding box (+ margin)
    "crop_to_seg_bbox": True,
    "crop_margin": 2,            # voxels of padding around the heart bbox

    # visualization
    "html_output_path": Path("./checkpoints/html_visualizations"),

    # misc
    "print_every": 50,
    "save_nifti": True,
    "mesh_threshold": 0.5,       # threshold for occupancy -> mesh
}

CONFIG["checkpoint_path"].mkdir(exist_ok=True, parents=True)
CONFIG["html_output_path"].mkdir(exist_ok=True, parents=True)


# ----------------------------
# Utilities: file discovery
# ----------------------------
def find_mitea_image_files(data_path: Path) -> List[Tuple[Path, Path]]:
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        print("ERROR: Expected images/ and labels/ subdirectories under", data_path)
        if images_dir.exists():
            print(" images/ exists; contents:", list(images_dir)[:5])
        if labels_dir.exists():
            print(" labels/ exists; contents:", list(labels_dir)[:5])
        return []

    image_files = sorted(images_dir.glob("*.nii*"))
    pairs: List[Tuple[Path, Path]] = []
    for img in image_files:
        stem = img.stem
        candidates = list(labels_dir.glob(f"{stem}*"))
        if candidates:
            pairs.append((img, candidates[0]))
        else:
            if stem.endswith(".nii"):
                alt = stem[:-4]
                candidates2 = list(labels_dir.glob(f"{alt}*"))
                if candidates2:
                    pairs.append((img, candidates2[0]))
    print(f"Found {len(pairs)} image-label pairs in {images_dir}")
    return pairs


# ----------------------------
# Models
# ----------------------------
class PositionalEncoding:
    def __init__(self, num_freqs: int = 4):
        self.num_freqs = num_freqs

    def encode(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (N,3) or (...,3)
        pe = []
        for i in range(self.num_freqs):
            freq = 2.0 ** i
            pe.append(torch.sin(freq * np.pi * coords))
            pe.append(torch.cos(freq * np.pi * coords))
        return torch.cat(pe, dim=-1)


class ImplicitNeuralRepresentation(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_layers: int = 4, pe_freqs: int = 4):
        super().__init__()
        self.pe = PositionalEncoding(num_freqs=pe_freqs)
        input_dim = 3 * 2 * pe_freqs
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else 1
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        orig_shape = coords.shape
        coords_flat = coords.view(-1, 3)
        pe = self.pe.encode(coords_flat)
        out = torch.sigmoid(self.mlp(pe))
        return out.view(*orig_shape[:-1], 1)

    def sample_grid(
        self,
        resolution: int = 64,
        device=None,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        resolution = int(resolution)
        if isinstance(device, str):
            device = torch.device(device)
        if device is None:
            device = torch.device("cpu")
        lin = torch.linspace(-1, 1, resolution, device=device, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), dim=-1)  # (R,R,R,3)
        if requires_grad:
            occ = self.forward(grid).squeeze(-1)
        else:
            with torch.no_grad():
                occ = self.forward(grid).squeeze(-1)
        return occ


class PoseParameters(nn.Module):
    """
    Per-view Euler (rx,ry,rz) and translation (tx,ty,tz).
    Initialized near identity with small noise.
    """
    def __init__(self, num_views: int, init_sigma: float = 1e-4):
        super().__init__()
        init = torch.zeros(num_views, 6)
        init += init_sigma * torch.randn_like(init)
        self.pose = nn.Parameter(init)

    def get_matrices(self, device=None) -> torch.Tensor:
        if device is None:
            device = self.pose.device
        p = self.pose.to(device)
        rx, ry, rz = p[:, 0], p[:, 1], p[:, 2]
        tx, ty, tz = p[:, 3], p[:, 4], p[:, 5]

        cx, sx = torch.cos(rx), torch.sin(rx)
        cy, sy = torch.cos(ry), torch.sin(ry)
        cz, sz = torch.cos(rz), torch.sin(rz)

        R = torch.zeros((p.shape[0], 3, 3), device=device)
        R[:, 0, 0] = cz * cy
        R[:, 0, 1] = cz * sy * sx - sz * cx
        R[:, 0, 2] = cz * sy * cx + sz * sx
        R[:, 1, 0] = sz * cy
        R[:, 1, 1] = sz * sy * sx + cz * cx
        R[:, 1, 2] = sz * sy * cx - cz * sx
        R[:, 2, 0] = -sy
        R[:, 2, 1] = cy * sx
        R[:, 2, 2] = cy * cx

        extrinsics = torch.eye(4, device=device).unsqueeze(0).repeat(p.shape[0], 1, 1)
        extrinsics[:, :3, :3] = R
        extrinsics[:, :3, 3] = torch.stack([tx, ty, tz], dim=-1)
        return extrinsics


# ----------------------------
# Data loading + strict slice selection
# ----------------------------
def load_mitea_subject(img_file: Path, label_file: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    vol = torch.tensor(nib.load(str(img_file)).get_fdata(), dtype=torch.float32)
    seg = torch.tensor(nib.load(str(label_file)).get_fdata(), dtype=torch.float32)

    # Normalize volume to [0,1]
    if vol.max() - vol.min() > 0:
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
    else:
        vol = torch.zeros_like(vol)

    print("Volume (before normalization):")
    print(f" Min: {vol.min().item()}, Max: {vol.max().item()}, Mean: {vol.mean().item()}")
    print("Segmentation (raw):")
    print(f" Min: {seg.min().item()}, Max: {seg.max().item()}, Mean: {seg.mean().item()}")

    # Binarize seg (LV/RV foreground)
    seg_bin = (seg > 0).float()
    print("Segmentation (binarized):")
    print(f" Min: {seg_bin.min().item()}, Max: {seg_bin.max().item()}, Mean: {seg_bin.mean().item()}")
    return vol, seg_bin


def crop_to_seg_bbox(
    vol: torch.Tensor,
    seg: torch.Tensor,
    margin: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int, int, int, int]]:
    """
    Crop volume and segmentation to a tight bounding box around seg>0,
    with 'margin' voxels of padding in each direction.

    Returns:
        vol_cropped, seg_cropped, bbox=(dmin,dmax,hmin,hmax,wmin,wmax)
    """
    mask = seg > 0
    if not mask.any():
        # nothing to crop, return as-is
        D, H, W = vol.shape
        bbox = (0, D - 1, 0, H - 1, 0, W - 1)
        print("No foreground in segmentation, skipping cropping.")
        return vol, seg, bbox

    nz = mask.nonzero(as_tuple=False)
    dmin, hmin, wmin = nz.min(dim=0).values.tolist()
    dmax, hmax, wmax = nz.max(dim=0).values.tolist()

    dmin = max(dmin - margin, 0)
    hmin = max(hmin - margin, 0)
    wmin = max(wmin - margin, 0)
    dmax = min(dmax + margin, vol.shape[0] - 1)
    hmax = min(hmax + margin, vol.shape[1] - 1)
    wmax = min(wmax + margin, vol.shape[2] - 1)

    vol_c = vol[dmin : dmax + 1, hmin : hmax + 1, wmin : wmax + 1].contiguous()
    seg_c = seg[dmin : dmax + 1, hmin : hmax + 1, wmin : wmax + 1].contiguous()
    bbox = (dmin, dmax, hmin, hmax, wmin, wmax)

    print(f" Cropped to bbox {bbox}; new vol shape {vol_c.shape}")
    return vol_c, seg_c, bbox


def select_strict_slices(
    seg: torch.Tensor,
    num_views: int,
    min_pixels: int = 100,
    strategy: str = "stratified",
) -> List[int]:
    """
    Return indices of axial slices (along D) with >= min_pixels
    foreground pixels.

    Strategies:
        - "topk":      strictly take the top-K slices by mask area.
        - "stratified": spread K slices across the non-empty z-range
                        (recommended: better 3D coverage).
    Strict: must return exactly num_views or raise error.
    """
    D = seg.shape[0]
    seg_cont = seg.contiguous()
    counts = seg_cont.reshape(D, -1).sum(dim=1)
    counts_np = counts.cpu().numpy()

    valid_idxs = np.where(counts_np >= min_pixels)[0]
    if len(valid_idxs) < num_views:
        raise ValueError(
            f"STRICT mode: not enough non-empty slices (found {len(valid_idxs)}, required {num_views}). "
            f"Try lowering min_pixels or use a different subject."
        )

    strategy = strategy.lower()
    if strategy == "topk":
        sorted_idxs = valid_idxs[np.argsort(-counts_np[valid_idxs])]
        chosen = sorted_idxs[:num_views].tolist()
        chosen_sorted = sorted(chosen)
    elif strategy == "stratified":
        # Spread indices across the valid z-range by indexing valid_idxs
        lin_idx = np.linspace(0, len(valid_idxs) - 1, num_views, dtype=int)
        chosen_sorted = sorted(valid_idxs[lin_idx].tolist())
    else:
        raise ValueError(f"Unknown slice_selection_strategy: {strategy}")

    print(
        f"Strict-selected slice indices (top {num_views}, strategy={strategy}): "
        f"{chosen_sorted}"
    )
    return chosen_sorted


def extract_synthetic_2d_slices_strict(
    vol: torch.Tensor,
    seg: torch.Tensor,
    num_views: int,
    min_pixels: int,
    out_size: int = 256,
    strategy: str = "stratified",
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Strict selection: pick exactly num_views axial indices with seg pixels >= min_pixels.
    Returns: slices_2d (V,H,W), contours_2d (V,H,W), chosen_indices (list of axial indices)
    """
    vol = vol.contiguous()
    seg = seg.contiguous()

    D, H, W = vol.shape
    chosen = select_strict_slices(
        seg,
        num_views=num_views,
        min_pixels=min_pixels,
        strategy=strategy,
    )
    slices = []
    contours = []
    for idx in chosen:
        slice_img = vol[idx, :, :].contiguous()
        slice_seg = seg[idx, :, :].contiguous()

        slice_img_resized = F.interpolate(
            slice_img.unsqueeze(0).unsqueeze(0),
            size=(out_size, out_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        slice_seg_resized = F.interpolate(
            slice_seg.unsqueeze(0).unsqueeze(0),
            size=(out_size, out_size),
            mode="nearest",
        ).squeeze()

        slices.append(slice_img_resized)
        contours.append(slice_seg_resized)

    slices_tensor = torch.stack(slices)
    contours_tensor = torch.stack(contours)
    return slices_tensor, contours_tensor, chosen


# ----------------------------
# Losses / Regularizers
# ----------------------------
def contour_bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.clamp(1e-6, 1 - 1e-6)
    return F.binary_cross_entropy(pred, target)


def laplacian_smoothness_loss(occupancy_grid: torch.Tensor, weight: float = 0.02) -> torch.Tensor:
    lap = (
        torch.roll(occupancy_grid, 1, dims=0)
        + torch.roll(occupancy_grid, -1, dims=0)
        + torch.roll(occupancy_grid, 1, dims=1)
        + torch.roll(occupancy_grid, -1, dims=1)
        + torch.roll(occupancy_grid, 1, dims=2)
        + torch.roll(occupancy_grid, -1, dims=2)
        - 6 * occupancy_grid
    )
    surface_mask = (occupancy_grid > 0.2) & (occupancy_grid < 0.8)
    if surface_mask.any():
        loss = torch.mean(lap[surface_mask] ** 2)
    else:
        loss = torch.mean(lap ** 2)
    return weight * loss


def volume_entropy_loss(occupancy_grid: torch.Tensor, weight: float = 0.01) -> torch.Tensor:
    eps = 1e-6
    entropy = -(
        occupancy_grid * torch.log(occupancy_grid + eps)
        + (1 - occupancy_grid) * torch.log(1 - occupancy_grid + eps)
    )
    return weight * torch.mean(entropy)


def surface_area_loss(occupancy_grid: torch.Tensor, weight: float = 0.005) -> torch.Tensor:
    grad_x = torch.diff(occupancy_grid, dim=0, prepend=occupancy_grid[:1])
    grad_y = torch.diff(occupancy_grid, dim=1, prepend=occupancy_grid[:, :1])
    grad_z = torch.diff(occupancy_grid, dim=2, prepend=occupancy_grid[:, :, :1])
    grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-6)
    return weight * torch.mean(grad_norm)


# ----------------------------
# Differentiable projection (plane sampling)
# ----------------------------
def project_slice_from_inr(
    model: ImplicitNeuralRepresentation,
    pose_matrix: torch.Tensor,
    resolution: int = 256,
    device="cpu",
    z_depth: float = 0.0,
) -> torch.Tensor:
    """
    Sample a planar grid in plane coordinates (x,y,z=z_depth), then map via pose_matrix (4x4)
    to world coords and query model. Return occupancy map (H,W) in [0,1].

    pose_matrix: (4,4) torch tensor (plane -> world)
    """
    if isinstance(device, str):
        device = torch.device(device)

    lin = torch.linspace(-1.0, 1.0, resolution, device=device, dtype=torch.float32)
    xv, yv = torch.meshgrid(lin, lin, indexing="ij")
    zv = torch.full_like(xv, fill_value=z_depth)
    ones = torch.ones_like(xv)

    pts_plane = torch.stack([xv, yv, zv, ones], dim=-1)  # (H,W,4)
    pts_world = pts_plane @ pose_matrix.T                # (H,W,4)
    pts_world = pts_world[..., :3]                       # (H,W,3)
    coords = pts_world.view(-1, 3)

    coords = coords.to(next(model.parameters()).device)

    batch = 4096
    outs = []
    for i in range(0, coords.shape[0], batch):
        outs.append(model(coords[i : i + batch]))
    occ = torch.cat(outs, dim=0).view(resolution, resolution)
    return occ


# ----------------------------
# Optimization (alternating, hybrid 2D + 3D)
# ----------------------------
def optimize_single_subject(
    model: ImplicitNeuralRepresentation,
    slices_2d: torch.Tensor,
    contours_2d: torch.Tensor,
    pose_layer: PoseParameters,
    chosen: List[int],
    config: Dict,
    num_steps: int,
    D: int,
    seg_vol: torch.Tensor,
) -> Tuple[ImplicitNeuralRepresentation, PoseParameters, List[float]]:
    """
    Alternating optimization:
    - shape step: update INR weights using 2D proj + volumetric regularizers (+ optional 3D BCE)
    - pose step:  update pose parameters using proj-only loss (plus pose regularization)

    If config["learn_pose"] is False, only shape is optimized and pose is kept at identity.

    seg_vol:
        The (possibly cropped) segmentation volume (D,H,W) used for
        optional volumetric BCE supervision.
    """
    device = torch.device(config["device"])
    learn_pose: bool = bool(config.get("learn_pose", True))
    pose_reg_weight: float = float(config.get("pose_reg_weight", 0.0))

    # Hybrid volumetric supervision config
    vol_weight: float = float(config.get("vol_supervision_weight", 0.0))
    vol_samples: int = int(config.get("vol_supervision_samples", 0))

    model = model.to(device)
    pose_layer = pose_layer.to(device)
    seg_device = seg_vol.to(device)

    D_seg, H_seg, W_seg = seg_device.shape
    if D_seg != D:
        print(f"Warning: D ({D}) != seg_vol.shape[0] ({D_seg}), using seg_vol D.")
        D = D_seg

    optimizer_shape = optim.Adam(model.parameters(), lr=config["learning_rate"])

    if learn_pose:
        optimizer_pose = optim.Adam(pose_layer.parameters(), lr=config["pose_learning_rate"])
        scheduler_pose = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_pose,
            T_max=num_steps,
            eta_min=config["pose_learning_rate"] / 10,
        )
    else:
        optimizer_pose = None
        scheduler_pose = None

    scheduler_shape = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_shape, T_max=num_steps, eta_min=config["learning_rate"] / 10
    )

    contours_device = contours_2d.to(device)
    losses: List[float] = []

    for step in range(num_steps):
        if learn_pose:
            # Alternate blocks if we actually learn pose
            shape_step = ((step // config.get("alternate_every", 1)) % 2 == 0)
        else:
            # Canonical mode: only shape optimization
            shape_step = True

        if shape_step:
            # For shape step, we don't want gradients wrt pose
            extrinsics = pose_layer.get_matrices(device=device).detach()
        else:
            # For pose step, backprop into pose
            extrinsics = pose_layer.get_matrices(device=device)

        loss_projection = torch.tensor(0.0, device=device)
        preds_debug = []
        vol_loss = torch.tensor(0.0, device=device)

        for v in range(contours_device.shape[0]):
            slice_index = chosen[v]
            # Map slice index [0, D-1] -> z in [-1, 1] consistently
            z_depth = 2.0 * (slice_index / (D - 1)) - 1.0
            pred = project_slice_from_inr(
                model,
                extrinsics[v],
                resolution=config["proj_resolution"],
                device=device,
                z_depth=z_depth,
            )
            preds_debug.append(pred)
            target = contours_device[v]
            loss_projection = loss_projection + contour_bce_loss(pred, target)

        if shape_step:
            # ---- Optional 3D volumetric BCE supervision on random 3D points ----
            if vol_weight > 0.0 and vol_samples > 0:
                # sample coords in canonical cube [-1,1]^3
                coords3d = torch.rand(vol_samples, 3, device=device) * 2.0 - 1.0
                pred_vol = model(coords3d).squeeze(-1)  # (N,)

                # map [-1,1] -> voxel indices [0, size-1]
                x_norm = (coords3d[:, 0] + 1.0) * 0.5  # in [0,1]
                y_norm = (coords3d[:, 1] + 1.0) * 0.5
                z_norm = (coords3d[:, 2] + 1.0) * 0.5

                x_idx = torch.clamp((x_norm * (H_seg - 1)).long(), 0, H_seg - 1)
                y_idx = torch.clamp((y_norm * (W_seg - 1)).long(), 0, W_seg - 1)
                z_idx = torch.clamp((z_norm * (D_seg - 1)).long(), 0, D_seg - 1)

                gt_vol = seg_device[z_idx, x_idx, y_idx].view(-1)

                vol_loss = F.binary_cross_entropy(pred_vol, gt_vol)

            # ---- Volumetric regularizers ----
            occ_grid = model.sample_grid(
                resolution=config["grid_resolution"],
                device=device,
                requires_grad=True,
            )
            loss_smooth = laplacian_smoothness_loss(occ_grid, weight=0.02)
            loss_entropy = volume_entropy_loss(occ_grid, weight=0.01)
            loss_area = surface_area_loss(occ_grid, weight=0.005)

            loss = (
                loss_projection
                + loss_smooth
                + loss_entropy
                + loss_area
                + vol_weight * vol_loss  # volumetric supervision term (may be 0)
            )
        else:
            # Pose update: projection loss + optional L2 regularization
            loss = loss_projection
            if pose_reg_weight > 0.0:
                loss = loss + pose_reg_weight * (pose_layer.pose ** 2).mean()

        optimizer_shape.zero_grad()
        if learn_pose and optimizer_pose is not None:
            optimizer_pose.zero_grad()

        loss.backward()

        if shape_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_shape.step()
            scheduler_shape.step()
        else:
            if learn_pose and optimizer_pose is not None:
                torch.nn.utils.clip_grad_norm_(pose_layer.parameters(), max_norm=1.0)
                optimizer_pose.step()
                scheduler_pose.step()

        losses.append(loss.item())

        if step % config.get("print_every", 50) == 0 or step == num_steps - 1:
            if shape_step:
                print(
                    f"Step {step}/{num_steps}: "
                    f"total={loss.item():.6f}, proj={loss_projection.item():.6f}, "
                    f"vol={vol_loss.item():.6f}, shape-step"
                )
            else:
                print(
                    f"Step {step}/{num_steps}: "
                    f"total={loss.item():.6f}, proj={loss_projection.item():.6f}, "
                    f"pose-step"
                )

        if step % 200 == 0:
            with torch.no_grad():
                p = preds_debug[0].detach().cpu()
                t = contours_device[0].detach().cpu()
                print(
                    f" Pred range: min {p.min().item():.4f}, max {p.max().item():.4f}, "
                    f"mean {p.mean().item():.4f}"
                )
                print(
                    f" GT range:  min {t.min().item():.4f}, max {t.max().item():.4f}, "
                    f"mean {t.mean().item():.4f}"
                )

    return model, pose_layer, losses


# ----------------------------
# 2D evaluation (consistent z-depth)
# ----------------------------
def evaluate_subject_2d(
    model: ImplicitNeuralRepresentation,
    contours_2d: torch.Tensor,
    pose_layer: PoseParameters,
    chosen: List[int],
    D: int,
    config: Dict,
) -> Dict[str, float]:
    """
    2D contour-level Dice / IoU, using the *same* z-depth mapping as training.
    """
    device = torch.device(config["device"])
    model = model.to(device)
    contours = contours_2d.to(device)
    extrinsics = pose_layer.get_matrices(device=device)

    dices: List[float] = []
    ious: List[float] = []
    for v in range(contours.shape[0]):
        slice_index = chosen[v]
        z_depth = 2.0 * (slice_index / (D - 1)) - 1.0

        pred = project_slice_from_inr(
            model,
            extrinsics[v],
            resolution=config["proj_resolution"],
            device=device,
            z_depth=z_depth,
        )
        pred_binary = (pred > 0.5).float()
        target_binary = (contours[v] > 0.5).float()

        intersection = torch.sum(pred_binary * target_binary)
        union = torch.sum(pred_binary) + torch.sum(target_binary)
        dice = (2 * intersection) / (union + 1e-6)
        iou = intersection / (union - intersection + 1e-6)
        dices.append(dice.item())
        ious.append(iou.item())

    metrics = {"dice_2d": float(np.mean(dices)), "iou_2d": float(np.mean(ious))}
    return metrics


# ----------------------------
# 3D evaluation + mesh export
# ----------------------------
def evaluate_subject_3d_and_mesh(
    model: ImplicitNeuralRepresentation,
    seg_vol: torch.Tensor,
    subject_id: str,
    config: Dict,
    chosen: List[int],
) -> Dict[str, float]:
    """
    - Resample GT seg to eval_grid_resolution^3
    - Sample INR on same cube
    - Align INR grid ordering to seg (z,y,x)
    - Compute 3D Dice / IoU over:
        * full volume
        * central z-range corresponding to supervised slices
    - Export Plotly Mesh3d for pred vs GT (full volume)
    """
    device = torch.device(config["device"])
    model = model.to(device)
    R = int(config.get("eval_grid_resolution", 64))
    thr = float(config.get("mesh_threshold", 0.5))

    # Sample occupancy grid from INR: (x,y,z)
    occ_grid = model.sample_grid(resolution=R, device=device, requires_grad=False)
    occ_grid_cpu = occ_grid.detach().cpu()

    # Reorder to (z,y,x) to match seg_vol (D,H,W) -> (z,y,x)
    occ_grid_aligned = occ_grid_cpu.permute(2, 1, 0).contiguous()

    # Resample GT segmentation to same resolution
    seg_t = seg_vol.unsqueeze(0).unsqueeze(0).float()  # (1,1,D,H,W)
    seg_resampled = F.interpolate(seg_t, size=(R, R, R), mode="nearest")
    seg_resampled = seg_resampled[0, 0]  # (R,R,R) as (z,y,x)

    # 3D metrics on aligned grids (full volume)
    pred_bin = (occ_grid_aligned > thr).float()
    gt_bin = (seg_resampled > 0.5).float()

    intersection = (pred_bin * gt_bin).sum().item()
    union = pred_bin.sum().item() + gt_bin.sum().item()
    dice_full = (2.0 * intersection) / (union + 1e-6)
    iou_full = intersection / (union - intersection + 1e-6)

    # 3D metrics restricted to central supervised z-range
    D_orig = int(seg_vol.shape[0])
    zmin_orig = int(min(chosen))
    zmax_orig = int(max(chosen))

    # Map original z indices [0, D_orig-1] -> resampled [0, R-1]
    zmin_R = int(round(zmin_orig / (D_orig - 1) * (R - 1)))
    zmax_R = int(round(zmax_orig / (D_orig - 1) * (R - 1)))
    zmin_R = max(0, min(zmin_R, R - 1))
    zmax_R = max(0, min(zmax_R, R - 1))
    if zmax_R < zmin_R:
        zmin_R, zmax_R = zmax_R, zmin_R

    pred_central = pred_bin[zmin_R : zmax_R + 1]
    gt_central = gt_bin[zmin_R : zmax_R + 1]

    intersection_c = (pred_central * gt_central).sum().item()
    union_c = pred_central.sum().item() + gt_central.sum().item()
    dice_central = (2.0 * intersection_c) / (union_c + 1e-6)
    iou_central = intersection_c / (union_c - intersection_c + 1e-6)

    print(
        f"  3D Dice (full): {dice_full:.4f}, 3D IoU (full): {iou_full:.4f}"
    )
    print(
        f"  3D Dice (central supervised z-range): {dice_central:.4f}, "
        f"3D IoU (central): {iou_central:.4f}"
    )

    # 3D mesh visualization (if scikit-image is available) – full volume
    try:
        from skimage import measure
    except ImportError:
        print(
            "  scikit-image not installed, skipping 3D mesh export. "
            "Install via `pip install scikit-image` if needed."
        )
        return {
            "dice_3d": dice_full,
            "iou_3d": iou_full,
            "dice_3d_central": dice_central,
            "iou_3d_central": iou_central,
        }

    occ_np = occ_grid_aligned.numpy()
    seg_np = seg_resampled.numpy()

    # marching cubes
    try:
        verts_pred, faces_pred, _, _ = measure.marching_cubes(occ_np, level=thr)
        verts_gt, faces_gt, _, _ = measure.marching_cubes(seg_np, level=0.5)
    except ValueError as e:
        print("  Marching cubes failed (likely empty volume):", e)
        return {
            "dice_3d": dice_full,
            "iou_3d": iou_full,
            "dice_3d_central": dice_central,
            "iou_3d_central": iou_central,
        }

    # normalize to [0,1] cube for visualization
    def norm_verts(v: np.ndarray) -> np.ndarray:
        scale = np.array([R - 1, R - 1, R - 1], dtype=np.float32)
        return v / scale

    vp = norm_verts(verts_pred)
    vg = norm_verts(verts_gt)

    fig = go.Figure()

    # GT mesh in green
    fig.add_trace(
        go.Mesh3d(
            x=vg[:, 0],
            y=vg[:, 1],
            z=vg[:, 2],
            i=faces_gt[:, 0],
            j=faces_gt[:, 1],
            k=faces_gt[:, 2],
            color="green",
            opacity=0.5,
            name="Ground Truth",
        )
    )

    # Pred mesh in red
    fig.add_trace(
        go.Mesh3d(
            x=vp[:, 0],
            y=vp[:, 1],
            z=vp[:, 2],
            i=faces_pred[:, 0],
            j=faces_pred[:, 1],
            k=faces_pred[:, 2],
            color="red",
            opacity=0.5,
            name="Prediction",
        )
    )

    fig.update_layout(
        title=f"{subject_id} – 3D mesh: Prediction (red) vs GT (green)",
        scene=dict(aspectmode="data"),
        legend=dict(x=0.02, y=0.98),
    )

    html_file = CONFIG["html_output_path"] / f"{subject_id}_3d_mesh.html"
    fig.write_html(str(html_file))
    print(f"  3D mesh HTML saved: {html_file}")

    return {
        "dice_3d": dice_full,
        "iou_3d": iou_full,
        "dice_3d_central": dice_central,
        "iou_3d_central": iou_central,
    }


# ----------------------------
# 2D HTML visualization
# ----------------------------
def visualize_2d_overlays(
    subject_id: str,
    model: ImplicitNeuralRepresentation,
    contours_2d: torch.Tensor,
    pose_layer: PoseParameters,
    chosen: List[int],
    D: int,
    config: Dict,
) -> None:
    """
    Create a single-row HTML figure with num_views columns:
    overlay of pred (red) and GT (green) per selected slice.
    Overlap region is highlighted (yellow).
    """
    device = torch.device(config["device"])
    model = model.to(device)
    extrinsics = pose_layer.get_matrices(device=device)

    num_views = len(chosen)
    fig = make_subplots(
        rows=1,
        cols=num_views,
        subplot_titles=[f"Slice index = {idx}" for idx in chosen],
    )

    for v in range(num_views):
        slice_index = chosen[v]
        z_depth = 2.0 * (slice_index / (D - 1)) - 1.0
        pred_slice = (
            project_slice_from_inr(
                model,
                extrinsics[v],
                resolution=config["proj_resolution"],
                device=device,
                z_depth=z_depth,
            )
            .detach()
            .cpu()
            .numpy()
        )
        gt_slice = contours_2d[v].detach().cpu().numpy()

        # Soft blending + explicit overlap highlighting
        overlay = np.zeros(
            (pred_slice.shape[0], pred_slice.shape[1], 3), dtype=np.float32
        )
        overlay[..., 0] = 0.7 * pred_slice  # red: prediction
        overlay[..., 1] = 0.7 * gt_slice    # green: ground truth

        mask_overlap = (pred_slice > 0.5) & (gt_slice > 0.5)
        overlay[..., 0][mask_overlap] = 1.0
        overlay[..., 1][mask_overlap] = 1.0

        fig.add_trace(
            go.Image(z=(overlay * 255).astype(np.uint8)),
            row=1,
            col=v + 1,
        )

    # add global legend text
    fig.update_layout(
        title=f"{subject_id} – 2D slices: Prediction vs Ground Truth",
        width=300 * num_views,
        height=320,
        margin=dict(l=0, r=0, t=40, b=40),
    )
    fig.add_annotation(
        text="Red = Prediction only, Green = GT only, Yellow = Overlap",
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.12,
        showarrow=False,
    )

    html_file = CONFIG["html_output_path"] / f"{subject_id}_2d_slices.html"
    fig.write_html(str(html_file))
    print(f"  2D slice HTML saved: {html_file}")


# ----------------------------
# MAIN
# ----------------------------
def main() -> None:
    device = torch.device(CONFIG["device"])
    print("Device:", device)
    print("Data path:", CONFIG["data_path"])
    print("Learn pose:", CONFIG.get("learn_pose", True))
    print("Slice selection strategy:", CONFIG.get("slice_selection_strategy", "stratified"))
    print("Vol supervision weight:", CONFIG.get("vol_supervision_weight", 0.0))
    print("Crop to seg bbox:", CONFIG.get("crop_to_seg_bbox", True))

    pairs = find_mitea_image_files(CONFIG["data_path"])
    max_subjects = CONFIG.get("max_subjects", None)
    if max_subjects is not None and max_subjects > 0:
        pairs = pairs[: max_subjects]

    if not pairs:
        print("No data pairs found - check data_path")
        return

    all_metrics_2d = {"dice": [], "iou": []}
    all_metrics_3d = {
        "dice": [],
        "iou": [],
        "dice_central": [],
        "iou_central": [],
    }

    for idx, (img_file, label_file) in enumerate(pairs):
        subject_id = img_file.stem
        print("\n" + "-" * 60)
        print(f"[{idx+1}/{len(pairs)}] Subject: {subject_id}")

        try:
            vol, seg = load_mitea_subject(img_file, label_file)

            # Optional cropping to heart bounding box
            if CONFIG.get("crop_to_seg_bbox", True):
                vol, seg, bbox = crop_to_seg_bbox(
                    vol, seg, margin=int(CONFIG.get("crop_margin", 2))
                )
                print(f" Using cropped volume for training/eval; bbox={bbox}")

            D, H, W = vol.shape
            print(f"Loaded (possibly cropped) vol shape: {vol.shape}, seg shape: {seg.shape}")

            # strict slice extraction with configurable strategy
            try:
                slices_2d, contours_2d, chosen = extract_synthetic_2d_slices_strict(
                    vol,
                    seg,
                    num_views=CONFIG["num_views"],
                    min_pixels=CONFIG["min_mask_pixels"],
                    out_size=CONFIG["image_size"],
                    strategy=CONFIG.get("slice_selection_strategy", "stratified"),
                )
            except Exception as e:
                print("Skipping subject due to strict-slice selection:", e)
                continue

            print(f"Extracted slices: {slices_2d.shape}, contours: {contours_2d.shape}")
            print(
                "Contours per-slice foreground pixels:",
                [int(contours_2d[v].sum().item()) for v in range(contours_2d.shape[0])],
            )

            # init model + pose
            model = ImplicitNeuralRepresentation(
                hidden_dim=CONFIG["hidden_dim"],
                num_layers=CONFIG["num_inr_layers"],
            )
            pose_layer = PoseParameters(CONFIG["num_views"])

            # In canonical mode (learn_pose=False), start exactly at identity.
            if not CONFIG.get("learn_pose", True):
                with torch.no_grad():
                    pose_layer.pose.zero_()

            # optimize (hybrid 2D + 3D)
            model, pose_layer, losses = optimize_single_subject(
                model,
                slices_2d,
                contours_2d,
                pose_layer,
                chosen,
                CONFIG,
                num_steps=CONFIG["num_optimization_steps"],
                D=D,
                seg_vol=seg,
            )

            # 2D evaluation
            metrics_2d = evaluate_subject_2d(
                model,
                contours_2d,
                pose_layer,
                chosen,
                D,
                CONFIG,
            )
            print(
                f"  2D Dice: {metrics_2d['dice_2d']:.4f}, "
                f"2D IoU: {metrics_2d['iou_2d']:.4f}"
            )
            all_metrics_2d["dice"].append(metrics_2d["dice_2d"])
            all_metrics_2d["iou"].append(metrics_2d["iou_2d"])

            # 3D evaluation + mesh (full + central)
            metrics_3d = evaluate_subject_3d_and_mesh(
                model,
                seg,
                subject_id,
                CONFIG,
                chosen,
            )
            all_metrics_3d["dice"].append(metrics_3d["dice_3d"])
            all_metrics_3d["iou"].append(metrics_3d["iou_3d"])
            all_metrics_3d["dice_central"].append(metrics_3d["dice_3d_central"])
            all_metrics_3d["iou_central"].append(metrics_3d["iou_3d_central"])

            # 2D overlays
            visualize_2d_overlays(
                subject_id,
                model,
                contours_2d,
                pose_layer,
                chosen,
                D,
                CONFIG,
            )

            # save predicted occupancy grid as NIfTI (for external inspection)
            if CONFIG.get("save_nifti", True):
                R = int(CONFIG.get("eval_grid_resolution", 64))
                occ_grid = model.sample_grid(
                    resolution=R,
                    device=device,
                    requires_grad=False,
                )
                # Align ordering to (z,y,x) to match seg_vol convention
                occ_np = (
                    occ_grid.detach()
                    .cpu()
                    .permute(2, 1, 0)
                    .numpy()
                    .astype(np.float32)
                )
                pred_nii = nib.Nifti1Image(occ_np, affine=np.eye(4))
                pred_file = (
                    CONFIG["checkpoint_path"]
                    / f"{subject_id}_pred_occ_{R}cubed.nii.gz"
                )
                nib.save(pred_nii, str(pred_file))
                print(f"  Saved predicted occupancy NIfTI: {pred_file}")

        except Exception as e:
            print("ERROR processing", subject_id, e)
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("FINAL RESULTS (2D)")
    if all_metrics_2d["dice"]:
        print(
            f"Mean 2D Dice: {np.mean(all_metrics_2d['dice']):.4f} "
            f"± {np.std(all_metrics_2d['dice']):.4f}"
        )
        print(
            f"Mean 2D IoU:  {np.mean(all_metrics_2d['iou']):.4f} "
            f"± {np.std(all_metrics_2d['iou']):.4f}"
        )
        print(f"Subjects processed: {len(all_metrics_2d['dice'])}")
    else:
        print("No successful subjects processed.")

    print("\nFINAL RESULTS (3D)")
    if all_metrics_3d["dice"]:
        print(
            f"Mean 3D Dice (full): {np.mean(all_metrics_3d['dice']):.4f} "
            f"± {np.std(all_metrics_3d['dice']):.4f}"
        )
        print(
            f"Mean 3D IoU  (full): {np.mean(all_metrics_3d['iou']):.4f} "
            f"± {np.std(all_metrics_3d['iou']):.4f}"
        )
        print(
            f"Mean 3D Dice (central): {np.mean(all_metrics_3d['dice_central']):.4f} "
            f"± {np.std(all_metrics_3d['dice_central']):.4f}"
        )
        print(
            f"Mean 3D IoU  (central): {np.mean(all_metrics_3d['iou_central']):.4f} "
            f"± {np.std(all_metrics_3d['iou_central']):.4f}"
        )
    else:
        print("No 3D metrics (likely scikit-image missing or all subjects skipped).")


if __name__ == "__main__":
    main()
