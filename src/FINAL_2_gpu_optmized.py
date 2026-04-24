#!/usr/bin/env python3
"""
Echo3D Hybrid – unified script with local/global/mixed/meta modes.

Modes (CONFIG["mode"]):
- "local"  : Baseline A – per-scan INR optimization from scratch (original behavior).
- "global" : Baseline B – single shared INR trained on train split, evaluated
             on val/test with NO per-scan fine-tuning (pure generalization).
- "mixed"  : Global INR prior + per-scan refinement:
             * Train shared INR on train split.
             * Evaluate global INR on test.
             * Then, for each test scan, copy global INR and run a smaller
               per-scan optimization (with pose) starting from that prior.
- "meta"   : Reptile-style meta-learning:
             * Learn a good initialization for quick per-heart adaptation.
"""

import copy
import csv
import random
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import os

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DATA_PATH = Path(
    os.getenv(
        "CARDIAC_DATA_PATH",
        str(Path(__file__).resolve().parents[1] / "cap-mitea" / "mitea"),
    )
).expanduser()
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ============================================================
# CONFIG
# ============================================================
CONFIG: Dict = {
    # device
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # data paths
    "data_path": DATA_PATH,
    "checkpoint_path": Path("./checkpoints5"),
    "html_output_path": Path("./checkpoints5/html_visualizations30"),

    # dataset control
    "max_subjects": None,   # None or <=0 => use all; otherwise limit num subjects
    "random_seed": 42,

    # subject-level split ratios (by patient ID, not scan)
    "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},

    # master mode:
    #   "local"  => per-scan optimization (Baseline A, original behavior)
    #   "global" => shared INR across hearts (Baseline B)
    #   "mixed"  => global INR + per-scan refinement from that prior
    #   "meta"   => meta-learning (Reptile-style)
    "mode": "mixed",

    # views / slices
    "num_views": 3,
    "min_mask_pixels": 100,
    # "topk" or "stratified"
    "slice_selection_strategy": "stratified",

    # network + training (shared with local & global)
    "image_size": 256,
    "hidden_dim": 128,
    "num_inr_layers": 4,
    "learning_rate": 1e-3,
    "pose_learning_rate": 1e-4,
    "num_optimization_steps": 1000,   # for local per-scan optimization
    "alternate_every": 20,            # steps per block before switching shape/pose
    "proj_resolution": 256,           # projection H,W
    "proj_batch_size": 65536,         # batch size for INR 2D projection

    # pose behavior
    "learn_pose": False,               # canonical pose default; used in local/mixed
    "pose_reg_weight": 1e-3,          # L2 pose regularization if learn_pose=True

    # Hybrid 3D volumetric supervision
    "vol_supervision_weight": 1.0,
    "vol_supervision_samples": 8192,  # reduced for speed

    # volumetric regularization grids
    "grid_resolution": 48,            # for Laplace/entropy/area
    "eval_grid_resolution": 96,      # for 3D metrics + NIfTI/mesh
    "reg_every": 4,                   # do volumetric regularizers every N steps

    # Cropping to seg bounding box
    "crop_to_seg_bbox": True,
    "crop_margin": 2,

    # visualization / outputs
    "mesh_threshold": 0.5,
    "save_nifti": True,
    "log_csv_basename": "l3d_metrics",

    # parallelism (across scans, not within a scan) – local mode only
    "num_workers": 1,  # keep 1 for GPU; >1 only makes sense if CPU-bound or multi-GPU

    # logging
    "print_every": 50,

    # ================= GLOBAL TRAINING (Baseline B) =================
    "num_epochs": 20,           # epochs over train scans
    "steps_per_epoch": 200,     # SGD steps per epoch (can set to len(train_scans))

    # ================= MIXED MODE (global prior + refinement) =======
    "mixed_refine_steps": 200,  # per-scan fine-tune steps starting from global INR

    # ================= META-LEARNING (Reptile-style) ================
    # Number of outer meta steps (episodes)
    "meta_outer_steps": 50,
    # Number of inner adaptation steps per task (per heart)
    "meta_inner_steps": 100,
    # How many scans (tasks) per outer step
    "meta_batch_size": 2,
    # Step size for moving base INR towards adapted INRs
    "meta_step_size": 0.5,
    # Test-time refinement steps in meta mode (like mixed)
    "meta_refine_steps": 200,

    # ================= FRAME & DEMOGRAPHICS FILTERING ===============
    # Which cardiac frame(s) to use:
    #   "ED"   -> only *_ED.nii*
    #   "ES"   -> only *_ES.nii*
    #   "both" -> use all frames
    "frame_filter": "ED",

    # Path to demographics CSV with columns like:
    #   mitea_id, category, age, height_cm, weight_kg, sex, image_quality
    "demographics_csv": DATA_PATH / "demographics.csv",

    # Subject category filter:
    #   "all"          -> no filtering
    #   "healthy_only" -> category == "healthy"
    #   "disease_only" -> category != "healthy"
    #   "whitelist"    -> category in subject_category_whitelist (case-insensitive)
    "subject_category_mode": "healthy_only",
    "subject_category_whitelist": ["healthy"],
}

    
    
    
# ----- Overrides you specified -----
CONFIG_OVERRIDES = {
    "alternate_every": 20,
    "eval_grid_resolution": 96,
    "frame_filter": "ED",
    "grid_resolution": 48,
    "hidden_dim": 128,
    "learn_pose": False,
    "learning_rate": 0.003535770674047263,
    "mesh_threshold": 0.4,
    "min_mask_pixels": 75,
    "mixed_refine_steps": 200,
    "num_epochs": 10,
    "num_inr_layers": 5,
    "num_views": 3,
    "pose_learning_rate": 0.0000632263199815192,
    "pose_reg_weight": 0.048687957736080374,
    "proj_batch_size": 32768,          # 32,768
    "proj_resolution": 224,
    "random_seed": 100,
    "reg_every": 1,
    "save_nifti": False,
    "slice_selection_strategy": "topk",
    "steps_per_epoch": 400,
    "subject_category_mode": "all",
    "vol_supervision_samples": 16384,  # 16,384
    "vol_supervision_weight": 0.4757693326449699,
}

# ----- Final config used for the run -----
RUN_CONFIG = copy.deepcopy(CONFIG)
RUN_CONFIG.update(CONFIG_OVERRIDES)  # standard Python dict update[web:1][web:3]


CONFIG = RUN_CONFIG
CONFIG["checkpoint_path"].mkdir(exist_ok=True, parents=True)
CONFIG["html_output_path"].mkdir(exist_ok=True, parents=True)

# cuDNN autotune: safe here because shapes are fixed-ish
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ============================================================
# Utility: seeding
# ============================================================


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Data discovery / grouping / splitting
# ============================================================


def find_mitea_image_files(data_path: Path) -> List[Tuple[Path, Path]]:
    """
    Find all image/label pairs under:
        data_path/images
        data_path/labels

    Matching is by stem prefix.

    Respects CONFIG["frame_filter"]:
      - "ED":  only files whose stem contains "_ED"
      - "ES":  only files whose stem contains "_ES"
      - "both": no frame filtering
    """
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        print("ERROR: Expected images/ and labels/ subdirectories under", data_path)
        if images_dir.exists():
            print(" images/ exists; sample contents:", list(images_dir)[:5])
        if labels_dir.exists():
            print(" labels/ exists; sample contents:", list(labels_dir)[:5])
        return []

    frame_filter = str(CONFIG.get("frame_filter", "both")).upper()
    if frame_filter not in {"ED", "ES", "BOTH"}:
        raise ValueError(
            f"Invalid CONFIG['frame_filter']={frame_filter!r}; "
            "expected 'ED', 'ES', or 'both'."
        )

    image_files = sorted(images_dir.glob("*.nii*"))
    filtered_images: List[Path] = []

    for img in image_files:
        stem = img.stem  # e.g. "MITEA_001_scan1_ED"
        if frame_filter == "BOTH":
            filtered_images.append(img)
        elif f"_{frame_filter}" in stem:
            filtered_images.append(img)

    if frame_filter != "BOTH":
        print(
            f"Frame filter = {frame_filter}; kept "
            f"{len(filtered_images)}/{len(image_files)} image files."
        )

    pairs: List[Tuple[Path, Path]] = []
    for img in filtered_images:
        stem = img.stem
        candidates = list(labels_dir.glob(f"{stem}*"))
        if candidates:
            pairs.append((img, candidates[0]))
        else:
            # handle .nii vs .nii.gz mismatch
            if stem.endswith(".nii"):
                alt = stem[:-4]
                candidates2 = list(labels_dir.glob(f"{alt}*"))
                if candidates2:
                    pairs.append((img, candidates2[0]))

    print(f"Found {len(pairs)} image-label pairs in {images_dir} after frame filtering")
    return pairs


def get_subject_id(img_path: Path) -> str:
    """
    Extract subject ID from filename.
    Example: MITEA_001_scan2_ED.nii.gz -> MITEA_001
    """
    stem = img_path.stem  # e.g., "MITEA_001_scan2_ED.nii" or "...ED"
    # Split on "_scan" and take the left part
    if "_scan" in stem:
        return stem.split("_scan")[0]
    # fallback: use first two tokens
    parts = stem.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return stem


def group_by_subject(
    pairs: List[Tuple[Path, Path]]
) -> Dict[str, List[Tuple[Path, Path]]]:
    """
    Group image/label pairs by subject ID.
    """
    subjects: Dict[str, List[Tuple[Path, Path]]] = defaultdict(list)
    for img, lab in pairs:
        sid = get_subject_id(img)
        subjects[sid].append((img, lab))

    print(f"Discovered {len(subjects)} subjects.")
    counts = [len(v) for v in subjects.values()]
    if counts:
        print(
            f"  Scans per subject: min={min(counts)}, max={max(counts)}, "
            f"mean={np.mean(counts):.2f}"
        )
    return subjects


def split_subjects(
    subject_ids: List[str],
    ratios: Dict[str, float],
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Subject-level train/val/test split with given ratios.
    """
    rng = np.random.RandomState(seed)
    subject_ids = list(subject_ids)
    rng.shuffle(subject_ids)
    n = len(subject_ids)

    r_train = ratios.get("train", 0.7)
    r_val = ratios.get("val", 0.15)
    r_test = ratios.get("test", 0.15)

    n_train = int(round(r_train * n))
    n_val = int(round(r_val * n))
    n_train = min(n_train, n)
    n_val = min(n_val, max(0, n - n_train))
    n_test = n - n_train - n_val

    train_ids = subject_ids[:n_train]
    val_ids = subject_ids[n_train : n_train + n_val]
    test_ids = subject_ids[n_train + n_val :]

    print("Subject-level split:")
    print(f"  Train: {len(train_ids)}")
    print(f"  Val:   {len(val_ids)}")
    print(f"  Test:  {len(test_ids)}")

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def load_demographics(csv_path_opt: Optional[Path]) -> Dict[str, Dict[str, str]]:
    """
    Load demographics.csv and return a mapping:
        mitea_id -> {"category": ..., "age": ..., "sex": ..., "image_quality": ...}
    """
    if csv_path_opt is None:
        print("No demographics CSV configured; skipping subject-category filtering.")
        return {}

    csv_path = Path(csv_path_opt)
    if not csv_path.exists():
        print(f"Demographics CSV {csv_path} not found; skipping subject-category filtering.")
        return {}

    mapping: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("mitea_id", "").strip()
            if not sid:
                continue
            mapping[sid] = {
                "category": row.get("category", "").strip(),
                "age": row.get("age", "").strip(),
                "sex": row.get("sex", "").strip(),
                "image_quality": row.get("image_quality", "").strip(),
            }

    print(f"Loaded demographics for {len(mapping)} subjects from {csv_path}")
    return mapping


# ============================================================
# Models: positional encoding + INR + pose layer
# ============================================================


class PositionalEncoding:
    def __init__(self, num_freqs: int = 4):
        self.num_freqs = num_freqs

    def encode(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (..., 3)
        pe = []
        for i in range(self.num_freqs):
            freq = 2.0 ** i
            pe.append(torch.sin(freq * np.pi * coords))
            pe.append(torch.cos(freq * np.pi * coords))
        return torch.cat(pe, dim=-1)


class ImplicitNeuralRepresentation(nn.Module):
    """
    Simple MLP-based occupancy network with positional encoding.
    """

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
        """
        Sample the INR on a uniform canonical grid in [-1,1]^3.

        Returns:
            occ: (R,R,R) in (x,y,z) ordering.
        """
        resolution = int(resolution)
        if isinstance(device, str):
            device = torch.device(device)
        if device is None:
            device = torch.device("cpu")

        lin = torch.linspace(-1, 1, resolution, device=device, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), dim=-1)
        if requires_grad:
            occ = self.forward(grid).squeeze(-1)
        else:
            with torch.no_grad():
                occ = self.forward(grid).squeeze(-1)
        return occ


class PoseParameters(nn.Module):
    """
    Per-view SE(3) parameters (Euler angles + translation).
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


# ============================================================
# Data loading / cropping / slice selection
# ============================================================


def load_mitea_subject(
    img_file: Path, label_file: Path
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, Tuple[int, int, int]]:
    """
    Load volume and segmentation, normalize volume to [0,1],
    binarize seg, and return affine + original shape.

    Returns:
        vol: (D,H,W) float32 in [0,1]
        seg_bin: (D,H,W) float32 in {0,1}
        affine: (4,4) np.ndarray
        orig_shape: (D,H,W)
    """
    img_nii = nib.load(str(img_file))
    seg_nii = nib.load(str(label_file))

    vol = torch.tensor(img_nii.get_fdata(), dtype=torch.float32)
    seg = torch.tensor(seg_nii.get_fdata(), dtype=torch.float32)

    # Normalize to [0,1]
    if (vol.max() - vol.min()) > 0:
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
    else:
        vol = torch.zeros_like(vol)

    print("Volume (normalized):")
    print(
        f" Min: {vol.min().item():.4f}, Max: {vol.max().item():.4f}, "
        f"Mean: {vol.mean().item():.4f}"
    )

    print("Segmentation (raw):")
    print(
        f" Min: {seg.min().item():.4f}, Max: {seg.max().item():.4f}, "
        f"Mean: {seg.mean().item():.4f}"
    )

    seg_bin = (seg > 0).float()
    print("Segmentation (binarized):")
    print(
        f" Min: {seg_bin.min().item():.4f}, Max: {seg_bin.max().item():.4f}, "
        f"Mean: {seg_bin.mean().item():.4f}"
    )

    affine = seg_nii.affine
    orig_shape = tuple(seg_bin.shape)
    return vol, seg_bin, affine, orig_shape


def crop_to_seg_bbox(
    vol: torch.Tensor,
    seg: torch.Tensor,
    margin: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int, int, int, int]]:
    """
    Crop volume and segmentation to tight bbox around seg>0, with margin.
    Returns:
        vol_c, seg_c, bbox=(dmin,dmax,hmin,hmax,wmin,wmax)
    """
    mask = seg > 0
    if not mask.any():
        D, H, W = vol.shape
        bbox = (0, D - 1, 0, H - 1, 0, W - 1)
        print("No foreground in seg; skipping cropping.")
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

    print(f" Cropped to bbox {bbox}; new vol shape {tuple(vol_c.shape)}")
    return vol_c, seg_c, bbox


def select_strict_slices(
    seg: torch.Tensor,
    num_views: int,
    min_pixels: int = 100,
    strategy: str = "stratified",
) -> List[int]:
    """
    Select exactly num_views axial slices with >= min_pixels foreground.

    strategy:
        - "topk": choose slices with largest foreground area.
        - "stratified": spread slices across z-range of valid slices.
    """
    D = seg.shape[0]
    counts = seg.contiguous().view(D, -1).sum(dim=1)
    counts_np = counts.cpu().numpy()

    valid_idxs = np.where(counts_np >= min_pixels)[0]
    if len(valid_idxs) < num_views:
        raise ValueError(
            f"STRICT: only {len(valid_idxs)} valid slices, "
            f"need {num_views}. Lower min_mask_pixels or skip this subject."
        )

    strategy = strategy.lower()
    if strategy == "topk":
        sorted_idxs = valid_idxs[np.argsort(-counts_np[valid_idxs])]
        chosen = sorted_idxs[:num_views].tolist()
        chosen_sorted = sorted(chosen)
    elif strategy == "stratified":
        lin_idx = np.linspace(0, len(valid_idxs) - 1, num_views, dtype=int)
        chosen_sorted = sorted(valid_idxs[lin_idx].tolist())
    else:
        raise ValueError(f"Unknown slice_selection_strategy: {strategy}")

    print(
        f"Strict-selected slice indices (strategy={strategy}): {chosen_sorted}"
    )
    return chosen_sorted


def extract_slices_strict(
    vol: torch.Tensor,
    seg: torch.Tensor,
    num_views: int,
    min_pixels: int,
    out_size: int,
    strategy: str = "stratified",
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Extract num_views axial slices with strict constraints.
    Returns:
        slices_2d: (V,H,W) normalized images
        contours_2d: (V,H,W) {0,1} masks
        chosen_idxs: list of z indices in [0, D-1]
    """
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

    slices_2d = torch.stack(slices)
    contours_2d = torch.stack(contours)
    return slices_2d, contours_2d, chosen


# ============================================================
# Losses / regularizers
# ============================================================


def contour_bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.clamp(1e-6, 1 - 1e-6)
    return F.binary_cross_entropy(pred, target)


def laplacian_smoothness_loss(
    occupancy_grid: torch.Tensor, weight: float = 0.02
) -> torch.Tensor:
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


def volume_entropy_loss(
    occupancy_grid: torch.Tensor, weight: float = 0.01
) -> torch.Tensor:
    eps = 1e-6
    entropy = -(
        occupancy_grid * torch.log(occupancy_grid + eps)
        + (1 - occupancy_grid) * torch.log(1 - occupancy_grid + eps)
    )
    return weight * torch.mean(entropy)


def surface_area_loss(
    occupancy_grid: torch.Tensor, weight: float = 0.005
) -> torch.Tensor:
    grad_x = torch.diff(occupancy_grid, dim=0, prepend=occupancy_grid[:1])
    grad_y = torch.diff(occupancy_grid, dim=1, prepend=occupancy_grid[:, :1])
    grad_z = torch.diff(occupancy_grid, dim=2, prepend=occupancy_grid[:, :, :1])
    grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-6)
    return weight * torch.mean(grad_norm)


# ============================================================
# Differentiable projection: INR -> planar slices (vectorized)
# ============================================================


def project_slices_from_inr_batch(
    model: ImplicitNeuralRepresentation,
    extrinsics: torch.Tensor,
    chosen: List[int],
    D: int,
    resolution: int,
    device,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Vectorized projection of all supervised slices in one INR call.

    Args:
        model: INR
        extrinsics: (V,4,4) pose matrices
        chosen: list[int] slice indices in [0, D-1]
        D: depth of volume used for mapping z -> [-1,1]
        resolution: H=W of projection
        device: torch.device or str
        batch_size: optional chunk size for INR eval

    Returns:
        occ: (V,H,W) occupancy in [0,1]
    """
    if isinstance(device, str):
        device = torch.device(device)

    V = len(chosen)
    resolution = int(resolution)

    lin = torch.linspace(-1.0, 1.0, resolution, device=device, dtype=torch.float32)
    xv, yv = torch.meshgrid(lin, lin, indexing="ij")  # (H,W)

    xv = xv.unsqueeze(0).expand(V, -1, -1)  # (V,H,W)
    yv = yv.unsqueeze(0).expand(V, -1, -1)

    # Map integer slice index -> canonical z depth in [-1,1]
    z_depths = torch.tensor(
        [2.0 * (float(z) / max(D - 1, 1)) - 1.0 for z in chosen],
        device=device,
        dtype=torch.float32,
    ).view(V, 1, 1)
    zv = torch.ones_like(xv) * z_depths

    ones = torch.ones_like(xv)

    pts_plane = torch.stack([xv, yv, zv, ones], dim=-1)  # (V,H,W,4)
    V, H, W, _ = pts_plane.shape
    pts_plane_flat = pts_plane.view(V, H * W, 4)

    # extrinsics: (V,4,4), match semantics pts @ extrinsics^T
    pts_world = torch.bmm(pts_plane_flat, extrinsics.transpose(1, 2))  # (V,HW,4)
    pts_world = pts_world[..., :3]  # (V,HW,3)

    coords = pts_world.reshape(V * H * W, 3)
    coords = coords.to(next(model.parameters()).device)

    if batch_size is None:
        batch_size = 65536

    outs = []
    for i in range(0, coords.shape[0], batch_size):
        outs.append(model(coords[i : i + batch_size]))
    occ = torch.cat(outs, dim=0).view(V, H, W)
    return occ


# ============================================================
# Optimization (alternating shape/pose, hybrid supervision)
# ============================================================


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
    Alternating optimization of INR (shape) and pose parameters.

    seg_vol: cropped segmentation volume used for volumetric BCE.
    """
    device = torch.device(config["device"])
    learn_pose: bool = bool(config["learn_pose"])
    pose_reg_weight: float = float(config.get("pose_reg_weight", 0.0))
    vol_weight: float = float(config.get("vol_supervision_weight", 0.0))
    vol_samples: int = int(config.get("vol_supervision_samples", 0))
    reg_every: int = int(config.get("reg_every", 1))
    proj_batch_size: int = int(config.get("proj_batch_size", 65536))

    model = model.to(device)
    pose_layer = pose_layer.to(device)
    seg_device = seg_vol.to(device)

    D_seg, H_seg, W_seg = seg_device.shape
    if D_seg != D:
        print(f"Warning: D={D} != seg_vol.shape[0]={D_seg}; using seg_vol D.")
        D = D_seg

    optimizer_shape = optim.Adam(model.parameters(), lr=config["learning_rate"])

    if learn_pose:
        optimizer_pose = optim.Adam(
            pose_layer.parameters(), lr=config["pose_learning_rate"]
        )
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
            shape_step = ((step // config.get("alternate_every", 1)) % 2 == 0)
        else:
            shape_step = True

        if shape_step:
            extrinsics = pose_layer.get_matrices(device=device).detach()
        else:
            extrinsics = pose_layer.get_matrices(device=device)

        # Vectorized projection of all supervised slices
        preds = project_slices_from_inr_batch(
            model,
            extrinsics,
            chosen,
            D,
            config["proj_resolution"],
            device,
            batch_size=proj_batch_size,
        )  # (V,H,W)

        loss_projection = contour_bce_loss(preds, contours_device)

        # Initialize regularizer terms
        vol_loss = torch.tensor(0.0, device=device)
        loss_smooth = torch.tensor(0.0, device=device)
        loss_entropy = torch.tensor(0.0, device=device)
        loss_area = torch.tensor(0.0, device=device)

        if shape_step:
            # Only do heavy volumetric regularization every reg_every steps
            do_reg = (reg_every <= 1) or (step % reg_every == 0)

            if do_reg:
                # Optional volumetric BCE
                if vol_weight > 0.0 and vol_samples > 0:
                    coords3d = torch.rand(vol_samples, 3, device=device) * 2.0 - 1.0
                    pred_vol = model(coords3d).squeeze(-1)

                    x_norm = (coords3d[:, 0] + 1.0) * 0.5
                    y_norm = (coords3d[:, 1] + 1.0) * 0.5
                    z_norm = (coords3d[:, 2] + 1.0) * 0.5

                    x_idx = torch.clamp((x_norm * (H_seg - 1)).long(), 0, H_seg - 1)
                    y_idx = torch.clamp((y_norm * (W_seg - 1)).long(), 0, W_seg - 1)
                    z_idx = torch.clamp((z_norm * (D_seg - 1)).long(), 0, D_seg - 1)

                    gt_vol = seg_device[z_idx, x_idx, y_idx].view(-1)
                    vol_loss = F.binary_cross_entropy(pred_vol, gt_vol)

                # Volumetric regularizers
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
                + vol_weight * vol_loss
            )
        else:
            # Pose-only step
            loss = loss_projection
            if pose_reg_weight > 0.0:
                loss = loss + pose_reg_weight * (pose_layer.pose**2).mean()

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
                    f"Step {step}/{num_steps}: total={loss.item():.6f}, "
                    f"proj={loss_projection.item():.6f}, vol={vol_loss.item():.6f}, "
                    f"shape-step"
                )
            else:
                print(
                    f"Step {step}/{num_steps}: total={loss.item():.6f}, "
                    f"proj={loss_projection.item():.6f}, pose-step"
                )

        if step % 200 == 0:
            with torch.no_grad():
                p = preds[0].detach().cpu()
                t = contours_device[0].detach().cpu()
                print(
                    f" Pred range: min {p.min().item():.4f}, "
                    f"max {p.max().item():.4f}, mean {p.mean().item():.4f}"
                )
                print(
                    f" GT range:  min {t.min().item():.4f}, "
                    f"max {t.max().item():.4f}, mean {t.mean().item():.4f}"
                )

    return model, pose_layer, losses


# ============================================================
# Evaluation helpers
# ============================================================


def sample_eval_grid_aligned(
    model: ImplicitNeuralRepresentation,
    config: Dict,
) -> torch.Tensor:
    """
    Sample the INR on canonical eval grid and return a volume aligned as (z,y,x)
    to match the (D,H,W) convention of seg volumes.
    """
    device = torch.device(config["device"])
    R = int(config.get("eval_grid_resolution", 64))
    occ_grid = model.sample_grid(resolution=R, device=device, requires_grad=False)
    # occ_grid: (R,R,R) in (x,y,z); align to (z,y,x)
    occ_aligned = occ_grid.detach().cpu().permute(2, 1, 0).contiguous()
    return occ_aligned  # (R,R,R) as (z,y,x)


def evaluate_subject_2d(
    model: ImplicitNeuralRepresentation,
    contours_2d: torch.Tensor,
    pose_layer: PoseParameters,
    chosen: List[int],
    D: int,
    config: Dict,
) -> Dict[str, float]:
    """
    2D Dice / IoU on the supervised slices (consistent z-depth mapping).
    """
    device = torch.device(config["device"])
    model = model.to(device)
    contours = contours_2d.to(device)
    extrinsics = pose_layer.get_matrices(device=device)

    preds = project_slices_from_inr_batch(
        model,
        extrinsics,
        chosen,
        D,
        config["proj_resolution"],
        device,
        batch_size=config.get("proj_batch_size", 65536),
    )  # (V,H,W)

    dices: List[float] = []
    ious: List[float] = []

    for v in range(contours.shape[0]):
        pred = preds[v]
        pred_binary = (pred > 0.5).float()
        target_binary = (contours[v] > 0.5).float()

        intersection = torch.sum(pred_binary * target_binary)
        union = torch.sum(pred_binary) + torch.sum(target_binary)
        dice = (2 * intersection) / (union + 1e-6)
        iou = intersection / (union - intersection + 1e-6)

        dices.append(dice.item())
        ious.append(iou.item())

    return {"dice_2d": float(np.mean(dices)), "iou_2d": float(np.mean(ious))}


def evaluate_subject_3d_and_mesh(
    model: ImplicitNeuralRepresentation,
    seg_vol: torch.Tensor,
    subject_scan_id: str,
    config: Dict,
    chosen: List[int],
) -> Dict[str, float]:
    """
    3D Dice/IoU (full + central) and 3D mesh HTML export.
    seg_vol is the (cropped) segmentation volume used for training.
    """
    device = torch.device(config["device"])
    model = model.to(device)
    thr = float(config.get("mesh_threshold", 0.5))

    occ_aligned = sample_eval_grid_aligned(model, config)  # (R,R,R) (z,y,x)
    R = occ_aligned.shape[0]

    # Resample seg to (R,R,R)
    seg_t = seg_vol.unsqueeze(0).unsqueeze(0).float()  # (1,1,D,H,W)
    seg_resampled = F.interpolate(seg_t, size=(R, R, R), mode="nearest")[0, 0]

    pred_bin = (occ_aligned > thr).float()
    gt_bin = (seg_resampled > 0.5).float()

    intersection = (pred_bin * gt_bin).sum().item()
    union = pred_bin.sum().item() + gt_bin.sum().item()
    dice_full = (2.0 * intersection) / (union + 1e-6)
    iou_full = intersection / (union - intersection + 1e-6)

    # central z-range (supervised slices)
    D_orig = seg_vol.shape[0]
    zmin_orig = int(min(chosen))
    zmax_orig = int(max(chosen))

    zmin_R = int(round(zmin_orig / max(D_orig - 1, 1) * (R - 1)))
    zmax_R = int(round(zmax_orig / max(D_orig - 1, 1) * (R - 1)))
    zmin_R = max(0, min(zmin_R, R - 1))
    zmax_R = max(0, min(zmax_R, R - 1))
    if zmax_R < zmin_R:
        zmin_R, zmax_R = zmax_R, zmin_R

    pred_c = pred_bin[zmin_R : zmax_R + 1]
    gt_c = gt_bin[zmin_R : zmax_R + 1]

    intersection_c = (pred_c * gt_c).sum().item()
    union_c = pred_c.sum().item() + gt_c.sum().item()
    dice_c = (2.0 * intersection_c) / (union_c + 1e-6)
    iou_c = intersection_c / (union_c - intersection_c + 1e-6)

    print(f"  3D Dice (full): {dice_full:.4f}, 3D IoU (full): {iou_full:.4f}")
    print(
        f"  3D Dice (central supervised z-range): {dice_c:.4f}, "
        f"3D IoU (central): {iou_c:.4f}"
    )

    # 3D mesh visualization via marching cubes
    try:
        from skimage import measure
    except ImportError:
        print(
            "  scikit-image not installed; skipping 3D mesh. "
            "Install with `pip install scikit-image` if needed."
        )
        return {
            "dice_3d": dice_full,
            "iou_3d": iou_full,
            "dice_3d_central": dice_c,
            "iou_3d_central": iou_c,
        }

    occ_np = occ_aligned.numpy()
    seg_np = seg_resampled.numpy()

    try:
        verts_pred, faces_pred, _, _ = measure.marching_cubes(occ_np, level=thr)
        verts_gt, faces_gt, _, _ = measure.marching_cubes(seg_np, level=0.5)
    except ValueError as e:
        print("  Marching cubes failed (likely empty volume):", e)
        return {
            "dice_3d": dice_full,
            "iou_3d": iou_full,
            "dice_3d_central": dice_c,
            "iou_3d_central": iou_c,
        }

    def norm_verts(v: np.ndarray) -> np.ndarray:
        scale = np.array([R - 1, R - 1, R - 1], dtype=np.float32)
        return v / scale

    vp = norm_verts(verts_pred)
    vg = norm_verts(verts_gt)

    fig = go.Figure()
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
        title=f"{subject_scan_id} – 3D mesh: Prediction (red) vs GT (green)",
        scene=dict(aspectmode="data"),
        legend=dict(x=0.02, y=0.98),
    )

    html_file = CONFIG["html_output_path"] / f"{subject_scan_id}_3d_mesh.html"
    fig.write_html(str(html_file))
    print(f"  3D mesh HTML saved: {html_file}")

    return {
        "dice_3d": dice_full,
        "iou_3d": iou_full,
        "dice_3d_central": dice_c,
        "iou_3d_central": iou_c,
    }


def visualize_2d_overlays(
    subject_scan_id: str,
    model: ImplicitNeuralRepresentation,
    contours_2d: torch.Tensor,
    pose_layer: PoseParameters,
    chosen: List[int],
    D: int,
    config: Dict,
) -> None:
    """
    One-row HTML figure: per selected slice overlay prediction vs GT.
    """
    device = torch.device(config["device"])
    model = model.to(device)
    extrinsics = pose_layer.get_matrices(device=device)

    preds = project_slices_from_inr_batch(
        model,
        extrinsics,
        chosen,
        D,
        config["proj_resolution"],
        device,
        batch_size=config.get("proj_batch_size", 65536),
    )  # (V,H,W)

    num_views = len(chosen)
    fig = make_subplots(
        rows=1,
        cols=num_views,
        subplot_titles=[f"z = {idx}" for idx in chosen],
    )

    for v in range(num_views):
        pred_slice = preds[v].detach().cpu().numpy()
        gt_slice = contours_2d[v].detach().cpu().numpy()

        overlay = np.zeros(
            (pred_slice.shape[0], pred_slice.shape[1], 3), dtype=np.float32
        )
        overlay[..., 0] = 0.7 * pred_slice  # red: prediction
        overlay[..., 1] = 0.7 * gt_slice    # green: GT

        mask_overlap = (pred_slice > 0.5) & (gt_slice > 0.5)
        overlay[..., 0][mask_overlap] = 1.0
        overlay[..., 1][mask_overlap] = 1.0

        fig.add_trace(
            go.Image(z=(overlay * 255).astype(np.uint8)),
            row=1,
            col=v + 1,
        )

    fig.update_layout(
        title=f"{subject_scan_id} – 2D slices: Prediction vs Ground Truth",
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

    html_file = CONFIG["html_output_path"] / f"{subject_scan_id}_2d_slices.html"
    fig.write_html(str(html_file))
    print(f"  2D slice HTML saved: {html_file}")


# ============================================================
# NIfTI export – full resolution, correct affine / cropping
# ============================================================


def save_pred_nifti_fullres(
    model: ImplicitNeuralRepresentation,
    affine_full: np.ndarray,
    orig_shape: Tuple[int, int, int],
    bbox: Tuple[int, int, int, int, int, int],
    subject_scan_id: str,
    config: Dict,
) -> None:
    """
    Save predicted occupancy NIfTI aligned with the original volume:

    1. Sample INR on canonical eval grid (128^3 by default).
    2. Align to (z,y,x), upsample/downsample to cropped bbox size.
    3. Embed into full-volume array at bbox location.
    4. Save with original affine so overlays match in viewers.
    """
    device = torch.device(config["device"])
    model = model.to(device)

    occ_aligned = sample_eval_grid_aligned(model, config)  # (R,R,R) (z,y,x)

    # Resize to cropped bbox size
    dmin, dmax, hmin, hmax, wmin, wmax = bbox
    D_c = dmax - dmin + 1
    H_c = hmax - hmin + 1
    W_c = wmax - wmin + 1

    occ_cropped = F.interpolate(
        occ_aligned.unsqueeze(0).unsqueeze(0),
        size=(D_c, H_c, W_c),
        mode="trilinear",
        align_corners=False,
    )[0, 0]

    full_pred = np.zeros(orig_shape, dtype=np.float32)
    full_pred[dmin : dmax + 1, hmin : hmax + 1, wmin : wmax + 1] = (
        occ_cropped.detach().cpu().numpy()
    )

    pred_nii = nib.Nifti1Image(full_pred, affine_full)
    R = int(config.get("eval_grid_resolution", 64))
    out_file = (
        config["checkpoint_path"]
        / f"{subject_scan_id}_pred_occ_fullres_from_{R}cubed.nii.gz"
    )
    nib.save(pred_nii, str(out_file))
    print(f"  Saved predicted occupancy NIfTI (fullres): {out_file}")


# ============================================================
# LOCAL MODE: per-scan job + main (Baseline A)
# ============================================================


def process_scan_job(job: Tuple[str, str, Path, Path]) -> Optional[Dict[str, Any]]:
    """
    Run the full pipeline for a single scan (one image/label pair).

    Args:
        job: (split, subj_id, img_file, label_file)

    Returns:
        dict with metrics + metadata, or None if skipped/failed.
    """
    split, subj_id, img_file, label_file = job
    subject_scan_id = img_file.stem

    print("\n" + "-" * 60)
    print(f"[{split}] Subject: {subj_id} | Scan: {subject_scan_id}")

    try:
        # -------------------------
        # Load volume + seg
        # -------------------------
        vol, seg, affine_full, orig_shape = load_mitea_subject(
            img_file, label_file
        )

        # -------------------------
        # Optional cropping to seg bbox
        # -------------------------
        if CONFIG.get("crop_to_seg_bbox", True):
            vol_c, seg_c, bbox = crop_to_seg_bbox(
                vol,
                seg,
                margin=int(CONFIG.get("crop_margin", 2)),
            )
            print(f" Using cropped volume for training/eval; bbox={bbox}")
        else:
            D0, H0, W0 = vol.shape
            bbox = (0, D0 - 1, 0, H0 - 1, 0, W0 - 1)
            vol_c, seg_c = vol, seg
            print(" Using full volume (no cropping).")

        D, H, W = vol_c.shape
        print(
            f"Loaded (possibly cropped) vol shape: {tuple(vol_c.shape)}, "
            f"seg shape: {tuple(seg_c.shape)}"
        )

        # -------------------------
        # Strict slice selection
        # -------------------------
        try:
            slices_2d, contours_2d, chosen = extract_slices_strict(
                vol_c,
                seg_c,
                num_views=CONFIG["num_views"],
                min_pixels=CONFIG["min_mask_pixels"],
                out_size=CONFIG["image_size"],
                strategy=CONFIG.get(
                    "slice_selection_strategy", "stratified"
                ),
            )
        except Exception as e:
            print("Skipping scan due to strict-slice selection failure:", e)
            return None

        print(f"Extracted slices: {slices_2d.shape}, contours: {contours_2d.shape}")
        print(
            "Contours per-slice foreground pixels:",
            [int(contours_2d[v].sum().item()) for v in range(contours_2d.shape[0])],
        )

        # -------------------------
        # Init model + pose
        # -------------------------
        model = ImplicitNeuralRepresentation(
            hidden_dim=CONFIG["hidden_dim"],
            num_layers=CONFIG["num_inr_layers"],
        )
        pose_layer = PoseParameters(CONFIG["num_views"])

        # Canonical mode: start at exact identity if learn_pose False
        if not CONFIG.get("learn_pose", True):
            with torch.no_grad():
                pose_layer.pose.zero_()

        # -------------------------
        # Optimize INR (+ pose if enabled)
        # -------------------------
        model, pose_layer, _ = optimize_single_subject(
            model,
            slices_2d,
            contours_2d,
            pose_layer,
            chosen,
            CONFIG,
            num_steps=CONFIG["num_optimization_steps"],
            D=D,
            seg_vol=seg_c,
        )

        # -------------------------
        # 2D evaluation
        # -------------------------
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

        # -------------------------
        # 3D evaluation + mesh
        # -------------------------
        metrics_3d = evaluate_subject_3d_and_mesh(
            model,
            seg_c,
            subject_scan_id,
            CONFIG,
            chosen,
        )

        # -------------------------
        # 2D overlay HTML
        # -------------------------
        visualize_2d_overlays(
            subject_scan_id,
            model,
            contours_2d,
            pose_layer,
            chosen,
            D,
            CONFIG,
        )

        # -------------------------
        # Full-res NIfTI export
        # -------------------------
        if CONFIG.get("save_nifti", True):
            save_pred_nifti_fullres(
                model,
                affine_full,
                orig_shape,
                bbox,
                subject_scan_id,
                CONFIG,
            )

        # -------------------------
        # Package result for aggregation
        # -------------------------
        return {
            "split": split,
            "subject_id": subj_id,
            "scan_id": subject_scan_id,
            "img_path": str(img_file),
            "label_path": str(label_file),
            "metrics_2d": metrics_2d,
            "metrics_3d": metrics_3d,
            "num_slices": len(chosen),
            "z_indices": chosen,
        }

    except Exception as e:
        print(f"ERROR processing {subject_scan_id}:", e)
        traceback.print_exc()
        return None


def main_local() -> None:
    """
    Original behavior: Baseline A (per-scan INR optimization),
    with added ED/ES frame filter and demographics-based subject filtering.
    """
    set_global_seed(CONFIG["random_seed"])
    device = torch.device(CONFIG["device"])

    print("LOCAL MODE (per-scan optimization)")
    print("Device:", device)
    print("Data path:", CONFIG["data_path"])
    print("Learn pose:", CONFIG["learn_pose"])
    print("Slice selection strategy:", CONFIG.get("slice_selection_strategy", "stratified"))
    print("Vol supervision weight:", CONFIG.get("vol_supervision_weight", 0.0))
    print("Crop to seg bbox:", CONFIG.get("crop_to_seg_bbox", True))
    print("Num workers:", CONFIG.get("num_workers", 1))
    print("Proj batch size:", CONFIG.get("proj_batch_size", 65536))
    print("Reg every:", CONFIG.get("reg_every", 1))
    print("Frame filter (ED/ES/both):", CONFIG.get("frame_filter", "both"))
    print("Subject category mode:", CONFIG.get("subject_category_mode", "all"))

    # Load demographics mapping once
    demographics = load_demographics(CONFIG.get("demographics_csv", None))

    # Discover pairs and group by subject
    pairs = find_mitea_image_files(CONFIG["data_path"])
    if not pairs:
        print("No data pairs found – check CONFIG['data_path'].")
        return

    subjects = group_by_subject(pairs)
    subject_ids = sorted(subjects.keys())

    # Optional subject limit
    max_subjects = CONFIG.get("max_subjects", None)
    if max_subjects is not None and max_subjects > 0:
        subject_ids = subject_ids[:max_subjects]
        subjects = {sid: subjects[sid] for sid in subject_ids}
        print(
            f"Restricting to first {len(subject_ids)} subjects due to "
            f"max_subjects={max_subjects}"
        )

    # Demographics-based subject filtering ("same boat" subjects)
    subject_category_mode = str(CONFIG.get("subject_category_mode", "all")).lower()
    if subject_category_mode != "all" and demographics:
        filtered_ids: List[str] = []
        whitelist = {
            c.lower() for c in CONFIG.get("subject_category_whitelist", [])
        }

        for sid in subject_ids:
            info = demographics.get(sid)
            if info is None:
                continue
            cat = info.get("category", "").lower()

            keep = False
            if subject_category_mode == "healthy_only":
                keep = (cat == "healthy")
            elif subject_category_mode == "disease_only":
                keep = (cat != "healthy" and cat != "")
            elif subject_category_mode == "whitelist":
                keep = (cat in whitelist)

            if keep:
                filtered_ids.append(sid)

        print(
            f"Subject category filter '{subject_category_mode}' kept "
            f"{len(filtered_ids)}/{len(subject_ids)} subjects."
        )
        subject_ids = filtered_ids
        subjects = {sid: subjects[sid] for sid in subject_ids}
    else:
        print("No subject-category filtering applied (mode='all' or no demographics).")

    if not subject_ids:
        print("No subjects left after filtering – nothing to do.")
        return

    # Subject-level split
    splits = split_subjects(
        subject_ids,
        CONFIG.get("split_ratios", {"train": 0.7, "val": 0.15, "test": 0.15}),
        seed=CONFIG.get("random_seed", 42),
    )

    # Which splits to process (for local mode, "all" makes sense)
    process_cfg = "all"  # keep original behavior here
    if process_cfg == "all":
        splits_to_run = ["train", "val", "test"]
    else:
        if process_cfg not in splits:
            print(
                f"Invalid process_split='{process_cfg}'. "
                "Expected one of 'all', 'train', 'val', 'test'."
            )
            return
        splits_to_run = [process_cfg]

    # Metric accumulators + CSV rows
    split_metrics = {
        split: {
            "dice_2d": [],
            "iou_2d": [],
            "dice_3d": [],
            "iou_3d": [],
            "dice_3d_central": [],
            "iou_3d_central": [],
        }
        for split in splits_to_run
    }
    csv_rows: List[Dict[str, Any]] = []

    # Build list of scan jobs
    jobs: List[Tuple[str, str, Path, Path]] = []
    for split in splits_to_run:
        for subj_id in splits[split]:
            scans = subjects[subj_id]
            for img_file, label_file in scans:
                jobs.append((split, subj_id, img_file, label_file))

    print(f"\nTotal scans to process: {len(jobs)}")

    num_workers = int(CONFIG.get("num_workers", 1))
    if num_workers > 1 and "cuda" in str(CONFIG["device"]).lower():
        print(
            f"NOTE: Running {num_workers} workers on a single GPU. "
            "This usually does NOT reduce wall-clock per scan; reduce if you see OOM."
        )

    # Run jobs (parallel or sequential)
    results: List[Dict[str, Any]] = []

    if num_workers > 1:
        print(f"Running in parallel with {num_workers} worker threads...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_job = {
                executor.submit(process_scan_job, job): job for job in jobs
            }
            for future in as_completed(future_to_job):
                res = future.result()
                if res is not None:
                    results.append(res)
    else:
        print("Running sequentially (num_workers=1)...")
        for job in jobs:
            res = process_scan_job(job)
            if res is not None:
                results.append(res)

    if not results:
        print("\nNo scans processed successfully – no metrics to report.")
        return

    # Aggregate metrics into split_metrics + csv_rows
    for res in results:
        split = res["split"]
        metrics_2d = res["metrics_2d"]
        metrics_3d = res["metrics_3d"]

        split_metrics[split]["dice_2d"].append(metrics_2d["dice_2d"])
        split_metrics[split]["iou_2d"].append(metrics_2d["iou_2d"])
        split_metrics[split]["dice_3d"].append(metrics_3d["dice_3d"])
        split_metrics[split]["iou_3d"].append(metrics_3d["iou_3d"])
        split_metrics[split]["dice_3d_central"].append(
            metrics_3d["dice_3d_central"]
        )
        split_metrics[split]["iou_3d_central"].append(
            metrics_3d["iou_3d_central"]
        )

        demo = demographics.get(res["subject_id"], {})

        csv_rows.append(
            {
                "split": split,
                "subject_id": res["subject_id"],
                "scan_id": res["scan_id"],
                "img_path": res["img_path"],
                "label_path": res["label_path"],
                "dice_2d": metrics_2d["dice_2d"],
                "iou_2d": metrics_2d["iou_2d"],
                "dice_3d": metrics_3d["dice_3d"],
                "iou_3d": metrics_3d["iou_3d"],
                "dice_3d_central": metrics_3d["dice_3d_central"],
                "iou_3d_central": metrics_3d["iou_3d_central"],
                "num_slices": res["num_slices"],
                "z_indices": ";".join(str(z) for z in res["z_indices"]),
                "category": demo.get("category", ""),
                "age": demo.get("age", ""),
                "sex": demo.get("sex", ""),
                "image_quality": demo.get("image_quality", ""),
            }
        )

    # Split-wise summary stats
    print("\n" + "=" * 60)
    print("FINAL SPLIT-WISE METRICS (LOCAL MODE)")

    def mean_std(x: List[float]) -> Tuple[float, float]:
        return float(np.mean(x)), float(np.std(x))

    for split in splits_to_run:
        m = split_metrics[split]
        if not m["dice_2d"]:
            print(f"  {split}: no successful scans.")
            continue

        d2m, d2s = mean_std(m["dice_2d"])
        i2m, i2s = mean_std(m["iou_2d"])
        d3m, d3s = mean_std(m["dice_3d"])
        i3m, i3s = mean_std(m["iou_3d"])
        d3cm, d3cs = mean_std(m["dice_3d_central"])
        i3cm, i3cs = mean_std(m["iou_3d_central"])

        print(f"\nSplit: {split} (n={len(m['dice_2d'])})")
        print(f"  2D Dice:           {d2m:.4f} ± {d2s:.4f}")
        print(f"  2D IoU:            {i2m:.4f} ± {i2s:.4f}")
        print(f"  3D Dice (full):    {d3m:.4f} ± {d3s:.4f}")
        print(f"  3D IoU  (full):    {i3m:.4f} ± {i3s:.4f}")
        print(f"  3D Dice (central): {d3cm:.4f} ± {d3cs:.4f}")
        print(f"  3D IoU  (central): {i3cm:.4f} ± {i3cs:.4f}")

    # CSV export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = (
        CONFIG["checkpoint_path"]
        / f"{CONFIG['log_csv_basename']}_{timestamp}_local.csv"
    )

    fieldnames = [
        "split",
        "subject_id",
        "scan_id",
        "img_path",
        "label_path",
        "dice_2d",
        "iou_2d",
        "dice_3d",
        "iou_3d",
        "dice_3d_central",
        "iou_3d_central",
        "num_slices",
        "z_indices",
        "category",
        "age",
        "sex",
        "image_quality",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print(f"\nPer-scan metrics CSV (local) saved to: {csv_path}")


# ============================================================
# GLOBAL DATASET STRUCTURE (for global/mixed/meta modes)
# ============================================================


@dataclass
class ScanDatum:
    split: str
    subject_id: str
    scan_id: str
    vol_c: torch.Tensor        # (D,H,W) cropped volume, CPU
    seg_c: torch.Tensor        # (D,H,W) cropped segmentation, CPU
    slices_2d: torch.Tensor    # (V,Hs,Ws) selected slices, CPU
    contours_2d: torch.Tensor  # (V,Hs,Ws) selected masks, CPU
    chosen_z: List[int]        # slice indices in [0,D-1] wrt cropped volume
    D: int                     # depth of cropped volume
    affine_full: np.ndarray    # (4,4) original seg affine
    orig_shape: Tuple[int, int, int]  # original (D,H,W) before cropping
    bbox: Tuple[int, int, int, int, int, int]  # cropping bbox in original space


def build_scan_dataset(config: Dict) -> Dict[str, List[ScanDatum]]:
    """
    Discover all scans, split by subject into train/val/test, and
    precompute cropped volumes + selected slices for each scan.

    Applies the same ED/ES frame filter and demographics-based
    subject filtering as local mode.

    Returns:
        dataset["train"], dataset["val"], dataset["test"]
    """
    set_global_seed(config.get("random_seed", 42))

    pairs = find_mitea_image_files(config["data_path"])
    if not pairs:
        raise RuntimeError("No data pairs found – check CONFIG['data_path'].")

    subjects = group_by_subject(pairs)
    subject_ids = sorted(subjects.keys())

    # Optional subject limit
    max_subjects = config.get("max_subjects", None)
    if max_subjects is not None and max_subjects > 0:
        subject_ids = subject_ids[:max_subjects]
        subjects = {sid: subjects[sid] for sid in subject_ids}
        print(
            f"Restricting to first {len(subject_ids)} subjects due to "
            f"max_subjects={max_subjects}"
        )

    # Demographics-based subject filtering
    demographics = load_demographics(config.get("demographics_csv", None))
    subject_category_mode = str(config.get("subject_category_mode", "all")).lower()
    if subject_category_mode != "all" and demographics:
        filtered_ids: List[str] = []
        whitelist = {
            c.lower() for c in config.get("subject_category_whitelist", [])
        }

        for sid in subject_ids:
            info = demographics.get(sid)
            if info is None:
                continue
            cat = info.get("category", "").lower()

            keep = False
            if subject_category_mode == "healthy_only":
                keep = (cat == "healthy")
            elif subject_category_mode == "disease_only":
                keep = (cat != "healthy" and cat != "")
            elif subject_category_mode == "whitelist":
                keep = (cat in whitelist)

            if keep:
                filtered_ids.append(sid)

        print(
            f"Subject category filter '{subject_category_mode}' kept "
            f"{len(filtered_ids)}/{len(subject_ids)} subjects."
        )
        subject_ids = filtered_ids
        subjects = {sid: subjects[sid] for sid in subject_ids}
    else:
        print("No subject-category filtering applied in dataset builder (mode='all' or no demographics).")

    if not subject_ids:
        raise RuntimeError("No subjects left after demographics/frame filtering.")

    splits = split_subjects(
        subject_ids,
        config.get("split_ratios", {"train": 0.7, "val": 0.15, "test": 0.15}),
        seed=config.get("random_seed", 42),
    )

    dataset: Dict[str, List[ScanDatum]] = {"train": [], "val": [], "test": []}

    for split_name, subj_list in splits.items():
        print(f"\nPreparing split '{split_name}' with {len(subj_list)} subjects...")
        for subj_id in subj_list:
            scans = subjects[subj_id]
            for img_file, label_file in scans:
                scan_id = img_file.stem
                print(f"  [{split_name}] {subj_id} | {scan_id}")

                try:
                    vol, seg, affine_full, orig_shape = load_mitea_subject(
                        img_file, label_file
                    )

                    # Cropping
                    if config.get("crop_to_seg_bbox", True):
                        vol_c, seg_c, bbox = crop_to_seg_bbox(
                            vol,
                            seg,
                            margin=int(config.get("crop_margin", 2)),
                        )
                        print(f"    Using cropped volume; bbox={bbox}")
                    else:
                        D0, H0, W0 = vol.shape
                        bbox = (0, D0 - 1, 0, H0 - 1, 0, W0 - 1)
                        vol_c, seg_c = vol, seg
                        print("    Using full volume (no cropping).")

                    D_c, H_c, W_c = vol_c.shape
                    print(f"    Cropped vol shape: {vol_c.shape}")

                    # Strict slice selection
                    slices_2d, contours_2d, chosen = extract_slices_strict(
                        vol_c,
                        seg_c,
                        num_views=config["num_views"],
                        min_pixels=config["min_mask_pixels"],
                        out_size=config["image_size"],
                        strategy=config.get(
                            "slice_selection_strategy", "stratified"
                        ),
                    )

                    print(
                        f"    Slices_2d: {tuple(slices_2d.shape)}, "
                        f"Contours_2d: {tuple(contours_2d.shape)}, "
                        f"chosen_z={chosen}"
                    )

                    datum = ScanDatum(
                        split=split_name,
                        subject_id=subj_id,
                        scan_id=scan_id,
                        vol_c=vol_c.cpu(),
                        seg_c=seg_c.cpu(),
                        slices_2d=slices_2d.cpu(),
                        contours_2d=contours_2d.cpu(),
                        chosen_z=chosen,
                        D=D_c,
                        affine_full=affine_full,
                        orig_shape=orig_shape,
                        bbox=bbox,
                    )
                    dataset[split_name].append(datum)

                except Exception as e:
                    print(
                        f"    Skipping scan {scan_id} due to error "
                        f"during preprocessing: {e}"
                    )
                    traceback.print_exc()

    print("\nDataset sizes:")
    for split_name in ["train", "val", "test"]:
        print(f"  {split_name}: {len(dataset[split_name])} scans")

    return dataset


# ============================================================
# GLOBAL TRAINING (Baseline B)
# ============================================================


def compute_scan_loss_global(
    model: ImplicitNeuralRepresentation,
    scan: ScanDatum,
    config: Dict,
    global_step: int,
) -> torch.Tensor:
    """
    Compute the hybrid loss for a single ScanDatum, using the
    shared INR model (no per-scan pose learning here).
    """
    device = torch.device(config["device"])
    model = model.to(device)

    contours = scan.contours_2d.to(device)  # (V,H,W)
    seg_vol = scan.seg_c.to(device)         # (D,H,W)

    D_seg, H_seg, W_seg = seg_vol.shape
    V = contours.shape[0]

    # Identity extrinsics for all views (canonical alignment assumption)
    extrinsics = torch.eye(4, device=device).unsqueeze(0).repeat(V, 1, 1)

    # 2D projection loss
    preds = project_slices_from_inr_batch(
        model,
        extrinsics,
        scan.chosen_z,
        scan.D,
        config["proj_resolution"],
        device,
        batch_size=config.get("proj_batch_size", 65536),
    )  # (V,H,W)

    loss_projection = contour_bce_loss(preds, contours)

    # Volumetric BCE + regularizers (optional)
    vol_weight = float(config.get("vol_supervision_weight", 0.0))
    vol_samples = int(config.get("vol_supervision_samples", 0))
    reg_every = int(config.get("reg_every", 1))

    vol_loss = torch.tensor(0.0, device=device)
    loss_smooth = torch.tensor(0.0, device=device)
    loss_entropy = torch.tensor(0.0, device=device)
    loss_area = torch.tensor(0.0, device=device)

    # Optional volumetric BCE
    if vol_weight > 0.0 and vol_samples > 0:
        coords3d = torch.rand(vol_samples, 3, device=device) * 2.0 - 1.0
        pred_vol = model(coords3d).squeeze(-1)

        x_norm = (coords3d[:, 0] + 1.0) * 0.5
        y_norm = (coords3d[:, 1] + 1.0) * 0.5
        z_norm = (coords3d[:, 2] + 1.0) * 0.5

        x_idx = torch.clamp((x_norm * (H_seg - 1)).long(), 0, H_seg - 1)
        y_idx = torch.clamp((y_norm * (W_seg - 1)).long(), 0, W_seg - 1)
        z_idx = torch.clamp((z_norm * (D_seg - 1)).long(), 0, D_seg - 1)

        gt_vol = seg_vol[z_idx, x_idx, y_idx].view(-1)
        vol_loss = F.binary_cross_entropy(pred_vol, gt_vol)

    # Volumetric regularizers every reg_every global steps
    do_reg = (reg_every <= 1) or (global_step % reg_every == 0)
    if do_reg:
        occ_grid = model.sample_grid(
            resolution=config["grid_resolution"],
            device=device,
            requires_grad=True,
        )
        loss_smooth = laplacian_smoothness_loss(occ_grid, weight=0.02)
        loss_entropy = volume_entropy_loss(occ_grid, weight=0.01)
        loss_area = surface_area_loss(occ_grid, weight=0.005)

    total_loss = (
        loss_projection
        + vol_weight * vol_loss
        + loss_smooth
        + loss_entropy
        + loss_area
    )
    return total_loss


def train_global_inr(
    train_scans: List[ScanDatum],
    val_scans: List[ScanDatum],
    config: Dict,
) -> ImplicitNeuralRepresentation:
    """
    Train a single shared INR over all training scans (Baseline B).
    """
    device = torch.device(config["device"])
    model = ImplicitNeuralRepresentation(
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_inr_layers"],
    ).to(device)

    lr = config["learning_rate"]
    num_epochs = int(config.get("num_epochs", 20))
    steps_per_epoch = int(config.get("steps_per_epoch", max(1, len(train_scans))))
    total_steps = num_epochs * steps_per_epoch

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr / 10.0
    )

    global_step = 0
    for epoch in range(num_epochs):
        # Shuffle scans each epoch
        random.shuffle(train_scans)
        running_loss = 0.0

        for step in range(steps_per_epoch):
            scan = train_scans[step % len(train_scans)]
            model.train()
            loss = compute_scan_loss_global(model, scan, config, global_step)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % config.get("print_every", 50) == 0:
                print(
                    f"[global step {global_step}/{total_steps}] "
                    f"train loss = {loss.item():.6f}"
                )

        avg_loss = running_loss / steps_per_epoch
        print(f"\nEpoch {epoch+1}/{num_epochs}: mean train loss = {avg_loss:.6f}")

        # Quick val metrics each epoch (optional, fast-ish)
        if val_scans:
            val_summary = evaluate_split_global(
                model, val_scans, config, split_name="val"
            )
            print(
                f"  Val 2D Dice: {val_summary['dice_2d_mean']:.4f}, "
                f"3D Dice: {val_summary['dice_3d_mean']:.4f}"
            )

    return model


def meta_train_reptile(
    train_scans: List[ScanDatum],
    val_scans: List[ScanDatum],
    config: Dict,
) -> ImplicitNeuralRepresentation:
    """
    Reptile-style meta-learning over scans.

    Each ScanDatum is a 'task':
      - Clone the current base INR.
      - Run a short per-scan optimization with optimize_single_subject.
      - Move the base INR parameters slightly towards the adapted INR.

    This trains an initialization that is explicitly good *after* a few
    per-scan steps on a new heart.
    """
    device = torch.device(config["device"])

    # Base INR initialization
    base_model = ImplicitNeuralRepresentation(
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_inr_layers"],
    ).to(device)

    outer_steps = int(config.get("meta_outer_steps", 50))
    inner_steps = int(config.get("meta_inner_steps", 100))
    meta_batch_size = int(config.get("meta_batch_size", 2))
    meta_step_size = float(config.get("meta_step_size", 0.5))

    print("\nStarting Reptile-style meta-training:")
    print(f"  outer_steps    = {outer_steps}")
    print(f"  inner_steps    = {inner_steps}")
    print(f"  meta_batch_size= {meta_batch_size}")
    print(f"  meta_step_size = {meta_step_size}")

    if not train_scans:
        raise RuntimeError("No training scans available for meta-learning.")

    for outer in range(outer_steps):
        # Current base weights snapshot
        base_state = {k: v.clone() for k, v in base_model.state_dict().items()}
        # Accumulate deltas across tasks
        agg_deltas = {k: torch.zeros_like(v) for k, v in base_state.items()}

        # Sample a batch of tasks (scans)
        batch_size = min(meta_batch_size, len(train_scans))
        batch_scans = random.sample(train_scans, batch_size)

        print(f"\n[Meta outer step {outer+1}/{outer_steps}] using {batch_size} scans")

        for scan_idx, scan in enumerate(batch_scans):
            print(f"  Task {scan_idx+1}/{batch_size}: {scan.scan_id}")

            # Clone base INR
            adapted_model = ImplicitNeuralRepresentation(
                hidden_dim=config["hidden_dim"],
                num_layers=config["num_inr_layers"],
            ).to(device)
            adapted_model.load_state_dict(base_state)

            # Fresh pose layer for this scan (inner loop can learn pose)
            pose_layer = PoseParameters(config["num_views"])
            if not config.get("learn_pose", True):
                with torch.no_grad():
                    pose_layer.pose.zero_()

            # Inner adaptation steps on this scan
            # (reuse optimize_single_subject but with fewer steps)
            inner_config = dict(config)  # shallow copy is fine
            inner_config["print_every"] = max(50, inner_steps // 2)  # avoid spam

            adapted_model, pose_layer, inner_losses = optimize_single_subject(
                adapted_model,
                scan.slices_2d,
                scan.contours_2d,
                pose_layer,
                scan.chosen_z,
                inner_config,
                num_steps=inner_steps,
                D=scan.D,
                seg_vol=scan.seg_c,
            )

            if inner_losses:
                print(
                    f"    Inner final loss (task {scan.scan_id}): "
                    f"{inner_losses[-1]:.6f}"
                )

            # Compute parameter delta: adapted - base
            adapted_state = adapted_model.state_dict()
            for name in base_state.keys():
                agg_deltas[name] += (adapted_state[name] - base_state[name])

        # Average delta across tasks and apply meta update
        with torch.no_grad():
            for name, base_tensor in base_state.items():
                delta = agg_deltas[name] / float(batch_size)
                base_state[name] = base_tensor + meta_step_size * delta

        # Load updated parameters back into base_model
        base_model.load_state_dict(base_state)

        # Optional: quick validation every few outer steps
        if val_scans and ((outer + 1) % max(1, outer_steps // 5) == 0):
            print("\n  Quick val evaluation (meta init, no adaptation):")
            _ = evaluate_split_global(
                base_model,
                val_scans,
                config,
                split_name=f"meta_val_outer{outer+1}",
            )

    print("\nMeta-training finished.")
    return base_model


def evaluate_scan_global(
    model: ImplicitNeuralRepresentation,
    scan: ScanDatum,
    config: Dict,
) -> Dict[str, Any]:
    """
    Evaluate a trained global INR on a single scan (no INR updates).
    Uses identity pose (canonical alignment).
    """
    # Pose layer with exact identity transforms
    pose_layer = PoseParameters(config["num_views"])
    with torch.no_grad():
        pose_layer.pose.zero_()

    metrics_2d = evaluate_subject_2d(
        model,
        scan.contours_2d,
        pose_layer,
        scan.chosen_z,
        scan.D,
        config,
    )

    metrics_3d = evaluate_subject_3d_and_mesh(
        model,
        scan.seg_c,
        scan.scan_id,
        config,
        scan.chosen_z,
    )

    return {
        "metrics_2d": metrics_2d,
        "metrics_3d": metrics_3d,
    }


def evaluate_split_global(
    model: ImplicitNeuralRepresentation,
    scans: List[ScanDatum],
    config: Dict,
    split_name: str = "test",
) -> Dict[str, float]:
    """
    Run 2D/3D metrics on all scans in a split, aggregate mean/std.
    """
    if not scans:
        print(f"No scans in split '{split_name}' to evaluate.")
        return {
            "dice_2d_mean": 0.0,
            "dice_2d_std": 0.0,
            "iou_2d_mean": 0.0,
            "iou_2d_std": 0.0,
            "dice_3d_mean": 0.0,
            "dice_3d_std": 0.0,
            "iou_3d_mean": 0.0,
            "iou_3d_std": 0.0,
            "dice_3d_central_mean": 0.0,
            "dice_3d_central_std": 0.0,
            "iou_3d_central_mean": 0.0,
            "iou_3d_central_std": 0.0,
        }

    dice_2d_list, iou_2d_list = [], []
    dice_3d_list, iou_3d_list = [], []
    dice_3d_c_list, iou_3d_c_list = [], []

    print(f"\nEvaluating split '{split_name}' with {len(scans)} scans...")
    for scan in scans:
        res = evaluate_scan_global(model, scan, config)
        m2 = res["metrics_2d"]
        m3 = res["metrics_3d"]

        dice_2d_list.append(m2["dice_2d"])
        iou_2d_list.append(m2["iou_2d"])
        dice_3d_list.append(m3["dice_3d"])
        iou_3d_list.append(m3["iou_3d"])
        dice_3d_c_list.append(m3["dice_3d_central"])
        iou_3d_c_list.append(m3["iou_3d_central"])

        print(
            f"  {scan.scan_id}: "
            f"2D Dice={m2['dice_2d']:.4f}, 3D Dice={m3['dice_3d']:.4f}, "
            f"3D Dice(c)={m3['dice_3d_central']:.4f}"
        )

    def mean_std(arr: List[float]) -> Tuple[float, float]:
        return float(np.mean(arr)), float(np.std(arr))

    d2m, d2s = mean_std(dice_2d_list)
    i2m, i2s = mean_std(iou_2d_list)
    d3m, d3s = mean_std(dice_3d_list)
    i3m, i3s = mean_std(iou_3d_list)
    d3cm, d3cs = mean_std(dice_3d_c_list)
    i3cm, i3cs = mean_std(iou_3d_c_list)

    print(f"\nSplit '{split_name}' summary (n={len(scans)}):")
    print(f"  2D Dice:           {d2m:.4f} ± {d2s:.4f}")
    print(f"  2D IoU:            {i2m:.4f} ± {i2s:.4f}")
    print(f"  3D Dice (full):    {d3m:.4f} ± {d3s:.4f}")
    print(f"  3D IoU  (full):    {i3m:.4f} ± {i3s:.4f}")
    print(f"  3D Dice (central): {d3cm:.4f} ± {d3cs:.4f}")
    print(f"  3D IoU  (central): {i3cm:.4f} ± {i3cs:.4f}")

    return {
        "dice_2d_mean": d2m,
        "dice_2d_std": d2s,
        "iou_2d_mean": i2m,
        "iou_2d_std": i2s,
        "dice_3d_mean": d3m,
        "dice_3d_std": d3s,
        "iou_3d_mean": i3m,
        "iou_3d_std": i3s,
        "dice_3d_central_mean": d3cm,
        "dice_3d_central_std": d3cs,
        "iou_3d_central_mean": i3cm,
        "iou_3d_central_std": i3cs,
    }


def main_global() -> None:
    """
    Global-training mode (Baseline B):
      - build dataset (train/val/test) with ED/ES & demographics filters
      - train one shared INR on train split
      - evaluate on val + test without INR updates
    """
    set_global_seed(CONFIG["random_seed"])
    device = torch.device(CONFIG["device"])

    print("GLOBAL MODE (shared INR across hearts)")
    print("Device:", device)
    print("Data path:", CONFIG["data_path"])
    print("Crop to seg bbox:", CONFIG.get("crop_to_seg_bbox", True))
    print("Num views per scan:", CONFIG["num_views"])
    print("Frame filter (ED/ES/both):", CONFIG.get("frame_filter", "both"))
    print("Subject category mode:", CONFIG.get("subject_category_mode", "all"))

    dataset = build_scan_dataset(CONFIG)
    train_scans = dataset["train"]
    val_scans = dataset["val"]
    test_scans = dataset["test"]

    if not train_scans:
        print("No training scans available – aborting.")
        return

    # Train shared INR on train split
    model = train_global_inr(train_scans, val_scans, CONFIG)

    # Final evaluation on test split
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SPLIT (GLOBAL MODE, NO REFINEMENT)")
    _ = evaluate_split_global(model, test_scans, CONFIG, split_name="test")


# ============================================================
# MIXED MODE: global INR prior + per-scan refinement
# ============================================================


def main_mixed() -> None:
    """
    Mixed mode:
      1) Train shared INR on train split (Baseline B).
      2) Evaluate global INR on test split (no refinement).
      3) For each test scan, copy global INR and run a shorter per-scan
         optimization (with pose) using optimize_single_subject.
    """
    set_global_seed(CONFIG["random_seed"])
    device = torch.device(CONFIG["device"])

    print("MIXED MODE (global INR prior + per-scan refinement)")
    print("Device:", device)
    print("Data path:", CONFIG["data_path"])
    print("Crop to seg bbox:", CONFIG.get("crop_to_seg_bbox", True))
    print("Num views per scan:", CONFIG["num_views"])
    print("Mixed refine steps:", CONFIG.get("mixed_refine_steps", 200))
    print("Frame filter (ED/ES/both):", CONFIG.get("frame_filter", "both"))
    print("Subject category mode:", CONFIG.get("subject_category_mode", "all"))

    dataset = build_scan_dataset(CONFIG)
    train_scans = dataset["train"]
    val_scans = dataset["val"]
    test_scans = dataset["test"]

    if not train_scans:
        print("No training scans available – aborting.")
        return

    # 1) Train shared INR on train split
    model_global = train_global_inr(train_scans, val_scans, CONFIG)

    # 2) Evaluate global INR on test split
    print("\n" + "=" * 60)
    print("TEST SPLIT – GLOBAL INR (NO REFINEMENT)")
    global_summary = evaluate_split_global(
        model_global, test_scans, CONFIG, split_name="test_global"
    )

    # 3) Per-scan refinement starting from global INR
    print("\n" + "=" * 60)
    print("TEST SPLIT – MIXED MODE (GLOBAL PRIOR + PER-SCAN REFINEMENT)")

    dice_2d_list, iou_2d_list = [], []
    dice_3d_list, iou_3d_list = [], []
    dice_3d_c_list, iou_3d_c_list = [], []

    refine_steps = int(CONFIG.get("mixed_refine_steps", 200))

    for scan in test_scans:
        print("\n" + "-" * 60)
        print(f"Refining scan {scan.scan_id} starting from global INR...")

        # Clone global INR weights
        model_local = ImplicitNeuralRepresentation(
            hidden_dim=CONFIG["hidden_dim"],
            num_layers=CONFIG["num_inr_layers"],
        )
        model_local.load_state_dict(model_global.state_dict())

        # New pose layer (we allow pose learning here)
        pose_layer = PoseParameters(CONFIG["num_views"])
        if not CONFIG.get("learn_pose", True):
            with torch.no_grad():
                pose_layer.pose.zero_()

        # Per-scan refinement config: same as CONFIG, but with fewer steps
        refine_config = dict(CONFIG)
        # we keep lr etc; just change number of steps
        model_local, pose_layer, _ = optimize_single_subject(
            model_local,
            scan.slices_2d,
            scan.contours_2d,
            pose_layer,
            scan.chosen_z,
            refine_config,
            num_steps=refine_steps,
            D=scan.D,
            seg_vol=scan.seg_c,
        )

        # Evaluate refined model
        metrics_2d = evaluate_subject_2d(
            model_local,
            scan.contours_2d,
            pose_layer,
            scan.chosen_z,
            scan.D,
            CONFIG,
        )
        metrics_3d = evaluate_subject_3d_and_mesh(
            model_local,
            scan.seg_c,
            scan.scan_id + "_mixed",
            CONFIG,
            scan.chosen_z,
        )

        dice_2d_list.append(metrics_2d["dice_2d"])
        iou_2d_list.append(metrics_2d["iou_2d"])
        dice_3d_list.append(metrics_3d["dice_3d"])
        iou_3d_list.append(metrics_3d["iou_3d"])
        dice_3d_c_list.append(metrics_3d["dice_3d_central"])
        iou_3d_c_list.append(metrics_3d["iou_3d_central"])

        print(
            f"  Mixed (refined) – 2D Dice={metrics_2d['dice_2d']:.4f}, "
            f"3D Dice={metrics_3d['dice_3d']:.4f}, "
            f"3D Dice(c)={metrics_3d['dice_3d_central']:.4f}"
        )

        # Optional: overlays + full-res NIfTI for each refined scan
        pose_for_vis = pose_layer
        visualize_2d_overlays(
            scan.scan_id + "_mixed",
            model_local,
            scan.contours_2d,
            pose_for_vis,
            scan.chosen_z,
            scan.D,
            CONFIG,
        )
        if CONFIG.get("save_nifti", True):
            save_pred_nifti_fullres(
                model_local,
                scan.affine_full,
                scan.orig_shape,
                scan.bbox,
                scan.scan_id + "_mixed",
                CONFIG,
            )

    def mean_std(arr: List[float]) -> Tuple[float, float]:
        return float(np.mean(arr)), float(np.std(arr))

    d2m, d2s = mean_std(dice_2d_list)
    i2m, i2s = mean_std(iou_2d_list)
    d3m, d3s = mean_std(dice_3d_list)
    i3m, i3s = mean_std(iou_3d_list)
    d3cm, d3cs = mean_std(dice_3d_c_list)
    i3cm, i3cs = mean_std(iou_3d_c_list)

    print("\nMIXED MODE SUMMARY (TEST SPLIT):")
    print(f"  2D Dice:           {d2m:.4f} ± {d2s:.4f}")
    print(f"  2D IoU:            {i2m:.4f} ± {i2s:.4f}")
    print(f"  3D Dice (full):    {d3m:.4f} ± {d3s:.4f}")
    print(f"  3D IoU  (full):    {i3m:.4f} ± {i3s:.4f}")
    print(f"  3D Dice (central): {d3cm:.4f} ± {d3cs:.4f}")
    print(f"  3D IoU  (central): {i3cm:.4f} ± {i3cs:.4f}")


def main_meta() -> None:
    """
    Meta-learning mode:
      1) Build dataset (train/val/test).
      2) Meta-train an INR initialization using Reptile-style updates.
      3) Evaluate meta-trained INR on test split with:
         - no adaptation (just forward pass with identity pose),
         - per-scan refinement starting from the meta initialization.
    """
    set_global_seed(CONFIG["random_seed"])
    device = torch.device(CONFIG["device"])

    print("META MODE (Reptile-style meta-learning over hearts)")
    print("Device:", device)
    print("Data path:", CONFIG["data_path"])
    print("Crop to seg bbox:", CONFIG.get("crop_to_seg_bbox", True))
    print("Num views per scan:", CONFIG["num_views"])
    print("Meta outer steps:", CONFIG.get("meta_outer_steps", 50))
    print("Meta inner steps:", CONFIG.get("meta_inner_steps", 100))
    print("Meta refine steps:", CONFIG.get("meta_refine_steps", 200))
    print("Frame filter (ED/ES/both):", CONFIG.get("frame_filter", "both"))
    print("Subject category mode:", CONFIG.get("subject_category_mode", "all"))

    dataset = build_scan_dataset(CONFIG)
    train_scans = dataset["train"]
    val_scans = dataset["val"]
    test_scans = dataset["test"]

    if not train_scans:
        print("No training scans available – aborting meta mode.")
        return

    # 1) Meta-train INR
    meta_model = meta_train_reptile(train_scans, val_scans, CONFIG)

    # 2) Evaluate meta-trained INR on test split without adaptation
    print("\n" + "=" * 60)
    print("TEST SPLIT – META INIT (NO PER-SCAN REFINEMENT)")
    _ = evaluate_split_global(
        meta_model,
        test_scans,
        CONFIG,
        split_name="test_meta_no_refine",
    )

    # 3) Per-scan refinement starting from meta initialization
    print("\n" + "=" * 60)
    print("TEST SPLIT – META INIT + PER-SCAN REFINEMENT")

    dice_2d_list, iou_2d_list = [], []
    dice_3d_list, iou_3d_list = [], []
    dice_3d_c_list, iou_3d_c_list = [], []

    refine_steps = int(CONFIG.get("meta_refine_steps", 200))

    for scan in test_scans:
        print("\n" + "-" * 60)
        print(f"Refining scan {scan.scan_id} from meta initialization...")

        # Clone meta-trained INR
        model_local = ImplicitNeuralRepresentation(
            hidden_dim=CONFIG["hidden_dim"],
            num_layers=CONFIG["num_inr_layers"],
        ).to(device)
        model_local.load_state_dict(meta_model.state_dict())

        # Pose layer (inner refinement can learn pose)
        pose_layer = PoseParameters(CONFIG["num_views"])
        if not CONFIG.get("learn_pose", True):
            with torch.no_grad():
                pose_layer.pose.zero_()

        refine_config = dict(CONFIG)
        refine_config["print_every"] = max(50, refine_steps // 2)

        model_local, pose_layer, _ = optimize_single_subject(
            model_local,
            scan.slices_2d,
            scan.contours_2d,
            pose_layer,
            scan.chosen_z,
            refine_config,
            num_steps=refine_steps,
            D=scan.D,
            seg_vol=scan.seg_c,
        )

        # Evaluate refined model
        metrics_2d = evaluate_subject_2d(
            model_local,
            scan.contours_2d,
            pose_layer,
            scan.chosen_z,
            scan.D,
            CONFIG,
        )
        metrics_3d = evaluate_subject_3d_and_mesh(
            model_local,
            scan.seg_c,
            scan.scan_id + "_meta",
            CONFIG,
            scan.chosen_z,
        )

        dice_2d_list.append(metrics_2d["dice_2d"])
        iou_2d_list.append(metrics_2d["iou_2d"])
        dice_3d_list.append(metrics_3d["dice_3d"])
        iou_3d_list.append(metrics_3d["iou_3d"])
        dice_3d_c_list.append(metrics_3d["dice_3d_central"])
        iou_3d_c_list.append(metrics_3d["iou_3d_central"])

        print(
            f"  Meta-refined – 2D Dice={metrics_2d['dice_2d']:.4f}, "
            f"3D Dice={metrics_3d['dice_3d']:.4f}, "
            f"3D Dice(c)={metrics_3d['dice_3d_central']:.4f}"
        )

        # Optional: visualize overlays + save NIfTI for each refined scan
        visualize_2d_overlays(
            scan.scan_id + "_meta",
            model_local,
            scan.contours_2d,
            pose_layer,
            scan.chosen_z,
            scan.D,
            CONFIG,
        )
        if CONFIG.get("save_nifti", True):
            save_pred_nifti_fullres(
                model_local,
                scan.affine_full,
                scan.orig_shape,
                scan.bbox,
                scan.scan_id + "_meta",
                CONFIG,
            )

    def mean_std(arr: List[float]) -> Tuple[float, float]:
        return float(np.mean(arr)), float(np.std(arr))

    d2m, d2s = mean_std(dice_2d_list)
    i2m, i2s = mean_std(iou_2d_list)
    d3m, d3s = mean_std(dice_3d_list)
    i3m, i3s = mean_std(iou_3d_list)
    d3cm, d3cs = mean_std(dice_3d_c_list)
    i3cm, i3cs = mean_std(iou_3d_c_list)

    print("\nMETA MODE SUMMARY (TEST SPLIT, AFTER REFINEMENT):")
    print(f"  2D Dice:           {d2m:.4f} ± {d2s:.4f}")
    print(f"  2D IoU:            {i2m:.4f} ± {i2s:.4f}")
    print(f"  3D Dice (full):    {d3m:.4f} ± {d3s:.4f}")
    print(f"  3D IoU  (full):    {i3m:.4f} ± {i3s:.4f}")
    print(f"  3D Dice (central): {d3cm:.4f} ± {d3cs:.4f}")
    print(f"  3D IoU  (central): {i3cm:.4f} ± {i3cs:.4f}")


# ============================================================
# ENTRYPOINT
# ============================================================


if __name__ == "__main__":
    mode = CONFIG.get("mode", "local").lower()
    if mode == "local":
        main_local()
    elif mode == "global":
        main_global()
    elif mode == "mixed":
        main_mixed()
    elif mode == "meta":
        main_meta()
    else:
        raise ValueError(
            f"Unknown CONFIG['mode'] = {mode!r} "
            "(expected 'local', 'global', 'mixed', or 'meta')"
        )
