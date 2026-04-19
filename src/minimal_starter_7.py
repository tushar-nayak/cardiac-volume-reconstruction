# # #!/usr/bin/env python3
# # """
# # Hybrid Echo3D minimal starter (Hybrid C) - STRICT non-empty slice selection

# # This script:
# # - Uses a hybrid of Version1 stability tricks + Version2 geometry/pose
# # - Uses STRICT non-empty-only slice selection: will pick exactly `num_views`
# #   slices that contain segmentation pixels above a configurable threshold.
# # - Alternating optimization of INR (shape) and PoseParameters (pose)
# # - Differentiable planar sampling projection from INR -> 2D slices
# # - Strong regularizers (laplacian, entropy, surface area)
# # - Robust debugging prints to detect empty masks

# # Usage:
# #     - Set CONFIG['num_views'] and CONFIG['view_mode']='strict'
# #     - Run: python hybrid_strict.py

# # Important:
# #     - Requires nibabel, torch, matplotlib (optional for debugging)
# #     - Expects MITEA data layout: data_path/images/*.nii* and data_path/labels/*.nii*
# # """

# # import os
# # import traceback
# # from pathlib import Path

# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import torch.optim as optim


# # # ----------------------------
# # # CONFIG
# # # ----------------------------
# # CONFIG = {
# #     # device
# #     "device": "cuda" if torch.cuda.is_available() else "cpu",
# #     # data
# #     "data_path": Path("/home/sofa/host_dir/cap-mitea/mitea"),
# #     "checkpoint_path": Path("./checkpoints"),
# #     # views / slices
# #     "view_mode": "strict",   # strict / auto / fixed (we will use strict per your choice)
# #     "num_views": 3,          # choose 1,2,3,4,6 as you like for ablation
# #     "min_mask_pixels": 100,  # threshold to consider a slice "non-empty"
# #     # network + training
# #     "image_size": 256,
# #     "hidden_dim": 64,
# #     "num_inr_layers": 4,
# #     "learning_rate": 1e-5,
# #     "pose_learning_rate": 1e-6,
# #     "num_optimization_steps": 1200,
# #     "alternate_every": 20,       # alternate shape/pose every N steps
# #     "proj_resolution": 256,      # projection resolution (H,W)
# #     # sampling / grid
# #     "grid_resolution": 64,       # for regularizers and evaluation
# #     # misc
# #     "print_every": 50,
# # }

# # # ----------------------------
# # # Utilities: file discovery
# # # ----------------------------
# # def find_mitea_image_files(data_path: Path):
# #     images_dir = data_path / "images"
# #     labels_dir = data_path / "labels"
# #     if not images_dir.exists() or not labels_dir.exists():
# #         print("ERROR: Expected images/ and labels/ subdirectories under", data_path)
# #         if images_dir.exists():
# #             print(" images/ exists; contents:", list(images_dir)[:5])
# #         if labels_dir.exists():
# #             print(" labels/ exists; contents:", list(labels_dir)[:5])
# #         return []

# #     image_files = sorted(images_dir.glob("*.nii*"))
# #     pairs = []
# #     for img in image_files:
# #         stem = img.stem
# #         # locate label file candidate(s)
# #         candidates = list(labels_dir.glob(f"{stem}*"))
# #         if candidates:
# #             pairs.append((img, candidates[0]))
# #         else:
# #             # try removing possible double .nii in stem
# #             if stem.endswith(".nii"):
# #                 alt = stem[:-4]
# #                 candidates2 = list(labels_dir.glob(f"{alt}*"))
# #                 if candidates2:
# #                     pairs.append((img, candidates2[0]))
# #     print(f"Found {len(pairs)} image-label pairs in {images_dir}")
# #     return pairs


# # # ----------------------------
# # # Models
# # # ----------------------------
# # class PositionalEncoding:
# #     def __init__(self, num_freqs=4):
# #         self.num_freqs = num_freqs

# #     def encode(self, coords):
# #         # coords: (N,3) or (...,3)
# #         # produce (..., 3 * 2 * num_freqs)
# #         pe = []
# #         for i in range(self.num_freqs):
# #             freq = 2.0 ** i
# #             pe.append(torch.sin(freq * np.pi * coords))
# #             pe.append(torch.cos(freq * np.pi * coords))
# #         return torch.cat(pe, dim=-1)


# # class ImplicitNeuralRepresentation(nn.Module):
# #     def __init__(self, hidden_dim=64, num_layers=4, pe_freqs=4):
# #         super().__init__()
# #         self.pe = PositionalEncoding(num_freqs=pe_freqs)
# #         input_dim = 3 * 2 * pe_freqs
# #         layers = []
# #         for i in range(num_layers):
# #             in_dim = input_dim if i == 0 else hidden_dim
# #             out_dim = hidden_dim if i < num_layers - 1 else 1
# #             layers.append(nn.Linear(in_dim, out_dim))
# #             if i < num_layers - 1:
# #                 layers.append(nn.ReLU(inplace=True))
# #         self.mlp = nn.Sequential(*layers)

# #     def forward(self, coords):
# #         # coords: (N,3) or (...,3)
# #         orig_shape = coords.shape
# #         coords_flat = coords.view(-1, 3)
# #         pe = self.pe.encode(coords_flat)
# #         out = torch.sigmoid(self.mlp(pe))
# #         return out.view(*orig_shape[:-1], 1)

# #     def sample_grid(self, resolution=64, device=None, requires_grad=False):
# #         resolution = int(resolution)
# #         if isinstance(device, str):
# #             device = torch.device(device)
# #         if device is None:
# #             device = torch.device("cpu")
# #         lin = torch.linspace(-1, 1, resolution, device=device, dtype=torch.float32)
# #         grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), dim=-1)  # (R,R,R,3)
# #         if requires_grad:
# #             occ = self.forward(grid).squeeze(-1)
# #         else:
# #             with torch.no_grad():
# #                 occ = self.forward(grid).squeeze(-1)
# #         return occ


# # class PoseParameters(nn.Module):
# #     """
# #     Per-view Euler (rx,ry,rz) and translation (tx,ty,tz).
# #     Initialized near identity with small noise.
# #     """
# #     def __init__(self, num_views, init_sigma=1e-4):
# #         super().__init__()
# #         init = torch.zeros(num_views, 6)
# #         init += init_sigma * torch.randn_like(init)
# #         # Small default z-translation to move plane slightly in front (optional)
# #         # init[:, 5] = 0.0
# #         self.pose = nn.Parameter(init)

# #     def get_matrices(self, device=None):
# #         if device is None:
# #             device = self.pose.device
# #         p = self.pose.to(device)
# #         rx, ry, rz = p[:, 0], p[:, 1], p[:, 2]
# #         tx, ty, tz = p[:, 3], p[:, 4], p[:, 5]

# #         cx, sx = torch.cos(rx), torch.sin(rx)
# #         cy, sy = torch.cos(ry), torch.sin(ry)
# #         cz, sz = torch.cos(rz), torch.sin(rz)

# #         R = torch.zeros((p.shape[0], 3, 3), device=device)
# #         R[:, 0, 0] = cz * cy
# #         R[:, 0, 1] = cz * sy * sx - sz * cx
# #         R[:, 0, 2] = cz * sy * cx + sz * sx
# #         R[:, 1, 0] = sz * cy
# #         R[:, 1, 1] = sz * sy * sx + cz * cx
# #         R[:, 1, 2] = sz * sy * cx - cz * sx
# #         R[:, 2, 0] = -sy
# #         R[:, 2, 1] = cy * sx
# #         R[:, 2, 2] = cy * cx

# #         extrinsics = torch.eye(4, device=device).unsqueeze(0).repeat(p.shape[0], 1, 1)
# #         extrinsics[:, :3, :3] = R
# #         extrinsics[:, :3, 3] = torch.stack([tx, ty, tz], dim=-1)
# #         return extrinsics


# # # ----------------------------
# # # Data loading + strict slice selection
# # # ----------------------------
# # def load_mitea_subject(img_file: Path, label_file: Path):
# #     try:
# #         import nibabel as nib
# #     except ImportError:
# #         raise ImportError("nibabel required: pip install nibabel")

# #     vol = torch.tensor(nib.load(str(img_file)).get_fdata(), dtype=torch.float32)
# #     seg = torch.tensor(nib.load(str(label_file)).get_fdata(), dtype=torch.float32)

# #     # Normalize volume to [0,1] (safe)
# #     if vol.max() - vol.min() > 0:
# #         vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
# #     else:
# #         vol = torch.zeros_like(vol)

# #     # Print diagnostics
# #     print("Volume (before normalization):")
# #     print(f" Min: {vol.min().item()}, Max: {vol.max().item()}, Mean: {vol.mean().item()}")
# #     print("Segmentation (raw):")
# #     print(f" Min: {seg.min().item()}, Max: {seg.max().item()}, Mean: {seg.mean().item()}")

# #     # Some labels may use integer classes >1 (e.g., multi-class). Binarize:
# #     seg_bin = (seg > 0).float()

# #     # Print statistics for binarized seg
# #     print("Segmentation (binarized):")
# #     print(f" Min: {seg_bin.min().item()}, Max: {seg_bin.max().item()}, Mean: {seg_bin.mean().item()}")
# #     return vol, seg_bin


# # def select_strict_slices(seg: torch.Tensor, num_views: int, min_pixels: int = 100):
# #     """
# #     Return indices of axial slices (along D dimension) that have >= min_pixels
# #     segmentation pixels. Strict mode: must return exactly num_views or raise error.
# #     Selection picks slices with largest segmented area.
# #     """
# #     D = seg.shape[0]
# #     # ensure contiguous before reshape/view operations
# #     seg_cont = seg.contiguous()
# #     counts = seg_cont.reshape(D, -1).sum(dim=1)  # use reshape (safe)
# #     counts_np = counts.cpu().numpy()

# #     # find indices with counts >= min_pixels
# #     valid_idxs = np.where(counts_np >= min_pixels)[0]
# #     if len(valid_idxs) < num_views:
# #         raise ValueError(
# #             f"STRICT mode: not enough non-empty slices (found {len(valid_idxs)}, required {num_views})."
# #             f" Try lowering min_pixels or use different subject."
# #         )
# #     # sort valid indices by descending count and pick top num_views
# #     sorted_idxs = valid_idxs[np.argsort(-counts_np[valid_idxs])]
# #     chosen = sorted_idxs[:num_views].tolist()
# #     chosen_sorted = sorted(chosen)  # keep anatomical order
# #     print(f"Strict-selected slice indices (top {num_views}): {chosen_sorted}")
# #     return chosen_sorted


# # def extract_synthetic_2d_slices_strict(vol: torch.Tensor, seg: torch.Tensor, num_views: int, min_pixels: int, out_size: int = 256):
# #     """
# #     Strict selection: pick exactly num_views axial indices with seg pixels >= min_pixels.
# #     Returns: slices_2d (V,H,W), contours_2d (V,H,W)
# #     """
# #     # make sure vol/seg are contiguous to avoid later non-contiguous errors
# #     vol = vol.contiguous()
# #     seg = seg.contiguous()

# #     D, H, W = vol.shape
# #     chosen = select_strict_slices(seg, num_views=num_views, min_pixels=min_pixels)
# #     slices = []
# #     contours = []
# #     for idx in chosen:
# #         # get contiguous slices
# #         slice_img = vol[idx, :, :].contiguous()
# #         slice_seg = seg[idx, :, :].contiguous()
# #         # resize to out_size using interpolate (expects float tensors)
# #         slice_img_resized = F.interpolate(slice_img.unsqueeze(0).unsqueeze(0),
# #                                           size=(out_size, out_size),
# #                                           mode="bilinear",
# #                                           align_corners=False).squeeze()
# #         slice_seg_resized = F.interpolate(slice_seg.unsqueeze(0).unsqueeze(0),
# #                                           size=(out_size, out_size),
# #                                           mode="nearest").squeeze()
# #         slices.append(slice_img_resized)
# #         contours.append(slice_seg_resized)
# #     slices_tensor = torch.stack(slices)
# #     contours_tensor = torch.stack(contours)
# #     return slices_tensor, contours_tensor


# # # ----------------------------
# # # Losses / Regularizers
# # # ----------------------------
# # def contour_bce_loss(pred, target):
# #     pred = pred.clamp(1e-6, 1 - 1e-6)
# #     return F.binary_cross_entropy(pred, target)


# # def laplacian_smoothness_loss(occupancy_grid, weight=0.02):
# #     lap = (
# #         torch.roll(occupancy_grid, 1, dims=0) +
# #         torch.roll(occupancy_grid, -1, dims=0) +
# #         torch.roll(occupancy_grid, 1, dims=1) +
# #         torch.roll(occupancy_grid, -1, dims=1) +
# #         torch.roll(occupancy_grid, 1, dims=2) +
# #         torch.roll(occupancy_grid, -1, dims=2) -
# #         6 * occupancy_grid
# #     )
# #     surface_mask = (occupancy_grid > 0.2) & (occupancy_grid < 0.8)
# #     if surface_mask.any():
# #         loss = torch.mean(lap[surface_mask] ** 2)
# #     else:
# #         loss = torch.mean(lap ** 2)
# #     return weight * loss


# # def volume_entropy_loss(occupancy_grid, weight=0.01):
# #     eps = 1e-6
# #     entropy = -(occupancy_grid * torch.log(occupancy_grid + eps) + (1 - occupancy_grid) * torch.log(1 - occupancy_grid + eps))
# #     return weight * torch.mean(entropy)


# # def surface_area_loss(occupancy_grid, weight=0.005):
# #     grad_x = torch.diff(occupancy_grid, dim=0, prepend=occupancy_grid[:1])
# #     grad_y = torch.diff(occupancy_grid, dim=1, prepend=occupancy_grid[:, :1])
# #     grad_z = torch.diff(occupancy_grid, dim=2, prepend=occupancy_grid[:, :, :1])
# #     grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-6)
# #     return weight * torch.mean(grad_norm)


# # # ----------------------------
# # # Differentiable projection (plane sampling)
# # # ----------------------------
# # def project_slice_from_inr(model: ImplicitNeuralRepresentation, pose_matrix: torch.Tensor, resolution=256, device="cpu"):
# #     """
# #     Sample a planar grid in plane coordinates (x,y, z=0), then map via pose_matrix (4x4)
# #     to world coords and query model. Return occupancy map (H,W) in [0,1].
# #     pose_matrix: (4,4) torch tensor (plane -> world)
# #     """
# #     if isinstance(device, str):
# #         device = torch.device(device)
# #     lin = torch.linspace(-1.0, 1.0, resolution, device=device, dtype=torch.float32)
# #     xv, yv = torch.meshgrid(lin, lin, indexing="ij")  # (H,W)
# #     zv = torch.zeros_like(xv)
# #     ones = torch.ones_like(xv)
# #     pts_plane = torch.stack([xv, yv, zv, ones], dim=-1)  # (H,W,4)

# #     # Transform
# #     pts_world = pts_plane @ pose_matrix.T    # (H,W,4)
# #     pts_world = pts_world[..., :3]            # (H,W,3)
# #     coords = pts_world.view(-1, 3)
# #     coords = coords.to(next(model.parameters()).device)

# #     # Query model in batches
# #     batch = 4096
# #     out_chunks = []
# #     for i in range(0, coords.shape[0], batch):
# #         c = coords[i:i+batch]
# #         out = model(c)   # (N,1)
# #         out_chunks.append(out)
# #     occ = torch.cat(out_chunks, dim=0).view(resolution, resolution)
# #     return occ


# # # ----------------------------
# # # Optimization (alternating)
# # # ----------------------------
# # def optimize_single_subject(model: ImplicitNeuralRepresentation, slices_2d: torch.Tensor, contours_2d: torch.Tensor, pose_layer: PoseParameters, config: dict, num_steps: int = None):
# #     device = torch.device(config["device"])
# #     model = model.to(device)
# #     pose_layer = pose_layer.to(device)

# #     if num_steps is None:
# #         num_steps = config["num_optimization_steps"]

# #     optimizer_shape = optim.Adam(model.parameters(), lr=config["learning_rate"])
# #     optimizer_pose = optim.Adam(pose_layer.parameters(), lr=config["pose_learning_rate"])

# #     scheduler_shape = optim.lr_scheduler.CosineAnnealingLR(optimizer_shape, T_max=num_steps, eta_min=config["learning_rate"] / 10)
# #     scheduler_pose = optim.lr_scheduler.CosineAnnealingLR(optimizer_pose, T_max=num_steps, eta_min=config["pose_learning_rate"] / 10)

# #     losses = []
# #     contours_device = contours_2d.to(device)

# #     for step in range(num_steps):
# #         extrinsics = pose_layer.get_matrices(device=device)  # (V,4,4)
# #         loss_projection = torch.tensor(0.0, device=device)
# #         preds = []
# #         # project each view (differentiable)
# #         for v in range(contours_device.shape[0]):
# #             pred = project_slice_from_inr(model, extrinsics[v], resolution=config["proj_resolution"], device=device)
# #             preds.append(pred)
# #             target = contours_device[v].to(device)
# #             loss_projection = loss_projection + contour_bce_loss(pred, target)

# #         occ_grid = model.sample_grid(resolution=config["grid_resolution"], device=device, requires_grad=True)
# #         loss_smooth = laplacian_smoothness_loss(occ_grid, weight=0.02)
# #         loss_entropy = volume_entropy_loss(occ_grid, weight=0.01)
# #         loss_area = surface_area_loss(occ_grid, weight=0.005)

# #         loss = loss_projection + loss_smooth + loss_entropy + loss_area

# #         # Alternating updates
# #         if (step // config.get("alternate_every", 1)) % 2 == 0:
# #             optimizer_shape.zero_grad()
# #             loss.backward()
# #             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# #             optimizer_shape.step()
# #             scheduler_shape.step()
# #         else:
# #             optimizer_pose.zero_grad()
# #             loss.backward()
# #             torch.nn.utils.clip_grad_norm_(pose_layer.parameters(), max_norm=1.0)
# #             optimizer_pose.step()
# #             scheduler_pose.step()

# #         losses.append(loss.item())

# #         # logging
# #         if step % config.get("print_every", 50) == 0 or step == num_steps - 1:
# #             print(f"Step {step}/{num_steps}: total loss={loss.item():.6f}, proj={loss_projection.item():.6f}")
# #         if step % 200 == 0:
# #             # print a small stats for pred and target for debugging
# #             with torch.no_grad():
# #                 p = preds[0].detach().cpu()
# #                 t = contours_device[0].detach().cpu()
# #                 print(f" Pred range: min {p.min().item():.4f}, max {p.max().item():.4f}, mean {p.mean().item():.4f}")
# #                 print(f" GT range:  min {t.min().item():.4f}, max {t.max().item():.4f}, mean {t.mean().item():.4f}")
# #     return model, pose_layer, losses


# # # ----------------------------
# # # Evaluation
# # # ----------------------------
# # def evaluate_subject(model: ImplicitNeuralRepresentation, contours_2d: torch.Tensor, pose_layer: PoseParameters, config: dict):
# #     device = torch.device(config["device"])
# #     model = model.to(device)
# #     contours = contours_2d.to(device)
# #     extrinsics = pose_layer.get_matrices(device=device)

# #     dices = []
# #     ious = []
# #     for v in range(contours.shape[0]):
# #         pred = project_slice_from_inr(model, extrinsics[v], resolution=config["proj_resolution"], device=device)
# #         pred_binary = (pred > 0.5).float()
# #         target_binary = (contours[v] > 0.5).float()
# #         intersection = torch.sum(pred_binary * target_binary)
# #         union = torch.sum(pred_binary) + torch.sum(target_binary)
# #         dice = (2 * intersection) / (union + 1e-6)
# #         iou = intersection / (union - intersection + 1e-6)
# #         dices.append(dice.item())
# #         ious.append(iou.item())
# #     metrics = {"dice": float(np.mean(dices)), "iou": float(np.mean(ious))}
# #     return metrics


# # # ----------------------------
# # # MAIN
# # # ----------------------------
# # def main():
# #     device = torch.device(CONFIG["device"])
# #     print("Device:", device)
# #     print("Data path:", CONFIG["data_path"])
# #     CONFIG["checkpoint_path"].mkdir(exist_ok=True)

# #     pairs = find_mitea_image_files(CONFIG["data_path"])[:10]
# #     if not pairs:
# #         print("No data pairs found - check data_path")
# #         return

# #     all_metrics = {"dice": [], "iou": []}

# #     for idx, (img_file, label_file) in enumerate(pairs):
# #         subject_id = img_file.stem
# #         print("\n" + "-" * 60)
# #         print(f"[{idx+1}/{len(pairs)}] Subject: {subject_id}")
# #         try:
# #             vol, seg = load_mitea_subject(img_file, label_file)
# #             D, H, W = vol.shape
# #             print(f"Loaded vol shape: {vol.shape}, seg shape: {seg.shape}")

# #             # Strict slice extraction
# #             try:
# #                 slices_2d, contours_2d = extract_synthetic_2d_slices_strict(
# #                     vol, seg, num_views=CONFIG["num_views"], min_pixels=CONFIG["min_mask_pixels"], out_size=CONFIG["image_size"]
# #                 )
# #             except Exception as e:
# #                 print("Skipping subject due to strict-slice selection:", e)
# #                 continue

# #             print(f"Extracted slices: {slices_2d.shape}, contours: {contours_2d.shape}")
# #             print(f"Contours per-slice mean pixels: {[int(contours_2d[v].sum().item()) for v in range(contours_2d.shape[0])]}")
# #             # init model + pose
# #             model = ImplicitNeuralRepresentation(hidden_dim=CONFIG["hidden_dim"], num_layers=CONFIG["num_inr_layers"])
# #             pose_layer = PoseParameters(CONFIG["num_views"])

# #             # Optimize
# #             model, pose_layer, losses = optimize_single_subject(
# #                 model, slices_2d, contours_2d, pose_layer, CONFIG, num_steps=CONFIG["num_optimization_steps"]
# #             )

# #             # Evaluate
# #             metrics = evaluate_subject(model, contours_2d, pose_layer, CONFIG)
# #             print(f"  Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}")
# #             all_metrics["dice"].append(metrics["dice"])
# #             all_metrics["iou"].append(metrics["iou"])

# #         except Exception as e:
# #             print("ERROR processing", subject_id, e)
# #             traceback.print_exc()

# #     print("\n" + "=" * 60)
# #     print("FINAL RESULTS")
# #     if all_metrics["dice"]:
# #         print(f"Mean Dice: {np.mean(all_metrics['dice']):.4f} ± {np.std(all_metrics['dice']):.4f}")
# #         print(f"Mean IoU:  {np.mean(all_metrics['iou']):.4f} ± {np.std(all_metrics['iou']):.4f}")
# #         print(f"Subjects processed: {len(all_metrics['dice'])}")
# #     else:
# #         print("No successful subjects processed.")


# # if __name__ == "__main__":
# #     main()

# #!/usr/bin/env python3
# """
# Hybrid Echo3D minimal starter (Hybrid C) - STRICT non-empty slice selection

# This script:
# - Uses STRICT non-empty-only slice selection: will pick exactly `num_views`
#   axial slices that contain segmentation pixels above a configurable threshold.
# - Alternating optimization of INR (shape) and PoseParameters (pose)
# - Differentiable planar sampling projection from INR -> 2D slices
# - Strong regularizers (laplacian, entropy, surface area)
# - Robust debugging prints to detect empty masks

# Usage:
#     - Set CONFIG['num_views'] and CONFIG['view_mode']='strict'
#     - Run: python hybrid_strict.py

# Notes:
#     - Requires nibabel, torch, matplotlib (optional for debugging)
#     - Expects MITEA data layout: data_path/images/*.nii* and data_path/labels/*.nii*
# """

# import os
# import traceback
# from pathlib import Path

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# # # ----------------------------
# # # CONFIG
# # # ----------------------------
# # CONFIG = {
# #     # device
# #     "device": "cuda" if torch.cuda.is_available() else "cpu",
# #     # data
# #     "data_path": Path("/home/sofa/host_dir/cap-mitea/mitea"),
# #     "checkpoint_path": Path("./checkpoints"),
# #     # views / slices
# #     "view_mode": "strict",   # strict / auto / fixed (we use strict here)
# #     "num_views": 6,          # choose 1,2,3,4,6 as you like for ablation
# #     "min_mask_pixels": 100,  # threshold to consider a slice "non-empty"
# #     # network + training
# #     "image_size": 256,
# #     "hidden_dim": 64,
# #     "num_inr_layers": 4,
# #     "learning_rate": 1e-5,
# #     "pose_learning_rate": 1e-6,
# #     "num_optimization_steps": 10000,
# #     "alternate_every": 20,       # alternate shape/pose every N steps
# #     "proj_resolution": 256,      # projection resolution (H,W)
# #     # sampling / grid
# #     "grid_resolution": 64,       # for regularizers and evaluation
# #     # misc
# #     "print_every": 50,
# # }
# import os
# from pathlib import Path
# import traceback


# import numpy as np
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# import nibabel as nib
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots


# # ----------------------------
# # CONFIG
# # ----------------------------
# CONFIG = {
# 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
# 'data_path': Path('/home/sofa/host_dir/cap-mitea/mitea'),
# 'checkpoint_path': Path('./checkpoints'),
# 'num_views': 6,
# 'min_mask_pixels': 100,
# 'image_size': 256,
# 'hidden_dim': 64,
# 'num_inr_layers': 4,
# 'learning_rate': 1e-3,
# 'pose_learning_rate': 1e-4,
# 'num_optimization_steps': 1000, # reduce for testing
# 'alternate_every': 20,
# 'proj_resolution': 256,
# 'grid_resolution': 64,
# 'print_every': 50,
# 'html_output_path': Path('./html_viz')
# }
# CONFIG['checkpoint_path'].mkdir(exist_ok=True)
# CONFIG['html_output_path'].mkdir(exist_ok=True)

# # ----------------------------
# # Utilities: file discovery
# # ----------------------------
# def find_mitea_image_files(data_path: Path):
#     images_dir = data_path / "images"
#     labels_dir = data_path / "labels"
#     if not images_dir.exists() or not labels_dir.exists():
#         print("ERROR: Expected images/ and labels/ subdirectories under", data_path)
#         if images_dir.exists():
#             print(" images/ exists; contents:", list(images_dir)[:5])
#         if labels_dir.exists():
#             print(" labels/ exists; contents:", list(labels_dir)[:5])
#         return []

#     image_files = sorted(images_dir.glob("*.nii*"))
#     pairs = []
#     for img in image_files:
#         stem = img.stem
#         # locate label file candidate(s)
#         candidates = list(labels_dir.glob(f"{stem}*"))
#         if candidates:
#             pairs.append((img, candidates[0]))
#         else:
#             # try removing possible double .nii in stem
#             if stem.endswith(".nii"):
#                 alt = stem[:-4]
#                 candidates2 = list(labels_dir.glob(f"{alt}*"))
#                 if candidates2:
#                     pairs.append((img, candidates2[0]))
#     print(f"Found {len(pairs)} image-label pairs in {images_dir}")
#     return pairs


# # ----------------------------
# # Models
# # ----------------------------
# class PositionalEncoding:
#     def __init__(self, num_freqs=4):
#         self.num_freqs = num_freqs

#     def encode(self, coords):
#         # coords: (N,3) or (...,3)
#         # produce (..., 3 * 2 * num_freqs)
#         pe = []
#         for i in range(self.num_freqs):
#             freq = 2.0 ** i
#             pe.append(torch.sin(freq * np.pi * coords))
#             pe.append(torch.cos(freq * np.pi * coords))
#         return torch.cat(pe, dim=-1)


# class ImplicitNeuralRepresentation(nn.Module):
#     def __init__(self, hidden_dim=64, num_layers=4, pe_freqs=4):
#         super().__init__()
#         self.pe = PositionalEncoding(num_freqs=pe_freqs)
#         input_dim = 3 * 2 * pe_freqs
#         layers = []
#         for i in range(num_layers):
#             in_dim = input_dim if i == 0 else hidden_dim
#             out_dim = hidden_dim if i < num_layers - 1 else 1
#             layers.append(nn.Linear(in_dim, out_dim))
#             if i < num_layers - 1:
#                 layers.append(nn.ReLU(inplace=True))
#         self.mlp = nn.Sequential(*layers)

#     def forward(self, coords):
#         # coords: (N,3) or (...,3)
#         orig_shape = coords.shape
#         coords_flat = coords.view(-1, 3)
#         pe = self.pe.encode(coords_flat)
#         out = torch.sigmoid(self.mlp(pe))
#         return out.view(*orig_shape[:-1], 1)

#     def sample_grid(self, resolution=64, device=None, requires_grad=False):
#         resolution = int(resolution)
#         if isinstance(device, str):
#             device = torch.device(device)
#         if device is None:
#             device = torch.device("cpu")
#         lin = torch.linspace(-1, 1, resolution, device=device, dtype=torch.float32)
#         grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), dim=-1)  # (R,R,R,3)
#         if requires_grad:
#             occ = self.forward(grid).squeeze(-1)
#         else:
#             with torch.no_grad():
#                 occ = self.forward(grid).squeeze(-1)
#         return occ


# class PoseParameters(nn.Module):
#     """
#     Per-view Euler (rx,ry,rz) and translation (tx,ty,tz).
#     Initialized near identity with small noise.
#     """
#     def __init__(self, num_views, init_sigma=1e-4):
#         super().__init__()
#         init = torch.zeros(num_views, 6)
#         init += init_sigma * torch.randn_like(init)
#         # Small default z-translation to move plane slightly in front (optional)
#         # init[:, 5] = 0.0
#         self.pose = nn.Parameter(init)

#     def get_matrices(self, device=None):
#         if device is None:
#             device = self.pose.device
#         p = self.pose.to(device)
#         rx, ry, rz = p[:, 0], p[:, 1], p[:, 2]
#         tx, ty, tz = p[:, 3], p[:, 4], p[:, 5]

#         cx, sx = torch.cos(rx), torch.sin(rx)
#         cy, sy = torch.cos(ry), torch.sin(ry)
#         cz, sz = torch.cos(rz), torch.sin(rz)

#         R = torch.zeros((p.shape[0], 3, 3), device=device)
#         R[:, 0, 0] = cz * cy
#         R[:, 0, 1] = cz * sy * sx - sz * cx
#         R[:, 0, 2] = cz * sy * cx + sz * sx
#         R[:, 1, 0] = sz * cy
#         R[:, 1, 1] = sz * sy * sx + cz * cx
#         R[:, 1, 2] = sz * sy * cx - cz * sx
#         R[:, 2, 0] = -sy
#         R[:, 2, 1] = cy * sx
#         R[:, 2, 2] = cy * cx

#         extrinsics = torch.eye(4, device=device).unsqueeze(0).repeat(p.shape[0], 1, 1)
#         extrinsics[:, :3, :3] = R
#         extrinsics[:, :3, 3] = torch.stack([tx, ty, tz], dim=-1)
#         return extrinsics


# # ----------------------------
# # Data loading + strict slice selection
# # ----------------------------
# def load_mitea_subject(img_file: Path, label_file: Path):
#     try:
#         import nibabel as nib
#     except ImportError:
#         raise ImportError("nibabel required: pip install nibabel")

#     vol = torch.tensor(nib.load(str(img_file)).get_fdata(), dtype=torch.float32)
#     seg = torch.tensor(nib.load(str(label_file)).get_fdata(), dtype=torch.float32)

#     # Normalize volume to [0,1] (safe)
#     if vol.max() - vol.min() > 0:
#         vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
#     else:
#         vol = torch.zeros_like(vol)

#     # Print diagnostics
#     print("Volume (before normalization):")
#     print(f" Min: {vol.min().item()}, Max: {vol.max().item()}, Mean: {vol.mean().item()}")
#     print("Segmentation (raw):")
#     print(f" Min: {seg.min().item()}, Max: {seg.max().item()}, Mean: {seg.mean().item()}")

#     # Some labels may use integer classes >1 (e.g., multi-class). Binarize:
#     seg_bin = (seg > 0).float()

#     # Print statistics for binarized seg
#     print("Segmentation (binarized):")
#     print(f" Min: {seg_bin.min().item()}, Max: {seg_bin.max().item()}, Mean: {seg_bin.mean().item()}")
#     return vol, seg_bin


# def select_strict_slices(seg: torch.Tensor, num_views: int, min_pixels: int = 100):
#     """
#     Return indices of axial slices (along D dimension) that have >= min_pixels
#     segmentation pixels. Strict mode: must return exactly num_views or raise error.
#     Selection picks slices with largest segmented area.
#     """
#     D = seg.shape[0]
#     # ensure contiguous before reshape/view operations
#     seg_cont = seg.contiguous()
#     counts = seg_cont.reshape(D, -1).sum(dim=1)  # use reshape (safe)
#     counts_np = counts.cpu().numpy()

#     # find indices with counts >= min_pixels
#     valid_idxs = np.where(counts_np >= min_pixels)[0]
#     if len(valid_idxs) < num_views:
#         raise ValueError(
#             f"STRICT mode: not enough non-empty slices (found {len(valid_idxs)}, required {num_views})."
#             f" Try lowering min_pixels or use different subject."
#         )
#     # sort valid indices by descending count and pick top num_views
#     sorted_idxs = valid_idxs[np.argsort(-counts_np[valid_idxs])]
#     chosen = sorted_idxs[:num_views].tolist()
#     chosen_sorted = sorted(chosen)  # keep anatomical order
#     print(f"Strict-selected slice indices (top {num_views}): {chosen_sorted}")
#     return chosen_sorted


# def extract_synthetic_2d_slices_strict(vol: torch.Tensor, seg: torch.Tensor, num_views: int, min_pixels: int, out_size: int = 256):
#     """
#     Strict selection: pick exactly num_views axial indices with seg pixels >= min_pixels.
#     Returns: slices_2d (V,H,W), contours_2d (V,H,W)
#     """
#     # make sure vol/seg are contiguous to avoid later non-contiguous errors
#     vol = vol.contiguous()
#     seg = seg.contiguous()

#     D, H, W = vol.shape
#     chosen = select_strict_slices(seg, num_views=num_views, min_pixels=min_pixels)
#     slices = []
#     contours = []
#     for idx in chosen:
#         # get contiguous slices
#         slice_img = vol[idx, :, :].contiguous()
#         slice_seg = seg[idx, :, :].contiguous()
#         # resize to out_size using interpolate (expects float tensors)
#         slice_img_resized = F.interpolate(slice_img.unsqueeze(0).unsqueeze(0),
#                                           size=(out_size, out_size),
#                                           mode="bilinear",
#                                           align_corners=False).squeeze()
#         slice_seg_resized = F.interpolate(slice_seg.unsqueeze(0).unsqueeze(0),
#                                           size=(out_size, out_size),
#                                           mode="nearest").squeeze()
#         slices.append(slice_img_resized)
#         contours.append(slice_seg_resized)
#     slices_tensor = torch.stack(slices)
#     contours_tensor = torch.stack(contours)
#     return slices_tensor, contours_tensor, chosen


# # ----------------------------
# # Losses / Regularizers
# # ----------------------------
# def contour_bce_loss(pred, target):
#     pred = pred.clamp(1e-6, 1 - 1e-6)
#     return F.binary_cross_entropy(pred, target)


# def laplacian_smoothness_loss(occupancy_grid, weight=0.02):
#     lap = (
#         torch.roll(occupancy_grid, 1, dims=0) +
#         torch.roll(occupancy_grid, -1, dims=0) +
#         torch.roll(occupancy_grid, 1, dims=1) +
#         torch.roll(occupancy_grid, -1, dims=1) +
#         torch.roll(occupancy_grid, 1, dims=2) +
#         torch.roll(occupancy_grid, -1, dims=2) -
#         6 * occupancy_grid
#     )
#     surface_mask = (occupancy_grid > 0.2) & (occupancy_grid < 0.8)
#     if surface_mask.any():
#         loss = torch.mean(lap[surface_mask] ** 2)
#     else:
#         loss = torch.mean(lap ** 2)
#     return weight * loss


# def volume_entropy_loss(occupancy_grid, weight=0.01):
#     eps = 1e-6
#     entropy = -(occupancy_grid * torch.log(occupancy_grid + eps) + (1 - occupancy_grid) * torch.log(1 - occupancy_grid + eps))
#     return weight * torch.mean(entropy)


# def surface_area_loss(occupancy_grid, weight=0.005):
#     grad_x = torch.diff(occupancy_grid, dim=0, prepend=occupancy_grid[:1])
#     grad_y = torch.diff(occupancy_grid, dim=1, prepend=occupancy_grid[:, :1])
#     grad_z = torch.diff(occupancy_grid, dim=2, prepend=occupancy_grid[:, :, :1])
#     grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-6)
#     return weight * torch.mean(grad_norm)


# # ----------------------------
# # Differentiable projection (plane sampling)
# # ----------------------------
# def project_slice_from_inr(model, pose_matrix, resolution=256, device="cpu", z_depth=0.0):
#     if isinstance(device, str):
#         device = torch.device(device)

#     lin = torch.linspace(-1.0, 1.0, resolution, device=device)
#     xv, yv = torch.meshgrid(lin, lin, indexing="ij")

#     # --- FIX: use z_depth instead of 0 ---
#     zv = torch.full_like(xv, fill_value=z_depth)

#     ones = torch.ones_like(xv)
#     pts_plane = torch.stack([xv, yv, zv, ones], dim=-1)

#     pts_world = pts_plane @ pose_matrix.T
#     pts_world = pts_world[..., :3]
#     coords = pts_world.reshape(-1, 3)

#     coords = coords.to(next(model.parameters()).device)

#     # batch query
#     batch = 4096
#     outs = []
#     for i in range(0, coords.shape[0], batch):
#         outs.append(model(coords[i:i+batch]))
#     occ = torch.cat(outs, dim=0).view(resolution, resolution)

#     return occ


# # ----------------------------
# # Optimization (alternating)
# # ----------------------------
# def optimize_single_subject(model: ImplicitNeuralRepresentation,
#                             slices_2d: torch.Tensor,
#                             contours_2d: torch.Tensor,
#                             pose_layer: PoseParameters,
#                             chosen,
#                             config: dict,
#                             num_steps: int = None,
#                             D: int = None):  # D: number of axial slices
#     if D is None:
#         raise ValueError("D (number of axial slices) must be provided.")

#     device = torch.device(config["device"])
#     model = model.to(device)
#     pose_layer = pose_layer.to(device)

#     if num_steps is None:
#         num_steps = config["num_optimization_steps"]

#     optimizer_shape = optim.Adam(model.parameters(), lr=config["learning_rate"])
#     optimizer_pose = optim.Adam(pose_layer.parameters(), lr=config["pose_learning_rate"])

#     scheduler_shape = optim.lr_scheduler.CosineAnnealingLR(
#         optimizer_shape, T_max=num_steps, eta_min=config["learning_rate"] / 10
#     )
#     scheduler_pose = optim.lr_scheduler.CosineAnnealingLR(
#         optimizer_pose, T_max=num_steps, eta_min=config["pose_learning_rate"] / 10
#     )

#     losses = []
#     contours_device = contours_2d.to(device)

#     for step in range(num_steps):
#         extrinsics = pose_layer.get_matrices(device=device)  # (V,4,4)
#         loss_projection = torch.tensor(0.0, device=device)
#         preds = []

#         # project each view (differentiable)
#         for v in range(contours_device.shape[0]):
#             slice_index = chosen[v]  # axial slice index
#             z_depth = 2 * (slice_index / (D - 1)) - 1  # map [0, D-1] -> [-1, 1]

#             pred = project_slice_from_inr(
#                 model,
#                 extrinsics[v],
#                 resolution=config["proj_resolution"],
#                 device=device,
#                 z_depth=z_depth
#             )
#             preds.append(pred)
#             target = contours_device[v].to(device)
#             loss_projection += contour_bce_loss(pred, target)

#         occ_grid = model.sample_grid(resolution=config["grid_resolution"], device=device, requires_grad=True)
#         loss_smooth = laplacian_smoothness_loss(occ_grid, weight=0.02)
#         loss_entropy = volume_entropy_loss(occ_grid, weight=0.01)
#         loss_area = surface_area_loss(occ_grid, weight=0.005)

#         loss = loss_projection + loss_smooth + loss_entropy + loss_area

#         # Alternating updates
#         if (step // config.get("alternate_every", 1)) % 2 == 0:
#             optimizer_shape.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer_shape.step()
#             scheduler_shape.step()
#         else:
#             optimizer_pose.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(pose_layer.parameters(), max_norm=1.0)
#             optimizer_pose.step()
#             scheduler_pose.step()

#         losses.append(loss.item())

#         # logging
#         if step % config.get("print_every", 50) == 0 or step == num_steps - 1:
#             print(f"Step {step}/{num_steps}: total loss={loss.item():.6f}, proj={loss_projection.item():.6f}")
#         if step % 200 == 0:
#             with torch.no_grad():
#                 p = preds[0].detach().cpu()
#                 t = contours_device[0].detach().cpu()
#                 print(f" Pred range: min {p.min().item():.4f}, max {p.max().item():.4f}, mean {p.mean().item():.4f}")
#                 print(f" GT range:  min {t.min().item():.4f}, max {t.max().item():.4f}, mean {t.mean().item():.4f}")

#     return model, pose_layer, losses

# # ----------------------------
# # Evaluation
# # ----------------------------
# def evaluate_subject(model: ImplicitNeuralRepresentation, contours_2d: torch.Tensor, pose_layer: PoseParameters, config: dict):
#     device = torch.device(config["device"])
#     model = model.to(device)
#     contours = contours_2d.to(device)
#     extrinsics = pose_layer.get_matrices(device=device)

#     dices = []
#     ious = []
#     for v in range(contours.shape[0]):
#         pred = project_slice_from_inr(model, extrinsics[v], resolution=config["proj_resolution"], device=device)
#         pred_binary = (pred > 0.5).float()
#         target_binary = (contours[v] > 0.5).float()
#         intersection = torch.sum(pred_binary * target_binary)
#         union = torch.sum(pred_binary) + torch.sum(target_binary)
#         dice = (2 * intersection) / (union + 1e-6)
#         iou = intersection / (union - intersection + 1e-6)
#         dices.append(dice.item())
#         ious.append(iou.item())
#     metrics = {"dice": float(np.mean(dices)), "iou": float(np.mean(ious))}
#     return metrics

# def save_patient_html(patient_id, scan_results, html_path):
        
#     # scan_results: dict {scan_name: {'pred': tensor, 'gt': tensor}}
#     fig = make_subplots(rows=len(scan_results), cols=CONFIG['num_views'],
#     subplot_titles=[f'{k} Slice {i}' for k in scan_results.keys() for i in range(CONFIG['num_views'])])


#     row = 1
#     for scan_name, data in scan_results.items():
#         pred = data['pred'].cpu().numpy()
#         gt = data['gt'].cpu().numpy()
#     for col in range(CONFIG['num_views']):
#         slice_pred = pred[col]
#         slice_gt = gt[col]
#         # Combine GT and prediction as RGB overlay
#         combined = np.stack([slice_pred, slice_gt, np.zeros_like(slice_pred)], axis=-1)
#         fig.add_trace(go.Image(z=(combined * 255).astype(np.uint8)), row=row, col=col+1)
#         row += 1


#     html_file = html_path / f'{patient_id}_comparison.html'
#     fig.update_layout(height=300*len(scan_results), width=300*CONFIG['num_views'], title_text=f'Patient {patient_id} Recon vs GT')
#     fig.write_html(str(html_file))
#     print(f'Saved HTML visualization for patient {patient_id} at {html_file}')

# # ----------------------------
# # MAIN
# # ----------------------------
# # def main():
# #     device = torch.device(CONFIG["device"])
# #     print("Device:", device)
# #     print("Data path:", CONFIG["data_path"])
# #     CONFIG["checkpoint_path"].mkdir(exist_ok=True)

# #     pairs = find_mitea_image_files(CONFIG["data_path"])[:10]
# #     if not pairs:
# #         print("No data pairs found - check data_path")
# #         return

# #     all_metrics = {"dice": [], "iou": []}

# #     for idx, (img_file, label_file) in enumerate(pairs):
# #         subject_id = img_file.stem
# #         print("\n" + "-" * 60)
# #         print(f"[{idx+1}/{len(pairs)}] Subject: {subject_id}")
# #         try:
# #             vol, seg = load_mitea_subject(img_file, label_file)
# #             D, H, W = vol.shape
# #             print(f"Loaded vol shape: {vol.shape}, seg shape: {seg.shape}")

# #             # Strict slice extraction
# #             try:
# #                 slices_2d, contours_2d, chosen = extract_synthetic_2d_slices_strict(
# #                     vol, seg, num_views=CONFIG["num_views"], min_pixels=CONFIG["min_mask_pixels"], out_size=CONFIG["image_size"]
# #                 )
# #             except Exception as e:
# #                 print("Skipping subject due to strict-slice selection:", e)
# #                 continue

# #             print(f"Extracted slices: {slices_2d.shape}, contours: {contours_2d.shape}")
# #             print(f"Contours per-slice mean pixels: {[int(contours_2d[v].sum().item()) for v in range(contours_2d.shape[0])]}")
# #             # init model + pose
# #             model = ImplicitNeuralRepresentation(hidden_dim=CONFIG["hidden_dim"], num_layers=CONFIG["num_inr_layers"])
# #             pose_layer = PoseParameters(CONFIG["num_views"])

# #             # Optimize
# #             model, pose_layer, losses = optimize_single_subject(
# #                 model, slices_2d, contours_2d, pose_layer,chosen, CONFIG, num_steps=CONFIG["num_optimization_steps"],D=D
# #             )

# #             # Evaluate
# #             metrics = evaluate_subject(model, contours_2d, pose_layer, CONFIG)
# #             print(f"  Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}")
# #             all_metrics["dice"].append(metrics["dice"])
# #             all_metrics["iou"].append(metrics["iou"])

# #         except Exception as e:
# #             print("ERROR processing", subject_id, e)
# #             traceback.print_exc()

# #     print("\n" + "=" * 60)
# #     print("FINAL RESULTS")
# #     if all_metrics["dice"]:
# #         print(f"Mean Dice: {np.mean(all_metrics['dice']):.4f} ± {np.std(all_metrics['dice']):.4f}")
# #         print(f"Mean IoU:  {np.mean(all_metrics['iou']):.4f} ± {np.std(all_metrics['iou']):.4f}")
# #         print(f"Subjects processed: {len(all_metrics['dice'])}")
# #     else:
# #         print("No successful subjects processed.")


# def main():
#     import plotly.graph_objects as go
#     from plotly.subplots import make_subplots
#     import nibabel as nib

#     device = torch.device(CONFIG["device"])
#     print("Device:", device)
#     print("Data path:", CONFIG["data_path"])
#     CONFIG["checkpoint_path"].mkdir(exist_ok=True)
#     html_dir = CONFIG["checkpoint_path"] / "html_visualizations"
#     html_dir.mkdir(exist_ok=True)

#     pairs = find_mitea_image_files(CONFIG["data_path"])[:10]
#     if not pairs:
#         print("No data pairs found - check data_path")
#         return

#     all_metrics = {"dice": [], "iou": []}

#     for idx, (img_file, label_file) in enumerate(pairs):
#         subject_id = img_file.stem
#         print("\n" + "-" * 60)
#         print(f"[{idx+1}/{len(pairs)}] Subject: {subject_id}")
#         try:
#             vol, seg = load_mitea_subject(img_file, label_file)
#             D, H, W = vol.shape
#             print(f"Loaded vol shape: {vol.shape}, seg shape: {seg.shape}")

#             # Strict slice extraction
#             try:
#                 slices_2d, contours_2d, chosen = extract_synthetic_2d_slices_strict(
#                     vol, seg, num_views=CONFIG["num_views"], min_pixels=CONFIG["min_mask_pixels"], out_size=CONFIG["image_size"]
#                 )
#             except Exception as e:
#                 print("Skipping subject due to strict-slice selection:", e)
#                 continue

#             print(f"Extracted slices: {slices_2d.shape}, contours: {contours_2d.shape}")
#             print(f"Contours per-slice mean pixels: {[int(contours_2d[v].sum().item()) for v in range(contours_2d.shape[0])]}")
            
#             # init model + pose
#             model = ImplicitNeuralRepresentation(hidden_dim=CONFIG["hidden_dim"], num_layers=CONFIG["num_inr_layers"])
#             pose_layer = PoseParameters(CONFIG["num_views"])

#             # Optimize
#             model, pose_layer, losses = optimize_single_subject(
#                 model, slices_2d, contours_2d, pose_layer, chosen, CONFIG,
#                 num_steps=CONFIG["num_optimization_steps"], D=D
#             )

#             # Evaluate
#             metrics = evaluate_subject(model, contours_2d, pose_layer, CONFIG)
#             print(f"  Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}")
#             all_metrics["dice"].append(metrics["dice"])
#             all_metrics["iou"].append(metrics["iou"])

#             # Save predicted volume as .nii.gz
#             print("Saving predicted occupancy grid as NIfTI...")
#             occ_grid = model.sample_grid(resolution=D, device=device, requires_grad=False).cpu().numpy()
#             pred_nii = nib.Nifti1Image(occ_grid.astype(np.float32), affine=np.eye(4))
#             pred_file = CONFIG["checkpoint_path"] / f"{subject_id}_pred.nii.gz"
#             nib.save(pred_nii, str(pred_file))
#             print(f" Saved: {pred_file}")

#             # ----------------------------
#             # Generate HTML visualization
#             # ----------------------------
#             fig = make_subplots(rows=1, cols=CONFIG["num_views"], subplot_titles=[f"Slice {s}" for s in chosen])

#             extrinsics = pose_layer.get_matrices(device=device)
#             for v in range(CONFIG["num_views"]):
#                 slice_index = chosen[v]
#                 z_depth = 2 * (slice_index / (D - 1)) - 1
#                 pred_slice = project_slice_from_inr(model, extrinsics[v], resolution=CONFIG["proj_resolution"], device=device, z_depth=z_depth).detach().cpu().numpy()
#                 gt_slice = contours_2d[v].detach().cpu().numpy()

#                 # Overlay: red = prediction, green = GT
#                 overlay = np.zeros((pred_slice.shape[0], pred_slice.shape[1], 3))
#                 overlay[..., 0] = pred_slice  # red channel
#                 overlay[..., 1] = gt_slice    # green channel

#                 fig.add_trace(
#                     go.Image(z=(overlay*255).astype(np.uint8)),
#                     row=1, col=v+1
#                 )

#             html_file = html_dir / f"{subject_id}_slices.html"
#             fig.update_layout(title_text=f"{subject_id} - Pred vs GT slices", width=300*CONFIG["num_views"], height=300)
#             fig.write_html(str(html_file))
#             print(f"HTML visualization saved: {html_file}")

#         except Exception as e:
#             print("ERROR processing", subject_id, e)
#             traceback.print_exc()

#     print("\n" + "=" * 60)
#     print("FINAL RESULTS")
#     if all_metrics["dice"]:
#         print(f"Mean Dice: {np.mean(all_metrics['dice']):.4f} ± {np.std(all_metrics['dice']):.4f}")
#         print(f"Mean IoU:  {np.mean(all_metrics['iou']):.4f} ± {np.std(all_metrics['iou']):.4f}")
#         print(f"Subjects processed: {len(all_metrics['dice'])}")
#     else:
#         print("No successful subjects processed.")

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
Hybrid Echo3D minimal starter (Hybrid C) - STRICT non-empty slice selection
with consistent z-depth usage, 2D HTML overlays, and 3D mesh visualizations.

- Strict non-empty slice selection (top-K slices by mask area)
- Alternating optimization of INR (shape) and PoseParameters (pose)
- Differentiable planar sampling projection from INR -> 2D slices
- Strong regularizers (laplacian, entropy, surface area) for shape steps
- Consistent z-depth mapping in training and evaluation
- 2D HTML overlays: red = prediction, green = ground truth
- 3D mesh HTML overlays: red = prediction, green = ground truth
"""

import os
import traceback
from pathlib import Path

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
CONFIG = {
    # device
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # data paths
    "data_path": Path("/home/sofa/host_dir/cap-mitea/mitea"),
    "checkpoint_path": Path("./checkpoints"),

    # views / slices
    "num_views": 6,
    "min_mask_pixels": 100,

    # network + training
    "image_size": 256,
    "hidden_dim": 64,
    "num_inr_layers": 4,
    "learning_rate": 1e-3,
    "pose_learning_rate": 1e-4,
    "num_optimization_steps": 1000,
    "alternate_every": 20,       # alternate shape/pose every N steps
    "proj_resolution": 256,      # projection resolution (H,W)

    # sampling / grid
    "grid_resolution": 64,       # for regularizers (Laplace / entropy / area)
    "eval_grid_resolution": 128, # for 3D volume metrics + mesh export

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
def find_mitea_image_files(data_path: Path):
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
    pairs = []
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
    def __init__(self, num_freqs=4):
        self.num_freqs = num_freqs

    def encode(self, coords):
        # coords: (N,3) or (...,3)
        pe = []
        for i in range(self.num_freqs):
            freq = 2.0 ** i
            pe.append(torch.sin(freq * np.pi * coords))
            pe.append(torch.cos(freq * np.pi * coords))
        return torch.cat(pe, dim=-1)


class ImplicitNeuralRepresentation(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4, pe_freqs=4):
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

    def forward(self, coords):
        orig_shape = coords.shape
        coords_flat = coords.view(-1, 3)
        pe = self.pe.encode(coords_flat)
        out = torch.sigmoid(self.mlp(pe))
        return out.view(*orig_shape[:-1], 1)

    def sample_grid(self, resolution=64, device=None, requires_grad=False):
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
    def __init__(self, num_views, init_sigma=1e-4):
        super().__init__()
        init = torch.zeros(num_views, 6)
        init += init_sigma * torch.randn_like(init)
        self.pose = nn.Parameter(init)

    def get_matrices(self, device=None):
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
def load_mitea_subject(img_file: Path, label_file: Path):
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


def select_strict_slices(seg: torch.Tensor,
                         num_views: int,
                         min_pixels: int = 100):
    """
    Return indices of axial slices (along D) with >= min_pixels
    foreground pixels. Strict: must return exactly num_views or raise error.
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
    sorted_idxs = valid_idxs[np.argsort(-counts_np[valid_idxs])]
    chosen = sorted_idxs[:num_views].tolist()
    chosen_sorted = sorted(chosen)
    print(f"Strict-selected slice indices (top {num_views}): {chosen_sorted}")
    return chosen_sorted


def extract_synthetic_2d_slices_strict(vol: torch.Tensor,
                                       seg: torch.Tensor,
                                       num_views: int,
                                       min_pixels: int,
                                       out_size: int = 256):
    """
    Strict selection: pick exactly num_views axial indices with seg pixels >= min_pixels.
    Returns: slices_2d (V,H,W), contours_2d (V,H,W), chosen_indices (list of axial indices)
    """
    vol = vol.contiguous()
    seg = seg.contiguous()

    D, H, W = vol.shape
    chosen = select_strict_slices(seg, num_views=num_views, min_pixels=min_pixels)
    slices = []
    contours = []
    for idx in chosen:
        slice_img = vol[idx, :, :].contiguous()
        slice_seg = seg[idx, :, :].contiguous()

        slice_img_resized = F.interpolate(
            slice_img.unsqueeze(0).unsqueeze(0),
            size=(out_size, out_size),
            mode="bilinear",
            align_corners=False
        ).squeeze()

        slice_seg_resized = F.interpolate(
            slice_seg.unsqueeze(0).unsqueeze(0),
            size=(out_size, out_size),
            mode="nearest"
        ).squeeze()

        slices.append(slice_img_resized)
        contours.append(slice_seg_resized)

    slices_tensor = torch.stack(slices)
    contours_tensor = torch.stack(contours)
    return slices_tensor, contours_tensor, chosen


# ----------------------------
# Losses / Regularizers
# ----------------------------
def contour_bce_loss(pred, target):
    pred = pred.clamp(1e-6, 1 - 1e-6)
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
    entropy = -(occupancy_grid * torch.log(occupancy_grid + eps) +
                (1 - occupancy_grid) * torch.log(1 - occupancy_grid + eps))
    return weight * torch.mean(entropy)


def surface_area_loss(occupancy_grid, weight=0.005):
    grad_x = torch.diff(occupancy_grid, dim=0, prepend=occupancy_grid[:1])
    grad_y = torch.diff(occupancy_grid, dim=1, prepend=occupancy_grid[:, :1])
    grad_z = torch.diff(occupancy_grid, dim=2, prepend=occupancy_grid[:, :, :1])
    grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-6)
    return weight * torch.mean(grad_norm)


# ----------------------------
# Differentiable projection (plane sampling)
# ----------------------------
def project_slice_from_inr(model,
                           pose_matrix: torch.Tensor,
                           resolution=256,
                           device="cpu",
                           z_depth=0.0):
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
        outs.append(model(coords[i:i+batch]))
    occ = torch.cat(outs, dim=0).view(resolution, resolution)
    return occ


# ----------------------------
# Optimization (alternating)
# ----------------------------
def optimize_single_subject(model: ImplicitNeuralRepresentation,
                            slices_2d: torch.Tensor,
                            contours_2d: torch.Tensor,
                            pose_layer: PoseParameters,
                            chosen,
                            config: dict,
                            num_steps: int,
                            D: int):
    """
    Alternating optimization:
    - shape step: update INR weights using proj + regularizers
    - pose step:  update pose parameters using proj-only loss
    """
    device = torch.device(config["device"])
    model = model.to(device)
    pose_layer = pose_layer.to(device)

    optimizer_shape = optim.Adam(model.parameters(), lr=config["learning_rate"])
    optimizer_pose = optim.Adam(pose_layer.parameters(), lr=config["pose_learning_rate"])

    scheduler_shape = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_shape, T_max=num_steps, eta_min=config["learning_rate"] / 10
    )
    scheduler_pose = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_pose, T_max=num_steps, eta_min=config["pose_learning_rate"] / 10
    )

    contours_device = contours_2d.to(device)
    losses = []

    for step in range(num_steps):
        # decide which block to update
        shape_step = ((step // config.get("alternate_every", 1)) % 2 == 0)

        if shape_step:
            # use extrinsics detached so pose doesn't receive grads
            extrinsics = pose_layer.get_matrices(device=device).detach()
        else:
            extrinsics = pose_layer.get_matrices(device=device)

        loss_projection = torch.tensor(0.0, device=device)
        preds_debug = []

        for v in range(contours_device.shape[0]):
            slice_index = chosen[v]
            z_depth = 2.0 * (slice_index / (D - 1)) - 1.0  # map [0, D-1] -> [-1, 1]
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
            occ_grid = model.sample_grid(
                resolution=config["grid_resolution"],
                device=device,
                requires_grad=True
            )
            loss_smooth = laplacian_smoothness_loss(occ_grid, weight=0.02)
            loss_entropy = volume_entropy_loss(occ_grid, weight=0.01)
            loss_area = surface_area_loss(occ_grid, weight=0.005)
            loss = loss_projection + loss_smooth + loss_entropy + loss_area
        else:
            # pose update: projection loss only
            loss = loss_projection

        optimizer_shape.zero_grad()
        optimizer_pose.zero_grad()
        loss.backward()

        if shape_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_shape.step()
            scheduler_shape.step()
        else:
            torch.nn.utils.clip_grad_norm_(pose_layer.parameters(), max_norm=1.0)
            optimizer_pose.step()
            scheduler_pose.step()

        losses.append(loss.item())

        if step % config.get("print_every", 50) == 0 or step == num_steps - 1:
            print(f"Step {step}/{num_steps}: "
                  f"total={loss.item():.6f}, proj={loss_projection.item():.6f}, "
                  f"{'shape' if shape_step else 'pose'}-step")

        if step % 200 == 0:
            with torch.no_grad():
                p = preds_debug[0].detach().cpu()
                t = contours_device[0].detach().cpu()
                print(f" Pred range: min {p.min().item():.4f}, max {p.max().item():.4f}, "
                      f"mean {p.mean().item():.4f}")
                print(f" GT range:  min {t.min().item():.4f}, max {t.max().item():.4f}, "
                      f"mean {t.mean().item():.4f}")

    return model, pose_layer, losses


# ----------------------------
# 2D evaluation (consistent z-depth)
# ----------------------------
def evaluate_subject_2d(model: ImplicitNeuralRepresentation,
                        contours_2d: torch.Tensor,
                        pose_layer: PoseParameters,
                        chosen,
                        D: int,
                        config: dict):
    """
    2D contour-level Dice / IoU, using the *same* z-depth mapping as training.
    """
    device = torch.device(config["device"])
    model = model.to(device)
    contours = contours_2d.to(device)
    extrinsics = pose_layer.get_matrices(device=device)

    dices = []
    ious = []
    for v in range(contours.shape[0]):
        slice_index = chosen[v]
        z_depth = 2.0 * (slice_index / (D - 1)) - 1.0

        pred = project_slice_from_inr(
            model,
            extrinsics[v],
            resolution=config["proj_resolution"],
            device=device,
            z_depth=z_depth
        )
        pred_binary = (pred > 0.5).float()
        target_binary = (contours[v] > 0.5).float()

        intersection = torch.sum(pred_binary * target_binary)
        union = torch.sum(pred_binary) + torch.sum(target_binary)
        dice = (2 * intersection) / (union + 1e-6)
        iou = intersection / (union - intersection + 1e-6)
        dices.append(dice.item())
        ious.append(iou.item())

    metrics = {"dice_2d": float(np.mean(dices)),
               "iou_2d": float(np.mean(ious))}
    return metrics


# ----------------------------
# 3D evaluation + mesh export
# ----------------------------
def evaluate_subject_3d_and_mesh(model: ImplicitNeuralRepresentation,
                                 seg_vol: torch.Tensor,
                                 subject_id: str,
                                 config: dict):
    """
    - Resample GT seg to eval_grid_resolution^3
    - Sample INR on same cube
    - Compute 3D Dice / IoU
    - Export Plotly Mesh3d for pred vs GT
    """
    device = torch.device(config["device"])
    model = model.to(device)
    R = int(config.get("eval_grid_resolution", 64))
    thr = float(config.get("mesh_threshold", 0.5))

    # Sample occupancy grid from INR
    occ_grid = model.sample_grid(resolution=R, device=device, requires_grad=False)
    occ_grid_cpu = occ_grid.detach().cpu()  # (R,R,R)

    # Resample GT segmentation to same resolution
    seg_t = seg_vol.unsqueeze(0).unsqueeze(0).float()  # (1,1,D,H,W)
    seg_resampled = F.interpolate(seg_t, size=(R, R, R), mode="nearest")
    seg_resampled = seg_resampled[0, 0]  # (R,R,R)

    pred_bin = (occ_grid_cpu > thr).float()
    gt_bin = (seg_resampled > 0.5).float()

    intersection = (pred_bin * gt_bin).sum().item()
    union = pred_bin.sum().item() + gt_bin.sum().item()
    dice = (2.0 * intersection) / (union + 1e-6)
    iou = intersection / (union - intersection + 1e-6)

    print(f"  3D Dice: {dice:.4f}, 3D IoU: {iou:.4f}")

    # 3D mesh visualization (if scikit-image is available)
    try:
        from skimage import measure
    except ImportError:
        print("  scikit-image not installed, skipping 3D mesh export. "
              "Install via `pip install scikit-image` if needed.")
        return {"dice_3d": dice, "iou_3d": iou}

    occ_np = occ_grid_cpu.numpy()
    seg_np = seg_resampled.numpy()

    # marching cubes
    try:
        verts_pred, faces_pred, _, _ = measure.marching_cubes(occ_np, level=thr)
        verts_gt, faces_gt, _, _ = measure.marching_cubes(seg_np, level=0.5)
    except ValueError as e:
        print("  Marching cubes failed (likely empty volume):", e)
        return {"dice_3d": dice, "iou_3d": iou}

    # normalize to [0,1] cube for visualization
    def norm_verts(v):
        scale = np.array([R - 1, R - 1, R - 1], dtype=np.float32)
        return v / scale

    vp = norm_verts(verts_pred)
    vg = norm_verts(verts_gt)

    fig = go.Figure()

    # GT mesh in green
    fig.add_trace(go.Mesh3d(
        x=vg[:, 0], y=vg[:, 1], z=vg[:, 2],
        i=faces_gt[:, 0], j=faces_gt[:, 1], k=faces_gt[:, 2],
        color="green", opacity=0.5, name="Ground Truth"
    ))

    # Pred mesh in red
    fig.add_trace(go.Mesh3d(
        x=vp[:, 0], y=vp[:, 1], z=vp[:, 2],
        i=faces_pred[:, 0], j=faces_pred[:, 1], k=faces_pred[:, 2],
        color="red", opacity=0.5, name="Prediction"
    ))

    fig.update_layout(
        title=f"{subject_id} – 3D mesh: Prediction (red) vs GT (green)",
        scene=dict(aspectmode="data"),
        legend=dict(x=0.02, y=0.98)
    )

    html_file = CONFIG["html_output_path"] / f"{subject_id}_3d_mesh.html"
    fig.write_html(str(html_file))
    print(f"  3D mesh HTML saved: {html_file}")

    return {"dice_3d": dice, "iou_3d": iou}


# ----------------------------
# 2D HTML visualization
# ----------------------------
def visualize_2d_overlays(subject_id: str,
                          model: ImplicitNeuralRepresentation,
                          contours_2d: torch.Tensor,
                          pose_layer: PoseParameters,
                          chosen,
                          D: int,
                          config: dict):
    """
    Create a single-row HTML figure with num_views columns:
    overlay of pred (red) and GT (green) per selected slice.
    """
    device = torch.device(config["device"])
    model = model.to(device)
    extrinsics = pose_layer.get_matrices(device=device)

    num_views = len(chosen)
    fig = make_subplots(
        rows=1,
        cols=num_views,
        subplot_titles=[f"Slice index = {idx}" for idx in chosen]
    )

    for v in range(num_views):
        slice_index = chosen[v]
        z_depth = 2.0 * (slice_index / (D - 1)) - 1.0
        pred_slice = project_slice_from_inr(
            model,
            extrinsics[v],
            resolution=config["proj_resolution"],
            device=device,
            z_depth=z_depth
        ).detach().cpu().numpy()
        gt_slice = contours_2d[v].detach().cpu().numpy()

        overlay = np.zeros((pred_slice.shape[0], pred_slice.shape[1], 3), dtype=np.float32)
        overlay[..., 0] = pred_slice  # red channel: prediction
        overlay[..., 1] = gt_slice    # green channel: ground truth

        fig.add_trace(
            go.Image(z=(overlay * 255).astype(np.uint8)),
            row=1, col=v + 1
        )

    # add global legend text
    fig.update_layout(
        title=f"{subject_id} – 2D slices: Prediction (red) vs GT (green)",
        width=300 * num_views,
        height=320,
        margin=dict(l=0, r=0, t=40, b=40),
    )
    fig.add_annotation(
        text="Red = Prediction, Green = Ground Truth",
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        showarrow=False
    )

    html_file = CONFIG["html_output_path"] / f"{subject_id}_2d_slices.html"
    fig.write_html(str(html_file))
    print(f"  2D slice HTML saved: {html_file}")


# ----------------------------
# MAIN
# ----------------------------
def main():
    device = torch.device(CONFIG["device"])
    print("Device:", device)
    print("Data path:", CONFIG["data_path"])

    pairs = find_mitea_image_files(CONFIG["data_path"])[:10]
    if not pairs:
        print("No data pairs found - check data_path")
        return

    all_metrics_2d = {"dice": [], "iou": []}
    all_metrics_3d = {"dice": [], "iou": []}

    for idx, (img_file, label_file) in enumerate(pairs):
        subject_id = img_file.stem
        print("\n" + "-" * 60)
        print(f"[{idx+1}/{len(pairs)}] Subject: {subject_id}")

        try:
            vol, seg = load_mitea_subject(img_file, label_file)
            D, H, W = vol.shape
            print(f"Loaded vol shape: {vol.shape}, seg shape: {seg.shape}")

            # strict slice extraction
            try:
                slices_2d, contours_2d, chosen = extract_synthetic_2d_slices_strict(
                    vol, seg,
                    num_views=CONFIG["num_views"],
                    min_pixels=CONFIG["min_mask_pixels"],
                    out_size=CONFIG["image_size"]
                )
            except Exception as e:
                print("Skipping subject due to strict-slice selection:", e)
                continue

            print(f"Extracted slices: {slices_2d.shape}, contours: {contours_2d.shape}")
            print("Contours per-slice foreground pixels:",
                  [int(contours_2d[v].sum().item()) for v in range(contours_2d.shape[0])])

            # init model + pose
            model = ImplicitNeuralRepresentation(
                hidden_dim=CONFIG["hidden_dim"],
                num_layers=CONFIG["num_inr_layers"]
            )
            pose_layer = PoseParameters(CONFIG["num_views"])

            # optimize
            model, pose_layer, losses = optimize_single_subject(
                model, slices_2d, contours_2d,
                pose_layer, chosen,
                CONFIG,
                num_steps=CONFIG["num_optimization_steps"],
                D=D
            )

            # 2D evaluation
            metrics_2d = evaluate_subject_2d(
                model, contours_2d, pose_layer, chosen, D, CONFIG
            )
            print(f"  2D Dice: {metrics_2d['dice_2d']:.4f}, 2D IoU: {metrics_2d['iou_2d']:.4f}")
            all_metrics_2d["dice"].append(metrics_2d["dice_2d"])
            all_metrics_2d["iou"].append(metrics_2d["iou_2d"])

            # 3D evaluation + mesh
            metrics_3d = evaluate_subject_3d_and_mesh(
                model, seg, subject_id, CONFIG
            )
            all_metrics_3d["dice"].append(metrics_3d["dice_3d"])
            all_metrics_3d["iou"].append(metrics_3d["iou_3d"])

            # 2D overlays
            visualize_2d_overlays(
                subject_id, model, contours_2d, pose_layer, chosen, D, CONFIG
            )

            # save predicted occupancy grid as NIfTI (for external inspection)
            if CONFIG.get("save_nifti", True):
                R = int(CONFIG.get("eval_grid_resolution", 64))
                occ_grid = model.sample_grid(resolution=R, device=device, requires_grad=False)
                occ_np = occ_grid.detach().cpu().numpy()
                pred_nii = nib.Nifti1Image(occ_np.astype(np.float32), affine=np.eye(4))
                pred_file = CONFIG["checkpoint_path"] / f"{subject_id}_pred_occ_{R}cubed.nii.gz"
                nib.save(pred_nii, str(pred_file))
                print(f"  Saved predicted occupancy NIfTI: {pred_file}")

        except Exception as e:
            print("ERROR processing", subject_id, e)
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("FINAL RESULTS (2D)")
    if all_metrics_2d["dice"]:
        print(f"Mean 2D Dice: {np.mean(all_metrics_2d['dice']):.4f} ± {np.std(all_metrics_2d['dice']):.4f}")
        print(f"Mean 2D IoU:  {np.mean(all_metrics_2d['iou']):.4f} ± {np.std(all_metrics_2d['iou']):.4f}")
        print(f"Subjects processed: {len(all_metrics_2d['dice'])}")
    else:
        print("No successful subjects processed.")

    print("\nFINAL RESULTS (3D)")
    if all_metrics_3d["dice"]:
        print(f"Mean 3D Dice: {np.mean(all_metrics_3d['dice']):.4f} ± {np.std(all_metrics_3d['dice']):.4f}")
        print(f"Mean 3D IoU:  {np.mean(all_metrics_3d['iou']):.4f} ± {np.std(all_metrics_3d['iou']):.4f}")
    else:
        print("No 3D metrics (likely scikit-image missing or all subjects skipped).")


if __name__ == "__main__":
    main()
